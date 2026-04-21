"""Rigorous bake-off: which LLM should do query-time topic extraction?

Runs every golden question through a set of candidate models twice and
reports:

    - success rate   — fraction of calls returning >=1 parseable topic
    - consistency    — fraction of queries where the two runs produced
                       the same topic set (order-insensitive). A proxy
                       for determinism.
    - match rate     — fraction of extracted topics that resolve to an
                       existing topic in the DB via exact name match +
                       HNSW similarity. Measures semantic usefulness.
    - p50 / p95      — latency percentiles over all calls (N ≈ 74 per
                       model)
    - $/1k queries   — input+output tokens * published OpenRouter price

Design notes

    - Calls within a single model are gated by an asyncio.Semaphore
      (concurrency=3) so provider rate limits don't distort latency
      numbers. Between models we run serially so each model sees a
      clean load profile.
    - Unique extracted topic strings are resolved once per run (cached
      across models via TopicMatcher's name cache) so the match-rate
      check doesn't blow up the DB.
    - Prompt is the existing ``TopicExtractor.QUERY_PROMPT``, so the
      result is directly comparable to the live system.

Run:

    uv run python scripts/bakeoff_topic_extractor.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MTSS_ENV_FILE", ".env.test")
from tests.eval.env import setup_eval_env  # noqa: E402

setup_eval_env(enable_langfuse=False)


# (model_id, $/1M tokens in, $/1M tokens out)
CANDIDATES: List[Tuple[str, float, float]] = [
    ("openrouter/openai/gpt-5-nano",             0.05,  0.40),
    ("openrouter/openai/gpt-5.4-nano",           0.04,  0.32),
    ("openrouter/openai/gpt-5-mini",             0.25,  2.00),
    ("openrouter/google/gemini-2.5-flash-lite",  0.075, 0.30),
    ("openrouter/google/gemini-2.5-flash",       0.30,  2.50),
    ("openrouter/anthropic/claude-haiku-4.5",    1.00,  5.00),
    ("openrouter/x-ai/grok-4-fast",              0.20,  0.50),
    ("openrouter/deepseek/deepseek-chat",        0.27,  1.10),
]

RUNS_PER_QUERY = 2
PER_MODEL_CONCURRENCY = 3

# Prompt call uses ~350 input, ~30 output tokens for these short queries.
INPUT_TOK_EST = 350
OUTPUT_TOK_EST = 30


def load_questions() -> List[str]:
    path = REPO_ROOT / "tests" / "eval" / "goldens" / "questions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [q["question"] for q in data["questions"]]


async def call_extractor(model: str, query: str) -> Dict:
    from mtss.processing.topics import TopicExtractor

    extractor = TopicExtractor(llm_model=model)
    t0 = time.perf_counter()
    try:
        topics = await extractor.extract_topics_from_query(query)
        err = None
    except Exception as e:  # catch-all so one bad call doesn't nuke the run
        topics = []
        err = type(e).__name__ + ": " + str(e)[:100]
    return {
        "latency": time.perf_counter() - t0,
        "topics": [t.name for t in topics],
        "err": err,
    }


async def resolve_match_rate(
    extracted_names: List[str],
) -> Tuple[int, int]:
    """Return (matched, total) for the set of unique extracted topic names.

    Uses ``TopicMatcher.find_topic_by_name`` so the cached name-match
    + HNSW fallback gives us the same answer the real filter would.
    """
    from mtss.processing.embeddings import EmbeddingGenerator
    from mtss.processing.topics import TopicMatcher
    from mtss.storage.supabase_client import SupabaseClient

    client = SupabaseClient()
    matcher = TopicMatcher(client, EmbeddingGenerator())
    unique = sorted(set(extracted_names))
    matched = 0
    for name in unique:
        try:
            topic = await matcher.find_topic_by_name(name)
        except Exception:
            topic = None
        if topic is not None:
            matched += 1
    await client.close()
    return matched, len(unique)


def summarise_model(model: str, runs: List[List[Dict]], match_rate_pct: float, price_in: float, price_out: float) -> Dict:
    """Aggregate per-query runs (list of RUNS_PER_QUERY dicts) into a summary row."""
    flat = [r for q_runs in runs for r in q_runs]
    latencies = [r["latency"] for r in flat]
    successes = sum(1 for r in flat if not r["err"] and r["topics"])
    failures = sum(1 for r in flat if r["err"])
    empties = sum(1 for r in flat if not r["err"] and not r["topics"])

    # Consistency: for each query, do the RUNS_PER_QUERY runs agree on topic set?
    consistent = 0
    for q_runs in runs:
        if len(q_runs) < 2:
            continue
        sets = [frozenset(r["topics"]) for r in q_runs]
        if all(s == sets[0] for s in sets):
            consistent += 1

    cost_per_1k = (
        (INPUT_TOK_EST * price_in / 1_000_000)
        + (OUTPUT_TOK_EST * price_out / 1_000_000)
    ) * 1000

    return {
        "success_pct": 100.0 * successes / len(flat),
        "consistency_pct": 100.0 * consistent / len(runs),
        "match_pct": match_rate_pct,
        "fail_count": failures,
        "empty_count": empties,
        "p50": statistics.median(latencies),
        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        "max": max(latencies),
        "cost_per_1k": cost_per_1k,
    }


async def run_model(model: str, queries: List[str]) -> List[List[Dict]]:
    """Return list-of-lists: runs[query_idx][run_idx] = call result."""
    sem = asyncio.Semaphore(PER_MODEL_CONCURRENCY)
    runs: List[List[Dict]] = [[] for _ in queries]

    async def _one(qi: int, _run_idx: int, q: str) -> None:
        async with sem:
            r = await call_extractor(model, q)
            runs[qi].append(r)

    tasks = [
        asyncio.create_task(_one(qi, ri, q))
        for qi, q in enumerate(queries)
        for ri in range(RUNS_PER_QUERY)
    ]
    await asyncio.gather(*tasks)
    return runs


async def main():
    from mtss.processing.entity_cache import warm_caches
    from mtss.storage.supabase_client import SupabaseClient

    queries = load_questions()
    print(f"Loaded {len(queries)} golden questions, {len(CANDIDATES)} models, "
          f"{RUNS_PER_QUERY} runs per query = {len(queries) * len(CANDIDATES) * RUNS_PER_QUERY} calls total\n")

    # Pre-warm topic cache so match-rate resolution is fast.
    client = SupabaseClient()
    t0 = time.perf_counter()
    await warm_caches(client)
    print(f"Topic cache warmed in {time.perf_counter()-t0:.2f}s\n")
    await client.close()

    summary = {}
    for model, pin, pout in CANDIDATES:
        print(f"=== {model}")
        t0 = time.perf_counter()
        runs = await run_model(model, queries)
        dt = time.perf_counter() - t0

        # Collect all extracted topic names for match-rate resolution
        all_names = [n for q_runs in runs for r in q_runs for n in r["topics"]]
        if all_names:
            matched, total_unique = await resolve_match_rate(all_names)
            match_pct = 100.0 * matched / total_unique if total_unique else 0.0
        else:
            match_pct = 0.0

        summary[model] = summarise_model(model, runs, match_pct, pin, pout)
        s = summary[model]
        print(
            f"   success={s['success_pct']:5.1f}%  consistent={s['consistency_pct']:5.1f}%  "
            f"match={s['match_pct']:5.1f}%  p50={s['p50']:5.2f}s  p95={s['p95']:5.2f}s  "
            f"${s['cost_per_1k']:.3f}/1k  ({dt:.1f}s wall)"
        )

    print()
    print("=" * 112)
    print(f"{'model':<46} {'succ':>6} {'cons':>6} {'match':>6} {'p50':>6} {'p95':>6} {'max':>6} {'fail':>5} {'empty':>6} {'$/1k':>8}")
    print("=" * 112)
    # Sort: consistent first, then fastest, then cheapest
    for model, s in sorted(summary.items(), key=lambda kv: (-kv[1]["success_pct"], -kv[1]["consistency_pct"], kv[1]["p50"], kv[1]["cost_per_1k"])):
        print(
            f"{model:<46} {s['success_pct']:5.1f}% {s['consistency_pct']:5.1f}% "
            f"{s['match_pct']:5.1f}% {s['p50']:5.2f}s {s['p95']:5.2f}s {s['max']:5.2f}s "
            f"{s['fail_count']:>5} {s['empty_count']:>6} ${s['cost_per_1k']:6.3f}"
        )
    print("=" * 112)


if __name__ == "__main__":
    asyncio.run(main())

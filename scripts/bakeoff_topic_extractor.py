"""Bake-off: which LLM should do query-time topic extraction?

Current default is gpt-5-nano, which returns ``content=None`` on roughly
half of JSON-mode prompts (reasoning mode "thinks" then produces no
output). That reliability gap forces broad-search fallback and erases the
topic-filter speed win. This script compares three candidates on the
same set of real eval queries and reports success rate, latency, match
rate, and projected cost per 1000 queries.

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
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MTSS_ENV_FILE", ".env.test")
from tests.eval.env import setup_eval_env
setup_eval_env(enable_langfuse=False)


# (model_id, pricing $/1M tokens in, $/1M tokens out)
# Pricing as of 2026-04 per OpenRouter listings; adjust if stale.
CANDIDATES: List[tuple[str, float, float]] = [
    ("openrouter/openai/gpt-5-nano",          0.05, 0.40),   # current default
    ("openrouter/openai/gpt-5.4-nano",        0.04, 0.32),   # cheaper than 5-mini, better reasoning than 5-nano
    ("openrouter/google/gemini-2.5-flash-lite", 0.075, 0.30),  # fast, strong JSON
    ("openrouter/openai/gpt-5-mini",          0.25, 2.00),   # known-good baseline
]

# Real eval queries spanning match / partial / no-match scenarios.
TEST_QUERIES = [
    "What maintenance issues are currently open for MARAN CANOPUS?",
    "Any recent lubricant supply orders?",
    "Summarise safety incidents for CANOPUS",
    "Tell me about cargo damage reports",
    "What engine problems happened on the VLCC fleet?",
    "Show me hull inspection findings",
    "Lloyd's Register audit findings for 2024",
    "Crew injury reports this quarter",
    "Ballast water treatment system failures",
    "Port state control deficiencies and detentions",
]


async def run_one(model: str, query: str) -> Dict:
    """Time a single extraction and return latency + token counts."""
    from mtss.processing.topics import TopicExtractor

    extractor = TopicExtractor(llm_model=model)
    t0 = time.perf_counter()
    try:
        topics = await extractor.extract_topics_from_query(query)
        ok = True
        err = None
    except Exception as e:
        topics = []
        ok = False
        err = str(e)[:120]
    dt = time.perf_counter() - t0
    return {
        "latency": dt,
        "ok": ok,
        "topic_count": len(topics),
        "topics": [t.name for t in topics],
        "err": err,
    }


async def main():
    print(f"Running {len(TEST_QUERIES)} queries × {len(CANDIDATES)} models…\n")

    # Token count estimation: query prompt is ~350 tokens input; response is
    # ~30 tokens for 1-3 topics. Use these as rough cost denominators —
    # real production sees similar magnitudes for this prompt.
    INPUT_TOK_EST = 350
    OUTPUT_TOK_EST = 30

    results = {}
    for model, price_in, price_out in CANDIDATES:
        print(f"=== {model} ===")
        runs = []
        for q in TEST_QUERIES:
            r = await run_one(model, q)
            runs.append(r)
            status = "OK " if r["ok"] and r["topic_count"] > 0 else "EMPTY" if r["ok"] else "FAIL"
            print(f"  [{status}] {r['latency']:5.2f}s  topics={r['topics']!s:<50}  q={q[:50]!r}")

        latencies = [r["latency"] for r in runs]
        ok_non_empty = sum(1 for r in runs if r["ok"] and r["topic_count"] > 0)
        failures = sum(1 for r in runs if not r["ok"])
        empties = sum(1 for r in runs if r["ok"] and r["topic_count"] == 0)

        cost_per_1k = (
            (INPUT_TOK_EST * price_in / 1_000_000) +
            (OUTPUT_TOK_EST * price_out / 1_000_000)
        ) * 1000

        results[model] = {
            "p50_latency": statistics.median(latencies),
            "max_latency": max(latencies),
            "success_rate": ok_non_empty / len(runs),
            "fail_count": failures,
            "empty_count": empties,
            "cost_per_1k_queries": cost_per_1k,
        }

    print("\n" + "=" * 90)
    print(f"{'model':<45} {'p50':>7} {'max':>7} {'ok':>6} {'fail':>5} {'empty':>6} {'$/1k':>8}")
    print("=" * 90)
    for model, s in results.items():
        print(
            f"{model:<45} {s['p50_latency']:6.2f}s {s['max_latency']:6.2f}s "
            f"{s['success_rate']*100:5.0f}% {s['fail_count']:>5} {s['empty_count']:>6} "
            f"${s['cost_per_1k_queries']:6.3f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())

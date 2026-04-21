"""Tune the topic filter: threshold * top-K * judged precision.

Measures on the 37 golden questions:

  For each (threshold, K) combination:
    - match rate      — fraction of queries with >=1 matched DB topic
    - avg topics/q    — mean number of matched topics per query (coverage)
    - avg chunks/q    — mean total chunk_count across matched topics
    - precision@K     — fraction of matched topics judged relevant by LLM

Uses gemini-2.5-flash-lite (the post-Haiku reliability/cost winner) as
the fixed extractor, so the tuning isolates filter behavior from
extractor noise. Judge is the same model — its job is binary relevance
over (user query, DB topic name) pairs, which is a much simpler task
than answer judgment and does not need a heavy model.
"""

from __future__ import annotations

import asyncio
import json
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


EXTRACTOR_MODEL = "openrouter/google/gemini-2.5-flash-lite"
JUDGE_MODEL = "openrouter/google/gemini-2.5-flash-lite"

# Sweep grid — threshold × K. Strict top-1 @ 0.70 is the current behavior.
THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
KS = [1, 3, 5, 10]


def load_questions() -> List[str]:
    path = REPO_ROOT / "tests" / "eval" / "goldens" / "questions.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [q["question"] for q in data["questions"]]


async def judge_relevance(query: str, candidates: List[str]) -> List[bool]:
    """Ask the judge: for each candidate DB topic, is it a relevant
    filter target for the user's query?

    Returns list[bool] in candidate order.
    """
    if not candidates:
        return []
    from litellm import acompletion

    prompt = (
        "You are scoring retrieval filter relevance for a maritime email search system.\n"
        "User asked: " + json.dumps(query) + "\n\n"
        "For each candidate topic below, answer YES if it is directly relevant\n"
        "to the user's question (would return useful results), NO if it is unrelated\n"
        "or tangential.\n\n"
        "Candidates:\n"
        + "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        + "\n\nReturn a JSON array of YES/NO in order, e.g. [\"YES\",\"NO\",\"YES\"]."
    )
    try:
        resp = await acompletion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = resp.choices[0].message.content or "[]"
        if "[" in text:
            text = text[text.index("[") : text.rindex("]") + 1]
        data = json.loads(text)
        return [str(v).strip().upper().startswith("Y") for v in data[: len(candidates)]] + [
            False
        ] * max(0, len(candidates) - len(data))
    except Exception as e:
        print(f"  judge error: {e}")
        return [False] * len(candidates)


async def main():
    from mtss.processing.entity_cache import warm_caches, get_topic_cache
    from mtss.processing.embeddings import EmbeddingGenerator
    from mtss.processing.topics import TopicExtractor
    from mtss.storage.supabase_client import SupabaseClient

    queries = load_questions()
    print(f"Tuning on {len(queries)} queries × {len(THRESHOLDS)} thresholds × {len(KS)} K values\n")

    client = SupabaseClient()
    await warm_caches(client)
    tc = get_topic_cache()
    emb_gen = EmbeddingGenerator()
    extractor = TopicExtractor(llm_model=EXTRACTOR_MODEL)

    # Phase 1: extract topics for each query (once) and embed each
    # extracted name (once). Deterministic enough for this purpose.
    extracted_by_query: Dict[str, List[str]] = {}
    emb_by_name: Dict[str, List[float]] = {}
    print("Phase 1: extracting topics + embedding names…")
    t0 = time.perf_counter()
    for q in queries:
        topics = await extractor.extract_topics_from_query(q)
        names = [t.name for t in topics]
        extracted_by_query[q] = names
        for n in names:
            if n not in emb_by_name:
                emb_by_name[n] = await emb_gen.generate_embedding(n)
    print(f"  done in {time.perf_counter()-t0:.1f}s; {sum(len(v) for v in extracted_by_query.values())} extractions, {len(emb_by_name)} uniq names")

    # Phase 2: for each threshold, find top-K_max candidates per extracted name once
    # (we can slice down to smaller K). Cache results by (threshold, name).
    candidates_cache: Dict[Tuple[float, str], List[dict]] = {}
    K_MAX = max(KS)
    print(f"\nPhase 2: candidate lookup (threshold × name) at K={K_MAX}…")
    t0 = time.perf_counter()
    for T in THRESHOLDS:
        for name, e in emb_by_name.items():
            sims = await client.find_similar_topics(e, threshold=T, limit=K_MAX)
            candidates_cache[(T, name)] = sims
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Phase 3: LLM-judge relevance for each (query, candidate_name) pair.
    # Collect unique pairs first so we judge each only once, not per grid cell.
    unique_pairs: Dict[Tuple[str, str], bool] = {}
    for q in queries:
        for name in extracted_by_query[q]:
            for T in THRESHOLDS:
                for c in candidates_cache[(T, name)][:K_MAX]:
                    unique_pairs.setdefault((q, c["name"]), None)  # None = not judged yet
    print(f"\nPhase 3: judging {len(unique_pairs)} unique (query, candidate) pairs…")
    # Judge in batches per query to give the LLM shared context.
    by_query_pairs: Dict[str, List[str]] = {}
    for q, name in unique_pairs.keys():
        by_query_pairs.setdefault(q, []).append(name)
    t0 = time.perf_counter()
    for q, names in by_query_pairs.items():
        uniq = list(dict.fromkeys(names))  # preserve order
        verdicts = await judge_relevance(q, uniq)
        for name, v in zip(uniq, verdicts):
            unique_pairs[(q, name)] = v
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    # Phase 4: grid over (threshold, K)
    print("\nResults grid (rows = threshold, cols = K):\n")
    rows = []
    for T in THRESHOLDS:
        for K in KS:
            matched_queries = 0
            topics_per_q: List[int] = []
            chunks_per_q: List[int] = []
            relevant = 0
            total_matches = 0
            for q in queries:
                pooled: Dict[str, float] = {}  # topic name -> best similarity across extracted names
                pooled_ids: Dict[str, str] = {}  # topic name -> id
                for name in extracted_by_query[q]:
                    for c in candidates_cache[(T, name)][:K]:
                        if c["name"] not in pooled or pooled[c["name"]] < c["similarity"]:
                            pooled[c["name"]] = c["similarity"]
                            pooled_ids[c["name"]] = c["id"]
                if pooled:
                    matched_queries += 1
                    topics_per_q.append(len(pooled))
                    total_chunks = 0
                    for tname, tid in pooled_ids.items():
                        from uuid import UUID
                        t = tc.get_by_id(UUID(tid) if isinstance(tid, str) else tid)
                        if t:
                            total_chunks += t.chunk_count
                        total_matches += 1
                        if unique_pairs.get((q, tname)):
                            relevant += 1
                    chunks_per_q.append(total_chunks)
                else:
                    topics_per_q.append(0)
                    chunks_per_q.append(0)

            match_rate = matched_queries / len(queries)
            precision = (relevant / total_matches) if total_matches else 0.0
            rows.append({
                "T": T, "K": K,
                "match_rate": match_rate,
                "avg_topics": statistics.mean(topics_per_q),
                "avg_chunks": statistics.mean(chunks_per_q),
                "precision": precision,
                "total_matches": total_matches,
            })

    print(f"{'T':>5} {'K':>3} {'match%':>7} {'avgTopics':>10} {'avgChunks':>10} {'prec@K':>7} {'total_matches':>14}")
    print("-" * 62)
    for r in rows:
        print(f"{r['T']:>5.2f} {r['K']:>3d} {r['match_rate']*100:>6.1f}% {r['avg_topics']:>10.2f} {r['avg_chunks']:>10.1f} {r['precision']*100:>6.1f}% {r['total_matches']:>14d}")

    # Highlight: best (T, K) by a weighted F1-like score
    def score(r):
        return r["match_rate"] * r["precision"] * (r["avg_chunks"] ** 0.3)

    best = max(rows, key=score)
    print(f"\nBest by match_rate * precision * chunks^0.3: T={best['T']} K={best['K']} "
          f"match={best['match_rate']*100:.1f}% precision={best['precision']*100:.1f}% "
          f"avgChunks={best['avg_chunks']:.1f}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

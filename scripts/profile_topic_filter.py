"""Time-profile each step of TopicFilter.analyze_query on a real question.

Run: uv run python scripts/profile_topic_filter.py "your question here"
Defaults to the CANOPUS maintenance question.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("MTSS_ENV_FILE", ".env.test")
from tests.eval.env import setup_eval_env
setup_eval_env(enable_langfuse=False)


async def profile(query: str) -> None:
    from mtss.processing.topics import TopicExtractor, TopicMatcher
    from mtss.processing.embeddings import EmbeddingGenerator
    from mtss.storage.supabase_client import SupabaseClient
    from mtss.rag.topic_filter import TopicFilter

    client = SupabaseClient()
    extractor = TopicExtractor()
    embeddings = EmbeddingGenerator()
    matcher = TopicMatcher(client, embeddings)

    timings = []
    def mark(label: str, start: float) -> None:
        timings.append((label, time.perf_counter() - start))

    # 1. Extract topics
    t = time.perf_counter()
    extracted = await extractor.extract_topics_from_query(query)
    mark(f"extract_topics_from_query (got {len(extracted)} topics)", t)
    names = [e.name for e in extracted]
    print(f"  extracted: {names}")

    # 2. For each extracted name, separately time the sub-steps of find_topic_by_name
    for name in names:
        # 2a. Exact DB match
        t = time.perf_counter()
        normalized = matcher._normalize_name(name)
        existing = await client.get_topic_by_name(normalized)
        mark(f"  exact get_topic_by_name({name!r})  hit={existing is not None}", t)

        if not existing:
            # 2b. Embed topic name
            t = time.perf_counter()
            emb = await embeddings.generate_embedding(name)
            mark(f"  embed topic name {name!r}", t)

            # 2c. Vector similarity search on topics table
            t = time.perf_counter()
            sims = await client.find_similar_topics(emb, threshold=0.5, limit=1)
            mark(f"  find_similar_topics({name!r})  hits={len(sims)}", t)

    # 3. Full analyze_query (end-to-end)
    t = time.perf_counter()
    tf = TopicFilter(extractor, matcher, client)
    result = await tf.analyze_query(query)
    mark(f"FULL analyze_query  detected={len(result.detected_topics)} matched={len(result.matched_topics)} skip={result.should_skip_rag}", t)

    print("\n===== TIMING =====")
    for label, secs in timings:
        print(f"  {secs:6.2f}s  {label}")
    print(f"\nquery: {query!r}")
    print(f"result.should_skip_rag = {result.should_skip_rag}")
    print(f"result.total_chunk_count = {result.total_chunk_count}")

    await client.close()


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What maintenance issues are currently open for MARAN CANOPUS?"
    asyncio.run(profile(query))

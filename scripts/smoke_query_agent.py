"""Smoke-test the RAG agent with one ad-hoc question.

Invokes the same LangGraph used by the API. Prints the final answer plus
the vessel names mentioned in each retrieved chunk's source metadata so we
can tell at a glance whether the response cites real fleet vessels or a
hallucinated name.

Usage:
    set -a; source .env.test; set +a
    uv run python scripts/smoke_query_agent.py "when was the last oil change on any vessel"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from mtss.api.agent import create_graph


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="The question to ask the agent")
    args = ap.parse_args()
    question = " ".join(args.question)

    graph = create_graph(MemorySaver())
    config: dict[str, Any] = {"configurable": {"thread_id": "smoke-test"}}

    citation_map: dict[str, Any] | None = None
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            if node_name == "search_node" and update.get("citation_map"):
                citation_map = update["citation_map"]

    final_state = await graph.aget_state(config)
    messages = final_state.values.get("messages", []) if final_state.values else []
    answer = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and getattr(msg, "type", "") == "ai":
            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                continue
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    print("=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    print(f"Retrieved {len(citation_map or {})} chunks")
    if citation_map:
        print("\nTop sources (email_subject / file_path):")
        for i, (cid, data) in enumerate(list(citation_map.items())[:10], 1):
            subj = data.get("email_subject") or data.get("source_title") or "?"
            fp = data.get("file_path") or ""
            print(f"  {i:2d}. {cid[:8]}  {subj}  [{fp}]")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

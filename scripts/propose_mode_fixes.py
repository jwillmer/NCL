"""Investigate embedding-mode drift: for each non-image doc with an
archived markdown, run the current decider against that markdown and flag
any doc whose stored mode disagrees with what the decider would choose now.

Emits a proposals JSONL (one line per flagged doc). Review the list, then
apply via ``mtss re-embed --doc-id X --mode Y --force`` per row, or feed
the list to a batch applier.

Read-only: does not modify the ingest.db.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# Filename tokens that strongly suggest a log/data dump — used only to
# sort the output, not to override decider results.
_LOG_HINTS = (
    "log", "report", "analysis", "icmp", "iccp", "bwts", "bwms",
    "scrubber", "sensor", "datalogger", "data_logger", "weekly",
    "monthly", "friday", "composite", "boiler", "auxiliary",
    "datareport", "ecs_",
)


def _has_log_hint(file_name: str) -> bool:
    low = (file_name or "").lower()
    return any(h in low for h in _LOG_HINTS)


def _extract_content_from_md(md: str) -> str:
    """Return the ``## Content`` section body, or full markdown if missing."""
    marker = "## Content\n"
    idx = md.find(marker)
    if idx == -1:
        return md
    return md[idx + len(marker):].strip()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/mode_fix_proposals.jsonl"),
        help="Where to write the proposals JSONL",
    )
    parser.add_argument(
        "--only-types",
        default="attachment_pdf,attachment_xlsx,attachment_xls,attachment_docx,attachment_csv",
        help="Comma-separated document_types to include",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N docs scanned (0 = all)",
    )
    args = parser.parse_args()

    from mtss.ingest.embedding_decider import decide_embedding_mode
    from mtss.parsers.llamaparse_parser import strip_llamaparse_image_refs
    from mtss.storage.sqlite_client import SqliteStorageClient

    db_path = args.output_dir / "ingest.db"
    archive_root = args.output_dir / "archive"
    if not db_path.exists():
        print(f"missing ingest.db in {args.output_dir}")
        return 1

    allowed_types = {t.strip() for t in args.only_types.split(",") if t.strip()}

    proposals: List[Dict[str, Any]] = []
    scanned = 0
    unchanged = 0
    missing_md = 0
    mode_transitions: Counter[str] = Counter()

    client = SqliteStorageClient(output_dir=args.output_dir)
    try:
        for d in client.iter_documents():
            if d.get("document_type") not in allowed_types:
                continue
            if d.get("status") == "failed":
                continue
            uri = d.get("archive_browse_uri")
            if not uri:
                continue
            rel = uri.removeprefix("/archive/")
            md_path = archive_root / rel
            if not md_path.is_file():
                missing_md += 1
                continue

            try:
                md = md_path.read_text(encoding="utf-8")
            except OSError:
                missing_md += 1
                continue
            md = strip_llamaparse_image_refs(_extract_content_from_md(md))
            if not md.strip():
                missing_md += 1
                continue

            current = d.get("embedding_mode")
            try:
                decision = await decide_embedding_mode(md, None)
            except Exception as e:
                proposals.append({
                    "doc_id": d.get("doc_id"),
                    "file_name": d.get("file_name"),
                    "document_type": d.get("document_type"),
                    "current_mode": current,
                    "proposed_mode": None,
                    "reason": f"decider_error: {e}",
                    "log_hint": _has_log_hint(d.get("file_name") or ""),
                })
                scanned += 1
                continue

            proposed = decision.mode.value
            transition = f"{current} -> {proposed}"
            if current == proposed:
                unchanged += 1
            else:
                mode_transitions[transition] += 1
                proposals.append({
                    "doc_id": d.get("doc_id"),
                    "file_name": d.get("file_name"),
                    "document_type": d.get("document_type"),
                    "current_mode": current,
                    "proposed_mode": proposed,
                    "reason": decision.reason,
                    "signals": {
                        k: decision.signals.get(k)
                        for k in (
                            "total_tokens", "digit_ratio", "table_char_pct",
                            "repetition_score", "short_line_ratio", "prose_ratio",
                        )
                    },
                    "log_hint": _has_log_hint(d.get("file_name") or ""),
                })
            scanned += 1
            if args.limit and scanned >= args.limit:
                break
    finally:
        try:
            client._conn.close()
        except Exception:
            pass

    # Write proposals
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"scanned: {scanned}")
    print(f"unchanged (current == proposed): {unchanged}")
    print(f"missing archive md: {missing_md}")
    print(f"proposed changes: {sum(mode_transitions.values())}")
    print()
    print("transitions:")
    for t, n in mode_transitions.most_common():
        print(f"  {n:5d}  {t}")
    print()
    print(f"proposals written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

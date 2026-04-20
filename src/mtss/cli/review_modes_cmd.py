"""mtss review-modes — surface embedding-mode decisions for human review.

Every ingest stamps an ``embedding_mode_decided`` event carrying the reason
(deterministic rule name or triage_* letter). This command aggregates those
signals by risk tier so the operator can spot-check a small set rather than
the full corpus:

  1. Triage-sourced decisions (inherently non-deterministic) — every one.
  2. SUMMARY / METADATA_ONLY docs — potential false negatives where we
     may have lost prose.
  3. Large ``full``-mode docs — potential false positives where a sensor
     log slipped through as prose.

Any flagged doc can be overridden via ``mtss re-embed --doc-id X --mode Y
--force``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer
from rich.console import Console
from rich.table import Table


console = Console()


_TRIAGE_REASONS = {"triage_prose", "triage_dense", "triage_noise", "triage_failed"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load rows from ``ingest.db``. ``path`` is parsed for its parent
    directory and table name; the JSONL file itself is no longer consulted."""
    from ..storage.sqlite_client import SqliteStorageClient

    output_dir = path.parent
    table = path.stem
    db_path = output_dir / "ingest.db"
    if not db_path.exists():
        raise FileNotFoundError(f"ingest.db not found in {output_dir}")

    client = SqliteStorageClient(output_dir=output_dir)
    try:
        if table == "documents":
            return list(_docs_from_db(client))
        if table == "ingest_events":
            return [dict(r) for r in client._conn.execute("SELECT * FROM ingest_events ORDER BY rowid")]
        return []
    finally:
        try:
            client._conn.close()
        except Exception:
            pass


def _docs_from_db(client) -> Iterable[Dict[str, Any]]:
    for row in client.iter_documents():
        meta = row.get("metadata_json")
        if meta:
            try:
                parsed = json.loads(meta)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        row.setdefault(k, v)
            except (TypeError, ValueError):
                pass
        row.pop("metadata_json", None)
        yield row


def _latest_decision_by_doc(events: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Last ``embedding_mode_decided`` reason per doc_id. Messages look like
    ``"<mode>: <reason>"``; only the reason is needed for bucketing."""
    by_doc: Dict[str, str] = {}
    for e in events:
        if e.get("event_type") != "embedding_mode_decided":
            continue
        did = e.get("document_id")
        msg = e.get("message") or ""
        if not did or ":" not in msg:
            continue
        _, _, reason = msg.partition(":")
        by_doc[did] = reason.strip()
    return by_doc


def _candidate_docs(
    docs: List[Dict[str, Any]], include_images: bool
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in docs:
        if not include_images and d.get("document_type") == "attachment_image":
            continue
        if d.get("status") == "failed":
            continue
        if not d.get("embedding_mode"):
            continue
        out.append(d)
    return out


def _format_row(d: Dict[str, Any], reason: Optional[str]) -> tuple[str, str, str, str]:
    """Return (mode, reason, doc_id, label) for one row."""
    mode = d.get("embedding_mode") or "?"
    doc_id = (d.get("doc_id") or d.get("id") or "?")[:16]
    label = d.get("source_title") or d.get("file_name") or d.get("doc_id") or "?"
    return mode, reason or "-", doc_id, label


def _render_table(title: str, rows: List[tuple[str, str, str, str]], limit: int) -> None:
    if not rows:
        console.print(f"[dim]{title}: none[/dim]")
        return
    t = Table(title=f"{title} ({len(rows)})", title_justify="left")
    t.add_column("mode", style="cyan")
    t.add_column("reason", style="yellow")
    t.add_column("doc_id")
    t.add_column("label")
    for mode, reason, doc_id, label in rows[:limit]:
        t.add_row(mode, reason, doc_id, label)
    if len(rows) > limit:
        t.caption = f"…{len(rows) - limit} more (rerun with --limit to see more)"
    console.print(t)


def register(app: "typer.Typer") -> None:
    @app.command("review-modes")
    def review_modes(
        output_dir: Path = typer.Option(
            Path("data/output"), "--output-dir", "-o",
            help="Output directory containing ingest.db",
        ),
        limit: int = typer.Option(
            50, "--limit", help="Max rows per tier (default 50)",
        ),
        include_images: bool = typer.Option(
            False, "--include-images",
            help="Also list image attachments (skipped by default)",
        ),
        full_min_tokens: int = typer.Option(
            50_000, "--full-min-tokens",
            help="Flag FULL-mode docs whose parsed content exceeds this "
                 "token estimate (possible log mis-classified as prose). "
                 "Used only when the decider emitted 'effective_tokens' in "
                 "the event payload; falls back to approx 4 chars/token.",
        ),
    ) -> None:
        """Audit embedding-mode decisions by risk tier.

        Tier 1 — Triage-sourced: non-deterministic LLM decisions.
        Tier 2 — SUMMARY / METADATA_ONLY: possible lost-prose cases.
        Tier 3 — Oversized FULL: possible embedded-sensor-log cases.
        """
        docs_path = output_dir / "documents.jsonl"
        events_path = output_dir / "ingest_events.jsonl"
        if not (output_dir / "ingest.db").exists():
            console.print(f"[red]ingest.db not found in {output_dir}[/red]")
            raise typer.Exit(1)

        docs = _load_jsonl(docs_path)
        events = _load_jsonl(events_path)
        reason_by_doc = _latest_decision_by_doc(events)

        candidates = _candidate_docs(docs, include_images)

        # Mode distribution for the summary header.
        dist = Counter(d.get("embedding_mode") or "?" for d in candidates)
        header = Table(title="Embedding-mode distribution", title_justify="left")
        header.add_column("mode", style="cyan")
        header.add_column("count", justify="right")
        for mode, n in dist.most_common():
            header.add_row(mode, str(n))
        console.print(header)

        tier1: List[tuple[str, str, str, str]] = []
        tier2: List[tuple[str, str, str, str]] = []
        tier3: List[tuple[str, str, str, str]] = []

        for d in candidates:
            # Match event by the full id (UUID). attachment_handler logs the
            # document.id, which is the UUID; docs.jsonl keys that as "id".
            uuid = d.get("id")
            reason = reason_by_doc.get(uuid)

            if reason in _TRIAGE_REASONS:
                tier1.append(_format_row(d, reason))
                continue
            if d.get("embedding_mode") in {"summary", "metadata_only"}:
                tier2.append(_format_row(d, reason))
                continue
            if d.get("embedding_mode") == "full":
                size = int(d.get("attachment_size_bytes") or 0)
                # Crude token estimate — 4 chars/token. Good enough for a
                # yellow flag that an oversized "prose" doc deserves a look.
                approx_tokens = size // 4
                if approx_tokens >= full_min_tokens:
                    tier3.append(_format_row(d, reason or f"~{approx_tokens:,} tokens"))

        _render_table("Tier 1 — triage-sourced (non-deterministic)", tier1, limit)
        _render_table("Tier 2 — SUMMARY / METADATA_ONLY (possible lost prose)", tier2, limit)
        _render_table(
            f"Tier 3 — oversized FULL (possible embedded log, >{full_min_tokens:,} tokens est.)",
            tier3,
            limit,
        )

        console.print()
        console.print(
            "[dim]Override any row with: "
            "[bold]mtss re-embed --doc-id <doc_id> --mode <full|summary|metadata_only> --force[/bold][/dim]"
        )

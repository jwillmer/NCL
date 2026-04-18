"""Classify ingested docs by their usefulness for RAG retrieval.

Why
---
Long PDFs are the biggest signal-to-noise risk in the corpus. A 500-page
sensor log gets chunked and embedded identically to a 500-page technical
manual, but only one of the two is useful for Q&A retrieval. Embeddings
derived from numeric log rows pollute retrieval results and burn token
budget. This script inspects already-ingested documents, classifies them,
and writes a review report so the operator can decide what actually lands
in Supabase.

What
----
For each target document (default: text content >= 20k chars), the analyzer:

1. Loads the doc's concatenated chunk content from ``chunks.jsonl``.
2. Computes structural heuristics — numeric density, unique-word ratio,
   line regularity, sentence density. These catch the obvious extremes:
   sensor logs (numeric + repetitive), tabular dumps (tight short lines),
   OCR noise (low unique-word ratio).
3. If heuristics are inconclusive, samples three excerpts (head / middle /
   tail) and asks a small LLM to classify: keep / summary_only / skip.
4. Emits ``rag_utility_report.jsonl`` with the classification, the reason,
   and the raw metrics for audit.

The report is purely informational — this script NEVER modifies
``documents.jsonl`` or ``chunks.jsonl``. A separate apply step (future)
will flip a ``rag_utility`` field once the operator reviews the report.

Classification scale
--------------------
- ``keep``         — narrative / technical / instructional; embed as-is.
- ``summary_only`` — structured data that's worth discovering but noisy
                     at chunk granularity (certificates, forms, tables).
- ``skip``         — sensor/telemetry/OCR-noise; no Q&A value.

Usage
-----
    uv run python scripts/analyze_rag_utility.py --output-dir data/output
    uv run python scripts/analyze_rag_utility.py --output-dir data/output \\
        --min-chars 50000 --no-llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger("rag_utility")


CLASSIFICATION_KEEP = "keep"
CLASSIFICATION_SUMMARY = "summary_only"
CLASSIFICATION_SKIP = "skip"
CLASSIFICATION_VALUES = {CLASSIFICATION_KEEP, CLASSIFICATION_SUMMARY, CLASSIFICATION_SKIP}

SOURCE_HEURISTIC = "heuristic"
SOURCE_LLM = "llm"
SOURCE_FALLBACK = "fallback"


@dataclass
class Metrics:
    char_count: int
    word_count: int
    unique_word_ratio: float   # unique lowercase words / total words
    numeric_char_ratio: float  # digit/.,:;- chars / alphanumeric chars
    avg_line_length: float
    short_line_ratio: float    # lines with < 5 words / total non-empty lines
    avg_sentence_words: float  # words / sentences (split on .!?)

    def as_dict(self) -> dict:
        return {
            "char_count": self.char_count,
            "word_count": self.word_count,
            "unique_word_ratio": round(self.unique_word_ratio, 4),
            "numeric_char_ratio": round(self.numeric_char_ratio, 4),
            "avg_line_length": round(self.avg_line_length, 2),
            "short_line_ratio": round(self.short_line_ratio, 4),
            "avg_sentence_words": round(self.avg_sentence_words, 2),
        }


def compute_metrics(text: str) -> Metrics:
    """Structural metrics over a doc's full concatenated content."""
    char_count = len(text)
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    word_count = len(words)
    lowered = [w.lower() for w in words]
    unique_word_ratio = (len(set(lowered)) / word_count) if word_count else 0.0

    alnum_chars = sum(1 for c in text if c.isalnum())
    numeric_chars = sum(1 for c in text if c.isdigit() or c in ".,:;-+")
    # Only count numeric chars among alnum+numeric-punct to get a
    # scale-invariant ratio regardless of whitespace volume.
    numeric_denom = alnum_chars + sum(1 for c in text if c in ".,:;-+")
    numeric_char_ratio = (numeric_chars / numeric_denom) if numeric_denom else 0.0

    lines = [ln for ln in text.splitlines() if ln.strip()]
    avg_line_length = (sum(len(ln) for ln in lines) / len(lines)) if lines else 0.0
    short_lines = sum(1 for ln in lines if len(ln.split()) < 5)
    short_line_ratio = (short_lines / len(lines)) if lines else 0.0

    sentences = re.split(r"[.!?]+\s+", text)
    sentence_count = sum(1 for s in sentences if s.strip())
    avg_sentence_words = (word_count / sentence_count) if sentence_count else 0.0

    return Metrics(
        char_count=char_count,
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        numeric_char_ratio=numeric_char_ratio,
        avg_line_length=avg_line_length,
        short_line_ratio=short_line_ratio,
        avg_sentence_words=avg_sentence_words,
    )


def heuristic_classify(m: Metrics) -> tuple[str | None, str]:
    """Return (classification, reason) — or (None, reason) if inconclusive.

    Strong signals short-circuit the LLM call:
      - numeric-heavy + repetitive  -> skip (sensor log / telemetry)
      - mostly short tight lines    -> summary_only (tabular data)
      - narrative-rich              -> keep (prose)
    """
    if m.word_count < 100:
        return CLASSIFICATION_SKIP, "almost_no_text"
    # Sensor/telemetry log: lots of numbers *and* low vocabulary. Catch this
    # before the tabular branch so a pure-number table doesn't silently get
    # tagged summary_only and waste a summarisation call.
    if m.numeric_char_ratio > 0.55 and m.unique_word_ratio < 0.08:
        return CLASSIFICATION_SKIP, (
            f"numeric_heavy_repetitive "
            f"(num={m.numeric_char_ratio:.2f}, uniq={m.unique_word_ratio:.2f})"
        )
    # Certificates / forms / field lists: mostly short labelled lines. These
    # also show low unique-word ratios (field names repeat), but the structure
    # is worth a single summary line rather than a hard skip.
    if m.short_line_ratio > 0.85 and m.avg_line_length < 30:
        return CLASSIFICATION_SUMMARY, (
            f"mostly_short_lines "
            f"(short={m.short_line_ratio:.2f}, avg_len={m.avg_line_length:.1f})"
        )
    # Pure degenerate repetition ("notice intentionally left blank" x500).
    if m.unique_word_ratio < 0.03:
        return CLASSIFICATION_SKIP, f"extreme_repetition (uniq={m.unique_word_ratio:.3f})"
    # Narrative-rich prose. Threshold calibrated against multi-paragraph
    # technical emails — longer docs naturally re-use domain vocabulary, so
    # requiring 0.4 misses the vast majority of real keep-worthy content.
    if m.unique_word_ratio > 0.15 and m.avg_sentence_words > 8:
        return CLASSIFICATION_KEEP, (
            f"narrative_rich "
            f"(uniq={m.unique_word_ratio:.2f}, avg_sw={m.avg_sentence_words:.1f})"
        )
    return None, "needs_llm_review"


def sample_excerpts(text: str, n_excerpts: int = 3, each_chars: int = 1500) -> list[str]:
    """Head / middle / tail samples for the LLM classifier."""
    if len(text) <= n_excerpts * each_chars:
        return [text]
    if n_excerpts == 1:
        return [text[:each_chars]]
    mid_start = max(each_chars, (len(text) // 2) - (each_chars // 2))
    return [
        text[:each_chars],
        text[mid_start : mid_start + each_chars],
        text[-each_chars:],
    ]


LLM_SYSTEM_PROMPT = """You decide whether a document's content is useful for Retrieval-Augmented Generation (RAG) Q&A.

You will see 1-3 excerpts from a longer document (head, middle, tail). Classify the whole document based on these samples:

- "keep": narrative / technical / instructional content. Users would ask Q&A about it. Emails, manuals, procedures, incident reports, decisions, correspondence.
- "summary_only": structured data worth discovering as a reference but noisy at chunk level. Certificates, forms, lists, dense tables that are mostly cells/labels.
- "skip": no Q&A value. Sensor / telemetry / log dumps, pure numeric data, OCR garbage, repetitive boilerplate.

Return strict JSON: {"classification": "keep"|"summary_only"|"skip", "reason": "one short sentence"}"""


async def llm_classify(
    excerpts: list[str],
    *,
    acompletion_fn: Callable[..., Awaitable[Any]] | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """Ask the LLM to classify — returns (classification, reason).

    ``acompletion_fn`` and ``model`` are injectable for tests.
    """
    if acompletion_fn is None:
        from litellm import acompletion
        from mtss.config import get_settings
        from mtss.llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY

        acompletion_fn = acompletion
        settings = get_settings()
        model = model or settings.get_model(settings.email_cleaner_model)
        extra_body = OPENROUTER_PRIVACY_EXTRA_BODY
    else:
        model = model or "test-model"
        extra_body = None

    user_content = "\n\n---\n\n".join(
        f"Excerpt {i + 1}:\n{ex}" for i, ex in enumerate(excerpts)
    )
    try:
        response = await acompletion_fn(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=200,
            reasoning_effort="minimal",
            drop_params=True,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        raw = response.choices[0].message.content or ""
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not m:
            return CLASSIFICATION_KEEP, f"llm_no_json:{raw[:80]}"
        result = json.loads(m.group())
        cls = result.get("classification")
        reason = (result.get("reason") or "").strip() or "llm_no_reason"
        if cls not in CLASSIFICATION_VALUES:
            return CLASSIFICATION_KEEP, f"llm_bad_classification:{cls}"
        return cls, reason
    except Exception as e:  # noqa: BLE001
        return CLASSIFICATION_KEEP, f"llm_error:{type(e).__name__}:{e}"


@dataclass
class ReportRow:
    doc_id: str
    file_name: str
    document_type: str
    size_chars: int
    classification: str
    source: str
    reason: str
    metrics: dict
    excerpts_head: str = ""

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "size_chars": self.size_chars,
            "classification": self.classification,
            "source": self.source,
            "reason": self.reason,
            "metrics": self.metrics,
            "excerpts_head": self.excerpts_head,
        }


def load_chunks_by_doc(output_dir: Path) -> dict[str, list[str]]:
    """Return document_id -> list of chunk contents, ordered."""
    content_by_doc: dict[str, list[tuple[int, str]]] = {}
    with (output_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            did = c.get("document_id")
            if not did:
                continue
            content_by_doc.setdefault(did, []).append(
                (c.get("chunk_index") or 0, c.get("content") or "")
            )
    return {
        did: [content for _, content in sorted(pairs, key=lambda p: p[0])]
        for did, pairs in content_by_doc.items()
    }


def select_targets(
    output_dir: Path, min_chars: int, min_word_count: int = 50
) -> list[tuple[dict, str]]:
    """Return (doc, concatenated_content) for docs worth analysing.

    Excludes:
      - emails (already curated by the cleaner)
      - image attachments (content is always a single LLM description; not
        the source of bulk-content noise)
      - failed / filtered docs
    """
    chunks_by_doc = load_chunks_by_doc(output_dir)
    results: list[tuple[dict, str]] = []
    with (output_dir / "documents.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") != "completed":
                continue
            doc_type = d.get("document_type") or ""
            if doc_type == "email":
                continue
            if doc_type == "attachment_image":
                continue
            chunks = chunks_by_doc.get(d["id"], [])
            if not chunks:
                continue
            content = "\n\n".join(chunks)
            if len(content) < min_chars:
                continue
            words = len(content.split())
            if words < min_word_count:
                continue
            results.append((d, content))
    return results


async def analyse_one(
    doc: dict,
    content: str,
    *,
    use_llm: bool,
    llm_semaphore: asyncio.Semaphore,
    acompletion_fn: Callable[..., Awaitable[Any]] | None = None,
) -> ReportRow:
    metrics = compute_metrics(content)
    pre_cls, pre_reason = heuristic_classify(metrics)

    if pre_cls is not None:
        return ReportRow(
            doc_id=doc["id"],
            file_name=doc.get("file_name", "?"),
            document_type=doc.get("document_type", "?"),
            size_chars=metrics.char_count,
            classification=pre_cls,
            source=SOURCE_HEURISTIC,
            reason=pre_reason,
            metrics=metrics.as_dict(),
            excerpts_head=content[:240].replace("\n", " "),
        )

    if not use_llm:
        return ReportRow(
            doc_id=doc["id"],
            file_name=doc.get("file_name", "?"),
            document_type=doc.get("document_type", "?"),
            size_chars=metrics.char_count,
            classification=CLASSIFICATION_KEEP,
            source=SOURCE_FALLBACK,
            reason="borderline_no_llm",
            metrics=metrics.as_dict(),
            excerpts_head=content[:240].replace("\n", " "),
        )

    excerpts = sample_excerpts(content)
    async with llm_semaphore:
        cls, reason = await llm_classify(excerpts, acompletion_fn=acompletion_fn)
    return ReportRow(
        doc_id=doc["id"],
        file_name=doc.get("file_name", "?"),
        document_type=doc.get("document_type", "?"),
        size_chars=metrics.char_count,
        classification=cls,
        source=SOURCE_LLM,
        reason=reason,
        metrics=metrics.as_dict(),
        excerpts_head=content[:240].replace("\n", " "),
    )


def write_report(output_dir: Path, rows: list[ReportRow]) -> Path:
    report_path = output_dir / "rag_utility_report.jsonl"
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")
    tmp.replace(report_path)
    return report_path


def print_summary(rows: list[ReportRow]) -> None:
    from collections import Counter

    by_cls = Counter(r.classification for r in rows)
    by_src = Counter(r.source for r in rows)
    print(f"\n  total analysed: {len(rows)}")
    print(f"  keep:         {by_cls[CLASSIFICATION_KEEP]}")
    print(f"  summary_only: {by_cls[CLASSIFICATION_SUMMARY]}")
    print(f"  skip:         {by_cls[CLASSIFICATION_SKIP]}")
    print(f"  by source — heuristic: {by_src[SOURCE_HEURISTIC]}   "
          f"llm: {by_src[SOURCE_LLM]}   fallback: {by_src[SOURCE_FALLBACK]}")
    print("\n  top 10 skip/summary candidates by size:")
    nonkeep = [r for r in rows if r.classification != CLASSIFICATION_KEEP]
    nonkeep.sort(key=lambda r: r.size_chars, reverse=True)
    for r in nonkeep[:10]:
        print(
            f"    [{r.classification}]  {r.size_chars:>8d}  "
            f"{r.file_name}  ({r.source}: {r.reason})"
        )


async def _run(
    output_dir: Path,
    min_chars: int,
    use_llm: bool,
    llm_concurrency: int,
    limit: int | None,
) -> int:
    targets = select_targets(output_dir, min_chars=min_chars)
    if limit is not None:
        targets = targets[:limit]
    print(f"targets: {len(targets)} (min_chars={min_chars}, use_llm={use_llm})")
    if not targets:
        return 0

    sem = asyncio.Semaphore(llm_concurrency)
    rows = await asyncio.gather(
        *(
            analyse_one(doc, content, use_llm=use_llm, llm_semaphore=sem)
            for doc, content in targets
        )
    )
    path = write_report(output_dir, rows)
    print(f"report written: {path}")
    print_summary(rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/output"),
        help="Ingest output dir (default: data/output)",
    )
    parser.add_argument(
        "--min-chars", type=int, default=20000,
        help="Skip docs whose concatenated chunk content is smaller (default: 20000)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Heuristics only; borderline docs default to 'keep' with source=fallback",
    )
    parser.add_argument(
        "--llm-concurrency", type=int, default=5,
        help="Parallel LLM calls for borderline docs (default: 5)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Analyse at most N targets (for pilot runs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return asyncio.run(
        _run(
            args.output_dir,
            args.min_chars,
            not args.no_llm,
            args.llm_concurrency,
            args.limit,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

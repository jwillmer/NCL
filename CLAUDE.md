# CLAUDE.md — MTSS project guide

This file gives future Claude sessions quick orientation for the MTSS codebase. Full docs: `README.md`, `docs/features.md`, `docs/architecture.md`.

## What MTSS does

Email RAG pipeline. Ingests `.eml` files (+ attachments: PDF, Office docs, images, ZIPs) into a local SQLite store (`data/output/ingest.db`), optionally pushes to Supabase. Tiered parser chain: local parsers → Gemini 2.5 Flash via OpenRouter for complex PDFs and modern non-local formats → LlamaParse only for legacy binary `.doc`/`.xls`/`.ppt`. Vision API for images, LiteLLM for embeddings/context/topics.

## Output layout (`data/output/`)

- `ingest.db` — single SQLite file (WAL, `PRAGMA foreign_keys=ON`, `busy_timeout=30000`, `cache_size=-64000` (~250 MB), `mmap_size=256 MB`, `journal_size_limit=64 MB`; the progress-tracker connection uses 1/8 of cache+mmap). Override via `MTSS_SQLITE_CACHE_SIZE_KIB` / `MTSS_SQLITE_MMAP_SIZE_BYTES` / `MTSS_SQLITE_JOURNAL_SIZE_LIMIT_BYTES`. Tables: `documents`, `chunks`, `topics`, `chunk_topics` (M:N join + single source of truth for chunk↔topic links), `ingest_events`, `processing_log`, `run_history`, `manifest`. ACID transactions + `UNIQUE(chunk_id)` + FK CASCADE structurally prevent the orphan-chunk / dupe-chunk classes that the pre-SQLite JSONL flushes leaked under Windows file locks.
- `archive/<folder_id>/` — human-readable markdown snapshots + original attachments. `folder_id` is a 32-char hash from `compute_folder_id(doc_id)` in `src/mtss/utils.py` (distinct keyspace from `doc_id` so orphan-cleanup/URI tooling can't conflate them).
- Legacy JSONLs (`documents.jsonl` / `chunks.jsonl` / `topics.jsonl` / `ingest_events.jsonl` / `processing_log.jsonl` / `run_history.jsonl` / `manifest.json`) are no longer written; if present they're residue from a pre-SQLite install. Migrate with `uv run python scripts/migrate_to_sqlite.py --apply` (dry-run by default).

## Ingest-side CLI (`src/mtss/cli/ingest_cmd.py`)

| Command | Purpose |
|---|---|
| `mtss ingest` | Main ingest. `--retry-failed` forces re-parse of FAILED entries. `--reprocess-outdated` re-parses below current ingest version. `--lenient` continues past errors. |
| `mtss estimate` | Cost estimate before running ingest. |
| `mtss import` | Push local output to Supabase. |
| `mtss failures` / `mtss reset-stale` / `mtss reset-failures` | Report + recover from crashed / failed runs. |

## Validate (`src/mtss/cli/validate_cmd.py`)

`mtss validate ingest` — runs ~20 integrity checks against `data/output/`. Returns issues (must-fix) + warnings (informational). Covers duplicate doc_ids/chunk_ids, missing embeddings, missing context_summary/embedding_text, orphan chunks, broken archive URIs, broken markdown links, stale topic counts, etc. Broken-link warnings attribute each folder to its source email via `build_folder_to_email_map(docs)`.

## Maintenance CLI (`src/mtss/cli/maintenance_cmd.py`)

Targeted repair commands — preferred over full reingest (API costs).

| Command | What it does | API cost |
|---|---|---|
| `mtss mark-failed <file.eml>...` | Set entries in the `processing_log` table to `FAILED`. Paired with `mtss ingest --retry-failed` to re-process just those emails (triggers `force_reparse` → cleans existing docs/chunks via FK CASCADE first). Matches by exact path or basename. | None (the retry does). |
| `mtss clean-archive-md` | Walk every `archive/**/*.md`, reapply `strip_llamaparse_image_refs()`. Idempotent. `--dry-run` available. Use after bumping the stripping regex to retroactively clean older archives. | Zero. |
| `mtss ingest-update` | Validate + auto-repair (orphaned docs, missing archives, missing context). | LLM only where needed. |
| `mtss reprocess` | Re-ingest docs below a target ingest version. | Full re-parse. |
| `mtss reindex-chunks` | Re-chunk from archived markdown (adds line numbers, regenerates context). | LLM for context. |
| `mtss re-embed` | Re-classify embedding mode + re-chunk + re-embed against archived markdown. No re-parse. `--dry-run`, `--limit`, `--mode`, `--force`. | Embeddings + (LLM triage on the medium-confidence band only). |
| `mtss backup` | Snapshot `ingest.db` (via `VACUUM INTO`, safe during reads) + zip `archive/` into `data/backup/<YYYYMMDD-HHMMSS>/`. DB alone is useless without archive/, so both are captured together. `--dry-run`, `--dest`, `--compression-level`. | None. |

## Parsers (`src/mtss/parsers/`)

- **Tiered routing** in `attachment_processor.py:_get_tiered_parser`:
  - `.pdf` simple (PyMuPDF text-layer + no form fields) → PyMuPDF4LLM (free, local)
  - `.pdf` complex *or* PyMuPDF4LLM returns empty → `GeminiPDFParser` (Gemini 2.5 Flash via OpenRouter, ~$0.0025/page)
  - `.docx`, `.xlsx`, `.csv`, `.html`, `.htm` → local parsers (free)
  - Legacy binary `.doc`, `.xls`, `.ppt` → `LlamaParseParser` (only remaining LlamaParse use)
  - Anything else without a local parser → Gemini if available, else unsupported
- **`GeminiPDFParser`** (`parsers/gemini_pdf_parser.py`) — uploads PDFs as base64 `type:"file"` content blocks via LiteLLM `acompletion`. Page-range pagination (default 25 pages/batch) with adaptive halving on `finish_reason="length"`; `asyncio.Semaphore(max_concurrent_gemini_pdf)` caps parallel calls per doc. `gemini_pdf_hard_page_ceiling` (default 200) caps parser-triggered spend; above that the caller falls back to `SUMMARY` mode.
- **`EmptyContentError`** (`parsers/base.py`) — raised by local parsers when they open a file but extract zero text (e.g. image-only docx). `process_attachment` catches it and falls back to Gemini (if available) before giving up. Fallback does not trigger for generic `ValueError` (corrupt file).
- `strip_llamaparse_image_refs(text)` (`parsers/llamaparse_parser.py`) — shared helper. Strips `<img>` HTML tags, `![alt](page_N_image_N.jpg)`, and bare `![alt](image)` — preserves alt-text. Used by the live parser, `clean-archive-md`, and the in-memory pre-clean inside `mtss re-embed`.

## Embedding modes (`src/mtss/ingest/embedding_decider.py`)

Each attachment is classified at ingest time into one of three modes — stamped on the `Document` and inherited by every `Chunk`:

| Mode | Meaning | Typical content | Cost |
|---|---|---|---|
| `full` | Standard chunk + context + embed | Prose, contracts, reports | Default — every chunk embedded |
| `summary` | One synthesized summary chunk | Sensor-log dumps, repetitive tabular | One LLM summary + one embedding |
| `metadata_only` | One filename-only stub chunk | Empty/noise PDFs | Trivial, no LLM |

Decision tree (deterministic rules first; LLM triage *always* runs on the medium-confidence band — not flag-gated):
1. <50 tokens, *or* `prose_ratio < 0.15 && heading_count == 0` → `metadata_only`
2. `>20K tokens` *and* (`digit_ratio > 0.40` *or* `table_char_pct > 0.50` *or* `repetition_score > 0.92` *or* `short_line_ratio > 0.95`) → `summary`
3. `>50K tokens && prose_ratio < 0.50` → LLM triage (A/B/C → full/summary/metadata_only); on LLM failure or unparseable response → `summary` with `confidence=0.3`
4. Default → `full`

LLM triage runs on the first 2000 chars of the markdown (hardcoded in `embedding_decider.py`). The preview is wrapped in `<document>…</document>` tags with an explicit anti-injection preamble; the response parser consumes only the first letter, so a crafted attachment cannot steer the classification by embedding instructions in its content.

The `repetition_score` and `short_line_ratio` thresholds were tuned 2026-04-19 from real-corpus dry-run data — earlier values (0.60 / 0.70) demoted real prose reports whose section headers/footers repeated across pages. Sensor logs (the intended target) are caught by `digit_ratio` and `table_char_pct`, not by repetition.

Email bodies skip the decider — they're always `full`. All thresholds are Pydantic settings with `DECIDER_*` env aliases.

`mtss re-embed` re-runs the decider + chunker + embedder against existing archive markdown (no re-parse). It is the successor to the deleted `scripts/repair_failed_llamaparse_attachments.py`.

## Ingest-time pipeline gotchas

- `_process_non_zip_attachment` and `_process_zip_member` (both in `src/mtss/ingest/attachment_handler.py`) route through shared helpers (`_decide_and_build_chunks` for context + chunking, `_write_attachment_archive_md` for the .md write + URI stamps, `_apply_vessel_metadata_to_chunks` for vessel fields). Don't inline these steps in a new branch — the helpers exist specifically to prevent the ZIP path from drifting (a recurring historical bug).
- LLM-backed ingest components (`EmbeddingGenerator`, `ContextGenerator`, `ImageProcessor`, `TopicExtractor`, every `BaseParser` subclass) expose `.model_name: str | None` for `ProcessingTrail` stamp sites. Use `component.model_name` when stamping the trail — don't reach for `.model` / `.llm_model` / `settings.get_model(...)` directly.
- `SqliteStorageClient` (`src/mtss/storage/sqlite_client.py`) is the canonical ingest writer. Every insert is a single transactional SQL write — no prior-vs-current in-memory split, no JSONL flush merge, no atomic-rewrite of a monolithic file. `insert_chunks` uses `INSERT OR REPLACE ON chunk_id`; dedup is enforced by `UNIQUE(chunk_id)` rather than a flush-time filter.
- `SqliteProgressTracker.get_pending_files` uses file hash (not path) to decide pending vs completed — same email re-sent gets re-ingested only if its content hash changes.

## Tests

- `uv run pytest` — full suite (~750 tests, ~10s).
- Parser/strip tests live in `tests/test_sanitize_migration.py` (`TestLlamaParseImageStripping`, `TestValidateNewChecks`).
- Attachment/fallback tests in `tests/test_attachment_processor.py` (`TestGeminiFallback`, `TestComplexPdfRoutesToGemini`, `TestLegacyOfficeRoutesToLlamaParse`, `TestLocalParserEmptyContentError`).
- Embedding-mode tests in `tests/test_embedding_decider.py` (decision-tree branches + LLM triage), `tests/test_chunk_strategies.py`, `tests/test_reembed_cmd.py`.
- Gemini parser tests in `tests/test_gemini_pdf_parser.py` (availability, single-call, paginated, adaptive halving, base64 payload structure).
- Storage + maintenance command tests in `tests/test_ingest_storage.py` (`TestMarkFailedCommand`, `TestCleanArchiveMdCommand`) and `tests/test_sqlite_client.py` (schema/constraint coverage).
- Ingest flow tests in `tests/test_ingest_processing.py` (`TestZipAttachmentContextGeneration`, `TestThreadDigest`, `TestEmbeddingModeStamping`).

## Workflow: fixing a data-integrity issue surfaced by validate

1. `mtss validate ingest` — identify affected emails (warnings now list folder + source_id).
2. Decide repair path:
   - **In-place** (preferred): if the fix is a regex/metadata tweak, use `clean-archive-md`, `ingest-update`, or a small script. Zero API cost.
   - **Targeted re-ingest**: if data is genuinely missing (empty content, broken parse), `mtss mark-failed <eml>...` then `mtss ingest --retry-failed`. Only those emails re-parse.
   - **Full reprocess**: last resort. Expensive at production scale (~50 GB ingest, ~100 GB DB).
3. Re-run `mtss validate ingest` to confirm clean.

## User preferences (see also `.claude/.../memory/`)

- Always ask before starting an ingest run.
- Prefer targeted repair over full reingest.
- **Treat `./data/` as irreplaceable production state.** It is not in git and has no backups; regeneration costs real API money and hours of wall-clock time.
  - No modifications — delete, move, rename, in-place edit — without explicit confirmation of the specific change.
  - **Small issues** (one or a few emails): `mtss mark-failed <file.eml>...` then `mtss ingest --retry-failed`, or an existing idempotent maintenance command.
  - **Large / structural changes**: write a dedicated migration script under `scripts/`, test it (dry-run, unit tests, or a copied subset) before running against `./data/`, confirm with the user before executing, and fix the root cause in the pipeline with a regression unit test so the drift can't reappear.

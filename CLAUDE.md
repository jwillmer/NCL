# CLAUDE.md — MTSS project guide

This file gives future Claude sessions quick orientation for the MTSS codebase. Full docs: `README.md`, `docs/features.md`, `docs/architecture.md`.

## What MTSS does

Email RAG pipeline. Ingests `.eml` files (+ attachments: PDF, Office docs, images, ZIPs) into local JSONL output (`data/output/`), optionally pushes to Supabase. Uses LlamaParse + local parsers for text extraction, vision API for images, LiteLLM for embeddings/context/topics.

## Output layout (`data/output/`)

- `documents.jsonl` — one row per email + per attachment
- `chunks.jsonl` — all text chunks with embeddings
- `topics.jsonl` — extracted topics with chunk_count/document_count
- `ingest_events.jsonl` — skip reasons, extraction failures, non-content classifications
- `processing_log.jsonl` — per-file progress (PROCESSING / COMPLETED / FAILED)
- `archive/<doc_id[:16]>/` — human-readable markdown snapshots + original attachments
- `manifest.json` + `run_history.jsonl` — run metadata

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
| `mtss mark-failed <file.eml>...` | Set entries in `processing_log.jsonl` to `FAILED`. Paired with `mtss ingest --retry-failed` to re-process just those emails (triggers `force_reparse` → cleans existing docs/chunks first). Matches by exact path or basename. | None (the retry does). |
| `mtss clean-archive-md` | Walk every `archive/**/*.md`, reapply `strip_llamaparse_image_refs()`. Idempotent. `--dry-run` available. Use after bumping the stripping regex to retroactively clean older archives. | Zero. |
| `mtss ingest-update` | Validate + auto-repair (orphaned docs, missing archives, missing context). | LLM only where needed. |
| `mtss reprocess` | Re-ingest docs below a target ingest version. | Full re-parse. |
| `mtss reindex-chunks` | Re-chunk from archived markdown (adds line numbers, regenerates context). | LLM for context. |

## Parsers (`src/mtss/parsers/`)

- Tiered routing in `attachment_processor.py:_get_tiered_parser`: local parsers preferred for `.pdf`/`.docx`/`.xlsx`/`.csv`/`.html`; LlamaParse via registry for everything else.
- **`EmptyContentError`** (`parsers/base.py`) — raised by local parsers when they open a file but extract zero text (e.g. image-only docx). `process_attachment` catches it and falls back to `LlamaParseParser()` if `is_available`. Fallback does not trigger for generic `ValueError` (corrupt file) or when primary is already LlamaParse.
- `strip_llamaparse_image_refs(text)` (`parsers/llamaparse_parser.py`) — shared helper. Strips `<img>` HTML tags, `![alt](page_N_image_N.jpg)`, and bare `![alt](image)` — preserves alt-text. Both the live parser and `clean-archive-md` call this single function.

## Ingest-time pipeline gotchas

- `process_attachment` and `process_zip_attachment` (both in `src/mtss/ingest/attachment_handler.py`) must stay in sync: both call `context_generator.generate_context` + set `archive_browse_uri/download_uri`. The ZIP path historically forgot both — keep them mirrored.
- `LocalClient.flush()` (`src/mtss/storage/local_client.py`) dedupes chunks by `chunk_id` for both prior-run AND current-run writes. Relies on the deterministic chunk_id (`doc_id + char_start + char_end`) from `compute_chunk_id`.
- `LocalProgressTracker.get_pending_files` uses file hash (not path) to decide pending vs completed — same email re-sent gets re-ingested only if its content hash changes.

## Tests

- `uv run pytest` — full suite (490+ tests, ~6s).
- Parser/strip tests live in `tests/test_sanitize_migration.py` (`TestLlamaParseImageStripping`, `TestValidateNewChecks`).
- Attachment/fallback tests in `tests/test_attachment_processor.py` (`TestLlamaParseFallback`, `TestLocalParserEmptyContentError`).
- Storage + maintenance command tests in `tests/test_ingest_storage.py` (`TestLocalClientFlushChunkDedup`, `TestMarkFailedCommand`, `TestCleanArchiveMdCommand`).
- Ingest flow tests in `tests/test_ingest_processing.py` (`TestZipAttachmentContextGeneration`, `TestThreadDigest`).

## Workflow: fixing a data-integrity issue surfaced by validate

1. `mtss validate ingest` — identify affected emails (warnings now list folder + source_id).
2. Decide repair path:
   - **In-place** (preferred): if the fix is a regex/metadata tweak, use `clean-archive-md`, `ingest-update`, or a small script. Zero API cost.
   - **Targeted re-ingest**: if data is genuinely missing (empty content, broken parse), `mtss mark-failed <eml>...` then `mtss ingest --retry-failed`. Only those emails re-parse.
   - **Full reprocess**: last resort. Expensive at production scale (~50 GB ingest, ~100 GB DB).
3. Re-run `mtss validate ingest` to confirm clean.

## User preferences (see also `.claude/.../memory/`)

- Always ask before starting an ingest run.
- Never remove anything from `./data/` without explicit confirmation — reprocessing costs time and money.
- Prefer targeted repair over full reingest.

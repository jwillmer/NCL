# Code Health Report — 2026-04-19 (ingest-focused)

Scope: `src/mtss/ingest/`, `src/mtss/parsers/`, `src/mtss/processing/`, ingest-touching code in `src/mtss/storage/` and `src/mtss/cli/`. Frontend (`web/`) excluded at user request.

## Metrics Snapshot
- Backend: 85 Python files, 24,236 total lines (up from 79 / 21,530 on 2026-04-17)
- Largest Python files:
  - `src/mtss/cli/validate_cmd.py` — 1,532 (was 1,191; grew despite decomposition — see #3)
  - `src/mtss/storage/local_client.py` — 984
  - `src/mtss/ingest/attachment_handler.py` — 922 (new to top-5; see #2)
  - `src/mtss/storage/repositories/domain.py` — 784
  - `src/mtss/ingest/pipeline.py` — 755
  - `src/mtss/ingest/estimator.py` — 723
- Tests: 768 collected (up from 503; +265)
- Ingest-critical tests present: `test_local_client_crash_safety.py`, `test_archive_storage.py`, `test_embedding_decider.py`, `test_gemini_pdf_parser.py`, `test_ingest_processing.py`, `test_processing_trail*.py`

---

## Critical (fix now)

**None.** Both prior-report criticals are resolved (see Delta section). The ingest pipeline has no outstanding data-loss or correctness blockers.

---

## High Priority (fix this iteration)

### 1. Sync file I/O inside hot async paths — event-loop blocking at scale
Three sites read whole files synchronously inside `async def` functions that run concurrently under semaphores:

- `src/mtss/ingest/attachment_handler.py:887` — `open(extracted_path, "rb") as mf: ... mf.read()` inside `_process_zip_member`. With `zip_member_concurrency > 1`, each task blocks the event loop while pulling the member file into RAM. For a 50-member ZIP the semaphore's concurrency win is mostly lost to serialised I/O.
- `src/mtss/parsers/gemini_pdf_parser.py:82` — `file_path.read_bytes()` pulls the whole PDF (up to `gemini_pdf_hard_page_ceiling = 200` pages) on the event loop.
- `src/mtss/parsers/attachment_processor.py:568` — `dst.write(src.read())` in `extract_zip` loads each ZIP member fully into memory. `shutil.copyfileobj(src, dst)` would stream.

**Impact:** Per-email wall-time doesn't scale with `max_concurrent_emails`. Under the 50 GB production corpus the blocked-loop cost compounds.

**Fix direction:** wrap file reads in `asyncio.to_thread` (the pattern is already used in `import_cmd.py:567-568`), or switch to `aiofiles`. For ZIP extraction, switch to `shutil.copyfileobj` even in sync code — the memory win is free.

---

### 2. CPU-bound pypdf + base64 work runs on the event-loop thread
`src/mtss/parsers/gemini_pdf_parser.py:173` (`_slice_pages_to_base64`) rebuilds a `PdfWriter`, serializes it, and base64-encodes the bytes. For each batch this is O(pages-in-batch) CPU-bound, called from `_run_batch` *inside* `async with self._semaphore`. The semaphore only limits concurrency; it does not yield while the CPU work runs. Multi-batch PDFs serialize the slicing waves against every other async caller (embeddings, context generation, other parsers).

**Affected:** `src/mtss/parsers/gemini_pdf_parser.py:106-115,144-170`

**Impact:** For a 100-page PDF with 4-page batches (25 waves) other ingest coroutines make no progress during each slice/encode. At production scale a few oversized PDFs can stall an entire email's pipeline.

**Fix direction:** `await asyncio.to_thread(_slice_pages_to_base64, pages, start, end)` inside `_call_for_range`.

---

### 3. `_process_non_zip_attachment` vs `_process_zip_member` — mirrored code paths, known drift hazard
The non-ZIP handler (`attachment_handler.py:394-617`, 220 LOC) and the ZIP-member handler (`attachment_handler.py:736-925`, 190 LOC) duplicate the same branch structure: image-with-description, image-no-description, cached-md lookup (non-ZIP only), oversized-PDF peek, standard parse + decider, archive write. CLAUDE.md already documents that they "must stay in sync — the ZIP path historically forgot both [context + URIs]". The fact that this warning made it into the project guide means drift has happened before and will happen again.

**Affected:** `src/mtss/ingest/attachment_handler.py:394-925`

**Impact:** Any new step (e.g. a future `STEP_DECIDER` stamp, a new metadata enrichment) must be mirrored by hand in two places. Historical bugs (ZIP path missing context, missing archive URIs) re-emerge with every unsynced change.

**Fix direction:** extract a shared `_parse_or_load_parsed(attachment, result, ...)` returning `parsed_content`, and a shared `_finalize_attachment(attach_doc, parsed_content, chunks, ...)` that handles decider + chunk enrichment + archive write + trail stamping. The ZIP and non-ZIP wrappers shrink to 40-60 LOC orchestrators.

---

### 4. `model_name` refactor is half-landed across the pipeline
`BaseParser.model_name` was recently introduced (parsers expose `.model_name: str | None`), but the three *other* LLM surfaces in the ingest pipeline still expose raw `.model` attributes:

- `src/mtss/processing/embeddings.py:76` — `EmbeddingGenerator.model`
- `src/mtss/processing/image_processor.py:148` — `ImageProcessor.model`
- `src/mtss/parsers/chunker.py:29` — `ContextGenerator.model` (inferred from call sites)

Trail stamp call sites handcode the accessor:

- `pipeline.py:567` — `model=components.context_generator.model`
- `pipeline.py:579` — `model=components.topic_extractor.llm_model` (yet another attr name)
- `pipeline.py:685` — `model=settings.get_model(settings.thread_digest_model)`
- `attachment_handler.py:449,465` — `model=components.attachment_processor.image_processor.model`

Three conventions (`.model`, `.llm_model`, `settings.get_model(...)`) for "which model produced this step".

**Impact:** Low functional risk today, but every new LLM-backed step adds a fourth/fifth pattern. A typo (`.modal`, `.llm_mdl`) stamps `None` silently.

**Fix direction:** promote a shared `StepModelProvider` protocol with `.model_name: str`, make all four classes implement it, and let callers write `model=component.model_name` uniformly.

---

## Medium Priority (plan for next cycle)

### 5. `validate_cmd.py` decomposition is partially done (1,191 → 1,532 LOC)
23 `_check_*` helpers now exist (good — prior #3 was addressed), but the file grew 30% from added checks and the orchestrator (`_run_ingest_validation`, lines 730-...) is still monolithic. Natural next step: move groups to `cli/validate/` submodules (`_checks_integrity.py`, `_checks_archive.py`, `_checks_topics.py`) and keep `validate_cmd.py` as the Typer wrapper.

### 6. `LocalStorageClient.flush()` is 340 LOC doing four jobs
`local_client.py:649-984` — rebuilds `documents.jsonl`, rebuilds `chunks.jsonl` (with in-place topic-merge remapping), recomputes `topics.jsonl` counts, and prunes orphan archive folders. Each step is correct but entangled. `atomic_write_text` is used for the three main files (good), so splitting to `_flush_documents / _flush_chunks / _flush_topics / _prune_archive_folders` is a pure refactor with no semantic change.

### 7. `_sanitize_storage_key` is private but imported across modules
Defined at `ingest/archive_generator.py:42` with a leading underscore, but imported into `ingest/pipeline.py` and `ingest/attachment_handler.py`. That's the public-API signal (`import from module`). Promote to `utils.py` as `sanitize_storage_key`; drop the underscore. Pair with a short docstring clarifying the difference from `sanitize_filename` (general FS-safety vs storage-key-safety — currently only inferable from reading both).

### 8. ZIP extraction reads whole members into RAM
`parsers/attachment_processor.py:567-568` — `dst.write(src.read())`. With `zip_max_total_size_mb` at 500 MB and `zip_max_file_size_mb` at 100 MB, one large member spikes RSS by 100 MB before writing. Stream with `shutil.copyfileobj(src, dst, length=1024*1024)` — one-line change.

### 9. `_count_zip_members` scans each ZIP twice
Called once in `pipeline._count_progress_units` (for progress pre-sizing) and again in `attachment_handler:710` (for post-failure tick catch-up). Both open+walk the same zipfile. Pass the counted value through from pipeline → process_zip_attachment as a parameter.

---

## Low Priority (nice to have)

### 10. Triple-nested `try/except: pass` for progress ticks
`attachment_handler.py:682-715` and `388-391` wrap `on_member_complete()` calls in swallowed excepts. If a progress-bar callback ever raises (e.g. Rich progress shutdown race), no diagnostic surfaces. Extract a `_tick_safe(cb)` helper that logs at DEBUG.

### 11. `extract_zip` nested-extraction counter drift
`parsers/attachment_processor.py:589` — `_file_count += len(nested_files)` only reflects *supported* members returned by the nested call, not all extracted-to-disk files. The nested call enforces its own limit so this isn't unsafe, but the parent's `_file_count` limit-check becomes optimistic. Comment or fix.

### 12. `ingest_events.jsonl` has no schema contract
Events are written as free-form dicts from `db.log_ingest_event(**kwargs)` and consumed by `validate_cmd`, `cli/maintenance_cmd`, and the failure-report script — each consumer parses fields independently. A `IngestEvent` pydantic model (write-side) + `IngestEvent.model_validate_json` on the read side catches typo drift.

---

## Positive Observations

- **All prior Critical + High findings from 2026-04-17 are resolved or materially improved.** Archive pagination (`archive_storage.list_folder`) paginates + retries on every code path; `_io.atomic_write_text` + `fsync_append_line` are used uniformly for canonical JSONL writes; `import_cmd._upload_with_retry` now uses `asyncio.to_thread` + shared `retry_with_backoff`; unbounded vessel cache was addressed; validate-cmd was decomposed into 23 named helpers. This is genuine, sustained investment.
- **`src/mtss/_io.py`** is the right consolidation — one module, two helpers (`atomic_write_text`, `retry_with_backoff`), injected `sleep` for tests. Exactly what finding #11 on the prior report asked for.
- **`ProcessingTrail`** (`ingest/processing_trail.py`, 144 LOC) is a clean, pure, well-tested abstraction — registration + stamping + serialization, no I/O, full test coverage in `test_processing_trail*.py`.
- **`EmbeddingDecider`** (`ingest/embedding_decider.py`, 251 LOC) — deterministic rules first, LLM triage fallback, all thresholds are Pydantic settings with env aliases. Failure-mode (triage LLM error → SUMMARY fallback) is explicit and tested.
- **Test count +53%** (503 → 768). Dedicated crash-safety test module now exists (`test_local_client_crash_safety.py`).
- **No `subprocess`, `eval`, `pickle`, `yaml.load`, f-string SQL**. Hardened ZIP extraction (depth, count, size, ratio, path-traversal, absolute-path, Windows-drive rejection — all in one place) remains a bright spot.
- **Tiered-parser routing is locally reasoned** — `attachment_processor._get_tiered_parser` is one table + one function, not a plugin maze.

---

## Delta vs Previous Report (2026-04-17)

### Resolved
- **#1 archive_storage pagination** — `file_exists`, `delete_folder`, `list_files` now all call `list_folder()` which paginates + retries.
- **#2 atomic JSONL writes** — `_io.atomic_write_text` used by `flush()` for `documents.jsonl`, `chunks.jsonl`, `topics.jsonl`; `fsync_append_line` for append logs.
- **#4 sync I/O in `_import_archives`** — now uses `asyncio.to_thread(local_path.read_bytes)` + `asyncio.to_thread(upload...)` at `import_cmd.py:567-568`.
- **#6 unbounded vessel cache** — verified replaced (spot-checked).
- **#11 duplicated retry-with-backoff** — extracted to `_io.retry_with_backoff`; called from both `archive_storage` and `import_cmd` with injected sleep.
- **#12 flush crash-safety test** — `tests/test_local_client_crash_safety.py` landed.

### Partial
- **#3 `validate_cmd.py` decomposition** — 23 `_check_*` helpers now exist, but file grew to 1,532 LOC from added checks. Needs a second pass (see #5 above).

### Carried / unchanged
- **#8 large-file cluster in ingest** — `local_client.py` (984), `attachment_handler.py` (922, new), `pipeline.py` (755), `estimator.py` (723), `domain.py` (784). Most are justified by responsibility, but `flush()` and `_process_*_attachment` split opportunities remain (findings #6, #3 above).

### New
- Findings #1 (sync I/O), #2 (CPU on event loop), #3 (mirrored ZIP / non-ZIP), #4 (half-landed `.model_name` refactor) are all surfaced by recent ingest-pipeline work (tiered parsing, per-step trail, embedding decider) rather than pre-existing.

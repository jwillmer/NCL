# Code Health Report — 2026-04-17

## Metrics Snapshot
- Backend: 79 Python files, 21,530 total lines
- Frontend: 16 TS/TSX files, 2,649 total lines
- Largest Python files:
  - `src/mtss/cli/validate_cmd.py` — 1,191
  - `src/mtss/storage/local_client.py` — 946
  - `src/mtss/storage/repositories/domain.py` — 784
  - `src/mtss/ingest/estimator.py` — 752
  - `src/mtss/storage/repositories/documents.py` — 719
- Largest frontend files:
  - `web/src/components/Sources.tsx` — 692
  - `web/src/pages/ChatPage.tsx` — 535
- Tests: 503 collected (pytest)
- Frontend build: passes (warns: single chunk 1.15 MB / 337 KB gzip, no code-splitting)
- Dependency audit (npm, prod only): 0 vulnerabilities
- No prior `code-health-report.md` found — this is the baseline

---

## Critical (fix now)

### 1. Archive storage pagination inconsistency — latent data loss at scale
**Overview:** `ArchiveStorage.list_folder()` was hardened with pagination + retry (tests added in `tests/test_archive_storage.py`), but three sibling methods still call `self.bucket.list(folder)` directly, which silently caps results at the storage3 default of 100 items.

**Affected files:**
- `src/mtss/storage/archive_storage.py:180` (`file_exists`)
- `src/mtss/storage/archive_storage.py:201,213` (`delete_folder`)
- `src/mtss/storage/archive_storage.py:243,248` (`list_files`)

**Impact:** At production scale (~50 GB ingest, zip-heavy emails with many extracted attachments):
- `file_exists` returns false-negatives for folders >100 files → phantom re-uploads during import.
- `delete_folder` leaves orphan files in the bucket undeleted (silent) when a doc has >100 attachments.
- `list_files` under-reports what exists.
The new retry/pagination work is half-landed; finish the migration.

---

### 2. No atomic writes or fsync for canonical JSONL output
**Overview:** `LocalStorageClient.flush()` rewrites `documents.jsonl`, `chunks.jsonl`, and `topics.jsonl` via plain `open(path, "w")` with no fsync and no temp-file-plus-rename. `_append_jsonl()` similarly does not fsync. If the process crashes or power fails mid-flush, these files can be truncated. `local_progress_tracker.py` *does* fsync — so the pattern is known internally, just inconsistent.

**Affected files:**
- `src/mtss/storage/local_client.py:107` (`_append_jsonl`)
- `src/mtss/storage/local_client.py:619` (`flush`, ~150 LOC rewriting the three primary files)

**Impact:** CLAUDE.md explicitly flags `./data/` as costly to regenerate (~50 GB ingest, ~100 GB DB). A crashed flush can corrupt the source-of-truth JSONL files and force a paid re-ingest. Atomic-rename pattern plus `os.fsync` for the flush targets is cheap insurance.

---

## High Priority (fix this iteration)

### 3. `validate_cmd.py` — 22 checks in one monolithic function
**Overview:** `_run_ingest_validation` bundles all 22 ingest-integrity checks inline in a single function; the file is 1,191 lines and most of the logic is one function. Adding or tweaking a check requires scrolling a monster.

**Affected files:** `src/mtss/cli/validate_cmd.py:121`

**Impact:** Maintenance friction on the most load-bearing integrity tool. Each check is independent and pure over `docs/chunks/topics/events/archive_dir` — they extract cleanly into named functions returning `(issues, warnings)` pairs. Also unlocks per-check unit tests (currently only the LlamaParse strip / new-check pair has targeted tests).

---

### 4. Sync I/O inside `async def` (event loop blocking)
**Overview:**
- `_import_archives` is `async` but calls `_upload_with_retry` which is sync and uses `time.sleep(delay)` for backoff.
- Same function reads file payloads via `local_path.read_bytes()` (sync).
- Uploads run serially.

**Affected files:**
- `src/mtss/cli/import_cmd.py:416` (sync `_upload_with_retry` + `time.sleep`)
- `src/mtss/cli/import_cmd.py:443,541-548` (async `_import_archives` calling it synchronously)

**Impact:** The event loop is blocked for every archive upload, so no concurrency is achievable. At ~100 GB DB / many-thousands-of-files scale this dominates import wall-time. Either make the path fully sync (it's a CLI, not a server) and drop the `async` facade, or convert to `anyio.to_thread.run_sync` / `asyncio.gather` with a semaphore.

---

### 5. Missing request size bounds on `/api/agent`
**Overview:** `AgentRequest` / `Message` / `MessagePart` use only `min_length=1`, no `max_length`. An authenticated client can post arbitrarily large `parts[].text`, which flows directly to the LLM. `conversations.py` and `feedback.py` *do* bound their string fields (title 200, content 500, message_id 100) — the agent streaming route is the outlier.

**Affected files:** `src/mtss/api/streaming.py:51-74`

**Impact:** Burn LLM tokens (cost) and slow checkpointer writes. `search_node` already truncates the *tool argument* to 2000 chars (`agent.py:369`), but the raw message reaches the LLM via `HumanMessage(content=msg.content)` unbounded. Add explicit `max_length` on `MessagePart.text`, `Message.parts`, `AgentRequest.messages`, and `thread_id` (UUID shape).

---

### 6. Unbounded module-level vessel cache
**Overview:** `_vessel_cache: Dict[str, Optional[Vessel]]` at module scope grows with every unique vessel_id ever seen; never evicted.

**Affected files:** `src/mtss/api/agent.py:159`

**Impact:** Slow memory growth in long-running API processes. Cardinality is probably small today, but any caller-controlled id (the vessel_id comes from frontend state) + no eviction = a latent leak. Replace with `functools.lru_cache` or TTL cache.

---

### 7. SPA auth bypass list drifts from React Router
**Overview:** `AuthMiddleware.dispatch` hardcodes `/chat`, `/chat/`, `/conversations`, `/conversations/` as auth-exempt (main.py:94). `SPAStaticFiles` already falls back to `index.html` for unknown paths — so the duplication is both brittle (new routes need manual adding) and arguably wrong (any new page route 401s until listed).

**Affected files:** `src/mtss/api/main.py:90-99`

**Impact:** Correctness regression risk whenever the frontend adds a route. The SPA shell is public anyway (config.js, index.html both served unauthenticated). A single rule like "non-API unknown paths → index.html, no auth" would remove the drift surface.

---

## Medium Priority (plan for next cycle)

### 8. Large-file cluster in storage/ingest
Six files >600 LOC: `local_client.py` (946), `repositories/domain.py` (784), `estimator.py` (752), `repositories/documents.py` (719), `maintenance_cmd.py` (681), `pipeline.py` (659). None are obviously "doing two jobs", but `local_client.py`'s `flush()` (150 LOC) and `repositories/domain.py` (topic + vessel + chunk updates mingled) are natural split points.

### 9. Frontend monoliths
`web/src/components/Sources.tsx` (692) and `web/src/pages/ChatPage.tsx` (535) each mix data-fetching, local state, memoized transforms, and render. Pattern is consistent — not broken — but the next feature-add here is likely to push one over a thousand lines. Candidate split: extract `useConversation`, `useFilters`, `useChatHistory` hooks out of ChatPage; split `CitationProvider` / `SourcesAccordion` / `SourceViewDialog` out of Sources.

### 10. Frontend bundle: no code-splitting
`vite build` emits a single 1,149 KB (gzip 337 KB) `index-*.js`. Vite itself flags this. Low-effort wins with `manualChunks` or route-level `React.lazy` (especially for `ConversationsPage`).

### 11. Duplicated retry-with-backoff loops
`ArchiveStorage.list_folder` (archive_storage.py:282-302) and `_upload_with_retry` (import_cmd.py:424-440) implement structurally identical exponential-backoff loops. Extract to a shared `retry_with_backoff` helper with an injectable sleep, so tests don't need `patch.object(module, "time")` per site.

### 12. Flush crash-safety lacks a regression test
Tied to finding #2. `TestLocalClientFlushChunkDedup` covers dedup but not the power-failure / partial-write scenario. If atomic writes are added, pair with a test that truncates the temp file mid-write and asserts the canonical file still reflects the previous successful flush.

### 13. `_init_api_keys()` swallows all errors at import time
`src/mtss/__init__.py:38` — bare `except Exception: pass` during module-load hides real config errors behind silent failures that surface later as confusing "API key missing" messages. At least log at DEBUG so the root cause is inspectable.

---

## Low Priority (nice to have)

### 14. Unvalidated `thread_id`
`/api/agent` and `/api/conversations/...` accept `thread_id` as free-form strings. Adding a UUID regex at the Pydantic layer costs nothing and rejects cache-key pollution from bad actors / malformed clients.

### 15. Two `as any` casts in frontend
`web/src/components/Sources.tsx:657` and `web/src/pages/ChatPage.tsx:170` — both `rehypeRaw as any` for a plugin typing gap. Upstream issue, low value to fix locally.

### 16. 16 `except Exception:` handlers
All are paired with a log call and a sane fallback — not bare catches. Worth a periodic sweep to confirm none mask bugs that should propagate (in particular the three in `estimator.py` at 490/500/533/539).

---

## Positive Observations

- **ZIP extraction is defense-in-depth** (`parsers/attachment_processor.py:460-576`): depth limit, file-count limit, total-size limit, compression-ratio bomb detection, path-traversal and absolute-path rejection, explicit lenient-vs-strict error policy. One of the better-hardened surfaces in the codebase.
- **No unsafe primitives**: no `subprocess`, `os.system`, `shell=True`, `eval`, `exec`, `pickle.loads`, or `yaml.load` anywhere in `src/mtss/` or `scripts/`.
- **SQL is consistently parameterized**: asyncpg calls use `$1`-style params exclusively; Supabase SDK fluent API everywhere else. No f-string SQL construction.
- **API security posture is solid**: JWT middleware, `X-Content-Type-Options`, `X-Frame-Options: DENY`, HSTS (prod-only gating), CSP with narrow `connect-src`, rate limiting via slowapi, path-traversal rejection on `/api/archive/`, chunk_id regex + length validation on `/api/citations/`.
- **Recent archive-storage hardening is good work** — pagination + retry with backoff, and the targeted tests in `tests/test_archive_storage.py` are precise. Finding #1 is about finishing that migration, not critiquing the approach.
- **`EmptyContentError` + LlamaParse fallback pattern** (`parsers/base.py`, `parsers/attachment_processor.py`) is a clean separation: local parsers raise a typed exception, pipeline catches that specific type only, ValueError still fails hard. Good discipline.
- **503 tests pass, npm audit clean, frontend builds**: CI-adjacent health is green.

---

## Delta vs Previous Report
No prior `code-health-report.md` — this establishes the baseline.

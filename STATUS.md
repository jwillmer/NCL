# Session Status — Chat UI fixes

_Last updated: 2026-04-22 — Claude keeps this file in sync; scroll is optional._
_ID scheme: `I#` = issue, `D#` = pending decision, `R#` = reference note, `C#` = commit._

## Active issues

_None — all chat-UI fixes confirmed by user._

## Pending decisions (need user input)

_None._

## Working theories / active investigations

_None._

## Reference notes

- **R1 — LangGraph checkpoint tables.** `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`, `checkpoint_migrations` in Supabase Postgres are created at API startup by `AsyncPostgresSaver.setup()` at `src/mtss/api/main.py:244-246`. Owned by LangGraph; do not hand-edit. Persist agent state across chat turns. Separate from app-owned `conversations` / `messages` tables and from `archive*` buckets.
- **R2 — AI SDK v6 protocol cheat sheet.** `data: {"type":"start","messageId":"..."}` → `data: {"type":"text-start","id":"..."}` → repeated `text-delta` with `{id, delta}` → `text-end` → optional `data-<name>` events → `finish` → `[DONE]`. Response header: `x-vercel-ai-ui-message-stream: v1`.
- **R3 — Canonical chunk_id.** 12-char lowercase hex (`[a-f0-9]{12}`), SHA-256 of `doc_id:char_start:char_end`.
- **R4 — UI proposals.** 300 triaged proposals available in `UI-PROPOSALS.md` across 20 categories; not yet prioritized.
- **R5 — Root logging.** API now configures root logger at INFO by default (`src/mtss/api/main.py`); override with `LOG_LEVEL=DEBUG`/`WARNING`. Operational breadcrumbs (`chat_node:` tool routing, `set_filter invoked:`, Langfuse init) surface to stdout.
- **R6 — Filter routing.** For factual queries, `chat_node` pins `tool_choice=search_documents`, which used to prevent `set_filter` from ever running. A deterministic pre-scan now redirects `tool_choice=set_filter` when the user's message contains a single unambiguous vessel type/class from the cache; the `filter_pending_search` flag brings the next pass back to search.

---

## Archive — resolved + confirmed

### Commits landed
- **C1 / `e10597b`** `fix(chat): port streaming to AI SDK v6 UI Message Stream` — resolves **I1**, **I2**, partial **I13**.
- **C2 / `34f5bc6`** `fix(chat): citations round-trip and downloads work from new tab` — resolves **I3**, **I4**, **I5**, partial **I6**.
- **C3 / `3600bf0`** `fix(chat): agent-pushed filter updates + test coverage for stream/citations` — resolves **I8** (code), partial **I12** (backend tests).
- **C4 / `ce5d2b8`** `feat(validate): opt-in check for remote archive URI presence` — resolves **I10** (detection side).
- **C5 / `8d80c29`** `feat(maintenance): heal-archive command to push missing files to bucket` — resolves **I10** (remediation side).
- **C6 / `f9097d9`** `fix(heal-archive): sanitize local keys before diffing against the bucket`.
- **C7 / `6387437`** `fix(chat): progress + timestamps + feedback toggle + download path` — resolves **I14**, **I15**, **I17**, **I18**, **I19**, partial **I21**; partial **I16** (frontend-only attempt).
- **C8 / `02bb060`** `feat(feedback): persist clear action to backend` — resolves **I20**.
- **C9 / `e08ca88`** `ui(chat): merge timestamp + duration onto the feedback row`.
- **C10 / `4d62d95` + `c6aeb12`** `chore(agent): log set_filter / chat_node tool-routing decisions`.
- **C11 / `693840f`** `chore(api): configure root logging so INFO breadcrumbs surface`.
- **C12 / `d315879`** `fix(agent): redirect to set_filter when user names a vessel type/class` — final fix for **I16**.

### Archived issues

| ID | Issue | Resolution |
|---|---|---|
| **I1** | Chat message sent → spinner flashes, nothing appears; refresh shows it | Restored `DefaultChatTransport` with async `headers` and functional `body`. |
| **I2** | Tokens not streaming progressively | Rewrote `_stream_agent` to v6 typed SSE. |
| **I3** | `/api/citations/user-content-<hex>` → 400 | Set `clobberPrefix: ""` in the sanitize schema. |
| **I4** | Clicking a source → "Failed to fetch: Bad Request" | Same clobber fix as I3. |
| **I5** | Download new-tab → 401 | Added `ArchiveStorage.create_signed_url()` (5 min TTL); citation response exposes `archive_download_signed_url`. |
| **I6** | Citation response had `content: null` | Replaced silent warning with `logger.exception`; test bucket populated by I11. |
| **I7** | User hypothesis: proxy/URL wrong | RULED OUT — Vite proxy + `/api` prefix + FastAPI routes are aligned. |
| **I8** | Agent sets vessel filter but UI dropdown doesn't update | Moved handler to `useChat`'s `onData` callback. (Re-surfaced as I16.) |
| **I9** | Orphan chunk in remote Supabase | RESOLVED by D2 — full `mtss import --skip-vessels` backfilled. |
| **I10** | `validate ingest` didn't check remote storage | Check #35 `_check_remote_archive_uris` (`--check-storage`, opt-in) + `mtss heal-archive`. |
| **I11** | Populate test Supabase bucket | 66,140 files uploaded to `archive-eval`. |
| **I12** | Unit tests for the fixes | Backend DONE — 951 passed / 4 skipped / 0 failed. Frontend vitest setup deferred. |
| **I13** | Debug leftovers (`test_payload.py`, `/test-agent`) | DONE — `/test-agent` removed; `test_payload.py` deleted. |
| **I14** | Progress status not shown during streaming | `useChat.onData` handles `data-progress`; rendered above the streaming skeleton. |
| **I15** | Loading animation gap between send and first token | Extended skeleton condition to cover `isStreaming` *and* no assistant-text-yet. |
| **I16** | Filter UI not updating for "vlcc" question | Two root causes: (a) `onData` ref-null on first call (fixed in C3/C7); (b) real root cause — `chat_node` pinned `tool_choice=search_documents` for factual queries, blocking `set_filter`. Fixed in C12 with deterministic prefilter redirect. |
| **I17** | Doubled folder path in "Download Original" | `resolveArchiveUrl` regex `{16}` → `{16,}`; dialog markdown swaps to `archive_download_signed_url`. |
| **I18** | Source name (not just icon) should open dialog | Row wrapped in `<button>`. |
| **I19** | Feedback buttons 405 Method Not Allowed | Backend `@router.post("")` (no trailing slash) to avoid 307 losing POST. |
| **I20** | Feedback cannot be unset or switched | Buttons no longer `disabled`; same-click clears + submits `value=-1`, other-click switches. Cleared events logged separately in Langfuse so analytics aren't skewed. |
| **I21** | Per-message timestamps + AI response duration | Backend stamps `HumanMessage.additional_kwargs["sent_at"]`; `get_messages` exposes `sent_at` (AIMessage = checkpoint ts fallback); frontend hydrates `messageTimestamps` on history load. Duration stays local. Timestamp + duration now render right-aligned on the feedback row. |

### Archived decisions

| ID | Question | Relates to | Status |
|---|---|---|---|
| **D1** | Fix **I8** via `onData` on `useChat`? | I8 | APPROVED — applied (later superseded by C12) |
| **D2** | Approve `mtss import --skip-vessels` against `.env.test` | I11 | APPROVED — complete |
| **D3** | Start backend tests now | I12 | APPROVED — complete |
| **D4** | Delete `test_payload.py`? | I13 | APPROVED — done |

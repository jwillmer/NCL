---
title: 'mtss-improvements'
project_folder: '/home/jwi/GitHub/MTSS'
claude_session: '77d35229-ff7c-4166-89d8-fd22809d70b3'
created: '2026-04-29T22:02:15+02:00'
focus: 'All feature work pushed on feat/reply-latency-and-vessel-validation; awaiting confirmation on Done items + apply-pass on vessel mismatches'
---

## Pick up here (resume-from-doc-only)

If you're returning to this work cold — or a different operator is — read this section first.

**Re-attach the orchestrator on resume**: in a new Claude Code session, run `/status-orchestrator /home/jwi/GitHub/MTSS/wip/1777492935-mtss-improvements.md` (or paste the path from the AI Status app). The skill keeps this file as the single source of truth for task status — do not edit Active/Done/Completed tables by hand once the orchestrator is driving. To resume the *original conversation* (with full tool history) instead of starting fresh, run `/resume 77d35229-ff7c-4166-89d8-fd22809d70b3` in Claude Code first; the YAML front matter at the top of this file carries the same UUID.

**Branch**: `feat/reply-latency-and-vessel-validation` on `origin` (8 commits, head `6b450fb`). PR not opened. `main` is clean at `origin/main` (`95467ed`).

**State**: All implementation landed. Two human-driven gates remain before the DB-apply pass on the vessel mismatches; one human confirmation pass for the 7 Done items before the PR merges.

**Key artefacts** (all under repo root unless noted):
- `wip/1777492935-mtss-improvements.md` — *this file*. Single source of truth for status.
- `wip/vessel-mismatch/README.md` — apply playbook (test → prod via `.env.test` → `.env`). Describes every step + rollback.
- `wip/vessel-mismatch/web_lookup_results.csv` — 122 untracked entries with verdicts (47 REAL / 11 UNCLEAR / 64 NOT_VESSEL); IMO + type + class + evidence URL where available.
- `wip/vessel-mismatch/web_lookup_summary.md` — top REAL findings + structural narrative (Maran Dry bulker fleet absence; legacy VLCC reconciliation).
- `wip/vessel-mismatch/typo_mapping.csv` — 45 auto-generated typo → canonical mappings (difflib cutoff 0.85; needs hand review per #24).
- `wip/vessel-mismatch/untracked_parking.txt` — 122 names to park in `metadata.unknown_vessel_names`.
- `wip/vessel-mismatch/extractor_noise.txt` — 131 noise entries (now suppressed upstream by #20 — kept as audit record).
- `wip/vessel-mismatch/mismatch_report.md` — categorised report from check #36.
- `wip/vessel-mismatch/generate.py` — reproducible regenerator (`uv run python wip/vessel-mismatch/generate.py`).
- `data/vessel-list.csv` — canonical register (single source of truth per user policy).
- `data/ingest.db` — production-state corpus (READ-ONLY for this work).

**To re-run check #36 against the corpus** (sanity-check post-tightening counts):
```
OPENROUTER_API_KEY=test-key uv run python -c "
import sys, sqlite3
sys.path.insert(0,'src')
from mtss.cli.validate_cmd import _check_unknown_vessel_mentions, _load_canonical_vessel_names, build_folder_to_email_map
from pathlib import Path
conn = sqlite3.connect('file:data/ingest.db?mode=ro', uri=True); conn.row_factory = sqlite3.Row
docs = [dict(r) for r in conn.execute('SELECT * FROM documents')]
chunks = [dict(r) for r in conn.execute('SELECT id, document_id, chunk_id, content, source_title, metadata_json FROM chunks')]
issues, warnings = _check_unknown_vessel_mentions(chunks, docs, _load_canonical_vessel_names(), build_folder_to_email_map(docs), sample_limit=5)
print(warnings[0])"
```
Expected after #20 tightening: `167 vessel-like name(s) … (336 occurrence(s) …)`.

**To re-run the full backend test bundle** touched by this branch:
```
OPENROUTER_API_KEY=test-key uv run pytest \
  tests/test_validate_checks.py tests/test_sanitize_migration.py \
  tests/test_citation_processor.py tests/test_api_agent.py \
  tests/test_api_streaming.py tests/test_sqlite_client.py -q
```
Expected: 238 passed.

**To re-run the frontend test bundle**:
```
cd web && npm test -- --run
```
Expected: 3/3 vitest pass; `tsc --noEmit` clean; `npm run build` clean.

**Gates blocking the apply-pass** (do NOT run `--apply` until all three are done):
1. **#24** — hand-curate `wip/vessel-mismatch/typo_mapping.csv`. Drop/fix difflib mistakes.
2. **#23** — hand-verify the 75 non-REAL entries (11 UNCLEAR + 64 NOT_VESSEL); produce `wip/vessel-mismatch/web_lookup_review.csv`.
3. **#25** — decide register additions from the 47 REAL entries (add / alias-on-existing / leave parked). Update `data/vessel-list.csv` accordingly before retag.

**After the gates clear**, follow `wip/vessel-mismatch/README.md` step-by-step: snapshot → test env → validate → snapshot → diff → prod env → snapshot → diff. The README is prescriptive on every command.

**To merge the PR**: complete #26 (user confirms each Done item). Until then, don't merge `feat/reply-latency-and-vessel-validation`.

## Active tasks

| # | Task | Owner | Input artefact | Expected output | Acceptance |
|---|------|-------|----------------|-----------------|------------|
| #23 | Re-validate the 11 UNCLEAR + 64 NOT_VESSEL entries before any DB write. No subagent — this is a hand-verification gate on #22's findings. | human | `wip/vessel-mismatch/web_lookup_results.csv` (filter to verdict ∈ {UNCLEAR, NOT_VESSEL}) | `wip/vessel-mismatch/web_lookup_review.csv` with a final verdict column the user signs off on | Every non-REAL row carries an explicit final-verdict (REAL-with-correction / NOT_VESSEL-confirmed / drop / leave-parked). |
| #24 | Hand-verify the 45 difflib-generated typo mappings. Several are guesses (`MARAN THE → MARAN THETIS`, `MARAN APO → MARAN APOLLO`, `MARAN MARIA → MARAN MIRA`). | human | `wip/vessel-mismatch/typo_mapping.csv` | The same CSV, curated — wrong rows deleted or corrected. | Every remaining row has been confirmed by a human to be a correct fix. |
| #25 | Decide register treatment for each of the 47 REAL entries. Per source-of-truth policy, no auto-add — you choose: add new row, alias existing canonical, or leave parked. Notable clusters: ~13 Maran Dry bulkers absent today; ARCTURUS/ANTARES/LIBRA VOYAGER are alias candidates for MARAN ARCTURUS/ANTARES/LIBRA; CALLISTO/REGULUS sold/renamed and need ownership reconciliation. | human | `wip/vessel-mismatch/web_lookup_results.csv` (filter verdict=REAL) + `wip/vessel-mismatch/web_lookup_summary.md` for narrative | Updated `data/vessel-list.csv` (and Supabase `vessels` after import) reflecting the additions/aliases | Every REAL row has a recorded decision; `data/vessel-list.csv` mtime newer than `web_lookup_results.csv`. |
| #26 | Confirm each of the 7 Done items below. Per orchestrator skill §6, no auto-promote — needs explicit "ship it" / "confirmed". | human | The Done (awaiting confirmation) table | Reply with confirmation; doc rows move to Completed. | All 7 rows promoted; PR ready to merge. |

## Done (awaiting confirmation)

| # | Task | Finished | Result | Tested |
|---|------|----------|--------|--------|
| #12+#14 | Frontend data-citations patch + Vitest smoke | 2026-04-29 | Commit `8a23102`. `web/src/types/rag.ts` (CitationsFrame types), `web/src/lib/utils.ts` (`applyCitationsToMarkdown` + `buildCiteTag`), `web/src/pages/ChatPage.tsx` (data-citations onData branch + 3s onFinish fallback), `web/src/lib/utils.test.ts`. | 3/3 vitest pass; `tsc --noEmit` clean; `npm run build` succeeds |
| #15 | context_summary Phase-1 measurement harness | 2026-04-29 | Commit `6e1394c`. `build_context_hybrid` in `citation_processor.py`; `embedding_mode` field on `RetrievalResult` + populated in retriever; `scripts/measure_context_summary_swap.py` CLI; smoke fixture saves 86% of tokens on a contrived sensor-log chunk. Production stays on `build_context` until the harness produces real numbers. | 7 new + 8 pre-existing tests pass on `test_citation_processor.py` |
| #20 | Tighten vessel mention extractor | 2026-04-29 | Commit `8166933`. `HARDCODED_NOISE` expanded to 106 entries, `_BIZ_PHRASE_BLOCKLIST` added, `filter_canonical_concats` helper. **Verified: 298→167 unique unknowns, 1045→336 occurrences (~70% noise reduction)** on the test-env corpus. | 238/238 backend tests pass on the consolidated suite |
| reports | Vessel-mismatch lists + apply playbook | 2026-04-29 | Commit `aeacc2f`. `wip/vessel-mismatch/` carries typo_mapping.csv (45), untracked_parking.txt (122), extractor_noise.txt (131), web_lookup_results.csv (47 REAL / 11 UNCLEAR / 64 NOT_VESSEL), web_lookup_summary.md, mismatch_report.md, generate.py, README.md (test→prod playbook). | n/a (artefacts) |
| #9-#11 | Async-citation streaming wire-up (`emit_citations` helper + `chat_node` rewire + `data-citations` SSE frame, gated by `ASYNC_CITATIONS_ENABLED`) | 2026-04-29 | Branch `feat/async-citations-wire` (commit `21919c1`). Sync path preserved for flag-off. Files: `src/mtss/api/agent.py`, `src/mtss/api/streaming.py`, `tests/test_api_agent.py`, `tests/test_api_streaming.py`. | 34/34 targeted tests pass; full suite 1010 pass + 1 pre-existing unrelated fail (`test_entities_cli.py`) |
| #22 | Web-lookup of 122 untracked vessel mentions | 2026-04-29 | **All 122 untracked entries cross-checked against AIS / vessel-database web sources** (vesselfinder, marinetraffic, equasis). `wip/vessel-mismatch/web_lookup_results.csv` + `web_lookup_summary.md`. **47 REAL** (with IMO/type/class), **11 UNCLEAR**, **64 NOT_VESSEL** (extraction noise). Structural finding: ~13 Maran Dry bulk carriers + several older Maran VLCCs missing from canonical register; ARCTURUS/ANTARES/LIBRA VOYAGER are former names (alias candidates). | yes — every entry has an explicit web-source verdict |
| #21 | Vessel-metadata mismatch report | 2026-04-29 | `wip/vessel-mismatch/mismatch_report.md` (moved from repo root). Buckets: 45 typos, 122 untracked, 131 noise. | yes (ran against test env) |

## Completed

| # | Task | Confirmed | Result |
|---|------|-----------|--------|
| audit | Multi-agent reply-latency audit (RAG / storage / LLM+stream / arch) | 2026-04-28 | Ranked recommendations + anti-recommendations; basis for #1, #3 (investigation), #5 |
| #5  | SQLite PRAGMA tuning (cache=-64000, mmap=256 MB, journal_size_limit=64 MB; tracker uses 1/8) | 2026-04-29 | Commit `48f5399`. 34/34 sqlite tests pass. CLAUDE.md updated. |
| #1 (backend foundation) | Async citation processing — `CitationProcessor` refactor (parallel HEADs, `process_response_async`, `serialize_citations_payload`) + `Settings.async_citations_enabled` flag | 2026-04-29 | Commit `b16f11c`. Sync API preserved. Frame wiring still pending (#9–#14). |
| vessel-check | Check #36 `_check_unknown_vessel_mentions` + shared `vessel_mention_extractor` module + 4 unit tests | 2026-04-29 | Commit `cd86963`. 117/117 validate+sqlite tests pass. Test-env run found 298 unique unknowns / 1,045 occurrences. |
| #16 | `.env.template` + `MTSS_SQLITE_*` + `ASYNC_CITATIONS_ENABLED` | 2026-04-29 | Folded into commits 48f5399 / b16f11c. |
| #3  | `context_summary` synthesis swap **investigation only** (verdict: hybrid; gated on Phase-1 measurement = #15) | 2026-04-29 | No code change yet by design. |

## Blocked / needs input

### Vessel-mismatch action queue — awaiting your confirmation

`data/vessel-list.csv` is the single source of truth. Every unknown mention is either a typo (fix) or an untracked vessel (park in `metadata.unknown_vessel_names`).

**Web validation complete**: every one of the 122 untracked entries was cross-checked against external vessel databases (vesselfinder, marinetraffic, equasis). Verdicts in `wip/vessel-mismatch/web_lookup_results.csv` — **47 REAL** (with IMO + type + class), **11 UNCLEAR**, **64 NOT_VESSEL**. The 64 NOT_VESSEL entries are mostly extraction noise (sentence fragments, address bleed, technical terms); they would still be parked harmlessly in `unknown_vessel_names`, but task #20's extractor tightening already collapses most of these upstream so they no longer surface — re-running the generator from the new path produced 0 noise entries.

**Apply-pass gate**: task **#23** is a **human-review-only** step — the 75 non-REAL entries (11 UNCLEAR + 64 NOT_VESSEL) need a hand-verified second pass before any DB write. No subagent will be dispatched for this; the user (or a domain expert) reviews `wip/vessel-mismatch/web_lookup_results.csv` directly and produces an annotated `web_lookup_review.csv` with final verdicts. Until that review lands, do not run `apply_vessel_name_mapping.py --apply` or `annotate_unknown_vessel_mentions.py --apply` against test or prod.

| Bucket | Unique | Occurrences | Proposed action | Draft file |
|---|---:|---:|---|---|
| **Typos to fix** | 45 | 111 | Map detected → canonical via `scripts/apply_vessel_name_mapping.py`, then retag | `wip/vessel-mismatch/typo_mapping.csv` |
| **Untracked to park** | 122 | 225 | Stamp on `metadata.unknown_vessel_names` via `scripts/annotate_unknown_vessel_mentions.py` | `wip/vessel-mismatch/untracked_parking.txt` |
| **Extractor noise (no DB action)** | 131 | 709 | Feed into `HARDCODED_NOISE` / regex tightening — task #20 | `wip/vessel-mismatch/extractor_noise.txt` |

All four files plus a `README.md` (apply playbook for `.env.test` → `.env`) and a reproducible `generate.py` live in `wip/vessel-mismatch/`.

Top typos (full list in CSV):
- `MARAN ARIES` ×15 → `MARAN ARES`
- `MARAN HERCUES` ×6 → `MARAN HERCULES`
- `MARAN ORPEHUS` ×6, `MARAN ORHEUS` ×5 → `MARAN ORPHEUS`
- `MARAN TRUST` ×6 → `MARAN TAURUS`
- `MARAN POSEIDON AND` ×6 → `MARAN POSEIDON`
- `MARAN HELLEN` ×4 → `MARAN HELEN`
- `MARAN APO` ×4 → `MARAN APOLLO`
- `MARAN APRHRODITE` ×4 → `MARAN APHRODITE`
- `MARAN MARIA` ×5 → `MARAN MIRA`

Top untracked (full list in text file):
- `ANTONIS L. ANGELICOUSSIS` ×16, `PEGASUS VOYAGER` ×7, `MARAN MIM` ×7, `POLARIS VOYAGER` ×6, `MARAN SOPHIA` ×5

**Confirmation needed**: review the three draft files at the repo root and reply with one of:
- "ok, apply both" → run typo fix on test env, then park untracked, then promote to prod
- "ok, typos only" / "ok, parking only" → partial
- list specific entries to drop / fix differently before applying

## Agent log

_(append-only, newest first)_

- 2026-04-29 — Added "Pick up here" resume block: re-attach instructions (`/status-orchestrator <path>` and `/resume <claude_session>`), key-artefact map, verification commands, gate sequencing. Doc is now self-sufficient for cold resumption.
- 2026-04-29 — Surfaced three additional human-validation gates: #24 (hand-review the difflib-generated typo mapping), #25 (decide register additions from the 47 REAL web findings), #26 (confirm the 7 Done items). The original chat summary called out only #23, which understated the queue.
- 2026-04-29 — Cleaned up Active section: rows for #9–#15 were stale duplicates of entries already in Done (awaiting confirmation). Only #23 (human-review gate) remains active.
- 2026-04-29 — Added task #23 — **human-review-only gate** on the DB apply-pass. The 75 non-REAL entries from #22 (11 UNCLEAR + 64 NOT_VESSEL) must be hand-verified before any typo-fix or parking write hits test/prod. Will not be delegated to a subagent.
- 2026-04-29 — Status file + vessel reports relocated into `wip/` (commit `6b450fb`). Status file moved via the AI-Status dashboard's `/move` endpoint to keep its index valid; vessel reports moved via `git mv reports/vessel-mismatch wip/vessel-mismatch`. All path references rewritten across docs and code; generator script confirmed to run from the new location (45 typos / 122 untracked / 0 noise — extractor tightening already collapses noise upstream).
- 2026-04-29 — Branch consolidation complete. `feat/reply-latency-and-vessel-validation` pushed to origin (8 commits). `main` reset to `origin/main`. The bundled commit `087eaf9` (vessel + measurement work entangled by an autocommit) was split via `git reset --soft HEAD~1` into two clean commits (`8166933` vessel, `6e1394c` measurement). Reports + .claude permission additions consolidated into `aeacc2f`.
- 2026-04-29 — All four parallel agents (#12+#14 frontend, #15 measurement, #20 extractor, #22 web-lookup) completed cleanly. Frontend tests run by orchestrator (sandboxed for the agent) confirmed 3/3 pass; backend full-suite spot-check at end-of-session shows 238/238 across all touched test files. Extractor count drop verified inline: 298→167 unique / 1045→336 occurrences against the test corpus.
- 2026-04-29 — Vessel web-lookup agent finished (`ae453779e81036834`): 47 REAL vessels found missing from the register (notably the entire Maran Dry bulker fleet — only 5 of ~18 listed today; plus older Maran VLCCs CYGNUS/GEMINI/ANDROMEDA/CARINA/CASTOR/CALLISTO/AQUARIUS/CORONA/CASSIOPEIA/SAGITTA/REGULUS/TRITON, several already sold or renamed). Top finding: `ANTONIS L. ANGELICOUSSIS` is a misspelling of `ANTONIS I. ANGELICOUSSIS` (real VLCC IMO 9930777). 64 NOT_VESSEL (extraction noise: maritime jargon, address concatenations, doc-text bleed).
- 2026-04-29 — Citation wire-up agent finished (`a0f308103ba334550`): commit `21919c1` on local branch `feat/async-citations-wire`. 34/34 targeted tests pass. Quality bar met (gated, sync path preserved, tests both branches, wire format documented).
- 2026-04-29 — Status orchestrator engaged; YAML front matter now carries `claude_session` (`77d35229-ff7c-4166-89d8-fd22809d70b3`) and `project_folder` for future `/resume`.
- 2026-04-29 — Generated `vessel_mismatch_report.md` via inline categorisation (difflib for typo bucket, regex heuristics for the rest). Auto-mode exited just before writing the report.
- 2026-04-29 — Three commits landed in order: `perf(sqlite)` → `perf(rag)` (async citations backend) → `feat(validate)` (check #36). Mixed files (`config.py`, `.env.template`) split via temporary edit + restore.
- 2026-04-29 — Vessel research subagent reported back: canonical register is the Supabase `vessels` table (UUID id, UNIQUE name, aliases TEXT[]). CSV-minted UUIDs in `data/vessel-list.csv` drift per load — names+aliases are the stable identity for this check.
- 2026-04-29 — `uv sync` was needed mid-session (`nest_asyncio`, `typer`, `langchain_text_splitters` missing); ran `uv sync --extra ingest` once, then 117 tests pass with `OPENROUTER_API_KEY=test-key`.
- 2026-04-28 — Pause + resume happened cleanly; background vessel-research agent finished while paused.

## Branch status

- **Pushed**: `feat/reply-latency-and-vessel-validation` → `origin/feat/reply-latency-and-vessel-validation` (8 commits ahead of `main`).
- **Open PR URL**: https://github.com/jwillmer/MTSS/pull/new/feat/reply-latency-and-vessel-validation
- **`main`**: clean at `origin/main` (`95467ed`).

## Notes

- **Don't touch `./data/`** without explicit confirmation per CLAUDE.md.
- Pre-existing test failures unrelated to this work: `test_ingest_storage.py::TestFixTableMdCommand::*` collection errors stem from a markdown-table normalisation gap (not a regression — confirmed by stashing).
- Async citations is gated by `ASYNC_CITATIONS_ENABLED=false` so the next chunk of work (#9–#14) can land behind the flag without breaking prod.
- Check #36 emits **warnings**, not issues, by deliberate choice — the canonical register is curated separately from ingest, so an unknown mention is a signal not a defect.
- Vessel extractor improvements tracked in #20; expected ~10× false-positive reduction (298 → ~30–50 actionable).

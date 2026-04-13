---
purpose: Identify overlaps, conflicts, and redundancies across all plan documents and propose consolidation
status: proposal
date: 2026-04-13
---

# Plan Consolidation Proposal

## Problems Found

### Problem 1: Search/Retrieval Fixes Are Duplicated Across Two Plans

**Files:** `00-critical-fixes-plan.md` (SR-0 through SR-3 equivalent as Fix 0.1, 1.1, 1.2, 1.3) and `optimization-plan.md` (lines 362-416, section "Search/Retrieval Fixes" SR-0 through SR-3).

Both plans describe the exact same four search/retrieval fixes:

| Fix | `00-critical-fixes-plan.md` | `optimization-plan.md` |
|-----|---------------------------|----------------------|
| Reranker bug (top_k) | Fix 0.1 (lines 36-93) | SR-0 (lines 369-376) |
| Enriched rerank context | Fix 1.1 (lines 98-141) | SR-1 (lines 378-386) |
| max_tokens 1000->2000 | Fix 1.2 (lines 143-169) | SR-2 (lines 388-397) |
| rerank_top_n 5->8 | Fix 1.3 (lines 171-198) | SR-3 (lines 399-406) |

**Conflict:** `00-critical-fixes-plan.md` provides far more detail (exact code, tests, edge cases, review findings). `optimization-plan.md` provides a summary. An implementer following both plans would do the work twice, or be confused about which version is authoritative.

**Additionally:** `00-critical-fixes-plan.md` includes five more fixes (Fix 2.1 through Fix 2.5) that `optimization-plan.md` does not cover. These are the Phase 2 items (top_k to 40, HNSW ef_search, parallel query embedding, rerank score floor, completeness transparency). So `00-critical-fixes-plan.md` is a strict superset of the search content in `optimization-plan.md`.

**Resolution:** The SR-0 through SR-3 section in `optimization-plan.md` (lines 362-416) should be replaced with a cross-reference to `00-critical-fixes-plan.md`. The optimization plan should say "Search/retrieval fixes are in `00-critical-fixes-plan.md`" and remove the duplicated detail.

---

### Problem 2: Retrieval Quality Quick Wins Exist in Three Places

**Files:** `optimization-plan.md` (lines 242-331, section "Retrieval Quality Quick Wins"), `06b-review-findings.md` (lines 30-88), and `implementation-plan.md` (step 6 in `decisions-and-progress.md` line 177).

All three documents describe the same three retrieval quality changes:

| Change | `optimization-plan.md` | `06b-review-findings.md` | `implementation-plan.md` |
|--------|----------------------|-------------------------|------------------------|
| RQ-1/P6: Attachment context inheritance | Lines 248-269 (full code) | Lines 32-49 (feasibility) | Referenced in decisions-and-progress.md line 177 |
| RQ-2/P1-A: Minimum content filter | Lines 271-297 (full code) | Lines 51-70 (feasibility) | Referenced in decisions-and-progress.md line 177 |
| RQ-3/P8-A: Date in embedding text | Lines 299-319 (full code) | Lines 73-88 (feasibility) | Referenced in decisions-and-progress.md line 177 |

**Conflict:** `optimization-plan.md` has full implementation code. `06b-review-findings.md` has the feasibility analysis and edge case notes. Neither references the other. An implementer might follow one and miss edge cases documented in the other (e.g., `06b-review-findings.md` line 42-43 notes ZIP attachments also need context inheritance, which `optimization-plan.md` does not mention).

**Resolution:** The implementation code in `optimization-plan.md` (RQ-1/RQ-2/RQ-3) should cross-reference `06b-review-findings.md` for edge cases. Or better: move the implementation details into `implementation-plan.md` Phase 4 (where they are executed) and have both `optimization-plan.md` and `06b-review-findings.md` reference the implementation plan.

---

### Problem 3: `local-only-ingest-plan.md` Is Superseded by `implementation-plan.md`

**Files:** `local-only-ingest-plan.md` (14 steps, ~1100 lines) and `implementation-plan.md` (6 phases, ~1200 lines).

`implementation-plan.md` explicitly states in its header (line 10): it merges "local-only ingest" and "cost optimizations" into a single plan. Its Phase 3 (lines 760-886) covers "Local-Only Storage Backend" and says (line 763): "This phase implements Steps 1-6 and 10-13 from the `local-only-ingest-plan.md`."

However, `local-only-ingest-plan.md` remains as a standalone document with its own step numbering, implementation code, and test plans. The two documents contain:

- **Overlapping content:** Step 1 (extend LocalStorageClient) = Phase 3.2. Step 2 (embeddings in chunks) = Phase 3.3. Step 3 (progress tracker) = Phase 3.6. Step 4 (unsupported logger) = Phase 3.7. Step 5 (archive generator DI) = Phase 3.8. Step 6 (topic similarity) = Phase 3.4. Step 7 (component factory) = Phase 4.1. Step 8 (CLI flag) = Phase 4.2. Step 9 (manifest) = Phase 4.3. Step 10 (flush) = Phase 3.5. Step 11 (version manager) = Phase 4.4. Step 12 (hierarchy manager) = Phase 3.8. Step 13 (move local_storage.py) = Phase 3.1.
- **Conflicting detail levels:** `local-only-ingest-plan.md` has more implementation detail (full code for each step, all 17 review findings). `implementation-plan.md` has a higher-level view with less code but better phase ordering.

**Stale content in `local-only-ingest-plan.md`:**
- Step 2 comment says "1536-dim" (line 167), but `implementation-plan.md` Phase 0.2 changed dimensions to 512. The implementation plan notes this dependency (line 77: "Update ... from '1536-dim' to '512-dim'") but the local-only plan itself was never updated.
- Step 6 performance section says "1536-dim vectors" (line 443-444), but dimensions are now 512.
- Step 9 manifest says `"embedding_dimensions": 1536` (line 656), but should be 512.

**Resolution:** `local-only-ingest-plan.md` should be archived. `implementation-plan.md` is the authoritative plan. The detailed implementation code and review findings from `local-only-ingest-plan.md` that are not in `implementation-plan.md` should be treated as reference material, not as an active plan.

---

### Problem 4: `optimization-plan.md` Cost Phases Overlap with `implementation-plan.md`

**Files:** `optimization-plan.md` (3 cost optimization phases) and `implementation-plan.md` (Phases 0-2 covering the same optimizations).

| Optimization | `optimization-plan.md` | `implementation-plan.md` |
|-------------|----------------------|------------------------|
| Image pre-filtering | Phase 1a (lines 40-46) | Phase 1 (lines 90-258) |
| Image model switch (GPT-4.1-nano) | Phase 1b (lines 48-51) | Not explicitly included |
| Local parsing for trivial formats | Phase 1c (lines 53-58) | Phase 2.3 (lines 427-608) |
| Tiered PDF parsing | Phase 2a (lines 78-84) | Phase 2.1-2.2 (lines 263-418) |
| Local Office parsers | Phase 2b (lines 86-91) | Phase 2.3 (lines 427-608) |
| GPT-4.1-mini batch for complex PDFs | Phase 2c (lines 93-94) | Phase 2.4 mentions fallback |
| LLM consolidation | Phase 3a-3c (lines 116-127) | Not included (optimization-plan says "optional") |

**Conflict:** `implementation-plan.md` has detailed implementation code for each optimization. `optimization-plan.md` has the cost analysis (before/after tables) but only brief descriptions. An implementer should follow `implementation-plan.md` for the code and `optimization-plan.md` for the cost justification.

**Missing in implementation-plan.md:** The image model switch to GPT-4.1-nano (optimization-plan.md Phase 1b) is not explicitly covered in `implementation-plan.md`. It is a one-line config change but could be missed.

**Resolution:** `optimization-plan.md` should be retained as the cost analysis document (read-only reference) with a clear note that implementation details are in `implementation-plan.md`. Add the GPT-4.1-nano model switch to `implementation-plan.md` Phase 1.

---

### Problem 5: `optimization-plan.md` Processing Time Estimates Contradict `06c-review-findings.md`

**Files:** `optimization-plan.md` (lines 165-200, "Processing Time Before & After") and `06c-review-findings.md` (lines 207-248, "Revised Speed Estimates").

| Estimate | `optimization-plan.md` | `06c-review-findings.md` |
|----------|----------------------|-------------------------|
| Current pipeline | ~10.5 hours | ~10.5 hours (agrees) |
| After speed Phase 1 | ~3-4 hours | Not directly comparable |
| After all speed phases | ~1.5-2.5 hours | ~53 minutes |
| Final with cost optimization | ~1-2 hours | ~53 minutes |

`06c-review-findings.md` was written AFTER `optimization-plan.md` and accounts for the bottleneck profile shift (local parsers replacing LlamaParse, local storage replacing Supabase). Its estimates are more accurate. The `optimization-plan.md` speed estimates are stale.

**Resolution:** `optimization-plan.md` processing time section should note that estimates are superseded by `06c-review-findings.md` which accounts for the local-only architecture.

---

### Problem 6: Review Findings Documents Contain Implementation Details That Belong in Plans

**Files:** `06b-review-findings.md`, `06c-review-findings.md`, `07a-review-findings.md`

These documents serve two purposes:
1. **Decision records:** Which proposals were approved/deferred and why (valuable, should be kept).
2. **Implementation code:** Exact code changes, line numbers, new methods (duplicates the implementation plan).

Specifically:
- `07a-review-findings.md` lines 243-297 contain implementation code for P0, P1a, P1b, P1c and a tsvector migration. This same code appears in `00-critical-fixes-plan.md` (more detailed) and is also referenced in `decisions-and-progress.md`.
- `06c-review-findings.md` lines 265-275 describe where approved proposals should go in the implementation plan. This integration guidance is useful but should be reflected in `implementation-plan.md` itself.

**Resolution:** The review-findings docs are fine as decision records. Their implementation code snippets are reference material, not authoritative implementation instructions. This is not a conflict that needs fixing -- it is an information architecture clarification.

---

### Problem 7: `decisions-and-progress.md` Has Stale References

**File:** `decisions-and-progress.md`

Issues found:
1. **Line 26:** "09 -- Test validation plan" shows status "Pending" for decision, but `09-test-validation-execution.md` is a fully written execution plan. The status should reflect that the plan exists but has not been executed.
2. **Line 100:** Implementation Order step 6 says "Retrieval quality quick wins (06b) -- P6, P1-A, P8-A (~2 hours, MUST be before first ingest)". This is correct but does not mention that the implementation details are in `optimization-plan.md` RQ-1/RQ-2/RQ-3, creating ambiguity about where to find the code.
3. **Line 198-199:** Plan Documents Index lists `optimization-plan.md` status as "Complete" and `local-only-ingest-plan.md` status as "Complete". These should be marked as "Superseded by implementation-plan.md" or "Reference only" to avoid confusion about which plans to follow.
4. **Line 14:** Decision D-07 reference says `"D-07: chunk_size_tokens 512 -> 1024, overlap 50 -> 100"` but the decisions log calls this D-08 (line 94). The frontmatter and the decisions log disagree on numbering. The implementation plan frontmatter (line 14) says D-07 for chunk size, but `decisions-and-progress.md` line 94 says D-08.

**Resolution:** Update the decisions tracker to reflect current plan statuses and fix the decision numbering inconsistency.

---

### Problem 8: `00-critical-fixes-plan.md` Fix 0.1 vs `implementation-plan.md` Config Changes

**Files:** `00-critical-fixes-plan.md` (Fix 0.1, line 56) and `implementation-plan.md` (Phase 0.2, line 64).

`00-critical-fixes-plan.md` Fix 0.1 adds a new config field `retrieval_top_k` with default 20 to `config.py`. `implementation-plan.md` Phase 0 also modifies `config.py` (chunk_size, embedding_dimensions). Fix 2.1 then changes `retrieval_top_k` default to 40. `implementation-plan.md` Phase 4.5 adds `max_concurrent_files` change to `config.py`.

**No actual conflict:** These are additive changes to different fields in `config.py`. However, both plans modify the same file without cross-referencing each other. An implementer should be aware that `config.py` is modified by BOTH plans.

**Resolution:** Minor -- add a note to each plan that `config.py` is a shared modification target.

---

### Problem 9: `09-test-validation-execution.md` Prerequisites Reference Both Plans

**File:** `09-test-validation-execution.md` (lines 20-29)

The prerequisites list items from multiple plans:
- "Plan 00 (Critical Fixes)" -- from `00-critical-fixes-plan.md`
- "Implementation Plan Phase 0-5" -- from `implementation-plan.md`
- "06b Quality Wins" -- from `optimization-plan.md` RQ section
- "Optimization Plan SR-0 through SR-3" -- from `optimization-plan.md`

This creates confusion because SR-0 through SR-3 in `optimization-plan.md` are the same fixes as Plan 00. The prerequisites list them as if they are separate work items.

**Resolution:** Remove "Optimization Plan SR-0 through SR-3" from prerequisites since it duplicates "Plan 00 (Critical Fixes)".

---

### Problem 10: `implementation-plan.md` Frontmatter Decision Numbering Mismatch

**File:** `implementation-plan.md` (lines 13-15)

The frontmatter says:
```
decisions:
  - "D-07: chunk_size_tokens 512 -> 1024, overlap 50 -> 100"
  - "D-08: embedding_dimensions 1536 -> 512"
  - "D-05: implement cost optimization Phase 1+2 before first ingest"
```

But in `decisions-and-progress.md`:
- D-07 is "Retrieval quality proposals (06b)" (line 74)
- D-08 is "Chunk size -- 1024 tokens" (line 94)
- D-09 is "Embedding dimensions -- 512" (line 101)

So `implementation-plan.md` labels chunk size as D-07 but it is actually D-08. Embedding dimensions is labeled D-08 but is actually D-09.

**Resolution:** Fix the frontmatter in `implementation-plan.md` to use D-08 and D-09.

---

## Proposed Plan Structure

After consolidation, the plan documents should be organized as follows:

```
docs/investigation/plans/
  00-critical-fixes-plan.md        -- ACTIVE: Search/retrieval fixes (execute first)
  implementation-plan.md           -- ACTIVE: Main implementation plan (Phases 0-5)
  09-test-validation-execution.md  -- ACTIVE: Test validation (execute after implementation)
  decisions-and-progress.md        -- ACTIVE: Decisions tracker (living document)

  reference/                       -- Reference material (not action plans)
    optimization-plan.md           -- Cost analysis with before/after numbers
    estimate-full-results.md       -- Raw estimate data
    local-storage-sqlite-vs-files.md -- Storage decision rationale
    06b-review-findings.md         -- Retrieval quality feasibility analysis
    06c-review-findings.md         -- Processing speed feasibility analysis
    07a-review-findings.md         -- Search optimization feasibility analysis

  archive/                         -- Superseded plans
    local-only-ingest-plan.md      -- Superseded by implementation-plan.md
```

## What to Keep (Active Plans)

### `00-critical-fixes-plan.md`
- **Status:** Clean, standalone, ready for implementation.
- **Role:** The authoritative source for all search/retrieval fixes (Fix 0.1 through Fix 2.5).
- **No changes needed** to the plan itself.

### `implementation-plan.md`
- **Status:** The main implementation plan. Phases 0-5 cover all ingest work.
- **Role:** The authoritative source for all ingest pipeline changes.
- **Needs minor updates** (see "What to Update" section).

### `09-test-validation-execution.md`
- **Status:** Complete plan, ready for execution after implementation.
- **Role:** Post-implementation validation.
- **Needs minor update** to prerequisites (remove duplicate SR-0/SR-3 reference).

### `decisions-and-progress.md`
- **Status:** Active living document.
- **Role:** Session continuity tracker.
- **Needs updates** (see "What to Update" section).

## What to Merge

No plans need to be merged. The overlap issues are better resolved by:
1. Removing duplicate content from `optimization-plan.md` (search fixes, retrieval quality implementation code) and replacing with cross-references.
2. Keeping `optimization-plan.md` as a reference document for cost analysis.

Merging `optimization-plan.md` into `implementation-plan.md` would make the implementation plan too long and would mix cost justification (reference material) with implementation instructions (action items).

## What to Archive

### `local-only-ingest-plan.md` -- Move to `archive/`

**Reason:** Fully superseded by `implementation-plan.md` which merges this plan with cost optimizations and provides a better-organized phase structure. The original plan has stale content (1536-dim references) that was never updated.

**Value preserved:** The 17 review findings (lines 936-1145) contain valuable edge case documentation. These should remain accessible as reference material. However, since `implementation-plan.md` already incorporates the key decisions, the archive copy is sufficient.

## What to Move to `reference/`

### `optimization-plan.md`

**Reason:** Its primary value is the cost analysis tables (before/after comparisons). The implementation details it contains are duplicated (and more detailed) in `00-critical-fixes-plan.md` and `implementation-plan.md`. It should not be treated as an action plan.

### `estimate-full-results.md`

**Reason:** Raw data. Already complete, no action items.

### `local-storage-sqlite-vs-files.md`

**Reason:** Decision rationale. Already complete, decision made (JSONL).

### `06b-review-findings.md`, `06c-review-findings.md`, `07a-review-findings.md`

**Reason:** Feasibility analyses. Decisions are captured in `decisions-and-progress.md`. These contain useful reference material (edge cases, code review notes) but are not action plans.

## What to Update

### 1. `optimization-plan.md` -- Remove Duplicate Content

**Change 1:** Replace lines 362-416 (Search/Retrieval Fixes section) with:
```markdown
## Search/Retrieval Fixes

See `00-critical-fixes-plan.md` for the authoritative implementation plan covering:
- SR-0/Fix 0.1: Reranker bug fix (P0 CRITICAL)
- SR-1/Fix 1.1: Enriched rerank context
- SR-2/Fix 1.2: max_tokens increase
- SR-3/Fix 1.3: rerank_top_n increase
- Fix 2.1-2.5: Additional search improvements (top_k, HNSW, parallel, score floor, transparency)
```

**Change 2:** Add note to lines 242-331 (Retrieval Quality section):
```markdown
> **Implementation:** See `implementation-plan.md` for code-level details.
> See `06b-review-findings.md` for edge case analysis (e.g., ZIP attachment handling for RQ-1).
```

**Change 3:** Add note to lines 165-200 (Processing Time section):
```markdown
> **Note:** These estimates pre-date the local-only architecture decision. See
> `06c-review-findings.md` for revised estimates (~53 minutes with local parsers
> and local storage).
```

### 2. `implementation-plan.md` -- Fix Frontmatter and Add Missing Items

**Change 1:** Fix decision numbering in frontmatter (lines 13-15):
```yaml
decisions:
  - "D-08: chunk_size_tokens 512 -> 1024, overlap 50 -> 100"
  - "D-09: embedding_dimensions 1536 -> 512"
  - "D-05: implement cost optimization Phase 1+2 before first ingest"
```

**Change 2:** Add GPT-4.1-nano image model switch to Phase 1 (after line 258). This is optimization-plan.md Phase 1b, currently missing from the implementation plan:
```markdown
### 1.4 Switch image model to GPT-4.1-nano

**File:** `src/mtss/config.py` or `.env`

Change `IMAGE_LLM_MODEL` from `gpt-4o-mini` to `gpt-4.1-nano`. Per-image cost drops
from ~$0.01 to ~$0.0002. Simple classify+describe tasks do not require the larger model.

**Effort:** 1 line, 1 minute.
```

**Change 3:** Add a note that `config.py` is also modified by `00-critical-fixes-plan.md` (new `retrieval_top_k` and `rerank_score_floor` fields).

### 3. `09-test-validation-execution.md` -- Fix Duplicate Prerequisite

**Change:** Remove line 29 ("Optimization Plan SR-0 through SR-3: Search/retrieval fixes applied") since it duplicates line 20 ("Plan 00 (Critical Fixes): Reranker bug fixed...").

### 4. `decisions-and-progress.md` -- Update Statuses and Fix References

**Change 1:** Line 26: Change doc 09 status from "Pending" to "Plan created (`09-test-validation-execution.md`), execution pending."

**Change 2:** Lines 196-204: Update Plan Documents Index:
- `optimization-plan.md`: Change status from "Complete" to "Reference (cost analysis)"
- `local-only-ingest-plan.md`: Change status from "Complete" to "Archived (superseded by implementation-plan.md)"
- Add a note: "For implementation, follow `00-critical-fixes-plan.md` then `implementation-plan.md`."

**Change 3:** Fix decision numbering cross-reference issue. In the frontmatter of `implementation-plan.md`, D-07 should be D-08 and D-08 should be D-09 (see Problem 10 above).

## Execution Order

After consolidation, the clean execution order is:

```
1. 00-critical-fixes-plan.md
   Phase 0: Fix 0.1 (reranker bug)                    [5 min]
   Phase 1: Fix 1.1-1.3 (enriched rerank, tokens, top_n) [30 min]
   Phase 2: Fix 2.1-2.5 (top_k, HNSW, parallel, floor, transparency) [2-3 hrs]

2. implementation-plan.md
   Phase 0: Config changes (chunk 1024, dims 512)      [15 min]
   Phase 1: Image pre-filtering + model switch          [2-3 hrs]
   Phase 2: Local parsers (PDF/DOCX/XLSX/CSV/HTML)      [4-5 days]
   Phase 3: Local storage backend (parallel with 2)     [5-7 hrs]
   Phase 4: Pipeline wiring + speed optimizations       [4-5 hrs]
     - Includes 06b quality wins (P6, P1-A, P8-A)
     - Includes 06c speed wins (P1 parallel, P4 concurrent)
   Phase 5: Validation                                  [4-6 hrs]

3. 09-test-validation-execution.md
   Phase 1-8: Test subset validation                    [~1 hr]

4. Full ingest run
   6,289 emails, local-only mode                        [~53 min, ~$6-10]

5. Production import (when ready)
```

Each step depends on the previous one completing. Steps 1 and 2 are the implementation work. Step 3 validates the implementation. Steps 4-5 are execution.

**No ambiguity:** Each file/function is modified by exactly one plan. `config.py` is the only file modified by both plans, but they change different fields (Plan 00 adds `retrieval_top_k` and `rerank_score_floor`; implementation plan changes `chunk_size_tokens`, `embedding_dimensions`, and `max_concurrent_files`).

---

## Summary of All Issues

| # | Type | Severity | Description |
|---|------|----------|-------------|
| 1 | Duplicate work | High | Search fixes duplicated in `00-critical-fixes-plan.md` and `optimization-plan.md` |
| 2 | Overlapping scope | Medium | Retrieval quality wins in three places with different detail levels |
| 3 | Superseded content | High | `local-only-ingest-plan.md` fully absorbed into `implementation-plan.md` with stale 1536-dim refs |
| 4 | Overlapping scope | Medium | Cost optimization phases in both `optimization-plan.md` and `implementation-plan.md` |
| 5 | Stale content | Medium | Processing time estimates in `optimization-plan.md` pre-date local-only architecture |
| 6 | Overlapping scope | Low | Review findings docs contain implementation code that duplicates plan content |
| 7 | Stale references | Medium | `decisions-and-progress.md` has outdated statuses and missing references |
| 8 | Missing cross-ref | Low | Both plans modify `config.py` without noting the other |
| 9 | Duplicate reference | Low | `09-test-validation-execution.md` lists same fixes twice in prerequisites |
| 10 | Data inconsistency | Medium | Decision numbering mismatch between `implementation-plan.md` frontmatter and `decisions-and-progress.md` |

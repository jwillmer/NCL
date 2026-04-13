---
purpose: Code-level review of 06b retrieval quality proposals against actual implementation
status: complete
date: 2026-04-13
source: 06b-optimization-retrieval-quality.md
---

# 06b Review Findings: Retrieval Quality Proposals

## What Was Reviewed

All 8 proposals from `06b-optimization-retrieval-quality.md` were reviewed against the actual
codebase. For each approved proposal, the specific source files were read and the implementation
path was verified. For deferred proposals, the deferral reasoning was confirmed.

### Source Files Examined

- `src/mtss/ingest/pipeline.py` — main email processing loop, context generation, chunk creation
- `src/mtss/ingest/attachment_handler.py` — attachment processing, context application
- `src/mtss/parsers/chunker.py` — ContextGenerator, DocumentChunker, build_embedding_text
- `src/mtss/processing/vessel_matcher.py` — word-boundary regex matching
- `src/mtss/models/vessel.py` — Vessel model (name, type, class; no aliases field)
- `src/mtss/models/document.py` — Document, EmailMetadata (has date_start, date_end)
- `src/mtss/models/chunk.py` — Chunk model, embedding_text field
- `src/mtss/parsers/email_cleaner.py` — split_into_messages, remove_boilerplate_from_message
- `data/vessel-list.csv` — vessel registry (NAME;TYPE;CLASS, no aliases column)

---

## Approved Proposals: Feasibility Verification

### P6 — Attachment Context Inheritance: CONFIRMED FEASIBLE

**Verification:** The parent email's `context_summary` is generated at `pipeline.py:199-206`,
before the attachment processing loop at `pipeline.py:342-376`. The variable is in scope and
simply needs to be passed as a parameter.

**Hidden dependencies found:** None. The `process_attachment()` function already accepts
`email_doc` which carries `email_metadata`, but `context_summary` is a separate local variable
not attached to the document object. It must be passed explicitly.

**Edge cases:**
- Context summary can be `None` (if LLM call fails or body_text is empty). The attachment
  handler already handles `None` context gracefully in lines 214-228.
- For ZIP attachments (`process_zip_attachment`), the email context should also be passed through.
  The current `process_zip_attachment` function does not receive any context. This is a minor gap
  that should be fixed at the same time — add the same `email_context_summary` parameter.

**Conflict with other changes:** None. The local-only ingest plan does not modify these code paths.

### P1-A — Minimum Content Filter: CONFIRMED FEASIBLE

**Verification:** The filter point is at `pipeline.py:289`, after `remove_boilerplate_from_message()`.
The cleaned message is checked for `strip()` emptiness on line 290 but not for minimum length.

**Hidden dependencies found:**
- The `DocumentChunker` has `count_tokens()` but is not directly available in the pipeline
  function (it is on `components.attachment_processor.chunker`). Using a word count heuristic
  (`len(cleaned_message.split()) < 20`) is simpler and avoids coupling.
- The chunk index (`msg_idx`) is assigned from `enumerate(messages)`, so skipping a message
  will create gaps in chunk indices (e.g., 0, 2, 3). This is harmless — chunk_index is only
  for ordering, not identity. The `chunk_id` is computed from character positions.

**Edge cases:**
- A message with exactly 20 words that is all boilerplate could slip through. Acceptable — the
  existing `remove_boilerplate_from_message()` already handles most boilerplate patterns.
- Thread messages with only "From: X / Sent: Y / Subject: Z" headers after cleaning are
  already caught by `if not cleaned_message.strip(): continue` on line 290.

**Conflict with other changes:** None. If chunk size is later changed (PD-01), the minimum
content filter is independent — it operates on email body messages, not attachment chunks.

### P8-A — Date in Embedding Text: CONFIRMED FEASIBLE

**Verification:** `email_doc.email_metadata.date_start` is populated by the `EmlParser` and
available at the point where embedding text is constructed (`pipeline.py:304-306`). The
`EmailMetadata.date_start` field is `Optional[datetime]` so a None check is needed.

**Hidden dependencies found:** None. The date is already a first-class field on the document.

**Edge cases:**
- Some emails may have `date_start = None` if the email headers are malformed. The implementation
  should gracefully skip the date prefix in this case (which the proposed code does).
- For attachment chunks in `attachment_handler.py`, the email date is accessible via
  `email_doc.email_metadata.date_start`. No additional data passing needed.

**Conflict with other changes:** None. The date prefix is prepended before the context summary,
so it works regardless of whether P6 (attachment context inheritance) is also applied.

---

## Deferred Proposals: Deferral Confirmation

### P2 — Structured Metadata: DEFERRAL CORRECT

Would require: new extraction prompt, metadata schema in chunk JSONB, query-time filter logic,
UI filter components. This is at least 2-3 days of work and needs iteration on prompt quality.
Cannot be trivially included now. Does not create technical debt — can be added as a metadata
enrichment step later without re-embedding (metadata is stored separately from embedding text,
unless we also want structured terms in the embedding text, which would require re-embedding).

**Re-embedding consideration:** If structured metadata is eventually added to embedding text
(P2-C), it would require re-embedding. However, the structured metadata extraction itself can
be done as a post-processing step on existing chunks (just add to metadata JSONB). Only adding
it to embedding text requires re-ingest. Recommendation: when implementing P2, decide whether
to include structured terms in embedding text. If yes, batch it with the next re-ingest.

### P1-B/C — Adaptive Chunk Sizes: DEFERRAL CORRECT

Linked to PD-01 (512 vs 1024 token decision). Needs real-data testing. The chunk size is a
config setting (`settings.chunk_size_tokens`), so changing it is trivial, but validating
retrieval quality requires running comparison queries. Deferral is correct — do not change
chunk size without test validation.

### P8-B — Per-message Date Preservation: DEFERRAL CORRECT

Would require refactoring `split_into_messages()` to return `(date, content)` tuples instead
of plain strings. The function currently uses regex to detect boundaries like
"On DATE, NAME wrote:" but discards the captured date. Moderate refactor. Not urgent since
P8-A (email-level date) covers the most common case.

### P5 — Document Linking: DEFERRAL CORRECT

High complexity. The `EmailMetadata` model already has `in_reply_to` and `references` fields,
so header-based threading data is available. But building the incident entity, cross-thread
linking heuristics, and the threading UI are significant work. Correct to defer to Phase 3.

**Note:** The existing `in_reply_to` and `references` fields are stored but never used. When
implementing P5, these fields are the starting point — no schema changes needed for header-based
threading.

### P3 — Topic Taxonomy: DEFERRAL CORRECT

Requires maritime domain expertise to define the category hierarchy. The current free-form
topic system works well enough for initial ingest. No technical debt from deferring — the
free-form topics can be retroactively mapped to a taxonomy without re-ingesting.

### P7 — Multi-Document Aggregation: DEFERRAL CORRECT

Depends on P5 for grouping signal. Cannot be meaningfully implemented without incident
linking. Correct to defer.

### P8-C — Temporal Query Parsing: DEFERRAL CORRECT

This is a query-time feature, not an ingest feature. Can be added at any time without
affecting stored data. No urgency.

---

## P4 — Vessel Matching: Detailed Assessment

P4 was not explicitly approved or deferred in the initial decisions. Assessment:

### P4-A (Aliases): LOW EFFORT, NOT BLOCKING

Current state:
- `Vessel` model has: `name`, `vessel_type`, `vessel_class` — no `aliases` field
- `vessel-list.csv` has: `NAME;TYPE;CLASS` — no aliases column
- `VesselMatcher._build_lookup()` only indexes the primary name

Adding aliases requires:
1. Add `aliases: List[str] = []` to `Vessel` model
2. Add ALIASES column to `vessel-list.csv` (semicolon-separated within the field, or pipe-separated)
3. Update `_build_lookup()` to index aliases in addition to primary names
4. Populate aliases from domain knowledge (e.g., "MARAN CASTOR" -> "CASTOR", "M. CASTOR")

**Recommendation:** Note for future. Vessel IDs are stored in chunk metadata (not embedding
text), so aliases can be added at any time and retroactively applied via a re-tagging pass
(the `entities_cmd.py` already supports this). No re-embedding needed.

### P4-B (IMO Numbers): MEDIUM EFFORT, DEFERRED

Requires IMO numbers in the vessel registry (not currently available in `vessel-list.csv`)
and regex scanning in email text. Defer until vessel data is enriched.

### P4-C (Contextual Disambiguation): MEDIUM EFFORT, DEFERRED

Requires proximity analysis with maritime terms. Over-engineering for current corpus size.

### P4-D (Minimum Name Length): LOW EFFORT, QUICK WIN

Add `if len(name) < 4: continue` in `_build_lookup()`. Prevents false positives from short
vessel names matching common words. Worth doing but not urgent — review the vessel list to
see if any vessel names are shorter than 4 characters first.

---

## Issues Found and Addressed

### Issue 1: ZIP attachments miss context inheritance (P6)

`process_zip_attachment()` does not receive or pass through the email context summary.
When P6 is implemented, both `process_attachment()` and `process_zip_attachment()` must be
updated to accept and use `email_context_summary`.

**Resolution:** Noted in the implementation plan (RQ-1 step 4 mentions image chunks; ZIP
handler should be explicitly listed too).

### Issue 2: Minimum content filter creates chunk_index gaps (P1-A)

Skipping messages in the enumeration loop means `chunk_index` values have gaps. This is
harmless but may look unexpected in debug output.

**Resolution:** Accepted. chunk_index is for ordering only; chunk_id (hash-based) is the
stable identifier. No action needed.

### Issue 3: Word count vs token count for minimum filter (P1-A)

The proposal says "30 tokens" but the pipeline does not have a tokenizer readily available.
Using word count (~20 words to approximate 30 tokens) is simpler.

**Resolution:** Recommended word count approach in the implementation plan.

### Issue 4: Date format in embedding text (P8-A)

The ISO format `[Date: 2024-03-15]` is clear but alternatives exist (e.g., natural language
"March 15, 2024"). ISO format is preferred because:
- Embedding models handle it well
- It is unambiguous across locales
- It sorts correctly as a string

**Resolution:** Use ISO format `[Date: YYYY-MM-DD]` as specified.

---

## Proposals That Should Be Reconsidered

### P1-D (Late Chunking) — Worth Revisiting

The proposal suggests embedding whole documents for short emails (<2000 tokens). This is
worth revisiting because many email bodies are short (single message, <500 tokens). Instead
of splitting into chunks and prepending context to each, the entire email body could be one
chunk with context prepended. This would:
- Eliminate fragmentation for simple emails
- Reduce total chunk count
- Improve retrieval for single-message emails

However, this conflicts with the current approach where `split_into_messages` creates one
chunk per message in a thread. For single-message emails, this is already effectively "late
chunking" (one chunk = one message). The proposal mainly benefits short threads where the
thread splitting creates unnecessarily small fragments.

**Recommendation:** Keep deferred. The P1-A minimum content filter addresses the worst case
(very short messages). True late chunking needs architectural changes to the chunking pipeline.

### P4-D (Minimum Vessel Name Length) — Could Be Quick

A one-line change (`if len(name) < 4: continue`) with no side effects. Check the vessel
list first to ensure no legitimate vessel names are shorter than 4 characters. If confirmed
safe, add during any vessel matcher refactoring.

---

## Summary

| Proposal | Decision | Verified | Notes |
|----------|----------|----------|-------|
| P6 Attachment context | APPROVED | Yes, feasible | Also update ZIP handler |
| P1-A Min content filter | APPROVED | Yes, feasible | Use word count heuristic |
| P8-A Date in embedding | APPROVED | Yes, feasible | ISO format, handle None dates |
| P4-A Vessel aliases | NOTED | Yes, feasible later | No re-embedding needed |
| P2 Structured metadata | DEFERRED Phase 2 | Correct | May need re-embed if added to embedding text |
| P1-B/C Chunk sizes | DEFERRED Phase 2 | Correct | Linked to PD-01 |
| P8-B Per-message dates | DEFERRED Phase 2 | Correct | Needs split_into_messages refactor |
| P5 Doc linking | DEFERRED Phase 3 | Correct | in_reply_to/references already stored |
| P3 Topic taxonomy | DEFERRED Phase 3 | Correct | Needs domain expertise |
| P7 Multi-doc aggregation | DEFERRED Phase 3 | Correct | Depends on P5 |
| P8-C Temporal queries | DEFERRED Phase 3 | Correct | Query-time feature, no ingest impact |

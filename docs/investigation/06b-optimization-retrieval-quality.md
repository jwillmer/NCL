# Retrieval Quality Optimization Proposals

Investigation into ingest pipeline improvements focused on improving retrieval recall, precision, and user satisfaction for the MTSS Email RAG system.

**Date:** 2026-04-13
**Status:** Proposal
**Scope:** Research only (no code changes)

---

## Table of Contents

1. [Current Pipeline Summary](#current-pipeline-summary)
2. [Proposal 1: Chunking Strategy for Incident Reports](#proposal-1-chunking-strategy-for-incident-reports)
3. [Proposal 2: Structured Metadata Extraction](#proposal-2-structured-metadata-extraction)
4. [Proposal 3: Topic Taxonomy — Controlled Vocabulary](#proposal-3-topic-taxonomy--controlled-vocabulary)
5. [Proposal 4: Vessel Matching Improvements](#proposal-4-vessel-matching-improvements)
6. [Proposal 5: Document Relationship Linking](#proposal-5-document-relationship-linking)
7. [Proposal 6: Attachment Context Inheritance](#proposal-6-attachment-context-inheritance)
8. [Proposal 7: Multi-Document Incident Aggregation](#proposal-7-multi-document-incident-aggregation)
9. [Proposal 8: Temporal Context Preservation](#proposal-8-temporal-context-preservation)
10. [Priority Ranking](#priority-ranking)

---

## Current Pipeline Summary

| Component | Current Approach |
|---|---|
| Chunking | Email thread split into 1 chunk per message via regex; attachments chunked at 512 tokens with 50-token overlap using `RecursiveCharacterTextSplitter`/`MarkdownTextSplitter` |
| Context | 2-3 sentence LLM summary prepended to each chunk's embedding text |
| Topics | Free-form LLM extraction (1-5 per email), deduplicated via embedding similarity >= 0.85 |
| Vessels | Word-boundary regex matching against vessel registry names (case-insensitive) |
| Embeddings | `text-embedding-3-small`, 1536 dims, max 8000 tokens |
| Search | pgvector similarity -> optional metadata filter -> Cohere cross-encoder rerank (top 5) -> LLM generation |
| Citations | `[C:chunk_id]` markers validated in retry loop (max 2 retries) |

---

## Proposal 1: Chunking Strategy for Incident Reports

### Problem

Email body chunks are created per thread message, regardless of length. A short "OK, noted" reply becomes its own chunk with the same weight as a 2000-word incident description. Meanwhile, attachment chunks use fixed 512-token windows that can split a root-cause analysis mid-sentence.

Specific issues:

1. **Short, low-value messages pollute search results.** Acknowledgments ("Noted, thanks", "Please see attached") become standalone chunks that get the full context summary prepended, making them appear semantically rich when they carry no incident information.
2. **512-token attachment chunks may split incident details.** A PDF incident report with Problem, Cause, and Resolution sections could have "Resolution" split across two chunks, reducing retrieval precision for "what was done to fix it" queries.
3. **No minimum content threshold.** A 10-word boilerplate message passes through `remove_boilerplate_from_message` and still becomes a chunk.

### Proposal

**A. Minimum content filtering.** Skip chunks below a minimum meaningful content threshold (e.g., 30 tokens after boilerplate removal). These short messages rarely contain incident details.

**B. Adaptive chunk sizes for attachments.** Increase chunk size to 1024 tokens for structured documents (PDFs, DOCX) while keeping 512 for email bodies. Incident reports typically have sections (Problem Description, Root Cause, Corrective Action) that are 200-400 words each and should stay intact.

**C. Section-aware chunking for markdown attachments.** The `MarkdownTextSplitter` already splits on headings, but the current 512-token limit can still break mid-section. Use heading boundaries as hard split points up to 1024 tokens, only subdividing when a section exceeds that limit.

**D. Consider late chunking (embed full document, chunk later).** For shorter emails and attachments (< 2000 tokens), skip chunking entirely and embed the whole document as one chunk. This preserves all context for retrieval and avoids artificial fragmentation.

### Expected Impact

| Metric | Improvement |
|---|---|
| Precision | +15-25% (fewer junk chunks in results) |
| Recall | +5-10% (intact sections match more queries) |
| User satisfaction | Higher (fewer "noted, thanks" fragments in sources) |

### Cost & Complexity

- **LLM calls:** No change (chunking is local)
- **Storage:** ~10-20% fewer chunks (filtering short messages), but larger chunks
- **Embedding cost:** Roughly neutral (fewer chunks but slightly more tokens per chunk)
- **Implementation:** Low-Medium. Minimum filter is trivial. Adaptive chunk sizes require config changes and testing. Section-aware chunking needs `MarkdownTextSplitter` configuration tuning.

---

## Proposal 2: Structured Metadata Extraction

### Problem

The current context summary is a free-form 2-3 sentence text blob. It helps embeddings but misses structured fields that enable precise filtering and better LLM prompts. When a user asks "what engine failures happened on VLCCs last year?", the system must rely entirely on vector similarity to match "engine failure" concepts rather than filtering on structured fields.

Additionally, the context summary prompt (`ContextGenerator.CONTEXT_PROMPT`) asks for "document type, author, date, main topic" but does not extract domain-specific fields critical for maritime incident reports:

- Incident type (equipment failure, cargo damage, collision, grounding)
- Equipment involved (main engine, aux boiler, ballast pump, crane)
- Root cause (material fatigue, human error, design deficiency)
- Resolution status (completed, pending, temporary fix)
- Location/port
- Severity/classification

### Proposal

**A. Extract structured incident metadata via LLM at ingest time.** Add a dedicated extraction step (after context summary generation) that produces structured JSON:

```json
{
  "incident_type": "equipment_failure",
  "equipment": ["main engine", "fuel injection pump"],
  "root_cause": "material_fatigue",
  "resolution": "replaced_component",
  "resolution_status": "completed",
  "port": "Singapore",
  "severity": "operational_impact"
}
```

**B. Store structured metadata in chunk metadata JSONB.** This enables precise filtering at query time: `WHERE metadata->>'incident_type' = 'equipment_failure'`.

**C. Include structured metadata in embedding text.** Prepend key fields to the embedding text so vector search also benefits from explicit categorization.

**D. Use structured fields for faceted search in the UI.** Enable dropdown filters for incident type, equipment category, resolution status.

### Expected Impact

| Metric | Improvement |
|---|---|
| Precision | +20-30% (faceted filtering eliminates irrelevant categories) |
| Recall | +10-15% (structured terms catch vocabulary mismatches) |
| User satisfaction | Significant (faceted search, structured answers) |

### Cost & Complexity

- **LLM calls:** +1 LLM call per email (can combine with context generation to save calls)
- **Storage:** Minimal increase (JSONB fields in existing metadata column)
- **Implementation:** Medium. Requires new extraction prompt, metadata schema, query-time filter logic, and UI work for faceted search.
- **Risk:** LLM extraction quality varies; need fallback for emails that are not incident reports (e.g., routine correspondence, purchase orders).

---

## Proposal 3: Topic Taxonomy — Controlled Vocabulary

### Problem

The current free-form topic extraction creates a growing, uncontrolled taxonomy. The deduplication threshold (0.85 similarity) catches obvious duplicates ("Engine Failure" vs "Engine Failures") but misses semantic overlaps that a human would group together. Over time this leads to:

1. **Topic fragmentation.** "Main Engine Breakdown", "Engine Room Malfunction", "Propulsion System Failure", and "Engine Mechanical Fault" may all exist as separate topics with slightly different embeddings below the 0.85 threshold.
2. **Inconsistent granularity.** Some topics are very broad ("Maintenance") while others are very specific ("Cargo Hold Bilge Pump Suction Valve Replacement").
3. **No hierarchical structure.** "Auxiliary Boiler Flame Failure" cannot be browsed under a parent category like "Boiler Issues" > "Auxiliary Boiler".

The query-time matching threshold (0.70) is lenient enough to partially compensate, but this shifts the problem to false positives in topic matching.

### Proposal

**A. Define a two-level maritime incident taxonomy.** Create a curated list of ~30-50 categories based on industry standards:

- Level 1 (10-15): Hull & Structure, Propulsion, Electrical, Navigation, Cargo, Safety Equipment, Environmental, Manning, Regulatory, Commercial
- Level 2 (3-5 per L1): e.g., under Propulsion: Main Engine, Auxiliary Engine, Propeller/Shaft, Fuel System, Exhaust System

Reference sources: IMO casualty categories, SIRE/CDI inspection areas, P&I loss categories.

**B. Map free-form LLM topics to the taxonomy.** Keep the LLM extraction as-is (it captures nuances), but add a mapping step that assigns each extracted topic to one or more taxonomy categories. This can be done via:
- Embedding similarity to taxonomy category descriptions (cheapest)
- LLM classification call with the taxonomy as context (most accurate)

**C. Store both the free-form topic and the taxonomy mapping.** The free-form topic is useful for display and nuanced search. The taxonomy category enables hierarchical browsing and consistent filtering.

**D. Allow taxonomy to grow via admin curation.** When the mapper fails (low confidence), flag for human review. New categories can be added to the taxonomy over time.

### Expected Impact

| Metric | Improvement |
|---|---|
| Precision | +10-20% (consistent categories reduce fragmentation) |
| Recall | +15-25% (hierarchical categories catch broader queries) |
| User satisfaction | High (browsable category tree, predictable filtering) |

### Cost & Complexity

- **LLM calls:** +0-1 per email (can use embedding similarity for mapping, avoiding extra LLM call)
- **Storage:** New taxonomy table, mapping table
- **Implementation:** Medium-High. Defining the taxonomy requires maritime domain expertise. Mapping logic needs testing. UI needs category browser.
- **Risk:** Taxonomy may not cover all edge cases initially; some emails genuinely don't fit a category.

---

## Proposal 4: Vessel Matching Improvements

### Problem

The current `VesselMatcher` uses word-boundary regex (`\b` + vessel name + `\b`) against a lookup of primary vessel names. This works well for standard references but has gaps:

1. **No alias/abbreviation support.** Vessels are often referred to by abbreviations, informal names, or hull numbers. "MARAN CASTOR" might appear as "M. CASTOR", "CASTOR", or hull number "H-2340". The current lookup only stores the primary name.
2. **No IMO number matching.** IMO numbers (7 digits, often prefixed "IMO" or "IMO No.") are definitive vessel identifiers but are not used in matching.
3. **No fleet-level grouping.** "Maran fleet" or "all VLCCs" is not matched to specific vessels (though `vessel_type` filtering partially covers this).
4. **Word-boundary issues with special characters.** Vessel names containing dots, hyphens, or parentheses (e.g., "MARAN PLATO (ex-OLYMPIC PLATO)") may not match correctly due to regex word boundary behavior with special characters.
5. **Short name false positives.** A 3-4 letter vessel name could match common words. The current code doesn't have a minimum name length check.

### Proposal

**A. Add vessel aliases.** Extend the `Vessel` model with an `aliases` field (list of strings). Populate from vessel registry data or learn from ingested text. Build aliases into the lookup.

**B. Add IMO number matching.** Scan for patterns like `IMO\s*\d{7}` or standalone 7-digit numbers near vessel context words. Cross-reference against vessel registry.

**C. Contextual vessel disambiguation.** For short or ambiguous names, require contextual proximity to maritime terms (e.g., "vessel", "ship", "MT", "MV", "v/") within N words.

**D. Add minimum name length threshold.** Skip word-boundary matching for names shorter than 4 characters to avoid false positives (or require them to appear with a vessel prefix like "MV" or "MT").

### Expected Impact

| Metric | Improvement |
|---|---|
| Precision | +5-10% (fewer false positive vessel matches) |
| Recall | +10-20% (aliases and IMO numbers catch missed references) |
| User satisfaction | Moderate (correct vessel attribution) |

### Cost & Complexity

- **LLM calls:** None
- **Storage:** Small increase (aliases column in vessel table)
- **Implementation:** Low-Medium. Alias lookup is straightforward. IMO matching requires regex + registry lookup. Contextual disambiguation adds complexity.

---

## Proposal 5: Document Relationship Linking

### Problem

The same incident is often discussed across multiple emails over days or weeks. Currently, each email is processed independently. There is no way to:

1. Find all emails related to a specific incident
2. See the timeline of an incident from report to resolution
3. Aggregate knowledge from multiple emails about the same event

The `EmailMetadata` model has `in_reply_to` and `references` fields from email headers, but these are stored and never used for linking.

### Proposal

**A. Build an incident thread graph using email references.** Use `In-Reply-To` and `References` headers to link emails in the same thread. This creates a thread tree that can be traversed to find all related messages.

```
Email A (initial report)
  ├── Email B (follow-up inspection)
  │     └── Email C (inspection results)
  └── Email D (superintendent response)
        └── Email E (resolution confirmation)
```

**B. Cross-thread incident linking via subject similarity.** Emails about the same incident may not share RFC headers (forwarded from different sources, different mail clients). Use subject line similarity (after stripping RE:/FW: prefixes) + vessel match + temporal proximity (within 30 days) to suggest linkages.

**C. Create an "incident" entity.** Group related emails into incidents with:
- Incident ID
- Title (derived from first email subject)
- Vessel(s)
- Date range (first report to last update)
- Status (open/closed)
- Linked email IDs

**D. Inject cross-reference context into chunks.** When embedding a chunk, prepend "This is part of an incident thread with N related emails spanning DATE_START to DATE_END" to improve retrieval for temporal and aggregation queries.

### Expected Impact

| Metric | Improvement |
|---|---|
| Recall | +20-30% (query about incident finds all related emails) |
| Precision | +10-15% (incident context disambiguates similar-looking chunks) |
| User satisfaction | Very high (complete incident timeline, not just fragments) |

### Cost & Complexity

- **LLM calls:** +0-1 per email for cross-thread linking (subject similarity can use embeddings)
- **Storage:** New incident table, email-incident mapping table
- **Implementation:** High. Header-based linking is straightforward. Cross-thread heuristics require tuning. Incident entity creation requires either automated or semi-manual workflow. Re-embedding of existing chunks when incident links are discovered.
- **Risk:** False thread linkages could pollute retrieval. Needs conservative linking thresholds.

---

## Proposal 6: Attachment Context Inheritance

### Problem

Attachment chunks currently receive:
- Their own context summary (generated from attachment content alone)
- Vessel IDs from the parent email
- Topic IDs from the parent email

However, the attachment context summary (`ContextGenerator`) only sees the attachment content, not the parent email context. A PDF inspection report attached to an email titled "RE: Main Engine Cylinder 3 Crack - MARAN CASTOR" would be summarized as "This is a PDF document containing an inspection report with tables and measurements" without mentioning the vessel or the specific engine problem.

When a user searches for "cylinder crack inspection on MARAN CASTOR", the email body chunk matches well, but the attachment chunk (which has the actual inspection data) may score lower because its embedding text lacks the incident context.

### Proposal

**A. Inherit parent email context in attachment embedding text.** Prepend the parent email's context summary (or a condensed version) to each attachment chunk's embedding text. This connects the attachment content to the incident context.

Current attachment embedding text:
```
[attachment context summary]
[chunk content]
```

Proposed attachment embedding text:
```
[parent email context: "Email about main engine cylinder 3 crack on MARAN CASTOR, reported by Chief Engineer on 2024-03-15"]
[attachment context summary]
[chunk content]
```

**B. Include parent email subject in attachment context generation prompt.** Pass the email subject and initiator to the `ContextGenerator` when generating attachment context so the LLM can reference it.

**C. Store parent email metadata on attachment chunks.** The `email_subject`, `email_date`, `email_initiator` fields are already available in `RetrievalResult` via the search query but are populated from the document hierarchy join. Denormalizing them onto the chunk metadata would make them available for the embedding text without an extra DB lookup.

### Expected Impact

| Metric | Improvement |
|---|---|
| Recall | +15-25% (attachment chunks match incident-specific queries) |
| Precision | +5-10% (attachment context disambiguates generic document content) |
| User satisfaction | High (relevant attachments surface alongside email body) |

### Cost & Complexity

- **LLM calls:** None (reuse existing context summary from parent email)
- **Storage:** Small increase (parent context stored in chunk embedding_text)
- **Embedding cost:** Slightly more tokens per chunk due to prepended context
- **Implementation:** Low. The parent email context summary is already computed before attachment processing. Pass it through and prepend.

---

## Proposal 7: Multi-Document Incident Aggregation

### Problem

When the LLM generates an answer about an incident, it sees individual chunks from potentially different emails. It has no signal that these chunks belong to the same incident or should be synthesized into a coherent narrative. The LLM may:

1. Present contradictory information from different stages of the incident without noting the temporal sequence
2. Miss that a "resolution" chunk from a later email answers the "problem" described in an earlier email
3. Cite 5 chunks from 3 emails without indicating they're all about the same incident

### Proposal

**A. Post-retrieval incident grouping.** After retrieval and reranking, group results by incident (using email thread linking from Proposal 5 or subject/vessel similarity). Present grouped chunks to the LLM with explicit incident headers:

```
=== Incident: Main Engine Cylinder 3 Crack (MARAN CASTOR, 2024-03-15 to 2024-03-22) ===

[Chunk 1 - Initial Report, 2024-03-15]
...
[Chunk 2 - Inspection Report, 2024-03-17]
...
[Chunk 3 - Resolution, 2024-03-22]
...
```

**B. Chronological ordering within groups.** Sort chunks within each incident group by date (email date or message index). This helps the LLM construct temporal narratives.

**C. Update the RAG system prompt.** Add instructions for the LLM to synthesize incident timelines and distinguish between initial reports, follow-ups, and resolutions.

### Expected Impact

| Metric | Improvement |
|---|---|
| Answer quality | +25-35% (coherent narrative vs. fragmented facts) |
| User satisfaction | Very high (temporal incident stories) |
| Citation quality | +10-15% (grouped citations make more sense) |

### Cost & Complexity

- **LLM calls:** None (restructuring happens post-retrieval, pre-generation)
- **Storage:** None (uses existing date metadata)
- **Implementation:** Medium. Requires incident grouping logic (depends on Proposal 5). Prompt engineering for temporal synthesis. Testing with real incident threads.
- **Dependency:** Benefits greatly from Proposal 5 (Document Relationship Linking). Without it, grouping must rely on heuristics.

---

## Proposal 8: Temporal Context Preservation

### Problem

Date/time context is partially preserved but has gaps:

1. **Email dates are stored but not in chunk embedding text.** The embedding for "we replaced the pump" doesn't include when it happened, so "what was done about the pump last year" relies entirely on vector similarity without temporal signal.
2. **Thread message dates may be lost.** `split_into_messages` extracts messages but the `remove_boilerplate_from_message` function removes `Date:`, `Sent:` header lines. The individual message dates are not passed to the chunk.
3. **No relative temporal indicators.** "Last week we noticed..." in an email from 2024-03-15 means the event was around 2024-03-08, but this context is lost in the chunk.
4. **Attachment dates are not preserved.** An inspection report PDF might have its own date that differs from the email date.
5. **The `email_date` field in `RetrievalResult` is populated from the email document, but not included in the embedding text or the context summary systematically.**

### Proposal

**A. Include date in embedding text.** Prepend the email date (or message-specific date when available) to each chunk's embedding text. This gives the embedding model temporal signal:

```
[Date: 2024-03-15]
[Context: Email about main engine cylinder crack on MARAN CASTOR...]
[Chunk content]
```

**B. Preserve per-message dates.** When `split_into_messages` detects a message boundary with a date, pass that date to the chunk metadata rather than stripping it during boilerplate removal.

**C. Add date-range filtering at query time.** Parse temporal expressions in user queries ("last year", "in 2023", "past 6 months") and convert to date filters for the vector search. This is more efficient and accurate than relying on embedding similarity for temporal matching.

**D. Include date in context summary prompt.** The `ContextGenerator` already includes `Date: {date_start}` in hints, but make it mandatory rather than optional.

### Expected Impact

| Metric | Improvement |
|---|---|
| Recall | +10-15% for temporal queries |
| Precision | +15-20% with date-range filtering |
| User satisfaction | High (accurate temporal answers) |

### Cost & Complexity

- **LLM calls:** None (dates are already available metadata)
- **Storage:** Minimal (date in chunk metadata)
- **Embedding cost:** Negligible increase from prepended date
- **Implementation:** Low-Medium. Date prepending is trivial. Temporal query parsing requires NLP or LLM-based date extraction.

---

## Priority Ranking

Ranked by expected quality improvement relative to implementation effort:

| Priority | Proposal | Impact | Effort | Rationale |
|---|---|---|---|---|
| 1 | **P6: Attachment Context Inheritance** | High | Low | Quick win. Parent email context already computed; just pass it to attachment chunks. Directly improves retrieval for the core use case (finding incident details in attachments). |
| 2 | **P1: Chunking Strategy** | High | Low-Med | Filtering short messages and increasing attachment chunk size are low-effort changes with immediate precision improvement. |
| 3 | **P8: Temporal Context** | Med-High | Low-Med | Adding dates to embedding text is trivial and directly addresses "what happened last time" queries. Date-range filtering can be deferred. |
| 4 | **P2: Structured Metadata** | High | Medium | Biggest precision win via faceted filtering, but requires careful prompt engineering and schema design. Combine with context generation LLM call to minimize cost. |
| 5 | **P4: Vessel Matching** | Med | Low-Med | Alias support and IMO matching are straightforward improvements. Low risk. |
| 6 | **P5: Document Relationship Linking** | Very High | High | Highest overall impact but most complex. Start with header-based threading (low effort), defer cross-thread heuristics. |
| 7 | **P3: Topic Taxonomy** | Med-High | Med-High | Valuable for long-term consistency but requires domain expertise and is less urgent than retrieval fixes. |
| 8 | **P7: Multi-Document Aggregation** | Very High | Medium | High impact on answer quality but depends on P5 for grouping signal. Implement after P5. |

### Recommended Implementation Phases

**Phase 1 (Quick Wins, 1-2 weeks):**
- P6: Attachment context inheritance (pass parent email context summary to attachment chunks)
- P1-A: Minimum content threshold for email body chunks
- P8-A: Date prepending in embedding text

**Phase 2 (Core Improvements, 2-4 weeks):**
- P1-B/C: Adaptive chunk sizes for attachments
- P2: Structured metadata extraction (combine with context generation call)
- P4: Vessel alias support and IMO number matching
- P8-B: Per-message date preservation

**Phase 3 (Advanced Features, 4-8 weeks):**
- P5: Document relationship linking (start with header-based threading)
- P3: Topic taxonomy definition and mapping
- P7: Post-retrieval incident grouping (after P5)
- P8-C: Temporal query parsing

### Cost Summary (All Proposals)

| Resource | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Additional LLM calls per email | 0 | +0-1 | +0-1 |
| Additional embedding calls | 0 | 0 | 0 |
| Storage increase | ~0% | ~5-10% | ~10-15% |
| Re-ingest required | Partial (re-embed) | Partial | Full |
| New DB tables | 0 | 0 | 2-3 |

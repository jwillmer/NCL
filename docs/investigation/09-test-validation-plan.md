---
purpose: Test subset selection and validation plan for small-batch ingest before full 6,289-EML production run
status: research-complete
date: 2026-04-13
---

# Test Validation Plan: Subset Ingest & Quality Check

## 1. Selected Test Documents (15 files)

The following emails were selected from `data/emails/` to cover all attachment types, content patterns, and edge cases the pipeline must handle.

### Group A: Incident / Technical Reports (core domain)

| # | Filename | Why Selected |
|---|----------|-------------|
| 1 | `100000922_vp4qcvdw.abx.eml` | **Vessel name in subject + long thread + attachments.** Subject: "MARAN CANOPUS ELREP MAIN ENGINE GOVERNOR FAILURE". Has `X-MS-Has-Attach: yes`, multipart/mixed with photos. References chain has 15+ message IDs -- stress-tests threading. The word "FAILURE" validates incident detection. Vessel name "MARAN CANOPUS" validates vessel matcher. |
| 2 | `100000366_qhlztvyi.alr.eml` | **Equipment repair report with PDF/image attachments.** Subject: "DANAE ELREP STARTER PANEL OF M/E AUX BLOWER No2". Has `X-MS-Has-Attach: yes`. Contains Keywords header (`ELREP,MTM,TECHNICALMTM`). In-Reply-To present -- tests threading detection. Vessel name "DANAE" in subject. |
| 3 | `100000660_vpirjft0.vyg.eml` | **Long email thread (15+ References).** Subject: "MARAN CANOPUS ELREP MAIN ENGINE GOVERNOR FAILURE". Multiple References headers spanning weeks of conversation. Attachment-bearing. Tests the In-Reply-To/References thread reconstruction. |
| 4 | `100000602_krtofd3g.nuw.eml` | **Classification/survey email with vessel IMO number.** Subject: "MARAN HELEN IMO 9779381 CLASS LR INCINERATOR". From classification society (Lloyd's Register). Contains IMO number -- useful for future entity extraction. Vessel name "MARAN HELEN" in subject. |

### Group B: Attachment Diversity

| # | Filename | Why Selected |
|---|----------|-------------|
| 5 | `100000440_0lbrmluh.10x.eml` | **Text-only reply (no attachments, multipart/alternative).** Subject: "DANAE ELREP STARTER PANEL OF M/E AUX BLOWER No2". `X-MS-Has-Attach:` is empty (no value). Content is base64 plain text. Tests the simple-email path with no attachment processing. |
| 6 | `100000376_lgqfor05.2fi.eml` | **Forwarded email with attachments.** Subject: "CAPRICORN ENRER ME CHAIN MOMENT COMPENSATOR". `X-MS-Has-Attach: yes`. multipart/mixed. Multiple CC recipients (7+). Keywords: `ARCTURUS,DDOCK,MTM`. Tests large participant list extraction. |
| 7 | `100284018_5vcg2bgp.jwm.eml` | **Logistics/shipping email with attachments.** Subject: "PRE ALERT- MARS - AMS / SIN". `X-MS-Has-Attach: yes`. Content-Type: multipart/related. Tests the multipart/related parsing path (inline images). Vessel name "MARS" in subject. |
| 8 | `100297210_2ef23hqq.cni.eml` | **Safety equipment email with attachments.** Subject: "THETIS SAFETY FFE - LOW EXPANSION FOAM REPLACEMENT". `X-MS-Has-Attach: yes`. Keywords: `Thetis`. Long References chain (10+ messages). Tests safety topic extraction. |

### Group C: Edge Cases & External Senders

| # | Filename | Why Selected |
|---|----------|-------------|
| 9 | `100000546_ccinfj2i.kew.eml` | **External sender (Wilhelmsen agent) with many recipients.** From: `ju-yong.lee@wilhelmsen.com`. 10+ CC recipients. Subject contains vessel name "ATHENA". Tests external-sender metadata, large participant lists. |
| 10 | `100000566_bhofcmo4.zug.eml` | **Non-maritime spam/newsletter.** From: "EAMA - European Anger Management Association". Subject is base64-encoded Greek text. Tests that the pipeline handles non-relevant content gracefully (topic extraction should yield nothing maritime). |
| 11 | `100293054_ueg1nbfp.amu.eml` | **Marketing email (Qatar Airways).** Subject is base64-encoded Greek. No In-Reply-To. Tests non-maritime content filtering. Pipeline should process but yield low-value chunks. |
| 12 | `100297520_gts3adxm.eid.eml` | **PSC (Port State Control) safety bulletin.** Subject: "[Korean Register/PSC Information] Lifeboat Release Mechanism". From: `psc@krs.co.kr`. External regulatory content. Tests topic extraction for safety/compliance topics. |
| 13 | `100297780_avnjkyr4.40y.eml` | **Worklogs email with attachments.** Subject: "HERMIONE WORKLOGS FOR week 26/25". `X-MS-Has-Attach: yes`. multipart/related. Tests operational/routine content. Vessel name "HERMIONE" in subject. |

### Group D: Pre-built Test Fixtures

| # | Filename | Why Selected |
|---|----------|-------------|
| 14 | `tests/fixtures/test_email.eml` | **Synthetic test email with PDF + ZIP + PNG attachments.** Subject: "MV TEST VESSEL EQUIPMENT INSPECTION". Contains all three attachment types in one email. Already used by `TestRealEmailParsing`. Includes multi-message thread in body text. Known-good baseline. |
| 15 | `100040671_drxriffw.1u2.eml` | **Third-party operations email.** Subject: "ARES, SHANGHAI- SPORE". From: SwiftMarine agent. Tests external sender with vessel "ARES" in subject. |

### Coverage Matrix

| Criterion | Files |
|-----------|-------|
| PDF attachment | 1, 2, 6, 7, 8, 14 |
| Image attachment | 1, 3, 14 |
| ZIP attachment | 14 |
| Office doc (.xls/.doc/.ppt) | 7, 8, 13 (if present in MIME parts) |
| Text-only email | 5, 10, 11 |
| Email thread (In-Reply-To) | 1, 2, 3, 4, 6, 8 |
| Vessel name in subject | 1, 2, 3, 4, 7, 8, 9, 13, 15 |
| Incident/problem described | 1, 3, 4 |
| External sender | 9, 10, 11, 12, 15 |
| Non-maritime content | 10, 11 |
| Base64-encoded subject | 10, 11 |

---

## 2. Test Scenarios

### Scenario 1: Basic Ingest Pipeline (Parse -> Chunk -> Embed -> Store)

**Input:** Files 5 (text-only) and 14 (multi-attachment)

**Expected behavior:**
- EML parser extracts subject, participants, date, message body
- Chunker splits body into chunks with line numbers and char positions
- Context generator produces a 1-2 sentence summary
- Embedding generator creates 1536-dim vectors
- Documents and chunks are stored (locally via `LocalIngestOutput`)

**Validation checks:**
- `documents.jsonl` has one document per email + one per attachment
- `chunks.jsonl` has at least 1 chunk per document
- Each chunk has non-null `content`, `chunk_index`, `line_from`, `line_to`
- `context_summary` is non-empty for each chunk
- `embedding_text` = context_summary + "\n\n" + content
- File hash is deterministic (run twice, same hash)
- `doc_id` is deterministic (same source_id + hash = same doc_id)

### Scenario 2: Local-Only Ingest Output Format

**Input:** Files 1, 5, 14

**Expected behavior:**
- JSONL files created in `{output_root}/database/` directory
- Archive files created in `{output_root}/bucket/` directory
- `documents.jsonl` lines are valid JSON with required fields
- `chunks.jsonl` lines are valid JSON (embeddings may be omitted for size)
- Manifest/status updates written to `status_updates.jsonl`
- Ingest events logged to `ingest_events.jsonl`

**Validation checks:**
- All JSONL files parse without errors
- Document count = number of emails + number of extracted attachments
- Each document has `id`, `source_id`, `doc_id`, `document_type`, `file_hash`
- Bucket directory has `{doc_id}/email.eml.md` for each email
- Bucket directory has `{doc_id}/attachments/{filename}.md` for each attachment with parsed content

### Scenario 3: Attachment Processing

**Input:** File 14 (PDF + ZIP + PNG), Files 1/2 (equipment photos)

**Expected behavior:**
- PDF attachments: parsed via pypdf (simple) or LlamaParse (complex), text extracted, chunked
- ZIP attachments: extracted recursively (max depth 3), inner files processed
- Image attachments: classified, optionally described via Vision API, stored
- Attachment documents linked to parent email via `parent_id`/`root_id`

**Validation checks:**
- Attachment documents have `depth=1`, `parent_id=email.id`
- PDF chunks contain text content (not raw bytes)
- ZIP inner files appear as separate documents with `depth >= 2`
- Image documents have `document_type=attachment_image`
- `archive_download_uri` points to the original file in bucket
- Hidden files (`.DS_Store`, `__MACOSX/`) are excluded from ZIP extraction

### Scenario 4: Vessel Matching

**Input:** Files 1 (MARAN CANOPUS), 4 (MARAN HELEN), 8 (THETIS), 9 (ATHENA)

**Expected behavior:**
- Vessel names extracted from subject and/or body
- Matched against the vessel lookup table
- Vessel IDs stored in chunk metadata

**Validation checks:**
- Chunks from file 1 have `vessel_ids` containing the ID for "MARAN CANOPUS"
- Chunks from file 4 have `vessel_ids` containing the ID for "MARAN HELEN"
- Case-insensitive matching works (subjects use all-caps)
- Word-boundary matching avoids false positives
- Empty/None subjects do not crash

### Scenario 5: Topic Extraction

**Input:** Files 1 (engine failure), 4 (incinerator), 8 (safety equipment), 12 (lifeboat safety)

**Expected behavior:**
- LLM extracts 1-3 maritime topics per email
- Topics are matched or created in the topic table
- Topic IDs are associated with chunks

**Validation checks:**
- File 1 yields topics related to "Engine Failure" or "Mechanical Breakdown"
- File 4 yields topics related to "Incinerator" or "Classification Survey"
- File 8 yields topics related to "Safety Equipment" or "Fire Fighting Equipment"
- File 12 yields topics related to "Lifeboat Safety" or "Port State Control"
- Non-maritime emails (files 10, 11) yield empty or generic topics
- Topic deduplication works (similar names merge)

### Scenario 6: Context Summary Generation

**Input:** Files 5 (simple text), 1 (complex technical report)

**Expected behavior:**
- Context generator produces a concise 1-2 sentence summary
- Summary includes date, participants, and topic hint
- Summary is prepended to chunk content for embedding

**Validation checks:**
- Context is non-empty string
- Context mentions the email date (or "unknown" if missing)
- Context mentions the sender or subject
- `embedding_text` = context + "\n\n" + content (for non-empty context)
- Fallback to metadata-based context on LLM failure

### Scenario 7: Document Hierarchy

**Input:** File 14 (email with 3 attachments)

**Expected behavior:**
- Root document (email): `depth=0`, `parent_id=None`, `root_id=self.id`
- Attachment documents: `depth=1`, `parent_id=root.id`, `root_id=root.id`
- ZIP inner files: `depth=2`, `parent_id=zip_doc.id`, `root_id=root.id`
- `path` array traces ancestry

**Validation checks:**
- Email document has empty parent chain
- Each attachment's `root_id` equals the email's `id`
- `path` array length = `depth + 1`
- `source_id` follows the format `{email_filename}/{attachment_filename}`

### Scenario 8: Reranker Bug Fix Validation

**Input:** Any query against ingested data

**Expected behavior (CURRENT BUG):**
- `agent.py:460` calls `search_only(top_k=settings.rerank_top_n)` which sets `top_k=5`
- This means only 5 candidates are retrieved from the vector DB
- The reranker then reranks those 5 results and returns `top_n=5` of them
- Since `top_k == rerank_top_n`, the reranker adds no value (it just reorders 5 results)

**Expected behavior (AFTER FIX):**
- `top_k` should be significantly larger than `rerank_top_n` (e.g., `top_k=20`, `rerank_top_n=5`)
- The reranker should see 20 candidates and select the best 5
- This gives the reranker a meaningful pool to choose from

**Test assertion:**
```python
# In the search call, top_k MUST be > rerank_top_n
assert top_k > rerank_top_n, (
    f"top_k ({top_k}) must be > rerank_top_n ({rerank_top_n}) "
    f"or the reranker is silently disabled"
)
```

**Where to test:** Add a test in `test_query_engine.py` that verifies the `search_similar_chunks` call receives `top_k >= 20` when reranking is enabled.

### Scenario 9: Date Filtering (Future Implementation)

**Input:** Files spanning different dates (June 21, June 23, July 1, 2025)

**Expected behavior (once implemented):**
- Queries like "incidents in June 2025" filter to June emails only
- Date extracted from email headers or body, stored in chunk metadata
- `match_chunks` function accepts date range parameters

**Validation checks:**
- Each document has `email_metadata.date_start` extracted from the Date header
- Chunks have date information available in metadata or embedding text
- A date-filtered search excludes results outside the range
- Edge case: emails with no Date header get a sentinel or NULL date

### Scenario 10: Search + Retrieval Quality

**Input:** Ingest files 1-8, then query

**Sample queries and expected results:**

| Query | Expected Top Result Source | Why |
|-------|--------------------------|-----|
| "main engine governor failure on Maran Canopus" | File 1 or 3 | Exact match on vessel + incident |
| "incinerator issues on VLCC vessels" | File 4 | Direct topic match |
| "starter panel repair blower" | Files 2, 5 | Technical repair content |
| "safety equipment foam replacement" | File 8 | Safety/FFE topic |
| "lifeboat release mechanism inspection" | File 12 | PSC bulletin content |
| "spare parts request Hermes" | File 15 (if present) | Procurement content |

**Validation checks:**
- Top-1 result has similarity score > 0.5
- Results include source attribution (file path, subject)
- Results from non-maritime emails (10, 11) do NOT appear for maritime queries
- Vessel-filtered queries reduce result set appropriately

### Scenario 11: Citation Accuracy

**Input:** Run `query_with_citations` against ingested subset

**Expected behavior:**
- LLM response contains `[C:chunk_id]` markers
- Citation processor validates each chunk_id exists in the citation map
- `<cite>` tags are generated with correct attributes
- Sources section lists all cited sources with page/line references

**Validation checks:**
- No `[C:...]` markers remain in the final response (all replaced with `<cite>`)
- Each citation's `chunk_id` maps to a real chunk in the database
- `archive_browse_uri` and `archive_download_uri` are present and valid
- No double `/archive/` prefix in URIs
- No double-encoding (`%2520`) in URIs
- Invalid citations (chunk_id not in map) are stripped
- `needs_retry` triggers when >50% of citations are invalid

---

## 3. Existing Test Coverage Gap Analysis

### What IS Covered (17 test files, ~180 tests)

| Area | Test File | Coverage |
|------|-----------|----------|
| EML parsing | `test_eml_parser.py` | Headers, body, attachments, BOM handling, real email fixture |
| Attachment processing | `test_attachment_processor.py` | ZIP extraction, path traversal, supported formats |
| Document chunking | `test_ingest_processing.py` | Empty/short/long text, headings, line numbers, char positions |
| Context generation | `test_ingest_processing.py` | Success, fallback, retry, embedding text composition |
| Embeddings | `test_ingest_processing.py` | Single, batch, truncation, empty list |
| Hierarchy management | `test_ingest_processing.py` | Email doc creation, attachment doc creation, ancestry |
| Vessel matching | `test_vessel_matcher.py` | Subject, body, case-insensitive, multi-vessel, edge cases |
| Topic extraction | `test_topics.py` | Extract from content, from query, error handling, dedup |
| Topic filtering | `test_topic_filter.py` | No match, partial match, empty topics, vessel filter |
| Reranker | `test_reranker.py` | Init, rerank results, disabled, custom top_n, metadata preservation |
| Query engine | `test_query_engine.py` | Query, no results, reranking, search_only, context header |
| Citation processor | `test_citation_processor.py` | Header format, context building, validation, retry, markers |
| Ingest flow | `test_ingest_flow.py` | Full pipeline integration, error handling, hierarchy |
| Version manager | `test_version_manager.py` | Insert, skip, reprocess, update decisions |
| Archive generator | `test_archive_generator.py` | Markdown gen, attachment archives, URI construction |
| Archive URIs | `test_archive_uris.py` | No double prefix, no double encoding, chunk propagation |
| Ingest consistency | `test_ingest_consistency.py` | Deterministic output, doc comparison, chunk comparison |
| Ingest storage | `test_ingest_storage.py` | Supabase client CRUD, issue tracker, helpers |
| Update flow | `test_ingest_update_flow.py` | Orphan detection, repair operations |
| Local storage | `local_storage.py` | JSONL write/read, bucket storage mock |

### What is NOT Covered (Gaps)

| Gap | Priority | Why It Matters |
|-----|----------|----------------|
| **Reranker bug: top_k == rerank_top_n** | CRITICAL | `agent.py:460` passes `top_k=rerank_top_n` which silently disables the reranker. No test catches this. |
| **End-to-end ingest with real EML files** | HIGH | All existing ingest tests use synthetic fixtures. No test feeds a real `.eml` file through parse -> chunk -> embed -> store. |
| **Local-only ingest output validation** | HIGH | `LocalIngestOutput` exists but no test verifies the complete local pipeline produces valid JSONL + bucket structure. |
| **Attachment context enrichment** | HIGH | Investigation 06b found attachment chunks miss parent email context. No test verifies context propagation from parent email to attachment chunks. |
| **Low-token chunk filtering** | MEDIUM | Investigation 06b found <30-token chunks pollute results. No test verifies filtering or flagging. |
| **Date extraction into embedding text** | MEDIUM | Investigation 06b found dates absent from embedding text. No test verifies date presence. |
| **Threading via In-Reply-To/References** | MEDIUM | Headers are stored but unused. No test verifies thread reconstruction. |
| **Image pre-filtering heuristic** | LOW | Investigation 06a found it exists but is unused in pipeline. No test for integration. |
| **Parallel attachment processing** | LOW | Investigation 06c identified this optimization. No performance/concurrency test. |
| **Date filtering in search** | MEDIUM | Investigation 07b found no date filtering in `match_chunks`. No test for date range queries. |
| **Vessel name extraction from query** | MEDIUM | Investigation 07b found this is missing. No test. |
| **Aggregation queries** | LOW | Investigation 07b found no aggregation path. No test. |
| **Full-text search** | MEDIUM | Investigation 07a found no full-text index. No test for keyword fallback. |
| **Non-ASCII/BOM in real emails** | LOW | BOM test exists for synthetic emails but no test with real BOM-corrupted EMLs from the corpus. |
| **Config path mismatch** | LOW | Investigation 03 found DATA_SOURCE_DIR vs actual path discrepancy. No test validates config consistency. |

---

## 4. Recommended Test Additions

### 4.1 CRITICAL: Reranker Top-K Bug Test

File: `tests/test_query_engine.py`

```python
async def test_reranker_gets_larger_candidate_pool(patches):
    """top_k passed to search_similar_chunks must exceed rerank_top_n.

    Regression test for agent.py:460 bug where top_k=rerank_top_n
    effectively disables the reranker by giving it no extra candidates.
    """
    patches.db.search_similar_chunks = AsyncMock(return_value=_make_db_rows(20))

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Answer."))]
        )
        engine = RAGQueryEngine()
        await engine.query("test query", use_rerank=True)

    # Verify search was called with top_k > rerank_top_n
    call_kwargs = patches.db.search_similar_chunks.call_args.kwargs
    top_k = call_kwargs.get("top_k", call_kwargs.get("match_count", 20))
    rerank_top_n = patches.settings.rerank_top_n  # default 3

    assert top_k > rerank_top_n, (
        f"top_k ({top_k}) must be > rerank_top_n ({rerank_top_n}) "
        f"or the reranker is silently disabled"
    )
```

### 4.2 HIGH: End-to-End Local Ingest Test

File: `tests/test_ingest_flow.py` (new class)

```python
class TestLocalIngestEndToEnd:
    """End-to-end test: real EML -> parse -> chunk -> local storage."""

    @pytest.mark.asyncio
    async def test_real_email_to_local_storage(
        self, real_eml_file, local_ingest_output, comprehensive_mock_settings
    ):
        """Process the test fixture EML through full pipeline to local storage."""
        # Parse
        parser = EMLParser(attachments_dir=tmp_path / "attachments")
        parsed = parser.parse_file(real_eml_file)
        assert parsed.metadata.subject is not None

        # Create document
        manager = HierarchyManager(local_ingest_output.db, tmp_path)
        email_doc = await manager.create_email_document(real_eml_file, parsed)

        # Chunk
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(parsed.full_text, email_doc.id, str(real_eml_file))
        assert len(chunks) >= 1

        # Store
        await local_ingest_output.db.insert_chunks(chunks)

        # Verify output
        docs = local_ingest_output.read_documents_jsonl()
        chunks_out = local_ingest_output.read_chunks_jsonl()
        assert len(docs) >= 1
        assert len(chunks_out) >= 1
```

### 4.3 HIGH: Attachment Context Propagation Test

File: `tests/test_ingest_flow.py` (new test)

```python
async def test_attachment_chunks_include_parent_context(self):
    """Attachment chunks must include parent email context in embedding_text."""
    # After processing an email with PDF attachment:
    # - Email context: "Email from X about Y dated Z"
    # - PDF chunk content: "Table of maintenance items..."
    # - PDF embedding_text SHOULD be: context + "\n\n" + content
    # Currently: PDF chunks get their OWN context (or none), missing email context
```

### 4.4 MEDIUM: Date in Embedding Text Test

File: `tests/test_ingest_processing.py`

```python
def test_context_includes_date(self, context_generator, sample_document):
    """Context summary should include the email date for temporal queries."""
    # Verify the context generation prompt asks for date
    # Verify the resulting context string contains a date reference
```

### 4.5 MEDIUM: Agent Top-K Integration Test

File: `tests/test_query_engine.py`

```python
async def test_agent_search_top_k_not_equals_rerank_top_n(patches):
    """Verify that the agent code path does not set top_k == rerank_top_n.

    This is a characterization test for the bug at agent.py:460.
    """
    # Simulate what agent.py does
    settings = patches.settings
    top_k_value = settings.rerank_top_n if settings.rerank_enabled else 10
    # This SHOULD fail until the bug is fixed
    assert top_k_value != settings.rerank_top_n or not settings.rerank_enabled
```

---

## 5. Sample Queries for UI Testing

After ingesting the 15-file subset, test these queries through the chat UI:

### Query 1: Specific Incident
**Query:** "What happened with the main engine governor on Maran Canopus?"
**Expected:** Response describes the governor failure incident from file 1/3. Should cite the ELREP report. Should mention the vessel name.

### Query 2: Equipment Type
**Query:** "Tell me about starter panel repairs on aux blowers"
**Expected:** Response references the DANAE aux blower starter panel repair (files 2, 5). Should describe the melted current transformer, power cable damage, and repair actions.

### Query 3: Vessel Filter
**Query:** "What classification surveys are pending for Maran Helen?"
**Expected:** Response references the incinerator survey with Lloyd's Register (file 4). Should mention IMO 9779381.

### Query 4: Safety Topic
**Query:** "What safety equipment issues have been reported?"
**Expected:** Response references foam replacement (file 8) and possibly lifeboat safety (file 12). Should NOT include non-safety content.

### Query 5: Date Range (once implemented)
**Query:** "What incidents were reported in June 2025?"
**Expected:** Files 1-4, 6 are from June 21, 2025. Should not include July 2025 files.

### Query 6: Negative Case
**Query:** "What cargo damage incidents occurred?"
**Expected:** No relevant results (none of the test files describe cargo damage). Response should say "couldn't find any relevant information" or similar.

### Query 7: Cross-Attachment
**Query:** "Show me photos related to the blower panel repair on Danae"
**Expected:** Response should reference the image attachments from the DANAE ELREP email. Citation should link to the image in the archive.

---

## 6. Step-by-Step Validation Procedure

### Phase 1: Infrastructure Verification (no API calls)

1. **Run existing tests:**
   ```bash
   uv run pytest tests/ -x -v
   ```
   All tests must pass. This establishes the baseline.

2. **Verify data directory:**
   ```bash
   ls data/emails/ | wc -l  # Should show 6,289
   ls data/emails/ | head -15  # Verify files match selection above
   ```

3. **Verify config path:**
   Confirm `.env` has `DATA_SOURCE_DIR=./data/emails` (not `./data/source`). This was flagged in investigation 03.

### Phase 2: Copy Test Subset

4. **Create test subset directory:**
   ```bash
   mkdir -p data/test-subset
   cp data/emails/100000366_qhlztvyi.alr.eml data/test-subset/
   cp data/emails/100000376_lgqfor05.2fi.eml data/test-subset/
   cp data/emails/100000440_0lbrmluh.10x.eml data/test-subset/
   cp data/emails/100000546_ccinfj2i.kew.eml data/test-subset/
   cp data/emails/100000566_bhofcmo4.zug.eml data/test-subset/
   cp data/emails/100000602_krtofd3g.nuw.eml data/test-subset/
   cp data/emails/100000660_vpirjft0.vyg.eml data/test-subset/
   cp data/emails/100000922_vp4qcvdw.abx.eml data/test-subset/
   cp data/emails/100040671_drxriffw.1u2.eml data/test-subset/
   cp data/emails/100284018_5vcg2bgp.jwm.eml data/test-subset/
   cp data/emails/100293054_ueg1nbfp.amu.eml data/test-subset/
   cp data/emails/100297210_2ef23hqq.cni.eml data/test-subset/
   cp data/emails/100297520_gts3adxm.eid.eml data/test-subset/
   cp data/emails/100297780_avnjkyr4.40y.eml data/test-subset/
   cp data/emails/100298198_va10u0fa.qfv.eml data/test-subset/
   ```

5. **Point config to subset:**
   Temporarily set `DATA_SOURCE_DIR=./data/test-subset` in `.env`.

### Phase 3: Dry Run (estimate mode)

6. **Run estimate command:**
   ```bash
   uv run mtss estimate
   ```
   Expected output: 15 EMLs, estimated cost < $1, lists attachment counts by type.

### Phase 4: Ingest Subset

7. **Run ingest in local-only mode** (once implemented):
   ```bash
   uv run mtss ingest --local-only --output ./data/test-output
   ```

8. **If local-only mode not yet available**, run against dev Supabase:
   ```bash
   uv run mtss ingest --lenient
   ```

9. **Verify ingest output:**
   - Check console for errors/warnings
   - Verify all 15 emails processed (no skips unless expected)
   - Count documents: should be 15 emails + N attachments
   - Check for any parse failures in the issue tracker summary

### Phase 5: Search Quality Validation

10. **Start the API server:**
    ```bash
    uv run mtss serve
    ```

11. **Open the chat UI** and run each query from Section 5.

12. **For each query, verify:**
    - Response is relevant (not hallucinated)
    - Citations reference the correct source files
    - Citation links work (browse and download)
    - No `[C:...]` raw markers in the response
    - Source section at the bottom lists all cited sources

### Phase 6: Regression Checks

13. **Verify reranker is functioning:**
    - Enable debug logging for `mtss.rag.reranker`
    - Run a query and verify log shows "Reranking N candidates -> M results" where N > M
    - If N == M, the reranker bug (Scenario 8) is still present

14. **Verify no double-encoding in URIs:**
    - Click citation download links
    - Check browser URL bar for `%2520` (bad) vs `%20` (good)
    - Check for double `/archive/archive/` in paths

15. **Restore config:**
    Set `DATA_SOURCE_DIR` back to `./data/emails` for production.

---

## 7. Success Criteria

The test subset validation passes if ALL of the following are true:

- [ ] All 17 existing test files pass (`pytest tests/ -x`)
- [ ] All 15 test emails parse without errors
- [ ] At least 12/15 emails produce at least 1 chunk each
- [ ] Vessel names are matched for files 1, 3, 4, 8, 9, 13
- [ ] Topics are extracted for maritime emails (files 1-4, 6, 8, 12)
- [ ] Non-maritime emails (10, 11) do not produce maritime topics
- [ ] Sample queries 1-4 return relevant results as top-1
- [ ] Sample query 6 (negative) returns "no information" response
- [ ] All citation links are valid (no broken browse/download)
- [ ] No double-encoding or double-prefix in any URI
- [ ] Reranker log shows meaningful candidate reduction (top_k > rerank_top_n)
- [ ] Local output (if available) produces valid JSONL files

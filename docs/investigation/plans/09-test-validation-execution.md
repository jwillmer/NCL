---
purpose: Step-by-step execution plan for test subset validation after all implementation plans are complete
status: pending
date: 2026-04-13
depends_on: [implementation-plan.md, 00-critical-fixes-plan.md, optimization-plan.md]
execute_after: All implementation plans (Phases 0-5) and critical fixes (SR-0 through SR-3) are complete
---

# Plan 09: Test Validation Execution

This plan executes AFTER all implementation plans are complete. It validates the full pipeline
using a 15-document subset before the production run of 6,289 emails.

**Source:** `docs/investigation/09-test-validation-plan.md`

## Prerequisites

Before starting this plan, verify ALL of the following are done:

- [ ] **Plan 00 (Critical Fixes):** Reranker bug fixed (agent.py), max_tokens increased, rerank_top_n=8
- [ ] **Implementation Plan Phase 0:** Config updated (chunk 1024, dims 512)
- [ ] **Implementation Plan Phase 1:** Image pre-filtering in pipeline
- [ ] **Implementation Plan Phase 2:** Local PDF/Office parsers working
- [ ] **Implementation Plan Phase 3:** LocalStorageClient extended, progress tracker, loggers
- [ ] **Implementation Plan Phase 4:** Pipeline wired with --local-only flag, parallel attachments
- [ ] **Implementation Plan Phase 5:** Unit tests passing
- [ ] **06b Quality Wins:** P6 (attachment context), P1-A (min content filter), P8-A (dates in embedding)
- [ ] **Optimization Plan SR-0 through SR-3:** Search/retrieval fixes applied

## Phase 1: Infrastructure Verification (~10 min)

### Step 1.1: Run existing test suite
```bash
uv run pytest tests/ -x -v
```
All tests must pass. If any fail, fix before proceeding.

### Step 1.2: Verify data directory
```bash
ls data/emails/ | wc -l          # Should show 6,289
grep DATA_SOURCE_DIR .env         # Should show ./data/emails
```

### Step 1.3: Verify config changes
```bash
grep -E "CHUNK_SIZE|CHUNK_OVERLAP|EMBEDDING_DIM" .env
# Expected: CHUNK_SIZE_TOKENS=1024, CHUNK_OVERLAP_TOKENS=100, EMBEDDING_DIMENSIONS=512
```

## Phase 2: Prepare Test Subset (~5 min)

### Step 2.1: Create test subset directory
```bash
mkdir -p data/test-subset
```

### Step 2.2: Copy selected test documents
```bash
cp data/emails/100000922_vp4qcvdw.abx.eml data/test-subset/  # MARAN CANOPUS engine governor
cp data/emails/100000366_qhlztvyi.alr.eml data/test-subset/  # DANAE starter panel repair
cp data/emails/100000660_vpirjft0.vyg.eml data/test-subset/  # Long thread (15+ refs)
cp data/emails/100000602_krtofd3g.nuw.eml data/test-subset/  # MARAN HELEN incinerator survey
cp data/emails/100000440_0lbrmluh.10x.eml data/test-subset/  # Text-only reply
cp data/emails/100000376_lgqfor05.2fi.eml data/test-subset/  # Forwarded with attachments
cp data/emails/100284018_5vcg2bgp.jwm.eml data/test-subset/  # Logistics, inline images
cp data/emails/100297210_2ef23hqq.cni.eml data/test-subset/  # Safety equipment
cp data/emails/100000546_ccinfj2i.kew.eml data/test-subset/  # External sender (Wilhelmsen)
cp data/emails/100000566_bhofcmo4.zug.eml data/test-subset/  # Non-maritime newsletter
cp data/emails/100293054_ueg1nbfp.amu.eml data/test-subset/  # Marketing (Qatar Airways)
cp data/emails/100297520_gts3adxm.eid.eml data/test-subset/  # PSC safety bulletin
cp data/emails/100297780_avnjkyr4.40y.eml data/test-subset/  # Worklogs with attachments
cp data/emails/100040671_drxriffw.1u2.eml data/test-subset/  # Third-party operations
# File 14 is tests/fixtures/test_email.eml — used directly, no copy needed
```

### Step 2.3: Verify subset
```bash
ls data/test-subset/ | wc -l   # Should show 14 (15th is in tests/fixtures/)
```

## Phase 3: Dry Run — Estimate (~2 min)

### Step 3.1: Run estimate on subset
```bash
uv run mtss estimate --source ./data/test-subset
```

**Expected:** 14 EMLs, estimated cost <$1, lists attachment counts by type.
If cost is significantly higher, investigate before proceeding.

## Phase 4: Local-Only Ingest (~5-15 min)

### Step 4.1: Run local-only ingest
```bash
uv run mtss ingest --source ./data/test-subset --local-only --output ./data/test-output --lenient --verbose
```

### Step 4.2: Monitor output
Watch for:
- All 14 emails processed (+ test fixture if included)
- No unexpected errors (warnings OK for non-maritime content)
- Attachment processing completing (PDFs parsed locally, images pre-filtered)
- Context summaries generated
- Topics extracted
- Embeddings generated

### Step 4.3: Verify local output structure
```bash
# Check JSONL files exist and have content
wc -l data/test-output/documents.jsonl     # Should be 14+ (emails + attachments)
wc -l data/test-output/chunks.jsonl        # Should be 20+ chunks
wc -l data/test-output/topics.jsonl        # Should be 5-15 topics
wc -l data/test-output/processing_log.jsonl # Should be 14 entries
ls data/test-output/archive/ | wc -l       # Should be 14 archive folders

# Verify JSONL is valid JSON
head -1 data/test-output/documents.jsonl | python -m json.tool > /dev/null && echo "Valid JSON"
head -1 data/test-output/chunks.jsonl | python -m json.tool > /dev/null && echo "Valid JSON"
```

### Step 4.4: Validate output content
```bash
# Check embeddings are present (not omitted)
python -c "
import json
with open('data/test-output/chunks.jsonl') as f:
    chunk = json.loads(f.readline())
    emb = chunk.get('embedding', [])
    print(f'Embedding dims: {len(emb)}')  # Should be 512
    print(f'Has content: {bool(chunk.get(\"content\"))}')
    print(f'Has context: {bool(chunk.get(\"context_summary\"))}')
    print(f'Has embedding_text: {bool(chunk.get(\"embedding_text\"))}')
"
```

### Step 4.5: Validate quality improvements (06b)
```bash
python -c "
import json

# Check minimum content filter (P1-A)
with open('data/test-output/chunks.jsonl') as f:
    chunks = [json.loads(line) for line in f]
    short = [c for c in chunks if len(c['content'].split()) < 20]
    print(f'Chunks with <20 words: {len(short)} (should be 0)')

# Check date in embedding text (P8-A)
    with_date = [c for c in chunks if 'Date:' in c.get('embedding_text', '')]
    print(f'Chunks with date in embedding: {len(with_date)}/{len(chunks)}')

# Check attachment context inheritance (P6)
    att_chunks = [c for c in chunks if c.get('metadata', {}).get('type', '').startswith('attachment')]
    if att_chunks:
        has_parent_ctx = sum(1 for c in att_chunks if 'Email' in c.get('embedding_text', '')[:200])
        print(f'Attachment chunks with parent context: {has_parent_ctx}/{len(att_chunks)}')
"
```

## Phase 5: Import to Database & Test Search (~15 min)

> **Note:** This phase requires the production database to be accessible.
> Ask the user to confirm database is ready before proceeding.

### Step 5.1: Ask user to confirm DB readiness
```
Database services needed:
- Supabase PostgreSQL (documents, chunks, topics tables)
- Supabase Storage (archive bucket) — optional for local-tested data
```

### Step 5.2: Import local data to DB
```bash
# If import tool exists:
uv run mtss import --from ./data/test-output

# If not yet implemented, re-ingest directly to DB:
uv run mtss ingest --source ./data/test-subset --lenient --verbose
```

### Step 5.3: Verify data in DB
```bash
uv run mtss stats
```
Expected: 14+ documents, 20+ chunks, 5-15 topics.

### Step 5.4: Start the API server
```bash
uv run mtss serve
```

### Step 5.5: Open the chat UI
Navigate to http://localhost:5173 (or configured port).

## Phase 6: Query Validation (~15 min)

Run each query and verify results. Record pass/fail.

### Query 1: Specific Incident
**Query:** "What happened with the main engine governor on Maran Canopus?"
- [ ] Response describes governor failure incident
- [ ] Cites ELREP report (files 1 or 3)
- [ ] Mentions vessel name "MARAN CANOPUS"
- [ ] Citation links work

### Query 2: Equipment Type
**Query:** "Tell me about starter panel repairs on aux blowers"
- [ ] References DANAE aux blower starter panel (files 2, 5)
- [ ] Describes melted transformer, cable damage, repairs
- [ ] Citations reference correct source files

### Query 3: Vessel Filter
**Query:** (select MARAN HELEN in UI dropdown) "What classification surveys are pending?"
- [ ] References incinerator survey with Lloyd's Register (file 4)
- [ ] Mentions IMO 9779381

### Query 4: Safety Topic
**Query:** "What safety equipment issues have been reported?"
- [ ] References foam replacement (file 8) and/or lifeboat safety (file 12)
- [ ] Does NOT include unrelated content

### Query 5: Negative Case
**Query:** "What cargo damage incidents occurred?"
- [ ] Returns "no relevant information" or similar
- [ ] Does NOT hallucinate cargo damage incidents

### Query 6: Cross-Attachment
**Query:** "Show me details from the blower panel repair report on Danae"
- [ ] References attachment content (PDF/image from file 2)
- [ ] Attachment chunks appear in results (validates P6 attachment context)

### Query 7: Reranker Verification
- [ ] Enable debug logging: `PYTHONLOGGING=DEBUG uv run mtss serve`
- [ ] Run any query
- [ ] Check logs for "Reranking N candidates -> M results" where N > M
- [ ] If N == M: FAIL — reranker bug (SR-0) not fixed

## Phase 7: Regression Checks (~5 min)

### Step 7.1: Citation integrity
- [ ] No raw `[C:...]` markers in any response
- [ ] All citation links resolve (no 404)
- [ ] No double-encoding (`%2520`) in URIs
- [ ] No double `/archive/archive/` prefixes

### Step 7.2: Verify config took effect
```bash
python -c "
import json
with open('data/test-output/chunks.jsonl') as f:
    chunk = json.loads(f.readline())
    print(f'Embedding dims: {len(chunk.get(\"embedding\", []))}')  # Should be 512
    print(f'Content length: {len(chunk.get(\"content\", \"\"))} chars')
"
```
- [ ] Embedding dimensions = 512 (not 1536)
- [ ] Chunks are larger (~1024 tokens, not ~512)

## Phase 8: Cleanup & Report

### Step 8.1: Record results
Update this document with pass/fail for each check.

### Step 8.2: Restore config for production
```bash
# If DATA_SOURCE_DIR was changed for testing, restore:
# DATA_SOURCE_DIR=./data/emails
```

### Step 8.3: Decision gate
If ALL checks pass:
- [ ] Proceed to full ingest of 6,289 emails
- [ ] Use local-only mode: `uv run mtss ingest --local-only --output ./data/full-ingest --lenient`
- [ ] Estimated cost: ~$6-10
- [ ] Estimated time: ~53 minutes

If ANY critical checks fail:
- [ ] Document the failure
- [ ] Fix the root cause
- [ ] Re-run the test subset (do NOT proceed to full ingest)

## Success Criteria Summary

| Category | Criteria | Required |
|----------|----------|----------|
| Existing tests | All pytest tests pass | Yes |
| Ingest | All 14 test emails process without error | Yes |
| Chunks | At least 1 chunk per email (12+ of 15 total) | Yes |
| Vessel matching | Correct vessels matched for files 1, 3, 4, 8, 9, 13 | Yes |
| Topics | Maritime topics extracted for files 1-4, 6, 8, 12 | Yes |
| Non-maritime | Files 10, 11 do not produce maritime topics | Yes |
| Queries 1-4 | Relevant results as top-1 | Yes |
| Query 5 | Negative case returns "no information" | Yes |
| Reranker | Log shows N > M candidate reduction | Yes |
| Citations | No broken links, no double-encoding | Yes |
| Local output | Valid JSONL with embeddings | Yes |
| Config | Embedding dims=512, chunk size=1024 | Yes |
| Quality wins | No <20-word chunks, dates in embeddings, attachment context | Yes |

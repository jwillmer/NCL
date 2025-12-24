# Maritime Metadata Enrichment System

> **Status:** Planned (not yet implemented)

Extract vessel identification and problem/solution classification from documents, with a vessel register that improves over time.

## Overview

The maritime industry deals with documents about specific vessels and technical problems/solutions. This enrichment system extracts:
- **Vessel identification** (primary + related vessels)
- **Problem/solution classification**
- **Vessel register** - master database that accumulates data

## Architecture

```
Existing Pipeline                    Maritime Enrichment (New)
─────────────────                    ────────────────────────
ncl ingest                           ncl enrich
    ↓                                    ↓
documents table  ──────────────────→ Query unenriched docs
    ↓                                    ↓
chunks table     ──────────────────→ Aggregate content
                                         ↓
                                    Maritime Extractor (LLM)
                                         ↓
                                    Vessel Registry (match/create)
                                         ↓
                                    ┌────┴────┐
                                    ↓         ↓
                              vessels    maritime_metadata
                                    ↓
                              document_vessel_mentions
```

## Processing Model

Enrichment runs **separately from ingestion**:

1. `ncl ingest` processes documents normally
2. `ncl enrich` extracts maritime metadata from unenriched documents
3. Status tracking prevents duplicate processing
4. `--retry-failed` reprocesses failed extractions

## Database Schema

### vessels (Vessel Register)

```sql
CREATE TABLE vessels (
    id UUID PRIMARY KEY,

    -- Identifiers
    imo TEXT UNIQUE,              -- 7 digits, globally unique
    mmsi TEXT,                    -- 9 digits
    call_sign TEXT,
    name TEXT NOT NULL,

    -- Additional info
    flag TEXT,
    vessel_type TEXT,             -- tanker, cargo, container, etc.
    aliases TEXT[],               -- Alternative names seen

    -- Quality tracking
    first_seen_at TIMESTAMPTZ,
    last_seen_at TIMESTAMPTZ,
    document_count INT DEFAULT 0,
    data_quality_score FLOAT      -- 0-1
);
```

### maritime_metadata (per document)

```sql
CREATE TABLE maritime_metadata (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE UNIQUE,

    -- Primary vessel
    primary_vessel_id UUID REFERENCES vessels(id),

    -- Problem/Solution
    problem_category TEXT,        -- propulsion, electrical, hull, etc.
    problem_description TEXT,
    solution_summary TEXT,
    solution_effectiveness TEXT,  -- resolved, partial, pending, unknown

    -- Extraction tracking
    extraction_status TEXT DEFAULT 'pending',
        -- pending, completed, failed, needs_review
    extraction_error TEXT,
    extraction_confidence FLOAT,
    extracted_at TIMESTAMPTZ
);
```

### document_vessel_mentions (many-to-many)

```sql
CREATE TABLE document_vessel_mentions (
    document_id UUID REFERENCES documents(id),
    vessel_id UUID REFERENCES vessels(id),

    is_primary BOOLEAN DEFAULT FALSE,
    mention_context TEXT,
    relationship_type TEXT,       -- sister_ship, fleet_member, referenced

    UNIQUE(document_id, vessel_id)
);
```

## Vessel Register Logic

### Matching Priority

1. **IMO** (exact) - Most reliable, globally unique
2. **MMSI** (exact) - Reliable but can change
3. **Call sign** (exact)
4. **Name** (fuzzy) - Case-insensitive, strip M/V, MT prefixes

### Merge Strategy

- Never overwrite non-null with null
- Newer data takes precedence for conflicts
- Collect all names in `aliases[]`
- Increment `document_count`, update `last_seen_at`

### Example

```
Doc 1: name="PACIFIC STAR", imo=null
  → Creates: {name: "PACIFIC STAR", doc_count: 1}

Doc 2: name="M/V Pacific Star", imo="1234567"
  → Fuzzy matches, updates: {imo: "1234567", aliases: ["M/V Pacific Star"], doc_count: 2}

Doc 3: name="PACIFIC STAR", mmsi="123456789"
  → IMO match, updates: {mmsi: "123456789", doc_count: 3}
```

## LLM Extraction

Uses OpenAI Agents SDK with structured outputs (same pattern as `image_processor.py`).

### Extraction Prompt

```
You are a maritime industry expert extracting structured information.

Extract:
1. PRIMARY VESSEL - name, IMO (7 digits), MMSI (9 digits), call sign, flag, type
2. RELATED VESSELS - other vessels mentioned with context
3. PROBLEM/SOLUTION - category, description, solution, effectiveness

Rules:
- IMO numbers are ALWAYS 7 digits
- MMSI numbers are ALWAYS 9 digits
- Leave fields null if not explicitly stated
- Set confidence based on clarity
```

## CLI Commands

### ncl enrich

```
ncl enrich [OPTIONS]

Options:
  -b, --batch-size INT     Documents per batch (default: 50)
  --retry-failed           Retry failed extractions
  --reprocess-review       Reprocess needs_review documents
  -v, --verbose            Detailed output
```

### ncl vessels (optional)

```
ncl vessels [OPTIONS]

Options:
  -s, --search TEXT    Search by name or IMO
  --stats              Show vessel register statistics
  --export FILE        Export to CSV
```

## Data Quality Handling

| Scenario | Handling |
|----------|----------|
| No vessel found | `completed` with null fields |
| Low confidence (<0.3) | `needs_review` status |
| LLM error | `failed` status, error logged |
| Invalid IMO format | Field left null |

## Files to Create/Modify

| File | Action |
|------|--------|
| `migrations/002_maritime_metadata.sql` | Create |
| `src/ncl/models/maritime.py` | Create |
| `src/ncl/processing/maritime_extractor.py` | Create |
| `src/ncl/storage/vessel_registry.py` | Create |
| `src/ncl/storage/supabase_client.py` | Modify |
| `src/ncl/cli.py` | Modify |
| `README.md` | Update |

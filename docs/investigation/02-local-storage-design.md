---
purpose: Design local backup/fallback storage and local-only ingest pipeline
status: proposal
date: 2026-04-13
scope: ingest pipeline, storage layer, data portability
---

# Local Storage Design: Backup & Provider-Independent Ingest

## 1. Data Inventory

Everything the ingest pipeline produces, mapped from source code analysis.

### 1.1 Documents (table: `documents`)

Each email and attachment becomes a Document record. Fields stored:

| Field | Type | Source |
|---|---|---|
| `id` | UUID | Generated at creation |
| `source_id` | string | `hash(relative_path.lower())` via `normalize_source_id()` |
| `doc_id` | string | `hash(source_id + file_hash)` via `compute_doc_id()` |
| `content_version` | int | Always 1 currently |
| `ingest_version` | int | From settings (currently 4) |
| `parent_id` | UUID | null for emails, parent UUID for attachments |
| `root_id` | UUID | Self-referencing for emails, root email for attachments |
| `depth` | int | 0 for emails, 1+ for attachments |
| `path` | string[] | Ancestry path as list of UUIDs |
| `document_type` | enum | `email`, `attachment_pdf`, `attachment_image`, etc. |
| `file_path` | string | Original EML path or archive path for attachments |
| `file_name` | string | Original filename |
| `file_hash` | string | SHA-256 of file content |
| `source_title` | string | Email subject or attachment filename |
| `archive_path` | string | Relative path to archive folder (e.g., `abc123def456`) |
| `archive_browse_uri` | string | URI to browsable .md file |
| `archive_download_uri` | string | URI to original file |
| `status` | enum | `pending`, `processing`, `completed`, `failed`, `skipped` |
| `error_message` | string | Error details if status=failed |
| `email_subject` | string | From email_metadata |
| `email_participants` | string[] | All unique email addresses |
| `email_initiator` | string | First sender in thread |
| `email_date_start` | datetime | Earliest message date |
| `email_date_end` | datetime | Latest message date |
| `email_message_count` | int | Number of messages in conversation |
| `attachment_content_type` | string | MIME type |
| `attachment_size_bytes` | int | File size |

### 1.2 Chunks (table: `chunks`)

Text segments with embeddings for vector search:

| Field | Type | Source |
|---|---|---|
| `id` | UUID | Generated at creation |
| `chunk_id` | string | `hash(doc_id + char_start + char_end)` -- deterministic |
| `document_id` | UUID | Foreign key to documents |
| `content` | string | Original text content |
| `chunk_index` | int | Position within document |
| `context_summary` | string | LLM-generated document-level context |
| `embedding_text` | string | Full text used for embedding (context + content) |
| `section_path` | string[] | Heading hierarchy (e.g., `["Email Body", "Message 1"]`) |
| `section_title` | string | Current section heading |
| `source_title` | string | Denormalized from document |
| `source_id` | string | Denormalized from document |
| `page_number` | int | For PDF/doc attachments |
| `line_from` | int | Start line |
| `line_to` | int | End line |
| `char_start` | int | Start character offset |
| `char_end` | int | End character offset |
| `archive_browse_uri` | string | Denormalized from document |
| `archive_download_uri` | string | Denormalized from document |
| `embedding` | float[] | 1536-dim vector (text-embedding-3-small) |
| `metadata` | JSON | Contains `vessel_ids`, `vessel_types`, `vessel_classes`, `topic_ids`, `type` |

### 1.3 Topics (table: `topics`)

Semantic categories extracted per email:

| Field | Type | Source |
|---|---|---|
| `id` | UUID | Generated at creation |
| `name` | string | Canonical lowercase |
| `display_name` | string | User-friendly display form |
| `description` | string | Brief description from LLM |
| `embedding` | float[] | 1536-dim vector for similarity matching |
| `chunk_count` | int | Number of chunks tagged with this topic |
| `document_count` | int | Number of documents tagged with this topic |

### 1.4 Archive Files (Supabase Storage bucket: `archive`)

Per-email folder structure in object storage:

```
{folder_id}/                      # doc_id[:16]
  email.eml                       # Original EML file
  email.eml.md                    # Browsable markdown preview
  metadata.json                   # Programmatic access metadata
  attachments/
    report.pdf                    # Original attachment
    report.pdf.md                 # Markdown preview of parsed content
    image.png                     # Original image
    image.png.md                  # Image description as markdown
```

### 1.5 Processing Log (table: `processing_log`)

Tracks per-file processing state for resumability:

| Field | Type |
|---|---|
| `file_path` | string (primary key) |
| `file_hash` | string |
| `status` | enum (pending/processing/completed/failed) |
| `started_at` | datetime |
| `completed_at` | datetime |
| `last_error` | string |
| `attempts` | int |

### 1.6 Ingest Events (table: `ingest_events`)

Unsupported/failed attachment tracking:

| Field | Type |
|---|---|
| `file_path` | string (primary key, upsert) |
| `file_name` | string |
| `file_size_bytes` | int |
| `mime_type` | string |
| `file_extension` | string |
| `reason` | string |
| `source_eml_path` | string |
| `source_zip_path` | string |
| `parent_document_id` | UUID |
| `discovered_at` | datetime |

### 1.7 Vessels (table: `vessels`)

Static reference data (loaded separately, not produced by ingest):

| Field | Type |
|---|---|
| `id` | UUID |
| `name` | string |
| `vessel_type` | string |
| `vessel_class` | string |

### 1.8 Failure Reports (local files)

Already written to `data/reports/` as JSON + CSV during ingest. No change needed.

---

## 2. Proposed Local Folder Structure

Each ingest run produces a self-contained output directory:

```
data/local-ingest/
  manifest.json                   # Index of this ingest run
  documents.jsonl                 # All document records (1 JSON per line)
  chunks.jsonl                    # All chunk records WITH embeddings
  topics.jsonl                    # All topics WITH embeddings
  processing_log.jsonl            # File processing status
  ingest_events.jsonl             # Unsupported/failed files
  vessels.json                    # Vessel reference data snapshot
  archive/                        # Mirrors Supabase Storage structure
    {folder_id}/
      email.eml
      email.eml.md
      metadata.json
      attachments/
        report.pdf
        report.pdf.md
```

### Why this structure

- **JSONL** for documents/chunks/topics: append-friendly, streaming reads, one record per line
- **Embeddings included** in chunks.jsonl and topics.jsonl: these are the expensive part (API calls to OpenAI). Each 1536-dim vector is ~12KB as JSON. For 100K chunks this is ~1.2GB of embeddings data -- large but portable
- **Archive folder** mirrors current Supabase Storage layout exactly: no path translation needed on import
- **manifest.json** provides metadata and quick validation without reading all files

---

## 3. JSON Schema Examples

### 3.1 manifest.json

```json
{
  "version": "1.0",
  "created_at": "2026-04-13T10:30:00Z",
  "ingest_version": 4,
  "embedding_model": "text-embedding-3-small",
  "embedding_dimensions": 1536,
  "source_dir": "data/source",
  "counts": {
    "documents": 1250,
    "chunks": 45000,
    "topics": 87,
    "archive_files": 8500
  },
  "settings_snapshot": {
    "chunk_size_tokens": 512,
    "chunk_overlap_tokens": 50,
    "context_llm_model": "gpt-4o-mini",
    "current_ingest_version": 4
  }
}
```

### 3.2 documents.jsonl (one line per record)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "source_id": "a1b2c3d4e5f6",
  "doc_id": "f6e5d4c3b2a1",
  "content_version": 1,
  "ingest_version": 4,
  "parent_id": null,
  "root_id": "550e8400-e29b-41d4-a716-446655440000",
  "depth": 0,
  "path": ["550e8400-e29b-41d4-a716-446655440000"],
  "document_type": "email",
  "file_path": "data/source/reports/incident-001.eml",
  "file_name": "incident-001.eml",
  "file_hash": "sha256_hex_string",
  "source_title": "RE: Main Engine Failure - MV Maran",
  "archive_path": "f6e5d4c3b2a10000",
  "archive_browse_uri": "/archive/f6e5d4c3b2a10000/email.eml.md",
  "archive_download_uri": "/archive/f6e5d4c3b2a10000/email.eml",
  "status": "completed",
  "email_metadata": {
    "subject": "RE: Main Engine Failure - MV Maran",
    "participants": ["tech@example.com", "ops@example.com"],
    "initiator": "tech@example.com",
    "date_start": "2025-03-15T08:00:00Z",
    "date_end": "2025-03-16T14:30:00Z",
    "message_count": 3
  },
  "attachment_metadata": null
}
```

### 3.3 chunks.jsonl (one line per record)

```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "chunk_id": "hash_based_stable_id",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "content": "The main engine fuel pump failed during transit...",
  "chunk_index": 0,
  "context_summary": "Email thread about main engine fuel pump failure on MV Maran during transit.",
  "embedding_text": "Context: Email thread about main engine fuel pump failure...\n\nContent: The main engine fuel pump failed during transit...",
  "section_path": ["Email Body", "Message 1"],
  "section_title": null,
  "source_title": "RE: Main Engine Failure - MV Maran",
  "source_id": "a1b2c3d4e5f6",
  "page_number": null,
  "line_from": null,
  "line_to": null,
  "char_start": 0,
  "char_end": 450,
  "archive_browse_uri": "/archive/f6e5d4c3b2a10000/email.eml.md",
  "archive_download_uri": "/archive/f6e5d4c3b2a10000/email.eml",
  "embedding": [0.0123, -0.0456, 0.0789, "... 1536 floats total ..."],
  "metadata": {
    "type": "email_body",
    "message_index": 0,
    "vessel_ids": ["vessel-uuid-1"],
    "vessel_types": ["VLCC"],
    "vessel_classes": ["Canopus Class"],
    "topic_ids": ["topic-uuid-1", "topic-uuid-2"]
  }
}
```

### 3.4 topics.jsonl (one line per record)

```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "name": "engine issues",
  "display_name": "Engine Issues",
  "description": "Problems related to main and auxiliary engine systems",
  "embedding": [0.0111, -0.0222, 0.0333, "... 1536 floats total ..."],
  "chunk_count": 234,
  "document_count": 45
}
```

### 3.5 processing_log.jsonl

```json
{
  "file_path": "data/source/reports/incident-001.eml",
  "file_hash": "sha256_hex",
  "status": "completed",
  "started_at": "2026-04-13T10:30:00Z",
  "completed_at": "2026-04-13T10:31:15Z",
  "last_error": null,
  "attempts": 1
}
```

### 3.6 ingest_events.jsonl

```json
{
  "file_path": "C:/data/attachments/logo.png",
  "file_name": "logo.png",
  "file_size_bytes": 15234,
  "mime_type": "image/png",
  "file_extension": ".png",
  "reason": "classified_as_non_content",
  "source_eml_path": "data/source/reports/incident-001.eml",
  "parent_document_id": "550e8400-e29b-41d4-a716-446655440000",
  "discovered_at": "2026-04-13T10:30:45Z"
}
```

---

## 4. Local-Only Pipeline Modification Plan

### 4.1 Component Analysis

| Component | Current Backend | Local-Only Change | Effort |
|---|---|---|---|
| `SupabaseClient` (db) | PostgreSQL via Supabase REST + asyncpg | Replace with `LocalStorageClient` | **Extend existing** |
| `ArchiveStorage` | Supabase Storage bucket | Replace with `LocalBucketStorage` | **Already exists** |
| `ArchiveGenerator` | Uses `ArchiveStorage` internally | Inject `LocalBucketStorage` | Minimal wiring |
| `HierarchyManager` | Uses `SupabaseClient` | Receives db via constructor | No change needed |
| `ProgressTracker` | Uses `SupabaseClient.client.table()` | Needs local implementation | **New adapter** |
| `UnsupportedFileLogger` | Uses `SupabaseClient.client.table()` | Needs local implementation | **New adapter** |
| `EmbeddingGenerator` | OpenAI API via LiteLLM | Keep as-is (still need embeddings) | No change |
| `TopicExtractor` | LLM via LiteLLM | Keep as-is | No change |
| `TopicMatcher` | Uses `SupabaseClient` for topic CRUD | Needs local implementation | **New adapter** |
| `VesselMatcher` | In-memory (reads from DB at start) | Keep as-is | No change |
| `VersionManager` | Uses `SupabaseClient` | Needs local implementation | **New adapter** |
| `EMLParser` | Pure Python parsing | Keep as-is | No change |
| `AttachmentProcessor` | Pure Python + LlamaParse | Keep as-is | No change |
| `ContextGenerator` | LLM via LiteLLM | Keep as-is | No change |
| `DocumentChunker` | Pure Python | Keep as-is | No change |

### 4.2 What Already Exists

The test infrastructure (`tests/local_storage.py`) already provides:

- `LocalStorageClient`: Handles `insert_document`, `insert_chunks`, `update_document_status`, `update_document_archive_uris`, `log_ingest_event`, `get_document_by_doc_id`, `get_document_by_source_id`, `get_chunks_by_document`, `close`
- `LocalBucketStorage`: Full file operations (`upload_file`, `upload_text`, `download_file`, `download_text`, `file_exists`, `delete_folder`, `list_files`)
- `LocalIngestOutput`: Coordinator with `create()` factory and summary/read helpers

### 4.3 Gaps in Existing LocalStorageClient

Methods the pipeline calls but `LocalStorageClient` does NOT implement:

1. **`get_document_by_hash(file_hash)`** -- used for legacy skip check
2. **`get_document_by_id(doc_id: UUID)`** -- used for version checks
3. **`delete_document_for_reprocess(doc_id)`** -- used for cleanup
4. **`insert_topic(topic)`** -- topic persistence
5. **`get_topic_by_name(name)`** -- topic deduplication
6. **`find_similar_topics(embedding, threshold, limit)`** -- topic matching
7. **`increment_topic_counts(topic_ids, chunk_delta, document_delta)`** -- topic stats
8. **`get_topic_by_id(topic_id)`** -- topic lookup

Additionally missing for full local operation:
- `ProgressTracker` local adapter (currently uses `processing_log` DB table)
- `UnsupportedFileLogger` local adapter (currently uses `ingest_events` DB table)

### 4.4 Chunks JSONL Must Include Embeddings

The existing `LocalStorageClient._chunk_to_dict()` explicitly omits the embedding:

```python
# Embedding is omitted (too large for JSONL)
```

For a production local backup, this must be changed to include embeddings. The comment notes size concerns, but embeddings are the most expensive data to regenerate (~$0.02 per 1M tokens). For ~50K chunks at ~512 tokens each, that is roughly $0.50 in API cost -- tolerable to re-embed but not to discard.

**Recommendation**: Include embeddings by default. Add a `--skip-embeddings` flag for debugging use cases where size matters.

### 4.5 Implementation Steps (Ordered)

**Phase 1: Extend LocalStorageClient** (the minimum for local-only ingest)

1. Add missing document query methods (`get_document_by_hash`, `get_document_by_id`)
2. Add `delete_document_for_reprocess` (remove from in-memory dict + append tombstone to JSONL)
3. Add topic CRUD methods with in-memory deduplication:
   - `insert_topic` / `get_topic_by_name` / `get_topic_by_id` / `find_similar_topics`
   - Use numpy or manual cosine similarity for in-memory topic matching
   - Write topics to `topics.jsonl`
4. Add `increment_topic_counts` (update in-memory, rewrite JSONL on flush)
5. Include embeddings in `_chunk_to_dict()`
6. Add `_topic_to_dict()` serialization including embedding

**Phase 2: Local ProgressTracker & UnsupportedFileLogger**

7. Create `LocalProgressTracker` that writes to `processing_log.jsonl`
   - Same interface as `ProgressTracker` but uses JSONL instead of DB table
   - On startup, reads existing JSONL to populate in-memory hash set
8. Create `LocalUnsupportedFileLogger` that writes to `ingest_events.jsonl`
   - Same interface as `UnsupportedFileLogger` but appends to JSONL

**Phase 3: Wire It Together**

9. Create `create_local_ingest_components()` factory in `components.py`:
   - Mirrors `create_ingest_components()` but uses local storage
   - Injects `LocalStorageClient` as `db`
   - Injects `LocalBucketStorage` into `ArchiveGenerator`
   - Creates local tracker and logger
10. Add `manifest.json` writer that runs at end of ingest
11. Add `--local-only` flag to CLI ingest command

**Phase 4: Finalization**

12. Flush method on `LocalStorageClient` to write final consistent state
13. Write `vessels.json` snapshot at start of ingest (reference data)
14. Validation script to verify local ingest output completeness

### 4.6 ArchiveGenerator Wiring

`ArchiveGenerator.__init__()` creates its own `ArchiveStorage()` instance internally:

```python
self.storage = ArchiveStorage()
```

For local operation, this needs to accept an injected storage backend:

```python
def __init__(self, ingest_root=None, storage=None):
    self.storage = storage or ArchiveStorage()
```

`LocalBucketStorage` already has the same method signatures (`upload_file`, `upload_text`, `download_file`, `download_text`, `file_exists`, `delete_folder`, `list_files`), so this is a drop-in replacement via duck typing.

### 4.7 TopicMatcher Local Operation

`TopicMatcher` uses `db.find_similar_topics()` which calls a pgvector `<=>` operator in PostgreSQL. For local operation, this needs an in-memory implementation:

```python
# In LocalStorageClient
async def find_similar_topics(self, embedding, threshold=0.85, limit=5):
    """Find similar topics using cosine similarity in-memory."""
    results = []
    for topic in self._topics.values():
        if topic.embedding:
            sim = cosine_similarity(embedding, topic.embedding)
            if sim >= threshold:
                results.append({"id": topic.id, "similarity": sim})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]
```

This is straightforward since topic count is small (typically <200 topics).

---

## 5. Importing Local Data to Any Provider

### 5.1 Import Strategy

The local format is designed for one-pass sequential import:

1. **Read `manifest.json`** to validate compatibility (embedding model, dimensions)
2. **Import `vessels.json`** (reference data, import first)
3. **Import `topics.jsonl`** line by line:
   - Insert each topic with its pre-computed embedding
   - Map old UUID to new UUID if target DB generates its own IDs
4. **Import `documents.jsonl`** line by line:
   - Process in depth order (depth=0 first, then depth=1, etc.)
   - This ensures parent documents exist before children
   - Map old UUIDs to new UUIDs
5. **Import `chunks.jsonl`** line by line:
   - Map `document_id` to new UUID
   - Insert with pre-computed embedding (no re-embedding needed)
   - Map topic_ids in metadata to new UUIDs
6. **Import `archive/`** folder:
   - Upload each file to target storage (S3, GCS, Supabase Storage, etc.)
   - Path structure is preserved as-is

### 5.2 Provider-Specific Considerations

| Target | Embedding Import | Notes |
|---|---|---|
| Supabase (pgvector) | Direct array insert | Same as current production |
| Pinecone | `upsert(vectors=[...])` | Map chunk_id to Pinecone ID, metadata as payload |
| Qdrant | `upsert(points=[...])` | Similar to Pinecone, supports payload filtering |
| Weaviate | Batch import with vectors | Supports bringing pre-computed vectors |
| ChromaDB | `add(embeddings=[...])` | Simple local alternative |
| Plain PostgreSQL + pgvector | SQL INSERT | Identical to Supabase path |

### 5.3 Re-embedding Considerations

If switching embedding models (e.g., from `text-embedding-3-small` to a different model):
- `embedding_text` is preserved in chunks.jsonl -- this is the full text used for embedding
- Re-embed using the new model on `embedding_text` field
- Topic embeddings also need re-generation from `name` field
- No need to re-run LLM steps (context generation, topic extraction)

---

## 6. Risks and Considerations

### 6.1 Data Size

- **Embeddings dominate storage**: 1536 floats * 4 bytes * ~50K chunks = ~300MB binary, ~1.2GB as JSON text
- **Archive files**: Original attachments (PDFs, images) will be the largest component -- depends on email corpus size
- **Mitigation**: Embeddings can be stored in a separate binary file (numpy .npy) with a flag, but JSON is more portable

### 6.2 JSONL Consistency on Crash

- JSONL is append-only, so partial writes on crash leave a valid prefix
- But documents may reference chunks/topics not yet written
- **Mitigation**: The manifest.json is written last and includes counts. If manifest is missing or counts don't match, the ingest was incomplete

### 6.3 UUID Mapping on Import

- Local ingest generates UUIDs that may conflict with target DB
- Parent-child relationships use these UUIDs
- **Mitigation**: Import script must maintain an old-UUID-to-new-UUID mapping and rewrite `parent_id`, `root_id`, `path`, and `document_id` references

### 6.4 Topic Matching Quality

- In-memory cosine similarity is mathematically identical to pgvector's `<=>` operator
- No quality loss for local operation
- Topic count is small enough (<500) that brute-force search is fast

### 6.5 Processing Log State

- `ProgressTracker` uses `processing_log` table for resumability
- Local version needs to support resume (read existing JSONL on startup)
- File hash-based deduplication works the same way locally

### 6.6 Archive URIs

- Current URIs are relative paths like `/archive/{folder_id}/email.eml.md`
- These are provider-agnostic already -- they just need a base URL prefix per deployment
- No change needed for local storage

### 6.7 Concurrent Access

- Local JSONL files are NOT safe for concurrent writes
- Single-process ingest only (which is the current model anyway)
- No issue for the use case

### 6.8 What NOT to Store Locally

- API keys, credentials (obviously)
- Supabase connection strings
- Langfuse observability data (already goes to Langfuse cloud)

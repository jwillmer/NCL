---
purpose: Implementation plan for local-only ingest pipeline (JSONL-based)
status: approved
date: 2026-04-13
scope: extend LocalStorageClient, create adapters, wire into CLI
depends_on:
  - 02-local-storage-design.md
  - local-storage-sqlite-vs-files.md
---

# Local-Only Ingest Implementation Plan

## Overview

Extend the existing `tests/local_storage.py` code (~60% done) to support full local-only
ingest. The pipeline writes all output to JSONL files and a local archive folder, with no
Supabase dependency. Data can later be imported to any provider.

**Key decisions already made:**
- JSONL over SQLite (see `local-storage-sqlite-vs-files.md`)
- Embeddings included in chunks.jsonl (expensive to regenerate)
- In-memory cosine similarity for topic matching (~200 topics)
- Single-process, append-only writes

---

## Step 1: Extend LocalStorageClient with Missing Methods

**File:** `tests/local_storage.py` (move to `src/mtss/storage/local_client.py`)
**Depends on:** Nothing
**Effort:** Medium (2-3 hours)

The pipeline calls these methods on `db` that `LocalStorageClient` does not yet implement.
Each method below lists the caller and the exact signature to match from `SupabaseClient`.

### 1.1 Add In-Memory Indexes

Add these fields to `LocalStorageClient.__init__` / `__post_init__`:

```python
_documents_by_hash: Dict[str, Any] = field(default_factory=dict)   # file_hash -> doc
_documents_by_doc_id: Dict[str, Any] = field(default_factory=dict)  # doc_id -> doc
_documents_by_source_id: Dict[str, Any] = field(default_factory=dict)  # source_id -> doc
_topics: Dict[UUID, Topic] = field(default_factory=dict)            # id -> Topic
_topics_by_name: Dict[str, Topic] = field(default_factory=dict)     # name -> Topic
_chunks_by_document: Dict[UUID, List[Any]] = field(default_factory=dict)  # doc_id -> [chunks]
```

Update `insert_document` to populate all three document indexes.
Update `insert_chunks` to populate `_chunks_by_document`.

### 1.2 Missing Document Methods

**`get_document_by_hash(file_hash: str) -> Optional[Document]`**
- Called by: `pipeline.py:122` (legacy skip check when no VersionManager)
- Implementation: `return self._documents_by_hash.get(file_hash)`

**`get_document_by_id(doc_id: UUID) -> Optional[Document]`**
- Called by: `pipeline.py:95` (check existing doc status), `version_manager.py:68`,
  `hierarchy_manager.py:276`, `repair.py` passim
- Implementation: `return self._documents.get(doc_id)`

**`delete_document_for_reprocess(doc_id: UUID)`**
- Called by: `pipeline.py:98,113,118,131,144` (cleanup before re-insert)
- Synchronous method (not async) -- matches `SupabaseClient` signature
- Implementation:
  1. Remove doc from `_documents`, `_documents_by_hash`, `_documents_by_doc_id`,
     `_documents_by_source_id`
  2. Remove children (docs with `parent_id == doc_id`) from all indexes
  3. Remove chunks for doc and children from `_chunks` and `_chunks_by_document`
  4. Append tombstone record to `documents.jsonl`:
     `{"_deleted": true, "id": str(doc_id), "timestamp": ...}`

**`get_document_children(doc_id: UUID) -> List[Document]`**
- Called by: `repair.py:112`, `hierarchy_manager.py:244`
- Implementation: `return [d for d in self._documents.values() if d.parent_id == doc_id]`

**`get_document_by_source_id(source_id: str) -> Optional[Document]`**
- Already implemented but uses linear scan. Update to use `_documents_by_source_id` index.

**`get_document_by_doc_id(doc_id: str) -> Optional[Document]`**
- Already implemented but uses linear scan. Update to use `_documents_by_doc_id` index.

### 1.3 Missing Topic Methods

**`insert_topic(topic: Topic) -> Topic`**
- Called by: `topics.py:367` (TopicMatcher.get_or_create_topic)
- Implementation:
  1. Store in `_topics[topic.id]` and `_topics_by_name[topic.name]`
  2. Append to `topics.jsonl` via `_append_jsonl("topics.jsonl", self._topic_to_dict(topic))`
  3. Return topic

**`get_topic_by_name(name: str) -> Optional[Topic]`**
- Called by: `topics.py:341` (exact match check)
- Implementation: `return self._topics_by_name.get(name.lower().strip())`

**`get_topic_by_id(topic_id: UUID) -> Optional[Topic]`**
- Called by: `topics.py:394,440` (lookup after similarity match)
- Implementation: `return self._topics.get(topic_id)`

**`find_similar_topics(embedding: List[float], threshold: float = 0.85, limit: int = 5) -> List[Dict[str, Any]]`**
- Called by: `topics.py:349,389,432` (TopicMatcher dedup and query matching)
- Implementation: brute-force cosine similarity over `_topics.values()`
- Returns: `[{"id": UUID, "name": str, "display_name": str, "similarity": float}, ...]`
- See Step 6 for details.

**`increment_topic_counts(topic_ids: List[UUID], chunk_delta: int = 0, document_delta: int = 0) -> None`**
- Called by: `pipeline.py:407`, `repair.py:728`
- Implementation: update `topic.chunk_count` and `topic.document_count` in memory.
  Persist on flush (topic records rewritten).

**`update_chunks_topic_ids(document_id: UUID, topic_ids: List[str]) -> int`**
- Called by: `repair.py:724` (fix_missing_topics)
- Implementation: update `metadata["topic_ids"]` on all chunks for this document
  (and children where `root_id == document_id`) in `_chunks_by_document`.
  Return count of updated chunks.

**`update_chunks_topics_checked(document_id: UUID) -> int`**
- Called by: `repair.py:710` (mark as checked even when no topics found)
- Implementation: set `metadata["topics_checked"] = True` on all chunks for
  document and children. Return count of updated chunks.

### 1.4 Missing Vessel/Chunk Metadata Methods

**`get_all_vessels() -> List[Vessel]`**
- Called by: `ingest_cmd.py:185` (load vessel registry)
- Implementation: Read from `vessels.json` if it exists, else return `[]`.
  For local-only ingest, the caller passes vessels into
  `create_local_ingest_components()` so this just returns the stored list.

**`update_chunks_vessel_metadata(document_id, vessel_ids, vessel_types, vessel_classes) -> int`**
- NOT called during normal ingest pipeline (vessel metadata is set at chunk creation
  time in `pipeline.py:310-316` and `attachment_handler.py:276-284`).
- Only called by `entities_cmd.py` for re-tagging. NOT needed for local-only ingest.
- Implementation: skip (raise `NotImplementedError` or return 0).

**`update_document_archive_uris(doc_id, browse_uri, download_uri) -> None`**
- Already implemented but writes to a separate `archive_updates.jsonl`. For local-only
  ingest, also update the in-memory document object so subsequent lookups see the URIs.

**`update_document_status(doc_id, status, error_message) -> None`**
- Already implemented but writes to `status_updates.jsonl`. Also update in-memory doc.

### 1.5 Test Strategy

- Unit test each new method with in-memory data
- Verify JSONL output matches expected schema from `02-local-storage-design.md`
- Test `delete_document_for_reprocess` tombstone behavior
- Test `find_similar_topics` with known embeddings and threshold values

---

## Step 2: Add Embeddings to Chunk Serialization

**File:** `src/mtss/storage/local_client.py` (`_chunk_to_dict`)
**Depends on:** Step 1
**Effort:** Small (15 minutes)

### Changes

In `_chunk_to_dict`, replace:
```python
# Embedding is omitted (too large for JSONL)
```
with:
```python
"embedding": chunk.embedding,  # 1536-dim float vector
```

Add `page_number` to `_chunk_to_dict` (currently missing):
```python
"page_number": chunk.page_number,
```

### New: `_topic_to_dict`

```python
def _topic_to_dict(self, topic: Topic) -> Dict[str, Any]:
    return {
        "id": str(topic.id),
        "name": topic.name,
        "display_name": topic.display_name,
        "description": topic.description,
        "embedding": topic.embedding,
        "chunk_count": topic.chunk_count,
        "document_count": topic.document_count,
    }
```

### Test Strategy

- Verify chunks.jsonl includes embedding arrays
- Verify round-trip: write chunk -> read JSONL -> parse -> verify embedding length is 1536

---

## Step 3: Create LocalProgressTracker Adapter

**File:** `src/mtss/storage/local_progress_tracker.py` (new file)
**Depends on:** Step 1
**Effort:** Medium (1-2 hours)

### Interface

Must match `ProgressTracker` (from `src/mtss/storage/progress_tracker.py`). Methods
called by the pipeline:

| Method | Called by |
|---|---|
| `compute_file_hash(file_path: Path) -> str` | `pipeline.py:80` |
| `mark_started(file_path: Path, file_hash: str)` | `pipeline.py:146` |
| `mark_completed(file_path: Path)` | `pipeline.py:415` |
| `mark_failed(file_path: Path, error: str)` | `ingest_cmd.py:338,400` |
| `get_pending_files(source_dir: Path) -> List[Path]` | `ingest_cmd.py:223` |
| `get_failed_files(max_attempts: int = 3) -> List[Path]` | `ingest_cmd.py:221` |
| `get_processing_stats() -> Dict[str, int]` | `ingest_cmd.py:405` |
| `reset_stale_processing(max_age_minutes: int) -> int` | `ingest_cmd.py:215` |
| `get_outdated_files(source_dir, target_version) -> List[Path]` | `ingest_cmd.py:211` |

### Implementation

```python
class LocalProgressTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._log: Dict[str, Dict] = {}  # file_path -> record
        self._load_existing()

    def _load_existing(self):
        """Load processing_log.jsonl on startup for resume."""
        log_file = self.output_dir / "processing_log.jsonl"
        if log_file.exists():
            for line in log_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._log[record["file_path"]] = record

    def compute_file_hash(self, file_path: Path) -> str:
        """Same SHA-256 implementation as ProgressTracker."""
        # Identical to ProgressTracker.compute_file_hash

    async def get_pending_files(self, source_dir: Path) -> List[Path]:
        """Get files not yet processed (by hash)."""
        all_files = list(source_dir.glob("**/*.eml"))
        processed_hashes = {
            r["file_hash"] for r in self._log.values()
            if r["status"] in ("completed", "processing")
        }
        return [f for f in all_files if self.compute_file_hash(f) not in processed_hashes]

    async def mark_started(self, file_path: Path, file_hash: str):
        record = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
            "attempts": self._log.get(str(file_path), {}).get("attempts", 0),
        }
        self._log[str(file_path)] = record
        self._append_jsonl(record)

    async def mark_completed(self, file_path: Path):
        record = self._log.get(str(file_path), {})
        record["status"] = "completed"
        record["completed_at"] = datetime.utcnow().isoformat()
        self._log[str(file_path)] = record
        self._append_jsonl(record)

    async def mark_failed(self, file_path: Path, error: str):
        record = self._log.get(str(file_path), {})
        record["status"] = "failed"
        record["last_error"] = error[:500]
        record["attempts"] = record.get("attempts", 0) + 1
        self._log[str(file_path)] = record
        self._append_jsonl(record)

    async def get_outdated_files(self, source_dir, target_version) -> List[Path]:
        """Not applicable for local-only (no version tracking in local DB)."""
        return []
```

### Notes

- `get_outdated_files` returns `[]` because local-only ingest does not track
  ingest versions in a queryable database. Reprocessing outdated files is a
  remote-DB operation.
- `_append_jsonl` writes to `processing_log.jsonl`. On resume, the file may
  contain multiple records for the same `file_path` (started then completed).
  The in-memory dict always holds the latest state. A final flush could
  compact the file but is not required.

### Test Strategy

- Test resume: write some records, create new tracker, verify pending files excludes them
- Test mark_started -> mark_completed flow
- Test mark_failed increments attempts

---

## Step 4: Create LocalUnsupportedFileLogger Adapter

**File:** `src/mtss/storage/local_unsupported_logger.py` (new file)
**Depends on:** Step 1
**Effort:** Small (30 minutes)

### Interface

Must match `UnsupportedFileLogger` (from `src/mtss/storage/unsupported_file_logger.py`).

| Method | Called by |
|---|---|
| `log_unsupported_file(file_path, reason, source_eml_path, source_zip_path, parent_document_id)` | `attachment_handler.py:108,265,326,396,407` |

### Implementation

```python
class LocalUnsupportedFileLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    async def log_unsupported_file(
        self,
        file_path: Path,
        reason: str,
        source_eml_path: Optional[str] = None,
        source_zip_path: Optional[str] = None,
        parent_document_id: Optional[UUID] = None,
    ):
        data = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size_bytes": file_path.stat().st_size if file_path.exists() else None,
            "mime_type": mimetypes.guess_type(str(file_path))[0],
            "file_extension": file_path.suffix.lower() if file_path.suffix else None,
            "reason": reason,
            "source_eml_path": source_eml_path,
            "source_zip_path": source_zip_path,
            "parent_document_id": str(parent_document_id) if parent_document_id else None,
            "discovered_at": datetime.utcnow().isoformat(),
        }
        self._append_jsonl("ingest_events.jsonl", data)
```

### Notes

- The `log_ingest_event` method on `LocalStorageClient` (line 132-154 of current code)
  already handles a different kind of event logging. `UnsupportedFileLogger` uses a
  different table (`ingest_events` with upsert on `file_path`). The local adapter
  simply appends to the same `ingest_events.jsonl` but with the schema from the design doc.

### Test Strategy

- Verify JSONL output schema matches `02-local-storage-design.md` section 3.6
- Verify file_size_bytes is captured when file exists

---

## Step 5: Add Dependency Injection to ArchiveGenerator

**File:** `src/mtss/ingest/archive_generator.py` (line 103-113)
**Depends on:** Nothing
**Effort:** Small (15 minutes)

### Change

```python
# Before (line 103-113):
class ArchiveGenerator:
    def __init__(self, ingest_root: Optional[Path] = None):
        from ..config import get_settings
        settings = get_settings()
        self.storage = ArchiveStorage()
        self.ingest_root = ingest_root or settings.eml_source_dir

# After:
class ArchiveGenerator:
    def __init__(self, ingest_root: Optional[Path] = None, storage=None):
        from ..config import get_settings
        settings = get_settings()
        self.storage = storage or ArchiveStorage()
        self.ingest_root = ingest_root or settings.eml_source_dir
```

This is a backward-compatible change. Existing callers that do not pass `storage`
get the default `ArchiveStorage()` behavior. Local-only callers pass a
`LocalBucketStorage` instance.

`LocalBucketStorage` already has the same method signatures as `ArchiveStorage`:
`upload_file`, `upload_text`, `download_file`, `download_text`, `file_exists`,
`delete_folder`, `list_files`. Duck typing works.

### Test Strategy

- Verify existing tests still pass (no change to default behavior)
- Verify `ArchiveGenerator(storage=LocalBucketStorage(...))` works

---

## Step 6: In-Memory Topic Similarity Search

**File:** `src/mtss/storage/local_client.py` (inside LocalStorageClient)
**Depends on:** Step 1, Step 2
**Effort:** Small (30 minutes)

### Implementation

```python
import math

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

async def find_similar_topics(
    self,
    embedding: List[float],
    threshold: float = 0.85,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    results = []
    for topic in self._topics.values():
        if topic.embedding:
            sim = _cosine_similarity(embedding, topic.embedding)
            if sim >= threshold:
                results.append({
                    "id": topic.id,
                    "name": topic.name,
                    "display_name": topic.display_name,
                    "similarity": sim,
                })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]
```

### Performance

- ~200 topics, 1536-dim vectors
- Brute-force: ~200 * 1536 multiply-adds = ~300K FLOPs
- Pure Python: < 1ms per query
- No numpy dependency needed

### Test Strategy

- Test with 2 known similar embeddings, verify match
- Test threshold filtering (below threshold -> empty result)
- Test limit parameter

---

## Step 7: Create `create_local_ingest_components()` Factory

**File:** `src/mtss/ingest/components.py`
**Depends on:** Steps 1-6
**Effort:** Medium (1 hour)

### New Function

```python
def create_local_ingest_components(
    output_dir: Path,
    source_dir: Path,
    vessels: Optional[list] = None,
    enable_topics: bool = True,
) -> IngestComponents:
    """Create ingest components backed by local storage.

    Mirrors create_ingest_components() but uses LocalStorageClient,
    LocalBucketStorage, and LocalProgressTracker instead of Supabase.

    Args:
        output_dir: Root directory for local output (JSONL + archive).
        source_dir: Root directory for email ingestion.
        vessels: Optional list of Vessel objects for VesselMatcher.
        enable_topics: Whether to enable topic extraction.

    Returns:
        IngestComponents with local storage backends.
    """
    from ..parsers.attachment_processor import AttachmentProcessor
    from ..parsers.chunker import ContextGenerator, DocumentChunker
    from ..parsers.eml_parser import EMLParser
    from ..processing.embeddings import EmbeddingGenerator
    from ..processing.topics import TopicExtractor, TopicMatcher
    from ..processing.vessel_matcher import VesselMatcher
    from ..storage.local_client import LocalStorageClient, LocalBucketStorage

    db = LocalStorageClient(output_dir=output_dir / "database")
    bucket = LocalBucketStorage(bucket_dir=output_dir / "archive")
    embeddings = EmbeddingGenerator()

    topic_extractor = None
    topic_matcher = None
    if enable_topics:
        topic_extractor = TopicExtractor()
        topic_matcher = TopicMatcher(db, embeddings)

    return IngestComponents(
        db=db,
        eml_parser=EMLParser(),
        attachment_processor=AttachmentProcessor(),
        hierarchy_manager=HierarchyManager(db, ingest_root=source_dir),
        embeddings=embeddings,
        archive_generator=ArchiveGenerator(ingest_root=source_dir, storage=bucket),
        context_generator=ContextGenerator(),
        chunker=DocumentChunker(),
        archive_storage=bucket,  # Duck-typed: same interface as ArchiveStorage
        vessel_matcher=VesselMatcher(vessels) if vessels else None,
        topic_extractor=topic_extractor,
        topic_matcher=topic_matcher,
    )
```

### Type Annotations

The `IngestComponents` dataclass uses `SupabaseClient` and `ArchiveStorage` type hints.
For local-only to work, the type hints need to be relaxed or the dataclass needs to
use `Any` / Protocol types. Since the type hints are in `TYPE_CHECKING` blocks (line 8
of `components.py`), they are strings at runtime and do not cause errors. Duck typing
works.

However, `HierarchyManager.__init__` (line 33 of `hierarchy_manager.py`) has an explicit
type annotation `db_client: SupabaseClient`. This is a runtime import. Two options:

1. Change the type hint to `Any` (simple, backward-compatible)
2. Create a `Protocol` class that both `SupabaseClient` and `LocalStorageClient` implement

**Recommendation:** Change to `Any` for now. The Protocol approach is cleaner but adds
scope. The `db_client` is used only via method calls that both classes implement.

Similarly, `VersionManager.__init__` (line 40) takes `db: Optional[SupabaseClient]`.
For local-only, we create a `LocalVersionManager` (see below).

### Test Strategy

- Verify function creates all components successfully
- Verify `IngestComponents.db` is a `LocalStorageClient` instance
- Verify `IngestComponents.archive_storage` is a `LocalBucketStorage` instance

---

## Step 8: Add `--local-only` Flag to CLI Ingest Command

**File:** `src/mtss/cli/ingest_cmd.py`
**Depends on:** Steps 1-7
**Effort:** Medium (1-2 hours)

### CLI Changes

Add parameter to the `ingest` command:

```python
local_only: bool = typer.Option(
    False,
    "--local-only",
    help="Write output to local JSONL files instead of Supabase",
),
output_dir: Optional[Path] = typer.Option(
    None,
    "--output-dir",
    "-o",
    help="Output directory for local-only mode (default: data/local-ingest)",
),
```

### `_ingest()` Changes

Add a branch at the component initialization section (around line 179-193):

```python
if local_only:
    from ..ingest.components import create_local_ingest_components
    from ..storage.local_client import LocalStorageClient
    from ..storage.local_progress_tracker import LocalProgressTracker
    from ..storage.local_unsupported_logger import LocalUnsupportedFileLogger
    from ..ingest.local_version_manager import LocalVersionManager

    output = output_dir or Path("data/local-ingest")
    output.mkdir(parents=True, exist_ok=True)

    # Load vessels from file if exists, or empty list
    vessels = _load_local_vessels(output)

    components = create_local_ingest_components(
        output_dir=output,
        source_dir=source_dir,
        vessels=vessels,
        enable_topics=True,
    )
    db = components.db
    tracker = LocalProgressTracker(output)
    unsupported_logger = LocalUnsupportedFileLogger(output)
    version_manager = LocalVersionManager(db)

    console.print(f"[green]Local-only mode: writing to {output}[/green]")
else:
    # Existing Supabase initialization (unchanged)
    db = SupabaseClient()
    tracker = ProgressTracker(db)
    unsupported_logger = UnsupportedFileLogger(db)
    version_manager = VersionManager(db)
    vessels = await db.get_all_vessels()
    components = create_ingest_components(
        db=db, source_dir=source_dir, vessels=vessels, enable_topics=True,
    )
```

The rest of the `_ingest()` function works unchanged because all components share
the same interfaces via duck typing.

### Cleanup Section

Replace the `await db.close()` in the `finally` block with:

```python
finally:
    if local_only:
        await db.flush()  # Write final state
        _write_manifest(output, db, source_dir)
    else:
        await db.close()
```

### Test Strategy

- Test `--local-only` flag creates output directory
- Test that ingest produces `documents.jsonl`, `chunks.jsonl`, `topics.jsonl`,
  `processing_log.jsonl`, `ingest_events.jsonl`, `manifest.json`
- Test that archive folder contains expected structure

---

## Step 9: Add `manifest.json` Writer

**File:** `src/mtss/storage/local_client.py` or `src/mtss/cli/ingest_cmd.py`
**Depends on:** Steps 1, 8
**Effort:** Small (30 minutes)

### Implementation

```python
def write_manifest(output_dir: Path, db: LocalStorageClient, source_dir: Path):
    """Write manifest.json after successful ingest."""
    from ..config import get_settings
    settings = get_settings()

    manifest = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "ingest_version": settings.current_ingest_version,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "source_dir": str(source_dir),
        "counts": {
            "documents": len(db._documents),
            "chunks": len(db._chunks),
            "topics": len(db._topics),
            "archive_files": _count_archive_files(output_dir / "archive"),
        },
        "settings_snapshot": {
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "context_llm_model": settings.context_llm_model,
            "current_ingest_version": settings.current_ingest_version,
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

### Notes

- Written LAST, after all JSONL files are complete
- If manifest exists and counts match, the ingest completed successfully
- If manifest is missing, the ingest was interrupted

### Test Strategy

- Verify manifest.json is valid JSON with all required fields
- Verify counts match actual JSONL record counts

---

## Step 10: Flush and Validation Logic

**File:** `src/mtss/storage/local_client.py`
**Depends on:** Steps 1-2
**Effort:** Small (30 minutes)

### `flush()` Method

```python
async def flush(self) -> None:
    """Write final consistent state to JSONL files.

    Called at end of ingest to ensure:
    1. Topic counts reflect final state (may have been incremented in-memory)
    2. Document status updates are reflected in main documents.jsonl
    """
    # Rewrite topics.jsonl with final counts
    topics_path = self.output_dir / "topics.jsonl"
    with open(topics_path, "w", encoding="utf-8") as f:
        for topic in self._topics.values():
            f.write(json.dumps(self._topic_to_dict(topic), default=str) + "\n")

    # Write vessels.json snapshot
    if self._vessels:
        vessels_path = self.output_dir / "vessels.json"
        vessels_data = [self._vessel_to_dict(v) for v in self._vessels]
        vessels_path.write_text(json.dumps(vessels_data, indent=2), encoding="utf-8")
```

### Notes

- Topics are written as append-only during ingest (new topics), but `increment_topic_counts`
  only updates in-memory. The flush rewrites `topics.jsonl` with final counts.
- `documents.jsonl` and `chunks.jsonl` are append-only and do not need rewriting.
  Status updates go to `status_updates.jsonl`. On import, the consumer reads both
  files and applies updates.
- Alternative: rewrite `documents.jsonl` with final status merged in. This is simpler
  for the consumer but more I/O during flush. **Decision:** Rewrite documents.jsonl
  during flush to produce a single source of truth per entity.

### Test Strategy

- Test that flush produces valid JSONL files
- Test that topic counts in flushed file match accumulated deltas

---

## Step 11: LocalVersionManager

**File:** `src/mtss/ingest/local_version_manager.py` (new file)
**Depends on:** Step 1
**Effort:** Small (30 minutes)

### Rationale

`VersionManager` (line 40-42) creates a `SupabaseClient()` if no `db` is provided,
and its `check_document` method calls `db.get_document_by_doc_id` and
`db.get_document_by_source_id`. These methods will exist on `LocalStorageClient`
after Step 1.

However, `VersionManager.__init__` has an explicit `SupabaseClient` type hint and
the default `db or SupabaseClient()` would fail without Supabase credentials.

**Solution:** Create a thin `LocalVersionManager` that wraps the same logic but
accepts `LocalStorageClient`.

```python
class LocalVersionManager:
    def __init__(self, db):
        self.db = db
        from ..config import get_settings
        self.current_version = get_settings().current_ingest_version

    async def check_document(self, source_id: str, file_hash: str) -> IngestDecision:
        """Same logic as VersionManager.check_document."""
        doc_id = compute_doc_id(source_id, file_hash)
        existing = await self.db.get_document_by_doc_id(doc_id)
        if existing:
            existing_version = getattr(existing, "ingest_version", 1)
            if existing_version < self.current_version:
                return IngestDecision(
                    action="reprocess",
                    reason=f"Ingest logic upgraded from v{existing_version} to v{self.current_version}",
                    existing_doc_id=existing.id,
                )
            return IngestDecision(
                action="skip", reason="Already processed", existing_doc_id=existing.id,
            )
        old_version = await self.db.get_document_by_source_id(source_id)
        if old_version:
            return IngestDecision(
                action="update", reason="Content changed", existing_doc_id=old_version.id,
            )
        return IngestDecision(action="insert", reason="New document")
```

### Alternative

Instead of creating a new class, make `VersionManager` accept `Any` for `db`
and remove the default `SupabaseClient()` construction. This is simpler but changes
the existing class.

**Recommendation:** Change `VersionManager.__init__` signature to accept `Any`:

```python
def __init__(self, db=None):
    if db is None:
        from ..storage.supabase_client import SupabaseClient
        db = SupabaseClient()
    self.db = db
```

This lazy import avoids requiring Supabase credentials when a local db is passed.
Then local-only code just does `VersionManager(db=local_db)`.

### Test Strategy

- Test that `VersionManager(db=local_db)` does not import SupabaseClient
- Test check_document returns correct actions for insert/skip/reprocess/update

---

## Step 12: HierarchyManager Type Relaxation

**File:** `src/mtss/ingest/hierarchy_manager.py` (line 33)
**Depends on:** Nothing
**Effort:** Tiny (5 minutes)

### Change

```python
# Before (line 18):
from ..storage.supabase_client import SupabaseClient

# Before (line 33):
def __init__(self, db_client: SupabaseClient, ingest_root: Optional[Path] = None):

# After (line 18): remove or move to TYPE_CHECKING
# After (line 33):
def __init__(self, db_client, ingest_root: Optional[Path] = None):
```

Move the import into `TYPE_CHECKING` block if type hints are desired:

```python
if TYPE_CHECKING:
    from ..storage.supabase_client import SupabaseClient
```

### Test Strategy

- Verify existing tests still pass
- Verify `HierarchyManager(db_client=local_db)` works

---

## Step 13: Move LocalStorageClient Out of tests/

**File:** Move `tests/local_storage.py` -> `src/mtss/storage/local_client.py`
**Depends on:** Nothing (do first or last)
**Effort:** Small (15 minutes)

### Rationale

The code is production code (used by the ingest pipeline), not test infrastructure.
Moving it to `src/mtss/storage/` makes imports cleaner and follows package conventions.

### Steps

1. Copy `tests/local_storage.py` to `src/mtss/storage/local_client.py`
2. Update all imports referencing `tests.local_storage`
3. Keep `tests/local_storage.py` as a re-export for backward compatibility (or delete
   if nothing imports from it in tests)

### Test Strategy

- All existing tests that use `LocalStorageClient` still pass

---

## Step 14: Test Plan

### Unit Tests

| Test | Covers |
|---|---|
| `test_local_client_insert_document` | Insert + index population |
| `test_local_client_get_document_by_hash` | Hash-based lookup |
| `test_local_client_get_document_by_id` | UUID lookup |
| `test_local_client_delete_for_reprocess` | Deletion + tombstone |
| `test_local_client_topic_crud` | insert/get_by_name/get_by_id |
| `test_local_client_find_similar_topics` | Cosine similarity |
| `test_local_client_increment_topic_counts` | Counter updates |
| `test_local_client_chunk_with_embedding` | Embedding serialization |
| `test_local_client_flush` | Final state write |
| `test_local_progress_tracker_resume` | Load existing + filter |
| `test_local_progress_tracker_lifecycle` | started -> completed flow |
| `test_local_unsupported_logger` | JSONL output schema |
| `test_archive_generator_injection` | LocalBucketStorage works |
| `test_create_local_components` | Factory produces valid components |
| `test_manifest_writer` | JSON output + counts |

### Integration Test

1. Create a small test fixture with 2-3 EML files (1 with attachments)
2. Run `_ingest()` with `local_only=True`
3. Verify:
   - `documents.jsonl` has correct record count
   - `chunks.jsonl` has embeddings (check first record)
   - `topics.jsonl` has entries (if topics were extracted)
   - `processing_log.jsonl` shows all files as completed
   - `archive/` folder has expected structure
   - `manifest.json` counts match JSONL line counts
4. Verify round-trip: read all JSONL files and validate schema

### Regression Safety

- All existing tests must pass unmodified (changes are additive)
- The `ArchiveGenerator` change (Step 5) is backward-compatible
- `HierarchyManager` type relaxation (Step 12) is backward-compatible
- `VersionManager` lazy import (Step 11) is backward-compatible

---

## Implementation Order

```
Step 13: Move local_storage.py to src/          (enables clean imports)
Step 5:  ArchiveGenerator DI                     (tiny, unblocks Step 7)
Step 12: HierarchyManager type relaxation        (tiny, unblocks Step 7)
Step 11: VersionManager lazy import              (tiny, unblocks Step 8)
Step 1:  Extend LocalStorageClient               (core work)
Step 2:  Embeddings in chunk serialization        (depends on Step 1)
Step 6:  Topic similarity search                  (depends on Step 1)
Step 3:  LocalProgressTracker                     (independent)
Step 4:  LocalUnsupportedFileLogger               (independent)
Step 10: Flush/validation logic                   (depends on Steps 1-2)
Step 7:  create_local_ingest_components()         (depends on Steps 1-6)
Step 8:  CLI --local-only flag                    (depends on Step 7)
Step 9:  Manifest writer                          (depends on Step 8)
Step 14: Tests                                    (last)
```

Total estimated effort: 8-12 hours

---

## Review Findings

The following issues were discovered during line-by-line review of the pipeline and
its callers, and have been addressed in the plan above.

### Finding 1: `update_document_status` and `update_document_archive_uris` Must Update In-Memory State

**Problem:** The existing `LocalStorageClient` writes status updates and archive URI
updates to separate JSONL files (`status_updates.jsonl`, `archive_updates.jsonl`) but
does NOT update the in-memory `_documents` dict. This means:
- `pipeline.py:95` calls `db.get_document_by_id(decision.existing_doc_id)` and checks
  `existing.status != ProcessingStatus.COMPLETED`. If status was set to COMPLETED but
  only in the JSONL file, the in-memory doc still shows `pending`, causing spurious
  re-processing.
- `attachment_handler.py:182` reads `components.archive_generator.storage.file_exists(cached_md_path)`
  which works fine with local bucket storage, but the doc's `archive_browse_uri` accessed at
  `repair.py:159` would be stale.

**Resolution:** Added to Step 1.4 -- both `update_document_status` and
`update_document_archive_uris` must also update the in-memory document object.
The JSONL append can remain for audit trail, but the in-memory state must be
authoritative.

### Finding 2: `attachment_handler.py` Accesses `components.archive_generator.storage` Directly

**Problem:** `attachment_handler.py:182` does:
```python
if components.archive_generator.storage.file_exists(cached_md_path):
    cached_md = components.archive_generator.storage.download_text(cached_md_path)
```

This accesses the storage backend directly via the archive generator. With DI (Step 5),
`archive_generator.storage` will be a `LocalBucketStorage` instance, which has the same
`file_exists` and `download_text` methods. No additional change needed.

**Resolution:** Already handled by Step 5 (duck typing).

### Finding 3: `repair.py` Uses `components.archive_storage` Directly

**Problem:** `repair.py` lines 299, 334, 359, 374 access `components.archive_storage`
directly for bucket operations (`file_exists`, `delete_folder`, `upload_file`,
`download_file`).

The `IngestComponents` dataclass has an `archive_storage` field (line 42 of `components.py`).
In `create_local_ingest_components` (Step 7), this must be set to the same
`LocalBucketStorage` instance used by the `ArchiveGenerator`.

**Resolution:** Added to Step 7 -- `archive_storage=bucket` ensures `repair.py` gets
the local bucket. Already in the plan.

### Finding 4: `VersionManager` Default Constructor Creates SupabaseClient

**Problem:** `VersionManager.__init__` (line 46) does `self.db = db or SupabaseClient()`.
If `db` is `None`, this imports and constructs a `SupabaseClient`, which requires
Supabase environment variables. In local-only mode, these variables may not be set.

**Resolution:** Added Step 11 -- lazy import of `SupabaseClient` only when `db is None`.
This is backward-compatible (existing code always passes `db`).

### Finding 5: `HierarchyManager` Has Runtime Import of SupabaseClient

**Problem:** `hierarchy_manager.py` line 18 has `from ..storage.supabase_client import SupabaseClient`.
This is a module-level import, not inside `TYPE_CHECKING`. Even if the type hint is
relaxed, the import runs at module load time. If Supabase SDK is installed (it is), this
import succeeds even without credentials. But it is conceptually wrong to import
`SupabaseClient` in a module that should work without Supabase.

**Resolution:** Added Step 12 -- move import to `TYPE_CHECKING` block. The `self.db`
attribute is used purely via duck-typed method calls.

### Finding 6: `log_ingest_event` on `LocalStorageClient` vs `UnsupportedFileLogger`

**Problem:** There are TWO different event logging mechanisms:
1. `db.log_ingest_event(document_id, event_type, severity, ...)` -- called by
   `SupabaseClient` directly (line 138-149 of `supabase_client.py`). The existing
   `LocalStorageClient` already implements this (line 132-154 of current code).
2. `UnsupportedFileLogger.log_unsupported_file(file_path, reason, ...)` -- uses
   `db.client.table("ingest_events").upsert(...)` which bypasses the facade.

For local-only mode, the `LocalUnsupportedFileLogger` (Step 4) writes to the same
`ingest_events.jsonl` file but with a different schema. The `log_ingest_event` on
`LocalStorageClient` also writes to `ingest_events.jsonl` but with a different schema.

**Resolution:** These are conceptually different event types going to the same file.
The JSONL format handles this naturally -- each line is self-describing. The consumer
can differentiate by checking for `"event_type"` vs `"reason"` fields.

### Finding 7: `repair.py` and `ingest-update` -- Do We Need Local Versions?

**Problem:** The `ingest-update` command (`maintenance_cmd.py:116`) uses
`create_ingest_components()` (the Supabase version) and runs repair operations. It calls
`find_orphaned_documents()`, `scan_ingest_issues()`, `fix_document_issues()`.

For local-only mode, `ingest-update` is not needed because:
- Orphaned documents: local ingest is a fresh run, no orphans exist
- Missing archives: local ingest writes archives as part of processing
- Missing context/chunks/topics: same -- fresh run includes everything

**Resolution:** The `ingest-update` and `repair` commands are NOT needed for local-only
mode. They are maintenance commands for an existing Supabase database. The `--local-only`
flag only applies to the `ingest` command. No changes needed to `repair.py` or
`maintenance_cmd.py`.

### Finding 8: `pipeline.py` line 391 Calls `archive_generator.regenerate_email_markdown`

**Problem:** This method (line 502-549 of `archive_generator.py`) calls
`self.storage.file_exists()` and `self.storage.upload_text()`. With DI, `self.storage`
is a `LocalBucketStorage`. This works via duck typing.

**Resolution:** No additional change needed. Already handled by Step 5.

### Finding 9: `get_all_root_source_ids` and `delete_orphaned_documents`

**Problem:** These methods are used by `repair.py:36,56` for orphan detection. They are
NOT called during normal ingest.

**Resolution:** Not needed for local-only ingest. Can be left unimplemented (or
raise `NotImplementedError`).

### Finding 10: Chunk `heading_path` vs `section_path` Constructor Argument

**Problem:** `pipeline.py:327` passes `heading_path=["Email Body", ...]` to the `Chunk`
constructor, but the `Chunk` model (line 31) has the field named `section_path` with a
`@property` alias for `heading_path`. Pydantic models accept field name OR alias in
the constructor depending on configuration.

**Resolution:** This is an existing pattern that works. No change needed for local-only.

### Finding 11: `_doc_to_dict` Missing Fields

**Problem:** The existing `_doc_to_dict` (line 43-61 of `local_storage.py`) is missing
several fields that are in the design doc schema:
- `content_version` (included but as `ingest_version`)
- `email_metadata` (full object not serialized)
- `attachment_metadata` (not serialized)
- `archive_path` (not serialized)
- `path` (hierarchy path as list of UUIDs, not serialized)

**Resolution:** Extend `_doc_to_dict` to include all fields from the design doc schema
(section 3.2). This is part of Step 1 work. Add:

```python
"content_version": doc.content_version,
"path": doc.path,
"archive_path": doc.archive_path,
"email_metadata": {
    "subject": doc.email_metadata.subject,
    "participants": doc.email_metadata.participants,
    ...
} if doc.email_metadata else None,
"attachment_metadata": {
    "content_type": doc.attachment_metadata.content_type,
    ...
} if doc.attachment_metadata else None,
```

### Finding 12: `db.close()` vs `db.flush()`

**Problem:** The pipeline calls `await db.close()` in the `finally` block
(`ingest_cmd.py:415`). For local storage, we need `flush()` instead of `close()`.
The existing `LocalStorageClient.close()` is a no-op.

**Resolution:** Change `close()` to call `flush()` internally, or add explicit
`flush()` call before `close()` in the local-only code path. Added to Step 8
(CLI changes) and Step 10 (flush logic).

### Finding 13: `pipeline.py` Passes `on_progress` Callback

**Problem:** The `on_progress` callback in `pipeline.py:59` and used throughout for
progress bar updates. This works identically for local-only since it is a display
concern, not a storage concern.

**Resolution:** No change needed.

### Finding 14: Concurrent File Processing Safety

**Problem:** The ingest command uses `asyncio.gather` with multiple workers
(`ingest_cmd.py:383`). JSONL append-only writes are not thread-safe. However, the
workers are asyncio coroutines running in a single thread (the event loop). Since
`_append_jsonl` uses synchronous file I/O (`open(..., "a")`), all writes are
serialized by the GIL and the event loop. No concurrency issue.

**Resolution:** No change needed. But add a comment to `LocalStorageClient` noting
this assumption.

### Finding 15: `LaneClassifier` in `ingest_cmd.py`

**Problem:** `LaneClassifier` (line 243-255) uses `eml_parser` to classify files
into fast/slow queues. This is pure Python with no storage dependency.

**Resolution:** No change needed. Works identically for local-only.

### Finding 16: `IngestReportWriter` in `ingest_cmd.py`

**Problem:** `IngestReportWriter` (line 205-206) writes failure reports to
`data/reports/`. This is already local file I/O, not Supabase. Works unchanged.

**Resolution:** No change needed.

### Finding 17: `vessels.json` Loading for Local-Only

**Problem:** In Supabase mode, `ingest_cmd.py:185` calls `await db.get_all_vessels()`
to load the vessel registry. In local-only mode, there is no Supabase to query.

**Resolution:** Add a `_load_local_vessels(output_dir)` helper that reads `vessels.json`
from the output directory if it exists (from a previous run), or an empty list. Users
can pre-populate `vessels.json` manually or via a separate setup step. Added to Step 8.

Alternatively, accept a `--vessels-file` CLI option for local-only mode.

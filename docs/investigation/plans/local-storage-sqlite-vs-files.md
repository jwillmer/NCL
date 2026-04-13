---
purpose: Evaluate SQLite vs JSONL vs hybrid for local-only ingest storage
status: recommendation
date: 2026-04-13
scope: local ingest storage backend decision
---

# Local Storage Backend: SQLite vs JSONL vs Hybrid

## 1. Actual Use Cases During Local Ingest

Before comparing technologies, clarify what local storage actually needs to do:

| Use Case | Access Pattern | Frequency |
|---|---|---|
| Resume/skip detection | Lookup by `file_hash` or `source_id` | Every file (6,289x) |
| Duplicate document check | Lookup by `doc_id` or `file_hash` | Every file |
| Topic deduplication | Cosine similarity on ~200 topic vectors (1536-dim) | Every email with topics |
| Topic CRUD | Insert/get by name/id | Per extracted topic |
| Increment topic counts | Update counters | Per email |
| Document insert | Append | Per email + attachments |
| Chunk insert (with embeddings) | Batch append | Per document |
| Status updates | Update by document ID | Per email |
| Progress tracking | Upsert by file_path, lookup by hash | Per file |
| Ingest event logging | Append | Per warning/error |
| Post-ingest export to production | Sequential full scan | Once |
| Post-ingest verification | Count records, check completeness | Once |

**Not needed locally:** Full vector search on chunks (production only), RAG queries (production only).

## 2. Technology Assessment

### 2.1 sqlite-vec (Vector Search Extension)

**Current state (2025-2026):**
- sqlite-vec v0.1.x is the successor to sqlite-vss (which is deprecated)
- Written in pure C, no external dependencies
- Installed via `pip install sqlite-vec`
- Brute-force KNN search (no index training required)
- Supports cosine similarity on float vectors of any dimension

**Performance with 1536-dim vectors:**
- 100K vectors at 1536-dim: ~105ms per KNN query (brute-force scan)
- For ~200 topic vectors: sub-millisecond (trivially fast)
- For ~50K chunk vectors: ~50ms per query (but we do NOT need this locally)

**Storage size for the dataset:**
- 50K vectors x 1536 dims x 4 bytes = ~307 MB for vector data alone
- Plus metadata, indexes, SQLite overhead: estimated ~400-500 MB total for chunks table
- Topic vectors (200 x 1536 x 4 bytes) = ~1.2 MB (negligible)

**Windows compatibility risk:**
- Known issues loading sqlite-vec on Windows 11 (GitHub issues #13, #45)
- The pre-compiled extension sometimes fails with "The specified module could not be found"
- Windows statically compiles its own SQLite, making extension loading fragile
- This is a development-environment risk since the project runs on Windows

### 2.2 aiosqlite (Async SQLite)

- Latest release: v0.21.0 (December 2025), actively maintained
- Wraps standard `sqlite3` module with async/await via a background thread
- Single shared thread per connection (serialized writes)
- Supports `enable_load_extension()` for loading sqlite-vec
- Compatible with Python 3.9+
- Drop-in async interface: `await db.execute(...)`, `await db.fetchone()`

### 2.3 JSONL (Current Proposal)

- Already partially implemented in `tests/local_storage.py`
- Append-only files, human-readable
- No query capability beyond full scan
- No crash recovery (partial line writes produce corrupt records)
- In-memory dicts for runtime lookups (loaded from JSONL on startup for resume)

## 3. Comparison Table

| Criterion | SQLite + sqlite-vec | JSONL (current proposal) | Hybrid (SQLite metadata + JSONL/binary embeddings) |
|---|---|---|---|
| **Async compatibility** | Good (aiosqlite wraps in background thread) | Good (file I/O via `anyio.to_thread`) | Good (same as SQLite) |
| **Crash recovery** | Excellent (WAL mode, ACID transactions) | Poor (partial writes corrupt last line, no rollback) | Good (SQLite for metadata) |
| **Resume capability** | Excellent (indexed lookups by hash/source_id) | Adequate (load full JSONL into memory on restart) | Good (same as SQLite) |
| **Topic similarity search** | Native via sqlite-vec cosine distance | Manual in-memory cosine similarity (numpy/math) | Either approach works |
| **Chunk vector search** | Supported but NOT NEEDED locally | Not supported | N/A |
| **Portability** | Single `.db` file + archive folder | Multiple `.jsonl` files + archive folder | `.db` + `.jsonl` + archive folder |
| **Human readability** | Requires SQLite viewer | Text editor / `jq` | Mixed |
| **Import to production** | Read rows, transform, insert to PostgreSQL | Read lines, parse JSON, insert | Read from both sources |
| **Implementation effort** | High: new schema, new repository class, extension loading, Windows risk | Low: extend existing `LocalStorageClient` (~8 missing methods) | Medium: new schema + embedding file handling |
| **Existing code reuse** | Minimal (new repository layer needed) | High (`tests/local_storage.py` is 80% there) | Medium |
| **Windows reliability** | Risk (sqlite-vec extension loading issues on Win11) | No risk (pure Python) | Same Windows risk if using sqlite-vec |
| **Data integrity** | Strong (foreign keys, constraints, transactions) | Weak (append-only, no referential checks) | Medium |
| **File size (full dataset)** | ~500 MB single `.db` file | ~1.2 GB across JSONL files (JSON float encoding is verbose) | ~200 MB `.db` + ~300 MB binary embeddings |
| **Dependencies** | `aiosqlite`, `sqlite-vec` (+ C extension) | None (stdlib `json` + `pathlib`) | `aiosqlite` + numpy (for binary embeddings) |

## 4. Analysis of Each Option

### 4.1 SQLite + sqlite-vec

**Advantages:**
- ACID transactions mean no corrupt state on crash/interruption
- Indexed lookups for resume/skip detection are O(log n) vs O(n) for JSONL
- Schema mirrors PostgreSQL tables closely, reducing impedance mismatch
- Single `.db` file is easier to manage than 6+ JSONL files
- Topic similarity search works natively without custom code

**Disadvantages:**
- sqlite-vec has known Windows loading issues -- this is a real risk for this project
- Requires writing a complete new `SqliteStorageClient` repository layer
- The existing `LocalStorageClient` code cannot be reused
- sqlite-vec is only needed for topic similarity (~200 vectors) -- massive overkill
- Adds 2 new dependencies (aiosqlite + sqlite-vec C extension)
- For a one-time bulk ingest followed by export, ACID transactions are nice-to-have, not essential

**Verdict:** Architecturally clean but over-engineered for the actual use case. The Windows extension loading risk is a real problem.

### 4.2 JSONL (Current Proposal)

**Advantages:**
- 80% of the code already exists in `tests/local_storage.py`
- Zero new dependencies
- Human-readable output for debugging
- Trivial to import to any target system (read line by line)
- No Windows compatibility concerns
- Matches the design document already written (`02-local-storage-design.md`)

**Disadvantages:**
- No crash recovery on individual records (last line may be truncated)
- Resume requires loading entire JSONL into memory on restart
- Topic similarity search must be implemented manually (cosine similarity in Python)
- No referential integrity checks
- File sizes are larger due to JSON encoding of floats

**Mitigations for disadvantages:**
- Crash recovery: manifest.json written last validates completeness; truncated last line is detectable and skippable on re-read
- Resume: 6,289 documents fit easily in memory as a hash set (~1 MB)
- Topic similarity: Only ~200 topics, brute-force cosine similarity in pure Python takes <1ms
- Referential integrity: Single-process sequential pipeline means references are always valid
- File size: 1.2 GB is acceptable for a one-time local ingest artifact

**Verdict:** Good enough for the actual requirements. The simplicity and existing code advantage is significant.

### 4.3 Hybrid (SQLite metadata + separate embeddings)

**Advantages:**
- SQLite provides crash recovery and indexed lookups for structured data
- Embeddings stored efficiently in binary format (numpy `.npy` or raw float32)
- Best of both: fast lookups + compact embedding storage

**Disadvantages:**
- Most complex to implement (two storage systems to coordinate)
- Still requires aiosqlite dependency
- Coordination between SQLite and embedding files adds failure modes
- Import to production needs to read from two sources
- No clear advantage over JSONL for the actual data volumes

**Verdict:** Over-engineered. The complexity does not pay for itself at this scale.

## 5. Key Insight: Topic Similarity Does Not Require sqlite-vec

The only vector search needed during local ingest is `find_similar_topics()`, which operates on ~200 topic vectors. This is trivially solvable in pure Python:

```python
# ~200 topics, each with a 1536-dim embedding
# Brute-force cosine similarity takes <1ms
import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

The existing design document (section 4.7) already proposes exactly this approach. For ~200 vectors, even without numpy, this runs in well under 1ms per query.

Full chunk vector search (~50K vectors) is only needed in production for RAG queries, not during local ingest.

## 6. Recommendation: JSONL (Extend Current Proposal)

**Use the JSONL approach as designed in `02-local-storage-design.md`.** The rationale:

1. **The code is 80% written.** `LocalStorageClient` needs ~8 more methods, not a ground-up rewrite.

2. **The data volumes are manageable.** 6,289 emails producing ~50K chunks fit comfortably in memory for runtime lookups, and ~1.2 GB on disk is acceptable.

3. **No new dependencies.** Pure Python, no C extensions, no Windows compatibility risk.

4. **Topic similarity is trivial without sqlite-vec.** ~200 topics with brute-force cosine similarity is sub-millisecond.

5. **Crash recovery is adequate.** JSONL append-only files preserve all complete records. The manifest validates completeness. The pipeline already supports resume via hash-based deduplication.

6. **Import to production is simpler.** Read JSONL line by line, insert to PostgreSQL. No SQLite-to-PostgreSQL migration tooling needed.

7. **The local storage is temporary.** It exists only between local ingest and production import. It does not serve queries. Engineering for durability beyond "resumable single-process batch job" is wasted effort.

### When to reconsider SQLite

Revisit this decision if:
- Local storage needs to support concurrent access (multiple ingest workers)
- Local storage needs to serve queries directly (local RAG without cloud)
- Dataset grows beyond ~500K chunks (memory pressure for in-memory lookups)
- Topic count grows beyond ~10,000 (brute-force similarity becomes slow)

None of these conditions apply to the current ~6,289 email / ~50K chunk dataset.

## 7. Gaps to Close in the JSONL Implementation

These are the methods that `LocalStorageClient` (in `tests/local_storage.py`) still needs, as identified in `02-local-storage-design.md` section 4.3:

| Missing Method | Implementation Approach |
|---|---|
| `get_document_by_hash(file_hash)` | Add `_documents_by_hash` dict, index on insert |
| `get_document_by_id(doc_id: UUID)` | Direct lookup in existing `_documents` dict |
| `delete_document_for_reprocess(doc_id)` | Remove from dicts, append tombstone to JSONL |
| `insert_topic(topic)` | Add `_topics` dict + append to `topics.jsonl` |
| `get_topic_by_name(name)` | Lookup in `_topics_by_name` dict |
| `get_topic_by_id(topic_id)` | Lookup in `_topics` dict |
| `find_similar_topics(embedding, threshold, limit)` | Brute-force cosine similarity over `_topics` |
| `increment_topic_counts(topic_ids, chunk_delta, document_delta)` | Update in-memory, persist on flush |
| `update_chunks_topic_ids(document_id, topic_ids)` | Update in-memory chunk metadata |
| `update_chunks_topics_checked(document_id)` | Update in-memory chunk metadata |

Additionally needed:
- `LocalProgressTracker` wrapping `processing_log.jsonl`
- Include embeddings in `_chunk_to_dict()` (currently omitted)
- `_topic_to_dict()` serialization including embedding
- Startup loading of existing JSONL files for resume capability

## 8. Summary

| Option | Recommendation | Reason |
|---|---|---|
| SQLite + sqlite-vec | No | Over-engineered, Windows risk, no code reuse |
| JSONL (extend current) | **Yes** | 80% done, zero dependencies, adequate for scale |
| Hybrid | No | Unnecessary complexity for the data volumes |

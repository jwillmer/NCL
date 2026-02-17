"""Local storage mocks for testing ingest output without Supabase.

Provides mock implementations of SupabaseClient and ArchiveStorage that
write to local files instead of the database/bucket. Useful for:
- Debugging ingest behavior locally
- Capturing test output for inspection
- Running integration tests without external dependencies
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass
class LocalStorageClient:
    """Mock SupabaseClient that writes to local JSONL files.

    Provides the same interface as SupabaseClient but stores data
    in local JSONL files for easy inspection.
    """

    output_dir: Path
    _documents: Dict[UUID, Any] = field(default_factory=dict)
    _chunks: Dict[UUID, Any] = field(default_factory=dict)
    _events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, filename: str, data: Dict[str, Any]) -> None:
        """Append a JSON object as a line to a JSONL file."""
        file_path = self.output_dir / filename
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _doc_to_dict(self, doc: Any) -> Dict[str, Any]:
        """Convert Document model to dictionary."""
        return {
            "id": str(doc.id),
            "source_id": doc.source_id,
            "doc_id": doc.doc_id,
            "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else doc.document_type,
            "file_path": doc.file_path,
            "file_name": doc.file_name,
            "file_hash": doc.file_hash,
            "parent_id": str(doc.parent_id) if doc.parent_id else None,
            "root_id": str(doc.root_id) if doc.root_id else None,
            "depth": doc.depth,
            "source_title": doc.source_title,
            "archive_browse_uri": doc.archive_browse_uri,
            "archive_download_uri": doc.archive_download_uri,
            "status": doc.status.value if hasattr(doc.status, "value") else doc.status,
            "ingest_version": doc.ingest_version,
        }

    def _chunk_to_dict(self, chunk: Any) -> Dict[str, Any]:
        """Convert Chunk model to dictionary."""
        return {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "context_summary": chunk.context_summary,
            "embedding_text": chunk.embedding_text,
            "section_path": chunk.section_path,
            "section_title": chunk.section_title,
            "source_title": chunk.source_title,
            "source_id": chunk.source_id,
            "line_from": chunk.line_from,
            "line_to": chunk.line_to,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "archive_browse_uri": chunk.archive_browse_uri,
            "archive_download_uri": chunk.archive_download_uri,
            "metadata": chunk.metadata,
            # Embedding is omitted (too large for JSONL)
        }

    async def insert_document(self, doc: Any) -> Any:
        """Insert a document record to local storage."""
        self._documents[doc.id] = doc
        self._append_jsonl("documents.jsonl", self._doc_to_dict(doc))
        return doc

    async def insert_chunks(self, chunks: List[Any]) -> List[Any]:
        """Insert chunks to local storage."""
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._append_jsonl("chunks.jsonl", self._chunk_to_dict(chunk))
        return chunks

    async def update_document_status(
        self,
        doc_id: UUID,
        status: Any,
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status in local storage."""
        event = {
            "type": "status_update",
            "doc_id": str(doc_id),
            "status": status.value if hasattr(status, "value") else status,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._append_jsonl("status_updates.jsonl", event)

    async def update_document_archive_uris(
        self,
        doc_id: UUID,
        archive_browse_uri: str,
        archive_download_uri: str | None = None,
    ) -> None:
        """Update document archive URIs."""
        event = {
            "type": "archive_uri_update",
            "doc_id": str(doc_id),
            "archive_browse_uri": archive_browse_uri,
            "archive_download_uri": archive_download_uri,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._append_jsonl("archive_updates.jsonl", event)

    def log_ingest_event(
        self,
        document_id: UUID,
        event_type: str,
        severity: str = "warning",
        message: Optional[str] = None,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        source_eml_path: Optional[str] = None,
    ) -> None:
        """Log an ingest event to local storage."""
        event = {
            "document_id": str(document_id),
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "file_path": file_path,
            "file_name": file_name,
            "source_eml_path": source_eml_path,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._events.append(event)
        self._append_jsonl("ingest_events.jsonl", event)

    async def get_document_by_doc_id(self, doc_id: str) -> Optional[Any]:
        """Get document by doc_id from local storage."""
        for doc in self._documents.values():
            if doc.doc_id == doc_id:
                return doc
        return None

    async def get_document_by_source_id(self, source_id: str) -> Optional[Any]:
        """Get document by source_id from local storage."""
        for doc in self._documents.values():
            if doc.source_id == source_id:
                return doc
        return None

    async def get_chunks_by_document(self, doc_id: UUID) -> List[Any]:
        """Get chunks for a document from local storage."""
        return [c for c in self._chunks.values() if c.document_id == doc_id]

    async def close(self) -> None:
        """No-op close for local storage."""
        pass


@dataclass
class LocalBucketStorage:
    """Mock ArchiveStorage that writes to local folders.

    Provides the same interface as ArchiveStorage but stores files
    in a local directory structure.
    """

    bucket_dir: Path

    def __post_init__(self):
        """Ensure bucket directory exists."""
        self.bucket_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(
        self,
        path: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Upload file to local bucket directory."""
        file_path = self.bucket_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return path

    def upload_text(
        self,
        path: str,
        text: str,
        content_type: str = "text/markdown; charset=utf-8",
    ) -> str:
        """Upload text file to local bucket directory."""
        return self.upload_file(path, text.encode("utf-8"), content_type)

    def download_file(self, path: str) -> bytes:
        """Download file from local bucket directory."""
        file_path = self.bucket_dir / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_bytes()

    def download_text(self, path: str, encoding: str = "utf-8") -> str:
        """Download text file from local bucket directory."""
        return self.download_file(path).decode(encoding)

    def file_exists(self, path: str) -> bool:
        """Check if file exists in local bucket directory."""
        return (self.bucket_dir / path).exists()

    def delete_folder(self, doc_id: str, preserve_md: bool = False) -> None:
        """Delete folder from local bucket directory."""
        folder_path = self.bucket_dir / doc_id
        if folder_path.exists():
            import shutil
            if preserve_md:
                # Delete non-md files only
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file() and not file_path.suffix == ".md":
                        file_path.unlink()
            else:
                shutil.rmtree(folder_path)

    def list_files(self, doc_id: str) -> List[Dict[str, Any]]:
        """List files in a folder."""
        folder_path = self.bucket_dir / doc_id
        if not folder_path.exists():
            return []

        files = []
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.bucket_dir)),
                    "size": file_path.stat().st_size,
                })
        return files


@dataclass
class LocalIngestOutput:
    """Coordinates local storage for test capture.

    Provides a unified interface for capturing all ingest output
    to local files for inspection and debugging.
    """

    db: LocalStorageClient
    bucket: LocalBucketStorage

    @classmethod
    def create(cls, output_root: Path) -> "LocalIngestOutput":
        """Create a LocalIngestOutput with default directory structure.

        Args:
            output_root: Root directory for all output.

        Returns:
            Configured LocalIngestOutput instance.
        """
        return cls(
            db=LocalStorageClient(output_root / "database"),
            bucket=LocalBucketStorage(output_root / "bucket"),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of captured output."""
        return {
            "documents": len(self.db._documents),
            "chunks": len(self.db._chunks),
            "events": len(self.db._events),
            "bucket_files": len(list(self.bucket.bucket_dir.rglob("*"))),
        }

    def read_documents_jsonl(self) -> List[Dict[str, Any]]:
        """Read all documents from JSONL file."""
        file_path = self.db.output_dir / "documents.jsonl"
        if not file_path.exists():
            return []
        documents = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
        return documents

    def read_chunks_jsonl(self) -> List[Dict[str, Any]]:
        """Read all chunks from JSONL file."""
        file_path = self.db.output_dir / "chunks.jsonl"
        if not file_path.exists():
            return []
        chunks = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def read_events_jsonl(self) -> List[Dict[str, Any]]:
        """Read all ingest events from JSONL file."""
        file_path = self.db.output_dir / "ingest_events.jsonl"
        if not file_path.exists():
            return []
        events = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

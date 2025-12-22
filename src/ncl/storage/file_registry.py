"""File registry for tracking processed and unsupported files."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import UUID

from ..models.document import ProcessingStatus
from .supabase_client import SupabaseClient


class FileRegistry:
    """Registry for tracking all files encountered during processing.

    Provides fast lookup to determine if files need processing, and
    tracks unsupported files for visibility.

    Features:
    - Hash-based deduplication
    - Support for subdirectories
    - Unsupported file logging
    - Processing status tracking
    """

    def __init__(self, db_client: SupabaseClient):
        """Initialize the file registry.

        Args:
            db_client: Supabase client for database operations.
        """
        self.db = db_client

    # =========================================================================
    # File Registry Operations
    # =========================================================================

    async def needs_processing(self, file_path: Path, file_hash: Optional[str] = None) -> bool:
        """Check if a file needs to be processed.

        Args:
            file_path: Path to the file.
            file_hash: Pre-computed hash (optional, will compute if not provided).

        Returns:
            True if file should be processed, False if already completed.
        """
        if file_hash is None:
            file_hash = self.compute_file_hash(file_path)

        result = (
            self.db.client.table("file_registry")
            .select("status, file_hash")
            .eq("file_path", str(file_path))
            .limit(1)
            .execute()
        )

        if not result.data:
            return True  # New file

        record = result.data[0]

        # If hash changed, needs reprocessing
        if record["file_hash"] != file_hash:
            return True

        # Only skip if completed
        return record["status"] not in [
            ProcessingStatus.COMPLETED.value,
            ProcessingStatus.SKIPPED.value,
        ]

    async def register_file(
        self,
        file_path: Path,
        source_type: str,
        parent_file_id: Optional[UUID] = None,
        root_eml_id: Optional[UUID] = None,
        mime_type: Optional[str] = None,
        is_supported: bool = True,
    ) -> UUID:
        """Register a file in the registry.

        Args:
            file_path: Path to the file.
            source_type: Type of source ('eml', 'attachment', 'zip_extracted').
            parent_file_id: ID of parent file (for attachments/extracted).
            root_eml_id: ID of root EML file.
            mime_type: MIME type of the file.
            is_supported: Whether the file format is supported.

        Returns:
            UUID of the registry entry.
        """
        file_hash = self.compute_file_hash(file_path)
        file_size = file_path.stat().st_size

        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(str(file_path))

        data = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "file_name": file_path.name,
            "file_size_bytes": file_size,
            "source_type": source_type,
            "mime_type": mime_type,
            "is_supported": is_supported,
            "status": ProcessingStatus.PENDING.value,
            "first_seen_at": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        if parent_file_id:
            data["parent_file_id"] = str(parent_file_id)
        if root_eml_id:
            data["root_eml_id"] = str(root_eml_id)

        result = (
            self.db.client.table("file_registry")
            .upsert(data, on_conflict="file_path,file_hash")
            .execute()
        )

        return UUID(result.data[0]["id"])

    async def mark_processing(self, file_path: Path):
        """Mark a file as currently being processed.

        Args:
            file_path: Path to the file.
        """
        self.db.client.table("file_registry").update(
            {
                "status": ProcessingStatus.PROCESSING.value,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("file_path", str(file_path)).execute()

    async def mark_completed(self, file_path: Path, document_id: Optional[UUID] = None):
        """Mark a file as successfully processed.

        Args:
            file_path: Path to the file.
            document_id: ID of the created document.
        """
        data = {
            "status": ProcessingStatus.COMPLETED.value,
            "last_processed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        if document_id:
            data["document_id"] = str(document_id)

        self.db.client.table("file_registry").update(data).eq(
            "file_path", str(file_path)
        ).execute()

    async def mark_failed(self, file_path: Path, error: str):
        """Mark a file as failed with error message.

        Args:
            file_path: Path to the file.
            error: Error message describing the failure.
        """
        # Get current attempts
        result = (
            self.db.client.table("file_registry")
            .select("attempts")
            .eq("file_path", str(file_path))
            .limit(1)
            .execute()
        )

        attempts = (result.data[0].get("attempts", 0) if result.data else 0) + 1

        self.db.client.table("file_registry").update(
            {
                "status": ProcessingStatus.FAILED.value,
                "error_message": error[:1000],
                "attempts": attempts,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("file_path", str(file_path)).execute()

    async def get_pending_files(self, source_dir: Path) -> List[Path]:
        """Get list of EML files that haven't been processed yet.

        Supports subdirectories - scans the entire source directory tree.

        Args:
            source_dir: Root directory to scan for EML files.

        Returns:
            List of file paths that need processing.
        """
        # Get all EML files in source directory (including subdirectories)
        all_files = list(source_dir.glob("**/*.eml"))

        # Get already processed file hashes
        processed_hashes = await self._get_processed_hashes()

        # Filter out already processed files
        pending = []
        for file_path in all_files:
            file_hash = self.compute_file_hash(file_path)
            if file_hash not in processed_hashes:
                pending.append(file_path)

        return pending

    async def get_failed_files(self, max_attempts: int = 3) -> List[Path]:
        """Get files that failed but haven't exceeded retry limit.

        Args:
            max_attempts: Maximum number of retry attempts.

        Returns:
            List of file paths eligible for retry.
        """
        result = (
            self.db.client.table("file_registry")
            .select("file_path")
            .eq("status", ProcessingStatus.FAILED.value)
            .eq("source_type", "eml")
            .lt("attempts", max_attempts)
            .execute()
        )

        return [Path(row["file_path"]) for row in result.data]

    async def _get_processed_hashes(self) -> Set[str]:
        """Get set of file hashes that have been processed.

        Returns:
            Set of SHA-256 hashes.
        """
        result = (
            self.db.client.table("file_registry")
            .select("file_hash")
            .eq("source_type", "eml")
            .in_(
                "status",
                [ProcessingStatus.COMPLETED.value, ProcessingStatus.PROCESSING.value],
            )
            .execute()
        )

        return {row["file_hash"] for row in result.data}

    # =========================================================================
    # Unsupported Files Operations
    # =========================================================================

    async def log_unsupported_file(
        self,
        file_path: Path,
        reason: str,
        source_eml_path: Optional[str] = None,
        source_zip_path: Optional[str] = None,
        parent_document_id: Optional[UUID] = None,
    ):
        """Log an unsupported file for visibility.

        Args:
            file_path: Path to the unsupported file.
            reason: Why the file is unsupported.
            source_eml_path: Path to the source EML file.
            source_zip_path: Path to the source ZIP file (if extracted from ZIP).
            parent_document_id: ID of the parent document.
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_extension = file_path.suffix.lower() if file_path.suffix else None

        try:
            file_size = file_path.stat().st_size if file_path.exists() else None
            file_hash = self.compute_file_hash(file_path) if file_path.exists() else None
        except Exception:
            file_size = None
            file_hash = None

        data = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_hash": file_hash,
            "file_size_bytes": file_size,
            "mime_type": mime_type,
            "file_extension": file_extension,
            "reason": reason,
            "discovered_at": datetime.utcnow().isoformat(),
        }

        if source_eml_path:
            data["source_eml_path"] = source_eml_path
        if source_zip_path:
            data["source_zip_path"] = source_zip_path
        if parent_document_id:
            data["parent_document_id"] = str(parent_document_id)

        self.db.client.table("unsupported_files").upsert(
            data, on_conflict="file_path"
        ).execute()

    async def get_unsupported_files_stats(self) -> Dict[str, int]:
        """Get statistics about unsupported files.

        Returns:
            Dictionary with counts by reason and mime_type.
        """
        result = (
            self.db.client.table("unsupported_files")
            .select("reason, mime_type")
            .execute()
        )

        stats = {
            "total": len(result.data),
            "by_reason": {},
            "by_mime_type": {},
        }

        for row in result.data:
            reason = row.get("reason", "unknown")
            mime_type = row.get("mime_type", "unknown")

            stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
            stats["by_mime_type"][mime_type] = stats["by_mime_type"].get(mime_type, 0) + 1

        return stats

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_registry_stats(self) -> Dict[str, int]:
        """Get overall file registry statistics.

        Returns:
            Dictionary with counts by status and source type.
        """
        result = (
            self.db.client.table("file_registry")
            .select("status, source_type, is_supported")
            .execute()
        )

        stats = {
            "total": len(result.data),
            "by_status": {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "skipped": 0,
            },
            "by_source_type": {
                "eml": 0,
                "attachment": 0,
                "zip_extracted": 0,
            },
            "supported": 0,
            "unsupported": 0,
        }

        for row in result.data:
            status = row.get("status", "pending")
            source_type = row.get("source_type", "unknown")
            is_supported = row.get("is_supported", True)

            if status in stats["by_status"]:
                stats["by_status"][status] += 1
            if source_type in stats["by_source_type"]:
                stats["by_source_type"][source_type] += 1

            if is_supported:
                stats["supported"] += 1
            else:
                stats["unsupported"] += 1

        return stats

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal hash string.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def reset_stale_processing(self, max_age_minutes: int = 60):
        """Reset files stuck in 'processing' state for too long.

        Args:
            max_age_minutes: Maximum age in minutes before considering stale.
        """
        from datetime import timezone

        cutoff = datetime.utcnow()

        result = (
            self.db.client.table("file_registry")
            .select("file_path, updated_at")
            .eq("status", ProcessingStatus.PROCESSING.value)
            .execute()
        )

        for row in result.data:
            updated_at = row.get("updated_at")
            if updated_at:
                updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age_minutes = (
                    cutoff.replace(tzinfo=timezone.utc) - updated
                ).total_seconds() / 60

                if age_minutes > max_age_minutes:
                    self.db.client.table("file_registry").update(
                        {
                            "status": ProcessingStatus.PENDING.value,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    ).eq("file_path", row["file_path"]).execute()

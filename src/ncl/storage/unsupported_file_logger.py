"""Logger for tracking unsupported files during processing."""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

from .supabase_client import SupabaseClient


class UnsupportedFileLogger:
    """Logger for tracking unsupported files for visibility.

    Records files that cannot be processed due to unsupported formats,
    allowing users to see what was skipped and why.
    """

    def __init__(self, db_client: SupabaseClient):
        """Initialize the logger.

        Args:
            db_client: Supabase client for database operations.
        """
        self.db = db_client

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
        except Exception:
            file_size = None

        data = {
            "file_path": str(file_path),
            "file_name": file_path.name,
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

"""Progress tracker for resumable email processing."""

from __future__ import annotations

import hashlib
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Set

import anyio

from ..config import get_settings
from ..models.document import ProcessingStatus
from .supabase_client import SupabaseClient


class ProgressTracker:
    """Track processing progress for resumable operations.

    Maintains a log of processed files by hash to enable:
    - Resumable processing after interruptions
    - Retry of failed files
    - Deduplication by content
    """

    def __init__(self, db_client: SupabaseClient):
        """Initialize the progress tracker.

        Args:
            db_client: Supabase client for database operations.
        """
        self.db = db_client

    async def get_pending_files(self, source_dir: Path) -> List[Path]:
        """Get list of files that haven't been processed yet.

        Args:
            source_dir: Directory to scan for EML files.

        Returns:
            List of file paths that need processing.
        """
        # Get all EML files in source directory
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

    async def mark_started(self, file_path: Path, file_hash: str):
        """Mark a file as started processing.

        Args:
            file_path: Path to the file.
            file_hash: SHA-256 hash of file content.
        """
        await anyio.to_thread.run_sync(
            partial(self._mark_started_sync, file_path, file_hash)
        )

    def _mark_started_sync(self, file_path: Path, file_hash: str):
        """Synchronous implementation of mark_started."""
        self.db.client.table("processing_log").upsert(
            {
                "file_path": str(file_path),
                "file_hash": file_hash,
                "status": ProcessingStatus.PROCESSING.value,
                "started_at": datetime.utcnow().isoformat(),
                "attempts": 1,
                "updated_at": datetime.utcnow().isoformat(),
            },
            on_conflict="file_path",
        ).execute()

    async def mark_completed(self, file_path: Path):
        """Mark a file as successfully processed.

        Args:
            file_path: Path to the file.
        """
        await anyio.to_thread.run_sync(partial(self._mark_completed_sync, file_path))

    def _mark_completed_sync(self, file_path: Path):
        """Synchronous implementation of mark_completed."""
        self.db.client.table("processing_log").update(
            {
                "status": ProcessingStatus.COMPLETED.value,
                "completed_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("file_path", str(file_path)).execute()

    async def mark_failed(self, file_path: Path, error: str):
        """Mark a file as failed with error message.

        Args:
            file_path: Path to the file.
            error: Error message describing the failure.
        """
        await anyio.to_thread.run_sync(partial(self._mark_failed_sync, file_path, error))

    def _mark_failed_sync(self, file_path: Path, error: str):
        """Synchronous implementation of mark_failed."""
        settings = get_settings()

        # Get current attempts
        result = (
            self.db.client.table("processing_log")
            .select("attempts")
            .eq("file_path", str(file_path))
            .limit(1)
            .execute()
        )

        attempts = (result.data[0].get("attempts", 0) if result.data else 0) + 1

        self.db.client.table("processing_log").update(
            {
                "status": ProcessingStatus.FAILED.value,
                "last_error": error[: settings.error_message_max_length],
                "attempts": attempts,
                "updated_at": datetime.utcnow().isoformat(),
            }
        ).eq("file_path", str(file_path)).execute()

    async def get_failed_files(self, max_attempts: int = 3) -> List[Path]:
        """Get files that failed but haven't exceeded retry limit.

        Args:
            max_attempts: Maximum number of retry attempts.

        Returns:
            List of file paths eligible for retry.
        """
        return await anyio.to_thread.run_sync(
            partial(self._get_failed_files_sync, max_attempts)
        )

    def _get_failed_files_sync(self, max_attempts: int) -> List[Path]:
        """Synchronous implementation of get_failed_files."""
        result = (
            self.db.client.table("processing_log")
            .select("file_path")
            .eq("status", ProcessingStatus.FAILED.value)
            .lt("attempts", max_attempts)
            .execute()
        )

        return [Path(row["file_path"]) for row in result.data]

    async def get_processing_stats(self) -> Dict[str, int]:
        """Get overall processing statistics.

        Returns:
            Dictionary with counts by status.
        """
        return await anyio.to_thread.run_sync(self._get_processing_stats_sync)

    def _get_processing_stats_sync(self) -> Dict[str, int]:
        """Synchronous implementation of get_processing_stats."""
        result = self.db.client.table("processing_log").select("status").execute()

        stats = {
            "total": len(result.data),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }

        for row in result.data:
            status = row["status"]
            if status in stats:
                stats[status] += 1

        return stats

    async def _get_processed_hashes(self) -> Set[str]:
        """Get set of file hashes that have been processed.

        Returns:
            Set of SHA-256 hashes.
        """
        return await anyio.to_thread.run_sync(self._get_processed_hashes_sync)

    def _get_processed_hashes_sync(self) -> Set[str]:
        """Synchronous implementation of _get_processed_hashes."""
        result = (
            self.db.client.table("processing_log")
            .select("file_hash")
            .in_(
                "status",
                [ProcessingStatus.COMPLETED.value, ProcessingStatus.PROCESSING.value],
            )
            .execute()
        )

        return {row["file_hash"] for row in result.data}

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
        await anyio.to_thread.run_sync(
            partial(self._reset_stale_processing_sync, max_age_minutes)
        )

    def _reset_stale_processing_sync(self, max_age_minutes: int):
        """Synchronous implementation of reset_stale_processing."""
        from datetime import timezone

        cutoff = datetime.utcnow()

        result = (
            self.db.client.table("processing_log")
            .select("file_path, started_at")
            .eq("status", ProcessingStatus.PROCESSING.value)
            .execute()
        )

        for row in result.data:
            started_at = row.get("started_at")
            if started_at:
                # Parse and check age
                started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                age_minutes = (
                    cutoff.replace(tzinfo=timezone.utc) - started
                ).total_seconds() / 60

                if age_minutes > max_age_minutes:
                    # Reset to pending
                    self.db.client.table("processing_log").update(
                        {
                            "status": ProcessingStatus.PENDING.value,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                    ).eq("file_path", row["file_path"]).execute()

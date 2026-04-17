"""Local progress tracker using JSONL files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LocalProgressTracker:
    """Track ingest progress using processing_log.jsonl."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._log_file = output_dir / "processing_log.jsonl"
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """Load existing progress entries from JSONL file."""
        if self._log_file.exists():
            with open(self._log_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._entries[entry["file_path"]] = entry

    def _save_entry(self, entry: Dict[str, Any]):
        """Save a progress entry to memory and append to JSONL file."""
        self._entries[entry["file_path"]] = entry
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def compact(self):
        """Rewrite processing_log.jsonl with one entry per file (removes duplicates)."""
        with open(self._log_file, "w", encoding="utf-8") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    async def mark_started(self, file_path: Path, file_hash: str):
        """Mark a file as started processing."""
        self._save_entry({
            "file_path": str(file_path),
            "file_hash": file_hash,
            "status": "PROCESSING",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
            "attempts": self._entries.get(str(file_path), {}).get("attempts", 0) + 1,
        })

    async def mark_completed(self, file_path: Path):
        """Mark a file as successfully processed."""
        entry = self._entries.get(str(file_path), {})
        now = datetime.now(timezone.utc)
        duration = None
        if entry.get("started_at"):
            try:
                started = datetime.fromisoformat(entry["started_at"])
                duration = round((now - started).total_seconds(), 1)
            except (ValueError, TypeError):
                pass
        entry.update({
            "file_path": str(file_path),
            "status": "COMPLETED",
            "completed_at": now.isoformat(),
            "duration_seconds": duration,
            "error": None,
        })
        self._save_entry(entry)

    async def mark_failed(self, file_path: Path, error: str):
        """Mark a file as failed with error message."""
        entry = self._entries.get(str(file_path), {})
        entry.update({
            "file_path": str(file_path),
            "status": "FAILED",
            "error": error[:1000],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save_entry(entry)

    async def get_pending_files(self, source_dir: Path) -> List[Path]:
        """Get list of files that haven't been processed yet."""
        completed_hashes = {
            e["file_hash"] for e in self._entries.values()
            if e.get("status") == "COMPLETED"
        }
        pending = []
        for eml in sorted(source_dir.rglob("*.eml")):
            file_hash = self.compute_file_hash(eml)
            if file_hash not in completed_hashes:
                pending.append(eml)
        return pending

    async def get_failed_files(self) -> List[Path]:
        """Return every file currently in FAILED state.

        ``--retry-failed`` is an explicit, user-driven command; it always
        retries whatever is FAILED. The ``attempts`` counter is kept on
        entries for visibility but does not gate retry eligibility.
        """
        return [
            Path(e["file_path"]) for e in self._entries.values()
            if e.get("status") == "FAILED"
        ]

    async def get_processing_stats(self) -> Dict[str, int]:
        """Get overall processing statistics."""
        stats = {"total": 0, "pending": 0, "processing": 0, "completed": 0, "failed": 0}
        for entry in self._entries.values():
            status = entry.get("status", "PENDING").upper()
            stats["total"] += 1
            key = status.lower()
            if key in stats:
                stats[key] += 1
        return stats

    async def reset_stale_processing(self, max_age_minutes: int = 60) -> int:
        """Reset files stuck in 'processing' state for too long."""
        count = 0
        now = datetime.now(timezone.utc)
        for entry in list(self._entries.values()):
            if entry.get("status") == "PROCESSING":
                started = entry.get("started_at")
                if started:
                    started_dt = datetime.fromisoformat(started)
                    if (now - started_dt) > timedelta(minutes=max_age_minutes):
                        entry["status"] = "FAILED"
                        entry["error"] = f"Stale processing (>{max_age_minutes}min)"
                        self._save_entry(entry)
                        count += 1
        return count

    async def get_outdated_files(self, source_dir: Path, target_version: int) -> List[Path]:
        """Get files processed with an older ingest version (not applicable for local-only)."""
        return []

"""Failure report generator for ingest runs."""

from __future__ import annotations

import csv
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings
from .supabase_client import SupabaseClient


@dataclass
class FailureRecord:
    """Unified failure record for reports."""

    type: str  # "eml_file", "attachment_skipped", "attachment_failed"
    file_path: str
    file_name: str
    parent_eml: Optional[str] = None
    mime_type: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    attempts: Optional[int] = None
    timestamp: Optional[str] = None


@dataclass
class FailureReport:
    """Complete failure report for an ingest run."""

    run_timestamp: str
    total_failures: int
    eml_failures: int
    attachment_failures: int
    failures: List[FailureRecord]

    # Run context
    source_dir: Optional[str] = None
    files_processed: int = 0
    files_succeeded: int = 0


class IngestReportWriter:
    """Writes ingest report continuously during processing.

    Creates the report file at initialization and updates it after each failure,
    ensuring error details are preserved even if the process crashes.
    """

    def __init__(self, source_dir: Optional[str] = None):
        self.settings = get_settings()
        self.source_dir = source_dir
        self.run_timestamp = datetime.now(timezone.utc)
        self.failures: List[FailureRecord] = []
        self.files_processed = 0
        self.files_succeeded = 0

        # Validate and create reports directory
        reports_dir = self.settings.failure_reports_dir.resolve()
        if ".." in str(self.settings.failure_reports_dir):
            raise ValueError(
                f"Invalid reports directory path: {self.settings.failure_reports_dir}"
            )
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Create file paths
        timestamp_str = self.run_timestamp.strftime("%Y%m%d_%H%M%S")
        self._json_path = reports_dir / f"ingest_{timestamp_str}.json"
        self._csv_path = reports_dir / f"ingest_{timestamp_str}.csv"

        # Write initial empty report
        self._write_report()

    def add_eml_failure(self, file_path: str, error: str) -> None:
        """Add an EML file failure and write to disk immediately."""
        self.failures.append(
            FailureRecord(
                type="eml_file",
                file_path=file_path,
                file_name=Path(file_path).name,
                error=error,
                attempts=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        self._write_report()

    def update_stats(self, processed: int, succeeded: int) -> None:
        """Update processing statistics and write to disk."""
        self.files_processed = processed
        self.files_succeeded = succeeded
        self._write_report()

    def _write_report(self) -> None:
        """Write report to JSON and CSV files (overwrites each time)."""
        eml_failures = sum(1 for f in self.failures if f.type == "eml_file")
        attachment_failures = len(self.failures) - eml_failures

        # Build report data
        data = {
            "run_timestamp": self.run_timestamp.isoformat(),
            "summary": {
                "total_failures": len(self.failures),
                "eml_failures": eml_failures,
                "attachment_failures": attachment_failures,
                "files_processed": self.files_processed,
                "files_succeeded": self.files_succeeded,
            },
            "source_dir": self.source_dir,
            "failures": [asdict(f) for f in self.failures],
        }

        # Write JSON
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Set restrictive file permissions
        try:
            os.chmod(self._json_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Ignore permission errors on Windows

        # Write CSV
        fieldnames = [
            "type",
            "file_name",
            "parent_eml",
            "mime_type",
            "reason",
            "error",
            "attempts",
            "file_path",
            "timestamp",
        ]
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for failure in self.failures:
                writer.writerow(asdict(failure))

        try:
            os.chmod(self._csv_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def get_path(self) -> Path:
        """Return the JSON report path."""
        return self._json_path

    def cleanup_old_reports(self) -> None:
        """Remove old reports beyond the configured keep count."""
        keep_count = self.settings.failure_reports_keep_count
        reports_dir = self._json_path.parent

        # Get all report files sorted by modification time
        json_files = sorted(
            reports_dir.glob("ingest_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        csv_files = sorted(
            reports_dir.glob("ingest_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove files beyond keep count
        for old_file in json_files[keep_count:]:
            old_file.unlink()
        for old_file in csv_files[keep_count:]:
            old_file.unlink()


class FailureReportGenerator:
    """Generates failure reports from multiple data sources."""

    def __init__(self, db_client: SupabaseClient):
        self.db = db_client
        self.settings = get_settings()

    async def collect_eml_failures(self) -> List[FailureRecord]:
        """Collect EML file failures from processing_log."""
        result = (
            self.db.client.table("processing_log")
            .select("file_path, file_hash, last_error, attempts, updated_at")
            .eq("status", "failed")
            .execute()
        )

        records = []
        for row in result.data:
            records.append(
                FailureRecord(
                    type="eml_file",
                    file_path=row["file_path"],
                    file_name=Path(row["file_path"]).name,
                    error=row.get("last_error"),
                    attempts=row.get("attempts", 1),
                    timestamp=row.get("updated_at"),
                )
            )
        return records

    async def collect_attachment_failures(self) -> List[FailureRecord]:
        """Collect attachment failures from unsupported_files."""
        result = (
            self.db.client.table("unsupported_files")
            .select(
                "file_path, file_name, source_eml_path, mime_type, reason, discovered_at"
            )
            .execute()
        )

        records = []
        for row in result.data:
            # Classify as skipped vs failed based on reason
            reason = row.get("reason", "")
            failure_type = (
                "attachment_failed"
                if reason in ("extraction_failed", "corrupted")
                else "attachment_skipped"
            )

            records.append(
                FailureRecord(
                    type=failure_type,
                    file_path=row["file_path"],
                    file_name=row["file_name"],
                    parent_eml=(
                        Path(row["source_eml_path"]).name
                        if row.get("source_eml_path")
                        else None
                    ),
                    mime_type=row.get("mime_type"),
                    reason=reason,
                    timestamp=row.get("discovered_at"),
                )
            )
        return records

    def add_in_memory_issues(
        self, records: List[FailureRecord], issues: List[Dict[str, str]]
    ) -> List[FailureRecord]:
        """Add in-memory processing issues to records."""
        for issue in issues:
            records.append(
                FailureRecord(
                    type="attachment_failed",
                    file_path=issue.get("attachment", ""),
                    file_name=issue.get("attachment", ""),
                    parent_eml=issue.get("email"),
                    error=issue.get("error"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        return records

    async def generate_report(
        self,
        in_memory_issues: Optional[List[Dict[str, str]]] = None,
        source_dir: Optional[str] = None,
        files_processed: int = 0,
        files_succeeded: int = 0,
    ) -> FailureReport:
        """Generate complete failure report."""
        eml_failures = await self.collect_eml_failures()
        attachment_failures = await self.collect_attachment_failures()

        # Add in-memory issues if provided
        if in_memory_issues:
            attachment_failures = self.add_in_memory_issues(
                attachment_failures, in_memory_issues
            )

        all_failures = eml_failures + attachment_failures

        return FailureReport(
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            total_failures=len(all_failures),
            eml_failures=len(eml_failures),
            attachment_failures=len(attachment_failures),
            failures=all_failures,
            source_dir=source_dir,
            files_processed=files_processed,
            files_succeeded=files_succeeded,
        )

    def _validate_reports_dir(self, reports_dir: Path) -> Path:
        """Validate reports directory is safe (no path traversal)."""
        # Resolve to absolute path
        resolved = reports_dir.resolve()

        # Ensure it's within a reasonable location (not system directories)
        # Allow any path that doesn't contain '..' after resolution
        if ".." in str(reports_dir):
            raise ValueError(f"Invalid reports directory path: {reports_dir}")

        return resolved

    def export_json(self, report: FailureReport, output_path: Path) -> Path:
        """Export report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "run_timestamp": report.run_timestamp,
            "summary": {
                "total_failures": report.total_failures,
                "eml_failures": report.eml_failures,
                "attachment_failures": report.attachment_failures,
                "files_processed": report.files_processed,
                "files_succeeded": report.files_succeeded,
            },
            "source_dir": report.source_dir,
            "failures": [asdict(f) for f in report.failures],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Set restrictive file permissions (owner read/write only)
        try:
            os.chmod(output_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Ignore permission errors on Windows

        return output_path

    def export_csv(self, report: FailureReport, output_path: Path) -> Path:
        """Export report to CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "type",
            "file_name",
            "parent_eml",
            "mime_type",
            "reason",
            "error",
            "attempts",
            "file_path",
            "timestamp",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for failure in report.failures:
                writer.writerow(asdict(failure))

        # Set restrictive file permissions (owner read/write only)
        try:
            os.chmod(output_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Ignore permission errors on Windows

        return output_path

    def export_report(
        self,
        report: FailureReport,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Path]:
        """Export report to both JSON and CSV formats."""
        timestamp = timestamp or datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        reports_dir = self._validate_reports_dir(self.settings.failure_reports_dir)

        json_path = reports_dir / f"ingest_{timestamp_str}.json"
        csv_path = reports_dir / f"ingest_{timestamp_str}.csv"

        self.export_json(report, json_path)
        self.export_csv(report, csv_path)

        # Cleanup old reports
        self._cleanup_old_reports(reports_dir)

        return {"json": json_path, "csv": csv_path}

    def _cleanup_old_reports(self, reports_dir: Path) -> None:
        """Remove old reports beyond keep count."""
        keep_count = self.settings.failure_reports_keep_count

        # Get all report files sorted by modification time
        json_files = sorted(
            reports_dir.glob("ingest_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        csv_files = sorted(
            reports_dir.glob("ingest_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove files beyond keep count
        for old_file in json_files[keep_count:]:
            old_file.unlink()
        for old_file in csv_files[keep_count:]:
            old_file.unlink()

    def list_reports(self) -> List[Dict[str, Any]]:
        """List available failure reports."""
        try:
            reports_dir = self._validate_reports_dir(self.settings.failure_reports_dir)
        except ValueError:
            return []

        if not reports_dir.exists():
            return []

        reports = []
        for json_file in sorted(reports_dir.glob("ingest_*.json"), reverse=True):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append(
                    {
                        "timestamp": data.get("run_timestamp"),
                        "total_failures": data.get("summary", {}).get(
                            "total_failures", 0
                        ),
                        "json_path": str(json_file),
                        "csv_path": str(json_file.with_suffix(".csv")),
                    }
                )
            except Exception:
                continue

        return reports

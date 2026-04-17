"""Supabase Storage client for archive file operations.

Provides upload, download, and deletion for the browsable archive.
Uses Supabase Storage API directly (no DB tracking needed).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import List

from storage3.utils import StorageException
from supabase import Client, create_client

from .._io import retry_with_backoff
from ..config import get_settings

logger = logging.getLogger(__name__)

# Retry policy for paginated list calls. Supabase Storage occasionally returns
# non-JSON bodies (gateway errors, transient 5xx) which surface as
# JSONDecodeError in storage3. Retry with exponential backoff.
_LIST_MAX_ATTEMPTS = 3
_LIST_BACKOFF_BASE = 1.0  # seconds; delays: 1s, 2s


@lru_cache(maxsize=1)
def _get_storage_client() -> Client:
    """Get cached Supabase client for storage operations."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


class ArchiveStorageError(Exception):
    """Base exception for archive storage operations."""

    pass


class ArchiveStorage:
    """Supabase Storage client for archive operations.

    Uses Storage API directly for all operations. The bucket is auto-created
    on first use if it doesn't exist.
    """

    # Track which buckets have been verified to exist (avoid repeated checks)
    _verified_buckets: set[str] = set()

    def __init__(self, bucket_name: str | None = None):
        """Initialize archive storage.

        Args:
            bucket_name: Optional bucket name override. Defaults to settings.archive_bucket.
        """
        settings = get_settings()
        self.client: Client = _get_storage_client()
        self.bucket_name = bucket_name or settings.archive_bucket
        self.bucket = self.client.storage.from_(self.bucket_name)
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist (cached per bucket name)."""
        if self.bucket_name in ArchiveStorage._verified_buckets:
            return

        try:
            self.client.storage.get_bucket(self.bucket_name)
        except StorageException:
            # Bucket doesn't exist, create it
            logger.info(f"Creating storage bucket: {self.bucket_name}")
            self.client.storage.create_bucket(self.bucket_name, options={"public": False})

        ArchiveStorage._verified_buckets.add(self.bucket_name)

    def upload_file(
        self,
        path: str,
        content: bytes,
        content_type: str,
    ) -> str:
        """Upload file to bucket.

        Args:
            path: Path in bucket (e.g., "abc123/email.eml").
            content: File content as bytes.
            content_type: MIME type (e.g., "application/pdf").

        Returns:
            The path in the bucket.

        Raises:
            ArchiveStorageError: If upload fails.
        """
        try:
            # Upload to bucket (upsert to handle re-ingestion)
            self.bucket.upload(
                path, content, {"content-type": content_type, "upsert": "true"}
            )
            logger.debug(f"Uploaded archive file: {path}")
            return path

        except StorageException as e:
            raise ArchiveStorageError(f"Failed to upload {path}: {e}") from e

    def upload_text(
        self,
        path: str,
        text: str,
        content_type: str = "text/markdown; charset=utf-8",
    ) -> str:
        """Upload text content as UTF-8 file.

        Args:
            path: Path in bucket (e.g., "abc123/email.eml.md").
            text: Text content.
            content_type: MIME type (default "text/markdown; charset=utf-8").

        Returns:
            The path in the bucket.
        """
        return self.upload_file(path, text.encode("utf-8"), content_type)

    def download_file(self, path: str) -> bytes:
        """Download file content from bucket.

        Args:
            path: Path in bucket.

        Returns:
            File content as bytes.

        Raises:
            ArchiveStorageError: If file not found or download fails.
        """
        try:
            return self.bucket.download(path)
        except StorageException as e:
            raise ArchiveStorageError(f"File not found: {path}") from e

    def download_text(self, path: str, encoding: str = "utf-8") -> str:
        """Download text file content from bucket.

        Args:
            path: Path in bucket.
            encoding: Text encoding (default UTF-8).

        Returns:
            File content as string.

        Raises:
            ArchiveStorageError: If file not found or download fails.
        """
        content = self.download_file(path)
        return content.decode(encoding)

    def file_exists(self, path: str) -> bool:
        """Check if file exists using Storage list API.

        Args:
            path: Path in bucket (e.g., "abc123/attachments/file.pdf").

        Returns:
            True if file exists in bucket.
        """
        from urllib.parse import unquote

        # Split path into folder and filename
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            folder, filename = parts
        else:
            folder, filename = "", parts[0]

        # Decode URL-encoded filename for comparison (Supabase stores decoded names)
        decoded_filename = unquote(filename)

        try:
            files = self.list_folder(folder)
            return any(f["name"] == decoded_filename for f in files)
        except (StorageException, ArchiveStorageError):
            return False

    def delete_folder(self, doc_id: str, preserve_md: bool = False) -> None:
        """Delete all files for a doc_id using Storage list + remove.

        Args:
            doc_id: Document ID (folder name) whose files should be deleted.
            preserve_md: If True, preserve .md files (cached parsed content).

        Raises:
            ArchiveStorageError: If deletion fails.
        """
        try:
            folder = doc_id
            paths: List[str] = []

            # List files in root folder (list_folder already filters subfolder
            # placeholders when files_only=True, the default).
            try:
                files = self.list_folder(folder)
                for f in files:
                    if preserve_md and f["name"].endswith(".md"):
                        continue
                    paths.append(f"{folder}/{f['name']}")
            except (StorageException, ArchiveStorageError):
                pass  # Folder may not exist

            # List files in attachments subfolder
            try:
                att_files = self.list_folder(f"{folder}/attachments")
                for f in att_files:
                    if preserve_md and f["name"].endswith(".md"):
                        continue
                    paths.append(f"{folder}/attachments/{f['name']}")
            except (StorageException, ArchiveStorageError):
                pass  # Attachments folder may not exist

            if paths:
                logger.debug(f"Deleting {len(paths)} archive files for folder: {folder}")
                self.bucket.remove(paths)

        except StorageException as e:
            raise ArchiveStorageError(f"Failed to delete folder {doc_id}: {e}") from e

    def list_files(self, doc_id: str) -> List[dict]:
        """List all files for a doc_id using Storage API.

        Args:
            doc_id: Document ID (folder name) to list files for.

        Returns:
            List of file records from Storage API.
        """
        folder = doc_id
        result: List[dict] = []

        try:
            files = self.list_folder(folder)
            result.extend(files)

            # Also list attachments subfolder
            try:
                att_files = self.list_folder(f"{folder}/attachments")
                result.extend(att_files)
            except (StorageException, ArchiveStorageError):
                pass
        except (StorageException, ArchiveStorageError):
            pass

        return result

    def list_folder(
        self,
        folder: str,
        files_only: bool = True,
        max_attempts: int = _LIST_MAX_ATTEMPTS,
        backoff_base: float = _LIST_BACKOFF_BASE,
    ) -> List[dict]:
        """List entries in a folder, paginating past the 100-item default.

        Retries each page fetch with exponential backoff on transient errors
        (e.g. JSONDecodeError when the storage gateway returns a non-JSON body).
        Raises ArchiveStorageError if a page cannot be fetched after all retries.

        Args:
            folder: Folder path to list (empty string = bucket root).
            files_only: If True, skip subfolder placeholders (id=None).
            max_attempts: Per-page retry budget.
            backoff_base: Seconds; delay is backoff_base * 2**attempt.
        """
        page_size = 100
        offset = 0
        results: List[dict] = []
        while True:
            def _fetch_page(_folder: str = folder, _offset: int = offset) -> List[dict]:
                return self.bucket.list(
                    _folder, {"limit": page_size, "offset": _offset}
                )

            def _log_retry(attempt: int, exc: BaseException, delay: float,
                           _folder: str = folder, _offset: int = offset) -> None:
                logger.warning(
                    f"List failed for {_folder!r} offset={_offset} "
                    f"(attempt {attempt}/{max_attempts}): {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )

            try:
                page = retry_with_backoff(
                    _fetch_page,
                    max_attempts=max_attempts,
                    backoff_base=backoff_base,
                    on_retry=_log_retry,
                    # Route through the module's `time` attribute so tests that
                    # patch `archive_storage.time` continue to observe sleeps.
                    sleep=lambda d: time.sleep(d),
                )
            except Exception as last_error:
                raise ArchiveStorageError(
                    f"Failed to list {folder!r} at offset={offset} "
                    f"after {max_attempts} attempts: {last_error}"
                ) from last_error

            if files_only:
                results.extend(f for f in page if f.get("id"))
            else:
                results.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return results

    def delete_all(self) -> int:
        """Delete all files from bucket. Used by clean command.

        Returns:
            Number of files deleted.
        """
        try:
            # List root level entries (doc_id folders) — paginated
            root_entries = self.list_folder("", files_only=False)
            count = 0

            for entry in root_entries:
                folder_name = entry["name"]
                paths: List[str] = []

                for f in self.list_folder(folder_name):
                    paths.append(f"{folder_name}/{f['name']}")

                for f in self.list_folder(f"{folder_name}/attachments"):
                    paths.append(f"{folder_name}/attachments/{f['name']}")

                if paths:
                    try:
                        self.bucket.remove(paths)
                        count += len(paths)
                    except StorageException as e:
                        logger.warning(f"Failed to clean folder {folder_name}: {e}")

            logger.info(f"Deleted {count} archive files")
            return count

        except StorageException as e:
            raise ArchiveStorageError(f"Failed to delete all archive files: {e}") from e

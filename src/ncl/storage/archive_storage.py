"""Supabase Storage client for archive file operations.

Provides upload, download, and deletion for the browsable archive.
Uses Supabase Storage API directly (no DB tracking needed).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from storage3.utils import StorageException
from supabase import Client, create_client

from ..config import get_settings

logger = logging.getLogger(__name__)


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
        # Split path into folder and filename
        parts = path.rsplit("/", 1)
        if len(parts) == 2:
            folder, filename = parts
        else:
            folder, filename = "", parts[0]

        try:
            files = self.bucket.list(folder)
            return any(f["name"] == filename for f in files)
        except StorageException:
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

            # List files in root folder
            try:
                files = self.bucket.list(folder)
                for f in files:
                    if preserve_md and f["name"].endswith(".md"):
                        continue
                    paths.append(f"{folder}/{f['name']}")
            except StorageException:
                pass  # Folder may not exist

            # List files in attachments subfolder
            try:
                att_files = self.bucket.list(f"{folder}/attachments")
                for f in att_files:
                    if preserve_md and f["name"].endswith(".md"):
                        continue
                    paths.append(f"{folder}/attachments/{f['name']}")
            except StorageException:
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
            files = self.bucket.list(folder)
            result.extend(files)

            # Also list attachments subfolder
            try:
                att_files = self.bucket.list(f"{folder}/attachments")
                result.extend(att_files)
            except StorageException:
                pass
        except StorageException:
            pass

        return result

    def delete_all(self) -> int:
        """Delete all files from bucket. Used by clean command.

        Returns:
            Number of files deleted.
        """
        try:
            # List root level folders (doc_id folders)
            folders = self.bucket.list("")
            count = 0

            for folder in folders:
                folder_name = folder["name"]
                paths: List[str] = []

                # List files in folder
                try:
                    files = self.bucket.list(folder_name)
                    paths.extend([f"{folder_name}/{f['name']}" for f in files])
                except StorageException:
                    pass

                # List files in attachments subfolder
                try:
                    att_files = self.bucket.list(f"{folder_name}/attachments")
                    paths.extend([f"{folder_name}/attachments/{f['name']}" for f in att_files])
                except StorageException:
                    pass

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

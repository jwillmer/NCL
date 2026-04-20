"""Filesystem-backed archive storage.

Drop-in for :class:`ArchiveStorage` (Supabase bucket) when running ingest
locally. Files land under ``<bucket_dir>/<folder_id>/...`` on disk — the
same relative paths ingest writes to Supabase in the cloud. Used by
``create_local_ingest_components`` to wire the local output directory's
``archive/`` subtree as the "bucket".
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LocalBucketStorage:
    """Mock ArchiveStorage that writes to local folders."""

    bucket_dir: Path

    def __post_init__(self) -> None:
        self.bucket_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, path: str, content: bytes, content_type: str) -> str:
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
        return self.upload_file(path, text.encode("utf-8"), content_type)

    def download_file(self, path: str) -> bytes:
        file_path = self.bucket_dir / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_bytes()

    def download_text(self, path: str, encoding: str = "utf-8") -> str:
        return self.download_file(path).decode(encoding)

    def file_exists(self, path: str) -> bool:
        return (self.bucket_dir / path).exists()

    def delete_folder(self, doc_id: str, preserve_md: bool = False) -> None:
        folder_path = self.bucket_dir / doc_id
        if not folder_path.exists():
            return
        if preserve_md:
            for file_path in folder_path.rglob("*"):
                if file_path.is_file() and file_path.suffix != ".md":
                    file_path.unlink()
        else:
            shutil.rmtree(folder_path)

    def list_files(self, doc_id: str) -> List[Dict[str, Any]]:
        folder_path = self.bucket_dir / doc_id
        if not folder_path.exists():
            return []
        files: List[Dict[str, Any]] = []
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.bucket_dir)),
                    "size": file_path.stat().st_size,
                })
        return files

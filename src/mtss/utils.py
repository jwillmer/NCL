"""Shared utility functions."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def normalize_source_id(file_path: str, ingest_root: Path) -> str:
    """Normalize file path to stable source_id relative to ingest root.

    Creates a consistent identifier for a file that remains stable across
    different machines and re-ingestion runs.

    Args:
        file_path: Absolute or relative path to the file.
        ingest_root: Root directory for ingestion (paths are relative to this).

    Returns:
        Normalized source ID (lowercase relative path with forward slashes).
    """
    abs_path = Path(file_path).resolve()
    try:
        rel_path = abs_path.relative_to(ingest_root.resolve())
    except ValueError:
        # Path is not relative to ingest_root, use absolute path
        rel_path = abs_path
    return rel_path.as_posix().lower()


def compute_doc_id(source_id: str, file_hash: str) -> str:
    """Generate content-addressable document ID.

    Creates a deterministic ID based on source location and content hash.
    This allows detection of content changes while maintaining stable references.

    Args:
        source_id: Normalized source path from normalize_source_id().
        file_hash: SHA-256 hash of file content.

    Returns:
        16-character hex string document ID.
    """
    combined = f"{source_id}:{file_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Archive folder name length. Doc_id is 16 hex chars; folder_id is derived
# separately so orphan cleanup and URI rewrites operate in a distinct
# keyspace from doc_id — harder to accidentally collide / misuse one for
# the other, and lays groundwork for expanding doc_id entropy later.
FOLDER_ID_LENGTH = 32


def compute_folder_id(doc_id: str) -> str:
    """Deterministic 32-char archive folder identifier derived from doc_id.

    Previously the archive folder name was ``doc_id[:16]`` — identical to
    the doc_id itself. Tools that conflated the two (orphan cleanup,
    validate reports) could produce confusing output. Using a distinct
    namespace makes the folder / doc_id relationship explicit.

    Args:
        doc_id: Document ID from ``compute_doc_id``.

    Returns:
        32-character hex string folder identifier.
    """
    return hashlib.sha256(f"folder:{doc_id}".encode()).hexdigest()[:FOLDER_ID_LENGTH]


# Length of chunk IDs used for citations (hex characters)
CHUNK_ID_LENGTH = 12


def compute_chunk_id(doc_id: str, char_start: int, char_end: int) -> str:
    """Generate deterministic chunk ID from document and character offsets.

    Creates a stable ID for a chunk that can be used for citation references.
    The ID is deterministic so the same chunk will always have the same ID.

    Args:
        doc_id: Document ID from compute_doc_id().
        char_start: Starting character offset in document.
        char_end: Ending character offset in document.

    Returns:
        12-character hex string chunk ID.
    """
    combined = f"{doc_id}:{char_start}:{char_end}"
    return hashlib.sha256(combined.encode()).hexdigest()[:CHUNK_ID_LENGTH]


# Synthetic chunk-id positions for chunks that don't map to a markdown range.
# Negative char_start sentinels keep these IDs disjoint from real-position IDs.
SUMMARY_CHUNK_POS: tuple[int, int] = (-1, 0)
METADATA_CHUNK_POS: tuple[int, int] = (-2, 0)


def strip_archive_prefix(uri: str) -> str:
    """Drop a leading ``/archive/`` so storage backends can resolve the key."""
    return uri.removeprefix("/archive/")


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename for safe filesystem use.

    Removes or replaces characters that are problematic on various filesystems.

    Args:
        filename: Original filename.
        max_length: Maximum allowed filename length.

    Returns:
        Sanitized filename safe for filesystem.
    """
    # Remove null bytes and path separators
    filename = filename.replace("\x00", "").replace("/", "_").replace("\\", "_")
    # Remove other problematic characters (Windows reserved: < > : " | ? *)
    filename = re.sub(r'[<>:"|?*]', "_", filename)
    # Limit length
    return filename[:max_length]

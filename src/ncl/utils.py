"""Shared utility functions."""

from __future__ import annotations

import re


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

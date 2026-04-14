"""Shared image filtering heuristics for pre-filtering before Vision API calls."""

from __future__ import annotations

import re
from pathlib import Path

# Hard skip: always filter regardless of file size
_HARD_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"logo", re.IGNORECASE),
    re.compile(r"banner", re.IGNORECASE),
    re.compile(r"signature", re.IGNORECASE),
    re.compile(r"icon", re.IGNORECASE),
]

# Soft skip: only filter when file is small (<100KB).
# Outlook names inline images image001.png, image002.png etc. — large ones
# (e.g. 243KB equipment photos) are meaningful content, not decorations.
_SOFT_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"image\d{3}", re.IGNORECASE),
]

_SOFT_SKIP_SIZE_THRESHOLD = 100_000  # 100 KB


def is_meaningful_image(path: Path) -> bool:
    """Heuristic: return True if the image is likely meaningful content.

    Filters out tracking pixels, icons, logos, banners, and email
    signature images based on filename patterns, file size, and dimensions.
    """
    stem = path.stem

    # Hard-skip patterns: always filter (no I/O needed)
    for pattern in _HARD_SKIP_PATTERNS:
        if pattern.search(stem):
            return False

    # Read file size (needed for soft-skip and small-file checks)
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = None  # can't check, assume meaningful

    # Soft-skip patterns: only filter when file is confirmed small
    for pattern in _SOFT_SKIP_PATTERNS:
        if pattern.search(stem):
            if file_size is not None and file_size < _SOFT_SKIP_SIZE_THRESHOLD:
                return False

    # Very small files are almost always tracking pixels or tiny icons
    if file_size is not None and file_size < 15_000:
        return False

    # Check dimensions with PIL for more accurate filtering
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        # Tiny images: icons, tracking pixels, small logos
        if max(w, h) < 100:
            return False
        # Banner-shaped: wide and short (email separators, signature strips)
        if h < 50 and w > 3 * h:
            return False
    except Exception:
        pass  # can't read dimensions, file size check is enough

    return True

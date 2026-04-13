"""Shared image filtering heuristics for pre-filtering before Vision API calls."""

from __future__ import annotations

import re
from pathlib import Path

_SKIP_FILENAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"logo", re.IGNORECASE),
    re.compile(r"banner", re.IGNORECASE),
    re.compile(r"signature", re.IGNORECASE),
    re.compile(r"icon", re.IGNORECASE),
    re.compile(r"image\d{3}", re.IGNORECASE),
]


def is_meaningful_image(path: Path) -> bool:
    """Heuristic: return True if the image is likely meaningful content.

    Filters out tracking pixels, icons, logos, banners, and email
    signature images based on filename patterns, file size, and dimensions.
    """
    # Filename pattern check
    stem = path.stem
    for pattern in _SKIP_FILENAME_PATTERNS:
        if pattern.search(stem):
            return False

    try:
        file_size = path.stat().st_size
    except OSError:
        return True  # can't check, assume meaningful

    # Very small files are almost always tracking pixels or tiny icons
    if file_size < 15_000:
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

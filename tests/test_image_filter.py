"""Tests for image_filter.is_meaningful_image heuristic."""

from pathlib import Path

from mtss.image_filter import is_meaningful_image


def _make_file(tmp_path: Path, name: str, size: int) -> Path:
    """Create a file with the given name and byte size."""
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return p


def test_small_file_filtered(tmp_path: Path) -> None:
    """Files under 15 KB are filtered as non-meaningful."""
    p = _make_file(tmp_path, "photo.png", 10_000)
    assert is_meaningful_image(p) is False


def test_logo_filename_filtered(tmp_path: Path) -> None:
    """Hard-skip pattern 'logo' always filters regardless of size."""
    p = _make_file(tmp_path, "company_logo.png", 200_000)
    assert is_meaningful_image(p) is False


def test_image_nnn_small_filtered(tmp_path: Path) -> None:
    """Small Outlook-style image002.png files are filtered."""
    p = _make_file(tmp_path, "image002.png", 10_000)
    assert is_meaningful_image(p) is False


def test_image_nnn_large_passes(tmp_path: Path) -> None:
    """Large Outlook-style image002.png files pass (the bug fix)."""
    p = _make_file(tmp_path, "image002.png", 200_000)
    assert is_meaningful_image(p) is True


def test_normal_image_passes(tmp_path: Path) -> None:
    """Normal-named images above 15 KB pass through."""
    p = _make_file(tmp_path, "equipment_photo.jpg", 50_000)
    assert is_meaningful_image(p) is True

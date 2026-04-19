"""Tests for narrowed except-clauses in mtss.ingest.estimator and the shared
PDF classifier in mtss.parsers.pdf_classifier.

Covers Fix #16 from code-health-report.md: four broad `except Exception:`
catches were narrowed to specific exception types. The PDF-reader-side
classification logic moved to ``parsers.pdf_classifier.classify_reader``;
these tests still cover its narrowed catches.

1. Positive — the specific exception family is caught (fallback taken).
2. Negative — an unrelated exception propagates (narrowing took effect).
"""

from __future__ import annotations

import binascii
from unittest.mock import MagicMock, patch

import pytest
from pypdf.errors import PdfReadError

from mtss.ingest.estimator import IngestEstimator
from mtss.parsers.pdf_classifier import PDFComplexity, classify_reader


# ---------------------------------------------------------------------------
# Catch #1 (line ~492): base64.b64decode — narrowed to (binascii.Error, ValueError)
# ---------------------------------------------------------------------------


class TestBase64DecodeNarrowing:
    """The base64-decode fallback should swallow binascii/ValueError only."""

    def test_binascii_error_is_caught_and_falls_through(self, tmp_path):
        """A b"JVBER"-prefixed file with invalid base64 triggers binascii.Error
        inside the narrowed catch. The function must NOT re-raise; it falls back
        to the regex-count path and returns a successful tuple."""
        pdf_file = tmp_path / "fake.pdf"
        # Starts with "JVBER" (so base64-decode path is taken) but the rest is
        # non-base64 garbage. base64.b64decode will raise binascii.Error.
        # Include a /Type /Page so regex fallback succeeds.
        payload = b"JVBER" + b"!!!!!!!!" + b"/Type /Page " * 3
        pdf_file.write_bytes(payload)

        estimator = IngestEstimator(source_dir=tmp_path, estimate_dir=tmp_path)
        count, issue, complexity = estimator._count_pdf_pages(pdf_file, "fake.pdf")

        # regex fallback wins, returning 3 pages with "complex" and no issue
        assert count == 3
        assert issue is None
        assert complexity == "complex"

    def test_unrelated_exception_propagates(self, tmp_path):
        """If base64.b64decode is patched to raise a KeyError (unrelated to
        the narrowed tuple), the exception must bubble up unmasked."""
        pdf_file = tmp_path / "fake.pdf"
        pdf_file.write_bytes(b"JVBER" + b"abcd" * 8)

        estimator = IngestEstimator(source_dir=tmp_path, estimate_dir=tmp_path)

        with patch("mtss.ingest.estimator.base64.b64decode", side_effect=KeyError("boom")):
            with pytest.raises(KeyError):
                estimator._count_pdf_pages(pdf_file, "fake.pdf")


# ---------------------------------------------------------------------------
# Catch #2 (line ~502): PdfReader open/pages — narrowed to (PyPdfError, OSError, ValueError)
# ---------------------------------------------------------------------------


class TestPdfReaderNarrowing:
    """Strategy 1 (pypdf) catch should handle pypdf failures but not random ones."""

    def test_pdf_read_error_is_caught_and_falls_through(self, tmp_path):
        """A corrupt-but-regex-matchable PDF: pypdf raises PdfReadError (subclass
        of PyPdfError), the function falls back to regex, returns a page count."""
        pdf_file = tmp_path / "corrupt.pdf"
        # No JVBER prefix → skip base64 branch. Malformed header, but includes
        # /Type /Page tokens so the regex fallback succeeds.
        payload = b"%PDF-1.4 bogus " + b"/Type /Page " * 2 + b"\nnot a real pdf"
        pdf_file.write_bytes(payload)

        estimator = IngestEstimator(source_dir=tmp_path, estimate_dir=tmp_path)
        count, issue, complexity = estimator._count_pdf_pages(pdf_file, "corrupt.pdf")

        # pypdf fails → regex finds 2 /Type /Page occurrences
        assert count == 2
        assert issue is None
        assert complexity == "complex"

    def test_unrelated_exception_propagates(self, tmp_path):
        """If PdfReader is patched to raise RuntimeError (not in narrowed tuple),
        the exception propagates out of _count_pdf_pages."""
        pdf_file = tmp_path / "any.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 junk")

        estimator = IngestEstimator(source_dir=tmp_path, estimate_dir=tmp_path)

        class _FakePdfReader:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("unexpected runtime failure")

        with patch("pypdf.PdfReader", _FakePdfReader):
            with pytest.raises(RuntimeError):
                estimator._count_pdf_pages(pdf_file, "any.pdf")

    def test_pypdf_error_family_caught_explicitly(self, tmp_path):
        """Verify PyPdfError subclasses are in the caught tuple by patching
        PdfReader to raise PdfReadError and asserting the function returns
        the regex-fallback result instead of raising."""
        pdf_file = tmp_path / "patched.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 /Type /Page dummy")

        estimator = IngestEstimator(source_dir=tmp_path, estimate_dir=tmp_path)

        class _FakePdfReader:
            def __init__(self, *args, **kwargs):
                raise PdfReadError("simulated pypdf failure")

        with patch("pypdf.PdfReader", _FakePdfReader):
            count, issue, complexity = estimator._count_pdf_pages(pdf_file, "patched.pdf")
            assert count == 1  # single /Type /Page match in our payload
            assert issue is None
            assert complexity == "complex"


# ---------------------------------------------------------------------------
# Catch #3 (line ~537): reader.get_fields() — narrowed to (PyPdfError, KeyError, AttributeError, TypeError)
# ---------------------------------------------------------------------------


class TestGetFieldsNarrowing:
    """classify_reader's get_fields() catch should swallow pypdf/type errors
    but propagate anything unrelated."""

    def _make_reader(self, get_fields_raises):
        reader = MagicMock()
        reader.pages = [MagicMock()]
        reader.get_fields.side_effect = get_fields_raises
        return reader

    def test_keyerror_is_caught_and_returns_complex(self):
        reader = self._make_reader(get_fields_raises=KeyError("/AcroForm"))
        assert classify_reader(reader) == PDFComplexity.COMPLEX

    def test_pdfreaderror_is_caught_and_returns_complex(self):
        reader = self._make_reader(get_fields_raises=PdfReadError("bad xref"))
        assert classify_reader(reader) == PDFComplexity.COMPLEX

    def test_unrelated_exception_propagates(self):
        reader = self._make_reader(get_fields_raises=RuntimeError("whoops"))
        with pytest.raises(RuntimeError):
            classify_reader(reader)


# ---------------------------------------------------------------------------
# Catch #4: page.extract_text() — narrowed to (PyPdfError, KeyError, AttributeError, TypeError, ValueError)
# ---------------------------------------------------------------------------


class TestExtractTextNarrowing:
    """classify_reader's extract_text() catch should swallow pypdf extraction
    errors but propagate anything unrelated."""

    def _make_reader_with_page(self, extract_text_raises):
        reader = MagicMock()
        reader.get_fields.return_value = None
        page = MagicMock()
        page.extract_text.side_effect = extract_text_raises
        reader.pages = [page]
        return reader

    def test_pdfreaderror_is_caught_and_returns_complex(self):
        reader = self._make_reader_with_page(
            extract_text_raises=PdfReadError("stream decode failed")
        )
        assert classify_reader(reader) == PDFComplexity.COMPLEX

    def test_valueerror_is_caught_and_returns_complex(self):
        reader = self._make_reader_with_page(extract_text_raises=ValueError("bad float"))
        assert classify_reader(reader) == PDFComplexity.COMPLEX

    def test_unrelated_exception_propagates(self):
        reader = self._make_reader_with_page(extract_text_raises=OSError("disk gone"))
        with pytest.raises(OSError):
            classify_reader(reader)


# ---------------------------------------------------------------------------
# Sanity: binascii.Error IS-A ValueError (both listed in the narrowed tuple
# defensively; this test documents that).
# ---------------------------------------------------------------------------


def test_binascii_error_is_valueerror_subclass():
    assert issubclass(binascii.Error, ValueError)

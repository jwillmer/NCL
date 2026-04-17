"""Tests for `mtss.parsers.email_cleaner`.

The cleaner is best-effort: every failure path (no JSON, truncated JSON,
anchors not found, network error) must fall back to ``(0, len(text))`` so
the pipeline still chunks the full body rather than losing content silently.
The tests here lock those fallback guarantees plus the ``max_tokens`` budget
that was bumped after we observed unterminated-JSON warnings on signatures
containing long tracking URLs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mtss.parsers import email_cleaner


def _mock_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_max_tokens_fits_url_bearing_signatures():
    """Regression: ``max_tokens=300`` truncated mid-string when ``last_words``
    included a long URL, producing unterminated-JSON warnings. The budget
    must stay well above 300 so the JSON closes even with realistic
    anchor lengths.
    """
    captured: dict = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _mock_response('{"first_words": "Hello", "last_words": "world"}')

    text = "Hello dear friend, this is the body. Goodbye world."
    with patch.object(email_cleaner, "acompletion", AsyncMock(side_effect=_capture)):
        await email_cleaner.extract_content_bounds(text, model="gpt-test")

    assert captured["max_tokens"] >= 600, (
        "max_tokens must stay >= 600 to avoid truncating JSON on long URL anchors"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_falls_back_to_full_text_when_json_truncated(caplog):
    """When the LLM response is unterminated JSON (previously the 300-token
    truncation symptom), ``extract_content_bounds`` must return full-text
    bounds, not raise. This keeps the pipeline safe even if the token bump
    is ever reverted.
    """
    text = "Body content here."
    truncated = '{"first_words": "Body content here.", "last_words": "very long url'
    with patch.object(
        email_cleaner, "acompletion", AsyncMock(return_value=_mock_response(truncated))
    ):
        start, end = await email_cleaner.extract_content_bounds(text, model="gpt-test")

    assert (start, end) == (0, len(text))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_falls_back_when_anchor_not_in_text():
    """LLM paraphrased ``last_words`` instead of quoting verbatim. The
    anchor can't be located, so bounds default to the full text.
    """
    text = "Dear John,\n\nPlease ship the spare parts.\n\nBest,\nAcme Ltd."
    response = '{"first_words": "Dear John", "last_words": "Sincerely yours, Acme"}'
    with patch.object(
        email_cleaner, "acompletion", AsyncMock(return_value=_mock_response(response))
    ):
        start, end = await email_cleaner.extract_content_bounds(text, model="gpt-test")

    assert end == len(text), "unfound last_words must fall back to end-of-text"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uses_anchors_when_found_verbatim():
    """Happy path: anchors match verbatim → bounds trim to the meaningful slice."""
    text = "JUNK HEADER\nDear John, please ship parts. Regards, Acme\nSIGNATURE BLOCK"
    response = '{"first_words": "Dear John", "last_words": "Regards, Acme"}'
    with patch.object(
        email_cleaner, "acompletion", AsyncMock(return_value=_mock_response(response))
    ):
        start, end = await email_cleaner.extract_content_bounds(text, model="gpt-test")

    assert text[start:end] == "Dear John, please ship parts. Regards, Acme"

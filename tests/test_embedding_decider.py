"""Tests for the embedding-mode decider + content-shape analyzer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ------------------------------ TestAnalyze ------------------------------ #


class TestAnalyze:
    """Pure shape analysis — no I/O, no mocks."""

    def test_empty_markdown_zero_everything(self):
        from mtss.ingest.embedding_decider import analyze

        s = analyze("")
        assert s.total_chars == 0
        assert s.total_tokens == 0
        assert s.digit_ratio == 0.0
        assert s.table_char_pct == 0.0
        assert s.prose_ratio == 0.0
        assert s.heading_count == 0

    def test_plain_prose_high_prose_ratio_low_digit_ratio(self):
        from mtss.ingest.embedding_decider import analyze

        md = (
            "# Contract\n\n"
            "This agreement between Party A and Party B governs the terms of "
            "services rendered during the period from January through December. "
            "All parties agree to the stipulations outlined below.\n"
        )
        s = analyze(md)
        assert s.prose_ratio > 0.7
        assert s.digit_ratio < 0.05
        assert s.heading_count == 1
        assert s.table_char_pct < 0.1

    def test_sensor_log_high_digit_ratio(self):
        from mtss.ingest.embedding_decider import analyze

        md = "\n".join(
            f"2025-06-{(i % 28) + 1:02d} 12:00 pH 6.{i % 9} flow {42 + i} temp {20 + (i % 10)}"
            for i in range(200)
        )
        s = analyze(md)
        assert s.digit_ratio > 0.4
        assert s.total_tokens > 1000

    def test_table_dominant_high_table_char_pct(self):
        from mtss.ingest.embedding_decider import analyze

        rows = ["| Col1 | Col2 | Col3 |", "|------|------|------|"]
        for i in range(50):
            rows.append(f"| r{i} | v{i} | x{i} |")
        md = "\n".join(rows)
        s = analyze(md)
        assert s.table_char_pct > 0.8

    def test_repetitive_lines_high_repetition_score(self):
        from mtss.ingest.embedding_decider import analyze

        md = "\n".join(["status: ok"] * 50 + ["status: ok"] * 50)
        s = analyze(md)
        assert s.repetition_score > 0.95

    def test_short_stub_doc_low_token_count(self):
        from mtss.ingest.embedding_decider import analyze

        s = analyze("OK")
        assert s.total_tokens < 50

    def test_heading_rich_prose_counts_headings(self):
        from mtss.ingest.embedding_decider import analyze

        md = "# H1\n\ntext\n\n## H2\n\nmore\n\n### H3\n\nend\n"
        s = analyze(md)
        assert s.heading_count == 3


# ------------------------------ TestDecider ------------------------------ #


class TestDecider:
    """Decision-tree branches — rules-based (no LLM reached)."""

    @pytest.fixture
    def settings(self, comprehensive_mock_settings):
        return comprehensive_mock_settings

    @pytest.mark.asyncio
    async def test_tiny_doc_returns_metadata_only(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        d = await decide_embedding_mode("Hello.", None, settings)
        assert d.mode == EmbeddingMode.METADATA_ONLY
        assert d.reason == "too_short"

    @pytest.mark.asyncio
    async def test_no_prose_returns_metadata_only(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        numbers = " ".join(str(i) for i in range(400))
        d = await decide_embedding_mode(numbers, None, settings)
        assert d.mode == EmbeddingMode.METADATA_ONLY
        assert d.reason == "no_prose"

    @pytest.mark.asyncio
    async def test_bulk_numeric_returns_summary(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        md = "# Sensor Log\n\n" + "\n".join(
            f"2025-06-{(i % 28) + 1:02d} 12:00 pH 6.{i % 9} flow {42 + i} temp {20 + (i % 10)}"
            for i in range(2500)
        )
        d = await decide_embedding_mode(md, None, settings)
        assert d.mode == EmbeddingMode.SUMMARY
        assert d.reason == "bulk_numeric"

    @pytest.mark.asyncio
    async def test_table_dominant_returns_summary(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        rows = ["# Report\n"]
        rows.append("| Col1 | Col2 |")
        rows.append("|------|------|")
        for i in range(5000):
            rows.append(f"| alpha{i} | beta{i} |")
        md = "\n".join(rows)
        d = await decide_embedding_mode(md, None, settings)
        assert d.mode == EmbeddingMode.SUMMARY
        assert d.reason in {"tabular", "bulk_numeric", "repetitive", "short_lines"}

    @pytest.mark.asyncio
    async def test_repetitive_returns_summary(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        md = "# Heading\n\n" + "\n".join(["status ok ready green normal"] * 5000)
        d = await decide_embedding_mode(md, None, settings)
        assert d.mode == EmbeddingMode.SUMMARY
        assert d.reason in {"repetitive", "tabular", "bulk_numeric", "short_lines"}

    @pytest.mark.asyncio
    async def test_prose_default_returns_full(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        paragraph = (
            "This agreement between Party A and Party B governs the terms of "
            "services rendered during the period from January through December. "
            "All parties agree to the stipulations outlined below. The contract "
            "is subject to applicable law in the jurisdiction of its signing.\n\n"
        )
        md = "# Contract\n\n" + paragraph * 20
        d = await decide_embedding_mode(md, None, settings)
        assert d.mode == EmbeddingMode.FULL
        assert d.reason == "default_prose"


# --------------------------- TestDeciderTriage --------------------------- #


class TestDeciderTriage:
    """Medium-confidence band — LLM triage always runs, no flag."""

    @pytest.fixture
    def settings(self, comprehensive_mock_settings):
        # Lower the medium bar so moderate-size mixed docs reach triage.
        comprehensive_mock_settings.decider_medium_token_threshold = 5_000
        comprehensive_mock_settings.decider_medium_prose_ratio = 0.70
        # Raise the bulk threshold so those rules don't pre-empt the triage branch.
        comprehensive_mock_settings.decider_bulk_token_threshold = 100_000
        return comprehensive_mock_settings

    def _mixed_markdown(self) -> str:
        prose = (
            "# Audit Report\n\n"
            "This audit covers the period from March to September. Findings "
            "were discussed with stakeholders and reviewed by legal. "
        )
        numbers = " ".join(f"{i}.{i + 1}" for i in range(3000))
        return prose + "\n\n" + numbers

    @pytest.mark.asyncio
    async def test_triage_A_returns_full(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "A — mostly prose"
        with patch(
            "mtss.ingest.embedding_decider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as m:
            d = await decide_embedding_mode(self._mixed_markdown(), None, settings)
            assert m.called
            assert d.mode == EmbeddingMode.FULL
            assert d.reason == "triage_prose"

    @pytest.mark.asyncio
    async def test_triage_B_returns_summary(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "B — mostly numeric"
        with patch(
            "mtss.ingest.embedding_decider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            d = await decide_embedding_mode(self._mixed_markdown(), None, settings)
            assert d.mode == EmbeddingMode.SUMMARY
            assert d.reason == "triage_dense"

    @pytest.mark.asyncio
    async def test_triage_C_returns_metadata_only(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "C — form with no real content"
        with patch(
            "mtss.ingest.embedding_decider.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            d = await decide_embedding_mode(self._mixed_markdown(), None, settings)
            assert d.mode == EmbeddingMode.METADATA_ONLY
            assert d.reason == "triage_noise"

    @pytest.mark.asyncio
    async def test_triage_failure_falls_back_to_summary(self, settings):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        with patch(
            "mtss.ingest.embedding_decider.litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("rate limited"),
        ):
            d = await decide_embedding_mode(self._mixed_markdown(), None, settings)
            assert d.mode == EmbeddingMode.SUMMARY
            assert d.reason == "triage_failed"


# -------------------- TestThresholdsRespectSettings --------------------- #


class TestThresholdsRespectSettings:
    """Thresholds must drive behavior — tuning via env, no code change."""

    @pytest.mark.asyncio
    async def test_raising_digit_ratio_prevents_bulk_numeric(
        self, comprehensive_mock_settings
    ):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        # Doc well above default thresholds: big, digit-heavy sensor log.
        md = "# Log\n\n" + "\n".join(
            f"2025-06-{(i % 28) + 1:02d} 12:00 pH 6.{i % 9} flow {42 + i} temp {20 + (i % 10)}"
            for i in range(2500)
        )

        # Default digit ratio (0.40) — flips to SUMMARY on bulk_numeric.
        d_default = await decide_embedding_mode(md, None, comprehensive_mock_settings)
        assert d_default.mode == EmbeddingMode.SUMMARY
        assert d_default.reason == "bulk_numeric"

        # Raise the threshold above what this doc has — should NOT flip for that reason.
        comprehensive_mock_settings.decider_digit_ratio = 0.99
        # Also neutralize the other SUMMARY rules so we isolate digit_ratio.
        comprehensive_mock_settings.decider_table_char_pct = 0.99
        comprehensive_mock_settings.decider_repetition_score = 0.99
        comprehensive_mock_settings.decider_short_line_ratio = 0.99
        d_high = await decide_embedding_mode(md, None, comprehensive_mock_settings)
        assert d_high.reason != "bulk_numeric"

    @pytest.mark.asyncio
    async def test_raising_short_token_threshold_changes_metadata_boundary(
        self, comprehensive_mock_settings
    ):
        from mtss.ingest.embedding_decider import decide_embedding_mode
        from mtss.models.document import EmbeddingMode

        md = "# H\n\n" + "This is a moderate length prose sentence for testing. " * 40
        # Default threshold (50) — this doc is well above it.
        d_default = await decide_embedding_mode(md, None, comprehensive_mock_settings)
        assert d_default.mode != EmbeddingMode.METADATA_ONLY

        # Raise the threshold so this doc counts as too-short.
        comprehensive_mock_settings.decider_short_token_threshold = 10_000
        d_raised = await decide_embedding_mode(md, None, comprehensive_mock_settings)
        assert d_raised.mode == EmbeddingMode.METADATA_ONLY
        assert d_raised.reason == "too_short"

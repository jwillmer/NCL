"""Tests for scripts/analyze_rag_utility.py.

These lock the analyzer's contract: the heuristics correctly classify the
obvious extremes, the LLM classifier handles borderline prose without
being called on content heuristics already decided, the report captures
metrics + source + reason, and target selection filters to the docs that
actually pose an embedding-noise risk.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "analyze_rag_utility.py"
)


def _borderline_fixture(seed: int = 1) -> str:
    """Synthetic maritime maintenance log whose structural metrics land
    squarely in heuristic-abstain territory: ~5% unique words, ~15% numeric,
    moderate line length, moderate sentence length. Varied enough to dodge
    the extreme_repetition skip but too short-sentenced and number-sprinkled
    to qualify as narrative_rich, so ``heuristic_classify`` returns None."""
    import random as _random

    r = _random.Random(seed)
    VERBS = [
        "observed", "inspected", "recorded", "monitored", "tested",
        "reviewed", "replaced", "adjusted", "calibrated", "isolated",
    ]
    SUBJECTS = [
        "port engine", "starboard engine", "auxiliary generator",
        "cargo pump", "ballast pump", "fire pump", "HVAC compressor",
        "fuel separator", "oil purifier", "steering gear",
    ]
    COMPONENTS = [
        "crankshaft", "camshaft", "piston", "bearing", "journal",
        "gasket", "seal", "filter", "valve", "nozzle",
    ]
    ADVERBS = [
        "during", "following", "after", "before", "alongside",
        "amidst", "concurrent with",
    ]
    STATES = ["nominal", "elevated", "critical", "pending review"]
    return "\n".join(
        f"{r.choice(VERBS).capitalize()} the {r.choice(COMPONENTS)} on "
        f"{r.choice(SUBJECTS)} {r.choice(ADVERBS)} the "
        f"{r.randint(1, 24):02d}:00 watch at {r.randint(50, 200)} psi. "
        f"Reading {r.randint(1, 999)} {r.choice(STATES)}."
        for _ in range(80)
    )


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("rag_utility", _SCRIPT_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["rag_utility"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------


class TestHeuristicClassify:
    """Structural classifier. The strong-signal branches must not misfire on
    normal prose, and the narrative-rich branch must not swallow obvious
    telemetry."""

    def test_sensor_log_classifies_as_skip(self, mod):
        # Rows like a real log export: timestamp + numeric columns, few unique
        # words ("ENG_TEMP", "FLOW_RPM"), high numeric density.
        log = "\n".join(
            f"2025-01-{i:02d} 12:00:00  ENG_TEMP 345.{i} FLOW_RPM 1200.{i} STATUS OK"
            for i in range(1, 200)
        )
        metrics = mod.compute_metrics(log)
        cls, reason = mod.heuristic_classify(metrics)
        assert cls == mod.CLASSIFICATION_SKIP
        assert "numeric" in reason or "repetition" in reason

    def test_repetitive_boilerplate_classifies_as_skip(self, mod):
        # Same line 500 times — degenerate case for unique_word_ratio.
        text = "This document intentionally left blank\n" * 500
        metrics = mod.compute_metrics(text)
        cls, reason = mod.heuristic_classify(metrics)
        assert cls == mod.CLASSIFICATION_SKIP

    def test_tabular_short_lines_classifies_as_summary_only(self, mod):
        # Realistic certificate: varied field names + short value lines.
        import random

        r = random.Random(0)
        labels = [
            "Vessel Name", "IMO Number", "Flag", "Port of Registry",
            "Classification Society", "Gross Tonnage", "Net Tonnage",
            "Date Built", "Keel Laid", "Builder", "Owner", "Operator",
            "Callsign", "MMSI", "Insurance", "P&I Club", "Last Docking",
            "Next Docking", "Engine Make", "Engine Power",
        ]
        values = [f"VAL-{r.randint(100, 9999)}" for _ in range(200)]
        tab = "\n".join(
            f"{labels[r.randint(0, len(labels) - 1)]}: {v}" for v in values
        )
        metrics = mod.compute_metrics(tab)
        cls, _ = mod.heuristic_classify(metrics)
        assert cls == mod.CLASSIFICATION_SUMMARY

    def test_narrative_prose_classifies_as_keep(self, mod):
        """A realistic technical email/description with long sentences and
        varied vocabulary."""
        paras = [
            "During the fuel transfer operation on tank 2 we observed "
            "abnormal pressure readings on the forward manifold. The chief "
            "engineer inspected the gauge and confirmed that the transducer "
            "had drifted by approximately twelve percent, which accounts "
            "for the elevated alarm threshold reported overnight.",
            "To prevent recurrence the shore office requested that we "
            "rotate the spare transducer from storage bin B12 into service "
            "at the next berthing. Technical staff will run a calibration "
            "procedure against the reference manometer once the vessel "
            "reaches the anchorage off Rotterdam.",
            "Please advise whether a replacement transducer should also be "
            "shipped to the Kalundborg agent for crew pickup. Our current "
            "spare inventory covers only the immediate replacement and the "
            "secondary circuit remains without redundancy.",
        ] * 5
        prose = "\n\n".join(paras)
        metrics = mod.compute_metrics(prose)
        cls, reason = mod.heuristic_classify(metrics)
        assert cls == mod.CLASSIFICATION_KEEP
        assert "narrative" in reason

    def test_borderline_content_returns_none(self, mod):
        """Mixed prose-plus-numbers with varied vocabulary: heuristics should
        abstain so the LLM gets the call."""
        mixed = _borderline_fixture()
        metrics = mod.compute_metrics(mixed)
        cls, reason = mod.heuristic_classify(metrics)
        assert cls is None
        assert reason == "needs_llm_review"

    def test_tiny_content_classifies_as_skip(self, mod):
        """Safety rail: too little text to judge — default to skip."""
        metrics = mod.compute_metrics("Hello world.")
        cls, _ = mod.heuristic_classify(metrics)
        assert cls == mod.CLASSIFICATION_SKIP


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------


class TestLLMClassify:
    """The LLM classifier must accept valid JSON, fall back safely on bad
    shapes, and never raise."""

    @pytest.mark.asyncio
    async def test_valid_json_response_returns_classification(self, mod):
        fake = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"classification": "skip", "reason": "pure telemetry rows"}'
                        )
                    )
                ]
            )
        )
        cls, reason = await mod.llm_classify(["some excerpt"], acompletion_fn=fake)
        assert cls == mod.CLASSIFICATION_SKIP
        assert reason == "pure telemetry rows"

    @pytest.mark.asyncio
    async def test_unknown_classification_falls_back_to_keep(self, mod):
        """Bad classification value must not poison the doc's status — we
        prefer a conservative keep + a traceable reason."""
        fake = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"classification": "uncertain", "reason": "weird"}'
                        )
                    )
                ]
            )
        )
        cls, reason = await mod.llm_classify(["x"], acompletion_fn=fake)
        assert cls == mod.CLASSIFICATION_KEEP
        assert "llm_bad_classification" in reason

    @pytest.mark.asyncio
    async def test_no_json_in_response_falls_back_to_keep(self, mod):
        fake = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="no json here, just text")
                    )
                ]
            )
        )
        cls, reason = await mod.llm_classify(["x"], acompletion_fn=fake)
        assert cls == mod.CLASSIFICATION_KEEP
        assert "llm_no_json" in reason

    @pytest.mark.asyncio
    async def test_exception_falls_back_to_keep(self, mod):
        """Transient API failure can't pull down a batch run."""
        fake = AsyncMock(side_effect=RuntimeError("network"))
        cls, reason = await mod.llm_classify(["x"], acompletion_fn=fake)
        assert cls == mod.CLASSIFICATION_KEEP
        assert "llm_error" in reason
        assert "network" in reason


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


class TestSelectTargets:
    """Excludes emails, image attachments, and docs below the size threshold.
    Must read chunk content through the same doc_id key the ingest writer uses."""

    @staticmethod
    def _seed_db(output, docs, chunks):
        """Write docs + chunks into a fresh ingest.db. Keeps tests tight: we
        only set the columns that ``select_targets`` looks at (id, status,
        document_type, file_name) plus the chunk join keys."""
        from mtss.storage.sqlite_client import SqliteStorageClient

        client = SqliteStorageClient(output_dir=output)
        try:
            conn = client._conn
            with conn:
                conn.execute("BEGIN")
                now = "2026-04-20T00:00:00"
                for d in docs:
                    conn.execute(
                        "INSERT INTO documents("
                        "id, doc_id, source_id, document_type, status, "
                        "file_name, root_id, created_at, updated_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            d["id"], d["id"][:16], d.get("file_name") or d["id"][:16],
                            d.get("document_type"), d.get("status") or "pending",
                            d.get("file_name"), d["id"], now, now,
                        ),
                    )
                for c in chunks:
                    cid = str(uuid4())
                    conn.execute(
                        "INSERT INTO chunks("
                        "id, chunk_id, document_id, content, chunk_index, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?)",
                        (cid, cid, c["document_id"], c["content"],
                         c.get("chunk_index", 0), now),
                    )
        finally:
            conn.close()

    def test_excludes_emails_images_and_small_docs(self, mod, tmp_path):
        output = tmp_path / "output"
        output.mkdir()

        email_id = str(uuid4())
        image_id = str(uuid4())
        small_id = str(uuid4())
        big_pdf_id = str(uuid4())
        big_docx_id = str(uuid4())

        docs = [
            {"id": email_id, "status": "completed", "document_type": "email", "file_name": "a.eml"},
            {"id": image_id, "status": "completed", "document_type": "attachment_image", "file_name": "b.jpg"},
            {"id": small_id, "status": "completed", "document_type": "attachment_pdf", "file_name": "small.pdf"},
            {"id": big_pdf_id, "status": "completed", "document_type": "attachment_pdf", "file_name": "big.pdf"},
            {"id": big_docx_id, "status": "completed", "document_type": "attachment_docx", "file_name": "big.docx"},
        ]

        big_content = "This is realistic prose. " * 2000  # ~50k chars
        chunks = [
            {"document_id": email_id, "chunk_index": 0, "content": "email body"},
            {"document_id": image_id, "chunk_index": 0, "content": "image desc"},
            {"document_id": small_id, "chunk_index": 0, "content": "short"},
            {"document_id": big_pdf_id, "chunk_index": 0, "content": big_content},
            {"document_id": big_docx_id, "chunk_index": 0, "content": big_content},
        ]
        self._seed_db(output, docs, chunks)

        targets = mod.select_targets(output, min_chars=20000)
        target_ids = {d["id"] for d, _ in targets}
        assert target_ids == {big_pdf_id, big_docx_id}

    def test_concatenates_chunks_in_chunk_index_order(self, mod, tmp_path):
        output = tmp_path / "output"
        output.mkdir()

        pdf_id = str(uuid4())
        docs = [{"id": pdf_id, "status": "completed", "document_type": "attachment_pdf", "file_name": "r.pdf"}]
        # Write out-of-order — select_targets must reassemble in chunk_index order.
        chunk_a = "First chunk content. " * 1000
        chunk_b = "Second chunk content. " * 1000
        chunks = [
            {"document_id": pdf_id, "chunk_index": 1, "content": chunk_b},
            {"document_id": pdf_id, "chunk_index": 0, "content": chunk_a},
        ]
        self._seed_db(output, docs, chunks)

        targets = mod.select_targets(output, min_chars=1000)
        assert len(targets) == 1
        _, content = targets[0]
        assert content.index("First chunk content.") < content.index("Second chunk content.")


# ---------------------------------------------------------------------------
# analyse_one integration
# ---------------------------------------------------------------------------


class TestAnalyseOne:
    """Sequence check: heuristic-strong cases short-circuit the LLM;
    borderline cases route to LLM; --no-llm routes borderline to keep with
    source=fallback."""

    @pytest.mark.asyncio
    async def test_strong_heuristic_skips_llm(self, mod):
        import asyncio

        log = "\n".join(
            f"2025-01-{i:02d} ENG {i:04d} RPM {i:04d}" for i in range(1, 500)
        )
        doc = {"id": "d1", "file_name": "log.pdf", "document_type": "attachment_pdf"}

        llm_calls: list = []

        async def _fake(**kwargs):
            llm_calls.append(kwargs)
            raise AssertionError("must not be called on strong heuristic")

        row = await mod.analyse_one(
            doc, log, use_llm=True,
            llm_semaphore=asyncio.Semaphore(1),
            acompletion_fn=_fake,
        )
        assert row.classification == mod.CLASSIFICATION_SKIP
        assert row.source == mod.SOURCE_HEURISTIC
        assert llm_calls == []

    @pytest.mark.asyncio
    async def test_borderline_routes_to_llm(self, mod):
        import asyncio

        mixed = _borderline_fixture()
        doc = {"id": "d2", "file_name": "mixed.pdf", "document_type": "attachment_pdf"}

        fake = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"classification": "summary_only", "reason": "mixed tabular+prose"}'
                        )
                    )
                ]
            )
        )
        row = await mod.analyse_one(
            doc, mixed, use_llm=True,
            llm_semaphore=asyncio.Semaphore(1),
            acompletion_fn=fake,
        )
        assert row.classification == mod.CLASSIFICATION_SUMMARY
        assert row.source == mod.SOURCE_LLM
        fake.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_llm_defaults_borderline_to_keep_with_fallback_source(self, mod):
        import asyncio

        mixed = _borderline_fixture(seed=2)
        doc = {"id": "d3", "file_name": "mixed.pdf", "document_type": "attachment_pdf"}
        row = await mod.analyse_one(
            doc, mixed, use_llm=False,
            llm_semaphore=asyncio.Semaphore(1),
        )
        assert row.classification == mod.CLASSIFICATION_KEEP
        assert row.source == mod.SOURCE_FALLBACK
        assert row.reason == "borderline_no_llm"


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


class TestWriteReport:
    def test_writes_jsonl_atomically(self, mod, tmp_path):
        rows = [
            mod.ReportRow(
                doc_id="a", file_name="a.pdf", document_type="attachment_pdf",
                size_chars=42000, classification=mod.CLASSIFICATION_SKIP,
                source=mod.SOURCE_HEURISTIC, reason="numeric_heavy",
                metrics={"unique_word_ratio": 0.05},
            ),
            mod.ReportRow(
                doc_id="b", file_name="b.pdf", document_type="attachment_pdf",
                size_chars=30000, classification=mod.CLASSIFICATION_KEEP,
                source=mod.SOURCE_LLM, reason="narrative technical content",
                metrics={"unique_word_ratio": 0.44},
            ),
        ]
        output = tmp_path / "output"
        output.mkdir()
        path = mod.write_report(output, rows)
        assert path == output / "rag_utility_report.jsonl"

        lines = [
            json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        assert [r["doc_id"] for r in lines] == ["a", "b"]
        assert lines[0]["classification"] == "skip"
        assert lines[1]["source"] == "llm"
        # .tmp cleaned up
        assert not (output / "rag_utility_report.jsonl.tmp").exists()

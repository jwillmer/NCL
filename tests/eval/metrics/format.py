"""Format-adherence deterministic scoring for RAG eval runs.

Checks the response against the MTSS system prompt's expected structure
(Component / Issue / Resolution Steps / Critical Notes / Related
Incidents headers, or the 'Based on your query' preamble). Also flags
the 'no results' disclaimer substrings and records total response
length. Cheap substring checks only — no parsing, no LLM.
"""

from __future__ import annotations

from typing import List

from ..types import RunResult


# Ordered so a caller can read the list as a ranked preference.
_STRUCTURE_MARKERS: tuple[str, ...] = (
    "Component:",
    "Issue:",
    "Resolution Steps:",
    "Most Relevant Solution",
    "Based on your query",
)

# Case-insensitive substrings that signal a "no results" / empty-index reply.
_NO_RESULTS_MARKERS: tuple[str, ...] = (
    "no relevant",
    "not found in sources",
    "we don't have any records",
    "broader search",
)

# Threshold above which a response is "long enough" to require citations.
_LONG_RESPONSE_CHARS = 400


def _has_no_results_disclaimer(response: str) -> bool:
    lower = response.lower()
    return any(marker in lower for marker in _NO_RESULTS_MARKERS)


def score_format(run: RunResult) -> dict:
    """Score response format adherence.

    Rules:
      - If response has citations, expect at least one structural marker.
      - If response is long (>400 chars) and has no citations and no
        no-results disclaimer, flag 'no citations despite long response'.
      - Structural markers individually missing aren't all failures — only
        the aggregate 'no structural header at all when citations present'
        is reported as 'missing Component header' (the most load-bearing
        marker in the system prompt).
    """
    response = run.response or ""
    length = len(response)
    has_no_results = _has_no_results_disclaimer(response)
    has_citations = bool(run.citations)

    violations: List[str] = []

    present_markers = [m for m in _STRUCTURE_MARKERS if m in response]

    if has_citations and not present_markers:
        violations.append("missing Component header")

    if (
        has_citations
        and "Component:" not in response
        and "Most Relevant Solution" not in response
        and "Based on your query" in response
    ):
        # Preamble present but the solution body wasn't structured.
        violations.append("missing Component header")

    if (
        length > _LONG_RESPONSE_CHARS
        and not has_citations
        and not has_no_results
    ):
        violations.append("no citations despite long response")

    # Dedup while preserving order (a single response could hit the same
    # violation twice via the two branches above).
    seen: set[str] = set()
    deduped: List[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            deduped.append(v)

    follows = not deduped and (has_citations or has_no_results or length == 0)

    return {
        "follows_response_format": follows,
        "format_violations": deduped,
        "has_no_results_disclaimer": has_no_results,
        "response_length_chars": length,
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    from ..types import CitationOccurrence, RunMetrics, RunResult as _RR

    good = _RR(
        question_id="q1",
        question="?",
        response=(
            "Based on your query about 'pump failure', I found 2 incidents.\n\n"
            "**Most Relevant Solution:**\n\n"
            "**Component:** Hydraulic Pump\n"
            "**Issue:** Pressure loss\n\n"
            "**Resolution Steps:**\n"
            "1. Replace seals [C:abc123def456]\n"
        ),
        citations=[
            CitationOccurrence(chunk_id="abc123def456", char_offset=0, is_valid=True)
        ],
        metrics=RunMetrics(latency_ms=1),
    )
    bad = _RR(
        question_id="q2",
        question="?",
        response="x" * 500,
        citations=[],
        metrics=RunMetrics(latency_ms=1),
    )
    no_results = _RR(
        question_id="q3",
        question="?",
        response="We don't have any records matching that query. Want a broader search?",
        citations=[],
        metrics=RunMetrics(latency_ms=1),
    )
    print("good:", score_format(good))
    print("bad:", score_format(bad))
    print("no_results:", score_format(no_results))

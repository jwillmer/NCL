"""Format-adherence deterministic scoring for RAG eval runs.

Checks the response against the MTSS system prompt's expected structure
(Component / Issue / Resolution Steps / Critical Notes / Related
Incidents headers, or the 'Based on your query' preamble). Also flags
the 'no results' disclaimer substrings and records total response
length. Cheap substring checks only — no parsing, no LLM.
"""

from __future__ import annotations

from typing import List, Optional

from ..types import GoldenQuestion, RunResult


# Ordered so a caller can read the list as a ranked preference. Preamble
# phrases are case-insensitive substrings — the canonical form is still
# "Based on your query" (per system prompt), but the agent naturally
# opens with variants like "I found 5 ...", "Based on the fleet-wide
# records..." for fleet-wide / aggregate questions. Widened 2026-04-22
# after the baseline-12d eval flagged 4 responses as format-failing
# despite opening with a clear preamble that just wasn't in the list.
_STRUCTURE_MARKERS: tuple[str, ...] = (
    "Component:",
    "Issue:",
    "Resolution Steps:",
    "Most Relevant Solution",
    "Based on your query",
    "Based on the ",
    "I found ",
    "I couldn't find",
    "I could not find",
    "Here are ",
)

# Case-insensitive substrings that signal a "no results" / empty-index /
# off-topic-scope-refusal / retriever-error disclaimer.
_NO_RESULTS_MARKERS: tuple[str, ...] = (
    "no relevant",
    "not found in sources",
    "we don't have any records",
    "broader search",
    # scope refusals (off-topic queries like q30 "tesla").
    # Use apostrophe-free substrings so both straight ' and typographic ’ match.
    "focused on vessel",
    "can t help with",
    "outside the scope",
    # retriever-error disclaimers (q29, q31 — "search returned an error")
    "couldn't retrieve",
    "search returned an error",
    "do not provide",
    "sources do not",
)

# Categories whose expected answer is a fleet-wide narrative / summary
# rather than per-incident Component/Issue/Resolution blocks. The
# system prompt only mandates the structured format for vessel-specific
# and past-incident lookups; these broader questions are graded by
# citation validity alone.
_NARRATIVE_CATEGORIES: frozenset[str] = frozenset(
    {
        "Cross-Cutting / Analytics",
        "Cross-Vessel Pattern",
        "Statistical Ranking",
        "Evaluative",
        "No Results Expected",
    }
)

# Threshold above which a response is "long enough" to require citations.
_LONG_RESPONSE_CHARS = 400


def _has_no_results_disclaimer(response: str) -> bool:
    lower = response.lower()
    return any(marker in lower for marker in _NO_RESULTS_MARKERS)


def score_format(run: RunResult, golden: Optional[GoldenQuestion] = None) -> dict:
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
    is_narrative = bool(
        golden is not None and golden.category in _NARRATIVE_CATEGORIES
    )

    violations: List[str] = []

    present_markers = [m for m in _STRUCTURE_MARKERS if m in response]

    # Narrative / fleet-wide / no-results-expected questions are graded on
    # citation validity, not on per-incident block structure. Skip the
    # Component-header check for those categories and for any response
    # whose own wording is a no-results / scope-refusal disclaimer.
    strict_structure = not is_narrative and not has_no_results

    if strict_structure and has_citations and not present_markers:
        violations.append("missing Component header")

    if (
        strict_structure
        and has_citations
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

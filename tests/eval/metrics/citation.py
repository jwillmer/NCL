"""Citation-level deterministic scoring for RAG eval runs.

Grades how well the agent's response cites its retrieved chunks. Computes
counts (total, valid, valid percentage) from the already-extracted
`run.citations`, then measures grounding by checking TF-IDF cosine
similarity between each cited chunk's text_preview and the sentence that
surrounds the [C:...] marker in the response. No LLM, no network.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from ..types import RetrievedChunk, RunResult

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only when sklearn missing
    _HAS_SKLEARN = False
    logger.warning(
        "sklearn not available; citation grounding will fall back to Jaccard similarity."
    )


# Sentence splitter: break on . ! ? (followed by whitespace/end), keep it simple
# and deterministic. Multi-char terminators ("...", "!!") collapse to one split.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_MARKER_RE = re.compile(r"\[C:[a-f0-9]+\]")


def _extract_sentence_around(text: str, char_offset: int) -> str:
    """Return the sentence that contains `char_offset` in `text`.

    Splits on . ! ? followed by whitespace. Returns the matching segment with
    all [C:...] markers stripped (so they don't skew the similarity score).
    """
    if not text:
        return ""
    # Find sentence boundaries — offsets of each boundary in the original text
    offsets: List[Tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        offsets.append((start, m.start()))
        start = m.end()
    offsets.append((start, len(text)))

    for s, e in offsets:
        if s <= char_offset < e:
            sentence = text[s:e]
            break
    else:
        sentence = text

    # Strip citation markers so they don't contribute to similarity tokens
    return _CITATION_MARKER_RE.sub(" ", sentence).strip()


def _jaccard_similarity(a: str, b: str) -> float:
    """Fallback similarity: Jaccard on lowercased word sets."""
    wa = {w for w in re.findall(r"\w+", a.lower()) if w}
    wb = {w for w in re.findall(r"\w+", b.lower()) if w}
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    union = wa | wb
    return len(inter) / len(union) if union else 0.0


def _tfidf_cosine(a: str, b: str) -> float:
    """TF-IDF cosine between two strings. Returns 0.0 if either is empty or
    the vectorizer finds no shared vocabulary."""
    if not a.strip() or not b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
    except ValueError:
        # Raised when documents contain only stop words / no tokens
        return 0.0
    sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    # Clamp — cosine on non-negative TF-IDF vectors is in [0, 1] but guard anyway.
    return float(max(0.0, min(1.0, sim)))


def _similarity(a: str, b: str) -> float:
    if _HAS_SKLEARN:
        return _tfidf_cosine(a, b)
    return _jaccard_similarity(a, b)


def _build_chunk_index(chunks: List[RetrievedChunk]) -> Dict[str, RetrievedChunk]:
    return {c.chunk_id: c for c in chunks}


def score_citations(run: RunResult) -> dict:
    """Score citation counts + grounding for one run.

    Returns a dict with:
        citations_count, citations_valid_count, citations_valid_pct,
        citation_grounding_score

    `citation_grounding_score` is the mean TF-IDF cosine similarity between
    the cited chunk's `text_preview` and the surrounding sentence in the
    response. 0.0 if there are no valid citations.
    """
    total = len(run.citations)
    valid = [c for c in run.citations if c.is_valid]
    valid_count = len(valid)
    valid_pct = (valid_count / total) if total else 0.0

    if not valid:
        return {
            "citations_count": total,
            "citations_valid_count": 0,
            "citations_valid_pct": valid_pct,
            "citation_grounding_score": 0.0,
        }

    chunk_index = _build_chunk_index(run.retrieval)
    sims: List[float] = []
    for occ in valid:
        chunk = chunk_index.get(occ.chunk_id)
        if chunk is None or not chunk.text_preview:
            # Valid per citation_map at run time, but chunk not in this run's
            # retrieval (shouldn't happen) or preview was blank — grounding = 0.
            sims.append(0.0)
            continue
        sentence = _extract_sentence_around(run.response, occ.char_offset)
        sims.append(_similarity(sentence, chunk.text_preview))

    grounding = sum(sims) / len(sims) if sims else 0.0

    return {
        "citations_count": total,
        "citations_valid_count": valid_count,
        "citations_valid_pct": valid_pct,
        "citation_grounding_score": grounding,
    }


if __name__ == "__main__":  # pragma: no cover - smoke test
    from ..types import CitationOccurrence, RunMetrics, RunResult as _RR

    response = (
        "The hydraulic pump lost pressure during cargo ops [C:8f3a2b1c4d5e]. "
        "The maintenance team replaced the seals [C:9a4b3c2d1e6f]."
    )
    run = _RR(
        question_id="q1",
        question="hydraulic pump?",
        response=response,
        retrieval=[
            RetrievedChunk(
                rank=1,
                chunk_id="8f3a2b1c4d5e",
                doc_id="d1",
                score=0.9,
                text_preview="hydraulic pump lost pressure during cargo operations",
            ),
            RetrievedChunk(
                rank=2,
                chunk_id="9a4b3c2d1e6f",
                doc_id="d2",
                score=0.8,
                text_preview="maintenance team replaced worn seals with new OEM seals",
            ),
        ],
        citations=[
            CitationOccurrence(
                chunk_id="8f3a2b1c4d5e",
                char_offset=response.index("[C:8f3a2b1c4d5e]"),
                is_valid=True,
            ),
            CitationOccurrence(
                chunk_id="9a4b3c2d1e6f",
                char_offset=response.index("[C:9a4b3c2d1e6f]"),
                is_valid=True,
            ),
        ],
        metrics=RunMetrics(latency_ms=100),
    )
    print(score_citations(run))

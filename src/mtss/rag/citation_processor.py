"""Citation processing for RAG responses.

Builds citation-aware context for LLM, validates citations in responses,
and formats final output with verified source links.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set

from ..models.chunk import (
    CitationValidationResult,
    RetrievalResult,
    ValidatedCitation,
)
from ..storage.archive_storage import ArchiveStorage
from ..utils import CHUNK_ID_LENGTH

logger = logging.getLogger(__name__)


class CitationProcessor:
    """Build context, format citations, and validate LLM responses."""

    # Pattern to extract citations from LLM response: [C:chunk_id]
    CITATION_PATTERN = re.compile(r"\[C:([a-f0-9]+)\]")

    # Maximum ratio of invalid citations before triggering retry
    MAX_INVALID_RATIO = 0.5

    # System prompt with citation rules
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

CITATION RULES (MANDATORY):
1. When stating a fact from the context, append a citation using the 12-character chunk_id: [C:chunk_id]
2. You may cite multiple chunks: [C:8f3a2b1c4d5e][C:9a4b3c2d1e6f]
3. If information is NOT in the context, say "Not found in sources"
4. Never invent or guess information not in the context
5. Use ONLY the chunk_id from the CITE: field in context headers (exactly 12 hex characters)

Example response:
"The GPS system requires calibration before each flight [C:8f3a2b1c4d5e]. This was confirmed by the maintenance team on Tuesday [C:9a4b3c2d1e6f]."
"""

    def __init__(self):
        """Initialize citation processor."""
        self.storage = ArchiveStorage()
        # Per-response cache: most answers cite the same folder many times
        # (e.g. 16 citations from 4 docs). Check the bucket once per URI.
        self._archive_exists_cache: Dict[str, bool] = {}

    def get_system_prompt(self) -> str:
        """Get system prompt with citation rules."""
        return self.SYSTEM_PROMPT

    def build_context(self, results: List[RetrievalResult]) -> str:
        """Build LLM context with citation headers from retrieval results.

        Args:
            results: List of retrieval results with citation metadata.

        Returns:
            Formatted context string for LLM.
        """
        context_blocks = []

        for result in results:
            header = self._build_citation_header(result)
            text = result.text
            if result.context_summary:
                text = f"[Context: {result.context_summary}]\n{text}"
            block = f"{header}\n{text}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def build_context_hybrid(self, results: List[RetrievalResult]) -> str:
        """Like :meth:`build_context`, but skips full-text for summary-mode chunks.

        Phase-1 measurement candidate (see ``scripts/measure_context_summary_swap.py``).
        For chunks whose ``embedding_mode == "summary"`` AND that carry a
        ``context_summary``, emit only ``[Context: <summary>]`` for that
        block — the full text is sensor-log / dense-tabular noise that
        the synthesizer cannot use anyway. All other modes (``full``,
        ``metadata_only``, or unknown/None) behave exactly like
        :meth:`build_context`.

        This method is NOT wired into the production agent; both builders
        coexist until the measurement harness produces numbers that
        justify a swap.
        """
        context_blocks = []

        for result in results:
            header = self._build_citation_header(result)
            mode = result.embedding_mode
            if mode == "summary" and result.context_summary:
                # Summary mode: only the synthesized summary is meaningful;
                # skip ``result.text`` entirely.
                text = f"[Context: {result.context_summary}]"
            else:
                text = result.text
                if result.context_summary:
                    text = f"[Context: {result.context_summary}]\n{text}"
            block = f"{header}\n{text}"
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def _build_citation_header(self, result: RetrievalResult) -> str:
        """Generate citation header for a retrieval result.

        Only includes the chunk_id (as CITE:) to avoid LLM confusion with
        other ID-like values. Additional metadata is provided for context
        but not in a format that could be mistaken for a citation ID.

        Args:
            result: Retrieval result with metadata.

        Returns:
            Formatted header string.
        """
        # Primary: chunk_id for citations (this is what the LLM should use)
        parts = [f"CITE:{result.chunk_id}"]

        # Secondary: human-readable context (not ID-like to avoid confusion)
        if result.source_title:
            # Truncate long titles
            title = result.source_title[:50]
            if len(result.source_title) > 50:
                title += "..."
            parts.append(f'title:"{title}"')

        # Email metadata for incident identification
        if result.email_subject:
            subject = result.email_subject[:40]
            if len(result.email_subject) > 40:
                subject += "..."
            parts.append(f'subject:"{subject}"')

        if result.email_date:
            parts.append(f"date:{result.email_date}")

        if result.email_initiator:
            parts.append(f"from:{result.email_initiator}")

        if result.page_number:
            parts.append(f"page:{result.page_number}")

        # Include image URI for image attachments so agent can embed them
        if result.image_uri:
            parts.append(f"img:{result.image_uri}")

        return f"[{' | '.join(parts)}]"

    def get_citation_map(
        self, results: List[RetrievalResult]
    ) -> Dict[str, RetrievalResult]:
        """Create mapping from chunk_id to retrieval result.

        Args:
            results: List of retrieval results.

        Returns:
            Dict mapping chunk_id to RetrievalResult.
        """
        return {r.chunk_id: r for r in results}

    def verify_archive_exists(self, archive_uri: Optional[str]) -> bool:
        """Check if the archive file exists in Supabase Storage.

        Args:
            archive_uri: Relative URI to archive file.

        Returns:
            True if file exists, False otherwise.
        """
        if not archive_uri:
            return False

        cached = self._archive_exists_cache.get(archive_uri)
        if cached is not None:
            return cached
        exists = self.storage.file_exists(archive_uri)
        self._archive_exists_cache[archive_uri] = exists
        return exists

    @staticmethod
    def _clean_response_text(response: str, invalid_citations: List[str]) -> str:
        """Strip invalid markers and collapse whitespace. Pure, no I/O."""
        cleaned = response
        for invalid_id in invalid_citations:
            cleaned = cleaned.replace(f"[C:{invalid_id}]", "")
        # Collapse runs of horizontal whitespace; preserve newlines.
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        # Drop trailing horizontal whitespace before a newline.
        cleaned = re.sub(r" +\n", "\n", cleaned)
        return cleaned.strip()

    def _collect_citations(
        self,
        response: str,
        citation_map: Dict[str, RetrievalResult],
    ) -> Dict[str, Any]:
        """Build provisional citations + validity bookkeeping with no I/O.

        Returns a dict with keys: ``citations`` (list of ValidatedCitation
        with ``archive_verified=False``), ``invalid_citations`` (list of bad
        chunk_ids), ``found`` (list of every chunk_id seen in the text),
        and ``unique_uris`` (set of distinct archive_browse_uris that need
        verification). Archive existence is *not* checked here — callers
        run :meth:`verify_archives_async` (or :meth:`verify_archive_exists`
        per-uri) and patch the ``archive_verified`` field afterwards.
        """
        found_citations = self.CITATION_PATTERN.findall(response)
        if not found_citations:
            return {
                "citations": [],
                "invalid_citations": [],
                "found": [],
                "unique_uris": set(),
            }

        provisional: List[ValidatedCitation] = []
        invalid_citations: List[str] = []
        seen_chunk_ids: Set[str] = set()
        unique_uris: Set[str] = set()
        document_indices: Dict[str, int] = {}
        next_document_index = 1

        for chunk_id in found_citations:
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            if len(chunk_id) != CHUNK_ID_LENGTH:
                logger.warning(
                    "Rejecting invalid chunk_id: '%s' (%d chars, expected %d)",
                    chunk_id, len(chunk_id), CHUNK_ID_LENGTH,
                )
                invalid_citations.append(chunk_id)
                continue
            if chunk_id not in citation_map:
                invalid_citations.append(chunk_id)
                continue

            result = citation_map[chunk_id]
            source_title = result.source_title or "Unknown Source"
            if source_title not in document_indices:
                document_indices[source_title] = next_document_index
                next_document_index += 1
            doc_index = document_indices[source_title]

            lines = None
            if result.line_from is not None and result.line_to is not None:
                lines = (result.line_from, result.line_to)

            if result.archive_browse_uri:
                unique_uris.add(result.archive_browse_uri)

            provisional.append(
                ValidatedCitation(
                    index=doc_index,
                    chunk_id=chunk_id,
                    source_title=result.source_title,
                    source_id=result.source_id or None,
                    page=result.page_number,
                    lines=lines,
                    archive_browse_uri=result.archive_browse_uri,
                    archive_download_uri=result.archive_download_uri,
                    archive_verified=False,
                )
            )

        return {
            "citations": provisional,
            "invalid_citations": invalid_citations,
            "found": found_citations,
            "unique_uris": unique_uris,
        }

    async def verify_archives_async(self, uris: List[str]) -> Dict[str, bool]:
        """Verify N archive URIs in parallel via ``asyncio.to_thread``.

        Each unique URI is checked once and the result populates the per-
        instance cache, so subsequent sync calls to
        :meth:`verify_archive_exists` for the same URI return immediately.
        Network HEAD calls dominate the wall-clock cost; running them
        concurrently turns N×~100 ms into ~max(times) instead of sum.
        """
        results: Dict[str, bool] = {}
        if not uris:
            return results
        # Reuse cached entries; only fetch the rest.
        uncached: List[str] = []
        for uri in uris:
            cached = self._archive_exists_cache.get(uri)
            if cached is None:
                uncached.append(uri)
            else:
                results[uri] = cached
        if uncached:
            fetched = await asyncio.gather(
                *(asyncio.to_thread(self.storage.file_exists, u) for u in uncached),
                return_exceptions=True,
            )
            for uri, val in zip(uncached, fetched):
                if isinstance(val, Exception):
                    logger.warning("verify_archives_async: %s failed (%s)", uri, val)
                    val = False
                self._archive_exists_cache[uri] = val
                results[uri] = val
        return results

    def _build_validation_result(
        self,
        response: str,
        collected: Dict[str, Any],
        archive_verified_by_uri: Dict[str, bool],
    ) -> CitationValidationResult:
        """Combine collected citations + archive-verification map into a
        :class:`CitationValidationResult`. Pure.
        """
        verified_citations: List[ValidatedCitation] = []
        missing_archives: List[str] = []
        for c in collected["citations"]:
            verified = (
                archive_verified_by_uri.get(c.archive_browse_uri, False)
                if c.archive_browse_uri else False
            )
            if c.archive_browse_uri and not verified:
                missing_archives.append(c.chunk_id)
            verified_citations.append(
                ValidatedCitation(
                    index=c.index,
                    chunk_id=c.chunk_id,
                    source_title=c.source_title,
                    source_id=c.source_id,
                    page=c.page,
                    lines=c.lines,
                    archive_browse_uri=c.archive_browse_uri,
                    archive_download_uri=c.archive_download_uri,
                    archive_verified=verified,
                )
            )

        invalid_citations = collected["invalid_citations"]
        cleaned_response = self._clean_response_text(response, invalid_citations)

        total_citations = len(collected["found"])
        invalid_count = len(invalid_citations)
        needs_retry = (
            invalid_count > 0
            and total_citations > 0
            and (invalid_count / total_citations) > self.MAX_INVALID_RATIO
        )
        return CitationValidationResult(
            response=cleaned_response,
            citations=verified_citations,
            invalid_citations=invalid_citations,
            missing_archives=missing_archives,
            needs_retry=needs_retry,
        )

    async def process_response_async(
        self,
        response: str,
        citation_map: Dict[str, RetrievalResult],
    ) -> CitationValidationResult:
        """Async variant of :meth:`process_response`.

        Verifies archive existence in parallel rather than sequentially.
        Use this on the user-reply hot path; the sync method is retained
        for callers that don't have an event loop handy.
        """
        collected = self._collect_citations(response, citation_map)
        if not collected["found"]:
            return CitationValidationResult(
                response=response,
                citations=[],
                invalid_citations=[],
                missing_archives=[],
                needs_retry=False,
            )
        archive_verified_by_uri = await self.verify_archives_async(
            sorted(collected["unique_uris"])
        )
        return self._build_validation_result(
            response, collected, archive_verified_by_uri
        )

    def process_response(
        self,
        response: str,
        citation_map: Dict[str, RetrievalResult],
    ) -> CitationValidationResult:
        """Validate citations in LLM response and build validation result.

        Sync variant; verifies archives serially. For the streaming RAG
        reply path, prefer :meth:`process_response_async` — same contract,
        ~N× faster on responses citing several documents.
        """
        collected = self._collect_citations(response, citation_map)
        if not collected["found"]:
            return CitationValidationResult(
                response=response,
                citations=[],
                invalid_citations=[],
                missing_archives=[],
                needs_retry=False,
            )
        archive_verified_by_uri = {
            uri: self.verify_archive_exists(uri) for uri in collected["unique_uris"]
        }
        return self._build_validation_result(
            response, collected, archive_verified_by_uri
        )

    @staticmethod
    def serialize_citations_payload(
        validation: CitationValidationResult,
    ) -> Dict[str, Any]:
        """Build the v1 wire payload for the ``data-citations`` SSE frame.

        The frontend uses this to swap ``[C:chunk_id]`` markers for full
        ``<cite>`` tags after the LLM stream completes. Field set must
        stay aligned with :meth:`replace_citation_markers` — the frontend
        builds the same tag attributes from this dict.
        """
        citations: List[Dict[str, Any]] = []
        for c in validation.citations:
            citations.append(
                {
                    "chunk_id": c.chunk_id,
                    "index": c.index,
                    "source_id": c.source_id,
                    "source_title": c.source_title,
                    "page": c.page,
                    "lines": list(c.lines) if c.lines else None,
                    "archive_browse_uri": c.archive_browse_uri,
                    "archive_download_uri": c.archive_download_uri,
                    "archive_verified": c.archive_verified,
                }
            )
        return {
            "version": 1,
            "citations": citations,
            "invalid_chunk_ids": list(validation.invalid_citations),
        }

    def replace_citation_markers(
        self,
        response: str,
        citations: List[ValidatedCitation],
    ) -> str:
        """Replace [C:chunk_id] markers with <cite> tags containing metadata.

        Output format: <cite id="chunk_id" title="Source Title" page="5">1</cite>

        These tags are rendered by the frontend via CopilotKit's markdownTagRenderers.

        Args:
            response: Response text with [C:chunk_id] markers.
            citations: List of validated citations with index numbers.

        Returns:
            Response with <cite> tags containing citation metadata.
        """
        result = response

        for c in citations:
            old_marker = f"[C:{c.chunk_id}]"

            # Build attributes for the cite tag
            attrs = [f'id="{c.chunk_id}"']

            if c.source_id:
                # Stable per-document identity — the UI dedupes chunks of
                # the same source by this attribute so a 4-chunk hit on
                # one report renders as one source card, not four.
                attrs.append(f'doc="{c.source_id}"')

            if c.source_title:
                # Escape quotes in title for HTML attribute
                safe_title = c.source_title.replace('"', "&quot;")
                attrs.append(f'title="{safe_title}"')

            if c.page:
                attrs.append(f'page="{c.page}"')

            if c.lines:
                attrs.append(f'lines="{c.lines[0]}-{c.lines[1]}"')

            if c.archive_download_uri:
                dl = c.archive_download_uri.removeprefix("/archive/")
                attrs.append(f'download="{dl}"')

            new_marker = f'<cite {" ".join(attrs)}>{c.index}</cite>'
            result = result.replace(old_marker, new_marker)

        return result

"""Citation processing for RAG responses.

Builds citation-aware context for LLM, validates citations in responses,
and formats final output with verified source links.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

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
            block = f"{header}\n{result.text}"
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

        return self.storage.file_exists(archive_uri)

    def process_response(
        self,
        response: str,
        citation_map: Dict[str, RetrievalResult],
    ) -> CitationValidationResult:
        """Validate citations in LLM response and build validation result.

        Args:
            response: Raw LLM response text.
            citation_map: Mapping from chunk_id to RetrievalResult.

        Returns:
            CitationValidationResult with validated citations and cleaned response.
        """
        # Find all citations in response
        found_citations = self.CITATION_PATTERN.findall(response)

        if not found_citations:
            return CitationValidationResult(
                response=response,
                citations=[],
                invalid_citations=[],
                missing_archives=[],
                needs_retry=False,
            )

        valid_citations: List[ValidatedCitation] = []
        invalid_citations: List[str] = []
        missing_archives: List[str] = []
        seen_chunk_ids: Set[str] = set()

        # Track document indices by source_title for consolidation
        # All citations from the same document share the same display index
        document_indices: Dict[str, int] = {}
        next_document_index = 1

        # Process each citation
        for chunk_id in found_citations:
            # Skip duplicate chunk_ids (same chunk cited multiple times)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            # Strict validation: chunk_id must be exactly 12 hex chars
            if len(chunk_id) != CHUNK_ID_LENGTH:
                logger.warning(
                    "Rejecting invalid chunk_id: '%s' (%d chars, expected %d)",
                    chunk_id,
                    len(chunk_id),
                    CHUNK_ID_LENGTH,
                )
                invalid_citations.append(chunk_id)
                continue

            # Check if chunk_id exists in retrieved results
            if chunk_id not in citation_map:
                invalid_citations.append(chunk_id)
                continue

            result = citation_map[chunk_id]

            # Check if archive exists
            archive_verified = self.verify_archive_exists(result.archive_browse_uri)
            if result.archive_browse_uri and not archive_verified:
                missing_archives.append(chunk_id)

            # Assign document index - same document gets same index
            source_title = result.source_title or "Unknown Source"
            if source_title not in document_indices:
                document_indices[source_title] = next_document_index
                next_document_index += 1
            doc_index = document_indices[source_title]

            # Build validated citation
            lines = None
            if result.line_from is not None and result.line_to is not None:
                lines = (result.line_from, result.line_to)

            valid_citations.append(
                ValidatedCitation(
                    index=doc_index,  # Shared index per document
                    chunk_id=chunk_id,
                    source_title=result.source_title,
                    page=result.page_number,
                    lines=lines,
                    archive_browse_uri=result.archive_browse_uri,
                    archive_download_uri=result.archive_download_uri,
                    archive_verified=archive_verified,
                )
            )

        # Remove invalid citations from response text
        cleaned_response = response
        for invalid_id in invalid_citations:
            cleaned_response = cleaned_response.replace(f"[C:{invalid_id}]", "")

        # Clean up any double spaces (but preserve newlines for formatting)
        cleaned_response = re.sub(r"[^\S\n]+", " ", cleaned_response)
        cleaned_response = re.sub(r" +\n", "\n", cleaned_response)  # Remove trailing spaces before newlines
        cleaned_response = cleaned_response.strip()

        # Determine if retry is needed
        total_citations = len(found_citations)
        invalid_count = len(invalid_citations)
        needs_retry = (
            invalid_count > 0
            and total_citations > 0
            and (invalid_count / total_citations) > self.MAX_INVALID_RATIO
        )

        return CitationValidationResult(
            response=cleaned_response,
            citations=valid_citations,
            invalid_citations=invalid_citations,
            missing_archives=missing_archives,
            needs_retry=needs_retry,
        )

    def build_retry_hint(self, valid_chunk_ids: List[str]) -> str:
        """Build a hint for retry prompts listing valid chunk IDs.

        Args:
            valid_chunk_ids: List of chunk IDs that are valid for citation.

        Returns:
            Hint string to add to retry prompt.
        """
        ids_str = ", ".join(valid_chunk_ids[:20])  # Limit to avoid token bloat
        return f"\n\nIMPORTANT: Only use these chunk IDs for citations: {ids_str}"

    def format_sources_section(
        self, citations: List[ValidatedCitation]
    ) -> str:
        """Format citations as a sources section for display.

        Args:
            citations: List of validated citations.

        Returns:
            Formatted sources section string.
        """
        if not citations:
            return ""

        lines = ["", "**Sources:**"]

        for citation in citations:
            line_parts = [f"[{citation.index}]"]

            if citation.source_title:
                line_parts.append(citation.source_title)

            if citation.page:
                line_parts.append(f"p.{citation.page}")

            if citation.lines:
                line_parts.append(f"lines {citation.lines[0]}-{citation.lines[1]}")

            # Add links
            links = []
            if citation.archive_browse_uri:
                links.append(f"[View](archive/{citation.archive_browse_uri})")
            if citation.archive_download_uri:
                links.append(f"[Download](archive/{citation.archive_download_uri})")

            if links:
                line_parts.append(" | ".join(links))

            lines.append(" | ".join(line_parts))

        return "\n".join(lines)

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

            if c.source_title:
                # Escape quotes in title for HTML attribute
                safe_title = c.source_title.replace('"', "&quot;")
                attrs.append(f'title="{safe_title}"')

            if c.page:
                attrs.append(f'page="{c.page}"')

            if c.lines:
                attrs.append(f'lines="{c.lines[0]}-{c.lines[1]}"')

            if c.archive_download_uri:
                attrs.append(f'download="{c.archive_download_uri}"')

            new_marker = f'<cite {" ".join(attrs)}>{c.index}</cite>'
            result = result.replace(old_marker, new_marker)

        return result

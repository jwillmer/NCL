"""Citation processing for RAG responses.

Builds citation-aware context for LLM, validates citations in responses,
and formats final output with verified source links.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..config import get_settings
from ..models.chunk import (
    CitationValidationResult,
    RetrievalResult,
    ValidatedCitation,
)

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
1. When stating a fact from the context, append a citation: [C:chunk_id]
2. You may cite multiple chunks: [C:abc123][C:def456]
3. If information is NOT in the context, say "Not found in sources"
4. Never invent or guess information not in the context
5. Use the chunk_id provided in each context block's header

Example response:
"The GPS system requires calibration before each flight [C:8f3a2b1c]. This was confirmed by the maintenance team on Tuesday [C:9a4b3c2d]."
"""

    def __init__(self, archive_dir: Optional[Path] = None):
        """Initialize citation processor.

        Args:
            archive_dir: Directory where archive files are stored for verification.
        """
        settings = get_settings()
        self.archive_dir = archive_dir or settings.archive_dir

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
        """Generate deterministic citation header for a retrieval result.

        Args:
            result: Retrieval result with metadata.

        Returns:
            Formatted header string.
        """
        parts = [
            f"S:{result.source_id[:8] if result.source_id else 'unknown'}",
            f"D:{result.doc_id[:8] if result.doc_id else 'unknown'}",
            f"C:{result.chunk_id}",
        ]

        if result.page_number:
            parts.append(f"p:{result.page_number}")

        if result.source_title:
            # Truncate long titles
            title = result.source_title[:50]
            if len(result.source_title) > 50:
                title += "..."
            parts.append(f'title:"{title}"')

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
        """Check if the archive file actually exists on disk.

        Args:
            archive_uri: Relative URI to archive file.

        Returns:
            True if file exists, False otherwise.
        """
        if not archive_uri:
            return False

        archive_path = self.archive_dir / archive_uri
        return archive_path.exists()

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

        # Process each citation
        citation_index = 1
        for chunk_id in found_citations:
            # Skip duplicates
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            # Check if chunk_id exists in retrieved results
            if chunk_id not in citation_map:
                invalid_citations.append(chunk_id)
                continue

            result = citation_map[chunk_id]

            # Check if archive exists
            archive_verified = self.verify_archive_exists(result.archive_browse_uri)
            if result.archive_browse_uri and not archive_verified:
                missing_archives.append(chunk_id)

            # Build validated citation
            lines = None
            if result.line_from is not None and result.line_to is not None:
                lines = (result.line_from, result.line_to)

            valid_citations.append(
                ValidatedCitation(
                    index=citation_index,
                    chunk_id=chunk_id,
                    source_title=result.source_title,
                    page=result.page_number,
                    lines=lines,
                    archive_browse_uri=result.archive_browse_uri,
                    archive_download_uri=result.archive_download_uri,
                    archive_verified=archive_verified,
                )
            )
            citation_index += 1

        # Remove invalid citations from response text
        cleaned_response = response
        for invalid_id in invalid_citations:
            cleaned_response = cleaned_response.replace(f"[C:{invalid_id}]", "")

        # Clean up any double spaces or orphaned punctuation
        cleaned_response = re.sub(r"\s+", " ", cleaned_response)
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
        """Replace [C:chunk_id] markers with numbered references [1], [2], etc.

        Args:
            response: Response text with [C:chunk_id] markers.
            citations: List of validated citations with index numbers.

        Returns:
            Response with numbered citation markers.
        """
        result = response

        # Build mapping from chunk_id to citation index
        id_to_index = {c.chunk_id: c.index for c in citations}

        # Replace each citation marker
        for chunk_id, index in id_to_index.items():
            result = result.replace(f"[C:{chunk_id}]", f"[{index}]")

        return result

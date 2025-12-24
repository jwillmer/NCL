"""Text chunking using LangChain text splitters."""

from __future__ import annotations

import logging
import re
from typing import List, Optional
from uuid import UUID

import tiktoken
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import get_settings
from ..models.chunk import Chunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk documents using LangChain text splitters.

    Uses MarkdownTextSplitter for markdown content and
    RecursiveCharacterTextSplitter for plain text.
    Token counting uses tiktoken with the same encoding as embeddings.
    """

    def __init__(self):
        """Initialize chunker with settings."""
        settings = get_settings()
        self.chunk_size = settings.chunk_size_tokens
        self.chunk_overlap = settings.chunk_overlap_tokens

        # Get tiktoken encoding for the embedding model
        model_name = settings.embedding_model
        if model_name.startswith("text-embedding-"):
            self._encoding = tiktoken.get_encoding("cl100k_base")
        else:
            try:
                self._encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self._encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize splitters
        self._markdown_splitter = MarkdownTextSplitter.from_tiktoken_encoder(
            encoding_name=self._encoding.name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self._encoding.name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(
        self,
        text: str,
        document_id: UUID,
        source_file: str,
        is_markdown: bool = True,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Text content to chunk.
            document_id: UUID of the parent document.
            source_file: Path to source file for metadata.
            is_markdown: Whether to use markdown-aware splitting.
            metadata: Additional metadata to include in chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        # Choose splitter based on content type
        splitter = self._markdown_splitter if is_markdown else self._text_splitter

        # Split the text
        splits = splitter.split_text(text)

        # Convert to Chunk objects
        chunks = []
        base_metadata = metadata.copy() if metadata else {}
        base_metadata["source_file"] = source_file

        for idx, content in enumerate(splits):
            # Extract heading context from markdown content
            heading_path = self._extract_heading_path(content) if is_markdown else []

            chunk = Chunk(
                document_id=document_id,
                content=content,
                chunk_index=idx,
                heading_path=heading_path,
                section_title=heading_path[-1] if heading_path else None,
                metadata=base_metadata.copy(),
            )
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    def _extract_heading_path(self, content: str) -> List[str]:
        """Extract heading hierarchy from chunk content.

        Finds markdown headings and builds a hierarchy path.

        Args:
            content: Chunk text content.

        Returns:
            List of heading strings from hierarchy.
        """
        headings = []
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        current_levels: dict[int, str] = {}

        for match in heading_pattern.finditer(content):
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Update the heading at this level
            current_levels[level] = heading_text

            # Clear any deeper levels
            for deeper_level in list(current_levels.keys()):
                if deeper_level > level:
                    del current_levels[deeper_level]

        # Build path from remaining headings
        if current_levels:
            for level in sorted(current_levels.keys()):
                headings.append(current_levels[level])

        return headings

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        return len(self._encoding.encode(text))

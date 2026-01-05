"""Text chunking using LangChain text splitters with contextual embeddings."""

from __future__ import annotations

import logging
import re
from typing import List, Optional
from uuid import UUID

import litellm
import tiktoken

# Drop unsupported parameters for models that don't support them (e.g., GPT-5 temperature)
litellm.drop_params = True
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ..config import get_settings
from ..models.chunk import Chunk
from ..models.document import Document

logger = logging.getLogger(__name__)


class ContextGenerator:
    """Generate document-level context summaries for improved retrieval.

    Creates 2-3 sentence summaries of documents that are prepended to each
    chunk before embedding. This technique (contextual chunking) improves
    retrieval accuracy by 35-67% according to Anthropic research.

    Uses a two-tier model approach:
    - Primary model for most content (fast/cheap)
    - Fallback model with larger context for long content
    """

    CONTEXT_PROMPT = """Summarize this document in 2-3 sentences for context.
Include: document type, author/source if available, date if available, and main topic.
Be concise and factual.

Document:
{content}"""

    def __init__(self):
        """Initialize context generator with settings."""
        settings = get_settings()
        self.model = settings.get_model(settings.context_llm_model)
        self.fallback_model = settings.get_model(settings.context_llm_fallback)
        self.max_input_tokens = settings.context_llm_max_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens.

        Returns:
            Truncated text.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def _build_prompt_content(
        self, document: Document, content_preview: str, max_preview_chars: int
    ) -> str:
        """Build the content portion of the prompt with metadata hints.

        Args:
            document: Document object with metadata.
            content_preview: Text content to summarize.
            max_preview_chars: Maximum characters for preview.

        Returns:
            Formatted content string with hints.
        """
        hints = []
        if document.document_type:
            hints.append(f"Type: {document.document_type.value}")
        if document.email_metadata:
            if document.email_metadata.subject:
                hints.append(f"Subject: {document.email_metadata.subject}")
            if document.email_metadata.initiator:
                hints.append(f"From: {document.email_metadata.initiator}")
            if document.email_metadata.date_start:
                hints.append(f"Date: {document.email_metadata.date_start.strftime('%Y-%m-%d')}")

        truncated_content = content_preview[:max_preview_chars]
        if hints:
            truncated_content = "\n".join(hints) + "\n\n" + truncated_content

        return truncated_content

    async def generate_context(
        self,
        document: Document,
        content_preview: str,
        max_preview_chars: int = 4000,
    ) -> str:
        """Generate a 2-3 sentence context summary for a document.

        Uses two-tier model approach: tries primary model first, falls back
        to larger context model if context window is exceeded.

        Args:
            document: Document object with metadata.
            content_preview: Text content to summarize (will be truncated).
            max_preview_chars: Maximum characters to send to LLM.

        Returns:
            Context summary string.
        """
        prompt_content = self._build_prompt_content(
            document, content_preview, max_preview_chars
        )

        # Try primary model first
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.CONTEXT_PROMPT.format(content=prompt_content),
                    }
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            # Check if context window error
            if "context" in error_str or "token" in error_str:
                logger.warning(
                    f"Context window exceeded with {self.model}, trying fallback model..."
                )
                return await self._try_fallback_model(document, content_preview)
            else:
                logger.warning(f"Failed to generate context: {e}")
                return self._fallback_context(document)

    async def _try_fallback_model(
        self, document: Document, content_preview: str
    ) -> str:
        """Try generating context with fallback model and token truncation.

        Args:
            document: Document object with metadata.
            content_preview: Full text content.

        Returns:
            Context summary string.
        """
        # Apply token truncation for fallback model (reserve 500 for prompt + response)
        safe_limit = self.max_input_tokens - 500
        truncated_content = self._truncate_to_tokens(content_preview, safe_limit)

        # Rebuild prompt with truncated content
        prompt_content = self._build_prompt_content(
            document, truncated_content, len(truncated_content)
        )

        try:
            response = await litellm.acompletion(
                model=self.fallback_model,
                messages=[
                    {
                        "role": "user",
                        "content": self.CONTEXT_PROMPT.format(content=prompt_content),
                    }
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as fallback_error:
            logger.warning(f"Fallback model also failed: {fallback_error}")
            return self._fallback_context(document)

    def _fallback_context(self, document: Document) -> str:
        """Generate fallback context from document metadata when LLM fails."""
        parts = []

        doc_type = document.document_type.value.replace("_", " ").title()
        parts.append(f"This is a {doc_type}.")

        if document.email_metadata:
            meta = document.email_metadata
            if meta.subject:
                parts.append(f"Subject: {meta.subject}.")
            if meta.initiator and meta.date_start:
                date_str = meta.date_start.strftime("%Y-%m-%d")
                parts.append(f"Sent by {meta.initiator} on {date_str}.")

        return " ".join(parts) if parts else ""

    def build_embedding_text(self, context: str, chunk_content: str) -> str:
        """Combine context with chunk content for embedding.

        Args:
            context: Document-level context summary.
            chunk_content: Original chunk text.

        Returns:
            Combined text for embedding.
        """
        if not context:
            return chunk_content
        return f"{context}\n\n{chunk_content}"


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

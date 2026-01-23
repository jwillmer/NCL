"""Embedding generator using OpenAI via LiteLLM."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, TypeVar

import tiktoken
from litellm import aembedding
from litellm.exceptions import APIConnectionError, RateLimitError, Timeout

from ..config import get_settings
from ..models.chunk import Chunk
from ..observability import get_langfuse_metadata

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def _call_with_retry(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Call an async function with exponential backoff retry.

    Retries on rate limits, timeouts, and transient connection errors.

    Args:
        func: Async function to call.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).

    Returns:
        The result of the function call.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except (RateLimitError, Timeout, APIConnectionError) as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"API call failed after {max_retries + 1} attempts: {e}")
                raise

            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    # This should never be reached, but satisfy type checker
    raise last_exception  # type: ignore


class EmbeddingGenerator:
    """Generate embeddings using OpenAI via LiteLLM.

    Supports batch processing with configurable batch sizes for efficiency.
    API keys are initialized via mtss.__init__ at module load.
    Automatically truncates text to fit within model's context window.
    """

    def __init__(self):
        """Initialize the embedding generator."""
        settings = get_settings()
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.max_tokens = settings.embedding_max_tokens
        self.max_concurrent = settings.max_concurrent_embeddings
        self.batch_size = settings.embedding_batch_size

        # Initialize tokenizer for truncation
        if self.model.startswith("text-embedding-"):
            self._encoding = tiktoken.get_encoding("cl100k_base")
        else:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")

    def _truncate_to_max_tokens(self, text: str) -> str:
        """Truncate text to fit within embedding model's token limit.

        Args:
            text: Text to truncate.

        Returns:
            Truncated text that fits within max_tokens.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= self.max_tokens:
            return text
        logger.debug(f"Truncating text from {len(tokens)} to {self.max_tokens} tokens")
        return self._encoding.decode(tokens[: self.max_tokens])

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Includes automatic retry with exponential backoff for transient failures.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        truncated = self._truncate_to_max_tokens(text)

        async def _call():
            return await aembedding(
                model=self.model,
                input=[truncated],
                dimensions=self.dimensions,
                metadata=get_langfuse_metadata(),
            )

        response = await _call_with_retry(_call)
        return response.data[0]["embedding"]

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.

        Includes automatic retry with exponential backoff for transient failures.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Truncate all texts to fit within token limit
        truncated_texts = [self._truncate_to_max_tokens(t) for t in texts]

        # Process in batches to respect API limits
        all_embeddings = []
        metadata = get_langfuse_metadata()

        for i in range(0, len(truncated_texts), self.batch_size):
            batch = truncated_texts[i : i + self.batch_size]

            async def _call(b=batch):  # Capture batch by value to avoid closure bug
                return await aembedding(
                    model=self.model,
                    input=b,
                    dimensions=self.dimensions,
                    metadata=metadata,
                )

            response = await _call_with_retry(_call)
            batch_embeddings = [item["embedding"] for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add embeddings to a list of chunks.

        Args:
            chunks: List of chunks to embed.

        Returns:
            Same chunks with embeddings added.
        """
        if not chunks:
            return chunks

        # Use embedding_text if available (contains cleaned/enriched content),
        # otherwise fall back to raw content
        texts = [chunk.embedding_text or chunk.content for chunk in chunks]
        embeddings = await self.generate_embeddings_batch(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks

    async def embed_chunks_with_progress(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Chunk]:
        """Embed chunks with progress reporting.

        Includes automatic retry with exponential backoff for transient failures.

        Args:
            chunks: List of chunks to embed.
            progress_callback: Optional callback(completed, total) for progress updates.

        Returns:
            Same chunks with embeddings added.
        """
        if not chunks:
            return chunks

        total = len(chunks)
        embedded_chunks: List[Chunk] = []
        metadata = get_langfuse_metadata()

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            # Use embedding_text if available (contains cleaned/enriched content),
            # otherwise fall back to raw content
            texts = [self._truncate_to_max_tokens(chunk.embedding_text or chunk.content) for chunk in batch]

            async def _call(t=texts):  # Capture texts by value to avoid closure bug
                return await aembedding(
                    model=self.model,
                    input=t,
                    dimensions=self.dimensions,
                    metadata=metadata,
                )

            response = await _call_with_retry(_call)

            for chunk, item in zip(batch, response.data):
                chunk.embedding = item["embedding"]
                embedded_chunks.append(chunk)

            if progress_callback:
                progress_callback(len(embedded_chunks), total)

        return embedded_chunks

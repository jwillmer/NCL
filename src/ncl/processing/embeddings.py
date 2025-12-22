"""Embedding generator using OpenAI via LiteLLM."""

from __future__ import annotations

import os
from typing import Callable, List, Optional

from ..config import get_settings
from ..models.chunk import Chunk


class EmbeddingGenerator:
    """Generate embeddings using OpenAI via LiteLLM.

    Supports batch processing with configurable batch sizes for efficiency.
    """

    def __init__(self):
        """Initialize the embedding generator."""
        settings = get_settings()
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.max_concurrent = settings.max_concurrent_embeddings
        self.batch_size = 100  # OpenAI allows up to 2048 inputs per request

        # Ensure API key is set for LiteLLM
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from litellm import embedding

        response = embedding(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
        )
        return response.data[0]["embedding"]

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        from litellm import embedding

        if not texts:
            return []

        # Process in batches to respect API limits
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = embedding(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )
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

        texts = [chunk.content for chunk in chunks]
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

        Args:
            chunks: List of chunks to embed.
            progress_callback: Optional callback(completed, total) for progress updates.

        Returns:
            Same chunks with embeddings added.
        """
        from litellm import embedding

        if not chunks:
            return chunks

        total = len(chunks)
        embedded_chunks: List[Chunk] = []

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [chunk.content for chunk in batch]

            response = embedding(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )

            for chunk, item in zip(batch, response.data):
                chunk.embedding = item["embedding"]
                embedded_chunks.append(chunk)

            if progress_callback:
                progress_callback(len(embedded_chunks), total)

        return embedded_chunks

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        # Rough approximation: ~4 characters per token for English
        return len(text) // 4

    def estimate_cost(self, chunks: List[Chunk]) -> float:
        """Estimate embedding cost for chunks.

        Args:
            chunks: List of chunks to estimate.

        Returns:
            Estimated cost in USD.
        """
        # OpenAI text-embedding-3-small pricing: $0.00002 per 1K tokens
        total_tokens = sum(self.estimate_tokens(chunk.content) for chunk in chunks)
        return (total_tokens / 1000) * 0.00002

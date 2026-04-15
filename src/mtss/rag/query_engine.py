"""RAG query engine for retrieval with source attribution."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..models.chunk import RetrievalResult
from ..processing.embeddings import EmbeddingGenerator
from ..storage.supabase_client import SupabaseClient
from .reranker import Reranker
from .retriever import Retriever

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """Query engine for RAG-based retrieval.

    Uses two-stage retrieval:
    1. Vector similarity search to retrieve candidates
    2. Cross-encoder reranking for improved accuracy (20-35% improvement)

    Answer generation is handled by the LangGraph agent (api/agent.py),
    not by this engine. This class focuses on retrieval only.
    """

    def __init__(self):
        """Initialize the RAG query engine."""
        db = SupabaseClient()
        embeddings = EmbeddingGenerator()
        reranker = Reranker()
        self.retriever = Retriever(db=db, embeddings=embeddings, reranker=reranker)

    async def search_only(
        self,
        question: str,
        top_k: int = 20,
        similarity_threshold: float = 0.3,
        rerank_top_n: Optional[int] = None,
        use_rerank: bool = True,
        vessel_id: Optional[str] = None,
        vessel_type: Optional[str] = None,
        vessel_class: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """Search for relevant chunks without generating an answer.

        Args:
            question: Search query.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score.
            rerank_top_n: Final results after reranking.
            use_rerank: Whether to use reranking.
            vessel_id: Optional vessel UUID to filter results by.
            vessel_type: Optional vessel type to filter results by.
            vessel_class: Optional vessel class to filter results by.
            metadata_filter: Optional pre-built metadata filter.
            on_progress: Optional async callback for progress updates.
            query_embedding: Optional pre-computed query embedding.

        Returns:
            List of retrieval results with citation metadata.
        """
        if metadata_filter is None:
            if vessel_id:
                metadata_filter = {"vessel_ids": [vessel_id]}
            elif vessel_type:
                metadata_filter = {"vessel_types": [vessel_type]}
            elif vessel_class:
                metadata_filter = {"vessel_classes": [vessel_class]}

        return await self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            rerank_top_n=rerank_top_n,
            use_rerank=use_rerank,
            metadata_filter=metadata_filter,
            on_progress=on_progress,
            query_embedding=query_embedding,
        )

    async def close(self):
        """Close database connections."""
        await self.retriever.db.close()

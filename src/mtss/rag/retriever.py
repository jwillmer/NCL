"""Retrieval pipeline: embed -> search -> convert -> rerank."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List

from ..config import get_settings
from ..models.chunk import RetrievalResult
from ..observability.step_timing import record_step
from ..processing.embeddings import EmbeddingGenerator
from ..storage.supabase_client import SupabaseClient
from .reranker import Reranker

logger = logging.getLogger(__name__)


class Retriever:
    """Two-stage retrieval: vector search followed by optional cross-encoder reranking.

    Encapsulates the embed -> search -> convert -> rerank pipeline so that
    both ``RAGQueryEngine`` and ``search_only`` callers share a single path.
    """

    def __init__(
        self,
        db: SupabaseClient,
        embeddings: EmbeddingGenerator,
        reranker: Reranker,
    ):
        self.db = db
        self.embeddings = embeddings
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
        similarity_threshold: float = 0.3,
        rerank_top_n: int | None = None,
        use_rerank: bool = True,
        metadata_filter: dict | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """Embed query (if needed), vector search, convert rows, rerank.

        Args:
            query: Search query text.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score (0-1).
            rerank_top_n: Final results after reranking (default from config).
            use_rerank: Whether to apply cross-encoder reranking.
            metadata_filter: Optional filter dict passed to the vector search.
            on_progress: Optional async callback for progress updates.
            query_embedding: Pre-computed embedding to skip the embed step.

        Returns:
            List of retrieval results, reranked if enabled.
        """
        if on_progress:
            await on_progress("Searching documents")

        if query_embedding is None:
            query_embedding = await self.embeddings.generate_embedding(query)

        settings = get_settings()
        async with record_step("db_search_ms"):
            results = await self.db.search_similar_chunks(
                query_embedding=query_embedding,
                match_threshold=similarity_threshold,
                match_count=top_k,
                metadata_filter=metadata_filter,
                query_text=query if settings.hybrid_search_enabled else None,
            )

        if not results:
            return []

        retrieval_results = _convert_to_retrieval_results(results)

        # Stage 2: Rerank if enabled (skip if too few results)
        effective_top_n = rerank_top_n or self.reranker.top_n
        if use_rerank and self.reranker.enabled and len(retrieval_results) > effective_top_n:
            if on_progress:
                await on_progress("Reranking results...")
            async with record_step("rerank_api_ms"):
                retrieval_results = await self.reranker.rerank_results(
                    query=query, results=retrieval_results, top_n=rerank_top_n
                )

        return retrieval_results

    async def embed_query(self, query: str) -> list[float]:
        """Generate query embedding for concurrent use with other async work."""
        return await self.embeddings.generate_embedding(query)


def _convert_to_retrieval_results(
    results: List[Dict[str, Any]],
) -> List[RetrievalResult]:
    """Convert database result dicts to RetrievalResult objects."""
    retrieval_results = []

    for result in results:
        # For image attachments, use archive_download_uri as the displayable image
        image_uri = None
        doc_type = result.get("document_type")
        if doc_type == "attachment_image" and result.get("archive_download_uri"):
            image_uri = result.get("archive_download_uri")

        retrieval_results.append(
            RetrievalResult(
                text=result["content"],
                score=result["similarity"],
                chunk_id=result.get("chunk_id", ""),
                doc_id=result.get("doc_id", ""),
                source_id=result.get("source_id", ""),
                source_title=result.get("source_title"),
                section_path=result.get("section_path") or [],
                page_number=result.get("page_number"),
                line_from=result.get("line_from"),
                line_to=result.get("line_to"),
                archive_browse_uri=result.get("archive_browse_uri"),
                archive_download_uri=result.get("archive_download_uri"),
                image_uri=image_uri,
                context_summary=result.get("context_summary"),
                document_type=result.get("document_type"),
                email_subject=result.get("email_subject"),
                email_initiator=result.get("email_initiator"),
                email_participants=result.get("email_participants"),
                email_date=(
                    result["email_date"].isoformat()
                    if result.get("email_date")
                    else None
                ),
                root_file_path=result.get("root_file_path"),
                file_path=result.get("file_path"),
            )
        )

    return retrieval_results

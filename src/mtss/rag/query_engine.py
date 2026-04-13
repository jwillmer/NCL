"""RAG query engine for question answering with source attribution."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import litellm

# Allow unsupported params to be dropped (e.g., temperature for gpt-5)
litellm.drop_params = True

from ..config import get_settings
from ..models.chunk import (
    EnhancedRAGResponse,
    RetrievalResult,
)
from ..observability import get_langfuse_metadata
from ..processing.embeddings import EmbeddingGenerator
from ..storage.supabase_client import SupabaseClient
from .citation_processor import CitationProcessor
from .reranker import Reranker
from .retriever import Retriever

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """Query engine for RAG-based question answering.

    Uses two-stage retrieval:
    1. Vector similarity search to retrieve candidates
    2. Cross-encoder reranking for improved accuracy (20-35% improvement)

    Then generates answers using LLM with source attribution.
    Includes citation validation with retry for improved accuracy.
    API keys are initialized via mtss.__init__ at module load.
    """

    MAX_CITATION_RETRIES = 2

    def __init__(self):
        """Initialize the RAG query engine."""
        settings = get_settings()
        db = SupabaseClient()
        embeddings = EmbeddingGenerator()
        reranker = Reranker()
        self.retriever = Retriever(db=db, embeddings=embeddings, reranker=reranker)
        self.citation_processor = CitationProcessor()
        self.llm_model = settings.get_model(settings.rag_llm_model)
        self.chunk_display_max_chars = settings.chunk_display_max_chars

    async def query(
        self,
        question: str,
        top_k: int = 20,
        similarity_threshold: float = 0.3,
        rerank_top_n: Optional[int] = None,
        use_rerank: bool = True,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> EnhancedRAGResponse:
        """Answer a question using RAG with two-stage retrieval and citations.

        Args:
            question: User's question.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score (0-1).
            rerank_top_n: Final results after reranking (default from config).
            use_rerank: Whether to use reranking (default: True).
            on_progress: Optional async callback for progress updates.

        Returns:
            EnhancedRAGResponse with answer and validated citations.
        """
        retrieval_results = await self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            rerank_top_n=rerank_top_n,
            use_rerank=use_rerank,
            on_progress=on_progress,
        )

        if not retrieval_results:
            return EnhancedRAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                query=question,
            )

        if on_progress:
            await on_progress("Generating answer...")

        return await self._generate_with_citations(question, retrieval_results)

    async def query_with_citations(
        self,
        question: str,
        top_k: int = 20,
        similarity_threshold: float = 0.5,
        rerank_top_n: Optional[int] = None,
        use_rerank: bool = True,
    ) -> EnhancedRAGResponse:
        """Answer a question with validated citations and retry logic.

        Args:
            question: User's question.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score (0-1).
            rerank_top_n: Final results after reranking (default from config).
            use_rerank: Whether to use reranking (default: True).

        Returns:
            EnhancedRAGResponse with validated citations and archive links.
        """
        retrieval_results = await self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            rerank_top_n=rerank_top_n,
            use_rerank=use_rerank,
        )

        if not retrieval_results:
            return EnhancedRAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                query=question,
            )

        return await self._generate_with_citations(question, retrieval_results)

    async def _generate_with_citations(
        self, question: str, retrieval_results: List[RetrievalResult]
    ) -> EnhancedRAGResponse:
        """Generate answer with citation validation and retry loop.

        Args:
            question: User's question.
            retrieval_results: Retrieved and optionally reranked results.

        Returns:
            EnhancedRAGResponse with validated citations.
        """
        citation_map = self.citation_processor.get_citation_map(retrieval_results)
        context = self.citation_processor.build_context(retrieval_results)

        retry_count = 0
        validation = None

        for attempt in range(self.MAX_CITATION_RETRIES + 1):
            current_context = context
            if attempt > 0 and validation and validation.invalid_citations:
                hint = self.citation_processor.build_retry_hint(list(citation_map.keys()))
                current_context = context + hint

            raw_response = await self._generate_answer_with_citations(
                question, current_context
            )

            validation = self.citation_processor.process_response(
                raw_response, citation_map
            )

            if not validation.needs_retry:
                break

            retry_count += 1
            logger.info(
                f"Citation retry {retry_count}: {len(validation.invalid_citations)} invalid citations"
            )

        formatted_answer = self.citation_processor.replace_citation_markers(
            validation.response, validation.citations
        )

        sources_section = self.citation_processor.format_sources_section(
            validation.citations
        )
        if sources_section:
            formatted_answer = formatted_answer + sources_section

        return EnhancedRAGResponse(
            answer=formatted_answer,
            citations=validation.citations,
            query=question,
            had_invalid_citations=len(validation.invalid_citations) > 0,
            retry_count=retry_count,
        )

    async def _generate_answer_with_citations(
        self, question: str, context: str
    ) -> str:
        """Generate answer using LLM with citation-aware prompt."""
        from litellm import completion

        system_prompt = self.citation_processor.get_system_prompt()

        user_prompt = f"""Context from emails and attachments:

{context}

---

Question: {question}

Please provide a comprehensive answer based on the above context. Remember to cite sources using [C:chunk_id] format."""

        response = completion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
            metadata=get_langfuse_metadata(),
        )

        return response.choices[0].message.content

    def _build_context_header(self, result: RetrievalResult) -> str:
        """Build a header describing the source of a retrieval result.

        Args:
            result: RetrievalResult with metadata.

        Returns:
            Formatted source header string.
        """
        parts = []

        if result.email_subject:
            parts.append(f"Email: {result.email_subject}")
        if result.email_participants:
            participants = ", ".join(result.email_participants[:3])
            if len(result.email_participants) > 3:
                participants += f" (+{len(result.email_participants) - 3} more)"
            parts.append(f"Participants: {participants}")

        parts.append(f"File: {result.file_path or 'unknown'}")

        if result.section_path:
            parts.append(f"Section: {' > '.join(result.section_path)}")

        return "[Source: " + " | ".join(parts) + "]"

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

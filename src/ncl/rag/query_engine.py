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
    RAGResponse,
    RetrievalResult,
    SourceReference,
)
from ..observability import get_langfuse_metadata
from ..processing.embeddings import EmbeddingGenerator
from ..processing.reranker import Reranker
from ..storage.supabase_client import SupabaseClient
from .citation_processor import CitationProcessor

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """Query engine for RAG-based question answering.

    Uses two-stage retrieval:
    1. Vector similarity search to retrieve candidates
    2. Cross-encoder reranking for improved accuracy (20-35% improvement)

    Then generates answers using LLM with source attribution.
    Includes citation validation with retry for improved accuracy.
    API keys are initialized via ncl.__init__ at module load.
    """

    MAX_CITATION_RETRIES = 2

    def __init__(self):
        """Initialize the RAG query engine."""
        settings = get_settings()
        self.db = SupabaseClient()
        self.embeddings = EmbeddingGenerator()
        self.reranker = Reranker()
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
    ) -> RAGResponse:
        """Answer a question using RAG with two-stage retrieval.

        Args:
            question: User's question.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score (0-1).
            rerank_top_n: Final results after reranking (default from config).
            use_rerank: Whether to use reranking (default: True).
            on_progress: Optional async callback for progress updates.

        Returns:
            RAGResponse with answer and source references.
        """
        # Progress: Vector search
        if on_progress:
            await on_progress("Searching documents")

        # Generate embedding for query
        query_embedding = await self.embeddings.generate_embedding(question)

        # Stage 1: Vector search (retrieve more candidates for reranking)
        results = await self.db.search_similar_chunks(
            query_embedding=query_embedding,
            match_threshold=similarity_threshold,
            match_count=top_k,
        )

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=question,
            )

        # Build source references from results
        sources = []
        result_map = {}  # Map chunk_content to full result for context building

        for result in results:
            source = SourceReference(
                file_path=result["file_path"],
                document_type=result["document_type"],
                email_subject=result.get("email_subject"),
                email_initiator=result.get("email_initiator"),
                email_participants=result.get("email_participants"),
                email_date=(
                    result["email_date"].isoformat() if result.get("email_date") else None
                ),
                chunk_content=result["content"][: self.chunk_display_max_chars],
                similarity_score=result["similarity"],
                heading_path=result.get("heading_path") or [],
                root_file_path=result.get("root_file_path"),
            )
            sources.append(source)
            result_map[source.chunk_content] = result

        # Stage 2: Rerank for improved accuracy (skip if too few results)
        effective_top_n = rerank_top_n or self.reranker.top_n
        if use_rerank and self.reranker.enabled and len(sources) > effective_top_n:
            if on_progress:
                await on_progress("Reranking results...")
            sources = self.reranker.rerank_results(
                query=question,
                sources=sources,
                top_n=rerank_top_n,
            )

        # Progress: Generating answer
        if on_progress:
            await on_progress("Generating answer...")

        # Build context from (reranked) sources
        context_parts = []
        for source in sources:
            result = result_map.get(source.chunk_content, {})
            if result:
                context_header = self._build_context_header(result)
                context_parts.append(f"{context_header}\n{result.get('content', source.chunk_content)}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer using LLM
        answer = await self._generate_answer(question, context, sources)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
        )

    async def query_with_citations(
        self,
        question: str,
        top_k: int = 20,
        similarity_threshold: float = 0.5,
        rerank_top_n: Optional[int] = None,
        use_rerank: bool = True,
    ) -> EnhancedRAGResponse:
        """Answer a question with validated citations and retry logic.

        Uses the enhanced citation system with:
        - Citation headers in context
        - Citation validation after LLM response
        - Automatic retry if too many invalid citations
        - Archive link verification

        Args:
            question: User's question.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score (0-1).
            rerank_top_n: Final results after reranking (default from config).
            use_rerank: Whether to use reranking (default: True).

        Returns:
            EnhancedRAGResponse with validated citations and archive links.
        """
        # Generate embedding for query
        query_embedding = await self.embeddings.generate_embedding(question)

        # Stage 1: Vector search
        results = await self.db.search_similar_chunks(
            query_embedding=query_embedding,
            match_threshold=similarity_threshold,
            match_count=top_k,
        )

        if not results:
            return EnhancedRAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                citations=[],
                query=question,
            )

        # Convert to RetrievalResult objects
        retrieval_results = self._convert_to_retrieval_results(results)

        # Stage 2: Rerank if enabled (skip if too few results)
        effective_top_n = rerank_top_n or self.reranker.top_n
        if use_rerank and self.reranker.enabled and len(retrieval_results) > effective_top_n:
            retrieval_results = self._rerank_retrieval_results(
                question, retrieval_results, rerank_top_n
            )

        # Build citation map and context
        citation_map = self.citation_processor.get_citation_map(retrieval_results)
        context = self.citation_processor.build_context(retrieval_results)

        # Generate answer with citation validation and retry
        retry_count = 0
        validation = None

        for attempt in range(self.MAX_CITATION_RETRIES + 1):
            # Add retry hint if this is a retry
            current_context = context
            if attempt > 0 and validation and validation.invalid_citations:
                hint = self.citation_processor.build_retry_hint(list(citation_map.keys()))
                current_context = context + hint

            # Generate LLM response
            raw_response = await self._generate_answer_with_citations(
                question, current_context
            )

            # Validate citations
            validation = self.citation_processor.process_response(
                raw_response, citation_map
            )

            if not validation.needs_retry:
                break

            retry_count += 1
            logger.info(
                f"Citation retry {retry_count}: {len(validation.invalid_citations)} invalid citations"
            )

        # Format final response
        formatted_answer = self.citation_processor.replace_citation_markers(
            validation.response, validation.citations
        )

        # Add sources section
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

    def _convert_to_retrieval_results(
        self, results: List[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """Convert database results to RetrievalResult objects."""
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

    def _rerank_retrieval_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank retrieval results using cross-encoder."""
        # Convert to SourceReference for reranker (it expects this type)
        sources = [
            SourceReference(
                file_path=r.file_path or "",
                document_type=r.document_type or "",
                email_subject=r.email_subject,
                email_initiator=r.email_initiator,
                email_participants=r.email_participants,
                email_date=r.email_date,
                chunk_content=r.text[: self.chunk_display_max_chars],
                similarity_score=r.score,
                heading_path=r.section_path,
                root_file_path=r.root_file_path,
            )
            for r in results
        ]

        # Rerank
        reranked_sources = self.reranker.rerank_results(
            query=query, sources=sources, top_n=top_n
        )

        # Map back to retrieval results with updated scores
        content_to_result = {r.text[: self.chunk_display_max_chars]: r for r in results}
        reranked_results = []

        for source in reranked_sources:
            original = content_to_result.get(source.chunk_content)
            if original:
                original.rerank_score = source.rerank_score
                reranked_results.append(original)

        return reranked_results

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
            max_tokens=1000,
            metadata=get_langfuse_metadata(),
        )

        return response.choices[0].message.content

    def _build_context_header(self, result: Dict[str, Any]) -> str:
        """Build a header describing the source of a chunk.

        Args:
            result: Search result dictionary.

        Returns:
            Formatted source header string.
        """
        parts = []

        if result.get("email_subject"):
            parts.append(f"Email: {result['email_subject']}")
        if result.get("email_participants"):
            participants = ", ".join(result["email_participants"][:3])
            if len(result["email_participants"]) > 3:
                participants += f" (+{len(result['email_participants']) - 3} more)"
            parts.append(f"Participants: {participants}")

        parts.append(f"File: {result['file_path']}")

        if result.get("heading_path"):
            parts.append(f"Section: {' > '.join(result['heading_path'])}")

        return "[Source: " + " | ".join(parts) + "]"

    async def _generate_answer(
        self,
        question: str,
        context: str,
        sources: List[SourceReference],
    ) -> str:
        """Generate answer using LLM with context.

        Args:
            question: User's question.
            context: Retrieved context from chunks.
            sources: List of source references.

        Returns:
            Generated answer string.
        """
        from litellm import completion

        system_prompt = """You are a helpful assistant that answers questions based on the provided context from emails and their attachments.

Key instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Reference the source documents when relevant (e.g., "According to the email from...")
4. Be concise but thorough
5. If there are conflicting pieces of information, note the discrepancy"""

        user_prompt = f"""Context from emails and attachments:

{context}

---

Question: {question}

Please provide a comprehensive answer based on the above context. If you reference specific information, mention which source it came from."""

        response = completion(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
            metadata=get_langfuse_metadata(),
        )

        return response.choices[0].message.content

    async def search_only(
        self,
        question: str,
        top_k: int = 20,
        similarity_threshold: float = 0.5,
        rerank_top_n: Optional[int] = None,
        use_rerank: bool = True,
        vessel_id: Optional[str] = None,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> List[RetrievalResult]:
        """Search for relevant chunks without generating an answer.

        Args:
            question: Search query.
            top_k: Number of candidates to retrieve for reranking.
            similarity_threshold: Minimum similarity score.
            rerank_top_n: Final results after reranking.
            use_rerank: Whether to use reranking.
            vessel_id: Optional vessel UUID to filter results by.
            on_progress: Optional async callback for progress updates.

        Returns:
            List of retrieval results with citation metadata.
        """
        if on_progress:
            await on_progress("Searching documents")

        query_embedding = await self.embeddings.generate_embedding(question)

        # Build metadata filter if vessel_id is specified
        metadata_filter = None
        if vessel_id:
            metadata_filter = {"vessel_ids": [vessel_id]}

        results = await self.db.search_similar_chunks(
            query_embedding=query_embedding,
            match_threshold=similarity_threshold,
            match_count=top_k,
            metadata_filter=metadata_filter,
        )

        if not results:
            return []

        # Convert to RetrievalResult (includes chunk_id, archive URIs for citations)
        retrieval_results = self._convert_to_retrieval_results(results)

        # Apply reranking if enabled (skip if too few results)
        effective_top_n = rerank_top_n or self.reranker.top_n
        if use_rerank and self.reranker.enabled and len(retrieval_results) > effective_top_n:
            if on_progress:
                await on_progress("Reranking results...")
            retrieval_results = self._rerank_retrieval_results(
                question, retrieval_results, rerank_top_n
            )

        return retrieval_results

    async def close(self):
        """Close database connections."""
        await self.db.close()


def format_response_with_sources(response: RAGResponse) -> str:
    """Format RAG response with source citations for display.

    Args:
        response: RAG response to format.

    Returns:
        Formatted string with answer and sources.
    """
    output = [response.answer, "", "---", "Sources:"]

    for i, source in enumerate(response.sources, 1):
        output.append(f"\n[{i}] {source.file_path}")
        if source.email_subject:
            output.append(f"    Subject: {source.email_subject}")
        if source.email_participants:
            participants = ", ".join(source.email_participants[:5])
            if len(source.email_participants) > 5:
                participants += f" (+{len(source.email_participants) - 5} more)"
            output.append(f"    Participants: {participants}")
        if source.email_date:
            output.append(f"    Date: {source.email_date}")
        # Show rerank score if available, otherwise show vector similarity
        if source.rerank_score is not None:
            output.append(f"    Relevance: {source.rerank_score:.1%} (reranked)")
        else:
            output.append(f"    Relevance: {source.similarity_score:.1%}")
        if source.heading_path:
            output.append(f"    Section: {' > '.join(source.heading_path)}")

    return "\n".join(output)

"""RAG query engine for question answering with source attribution."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..models.chunk import RAGResponse, SourceReference
from ..processing.embeddings import EmbeddingGenerator
from ..storage.supabase_client import SupabaseClient


class RAGQueryEngine:
    """Query engine for RAG-based question answering.

    Retrieves relevant chunks via vector similarity search and
    generates answers using LLM with source attribution.
    """

    def __init__(self):
        """Initialize the RAG query engine."""
        settings = get_settings()
        self.db = SupabaseClient()
        self.embeddings = EmbeddingGenerator()
        self.llm_model = settings.llm_model

        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    async def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> RAGResponse:
        """Answer a question using RAG.

        Args:
            question: User's question.
            top_k: Number of chunks to retrieve.
            similarity_threshold: Minimum similarity score (0-1).

        Returns:
            RAGResponse with answer and source references.
        """
        # Generate embedding for query
        query_embedding = await self.embeddings.generate_embedding(question)

        # Search for relevant chunks
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

        # Build context from retrieved chunks
        context_parts = []
        sources = []

        for result in results:
            # Build source reference
            source = SourceReference(
                file_path=result["file_path"],
                document_type=result["document_type"],
                email_subject=result.get("email_subject"),
                email_initiator=result.get("email_initiator"),
                email_participants=result.get("email_participants"),
                email_date=(
                    result["email_date"].isoformat() if result.get("email_date") else None
                ),
                chunk_content=result["content"][:500],  # Truncate for display
                similarity_score=result["similarity"],
                heading_path=result.get("heading_path") or [],
                root_file_path=result.get("root_file_path"),
            )
            sources.append(source)

            # Build context string
            context_header = self._build_context_header(result)
            context_parts.append(f"{context_header}\n{result['content']}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer using LLM
        answer = await self._generate_answer(question, context, sources)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
        )

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
        )

        return response.choices[0].message.content

    async def search_only(
        self,
        question: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
    ) -> List[SourceReference]:
        """Search for relevant chunks without generating an answer.

        Args:
            question: Search query.
            top_k: Number of results to return.
            similarity_threshold: Minimum similarity score.

        Returns:
            List of source references.
        """
        query_embedding = await self.embeddings.generate_embedding(question)

        results = await self.db.search_similar_chunks(
            query_embedding=query_embedding,
            match_threshold=similarity_threshold,
            match_count=top_k,
        )

        sources = []
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
                chunk_content=result["content"][:500],
                similarity_score=result["similarity"],
                heading_path=result.get("heading_path") or [],
                root_file_path=result.get("root_file_path"),
            )
            sources.append(source)

        return sources

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
        output.append(f"    Relevance: {source.similarity_score:.1%}")
        if source.heading_path:
            output.append(f"    Section: {' > '.join(source.heading_path)}")

    return "\n".join(output)

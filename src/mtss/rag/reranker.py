"""Reranker for two-stage retrieval.

Improves RAG accuracy by 20-35% using cross-encoder models that examine
query+document pairs together for deeper semantic understanding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from litellm import rerank

from ..config import get_settings

if TYPE_CHECKING:
    from ..models.chunk import RetrievalResult


class Reranker:
    """Reranks search results using cross-encoder model via LiteLLM.

    Supports multiple providers:
    - Cohere: cohere/rerank-english-v3.0
    - Azure AI: azure_ai/cohere-rerank-v3.5
    - AWS Bedrock: bedrock/rerank
    - Infinity (self-hosted): infinity/<model>

    API keys are initialized via mtss.__init__ at module load.
    """

    def __init__(self):
        """Initialize the reranker with settings."""
        settings = get_settings()
        self.enabled = settings.rerank_enabled
        self.model = settings.rerank_model
        self.top_n = settings.rerank_top_n

    def rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank results by relevance to query.

        Args:
            query: User's question.
            results: Initial search results from vector search.
            top_n: Number of results to return (default: self.top_n).

        Returns:
            Reranked and filtered results with rerank_score populated.
        """
        if not self.enabled or not results:
            return results[: top_n or self.top_n]

        top_n = top_n or self.top_n

        # Don't rerank if we have fewer results than requested
        if len(results) <= top_n:
            return results

        # Extract document texts for reranking
        documents = [r.text for r in results]

        # Call LiteLLM rerank
        response = rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        # Reorder results by rerank scores
        reranked = []
        for item in response.results:
            result = results[item.index]
            result.rerank_score = item.relevance_score
            reranked.append(result)

        return reranked

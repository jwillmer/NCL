"""Reranker for two-stage retrieval.

Improves RAG accuracy by 20-35% using cross-encoder models that examine
query+document pairs together for deeper semantic understanding.

Uses OpenRouter's rerank API directly via httpx.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from ..config import get_settings
from ..llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY

if TYPE_CHECKING:
    from ..models.chunk import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks search results using cross-encoder model via OpenRouter.

    Calls the OpenRouter /rerank endpoint directly. The response follows
    the Cohere rerank schema (results with index + relevance_score).
    """

    def __init__(self):
        """Initialize the reranker with settings."""
        settings = get_settings()
        self.enabled = settings.rerank_enabled
        self.model = settings.rerank_model
        self.top_n = settings.rerank_top_n
        self.score_floor = settings.rerank_score_floor
        self._api_key = settings.openrouter_api_key
        self._base_url = settings.openrouter_base_url

    async def rerank_results(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_n: Optional[int] = None,
    ) -> List["RetrievalResult"]:
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

        # Build enriched documents for reranking (subject + title provide context)
        documents = []
        for r in results:
            prefix_parts = []
            if r.email_subject:
                prefix_parts.append(r.email_subject)
            if r.source_title and r.source_title != r.email_subject:
                prefix_parts.append(r.source_title)
            prefix = " | ".join(prefix_parts)
            documents.append(f"{prefix}\n{r.text}" if prefix else r.text)

        # Call OpenRouter rerank endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._base_url}/rerank",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                        **OPENROUTER_PRIVACY_EXTRA_BODY,
                    },
                    timeout=30.0,
                )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

            # Reorder results by rerank scores (Cohere response schema)
            reranked = []
            for item in data["results"]:
                result = results[item["index"]]
                result.rerank_score = item["relevance_score"]
                reranked.append(result)
        except Exception as e:
            logger.warning("Reranking failed, returning unranked results: %s", e)
            return results[:top_n]

        # Filter out results below score floor (but keep at least 1 result)
        filtered = [r for r in reranked if r.rerank_score >= self.score_floor]
        return filtered if filtered else reranked[:1]

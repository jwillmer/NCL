"""Vector search and metadata filtering."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import BaseRepository


class SearchRepository(BaseRepository):
    """Handles vector similarity search operations."""

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.7,
        match_count: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity with optional BM25 hybrid scoring.

        Args:
            query_embedding: Query embedding vector.
            match_threshold: Minimum similarity score (0-1).
            match_count: Maximum number of results.
            metadata_filter: Optional JSONB filter (e.g., {"vessel_ids": ["uuid"]}).
            query_text: Optional query text for hybrid BM25 scoring.

        Returns:
            List of matching chunks with document context.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Convert metadata_filter to JSONB format for PostgreSQL
            filter_json = json.dumps(metadata_filter) if metadata_filter else None
            # Use explicit transaction so SET LOCAL applies to the query
            async with conn.transaction():
                # Increase HNSW search quality for this query
                # Default ef_search=40; 100 improves recall by ~5-10% with ~10-20ms cost
                await conn.execute("SET LOCAL hnsw.ef_search = 100")
                rows = await conn.fetch(
                    """
                    SELECT * FROM match_chunks($1, $2, $3, $4::jsonb, $5)
                    """,
                    query_embedding,
                    match_threshold,
                    match_count,
                    filter_json,
                    query_text,
                )

        return [dict(row) for row in rows]

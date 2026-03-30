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
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector (1536 dimensions).
            match_threshold: Minimum similarity score (0-1).
            match_count: Maximum number of results.
            metadata_filter: Optional JSONB filter (e.g., {"vessel_ids": ["uuid"]}).

        Returns:
            List of matching chunks with document context.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Convert metadata_filter to JSONB format for PostgreSQL
            filter_json = json.dumps(metadata_filter) if metadata_filter else None
            rows = await conn.fetch(
                """
                SELECT * FROM match_chunks($1, $2, $3, $4::jsonb)
                """,
                query_embedding,
                match_threshold,
                match_count,
                filter_json,
            )

        return [dict(row) for row in rows]

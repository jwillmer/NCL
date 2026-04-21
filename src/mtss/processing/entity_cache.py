"""Process-local, lazily-loaded caches for small, read-heavy entities.

Covers two tables:
- ``topics`` (~3k rows) — used on every query for exact-name lookup and
  stored ``chunk_count`` access; similarity search stays server-side on the
  HNSW index.
- ``vessels`` (~50 rows) — used on every query to populate the
  vessel-aware system prompt.

Design
------
- Module-level singletons, lazily populated on first use.
- The topic cache mirrors ``(id, name, display_name, chunk_count)`` — NOT
  the 1536-dim embeddings. Loading embeddings for 3k rows meant transferring
  ~92 MB of ``vector::text`` through Supavisor (session pooler), which took
  minutes and tripped the statement timeout. Similarity queries stay on
  the DB via the HNSW-indexed ``find_similar_topics`` RPC (migration 002
  rewrote it so the index is actually used).
- ``ensure_loaded()`` is idempotent + guarded by an ``asyncio.Lock`` so the
  first concurrent caller loads and the rest wait for the same load.
- ``invalidate()`` and ``upsert(...)`` keep the cache aligned with writes
  made by the *same process* (ingest path calling ``get_or_create_topic``).
- Cross-process writes are bounded by a 5-minute TTL: after TTL expires the
  next read reloads from DB.

Not thread-safe by design — asyncio single-threaded execution is assumed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from uuid import UUID

from ..models.topic import Topic
from ..models.vessel import Vessel

if TYPE_CHECKING:
    from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

# Reload from DB if the cache is older than this, even if not explicitly
# invalidated. Protects against cross-process staleness.
_DEFAULT_TTL_SECONDS = 300.0


def _normalize_topic_name(name: str) -> str:
    """Same normalization rule as TopicMatcher._normalize_name — lowercased,
    whitespace-collapsed, trimmed. Kept in sync via a unit test."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _normalize_vessel_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# ============================================================================
# Topic cache
# ============================================================================


class TopicCache:
    """In-memory mirror of the ``topics`` table (without embeddings).

    Hot path is ``match_by_name`` (O(1) dict lookup) and reading
    ``topic.chunk_count`` from the cached Topic model. Embedding-based
    similarity is delegated to the DB RPC via ``cosine_similar`` — that
    RPC uses the HNSW index so a top-k search completes in single-digit ms.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS):
        self._by_norm: Dict[str, Topic] = {}
        self._by_id: Dict[UUID, Topic] = {}
        self._loaded_at: float = 0.0
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    def is_fresh(self) -> bool:
        if not self._by_id:
            return False
        import time
        return (time.time() - self._loaded_at) < self._ttl

    async def ensure_loaded(self, db: "SupabaseClient") -> None:
        """Load if empty or stale. Safe to call on the hot path."""
        if self.is_fresh():
            return
        async with self._lock:
            if self.is_fresh():  # double-check after acquiring the lock
                return
            await self._load(db)

    async def _load(self, db: "SupabaseClient") -> None:
        import time
        topics = await db.list_all_topics_lightweight()
        by_norm: Dict[str, Topic] = {}
        by_id: Dict[UUID, Topic] = {}
        for t in topics:
            by_norm[_normalize_topic_name(t.name)] = t
            by_id[t.id] = t
        self._by_norm = by_norm
        self._by_id = by_id
        self._loaded_at = time.time()
        logger.info("TopicCache loaded %d topics (lightweight)", len(by_id))

    def invalidate(self) -> None:
        """Force a reload on the next ensure_loaded()."""
        self._loaded_at = 0.0

    def upsert(self, topic: Topic) -> None:
        """Reflect an in-process create/update without a full reload."""
        norm = _normalize_topic_name(topic.name)
        self._by_norm[norm] = topic
        self._by_id[topic.id] = topic

    def bump_chunk_count(self, topic_id: UUID, delta: int) -> None:
        t = self._by_id.get(topic_id)
        if t is not None:
            t.chunk_count = max(0, t.chunk_count + delta)

    def match_by_name(self, name: str) -> Optional[Topic]:
        """O(1) exact-name lookup on normalized name."""
        return self._by_norm.get(_normalize_topic_name(name))

    def get_by_id(self, topic_id: UUID) -> Optional[Topic]:
        return self._by_id.get(topic_id)

    async def cosine_similar(
        self,
        db: "SupabaseClient",
        query_embedding: List[float],
        threshold: float,
        limit: int = 1,
        exclude_ids: Optional[set[UUID]] = None,
    ) -> List[Tuple[Topic, float]]:
        """Top-k cosine-similar topics via the HNSW-indexed DB RPC.

        The RPC (``find_similar_topics``, migration 002) does the search on
        the server; we only transfer the top-k rows back. We hydrate the
        results from the cache so the caller gets the same Topic instance
        it would get from ``match_by_name`` (including chunk_count).
        """
        over_fetch = limit + (len(exclude_ids) if exclude_ids else 0)
        raw = await db.find_similar_topics(
            query_embedding, threshold=threshold, limit=max(over_fetch, 1)
        )
        out: List[Tuple[Topic, float]] = []
        for hit in raw:
            tid = hit["id"] if isinstance(hit, dict) else hit.id  # supabase-py returns dicts
            if isinstance(tid, str):
                tid = UUID(tid)
            if exclude_ids and tid in exclude_ids:
                continue
            topic = self._by_id.get(tid)
            if topic is None:
                # Cache miss — DB has a topic we didn't cache. Fall back to
                # a direct lookup so the caller still gets a Topic.
                topic = await db.get_topic_by_id(tid)
                if topic is not None:
                    self.upsert(topic)
            if topic is None:
                continue
            score = float(hit["similarity"]) if isinstance(hit, dict) else float(hit.similarity)
            out.append((topic, score))
            if len(out) >= limit:
                break
        return out


# ============================================================================
# Vessel cache
# ============================================================================


class VesselCache:
    """In-memory mirror of the ``vessels`` table (tiny — ~50 rows)."""

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS):
        self._by_id: Dict[UUID, Vessel] = {}
        self._by_norm: Dict[str, Vessel] = {}
        self._loaded_at: float = 0.0
        self._ttl = ttl_seconds
        self._lock = asyncio.Lock()

    def is_fresh(self) -> bool:
        import time
        return bool(self._by_id) and (time.time() - self._loaded_at) < self._ttl

    async def ensure_loaded(self, db: "SupabaseClient") -> None:
        if self.is_fresh():
            return
        async with self._lock:
            if self.is_fresh():
                return
            await self._load(db)

    async def _load(self, db: "SupabaseClient") -> None:
        import time
        vessels = await db.get_all_vessels()
        by_id: Dict[UUID, Vessel] = {}
        by_norm: Dict[str, Vessel] = {}
        for v in vessels:
            by_id[v.id] = v
            by_norm[_normalize_vessel_name(v.name)] = v
            for alias in v.aliases or ():
                by_norm.setdefault(_normalize_vessel_name(alias), v)
        self._by_id = by_id
        self._by_norm = by_norm
        self._loaded_at = time.time()
        logger.info("VesselCache loaded %d vessels", len(by_id))

    def invalidate(self) -> None:
        self._loaded_at = 0.0

    def upsert(self, vessel: Vessel) -> None:
        self._by_id[vessel.id] = vessel
        self._by_norm[_normalize_vessel_name(vessel.name)] = vessel
        for alias in vessel.aliases or ():
            self._by_norm.setdefault(_normalize_vessel_name(alias), vessel)

    def get_by_id(self, vessel_id: UUID) -> Optional[Vessel]:
        return self._by_id.get(vessel_id)

    def get_by_name(self, name: str) -> Optional[Vessel]:
        return self._by_norm.get(_normalize_vessel_name(name))

    def list_all(self) -> List[Vessel]:
        return list(self._by_id.values())


# ============================================================================
# Module-level singletons + pre-warm helper
# ============================================================================


_topic_cache = TopicCache()
_vessel_cache = VesselCache()


def get_topic_cache() -> TopicCache:
    return _topic_cache


def get_vessel_cache() -> VesselCache:
    return _vessel_cache


async def warm_caches(db: "SupabaseClient") -> None:
    """Pre-load both caches — called at API startup or eval setup so the
    first user request doesn't pay the full-load latency."""
    await asyncio.gather(
        _topic_cache.ensure_loaded(db),
        _vessel_cache.ensure_loaded(db),
    )

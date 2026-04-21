"""Topic extraction and matching for categorization.

Handles:
- LLM-based topic extraction (ingest and query time)
- Semantic deduplication via embeddings
- Topic matching for filtering
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from uuid import UUID

from ..llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY
from ..models.topic import ExtractedTopic, Topic
from .entity_cache import get_topic_cache

if TYPE_CHECKING:
    from ..processing.embeddings import EmbeddingGenerator
    from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


# ============================================
# INPUT SANITIZATION (Security)
# ============================================


_NAME_PUNCT_RE = re.compile(r"[^\w\s]+")
_NAME_WHITESPACE_RE = re.compile(r"\s+")


def normalize_topic_name(name: str) -> str:
    """Aggressive normalization for topic-name dedup.

    Collapses case, punctuation, and whitespace so that variants like
    ``"Pre-Inspection Documentation"``, ``"pre-inspection documentation"``,
    and ``"Pre Inspection, Documentation"`` all map to the same bucket.
    Keeps underscores (``\\w`` retains them) because they are used in
    machine-generated identifiers and are meaningful when present.
    """
    if not name:
        return ""
    lowered = name.lower().strip()
    # Replace any run of punctuation (excluding word chars + whitespace) with a space.
    without_punct = _NAME_PUNCT_RE.sub(" ", lowered)
    collapsed = _NAME_WHITESPACE_RE.sub(" ", without_punct).strip()
    return collapsed


def sanitize_input(text: str, max_length: int = 500) -> str:
    """Sanitize user input before sending to LLM.

    Security measures:
    - Truncates to max_length
    - Removes control characters
    - Strips whitespace
    - Basic prompt injection mitigation
    """
    # Remove control characters except newlines/tabs
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Basic prompt injection mitigation (not foolproof, but raises the bar)
    cleaned = re.sub(
        r"(ignore|disregard|forget).{0,20}(instructions|above|previous|system)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned[:max_length].strip()


# ============================================
# TOPIC EXTRACTOR
# ============================================


class TopicExtractor:
    """Extract topics using LLM.

    Used in TWO contexts:
    1. Ingest time: Extract 1-5 topics from email content
    2. Query time: Extract 1-3 topics from user question
    """

    INGEST_PROMPT = """Extract the main topics from this email for categorization.

You will receive structured input with:
- Subject line (contains key topic indicators)
- Original message (the initial problem/question - most important for topic detection)
- Summary (semantic overview)

Focus on the PROBLEM TYPE being reported, not the solution or vessel name.

Topics should be:
- Specific enough to be useful (e.g., "Cargo Damage" not just "Damage")
- General enough to apply to multiple emails (e.g., "Engine Maintenance" not "Main Engine Fuel Pump Repair")
- Maritime/shipping domain focused when applicable
- Based on the TYPE of issue (equipment failure, damage, maintenance, safety, etc.)

Return 1-5 topics as JSON array:
[{{"name": "Topic Name", "description": "Brief description"}}]

If no clear topics can be extracted, return: []

Content:
{content}"""

    QUERY_PROMPT = """Extract the main topic(s) from this user question.

If the question covers multiple distinct issue types, extract up to 3 topics.
If the question uses "OR" between similar terms, treat them as synonyms for ONE topic.
If the question is too general or doesn't have a clear topic, return empty array.

Examples:
- "What cargo damage incidents happened?" → [{{"name": "Cargo Damage"}}]
- "Engine problems on VLCC?" → [{{"name": "Engine Issues"}}]
- "Cargo damage and engine failures?" → [{{"name": "Cargo Damage"}}, {{"name": "Engine Issues"}}]
- "Hull damage, ballast issues, and engine problems" → [{{"name": "Hull Damage"}}, {{"name": "Ballast Systems"}}, {{"name": "Engine Issues"}}]
- "lubrication supply orders OR lube oil orders OR lubricant procurement" → [{{"name": "Lubricants Supply"}}]
- "engine failure OR motor breakdown OR propulsion issues" → [{{"name": "Engine Issues"}}]
- "Tell me about the Maran" → [] (too general, just a vessel name)
- "Hello" → [] (not a search query)

Return JSON array (1-3 items): [{{"name": "...", "description": "..."}}] or []

User question: {query}"""

    def __init__(self, llm_model: Optional[str] = None):
        """Initialize the topic extractor.

        Args:
            llm_model: Optional LLM model override. If None, uses config default.
        """
        from ..config import get_settings

        settings = get_settings()
        self.llm_model = llm_model or settings.context_llm_model

    @property
    def model_name(self) -> str:
        """Canonical accessor used by ProcessingTrail stamp sites."""
        return self.llm_model

    async def extract_topics(
        self, content: str, max_topics: int = 5
    ) -> List[ExtractedTopic]:
        """Extract 1-5 topics from document content (ingest time).

        Args:
            content: Document text to analyze
            max_topics: Maximum topics to extract

        Returns:
            List of extracted topics (may be empty)
        """
        sanitized = sanitize_input(content, max_length=4000)
        if not sanitized:
            return []

        try:
            from litellm import acompletion

            from ..observability import get_langfuse_metadata

            # Some models (gpt-5) don't support temperature parameter
            call_params = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": self.INGEST_PROMPT.format(content=sanitized),
                    }
                ],
                "max_tokens": 1500,  # gpt-5-nano needs more tokens for full 1-5 topics
                "metadata": get_langfuse_metadata(),
                "extra_body": OPENROUTER_PRIVACY_EXTRA_BODY,
            }
            # Only add temperature for models that support it
            if "gpt-5" not in self.llm_model:
                call_params["temperature"] = 0.3

            from ..cli._common import _service_counter
            _service_counter.add("llm_topics")

            response = await acompletion(**call_params)

            result = response.choices[0].message.content
            if not result:
                logger.debug("Topic extraction returned empty response")
                return []

            if not result:
                logger.warning("Topic extraction returned whitespace-only response")
                return []

            # Extract JSON from response (handle markdown code blocks)
            if "```" in result:
                # Extract content between code blocks
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result)
                if match:
                    result = match.group(1).strip()

            # Handle common non-JSON responses
            if not result or result.lower() in ("null", "none", "n/a", "no topics"):
                return []

            # Try to find JSON array in response (LLM might add text around it)
            if not result.startswith("["):
                match = re.search(r"\[[\s\S]*\]", result)
                if match:
                    result = match.group(0)
                else:
                    logger.debug("Topic extraction returned non-JSON: %s", result[:100])
                    return []

            # Parse JSON array
            topics_data = json.loads(result)

            # Handle case where LLM returns empty array
            if not topics_data:
                logger.debug("Topic extraction returned empty array []")
                return []

            topics = [
                ExtractedTopic(name=t["name"], description=t.get("description"))
                for t in topics_data[:max_topics]
                if isinstance(t, dict) and t.get("name")
            ]

            if not topics:
                logger.debug("Topic extraction parsed but no valid topics in: %s", result[:200])

            return topics
        except json.JSONDecodeError as e:
            logger.debug("Topic extraction JSON parse failed: %s (response: %s)", e, result[:100] if result else "empty")
            return []
        except Exception as e:
            logger.warning("Topic extraction failed: %s", e)
            return []

    async def extract_topics_from_query(
        self, query: str, max_topics: int = 3
    ) -> List[ExtractedTopic]:
        """Extract 1-3 topics from user query (query time).

        Supports multi-topic queries like:
        - "cargo damage and engine problems" → 2 topics
        - "hull damage, ballast issues, engine failures" → 3 topics

        Args:
            query: User's search question
            max_topics: Maximum topics to extract (default 3)

        Returns:
            List of extracted topics (may be empty if query too general)
        """
        sanitized = sanitize_input(query, max_length=500)
        if not sanitized or len(sanitized) < 3:
            return []

        try:
            from litellm import acompletion

            from ..observability import get_langfuse_metadata

            # Some models (gpt-5) don't support temperature parameter
            call_params = {
                "model": self.llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": self.QUERY_PROMPT.format(query=sanitized),
                    }
                ],
                "max_tokens": 500,  # gpt-5-nano needs more tokens
                "metadata": get_langfuse_metadata(),
                "extra_body": OPENROUTER_PRIVACY_EXTRA_BODY,
            }
            # Only add temperature for models that support it
            if "gpt-5" not in self.llm_model:
                call_params["temperature"] = 0.1

            response = await acompletion(**call_params)

            result = response.choices[0].message.content.strip()

            # Handle empty array or null
            if result.lower() in ("null", "[]", "") or not result:
                return []

            # Extract JSON from response (handle markdown code blocks)
            if "```" in result:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result)
                if match:
                    result = match.group(1).strip()

            topics_data = json.loads(result)

            # Ensure it's a list
            if isinstance(topics_data, dict):
                topics_data = [topics_data]  # Single topic returned as object

            return [
                ExtractedTopic(name=t["name"], description=t.get("description"))
                for t in topics_data[:max_topics]
                if t.get("name")
            ]
        except json.JSONDecodeError as e:
            logger.warning("Query topic extraction JSON parse failed: %s", e)
            return []
        except Exception as e:
            logger.warning("Query topic extraction failed: %s", e)
            return []


# ============================================
# TOPIC MATCHER
# ============================================


class TopicMatcher:
    """Match and deduplicate topics using embeddings.

    Thresholds come from ``Settings`` (``TOPIC_DEDUP_THRESHOLD`` and
    ``TOPIC_QUERY_MATCH_THRESHOLD``). The class-level attributes below are
    kept as module-load defaults so existing callers and tests that read
    them directly keep working; the instance-level overrides win at runtime.
    """

    SIMILARITY_THRESHOLD = 0.80  # For ingest deduplication (strict)
    QUERY_SIMILARITY_THRESHOLD = 0.70  # For query-time matching (lenient)

    def __init__(self, db: "SupabaseClient", embeddings: "EmbeddingGenerator"):
        """Initialize the topic matcher.

        Args:
            db: Database client for topic storage.
            embeddings: Embedding generator for semantic matching.
        """
        from ..config import get_settings

        settings = get_settings()
        self.db = db
        self.embeddings = embeddings
        self._name_cache: Dict[str, UUID] = {}  # name -> topic_id
        # Per-instance thresholds so env overrides apply without touching
        # class state (safer under pytest where multiple matchers coexist).
        self.similarity_threshold = settings.topic_dedup_threshold
        self.query_similarity_threshold = settings.topic_query_match_threshold

    def _normalize_name(self, name: str) -> str:
        """Normalize topic name for comparison.

        Delegates to :func:`normalize_topic_name` so the case/punctuation/
        whitespace rules stay identical between the runtime dedup path and
        the retroactive ``mtss topics consolidate --strategy name`` pass.
        """
        return normalize_topic_name(name)

    async def get_or_create_topic(
        self, name: str, description: Optional[str] = None
    ) -> UUID:
        """Get existing topic ID or create new topic.

        Process:
        1. Check cache and exact name match
        2. Generate embedding for candidate
        3. Search existing topics by similarity
        4. If similarity > 0.92: use existing
        5. If 0.85 < similarity < 0.92: auto-merge (close enough)
        6. Else: create new topic
        """
        normalized = self._normalize_name(name)

        # Check cache first
        if normalized in self._name_cache:
            return self._name_cache[normalized]

        # Check process-wide topic cache (in-memory mirror of topics table)
        topic_cache = get_topic_cache()
        await topic_cache.ensure_loaded(self.db)
        cached_match = topic_cache.match_by_name(name)
        if cached_match is not None:
            self._name_cache[normalized] = cached_match.id
            return cached_match.id

        # Fallback exact name match in DB (cache may be missing a fresh row)
        existing = await self.db.get_topic_by_name(normalized)
        if existing:
            self._name_cache[normalized] = existing.id
            topic_cache.upsert(existing)
            return existing.id

        # Generate embedding for similarity search
        embedding = await self.embeddings.generate_embedding(name)

        # Server-side HNSW-indexed similarity search (wrapped by the cache
        # so the returned Topic instance stays cache-consistent).
        cached_sim = await topic_cache.cosine_similar(
            self.db, embedding, threshold=self.similarity_threshold, limit=1
        )
        if cached_sim:
            matched_topic, _score = cached_sim[0]
            self._name_cache[normalized] = matched_topic.id
            return matched_topic.id

        # Create new topic
        topic = Topic(
            name=normalized,
            display_name=name.strip(),
            description=description,
            embedding=embedding,
        )
        created = await self.db.insert_topic(topic)
        self._name_cache[normalized] = created.id
        topic_cache.upsert(created)
        return created.id

    async def get_or_create_topics_batch(
        self, topics: List[Tuple[str, Optional[str]]]
    ) -> List[UUID]:
        """Batch version of get_or_create_topic.

        Reduces embedding API calls by batching all uncached topics into
        a single generate_embeddings_batch() call.

        Args:
            topics: List of (name, description) tuples.

        Returns:
            List of topic UUIDs in the same order as input.
        """
        if not topics:
            return []

        results: List[Optional[UUID]] = [None] * len(topics)
        needs_embedding: List[Tuple[int, str, Optional[str]]] = []  # (index, name, desc)

        topic_cache = get_topic_cache()
        await topic_cache.ensure_loaded(self.db)

        # Pass 1: Check cache and DB exact match
        for i, (name, description) in enumerate(topics):
            normalized = self._normalize_name(name)

            if normalized in self._name_cache:
                results[i] = self._name_cache[normalized]
                continue

            cached_match = topic_cache.match_by_name(name)
            if cached_match is not None:
                self._name_cache[normalized] = cached_match.id
                results[i] = cached_match.id
                continue

            existing = await self.db.get_topic_by_name(normalized)
            if existing:
                self._name_cache[normalized] = existing.id
                topic_cache.upsert(existing)
                results[i] = existing.id
                continue

            needs_embedding.append((i, name, description))

        if not needs_embedding:
            return results  # type: ignore[return-value]

        # Pass 2: Batch embed all uncached topics in one API call
        names_to_embed = [name for _, name, _ in needs_embedding]
        embeddings = await self.embeddings.generate_embeddings_batch(names_to_embed)

        # Pass 3: Similarity check per topic with pre-computed embeddings
        for (i, name, description), embedding in zip(needs_embedding, embeddings):
            normalized = self._normalize_name(name)

            # Re-check cache (earlier topic in this batch may have created it)
            if normalized in self._name_cache:
                results[i] = self._name_cache[normalized]
                continue

            cached_sim = await topic_cache.cosine_similar(
                self.db, embedding, threshold=self.similarity_threshold, limit=1
            )
            if cached_sim:
                matched_topic, _score = cached_sim[0]
                self._name_cache[normalized] = matched_topic.id
                results[i] = matched_topic.id
                continue

            # Create new topic
            topic = Topic(
                name=normalized,
                display_name=name.strip(),
                description=description,
                embedding=embedding,
            )
            created = await self.db.insert_topic(topic)
            self._name_cache[normalized] = created.id
            topic_cache.upsert(created)
            results[i] = created.id

        return results  # type: ignore[return-value]

    async def find_topic_by_name(
        self, name: str, threshold: float | None = None
    ) -> Optional[Topic]:
        """Find single topic by name using embedding similarity.

        Args:
            name: Topic name to search for
            threshold: Override similarity threshold (default: QUERY_SIMILARITY_THRESHOLD)

        Returns:
            Matching topic or None
        """
        normalized = self._normalize_name(name)

        topic_cache = get_topic_cache()
        await topic_cache.ensure_loaded(self.db)

        # Check exact-name match in the cache (O(1)).
        cached_match = topic_cache.match_by_name(name)
        if cached_match is not None:
            return cached_match

        # Cache miss on exact name — fall back to DB in case the cache is
        # stale for a just-created topic.
        existing = await self.db.get_topic_by_name(normalized)
        if existing:
            topic_cache.upsert(existing)
            return existing

        # Server-side HNSW-indexed cosine search.
        effective_threshold = threshold if threshold is not None else self.query_similarity_threshold
        embedding = await self.embeddings.generate_embedding(name)
        cached_sim = await topic_cache.cosine_similar(
            self.db, embedding, threshold=effective_threshold, limit=1
        )
        if cached_sim:
            return cached_sim[0][0]

        return None

    async def find_topics_by_names(
        self, names: List[str], threshold: float | None = None
    ) -> List[Tuple[str, Optional[Topic]]]:
        """Find multiple topics by name (for multi-topic queries).

        Args:
            names: List of topic names to search for
            threshold: Override similarity threshold for all lookups

        Returns:
            List of (original_name, matched_topic) tuples.
            matched_topic is None if no match found.
        """
        results = []
        for name in names:
            topic = await self.find_topic_by_name(name, threshold=threshold)
            results.append((name, topic))
        return results

    async def find_topic_clusters(
        self,
        names: List[str],
        top_k: int = 3,
        threshold: float | None = None,
    ) -> List[Tuple[str, List[Topic]]]:
        """Find the top-K closest DB topics for each query-extracted name.

        Used by the query-time topic filter. Returns a *cluster* of related
        topics per extracted name rather than a single winner. The topic
        ontology is highly fragmented (median chunk_count=1, common
        concepts like "spare parts" span ~7 near-synonymous topic rows),
        so top-1 matching leaves most of the relevant chunk cluster on the
        table. Top-K + a loose threshold (0.55 default) pools the cluster;
        the Cohere reranker downstream cleans up residual noise.

        Exact-name cache hits always win the cluster's first slot, then
        HNSW similarity fills the remaining slots.
        """
        topic_cache = get_topic_cache()
        await topic_cache.ensure_loaded(self.db)

        effective_threshold = (
            threshold if threshold is not None else self.query_similarity_threshold
        )

        out: List[Tuple[str, List[Topic]]] = []
        for name in names:
            seen: set[UUID] = set()
            cluster: List[Topic] = []

            exact = topic_cache.match_by_name(name)
            if exact is not None:
                cluster.append(exact)
                seen.add(exact.id)

            if len(cluster) < top_k:
                embedding = await self.embeddings.generate_embedding(name)
                sims = await topic_cache.cosine_similar(
                    self.db, embedding, threshold=effective_threshold, limit=top_k
                )
                for topic, _score in sims:
                    if topic.id in seen:
                        continue
                    cluster.append(topic)
                    seen.add(topic.id)
                    if len(cluster) >= top_k:
                        break

            out.append((name, cluster))
        return out

    async def find_related_topics(
        self,
        query: str,
        exclude_ids: Optional[List[UUID]] = None,
        limit: int = 3,
    ) -> List[Topic]:
        """Find topics related to a query for suggestions.

        Args:
            query: Search query to find related topics for
            exclude_ids: Topic IDs to exclude from results (e.g., already matched)
            limit: Maximum number of topics to return
        """
        embedding = await self.embeddings.generate_embedding(query)
        exclude_set = set(exclude_ids) if exclude_ids else set()

        topic_cache = get_topic_cache()
        await topic_cache.ensure_loaded(self.db)

        # Over-fetch so we can still return `limit` after filtering zero-chunk topics.
        cached_sim = await topic_cache.cosine_similar(
            self.db,
            embedding,
            threshold=0.5,
            limit=limit + len(exclude_set) + 5,
            exclude_ids=exclude_set,
        )

        results: List[Topic] = []
        for topic, _score in cached_sim:
            if topic.chunk_count > 0:
                results.append(topic)
            if len(results) >= limit:
                break

        return results

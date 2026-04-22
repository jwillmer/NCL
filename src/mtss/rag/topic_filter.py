"""Topic-based pre-filtering for RAG queries.

Extracts topic from user query, matches to existing topics,
and provides early return when no results are possible.

Topics are automatically detected from user questions -
users do not manually select categories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional
from uuid import UUID

from ..config import get_settings

from ..models.topic import TopicSummary

if TYPE_CHECKING:
    from ..processing.topics import TopicExtractor, TopicMatcher
    from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


# ============================================
# USER-FACING MESSAGES
# ============================================


class TopicMessages:
    """User-facing messages for topic filtering results.

    Topics are automatically detected - messages reflect this.
    Supports both single and multi-topic scenarios.
    """

    @staticmethod
    def _format_topics(names: List[str]) -> str:
        """Format topic names for display."""
        if len(names) == 1:
            return f"'{names[0]}'"
        elif len(names) == 2:
            return f"'{names[0]}' and '{names[1]}'"
        else:
            return ", ".join(f"'{n}'" for n in names[:-1]) + f", and '{names[-1]}'"

    @staticmethod
    def no_topic_match(detected: List[str]) -> str:
        """Detected topic(s) not found in database."""
        topics_str = TopicMessages._format_topics(detected)
        plural = len(detected) > 1
        category_text = "these categories aren't" if plural else "this category isn't"
        return (
            f"I detected you're asking about {topics_str}, but "
            f"{category_text} in our records yet.\n\n"
            f"Our knowledge base is continuously expanding as new incidents are reported.\n\n"
            f"Would you like me to search across all available categories instead?"
        )

    @staticmethod
    def partial_topic_match(
        matched: List[str], unmatched: List[str], count: int
    ) -> str:
        """Some topics matched, others didn't."""
        matched_str = TopicMessages._format_topics(matched)
        unmatched_str = TopicMessages._format_topics(unmatched)
        category_text = "aren't" if len(unmatched) > 1 else "isn't"
        return (
            f"Found {count} documents in {matched_str}. "
            f"Note: {unmatched_str} {category_text} "
            f"in our records yet."
        )

    @staticmethod
    def empty_topics(topic_names: List[str], suggested: List[str]) -> str:
        """Topic(s) exist but have no documents yet."""
        topics_str = TopicMessages._format_topics(topic_names)
        plural = len(topic_names) > 1
        msg = (
            f"I detected your question relates to {topics_str}, but we don't "
            f"have any documents in {'these categories' if plural else 'this category'} yet.\n\n"
            f"Our knowledge base is continuously expanding."
        )
        if suggested:
            related = ", ".join(f"'{s}'" for s in suggested[:3])
            msg += f"\n\n**Related categories with records:** {related}"
        msg += "\n\nWould you like me to search across all categories instead?"
        return msg

    @staticmethod
    def no_vessel_match(
        topic_names: List[str], topic_count: int, vessel_desc: str
    ) -> str:
        """Topic(s) have results but none match vessel filter."""
        topics_str = TopicMessages._format_topics(topic_names)
        return (
            f"I found **{topic_count} records** related to {topics_str}, "
            f"but none specifically for {vessel_desc}.\n\n"
            f"**I can:**\n"
            f"1. Show results from {topics_str} across all vessels "
            f"(recommended - solutions often apply)\n"
            f"2. Keep the vessel filter and search other related topics\n\n"
            f"Which would be more helpful?"
        )

    @staticmethod
    def topic_context(topic_names: List[str], count: int) -> str:
        """Context message when searching within topic(s)."""
        topics_str = TopicMessages._format_topics(topic_names)
        return f"Based on your question, searching {count} documents in {topics_str}..."


# ============================================
# TOPIC FILTER RESULT
# ============================================


@dataclass
class MatchedTopic:
    """A single matched topic with its counts."""

    id: UUID
    name: str  # Canonical
    display_name: str  # User-friendly
    chunk_count: int  # Total chunks in this topic
    filtered_count: int  # Chunks after vessel filter (if applied)


@dataclass
class TopicFilterResult:
    """Result of topic pre-filtering (supports multi-topic)."""

    detected_topics: List[str] = field(default_factory=list)  # Topics extracted from query
    matched_topics: List[MatchedTopic] = field(default_factory=list)  # Matched in database
    unmatched_topics: List[str] = field(default_factory=list)  # Detected but not in DB
    should_skip_rag: bool = False  # If True, don't run RAG
    message: Optional[str] = None  # User-facing message
    total_chunk_count: int = 0  # Combined across all matched topics
    filtered_chunk_count: int = 0  # After vessel filter
    suggested_topics: List[TopicSummary] = field(default_factory=list)

    @property
    def matched_topic_ids(self) -> List[UUID]:
        """Get list of matched topic IDs for filtering."""
        return [t.id for t in self.matched_topics]

    @property
    def matched_topic_names(self) -> List[str]:
        """Get display names for user messages."""
        return [t.display_name for t in self.matched_topics]


# ============================================
# TOPIC FILTER
# ============================================


class TopicFilter:
    """Pre-filter queries based on topic before running RAG.

    Benefits:
    - Early return when no results possible (faster queries)
    - User feedback about data availability
    - Combined topic + vessel filtering
    """

    def __init__(
        self,
        topic_extractor: "TopicExtractor",
        topic_matcher: "TopicMatcher",
        db: "SupabaseClient",
    ):
        """Initialize the topic filter.

        Args:
            topic_extractor: Extractor for topic detection from queries.
            topic_matcher: Matcher for finding existing topics.
            db: Database client for count queries.
        """
        self.extractor = topic_extractor
        self.matcher = topic_matcher
        self.db = db

    async def analyze_query(
        self,
        query: str,
        vessel_filter: Optional[Dict] = None,
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> TopicFilterResult:
        """Analyze query and determine if RAG should proceed.

        Supports multi-topic queries (1-3 topics).

        Steps:
        1. Extract topics from query using LLM (1-3) — skipped in
           embedding-only mode.
        2. Match each to existing topics via embedding similarity
        3. Pre-check counts for early return decision
        4. Use OR logic: proceed if ANY matched topic has results

        Args:
            query: User's search question
            vessel_filter: Optional vessel filter dict
            query_embedding: Pre-computed query embedding. When
                ``settings.topic_filter_embedding_only`` is True this
                is used to match topics directly against the topic
                cache, skipping the LLM extractor (~3s saved per
                call). The query text is still recorded as the
                detected name for display.

        Returns:
            TopicFilterResult with skip decision and context
        """
        settings = get_settings()

        if settings.topic_filter_embedding_only and query_embedding is not None:
            # Embedding-only path: skip the LLM extractor. The query
            # embedding is already computed in search_node's parallel
            # gather, so this path adds zero new API calls. Trade-off:
            # multi-topic queries ("cargo damage AND engine problems")
            # collapse into one cluster instead of N, but the no-result
            # fallback in search_node already catches over-narrowing.
            if on_progress:
                await on_progress("Matching topics")
            try:
                from ..processing.entity_cache import get_topic_cache

                topic_cache = get_topic_cache()
                await topic_cache.ensure_loaded(self.db)
                sims = await topic_cache.cosine_similar(
                    self.db,
                    query_embedding,
                    threshold=self.matcher.query_similarity_threshold,
                    limit=settings.topic_query_top_k,
                )
            except Exception as e:
                logger.warning(
                    "Embedding-only topic matching failed, broad search: %s", e
                )
                return TopicFilterResult()

            if not sims:
                return TopicFilterResult()

            pooled_topics = [t for t, _score in sims]
            detected_names = [query[:60]]  # surface the query as the label

            # Batch the vessel-filtered count into a single RPC
            # (count_chunks_by_topics — OR over topic_ids). The prior
            # per-topic loop did N sequential DB round-trips that dominated
            # topic_filter_ms on vessel-filtered queries.
            vessel_filtered_total = 0
            if vessel_filter and pooled_topics:
                try:
                    vessel_filtered_total = await self.db.get_chunks_count_for_topics(
                        [t.id for t in pooled_topics], vessel_filter
                    )
                except Exception as e:
                    logger.warning("Batched topic count failed: %s", e)

            matched_topics: List[MatchedTopic] = [
                MatchedTopic(
                    id=t.id,
                    name=t.name,
                    display_name=t.display_name,
                    chunk_count=t.chunk_count,
                    # Per-topic filtered count is display-only and unused
                    # by search_node; approximate with the unfiltered count
                    # to avoid N DB round-trips.
                    filtered_count=t.chunk_count,
                )
                for t in pooled_topics
            ]
            total_chunks = (
                vessel_filtered_total
                if vessel_filter
                else sum(t.chunk_count for t in matched_topics)
            )
            return TopicFilterResult(
                detected_topics=detected_names,
                matched_topics=matched_topics,
                unmatched_topics=[],
                total_chunk_count=total_chunks,
            )

        # Step 1: Extract topics from query (with error handling)
        if on_progress:
            await on_progress("Extracting topics")
        try:
            extracted = await self.extractor.extract_topics_from_query(query)
        except Exception as e:
            logger.warning("Topic extraction failed, using broad search: %s", e)
            return TopicFilterResult()  # Broad search on failure

        if not extracted:
            # Query too general, proceed with broad search
            return TopicFilterResult()

        detected_names = [t.name for t in extracted]

        # Step 2: Match each extracted name to a CLUSTER of DB topics (top-K).
        # The ontology has a lot of near-synonyms per concept (e.g. ~7 "spare
        # parts" rows), so top-1 matching historically missed ~40% of the
        # relevant chunks per query. Top-K at a loose threshold (0.55 default)
        # pools the cluster; the reranker downstream filters residual noise.
        try:
            clusters = await self.matcher.find_topic_clusters(
                detected_names, top_k=settings.topic_query_top_k
            )
        except Exception as e:
            logger.warning("Topic matching failed: %s", e)
            return TopicFilterResult(
                detected_topics=detected_names,
                message=TopicMessages.no_topic_match(detected_names),
            )

        # Pool across all extracted names and dedupe by topic id.
        pooled: Dict[UUID, "Topic"] = {}  # type: ignore[name-defined]
        unmatched_names: List[str] = []
        for original_name, cluster in clusters:
            if not cluster:
                unmatched_names.append(original_name)
                continue
            for t in cluster:
                pooled.setdefault(t.id, t)

        # Convert to MatchedTopic list; apply vessel_filter only when set
        # (stored chunk_count handles the no-filter case without a DB call).
        matched_topics: List[MatchedTopic] = []
        for t in pooled.values():
            count = t.chunk_count
            filtered = count
            if vessel_filter:
                filtered = await self.db.get_chunks_count_for_topic(
                    t.id, vessel_filter
                )
            matched_topics.append(
                MatchedTopic(
                    id=t.id,
                    name=t.name,
                    display_name=t.display_name,
                    chunk_count=count,
                    filtered_count=filtered,
                )
            )

        # Loosening safety: if the top-K / primary-threshold pass still came
        # back sparse, retry with the looser threshold setting. With the new
        # default (primary=0.55, min_chunks=1) this rarely fires — kept for
        # operators who tighten the primary threshold via env overrides.
        total_chunks_initial = sum(t.chunk_count for t in matched_topics)
        if total_chunks_initial < settings.topic_loosening_min_chunks and detected_names:
            try:
                loose_clusters = await self.matcher.find_topic_clusters(
                    detected_names,
                    top_k=settings.topic_query_top_k,
                    threshold=settings.topic_match_threshold_loose,
                )
                loose_pooled: Dict[UUID, "Topic"] = {}  # type: ignore[name-defined]
                loose_unmatched: List[str] = []
                for original_name, cluster in loose_clusters:
                    if not cluster:
                        loose_unmatched.append(original_name)
                        continue
                    for t in cluster:
                        loose_pooled.setdefault(t.id, t)
                loose_matched: List[MatchedTopic] = []
                for t in loose_pooled.values():
                    count = t.chunk_count
                    filtered = count
                    if vessel_filter:
                        filtered = await self.db.get_chunks_count_for_topic(
                            t.id, vessel_filter
                        )
                    loose_matched.append(
                        MatchedTopic(
                            id=t.id,
                            name=t.name,
                            display_name=t.display_name,
                            chunk_count=count,
                            filtered_count=filtered,
                        )
                    )
                loose_total = sum(t.chunk_count for t in loose_matched)
                if loose_total > total_chunks_initial:
                    logger.info(
                        "Topic loosening improved results: %d -> %d chunks (threshold %.2f)",
                        total_chunks_initial,
                        loose_total,
                        settings.topic_match_threshold_loose,
                    )
                    matched_topics = loose_matched
                    unmatched_names = loose_unmatched
            except Exception as e:
                logger.warning("Topic loosening failed, using original results: %s", e)

        # No matches at all → early return, ask user
        if not matched_topics:
            return TopicFilterResult(
                should_skip_rag=True,
                detected_topics=detected_names,
                unmatched_topics=unmatched_names,
                message=TopicMessages.no_topic_match(detected_names),
            )

        # Calculate totals
        total_chunks = sum(t.chunk_count for t in matched_topics)
        total_filtered = sum(t.filtered_count for t in matched_topics)
        matched_names = [t.display_name for t in matched_topics]

        # Step 3: ALL matched topics have 0 documents → early return
        if total_chunks == 0:
            # All topics empty - suggest related topics (exclude the matched ones)
            exclude_ids = [t.id for t in matched_topics]
            related = await self.matcher.find_related_topics(
                query, exclude_ids=exclude_ids, limit=3
            )
            return TopicFilterResult(
                should_skip_rag=True,
                detected_topics=detected_names,
                matched_topics=matched_topics,
                unmatched_topics=unmatched_names,
                total_chunk_count=0,
                message=TopicMessages.empty_topics(
                    matched_names, [t.display_name for t in related]
                ),
                suggested_topics=[
                    TopicSummary(
                        id=t.id,
                        name=t.name,
                        display_name=t.display_name,
                        chunk_count=t.chunk_count,
                    )
                    for t in related
                ],
            )

        # Step 4 (vessel-filter skip) intentionally removed.
        #
        # The previous behavior — "skip RAG if total_filtered == 0" — used
        # the `count_chunks_by_topic($topic, $vessel_filter)` RPC as a hard
        # veto. That RPC does a JSONB containment check
        # (`chunks.metadata @> vessel_filter`), which matches on a narrow
        # schema. The retrieval layer downstream applies a richer vessel
        # filter (ids, classes, type aliases). Baseline-01 (2026-04-21)
        # showed this false-skipping queries where the topic had 32+
        # chunks — users were told "no records" when plenty existed.
        # We now defer to the retriever to decide if nothing matches.

        # Step 5: Proceed with RAG (some results exist)
        # Build appropriate message
        if unmatched_names:
            message = TopicMessages.partial_topic_match(
                matched_names, unmatched_names, total_filtered or total_chunks
            )
        else:
            message = TopicMessages.topic_context(
                matched_names, total_filtered or total_chunks
            )

        return TopicFilterResult(
            detected_topics=detected_names,
            matched_topics=matched_topics,
            unmatched_topics=unmatched_names,
            total_chunk_count=total_chunks,
            filtered_chunk_count=total_filtered,
            message=message,
        )

    def _describe_vessel_filter(self, vessel_filter: Dict) -> str:
        """Format vessel filter for user message."""
        if "vessel_ids" in vessel_filter:
            # Would need to look up vessel name - simplified for now
            return "the selected vessel"
        elif "vessel_types" in vessel_filter:
            types = vessel_filter["vessel_types"]
            if isinstance(types, list) and types:
                return f"vessel type {types[0]}"
            return "the selected vessel type"
        elif "vessel_classes" in vessel_filter:
            classes = vessel_filter["vessel_classes"]
            if isinstance(classes, list) and classes:
                return f"vessel class {classes[0]}"
            return "the selected vessel class"
        return "the selected filter"

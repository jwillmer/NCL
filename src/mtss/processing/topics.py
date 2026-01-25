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

from ..models.topic import ExtractedTopic, Topic

if TYPE_CHECKING:
    from ..processing.embeddings import EmbeddingGenerator
    from ..storage.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


# ============================================
# INPUT SANITIZATION (Security)
# ============================================


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
If the question is too general or doesn't have a clear topic, return empty array.

Examples:
- "What cargo damage incidents happened?" → [{{"name": "Cargo Damage"}}]
- "Engine problems on VLCC?" → [{{"name": "Engine Issues"}}]
- "Cargo damage and engine failures?" → [{{"name": "Cargo Damage"}}, {{"name": "Engine Issues"}}]
- "Hull damage, ballast issues, and engine problems" → [{{"name": "Hull Damage"}}, {{"name": "Ballast Systems"}}, {{"name": "Engine Issues"}}]
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
            }
            # Only add temperature for models that support it
            if not self.llm_model.startswith("gpt-5"):
                call_params["temperature"] = 0.3

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
            }
            # Only add temperature for models that support it
            if not self.llm_model.startswith("gpt-5"):
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

    Deduplication threshold: 0.85
    - >= 0.85: Auto-merge with existing topic
    - < 0.85: Create new topic
    """

    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, db: "SupabaseClient", embeddings: "EmbeddingGenerator"):
        """Initialize the topic matcher.

        Args:
            db: Database client for topic storage.
            embeddings: Embedding generator for semantic matching.
        """
        self.db = db
        self.embeddings = embeddings
        self._name_cache: Dict[str, UUID] = {}  # name -> topic_id

    def _normalize_name(self, name: str) -> str:
        """Normalize topic name for comparison."""
        return name.lower().strip()

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

        # Check exact name match in DB
        existing = await self.db.get_topic_by_name(normalized)
        if existing:
            self._name_cache[normalized] = existing.id
            return existing.id

        # Generate embedding for similarity search
        embedding = await self.embeddings.generate_embedding(name)

        # Find similar topics
        similar = await self.db.find_similar_topics(
            embedding, threshold=self.SIMILARITY_THRESHOLD, limit=3
        )

        if similar:
            top_match = similar[0]
            if top_match["similarity"] >= self.SIMILARITY_THRESHOLD:
                # Auto-merge (same or close enough)
                self._name_cache[normalized] = top_match["id"]
                return top_match["id"]

        # Create new topic
        topic = Topic(
            name=normalized,
            display_name=name.strip(),
            description=description,
            embedding=embedding,
        )
        created = await self.db.insert_topic(topic)
        self._name_cache[normalized] = created.id
        return created.id

    async def find_topic_by_name(self, name: str) -> Optional[Topic]:
        """Find single topic by name using embedding similarity.

        Args:
            name: Topic name to search for

        Returns:
            Matching topic or None
        """
        normalized = self._normalize_name(name)

        # Check exact match first
        existing = await self.db.get_topic_by_name(normalized)
        if existing:
            return existing

        # Try embedding similarity
        embedding = await self.embeddings.generate_embedding(name)
        similar = await self.db.find_similar_topics(
            embedding, threshold=self.SIMILARITY_THRESHOLD, limit=1
        )

        if similar:
            return await self.db.get_topic_by_id(similar[0]["id"])

        return None

    async def find_topics_by_names(
        self, names: List[str]
    ) -> List[Tuple[str, Optional[Topic]]]:
        """Find multiple topics by name (for multi-topic queries).

        Args:
            names: List of topic names to search for

        Returns:
            List of (original_name, matched_topic) tuples.
            matched_topic is None if no match found.
        """
        results = []
        for name in names:
            topic = await self.find_topic_by_name(name)
            results.append((name, topic))
        return results

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

        similar = await self.db.find_similar_topics(
            embedding, threshold=0.5, limit=limit + len(exclude_set)
        )

        results = []
        for match in similar:
            if match["id"] in exclude_set:
                continue
            topic = await self.db.get_topic_by_id(match["id"])
            if topic and topic.chunk_count > 0:
                results.append(topic)
            if len(results) >= limit:
                break

        return results

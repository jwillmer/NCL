"""Query intent classifier for chat_node preprocessing.

Routes user turns into one of four intents so the agent can decide
whether to hit RAG at all:

- ``factual_query``    — answerable from the knowledge base; force RAG search.
- ``exploratory``      — user is thinking out loud or scoping a problem
                         ("I have an issue, not sure where it's from").
                         The agent replies conversationally and offers
                         to search when the query firms up.
- ``off_topic``        — outside the maritime-support domain (e.g. trivia,
                         personal questions). Short-circuit with a polite
                         scope reminder; no RAG, no wasted tokens.
- ``greeting``         — "hi", "thanks", small talk. Short canned reply,
                         no RAG.

Why this exists:
    Baseline-01 (2026-04-21) showed gpt-5-mini exercised its own tool-call
    discretion and skipped ``search_documents`` on 31/37 factual questions.
    Giving the router its own focused classifier + a ``tool_choice="required"``
    hand-off makes the search-or-not decision deterministic and observable,
    and lets us skip the tool-call round trip entirely on chit-chat /
    off-topic turns (saves ~10–20 s + model fees per wasted call).

Design notes:
    * Runs on every fresh user turn entering ``chat_node`` (not on
      returns from ``search_node`` — those already have a citation_map).
    * Uses a small, fast model (``gemini-2.5-flash-lite``) — same family
      that won the topic-extractor bake-off on latency/consistency.
    * On any failure (LLM error, bad JSON, timeout) we fall open to
      ``factual_query`` so a misclassifier never *prevents* a user
      from getting search results — worst case we hit RAG when we
      shouldn't have, which is strictly the old behavior.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..config import get_settings
from ..llm_privacy import OPENROUTER_PRIVACY_EXTRA_BODY
from ..processing.topics import sanitize_input

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTUAL_QUERY = "factual_query"
    EXPLORATORY = "exploratory"
    OFF_TOPIC = "off_topic"
    GREETING = "greeting"


@dataclass(frozen=True)
class IntentResult:
    intent: QueryIntent
    confidence: float  # 0.0–1.0, self-reported by the model
    reasoning: Optional[str] = None

    @property
    def should_search(self) -> bool:
        return self.intent is QueryIntent.FACTUAL_QUERY


CLASSIFIER_PROMPT = """You route user messages for a maritime technical support assistant (MTSS).
MTSS has a knowledge base of ship emails, incident reports, maintenance logs, surveys, and technical attachments.

Classify the latest user message into exactly ONE intent:

- factual_query: The user is asking a specific question answerable from records (incidents, emails, maintenance, vessels, surveys, parts, regulations, technical issues). ANYTHING that could be looked up against the knowledge base.
  Examples: "What cargo pump failures happened last quarter?", "show me PSC deficiencies", "open maintenance on MV Canopus", "how was the BWTS PLC recovery handled?"

- exploratory: The user is describing a problem without a concrete question yet — thinking out loud, scoping, or asking for help narrowing down. NOT a lookup yet; the agent should chat and offer to search once the query is specific.
  Examples: "I have a problem, the gauge shows wrong reading but I'm not sure where it comes from", "help me figure out what's wrong with engine 2", "can you help me troubleshoot?"

- off_topic: Not maritime/vessel/technical-support related. Trivia, politics, personal questions, other domains.
  Examples: "What is 2+2?", "who won the world cup?", "write me a poem", "what's your favorite color?"

- greeting: Small talk only — hello/hi/thanks/goodbye/how are you. No actual request.
  Examples: "hi", "thanks", "hello there", "bye"

Rules:
- When in doubt between factual_query and exploratory, pick factual_query (searching is cheap, missing a search is expensive).
- A vessel name alone ("tell me about MV Canopus") is factual_query.
- A question about the system itself ("what can you do?") is greeting.

Return ONLY valid JSON, no prose:
{{"intent": "<one of: factual_query, exploratory, off_topic, greeting>", "confidence": <0.0-1.0>, "reasoning": "<one short clause>"}}

User message:
{query}"""


class IntentClassifier:
    """Classify a user turn into a routing intent.

    Instantiate once per request (cheap; no cache). Call
    :meth:`classify` with the raw user question.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        self.model = model or settings.intent_classifier_model or settings.llm_model

    async def classify(self, query: str) -> IntentResult:
        """Classify a single user message.

        Falls open to ``factual_query`` on any failure so a broken
        classifier can never block search.
        """
        sanitized = sanitize_input(query, max_length=500)
        if not sanitized:
            return IntentResult(
                intent=QueryIntent.GREETING,
                confidence=1.0,
                reasoning="empty input",
            )

        try:
            from litellm import acompletion
            from ..observability import get_langfuse_metadata

            call_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": CLASSIFIER_PROMPT.format(query=sanitized),
                    }
                ],
                "max_tokens": 120,
                "metadata": get_langfuse_metadata(),
                "extra_body": OPENROUTER_PRIVACY_EXTRA_BODY,
            }
            if "gpt-5" not in self.model:
                call_params["temperature"] = 0.0

            response = await acompletion(**call_params)
            raw = (response.choices[0].message.content or "").strip()
            return self._parse(raw)
        except Exception as exc:
            logger.warning("Intent classification failed, falling open: %s", exc)
            return IntentResult(
                intent=QueryIntent.FACTUAL_QUERY,
                confidence=0.0,
                reasoning=f"classifier error: {type(exc).__name__}",
            )

    @staticmethod
    def _parse(raw: str) -> IntentResult:
        """Pull JSON payload out of the model response.

        Tolerates markdown fences and surrounding prose, since smaller
        models occasionally ignore "ONLY JSON" instructions.
        """
        if not raw:
            return IntentResult(
                intent=QueryIntent.FACTUAL_QUERY,
                confidence=0.0,
                reasoning="empty classifier response",
            )

        text = raw
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()
        if not text.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                text = match.group(0)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Intent classifier returned non-JSON: %s", raw[:200])
            return IntentResult(
                intent=QueryIntent.FACTUAL_QUERY,
                confidence=0.0,
                reasoning="unparseable classifier response",
            )

        intent_raw = str(payload.get("intent", "")).strip().lower()
        try:
            intent = QueryIntent(intent_raw)
        except ValueError:
            logger.warning("Intent classifier returned unknown intent: %r", intent_raw)
            intent = QueryIntent.FACTUAL_QUERY

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning = payload.get("reasoning")
        if reasoning is not None:
            reasoning = str(reasoning)[:240]

        return IntentResult(intent=intent, confidence=confidence, reasoning=reasoning)


def latest_user_text(messages: List[BaseMessage]) -> Optional[str]:
    """Return the text of the most recent HumanMessage, or None."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                return "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            return str(content) if content else None
    return None


def is_fresh_user_turn(messages: List[BaseMessage]) -> bool:
    """True iff the last message is a user turn with no pending tool calls.

    Guards against running the classifier on:
      * empty message lists
      * returns from ``search_node`` (last msg is ToolMessage/AIMessage)
      * mid-tool-call states
    """
    if not messages:
        return False
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return False
    # If the previous AIMessage had unresolved tool calls, we're mid-turn.
    for msg in reversed(messages[:-1]):
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                return False
            break
    return True

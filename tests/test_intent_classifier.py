"""Tests for src/mtss/rag/intent_classifier.py.

The classifier is the router that replaces the gpt-5-mini tool-call
discretion observed on baseline-01 (which skipped search on 31/37
factual questions). These tests cover:

- Response parsing (happy path, markdown fences, surrounding prose,
  malformed JSON, unknown intent strings, out-of-range confidence)
- Fall-open behavior (any LLM error → ``factual_query`` with conf=0)
- ``is_fresh_user_turn`` / ``latest_user_text`` message-history helpers
  that gate whether the classifier is invoked at all
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mtss.rag.intent_classifier import (
    IntentClassifier,
    IntentResult,
    QueryIntent,
    is_fresh_user_turn,
    latest_user_text,
)


def _mock_response(content: str) -> MagicMock:
    """Build a minimal litellm response shape."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestParser:
    def test_plain_json(self):
        r = IntentClassifier._parse(
            '{"intent": "factual_query", "confidence": 0.9, "reasoning": "lookup"}'
        )
        assert r.intent is QueryIntent.FACTUAL_QUERY
        assert r.confidence == 0.9
        assert r.reasoning == "lookup"

    def test_markdown_fence(self):
        r = IntentClassifier._parse(
            '```json\n{"intent": "greeting", "confidence": 1.0}\n```'
        )
        assert r.intent is QueryIntent.GREETING

    def test_surrounding_prose(self):
        r = IntentClassifier._parse(
            'Here you go: {"intent": "off_topic", "confidence": 0.5} ok?'
        )
        assert r.intent is QueryIntent.OFF_TOPIC

    def test_malformed_json_falls_open(self):
        r = IntentClassifier._parse("not json at all")
        assert r.intent is QueryIntent.FACTUAL_QUERY
        assert r.confidence == 0.0

    def test_unknown_intent_falls_open(self):
        r = IntentClassifier._parse('{"intent": "nonsense", "confidence": 0.8}')
        assert r.intent is QueryIntent.FACTUAL_QUERY

    def test_confidence_clamped(self):
        r = IntentClassifier._parse(
            '{"intent": "exploratory", "confidence": 5.0}'
        )
        assert r.confidence == 1.0

    def test_missing_confidence_defaults_to_zero(self):
        r = IntentClassifier._parse('{"intent": "exploratory"}')
        assert r.intent is QueryIntent.EXPLORATORY
        assert r.confidence == 0.0

    def test_empty_response_falls_open(self):
        r = IntentClassifier._parse("")
        assert r.intent is QueryIntent.FACTUAL_QUERY


class TestClassify:
    @pytest.mark.asyncio
    async def test_happy_path_factual(self):
        classifier = IntentClassifier(model="test/model")
        with patch(
            "mtss.rag.intent_classifier.acompletion",
            new=AsyncMock(
                return_value=_mock_response(
                    '{"intent": "factual_query", "confidence": 0.95}'
                )
            ),
            create=True,
        ):
            # litellm is imported lazily inside classify(), so patch via
            # sys.modules before the call.
            import sys
            from unittest.mock import MagicMock as _MM

            fake_litellm = _MM()
            fake_litellm.acompletion = AsyncMock(
                return_value=_mock_response(
                    '{"intent": "factual_query", "confidence": 0.95}'
                )
            )
            sys.modules["litellm"] = fake_litellm
            result = await classifier.classify("Show me PSC deficiencies")
        assert result.intent is QueryIntent.FACTUAL_QUERY
        assert result.should_search is True

    @pytest.mark.asyncio
    async def test_llm_failure_falls_open(self):
        """Any exception from the LLM call → factual_query (never block search)."""
        classifier = IntentClassifier(model="test/model")
        import sys
        from unittest.mock import MagicMock as _MM

        fake_litellm = _MM()
        fake_litellm.acompletion = AsyncMock(side_effect=RuntimeError("boom"))
        sys.modules["litellm"] = fake_litellm

        result = await classifier.classify("any query")
        assert result.intent is QueryIntent.FACTUAL_QUERY
        assert result.confidence == 0.0
        assert "RuntimeError" in (result.reasoning or "")

    @pytest.mark.asyncio
    async def test_empty_input_treated_as_greeting(self):
        classifier = IntentClassifier(model="test/model")
        result = await classifier.classify("   ")
        assert result.intent is QueryIntent.GREETING


class TestHelpers:
    def test_fresh_turn_true_for_lone_human(self):
        assert is_fresh_user_turn([HumanMessage(content="hello")]) is True

    def test_fresh_turn_true_after_prior_answer(self):
        msgs = [
            HumanMessage(content="first"),
            AIMessage(content="answer"),
            HumanMessage(content="follow-up"),
        ]
        assert is_fresh_user_turn(msgs) is True

    def test_fresh_turn_false_mid_tool_call(self):
        """AIMessage with tool_calls means we're awaiting a ToolMessage, not a user turn."""
        ai_with_tool = AIMessage(
            content="", tool_calls=[{"id": "t1", "name": "search_documents", "args": {}}]
        )
        msgs = [HumanMessage(content="q"), ai_with_tool, HumanMessage(content="x")]
        assert is_fresh_user_turn(msgs) is False

    def test_fresh_turn_false_when_last_is_tool_message(self):
        msgs = [
            HumanMessage(content="q"),
            ToolMessage(content="{}", tool_call_id="t1"),
        ]
        assert is_fresh_user_turn(msgs) is False

    def test_fresh_turn_false_for_empty(self):
        assert is_fresh_user_turn([]) is False

    def test_latest_user_text_returns_last_human(self):
        msgs = [
            HumanMessage(content="first"),
            AIMessage(content="answer"),
            HumanMessage(content="second"),
        ]
        assert latest_user_text(msgs) == "second"

    def test_latest_user_text_handles_list_content(self):
        msgs = [HumanMessage(content=[{"type": "text", "text": "hi"}, "world"])]
        assert latest_user_text(msgs) == "hiworld"

    def test_latest_user_text_none_if_no_human(self):
        assert latest_user_text([AIMessage(content="only ai")]) is None


class TestIntentResult:
    def test_should_search_only_on_factual(self):
        assert IntentResult(QueryIntent.FACTUAL_QUERY, 0.9).should_search is True
        assert IntentResult(QueryIntent.EXPLORATORY, 0.9).should_search is False
        assert IntentResult(QueryIntent.OFF_TOPIC, 0.9).should_search is False
        assert IntentResult(QueryIntent.GREETING, 0.9).should_search is False

"""LangChain callback for capturing token + cost stats during eval runs.

Production code uses Langfuse for this — but eval needs the numbers in-process
so they can be written to results.jsonl without round-tripping the Langfuse
API.
"""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class TokenCounterCallback(AsyncCallbackHandler):
    """Accumulates input/output tokens + estimated cost across all LLM calls.

    Reads usage from ``LLMResult.llm_output['token_usage']`` (OpenAI-format)
    and ``response_metadata`` on individual generations (LangChain v0.2+).
    """

    # Approximate per-1M-token pricing — only used when the LLM doesn't return
    # cost directly. Production cost data comes from Langfuse; this is a
    # local fallback so eval reports always have a number.
    PRICING_PER_M = {
        "openai/gpt-5-nano": (0.05, 0.40),
        "openai/gpt-5-mini": (0.25, 2.00),
        "openai/gpt-5": (1.25, 10.00),
        "anthropic/claude-haiku-4.5": (1.00, 5.00),
        "anthropic/claude-sonnet-4.6": (3.00, 15.00),
        "anthropic/claude-opus-4.7": (15.00, 75.00),
    }

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.tool_calls = 0
        self.calls: List[Dict[str, Any]] = []

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        token_usage: Dict[str, int] = {}
        model = ""
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {}) or {}
            model = response.llm_output.get("model_name", "")

        # Fallback to per-generation usage_metadata
        if not token_usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is None:
                        continue
                    usage = getattr(msg, "usage_metadata", None) or {}
                    if usage:
                        token_usage = {
                            "prompt_tokens": usage.get("input_tokens", 0),
                            "completion_tokens": usage.get("output_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                        break

        prompt = int(token_usage.get("prompt_tokens", 0))
        completion = int(token_usage.get("completion_tokens", 0))
        self.input_tokens += prompt
        self.output_tokens += completion

        cost = self._estimate_cost(model, prompt, completion)
        self.cost_usd += cost
        self.calls.append({
            "model": model,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cost_usd": cost,
        })

    async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self.tool_calls += 1

    @classmethod
    def _estimate_cost(cls, model: str, prompt: int, completion: int) -> float:
        if not model:
            return 0.0
        # Strip openrouter/ prefix if present
        key = model.removeprefix("openrouter/")
        rates = cls.PRICING_PER_M.get(key)
        if not rates:
            return 0.0
        in_rate, out_rate = rates
        return (prompt * in_rate + completion * out_rate) / 1_000_000

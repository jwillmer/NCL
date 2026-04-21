"""LLM-as-judge for answer quality.

Single combined call per question — returns all four criteria at once for cost
efficiency. Results are cached by SHA256 of (question + response + reference +
judge_model + rubric_version), so re-judging an existing run is free.

Run separately from auto-graders: this is the only metric that costs money.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import litellm

from ..types import GoldenQuestion, JudgeScores, RunResult

logger = logging.getLogger(__name__)

# Bump RUBRIC_VERSION whenever the prompt or score schema changes — invalidates
# the cache without needing to delete .judge_cache/ manually.
RUBRIC_VERSION = "v1"

JUDGE_PROMPT = """You are an expert evaluator scoring an AI assistant's answer to a maritime technical-support question.

You will be given:
1. The QUESTION asked.
2. The REFERENCE_ANSWER — a curated good answer.
3. EXPECTED_FACTS — atomic claims the answer should cover.
4. The CANDIDATE_RESPONSE produced by the assistant under test.
5. The RETRIEVED_CONTEXT — the source chunks the assistant had access to (each prefixed with its chunk_id).

Score the CANDIDATE_RESPONSE on four 1-5 criteria. Be strict — 5 means "essentially as good as the reference", 1 means "fundamentally wrong or empty".

CRITERIA:
- faithfulness (1-5): Every factual claim is supported by the RETRIEVED_CONTEXT. No fabrication, no hallucination. Cited [C:chunk_id] markers must point to chunks that actually contain the claim.
- completeness (1-5): Covers the EXPECTED_FACTS. Missing or skipping key facts lowers the score. Extra valid facts beyond the expected list do not lower it.
- relevance (1-5): Directly answers the question. Padding, off-topic preamble, or excessive hedging lowers the score.
- actionability (1-5): When the question implies "what should I do" / "how was this resolved", concrete actionable steps are present. For purely informational questions, score 5 if the information is clearly presented.

Also produce:
- missing_facts: list of items from EXPECTED_FACTS that are missing or only partially addressed in the response.
- fabricated_claims: list of factual claims in the response that are NOT supported by the RETRIEVED_CONTEXT (citation pointing at the wrong chunk also counts).
- rationale: one short paragraph (under 80 words) explaining the scores.

Respond with ONLY a JSON object matching this exact schema:
{{
  "faithfulness": <int 1-5>,
  "completeness": <int 1-5>,
  "relevance": <int 1-5>,
  "actionability": <int 1-5>,
  "missing_facts": [<string>, ...],
  "fabricated_claims": [<string>, ...],
  "rationale": <string>
}}

QUESTION:
{question}

REFERENCE_ANSWER:
{reference_answer}

EXPECTED_FACTS:
{expected_facts}

RETRIEVED_CONTEXT:
{retrieved_context}

CANDIDATE_RESPONSE:
{candidate_response}
"""


def _cache_dir() -> Path:
    root = Path(os.environ.get("MTSS_EVAL_CACHE_DIR", "tests/eval/.judge_cache"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(
    question: GoldenQuestion,
    run: RunResult,
    judge_model: str,
) -> str:
    payload = json.dumps({
        "qid": question.id,
        "question": question.question,
        "reference": question.reference_answer,
        "facts": sorted(question.expected_facts),
        "response": run.response,
        "retrieval_ids": [c.chunk_id for c in run.retrieval],
        "judge_model": judge_model,
        "rubric": RUBRIC_VERSION,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached(key: str) -> Optional[JudgeScores]:
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        return JudgeScores.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cached judge result %s unreadable: %s", key, exc)
        return None


def _save_cached(key: str, scores: JudgeScores) -> None:
    path = _cache_dir() / f"{key}.json"
    path.write_text(scores.model_dump_json(indent=2), encoding="utf-8")


def _build_retrieved_context(run: RunResult, max_chunks: int = 10) -> str:
    """Render the top retrieved chunks the same way the agent saw them."""
    if not run.retrieval:
        return "(no chunks retrieved)"
    blocks = []
    for chunk in run.retrieval[:max_chunks]:
        header = f"[C:{chunk.chunk_id}] (rank={chunk.rank}, score={chunk.score:.3f}"
        if chunk.rerank_score is not None:
            header += f", rerank={chunk.rerank_score:.3f}"
        if chunk.email_subject:
            header += f", subject={chunk.email_subject!r}"
        header += ")"
        blocks.append(f"{header}\n{chunk.text_preview}")
    return "\n\n---\n\n".join(blocks)


async def judge(
    question: GoldenQuestion,
    run: RunResult,
    *,
    judge_model: Optional[str] = None,
    use_cache: bool = True,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> JudgeScores:
    """Score one (question, response) pair using an LLM judge.

    Args:
        question: Golden question with reference answer + expected facts.
        run: The agent's run result for this question.
        judge_model: LiteLLM model id. Defaults to $EVAL_JUDGE_MODEL or
            $LLM_MODEL.
        use_cache: If True (default) skip the LLM call when a cached score
            exists for the exact (question, response, judge_model, rubric).
        api_key / api_base: Override OpenRouter credentials. Defaults to the
            settings already loaded into env.

    Returns:
        JudgeScores. On hard parse failure, returns 1s across the board with
        an explanatory rationale and ``judge_cost_usd=0`` (no retry — the
        orchestrator can decide whether to mark the run as judge-failed).
    """
    model = judge_model or os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("No judge model configured (set EVAL_JUDGE_MODEL or LLM_MODEL)")

    key = _cache_key(question, run, model)
    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            logger.debug("judge cache hit for %s", question.id)
            return cached

    prompt = JUDGE_PROMPT.format(
        question=question.question,
        reference_answer=question.reference_answer,
        expected_facts="\n".join(f"- {f}" for f in question.expected_facts) or "(none provided)",
        retrieved_context=_build_retrieved_context(run),
        candidate_response=run.response or "(empty response)",
    )

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        resp = await litellm.acompletion(**kwargs)
    except Exception as exc:
        logger.exception("Judge LLM call failed for %s", question.id)
        return JudgeScores(
            model=model,
            faithfulness=1, completeness=1, relevance=1, actionability=1,
            rationale=f"Judge call failed: {exc}",
        )

    text = resp.choices[0].message.content or "{}"
    parsed = _parse_judge_json(text)
    cost = float(getattr(resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0)

    scores = JudgeScores(
        model=model,
        faithfulness=int(parsed.get("faithfulness", 1)),
        completeness=int(parsed.get("completeness", 1)),
        relevance=int(parsed.get("relevance", 1)),
        actionability=int(parsed.get("actionability", 1)),
        missing_facts=list(parsed.get("missing_facts", []) or []),
        fabricated_claims=list(parsed.get("fabricated_claims", []) or []),
        rationale=str(parsed.get("rationale", ""))[:1000],
        judge_cost_usd=cost,
    )
    _save_cached(key, scores)
    return scores


def _parse_judge_json(text: str) -> dict:
    """Tolerant JSON parser — strips code fences if the model added them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    # Find first { and last } for further tolerance
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Judge JSON parse failed: %s\nRaw: %s", exc, text[:500])
        return {}

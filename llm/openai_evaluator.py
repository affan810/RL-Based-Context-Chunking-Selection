"""
ChatGPT-based answer evaluator.

Given: question, golden answer, model answer
Returns: quality score [0, 1] + detailed rubric breakdown.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional

import config as cfg


@dataclass
class EvalResult:
    score: float               # [0, 1] composite
    correctness: float         # [0, 1]
    completeness: float        # [0, 1]
    factual_alignment: float   # [0, 1]
    reasoning: str             # GPT explanation
    raw_response: str          # full GPT output for debugging


_EVAL_SYSTEM = """\
You are an expert evaluator for question-answering systems.
You will be given:
- A question
- A golden (reference) answer
- A model-generated answer

Evaluate the model answer on three dimensions, each scored 0–10:
1. Correctness: Is the answer factually accurate compared to the golden answer?
2. Completeness: Does the answer cover all key points from the golden answer?
3. Factual alignment: Are all stated facts consistent with the golden answer?

Respond ONLY with valid JSON in this exact format:
{
  "correctness": <int 0-10>,
  "completeness": <int 0-10>,
  "factual_alignment": <int 0-10>,
  "reasoning": "<brief explanation>"
}
"""

_EVAL_USER_TEMPLATE = """\
Question: {question}

Golden Answer: {golden}

Model Answer: {model_answer}
"""


def evaluate_answer(
    question: str,
    golden_answer: str,
    model_answer: str,
    api_key: str = cfg.OPENAI_API_KEY,
    model: str = cfg.OPENAI_MODEL,
) -> EvalResult:
    """Call ChatGPT to evaluate the model answer against the golden answer."""

    if not api_key:
        return _heuristic_fallback(question, golden_answer, model_answer)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        user_msg = _EVAL_USER_TEMPLATE.format(
            question=question,
            golden=golden_answer,
            model_answer=model_answer,
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or "{}"
        return _parse_response(raw)

    except Exception as e:
        # Fallback to heuristic if API fails
        return _heuristic_fallback(question, golden_answer, model_answer)


def _parse_response(raw: str) -> EvalResult:
    try:
        data = json.loads(raw)
        correctness = data.get("correctness", 5) / 10
        completeness = data.get("completeness", 5) / 10
        factual = data.get("factual_alignment", 5) / 10
        score = (correctness + completeness + factual) / 3
        return EvalResult(
            score=round(score, 4),
            correctness=round(correctness, 4),
            completeness=round(completeness, 4),
            factual_alignment=round(factual, 4),
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
        )
    except Exception:
        return EvalResult(
            score=0.5, correctness=0.5, completeness=0.5,
            factual_alignment=0.5, reasoning="Parse error", raw_response=raw,
        )


def _heuristic_fallback(question: str, golden: str, model_answer: str) -> EvalResult:
    """Token-overlap heuristic when OpenAI is unavailable."""
    g_terms = set(golden.lower().split())
    m_terms = set(model_answer.lower().split())
    if not g_terms:
        score = 0.0
    else:
        overlap = len(g_terms & m_terms) / len(g_terms)
        score = min(overlap, 1.0)

    return EvalResult(
        score=round(score, 4),
        correctness=round(score, 4),
        completeness=round(score, 4),
        factual_alignment=round(score, 4),
        reasoning="Heuristic token-overlap (no API key)",
        raw_response="",
    )

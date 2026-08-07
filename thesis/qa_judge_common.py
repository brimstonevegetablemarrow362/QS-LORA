"""Shared prompts and parsing for LLM-as-judge QA scoring."""

from __future__ import annotations

import json
import re
from typing import Any

JUDGE_PROMPT_VERSION = "qa_judge_rubric/v2"

# Eval predictions: judge model output against human gold + context (Bedrock eval runs).
JUDGE_PROMPT_VERSION_EVAL = "qa_judge_rubric/v3_eval_gold"
# Same rubric; gold/prediction block order swapped and scores averaged (position debias).
JUDGE_PROMPT_VERSION_EVAL_DEBIAS = "qa_judge_rubric/v3_eval_gold_debias"

# Listwise ranking: rank all model answers for one question (Bedrock eval runs).
LISTWISE_RANK_PROMPT_VERSION = "qa_listwise_rank/v1"

JUDGE_SYSTEM = (
    "You are an expert evaluator for document-grounded question answering. "
    "Score only from the provided context. Respond with a single JSON object, no markdown."
)

JUDGE_EVAL_SYSTEM = (
    "You are a strict, conservative evaluator for document-grounded question answering. "
    "Use ONLY the provided context, the human gold reference answer, and the model prediction. "
    "Do not rely on outside knowledge, assumptions, or guesswork. "
    "A score of 5 means the prediction is essentially perfect for that dimension; "
    "a score of 3 means clearly imperfect but still partially useful; "
    "a score of 1 means clearly wrong, off-topic, or unsupported. "
    "Err on the side of LOWER scores when in doubt. "
    "Respond with a single JSON object, no markdown."
)

JUDGE_USER_TEMPLATE = """Context:
{context}

Question:
{question}

Answer to evaluate:
{answer}

Score the answer on four dimensions (integers 1-5):

grounding: Is the answer supported by this specific context?
  5 = every claim is directly in the context
  3 = mostly supported, minor extrapolation
  1 = contradicts context or ignores it entirely

relevance: Does the answer directly address what was asked?
  5 = directly and completely answers the question
  3 = partially answers the question
  1 = off-topic or evasive

document_necessity: Would someone NEED this specific document to answer correctly?
  5 = impossible to answer without reading this document (specific facts, numbers, names, procedures)
  3 = document helps but general knowledge could partially answer
  1 = pure general knowledge, document is irrelevant

overall: Is this a valuable training example for document-specific QA?
  A high score requires ALL of: grounded in context, answers the question,
  AND requires the document. Generic Q/A pairs should score low even if correct.
  5 = excellent training example, teaches document-specific knowledge
  3 = mediocre, some training value
  1 = no training value, model already knows this

Set refuse_expected to true ONLY if the question cannot be answered from the context.

Respond with JSON only:
{{"grounding": <1-5>, "relevance": <1-5>, "document_necessity": <1-5>, "overall": <1-5>, "refuse_expected": <bool>, "brief_reason": "<one sentence>"}}
"""

JUDGE_EVAL_USER_TEMPLATE = """Context:
{context}

Question:
{question}

Gold reference answer (human benchmark — correct target for this question):
{gold}

Model prediction to evaluate:
{prediction}

Score the **model prediction** on five dimensions (integers 1-5):

grounding: Is the prediction supported by this specific context?
  5 = every claim is directly in the context
  3 = mostly supported, minor extrapolation
  1 = contradicts context or ignores it entirely

relevance: Does the prediction address the question as well as the gold answer does?
  5 = directly answers; same intent as gold
  3 = partially answers or misses nuance vs gold
  1 = off-topic or evasive

gold_alignment: Semantic match between prediction and gold reference
  5 = equivalent meaning to gold, including all critical facts (dates, numbers, entities); wording may differ
  3 = partially matches gold (some overlap or partial fact), but misses or alters important details
  1 = wrong vs gold, or gold says "not found" but prediction invents an answer (or vice versa), or contradicts gold

document_necessity: Would someone NEED this specific document to answer correctly?
  5 = impossible without this document
  3 = document helps; partial general knowledge possible
  1 = pure general knowledge

overall: Overall quality of the prediction for this benchmark item
  Consider grounding, match to gold, and completeness. Penalize hallucinations, contradictions, and unnecessary verbosity.
  5 = excellent and reliable; 3 = acceptable but clearly flawed; 1 = poor or misleading

Set refuse_expected to true ONLY if the gold says the answer is not in the document
AND the prediction should also refuse (not invent facts).

Respond with JSON only:
{{"grounding": <1-5>, "relevance": <1-5>, "gold_alignment": <1-5>, "document_necessity": <1-5>, "overall": <1-5>, "refuse_expected": <bool>, "brief_reason": "<one sentence>"}}
"""

JUDGE_EVAL_USER_TEMPLATE_PRED_FIRST = """Context:
{context}

Question:
{question}

Model prediction to evaluate:
{prediction}

Gold reference answer (human benchmark — correct target for this question):
{gold}

Score the **model prediction** on five dimensions (integers 1-5):

grounding: Is the prediction supported by this specific context?
  5 = every claim is directly in the context
  3 = mostly supported, minor extrapolation
  1 = contradicts context or ignores it entirely

relevance: Does the prediction address the question as well as the gold answer does?
  5 = directly answers; same intent as gold
  3 = partially answers or misses nuance vs gold
  1 = off-topic or evasive

gold_alignment: Semantic match between prediction and gold reference
  5 = equivalent meaning to gold, including all critical facts (dates, numbers, entities); wording may differ
  3 = partially matches gold (some overlap or partial fact), but misses or alters important details
  1 = wrong vs gold, or gold says "not found" but prediction invents an answer (or vice versa), or contradicts gold

document_necessity: Would someone NEED this specific document to answer correctly?
  5 = impossible without this document
  3 = document helps; partial general knowledge possible
  1 = pure general knowledge

overall: Overall quality of the prediction for this benchmark item
  Consider grounding, match to gold, and completeness. Penalize hallucinations, contradictions, and unnecessary verbosity.
  5 = excellent and reliable; 3 = acceptable but clearly flawed; 1 = poor or misleading

Set refuse_expected to true ONLY if the gold says the answer is not in the document
AND the prediction should also refuse (not invent facts).

Respond with JSON only:
{{"grounding": <1-5>, "relevance": <1-5>, "gold_alignment": <1-5>, "document_necessity": <1-5>, "overall": <1-5>, "refuse_expected": <bool>, "brief_reason": "<one sentence>"}}
"""

JUDGE_SCORE_KEYS = ("grounding", "relevance", "gold_alignment", "document_necessity", "overall")

LISTWISE_RANK_SYSTEM = (
    "You are a strict comparative evaluator for document-grounded question answering. "
    "Rank anonymous candidate answers from BEST to WORST for the question, using ONLY the context, "
    "the human gold reference, and the candidates. "
    "Prefer answers that match gold in meaning, are grounded in the context, and refuse when gold refuses. "
    "Penalize hallucinations, contradictions, missing critical facts, and unnecessary verbosity. "
    "Respond with a single JSON object, no markdown."
)

LISTWISE_RANK_USER_TEMPLATE = """Context:
{context}

Question:
{question}

Gold reference answer (human benchmark):
{gold}

Candidate answers (anonymous labels — order is arbitrary, do not favor earlier labels):
{candidates_block}

Task: Rank ALL {n_candidates} candidates from best (1) to worst ({n_candidates}) relative to the gold answer and context.

Rules:
- Each label must appear exactly once in "ranking" (best first).
- Rank 1 = best answer; rank {n_candidates} = worst.
- If gold says the answer is not in the document, prefer candidates that refuse rather than invent facts.

Respond with JSON only:
{{"ranking": ["<label_best>", "<label_2nd>", ...], "brief_reason": "<one sentence>"}}
"""


def rank_to_points(rank: int, n_candidates: int) -> int:
    """Convert 1-based rank to points: best=9 when n=8, worst=1 (points = (n+1) - rank)."""
    if rank < 1 or rank > n_candidates:
        raise ValueError(f"rank {rank} out of range 1..{n_candidates}")
    return (n_candidates + 1) - rank


def is_nan_answer(answer: str) -> bool:
    """True for failed generations (nan/none/empty) — skip API judging."""
    return str(answer).strip().lower() in ("nan", "none", "")


def skipped_nan_judge_block(*, provider: str, model: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "skipped": True,
        "skip_reason": "nan_answer",
        "quality_tier": "drop",
    }


def build_judge_user_message(
    *,
    context: str,
    question: str,
    answer: str,
    max_context_chars: int,
    gold: str | None = None,
    gold_first: bool = True,
) -> str:
    """Build judge user prompt. If ``gold`` is set, use eval rubric (pred vs gold + context)."""
    ctx = context[:max_context_chars] if max_context_chars > 0 else context
    if gold is not None:
        tmpl = JUDGE_EVAL_USER_TEMPLATE if gold_first else JUDGE_EVAL_USER_TEMPLATE_PRED_FIRST
        return tmpl.format(
            context=ctx,
            question=question,
            gold=gold,
            prediction=answer,
        )
    return JUDGE_USER_TEMPLATE.format(context=ctx, question=question, answer=answer)


def merge_position_swap_judge_blocks(
    block_gold_first: dict[str, Any],
    block_pred_first: dict[str, Any],
    *,
    answer: str,
) -> dict[str, Any]:
    """Average numeric scores from gold-first and prediction-first eval judge passes."""
    if block_gold_first.get("error") or block_pred_first.get("error"):
        return block_gold_first if not block_gold_first.get("error") else block_pred_first

    merged: dict[str, Any] = {
        "provider": block_gold_first.get("provider"),
        "model": block_gold_first.get("model"),
        "prompt_version": JUDGE_PROMPT_VERSION_EVAL_DEBIAS,
        "position_swap_debias": True,
        "pass_gold_first": {k: block_gold_first.get(k) for k in JUDGE_SCORE_KEYS},
        "pass_pred_first": {k: block_pred_first.get(k) for k in JUDGE_SCORE_KEYS},
    }
    for key in JUDGE_SCORE_KEYS:
        a = block_gold_first.get(key)
        b = block_pred_first.get(key)
        if a is not None and b is not None:
            merged[key] = round((float(a) + float(b)) / 2.0, 4)
    merged["refuse_expected"] = bool(
        block_gold_first.get("refuse_expected") or block_pred_first.get("refuse_expected")
    )
    reasons = [
        str(block_gold_first.get("brief_reason") or "").strip(),
        str(block_pred_first.get("brief_reason") or "").strip(),
    ]
    merged["brief_reason"] = " | ".join(r for r in reasons if r)[:500]
    if block_gold_first.get("gold_reference") is not None:
        merged["gold_reference"] = block_gold_first["gold_reference"]
    g = merged.get("grounding")
    r = merged.get("relevance")
    dn = merged.get("document_necessity")
    o = merged.get("overall")
    if g is not None and r is not None and dn is not None and o is not None:
        merged["quality_tier"] = quality_tier_from_scores(
            grounding=int(round(g)),
            relevance=int(round(r)),
            document_necessity=int(round(dn)),
            overall=int(round(o)),
            answer=answer,
        )
    return merged


def judge_prompt_version(*, gold: str | None) -> str:
    return JUDGE_PROMPT_VERSION_EVAL if gold is not None else JUDGE_PROMPT_VERSION


def build_listwise_rank_user_message(
    *,
    context: str,
    question: str,
    gold: str,
    labeled_preds: list[tuple[str, str]],
    max_context_chars: int,
    max_pred_chars: int,
) -> str:
    ctx = context[:max_context_chars] if max_context_chars > 0 else context
    lines: list[str] = []
    for label, pred in labeled_preds:
        p = pred.strip()
        if max_pred_chars > 0 and len(p) > max_pred_chars:
            p = p[:max_pred_chars] + "…"
        lines.append(f"[{label}]\n{p}")
    n = len(labeled_preds)
    return LISTWISE_RANK_USER_TEMPLATE.format(
        context=ctx,
        question=question,
        gold=gold,
        candidates_block="\n\n".join(lines),
        n_candidates=n,
    )


def parse_listwise_rank_json(raw: str, valid_labels: set[str]) -> dict[str, Any] | None:
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\"ranking\"[\s\S]*\}", s)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    ranking = obj.get("ranking")
    if not isinstance(ranking, list):
        return None
    labels = [str(x).strip() for x in ranking]
    if set(labels) != valid_labels or len(labels) != len(valid_labels):
        return None
    return obj


def parse_judge_json(raw: str) -> dict[str, Any] | None:
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "overall" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\"overall\"[\s\S]*\}", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def quality_tier_from_scores(
    *,
    grounding: int,
    relevance: int,
    document_necessity: int,
    overall: int,
    answer: str,
    high_min: int = 4,
    low_max: int = 2,
) -> str:
    if str(answer).strip().lower() in ("nan", "none", ""):
        return "drop"

    # Hard drop if grounding is very low (contradicts document)
    if grounding <= low_max:
        return "drop"

    scores = (grounding, relevance, document_necessity, overall)

    if min(scores) <= low_max:
        return "low"
    if min(scores) >= high_min:
        return "high"
    return "medium"


def coerce_judge_score(value: Any) -> int | None:
    """Parse a 1–5 rubric score; return None if value is missing or not coercible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return v if 1 <= v <= 5 else None
    s = str(value).strip()
    if not s:
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    return v if 1 <= v <= 5 else None


def normalize_judge_block(
    *,
    provider: str,
    model: str,
    parsed: dict[str, Any] | None,
    raw: str | None = None,
    error: str | None = None,
    answer: str = "",
    prompt_version: str | None = None,
    gold: str | None = None,
) -> dict[str, Any]:
    pv = prompt_version or judge_prompt_version(gold=gold)
    if error:
        out: dict[str, Any] = {
            "error": error,
            "provider": provider,
            "model": model,
            "prompt_version": pv,
        }
        if raw:
            out["raw_preview"] = raw[:400]
        return out
    if not parsed:
        return {
            "error": "parse_error",
            "provider": provider,
            "model": model,
            "prompt_version": pv,
            "raw_preview": (raw or "")[:400],
        }

    g = coerce_judge_score(parsed.get("grounding"))
    r = coerce_judge_score(parsed.get("relevance"))
    dn = coerce_judge_score(parsed.get("document_necessity"))
    o = coerce_judge_score(parsed.get("overall"))
    ga = coerce_judge_score(parsed.get("gold_alignment")) if "gold_alignment" in parsed else None
    if g is None or r is None or dn is None or o is None:
        return {
            "error": "parse_error",
            "provider": provider,
            "model": model,
            "prompt_version": pv,
            "raw_preview": (raw or "")[:400],
        }

    out_block: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "prompt_version": pv,
        "grounding": g,
        "relevance": r,
        "document_necessity": dn,
        "overall": o,
        "refuse_expected": bool(parsed.get("refuse_expected", False)),
        "brief_reason": str(parsed.get("brief_reason", ""))[:500],
        "quality_tier": quality_tier_from_scores(
            grounding=g,
            relevance=r,
            document_necessity=dn,
            overall=o,
            answer=answer,
        ),
    }
    if ga is not None:
        out_block["gold_alignment"] = ga
    if gold is not None:
        out_block["gold_reference"] = gold[:500]
    return out_block

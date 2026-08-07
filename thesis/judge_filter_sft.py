"""Shared helpers for training-filter judge distillation (qa_judge_rubric/v2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from thesis.qa_judge_common import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM,
    build_judge_user_message,
    is_nan_answer,
    quality_tier_from_scores,
)

DEFAULT_MAX_CONTEXT_CHARS = 8000


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_block(row: dict[str, Any]) -> dict[str, Any] | None:
    block = row.get("llm_judge")
    return block if isinstance(block, dict) else None


def is_v2_training_judge_row(row: dict[str, Any]) -> bool:
    """Usable distillation row: v2 rubric, scored, non-nan answer."""
    ans = str(row.get("answer") or "").strip()
    if not ans or is_nan_answer(ans):
        return False
    if not (row.get("context") or "").strip():
        return False
    if not (row.get("question") or "").strip():
        return False

    j = judge_block(row)
    if not j or j.get("skipped") or j.get("error"):
        return False
    pv = str(j.get("prompt_version") or "")
    if pv and pv != JUDGE_PROMPT_VERSION:
        return False
    if "gold_alignment" in j:
        return False
    for key in ("grounding", "relevance", "document_necessity", "overall"):
        if key not in j:
            return False
    return True


def teacher_scores(j: dict[str, Any]) -> dict[str, Any]:
    return {
        "grounding": int(j["grounding"]),
        "relevance": int(j["relevance"]),
        "document_necessity": int(j["document_necessity"]),
        "overall": int(j["overall"]),
        "refuse_expected": bool(j.get("refuse_expected", False)),
        "brief_reason": str(j.get("brief_reason") or "")[:500],
    }


def teacher_target_json(j: dict[str, Any]) -> str:
    return json.dumps(teacher_scores(j), ensure_ascii=False)


def teacher_quality_tier(j: dict[str, Any], answer: str) -> str:
    tier = j.get("quality_tier")
    if tier:
        return str(tier)
    s = teacher_scores(j)
    return quality_tier_from_scores(
        grounding=s["grounding"],
        relevance=s["relevance"],
        document_necessity=s["document_necessity"],
        overall=s["overall"],
        answer=answer,
    )


def row_to_judge_messages(
    row: dict[str, Any],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> list[dict[str, str]]:
    j = judge_block(row)
    if not j:
        raise ValueError("row missing llm_judge")
    user = build_judge_user_message(
        context=str(row.get("context") or ""),
        question=str(row.get("question") or ""),
        answer=str(row.get("answer") or ""),
        max_context_chars=max_context_chars,
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": teacher_target_json(j)},
    ]


def to_sft_row(row: dict[str, Any], *, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> dict[str, Any]:
    j = judge_block(row)
    assert j is not None
    ans = str(row.get("answer") or "").strip()
    return {
        "context": row.get("context"),
        "question": row.get("question"),
        "answer": ans,
        "source": row.get("source"),
        "document_id": row.get("document_id") or row.get("section_id"),
        "chunk_id": row.get("chunk_id"),
        "teacher_judge": j,
        "teacher_quality_tier": teacher_quality_tier(j, ans),
        "target_json": teacher_target_json(j),
        "messages": row_to_judge_messages(row, max_context_chars=max_context_chars),
    }

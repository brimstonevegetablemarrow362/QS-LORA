"""DROP validation eval helpers (multi-answer gold, context merge)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from thesis.qa_answer_metrics import REFUSAL_GOLD


def drop_gold_reference(row: dict[str, Any]) -> str:
    """Pick one human gold string for judge / logging (mode over DROP ``answers``)."""
    if row.get("unanswerable"):
        return REFUSAL_GOLD
    answers = row.get("answers")
    if isinstance(answers, list) and not answers:
        return REFUSAL_GOLD
    if isinstance(answers, list) and answers:
        texts: list[str] = []
        for a in answers:
            if isinstance(a, dict):
                t = str(a.get("text") or "").strip()
            else:
                t = str(a).strip()
            if t:
                texts.append(t)
        if texts:
            return Counter(texts).most_common(1)[0][0]
    return str(row.get("gold") or row.get("answer") or "").strip()


def load_eval_index(eval_jsonl: Path) -> dict[str, dict[str, Any]]:
    path = eval_jsonl.expanduser().resolve()
    index: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: {e}") from e
        key = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if key:
            index[key] = row
    return index


def enrich_rows_with_drop_eval(
    rows: list[dict[str, Any]],
    eval_index: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Attach context, answers, and canonical gold from DROP validation by eval_id."""
    n_merged = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        key = str(r.get("eval_id") or r.get("chunk_id") or "").strip()
        ref = eval_index.get(key) if key else None
        if ref is None:
            if not str(r.get("gold") or "").strip() and r.get("answers"):
                r["gold"] = drop_gold_reference(r)
            out.append(r)
            continue
        if overwrite or not str(r.get("context") or "").strip():
            ctx = str(ref.get("context") or "").strip()
            if ctx:
                r["context"] = ctx
                n_merged += 1
        if ref.get("answers") is not None:
            r["answers"] = ref["answers"]
        if overwrite or not str(r.get("gold") or "").strip():
            r["gold"] = drop_gold_reference(ref)
        if not str(r.get("question") or "").strip():
            r["question"] = ref.get("question")
        out.append(r)
    return out, n_merged

"""SQuAD-style answer metrics: exact match and token-level F1."""

from __future__ import annotations

import re
from collections import defaultdict

_INVALID_PRED = frozenset({"nan", "none", ""})

# Canonical refusal gold (RepLiQA + SQuAD 2.0 unanswerable); used by judge v3 rubric.
REFUSAL_GOLD = "The answer is not found in the document."


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def is_invalid_answer(s: str) -> bool:
    return normalize_text(s) in _INVALID_PRED


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_text(pred).split()
    gt = normalize_text(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pred_counts: dict[str, int] = defaultdict(int)
    gold_counts: dict[str, int] = defaultdict(int)
    for t in pt:
        pred_counts[t] += 1
    for t in gt:
        gold_counts[t] += 1
    common = 0
    for t, c in pred_counts.items():
        common += min(c, gold_counts.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(pt)
    recall = common / len(gt)
    return 2 * precision * recall / (precision + recall)


def is_refusal_gold(gold: str) -> bool:
    g = normalize_text(gold)
    return "not found in the document" in g or g in (
        "unanswerable",
        "not answerable",
        "cannot be answered from the document",
    )

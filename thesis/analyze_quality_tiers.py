"""
Characterize Haiku/Bedrock quality tiers (high / medium / low / drop) in judged SFT pools.

Usage:
  python -m thesis.cli analyze-quality-tiers --all
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.qa_answer_metrics import is_refusal_gold

THESIS_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THESIS_ROOT / "experiments"
SCHEMA = "thesis_quality_tier_analysis/v1"

DATASETS: dict[str, Path] = {
    "repliqa": EXPERIMENTS_ROOT
    / "repliqa/runs/repliqa_train_0-3/train/synthetic_qa_haiku_judge.jsonl",
    "quoref": EXPERIMENTS_ROOT
    / "quoref/runs/quoref_synthetic_deploy_v1/train/bedrock_judge.jsonl",
    "squad_v2": EXPERIMENTS_ROOT
    / "squad_v2/runs/squad_synthetic_deploy_v1/train/bedrock_judge.jsonl",
}

WH_WORDS = frozenset(
    {"what", "who", "whom", "whose", "when", "where", "why", "how", "which"}
)
YESNO_STARTERS = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "will",
        "would",
        "has",
        "have",
        "had",
        "am",
    }
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def question_starter(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "empty"
    first = re.match(r"^(\w+)", q)
    if not first:
        return "other"
    w = first.group(1)
    if w in WH_WORDS:
        return w
    if w in YESNO_STARTERS:
        return "yes_no"
    return "other"


def answer_type(row: dict[str, Any]) -> str:
    gold = str(row.get("answer") or "")
    if is_refusal_gold(gold):
        return "unanswerable_refusal"
    judge = row.get("llm_judge") or {}
    if judge.get("refuse_expected"):
        return "unanswerable_refusal"
    aw = word_count(gold)
    if aw <= 3:
        return "short_span"
    if re.fullmatch(r"[\d.,\-%+\s]+", gold.strip()):
        return "numeric"
    if aw <= 12 and not gold.strip().endswith("."):
        return "short_span"
    if aw <= 20:
        return "medium_span"
    return "explanatory"


def tier_of(row: dict[str, Any]) -> str:
    judge = row.get("llm_judge") or {}
    return str(judge.get("quality_tier") or "missing").lower()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def top_counter(counter: Counter[str], n: int = 8) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1
    return [
        {"label": k, "count": v, "pct": round(pct(v, total), 1)}
        for k, v in counter.most_common(n)
    ]


def analyze_dataset(name: str, path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[tier_of(row)].append(row)

    tier_stats: dict[str, Any] = {}
    order = ["high", "medium", "low", "drop", "missing"]
    tiers_present = [t for t in order if t in by_tier] + [
        t for t in sorted(by_tier) if t not in order
    ]

    for tier in tiers_present:
        tier_rows = by_tier[tier]
        n = len(tier_rows)
        if n == 0:
            continue

        q_words = [word_count(str(r.get("question") or "")) for r in tier_rows]
        a_words = [word_count(str(r.get("answer") or "")) for r in tier_rows]
        ctx_words = [word_count(str(r.get("context") or "")) for r in tier_rows]

        q_starters: Counter[str] = Counter()
        ans_types: Counter[str] = Counter()
        topics: Counter[str] = Counter()
        judge_dims: dict[str, list[int]] = defaultdict(list)

        for r in tier_rows:
            q_starters[question_starter(str(r.get("question") or ""))] += 1
            ans_types[answer_type(r)] += 1
            topic = str(r.get("document_topic") or r.get("source") or "unknown")
            topics[topic] += 1
            j = r.get("llm_judge") or {}
            for dim in ("grounding", "relevance", "document_necessity", "overall"):
                if dim in j and j[dim] is not None:
                    judge_dims[dim].append(int(j[dim]))

        tier_stats[tier] = {
            "n": n,
            "pct_of_pool": round(pct(n, len(rows)), 1),
            "question_words": {
                "mean": round(mean(q_words), 1),
                "median": sorted(q_words)[n // 2],
            },
            "answer_words": {
                "mean": round(mean(a_words), 1),
                "median": sorted(a_words)[n // 2],
            },
            "context_words": {
                "mean": round(mean(ctx_words), 1),
                "median": sorted(ctx_words)[n // 2],
            },
            "answer_type_distribution": top_counter(ans_types, n=6),
            "question_starter_distribution": top_counter(q_starters, n=10),
            "top_topics": top_counter(topics, n=5) if name == "repliqa" else [],
            "judge_score_means": {
                dim: round(mean(vals), 2) for dim, vals in judge_dims.items()
            },
            "refusal_gold_pct": round(
                pct(
                    sum(1 for r in tier_rows if is_refusal_gold(str(r.get("answer") or ""))),
                    n,
                ),
                1,
            ),
        }

    usable = sum(len(by_tier[t]) for t in ("high", "medium", "low"))
    return {
        "dataset": name,
        "judged_jsonl": str(path),
        "n_total": len(rows),
        "tier_counts": {t: len(by_tier[t]) for t in tiers_present},
        "n_usable_sft": usable,
        "usable_pct": round(pct(usable, len(rows)), 1),
        "merge_weights_default": {
            "high": round(len(by_tier["high"]) / usable, 3) if usable else 0,
            "medium": round(len(by_tier["medium"]) / usable, 3) if usable else 0,
            "low": round(len(by_tier["low"]) / usable, 3) if usable else 0,
        },
        "tiers": tier_stats,
    }


def run_all(*, output_path: Path | None = None) -> dict[str, Any]:
    out_path = output_path or (EXPERIMENTS_ROOT / "quality_tier_analysis.json")
    results = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tier_rule": (
            "drop: empty answer or grounding<=2; "
            "low: min(grounding,relevance,document_necessity,overall)<=2; "
            "high: all dims>=4; else medium"
        ),
        "datasets": [analyze_dataset(name, path) for name, path in DATASETS.items()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return results


def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "analyze-quality-tiers",
        help="Summarize judged SFT pool by quality tier (length, answer type, question words)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENTS_ROOT / "quality_tier_analysis.json",
    )
    p.add_argument("--all", action="store_true", help="Analyze RepLiQA, Quoref, SQuAD pools")

    def _run(ns: argparse.Namespace) -> int:
        run_all(output_path=ns.output)
        return 0

    p.set_defaults(fn=_run)


if __name__ == "__main__":
    run_all()

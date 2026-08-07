"""
Compile all triple hallucination/refusal proofs (Ours correct, B3+B5 wrong)
across reference + cross-model runs into one JSON catalog.

  python -m thesis.cli export-triple-hallucination-catalog
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.eval_export_hallucination_pack import (
    find_triple_judge_gap_cases,
    find_triple_refusal_cases,
    _load_jsonl_map,
    _judge_block,
)

THESIS_ROOT = Path(__file__).resolve().parent
DEFAULT_CROSS_ROOT = Path("/fs/ess/PAS2699/pratham2210/cross_model/runs")
SCHEMA = "triple_hallucination_catalog/v1"

MODEL_META: dict[str, dict[str, str]] = {
    "llama32_1b": {"family": "Llama", "size": "1B", "label": "Llama-3.2-1B"},
    "llama32_3b": {"family": "Llama", "size": "3B", "label": "Llama-3.2-3B"},
    "llama31_8b": {"family": "Llama", "size": "8B", "label": "Llama-3.1-8B"},
    "llama31_70b": {"family": "Llama", "size": "70B", "label": "Llama-3.1-70B"},
    "qwen25_3b": {"family": "Qwen2.5", "size": "3B", "label": "Qwen2.5-3B"},
    "qwen25_7b": {"family": "Qwen2.5", "size": "7B", "label": "Qwen2.5-7B"},
    "qwen25_14b": {"family": "Qwen2.5", "size": "14B", "label": "Qwen2.5-14B"},
    "gemma3_1b": {"family": "Gemma-3", "size": "1B", "label": "Gemma-3-1B"},
    "gemma3_4b": {"family": "Gemma-3", "size": "4B", "label": "Gemma-3-4B"},
    "gemma3_12b": {"family": "Gemma-3", "size": "12B", "label": "Gemma-3-12B"},
}

DATASET_CONDITIONS: dict[str, tuple[str, str, str]] = {
    "repliqa": ("B3_lora_all", "B5_adalora_all", "Ours_tier_merge"),
    "quoref": ("B3_lora_ctx", "B5_adalora_ctx", "Ours_tier_ctx"),
    "squad": ("B3_lora_ctx", "B5_adalora_ctx", "Ours_tier_ctx"),
}

REFERENCE_RUNS: list[tuple[str, str, Path, str]] = [
    ("llama32_3b", "repliqa", THESIS_ROOT / "experiments/repliqa/runs/repliqa_train_0-3", "reference"),
    ("llama32_3b", "quoref", THESIS_ROOT / "experiments/quoref/runs/quoref_qa_v1", "reference"),
    ("llama32_3b", "squad", THESIS_ROOT / "experiments/squad_v2/runs/squad_qa_v1", "reference"),
]

CROSS_DATASET_DIRS = {
    "repliqa": "repliqa",
    "quoref": "quoref_qa_v1",
    "squad": "squad_qa_v1",
}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pred_block(row: dict[str, Any], condition: str) -> dict[str, Any]:
    j = _judge_block(row)
    return {
        "condition": condition,
        "pred": row.get("pred"),
        "gold_alignment": j.get("gold_alignment"),
        "grounding": j.get("grounding"),
        "relevance": j.get("relevance"),
        "document_necessity": j.get("document_necessity"),
        "overall": j.get("overall"),
        "quality_tier": j.get("quality_tier"),
        "refusal_class": row.get("b3_refusal_class")
        or row.get("b5_refusal_class")
        or row.get("ours_refusal_class"),
        "judge_reason": j.get("brief_reason"),
    }


def _format_case(
    row: dict[str, Any],
    *,
    pattern: str,
    dataset: str,
    model_slug: str,
    run_root: Path,
    b3_condition: str,
    b5_condition: str,
    ours_condition: str,
    source: str,
) -> dict[str, Any]:
    meta = MODEL_META.get(model_slug, {"family": model_slug, "size": "?", "label": model_slug})
    eid = str(row["eval_id"])
    return {
        "case_id": f"{model_slug}/{dataset}/{pattern}/{eid}",
        "eval_id": eid,
        "dataset": dataset,
        "pattern": pattern,
        "model_slug": model_slug,
        "model_family": meta["family"],
        "model_size": meta["size"],
        "model_label": meta["label"],
        "source": source,
        "run_root": str(run_root),
        "question": row.get("question"),
        "gold": row.get("gold"),
        "context": row.get("context"),
        "B3": {
            "condition": b3_condition,
            "pred": row.get("b3_pred"),
            "gold_alignment": row.get("b3_gold_alignment"),
            "grounding": row.get("b3_grounding"),
            "refusal_class": row.get("b3_refusal_class"),
            "judge_reason": row.get("b3_judge_reason"),
        },
        "B5": {
            "condition": b5_condition,
            "pred": row.get("b5_pred"),
            "gold_alignment": row.get("b5_gold_alignment"),
            "grounding": row.get("b5_grounding"),
            "refusal_class": row.get("b5_refusal_class"),
            "judge_reason": row.get("b5_judge_reason"),
        },
        "Ours": {
            "condition": ours_condition,
            "pred": row.get("ours_pred"),
            "gold_alignment": row.get("ours_gold_alignment"),
            "grounding": row.get("ours_grounding"),
            "refusal_class": row.get("ours_refusal_class"),
            "judge_reason": row.get("ours_judge_reason"),
        },
    }


def _collect_run(
    *,
    model_slug: str,
    dataset: str,
    run_root: Path,
    source: str,
) -> list[dict[str, Any]]:
    b3_cond, b5_cond, ours_cond = DATASET_CONDITIONS[dataset]
    judged = run_root / "eval" / "judged"
    b3_path = judged / b3_cond / "bedrock_judge.jsonl"
    b5_path = judged / b5_cond / "bedrock_judge.jsonl"
    ours_path = judged / ours_cond / "bedrock_judge.jsonl"
    for p in (b3_path, b5_path, ours_path):
        if not p.is_file():
            return []

    b3_rows = _load_jsonl_map(b3_path)
    b5_rows = _load_jsonl_map(b5_path)
    ours_rows = _load_jsonl_map(ours_path)

    # Enrich triple rows with full context from judged files
    def enrich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            eid = row["eval_id"]
            base = ours_rows.get(eid) or b3_rows.get(eid) or {}
            merged = dict(row)
            merged["context"] = base.get("context")
            if not merged.get("question"):
                merged["question"] = base.get("question")
            if not merged.get("gold"):
                merged["gold"] = base.get("gold")
            out.append(merged)
        return out

    cases: list[dict[str, Any]] = []
    for pattern, finder in (
        ("refusal", find_triple_refusal_cases),
        ("judge_gap", find_triple_judge_gap_cases),
    ):
        raw = enrich(
            finder(b3_rows=b3_rows, b5_rows=b5_rows, ours_rows=ours_rows)
        )
        for row in raw:
            cases.append(
                _format_case(
                    row,
                    pattern=pattern,
                    dataset=dataset,
                    model_slug=model_slug,
                    run_root=run_root,
                    b3_condition=b3_cond,
                    b5_condition=b5_cond,
                    ours_condition=ours_cond,
                    source=source,
                )
            )
    return cases


def build_catalog(*, cross_root: Path = DEFAULT_CROSS_ROOT) -> dict[str, Any]:
    all_cases: list[dict[str, Any]] = []
    run_stats: list[dict[str, Any]] = []

    for model_slug, dataset, run_root, source in REFERENCE_RUNS:
        cases = _collect_run(
            model_slug=model_slug, dataset=dataset, run_root=run_root, source=source
        )
        refusal_n = sum(1 for c in cases if c["pattern"] == "refusal")
        gap_n = sum(1 for c in cases if c["pattern"] == "judge_gap")
        run_stats.append(
            {
                "model_slug": model_slug,
                "model_label": MODEL_META[model_slug]["label"],
                "dataset": dataset,
                "source": source,
                "n_refusal": refusal_n,
                "n_judge_gap": gap_n,
                "n_total": len(cases),
            }
        )
        all_cases.extend(cases)

    for model_slug in MODEL_META:
        if model_slug == "llama32_3b":
            continue
        for dataset, ds_dir in CROSS_DATASET_DIRS.items():
            run_root = cross_root / model_slug / ds_dir
            if not run_root.is_dir():
                continue
            cases = _collect_run(
                model_slug=model_slug, dataset=dataset, run_root=run_root, source="cross_model"
            )
            if not cases:
                continue
            refusal_n = sum(1 for c in cases if c["pattern"] == "refusal")
            gap_n = sum(1 for c in cases if c["pattern"] == "judge_gap")
            run_stats.append(
                {
                    "model_slug": model_slug,
                    "model_label": MODEL_META[model_slug]["label"],
                    "dataset": dataset,
                    "source": "cross_model",
                    "n_refusal": refusal_n,
                    "n_judge_gap": gap_n,
                    "n_total": len(cases),
                }
            )
            all_cases.extend(cases)

    by_model: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    by_pattern: dict[str, int] = {}
    for c in all_cases:
        by_model[c["model_label"]] = by_model.get(c["model_label"], 0) + 1
        by_dataset[c["dataset"]] = by_dataset.get(c["dataset"], 0) + 1
        by_pattern[c["pattern"]] = by_pattern.get(c["pattern"], 0) + 1

    return {
        "schema": SCHEMA,
        "created_at": utc_iso(),
        "description": (
            "Cases where Ours is correct (GA≥4 or proper refusal) but both B3 and B5 "
            "hallucinate or invent (GA≤2 on answerable gold, or invent on unanswerable)."
        ),
        "summary": {
            "n_cases_total": len(all_cases),
            "n_runs": len(run_stats),
            "by_model_label": dict(sorted(by_model.items())),
            "by_dataset": dict(sorted(by_dataset.items())),
            "by_pattern": dict(sorted(by_pattern.items())),
            "per_run": run_stats,
        },
        "cases": all_cases,
    }


def run_export_triple_hallucination_catalog(ns: argparse.Namespace) -> int:
    cross_root = Path(ns.cross_root).expanduser().resolve()
    out = Path(ns.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Building triple hallucination catalog...", flush=True)
    catalog = build_catalog(cross_root=cross_root)
    out.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"  cases={catalog['summary']['n_cases_total']}  "
        f"runs={catalog['summary']['n_runs']}"
    )
    return 0


def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export-triple-hallucination-catalog",
        help="JSON catalog of Ours-correct / B3+B5-wrong cases across all models",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=THESIS_ROOT / "experiments" / "analysis" / "triple_hallucination_catalog.json",
    )
    p.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT)
    p.set_defaults(fn=run_export_triple_hallucination_catalog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    raise SystemExit(parser.parse_args().fn(parser.parse_args()))

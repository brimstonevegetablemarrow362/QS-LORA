"""Abstention-controlled hallucination metrics from judged eval JSONLs.

Reframes hallucination as a selective-prediction problem. For every judged
response we decide whether the model *answered* or *abstained* (via the shared
`_classify_refusal` heuristic), then use the judge's `grounding` dimension
(1-5) to decide whether an answer is supported by the context.

Per (model, dataset, arm) we report:
  - UAR  (Unsupported-Answer Rate) = ungrounded answers / answered rows
        -> "how often it fabricates when it chooses to speak" (abstention-invariant)
  - coverage (answer rate)         = answered rows / total rows
  - correct_refusal (unans rows)   = abstained / unanswerable rows
  - over_refusal   (ans rows)      = abstained / answerable rows
  - grounded_accuracy (ans rows)   = grounded answers / answerable rows (context health)

Nothing is re-judged; this only re-aggregates existing bedrock_judge.jsonl files.

  python -m thesis.compute_grounded_hallucination
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.eval_export_hallucination_pack import (
    _classify_refusal,
    _judge_block,
    _load_jsonl_map,
)
from thesis.export_triple_hallucination_catalog import (
    CROSS_DATASET_DIRS,
    DATASET_CONDITIONS,
    DEFAULT_CROSS_ROOT,
    MODEL_META,
    REFERENCE_RUNS,
)
from thesis.qa_answer_metrics import is_refusal_gold

THESIS_ROOT = Path(__file__).resolve().parent
SCHEMA = "grounded_hallucination/v1"
ARMS = ("B3", "B5", "Ours")
MODEL_ORDER = [
    "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Llama-3.1-70B",
    "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B",
    "Gemma-3-1B", "Gemma-3-4B", "Gemma-3-12B",
]

# _classify_refusal buckets that mean "the model committed to an answer".
_ANSWERED_CLASSES = {"invented_answer", "context_dump"}


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _answered(pred: str) -> bool:
    return _classify_refusal(pred) in _ANSWERED_CLASSES


def _grounding(row: dict[str, Any]) -> float | None:
    g = _judge_block(row).get("grounding")
    try:
        return float(g)
    except (TypeError, ValueError):
        return None


def _blank_counts() -> dict[str, int]:
    return {
        "total": 0,
        "answerable": 0,
        "unanswerable": 0,
        "answered": 0,               # committed to an answer
        "answered_ungrounded": 0,    # answered AND grounding <= threshold
        "answered_no_grounding": 0,  # answered but judge grounding missing
        "abstained": 0,
        "abstained_on_unans": 0,     # correct refusals
        "abstained_on_ans": 0,       # over-refusals
        "ans_grounded": 0,           # answerable rows answered & grounded
    }


def _accumulate(counts: dict[str, int], rows: dict[str, dict[str, Any]], thr: float) -> None:
    for r in rows.values():
        gold = str(r.get("gold") or "")
        pred = str(r.get("pred") or "")
        is_unans = is_refusal_gold(gold)
        answered = _answered(pred)
        counts["total"] += 1
        counts["unanswerable" if is_unans else "answerable"] += 1
        if answered:
            counts["answered"] += 1
            g = _grounding(r)
            if g is None:
                counts["answered_no_grounding"] += 1
            elif g <= thr:
                counts["answered_ungrounded"] += 1
            if (not is_unans) and g is not None and g > thr:
                counts["ans_grounded"] += 1
        else:
            counts["abstained"] += 1
            counts["abstained_on_unans" if is_unans else "abstained_on_ans"] += 1


def _rates(c: dict[str, int]) -> dict[str, Any]:
    def div(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None
    return {
        "n": c["total"],
        "answerable_n": c["answerable"],
        "unanswerable_n": c["unanswerable"],
        "uar": div(c["answered_ungrounded"], c["answered"]),
        "coverage": div(c["answered"], c["total"]),
        "correct_refusal": div(c["abstained_on_unans"], c["unanswerable"]),
        "over_refusal": div(c["abstained_on_ans"], c["answerable"]),
        "grounded_accuracy": div(c["ans_grounded"], c["answerable"]),
        "answered_no_grounding": c["answered_no_grounding"],
    }


def compute(cross_root: Path, thr: float) -> dict[str, Any]:
    # counts[model_label][arm]["pooled" | dataset] -> counts dict
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(
        lambda: {a: defaultdict(_blank_counts) for a in ARMS}
    )
    slug_of: dict[str, str] = {}

    coverage_notes: list[dict[str, Any]] = []

    def process(model_slug: str, dataset: str, run_root: Path) -> None:
        conds = dict(zip(ARMS, DATASET_CONDITIONS[dataset]))
        judged = run_root / "eval" / "judged"
        paths = {a: judged / conds[a] / "bedrock_judge.jsonl" for a in ARMS}
        if not all(p.is_file() for p in paths.values()):
            return
        label = MODEL_META[model_slug]["label"]
        slug_of[label] = model_slug
        loaded = {a: _load_jsonl_map(paths[a]) for a in ARMS}
        sizes = {a: len(loaded[a]) for a in ARMS}
        # Paired: only score eval_ids present in ALL three arms so every rate
        # shares an identical denominator (fixes incomplete-judging skew).
        common = set.intersection(*(set(loaded[a]) for a in ARMS))
        if min(sizes.values()) != max(sizes.values()):
            coverage_notes.append(
                {"model_label": label, "dataset": dataset,
                 "arm_sizes": sizes, "paired_n": len(common)}
            )
        for arm in ARMS:
            rows = {eid: loaded[arm][eid] for eid in common}
            _accumulate(counts[label][arm][dataset], rows, thr)
            _accumulate(counts[label][arm]["pooled"], rows, thr)

    for model_slug, dataset, run_root, _src in REFERENCE_RUNS:
        process(model_slug, dataset, run_root)
    for model_slug in MODEL_META:
        if model_slug == "llama32_3b":
            continue
        for dataset, ds_dir in CROSS_DATASET_DIRS.items():
            run_root = cross_root / model_slug / ds_dir
            if run_root.is_dir():
                process(model_slug, dataset, run_root)

    per_model: list[dict[str, Any]] = []
    for label in MODEL_ORDER:
        if label not in counts:
            continue
        arms_out: dict[str, Any] = {}
        for arm in ARMS:
            by_ds = {
                ds: _rates(c)
                for ds, c in sorted(counts[label][arm].items())
                if ds != "pooled"
            }
            arms_out[arm] = {"pooled": _rates(counts[label][arm]["pooled"]), "by_dataset": by_ds}
        per_model.append({"model_label": label, "model_slug": slug_of[label], "arms": arms_out})

    return {
        "schema": SCHEMA,
        "created_at": utc_iso(),
        "grounding_ungrounded_threshold": thr,
        "note": "answer = _classify_refusal in {invented_answer, context_dump}; "
                "ungrounded = judge grounding <= threshold. Metrics are PAIRED: "
                "each (model,dataset) uses only eval_ids present in all three arms.",
        "incomplete_judging_notes": coverage_notes,
        "per_model": per_model,
    }


def _fmt(x: float | None) -> str:
    return f"{x * 100:.1f}" if isinstance(x, (int, float)) else "—"


def _delta(ours: float | None, base: float | None) -> str:
    if not isinstance(ours, (int, float)) or not isinstance(base, (int, float)):
        return "—"
    d = (ours - base) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f}"


def print_markdown(report: dict[str, Any]) -> None:
    pm = report["per_model"]

    print(f"\n## UAR — Unsupported-Answer Rate (pooled, lower better; grounding<= {report['grounding_ungrounded_threshold']:.0f})")
    print("| Model | B3 | B5 | Ours | Δ Ours−B3 | Δ Ours−B5 |")
    print("|---|---|---|---|---|---|")
    for m in pm:
        a = m["arms"]
        u = {k: a[k]["pooled"]["uar"] for k in ARMS}
        print(f"| {m['model_label']} | {_fmt(u['B3'])} | {_fmt(u['B5'])} | {_fmt(u['Ours'])} "
              f"| {_delta(u['Ours'], u['B3'])} | {_delta(u['Ours'], u['B5'])} |")

    print("\n## Coverage — answer rate (pooled)")
    print("| Model | B3 | B5 | Ours |")
    print("|---|---|---|---|")
    for m in pm:
        a = m["arms"]
        print(f"| {m['model_label']} | {_fmt(a['B3']['pooled']['coverage'])} "
              f"| {_fmt(a['B5']['pooled']['coverage'])} | {_fmt(a['Ours']['pooled']['coverage'])} |")

    print("\n## Correct-refusal rate on unanswerable rows (pooled, higher better)")
    print("| Model | B3 | B5 | Ours |")
    print("|---|---|---|---|")
    for m in pm:
        a = m["arms"]
        print(f"| {m['model_label']} | {_fmt(a['B3']['pooled']['correct_refusal'])} "
              f"| {_fmt(a['B5']['pooled']['correct_refusal'])} | {_fmt(a['Ours']['pooled']['correct_refusal'])} |")

    print("\n## Over-refusal rate on answerable rows (pooled, lower better)")
    print("| Model | B3 | B5 | Ours |")
    print("|---|---|---|---|")
    for m in pm:
        a = m["arms"]
        print(f"| {m['model_label']} | {_fmt(a['B3']['pooled']['over_refusal'])} "
              f"| {_fmt(a['B5']['pooled']['over_refusal'])} | {_fmt(a['Ours']['pooled']['over_refusal'])} |")

    # win tallies on UAR
    b3w = sum(1 for m in pm if _lt(m["arms"]["Ours"]["pooled"]["uar"], m["arms"]["B3"]["pooled"]["uar"]))
    b5w = sum(1 for m in pm if _lt(m["arms"]["Ours"]["pooled"]["uar"], m["arms"]["B5"]["pooled"]["uar"]))
    print(f"\nUAR win tally: Ours < B3 on {b3w}/{len(pm)} models; Ours < B5 on {b5w}/{len(pm)} models")


def _lt(a: float | None, b: float | None) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and a < b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-root", type=Path, default=DEFAULT_CROSS_ROOT)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="grounding <= threshold counts as ungrounded (default 2 on 1-5 scale)")
    ap.add_argument("--out", type=Path,
                    default=THESIS_ROOT / "experiments/analysis/grounded_hallucination_stats.json")
    ns = ap.parse_args()
    report = compute(ns.cross_root, ns.threshold)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {ns.out}")
    print_markdown(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

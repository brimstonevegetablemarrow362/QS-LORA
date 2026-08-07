"""
Slice RepLiQA eval metrics by Bedrock question categories.

Joins:
  - eval_subset_2000_classified_bedrock.jsonl (categories per eval_id)
  - listwise_rank/listwise_rank_results.jsonl (points + ranks per condition)
  - eval/metrics/<condition>/scored_predictions.jsonl (F1, EM, cosine)

Outputs per slice dimension (question_type, answer_evidence, …):
  mean listwise points, mean token F1, win rate vs baseline, rank per slice.

Run:
  python -m thesis.cli eval-repliqa-slice-metrics
  python -m thesis.cli eval-repliqa-slice-metrics --slice-field question_type
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis.eval_repliqa_score import _discover_prediction_files

SCHEMA = "repliqa_eval_slices/v1"
DEFAULT_SLICE_FIELDS = (
    "question_type",
    "answer_evidence",
    "document_necessity",
    "answer_form",
    "finetuning_expected_gain",
)


def _load_classifications(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        cats = row.get("categories") or {}
        if eid:
            out[eid] = cats if isinstance(cats, dict) else {}
    return out


def _load_listwise(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        eid = str(row.get("eval_id") or "").strip()
        if eid:
            out[eid] = row
    return out


def _load_metrics_by_condition(metrics_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """condition -> eval_id -> metrics dict"""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cond_dir in sorted(metrics_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        scored = cond_dir / "scored_predictions.jsonl"
        if not scored.is_file():
            continue
        cond = cond_dir.name
        by_eid: dict[str, dict[str, Any]] = {}
        for line in scored.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            eid = str(row.get("eval_id") or "").strip()
            m = row.get("metrics") or {}
            if eid and isinstance(m, dict):
                by_eid[eid] = m
        out[cond] = by_eid
    return out


def _leaderboard(
    per_cond: dict[str, dict[str, Any]],
    *,
    metric_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for cond, stats in per_cond.items():
        v = stats.get(metric_key)
        if v is not None:
            rows.append({"condition": cond, metric_key: v})
    rows.sort(key=lambda x: float(x[metric_key]), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _aggregate_slice(
    eval_ids: list[str],
    *,
    conditions: list[str],
    baseline: str,
    listwise: dict[str, dict[str, Any]],
    metrics_by_cond: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    per_cond: dict[str, dict[str, Any]] = {}
    for cond in conditions:
        pts: list[float] = []
        f1s: list[float] = []
        ems: list[float] = []
        cos: list[float] = []
        wins = losses = ties = 0

        for eid in eval_ids:
            lw = listwise.get(eid)
            if lw:
                pbc = lw.get("points_by_condition") or {}
                rbc = lw.get("ranks_by_condition") or {}
                if cond in pbc:
                    pts.append(float(pbc[cond]))
                if baseline in rbc and cond in rbc and cond != baseline:
                    rb, rc = int(rbc[baseline]), int(rbc[cond])
                    if rc < rb:
                        wins += 1
                    elif rc > rb:
                        losses += 1
                    else:
                        ties += 1

            m = metrics_by_cond.get(cond, {}).get(eid)
            if m:
                f1s.append(float(m.get("token_f1", 0)))
                ems.append(float(m.get("exact_match", 0)))
                c = m.get("pred_gold_cosine")
                if c is not None:
                    cos.append(float(c))

        n = len(eval_ids)
        n_pair = wins + losses + ties
        per_cond[cond] = {
            "n_questions": n,
            "mean_listwise_points": round(sum(pts) / len(pts), 4) if pts else None,
            "mean_token_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "mean_exact_match": round(sum(ems) / len(ems), 4) if ems else None,
            "mean_pred_gold_cosine": round(sum(cos) / len(cos), 4) if cos else None,
            "win_rate_vs_baseline": round(wins / n_pair, 4) if n_pair else None,
            "wins_vs_baseline": wins,
            "losses_vs_baseline": losses,
            "ties_vs_baseline": ties,
        }

    return {
        "n_questions": len(eval_ids),
        "per_condition": per_cond,
        "leaderboard_by_mean_listwise_points": _leaderboard(
            per_cond, metric_key="mean_listwise_points"
        ),
        "leaderboard_by_mean_token_f1": _leaderboard(per_cond, metric_key="mean_token_f1"),
    }


def run_eval_repliqa_slice_metrics(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = run_root / "eval"
    classified = (
        Path(ns.classified_jsonl).expanduser().resolve()
        if ns.classified_jsonl
        else eval_dir / "eval_subset_2000_classified_bedrock.jsonl"
    )
    listwise_path = (
        Path(ns.listwise_jsonl).expanduser().resolve()
        if ns.listwise_jsonl
        else eval_dir / "listwise_rank" / "listwise_rank_results.jsonl"
    )
    metrics_dir = Path(ns.metrics_dir).expanduser().resolve() if ns.metrics_dir else eval_dir / "metrics"
    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else eval_dir / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not classified.is_file():
        print(f"Not found: {classified}", file=sys.stderr)
        return 1
    if not listwise_path.is_file():
        print(f"Not found: {listwise_path}", file=sys.stderr)
        return 1

    classifications = _load_classifications(classified)
    listwise = _load_listwise(listwise_path)
    metrics_by_cond = _load_metrics_by_condition(metrics_dir) if metrics_dir.is_dir() else {}

    pred_dir = eval_dir / "predictions"
    files = _discover_prediction_files(predictions_dir=pred_dir, predictions_index=None, predictions_jsonl=None)
    conditions = sorted({c for c, _ in files})

    slice_fields = list(ns.slice_field) if ns.slice_field else list(DEFAULT_SLICE_FIELDS)
    baseline = str(ns.baseline)

    # eval_id -> list of slice keys per field
    by_field_value: dict[str, dict[str, list[str]]] = {
        f: defaultdict(list) for f in slice_fields
    }
    for eid, cats in classifications.items():
        for field in slice_fields:
            val = str(cats.get(field) or "unknown").strip() or "unknown"
            by_field_value[field][val].append(eid)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "baseline": baseline,
        "classified_jsonl": str(classified),
        "listwise_jsonl": str(listwise_path),
        "metrics_dir": str(metrics_dir),
        "conditions": conditions,
        "dimensions": {},
    }

    for field in slice_fields:
        dim: dict[str, Any] = {}
        for val in sorted(by_field_value[field].keys()):
            eids = sorted(by_field_value[field][val])
            # Only eval_ids present in listwise
            eids = [e for e in eids if e in listwise]
            if not eids:
                continue
            dim[val] = _aggregate_slice(
                eids,
                conditions=conditions,
                baseline=baseline,
                listwise=listwise,
                metrics_by_cond=metrics_by_cond,
            )
        payload["dimensions"][field] = dim

    out_json = out_dir / "slice_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_report(payload, slice_fields=slice_fields, baseline=baseline)
    print(f"\nWrote {out_json}", flush=True)
    return 0


def _print_report(payload: dict[str, Any], *, slice_fields: list[str], baseline: str) -> None:
    for field in slice_fields:
        dim = payload.get("dimensions", {}).get(field) or {}
        if not dim:
            continue
        print(f"\n{'=' * 72}", flush=True)
        print(f"SLICE: {field}  (baseline={baseline})", flush=True)
        print(f"{'=' * 72}", flush=True)
        for val in sorted(dim.keys(), key=lambda v: (-dim[v]["n_questions"], v)):
            block = dim[val]
            n = block["n_questions"]
            print(f"\n--- {val} (n={n}) ---", flush=True)
            lb = block.get("leaderboard_by_mean_listwise_points") or []
            print(f"{'rank':<4} {'condition':<26} {'pts':>6} {'F1':>6} {'win%B1':>8}", flush=True)
            per = block.get("per_condition") or {}
            for row in lb:
                cond = row["condition"]
                st = per.get(cond) or {}
                pts = st.get("mean_listwise_points")
                f1 = st.get("mean_token_f1")
                wr = st.get("win_rate_vs_baseline")
                pts_s = f"{pts:.3f}" if pts is not None else "n/a"
                f1_s = f"{f1:.3f}" if f1 is not None else "n/a"
                wr_s = f"{wr*100:5.1f}%" if wr is not None and cond != baseline else "  base"
                print(f"{row['rank']:<4} {cond:<26} {pts_s:>6} {f1_s:>6} {wr_s:>8}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Slice eval metrics by question category.")
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--classified-jsonl", type=Path, default=None)
    p.add_argument("--listwise-jsonl", type=Path, default=None)
    p.add_argument("--metrics-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--baseline", type=str, default="B3_lora_all")
    p.add_argument(
        "--slice-field",
        action="append",
        default=None,
        help="Repeatable; default: all 5 category fields.",
    )
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.run_root is None:
        ns.run_root = (
            Path(__file__).resolve().parent
            / "experiments/repliqa/runs/repliqa_train_0-3"
        )
    raise SystemExit(run_eval_repliqa_slice_metrics(ns))

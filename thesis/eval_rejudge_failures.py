"""
Re-judge rows where Ours underperforms and compare GA before vs after.

Use cases:
  - Test judge variance (plain re-judge same rubric)
  - Test position-swap debias on failure rows only

Usage:
  python -m thesis.cli eval-rejudge-failures \\
    --run-root /path/to/run \\
    --ours-condition Ours_tier_merge \\
    --baseline-condition B5_adalora_all \\
    --mode loses_to_baseline \\
    --position-swap-debias
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.bedrock_judge_qa_score import _aggregate_stats, _row_key, run_qa_bedrock_judge


def _default_eval_jsonl(run_root: Path) -> Path:
    finetune = Path(__file__).resolve().parent.parent
    name = run_root.name
    if name == "repliqa" or name.endswith("repliqa_train_0-3"):
        local = run_root / "eval" / "eval_subset_2000.jsonl"
        if local.is_file():
            return local
        return finetune / "thesis/experiments/repliqa/runs/repliqa_train_0-3/eval/eval_subset_2000.jsonl"
    if "quoref" in name:
        return finetune / "data/quoref/jsonl/validation.jsonl"
    if "squad" in name:
        return finetune / "data/squad_v2/jsonl/validation.jsonl"
    manifest = run_root / "cross_model_manifest.json"
    if manifest.is_file():
        ds = json.loads(manifest.read_text(encoding="utf-8")).get("dataset")
        if ds == "repliqa":
            return finetune / "thesis/experiments/repliqa/runs/repliqa_train_0-3/eval/eval_subset_2000.jsonl"
        if ds == "quoref":
            return finetune / "data/quoref/jsonl/validation.jsonl"
        if ds == "squad":
            return finetune / "data/squad_v2/jsonl/validation.jsonl"
    raise FileNotFoundError(f"Cannot infer --eval-jsonl from {run_root}")


def _ga_from_row(row: dict[str, Any]) -> float | None:
    block = row.get("llm_judge") or row.get("judge") or {}
    v = block.get("gold_alignment")
    return float(v) if v is not None else None


def load_judged_ga(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = _row_key(row)
        ga = _ga_from_row(row)
        if key and ga is not None:
            out[key] = ga
    return out


def load_judged_rows(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = _row_key(row)
        if key:
            out[key] = row
    return out


def select_failure_ids(
    ours_ga: dict[str, float],
    *,
    mode: str,
    low_ga_threshold: int,
    baseline_ga: dict[str, float] | None = None,
    b3_ga: dict[str, float] | None = None,
    b5_ga: dict[str, float] | None = None,
) -> list[str]:
    ids: list[str] = []
    for eid, ga_o in ours_ga.items():
        if mode == "low_ga":
            if ga_o <= low_ga_threshold:
                ids.append(eid)
            continue
        if mode == "loses_to_baseline":
            if baseline_ga and eid in baseline_ga and ga_o < baseline_ga[eid]:
                ids.append(eid)
            continue
        if mode == "loses_to_b3":
            if b3_ga and eid in b3_ga and ga_o < b3_ga[eid]:
                ids.append(eid)
            continue
        if mode == "loses_to_b5":
            if b5_ga and eid in b5_ga and ga_o < b5_ga[eid]:
                ids.append(eid)
            continue
        if mode == "loses_to_best":
            best = None
            if b3_ga and eid in b3_ga:
                best = b3_ga[eid]
            if b5_ga and eid in b5_ga:
                best = max(best, b5_ga[eid]) if best is not None else b5_ga[eid]
            if best is not None and ga_o < best:
                ids.append(eid)
            continue
        raise ValueError(f"Unknown mode: {mode}")
    return sorted(ids)


def _write_subset_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_rejudge_to_main(
    ours_judged: Path,
    rejudge_jsonl: Path,
    failure_ids: list[str],
    *,
    min_delta: float = 0.0,
    counterfactual_delta: float | None = None,
) -> dict[str, Any]:
    """Merge re-judged failure rows into main bedrock_judge.jsonl if GA improves."""
    if counterfactual_delta is not None and counterfactual_delta <= min_delta:
        return {
            "applied": False,
            "reason": f"counterfactual_delta {counterfactual_delta} <= min_delta {min_delta}",
        }

    new_rows = load_judged_rows(rejudge_jsonl)
    if not new_rows:
        return {"applied": False, "reason": "empty rejudge jsonl"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ours_judged.with_name(f"bedrock_judge.pre_rejudge.{ts}.jsonl")
    backup.write_bytes(ours_judged.read_bytes())

    merged_lines: list[str] = []
    n_replaced = 0
    for line in ours_judged.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = _row_key(row)
        if key in failure_ids and key in new_rows:
            merged_lines.append(json.dumps(new_rows[key], ensure_ascii=False))
            n_replaced += 1
        else:
            merged_lines.append(line)

    ours_judged.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")

    all_rows = [json.loads(ln) for ln in merged_lines]
    stats = _aggregate_stats(all_rows)
    summary_path = ours_judged.parent / "bedrock_judge_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    old_ga = (summary.get("stats") or {}).get("mean_gold_alignment")
    summary["stats"] = stats
    summary["n_rows"] = len(all_rows)
    notes = list(summary.get("notes") or [])
    notes.append(
        f"Re-judge merge {ts}: replaced {n_replaced} failure rows; "
        f"mean_gold_alignment {old_ga} -> {stats.get('mean_gold_alignment')}"
    )
    summary["notes"] = notes
    summary["rejudge_merge"] = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "backup_jsonl": str(backup),
        "rejudge_jsonl": str(rejudge_jsonl),
        "n_replaced": n_replaced,
        "mean_gold_alignment_before": old_ga,
        "mean_gold_alignment_after": stats.get("mean_gold_alignment"),
        "counterfactual_delta": counterfactual_delta,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "applied": True,
        "backup_jsonl": str(backup),
        "n_replaced": n_replaced,
        "mean_gold_alignment_before": old_ga,
        "mean_gold_alignment_after": stats.get("mean_gold_alignment"),
    }


def compare_rejudge(
    failure_ids: list[str],
    old_ga: dict[str, float],
    new_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    deltas: list[float] = []
    improved = worse = same = 0
    per_row: list[dict[str, Any]] = []
    for eid in failure_ids:
        old = old_ga.get(eid)
        new = _ga_from_row(new_rows.get(eid, {}))
        if old is None or new is None:
            continue
        d = new - old
        deltas.append(d)
        if d > 1e-9:
            improved += 1
        elif d < -1e-9:
            worse += 1
        else:
            same += 1
        per_row.append({"eval_id": eid, "ga_old": old, "ga_new": new, "delta": round(d, 4)})
    n = len(deltas)
    return {
        "n_compared": n,
        "mean_ga_old": round(sum(old_ga[eid] for eid in failure_ids if eid in old_ga) / max(1, len(failure_ids)), 4),
        "mean_ga_new": round(sum(_ga_from_row(new_rows[eid]) or 0 for eid in failure_ids if eid in new_rows and _ga_from_row(new_rows[eid]) is not None) / max(1, n), 4) if n else None,
        "mean_delta": round(sum(deltas) / n, 4) if n else None,
        "n_improved": improved,
        "n_worse": worse,
        "n_same": same,
        "pct_improved": round(100.0 * improved / n, 1) if n else None,
        "rows": per_row[:50],
        "rows_truncated": max(0, n - 50),
    }


def run_eval_rejudge_failures(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    ours_cond = str(ns.ours_condition)
    judged_dir = Path(ns.judged_dir).expanduser().resolve() if ns.judged_dir else run_root / "eval" / "judged"
    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else judged_dir / ours_cond / "rejudge_failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    ours_judged = judged_dir / ours_cond / "bedrock_judge.jsonl"
    ours_pred = run_root / "eval" / "predictions" / ours_cond / "predictions.jsonl"
    if not ours_judged.is_file():
        print(f"Missing judged: {ours_judged}", file=sys.stderr)
        return 1
    if not ours_pred.is_file():
        print(f"Missing predictions: {ours_pred}", file=sys.stderr)
        return 1

    ours_ga = load_judged_ga(ours_judged)
    b3_ga = b5_ga = baseline_ga = None
    if ns.b3_condition:
        b3_ga = load_judged_ga(judged_dir / str(ns.b3_condition) / "bedrock_judge.jsonl")
    if ns.b5_condition:
        b5_ga = load_judged_ga(judged_dir / str(ns.b5_condition) / "bedrock_judge.jsonl")
    if ns.baseline_condition:
        baseline_ga = load_judged_ga(judged_dir / str(ns.baseline_condition) / "bedrock_judge.jsonl")

    failure_ids = select_failure_ids(
        ours_ga,
        mode=str(ns.mode),
        low_ga_threshold=int(ns.low_ga_threshold),
        baseline_ga=baseline_ga,
        b3_ga=b3_ga,
        b5_ga=b5_ga,
    )
    if int(ns.max_rows) > 0:
        failure_ids = failure_ids[: int(ns.max_rows)]

    print(f"Failure mode={ns.mode}: {len(failure_ids)} rows selected", flush=True)
    if not failure_ids:
        print("No failure rows matched; nothing to re-judge.", flush=True)
        return 0

    if ns.dry_run:
        print(f"Dry run: would re-judge {len(failure_ids)} rows -> {out_dir}", flush=True)
        return 0

    # Load prediction rows for failure ids
    want = set(failure_ids)
    pred_rows: list[dict[str, Any]] = []
    for line in ours_pred.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = _row_key(row)
        if eid in want:
            pred_rows.append(row)
    if len(pred_rows) != len(failure_ids):
        print(f"Warning: predictions matched {len(pred_rows)}/{len(failure_ids)} failure ids", flush=True)

    subset_pred = out_dir / "failure_predictions.jsonl"
    _write_subset_jsonl(pred_rows, subset_pred)

    eval_jsonl = Path(ns.eval_jsonl).expanduser().resolve() if ns.eval_jsonl else _default_eval_jsonl(run_root)

    rejudge_jsonl = out_dir / "bedrock_judge_rejudge.jsonl"
    judge_ns = argparse.Namespace(
        predictions_jsonl=subset_pred,
        out_jsonl=rejudge_jsonl,
        summary_json=out_dir / "bedrock_judge_rejudge_summary.json",
        timing_json=out_dir / "bedrock_judge_rejudge_timing.json",
        model=ns.model,
        region=ns.region,
        answer_field="pred",
        max_rows=0,
        max_context_chars=int(ns.max_context_chars),
        max_tokens=int(ns.max_tokens),
        temperature=float(ns.temperature),
        concurrency=int(ns.concurrency),
        request_delay_s=float(ns.request_delay_s),
        dry_run=False,
        eval_jsonl=eval_jsonl,
        qa_jsonl=None,
        resume=False,
        force=True,
        position_swap_debias=bool(ns.position_swap_debias),
    )
    rc = run_qa_bedrock_judge(judge_ns)
    if rc != 0:
        return rc

    new_rows = load_judged_rows(rejudge_jsonl)
    comparison = compare_rejudge(failure_ids, ours_ga, new_rows)

    # Full-set counterfactual: replace failure rows' GA in overall mean
    all_ids = list(ours_ga.keys())
    counterfactual_sum = 0.0
    for eid in all_ids:
        if eid in failure_ids and eid in new_rows:
            ga = _ga_from_row(new_rows[eid])
            counterfactual_sum += ga if ga is not None else ours_ga[eid]
        else:
            counterfactual_sum += ours_ga[eid]
    counterfactual_mean = counterfactual_sum / max(1, len(all_ids))
    original_mean = sum(ours_ga.values()) / max(1, len(ours_ga))

    report = {
        "schema": "eval_rejudge_failures/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "ours_condition": ours_cond,
        "mode": ns.mode,
        "position_swap_debias": bool(ns.position_swap_debias),
        "n_failure_ids": len(failure_ids),
        "original_mean_ga_all": round(original_mean, 4),
        "counterfactual_mean_ga_all": round(counterfactual_mean, 4),
        "counterfactual_delta_all": round(counterfactual_mean - original_mean, 4),
        "failure_subset": comparison,
        "artifacts": {
            "failure_predictions": str(subset_pred),
            "rejudge_jsonl": str(rejudge_jsonl),
            "rejudge_summary": str(out_dir / "bedrock_judge_rejudge_summary.json"),
        },
    }
    if bool(ns.apply_if_improved):
        apply_result = apply_rejudge_to_main(
            ours_judged,
            rejudge_jsonl,
            failure_ids,
            min_delta=float(ns.apply_min_delta),
            counterfactual_delta=report["counterfactual_delta_all"],
        )
        report["apply_result"] = apply_result
        if apply_result.get("applied"):
            print(
                f"Applied re-judge to main: GA {apply_result['mean_gold_alignment_before']} "
                f"-> {apply_result['mean_gold_alignment_after']} ({apply_result['n_replaced']} rows)",
                flush=True,
            )
        else:
            print(f"Did not apply re-judge: {apply_result.get('reason', 'unknown')}", flush=True)

    report_path = out_dir / "rejudge_comparison.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== Re-judge failures ({ns.mode}) ===", flush=True)
    print(f"  failure rows: {len(failure_ids)}", flush=True)
    print(f"  mean GA (failures) old: {comparison['mean_ga_old']} -> new: {comparison['mean_ga_new']}", flush=True)
    print(f"  mean delta on failures: {comparison['mean_delta']}", flush=True)
    print(f"  improved/worse/same: {comparison['n_improved']}/{comparison['n_worse']}/{comparison['n_same']}", flush=True)
    print(f"  full-set GA: {original_mean:.4f} -> counterfactual {counterfactual_mean:.4f} ({report['counterfactual_delta_all']:+.4f})", flush=True)
    print(f"Wrote {report_path}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Re-judge Ours failure rows and compare GA.")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--ours-condition", type=str, required=True)
    p.add_argument("--judged-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--b3-condition", type=str, default=None)
    p.add_argument("--b5-condition", type=str, default=None)
    p.add_argument("--baseline-condition", type=str, default=None, help="Alias for single baseline (sets mode loses_to_baseline).")
    p.add_argument(
        "--mode",
        type=str,
        default="loses_to_best",
        choices=("low_ga", "loses_to_baseline", "loses_to_b3", "loses_to_b5", "loses_to_best"),
    )
    p.add_argument("--low-ga-threshold", type=int, default=2)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--position-swap-debias", action="store_true")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-delay-s", type=float, default=0.05)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--apply-if-improved",
        action="store_true",
        help="If counterfactual full-set GA improves, merge re-judged rows into bedrock_judge.jsonl.",
    )
    p.add_argument(
        "--apply-min-delta",
        type=float,
        default=0.0,
        help="Minimum counterfactual GA gain required before --apply-if-improved merges.",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run_eval_rejudge_failures(build_arg_parser().parse_args()))

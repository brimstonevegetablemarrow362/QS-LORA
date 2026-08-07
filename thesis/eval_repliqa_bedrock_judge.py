"""
Batch Bedrock LLM judge on all RepLiQA eval prediction sets.

External evaluator: pred vs human gold + context (rubric v3).
Primary thesis metric from judge: mean_gold_alignment, mean_overall.

Run (from finetuning/, after source_bedrock_env.sh):
  python -m thesis.cli eval-repliqa-bedrock-judge --max-rows 50   # smoke
  python -m thesis.cli eval-repliqa-bedrock-judge                   # all conditions × 2000
  python -m thesis.cli eval-repliqa-bedrock-judge --leaderboard-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.eval_repliqa_score import _discover_prediction_files, filter_prediction_files

SCHEMA = "repliqa_bedrock_judge_leaderboard/v1"


def _rank_key(row: dict[str, Any], *, primary: str) -> tuple[float, float, str]:
    ga = float(row.get("mean_gold_alignment") or 0.0)
    ov = float(row.get("mean_overall") or 0.0)
    gr = float(row.get("mean_grounding") or 0.0)
    if primary == "mean_overall":
        return (ov, ga, gr, str(row.get("condition", "")))
    if primary == "mean_grounding":
        return (gr, ga, ov, str(row.get("condition", "")))
    return (ga, ov, gr, str(row.get("condition", "")))


def _resolve_condition_name(s: dict[str, Any], summary_path: Path | None = None) -> str:
    """Basename from judged/<condition>/ or predictions path, not bedrock_judge_summary.json."""
    for key in ("condition", "model_id_from_rows"):
        v = str(s.get(key) or "").strip()
        if v and v not in ("bedrock_judge", "judged", "predictions"):
            return v
    inp = str(s.get("input_jsonl") or "").strip()
    if inp:
        parent = Path(inp).parent.name
        if parent and parent not in ("predictions", "judged"):
            return parent
    if summary_path is not None:
        parent = summary_path.parent.name
        if parent and parent != "judged":
            return parent
    return "unknown"


def aggregate_judge_summaries(summaries: list[dict[str, Any]], *, rank_by: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in summaries:
        stats = s.get("stats") or {}
        summary_path = Path(s["summary_json"]) if s.get("summary_json") else None
        cond = _resolve_condition_name(s, summary_path)
        rows.append(
            {
                "condition": cond,
                "input_jsonl": s.get("input_jsonl"),
                "out_jsonl": s.get("out_jsonl"),
                "summary_json": s.get("summary_json"),
                "prompt_version": s.get("prompt_version"),
                "n_rows": s.get("n_rows"),
                "n_judged_ok": stats.get("n_judged_ok"),
                "mean_gold_alignment": stats.get("mean_gold_alignment"),
                "mean_overall": stats.get("mean_overall"),
                "mean_grounding": stats.get("mean_grounding"),
                "mean_relevance": stats.get("mean_relevance"),
                "tier_counts": stats.get("tier_counts"),
            }
        )
    ranked = sorted(rows, key=lambda r: _rank_key(r, primary=rank_by), reverse=True)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked


def _load_summary(path: Path, *, condition: str | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["summary_json"] = str(path)
    data["condition"] = condition or _resolve_condition_name(data, path)
    return data


def collect_existing_summaries(judged_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    # One summary per immediate condition subdir (skip backup dirs like Ours_tier_merge.bad_*).
    for child in sorted(judged_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith(".") or ".bad_" in name:
            continue
        p = child / "bedrock_judge_summary.json"
        if not p.is_file():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(_load_summary(p))
    return out


def run_eval_repliqa_bedrock_judge(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = run_root / "eval"
    eval_jsonl = (
        Path(ns.eval_jsonl).expanduser().resolve()
        if ns.eval_jsonl
        else eval_dir / str(ns.eval_input_name)
    )
    judged_dir = Path(ns.judged_dir).expanduser().resolve() if ns.judged_dir else eval_dir / "judged"
    if getattr(ns, "position_swap_debias", False) and ns.judged_dir is None:
        judged_dir = eval_dir / "judged_debias"
    judged_dir.mkdir(parents=True, exist_ok=True)

    if ns.leaderboard_only:
        summaries = collect_existing_summaries(judged_dir)
        if not summaries:
            print(f"No judge summaries under {judged_dir}", file=sys.stderr)
            return 1
        leaderboard = aggregate_judge_summaries(summaries, rank_by=str(ns.rank_by))
        _write_leaderboard(judged_dir, leaderboard, rank_by=str(ns.rank_by), n_conditions=len(leaderboard))
        _print_leaderboard(leaderboard, rank_by=str(ns.rank_by))
        return 0

    pred_dir = Path(ns.predictions_dir).expanduser().resolve() if ns.predictions_dir else eval_dir / "predictions"
    files = _discover_prediction_files(
        predictions_dir=pred_dir,
        predictions_index=Path(ns.predictions_index).expanduser().resolve()
        if ns.predictions_index
        else None,
        predictions_jsonl=Path(ns.predictions_jsonl).expanduser().resolve()
        if ns.predictions_jsonl
        else None,
    )
    if not files:
        print("No prediction files found.", file=sys.stderr)
        return 1
    conditions_filter = getattr(ns, "conditions", None) or None
    if conditions_filter:
        files = filter_prediction_files(files, list(conditions_filter))
    if not eval_jsonl.is_file():
        print(f"Eval subset not found: {eval_jsonl}", file=sys.stderr)
        return 1

    from thesis.bedrock_judge_qa_score import run_qa_bedrock_judge

    summaries: list[dict[str, Any]] = []
    for cond, pj in files:
        if not pj.is_file() or pj.stat().st_size == 0:
            print(f"Skip {cond}: empty or missing predictions at {pj}", flush=True)
            continue

        out_dir = judged_dir / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        out_jsonl = out_dir / "bedrock_judge.jsonl"
        sum_path = out_dir / "bedrock_judge_summary.json"
        if ns.skip_existing and sum_path.is_file() and not ns.force and not ns.resume:
            print(f"Skip {cond} (summary exists)", flush=True)
            summaries.append(_load_summary(sum_path, condition=cond))
            continue

        print(f"\n=== Bedrock judge: {cond} ===", flush=True)
        judge_ns = argparse.Namespace(
            predictions_jsonl=pj,
            out_jsonl=out_jsonl,
            summary_json=sum_path,
            timing_json=out_dir / "bedrock_judge_timing.json",
            model=ns.model,
            region=ns.region,
            answer_field="pred",
            max_rows=int(ns.max_rows),
            max_context_chars=int(ns.max_context_chars),
            max_tokens=int(ns.max_tokens),
            temperature=float(ns.temperature),
            concurrency=int(ns.concurrency),
            request_delay_s=float(ns.request_delay_s),
            dry_run=bool(ns.dry_run),
            eval_jsonl=eval_jsonl,
            qa_jsonl=None,
            resume=bool(ns.resume),
            force=bool(ns.force),
            position_swap_debias=bool(getattr(ns, "position_swap_debias", False)),
        )
        rc = run_qa_bedrock_judge(judge_ns)
        if rc != 0:
            print(f"Skip {cond}: judge failed (rc={rc})", flush=True)
            continue
        if sum_path.is_file():
            summaries.append(_load_summary(sum_path, condition=cond))

    if not summaries:
        print("No judge summaries produced.", file=sys.stderr)
        return 1

    leaderboard = aggregate_judge_summaries(summaries, rank_by=str(ns.rank_by))
    _write_leaderboard(
        judged_dir,
        leaderboard,
        rank_by=str(ns.rank_by),
        n_conditions=len(leaderboard),
        position_swap_debias=bool(getattr(ns, "position_swap_debias", False)),
    )
    _print_leaderboard(leaderboard, rank_by=str(ns.rank_by))
    return 0


def _write_leaderboard(
    judged_dir: Path,
    leaderboard: list[dict[str, Any]],
    *,
    rank_by: str,
    n_conditions: int,
    position_swap_debias: bool = False,
) -> None:
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rank_by": rank_by,
        "n_conditions": n_conditions,
        "leaderboard": leaderboard,
        "notes": [
            "External LLM judge (Bedrock Haiku) with human gold reference (v3_eval_gold).",
            "gold_alignment: semantic match pred vs gold (1-5).",
            "overall: holistic prediction quality; grounding: supported by context.",
        ]
        + (
            [
                "position_swap_debias: gold-first and pred-first prompts averaged per row.",
            ]
            if position_swap_debias
            else []
        ),
    }
    path = judged_dir / "judge_leaderboard.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path}", flush=True)


def _print_leaderboard(leaderboard: list[dict[str, Any]], *, rank_by: str) -> None:
    print(f"\n=== Bedrock judge leaderboard (rank_by={rank_by}) ===", flush=True)
    hdr = f"{'rank':<5} {'condition':<28} {'gold_al':>8} {'overall':>8} {'ground':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in leaderboard:
        ga = row.get("mean_gold_alignment")
        ov = row.get("mean_overall")
        gr = row.get("mean_grounding")
        print(
            f"{row['rank']:<5} {row['condition']:<28} "
            f"{ga if ga is not None else 'n/a':>8} "
            f"{ov if ov is not None else 'n/a':>8} "
            f"{gr if gr is not None else 'n/a':>8}",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch Bedrock judge on RepLiQA eval predictions.")
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument("--predictions-index", type=Path, default=None)
    p.add_argument("--predictions-jsonl", type=Path, default=None)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--eval-input-name", type=str, default="eval_subset_2000.jsonl")
    p.add_argument("--judged-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-delay-s", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--force", action="store_true", help="Re-judge all rows even if summary exists.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful judged rows; Bedrock only for failed/missing rows in bedrock_judge.jsonl.",
    )
    p.add_argument(
        "--rank-by",
        type=str,
        default="mean_gold_alignment",
        choices=("mean_gold_alignment", "mean_overall", "mean_grounding"),
    )
    p.add_argument(
        "--leaderboard-only",
        action="store_true",
        help="Aggregate existing judged/*/bedrock_judge_summary.json only.",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Only judge these prediction subdir names (e.g. B3_lora_ctx Ours_tier_ctx).",
    )
    p.add_argument(
        "--position-swap-debias",
        action="store_true",
        help="Average scores from gold-first and prediction-first eval prompts (writes judged_debias/).",
    )
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.run_root is None:
        ns.run_root = (
            Path(__file__).resolve().parent
            / "experiments/repliqa/runs/repliqa_train_0-3"
        )
    raise SystemExit(run_eval_repliqa_bedrock_judge(ns))

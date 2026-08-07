"""
Listwise Bedrock ranking: for each eval question, rank all model predictions together.

Scoring: rank 1 (best) -> (n+1)-1 points; for n=8, best=9, worst=1.
Mean points per condition across questions = leaderboard.

Run (from finetuning/, after source_bedrock_env.sh):
  python -m thesis.cli eval-repliqa-listwise-rank --max-rows 10
  python -m thesis.cli eval-repliqa-listwise-rank --concurrency 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.bedrock_judge_qa_score import (
    DEFAULT_BEDROCK_MODEL_ID,
    PROVIDER,
    _bedrock_client,
    _check_aws_env,
    _invoke_bedrock_claude,
)
from thesis.eval_repliqa_score import _discover_prediction_files, filter_prediction_files
from thesis.qa_answer_metrics import is_invalid_answer
from thesis.qa_judge_common import (
    LISTWISE_RANK_PROMPT_VERSION,
    LISTWISE_RANK_SYSTEM,
    build_listwise_rank_user_message,
    parse_listwise_rank_json,
    rank_to_points,
)
from thesis.repliqa_eval_context import load_eval_index

SCHEMA = "repliqa_listwise_rank/v1"


def _parse_conditions(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split() if p.strip()]
        return parts or None
    parts = [str(p).strip() for p in raw if str(p).strip()]
    return parts or None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_preds_by_eval_id(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if not eid:
            continue
        pred = str(row.get("pred") or row.get("prediction") or "").strip()
        out[eid] = pred
    return out


def _build_tasks(
    eval_index: dict[str, dict[str, Any]],
    preds_by_condition: dict[str, dict[str, str]],
    *,
    require_all_conditions: bool,
    max_rows: int,
    seed: int,
    fixed_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    conditions = sorted(preds_by_condition.keys())
    n_cond = len(conditions)
    eval_ids = sorted(eval_index.keys())
    if max_rows > 0:
        eval_ids = eval_ids[:max_rows]

    tasks: list[dict[str, Any]] = []
    for eid in eval_ids:
        ref = eval_index[eid]
        preds: dict[str, str] = {}
        for cond in conditions:
            p = preds_by_condition.get(cond, {}).get(eid, "")
            if is_invalid_answer(p):
                p = ""
            preds[cond] = p
        present = [c for c in conditions if preds[c]]
        if require_all_conditions and len(present) != n_cond:
            continue
        if len(present) < 2:
            continue
        tasks.append(
            {
                "eval_id": eid,
                "context": str(ref.get("context") or ""),
                "question": str(ref.get("question") or ""),
                "gold": str(ref.get("gold") or ref.get("answer") or ""),
                "preds": preds,
                "conditions_present": present,
            }
        )
    rng = random.Random(seed)
    for t in tasks:
        present = list(t["conditions_present"])
        if fixed_order:
            order = [c for c in fixed_order if c in present]
            if len(order) < 2:
                order = present
        else:
            order = list(present)
            rng.shuffle(order)
        labels = [chr(ord("A") + i) for i in range(len(order))]
        label_to_cond = {lab: cond for lab, cond in zip(labels, order)}
        cond_to_label = {cond: lab for lab, cond in label_to_cond.items()}
        t["label_to_condition"] = label_to_cond
        t["condition_to_label"] = cond_to_label
        t["labels"] = labels
    return tasks


def _merge_position_swap_results(
    pass1: list[dict[str, Any]],
    pass2: list[dict[str, Any]],
    *,
    order1: list[str],
    order2: list[str],
) -> list[dict[str, Any]]:
    """Average ranks from both presentation orders to reduce positional bias."""
    by_id1 = {str(r["eval_id"]): r for r in pass1 if not r.get("error")}
    by_id2 = {str(r["eval_id"]): r for r in pass2 if not r.get("error")}
    merged: list[dict[str, Any]] = []
    for eid in sorted(set(by_id1) & set(by_id2)):
        r1 = by_id1[eid]
        r2 = by_id2[eid]
        ranks1 = r1.get("ranks_by_condition") or {}
        ranks2 = r2.get("ranks_by_condition") or {}
        conds = sorted(set(ranks1) | set(ranks2))
        avg_ranks: dict[str, float] = {}
        avg_points: dict[str, float] = {}
        for c in conds:
            if c in ranks1 and c in ranks2:
                avg_ranks[c] = (int(ranks1[c]) + int(ranks2[c])) / 2.0
                p1 = (r1.get("points_by_condition") or {}).get(c)
                p2 = (r2.get("points_by_condition") or {}).get(c)
                if p1 is not None and p2 is not None:
                    avg_points[c] = (int(p1) + int(p2)) / 2.0
        merged.append(
            {
                "eval_id": eid,
                "prompt_version": LISTWISE_RANK_PROMPT_VERSION,
                "position_swap_debias": True,
                "presentation_order_pass1": order1,
                "presentation_order_pass2": order2,
                "ranks_by_condition_pass1": ranks1,
                "ranks_by_condition_pass2": ranks2,
                "ranks_by_condition": avg_ranks,
                "points_by_condition": avg_points,
                "n_candidates": len(conds),
            }
        )
    return merged


def _rank_one(
    client: Any,
    task: dict[str, Any],
    *,
    model_id: str,
    max_context_chars: int,
    max_pred_chars: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    label_to_cond: dict[str, str] = task["label_to_condition"]
    labeled = [(lab, task["preds"][label_to_cond[lab]]) for lab in task["labels"]]
    user = build_listwise_rank_user_message(
        context=task["context"],
        question=task["question"],
        gold=task["gold"],
        labeled_preds=labeled,
        max_context_chars=max_context_chars,
        max_pred_chars=max_pred_chars,
    )
    raw = _invoke_bedrock_claude(
        client,
        model_id=model_id,
        user_message=user,
        max_tokens=max_tokens,
        temperature=temperature,
        system=LISTWISE_RANK_SYSTEM,
    )
    valid = set(task["labels"])
    parsed = parse_listwise_rank_json(raw, valid)
    n = len(task["labels"])
    result: dict[str, Any] = {
        "eval_id": task["eval_id"],
        "prompt_version": LISTWISE_RANK_PROMPT_VERSION,
        "n_candidates": n,
        "label_to_condition": label_to_cond,
        "raw_preview": (raw or "")[:400],
    }
    if not parsed:
        result["error"] = "parse_error"
        return result

    ranking_labels = [str(x).strip() for x in parsed["ranking"]]
    ranks_by_condition: dict[str, int] = {}
    points_by_condition: dict[str, int] = {}
    for rank_idx, lab in enumerate(ranking_labels, start=1):
        cond = label_to_cond[lab]
        ranks_by_condition[cond] = rank_idx
        points_by_condition[cond] = rank_to_points(rank_idx, n)

    result["ranking_labels"] = ranking_labels
    result["ranks_by_condition"] = ranks_by_condition
    result["points_by_condition"] = points_by_condition
    result["brief_reason"] = str(parsed.get("brief_reason", ""))[:500]
    return result


def _aggregate_points(results: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    sums = {c: 0 for c in conditions}
    counts = {c: 0 for c in conditions}
    n_ok = 0
    n_err = 0
    for r in results:
        if r.get("error"):
            n_err += 1
            continue
        pts = r.get("points_by_condition") or {}
        n_ok += 1
        for c, p in pts.items():
            if c in sums:
                sums[c] += float(p)
                counts[c] += 1
    per_cond: dict[str, Any] = {}
    for c in conditions:
        n = counts[c]
        per_cond[c] = {
            "n_ranked": n,
            "sum_points": sums[c],
            "mean_points": round(sums[c] / n, 4) if n else None,
        }
    return {"n_questions_ok": n_ok, "n_questions_error": n_err, "per_condition": per_cond}


def _execute_rank_tasks(
    client: Any,
    tasks: list[dict[str, Any]],
    *,
    model_id: str,
    ns: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    lock = threading.Lock()
    conc = max(1, int(ns.concurrency))

    def job(idx: int, task: dict[str, Any]) -> None:
        try:
            if float(ns.request_delay_s) > 0:
                time.sleep(float(ns.request_delay_s) * (idx % conc) * 0.05)
            res = _rank_one(
                client,
                task,
                model_id=model_id,
                max_context_chars=int(ns.max_context_chars),
                max_pred_chars=int(ns.max_pred_chars),
                max_tokens=int(ns.max_tokens),
                temperature=float(ns.temperature),
            )
        except Exception as e:
            res = {
                "eval_id": task["eval_id"],
                "error": f"api_error:{e}",
                "prompt_version": LISTWISE_RANK_PROMPT_VERSION,
            }
        with lock:
            results[idx] = res
            done = sum(1 for r in results if r is not None)
            if done % 10 == 0 or done == len(tasks):
                print(f"  ... {done}/{len(tasks)} ranked", flush=True)

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(job, i, t) for i, t in enumerate(tasks)]
        for f in as_completed(futs):
            f.result()
    return [r for r in results if r is not None]


def run_eval_repliqa_listwise_rank(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = run_root / "eval"
    eval_jsonl = (
        Path(ns.eval_jsonl).expanduser().resolve()
        if ns.eval_jsonl
        else eval_dir / str(ns.eval_input_name)
    )
    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else eval_dir / "listwise_rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "listwise_rank_results.jsonl"
    sum_path = out_dir / "listwise_rank_summary.json"
    lb_path = out_dir / "listwise_rank_leaderboard.json"

    if ns.leaderboard_only:
        if not sum_path.is_file():
            print(f"Missing {sum_path}", file=sys.stderr)
            return 1
        data = json.loads(sum_path.read_text(encoding="utf-8"))
        _print_leaderboard(data.get("leaderboard") or [])
        return 0

    region = (ns.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip()
    _check_aws_env(region)
    model_id = str(ns.model or DEFAULT_BEDROCK_MODEL_ID)

    if not eval_jsonl.is_file():
        print(f"Not found: {eval_jsonl}", file=sys.stderr)
        return 1

    pred_dir = Path(ns.predictions_dir).expanduser().resolve() if ns.predictions_dir else eval_dir / "predictions"
    files = _discover_prediction_files(
        predictions_dir=pred_dir,
        predictions_index=None,
        predictions_jsonl=None,
    )
    conditions_filter = _parse_conditions(getattr(ns, "conditions", None))
    if conditions_filter:
        files = filter_prediction_files(files, conditions_filter)
    if not files:
        print("No prediction files found.", file=sys.stderr)
        return 1
    if len(files) < 2:
        print("Listwise rank needs at least 2 conditions.", file=sys.stderr)
        return 1

    eval_index = load_eval_index(eval_jsonl)
    preds_by_condition = {cond: _load_preds_by_eval_id(pj) for cond, pj in files}
    conditions = sorted(preds_by_condition.keys())
    presentation_order = list(conditions_filter) if conditions_filter else conditions
    position_swap = bool(getattr(ns, "position_swap_debias", False))

    if position_swap:
        if len(presentation_order) != 2:
            print("position-swap debias requires exactly 2 --conditions", file=sys.stderr)
            return 1

    task_kwargs = dict(
        eval_index=eval_index,
        preds_by_condition=preds_by_condition,
        require_all_conditions=bool(ns.require_all_conditions),
        max_rows=int(ns.max_rows),
        seed=int(ns.seed),
    )

    if ns.dry_run:
        n_tasks = len(
            _build_tasks(**task_kwargs, fixed_order=presentation_order if position_swap else None)
        )
        print(
            f"Dry run: {n_tasks} questions, {len(conditions)} conditions, "
            f"require_all={ns.require_all_conditions}, position_swap={position_swap}",
            flush=True,
        )
        return 0

    client = _bedrock_client(region)
    wall0 = time.perf_counter()

    if position_swap:
        order2 = list(reversed(presentation_order))
        print(
            f"Listwise rank (position-swap debias): {len(presentation_order)} conditions, "
            f"orders={presentation_order} / {order2}, model={model_id}",
            flush=True,
        )
        tasks1 = _build_tasks(**task_kwargs, fixed_order=presentation_order)
        print(f"Pass 1: presentation order {presentation_order}", flush=True)
        pass1 = _execute_rank_tasks(client, tasks1, model_id=model_id, ns=ns)
        tasks2 = _build_tasks(**task_kwargs, fixed_order=order2)
        print(f"Pass 2: presentation order {order2}", flush=True)
        pass2 = _execute_rank_tasks(client, tasks2, model_id=model_id, ns=ns)
        rows_out = _merge_position_swap_results(pass1, pass2, order1=presentation_order, order2=order2)
        if not rows_out:
            print("No debiased rows after merging both presentation orders.", file=sys.stderr)
            return 1
    else:
        tasks = _build_tasks(**task_kwargs, fixed_order=None)
        if not tasks:
            print("No questions to rank.", file=sys.stderr)
            return 1
        print(
            f"Listwise rank: {len(tasks)} questions × {len(conditions)} conditions (present), "
            f"model={model_id}",
            flush=True,
        )
        rows_out = _execute_rank_tasks(client, tasks, model_id=model_id, ns=ns)

    agg = _aggregate_points(rows_out, conditions)
    leaderboard = sorted(
        [
            {
                "condition": c,
                "mean_points": agg["per_condition"][c]["mean_points"],
                "sum_points": agg["per_condition"][c]["sum_points"],
                "n_ranked": agg["per_condition"][c]["n_ranked"],
            }
            for c in conditions
        ],
        key=lambda x: float(x.get("mean_points") or 0),
        reverse=True,
    )
    for i, row in enumerate(leaderboard, start=1):
        row["rank"] = i

    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = {
        "schema": SCHEMA,
        "created_at": _utc_iso(),
        "prompt_version": LISTWISE_RANK_PROMPT_VERSION,
        "provider": PROVIDER,
        "model": model_id,
        "region": region,
        "eval_jsonl": str(eval_jsonl),
        "predictions_dir": str(pred_dir),
        "conditions": conditions,
        "n_questions_attempted": len(tasks),
        "require_all_conditions": bool(ns.require_all_conditions),
        "conditions_filter": conditions_filter,
        "scoring": (
            f"points = (n_candidates + 1) - rank; n={len(conditions)} => "
            f"best={len(conditions)}, worst=1"
        ),
        "stats": agg,
        "leaderboard": leaderboard,
        "timing": {"total_wall_s": round(time.perf_counter() - wall0, 3)},
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lb_path.write_text(
        json.dumps(
            {"schema": SCHEMA, "leaderboard": leaderboard},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    _print_leaderboard(leaderboard)
    return 0 if agg["n_questions_ok"] > 0 else 1


def _print_leaderboard(leaderboard: list[dict[str, Any]]) -> None:
    print("\n=== Listwise rank leaderboard (mean points, max=9) ===", flush=True)
    hdr = f"{'rank':<5} {'condition':<28} {'mean_pts':>10} {'n':>6}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in leaderboard:
        mp = row.get("mean_points")
        mp_s = f"{mp:.4f}" if mp is not None else "n/a"
        print(
            f"{row.get('rank', '?'):<5} {row['condition']:<28} {mp_s:>10} "
            f"{row.get('n_ranked', 0):>6}",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Listwise Bedrock rank of all model preds per question; mean points leaderboard."
    )
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Only rank these prediction subdirs (default: all under predictions/).",
    )
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--eval-input-name", type=str, default="eval_subset_2000.jsonl")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-context-chars", type=int, default=8000)
    p.add_argument("--max-pred-chars", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--request-delay-s", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42, help="Shuffle candidate label order per question.")
    p.add_argument(
        "--position-swap-debias",
        action="store_true",
        help="Rank twice with swapped fixed presentation order (2 conditions only); average ranks.",
    )
    p.add_argument(
        "--allow-partial-conditions",
        action="store_true",
        help="Rank when some models lack preds; default requires all conditions.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--leaderboard-only", action="store_true")
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.run_root is None:
        ns.run_root = (
            Path(__file__).resolve().parent
            / "experiments/repliqa/runs/repliqa_train_0-3"
        )
    if not hasattr(ns, "require_all_conditions"):
        ns.require_all_conditions = not bool(getattr(ns, "allow_partial_conditions", False))
    raise SystemExit(run_eval_repliqa_listwise_rank(ns))

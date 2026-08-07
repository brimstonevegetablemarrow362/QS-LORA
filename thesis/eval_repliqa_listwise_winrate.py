"""
Compute head-to-head listwise win rates vs a baseline (e.g., B3_lora_all).

Uses listwise_rank_results.jsonl from eval_repliqa_listwise_rank.py.
For each eval_id, compares ranks:
  - win:  rank_baseline > rank_other  (baseline worse)
  - loss: rank_baseline < rank_other  (baseline better)
  - tie:  equal ranks

Outputs JSON summary + prints a small table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _discover_conditions(rows: list[dict[str, Any]], *, baseline: str) -> list[str]:
    """Union condition names from all ranked rows (skip API/parse error rows)."""
    seen: set[str] = set()
    for row in rows:
        if row.get("error"):
            continue
        ranks = row.get("ranks_by_condition") or {}
        seen.update(ranks.keys())
    conditions = sorted(seen)
    if baseline not in conditions:
        raise ValueError(f"Baseline {baseline!r} not in conditions {conditions}")
    return conditions


def run_eval_repliqa_listwise_winrate(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = run_root / "eval"
    in_path = (
        Path(ns.results_jsonl).expanduser().resolve()
        if ns.results_jsonl
        else eval_dir / "listwise_rank" / "listwise_rank_results.jsonl"
    )
    if not in_path.is_file():
        print(f"Not found: {in_path}", flush=True)
        return 1

    baseline = str(ns.baseline)
    rows = _read_rows(in_path)
    if not rows:
        print("No rows in listwise_rank_results.jsonl", flush=True)
        return 1

    try:
        conditions = _discover_conditions(rows, baseline=baseline)
    except ValueError as e:
        print(str(e), flush=True)
        return 1

    stats: dict[str, dict[str, int]] = {}
    for cond in conditions:
        if cond == baseline:
            continue
        stats[cond] = {"wins": 0, "losses": 0, "ties": 0, "n": 0}

    for r in rows:
        ranks = r.get("ranks_by_condition") or {}
        if baseline not in ranks:
            continue
        rb = float(ranks[baseline])
        for cond in conditions:
            if cond == baseline or cond not in ranks:
                continue
            rc = float(ranks[cond])
            s = stats[cond]
            s["n"] += 1
            if rc < rb:
                s["wins"] += 1
            elif rc > rb:
                s["losses"] += 1
            else:
                s["ties"] += 1

    summary: dict[str, Any] = {
        "schema": "repliqa_listwise_winrate/v1",
        "baseline": baseline,
        "results_jsonl": str(in_path),
        "conditions": conditions,
        "pairwise": {},
    }
    for cond, s in stats.items():
        n = s["n"] or 1
        win_rate = s["wins"] / n
        loss_rate = s["losses"] / n
        tie_rate = s["ties"] / n
        summary["pairwise"][cond] = {
            **s,
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "tie_rate": round(tie_rate, 4),
        }

    out_path = in_path.with_name("listwise_winrate_vs_baseline.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Baseline: {baseline}", flush=True)
    print(f"Wrote {out_path}", flush=True)
    print("\n=== Head-to-head listwise win rates (baseline worse = win) ===", flush=True)
    hdr = f"{'cond':<24} {'wins':>8} {'losses':>8} {'ties':>8} {'win%':>8} {'loss%':>8} {'tie%':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for cond in conditions:
        if cond == baseline:
            continue
        s = summary["pairwise"][cond]
        print(
            f"{cond:<24} {s['wins']:>8} {s['losses']:>8} {s['ties']:>8} "
            f"{s['win_rate']*100:>7.2f} {s['loss_rate']*100:>7.2f} {s['tie_rate']*100:>7.2f}",
            flush=True,
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Head-to-head listwise win rates vs a baseline (using listwise_rank_results.jsonl)."
    )
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--results-jsonl", type=Path, default=None)
    p.add_argument("--baseline", type=str, default="B3_lora_all")
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.run_root is None:
        ns.run_root = (
            Path(__file__).resolve().parent
            / "experiments/repliqa/runs/repliqa_train_0-3"
        )
    raise SystemExit(run_eval_repliqa_listwise_winrate(ns))


"""
Export a human-readable comparison sample: gold vs Ours vs B3 LoRA.

Outputs:
  eval/study_samples/study_sample_<n>.jsonl
  eval/study_samples/study_sample_<n>.md

Run:
  python -m thesis.cli eval-repliqa-export-study-sample
  python -m thesis.cli eval-repliqa-export-study-sample --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OURS_CONDITION = "Ours_tier_merge"
B3_CONDITION = "B3_lora_all"


def _load_preds(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if eid:
            out[eid] = row
    return out


def _load_listwise_ranks(path: Path) -> dict[str, dict[str, int]]:
    """eval_id -> condition -> rank (1=best)."""
    out: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        eid = str(row.get("eval_id") or "").strip()
        ranks = row.get("ranks_by_condition") or row.get("ranks") or {}
        if eid and isinstance(ranks, dict):
            out[eid] = {str(k): int(v) for k, v in ranks.items()}
    return out


def _pick_balanced_ids(
    eval_ids: list[str],
    listwise: dict[str, dict[str, int]],
    *,
    n: int,
    seed: int,
) -> list[str]:
    """Mix: Ours beats B3, B3 beats Ours, random."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for eid in eval_ids:
        ranks = listwise.get(eid) or {}
        r_ours = ranks.get(OURS_CONDITION)
        r_b3 = ranks.get(B3_CONDITION)
        if r_ours is None or r_b3 is None:
            buckets["random"].append(eid)
            continue
        if r_ours < r_b3:
            buckets["ours_beats_b3"].append(eid)
        elif r_b3 < r_ours:
            buckets["b3_beats_ours"].append(eid)
        else:
            buckets["tie"].append(eid)

    rng = random.Random(seed)
    per_bucket = max(1, n // 3)
    chosen: list[str] = []
    seen: set[str] = set()

    for key in ("ours_beats_b3", "b3_beats_ours", "random"):
        pool = list(buckets[key])
        rng.shuffle(pool)
        for eid in pool:
            if eid in seen:
                continue
            chosen.append(eid)
            seen.add(eid)
            if len([x for x in chosen if x in buckets[key]]) >= per_bucket:
                break
            if len(chosen) >= n:
                break
        if len(chosen) >= n:
            break

    remaining = [e for e in eval_ids if e not in seen]
    rng.shuffle(remaining)
    for eid in remaining:
        if len(chosen) >= n:
            break
        chosen.append(eid)
        seen.add(eid)

    return chosen[:n]


def _build_row(
    eid: str,
    eval_row: dict[str, Any],
    ours: dict[str, Any],
    b3: dict[str, Any],
    listwise: dict[str, dict[str, int]],
) -> dict[str, Any]:
    ranks = listwise.get(eid) or {}
    return {
        "eval_id": eid,
        "document_topic": eval_row.get("document_topic") or ours.get("document_topic"),
        "question": eval_row.get("question") or ours.get("question"),
        "gold": eval_row.get("gold") or eval_row.get("answer") or ours.get("gold"),
        "ours_answer": ours.get("pred", ""),
        "ours_condition": OURS_CONDITION,
        "b3_lora_answer": b3.get("pred", ""),
        "b3_condition": B3_CONDITION,
        "listwise_rank_ours": ranks.get(OURS_CONDITION),
        "listwise_rank_b3": ranks.get(B3_CONDITION),
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path, *, n: int) -> None:
    lines = [
        "# RepLiQA study sample — gold vs Ours vs B3 LoRA",
        "",
        f"**N:** {len(rows)} questions  ",
        f"**Ours:** `{OURS_CONDITION}`  ",
        f"**LoRA (uniform):** `{B3_CONDITION}`  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "Listwise rank: 1 = best among 8 models on that question.",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        eid = row["eval_id"]
        r_o = row.get("listwise_rank_ours")
        r_b3 = row.get("listwise_rank_b3")
        rank_note = f"listwise rank — Ours:{r_o} B3:{r_b3}"
        lines.extend(
            [
                f"## {i}. `{eid}`",
                "",
                f"*{row.get('document_topic') or ''}* · {rank_note}",
                "",
                "### Question",
                "",
                str(row["question"]),
                "",
                "### Gold (human benchmark)",
                "",
                str(row["gold"]),
                "",
                f"### Ours (`{OURS_CONDITION}`)",
                "",
                str(row["ours_answer"]),
                "",
                f"### Uniform LoRA (`{B3_CONDITION}`)",
                "",
                str(row["b3_lora_answer"]),
                "",
                "---",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_eval_repliqa_export_study_sample(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = run_root / "eval"
    n = int(ns.n)
    seed = int(ns.seed)

    eval_jsonl = Path(ns.eval_jsonl) if ns.eval_jsonl else eval_dir / "eval_subset_2000.jsonl"
    preds_dir = Path(ns.predictions_dir) if ns.predictions_dir else eval_dir / "predictions"
    out_dir = Path(ns.output_dir) if ns.output_dir else eval_dir / "study_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    ours_path = preds_dir / OURS_CONDITION / "predictions.jsonl"
    b3_path = preds_dir / B3_CONDITION / "predictions.jsonl"
    for p in (ours_path, b3_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    eval_index: dict[str, dict[str, Any]] = {}
    for line in eval_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if eid:
            eval_index[eid] = row

    ours_preds = _load_preds(ours_path)
    b3_preds = _load_preds(b3_path)

    common = sorted(set(eval_index) & set(ours_preds) & set(b3_preds))
    listwise_path = eval_dir / "listwise_rank/listwise_rank_results.jsonl"
    listwise = _load_listwise_ranks(listwise_path)

    if ns.strategy == "balanced":
        picked = _pick_balanced_ids(common, listwise, n=n, seed=seed)
    else:
        rng = random.Random(seed)
        picked = rng.sample(common, min(n, len(common)))

    rows = [
        _build_row(eid, eval_index[eid], ours_preds[eid], b3_preds[eid], listwise)
        for eid in picked
    ]

    stem = f"study_sample_{len(rows)}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    md_path = out_dir / f"{stem}.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_markdown(rows, md_path, n=len(rows))

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {md_path}")
    print(f"  strategy={ns.strategy} seed={seed} n={len(rows)}")
    return 0


def _load_classifications(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        cats = row.get("categories") or {}
        if eid and isinstance(cats, dict):
            out[eid] = cats
    return out


def _load_judge_scores(path: Path) -> dict[str, dict[str, Any]]:
    """eval_id -> llm_judge block (gold_alignment, overall, …)."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or "").strip()
        judge = row.get("llm_judge") or {}
        if eid and isinstance(judge, dict) and judge.get("gold_alignment") is not None:
            out[eid] = judge
    return out


def _load_listwise_points(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        eid = str(row.get("eval_id") or "").strip()
        pts = row.get("points_by_condition") or {}
        if eid and isinstance(pts, dict):
            out[eid] = {str(k): int(v) for k, v in pts.items()}
    return out


def _load_token_f1(metrics_dir: Path, condition: str) -> dict[str, float]:
    scored = metrics_dir / condition / "scored_predictions.jsonl"
    out: dict[str, float] = {}
    if not scored.is_file():
        return out
    for line in scored.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or "").strip()
        m = row.get("metrics") or row
        f1 = m.get("token_f1")
        if eid and f1 is not None:
            out[eid] = float(f1)
    return out


def export_ours_beats_lora(run_root: Path, *, ours: str = OURS_CONDITION, lora: str = B3_CONDITION) -> dict[str, Any]:
    """All eval items where Ours listwise rank is better (lower) than uniform LoRA."""
    eval_dir = run_root / "eval"
    eval_jsonl = eval_dir / "eval_subset_2000.jsonl"
    preds_dir = eval_dir / "predictions"

    eval_index: dict[str, dict[str, Any]] = {}
    for line in eval_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if eid:
            eval_index[eid] = row

    ours_preds = _load_preds(preds_dir / ours / "predictions.jsonl")
    b3_preds = _load_preds(preds_dir / lora / "predictions.jsonl")

    listwise_ranks = _load_listwise_ranks(eval_dir / "listwise_rank/listwise_rank_results.jsonl")

    common = sorted(set(eval_index) & set(ours_preds) & set(b3_preds))
    items: list[dict[str, Any]] = []

    for eid in common:
        ranks = listwise_ranks.get(eid) or {}
        r_ours = ranks.get(ours)
        r_b3 = ranks.get(lora)
        if r_ours is None or r_b3 is None:
            continue
        if r_ours >= r_b3:
            continue

        items.append(
            {
                "question": eval_index[eid].get("question") or ours_preds[eid].get("question"),
                "gold_answer": eval_index[eid].get("gold") or eval_index[eid].get("answer"),
                "ours_answer": ours_preds[eid].get("pred", ""),
                "b3_lora_answer": b3_preds[eid].get("pred", ""),
            }
        )

    items.sort(key=lambda x: x["question"])

    return items


def run_eval_repliqa_export_ours_beats_lora(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = run_root / "eval"
    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else eval_dir / "study_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    ours = str(ns.ours_condition or OURS_CONDITION)
    lora = str(ns.lora_condition or B3_CONDITION)
    items = export_ours_beats_lora(run_root, ours=ours, lora=lora)

    out_path = out_dir / str(ns.output_name or "ours_beats_lora.json")
    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"  ours={ours} lora={lora} n={len(items)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export gold vs Ours vs B3 study sample")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategy", choices=("balanced", "random"), default="balanced")
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return run_eval_repliqa_export_study_sample(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

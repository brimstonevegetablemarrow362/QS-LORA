"""
Score DROP validation predictions vs human gold (max EM/F1 over multiple answers).

  python -m thesis.cli eval-drop-score --predictions-dir .../eval/predictions
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.qa_answer_metrics import exact_match, is_invalid_answer, token_f1

SCHEMA = "drop_eval_metrics/v1"


def gold_texts(row: dict[str, Any]) -> list[str]:
    if row.get("answers"):
        out: list[str] = []
        for a in row["answers"]:
            if isinstance(a, dict):
                t = str(a.get("text") or "").strip()
            else:
                t = str(a).strip()
            if t:
                out.append(t)
        if out:
            return out
    g = str(row.get("gold") or row.get("answer") or "").strip()
    return [g] if g else []


def drop_em_f1(pred: str, golds: list[str]) -> tuple[int, float]:
    if not golds:
        return 0, 0.0
    em = 1 if any(exact_match(pred, g) for g in golds) else 0
    f1 = max(token_f1(pred, g) for g in golds)
    return em, f1


def score_file(path: Path, *, write_scored: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    ems: list[int] = []
    f1s: list[float] = []
    n_invalid = 0
    scored: list[dict[str, Any]] = []
    for row in rows:
        pred = str(row.get("pred") or row.get("prediction") or "").strip()
        golds = gold_texts(row)
        if is_invalid_answer(pred):
            n_invalid += 1
        em, f1 = drop_em_f1(pred, golds)
        ems.append(em)
        f1s.append(f1)
        if write_scored:
            rec = dict(row)
            rec["exact_match"] = em
            rec["token_f1"] = round(f1, 6)
            scored.append(rec)
    n = len(rows)
    stats = {
        "n_rows": n,
        "n_invalid_pred": n_invalid,
        "exact_match": round(sum(ems) / n, 6) if n else None,
        "token_f1": round(sum(f1s) / n, 6) if n else None,
    }
    return stats, scored


def discover_prediction_files(predictions_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for sub in sorted(predictions_dir.iterdir()):
        if not sub.is_dir():
            continue
        pj = sub / "predictions.jsonl"
        if pj.is_file():
            out.append((sub.name, pj))
    return out


def run_eval_drop_score(ns: argparse.Namespace) -> int:
    if ns.predictions_jsonl:
        files = [(Path(ns.predictions_jsonl).parent.name, Path(ns.predictions_jsonl))]
    elif ns.predictions_dir:
        files = discover_prediction_files(Path(ns.predictions_dir))
    else:
        raise SystemExit("Pass --predictions-dir or --predictions-jsonl")

    if not files:
        raise SystemExit("No predictions.jsonl files found")

    run_root = Path(ns.run_root).expanduser().resolve() if ns.run_root else None
    eval_dir = (
        Path(ns.eval_dir).expanduser().resolve()
        if ns.eval_dir
        else (run_root / "eval" if run_root else Path("eval"))
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: dict[str, Any] = {
        "schema": SCHEMA,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "conditions": {},
    }

    for cond_id, path in files:
        stats, scored_rows = score_file(path, write_scored=bool(ns.write_scored))
        if ns.write_scored and scored_rows:
            scored_path = path.parent / "scored.jsonl"
            with scored_path.open("w", encoding="utf-8") as fp:
                for r in scored_rows:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        leaderboard["conditions"][cond_id] = {
            "predictions_jsonl": str(path),
            **stats,
        }
        print(
            f"{cond_id:30s} EM={stats['exact_match']}  F1={stats['token_f1']}  n={stats['n_rows']}",
            flush=True,
        )

    out_path = Path(ns.leaderboard_json) if ns.leaderboard_json else eval_dir / "leaderboard.json"
    out_path.write_text(json.dumps(leaderboard, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DROP validation EM/F1 (max over gold answers)")
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument("--predictions-jsonl", type=Path, default=None)
    p.add_argument("--leaderboard-json", type=Path, default=None)
    p.add_argument("--write-scored", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run_eval_drop_score(build_arg_parser().parse_args()))

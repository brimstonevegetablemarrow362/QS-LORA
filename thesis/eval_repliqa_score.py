"""
Score RepLiQA eval predictions vs human gold.

Metrics per row (pred vs gold):
  - exact_match (0/1)
  - token_f1 (0–1)
  - pred_gold_cosine (optional; sentence-transformers embedding similarity)

Aggregate per condition, rank baselines, write leaderboard JSON.

Run from finetuning/:
  python -m thesis.cli eval-repliqa-score --predictions-dir .../eval/predictions
  python -m thesis.cli eval-repliqa-score --predictions-jsonl .../B3_lora_all/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.qa_answer_metrics import (
    exact_match,
    is_invalid_answer,
    is_refusal_gold,
    token_f1,
)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCHEMA = "repliqa_eval_metrics/v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: {e}") from e
    return rows


def _resolve_pred_gold(row: dict[str, Any]) -> tuple[str, str]:
    pred = str(row.get("pred") or row.get("prediction") or "").strip()
    gold = str(row.get("gold") or row.get("answer") or "").strip()
    return pred, gold


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _discover_prediction_files(
    *,
    predictions_dir: Path | None,
    predictions_index: Path | None,
    predictions_jsonl: Path | None,
) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []

    if predictions_jsonl is not None:
        p = predictions_jsonl.expanduser().resolve()
        name = p.parent.name if p.name == "predictions.jsonl" else p.stem
        out.append((name, p))
        return out

    if predictions_index is not None:
        idx_path = predictions_index.expanduser().resolve()
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        conds = data.get("conditions") or {}
        for cond_id, meta in sorted(conds.items()):
            pj = meta.get("predictions_jsonl")
            if pj:
                out.append((str(cond_id), Path(pj).expanduser().resolve()))
        return out

    if predictions_dir is not None:
        root = predictions_dir.expanduser().resolve()
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            pj = sub / "predictions.jsonl"
            if pj.is_file():
                out.append((sub.name, pj))
        return out

    return out


def filter_prediction_files(
    files: list[tuple[str, Path]],
    conditions: list[str] | None,
) -> list[tuple[str, Path]]:
    """Keep only named prediction conditions (subdir names under predictions/)."""
    if not conditions:
        return files
    want = [str(c).strip() for c in conditions if str(c).strip()]
    if not want:
        return files
    by_name = {name: path for name, path in files}
    missing = [c for c in want if c not in by_name]
    if missing:
        raise SystemExit(
            f"Missing prediction conditions: {missing}. Available: {sorted(by_name.keys())}"
        )
    return [(c, by_name[c]) for c in want]


def _embed_cosine_pairs(
    pred_texts: list[str],
    gold_texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> list[float | None]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: sentence-transformers\n"
            "  pip install sentence-transformers\n"
            "  Or pass --no-embed for F1/EM only."
        ) from e

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embed model {model_name} on {device} …", flush=True)
    model = SentenceTransformer(model_name, device=device)
    pe = model.encode(
        pred_texts,
        batch_size=batch_size,
        show_progress_bar=len(pred_texts) > 128,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    ge = model.encode(
        gold_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    import numpy as np

    cos = np.sum(pe * ge, axis=1)
    return [float(c) for c in cos]


def score_predictions_file(
    predictions_jsonl: Path,
    *,
    condition: str,
    use_embed: bool,
    embed_model: str,
    embed_device: str,
    embed_batch_size: int,
    out_scored_jsonl: Path | None,
) -> dict[str, Any]:
    rows = _read_jsonl(predictions_jsonl)
    if not rows:
        raise ValueError(f"No rows in {predictions_jsonl}")

    scored_rows: list[dict[str, Any]] = []
    em_vals: list[float] = []
    f1_vals: list[float] = []
    cos_vals: list[float] = []
    refusal_em: list[float] = []
    refusal_f1: list[float] = []
    refusal_cos: list[float] = []
    answerable_em: list[float] = []
    answerable_f1: list[float] = []
    answerable_cos: list[float] = []
    n_invalid_pred = 0
    n_refusal_gold = 0

    embed_indices: list[int] = []
    pred_for_embed: list[str] = []
    gold_for_embed: list[str] = []

    for i, row in enumerate(rows):
        pred, gold = _resolve_pred_gold(row)
        refusal = is_refusal_gold(gold)
        if refusal:
            n_refusal_gold += 1

        if is_invalid_answer(pred):
            n_invalid_pred += 1
            em = 0.0
            f1 = 0.0
            cos: float | None = None
        else:
            em = 1.0 if exact_match(pred, gold) else 0.0
            f1 = token_f1(pred, gold)
            embed_indices.append(len(scored_rows))
            pred_for_embed.append(pred)
            gold_for_embed.append(gold)

        em_vals.append(em)
        f1_vals.append(f1)
        if refusal:
            refusal_em.append(em)
            refusal_f1.append(f1)
        else:
            answerable_em.append(em)
            answerable_f1.append(f1)

        scored_rows.append(
            {
                **row,
                "metrics": {
                    "exact_match": em,
                    "token_f1": round(f1, 6),
                    "pred_gold_cosine": None,
                    "gold_is_refusal": refusal,
                    "pred_invalid": is_invalid_answer(pred),
                },
            }
        )

    if use_embed and pred_for_embed:
        cos_list = _embed_cosine_pairs(
            pred_for_embed,
            gold_for_embed,
            model_name=embed_model,
            device=embed_device,
            batch_size=embed_batch_size,
        )
        for j, idx in enumerate(embed_indices):
            c = round(cos_list[j], 6)
            scored_rows[idx]["metrics"]["pred_gold_cosine"] = c
            cos_vals.append(c)
            if scored_rows[idx]["metrics"]["gold_is_refusal"]:
                refusal_cos.append(c)
            else:
                answerable_cos.append(c)

    if out_scored_jsonl is not None:
        out_scored_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_scored_jsonl, "w", encoding="utf-8") as fp:
            for r in scored_rows:
                fp.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    return {
        "condition": condition,
        "predictions_jsonl": str(predictions_jsonl.resolve()),
        "scored_jsonl": str(out_scored_jsonl.resolve()) if out_scored_jsonl else None,
        "n_rows": len(rows),
        "n_invalid_pred": n_invalid_pred,
        "n_refusal_gold": n_refusal_gold,
        "exact_match_mean": _mean(em_vals),
        "token_f1_mean": _mean(f1_vals),
        "pred_gold_cosine_mean": _mean(cos_vals) if cos_vals else None,
        "answerable_exact_match_mean": _mean(answerable_em),
        "answerable_token_f1_mean": _mean(answerable_f1),
        "answerable_pred_gold_cosine_mean": _mean(answerable_cos) if answerable_cos else None,
        "refusal_exact_match_mean": _mean(refusal_em),
        "refusal_token_f1_mean": _mean(refusal_f1),
        "refusal_pred_gold_cosine_mean": _mean(refusal_cos) if refusal_cos else None,
    }


def _rank_key(row: dict[str, Any], *, primary: str) -> tuple[float, float, float, str]:
    """Higher is better; condition name last for stable sort."""
    f1 = float(row.get("token_f1_mean") or 0.0)
    em = float(row.get("exact_match_mean") or 0.0)
    cos = float(row.get("pred_gold_cosine_mean") or 0.0)
    if primary == "exact_match":
        return (em, f1, cos, str(row.get("condition", "")))
    if primary == "pred_gold_cosine":
        return (cos, f1, em, str(row.get("condition", "")))
    return (f1, em, cos, str(row.get("condition", "")))


def build_leaderboard(
    per_condition: list[dict[str, Any]],
    *,
    rank_by: str,
) -> list[dict[str, Any]]:
    ranked = sorted(per_condition, key=lambda r: _rank_key(r, primary=rank_by), reverse=True)
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        out.append({**row, "rank": rank})
    return out


def run_eval_repliqa_score(ns: argparse.Namespace) -> int:
    pred_dir = Path(ns.predictions_dir).expanduser().resolve() if ns.predictions_dir else None
    pred_index = Path(ns.predictions_index).expanduser().resolve() if ns.predictions_index else None
    pred_jsonl = Path(ns.predictions_jsonl).expanduser().resolve() if ns.predictions_jsonl else None

    files = _discover_prediction_files(
        predictions_dir=pred_dir,
        predictions_index=pred_index,
        predictions_jsonl=pred_jsonl,
    )
    if not files:
        print(
            "No prediction files found. Pass --predictions-dir, --predictions-index, or --predictions-jsonl.",
            file=sys.stderr,
        )
        return 1

    metrics_root = (
        Path(ns.metrics_dir).expanduser().resolve()
        if ns.metrics_dir
        else (pred_dir.parent / "metrics" if pred_dir else pred_jsonl.parent.parent / "metrics")
    )
    metrics_root.mkdir(parents=True, exist_ok=True)

    use_embed = not bool(ns.no_embed)
    per_condition: list[dict[str, Any]] = []

    for cond, pj in files:
        out_scored = None
        if ns.write_scored:
            out_scored = metrics_root / cond / "scored_predictions.jsonl"
        print(f"Scoring {cond} ({pj.name}) …", flush=True)
        summary = score_predictions_file(
            pj,
            condition=cond,
            use_embed=use_embed,
            embed_model=str(ns.embed_model),
            embed_device=str(ns.embed_device),
            embed_batch_size=int(ns.embed_batch_size),
            out_scored_jsonl=out_scored,
        )
        per_condition.append(summary)
        cos_m = summary["pred_gold_cosine_mean"]
        cos_s = f"{cos_m:.4f}" if cos_m is not None else "n/a"
        print(
            f"  EM={summary['exact_match_mean']:.4f}  "
            f"F1={summary['token_f1_mean']:.4f}  cos={cos_s}",
            flush=True,
        )

    rank_by = str(ns.rank_by)
    leaderboard = build_leaderboard(per_condition, rank_by=rank_by)

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rank_by": rank_by,
        "use_embed": use_embed,
        "embed_model": str(ns.embed_model) if use_embed else None,
        "n_conditions": len(leaderboard),
        "leaderboard": leaderboard,
        "per_condition": {r["condition"]: r for r in per_condition},
        "notes": [
            "Primary ranking default: token_f1_mean (SQuAD-style).",
            "exact_match: normalize(pred)==normalize(gold).",
            "pred_gold_cosine: embedding similarity; optional with --no-embed.",
            "Refusal gold: substring 'not found in the document' (answerable_* excludes these).",
        ],
    }

    summary_path = metrics_root / "metrics_summary.json"
    leaderboard_path = metrics_root / "metrics_leaderboard.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    leaderboard_path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "rank_by": rank_by,
                "leaderboard": leaderboard,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== Leaderboard (best first) ===", flush=True)
    hdr = f"{'rank':<5} {'condition':<28} {'EM':>8} {'F1':>8} {'cosine':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in leaderboard:
        cos = row.get("pred_gold_cosine_mean")
        cos_s = f"{cos:.4f}" if cos is not None else "n/a"
        print(
            f"{row['rank']:<5} {row['condition']:<28} "
            f"{row['exact_match_mean']:>8.4f} {row['token_f1_mean']:>8.4f} {cos_s:>8}",
            flush=True,
        )
    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {leaderboard_path}", flush=True)
    if ns.write_scored:
        print(f"Per-condition scored JSONL under {metrics_root}/<condition>/", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score RepLiQA eval predictions: EM, token F1, pred↔gold cosine; rank baselines."
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Scan <dir>/<condition>/predictions.jsonl (default when run-root eval/predictions).",
    )
    src.add_argument("--predictions-index", type=Path, default=None)
    src.add_argument("--predictions-jsonl", type=Path, default=None)
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Output dir for summary + leaderboard (default: sibling metrics/ under eval/).",
    )
    p.add_argument("--run-root", type=Path, default=None, help="RepLiQA run dir (sets default paths).")
    p.add_argument("--no-embed", action="store_true", help="Skip pred↔gold embedding cosine.")
    p.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--embed-device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--embed-batch-size", type=int, default=64)
    p.add_argument(
        "--rank-by",
        type=str,
        default="token_f1",
        choices=("token_f1", "exact_match", "pred_gold_cosine"),
        help="Primary metric for leaderboard ordering.",
    )
    p.add_argument(
        "--write-scored",
        action="store_true",
        help="Write per-row scored JSONL under metrics/<condition>/scored_predictions.jsonl",
    )
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.run_root is not None:
        run_root = Path(ns.run_root).expanduser().resolve()
        eval_dir = run_root / "eval"
        if ns.predictions_dir is None and ns.predictions_jsonl is None and ns.predictions_index is None:
            ns.predictions_dir = eval_dir / "predictions"
        if ns.metrics_dir is None:
            ns.metrics_dir = eval_dir / "metrics"
    raise SystemExit(run_eval_repliqa_score(ns))

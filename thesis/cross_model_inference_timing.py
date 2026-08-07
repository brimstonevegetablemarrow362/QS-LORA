"""
Aggregate per-condition eval generate / inference timing for cross-model runs.

Reads eval/predictions/<condition>/timing.json (RepLiQA v2 or DROP compact schema)
and writes eval/inference_timing.json for thesis tables (mean seconds per question).

Usage:
  python -m thesis.cli cross-model-collect-inference --run-root /path/to/run
  python -m thesis.cli cross-model-collect-inference --cross-root /path/to/cross_model/runs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "cross_model_inference_timing/v1"

DATASET_DIRS = {
    "repliqa": "repliqa",
    "quoref": "quoref_qa_v1",
    "squad": "squad_qa_v1",
}

EVAL_N = {"repliqa": 2000, "quoref": 2418, "squad": 11873}


def _infer_dataset_key(run_root: Path) -> str | None:
    name = run_root.name
    for key, sub in DATASET_DIRS.items():
        if name == sub:
            return key
    return None


def _timing_reliable_flags(
    data: dict[str, Any],
    *,
    dataset_key: str | None,
    model_slug: str | None,
    n_preds: int | None,
) -> tuple[bool, str | None]:
    if model_slug and model_slug.startswith("gemma3_"):
        return False, "gemma_mixed_backends"
    backend = str(data.get("backend") or (data.get("decoding") or {}).get("backend") or "hf").lower()
    if backend == "vllm":
        return False, "vllm_batch"
    gen = (data.get("timing") or {}).get("generate_per_question") or data.get("generate") or {}
    n = data.get("n_questions") or data.get("n_rows") or gen.get("n")
    if dataset_key:
        expected = EVAL_N.get(dataset_key)
        if expected and n and int(n) < int(expected) * 0.95:
            return False, "partial_shard"
        if expected and n_preds and n_preds < int(expected) * 0.95:
            return False, "partial_preds"
    return True, None


def _fmt_hms(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def normalize_prediction_timing(
    data: dict[str, Any],
    *,
    timing_path: Path | None = None,
    dataset_key: str | None = None,
    model_slug: str | None = None,
    n_preds: int | None = None,
) -> dict[str, Any]:
    """Normalize RepLiQA v2 and DROP compact timing.json into one schema."""
    t = data.get("timing") or {}
    gen = t.get("generate_per_question") or data.get("generate") or {}
    decoding = data.get("decoding") or {}
    cond = str(data.get("condition") or data.get("model_id") or "").strip()
    mean_s = gen.get("mean_s")
    reliable, skip_reason = _timing_reliable_flags(
        data, dataset_key=dataset_key, model_slug=model_slug, n_preds=n_preds
    )
    return {
        "condition": cond,
        "load_type": data.get("load_type"),
        "backend": decoding.get("backend") or data.get("backend") or "hf",
        "n_questions": data.get("n_questions") or data.get("n_rows"),
        "load_model_s": t.get("load_model_s") if t.get("load_model_s") is not None else data.get("load_s"),
        "generate_loop_s": t.get("generate_loop_s"),
        "total_wall_s": t.get("total_wall_s") if t.get("total_wall_s") is not None else data.get("total_wall_s"),
        "total_wall_hms": t.get("total_wall_hms") or _fmt_hms(
            t.get("total_wall_s") if t.get("total_wall_s") is not None else data.get("total_wall_s")
        ),
        "mean_s_per_question": mean_s if reliable else None,
        "mean_s_per_question_hms": gen.get("mean_hms") or _fmt_hms(mean_s) if reliable else None,
        "p50_s_per_question": gen.get("p50_s"),
        "p90_s_per_question": gen.get("p90_s"),
        "timing_reliable": reliable,
        "timing_skip_reason": skip_reason,
        "host": data.get("host"),
        "env": data.get("env"),
        "memory": data.get("memory"),
        "timing_json": str(timing_path) if timing_path else None,
    }


def collect_run_inference_timing(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    preds_dir = run_root / "eval" / "predictions"
    dataset_key = _infer_dataset_key(run_root)
    by_condition: dict[str, Any] = {}
    if preds_dir.is_dir():
        for timing_path in sorted(preds_dir.glob("*/timing.json")):
            data = json.loads(timing_path.read_text(encoding="utf-8"))
            cond_dir = timing_path.parent
            pred_path = cond_dir / "predictions.jsonl"
            n_preds = sum(1 for _ in pred_path.open()) if pred_path.is_file() else None
            manifest_path = run_root / "cross_model_manifest.json"
            model_slug = None
            if manifest_path.is_file():
                model_slug = json.loads(manifest_path.read_text(encoding="utf-8")).get("model_slug")
            row = normalize_prediction_timing(
                data,
                timing_path=timing_path,
                dataset_key=dataset_key,
                model_slug=model_slug,
                n_preds=n_preds,
            )
            cond = row.get("condition") or timing_path.parent.name
            by_condition[str(cond)] = row

    manifest_path = run_root / "cross_model_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "model_slug": manifest.get("model_slug"),
        "base_model": manifest.get("base_model"),
        "dataset": manifest.get("dataset"),
        "protocol": manifest.get("protocol"),
        "n_conditions": len(by_condition),
        "by_condition": by_condition,
        "notes": [
            "mean_s_per_question: wall time per answer during greedy decode (batch size 1 for HF).",
            "load_type dense = merged weights (Ours); lora = PEFT adapter at inference.",
            "Source: eval/predictions/<condition>/timing.json from eval-repliqa-generate or eval-drop-generate.",
        ],
    }


def collect_cross_model_inference_timing(cross_root: Path) -> dict[str, Any]:
    cross_root = cross_root.resolve()
    runs: list[dict[str, Any]] = []
    for slug_dir in sorted(cross_root.iterdir()):
        if not slug_dir.is_dir():
            continue
        for dataset, subdir in DATASET_DIRS.items():
            run_root = slug_dir / subdir
            preds = run_root / "eval" / "predictions"
            if not preds.is_dir():
                continue
            row = collect_run_inference_timing(run_root)
            if not row.get("by_condition"):
                continue
            out = run_root / "eval" / "inference_timing.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _update_cross_model_manifest(run_root, row)
            runs.append(row)

    return {
        "schema": f"{SCHEMA}_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cross_root": str(cross_root),
        "n_runs": len(runs),
        "runs": runs,
    }


def _update_cross_model_manifest(run_root: Path, inference_doc: dict[str, Any]) -> None:
    manifest_path = run_root / "cross_model_manifest.json"
    doc: dict[str, Any] = {}
    if manifest_path.is_file():
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        cond: {
            "mean_s_per_question": row.get("mean_s_per_question"),
            "load_type": row.get("load_type"),
            "backend": row.get("backend"),
            "n_questions": row.get("n_questions"),
        }
        for cond, row in (inference_doc.get("by_condition") or {}).items()
    }
    doc["inference_timing"] = {
        "updated_at": inference_doc.get("created_at"),
        "path": str(run_root / "eval" / "inference_timing.json"),
        "by_condition": summary,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_collect_inference(ns: argparse.Namespace) -> int:
    cross_root = getattr(ns, "cross_root", None)
    if cross_root is not None:
        cross_root = Path(cross_root).expanduser().resolve()
        doc = collect_cross_model_inference_timing(cross_root)
        out = (
            Path(ns.output_json).expanduser().resolve()
            if ns.output_json
            else cross_root / "inference_timing_matrix.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out} ({doc['n_runs']} runs)", flush=True)
        return 0

    run_root = Path(ns.run_root).expanduser().resolve()
    doc = collect_run_inference_timing(run_root)
    if not doc.get("by_condition"):
        print(f"No timing.json under {run_root}/eval/predictions/", flush=True)
        return 1

    out = (
        Path(ns.output_json).expanduser().resolve()
        if ns.output_json
        else run_root / "eval" / "inference_timing.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _update_cross_model_manifest(run_root, doc)

    print(f"Wrote {out}", flush=True)
    hdr = f"{'condition':<24} {'load':<6} {'backend':<5} {'mean_s/q':>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for cond in sorted(doc["by_condition"]):
        row = doc["by_condition"][cond]
        print(
            f"{cond:<24} {str(row.get('load_type') or '-'):<6} "
            f"{str(row.get('backend') or '-'):<5} "
            f"{row.get('mean_s_per_question') if row.get('mean_s_per_question') is not None else 'n/a':>8}",
            flush=True,
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect eval generate / inference timing for cross-model runs.")
    p.add_argument("--run-root", type=Path, default=None, help="Single model/dataset run root.")
    p.add_argument(
        "--cross-root",
        type=Path,
        default=None,
        help="Aggregate all runs under cross_model/runs (writes inference_timing_matrix.json).",
    )
    p.add_argument("--output-json", type=Path, default=None)
    return p


if __name__ == "__main__":
    raise SystemExit(run_collect_inference(build_arg_parser().parse_args()))

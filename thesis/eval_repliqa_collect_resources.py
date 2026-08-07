"""
Aggregate train / generate / merge / judge wall times for a RepLiQA run.

Reads existing timing.json, run_manifest.json, qs_merge_timing_index.json, etc.
Writes eval/resource_timing.json (machine-readable).

Run:
  python -m thesis.cli eval-repliqa-collect-resources
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "repliqa_resource_timing/v1"

# Eval condition -> training artifacts (one-time shared costs noted in notes)
EVAL_TRAIN_SOURCES: dict[str, list[str]] = {
    "B3_lora_all": ["baselines/B3_all_lora_r16"],
    "Ours_equal_merge": [
        "baselines/qs_strat/QS_strat_high_lora_r32",
        "baselines/qs_strat/QS_strat_medium_lora_r16",
        "baselines/qs_strat/QS_strat_low_lora_r8",
        "merge:QS_merged_strat_dense",
    ],
    "Ours_tier_merge": [
        "baselines/qs_strat/QS_strat_high_lora_r32",
        "baselines/qs_strat/QS_strat_medium_lora_r16",
        "baselines/qs_strat/QS_strat_low_lora_r8",
        "merge:QS_merged_strat_dense_w60_30_10",
    ],
    "Ours_freq_merge": [
        "baselines/qs_strat/QS_strat_high_lora_r32",
        "baselines/qs_strat/QS_strat_medium_lora_r16",
        "baselines/qs_strat/QS_strat_low_lora_r8",
        "merge:QS_merged_strat_dense_freq",
    ],
    "Ours_high_medium_merge": [
        "baselines/qs_strat/QS_strat_high_lora_r32",
        "baselines/qs_strat/QS_strat_medium_lora_r16",
        "merge:QS_merged_strat_dense_high_med_w67_33_0",
    ],
    "Ours_high_only_lora": ["baselines/qs_strat/QS_strat_high_lora_r32"],
}

MERGE_DIR_BY_KEY = {
    "QS_merged_strat_dense": "baselines/qs_strat/QS_merged_strat_dense",
    "QS_merged_strat_dense_w60_30_10": "baselines/qs_strat/QS_merged_strat_dense_w60_30_10",
    "QS_merged_strat_dense_freq": "baselines/qs_strat/QS_merged_strat_dense_freq",
    "QS_merged_strat_dense_high_med_w67_33_0": (
        "baselines/qs_strat/QS_merged_strat_dense_high_med_w67_33_0"
    ),
}


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _read_train_manifest(run_root: Path, rel: str) -> dict[str, Any] | None:
    manifest = run_root / rel / "experiment/run_manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    timing = data.get("timing") or {}
    steps = timing.get("steps") or {}
    train_step = steps.get("lora_sft_train") or {}
    return {
        "path": str(rel),
        "baseline": data.get("baseline"),
        "total_wall_s": timing.get("total_wall_s"),
        "total_wall_hms": _fmt_hms(timing["total_wall_s"]) if timing.get("total_wall_s") else None,
        "lora_sft_train_s": train_step.get("total_s"),
        "hyperparameters": data.get("hyperparameters"),
        "host": data.get("host"),
        "env": data.get("env"),
        "finished_at": data.get("finished_at"),
    }


def _read_merge_timing(run_root: Path, merge_key: str) -> dict[str, Any] | None:
    rel = MERGE_DIR_BY_KEY.get(merge_key)
    if not rel:
        return None
    timing_path = run_root / rel / "qs_merge_timing.json"
    if timing_path.is_file():
        data = json.loads(timing_path.read_text(encoding="utf-8"))
        return {"path": rel, **data}
    index_path = run_root / "qs_merge_timing_index.json"
    if index_path.is_file():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        full = str((run_root / rel).resolve())
        if full in idx:
            return {"path": rel, **idx[full]}
    return None


def collect_resources(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    eval_dir = run_root / "eval"
    preds_dir = eval_dir / "predictions"

    generate: dict[str, Any] = {}
    for timing_path in sorted(preds_dir.glob("*/timing.json")):
        data = json.loads(timing_path.read_text(encoding="utf-8"))
        cond = data.get("condition") or timing_path.parent.name
        t = data.get("timing") or {}
        gen_q = t.get("generate_per_question") or {}
        generate[cond] = {
            "load_type": data.get("load_type"),
            "n_questions": data.get("n_questions") or data.get("n_rows"),
            "load_model_s": t.get("load_model_s") or data.get("load_s"),
            "generate_loop_s": t.get("generate_loop_s"),
            "total_wall_s": t.get("total_wall_s") or data.get("total_wall_s"),
            "total_wall_hms": t.get("total_wall_hms"),
            "mean_s_per_question": gen_q.get("mean_s") or (data.get("generate") or {}).get("mean_s"),
            "p50_s_per_question": gen_q.get("p50_s") or (data.get("generate") or {}).get("p50_s"),
            "host": data.get("host"),
            "env": data.get("env"),
            "memory": data.get("memory"),
        }

    train_runs: dict[str, Any] = {}
    for rel in sorted({s for sources in EVAL_TRAIN_SOURCES.values() for s in sources if not s.startswith("merge:")}):
        row = _read_train_manifest(run_root, rel)
        if row:
            train_runs[rel] = row

    merge_runs: dict[str, Any] = {}
    for key in MERGE_DIR_BY_KEY:
        row = _read_merge_timing(run_root, key)
        if row:
            merge_runs[key] = row

    judge: dict[str, Any] = {}
    judged_dir = eval_dir / "judged"
    if judged_dir.is_dir():
        for timing_path in sorted(judged_dir.glob("*/bedrock_judge_timing.json")):
            data = json.loads(timing_path.read_text(encoding="utf-8"))
            cond = timing_path.parent.name
            judge[cond] = {
                "total_wall_s": data.get("total_wall_s"),
                "total_wall_hms": _fmt_hms(data["total_wall_s"]) if data.get("total_wall_s") else None,
                "mean_request_s": data.get("mean_request_s"),
                "n_rows": 2000,
            }

    listwise: dict[str, Any] = {}
    lw_summary = eval_dir / "listwise_rank/listwise_rank_summary.json"
    if lw_summary.is_file():
        data = json.loads(lw_summary.read_text(encoding="utf-8"))
        lw_timing = (data.get("timing") or {})
        listwise = {
            "total_wall_s": lw_timing.get("total_wall_s"),
            "total_wall_hms": _fmt_hms(lw_timing["total_wall_s"]) if lw_timing.get("total_wall_s") else None,
            "n_questions": data.get("n_questions_attempted"),
            "model": data.get("model"),
        }

    # Per-condition train cost rollup (shared QS trains counted once per group in notes)
    qs_shared_s = sum(
        (train_runs.get(p) or {}).get("total_wall_s") or 0
        for p in (
            "baselines/qs_strat/QS_strat_high_lora_r32",
            "baselines/qs_strat/QS_strat_medium_lora_r16",
            "baselines/qs_strat/QS_strat_low_lora_r8",
        )
    )

    per_condition_train: dict[str, Any] = {}
    for cond, sources in EVAL_TRAIN_SOURCES.items():
        train_s = 0.0
        merge_s = 0.0
        parts: list[str] = []
        for src in sources:
            if src.startswith("merge:"):
                key = src.split(":", 1)[1]
                m = merge_runs.get(key) or {}
                ms = m.get("total_wall_s") or 0
                merge_s += ms
                parts.append(f"merge {key} ({ms:.0f}s)")
            else:
                tr = train_runs.get(src) or {}
                ts = tr.get("total_wall_s") or 0
                train_s += ts
                parts.append(f"train {src} ({ts:.0f}s)")
        per_condition_train[cond] = {
            "train_wall_s": train_s if train_s else None,
            "merge_wall_s": merge_s if merge_s else None,
            "incremental_wall_s": (train_s + merge_s) if (train_s or merge_s) else 0,
            "sources": parts,
            "notes": (
                "QS strat high/med/low trains are shared across all Ours_*_merge conditions; "
                f"combined one-time train wall ≈ {qs_shared_s:.0f}s ({_fmt_hms(qs_shared_s)})."
                if cond.startswith("Ours_") and "merge" in str(sources)
                else None
            ),
        }

    has_memory = any((g.get("memory") or {}).get("cuda_available") for g in generate.values())
    memory_note = (
        "Peak GPU memory logged in eval/predictions/*/timing.json (memory.*) when generate "
        "runs with thesis.gpu_memory_stats (after 2026-06-15)."
        if has_memory
        else (
            "Peak GPU memory was not logged in timing.json or run_manifest for this run. "
            "Re-run eval generate (or sbatch_gpu_memory_probe.sh) to populate memory fields."
        )
    )
    memory_note += (
        " SLURM eval jobs used 1× NVIDIA A100-SXM4-80GB (CUDA_VISIBLE_DEVICES=0). "
        "Train/generate use bf16 Llama-3.2-3B-Instruct, max_seq_length=4096, batch_size=1."
    )

    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "hardware": {
            "gpu": "NVIDIA A100-SXM4-80GB",
            "gpus_per_job": 1,
            "source": "SLURM eval/train job logs (nvidia-smi header)",
        },
        "memory_note": memory_note,
        "generate": generate,
        "train_runs": train_runs,
        "merge_runs": merge_runs,
        "per_condition_train": per_condition_train,
        "bedrock_judge": judge,
        "listwise_rank": listwise,
        "qs_strat_shared_train_wall_s": qs_shared_s,
    }


def run_eval_repliqa_collect_resources(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    out_path = Path(ns.output_json).expanduser().resolve() if ns.output_json else run_root / "eval/resource_timing.json"
    payload = collect_resources(run_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect RepLiQA train/generate/judge timing")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    return run_eval_repliqa_collect_resources(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

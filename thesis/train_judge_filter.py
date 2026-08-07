"""
Train Llama-3.2-3B LoRA training-filter judge (RepLiQA train/val splits).

  python -m thesis.cli prepare-judge-filter-sft ...
  python -m thesis.cli train-judge-filter --splits-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from thesis.experiment_log import ExperimentLogger

FINETUNING_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = FINETUNING_ROOT / "train_judge_filter_sft.py"


def run_train_judge_filter(ns: argparse.Namespace) -> int:
    wall0 = time.perf_counter()
    splits_dir = Path(ns.splits_dir).expanduser().resolve()
    output_dir = Path(ns.output_dir).expanduser().resolve()
    manifest_path = splits_dir / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "paths": {
                "train_jsonl": str(splits_dir / "train.jsonl"),
                "val_jsonl": str(splits_dir / "val.jsonl"),
            }
        }

    train_jsonl = Path(manifest["paths"]["train_jsonl"])
    val_jsonl = Path(manifest["paths"]["val_jsonl"])
    if not train_jsonl.is_file():
        print(f"Missing {train_jsonl} — run prepare-judge-filter-sft first", file=sys.stderr)
        return 1

    log = ExperimentLogger(run_dir=output_dir, baseline="judge_filter")
    log.update_section(
        "hyperparameters",
        {
            "model": str(ns.model),
            "lora_r": int(ns.lora_r),
            "lora_alpha": int(ns.lora_alpha),
            "lora_dropout": float(ns.lora_dropout),
            "epochs": int(ns.epochs),
            "learning_rate": float(ns.lr),
            "max_seq_length": int(ns.max_seq_length),
            "per_device_train_batch_size": int(ns.batch_size),
            "gradient_accumulation_steps": int(ns.grad_accum),
            "seed": int(ns.seed),
            "bf16": not ns.no_bf16,
        },
    )
    log.update_section(
        "data",
        {
            "splits_dir": str(splits_dir),
            "split_manifest": str(manifest_path),
            "n_train_rows": manifest.get("n_train_rows"),
            "n_val_rows": manifest.get("n_val_rows"),
            "train_judged_jsonl": manifest.get("train_judged_jsonl"),
            "test_judged_jsonl": manifest.get("test_judged_jsonl"),
        },
    )
    log.add_artifact("splits_train_jsonl", train_jsonl)
    log.add_artifact("splits_val_jsonl", val_jsonl)
    log.add_artifact("split_manifest", manifest_path)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_jsonl",
        str(train_jsonl),
        "--val_jsonl",
        str(val_jsonl),
        "--output_dir",
        str(output_dir),
        "--model_name",
        str(ns.model),
        "--num_train_epochs",
        str(ns.epochs),
        "--learning_rate",
        str(ns.lr),
        "--lora_r",
        str(ns.lora_r),
        "--lora_alpha",
        str(ns.lora_alpha),
        "--lora_dropout",
        str(ns.lora_dropout),
        "--max_seq_length",
        str(ns.max_seq_length),
        "--per_device_train_batch_size",
        str(ns.batch_size),
        "--gradient_accumulation_steps",
        str(ns.grad_accum),
        "--seed",
        str(ns.seed),
    ]
    if not ns.no_bf16:
        cmd.append("--bf16")
    if ns.use_qlora_4bit:
        cmd.append("--use_qlora_4bit")

    cmd_path = log.exp_dir / "train_command.txt"
    cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    log.add_artifact("train_command", cmd_path)
    print("=== Judge-filter LoRA SFT ===", flush=True)
    print("Command:", " ".join(cmd), flush=True)

    train_t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(FINETUNING_ROOT))
    train_wall_s = time.perf_counter() - train_t0
    log.emit_span(
        "judge_filter_lora_train",
        train_wall_s,
        status="ok" if rc == 0 else "error",
        extra={"exit_code": rc, "output_dir": str(output_dir)},
    )
    log.ingest_trainer_state(output_dir)
    log.add_artifact("lora_adapter_dir", output_dir)
    log.finalize(status="ok" if rc == 0 else "error", started_wall=wall0)
    print(f"Experiment log: {log.manifest_path}", flush=True)
    if rc == 0:
        print(f"Adapter saved to {output_dir}", flush=True)
    return int(rc)

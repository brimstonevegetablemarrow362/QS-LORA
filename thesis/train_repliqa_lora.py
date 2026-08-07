"""
Train answerer LoRA (B3: all usable synthetic Q/A) with experiment timing + manifest.

Run on a GPU node:
  python -m thesis.cli train-repliqa-lora \\
    --baseline B3 \\
    --qa-jsonl thesis/experiments/repliqa/runs/repliqa_train_0-3/train/synthetic_qa.jsonl \\
    --output-dir thesis/experiments/repliqa/runs/repliqa_train_0-3/baselines/B3_all_lora_r16
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
from thesis.prepare_repliqa_sft_splits import prepare_splits

FINETUNING_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = FINETUNING_ROOT / "train_normalized_qa_sft.py"


def run_train_repliqa_lora(ns: argparse.Namespace) -> int:
    wall0 = time.perf_counter()
    qa_jsonl = Path(ns.qa_jsonl).expanduser().resolve()
    output_dir = Path(ns.output_dir).expanduser().resolve()
    splits_dir = (
        Path(ns.splits_dir).expanduser().resolve()
        if ns.splits_dir
        else output_dir.parent / "splits" / "sft_all"
    )

    log = ExperimentLogger(run_dir=output_dir, baseline=str(ns.baseline))
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
            "effective_batch_size": int(ns.batch_size) * int(ns.grad_accum),
            "seed": int(ns.seed),
            "bf16": not ns.no_bf16,
            "use_qlora_4bit": bool(ns.use_qlora_4bit),
            "use_context": not bool(getattr(ns, "no_context", False)),
            "peft_type": str(getattr(ns, "peft_type", "lora") or "lora"),
            "adalora_init_r": int(getattr(ns, "adalora_init_r", 16)),
            "adalora_target_r": int(getattr(ns, "adalora_target_r", 16)),
            "adalora_tinit_ratio": float(getattr(ns, "adalora_tinit_ratio", 0.1)),
            "adalora_tfinal_ratio": float(getattr(ns, "adalora_tfinal_ratio", 0.9)),
            "adalora_delta_t": int(getattr(ns, "adalora_delta_t", 10)),
        },
    )
    train_filter = "all_usable_rows_skip_nan"
    if ns.quality_tier:
        train_filter = f"haiku_quality_tier_{ns.quality_tier}"
    log.update_section(
        "data",
        {
            "qa_jsonl": str(qa_jsonl),
            "splits_dir": str(splits_dir),
            "val_ratio": float(ns.val_ratio),
            "train_filter": train_filter,
            "quality_tier": ns.quality_tier or None,
            "sft_use_context": not bool(getattr(ns, "no_context", False)),
        },
    )
    log.add_artifact("qa_jsonl", qa_jsonl)

    split_manifest: dict[str, Any] = {}
    if not ns.skip_prepare:
        with log.span("prepare_sft_splits", val_ratio=float(ns.val_ratio)):
            split_manifest = prepare_splits(
                qa_jsonl=qa_jsonl,
                out_dir=splits_dir,
                val_ratio=float(ns.val_ratio),
                seed=int(ns.seed),
                quality_tier=ns.quality_tier or None,
            )
    else:
        manifest_path = splits_dir / "split_manifest.json"
        if manifest_path.is_file():
            split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            split_manifest = {
                "train_jsonl": str(splits_dir / "train.jsonl"),
                "val_jsonl": str(splits_dir / "val.jsonl"),
            }

    log.add_artifact("splits_train_jsonl", split_manifest.get("train_jsonl", splits_dir / "train.jsonl"))
    log.add_artifact("splits_val_jsonl", split_manifest.get("val_jsonl", splits_dir / "val.jsonl"))
    log.add_artifact("split_manifest", splits_dir / "split_manifest.json")
    log.update_section(
        "data",
        {
            "n_train_rows": split_manifest.get("n_train_rows"),
            "n_val_rows": split_manifest.get("n_val_rows"),
            "n_skipped_unusable": split_manifest.get("n_skipped_unusable"),
            "n_input_rows": split_manifest.get("n_input_rows"),
            "n_after_quality_tier": split_manifest.get("n_after_quality_tier"),
            "filter_quality_tier": split_manifest.get("filter_quality_tier"),
            "split_manifest": split_manifest,
        },
    )

    train_jsonl = Path(split_manifest.get("train_jsonl", splits_dir / "train.jsonl"))
    val_jsonl = Path(split_manifest.get("val_jsonl", splits_dir / "val.jsonl"))
    if not train_jsonl.is_file():
        print(f"Missing {train_jsonl}", file=sys.stderr)
        log.finalize(status="error", started_wall=wall0)
        return 1
    if not val_jsonl.is_file():
        val_jsonl.write_text("", encoding="utf-8")

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
    if getattr(ns, "no_context", False):
        cmd.append("--no-context")
    peft_type = str(getattr(ns, "peft_type", "lora") or "lora").lower()
    if peft_type == "adalora":
        cmd.extend(
            [
                "--peft_type",
                "adalora",
                "--adalora_init_r",
                str(int(getattr(ns, "adalora_init_r", 16))),
                "--adalora_target_r",
                str(int(getattr(ns, "adalora_target_r", 16))),
                "--adalora_tinit_ratio",
                str(float(getattr(ns, "adalora_tinit_ratio", 0.1))),
                "--adalora_tfinal_ratio",
                str(float(getattr(ns, "adalora_tfinal_ratio", 0.9))),
                "--adalora_delta_t",
                str(int(getattr(ns, "adalora_delta_t", 10))),
            ]
        )
    else:
        cmd.extend(["--peft_type", "lora"])

    cmd_path = log.exp_dir / "train_command.txt"
    cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    log.add_artifact("train_command", cmd_path)

    print("=== LoRA SFT ===", flush=True)
    print("Command:", " ".join(cmd), flush=True)
    train_t0 = time.perf_counter()
    rc = subprocess.call(cmd)
    train_wall_s = time.perf_counter() - train_t0
    log.emit_span(
        "lora_sft_train",
        train_wall_s,
        status="ok" if rc == 0 else "error",
        extra={"exit_code": rc, "output_dir": str(output_dir)},
    )

    log.ingest_trainer_state(output_dir)
    log.add_artifact("lora_adapter_dir", output_dir)

    manifest = log.finalize(status="ok" if rc == 0 else "error", started_wall=wall0)
    print(f"Experiment log: {log.manifest_path}", flush=True)
    print(f"Total wall time: {manifest['timing']['total_wall_s']}s", flush=True)
    if log._manifest.get("trainer", {}).get("metrics", {}).get("train_runtime_s"):
        print(f"HF train_runtime: {log._manifest['trainer']['metrics']['train_runtime_s']}s", flush=True)
    if rc == 0:
        print(f"Adapter saved to {output_dir}", flush=True)
    return int(rc)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RepLiQA LoRA SFT with experiment timing manifest.")
    p.add_argument("--baseline", type=str, default="B3", help="Baseline id for manifest (default B3).")
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="LoRA adapter + experiment/ logs.")
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--val-ratio", type=float, default=0.1, help="0 = train-only (no val docs).")
    p.add_argument(
        "--quality-tier",
        type=str,
        default=None,
        help="Filter to llm_judge tier (e.g. high). Use synthetic_qa_haiku_judge.jsonl as --qa-jsonl.",
    )
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--use-qlora-4bit", action="store_true")
    p.add_argument(
        "--no-context",
        action="store_true",
        help="SFT prompt is question-only -> answer (closed-book deploy track).",
    )
    p.add_argument(
        "--peft-type",
        type=str,
        choices=("lora", "adalora"),
        default="lora",
        help="LoRA (fixed rank) or AdaLoRA (adaptive rank budget).",
    )
    p.add_argument("--adalora-init-r", type=int, default=16, help="AdaLoRA initial rank (match B3 r=16).")
    p.add_argument("--adalora-target-r", type=int, default=16, help="AdaLoRA target average rank.")
    p.add_argument("--adalora-tinit-ratio", type=float, default=0.1, help="Fraction of steps before rank pruning.")
    p.add_argument("--adalora-tfinal-ratio", type=float, default=0.9, help="Fraction of steps when rank budget fixed.")
    p.add_argument("--adalora-delta-t", type=int, default=10, help="AdaLoRA rank adjustment interval.")
    return p


if __name__ == "__main__":
    raise SystemExit(run_train_repliqa_lora(build_arg_parser().parse_args()))

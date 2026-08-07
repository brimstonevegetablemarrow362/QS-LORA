"""
Train DROP domain CPT (Stage 1) with experiment logging.

  python -m thesis.cli prepare-drop-cpt-corpus ...
  python -m thesis.cli train-drop-cpt --corpus-dir ... --output-dir ...
  python -m thesis.cli merge-lora-dense --adapter-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from thesis.experiment_log import ExperimentLogger

FINETUNING_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = FINETUNING_ROOT / "train_domain_cpt.py"


def run_train_drop_cpt(ns: argparse.Namespace) -> int:
    wall0 = time.perf_counter()
    corpus_dir = Path(ns.corpus_dir).expanduser().resolve()
    output_dir = Path(ns.output_dir).expanduser().resolve()
    manifest_path = corpus_dir / "split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_jsonl = Path(manifest["paths"]["passages_train_jsonl"])
        val_jsonl = Path(manifest["paths"]["passages_val_jsonl"])
    else:
        train_jsonl = corpus_dir / "passages_train.jsonl"
        val_jsonl = corpus_dir / "passages_val.jsonl"

    if not train_jsonl.is_file():
        print(f"Missing {train_jsonl} — run prepare-drop-cpt-corpus first", file=sys.stderr)
        return 1

    log = ExperimentLogger(run_dir=output_dir, baseline="drop_cpt")
    log.update_section(
        "hyperparameters",
        {
            "model": str(ns.model),
            "task": "domain_cpt_causal_lm",
            "lora_r": int(ns.lora_r),
            "lora_alpha": int(ns.lora_alpha),
            "epochs": int(ns.epochs),
            "learning_rate": float(ns.lr),
            "max_seq_length": int(ns.max_seq_length),
            "per_device_train_batch_size": int(ns.batch_size),
            "gradient_accumulation_steps": int(ns.grad_accum),
            "seed": int(ns.seed),
            "bf16": not ns.no_bf16,
            "use_qlora_4bit": bool(getattr(ns, "use_qlora_4bit", False)),
        },
    )
    log.update_section(
        "data",
        {
            "corpus_dir": str(corpus_dir),
            "split_manifest": str(manifest_path) if manifest_path.is_file() else None,
            "n_train_passages": manifest.get("n_train_passages") if manifest_path.is_file() else None,
            "n_val_passages": manifest.get("n_val_passages") if manifest_path.is_file() else None,
        },
    )
    log.add_artifact("passages_train_jsonl", train_jsonl)
    log.add_artifact("passages_val_jsonl", val_jsonl)

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
    if getattr(ns, "use_qlora_4bit", False):
        cmd.append("--use_qlora_4bit")

    cmd_path = log.exp_dir / "train_command.txt"
    cmd_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    log.add_artifact("train_command", cmd_path)

    print("=== DROP domain CPT ===", flush=True)
    print("Command:", " ".join(cmd), flush=True)
    train_t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(FINETUNING_ROOT))
    train_wall_s = time.perf_counter() - train_t0
    log.emit_span(
        "drop_cpt_train",
        train_wall_s,
        status="ok" if rc == 0 else "error",
        extra={"exit_code": rc, "output_dir": str(output_dir)},
    )
    log.ingest_trainer_state(output_dir)
    log.add_artifact("lora_adapter_dir", output_dir)
    log.finalize(status="ok" if rc == 0 else "error", started_wall=wall0)
    print(f"Experiment log: {log.manifest_path}", flush=True)
    if rc == 0:
        print(f"CPT adapter saved to {output_dir}", flush=True)
    return int(rc)

"""
Merge a single LoRA adapter into a dense HuggingFace model folder.

Used after domain CPT (Stage 1) so Stage 2 QA training uses one base path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def merge_lora_dense(
    *,
    base_model: str,
    adapter_dir: Path,
    output_dir: Path,
    bf16: bool = True,
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = adapter_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Loading base {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu",
    )
    print(f"Loading adapter {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("Merging and unloading …", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "schema": "merge_lora_dense/v1",
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(output_dir),
        "bf16": bf16,
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote dense model to {output_dir}", flush=True)
    return manifest


def run_merge(ns: argparse.Namespace) -> int:
    manifest = merge_lora_dense(
        base_model=str(ns.base_model),
        adapter_dir=Path(ns.adapter_dir),
        output_dir=Path(ns.output_dir),
        bf16=not ns.no_bf16,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge one LoRA adapter into dense HF model")
    p.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--no-bf16", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run_merge(build_arg_parser().parse_args()))

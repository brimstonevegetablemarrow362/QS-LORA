#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into the base weights for vLLM (single-folder checkpoint).

Usage:
  python merge_lora.py \\
    --base meta-llama/Llama-3.2-3B-Instruct \\
    --adapter ../out/qa-sft-lora \\
    --out ../out/qa-sft-merged

Install: pip install -r ../requirements-train.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA into base model")
    ap.add_argument("--base", type=str, required=True, help="Base model HF id or local path")
    ap.add_argument("--adapter", type=Path, required=True, help="PEFT adapter directory")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for merged weights")
    ap.add_argument("--bf16", action="store_true", help="Load/merge in bfloat16")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.bf16 else None
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()

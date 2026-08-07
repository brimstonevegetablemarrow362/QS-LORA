#!/usr/bin/env python3
"""
LoRA SFT for the **generator** task on normalized JSONL (same format as custom/normalized/*.jsonl).

Each row: context, question, answer, source (source is not shown to the model).
Teaches: excerpt → JSON {"question","answer"} matching train_peerqa_generator_sft.py.

Use merged corpora or a single file, e.g.:
  custom/normalized/all.jsonl
  custom/normalized/pubmedqa.jsonl

Usage:
  python train_normalized_generator_sft.py --output_dir ./out/normalized-generator-lora --bf16

  # Use existing train/val split (from split_normalized_dataset.py); test.jsonl is not used here (held out).
  python train_normalized_generator_sft.py --use-normalized-splits --output_dir ./out/normalized-generator-lora --bf16
  # or explicit paths:
  python train_normalized_generator_sft.py \\
    --train_jsonl custom/normalized/splits/train.jsonl \\
    --val_jsonl custom/normalized/splits/val.jsonl \\
    --output_dir ./out/normalized-generator-lora --bf16

Then merge the LoRA into a single folder and serve with vLLM; synthetic Q/A uses
`generate_qa_from_chunks.py --vllm-model <merged-path>` (see domain_v1 docstring).

Evaluate on held-out test split:
  python eval_normalized_generator_greedy.py --adapter ./out/normalized-generator-lora --bf16

Requires: pip install -r requirements_sft.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "custom" / "normalized" / "all.jsonl"
DEFAULT_SPLIT_TRAIN = SCRIPT_DIR / "custom" / "normalized" / "splits" / "train.jsonl"
DEFAULT_SPLIT_VAL = SCRIPT_DIR / "custom" / "normalized" / "splits" / "val.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} JSON error: {e}") from e
    return rows


def filter_usable(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        ctx = (r.get("context") or "").strip()
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if ctx and q and a:
            out.append({"context": ctx, "question": q, "answer": a})
    return out


def row_to_messages(row: Dict[str, str], system_prompt: str) -> Dict[str, Any]:
    """Same target format as train_peerqa_generator_sft.row_to_messages."""
    payload = json.dumps(
        {"question": row["question"], "answer": row["answer"]},
        ensure_ascii=False,
    )
    user_block = (
        "You are given an excerpt from a scientific paper (evidence passages may be "
        "fragmented). Produce ONE question and ONE short answer that are fully "
        "grounded in the excerpt only.\n\n"
        "Excerpt:\n"
        + row["context"]
        + "\n\nRespond with a single JSON object with keys \"question\" and \"answer\" only. "
        "No markdown fences."
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
            {"role": "assistant", "content": payload},
        ]
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generator LoRA: normalized context → JSON question+answer"
    )
    p.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="One or more normalized .jsonl files (used if not using --train_jsonl / --use-normalized-splits)",
    )
    p.add_argument(
        "--train_jsonl",
        type=Path,
        default=None,
        help="Pre-split training JSONL (with --val_jsonl); ignores --input and --eval_fraction",
    )
    p.add_argument(
        "--val_jsonl",
        type=Path,
        default=None,
        help="Pre-split validation JSONL for eval during SFT (optional)",
    )
    p.add_argument(
        "--use-normalized-splits",
        action="store_true",
        help=f"Use {DEFAULT_SPLIT_TRAIN} and {DEFAULT_SPLIT_VAL}",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval_fraction",
        type=float,
        default=0.05,
        help="Random holdout fraction when loading from --input only (ignored if --train_jsonl set)",
    )
    p.add_argument("--max_rows", type=int, default=0, help="0 = use all rows (after filter)")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    args = p.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            "Missing deps. Install with:\n"
            f"  pip install -r {SCRIPT_DIR / 'requirements_sft.txt'}\n"
            f"Original error: {e}"
        ) from e

    train_rows: list[dict[str, str]]
    eval_rows: list[dict[str, str]]

    if args.use_normalized_splits:
        train_path, val_path = DEFAULT_SPLIT_TRAIN, DEFAULT_SPLIT_VAL
        if not train_path.is_file():
            raise SystemExit(f"Missing {train_path}; run custom/split_normalized_dataset.py first.")
        train_rows = filter_usable(load_jsonl(train_path))
        eval_rows = filter_usable(load_jsonl(val_path)) if val_path.is_file() else []
        print(f"From --use-normalized-splits: train {train_path} ({len(train_rows)} usable)", flush=True)
        if val_path.is_file():
            print(f"  val {val_path} ({len(eval_rows)} usable)", flush=True)
        else:
            print(f"  (no val file at {val_path})", flush=True)
    elif args.train_jsonl is not None:
        tp = Path(args.train_jsonl)
        if not tp.is_file():
            raise SystemExit(f"train_jsonl not found: {tp}")
        train_rows = filter_usable(load_jsonl(tp))
        eval_rows = []
        if args.val_jsonl is not None:
            vp = Path(args.val_jsonl)
            if not vp.is_file():
                raise SystemExit(f"val_jsonl not found: {vp}")
            eval_rows = filter_usable(load_jsonl(vp))
        print(f"From --train_jsonl: {len(train_rows)} train, {len(eval_rows)} val", flush=True)
    else:
        inputs = args.input if args.input is not None else [DEFAULT_INPUT]
        all_rows: list[dict[str, str]] = []
        for path in inputs:
            if not path.is_file():
                raise SystemExit(f"Input not found: {path}")
            raw = load_jsonl(path)
            usable = filter_usable(raw)
            all_rows.extend(usable)
            print(f"Loaded {len(raw)} lines from {path} -> {len(usable)} usable", flush=True)

        if not all_rows:
            raise SystemExit("No usable rows (need non-empty context, question, answer).")

        if args.max_rows > 0:
            rng = random.Random(args.seed)
            rng.shuffle(all_rows)
            all_rows = all_rows[: args.max_rows]
            print(f"Capped to max_rows={args.max_rows}", flush=True)

        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
        n = len(all_rows)
        if n <= 1:
            train_rows, eval_rows = all_rows, []
        else:
            n_eval = max(1, int(n * args.eval_fraction))
            if n_eval >= n:
                n_eval = max(1, n // 10)
            eval_rows = all_rows[:n_eval]
            train_rows = all_rows[n_eval:]
            if not train_rows:
                train_rows, eval_rows = all_rows[:-1], all_rows[-1:]
        print(f"Random split from --input: train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    if not train_rows:
        raise SystemExit("No training rows after filtering.")

    if args.max_rows > 0 and (args.train_jsonl is not None or args.use_normalized_splits):
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)
        train_rows = train_rows[: args.max_rows]
        print(f"Capped train to max_rows={args.max_rows}", flush=True)

    print(f"Final train={len(train_rows)} eval={len(eval_rows)}", flush=True)

    system_prompt = (
        "You write exam-style question and answer pairs for scientific text. "
        "Answers must not introduce facts outside the given excerpt."
    )
    train_ds = Dataset.from_list([row_to_messages(r, system_prompt) for r in train_rows])
    eval_ds = (
        Dataset.from_list([row_to_messages(r, system_prompt) for r in eval_rows])
        if eval_rows
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = os.environ.get("PEERQA_QLORA", "").strip() in ("1", "true", "yes")
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    torch_dtype = torch.bfloat16 if (args.bf16 and not args.no_bf16) else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto" if bnb_config else None,
        torch_dtype=torch_dtype,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    bf16 = args.bf16 and not args.no_bf16
    fp16 = args.fp16 and not bf16

    def formatting_func(example: Dict[str, Any]) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=bf16,
        fp16=fp16,
        max_length=args.max_seq_length,
        packing=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved generator adapter + tokenizer to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

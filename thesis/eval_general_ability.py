"""
Closed-book general-ability eval: base Instruct vs RepLiQA QS dense merge.

Generate answers for the frozen 100-question bank, then score with Bedrock Haiku
(v3_eval_gold; primary metric = gold_alignment).

  python -m thesis.cli eval-general-ability-generate \\
    --model-path meta-llama/Llama-3.2-3B-Instruct \\
    --condition-id base_llama32_3b \\
    --out-dir .../general_ability/llama32_3b/base

  python -m thesis.cli eval-general-ability-generate \\
    --model-path .../QS_merged_strat_dense_w60_30_10 \\
    --condition-id ft_repliqa_ours_llama32_3b \\
    --out-dir .../general_ability/llama32_3b/ft_repliqa
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parent
    / "experiments/analysis/general_ability/general_eval_questions_100.jsonl"
)

SYSTEM_NO_CTX = (
    "You are a helpful assistant. Answer the question clearly and concisely. "
    "Follow any formatting instructions exactly."
)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_user(row: dict[str, Any]) -> str:
    return (row.get("question") or "").strip()


def run_eval_general_ability_generate(ns: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    q_path = Path(ns.questions_jsonl).expanduser().resolve()
    out_dir = Path(ns.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    timing_path = out_dir / "timing.json"

    rows = _load_rows(q_path)
    if ns.max_rows and ns.max_rows > 0:
        rows = rows[: int(ns.max_rows)]

    model_path = str(ns.model_path)
    condition_id = str(ns.condition_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if ns.bf16 and device == "cuda" else None

    print(f"Loading {model_path} on {device} …", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s; n={len(rows)}", flush=True)

    max_new = int(ns.max_new_tokens)
    gen_times: list[float] = []
    with pred_path.open("w", encoding="utf-8") as fp:
        for i, row in enumerate(rows):
            messages = [
                {"role": "system", "content": SYSTEM_NO_CTX},
                {"role": "user", "content": _build_user(row)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            dt = time.perf_counter() - t1
            gen_times.append(dt)
            new_tokens = out[0][inputs["input_ids"].shape[-1] :]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            gold = (row.get("gold") or row.get("answer") or "").strip()
            rec = {
                "eval_id": row.get("eval_id"),
                "bucket": row.get("bucket"),
                "document_id": row.get("document_id"),
                "chunk_id": row.get("chunk_id"),
                "model_id": condition_id,
                "condition": condition_id,
                "question": row.get("question"),
                "gold": gold,
                "pred": pred,
                "context": row.get("context") or "",
                "use_context": False,
                "source": row.get("source"),
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i+1}/{len(rows)}  last={dt:.2f}s", flush=True)

    timing = {
        "schema": "general_ability_generate_timing/v1",
        "created_at": _utc_iso(),
        "model_path": model_path,
        "condition_id": condition_id,
        "questions_jsonl": str(q_path),
        "n_questions": len(rows),
        "load_s": round(load_s, 3),
        "mean_generate_s": round(sum(gen_times) / max(len(gen_times), 1), 3),
        "total_generate_s": round(sum(gen_times), 3),
        "device": device,
        "bf16": bool(ns.bf16),
        "max_new_tokens": max_new,
    }
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {timing_path}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate closed-book general-ability answers")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--condition-id", type=str, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--questions-jsonl", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_arg_parser().parse_args(argv)
    if ns.no_bf16:
        ns.bf16 = False
    return run_eval_general_ability_generate(ns)


if __name__ == "__main__":
    raise SystemExit(main())

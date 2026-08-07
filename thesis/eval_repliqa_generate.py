"""
Generate answers on RepLiQA eval subset (greedy by default; pass --temperature > 0 to sample).

Outputs per condition:
  eval/predictions/<model_id>/predictions.jsonl
  eval/predictions/<model_id>/timing.json

Conditions (see EVAL_CONDITIONS below):
  B3 — LoRA all-data
  B5 — AdaLoRA all-data
  Ours_tier / Ours_equal / Ours_freq — dense QS merges

Usage (from finetuning/, GPU node):
  python -m thesis.cli eval-repliqa-generate --list-conditions
  python -m thesis.cli eval-repliqa-generate --condition B3_lora_all --max-rows 20
  python -m thesis.cli eval-repliqa-generate --condition B3_lora_all
  python -m thesis.cli eval-repliqa-generate --condition Ours_tier_merge
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.gpu_memory_stats import (
    collect_gpu_memory_snapshot,
    merge_memory_phases,
    reset_peak_gpu_memory,
)
from thesis.eval_row_slice import resolve_pred_output_path, slice_eval_rows

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT_CTX = (
    "You answer questions using only the provided context. "
    "Stay grounded in that text and do not invent information."
)

SYSTEM_PROMPT_NO_CTX = (
    "You answer questions concisely. Use general knowledge; no document is provided."
)


@dataclass(frozen=True)
class EvalCondition:
    model_id: str
    description: str
    load_type: str  # base | lora | dense
    use_context: bool
    adapter_dir: str | None = None
    dense_dir: str | None = None


def _conditions(run_root: Path) -> dict[str, EvalCondition]:
    baselines = run_root / "baselines"
    qs = baselines / "qs_strat"
    return {
        "B3_lora_all": EvalCondition(
            model_id="B3_lora_all",
            description="B3 standard LoRA, all synthetic pairs r=16",
            load_type="lora",
            use_context=True,
            adapter_dir=str(baselines / "B3_all_lora_r16"),
        ),
        "B5_adalora_all": EvalCondition(
            model_id="B5_adalora_all",
            description="B5 AdaLoRA on base, all synthetic pairs r=16",
            load_type="lora",
            use_context=True,
            adapter_dir=str(baselines / "B5_adalora_r16"),
        ),
        "Ours_high_only_lora": EvalCondition(
            model_id="Ours_high_only_lora",
            description="QS-LoRA high tier specialist only (LoRA r=32, no dense merge)",
            load_type="lora",
            use_context=True,
            adapter_dir=str(qs / "QS_strat_high_lora_r32"),
        ),
        "Ours_high_medium_merge": EvalCondition(
            model_id="Ours_high_medium_merge",
            description="QS-LoRA dense merge high+medium only (low weight=0)",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_high_med_w67_33_0"),
        ),
        "Ours_tier_merge": EvalCondition(
            model_id="Ours_tier_merge",
            description="QS-LoRA dense merge weights 0.6/0.3/0.1",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_w60_30_10"),
        ),
        "Ours_equal_rank_merge": EvalCondition(
            model_id="Ours_equal_rank_merge",
            description="QS-LoRA equal-rank (r=16/16/16) dense merge 0.6/0.3/0.1",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_equal_rank_w60_30_10"),
        ),
        "Ours_equal_merge": EvalCondition(
            model_id="Ours_equal_merge",
            description="QS-LoRA dense merge equal 1/1/1",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense"),
        ),
        "Ours_freq_merge": EvalCondition(
            model_id="Ours_freq_merge",
            description="QS-LoRA dense merge frequency weights",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_freq"),
        ),
        "Ours_low_heavy_merge": EvalCondition(
            model_id="Ours_low_heavy_merge",
            description="QS-LoRA dense merge weights 0.4/0.4/0.2",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_w40_40_20"),
        ),
        "Ours_inverted_merge": EvalCondition(
            model_id="Ours_inverted_merge",
            description="QS-LoRA dense merge inverted weights 0.1/0.3/0.6",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_w10_30_60"),
        ),
    }


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_hms(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _timing_stats(times: list[float]) -> dict[str, Any]:
    if not times:
        return {"n": 0}
    st = sorted(times)
    n = len(st)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(p * n) - 1))
        return st[idx]

    return {
        "n": n,
        "sum_s": round(sum(st), 3),
        "mean_s": round(sum(st) / n, 3),
        "min_s": round(st[0], 3),
        "max_s": round(st[-1], 3),
        "p50_s": round(pct(0.50), 3),
        "p90_s": round(pct(0.90), 3),
        "mean_hms": _fmt_hms(sum(st) / n),
        "total_hms": _fmt_hms(sum(st)),
    }


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
                raise ValueError(f"{path}:{i} {e}") from e
    return rows


def build_user_block(
    row: dict[str, Any],
    *,
    use_context: bool,
    context_fraction: float = 1.0,
) -> str:
    q = (row.get("question") or "").strip()
    if not use_context:
        return f"Question: {q}\n\nAnswer concisely."
    ctx = (row.get("context") or "").strip()
    if context_fraction < 1.0:
        ctx = ctx[: max(0, int(len(ctx) * context_fraction))]
    return (
        "Context:\n"
        + ctx
        + "\n\nQuestion: "
        + q
        + "\n\nAnswer the question using only the context above. Be direct and concise."
    )


def load_model_and_tokenizer(
    cond: EvalCondition,
    *,
    base_model: str,
    bf16: bool,
) -> tuple[Any, Any, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    if cond.load_type == "dense":
        assert cond.dense_dir
        model_path = cond.dense_dir
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if bf16 else None
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
    elif cond.load_type == "lora":
        from peft import PeftModel

        assert cond.adapter_dir
        tokenizer = AutoTokenizer.from_pretrained(cond.adapter_dir, trust_remote_code=True)
        dtype = torch.bfloat16 if bf16 else None
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, cond.adapter_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        dtype = torch.bfloat16 if bf16 else None
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, time.perf_counter() - t0


def decoding_settings(
    *,
    temperature: float = 0.0,
    top_p: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    temp = float(temperature)
    greedy = temp <= 0.0
    return {
        "temperature": temp,
        "top_p": float(top_p),
        "seed": int(seed) if seed is not None else None,
        "greedy": greedy,
        "do_sample": not greedy,
    }


def generate_one(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    use_context: bool,
    context_fraction: float = 1.0,
    max_seq_length: int,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    seed: int | None = None,
    row_index: int = 0,
) -> tuple[str, float]:
    import torch
    from transformers import GenerationConfig

    dec = decoding_settings(temperature=temperature, top_p=top_p, seed=seed)
    system = SYSTEM_PROMPT_CTX if use_context else SYSTEM_PROMPT_NO_CTX
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_user_block(
                row, use_context=use_context, context_fraction=context_fraction
            ),
        },
    ]
    t0 = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    device = next(model.parameters()).device
    if isinstance(inputs, torch.Tensor):
        gen_in = {"input_ids": inputs.to(device)}
    else:
        gen_in = {k: v.to(device) for k, v in dict(inputs).items()}
    if "attention_mask" not in gen_in:
        gen_in["attention_mask"] = torch.ones_like(
            gen_in["input_ids"], dtype=torch.long, device=device
        )
    input_len = gen_in["input_ids"].shape[-1]
    if dec["seed"] is not None:
        torch.manual_seed(dec["seed"] + int(row_index))
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=dec["do_sample"],
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if dec["do_sample"]:
        gen_cfg.temperature = dec["temperature"]
        gen_cfg.top_p = dec["top_p"]
    with torch.no_grad():
        out = model.generate(**gen_in, generation_config=gen_cfg)
    pred = tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()
    return pred, time.perf_counter() - t0


def run_eval_repliqa_generate(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = Path(ns.eval_dir).expanduser().resolve()
    cond_map = _conditions(run_root)

    if ns.list_conditions:
        for cid, c in cond_map.items():
            print(f"  {cid}: {c.description} [{c.load_type}]", flush=True)
        return 0

    cond_id = str(ns.condition).strip()
    if cond_id not in cond_map:
        raise SystemExit(f"Unknown --condition {cond_id!r}. Use --list-conditions.")
    cond = cond_map[cond_id]

    use_context = bool(cond.use_context)
    if getattr(ns, "no_context", False):
        use_context = False
    context_fraction = float(getattr(ns, "context_fraction", 1.0) or 1.0)
    if context_fraction <= 0 or context_fraction > 1.0:
        raise SystemExit("--context-fraction must be in (0, 1].")
    if not use_context:
        context_fraction = 1.0

    model_id = str(getattr(ns, "condition_id", None) or cond.model_id).strip()
    if getattr(ns, "no_context", False) and not model_id.endswith("_no_ctx"):
        model_id = f"{cond.model_id}_no_ctx"
    elif context_fraction < 1.0 and "_ctx" not in model_id:
        pct = int(round(context_fraction * 100))
        model_id = f"{cond.model_id}_ctx{pct}"

    in_path = Path(ns.eval_jsonl).expanduser().resolve() if ns.eval_jsonl else eval_dir / ns.eval_input_name
    if not in_path.is_file():
        raise SystemExit(f"Eval subset not found: {in_path}")

    rows = load_jsonl(in_path)
    row_start = int(getattr(ns, "row_start", 0) or 0)
    row_end = int(getattr(ns, "row_end", 0) or 0)
    rows, row_start, row_end, total_rows = slice_eval_rows(
        rows, row_start=row_start, row_end=row_end, max_rows=int(ns.max_rows)
    )
    if not rows:
        raise SystemExit(f"No rows in slice [{row_start}:{row_end}) of {total_rows}")

    out_dir = (
        Path(ns.output_dir).expanduser().resolve()
        if ns.output_dir
        else eval_dir / "predictions" / model_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = out_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    pred_path = resolve_pred_output_path(
        out_dir, row_start=row_start, row_end=row_end, total_rows=total_rows
    )
    timing_path = out_dir / "timing.json"
    spans_path = exp_dir / "generate_spans.jsonl"
    manifest_path = exp_dir / "run_manifest.json"

    wall0 = time.perf_counter()
    started_at = _utc_iso()

    ctx_note = (
        "no context"
        if not use_context
        else (f"context first {context_fraction:.0%}" if context_fraction < 1.0 else "full context")
    )
    print(f"Condition: {model_id} ({cond_id}) — {cond.description} [{ctx_note}]", flush=True)
    print(
        f"Eval: {in_path} ({len(rows)} rows slice [{row_start}:{row_end}) of {total_rows})",
        flush=True,
    )

    decode = decoding_settings(
        temperature=float(getattr(ns, "temperature", 0.0) or 0.0),
        top_p=float(getattr(ns, "top_p", 0.95) or 0.95),
        seed=(int(ns.seed) if getattr(ns, "seed", None) is not None else None),
    )
    if decode["do_sample"]:
        print(
            f"Decoding: sample temperature={decode['temperature']} top_p={decode['top_p']} "
            f"seed={decode['seed']}",
            flush=True,
        )
    else:
        print("Decoding: greedy (temperature=0)", flush=True)

    backend = str(getattr(ns, "backend", "hf") or "hf").lower()
    load_s = 0.0
    mem_after_load: dict[str, Any] = {}
    gen_times: list[float] = []

    if backend == "vllm":
        from openai import OpenAI

        from thesis.eval_vllm_backend import check_vllm, generate_rows_vllm, openai_base_url

        vllm_base_url = getattr(ns, "vllm_base_url", None)
        vllm_model = getattr(ns, "vllm_model", None)
        if not vllm_base_url or not vllm_model:
            raise SystemExit("--backend vllm requires --vllm-base-url and --vllm-model")

        client = OpenAI(base_url=openai_base_url(str(vllm_base_url)), api_key="unused")
        print(
            f"vLLM backend: {openai_base_url(str(vllm_base_url))} model={vllm_model!r}",
            flush=True,
        )
        check_vllm(client)
        concurrency = int(getattr(ns, "concurrency", 4) or 4)
        t_gen0 = time.perf_counter()
        preds, gen_times = generate_rows_vllm(
            client=client,
            model=str(vllm_model),
            rows=rows,
            use_context=use_context,
            context_fraction=context_fraction,
            max_new_tokens=int(ns.max_new_tokens),
            concurrency=concurrency,
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            seed=decode["seed"],
            row_start=row_start,
        )
        generate_loop_s = time.perf_counter() - t_gen0
        mem_job_total: dict[str, Any] = {}
        with open(pred_path, "w", encoding="utf-8") as fp, open(
            spans_path, "w", encoding="utf-8"
        ) as span_f:
            for i, (row, pred, dt) in enumerate(zip(rows, preds, gen_times)):
                gold = (row.get("gold") or row.get("answer") or "").strip()
                eval_id = row.get("eval_id") or row.get("chunk_id")
                rec = {
                    "eval_id": eval_id,
                    "document_id": row.get("document_id"),
                    "chunk_id": row.get("chunk_id"),
                    "repliqa_split": row.get("repliqa_split"),
                    "document_topic": row.get("document_topic"),
                    "model_id": model_id,
                    "condition": cond_id,
                    "question": row.get("question"),
                    "gold": gold,
                    "pred": pred,
                    "use_context": use_context,
                    "context_fraction": context_fraction if use_context else None,
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                span_f.write(
                    json.dumps(
                        {
                            "step": "generate_one",
                            "index": i,
                            "eval_id": eval_id,
                            "duration_s": round(dt, 3),
                            "duration_hms": _fmt_hms(dt),
                            "wall_elapsed_s": round(time.perf_counter() - wall0, 3),
                            "ts": _utc_iso(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        reset_peak_gpu_memory()
        t_load0 = time.perf_counter()
        model, tokenizer, _ = load_model_and_tokenizer(
            cond, base_model=str(ns.base_model), bf16=bool(ns.bf16)
        )
        load_s = time.perf_counter() - t_load0
        mem_after_load = collect_gpu_memory_snapshot()
        print(f"Model loaded in {load_s:.1f}s ({_fmt_hms(load_s)})", flush=True)
        if mem_after_load.get("cuda_available"):
            print(
                f"  GPU after load: peak_alloc={mem_after_load['peak_allocated_gib']} GiB "
                f"peak_reserved={mem_after_load['peak_reserved_gib']} GiB",
                flush=True,
            )

        t_gen0 = time.perf_counter()
        with open(pred_path, "w", encoding="utf-8") as fp, open(
            spans_path, "w", encoding="utf-8"
        ) as span_f:
            for i, row in enumerate(rows):
                pred, dt = generate_one(
                    model,
                    tokenizer,
                    row,
                    use_context=use_context,
                    context_fraction=context_fraction,
                    max_seq_length=int(ns.max_seq_length),
                    max_new_tokens=int(ns.max_new_tokens),
                    temperature=decode["temperature"],
                    top_p=decode["top_p"],
                    seed=decode["seed"],
                    row_index=row_start + i,
                )
                gen_times.append(dt)
                gold = (row.get("gold") or row.get("answer") or "").strip()
                eval_id = row.get("eval_id") or row.get("chunk_id")
                rec = {
                    "eval_id": eval_id,
                    "document_id": row.get("document_id"),
                    "chunk_id": row.get("chunk_id"),
                    "repliqa_split": row.get("repliqa_split"),
                    "document_topic": row.get("document_topic"),
                    "model_id": model_id,
                    "condition": cond_id,
                    "question": row.get("question"),
                    "gold": gold,
                    "pred": pred,
                    "use_context": use_context,
                    "context_fraction": context_fraction if use_context else None,
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                span_f.write(
                    json.dumps(
                        {
                            "step": "generate_one",
                            "index": i,
                            "eval_id": eval_id,
                            "duration_s": round(dt, 3),
                            "duration_hms": _fmt_hms(dt),
                            "wall_elapsed_s": round(time.perf_counter() - wall0, 3),
                            "ts": _utc_iso(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if (i + 1) % 20 == 0 or i + 1 == len(rows):
                    mean_so_far = sum(gen_times) / len(gen_times)
                    print(
                        f"  generated {i + 1}/{len(rows)}  "
                        f"last={dt:.2f}s mean={mean_so_far:.2f}s",
                        flush=True,
                    )
        generate_loop_s = time.perf_counter() - t_gen0
        mem_job_total = collect_gpu_memory_snapshot()
        if mem_job_total.get("cuda_available"):
            print(
                f"  GPU job peak: alloc={mem_job_total['peak_allocated_gib']} GiB "
                f"reserved={mem_job_total['peak_reserved_gib']} GiB",
                flush=True,
            )

    total_s = time.perf_counter() - wall0
    finished_at = _utc_iso()
    gen_stats = _timing_stats(gen_times)

    timing = {
        "schema": "repliqa_eval_generate_timing/v2",
        "model_id": model_id,
        "condition": cond_id,
        "description": cond.description,
        "load_type": cond.load_type,
        "use_context": use_context,
        "context_fraction": context_fraction if use_context else None,
        "n_questions": len(rows),
        "row_start": row_start,
        "row_end": row_end,
        "total_rows": total_rows,
        "eval_jsonl": str(in_path),
        "predictions_jsonl": str(pred_path),
        "generate_spans_jsonl": str(spans_path),
        "run_manifest_json": str(manifest_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "host": socket.gethostname(),
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "base_model": str(ns.base_model) if cond.load_type != "dense" else None,
        "adapter_dir": cond.adapter_dir,
        "dense_dir": cond.dense_dir,
        "timing": {
            "load_model_s": round(load_s, 3),
            "load_model_hms": _fmt_hms(load_s),
            "generate_loop_s": round(generate_loop_s, 3),
            "generate_loop_hms": _fmt_hms(generate_loop_s),
            "generate_per_question": gen_stats,
            "total_wall_s": round(total_s, 3),
            "total_wall_hms": _fmt_hms(total_s),
            "overhead_s": round(max(0.0, total_s - load_s - generate_loop_s), 3),
        },
        "decoding": {
            "backend": backend,
            "greedy": decode["greedy"],
            "do_sample": decode["do_sample"],
            "temperature": decode["temperature"],
            "top_p": decode["top_p"],
            "seed": decode["seed"],
            "num_beams": 1,
            "max_seq_length": int(ns.max_seq_length),
            "max_new_tokens": int(ns.max_new_tokens),
            "bf16": bool(ns.bf16),
            "vllm_model": getattr(ns, "vllm_model", None) if backend == "vllm" else None,
            "concurrency": int(getattr(ns, "concurrency", 4) or 4) if backend == "vllm" else None,
        },
        "memory": merge_memory_phases(
            after_load=mem_after_load,
            after_generate=mem_job_total,
            job_total=mem_job_total,
        ),
    }
    timing_path.write_text(json.dumps(timing, indent=2), encoding="utf-8")

    cmd = (
        f"python -m thesis.cli eval-repliqa-generate --condition {cond_id}"
        + (f" --max-rows {int(ns.max_rows)}" if int(ns.max_rows) > 0 else "")
    )
    manifest = {
        "schema": "repliqa_eval_generate_manifest/v1",
        **{k: timing[k] for k in (
            "model_id", "condition", "description", "load_type", "use_context",
            "n_questions", "started_at", "finished_at", "host", "env",
        )},
        "paths": {
            "output_dir": str(out_dir),
            "predictions_jsonl": str(pred_path),
            "timing_json": str(timing_path),
            "generate_spans_jsonl": str(spans_path),
        },
        "timing": timing["timing"],
        "decoding": timing["decoding"],
        "command": cmd,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (exp_dir / "generate_command.txt").write_text(cmd + "\n", encoding="utf-8")

    idx_path = eval_dir / "predictions_index.json"
    idx: dict[str, Any] = {}
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    idx.setdefault("conditions", {})[model_id] = {
        "predictions_jsonl": str(pred_path),
        "timing_json": str(timing_path),
        "generate_spans_jsonl": str(spans_path),
        "run_manifest_json": str(manifest_path),
        "n_questions": len(rows),
        "finished_at": finished_at,
        "total_wall_s": timing["timing"]["total_wall_s"],
    }
    idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")

    pipeline_log = eval_dir / "eval_pipeline_log.jsonl"
    with open(pipeline_log, "a", encoding="utf-8") as plog:
        plog.write(
            json.dumps(
                {
                    "event": "eval_generate_done",
                    "ts": finished_at,
                    "model_id": model_id,
                    "n_questions": len(rows),
                    "timing": timing["timing"],
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {timing_path}", flush=True)
    print(f"Wrote {spans_path}", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    print(
        f"Done: n={len(rows)} gen_mean={gen_stats.get('mean_s')}s "
        f"gen_p50={gen_stats.get('p50_s')}s total_wall={total_s:.1f}s ({_fmt_hms(total_s)})",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
    eval_dir = run_root / "eval"
    p = argparse.ArgumentParser(description="Generate eval answers for one baseline condition.")
    p.add_argument("--condition", type=str, default=None, help="See --list-conditions")
    p.add_argument("--list-conditions", action="store_true")
    p.add_argument("--run-root", type=Path, default=run_root)
    p.add_argument("--eval-dir", type=Path, default=eval_dir)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--eval-input-name", type=str, default="eval_subset_2000.jsonl")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--row-start", type=int, default=0, help="0-based start index (inclusive).")
    p.add_argument("--row-end", type=int, default=0, help="0-based end index (exclusive); 0 = end.")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--context-fraction", type=float, default=1.0)
    p.add_argument("--condition-id", type=str, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    p.add_argument("--vllm-base-url", type=str, default=None)
    p.add_argument("--vllm-model", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy decode (default for reproducible eval); >0 enables sampling.",
    )
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling when temperature > 0.")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling (per-row offset applied). Omit for non-reproducible sample.",
    )
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.no_bf16:
        ns.bf16 = False
    raise SystemExit(run_eval_repliqa_generate(ns))

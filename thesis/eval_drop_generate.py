"""
Generate answers on DROP validation.jsonl (human gold).

Reuses decode helpers from eval_repliqa_generate.py.
Default: CPT merged base; B3/Ours eval without context (deployable setting).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from thesis.eval_repliqa_generate import (
    EvalCondition,
    _fmt_hms,
    _timing_stats,
    _utc_iso,
    build_user_block,
    decoding_settings,
    generate_one,
    load_jsonl,
    load_model_and_tokenizer,
)
from thesis.gpu_memory_stats import (
    collect_gpu_memory_snapshot,
    merge_memory_phases,
    reset_peak_gpu_memory,
)
from thesis.drop_eval_context import drop_gold_reference
from thesis.eval_row_slice import resolve_pred_output_path, slice_eval_rows
from thesis.paths import DROP_JSONL_DIR, DROP_RUN_CPT

DEFAULT_DROP_RUN = Path(__file__).resolve().parent / "experiments" / "drop" / "runs" / "drop_qa_v1"


def drop_conditions(
    run_root: Path,
    *,
    baselines_subdir: str = "qs_strat",
    high_adapter: str = "QS_strat_high_lora_r32",
    medium_adapter: str = "QS_strat_medium_lora_r16",
    low_adapter: str = "QS_strat_low_lora_r8",
) -> dict[str, EvalCondition]:
    baselines = run_root / "baselines"
    qs = baselines / baselines_subdir
    return {
        "B0_cpt_no_ctx": EvalCondition(
            model_id="B0_cpt_no_ctx",
            description="CPT merged base, question only",
            load_type="base",
            use_context=False,
        ),
        "B1_cpt_ctx": EvalCondition(
            model_id="B1_cpt_ctx",
            description="CPT merged base, gold context + question",
            load_type="base",
            use_context=True,
        ),
        "B3_lora_no_ctx": EvalCondition(
            model_id="B3_lora_no_ctx",
            description="B3 LoRA on CPT base, question only",
            load_type="lora",
            use_context=False,
            adapter_dir=str(baselines / "B3_all_lora_r16"),
        ),
        "B3_lora_ctx": EvalCondition(
            model_id="B3_lora_ctx",
            description="B3 LoRA on CPT base, gold context + question",
            load_type="lora",
            use_context=True,
            adapter_dir=str(baselines / "B3_all_lora_r16"),
        ),
        "B5_adalora_ctx": EvalCondition(
            model_id="B5_adalora_ctx",
            description="B5 AdaLoRA on CPT base (all synthetic QA), gold context + question",
            load_type="lora",
            use_context=True,
            adapter_dir=str(baselines / "B5_adalora_r16"),
        ),
        "Ours_tier_no_ctx": EvalCondition(
            model_id="Ours_tier_no_ctx",
            description="QS tier merge (0.6/0.3/0.1) on CPT base, question only",
            load_type="dense",
            use_context=False,
            dense_dir=str(qs / "QS_merged_strat_dense_w60_30_10"),
        ),
        "Ours_tier_ctx": EvalCondition(
            model_id="Ours_tier_ctx",
            description="QS tier merge (0.6/0.3/0.1) on CPT base, gold context + question",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_w60_30_10"),
        ),
        "Ours_equal_ctx": EvalCondition(
            model_id="Ours_equal_ctx",
            description="QS dense merge equal weights (1/1/1) on CPT base, gold context + question",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense"),
        ),
        "Ours_freq_ctx": EvalCondition(
            model_id="Ours_freq_ctx",
            description="QS dense merge frequency weights on CPT base, gold context + question",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_freq"),
        ),
        "Ours_high_only_ctx": EvalCondition(
            model_id="Ours_high_only_ctx",
            description="QS high-tier LoRA only on CPT base, gold context + question",
            load_type="lora",
            use_context=True,
            adapter_dir=str(qs / high_adapter),
        ),
        "Ours_equal_rank_ctx": EvalCondition(
            model_id="Ours_equal_rank_ctx",
            description="QS equal-rank (r=16/16/16) dense merge 0.6/0.3/0.1, gold context",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_equal_rank_w60_30_10"),
        ),
        "Ours_high_med_ctx": EvalCondition(
            model_id="Ours_high_med_ctx",
            description="QS dense merge high+medium only (0.67/0.33/0), gold context",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_high_med_w67_33_0"),
        ),
        "Ours_low_heavy_ctx": EvalCondition(
            model_id="Ours_low_heavy_ctx",
            description="QS dense merge weights 0.4/0.4/0.2, gold context",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_w40_40_20"),
        ),
        "Ours_inverted_ctx": EvalCondition(
            model_id="Ours_inverted_ctx",
            description="QS dense merge inverted weights 0.1/0.3/0.6, gold context",
            load_type="dense",
            use_context=True,
            dense_dir=str(qs / "QS_merged_strat_dense_w10_30_60"),
        ),
    }


def ohioline_tier_conditions(run_root: Path) -> dict[str, EvalCondition]:
    """OhioLine tier_matrix adapters (different naming from qs_strat)."""
    return drop_conditions(
        run_root,
        baselines_subdir="tier_matrix",
        high_adapter="high_only_r32",
        medium_adapter="medium_only_r16",
        low_adapter="low_only_r8",
    )


def run_eval_drop_generate(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve() if ns.run_root else DEFAULT_DROP_RUN
    if getattr(ns, "tier_matrix", False):
        cond_map = ohioline_tier_conditions(run_root)
    elif getattr(ns, "baselines_subdir", None):
        cond_map = drop_conditions(run_root, baselines_subdir=str(ns.baselines_subdir))
    else:
        cond_map = drop_conditions(run_root)

    if ns.list_conditions:
        for k, c in cond_map.items():
            print(f"  {k:22s}  ctx={c.use_context}  {c.load_type:5s}  {c.description}")
        return 0

    cond_id = str(ns.condition).strip()
    if cond_id not in cond_map:
        raise SystemExit(f"Unknown --condition {cond_id!r}. Use --list-conditions.")

    cond = cond_map[cond_id]
    use_context = cond.use_context
    use_context_override = getattr(ns, "use_context", None)
    if use_context_override is not None:
        use_context = bool(use_context_override)
    if ns.no_context:
        use_context = False

    base_model = str(ns.base_model)
    eval_path = (
        Path(ns.eval_jsonl).expanduser().resolve()
        if ns.eval_jsonl
        else Path(DROP_JSONL_DIR / "validation.jsonl")
    )
    if not eval_path.is_file():
        raise SystemExit(f"Missing eval jsonl: {eval_path}")

    rows = load_jsonl(eval_path)
    row_start = int(getattr(ns, "row_start", 0) or 0)
    row_end = int(getattr(ns, "row_end", 0) or 0)
    rows, row_start, row_end, total_rows = slice_eval_rows(
        rows, row_start=row_start, row_end=row_end, max_rows=int(ns.max_rows)
    )
    if not rows:
        raise SystemExit(f"No rows in slice [{row_start}:{row_end}) of {total_rows}")

    eval_dir = Path(ns.eval_dir).expanduser().resolve() if ns.eval_dir else run_root / "eval"
    model_id = str(ns.condition_id or cond.model_id)
    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir else eval_dir / "predictions" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = resolve_pred_output_path(
        out_dir, row_start=row_start, row_end=row_end, total_rows=total_rows
    )
    timing_path = out_dir / "timing.json"

    bf16 = bool(ns.bf16) and not bool(ns.no_bf16)
    backend = str(getattr(ns, "backend", "hf") or "hf").lower()
    print(
        f"Condition: {model_id} base={base_model} use_context={use_context} backend={backend}",
        flush=True,
    )
    print(f"Eval: {eval_path} ({len(rows)} rows slice [{row_start}:{row_end}) of {total_rows})", flush=True)

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

    wall0 = time.perf_counter()
    started_at = _utc_iso()
    load_s = 0.0
    mem_after_load: dict[str, Any] = {}
    gen_times: list[float] = []
    generate_loop_s = 0.0

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
            context_fraction=1.0,
            max_new_tokens=int(ns.max_new_tokens),
            concurrency=concurrency,
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            seed=decode["seed"],
            row_start=row_start,
        )
        generate_loop_s = time.perf_counter() - t_gen0
        mem_job_total: dict[str, Any] = {}
        with pred_path.open("w", encoding="utf-8") as fp:
            for row, pred in zip(rows, preds):
                answers = row.get("answers")
                rec = {
                    "eval_id": row.get("eval_id"),
                    "section_id": row.get("section_id"),
                    "model_id": model_id,
                    "condition": cond_id,
                    "question": row.get("question"),
                    "answers": answers,
                    "gold": drop_gold_reference(row),
                    "pred": pred,
                    "use_context": use_context,
                    "context": row.get("context") if use_context else None,
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        reset_peak_gpu_memory()
        model, tokenizer, load_s = load_model_and_tokenizer(cond, base_model=base_model, bf16=bf16)
        mem_after_load = collect_gpu_memory_snapshot()
        print(f"Model loaded in {load_s:.1f}s", flush=True)
        if mem_after_load.get("cuda_available"):
            print(
                f"  GPU after load: peak_alloc={mem_after_load['peak_allocated_gib']} GiB",
                flush=True,
            )

        t_gen0 = time.perf_counter()
        with pred_path.open("w", encoding="utf-8") as fp:
            for i, row in enumerate(rows):
                pred, dt = generate_one(
                    model,
                    tokenizer,
                    row,
                    use_context=use_context,
                    max_seq_length=int(ns.max_seq_length),
                    max_new_tokens=int(ns.max_new_tokens),
                    temperature=decode["temperature"],
                    top_p=decode["top_p"],
                    seed=decode["seed"],
                    row_index=row_start + i,
                )
                gen_times.append(dt)
                answers = row.get("answers")
                rec = {
                    "eval_id": row.get("eval_id"),
                    "section_id": row.get("section_id"),
                    "model_id": model_id,
                    "condition": cond_id,
                    "question": row.get("question"),
                    "answers": answers,
                    "gold": drop_gold_reference(row),
                    "pred": pred,
                    "use_context": use_context,
                    "context": row.get("context") if use_context else None,
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print(f"  ... {i + 1}/{len(rows)}", flush=True)
        generate_loop_s = time.perf_counter() - t_gen0
        mem_job_total = collect_gpu_memory_snapshot()

    total_s = time.perf_counter() - wall0
    finished_at = _utc_iso()
    gen_stats = _timing_stats(gen_times)
    timing = {
        "schema": "drop_eval_generate_timing/v2",
        "condition": cond_id,
        "model_id": model_id,
        "load_type": cond.load_type,
        "use_context": use_context,
        "n_rows": len(rows),
        "row_start": row_start,
        "row_end": row_end,
        "total_rows": total_rows,
        "backend": backend,
        "started_at": started_at,
        "finished_at": finished_at,
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
            "greedy": True,
            "max_seq_length": int(ns.max_seq_length),
            "max_new_tokens": int(ns.max_new_tokens),
            "bf16": bf16,
            "concurrency": int(getattr(ns, "concurrency", 4) or 4) if backend == "vllm" else None,
        },
        "memory": merge_memory_phases(
            after_load=mem_after_load,
            after_generate=mem_job_total,
            job_total=mem_job_total,
        ),
        # Legacy flat fields for older collectors.
        "load_s": round(load_s, 3),
        "total_wall_s": round(total_s, 3),
        "generate": gen_stats,
    }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pred_path}", flush=True)
    print(f"Total wall {_fmt_hms(total_s)}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate DROP validation predictions")
    p.add_argument("--condition", type=str, default=None)
    p.add_argument("--list-conditions", action="store_true")
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--condition-id", type=str, default=None)
    p.add_argument(
        "--base-model",
        type=str,
        default=str(DROP_RUN_CPT / "merged_base"),
    )
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--row-start", type=int, default=0, help="0-based start index (inclusive).")
    p.add_argument("--row-end", type=int, default=0, help="0-based end index (exclusive); 0 = end.")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--use-context", type=bool, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    p.add_argument("--vllm-base-url", type=str, default=None)
    p.add_argument("--vllm-model", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    if ns.no_bf16:
        ns.bf16 = False
    raise SystemExit(run_eval_drop_generate(ns))

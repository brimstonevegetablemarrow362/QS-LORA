"""
Generate eval answers with closed API models on Amazon Bedrock.

Dual-vendor ceilings (same AWS credentials as Haiku judge):
  REF_claude_opus   — anthropic.claude-opus-4-8 (Anthropic ceiling)
  REF_nova_2_lite   — amazon.nova-2-lite-v1:0 (Amazon ceiling)

Usage:
  source thesis/scripts/source_bedrock_env.sh
  python -m thesis.cli eval-api-generate \\
    --run-root $RUN --eval-jsonl $EVAL --condition-id REF_nova_2_lite --max-rows 20
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.drop_eval_context import drop_gold_reference
from thesis.eval_repliqa_generate import (
    SYSTEM_PROMPT_CTX,
    SYSTEM_PROMPT_NO_CTX,
    _fmt_hms,
    _timing_stats,
    _utc_iso,
    build_user_block,
    load_jsonl,
)

DEFAULT_ANTHROPIC_CEILING_MODEL = os.environ.get(
    "BEDROCK_ANTHROPIC_CEILING_MODEL_ID",
    os.environ.get("BEDROCK_CEILING_MODEL_ID", "us.anthropic.claude-opus-4-8"),
)
DEFAULT_NOVA_CEILING_MODEL = os.environ.get(
    "BEDROCK_NOVA_CEILING_MODEL_ID",
    "us.amazon.nova-2-lite-v1:0",
)

CEILING_PRESETS: dict[str, dict[str, str]] = {
    "REF_claude_opus": {
        "provider": "bedrock",
        "model": DEFAULT_ANTHROPIC_CEILING_MODEL,
    },
    "REF_nova_2_lite": {
        "provider": "bedrock",
        "model": DEFAULT_NOVA_CEILING_MODEL,
    },
    # Legacy aliases
    "REF_claude_sonnet": {
        "provider": "bedrock",
        "model": DEFAULT_ANTHROPIC_CEILING_MODEL,
    },
}

DEFAULT_CEILING_CONDITIONS = ("REF_claude_opus", "REF_nova_2_lite")


def _is_drop_row(row: dict[str, Any]) -> bool:
    return isinstance(row.get("answers"), list) or "section_id" in row


def _prediction_record(
    row: dict[str, Any],
    *,
    condition_id: str,
    pred: str,
    use_context: bool,
    api_provider: str,
    api_model: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "api_provider": api_provider,
        "api_model": api_model,
    }
    if _is_drop_row(row):
        return {
            **base,
            "eval_id": row.get("eval_id"),
            "section_id": row.get("section_id"),
            "model_id": condition_id,
            "condition": condition_id,
            "question": row.get("question"),
            "answers": row.get("answers"),
            "gold": drop_gold_reference(row),
            "pred": pred,
            "use_context": use_context,
            "context": row.get("context") if use_context else None,
        }
    gold = (row.get("gold") or row.get("answer") or "").strip()
    eval_id = row.get("eval_id") or row.get("chunk_id")
    return {
        **base,
        "eval_id": eval_id,
        "document_id": row.get("document_id"),
        "chunk_id": row.get("chunk_id"),
        "repliqa_split": row.get("repliqa_split"),
        "document_topic": row.get("document_topic"),
        "model_id": condition_id,
        "condition": condition_id,
        "question": row.get("question"),
        "gold": gold,
        "pred": pred,
        "use_context": use_context,
        "context_fraction": 1.0 if use_context else None,
        "context": row.get("context") if use_context else None,
    }


def _resolve_provider_model(
    *,
    condition_id: str,
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    preset = CEILING_PRESETS.get(condition_id, {})
    prov = (provider or preset.get("provider") or "bedrock").strip().lower()
    if prov != "bedrock":
        raise ValueError(f"unsupported provider {prov!r}; ceilings use bedrock only")
    mdl = model or preset.get("model") or DEFAULT_ANTHROPIC_CEILING_MODEL
    return prov, str(mdl)


def _generate_one_bedrock(
    client: Any,
    *,
    model_id: str,
    row: dict[str, Any],
    use_context: bool,
    max_tokens: int,
    temperature: float,
    nova_reasoning_effort: str,
    region: str,
) -> tuple[str, float]:
    from thesis.bedrock_judge_qa_score import invoke_bedrock_generate

    system = SYSTEM_PROMPT_CTX if use_context else SYSTEM_PROMPT_NO_CTX
    user = build_user_block(row, use_context=use_context, context_fraction=1.0)
    t0 = time.perf_counter()
    pred = invoke_bedrock_generate(
        client,
        model_id=model_id,
        system=system,
        user_message=user,
        max_tokens=max_tokens,
        temperature=temperature,
        nova_reasoning_effort=nova_reasoning_effort,
        region=region,
    )
    return pred.strip(), time.perf_counter() - t0


def run_eval_api_generate(ns: argparse.Namespace) -> int:
    run_root = Path(ns.run_root).expanduser().resolve() if ns.run_root else None
    eval_path = Path(ns.eval_jsonl).expanduser().resolve()
    if not eval_path.is_file():
        print(f"Missing eval jsonl: {eval_path}", file=sys.stderr)
        return 1

    rows = load_jsonl(eval_path)
    if int(ns.max_rows) > 0:
        rows = rows[: int(ns.max_rows)]
    if not rows:
        print("No eval rows.", file=sys.stderr)
        return 1

    use_context = not bool(ns.no_context)
    condition_id = str(ns.condition_id or "REF_claude_opus")
    provider, model_id = _resolve_provider_model(
        condition_id=condition_id,
        provider=getattr(ns, "provider", None),
        model=ns.model,
    )

    region = (ns.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        print("Set AWS_REGION or pass --region for Bedrock", file=sys.stderr)
        return 1

    eval_dir = (
        Path(ns.eval_dir).expanduser().resolve()
        if ns.eval_dir
        else (run_root / "eval" if run_root else eval_path.parent)
    )
    out_dir = (
        Path(ns.output_dir).expanduser().resolve()
        if ns.output_dir
        else eval_dir / "predictions" / condition_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    timing_path = out_dir / "timing.json"

    if ns.dry_run:
        print(
            f"Dry run OK: provider={provider} model={model_id} condition={condition_id} "
            f"rows={len(rows)} use_context={use_context} region={region}",
            flush=True,
        )
        return 0

    from thesis.bedrock_judge_qa_score import _bedrock_client

    client = _bedrock_client(region)
    conc = max(1, int(ns.concurrency))
    delay = float(ns.request_delay_s)
    max_tok = int(ns.max_tokens)
    temp = float(ns.temperature)
    nova_effort = str(getattr(ns, "nova_reasoning_effort", None) or "low")

    wall0 = time.perf_counter()
    started_at = _utc_iso()
    print(
        f"API generate: provider={provider} model={model_id} condition={condition_id} "
        f"rows={len(rows)} concurrency={conc} use_context={use_context} "
        f"nova_reasoning={nova_effort if 'nova' in model_id.lower() else 'n/a'}",
        flush=True,
    )

    results: list[dict[str, Any] | None] = [None] * len(rows)
    gen_times: list[float] = []
    n_err = 0
    lock = __import__("threading").Lock()

    def job(i: int, row: dict[str, Any]) -> None:
        nonlocal n_err
        try:
            if delay > 0:
                time.sleep(delay * (i % conc) * 0.05)
            pred, dt = _generate_one_bedrock(
                client,
                model_id=model_id,
                row=row,
                use_context=use_context,
                max_tokens=max_tok,
                temperature=temp,
                nova_reasoning_effort=nova_effort,
                region=region,
            )
            rec = _prediction_record(
                row,
                condition_id=condition_id,
                pred=pred,
                use_context=use_context,
                api_provider=provider,
                api_model=model_id,
            )
            with lock:
                results[i] = rec
                gen_times.append(dt)
        except Exception as e:
            with lock:
                n_err += 1
                results[i] = {
                    **_prediction_record(
                        row,
                        condition_id=condition_id,
                        pred="",
                        use_context=use_context,
                        api_provider=provider,
                        api_model=model_id,
                    ),
                    "api_error": str(e),
                }

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(job, i, r) for i, r in enumerate(rows)]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 20 == 0 or done == len(rows):
                print(f"  ... {done}/{len(rows)}", flush=True)

    rows_out = [r for r in results if r is not None]
    generate_loop_s = time.perf_counter() - wall0
    gen_stats = _timing_stats(gen_times)

    timing = {
        "schema": "api_eval_generate_timing/v1",
        "provider": provider,
        "condition": condition_id,
        "model_id": condition_id,
        "api_model": model_id,
        "load_type": "api",
        "use_context": use_context,
        "n_questions": len(rows),
        "n_api_errors": n_err,
        "started_at": started_at,
        "finished_at": _utc_iso(),
        "host": socket.gethostname(),
        "env": {
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "AWS_REGION": region,
        },
        "timing": {
            "load_model_s": 0.0,
            "generate_loop_s": round(generate_loop_s, 3),
            "generate_loop_hms": _fmt_hms(generate_loop_s),
            "generate_per_question": gen_stats,
            "total_wall_s": round(generate_loop_s, 3),
            "total_wall_hms": _fmt_hms(generate_loop_s),
        },
        "decoding": {
            "backend": "bedrock_api",
            "temperature": temp,
            "max_tokens": max_tok,
            "concurrency": conc,
            "nova_reasoning_effort": nova_effort if "nova" in model_id.lower() else None,
        },
    }
    timing_path.write_text(json.dumps(timing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with pred_path.open("w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {timing_path}", flush=True)
    print(
        f"Done: mean_s/q={gen_stats.get('mean_s')} api_errors={n_err} wall={_fmt_hms(generate_loop_s)}",
        flush=True,
    )
    return 0 if n_err < len(rows) else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate eval answers via Bedrock API ceilings.")
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--condition-id", type=str, default="REF_claude_opus")
    p.add_argument(
        "--provider",
        type=str,
        choices=("bedrock",),
        default="bedrock",
        help="Bedrock only (same credentials as judge).",
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-delay-s", type=float, default=0.05)
    p.add_argument(
        "--nova-reasoning-effort",
        type=str,
        default="low",
        choices=("low", "medium", "high"),
        help="Nova 2 extended thinking effort (default low for concise QA).",
    )
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run_eval_api_generate(build_arg_parser().parse_args()))

"""
Score QA or eval-prediction JSONL with Claude Haiku via **Amazon Bedrock** (boto3).

Designed to run from OSC (or any host) with outbound HTTPS to Bedrock; billing uses
AWS credits. Reuses ``qa_judge_common`` rubric v2 (same as Haiku training judge).

**Eval predictions** (``--answer-field pred``):
  Each row needs ``context``, ``question``, ``gold``, and ``pred``.
  The judge compares **pred vs gold reference** and grounding in context (rubric v3).

**AWS:** see ``thesis/experiments/repliqa/runs/repliqa_train_0-3/eval/BEDROCK_JUDGE_SETUP.md``

Usage (from ``finetuning/``, after ``pip install boto3``):
  source thesis/scripts/source_bedrock_env.sh   # loads eval/bedrock_credentials.env

  # Smoke test (3 rows)
  python -m thesis.cli qa-bedrock-judge \\
    --predictions-jsonl thesis/experiments/repliqa/runs/repliqa_train_0-3/eval/predictions/B3_lora_all/predictions.jsonl \\
    --answer-field pred --max-rows 3

  # Full run on eval preds
  python -m thesis.cli qa-bedrock-judge \\
    --predictions-jsonl .../eval/predictions/Ours_tier_merge/predictions.jsonl \\
    --answer-field pred --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.qa_judge_common import (
    JUDGE_EVAL_SYSTEM,
    JUDGE_PROMPT_VERSION,
    JUDGE_PROMPT_VERSION_EVAL,
    JUDGE_PROMPT_VERSION_EVAL_DEBIAS,
    JUDGE_SYSTEM,
    build_judge_user_message,
    is_nan_answer,
    judge_prompt_version,
    merge_position_swap_judge_blocks,
    normalize_judge_block,
    parse_judge_json,
    skipped_nan_judge_block,
)

# Common Bedrock model IDs (enable in console → Model access). Override with --model or env.
DEFAULT_BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_JUDGE_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)
PROVIDER = "bedrock"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no} JSON error: {e}") from e
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def resolve_answer_text(row: dict[str, Any], answer_field: str) -> str:
    """Text sent to the judge as the model prediction (eval) or answer to score (training)."""
    if answer_field == "pred":
        return str(row.get("pred") or row.get("prediction") or "").strip()
    if answer_field == "answer":
        return str(row.get("answer") or "").strip()
    if answer_field == "auto":
        return str(row.get("pred") or row.get("prediction") or row.get("answer") or "").strip()
    raise ValueError(f"unknown answer_field: {answer_field!r}")


def resolve_gold_text(row: dict[str, Any]) -> str:
    """Human reference answer (eval gold). Supports DROP multi-answer rows."""
    if "answers" in row or row.get("unanswerable"):
        from thesis.drop_eval_context import drop_gold_reference

        g = drop_gold_reference(row)
        if g:
            return g
    return str(row.get("gold") or row.get("answer") or "").strip()


def _row_key(row: dict[str, Any]) -> str:
    return str(row.get("eval_id") or row.get("chunk_id") or "").strip()


def is_judge_ok(row: dict[str, Any]) -> bool:
    """True when ``llm_judge`` has a usable score (not error / missing gold)."""
    block = row.get("llm_judge") or {}
    if block.get("skipped"):
        return True
    if block.get("error"):
        return False
    return block.get("overall") is not None


def load_existing_judged(out_jsonl: Path) -> dict[str, dict[str, Any]]:
    if not out_jsonl.is_file():
        return {}
    index: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(out_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{out_jsonl}:{line_no} JSON error: {e}") from e
        key = _row_key(row)
        if key:
            index[key] = row
    return index


def _reuse_judged_row(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    out = dict(current)
    for key in ("answer_judged", "gold_judged", "llm_judge"):
        if key in previous:
            out[key] = previous[key]
    return out


def _bedrock_client(region: str):
    try:
        import boto3
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: boto3\n  pip install 'boto3>=1.35.0'"
        ) from e
    return boto3.client("bedrock-runtime", region_name=region)


def _check_aws_env(region: str) -> None:
    if not region:
        print(
            "Set AWS region: export AWS_REGION=us-east-1 (or pass --region).\n"
            "  source thesis/scripts/source_bedrock_env.sh",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # boto3 uses default credential chain: env vars, ~/.aws/credentials, IAM role, etc.


def _is_nova_bedrock_model(model_id: str) -> bool:
    return "nova" in model_id.lower()


def _claude_omit_temperature(model_id: str) -> bool:
    """Some newer Claude models reject ``temperature`` on Bedrock Converse/Invoke."""
    mid = model_id.lower()
    return "opus-4-8" in mid


def _normalize_anthropic_bedrock_model_id(model_id: str, region: str) -> str:
    """Anthropic on-demand often requires a cross-region inference profile (e.g. us.anthropic.*)."""
    mid = model_id.strip()
    if not mid.startswith("anthropic."):
        return mid
    if mid.startswith(("us.", "eu.", "apac.", "global.")):
        return mid
    geo = "us"
    if region.startswith("eu-"):
        geo = "eu"
    elif region.startswith(("ap-", "apac")):
        geo = "apac"
    return f"{geo}.{mid}"


def _normalize_nova_bedrock_model_id(model_id: str, region: str) -> str:
    """Nova 2 on-demand requires a cross-region inference profile (e.g. us.amazon.*)."""
    mid = model_id.strip()
    if not _is_nova_bedrock_model(mid):
        return mid
    if mid.startswith(("us.", "eu.", "apac.", "global.")):
        return mid
    # amazon.nova-2-lite-v1:0 -> us.amazon.nova-2-lite-v1:0 in us-east-1
    geo = "us"
    if region.startswith("eu-"):
        geo = "eu"
    elif region.startswith(("ap-", "apac")):
        geo = "apac"
    if mid.startswith("amazon."):
        return f"{geo}.{mid}"
    return mid


def _invoke_bedrock_nova(
    client: Any,
    *,
    model_id: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    system: str | None = None,
    reasoning_effort: str = "low",
) -> str:
    """Bedrock Converse for Amazon Nova / Nova 2 (text generation)."""
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    if "nova-2" in model_id.lower():
        kwargs["additionalModelRequestFields"] = {
            "reasoningConfig": {"type": "enabled", "maxReasoningEffort": reasoning_effort},
        }
    resp = client.converse(**kwargs)
    parts = resp.get("output", {}).get("message", {}).get("content") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    return "".join(texts).strip()


def invoke_bedrock_generate(
    client: Any,
    *,
    model_id: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    system: str | None = None,
    nova_reasoning_effort: str = "low",
    region: str = "",
) -> str:
    """Generate text on Bedrock (Anthropic Claude or Amazon Nova)."""
    if _is_nova_bedrock_model(model_id):
        model_id = _normalize_nova_bedrock_model_id(model_id, region)
        return _invoke_bedrock_nova(
            client,
            model_id=model_id,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            reasoning_effort=nova_reasoning_effort,
        )
    model_id = _normalize_anthropic_bedrock_model_id(model_id, region)
    return _invoke_bedrock_claude(
        client,
        model_id=model_id,
        user_message=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
    )


def _invoke_bedrock_claude(
    client: Any,
    *,
    model_id: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    system: str | None = None,
) -> str:
    """Call Bedrock; prefer Converse API, fall back to InvokeModel Anthropic payload."""
    system_text = system if system is not None else JUDGE_SYSTEM
    inference_config: dict[str, Any] = {"maxTokens": max_tokens}
    if not _claude_omit_temperature(model_id):
        inference_config["temperature"] = temperature
    try:
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system_text}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_message}],
                }
            ],
            inferenceConfig=inference_config,
        )
        parts = resp.get("output", {}).get("message", {}).get("content") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
    except Exception as converse_err:
        payload: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_text,
            "messages": [{"role": "user", "content": user_message}],
        }
        if not _claude_omit_temperature(model_id):
            payload["temperature"] = temperature
        body = json.dumps(payload)
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            content = payload.get("content") or []
            if content and isinstance(content[0], dict):
                return str(content[0].get("text", "")).strip()
            return str(payload.get("completion", "")).strip()
        except Exception as invoke_err:
            raise RuntimeError(
                f"Bedrock converse failed: {converse_err}; invoke_model failed: {invoke_err}"
            ) from invoke_err


def _judge_one(
    client: Any,
    *,
    model_id: str,
    context: str,
    question: str,
    answer: str,
    max_context_chars: int,
    max_tokens: int,
    temperature: float,
    gold: str | None = None,
    gold_first: bool = True,
) -> dict[str, Any]:
    user = build_judge_user_message(
        context=context,
        question=question,
        answer=answer,
        max_context_chars=max_context_chars,
        gold=gold,
        gold_first=gold_first,
    )
    system = JUDGE_EVAL_SYSTEM if gold is not None else JUDGE_SYSTEM
    raw = _invoke_bedrock_claude(
        client,
        model_id=model_id,
        user_message=user,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
    )
    parsed = parse_judge_json(raw)
    return normalize_judge_block(
        provider=PROVIDER,
        model=model_id,
        parsed=parsed,
        raw=raw,
        answer=answer,
        gold=gold,
    )


def _aggregate_stats(rows_out: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {
        "n_parse_error": 0,
        "n_api_error": 0,
        "n_skipped_nan": 0,
        "n_missing_fields": 0,
        "n_judged_ok": 0,
        "tier_counts": {"high": 0, "medium": 0, "low": 0, "drop": 0, "error": 0},
        "sum_g": 0.0,
        "sum_r": 0.0,
        "sum_ga": 0.0,
        "sum_dn": 0.0,
        "sum_o": 0.0,
        "n_with_gold_alignment": 0,
    }
    for row in rows_out:
        block = row.get("llm_judge") or {}
        if block.get("skipped"):
            stats["n_skipped_nan"] += 1
            stats["tier_counts"]["drop"] += 1
            continue
        if block.get("error"):
            err = str(block.get("error", ""))
            if err == "parse_error":
                stats["n_parse_error"] += 1
            elif err == "missing_context_question_or_answer":
                stats["n_missing_fields"] += 1
            else:
                stats["n_api_error"] += 1
            stats["tier_counts"]["error"] += 1
            continue
        stats["n_judged_ok"] += 1
        stats["sum_g"] += float(block.get("grounding", 0))
        stats["sum_r"] += float(block.get("relevance", 0))
        stats["sum_dn"] += float(block.get("document_necessity", 0))
        stats["sum_o"] += float(block.get("overall", 0))
        if block.get("gold_alignment") is not None:
            stats["sum_ga"] += float(block["gold_alignment"])
            stats["n_with_gold_alignment"] += 1
        tier = block.get("quality_tier", "error")
        if tier in stats["tier_counts"]:
            stats["tier_counts"][tier] += 1
    n_ok = stats["n_judged_ok"]
    stats["mean_grounding"] = round(stats["sum_g"] / max(1, n_ok), 4)
    stats["mean_relevance"] = round(stats["sum_r"] / max(1, n_ok), 4)
    stats["mean_document_necessity"] = round(stats["sum_dn"] / max(1, n_ok), 4)
    stats["mean_overall"] = round(stats["sum_o"] / max(1, n_ok), 4)
    n_ga = stats["n_with_gold_alignment"]
    stats["mean_gold_alignment"] = round(stats["sum_ga"] / max(1, n_ga), 4) if n_ga else None
    return stats


def run_qa_bedrock_judge(ns: argparse.Namespace) -> int:
    in_path = Path(ns.predictions_jsonl or ns.qa_jsonl).expanduser().resolve()
    if not in_path.is_file():
        print(f"Not found: {in_path}", file=sys.stderr)
        return 1

    region = (ns.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip()

    out_jsonl = (
        Path(ns.out_jsonl).expanduser().resolve()
        if ns.out_jsonl
        else in_path.parent / f"{in_path.stem}_bedrock_judge.jsonl"
    )
    sum_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    )
    timing_path = (
        Path(ns.timing_json).expanduser().resolve()
        if ns.timing_json
        else out_jsonl.with_name(out_jsonl.stem + "_timing.json")
    )

    rows_in = _load_rows(in_path, int(ns.max_rows))
    if not rows_in:
        print("No rows to judge.", file=sys.stderr)
        return 1

    answer_field = str(ns.answer_field)

    eval_jsonl = getattr(ns, "eval_jsonl", None)
    if eval_jsonl is not None:
        eval_path = Path(eval_jsonl).expanduser().resolve()
        if not eval_path.is_file():
            print(f"Eval JSONL not found: {eval_path}", file=sys.stderr)
            return 1
        # DROP validation uses answers[]; RepLiQA uses gold on eval subset.
        sample = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])
        if isinstance(sample.get("answers"), list):
            from thesis.drop_eval_context import enrich_rows_with_drop_eval, load_eval_index

            eval_index = load_eval_index(eval_path)
            rows_in, n_ctx = enrich_rows_with_drop_eval(rows_in, eval_index)
            print(f"Merged DROP eval fields: {n_ctx}/{len(rows_in)} rows", flush=True)
        else:
            from thesis.repliqa_eval_context import enrich_rows_with_eval_context, load_eval_index

            eval_index = load_eval_index(eval_path)
            rows_in, n_ctx = enrich_rows_with_eval_context(rows_in, eval_index)
            print(f"Merged context from eval subset: {n_ctx}/{len(rows_in)} rows", flush=True)
    elif answer_field in ("pred", "auto") and not any(
        str(r.get("context") or "").strip() for r in rows_in[:5]
    ):
        print(
            "Warning: predictions lack 'context'. Pass --eval-jsonl eval_subset_2000.jsonl "
            "so the judge can score grounding.",
            file=sys.stderr,
        )

    model_id = str(ns.model or DEFAULT_BEDROCK_MODEL_ID)
    max_ctx = int(ns.max_context_chars)
    max_tok = int(ns.max_tokens)
    temp = float(ns.temperature)
    conc = max(1, int(ns.concurrency))
    delay = float(ns.request_delay_s)

    if ns.dry_run:
        resume = bool(getattr(ns, "resume", False)) and not bool(getattr(ns, "force", False))
        n_reuse = 0
        if resume:
            existing = load_existing_judged(out_jsonl)
            for row in rows_in:
                prev = existing.get(_row_key(row)) if _row_key(row) else None
                if prev and is_judge_ok(prev):
                    n_reuse += 1
        n_new = len(rows_in) - n_reuse
        print(
            f"Dry run OK: region={region} model={model_id} rows={len(rows_in)} "
            f"resume={resume} reuse={n_reuse} judge_new={n_new}",
            flush=True,
        )
        print("Credentials: boto3 default chain (env / ~/.aws/credentials)", flush=True)
        return 0

    _check_aws_env(region)

    resume = bool(getattr(ns, "resume", False)) and not bool(getattr(ns, "force", False))
    existing_by_key = load_existing_judged(out_jsonl) if resume else {}

    client = _bedrock_client(region)
    wall0 = time.perf_counter()
    started_at = _utc_iso()

    use_gold_rubric = answer_field == "pred" or (
        answer_field == "auto"
        and any(str(r.get("gold") or "").strip() for r in rows_in[:5])
    )
    position_swap_debias = bool(getattr(ns, "position_swap_debias", False)) and use_gold_rubric
    if bool(getattr(ns, "position_swap_debias", False)) and not use_gold_rubric:
        print(
            "Warning: --position-swap-debias requires gold-reference eval rubric; ignoring.",
            file=sys.stderr,
        )
    prompt_version = (
        JUDGE_PROMPT_VERSION_EVAL_DEBIAS
        if position_swap_debias
        else (JUDGE_PROMPT_VERSION_EVAL if use_gold_rubric else JUDGE_PROMPT_VERSION)
    )
    print(
        f"Bedrock judge: region={region} model={model_id} rows={len(rows_in)} "
        f"concurrency={conc} answer_field={answer_field} "
        f"rubric={prompt_version}"
        + (f" resume={resume}" if resume else "")
        + (f" position_swap_debias={position_swap_debias}" if position_swap_debias else ""),
        flush=True,
    )

    results: list[dict[str, Any] | None] = [None] * len(rows_in)
    to_judge: list[tuple[int, dict[str, Any]]] = []
    n_reused = 0
    for i, row in enumerate(rows_in):
        key = _row_key(row)
        prev = existing_by_key.get(key) if key and resume else None
        if prev and is_judge_ok(prev):
            results[i] = _reuse_judged_row(row, prev)
            n_reused += 1
        else:
            to_judge.append((i, row))

    if resume:
        print(
            f"Resume: reuse {n_reused}/{len(rows_in)}, Bedrock calls for {len(to_judge)} rows",
            flush=True,
        )

    lock = threading.Lock()
    latencies: list[float] = []
    n_api_err = 0
    n_bedrock_calls = 0

    def job(idx: int, row: dict[str, Any]) -> None:
        nonlocal n_api_err, n_bedrock_calls
        ctx = str(row.get("context") or "").strip()
        q = str(row.get("question") or "").strip()
        a = resolve_answer_text(row, answer_field)
        gold = resolve_gold_text(row) if use_gold_rubric else None
        out_row = {**row, "answer_judged": a}
        if gold:
            out_row["gold_judged"] = gold
        t0 = time.perf_counter()

        if is_nan_answer(a):
            block = skipped_nan_judge_block(provider=PROVIDER, model=model_id)
        elif not ctx or not q or not a:
            block = normalize_judge_block(
                provider=PROVIDER,
                model=model_id,
                parsed=None,
                error="missing_context_question_or_answer",
                answer=a,
                gold=gold,
            )
        elif use_gold_rubric and not gold:
            block = normalize_judge_block(
                provider=PROVIDER,
                model=model_id,
                parsed=None,
                error="missing_gold_reference",
                answer=a,
                gold=None,
            )
        else:
            try:
                if delay > 0:
                    time.sleep(delay * (idx % conc) * 0.05)
                if position_swap_debias:
                    block_gf = _judge_one(
                        client,
                        model_id=model_id,
                        context=ctx,
                        question=q,
                        answer=a,
                        max_context_chars=max_ctx,
                        max_tokens=max_tok,
                        temperature=temp,
                        gold=gold,
                        gold_first=True,
                    )
                    with lock:
                        n_bedrock_calls += 1
                    block_pf = _judge_one(
                        client,
                        model_id=model_id,
                        context=ctx,
                        question=q,
                        answer=a,
                        max_context_chars=max_ctx,
                        max_tokens=max_tok,
                        temperature=temp,
                        gold=gold,
                        gold_first=False,
                    )
                    with lock:
                        n_bedrock_calls += 1
                    block = merge_position_swap_judge_blocks(
                        block_gf, block_pf, answer=a
                    )
                else:
                    block = _judge_one(
                        client,
                        model_id=model_id,
                        context=ctx,
                        question=q,
                        answer=a,
                        max_context_chars=max_ctx,
                        max_tokens=max_tok,
                        temperature=temp,
                        gold=gold,
                    )
                    with lock:
                        n_bedrock_calls += 1
            except Exception as e:
                block = normalize_judge_block(
                    provider=PROVIDER,
                    model=model_id,
                    parsed=None,
                    error=f"api_error:{e}",
                    answer=a,
                    gold=gold,
                )
                with lock:
                    n_api_err += 1

        elapsed = time.perf_counter() - t0
        with lock:
            results[idx] = {**out_row, "llm_judge": block}
            latencies.append(elapsed)
            done = sum(1 for r in results if r is not None)
            if done % 20 == 0 or done == len(rows_in):
                print(f"  ... {done}/{len(rows_in)} rows ready ({n_reused} reused)", flush=True)

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(job, i, r) for i, r in to_judge]
        for f in as_completed(futs):
            f.result()

    wall_s = time.perf_counter() - wall0
    rows_out = [r for r in results if r is not None]
    stats = _aggregate_stats(rows_out)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    model_id_row = rows_in[0].get("model_id") if rows_in else None
    condition_name = (
        str(model_id_row or "").strip()
        or in_path.parent.name
        or out_jsonl.parent.name
    )
    if condition_name in ("predictions", "judged", "bedrock_judge"):
        condition_name = str(model_id_row or "unknown")
    summary = {
        "schema": "qa_bedrock_judge_summary/v1",
        "condition": condition_name,
        "input_jsonl": str(in_path),
        "out_jsonl": str(out_jsonl),
        "provider": PROVIDER,
        "model": model_id,
        "region": region,
        "prompt_version": prompt_version,
        "answer_field": answer_field,
        "includes_gold_reference": use_gold_rubric,
        "position_swap_debias": position_swap_debias,
        "model_id_from_rows": model_id_row,
        "settings": {
            "max_context_chars": max_ctx,
            "max_tokens": max_tok,
            "temperature": temp,
            "concurrency": conc,
            "request_delay_s": delay,
        },
        "n_rows": len(rows_in),
        "stats": stats,
        "resume": {
            "enabled": resume,
            "n_reused": n_reused,
            "n_judged_new": len(to_judge),
        },
        "timing": {
            "started_at": started_at,
            "finished_at": _utc_iso(),
            "total_wall_s": round(wall_s, 3),
            "mean_request_s": round(sum(latencies) / max(1, len(latencies)), 3),
            "n_api_errors_during_run": n_api_err,
            "n_bedrock_calls": n_bedrock_calls if position_swap_debias else len(to_judge),
        },
        "notes": [
            "Runs from OSC via HTTPS to Bedrock; charges AWS account (credits).",
            "For eval, use --answer-field pred on predictions.jsonl (gold sent as reference).",
            "Rubric v3 adds gold_alignment vs human gold; training curation stays v2.",
            "Pin model_id and date in thesis methods.",
        ],
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    timing_path.write_text(
        json.dumps(summary["timing"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(f"Wrote {timing_path}", flush=True)
    print(
        f"Judged ok: {stats['n_judged_ok']}/{len(rows_in)}  "
        f"mean_overall={stats['mean_overall']}  wall_s={wall_s:.1f}"
        + (f"  (reused {n_reused}, new Bedrock calls {len(to_judge)})" if resume else ""),
        flush=True,
    )
    return 0 if stats["n_judged_ok"] > 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score QA/predictions JSONL with Claude via Amazon Bedrock (Haiku)."
    )
    p.add_argument(
        "--predictions-jsonl",
        "--qa-jsonl",
        dest="predictions_jsonl",
        type=Path,
        required=True,
        help="Input JSONL (eval predictions or QA rows).",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--timing-json", type=Path, default=None)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Bedrock model ID (default: {DEFAULT_BEDROCK_MODEL_ID} or env BEDROCK_JUDGE_MODEL_ID).",
    )
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region (default: AWS_REGION or AWS_DEFAULT_REGION).",
    )
    p.add_argument(
        "--answer-field",
        type=str,
        default="auto",
        choices=("auto", "pred", "answer"),
        help="Row field to judge: pred for eval predictions, answer for synthetic QA.",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows.")
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-delay-s", type=float, default=0.0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/env only; no Bedrock calls.",
    )
    p.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="RepLiQA eval subset; merge context (and gold if missing) by eval_id.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful rows from --out-jsonl; Bedrock only for failed/missing rows.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-judge all rows (ignores --resume reuse).",
    )
    p.add_argument(
        "--position-swap-debias",
        action="store_true",
        help="Eval rubric only: judge with gold-first and pred-first prompts; average scores.",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run_qa_bedrock_judge(build_arg_parser().parse_args()))

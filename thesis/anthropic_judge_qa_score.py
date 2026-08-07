"""
Anthropic Haiku LLM-as-judge for QA JSONL (full corpus via Message Batches API).

Requires: pip install anthropic
Env: ANTHROPIC_API_KEY

Default model: claude-haiku-4-5 (50% batch pricing).

Run (submit + poll + write scored JSONL):
  export ANTHROPIC_API_KEY=...
  python -m thesis.cli qa-haiku-judge \\
    --qa-jsonl thesis/experiments/repliqa/runs/repliqa_train_0-3/train/synthetic_qa.jsonl

Resume polling only (batch ids saved in <out_stem>_batch_state.json):
  python -m thesis.cli qa-haiku-judge --qa-jsonl ... --resume-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from thesis.qa_judge_common import (
    JUDGE_SYSTEM,
    build_judge_user_message,
    is_nan_answer,
    normalize_judge_block,
    parse_judge_json,
    skipped_nan_judge_block,
)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
DEFAULT_BATCH_CHUNK_SIZE = 3000
DEFAULT_POLL_INTERVAL_S = 30.0
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _sanitize_custom_id(row: dict[str, Any], idx: int) -> str:
    raw = str(row.get("chunk_id") or row.get("document_id") or f"row_{idx}")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw).strip("_")
    if not safe:
        safe = f"row_{idx}"
    safe = safe[:64]
    if not CUSTOM_ID_RE.match(safe):
        safe = f"row_{idx}"
    return safe


def _extract_message_text(message: Any) -> str:
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts).strip()


def _load_rows(qa_path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(qa_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Line {line_no}: JSON error: {e}", file=sys.stderr)
    if max_rows > 0:
        rows = rows[:max_rows]
    return rows


def _build_batch_requests(
    rows: list[dict[str, Any]],
    *,
    model: str,
    max_context_chars: int,
    max_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (anthropic batch request dicts, custom_id -> row index)."""
    requests: list[dict[str, Any]] = []
    id_to_idx: dict[str, int] = {}
    used_ids: set[str] = set()

    for idx, row in enumerate(rows):
        ctx = str(row.get("context") or "").strip()
        q = str(row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        cid = _sanitize_custom_id(row, idx)
        if cid in used_ids:
            cid = f"{cid[:54]}_{idx}"[:64]
        used_ids.add(cid)
        id_to_idx[cid] = idx

        if not ctx or not q or not a:
            continue
        if is_nan_answer(a):
            continue

        user = build_judge_user_message(
            context=ctx, question=q, answer=a, max_context_chars=max_context_chars
        )
        requests.append(
            {
                "custom_id": cid,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": user}],
                },
            }
        )
    return requests, id_to_idx


def _submit_batches(
    client: Any,
    requests: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> list[str]:
    batch_ids: list[str] = []
    n = len(requests)
    for start in range(0, n, chunk_size):
        chunk = requests[start : start + chunk_size]
        print(f"Submitting batch {len(batch_ids) + 1} ({len(chunk)} requests) …", flush=True)
        batch = client.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"  batch_id={batch.id}  status={batch.processing_status}", flush=True)
    return batch_ids


def _poll_batches(client: Any, batch_ids: list[str], *, poll_interval_s: float) -> None:
    pending = set(batch_ids)
    while pending:
        time.sleep(poll_interval_s)
        still: set[str] = set()
        for bid in pending:
            batch = client.messages.batches.retrieve(bid)
            counts = getattr(batch, "request_counts", None)
            if counts:
                print(
                    f"  {bid}: {batch.processing_status} "
                    f"succeeded={getattr(counts, 'succeeded', '?')} "
                    f"errored={getattr(counts, 'errored', '?')} "
                    f"processing={getattr(counts, 'processing', '?')} "
                    f"expired={getattr(counts, 'expired', '?')}",
                    flush=True,
                )
            else:
                print(f"  {bid}: {batch.processing_status}", flush=True)
            if batch.processing_status != "ended":
                still.add(bid)
        pending = still
    print("All batches ended.", flush=True)


def _collect_batch_results(
    client: Any,
    batch_ids: list[str],
    id_to_idx: dict[str, int],
    rows: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge judge results into rows; return (rows_out, stats)."""
    results_by_idx: dict[int, dict[str, Any]] = {}
    stats = {
        "n_parse_error": 0,
        "n_api_error": 0,
        "n_expired": 0,
        "n_missing_request": 0,
        "n_skipped_nan": 0,
        "n_judged_ok": 0,
        "tier_counts": {"high": 0, "medium": 0, "low": 0, "drop": 0, "error": 0},
        "sum_g": 0.0,
        "sum_r": 0.0,
        "sum_dn": 0.0,
        "sum_o": 0.0,
    }

    for bid in batch_ids:
        print(f"Downloading results for {bid} …", flush=True)
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            idx = id_to_idx.get(cid)
            if idx is None:
                continue
            row = rows[idx]
            answer = str(row.get("answer") or "")
            rtype = result.result.type

            if rtype == "succeeded":
                raw = _extract_message_text(result.result.message)
                parsed = parse_judge_json(raw)
                block = normalize_judge_block(
                    provider=provider,
                    model=model,
                    parsed=parsed,
                    raw=raw,
                    answer=answer,
                )
            elif rtype == "errored":
                err = result.result.error
                err_type = getattr(err, "type", "unknown")
                block = normalize_judge_block(
                    provider=provider,
                    model=model,
                    parsed=None,
                    error=f"api_error:{err_type}",
                )
                stats["n_api_error"] += 1
            else:
                block = normalize_judge_block(
                    provider=provider,
                    model=model,
                    parsed=None,
                    error="expired",
                )
                stats["n_expired"] += 1

            results_by_idx[idx] = block

    rows_out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out = {**row}
        block = results_by_idx.get(idx)
        a = str(row.get("answer") or "").strip()
        if block is None:
            ctx = str(row.get("context") or "").strip()
            q = str(row.get("question") or "").strip()
            if is_nan_answer(a):
                block = skipped_nan_judge_block(provider=provider, model=model)
                stats["n_skipped_nan"] += 1
            elif not ctx or not q or not a:
                block = normalize_judge_block(
                    provider=provider,
                    model=model,
                    parsed=None,
                    error="missing_context_question_or_answer",
                    answer=a,
                )
                stats["n_missing_request"] += 1
            else:
                block = normalize_judge_block(
                    provider=provider,
                    model=model,
                    parsed=None,
                    error="no_batch_result",
                    answer=a,
                )
                stats["n_missing_request"] += 1

        out["llm_judge"] = block
        rows_out.append(out)

        if block.get("skipped"):
            stats["tier_counts"]["drop"] += 1
            continue

        if block.get("error"):
            stats["tier_counts"]["error"] += 1
            if block.get("error") == "parse_error":
                stats["n_parse_error"] += 1
            continue

        stats["n_judged_ok"] += 1
        stats["sum_g"] += float(block.get("grounding", 0))
        stats["sum_r"] += float(block.get("relevance", 0))
        stats["sum_dn"] += float(block.get("document_necessity", 0))
        stats["sum_o"] += float(block.get("overall", 0))
        tier = block.get("quality_tier", "error")
        if tier in stats["tier_counts"]:
            stats["tier_counts"][tier] += 1

    return rows_out, stats


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_qa_haiku_judge(ns: argparse.Namespace) -> int:
    qa_path = Path(ns.qa_jsonl).expanduser().resolve()
    if not qa_path.is_file():
        print(f"Not found: {qa_path}", file=sys.stderr)
        return 1

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in the environment.", file=sys.stderr)
        return 1

    try:
        import anthropic
    except ImportError:
        print("Missing dependency: anthropic\n  pip install anthropic", file=sys.stderr)
        return 1

    out_jsonl = (
        Path(ns.out_jsonl).expanduser().resolve()
        if ns.out_jsonl
        else qa_path.parent / f"{qa_path.stem}_haiku_judge.jsonl"
    )
    sum_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    )
    state_path = (
        Path(ns.state_json).expanduser().resolve()
        if ns.state_json
        else out_jsonl.with_name(out_jsonl.stem + "_batch_state.json")
    )

    model = ns.model
    provider = "anthropic"
    max_ctx = int(ns.max_context_chars)
    max_tok = int(ns.max_tokens)
    temp = float(ns.temperature)
    chunk_size = int(ns.batch_chunk_size)
    poll_s = float(ns.poll_interval_s)

    rows = _load_rows(qa_path, int(ns.max_rows))
    if not rows:
        print("No rows to judge.", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    batch_ids: list[str] = []
    id_to_idx: dict[str, int] = {}

    if ns.resume_only:
        if not state_path.is_file():
            print(f"No state file: {state_path}", file=sys.stderr)
            return 1
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch_ids = list(state.get("batch_ids") or [])
        id_to_idx = {k: int(v) for k, v in (state.get("custom_id_to_row_index") or {}).items()}
        print(f"Resume: {len(batch_ids)} batch(es), {len(rows)} rows", flush=True)
    else:
        requests, id_to_idx = _build_batch_requests(
            rows,
            model=model,
            max_context_chars=max_ctx,
            max_tokens=max_tok,
            temperature=temp,
        )
        n_skip_nan = sum(1 for r in rows if is_nan_answer(str(r.get("answer") or "")))
        print(
            f"Prepared {len(requests)} batch requests for {len(rows)} rows "
            f"(skipping {n_skip_nan} nan/empty answers, no API cost; "
            f"model={model}, chunk_size={chunk_size})",
            flush=True,
        )
        if not requests:
            print("No valid requests (all rows missing fields?).", file=sys.stderr)
            return 1

        if ns.submit_only:
            batch_ids = _submit_batches(client, requests, chunk_size=chunk_size)
            _save_state(
                state_path,
                {
                    "qa_jsonl": str(qa_path),
                    "model": model,
                    "batch_ids": batch_ids,
                    "custom_id_to_row_index": id_to_idx,
                    "n_requests": len(requests),
                },
            )
            print(f"Submitted only. State: {state_path}", flush=True)
            return 0

        batch_ids = _submit_batches(client, requests, chunk_size=chunk_size)
        _save_state(
            state_path,
            {
                "qa_jsonl": str(qa_path),
                "model": model,
                "batch_ids": batch_ids,
                "custom_id_to_row_index": id_to_idx,
                "n_requests": len(requests),
            },
        )

    _poll_batches(client, batch_ids, poll_interval_s=poll_s)

    rows_out, stats = _collect_batch_results(
        client, batch_ids, id_to_idx, rows, provider=provider, model=model
    )

    n_ok = stats["n_judged_ok"]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "schema": "qa_haiku_judge_summary/v1",
        "qa_jsonl": str(qa_path),
        "out_jsonl": str(out_jsonl),
        "state_json": str(state_path),
        "provider": provider,
        "model": model,
        "mode": "anthropic_message_batches",
        "settings": {
            "max_context_chars": max_ctx,
            "max_tokens": max_tok,
            "temperature": temp,
            "batch_chunk_size": chunk_size,
            "poll_interval_s": poll_s,
        },
        "batch_ids": batch_ids,
        "n_rows": len(rows),
        "stats": {
            **stats,
            "mean_grounding": round(stats["sum_g"] / max(1, n_ok), 4),
            "mean_relevance": round(stats["sum_r"] / max(1, n_ok), 4),
            "mean_document_necessity": round(stats["sum_dn"] / max(1, n_ok), 4),
            "mean_overall": round(stats["sum_o"] / max(1, n_ok), 4),
        },
        "tier_rules": {
            "high": "grounding,relevance,document_necessity,overall all >= 4",
            "medium": "otherwise above low",
            "low": "any score <= 2 (except grounding<=2 -> drop)",
            "drop": "nan/empty answer or grounding <= 2",
        },
        "notes": [
            "Batch API = 50% token pricing vs standard Messages API.",
            "Rows with answer nan/none/empty are not sent to the API (quality_tier=drop).",
            "Pin model id and run date in thesis methods.",
        ],
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(
        f"Judged ok: {n_ok}/{len(rows)}  tiers={stats['tier_counts']}  "
        f"mean overall={summary['stats']['mean_overall']}",
        flush=True,
    )
    return 0 if n_ok > 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score QA JSONL with Anthropic Haiku via Message Batches API."
    )
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--state-json", type=Path, default=None, help="Batch ids + id map for resume.")
    p.add_argument("--model", type=str, default=DEFAULT_ANTHROPIC_MODEL)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows.")
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--batch-chunk-size",
        type=int,
        default=DEFAULT_BATCH_CHUNK_SIZE,
        help="Requests per Anthropic batch (max 100k; keep under ~256MB payload).",
    )
    p.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    p.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit batch(es) and save state; do not poll or write results.",
    )
    p.add_argument(
        "--resume-only",
        action="store_true",
        help="Poll existing batch ids from state-json and write results.",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run_qa_haiku_judge(build_arg_parser().parse_args()))

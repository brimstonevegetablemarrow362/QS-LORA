#!/usr/bin/env python3
"""
LLM-as-judge for QA JSONL (thesis / research only — requires API key + network).

Scores each row on grounding in context, question relevance, and overall quality.
Writes ``llm_judge`` block per row + summary JSON.

Usage:
  export OPENAI_API_KEY=...
  python -m thesis.cli qa-llm-judge \\
    --qa-jsonl ./thesis/experiments/repliqa/runs/repliqa_train_0-3/train/synthetic_qa.jsonl \\
    --max-rows 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _call_openai(*, model: str, messages: list[dict[str, str]], temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=256,
    )
    return (r.choices[0].message.content or "").strip()


def _judge_one(
    *,
    provider: str,
    model: str,
    context: str,
    question: str,
    answer: str,
    max_context_chars: int,
    temperature: float,
) -> dict[str, Any]:
    user = build_judge_user_message(
        context=context, question=question, answer=answer, max_context_chars=max_context_chars
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    if provider == "openai":
        raw = _call_openai(model=model, messages=messages, temperature=temperature)
    else:
        raise ValueError(f"unsupported provider: {provider}")

    parsed = parse_judge_json(raw)
    return normalize_judge_block(
        provider=provider,
        model=model,
        parsed=parsed,
        raw=raw,
        answer=answer,
    )


def run_qa_llm_judge(ns: argparse.Namespace) -> int:
    qa_path = Path(ns.qa_jsonl).expanduser().resolve()
    if not qa_path.is_file():
        print(f"Not found: {qa_path}", file=sys.stderr)
        return 1

    if ns.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in the environment.", file=sys.stderr)
        return 1

    out_jsonl = (
        Path(ns.out_jsonl).expanduser().resolve()
        if ns.out_jsonl
        else qa_path.parent / f"{qa_path.stem}_llm_judge.jsonl"
    )
    sum_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    )

    rows_in: list[dict[str, Any]] = []
    for line_no, line in enumerate(qa_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows_in.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Line {line_no}: {e}", file=sys.stderr)

    if ns.max_rows > 0:
        rows_in = rows_in[: int(ns.max_rows)]

    provider = ns.provider
    model = ns.model
    max_ctx = int(ns.max_context_chars)
    temp = float(ns.temperature)
    conc = max(1, int(ns.concurrency))
    delay = float(ns.request_delay_s)

    print(f"LLM judge: provider={provider} model={model} rows={len(rows_in)} concurrency={conc}", flush=True)

    results: list[dict[str, Any] | None] = [None] * len(rows_in)
    lock = threading.Lock()
    n_ok = n_err = n_skipped_nan = 0
    sum_g = sum_r = sum_o = 0.0

    def job(idx: int, row: dict[str, Any]) -> None:
        nonlocal n_ok, n_err, n_skipped_nan, sum_g, sum_r, sum_o
        ctx = str(row.get("context") or "").strip()
        q = str(row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        out_row = {**row}
        if is_nan_answer(a):
            block = skipped_nan_judge_block(provider=provider, model=model)
        elif not ctx or not q or not a:
            block = {"error": "missing_context_question_or_answer"}
        else:
            try:
                if delay > 0:
                    time.sleep(delay * (idx % conc) * 0.1)
                block = _judge_one(
                    provider=provider,
                    model=model,
                    context=ctx,
                    question=q,
                    answer=a,
                    max_context_chars=max_ctx,
                    temperature=temp,
                )
            except Exception as e:
                block = {"error": f"api_error: {e}"}
        out_row["llm_judge"] = block
        with lock:
            results[idx] = out_row
            if block.get("skipped"):
                n_skipped_nan += 1
            elif block.get("error"):
                n_err += 1
            else:
                n_ok += 1
                sum_g += float(block.get("grounding", 0))
                sum_r += float(block.get("relevance", 0))
                sum_o += float(block.get("overall", 0))
            if (n_ok + n_err) % 20 == 0:
                print(f"  ... {n_ok + n_err}/{len(rows_in)} judged", flush=True)

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(job, i, r) for i, r in enumerate(rows_in)]
        for f in as_completed(futs):
            f.result()

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in results:
            if r is not None:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "schema": "qa_llm_judge_summary/v1",
        "qa_jsonl": str(qa_path),
        "out_jsonl": str(out_jsonl),
        "provider": provider,
        "model": model,
        "n_rows": len(rows_in),
        "n_judged_ok": n_ok,
        "n_skipped_nan": n_skipped_nan,
        "n_errors": n_err,
        "mean_grounding": round(sum_g / max(1, n_ok), 4),
        "mean_relevance": round(sum_r / max(1, n_ok), 4),
        "mean_overall": round(sum_o / max(1, n_ok), 4),
        "notes": [
            "Thesis/research use only; pin model name and date in your write-up.",
            "Prefer stratified samples (e.g. 200–500 rows) before full-corpus judging.",
        ],
    }
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(f"Judged ok: {n_ok}  errors: {n_err}  mean overall={summary['mean_overall']}", flush=True)
    return 0 if n_ok > 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-as-judge scores for QA JSONL (OpenAI API).")
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--provider", type=str, default="openai", choices=("openai",))
    p.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows (can be expensive).")
    p.add_argument("--max-context-chars", type=int, default=12000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-delay-s", type=float, default=0.0, help="Optional throttle between requests.")
    return p


if __name__ == "__main__":
    raise SystemExit(run_qa_llm_judge(build_arg_parser().parse_args()))

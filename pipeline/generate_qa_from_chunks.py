#!/usr/bin/env python3
"""
Generate normalized Q/A JSONL from chunk JSONL via a **local vLLM** OpenAI-compatible server.

Merge your generator LoRA into a single folder, then on the **same GPU node** run vLLM, e.g.:

  python -m vllm.entrypoints.openai.api_server \\
    --model /path/to/merged-generator \\
    --host 127.0.0.1 --port 8100 --dtype auto

This script does **not** load Transformers on the client — it only sends HTTP requests.

Each output row: context, question, answer, source, chunk_id.

Usage:
  python generate_qa_from_chunks.py \\
    --chunks chunks.jsonl --out qa/all.jsonl \\
    --vllm-base-url http://127.0.0.1:8100 \\
    --vllm-model /path/to/merged-generator

Defaults follow **pipeline/paths.py** (override with CLI flags or GENERATOR_VLLM_* env vars).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from paths import (
    QA_GEN_CONCURRENCY,
    QA_GEN_MAX_NEW_TOKENS,
    QA_GEN_TEMPERATURE,
    generator_vllm_base_url,
    generator_vllm_model_id,
)
from prompts import GENERATOR_SYSTEM, generator_user_block


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


def chunk_to_excerpt(row: dict[str, Any]) -> str:
    """Support domain_v1 chunk_markdown.jsonl and chunking.py-style rows."""
    parts: list[str] = []
    title = (row.get("title") or "").strip()
    if title and title != "INTRO":
        parts.append(title)
    text = (row.get("text") or "").strip()
    if text:
        parts.append(text)
    imgs = row.get("image_descriptions")
    if isinstance(imgs, list) and imgs:
        parts.append("\n[Figure / image notes]\n" + "\n".join(str(x) for x in imgs if str(x).strip()))
    return "\n\n".join(parts).strip()


def extract_json_object(raw: str) -> Optional[dict[str, Any]]:
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "question" in obj and "answer" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\"question\"[^{}]*\"answer\"[^{}]*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _openai_base_url(base_url: str) -> str:
    bu = base_url.rstrip("/")
    if not bu.endswith("/v1"):
        bu = bu + "/v1"
    return bu


def _check_vllm(client: Any) -> None:
    try:
        client.models.list()
    except Exception as e:
        raise SystemExit(
            f"Cannot reach vLLM OpenAI API: {e}\n"
            "Start the server on this node first, e.g.:\n"
            "  python -m vllm.entrypoints.openai.api_server --model <merged-generator> "
            "--host 127.0.0.1 --port 8100 --dtype auto"
        ) from e


def _vllm_chat(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
    }
    r = client.chat.completions.create(**kw)
    return (r.choices[0].message.content or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Q/A JSONL from chunks (vLLM OpenAI API only)")
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--vllm-base-url",
        type=str,
        default=generator_vllm_base_url(),
        help="Override paths.generator_vllm_base_url() / DOMAIN_GENERATOR_VLLM_BASE_URL.",
    )
    ap.add_argument(
        "--vllm-model",
        type=str,
        default=generator_vllm_model_id(),
        help="Override paths.generator_vllm_model_id() / DOMAIN_GENERATOR_VLLM_MODEL.",
    )
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks (dev cap)")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=QA_GEN_CONCURRENCY,
        help="Parallel requests (default paths.QA_GEN_CONCURRENCY).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=QA_GEN_MAX_NEW_TOKENS,
        help="max_tokens per completion (default paths.QA_GEN_MAX_NEW_TOKENS).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=QA_GEN_TEMPERATURE,
        help="Sampling temperature (default paths.QA_GEN_TEMPERATURE).",
    )
    ap.add_argument(
        "--source-tag",
        type=str,
        default="user_corpus",
        help="Single `source` value for all rows so train/val/test split is not per-chunk",
    )
    ap.add_argument(
        "--no-length-sort",
        action="store_true",
        help="Keep chunks in file order (default: sort by user message length for steadier batching).",
    )
    args = ap.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    from openai import OpenAI

    client = OpenAI(base_url=_openai_base_url(args.vllm_base_url), api_key="unused")
    print(f"vLLM base: {_openai_base_url(args.vllm_base_url)}  model={args.vllm_model!r}", flush=True)
    _check_vllm(client)

    rows = load_jsonl(args.chunks)
    if args.max_chunks > 0:
        rows = rows[: args.max_chunks]

    work: list[tuple[int, str, str, list[dict[str, str]]]] = []
    for i, row in enumerate(rows):
        excerpt = chunk_to_excerpt(row)
        if len(excerpt) < 40:
            print(f"skip {i}: excerpt too short", flush=True)
            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{len(rows)} chunks, 0 pairs written (short skips only)", flush=True)
            continue
        cid = (row.get("chunk_id") or row.get("source") or f"chunk_{i}") or f"chunk_{i}"
        messages = [
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user", "content": generator_user_block(excerpt)},
        ]
        work.append((i, excerpt, cid, messages))

    length_sort = not args.no_length_sort
    if length_sort and len(work) > 1:
        work.sort(key=lambda t: len(t[3][1]["content"]))

    if not work:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        print("No chunks to send to vLLM (all skipped or empty).", flush=True)
        return

    print(
        f"Generating via vLLM: concurrency={args.concurrency}, max_new_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}, length_sort={length_sort}, jobs={len(work)}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    lock = threading.Lock()
    done = 0

    def one_job(item: tuple[int, str, str, list[dict[str, str]]]) -> tuple[int, Optional[str]]:
        nonlocal n_ok, done
        i, excerpt, cid, messages = item
        try:
            raw = _vllm_chat(
                client=client,
                model=args.vllm_model,
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"skip {i} ({cid}): vLLM error: {e}", flush=True)
            line = None
        else:
            parsed = extract_json_object(raw)
            if not parsed:
                print(f"skip {i} ({cid}): could not parse JSON from model output", flush=True)
                line = None
            else:
                q = str(parsed.get("question", "")).strip()
                a = str(parsed.get("answer", "")).strip()
                if not q or not a:
                    print(f"skip {i} ({cid}): empty q/a", flush=True)
                    line = None
                else:
                    record = {
                        "context": excerpt,
                        "question": q,
                        "answer": a,
                        "source": args.source_tag,
                        "chunk_id": str(cid),
                    }
                    line = json.dumps(record, ensure_ascii=False)
                    with lock:
                        n_ok += 1
        with lock:
            done += 1
            if done % 10 == 0 or done == len(work):
                print(f"  ... {done}/{len(work)} jobs, {n_ok} pairs written", flush=True)
        return i, line

    lines_by_i: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_job, w) for w in work]
        for fut in as_completed(futures):
            i, line = fut.result()
            if line is not None:
                lines_by_i[i] = line

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i in sorted(lines_by_i):
            out_f.write(lines_by_i[i] + "\n")

    print(f"Done. Wrote {len(lines_by_i)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

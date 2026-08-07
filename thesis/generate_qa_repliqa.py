#!/usr/bin/env python3
"""
Generate synthetic Q/A from RepLiQA JSONL — **one vLLM call per unique document**
(``document_id``), not per human question row (5× duplicate contexts).

Outputs under ``finetuning/thesis/experiments/repliqa/runs/<run_name>/``:

  config.json
  train/documents_unique.jsonl
  train/synthetic_qa.jsonl
  train/generation_log.jsonl
  summary.json

Default train splits: repliqa_0 … repliqa_3 (hold out repliqa_4 for human eval).

Requires vLLM OpenAI API (same as generate_qa_from_chunks.py).

Usage (from finetuning/, GPU node with vLLM up):
  python -m thesis.cli generate-repliqa-synthetic \\
    --vllm-base-url http://127.0.0.1:8100 \\
    --vllm-model /path/to/merged-generator

  python -m thesis.cli generate-repliqa-synthetic --max-documents 50
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import thesis.bootstrap  # noqa: F401

from generate_qa_from_chunks import (
    _check_vllm,
    _openai_base_url,
    _vllm_chat,
    extract_json_object,
)
from paths import (
    QA_GEN_CONCURRENCY,
    QA_GEN_MAX_NEW_TOKENS,
    QA_GEN_TEMPERATURE,
    generator_vllm_base_url,
    generator_vllm_model_id,
)
from thesis.prompts import GENERATOR_SYSTEM, generator_user_block
from thesis.paths import DEFAULT_TRAIN_SPLITS, REPLIQA_EXP_ROOT, REPLIQA_JSONL_DIR

DEFAULT_JSONL_DIR = REPLIQA_JSONL_DIR
DEFAULT_EXP_ROOT = REPLIQA_EXP_ROOT


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


def dedupe_documents(
    jsonl_paths: list[Path],
    *,
    min_context_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One record per document_id; context taken from first row seen."""
    by_id: dict[str, dict[str, Any]] = {}
    stats = {
        "input_rows": 0,
        "unique_documents": 0,
        "skipped_short_context": 0,
        "skipped_missing_doc_id": 0,
        "splits_seen": [],
    }

    for path in jsonl_paths:
        split = path.stem
        if split not in stats["splits_seen"]:
            stats["splits_seen"].append(split)
        for row in load_jsonl(path):
            stats["input_rows"] += 1
            doc_id = (row.get("document_id") or "").strip()
            ctx = (row.get("context") or row.get("document_extracted") or "").strip()
            if not doc_id:
                stats["skipped_missing_doc_id"] += 1
                continue
            if doc_id not in by_id:
                if len(ctx) < min_context_chars:
                    stats["skipped_short_context"] += 1
                    continue
                by_id[doc_id] = {
                    "document_id": doc_id,
                    "context": ctx,
                    "document_topic": row.get("document_topic"),
                    "repliqa_splits": [split],
                    "human_qa_rows_in_source": 1,
                }
            else:
                rec = by_id[doc_id]
                if split not in rec["repliqa_splits"]:
                    rec["repliqa_splits"].append(split)
                rec["human_qa_rows_in_source"] = int(rec.get("human_qa_rows_in_source", 0)) + 1

    docs = list(by_id.values())
    stats["unique_documents"] = len(docs)
    return docs, stats


def _run_name_arg(name: str | None) -> str:
    if name and name.strip():
        return name.strip().replace(" ", "_")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_generate_repliqa(ns: argparse.Namespace) -> int:
    jsonl_dir = Path(ns.jsonl_dir).expanduser().resolve()
    exp_root = Path(ns.exp_root).expanduser().resolve()
    splits = list(ns.splits)
    paths = [jsonl_dir / f"{s}.jsonl" for s in splits]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print("Missing JSONL files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    docs, dedupe_stats = dedupe_documents(paths, min_context_chars=int(ns.min_context_chars))
    if ns.max_documents > 0:
        docs = docs[: int(ns.max_documents)]

    run_name = _run_name_arg(ns.run_name)
    run_dir = exp_root / "runs" / run_name
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    docs_path = train_dir / "documents_unique.jsonl"
    with open(docs_path, "w", encoding="utf-8") as fp:
        for d in docs:
            fp.write(json.dumps(d, ensure_ascii=False) + "\n")

    if not docs:
        print("No documents after dedupe; nothing to generate.", file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI(base_url=_openai_base_url(ns.vllm_base_url), api_key="unused")
    print(f"vLLM base: {_openai_base_url(ns.vllm_base_url)}  model={ns.vllm_model!r}", flush=True)
    _check_vllm(client)

    pairs_per_doc = max(1, int(ns.pairs_per_doc))
    concurrency = max(1, int(ns.concurrency))
    max_tok = int(ns.max_new_tokens)
    temp = float(ns.temperature)
    source_tag = str(ns.source_tag)

    # Jobs: one vLLM request per (document, pair_index)
    work: list[dict[str, Any]] = []
    for d in docs:
        doc_id = d["document_id"]
        ctx = d["context"]
        for k in range(pairs_per_doc):
            messages = [
                {"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": generator_user_block(ctx)},
            ]
            work.append(
                {
                    "document_id": doc_id,
                    "pair_index": k,
                    "context": ctx,
                    "document_topic": d.get("document_topic"),
                    "repliqa_splits": d.get("repliqa_splits"),
                    "messages": messages,
                }
            )

    if not ns.no_length_sort and len(work) > 1:
        work.sort(key=lambda j: len(j["messages"][1]["content"]))

    print(
        f"RepLiQA synthetic gen: documents={len(docs)} pairs_per_doc={pairs_per_doc} "
        f"jobs={len(work)} concurrency={concurrency} run_dir={run_dir}",
        flush=True,
    )

    synthetic_path = train_dir / "synthetic_qa.jsonl"
    log_path = train_dir / "generation_log.jsonl"
    lock = threading.Lock()
    n_ok = 0
    n_fail = 0
    done = 0
    synthetic_lines: list[str] = []

    def one_job(job: dict[str, Any]) -> dict[str, Any]:
        nonlocal n_ok, n_fail, done
        doc_id = job["document_id"]
        k = job["pair_index"]
        raw = ""
        log_entry: dict[str, Any] = {
            "document_id": doc_id,
            "pair_index": k,
            "status": "ok",
            "error": None,
        }
        record_line = None
        try:
            raw = _vllm_chat(
                client=client,
                model=ns.vllm_model,
                messages=job["messages"],
                max_tokens=max_tok,
                temperature=temp,
            )
            parsed = extract_json_object(raw)
            if not parsed:
                log_entry["status"] = "parse_error"
                log_entry["error"] = "could_not_parse_json"
            else:
                q = str(parsed.get("question", "")).strip()
                a = str(parsed.get("answer", "")).strip()
                if not q or not a:
                    log_entry["status"] = "empty_qa"
                    log_entry["error"] = "empty_question_or_answer"
                else:
                    record = {
                        "context": job["context"],
                        "question": q,
                        "answer": a,
                        "source": source_tag,
                        "chunk_id": f"{doc_id}::syn_{k:02d}",
                        "document_id": doc_id,
                        "document_topic": job.get("document_topic"),
                        "repliqa_splits": job.get("repliqa_splits"),
                        "synthetic_pair_index": k,
                        "generator_model": ns.vllm_model,
                    }
                    record_line = json.dumps(record, ensure_ascii=False)
                    log_entry["status"] = "ok"
        except Exception as e:
            log_entry["status"] = "vllm_error"
            log_entry["error"] = str(e)

        if raw and log_entry["status"] != "ok":
            log_entry["raw_preview"] = raw[:300] if len(raw) > 300 else raw

        with lock:
            done += 1
            if record_line is not None:
                n_ok += 1
                synthetic_lines.append(record_line)
            else:
                n_fail += 1
            if done % 20 == 0 or done == len(work):
                print(f"  ... {done}/{len(work)} jobs, {n_ok} ok, {n_fail} failed", flush=True)
        return log_entry

    log_entries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one_job, j) for j in work]
        for fut in as_completed(futs):
            log_entries.append(fut.result())

    with open(synthetic_path, "w", encoding="utf-8") as fp:
        for line in synthetic_lines:
            fp.write(line + "\n")

    with open(log_path, "w", encoding="utf-8") as fp:
        for entry in log_entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    config = {
        "schema": "repliqa_synthetic_run/v1",
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jsonl_dir": str(jsonl_dir),
        "splits": splits,
        "vllm_base_url": ns.vllm_base_url,
        "vllm_model": ns.vllm_model,
        "pairs_per_doc": pairs_per_doc,
        "min_context_chars": int(ns.min_context_chars),
        "max_documents": int(ns.max_documents),
        "concurrency": concurrency,
        "max_new_tokens": max_tok,
        "temperature": temp,
        "source_tag": source_tag,
        "paths": {
            "run_dir": str(run_dir),
            "documents_unique": str(docs_path),
            "synthetic_qa": str(synthetic_path),
            "generation_log": str(log_path),
        },
        "note": "Human eval remains in data/repliqa/jsonl (e.g. repliqa_4); do not use for synthetic train.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = {
        **dedupe_stats,
        "documents_scheduled": len(docs),
        "generation_jobs": len(work),
        "synthetic_pairs_written": n_ok,
        "generation_failures": n_fail,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        latest = exp_root / "latest"
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        (exp_root / "LATEST_RUN.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    print(f"Wrote {docs_path} ({len(docs)} documents)", flush=True)
    print(f"Wrote {synthetic_path} ({n_ok} pairs)", flush=True)
    print(f"Wrote {log_path}", flush=True)
    print(f"Run: {run_dir}", flush=True)
    return 0 if n_ok > 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic Q/A from RepLiQA (unique documents only).")
    p.add_argument(
        "--jsonl-dir",
        type=Path,
        default=DEFAULT_JSONL_DIR,
        help="Directory with repliqa_*.jsonl exports.",
    )
    p.add_argument(
        "--exp-root",
        type=Path,
        default=DEFAULT_EXP_ROOT,
        help="Root for experiments/repliqa/runs/<run_name>/",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_TRAIN_SPLITS),
        help="RepLiQA splits to use (default: repliqa_0..3, excludes test repliqa_4).",
    )
    p.add_argument("--run-name", type=str, default=None, help="Run folder name (default: UTC timestamp).")
    p.add_argument("--pairs-per-doc", type=int, default=1, help="Synthetic Q/A pairs per unique document.")
    p.add_argument("--max-documents", type=int, default=0, help="Cap unique documents (0 = all).")
    p.add_argument("--min-context-chars", type=int, default=40)
    p.add_argument("--vllm-base-url", type=str, default=generator_vllm_base_url())
    p.add_argument("--vllm-model", type=str, default=generator_vllm_model_id())
    p.add_argument("--concurrency", type=int, default=QA_GEN_CONCURRENCY)
    p.add_argument("--max-new-tokens", type=int, default=QA_GEN_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=QA_GEN_TEMPERATURE)
    p.add_argument(
        "--source-tag",
        type=str,
        default="repliqa/synthetic/train",
        help="`source` field on synthetic rows.",
    )
    p.add_argument("--no-length-sort", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(run_generate_repliqa(build_arg_parser().parse_args()))

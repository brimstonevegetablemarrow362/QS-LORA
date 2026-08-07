#!/usr/bin/env python3
"""
Download ServiceNow/repliqa into the finetuning tree and optionally export JSONL.

HF cache (Arrow shards):  <finetuning>/data/repliqa/hf_cache/
Exported JSONL (optional): <finetuning>/data/repliqa/jsonl/<split>.jsonl

Usage (from finetuning/):
  python -m thesis.cli download-repliqa --export-jsonl
  python thesis/cli.py download-repliqa --export-jsonl
  python domain_v1/download_repliqa.py --splits repliqa_0 --export-jsonl
  python domain_v1/download_repliqa.py --export-jsonl --drop-unanswerable

If the dataset is gated, run:  huggingface-cli login
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from thesis.paths import REPLIQA_DATA_DIR, REPLIQA_HF_CACHE_DIR, REPLIQA_JSONL_DIR

REPLIQA_ID = "ServiceNow/repliqa"
DEFAULT_DATA_DIR = REPLIQA_DATA_DIR
DEFAULT_CACHE_DIR = REPLIQA_HF_CACHE_DIR
DEFAULT_JSONL_DIR = REPLIQA_JSONL_DIR


def _row_to_record(row: dict, *, split: str, drop_unanswerable: bool) -> dict | None:
    ctx = (row.get("document_extracted") or "").strip()
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    if not ctx or not q:
        return None
    if drop_unanswerable and a.upper() == "UNANSWERABLE":
        return None
    doc_id = row.get("document_id") or ""
    qid = row.get("question_id") or ""
    return {
        "context": ctx,
        "question": q,
        "answer": a,
        "source": f"repliqa/{split}",
        "chunk_id": str(qid or doc_id),
        "document_id": doc_id,
        "document_topic": row.get("document_topic"),
        "long_answer": row.get("long_answer"),
        "repliqa_split": split,
    }


def download_and_export(
    *,
    cache_dir: Path,
    jsonl_dir: Path | None,
    splits: list[str] | None,
    export_jsonl: bool,
    drop_unanswerable: bool,
    max_rows_per_split: int,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets>=2.14.0", file=sys.stderr)
        return 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {REPLIQA_ID} (cache_dir={cache_dir}) …", flush=True)
    ds = load_dataset(REPLIQA_ID, cache_dir=str(cache_dir))

    available = list(ds.keys())
    print(f"Available splits: {available}", flush=True)

    use_splits = splits if splits else available
    unknown = [s for s in use_splits if s not in ds]
    if unknown:
        print(f"Unknown split(s): {unknown}", file=sys.stderr)
        return 1

    if export_jsonl:
        assert jsonl_dir is not None
        jsonl_dir.mkdir(parents=True, exist_ok=True)

    total_exported = 0
    for split in use_splits:
        part = ds[split]
        n = len(part)
        limit = n if max_rows_per_split <= 0 else min(n, max_rows_per_split)
        print(f"  {split}: {n} rows" + (f" (export cap {limit})" if limit < n else ""), flush=True)

        if not export_jsonl:
            continue

        out_path = jsonl_dir / f"{split}.jsonl"
        written = 0
        skipped = 0
        with open(out_path, "w", encoding="utf-8") as fp:
            for i in range(limit):
                row = part[i]
                rec = _row_to_record(row, split=split, drop_unanswerable=drop_unanswerable)
                if rec is None:
                    skipped += 1
                    continue
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        total_exported += written
        print(f"    → {out_path}  ({written} rows, {skipped} skipped)", flush=True)

    print("Done.", flush=True)
    if export_jsonl:
        print(f"Total exported rows: {total_exported}", flush=True)
    else:
        print(f"Dataset cached under: {cache_dir}", flush=True)
        print("Re-run with --export-jsonl to write domain_v1-compatible JSONL.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ServiceNow/repliqa into finetuning/data/repliqa")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face datasets cache directory.",
    )
    ap.add_argument(
        "--jsonl-dir",
        type=Path,
        default=DEFAULT_JSONL_DIR,
        help="Output directory for JSONL exports.",
    )
    ap.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Write context/question/answer JSONL per split.",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Subset of splits to export (default: all available).",
    )
    ap.add_argument(
        "--drop-unanswerable",
        action="store_true",
        help="Skip rows where answer is UNANSWERABLE.",
    )
    ap.add_argument(
        "--max-rows-per-split",
        type=int,
        default=0,
        help="Cap rows per split when exporting (0 = all).",
    )
    args = ap.parse_args()

    return download_and_export(
        cache_dir=args.cache_dir.expanduser().resolve(),
        jsonl_dir=args.jsonl_dir.expanduser().resolve() if args.export_jsonl else None,
        splits=args.splits,
        export_jsonl=args.export_jsonl,
        drop_unanswerable=args.drop_unanswerable,
        max_rows_per_split=args.max_rows_per_split,
    )


if __name__ == "__main__":
    raise SystemExit(main())

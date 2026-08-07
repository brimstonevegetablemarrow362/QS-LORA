#!/usr/bin/env python3
"""
Download allenai/quoref and export cleaned JSONL for thesis experiments.

JSONL:  <finetuning>/data/quoref/jsonl/{train,validation}.jsonl

Usage (from finetuning/):
  python -m thesis.cli download-quoref --export-jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUOREF_ID = "allenai/quoref"


def _passage_id(context: str, title: str) -> str:
    digest = hashlib.sha256(context.strip().encode("utf-8")).hexdigest()[:16]
    if title.strip():
        tslug = hashlib.sha256(title.strip().encode("utf-8")).hexdigest()[:8]
        return f"quoref_{tslug}_{digest}"
    return f"quoref_{digest}"


def _row_to_record(row: dict[str, Any], *, split: str) -> dict[str, Any] | None:
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    row_id = (row.get("id") or "").strip()
    title = (row.get("title") or "").strip()
    if not context or not question or not row_id:
        return None

    answers_raw = row.get("answers") or {}
    texts = answers_raw.get("text") if isinstance(answers_raw, dict) else None
    if not texts:
        return None

    answer_texts = [str(t).strip() for t in texts if str(t).strip()]
    if not answer_texts:
        return None
    answers = [{"text": t, "type": "span"} for t in answer_texts]

    return {
        "eval_id": row_id,
        "section_id": _passage_id(context, title),
        "title": title,
        "context": context,
        "question": question,
        "answers": answers,
        "source": f"quoref/{split}",
        "url": row.get("url"),
    }


def download_and_export(
    *,
    cache_dir: Path,
    jsonl_dir: Path | None,
    splits: list[str] | None,
    export_jsonl: bool,
    max_rows_per_split: int,
    dedupe_eval_id: bool,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets>=2.14.0", file=sys.stderr)
        return 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {QUOREF_ID} (cache_dir={cache_dir}) …", flush=True)
    # HF deprecated loading scripts; parquet conversion ref works with modern ``datasets``.
    try:
        ds = load_dataset(QUOREF_ID, cache_dir=str(cache_dir))
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        print("Falling back to parquet revision refs/convert/parquet …", flush=True)
        ds = load_dataset(QUOREF_ID, revision="refs/convert/parquet", cache_dir=str(cache_dir))

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

    manifest: dict[str, Any] = {
        "schema": "quoref_download_manifest/v1",
        "dataset_id": QUOREF_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "jsonl_dir": str(jsonl_dir) if jsonl_dir else None,
        "splits": {},
    }

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
        skipped_empty = 0
        skipped_dup = 0
        seen_id: set[str] = set()

        with out_path.open("w", encoding="utf-8") as fp:
            for i in range(limit):
                row = dict(part[i])
                rec = _row_to_record(row, split=split)
                if rec is None:
                    skipped_empty += 1
                    continue
                eid = rec["eval_id"]
                if dedupe_eval_id and eid in seen_id:
                    skipped_dup += 1
                    continue
                seen_id.add(eid)
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

        total_exported += written
        manifest["splits"][split] = {
            "hf_rows": n,
            "exported_rows": written,
            "skipped_empty": skipped_empty,
            "skipped_duplicate_eval_id": skipped_dup,
            "jsonl": str(out_path),
        }
        print(
            f"    → {out_path}  ({written} rows, {skipped_empty} skipped, {skipped_dup} dup)",
            flush=True,
        )

    manifest_path = cache_dir.parent / "quoref_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)
    print("Done.", flush=True)
    if export_jsonl:
        print(f"Total exported rows: {total_exported}", flush=True)
    return 0


def main() -> int:
    from thesis.paths import QUOREF_HF_CACHE_DIR, QUOREF_JSONL_DIR

    ap = argparse.ArgumentParser(description="Download and clean allenai/quoref")
    ap.add_argument("--cache-dir", type=Path, default=QUOREF_HF_CACHE_DIR)
    ap.add_argument("--jsonl-dir", type=Path, default=QUOREF_JSONL_DIR)
    ap.add_argument("--export-jsonl", action="store_true")
    ap.add_argument("--splits", nargs="+", default=None, help="train validation (default: all)")
    ap.add_argument("--max-rows-per-split", type=int, default=0, help="0 = all rows")
    ap.add_argument("--no-dedupe-eval-id", action="store_true")
    args = ap.parse_args()

    return download_and_export(
        cache_dir=args.cache_dir.expanduser().resolve(),
        jsonl_dir=args.jsonl_dir.expanduser().resolve() if args.export_jsonl else None,
        splits=args.splits,
        export_jsonl=args.export_jsonl,
        max_rows_per_split=args.max_rows_per_split,
        dedupe_eval_id=not args.no_dedupe_eval_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())

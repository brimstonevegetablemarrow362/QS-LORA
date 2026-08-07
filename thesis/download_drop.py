#!/usr/bin/env python3
"""
Download ucinlp/drop and export cleaned JSONL for thesis experiments.

HF cache:  <finetuning>/data/drop/hf_cache/
JSONL:     <finetuning>/data/drop/jsonl/{train,validation}.jsonl
Manifest:  <finetuning>/data/drop/drop_manifest.json

Cleaning:
  - require non-empty passage, question, and at least one answer span
  - strip whitespace; drop duplicate query_id (keep first)
  - normalize HF answers_spans → answers: [{text, type}, ...]
  - RepLiQA-compatible loaders can use answers[0].text as primary gold

Usage (from finetuning/):
  python -m thesis.cli download-drop --export-jsonl
  python -m thesis.cli download-drop --export-jsonl --max-rows-per-split 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DROP_ID = "ucinlp/drop"


def _normalize_answers_spans(raw: Any) -> tuple[list[str], list[str]] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        spans = raw.get("spans") or []
        types = raw.get("types") or []
    else:
        return None
    out_spans: list[str] = []
    out_types: list[str] = []
    for i, s in enumerate(spans):
        text = str(s).strip()
        if not text:
            continue
        out_spans.append(text)
        t = str(types[i]).strip() if i < len(types) else "span"
        out_types.append(t or "span")
    if not out_spans:
        return None
    return out_spans, out_types


def _row_to_record(row: dict[str, Any], *, split: str) -> dict[str, Any] | None:
    passage = (row.get("passage") or "").strip()
    question = (row.get("question") or "").strip()
    query_id = (row.get("query_id") or "").strip()
    section_id = (row.get("section_id") or "").strip()
    if not passage or not question or not query_id:
        return None

    parsed = _normalize_answers_spans(row.get("answers_spans"))
    if parsed is None:
        return None
    spans, types = parsed
    answers = [{"text": s, "type": t} for s, t in zip(spans, types)]

    return {
        "eval_id": query_id,
        "section_id": section_id,
        "context": passage,
        "question": question,
        "answers": answers,
        "source": f"drop/{split}",
    }


def download_and_export(
    *,
    cache_dir: Path,
    jsonl_dir: Path | None,
    splits: list[str] | None,
    export_jsonl: bool,
    max_rows_per_split: int,
    dedupe_query_id: bool,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets>=2.14.0", file=sys.stderr)
        return 1

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DROP_ID} (cache_dir={cache_dir}) …", flush=True)
    ds = load_dataset(DROP_ID, cache_dir=str(cache_dir))

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
        "schema": "drop_download_manifest/v1",
        "dataset_id": DROP_ID,
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
        type_counts: Counter[str] = Counter()
        seen_query: set[str] = set()

        with open(out_path, "w", encoding="utf-8") as fp:
            for i in range(limit):
                row = dict(part[i])
                rec = _row_to_record(row, split=split)
                if rec is None:
                    skipped_empty += 1
                    continue
                qid = rec["eval_id"]
                if dedupe_query_id and qid in seen_query:
                    skipped_dup += 1
                    continue
                seen_query.add(qid)
                types = {a["type"] for a in rec["answers"]}
                type_counts["mixed" if len(types) > 1 else rec["answers"][0]["type"]] += 1
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

        total_exported += written
        manifest["splits"][split] = {
            "hf_rows": n,
            "exported_rows": written,
            "skipped_empty": skipped_empty,
            "skipped_duplicate_query_id": skipped_dup,
            "answer_type_counts": dict(type_counts),
            "jsonl": str(out_path),
        }
        print(
            f"    → {out_path}  ({written} rows, "
            f"{skipped_empty} skipped empty, {skipped_dup} dup query_id)",
            flush=True,
        )

    manifest_path = cache_dir.parent / "drop_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    print("Done.", flush=True)
    if export_jsonl:
        print(f"Total exported rows: {total_exported}", flush=True)
    else:
        print(f"Dataset cached under: {cache_dir}", flush=True)
        print("Re-run with --export-jsonl to write cleaned JSONL.", flush=True)
    return 0


def main() -> int:
    from thesis.paths import DROP_HF_CACHE_DIR, DROP_JSONL_DIR

    ap = argparse.ArgumentParser(description="Download and clean ucinlp/drop")
    ap.add_argument("--cache-dir", type=Path, default=DROP_HF_CACHE_DIR)
    ap.add_argument("--jsonl-dir", type=Path, default=DROP_JSONL_DIR)
    ap.add_argument("--export-jsonl", action="store_true")
    ap.add_argument("--splits", nargs="+", default=None, help="train validation (default: all)")
    ap.add_argument("--max-rows-per-split", type=int, default=0, help="0 = all rows")
    ap.add_argument("--no-dedupe-query-id", action="store_true")
    args = ap.parse_args()

    return download_and_export(
        cache_dir=args.cache_dir.expanduser().resolve(),
        jsonl_dir=args.jsonl_dir.expanduser().resolve() if args.export_jsonl else None,
        splits=args.splits,
        export_jsonl=args.export_jsonl,
        max_rows_per_split=args.max_rows_per_split,
        dedupe_query_id=not args.no_dedupe_query_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())

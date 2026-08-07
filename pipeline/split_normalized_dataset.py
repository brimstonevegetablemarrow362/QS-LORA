#!/usr/bin/env python3
"""
Stratified train / validation / test split of normalized custom QA JSONL.

Reads rows grouped by `source` and splits each source separately so every corpus
appears in train, val, and test (when there are at least 3 rows for that source).

Default input:  custom/normalized/all.jsonl
Default output: custom/normalized/splits/{train,val,test}.jsonl

Run:
  python split_normalized_dataset.py
  python split_normalized_dataset.py --train 0.85 --val 0.075 --seed 123
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
# No default paths — callers pass --input and --out-dir (per-user run workspace).
DEFAULT_INPUT = None
DEFAULT_OUT_DIR = None


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
                raise ValueError(f"{path}:{i} JSON error: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_source_rows(
    items: list[dict[str, Any]],
    rng,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one source's rows into train / val / test with at least one val and one test when n >= 3."""
    n = len(items)
    shuffled = items[:]
    rng.shuffle(shuffled)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    if n == 0:
        return [], [], []
    if n == 1:
        return shuffled, [], []
    if n == 2:
        # Cannot populate train, val, and test; prefer train + val
        return [shuffled[0]], [shuffled[1]], []

    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * test_ratio))
    n_train = n - n_val - n_test
    while n_train < 1 and (n_val > 1 or n_test > 1):
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = 1, 1, 1

    a = 0
    b = n_train
    c = b + n_val
    return shuffled[a:b], shuffled[b:c], shuffled[c:]


def stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        src = (r.get("source") or "unknown").strip() or "unknown"
        by_source.setdefault(src, []).append(r)

    rng = random.Random(seed)
    train_all: list[dict[str, Any]] = []
    val_all: list[dict[str, Any]] = []
    test_all: list[dict[str, Any]] = []

    for src in sorted(by_source.keys()):
        t, v, te = split_source_rows(by_source[src], rng, train_ratio, val_ratio)
        train_all.extend(t)
        val_all.extend(v)
        test_all.extend(te)

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)
    return train_all, val_all, test_all


def count_by_source(rows: list[dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in rows:
        c[(r.get("source") or "unknown").strip() or "unknown"] += 1
    return c


def main() -> None:
    p = argparse.ArgumentParser(description="Stratified split of normalized all.jsonl by source.")
    p.add_argument("--input", type=Path, required=True, help="Path to all.jsonl")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for split files")
    p.add_argument("--train", type=float, default=0.8, dest="train_ratio", help="Train fraction per source")
    p.add_argument("--val", type=float, default=0.1, dest="val_ratio", help="Validation fraction per source")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible splits)")
    args = p.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise SystemExit("train and val ratios must be positive")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1")

    rows = load_jsonl(args.input)
    train_rows, val_rows, test_rows = stratified_split(
        rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    out_train = args.out_dir / "train.jsonl"
    out_val = args.out_dir / "val.jsonl"
    out_test = args.out_dir / "test.jsonl"
    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)
    write_jsonl(out_test, test_rows)

    print(f"Input:  {args.input} ({len(rows)} rows)")
    print(f"Wrote:  {out_train} ({len(train_rows)})")
    print(f"        {out_val} ({len(val_rows)})")
    print(f"        {out_test} ({len(test_rows)})")
    print("\nRows per source (train | val | test):")
    total_by_src = count_by_source(rows)
    train_by_src = count_by_source(train_rows)
    val_by_src = count_by_source(val_rows)
    test_by_src = count_by_source(test_rows)
    sources = sorted(total_by_src.keys())
    for s in sources:
        print(
            f"  {s:12}  {train_by_src[s]:5} | {val_by_src[s]:5} | {test_by_src[s]:5}"
        )

    warnings: list[str] = []
    for s in sources:
        n = total_by_src[s]
        if n >= 3 and (val_by_src[s] < 1 or test_by_src[s] < 1):
            warnings.append(f"{s}: expected val and test nonempty for n>=3")
        if n == 2 and test_by_src[s] < 1:
            warnings.append(f"{s}: only 2 rows — test split is empty for this source")
    if warnings:
        print("\nNote:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

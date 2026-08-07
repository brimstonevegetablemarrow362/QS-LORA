"""Row slicing and shard merge helpers for eval generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def slice_eval_rows(
    rows: list[dict[str, Any]],
    *,
    row_start: int = 0,
    row_end: int = 0,
    max_rows: int = 0,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Return (slice, row_start, row_end_exclusive, total_rows)."""
    total = len(rows)
    start = max(0, int(row_start))
    end = int(row_end) if int(row_end) > 0 else total
    end = min(end, total)
    if int(max_rows) > 0:
        end = min(end, start + int(max_rows))
    if start >= end:
        return [], start, end, total
    return rows[start:end], start, end, total


def shard_pred_path(pred_dir: Path, row_start: int, row_end: int) -> Path:
    return pred_dir / f"predictions.jsonl.shard{row_start:06d}_{row_end:06d}"


def is_sharded_run(row_start: int, row_end: int, total_rows: int) -> bool:
    return int(row_start) > 0 or (int(row_end) > 0 and int(row_end) < total_rows)


def resolve_pred_output_path(
    pred_dir: Path,
    *,
    row_start: int,
    row_end: int,
    total_rows: int,
) -> Path:
    if is_sharded_run(row_start, row_end, total_rows):
        return shard_pred_path(pred_dir, row_start, row_end)
    return pred_dir / "predictions.jsonl"


def merge_prediction_shards(pred_dir: Path, *, out_name: str = "predictions.jsonl") -> Path:
    """Concat shard files in row order into predictions.jsonl."""
    pred_dir = pred_dir.expanduser().resolve()
    shards = sorted(pred_dir.glob("predictions.jsonl.shard*"))
    if not shards:
        raise FileNotFoundError(f"No shard files under {pred_dir}")

    parsed: list[tuple[int, int, Path]] = []
    for p in shards:
        name = p.name.removeprefix("predictions.jsonl.shard")
        a, b = name.split("_", 1)
        parsed.append((int(a), int(b), p))
    parsed.sort(key=lambda x: x[0])

    # Verify contiguous coverage.
    expect = 0
    for start, end, p in parsed:
        if start != expect:
            raise ValueError(f"Shard gap: expected start {expect}, got {start} ({p.name})")
        expect = end

    out_path = pred_dir / out_name
    n_lines = 0
    with out_path.open("w", encoding="utf-8") as out_fp:
        for _start, _end, shard in parsed:
            with shard.open(encoding="utf-8") as in_fp:
                for line in in_fp:
                    if line.strip():
                        out_fp.write(line if line.endswith("\n") else line + "\n")
                        n_lines += 1

    meta = {
        "schema": "pred_shard_merge/v1",
        "pred_dir": str(pred_dir),
        "output": str(out_path),
        "n_shards": len(parsed),
        "n_lines": n_lines,
        "shards": [p.name for _, _, p in parsed],
        "row_end": expect,
    }
    (pred_dir / "predictions.shard_merge.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path

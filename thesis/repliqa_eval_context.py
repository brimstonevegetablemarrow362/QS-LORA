"""Join RepLiQA eval subset context onto prediction rows (by eval_id)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_index(eval_jsonl: Path) -> dict[str, dict[str, Any]]:
    path = eval_jsonl.expanduser().resolve()
    index: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: {e}") from e
        key = str(row.get("eval_id") or row.get("chunk_id") or "").strip()
        if key:
            index[key] = row
    return index


def enrich_rows_with_eval_context(
    rows: list[dict[str, Any]],
    eval_index: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Attach ``context`` (and missing gold) from eval subset. Returns (rows, n_merged)."""
    n_merged = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        key = str(r.get("eval_id") or r.get("chunk_id") or "").strip()
        ref = eval_index.get(key) if key else None
        if ref is None:
            out.append(r)
            continue
        if overwrite or not str(r.get("context") or "").strip():
            ctx = str(ref.get("context") or "").strip()
            if ctx:
                r["context"] = ctx
                n_merged += 1
        if not str(r.get("gold") or "").strip():
            g = str(ref.get("gold") or ref.get("answer") or "").strip()
            if g:
                r["gold"] = g
        out.append(r)
    return out, n_merged

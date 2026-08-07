"""
Build passage-only corpus for domain CPT (Stage 1).

**Full knowledge-base policy:** passages are collected from *all* provided QA JSONL
files (typically train + validation/dev). QA questions in the eval split remain
held out for LoRA training and final evaluation — only passage *text* is shared
with CPT, matching deploy-time access to the full document corpus.

Outputs under ``<out_dir>/``:
  passages_full.jsonl   — every unique passage (audit / provenance)
  passages_train.jsonl  — CPT gradient updates (default: all passages)
  passages_val.jsonl    — optional tiny slice for CPT training loss only
  split_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from thesis.prepare_repliqa_sft_splits import split_by_document


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def passage_id_for_row(row: dict[str, Any], *, dataset_tag: str) -> str | None:
    section_id = (row.get("section_id") or row.get("paragraph_id") or "").strip()
    if section_id:
        return section_id
    context = (row.get("context") or "").strip()
    if not context:
        return None
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
    title = (row.get("title") or "").strip()
    if title:
        slug = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        return f"{dataset_tag}_{slug}_{digest}"
    return f"{dataset_tag}_{digest}"


def dedupe_passages_from_qa_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_tag: str,
    min_context_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One passage per section_id; keep longest context."""
    by_section: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "input_qa_rows": len(rows),
        "skipped_short": 0,
        "skipped_missing_context": 0,
        "duplicate_section_rows": 0,
    }

    for row in rows:
        context = (row.get("context") or "").strip()
        if not context:
            stats["skipped_missing_context"] += 1
            continue
        if len(context) < min_context_chars:
            stats["skipped_short"] += 1
            continue
        section_id = passage_id_for_row(row, dataset_tag=dataset_tag)
        if not section_id:
            stats["skipped_missing_context"] += 1
            continue
        prev = by_section.get(section_id)
        if prev is None or len(context) > len(prev.get("context") or ""):
            if prev is not None:
                stats["duplicate_section_rows"] += 1
            by_section[section_id] = {**row, "section_id": section_id, "context": context}
        else:
            stats["duplicate_section_rows"] += 1

    passages: list[dict[str, Any]] = []
    for section_id, row in sorted(by_section.items()):
        text = (row.get("context") or "").strip()
        passages.append(
            {
                "section_id": section_id,
                "text": text + "\n",
                "source": row.get("source", dataset_tag),
                "char_count": len(text),
                "title": row.get("title"),
            }
        )

    stats["unique_passages"] = len(passages)
    return passages, stats


def prepare_qa_cpt_corpus(
    *,
    qa_jsonl_paths: list[Path],
    out_dir: Path,
    dataset_tag: str,
    cpt_monitor_val_ratio: float = 0.0,
    seed: int = 42,
    min_context_chars: int = 40,
    max_passages: int = 0,
) -> dict[str, Any]:
    if not qa_jsonl_paths:
        raise ValueError("At least one --qa-jsonl path is required")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in qa_jsonl_paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        all_rows.extend(load_jsonl(path))
        sources.append(str(path))

    passages, dedupe_stats = dedupe_passages_from_qa_rows(
        all_rows,
        dataset_tag=dataset_tag,
        min_context_chars=min_context_chars,
    )
    if max_passages > 0:
        passages = passages[:max_passages]

    for p in passages:
        p["document_id"] = p["section_id"]

    full_path = out_dir / "passages_full.jsonl"
    write_jsonl(full_path, passages)

    if cpt_monitor_val_ratio > 0 and len(passages) > 1:
        train_rows, val_rows, split_manifest = split_by_document(
            passages,
            val_ratio=cpt_monitor_val_ratio,
            seed=seed,
        )
    else:
        train_rows = passages
        val_rows = []
        split_manifest = {
            "schema": "repliqa_sft_split_manifest/v1",
            "split_level": "document_id",
            "seed": seed,
            "val_ratio": 0.0,
            "note": "all passages in CPT train (full knowledge base)",
            "n_train_rows": len(train_rows),
            "n_val_rows": 0,
        }

    train_path = out_dir / "passages_train.jsonl"
    val_path = out_dir / "passages_val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    manifest: dict[str, Any] = {
        "schema": "qa_cpt_corpus/v1",
        "dataset_tag": dataset_tag,
        "cpt_passage_policy": "full_kb",
        "qa_jsonl_sources": sources,
        "out_dir": str(out_dir.resolve()),
        "cpt_monitor_val_ratio": cpt_monitor_val_ratio,
        "seed": seed,
        "min_context_chars": min_context_chars,
        "dedupe_stats": dedupe_stats,
        "n_passages_full_kb": len(passages),
        "n_cpt_train_passages": len(train_rows),
        "n_cpt_monitor_val_passages": len(val_rows),
        "split_manifest": split_manifest,
        "paths": {
            "passages_full_jsonl": str(full_path.resolve()),
            "passages_train_jsonl": str(train_path.resolve()),
            "passages_val_jsonl": str(val_path.resolve()),
        },
        "notes": [
            "All unique passages from listed QA JSONLs are included in the CPT knowledge base.",
            "QA questions in eval splits remain held out for Stage 2 LoRA / final eval.",
            "passages_val.jsonl is only for CPT training loss monitoring (optional).",
        ],
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_prepare(ns: argparse.Namespace) -> int:
    paths = [Path(p).expanduser().resolve() for p in ns.qa_jsonl]
    manifest = prepare_qa_cpt_corpus(
        qa_jsonl_paths=paths,
        out_dir=Path(ns.out_dir).expanduser().resolve(),
        dataset_tag=str(ns.dataset_tag),
        cpt_monitor_val_ratio=float(ns.cpt_monitor_val_ratio),
        seed=int(ns.seed),
        min_context_chars=int(ns.min_context_chars),
        max_passages=int(ns.max_passages),
    )
    print(json.dumps(manifest, indent=2))
    print(
        f"Wrote CPT corpus: {manifest['n_passages_full_kb']} passages (full KB), "
        f"{manifest['n_cpt_train_passages']} in CPT train",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare full-KB passage corpus for domain CPT")
    p.add_argument(
        "--qa-jsonl",
        type=Path,
        action="append",
        required=True,
        help="QA JSONL(s); pass train + validation for full knowledge base",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dataset-tag", type=str, required=True, help="e.g. drop, squad_v2, quoref")
    p.add_argument(
        "--cpt-monitor-val-ratio",
        type=float,
        default=0.0,
        help="Fraction of passages held out only for CPT eval loss (0 = all passages in CPT train)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-context-chars", type=int, default=40)
    p.add_argument("--max-passages", type=int, default=0, help="0 = all unique passages")
    return p


if __name__ == "__main__":
    raise SystemExit(run_prepare(build_arg_parser().parse_args()))

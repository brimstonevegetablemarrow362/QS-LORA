"""
Chunk unique RepLiQA documents from one split into overlapping windows.

Each chunk is written as a passage row for ``generate-drop-synthetic``:
  section_id = "{document_id}::c{idx:02d}"
  context    = chunk text

Usage (from finetuning/):
  python -m thesis.cli chunk-repliqa-split \\
    --split repliqa_1 \\
    --out thesis/experiments/repliqa/runs/repliqa_split1_chunk_gen_ab/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.generate_qa_repliqa import dedupe_documents, load_jsonl
from thesis.paths import REPLIQA_JSONL_DIR


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def chunk_document(
    text: str,
    *,
    target_words: int,
    overlap_words: int,
    min_words: int,
) -> list[str]:
    words = split_words(text)
    if len(words) <= target_words:
        return [text.strip()] if len(words) >= min_words else []

    step = max(1, target_words - overlap_words)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target_words)
        piece = " ".join(words[start:end]).strip()
        if len(split_words(piece)) >= min_words:
            chunks.append(piece)
        if end >= len(words):
            break
        start += step
        # Avoid tiny trailing duplicate when last window almost covered
        if start < len(words) and len(words) - start < min_words:
            break
    return chunks


def run_chunk_repliqa_split(ns: argparse.Namespace) -> int:
    split = str(ns.split).strip()
    if not split.startswith("repliqa_"):
        split = f"repliqa_{split}" if split.isdigit() else split

    jsonl_dir = Path(ns.jsonl_dir).expanduser().resolve()
    src = jsonl_dir / f"{split}.jsonl"
    if not src.is_file():
        print(f"Missing source: {src}")
        return 1

    out = Path(ns.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    docs_out = out.parent / "documents_unique.jsonl"
    summary_out = out.parent / "chunk_summary.json"

    docs, dedupe_stats = dedupe_documents([src], min_context_chars=int(ns.min_context_chars))
    if int(ns.max_documents) > 0:
        docs = docs[: int(ns.max_documents)]

    target = int(ns.target_words)
    overlap = int(ns.overlap_words)
    min_w = int(ns.min_chunk_words)

    chunk_rows: list[dict[str, Any]] = []
    n_chunks_per_doc: list[int] = []

    with docs_out.open("w", encoding="utf-8") as fp_docs:
        for doc in docs:
            fp_docs.write(json.dumps(doc, ensure_ascii=False) + "\n")
            parts = chunk_document(
                doc["context"],
                target_words=target,
                overlap_words=overlap,
                min_words=min_w,
            )
            n_chunks_per_doc.append(len(parts))
            for i, ctx in enumerate(parts):
                section_id = f"{doc['document_id']}::c{i:02d}"
                chunk_rows.append(
                    {
                        "section_id": section_id,
                        "document_id": doc["document_id"],
                        "chunk_index": i,
                        "n_chunks_in_doc": len(parts),
                        "context": ctx,
                        "document_topic": doc.get("document_topic"),
                        "repliqa_split": split,
                        "source": f"repliqa/chunk/{split}",
                    }
                )

    with out.open("w", encoding="utf-8") as fp:
        for row in chunk_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema": "repliqa_chunk_summary/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source_jsonl": str(src),
        "documents_unique": str(docs_out),
        "chunks_jsonl": str(out),
        "settings": {
            "target_words": target,
            "overlap_words": overlap,
            "min_chunk_words": min_w,
            "min_context_chars": int(ns.min_context_chars),
            "max_documents": int(ns.max_documents),
        },
        "dedupe_stats": dedupe_stats,
        "n_documents": len(docs),
        "n_chunks": len(chunk_rows),
        "mean_chunks_per_doc": round(sum(n_chunks_per_doc) / max(len(n_chunks_per_doc), 1), 3),
        "min_chunks_per_doc": min(n_chunks_per_doc) if n_chunks_per_doc else 0,
        "max_chunks_per_doc": max(n_chunks_per_doc) if n_chunks_per_doc else 0,
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Documents: {len(docs)} → chunks: {len(chunk_rows)}", flush=True)
    print(f"Mean chunks/doc: {summary['mean_chunks_per_doc']}", flush=True)
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {docs_out}", flush=True)
    print(f"Wrote {summary_out}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunk one RepLiQA split into overlapping windows")
    p.add_argument("--split", type=str, default="repliqa_1")
    p.add_argument("--jsonl-dir", type=Path, default=REPLIQA_JSONL_DIR)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("thesis/experiments/repliqa/runs/repliqa_split1_chunk_gen_ab/chunks.jsonl"),
    )
    p.add_argument("--target-words", type=int, default=350)
    p.add_argument("--overlap-words", type=int, default=50)
    p.add_argument("--min-chunk-words", type=int, default=80)
    p.add_argument("--min-context-chars", type=int, default=40)
    p.add_argument("--max-documents", type=int, default=0)
    return p


if __name__ == "__main__":
    raise SystemExit(run_chunk_repliqa_split(build_arg_parser().parse_args()))

#!/usr/bin/env python3
"""
Chunk plain Markdown files for Q/A generation (no Docling / PDF deps).

Splits on # headings, then packs text into chunks up to --max-chars.
Output JSONL lines: chunk_id, title, text, source

Usage:
  python chunk_markdown.py --input doc.md --out chunks.jsonl
  python chunk_markdown.py --input ./mydocs/ --out chunks.jsonl --source-basename
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def split_md_sections(md: str) -> list[tuple[str, str]]:
    """Return list of (section_title, body) for each # heading; preamble uses INTRO."""
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = "INTRO"
    body: list[str] = []
    sections: list[tuple[str, str]] = []

    heading_re = re.compile(r"^#{1,6}\s+")

    for line in lines:
        if heading_re.match(line):
            if body:
                sections.append((title, "\n".join(body).strip()))
            title = line.strip()
            body = []
        else:
            body.append(line)
    if body:
        sections.append((title, "\n".join(body).strip()))
    return [(t, b) for t, b in sections if b]


def pack_chunks(
    section_title: str,
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    """Split long section bodies into <= max_chars pieces with optional overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_file(
    path: Path,
    max_chars: int,
    overlap: int,
    source_tag: str,
) -> list[dict[str, Any]]:
    md = path.read_text(encoding="utf-8", errors="replace")
    out: list[dict[str, Any]] = []
    idx = 0
    for sec_title, body in split_md_sections(md):
        for piece in pack_chunks(sec_title, body, max_chars, overlap):
            cid = f"{source_tag}::{idx}"
            out.append(
                {
                    "chunk_id": cid,
                    "title": sec_title,
                    "text": piece,
                    "source": source_tag,
                }
            )
            idx += 1
    return out


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.md")) + sorted(path.glob("*.markdown"))
        return [f for f in files if f.is_file()]
    raise FileNotFoundError(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Chunk markdown for domain Q/A pipeline")
    p.add_argument("--input", type=Path, required=True, help="Markdown file or directory")
    p.add_argument("--out", type=Path, required=True, help="Output chunks.jsonl")
    p.add_argument("--max-chars", type=int, default=6000, help="Max characters per chunk")
    p.add_argument("--overlap", type=int, default=200, help="Char overlap between chunks")
    p.add_argument(
        "--source-basename",
        action="store_true",
        help="Use file stem as `source` field instead of 'markdown'",
    )
    args = p.parse_args()

    paths = collect_inputs(args.input)
    if not paths:
        raise SystemExit(f"No .md files under {args.input}")

    all_chunks: list[dict[str, Any]] = []
    for fp in paths:
        tag = fp.stem if args.source_basename else "markdown"
        all_chunks.extend(chunk_file(fp, args.max_chars, args.overlap, tag))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

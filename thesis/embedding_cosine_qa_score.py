"""
Semantic similarity scoring for QA JSONL using **embedding cosine similarity**.

Unlike NLI (logical entailment), this measures whether two texts live in a similar
**meaning neighborhood** in the model's embedding space. It is cheap (local model, no API).

Logic
-----
1. Map each text to a fixed-size vector (embedding) with a sentence embedding model.
2. L2-normalize vectors so cosine similarity equals the dot product:
      cos(a, b) = (a · b) / (||a|| ||b||)  →  dot(â, b̂) when ||â|| = ||b̂|| = 1
3. Compare:
   - **context_vs_answer**: chunk vs answer — topical alignment / “about the same thing”.
   - **cq_vs_answer**: (context + question) vs answer — closer to “answer addresses this
     question given this passage” without entailment’s strictness.
   - **question_vs_answer**: question vs answer — relevance of the answer to the question
     (weak grounding signal; ignores whether the chunk supports it).

Interpretation
--------------
- High cosine: similar topics/wording (good if the retrieved chunk is correct).
- Low cosine: off-topic answer, wrong chunk, or a paraphrase far from the chunk surface form.
- **Does not detect contradiction** — use alongside NLI or manual checks.

Dependencies: ``pip install sentence-transformers`` (uses PyTorch).

Default pair for CLI: **cq_vs_answer** (context + question vs answer).

Run: python -m thesis.cli qa-embed-cosine --qa-jsonl ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _pair_texts(row: dict[str, Any], mode: str) -> tuple[str, str]:
    ctx = str(row.get("context") or "").strip()
    q = str(row.get("question") or "").strip()
    ans = str(row.get("answer") or "").strip()
    if mode == "context_vs_answer":
        return ctx, ans
    if mode == "cq_vs_answer":
        left = ctx
        if q:
            left = f"{ctx}\n\nQuestion: {q}"
        return left, ans
    if mode == "question_vs_answer":
        return q, ans
    raise ValueError(f"unknown mode: {mode}")


def run_qa_embed_cosine(ns: argparse.Namespace) -> int:
    qa_path = Path(ns.qa_jsonl).expanduser().resolve()
    if not qa_path.is_file():
        print(f"Not found: {qa_path}", file=sys.stderr)
        return 1

    out_jsonl = (
        Path(ns.out_jsonl).expanduser().resolve()
        if ns.out_jsonl
        else qa_path.parent / f"{qa_path.stem}_embed.jsonl"
    )
    sum_path = (
        Path(ns.summary_json).expanduser().resolve()
        if ns.summary_json
        else out_jsonl.with_name(out_jsonl.stem + "_summary.json")
    )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency: sentence-transformers\n"
            "  pip install sentence-transformers",
            file=sys.stderr,
        )
        return 1

    device = ns.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {ns.model} on {device} …", flush=True)
    model = SentenceTransformer(ns.model, device=device)

    mode = ns.pair_mode
    max_chars = int(ns.max_context_chars)

    rows_out: list[dict[str, Any]] = []
    left_texts: list[str] = []
    right_texts: list[str] = []
    fill_indices: list[int] = []
    n_parse_errors = 0

    lines = qa_path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            n_parse_errors += 1
            rows_out.append({"_line": line_no, "_parse_error": str(e)})
            continue

        left, right = _pair_texts(row, mode)
        if max_chars > 0 and len(left) > max_chars:
            left = left[:max_chars]

        pos = len(rows_out)
        if not left.strip() or not right.strip():
            rows_out.append(
                {
                    **row,
                    "embed_cosine": {
                        "error": "missing_left_or_right_text",
                        "pair_mode": mode,
                        "cosine_similarity": None,
                    },
                }
            )
            continue

        fill_indices.append(pos)
        left_texts.append(left)
        right_texts.append(right)
        rows_out.append({**row})

    stats = {
        "n_rows": len(rows_out),
        "sum_cos": 0.0,
        "count_low_cosine": 0,
        "cos_threshold": float(ns.low_threshold),
    }
    low_thr = float(ns.low_threshold)

    import numpy as np

    if left_texts:
        le = model.encode(
            left_texts,
            batch_size=int(ns.batch_size),
            show_progress_bar=len(left_texts) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        re = model.encode(
            right_texts,
            batch_size=int(ns.batch_size),
            show_progress_bar=len(right_texts) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        cos_arr = np.sum(le * re, axis=1)

        for i, idx in enumerate(fill_indices):
            c = float(cos_arr[i])
            row = rows_out[idx]
            rows_out[idx] = {
                **row,
                "embed_cosine": {
                    "cosine_similarity": round(c, 6),
                    "pair_mode": mode,
                    "embed_model": ns.model,
                    "max_context_chars": max_chars if max_chars > 0 else None,
                },
            }
            stats["sum_cos"] += c
            if c < low_thr:
                stats["count_low_cosine"] += 1

    n_missing_pair = 0
    for r in rows_out:
        ec = r.get("embed_cosine") if isinstance(r, dict) else None
        if isinstance(ec, dict) and ec.get("error"):
            n_missing_pair += 1

    n_scored = len(fill_indices)
    stats["n_parse_errors"] = n_parse_errors
    stats["n_missing_pair"] = n_missing_pair
    stats["n_scored"] = n_scored

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for r in rows_out:
            fp.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = {
        "schema": "qa_embed_cosine_summary/v1",
        "model": ns.model,
        "qa_jsonl": str(qa_path),
        "out_jsonl": str(out_jsonl),
        "device": str(device),
        "settings": {
            "pair_mode": mode,
            "max_context_chars": max_chars if max_chars > 0 else None,
            "batch_size": int(ns.batch_size),
            "low_threshold": low_thr,
        },
        "stats": {
            **{k: v for k, v in stats.items() if k not in ("sum_cos",)},
            "mean_cosine_similarity": (
                round(stats["sum_cos"] / n_scored, 6) if n_scored else None
            ),
            "sum_cos": stats["sum_cos"],
        },
        "notes": [
            "Cosine is computed on L2-normalized embeddings (dot product).",
            "High score ≈ similar semantics in embedding space; not entailment or factuality.",
            "pair_mode=context_vs_answer | cq_vs_answer | question_vs_answer — see module docstring.",
        ],
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_jsonl}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(
        f"Rows: {stats['n_rows']}  mean cosine={summary['stats']['mean_cosine_similarity']}  "
        f"cosine<{low_thr}: {stats['count_low_cosine']}",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score QA JSONL with embedding cosine similarity.")
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL)
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument(
        "--pair-mode",
        type=str,
        default="cq_vs_answer",
        choices=("context_vs_answer", "cq_vs_answer", "question_vs_answer"),
        help="Left vs right texts to embed (default: cq_vs_answer = context+question vs answer).",
    )
    p.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Truncate left text to this many chars (0 = no truncation). Models still cap tokens internally.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--low-threshold",
        type=float,
        default=0.35,
        help="Count rows with cosine below this for quick QA filtering.",
    )
    return p


if __name__ == "__main__":
    ns = build_arg_parser().parse_args()
    raise SystemExit(run_qa_embed_cosine(ns))

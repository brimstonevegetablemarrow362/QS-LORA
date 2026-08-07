"""
Build a fixed RepLiQA eval subset (default: 2000 Q/A = 400 docs × 5 questions).

Stratified across repliqa_0 … repliqa_3 (100 documents per split by default).
Same rows for every model at inference time.

Usage (from finetuning/):
  python -m thesis.cli prepare-repliqa-eval-subset
  python -m thesis.cli prepare-repliqa-eval-subset --docs-per-split 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from thesis.paths import DEFAULT_TRAIN_SPLITS, REPLIQA_JSONL_DIR


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                raise ValueError(f"{path}:{i} {e}") from e
    return rows


def load_train_document_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    ids: set[str] = set()
    for row in load_jsonl(path):
        did = (row.get("document_id") or "").strip()
        if did:
            ids.add(did)
    return ids


def index_by_split_and_document(
    jsonl_dir: Path,
    splits: list[str],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """split -> document_id -> list of human Q/A rows (same context)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {sp: defaultdict(list) for sp in splits}
    for sp in splits:
        path = jsonl_dir / f"{sp}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in load_jsonl(path):
            did = (row.get("document_id") or "").strip()
            if not did:
                continue
            out[sp][did].append(row)
    return out


def sample_document_ids(
    doc_ids: list[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    if n >= len(doc_ids):
        return sorted(doc_ids)
    return sorted(rng.sample(doc_ids, n))


def row_to_eval_record(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    gold = (row.get("answer") or "").strip()
    return {
        "eval_id": (row.get("chunk_id") or "").strip() or f"{row.get('document_id')}-q",
        "document_id": row.get("document_id"),
        "chunk_id": row.get("chunk_id"),
        "repliqa_split": split,
        "document_topic": row.get("document_topic"),
        "context": row.get("context") or "",
        "question": row.get("question") or "",
        "answer": gold,
        "gold": gold,
        "long_answer": row.get("long_answer"),
        "source": row.get("source") or f"repliqa/{split}",
    }


def run_prepare_repliqa_eval_subset(ns: argparse.Namespace) -> int:
    jsonl_dir = Path(ns.jsonl_dir).expanduser().resolve()
    run_root = Path(ns.run_root).expanduser().resolve()
    eval_dir = Path(ns.eval_dir).expanduser().resolve() if ns.eval_dir else run_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    splits = list(ns.splits)
    docs_per_split = int(ns.docs_per_split)
    seed = int(ns.seed)
    rng = random.Random(seed)

    by_split = index_by_split_and_document(jsonl_dir, splits)
    train_doc_path = (
        Path(ns.train_documents_jsonl).expanduser().resolve()
        if ns.train_documents_jsonl
        else run_root / "train/documents_unique.jsonl"
    )
    train_doc_ids = load_train_document_ids(train_doc_path)

    selected_docs: dict[str, list[str]] = {}
    eval_rows: list[dict[str, Any]] = []
    per_split_stats: dict[str, Any] = {}

    for sp in splits:
        docs_map = by_split[sp]
        all_ids = sorted(docs_map.keys())
        pool_ids = all_ids
        if ns.exclude_train_documents and train_doc_ids:
            pool_ids = [d for d in all_ids if d not in train_doc_ids]

        chosen = sample_document_ids(pool_ids, docs_per_split, rng)
        selected_docs[sp] = chosen
        n_q = 0
        q_per_doc_counts: list[int] = []
        for did in chosen:
            rows = docs_map[did]
            q_per_doc_counts.append(len(rows))
            for row in rows:
                eval_rows.append(row_to_eval_record(row, split=sp))
                n_q += 1

        per_split_stats[sp] = {
            "documents_in_split": len(all_ids),
            "documents_sampled": len(chosen),
            "questions_written": n_q,
            "questions_per_doc_min": min(q_per_doc_counts) if q_per_doc_counts else 0,
            "questions_per_doc_max": max(q_per_doc_counts) if q_per_doc_counts else 0,
            "pool_after_exclude_train": len(pool_ids),
        }

    out_jsonl = eval_dir / ns.output_name
    manifest_path = eval_dir / ns.manifest_name

    with open(out_jsonl, "w", encoding="utf-8") as fp:
        for rec in eval_rows:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    overlap_train = sum(
        1 for sp, ids in selected_docs.items() for d in ids if d in train_doc_ids
    )
    topic_counts = Counter(r.get("document_topic") or "unknown" for r in eval_rows)

    manifest = {
        "schema": "repliqa_eval_subset_manifest/v1",
        "created_utc": _utc_iso(),
        "jsonl_dir": str(jsonl_dir),
        "splits": splits,
        "sampling": {
            "seed": seed,
            "docs_per_split": docs_per_split,
            "expected_questions": docs_per_split * len(splits) * int(ns.questions_per_doc),
            "questions_per_doc_assumed": int(ns.questions_per_doc),
            "exclude_train_documents": bool(ns.exclude_train_documents),
        },
        "outputs": {
            "eval_subset_jsonl": str(out_jsonl),
            "manifest_json": str(manifest_path),
        },
        "train_documents_jsonl": str(train_doc_path) if train_doc_path.is_file() else None,
        "n_train_document_ids": len(train_doc_ids),
        "n_eval_questions": len(eval_rows),
        "n_eval_documents": sum(len(v) for v in selected_docs.values()),
        "selected_document_ids_by_split": selected_docs,
        "per_split": per_split_stats,
        "document_topic_counts": dict(sorted(topic_counts.items(), key=lambda x: -x[1])),
        "overlap_selected_docs_with_synthetic_train": overlap_train,
        "notes": [
            "All models must run inference on this exact JSONL (same eval_id order).",
            "Default: 100 docs × 5 Q × 4 splits = 2000 questions.",
            "Primary eval regime: RepLiQA splits 0–3 (in-distribution for user-document pipeline).",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_jsonl} ({len(eval_rows)} questions)", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    for sp in splits:
        st = per_split_stats[sp]
        print(
            f"  {sp}: {st['documents_sampled']} docs, {st['questions_written']} questions",
            flush=True,
        )
    if train_doc_ids:
        print(
            f"  Overlap with synthetic-train document_ids: {overlap_train} / {manifest['n_eval_documents']}",
            flush=True,
        )
    return 0 if len(eval_rows) > 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
    p = argparse.ArgumentParser(description="Sample fixed RepLiQA eval subset (default 2000 Q/A).")
    p.add_argument("--jsonl-dir", type=Path, default=REPLIQA_JSONL_DIR)
    p.add_argument("--run-root", type=Path, default=run_root)
    p.add_argument("--eval-dir", type=Path, default=None)
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_TRAIN_SPLITS),
        help="Default: repliqa_0 … repliqa_3",
    )
    p.add_argument(
        "--docs-per-split",
        type=int,
        default=100,
        help="Documents sampled per split (100 × 4 splits × 5 Q = 2000).",
    )
    p.add_argument("--questions-per-doc", type=int, default=5, help="Documentary only; all Q kept.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--exclude-train-documents",
        action="store_true",
        help="Exclude document_id present in train/documents_unique.jsonl.",
    )
    p.add_argument(
        "--train-documents-jsonl",
        type=Path,
        default=None,
        help="Default: <run-root>/train/documents_unique.jsonl",
    )
    p.add_argument("--output-name", type=str, default="eval_subset_2000.jsonl")
    p.add_argument("--manifest-name", type=str, default="eval_subset_manifest.json")
    return p


if __name__ == "__main__":
    raise SystemExit(run_prepare_repliqa_eval_subset(build_arg_parser().parse_args()))

"""
Build train/val JSONL for RepLiQA synthetic SFT (document-level split).

B3: all usable Q/A rows. B4: add ``--quality-tier high`` on Haiku judge JSONL.

Outputs under <run_dir>/splits/sft/:
  train.jsonl, val.jsonl, split_manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from thesis.qa_judge_common import is_nan_answer


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


def is_usable_row(row: dict[str, Any]) -> bool:
    if not (row.get("context") or "").strip():
        return False
    if not (row.get("question") or "").strip():
        return False
    ans = str(row.get("answer") or "").strip()
    if not ans or is_nan_answer(ans):
        return False
    return True


def document_id_for(row: dict[str, Any]) -> str:
    did = (row.get("document_id") or "").strip()
    if did:
        return did
    cid = (row.get("chunk_id") or "").strip()
    if "::" in cid:
        return cid.split("::", 1)[0]
    return cid or "unknown"


def split_by_document(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_doc[document_id_for(r)].append(r)

    doc_ids = sorted(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    n_docs = len(doc_ids)
    if val_ratio <= 0 or n_docs <= 1:
        n_val_docs = 0
    else:
        n_val_docs = min(n_docs - 1, max(1, round(n_docs * val_ratio)))
    val_doc_set = set(doc_ids[:n_val_docs])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for doc_id, doc_rows in by_doc.items():
        if doc_id in val_doc_set:
            val_rows.extend(doc_rows)
        else:
            train_rows.extend(doc_rows)

    manifest = {
        "schema": "repliqa_sft_split_manifest/v1",
        "split_level": "document_id",
        "seed": seed,
        "val_ratio": val_ratio,
        "n_documents": n_docs,
        "n_val_documents": len(val_doc_set),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "val_document_ids": sorted(val_doc_set),
    }
    return train_rows, val_rows, manifest


def parse_quality_tiers(quality_tier: str) -> set[str]:
    """Parse ``high``, ``high,medium``, or ``high+medium`` into a set of tiers."""
    raw = quality_tier.strip().lower().replace("+", ",")
    tiers = {t.strip() for t in raw.split(",") if t.strip()}
    return tiers


def filter_by_quality_tier(
    rows: list[dict[str, Any]],
    quality_tier: str,
) -> tuple[list[dict[str, Any]], int]:
    """Keep rows whose ``llm_judge.quality_tier`` is in the allowed set (e.g. high or high,medium)."""
    allowed = parse_quality_tiers(quality_tier)
    kept: list[dict[str, Any]] = []
    n_no_judge = 0
    for r in rows:
        judge = r.get("llm_judge") if isinstance(r.get("llm_judge"), dict) else {}
        row_tier = (judge.get("quality_tier") or "").strip().lower()
        if row_tier in allowed:
            kept.append(r)
        elif not row_tier:
            n_no_judge += 1
    return kept, n_no_judge


def prepare_splits(
    *,
    qa_jsonl: Path,
    out_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    quality_tier: str | None = None,
) -> dict[str, Any]:
    qa_jsonl = qa_jsonl.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    if not qa_jsonl.is_file():
        raise FileNotFoundError(qa_jsonl)

    all_rows = load_jsonl(qa_jsonl)
    if quality_tier:
        tier_rows, n_no_judge = filter_by_quality_tier(all_rows, quality_tier)
    else:
        tier_rows, n_no_judge = all_rows, 0

    usable = [r for r in tier_rows if is_usable_row(r)]
    train_rows, val_rows, manifest = split_by_document(usable, val_ratio=val_ratio, seed=seed)

    manifest["qa_jsonl"] = str(qa_jsonl)
    manifest["n_input_rows"] = len(all_rows)
    manifest["n_after_quality_tier"] = len(tier_rows)
    manifest["n_skipped_unusable"] = len(tier_rows) - len(usable)
    manifest["n_skipped_no_judge_tier"] = n_no_judge
    if quality_tier:
        manifest["filter_quality_tier"] = quality_tier
        sample_judge = next(
            (r.get("llm_judge") for r in usable if isinstance(r.get("llm_judge"), dict)),
            {},
        )
        if isinstance(sample_judge, dict):
            manifest["judge_prompt_version"] = sample_judge.get("prompt_version")
            manifest["judge_model"] = sample_judge.get("model")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    manifest_path = out_dir / "split_manifest.json"

    with train_path.open("w", encoding="utf-8") as fp:
        for r in train_rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as fp:
        for r in val_rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest["train_jsonl"] = str(train_path)
    manifest["val_jsonl"] = str(val_path)
    manifest["manifest_json"] = str(manifest_path)
    return manifest


def run_prepare(ns: argparse.Namespace) -> int:
    manifest = prepare_splits(
        qa_jsonl=Path(ns.qa_jsonl),
        out_dir=Path(ns.out_dir),
        val_ratio=float(ns.val_ratio),
        seed=int(ns.seed),
        quality_tier=ns.quality_tier,
    )
    print(f"Input rows: {manifest['n_input_rows']}", flush=True)
    if manifest.get("filter_quality_tier"):
        print(f"After tier={manifest['filter_quality_tier']}: {manifest['n_after_quality_tier']}", flush=True)
    print(f"Skipped unusable: {manifest['n_skipped_unusable']}", flush=True)
    print(f"Train: {manifest['n_train_rows']} rows ({manifest['n_documents'] - manifest['n_val_documents']} docs)", flush=True)
    print(f"Val:   {manifest['n_val_rows']} rows ({manifest['n_val_documents']} docs)", flush=True)
    print(f"Wrote {manifest['train_jsonl']}", flush=True)
    print(f"Wrote {manifest['val_jsonl']}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Document-level train/val split for RepLiQA synthetic SFT.")
    p.add_argument("--qa-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quality-tier",
        type=str,
        default=None,
        help="Filter tiers: high | medium | high,medium | high+medium (requires judge JSONL with llm_judge).",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(run_prepare(build_arg_parser().parse_args()))

"""
Export judge-filter SFT splits: RepLiQA train/val, OhioLine OOD test.

Baseline distillation setup:
  train + val  ← RepLiQA synthetic_qa_haiku_judge.jsonl (v2, document-level val)
  test         ← OhioLine bedrock_judge.jsonl (v2, full file)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from thesis.judge_filter_sft import (
    DEFAULT_MAX_CONTEXT_CHARS,
    is_v2_training_judge_row,
    load_jsonl,
    to_sft_row,
    write_jsonl,
)
from thesis.prepare_repliqa_sft_splits import document_id_for, split_by_document


def _tier_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c[str(r.get("teacher_quality_tier") or "?")] += 1
    return dict(sorted(c.items()))


def prepare_judge_filter_sft(
    *,
    train_judged_jsonl: Path,
    test_judged_jsonl: Path | None = None,
    out_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    extra_judged_jsonl: Path | None = None,
    extra_label: str = "ohioline",
    extra_judged_jsonls: list[tuple[Path, str]] | None = None,
    ood_extra_label: str = "ohioline",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    extras: list[tuple[Path, str]] = list(extra_judged_jsonls or [])
    if extra_judged_jsonl is not None:
        extras.append((extra_judged_jsonl, extra_label))

    raw_primary = load_jsonl(train_judged_jsonl)
    usable_primary = [r for r in raw_primary if is_v2_training_judge_row(r)]

    primary_train, primary_val, primary_manifest = split_by_document(
        usable_primary,
        val_ratio=val_ratio,
        seed=seed,
    )
    train_rows = list(primary_train)
    val_rows = list(primary_val)

    extra_manifests: list[dict[str, Any]] = []
    ood_test_rows: list[dict[str, Any]] = []
    for idx, (extra_path, label) in enumerate(extras):
        raw_extra = load_jsonl(extra_path)
        usable_extra = [r for r in raw_extra if is_v2_training_judge_row(r)]
        extra_train, extra_val, split_manifest = split_by_document(
            usable_extra,
            val_ratio=val_ratio,
            seed=seed + 1 + idx,
        )
        train_rows.extend(extra_train)
        val_rows.extend(extra_val)
        if label == ood_extra_label and test_judged_jsonl is None:
            ood_test_rows = extra_val
        extra_manifests.append(
            {
                **split_manifest,
                "label": label,
                "judged_jsonl": str(extra_path.resolve()),
                "n_raw_rows": len(raw_extra),
                "n_usable_rows": len(usable_extra),
                "ood_test": label == ood_extra_label and test_judged_jsonl is None,
            }
        )

    if test_judged_jsonl is not None:
        raw_test = load_jsonl(test_judged_jsonl)
        ood_test_rows = [r for r in raw_test if is_v2_training_judge_row(r)]

    train_sft = [to_sft_row(r, max_context_chars=max_context_chars) for r in train_rows]
    val_sft = [to_sft_row(r, max_context_chars=max_context_chars) for r in val_rows]
    test_sft = [to_sft_row(r, max_context_chars=max_context_chars) for r in ood_test_rows]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test_ohioline.jsonl"
    write_jsonl(train_path, train_sft)
    write_jsonl(val_path, val_sft)
    write_jsonl(test_path, test_sft)

    schema = "judge_filter_sft_split/v1"
    if extras:
        schema = "judge_filter_sft_split/v3" if len(extras) > 1 else "judge_filter_sft_split/v2"

    manifest: dict[str, Any] = {
        "schema": schema,
        "prompt_version": "qa_judge_rubric/v2",
        "train_judged_jsonl": str(train_judged_jsonl.resolve()),
        "test_judged_jsonl": str(test_judged_jsonl.resolve()) if test_judged_jsonl else None,
        "extra_judged_jsonls": [
            {"path": str(p.resolve()), "label": lbl} for p, lbl in extras
        ],
        "ood_extra_label": ood_extra_label if extras and test_judged_jsonl is None else None,
        "out_dir": str(out_dir.resolve()),
        "val_ratio": val_ratio,
        "seed": seed,
        "max_context_chars": max_context_chars,
        "n_raw_primary_rows": len(raw_primary),
        "n_usable_primary_rows": len(usable_primary),
        "n_train_rows": len(train_sft),
        "n_val_rows": len(val_sft),
        "n_test_rows": len(test_sft),
        "train_tier_counts": _tier_counts(train_sft),
        "val_tier_counts": _tier_counts(val_sft),
        "test_tier_counts": _tier_counts(test_sft),
        "primary_split_manifest": primary_manifest,
        "extra_split_manifests": extra_manifests,
        "ood_test_source": (
            "extra_val_holdout"
            if extras and test_judged_jsonl is None
            else "fixed_test_judged_jsonl"
            if test_judged_jsonl
            else "none"
        ),
        "paths": {
            "train_jsonl": str(train_path.resolve()),
            "val_jsonl": str(val_path.resolve()),
            "test_ohioline_jsonl": str(test_path.resolve()),
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_prepare(ns: argparse.Namespace) -> int:
    extra_paths = [Path(p).expanduser().resolve() for p in (ns.extra_judged_jsonls or [])]
    extra_labels = list(ns.extra_labels or [])
    while len(extra_labels) < len(extra_paths):
        extra_labels.append(f"extra_{len(extra_labels)}")

    extras = list(zip(extra_paths, extra_labels))
    legacy_extra = Path(ns.extra_judged_jsonl).expanduser().resolve() if ns.extra_judged_jsonl else None
    if legacy_extra:
        extras.append((legacy_extra, str(ns.extra_label)))

    test_path = (
        None
        if extras
        else (Path(ns.test_judged_jsonl).expanduser().resolve() if ns.test_judged_jsonl else None)
    )
    manifest = prepare_judge_filter_sft(
        train_judged_jsonl=Path(ns.train_judged_jsonl).expanduser().resolve(),
        test_judged_jsonl=test_path,
        out_dir=Path(ns.out_dir).expanduser().resolve(),
        val_ratio=float(ns.val_ratio),
        seed=int(ns.seed),
        max_context_chars=int(ns.max_context_chars),
        extra_judged_jsonls=extras or None,
        ood_extra_label=str(ns.ood_extra_label),
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote splits under {manifest['out_dir']}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare judge-filter SFT splits (RepLiQA train, OhioLine test)")
    p.add_argument("--train-judged-jsonl", type=Path, required=True)
    p.add_argument(
        "--test-judged-jsonl",
        type=Path,
        default=None,
        help="Fixed OOD test file (v1: full OhioLine). Omit when using --extra-judged-jsonl.",
    )
    p.add_argument(
        "--extra-judged-jsonl",
        type=Path,
        default=None,
        help="Single extra judged JSONL (legacy). Prefer --extra-judged-jsonls.",
    )
    p.add_argument(
        "--extra-judged-jsonls",
        type=Path,
        nargs="*",
        default=[],
        help="Additional judged JSONL files merged into train/val.",
    )
    p.add_argument(
        "--extra-labels",
        type=str,
        nargs="*",
        default=[],
        help="Labels parallel to --extra-judged-jsonls.",
    )
    p.add_argument(
        "--ood-extra-label",
        type=str,
        default="ohioline",
        help="Which extra source's val holdout becomes OOD test (default: ohioline).",
    )
    p.add_argument("--extra-label", type=str, default="ohioline")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    return p


if __name__ == "__main__":
    raise SystemExit(run_prepare(build_arg_parser().parse_args()))

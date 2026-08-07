"""
Batch Bedrock Haiku judge on DROP validation predictions.

External evaluator: pred vs human gold + context (rubric v3_eval_gold).
Same pipeline as RepLiQA; defaults point at drop_qa_v1 + validation.jsonl.

  source thesis/scripts/source_bedrock_env.sh
  python -m thesis.cli eval-drop-bedrock-judge --max-rows 50
"""

from __future__ import annotations

from thesis.eval_repliqa_bedrock_judge import (
    build_arg_parser,
    run_eval_repliqa_bedrock_judge,
)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from thesis.paths import DROP_JSONL_DIR, DROP_RUN_QA

    p = build_arg_parser()
    ns = p.parse_args()
    if ns.run_root is None:
        ns.run_root = DROP_RUN_QA
    if ns.eval_jsonl is None:
        ns.eval_jsonl = DROP_JSONL_DIR / "validation.jsonl"
    raise SystemExit(run_eval_repliqa_bedrock_judge(ns))

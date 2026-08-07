"""Paths for thesis / RepLiQA experiments (under finetuning/thesis/)."""

from __future__ import annotations

from pathlib import Path

FINETUNING_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = Path(__file__).resolve().parent

# Shared RepLiQA data (HF export + cache)
REPLIQA_DATA_DIR = FINETUNING_ROOT / "data" / "repliqa"
REPLIQA_JSONL_DIR = REPLIQA_DATA_DIR / "jsonl"
REPLIQA_HF_CACHE_DIR = REPLIQA_DATA_DIR / "hf_cache"

# Thesis experiment outputs
REPLIQA_EXP_ROOT = THESIS_ROOT / "experiments" / "repliqa"
SPLITS_DIR = THESIS_ROOT / "splits"

# Split policy (document-level holdout for val/test within 0–3; 4 = OOD generalization)
DEFAULT_TRAIN_SPLITS = ("repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3")
REPLIQA_TEST_SPLIT = "repliqa_4"

# DROP (ucinlp/drop)
DROP_DATA_DIR = FINETUNING_ROOT / "data" / "drop"
DROP_JSONL_DIR = DROP_DATA_DIR / "jsonl"
DROP_HF_CACHE_DIR = DROP_DATA_DIR / "hf_cache"
DROP_EXP_ROOT = THESIS_ROOT / "experiments" / "drop"
DROP_RUN_CPT = DROP_EXP_ROOT / "runs" / "drop_cpt_v1"
DROP_SYN_RUN = DROP_EXP_ROOT / "runs" / "drop_synthetic_full_v1"
DROP_RUN_QA = DROP_EXP_ROOT / "runs" / "drop_qa_v1"
DROP_TRAIN_SPLIT = "train"
DROP_VAL_SPLIT = "validation"

# SQuAD 2.0 (rajpurkar/squad_v2)
SQUAD_DATA_DIR = FINETUNING_ROOT / "data" / "squad_v2"
SQUAD_JSONL_DIR = SQUAD_DATA_DIR / "jsonl"
SQUAD_HF_CACHE_DIR = SQUAD_DATA_DIR / "hf_cache"
SQUAD_EXP_ROOT = THESIS_ROOT / "experiments" / "squad_v2"
SQUAD_RUN_CPT = SQUAD_EXP_ROOT / "runs" / "squad_cpt_v1"
SQUAD_SYN_RUN = SQUAD_EXP_ROOT / "runs" / "squad_synthetic_deploy_v1"
SQUAD_RUN_QA = SQUAD_EXP_ROOT / "runs" / "squad_qa_v1"
SQUAD_RUN_QA_NCTX = SQUAD_EXP_ROOT / "runs" / "squad_qa_v1_nctx"
SQUAD_VAL_SPLIT = "validation"

# Quoref (allenai/quoref)
QUOREF_DATA_DIR = FINETUNING_ROOT / "data" / "quoref"
QUOREF_JSONL_DIR = QUOREF_DATA_DIR / "jsonl"
QUOREF_HF_CACHE_DIR = QUOREF_DATA_DIR / "hf_cache"
QUOREF_EXP_ROOT = THESIS_ROOT / "experiments" / "quoref"
QUOREF_RUN_CPT = QUOREF_EXP_ROOT / "runs" / "quoref_cpt_v1"
QUOREF_SYN_RUN = QUOREF_EXP_ROOT / "runs" / "quoref_synthetic_deploy_v1"
QUOREF_RUN_QA = QUOREF_EXP_ROOT / "runs" / "quoref_qa_v1"
QUOREF_RUN_QA_NCTX = QUOREF_EXP_ROOT / "runs" / "quoref_qa_v1_nctx"
QUOREF_VAL_SPLIT = "validation"

# Ohioline (Ohio State Extension fact sheets)
OHIO_LINE_DATA_DIR = FINETUNING_ROOT / "data" / "ohioline"
OHIO_LINE_JSONL_DIR = OHIO_LINE_DATA_DIR / "jsonl"
OHIO_LINE_EXP_ROOT = THESIS_ROOT / "experiments" / "ohioline"

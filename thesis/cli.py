#!/usr/bin/env python3
"""
Thesis / RepLiQA experiment CLI (benchmarks, synthetic Q/A, quality scores).

Run from finetuning/:
  python -m thesis.cli download-repliqa --export-jsonl
  python -m thesis.cli generate-repliqa-synthetic --max-documents 50
  python -m thesis.cli qa-nli-score --qa-jsonl ./path/to/synthetic_qa.jsonl
  python -m thesis.cli qa-embed-cosine --qa-jsonl ./path/to/synthetic_qa.jsonl
  python -m thesis.cli qa-context-gap-vllm --qa-jsonl ./data/repliqa/jsonl/repliqa_1.jsonl
  python -m thesis.cli qa-llm-judge --qa-jsonl ./path/to/synthetic_qa.jsonl --max-rows 200
  python -m thesis.cli qa-haiku-judge --qa-jsonl ./path/to/synthetic_qa.jsonl
  python -m thesis.cli qa-bedrock-judge --predictions-jsonl ./eval/predictions/B3/predictions.jsonl --answer-field pred
  python -m thesis.cli eval-repliqa-score --write-scored
  python -m thesis.cli train-repliqa-lora --qa-jsonl .../synthetic_qa.jsonl --output-dir .../baselines/B3_all_lora_r16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_FINETUNING = Path(__file__).resolve().parent.parent
if str(_FINETUNING) not in sys.path:
    sys.path.insert(0, str(_FINETUNING))

import thesis.bootstrap  # noqa: F401

from paths import (  # domain_v1
    DEFAULT_BASE_MODEL_ID,
    QA_GEN_CONCURRENCY,
    QA_GEN_MAX_NEW_TOKENS,
    QA_GEN_TEMPERATURE,
    generator_vllm_base_url,
    generator_vllm_model_id,
)
from thesis.paths import (
    DEFAULT_TRAIN_SPLITS,
    DROP_EXP_ROOT,
    DROP_HF_CACHE_DIR,
    DROP_JSONL_DIR,
    DROP_RUN_CPT,
    OHIO_LINE_JSONL_DIR,
    QUOREF_HF_CACHE_DIR,
    QUOREF_JSONL_DIR,
    QUOREF_RUN_CPT,
    REPLIQA_EXP_ROOT,
    REPLIQA_HF_CACHE_DIR,
    REPLIQA_JSONL_DIR,
    SQUAD_HF_CACHE_DIR,
    SQUAD_JSONL_DIR,
    SQUAD_RUN_CPT,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Thesis / RepLiQA experiment tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("download-repliqa", help="Download RepLiQA HF dataset + optional JSONL export")
    p_rep.add_argument("--cache-dir", type=Path, default=None)
    p_rep.add_argument("--jsonl-dir", type=Path, default=None)
    p_rep.add_argument("--export-jsonl", action="store_true")
    p_rep.add_argument("--splits", nargs="+", default=None)
    p_rep.add_argument("--drop-unanswerable", action="store_true")
    p_rep.add_argument("--max-rows-per-split", type=int, default=0)

    def _run_download(ns: argparse.Namespace) -> int:
        from thesis.download_repliqa import download_and_export

        return download_and_export(
            cache_dir=(ns.cache_dir or REPLIQA_HF_CACHE_DIR).expanduser().resolve(),
            jsonl_dir=(ns.jsonl_dir or REPLIQA_JSONL_DIR).expanduser().resolve() if ns.export_jsonl else None,
            splits=ns.splits,
            export_jsonl=ns.export_jsonl,
            drop_unanswerable=ns.drop_unanswerable,
            max_rows_per_split=ns.max_rows_per_split,
        )

    p_rep.set_defaults(fn=_run_download)

    p_drop = sub.add_parser("download-drop", help="Download ucinlp/drop + export cleaned JSONL")
    p_drop.add_argument("--cache-dir", type=Path, default=None)
    p_drop.add_argument("--jsonl-dir", type=Path, default=None)
    p_drop.add_argument("--export-jsonl", action="store_true")
    p_drop.add_argument("--splits", nargs="+", default=None)
    p_drop.add_argument("--max-rows-per-split", type=int, default=0)
    p_drop.add_argument("--no-dedupe-query-id", action="store_true")

    def _run_download_drop(ns: argparse.Namespace) -> int:
        from thesis.download_drop import download_and_export

        return download_and_export(
            cache_dir=(ns.cache_dir or DROP_HF_CACHE_DIR).expanduser().resolve(),
            jsonl_dir=(ns.jsonl_dir or DROP_JSONL_DIR).expanduser().resolve()
            if ns.export_jsonl
            else None,
            splits=ns.splits,
            export_jsonl=ns.export_jsonl,
            max_rows_per_split=ns.max_rows_per_split,
            dedupe_query_id=not ns.no_dedupe_query_id,
        )

    p_drop.set_defaults(fn=_run_download_drop)

    p_squad = sub.add_parser("download-squad", help="Download rajpurkar/squad_v2 + export cleaned JSONL")
    p_squad.add_argument("--cache-dir", type=Path, default=None)
    p_squad.add_argument("--jsonl-dir", type=Path, default=None)
    p_squad.add_argument("--export-jsonl", action="store_true")
    p_squad.add_argument("--splits", nargs="+", default=None)
    p_squad.add_argument("--max-rows-per-split", type=int, default=0)
    p_squad.add_argument("--no-dedupe-eval-id", action="store_true")

    def _run_download_squad(ns: argparse.Namespace) -> int:
        from thesis.download_squad import download_and_export

        return download_and_export(
            cache_dir=(ns.cache_dir or SQUAD_HF_CACHE_DIR).expanduser().resolve(),
            jsonl_dir=(ns.jsonl_dir or SQUAD_JSONL_DIR).expanduser().resolve()
            if ns.export_jsonl
            else None,
            splits=ns.splits,
            export_jsonl=ns.export_jsonl,
            max_rows_per_split=ns.max_rows_per_split,
            dedupe_eval_id=not ns.no_dedupe_eval_id,
        )

    p_squad.set_defaults(fn=_run_download_squad)

    p_quoref = sub.add_parser("download-quoref", help="Download allenai/quoref + export cleaned JSONL")
    p_quoref.add_argument("--cache-dir", type=Path, default=None)
    p_quoref.add_argument("--jsonl-dir", type=Path, default=None)
    p_quoref.add_argument("--export-jsonl", action="store_true")
    p_quoref.add_argument("--splits", nargs="+", default=None)
    p_quoref.add_argument("--max-rows-per-split", type=int, default=0)
    p_quoref.add_argument("--no-dedupe-eval-id", action="store_true")

    def _run_download_quoref(ns: argparse.Namespace) -> int:
        from thesis.download_quoref import download_and_export

        return download_and_export(
            cache_dir=(ns.cache_dir or QUOREF_HF_CACHE_DIR).expanduser().resolve(),
            jsonl_dir=(ns.jsonl_dir or QUOREF_JSONL_DIR).expanduser().resolve()
            if ns.export_jsonl
            else None,
            splits=ns.splits,
            export_jsonl=ns.export_jsonl,
            max_rows_per_split=ns.max_rows_per_split,
            dedupe_eval_id=not ns.no_dedupe_eval_id,
        )

    p_quoref.set_defaults(fn=_run_download_quoref)

    def _add_cpt_prep_args(p: argparse.ArgumentParser, *, default_out: Path) -> None:
        p.add_argument("--out-dir", type=Path, default=default_out)
        p.add_argument("--cpt-monitor-val-ratio", type=float, default=0.0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--min-context-chars", type=int, default=40)
        p.add_argument("--max-passages", type=int, default=0)
        p.add_argument("--qa-jsonl", type=Path, action="append", default=None)

    def _run_qa_cpt_prep(ns: argparse.Namespace, *, dataset_tag: str, default_paths: list[Path]) -> int:
        from thesis.prepare_qa_cpt_corpus import prepare_qa_cpt_corpus

        paths = [Path(p).expanduser().resolve() for p in (ns.qa_jsonl or default_paths)]
        manifest = prepare_qa_cpt_corpus(
            qa_jsonl_paths=paths,
            out_dir=Path(ns.out_dir).expanduser().resolve(),
            dataset_tag=dataset_tag,
            cpt_monitor_val_ratio=float(ns.cpt_monitor_val_ratio),
            seed=int(ns.seed),
            min_context_chars=int(ns.min_context_chars),
            max_passages=int(ns.max_passages),
        )
        print(__import__("json").dumps(manifest, indent=2))
        print(f"Wrote CPT corpus under {manifest['out_dir']}", flush=True)
        return 0

    p_cpt_prep = sub.add_parser(
        "prepare-qa-cpt-corpus",
        help="Passages from train+eval QA JSONLs → full-KB CPT corpus",
    )
    p_cpt_prep.add_argument("--dataset-tag", type=str, required=True)
    _add_cpt_prep_args(p_cpt_prep, default_out=DROP_RUN_CPT / "cpt_corpus")

    def _run_generic_cpt_prep(ns: argparse.Namespace) -> int:
        from thesis.prepare_qa_cpt_corpus import run_prepare

        if not ns.qa_jsonl:
            print("prepare-qa-cpt-corpus requires at least one --qa-jsonl", file=sys.stderr)
            return 1
        return run_prepare(ns)

    p_cpt_prep.set_defaults(fn=_run_generic_cpt_prep)

    p_dcpt_prep = sub.add_parser(
        "prepare-drop-cpt-corpus",
        help="DROP train+validation passages → full-KB CPT corpus",
    )
    p_dcpt_prep.add_argument("--train-jsonl", type=Path, default=DROP_JSONL_DIR / "train.jsonl")
    p_dcpt_prep.add_argument("--eval-jsonl", type=Path, default=None)
    p_dcpt_prep.add_argument("--no-eval-jsonl", action="store_true")
    p_dcpt_prep.add_argument("--out-dir", type=Path, default=DROP_RUN_CPT / "cpt_corpus")
    p_dcpt_prep.add_argument("--cpt-monitor-val-ratio", type=float, default=0.0)
    p_dcpt_prep.add_argument("--val-ratio", type=float, default=None, help=argparse.SUPPRESS)
    p_dcpt_prep.add_argument("--seed", type=int, default=42)
    p_dcpt_prep.add_argument("--min-context-chars", type=int, default=40)
    p_dcpt_prep.add_argument("--max-passages", type=int, default=0)

    def _run_dcpt_prep(ns: argparse.Namespace) -> int:
        from thesis.prepare_drop_cpt_corpus import run_prepare

        return run_prepare(ns)

    p_dcpt_prep.set_defaults(fn=_run_dcpt_prep)

    p_scpt_prep = sub.add_parser(
        "prepare-squad-cpt-corpus",
        help="SQuAD 2.0 train+validation passages → full-KB CPT corpus",
    )
    _add_cpt_prep_args(p_scpt_prep, default_out=SQUAD_RUN_CPT / "cpt_corpus")

    def _run_scpt_prep(ns: argparse.Namespace) -> int:
        defaults = [SQUAD_JSONL_DIR / "train.jsonl", SQUAD_JSONL_DIR / "validation.jsonl"]
        return _run_qa_cpt_prep(ns, dataset_tag="squad_v2", default_paths=defaults)

    p_scpt_prep.set_defaults(fn=_run_scpt_prep)

    p_qcpt_prep = sub.add_parser(
        "prepare-quoref-cpt-corpus",
        help="Quoref train+validation passages → full-KB CPT corpus",
    )
    _add_cpt_prep_args(p_qcpt_prep, default_out=QUOREF_RUN_CPT / "cpt_corpus")

    def _run_qcpt_prep(ns: argparse.Namespace) -> int:
        defaults = [QUOREF_JSONL_DIR / "train.jsonl", QUOREF_JSONL_DIR / "validation.jsonl"]
        return _run_qa_cpt_prep(ns, dataset_tag="quoref", default_paths=defaults)

    p_qcpt_prep.set_defaults(fn=_run_qcpt_prep)

    p_dcpt_tr = sub.add_parser(
        "train-drop-cpt",
        help="Stage 1: LoRA domain CPT on DROP passages (next-token prediction)",
    )
    p_dcpt_tr.add_argument("--corpus-dir", type=Path, default=DROP_RUN_CPT / "cpt_corpus")
    p_dcpt_tr.add_argument("--output-dir", type=Path, default=DROP_RUN_CPT / "cpt_lora")
    p_dcpt_tr.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_dcpt_tr.add_argument("--lora-r", type=int, default=16)
    p_dcpt_tr.add_argument("--lora-alpha", type=int, default=32)
    p_dcpt_tr.add_argument("--lora-dropout", type=float, default=0.05)
    p_dcpt_tr.add_argument("--epochs", type=int, default=1)
    p_dcpt_tr.add_argument("--lr", type=float, default=1e-5)
    p_dcpt_tr.add_argument("--max-seq-length", type=int, default=4096)
    p_dcpt_tr.add_argument("--batch-size", type=int, default=1)
    p_dcpt_tr.add_argument("--grad-accum", type=int, default=16)
    p_dcpt_tr.add_argument("--seed", type=int, default=42)
    p_dcpt_tr.add_argument("--no-bf16", action="store_true")
    p_dcpt_tr.add_argument("--use-qlora-4bit", action="store_true")

    def _run_dcpt_tr(ns: argparse.Namespace) -> int:
        from thesis.train_drop_cpt import run_train_drop_cpt

        return run_train_drop_cpt(ns)

    p_dcpt_tr.set_defaults(fn=_run_dcpt_tr)

    p_merge = sub.add_parser(
        "merge-lora-dense",
        help="Merge one LoRA adapter into a dense HF model (post-CPT base for Stage 2)",
    )
    p_merge.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_merge.add_argument("--adapter-dir", type=Path, required=True)
    p_merge.add_argument("--output-dir", type=Path, required=True)
    p_merge.add_argument("--no-bf16", action="store_true")

    def _run_merge_dense(ns: argparse.Namespace) -> int:
        from thesis.merge_lora_dense import run_merge

        return run_merge(ns)

    p_merge.set_defaults(fn=_run_merge_dense)

    p_ohio = sub.add_parser(
        "scrape-ohioline",
        help="Scrape Ag Crops, Farm Management, and Insects/Pests fact sheets from Ohioline",
    )
    p_ohio.add_argument("--jsonl-dir", type=Path, default=OHIO_LINE_JSONL_DIR)
    p_ohio.add_argument("--delay-sec", type=float, default=0.75)
    p_ohio.add_argument("--min-text-chars", type=int, default=80)
    p_ohio.add_argument("--max-factsheets", type=int, default=0)
    p_ohio.add_argument("--resume", action="store_true")
    p_ohio.add_argument("--force", action="store_true")

    def _run_scrape_ohioline(ns: argparse.Namespace) -> int:
        from thesis.scrape_ohioline import run_scrape

        return run_scrape(
            jsonl_dir=(ns.jsonl_dir or OHIO_LINE_JSONL_DIR).expanduser().resolve(),
            delay_sec=ns.delay_sec,
            min_text_chars=ns.min_text_chars,
            max_factsheets=ns.max_factsheets,
            resume=ns.resume,
            force=ns.force,
        )

    p_ohio.set_defaults(fn=_run_scrape_ohioline)

    p_ochunk = sub.add_parser(
        "chunk-ohioline",
        help="Table-aware chunking of passages.jsonl → chunks.jsonl",
    )
    p_ochunk.add_argument("--input", type=Path, default=OHIO_LINE_JSONL_DIR / "passages.jsonl")
    p_ochunk.add_argument("--output", type=Path, default=OHIO_LINE_JSONL_DIR / "chunks.jsonl")
    p_ochunk.add_argument("--max-chars", type=int, default=6000)
    p_ochunk.add_argument("--overlap", type=int, default=200)

    def _run_chunk_ohioline(ns: argparse.Namespace) -> int:
        from thesis.chunk_ohioline import run_chunk

        return run_chunk(
            input_jsonl=ns.input.expanduser().resolve(),
            output_jsonl=ns.output.expanduser().resolve(),
            max_chars=ns.max_chars,
            overlap=ns.overlap,
        )

    p_ochunk.set_defaults(fn=_run_chunk_ohioline)

    p_rchunk = sub.add_parser(
        "chunk-repliqa-split",
        help="Chunk unique RepLiQA documents from one split into overlapping windows",
    )
    p_rchunk.add_argument("--split", type=str, default="repliqa_1")
    p_rchunk.add_argument("--jsonl-dir", type=Path, default=REPLIQA_JSONL_DIR)
    p_rchunk.add_argument(
        "--out",
        type=Path,
        default=Path("thesis/experiments/repliqa/runs/repliqa_split1_chunk_gen_ab/chunks.jsonl"),
    )
    p_rchunk.add_argument("--target-words", type=int, default=350)
    p_rchunk.add_argument("--overlap-words", type=int, default=50)
    p_rchunk.add_argument("--min-chunk-words", type=int, default=80)
    p_rchunk.add_argument("--min-context-chars", type=int, default=40)
    p_rchunk.add_argument("--max-documents", type=int, default=0)

    def _run_chunk_repliqa(ns: argparse.Namespace) -> int:
        from thesis.chunk_repliqa_split import run_chunk_repliqa_split

        return run_chunk_repliqa_split(ns)

    p_rchunk.set_defaults(fn=_run_chunk_repliqa)

    p_dcov = sub.add_parser(
        "analyze-document-coverage",
        help="Window coverage of synthetic QAs over parent documents",
    )
    p_dcov.add_argument("--docs-jsonl", type=Path, required=True)
    p_dcov.add_argument("--qa-jsonl", type=Path, required=True)
    p_dcov.add_argument("--out-dir", type=Path, required=True)
    p_dcov.add_argument("--n-windows", type=int, default=5)
    p_dcov.add_argument("--overlap-threshold", type=float, default=0.12)
    p_dcov.add_argument(
        "--mode",
        choices=("lexical", "embedding"),
        default="lexical",
        help="lexical token overlap, or embedding cosine (MiniLM).",
    )
    p_dcov.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.35,
        help="Window covered if QA↔window cosine >= this (embedding mode).",
    )
    p_dcov.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    p_dcov.add_argument("--device", type=str, default="auto")
    p_dcov.add_argument("--batch-size", type=int, default=64)
    p_dcov.add_argument(
        "--exclude-tiers",
        nargs="+",
        default=None,
        help="Skip QA rows with these judge tiers (e.g. drop).",
    )

    def _run_dcov(ns: argparse.Namespace) -> int:
        from thesis.analyze_document_coverage import run_analyze_document_coverage

        return run_analyze_document_coverage(ns)

    p_dcov.set_defaults(fn=_run_dcov)

    p_gagen = sub.add_parser(
        "eval-general-ability-generate",
        help="Closed-book general-ability answers (base vs RepLiQA FT)",
    )
    p_gagen.add_argument("--model-path", type=str, required=True)
    p_gagen.add_argument("--condition-id", type=str, required=True)
    p_gagen.add_argument("--out-dir", type=Path, required=True)
    p_gagen.add_argument(
        "--questions-jsonl",
        type=Path,
        default=None,
        help="Default: experiments/analysis/general_ability/general_eval_questions_100.jsonl",
    )
    p_gagen.add_argument("--max-rows", type=int, default=0)
    p_gagen.add_argument("--max-new-tokens", type=int, default=512)
    p_gagen.add_argument("--bf16", action="store_true", default=True)
    p_gagen.add_argument("--no-bf16", action="store_true")

    def _run_gagen(ns: argparse.Namespace) -> int:
        from thesis.eval_general_ability import (
            DEFAULT_QUESTIONS,
            run_eval_general_ability_generate,
        )

        if ns.questions_jsonl is None:
            ns.questions_jsonl = DEFAULT_QUESTIONS
        if ns.no_bf16:
            ns.bf16 = False
        return run_eval_general_ability_generate(ns)

    p_gagen.set_defaults(fn=_run_gagen)

    p_dgen = sub.add_parser(
        "generate-drop-synthetic",
        help="Unique section_id → synthetic Q/A via vLLM (passage JSONL sources)",
    )
    p_dgen.add_argument("--jsonl", type=Path, action="append", default=None)
    p_dgen.add_argument("--extra-jsonl", type=Path, action="append", default=None)
    p_dgen.add_argument(
        "--passage-policy",
        type=str,
        default="deploy_full_kb",
        choices=("deploy_full_kb", "train_only"),
    )
    p_dgen.add_argument("--exp-root", type=Path, default=DROP_EXP_ROOT)
    p_dgen.add_argument("--run-name", type=str, default=None)
    p_dgen.add_argument("--pairs-per-passage", type=int, default=2)
    p_dgen.add_argument("--max-passages", type=int, default=0)
    p_dgen.add_argument("--min-context-chars", type=int, default=40)
    p_dgen.add_argument("--backend", choices=("vllm", "local", "bedrock"), default="vllm")
    p_dgen.add_argument("--vllm-base-url", type=str, default=generator_vllm_base_url())
    p_dgen.add_argument("--vllm-model", type=str, default=generator_vllm_model_id())
    p_dgen.add_argument("--model-path", type=Path, default=None)
    p_dgen.add_argument("--bedrock-model", type=str, default=None)
    p_dgen.add_argument("--region", type=str, default=None)
    p_dgen.add_argument("--concurrency", type=int, default=QA_GEN_CONCURRENCY)
    p_dgen.add_argument("--max-new-tokens", type=int, default=QA_GEN_MAX_NEW_TOKENS)
    p_dgen.add_argument("--max-seq-length", type=int, default=4096)
    p_dgen.add_argument("--temperature", type=float, default=QA_GEN_TEMPERATURE)
    p_dgen.add_argument("--bf16", action="store_true", default=True)
    p_dgen.add_argument("--no-bf16", action="store_true")
    p_dgen.add_argument("--source-tag", type=str, default="drop/synthetic/train")
    p_dgen.add_argument("--prompt-profile", choices=("drop", "default", "ohioline"), default="drop")
    p_dgen.add_argument("--list-prompts", action="store_true")
    p_dgen.add_argument("--no-length-sort", action="store_true")

    def _run_drop_gen(ns: argparse.Namespace) -> int:
        from thesis.generate_qa_drop import run_generate_drop
        from paths import MERGED_GENERATOR_DIR

        if not ns.jsonl:
            ns.jsonl = [DROP_JSONL_DIR / "train.jsonl"]
        if ns.model_path is None:
            ns.model_path = MERGED_GENERATOR_DIR
        if ns.no_bf16:
            ns.bf16 = False
        return run_generate_drop(ns)

    p_dgen.set_defaults(fn=_run_drop_gen)

    p_rgen = sub.add_parser(
        "generate-repliqa-synthetic",
        help="Unique document_id → synthetic Q/A via vLLM (default splits 0–3)",
    )
    p_rgen.add_argument("--jsonl-dir", type=Path, default=REPLIQA_JSONL_DIR)
    p_rgen.add_argument("--exp-root", type=Path, default=REPLIQA_EXP_ROOT)
    p_rgen.add_argument("--splits", nargs="+", default=list(DEFAULT_TRAIN_SPLITS))
    p_rgen.add_argument("--run-name", type=str, default=None)
    p_rgen.add_argument("--pairs-per-doc", type=int, default=1)
    p_rgen.add_argument("--max-documents", type=int, default=0)
    p_rgen.add_argument("--min-context-chars", type=int, default=40)
    p_rgen.add_argument("--vllm-base-url", type=str, default=generator_vllm_base_url())
    p_rgen.add_argument("--vllm-model", type=str, default=generator_vllm_model_id())
    p_rgen.add_argument("--concurrency", type=int, default=QA_GEN_CONCURRENCY)
    p_rgen.add_argument("--max-new-tokens", type=int, default=QA_GEN_MAX_NEW_TOKENS)
    p_rgen.add_argument("--temperature", type=float, default=QA_GEN_TEMPERATURE)
    p_rgen.add_argument("--source-tag", type=str, default="repliqa/synthetic/train")
    p_rgen.add_argument("--no-length-sort", action="store_true")

    def _run_gen(ns: argparse.Namespace) -> int:
        from thesis.generate_qa_repliqa import run_generate_repliqa

        return run_generate_repliqa(ns)

    p_rgen.set_defaults(fn=_run_gen)

    p_nli = sub.add_parser("qa-nli-score", help="NLI faithfulness: context vs answer")
    p_nli.add_argument("--qa-jsonl", type=Path, required=True)
    p_nli.add_argument("--out-jsonl", type=Path, default=None)
    p_nli.add_argument("--summary-json", type=Path, default=None)
    p_nli.add_argument("--model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli")
    p_nli.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p_nli.add_argument("--max-length", type=int, default=512)
    p_nli.add_argument("--stride", type=int, default=128)
    p_nli.add_argument("--max-hypothesis-tokens", type=int, default=384)
    p_nli.add_argument(
        "--aggregate",
        type=str,
        default="max_entail",
        choices=("max_entail", "mean_probs", "max_contradict"),
    )
    p_nli.add_argument(
        "--sliding-window",
        action="store_true",
        help="Fixed sliding windows; default out: <stem>_nli_sliding_window.jsonl",
    )

    def _run_nli(ns: argparse.Namespace) -> int:
        from thesis.nli_qa_score import run_qa_nli_score

        return run_qa_nli_score(ns)

    p_nli.set_defaults(fn=_run_nli)

    p_emb = sub.add_parser("qa-embed-cosine", help="Embedding cosine (default: context+question vs answer)")
    p_emb.add_argument("--qa-jsonl", type=Path, required=True)
    p_emb.add_argument("--out-jsonl", type=Path, default=None)
    p_emb.add_argument("--summary-json", type=Path, default=None)
    p_emb.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_emb.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p_emb.add_argument(
        "--pair-mode",
        type=str,
        default="cq_vs_answer",
        choices=("context_vs_answer", "cq_vs_answer", "question_vs_answer"),
    )
    p_emb.add_argument("--max-context-chars", type=int, default=12000)
    p_emb.add_argument("--batch-size", type=int, default=32)
    p_emb.add_argument("--low-threshold", type=float, default=0.35)

    def _run_emb(ns: argparse.Namespace) -> int:
        from thesis.embedding_cosine_qa_score import run_qa_embed_cosine

        return run_qa_embed_cosine(ns)

    p_emb.set_defaults(fn=_run_emb)

    p_gap = sub.add_parser("qa-context-gap-vllm", help="vLLM: answer without vs with context + embed gap")
    p_gap.add_argument("--qa-jsonl", type=Path, required=True)
    p_gap.add_argument("--out-jsonl", type=Path, default=None)
    p_gap.add_argument("--summary-json", type=Path, default=None)
    p_gap.add_argument("--vllm-base-url", type=str, default=generator_vllm_base_url())
    p_gap.add_argument("--vllm-model", type=str, default=DEFAULT_BASE_MODEL_ID)
    p_gap.add_argument("--max-new-tokens", type=int, default=QA_GEN_MAX_NEW_TOKENS)
    p_gap.add_argument("--temperature", type=float, default=0.0)
    p_gap.add_argument("--concurrency", type=int, default=QA_GEN_CONCURRENCY)
    p_gap.add_argument("--max-rows", type=int, default=0)
    p_gap.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_gap.add_argument("--embed-device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p_gap.add_argument("--embed-batch-size", type=int, default=32)

    def _run_gap(ns: argparse.Namespace) -> int:
        from thesis.qa_context_gap_vllm import run_qa_context_gap_vllm

        return run_qa_context_gap_vllm(ns)

    p_gap.set_defaults(fn=_run_gap)

    p_judge = sub.add_parser(
        "qa-llm-judge",
        help="OpenAI LLM-as-judge (grounding/relevance/overall 1-5); thesis only, needs OPENAI_API_KEY",
    )
    p_judge.add_argument("--qa-jsonl", type=Path, required=True)
    p_judge.add_argument("--out-jsonl", type=Path, default=None)
    p_judge.add_argument("--summary-json", type=Path, default=None)
    p_judge.add_argument("--provider", type=str, default="openai", choices=("openai",))
    p_judge.add_argument("--model", type=str, default="gpt-4o-mini")
    p_judge.add_argument("--max-rows", type=int, default=0)
    p_judge.add_argument("--max-context-chars", type=int, default=12000)
    p_judge.add_argument("--temperature", type=float, default=0.0)
    p_judge.add_argument("--concurrency", type=int, default=4)
    p_judge.add_argument("--request-delay-s", type=float, default=0.0)

    def _run_judge(ns: argparse.Namespace) -> int:
        from thesis.llm_judge_qa_score import run_qa_llm_judge

        return run_qa_llm_judge(ns)

    p_judge.set_defaults(fn=_run_judge)

    p_haiku = sub.add_parser(
        "qa-haiku-judge",
        help="Anthropic Haiku judge (Message Batches API, full corpus); needs ANTHROPIC_API_KEY",
    )
    p_haiku.add_argument("--qa-jsonl", type=Path, required=True)
    p_haiku.add_argument("--out-jsonl", type=Path, default=None)
    p_haiku.add_argument("--summary-json", type=Path, default=None)
    p_haiku.add_argument("--state-json", type=Path, default=None)
    p_haiku.add_argument("--model", type=str, default="claude-haiku-4-5")
    p_haiku.add_argument("--max-rows", type=int, default=0)
    p_haiku.add_argument("--max-context-chars", type=int, default=12000)
    p_haiku.add_argument("--max-tokens", type=int, default=512)
    p_haiku.add_argument("--temperature", type=float, default=0.0)
    p_haiku.add_argument("--batch-chunk-size", type=int, default=3000)
    p_haiku.add_argument("--poll-interval-s", type=float, default=30.0)
    p_haiku.add_argument("--submit-only", action="store_true")
    p_haiku.add_argument("--resume-only", action="store_true")

    def _run_haiku(ns: argparse.Namespace) -> int:
        from thesis.anthropic_judge_qa_score import run_qa_haiku_judge

        return run_qa_haiku_judge(ns)

    p_haiku.set_defaults(fn=_run_haiku)

    p_br = sub.add_parser(
        "qa-bedrock-judge",
        help="Claude Haiku via Amazon Bedrock (OSC → AWS); scores pred or answer field",
    )
    p_br.add_argument(
        "--predictions-jsonl",
        "--qa-jsonl",
        dest="predictions_jsonl",
        type=Path,
        required=True,
    )
    p_br.add_argument("--out-jsonl", type=Path, default=None)
    p_br.add_argument("--summary-json", type=Path, default=None)
    p_br.add_argument("--timing-json", type=Path, default=None)
    p_br.add_argument("--model", type=str, default=None)
    p_br.add_argument("--region", type=str, default=None)
    p_br.add_argument(
        "--answer-field",
        type=str,
        default="auto",
        choices=("auto", "pred", "answer"),
    )
    p_br.add_argument("--max-rows", type=int, default=0)
    p_br.add_argument("--max-context-chars", type=int, default=12000)
    p_br.add_argument("--max-tokens", type=int, default=512)
    p_br.add_argument("--temperature", type=float, default=0.0)
    p_br.add_argument("--concurrency", type=int, default=4)
    p_br.add_argument("--request-delay-s", type=float, default=0.0)
    p_br.add_argument("--dry-run", action="store_true")
    p_br.add_argument(
        "--eval-jsonl",
        type=Path,
        default=None,
        help="Merge context from eval subset by eval_id (required for predictions without context).",
    )
    p_br.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful rows from --out-jsonl; Bedrock only for failed/missing rows.",
    )
    p_br.add_argument("--force", action="store_true", help="Re-judge all rows (ignores --resume).")

    def _run_bedrock(ns: argparse.Namespace) -> int:
        from thesis.bedrock_judge_qa_score import run_qa_bedrock_judge

        return run_qa_bedrock_judge(ns)

    p_br.set_defaults(fn=_run_bedrock)

    p_ebj = sub.add_parser(
        "eval-repliqa-bedrock-judge",
        help="Batch Bedrock judge on all eval predictions (external pred vs gold evaluator)",
    )
    p_ebj.add_argument("--run-root", type=Path, default=None)
    p_ebj.add_argument("--predictions-dir", type=Path, default=None)
    p_ebj.add_argument("--predictions-index", type=Path, default=None)
    p_ebj.add_argument("--predictions-jsonl", type=Path, default=None)
    p_ebj.add_argument("--eval-jsonl", type=Path, default=None)
    p_ebj.add_argument("--eval-input-name", type=str, default="eval_subset_2000.jsonl")
    p_ebj.add_argument("--judged-dir", type=Path, default=None)
    p_ebj.add_argument("--model", type=str, default=None)
    p_ebj.add_argument("--region", type=str, default=None)
    p_ebj.add_argument("--max-rows", type=int, default=0)
    p_ebj.add_argument("--max-context-chars", type=int, default=12000)
    p_ebj.add_argument("--max-tokens", type=int, default=512)
    p_ebj.add_argument("--temperature", type=float, default=0.0)
    p_ebj.add_argument("--concurrency", type=int, default=4)
    p_ebj.add_argument("--request-delay-s", type=float, default=0.0)
    p_ebj.add_argument("--dry-run", action="store_true")
    p_ebj.add_argument("--skip-existing", action="store_true", default=True)
    p_ebj.add_argument("--force", action="store_true")
    p_ebj.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful judged rows; Bedrock only for failed/missing rows.",
    )
    p_ebj.add_argument(
        "--rank-by",
        type=str,
        default="mean_gold_alignment",
        choices=("mean_gold_alignment", "mean_overall", "mean_grounding"),
    )
    p_ebj.add_argument("--leaderboard-only", action="store_true")
    p_ebj.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Only judge these prediction subdir names.",
    )
    p_ebj.add_argument(
        "--position-swap-debias",
        action="store_true",
        help="Average gold-first and pred-first eval prompts (default output: judged_debias/).",
    )

    def _run_eval_bedrock_judge(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_bedrock_judge import run_eval_repliqa_bedrock_judge

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_repliqa_bedrock_judge(ns)

    p_ebj.set_defaults(fn=_run_eval_bedrock_judge)

    p_lwr = sub.add_parser(
        "eval-repliqa-listwise-rank",
        help="Bedrock listwise rank: all 8 preds per question; mean points (best=9, worst=1)",
    )
    p_lwr.add_argument("--run-root", type=Path, default=None)
    p_lwr.add_argument("--predictions-dir", type=Path, default=None)
    p_lwr.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Only rank these prediction subdirs (default: all under predictions/).",
    )
    p_lwr.add_argument("--eval-jsonl", type=Path, default=None)
    p_lwr.add_argument("--eval-input-name", type=str, default="eval_subset_2000.jsonl")
    p_lwr.add_argument("--output-dir", type=Path, default=None)
    p_lwr.add_argument("--model", type=str, default=None)
    p_lwr.add_argument("--region", type=str, default=None)
    p_lwr.add_argument("--max-rows", type=int, default=0)
    p_lwr.add_argument("--max-context-chars", type=int, default=8000)
    p_lwr.add_argument("--max-pred-chars", type=int, default=600)
    p_lwr.add_argument("--max-tokens", type=int, default=1024)
    p_lwr.add_argument("--temperature", type=float, default=0.0)
    p_lwr.add_argument("--concurrency", type=int, default=2)
    p_lwr.add_argument("--request-delay-s", type=float, default=0.1)
    p_lwr.add_argument("--seed", type=int, default=42)
    p_lwr.add_argument("--position-swap-debias", action="store_true")
    p_lwr.add_argument("--allow-partial-conditions", action="store_true")
    p_lwr.add_argument("--dry-run", action="store_true")
    p_lwr.add_argument("--leaderboard-only", action="store_true")

    def _run_listwise_rank(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_listwise_rank import run_eval_repliqa_listwise_rank

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        ns.require_all_conditions = not bool(ns.allow_partial_conditions)
        return run_eval_repliqa_listwise_rank(ns)

    p_lwr.set_defaults(fn=_run_listwise_rank)

    p_lww = sub.add_parser(
        "eval-repliqa-listwise-winrate",
        help="Head-to-head listwise win rates vs a baseline (using listwise_rank_results.jsonl)",
    )
    p_lww.add_argument("--run-root", type=Path, default=None)
    p_lww.add_argument("--results-jsonl", type=Path, default=None)
    p_lww.add_argument("--baseline", type=str, default="B3_lora_all")

    def _run_listwise_winrate(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_listwise_winrate import run_eval_repliqa_listwise_winrate

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_repliqa_listwise_winrate(ns)

    p_lww.set_defaults(fn=_run_listwise_winrate)

    p_slc = sub.add_parser(
        "eval-repliqa-slice-metrics",
        help="Metrics sliced by question_type, answer_evidence, finetuning_expected_gain, etc.",
    )
    p_slc.add_argument("--run-root", type=Path, default=None)
    p_slc.add_argument("--classified-jsonl", type=Path, default=None)
    p_slc.add_argument("--listwise-jsonl", type=Path, default=None)
    p_slc.add_argument("--metrics-dir", type=Path, default=None)
    p_slc.add_argument("--output-dir", type=Path, default=None)
    p_slc.add_argument("--baseline", type=str, default="B3_lora_all")
    p_slc.add_argument("--slice-field", action="append", default=None)

    def _run_slice_metrics(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_slice_metrics import run_eval_repliqa_slice_metrics

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_repliqa_slice_metrics(ns)

    p_slc.set_defaults(fn=_run_slice_metrics)

    p_res = sub.add_parser(
        "eval-repliqa-collect-resources",
        help="Aggregate train/generate/merge/judge wall times into eval/resource_timing.json",
    )
    p_res.add_argument("--run-root", type=Path, default=None)
    p_res.add_argument("--output-json", type=Path, default=None)

    def _run_collect_resources(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_collect_resources import run_eval_repliqa_collect_resources

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_repliqa_collect_resources(ns)

    p_res.set_defaults(fn=_run_collect_resources)

    p_xinf = sub.add_parser(
        "cross-model-collect-inference",
        help="Aggregate eval generate timing (mean s/question) into eval/inference_timing.json",
    )
    p_xinf.add_argument("--run-root", type=Path, default=None)
    p_xinf.add_argument(
        "--cross-root",
        type=Path,
        default=None,
        help="If set, aggregate all model×dataset runs under cross_model/runs.",
    )
    p_xinf.add_argument("--output-json", type=Path, default=None)

    def _run_cross_model_inference(ns: argparse.Namespace) -> int:
        from thesis.cross_model_inference_timing import run_collect_inference

        if ns.run_root is None and ns.cross_root is None:
            ns.cross_root = Path("/fs/ess/PAS2699/pratham2210/cross_model/runs")
        elif ns.run_root is not None:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_collect_inference(ns)

    p_xinf.set_defaults(fn=_run_cross_model_inference)

    p_mshard = sub.add_parser(
        "merge-pred-shards",
        help="Concat predictions.jsonl.shard* files into predictions.jsonl",
    )
    p_mshard.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory containing predictions.jsonl.shard* files",
    )
    p_mshard.add_argument("--output-name", type=str, default="predictions.jsonl")

    def _run_merge_shards(ns: argparse.Namespace) -> int:
        from thesis.eval_row_slice import merge_prediction_shards

        out = merge_prediction_shards(Path(ns.pred_dir), out_name=str(ns.output_name))
        print(f"Merged -> {out}", flush=True)
        return 0

    p_mshard.set_defaults(fn=_run_merge_shards)

    p_api_gen = sub.add_parser(
        "eval-api-generate",
        help="Generate eval answers via Bedrock ceilings (Opus + Nova 2 Lite)",
    )
    p_api_gen.add_argument("--run-root", type=Path, default=None)
    p_api_gen.add_argument("--eval-jsonl", type=Path, required=True)
    p_api_gen.add_argument("--eval-dir", type=Path, default=None)
    p_api_gen.add_argument("--output-dir", type=Path, default=None)
    p_api_gen.add_argument("--condition-id", type=str, default="REF_claude_opus")
    p_api_gen.add_argument(
        "--provider",
        type=str,
        choices=("bedrock",),
        default="bedrock",
    )
    p_api_gen.add_argument("--model", type=str, default=None)
    p_api_gen.add_argument("--region", type=str, default=None)
    p_api_gen.add_argument("--max-rows", type=int, default=0)
    p_api_gen.add_argument("--max-tokens", type=int, default=512)
    p_api_gen.add_argument("--temperature", type=float, default=0.0)
    p_api_gen.add_argument("--concurrency", type=int, default=4)
    p_api_gen.add_argument("--request-delay-s", type=float, default=0.05)
    p_api_gen.add_argument(
        "--nova-reasoning-effort",
        type=str,
        default="low",
        choices=("low", "medium", "high"),
    )
    p_api_gen.add_argument("--no-context", action="store_true")
    p_api_gen.add_argument("--dry-run", action="store_true")

    def _run_api_gen(ns: argparse.Namespace) -> int:
        from thesis.eval_api_generate import run_eval_api_generate

        if ns.run_root is not None:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_api_generate(ns)

    p_api_gen.set_defaults(fn=_run_api_gen)

    p_ceil = sub.add_parser(
        "eval-ceiling-gap",
        help="Report judged GA gap vs REF_claude_opus / REF_nova_2_lite ceilings",
    )
    p_ceil.add_argument("--run-root", type=Path, required=True)
    p_ceil.add_argument("--judged-dir", type=Path, default=None)
    p_ceil.add_argument("--ceiling-condition", type=str, default="REF_claude_opus")
    p_ceil.add_argument(
        "--all-ceilings",
        action="store_true",
        help="Report vs both REF_claude_opus and REF_nova_2_lite.",
    )
    p_ceil.add_argument("--baseline-condition", type=str, default=None)
    p_ceil.add_argument("--output-json", type=Path, default=None)

    def _run_ceil_gap(ns: argparse.Namespace) -> int:
        from thesis.eval_ceiling_gap import run_eval_ceiling_gap

        ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_ceiling_gap(ns)

    p_ceil.set_defaults(fn=_run_ceil_gap)

    p_boot = sub.add_parser(
        "eval-bootstrap-ci",
        help="Bootstrap 95% CIs for judge GA (paired + per-condition) and listwise win rates",
    )
    p_boot.add_argument("--all", dest="all_runs", action="store_true")
    p_boot.add_argument("--experiments-root", type=Path, default=None)
    p_boot.add_argument("--output-json", type=Path, default=None)
    p_boot.add_argument("--leaderboard", type=Path, default=None)
    p_boot.add_argument("--n-bootstrap", type=int, default=10_000)
    p_boot.add_argument("--ci-level", type=float, default=0.95)
    p_boot.add_argument("--seed", type=int, default=42)

    def _run_bootstrap_ci(ns: argparse.Namespace) -> int:
        from thesis.eval_bootstrap_ci import run_eval_bootstrap_ci

        if ns.experiments_root is None:
            ns.experiments_root = Path(__file__).resolve().parent / "experiments"
        else:
            ns.experiments_root = Path(ns.experiments_root).expanduser().resolve()
        return run_eval_bootstrap_ci(ns)

    p_boot.set_defaults(fn=_run_bootstrap_ci)

    from thesis.analyze_quality_tiers import add_cli as _add_tier_analysis_cli

    _add_tier_analysis_cli(sub)

    from thesis.analyze_adapter_effective_rank import add_cli as _add_adapter_rank_cli

    _add_adapter_rank_cli(sub)

    from thesis.export_triple_hallucination_catalog import add_cli as _add_triple_catalog_cli

    _add_triple_catalog_cli(sub)

    from thesis.export_triple_hallucination_showcase import add_cli as _add_triple_showcase_cli

    _add_triple_showcase_cli(sub)

    from thesis.export_pairwise_hallucination_showcase import add_cli as _add_pairwise_showcase_cli

    _add_pairwise_showcase_cli(sub)

    from thesis.export_judge_filter_holdouts import add_cli as _add_judge_holdouts_cli

    _add_judge_holdouts_cli(sub)

    from thesis.compile_training_resources_doc import add_cli as _add_training_resources_cli

    _add_training_resources_cli(sub)

    p_study = sub.add_parser(
        "eval-repliqa-export-study-sample",
        help="Export JSONL + Markdown: gold vs Ours_tier vs B3 LoRA vs B1 (default 100 questions)",
    )
    p_study.add_argument("--run-root", type=Path, default=None)
    p_study.add_argument("--n", type=int, default=100)
    p_study.add_argument("--seed", type=int, default=42)
    p_study.add_argument("--strategy", choices=("balanced", "random"), default="balanced")
    p_study.add_argument("--eval-jsonl", type=Path, default=None)
    p_study.add_argument("--predictions-dir", type=Path, default=None)
    p_study.add_argument("--output-dir", type=Path, default=None)

    def _run_export_study_sample(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_export_study_sample import run_eval_repliqa_export_study_sample

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_repliqa_export_study_sample(ns)

    p_study.set_defaults(fn=_run_export_study_sample)

    p_obl = sub.add_parser(
        "eval-repliqa-export-ours-beats-lora",
        help="JSON of all questions where Ours listwise rank beats B3 LoRA (+ question_type)",
    )
    p_obl.add_argument("--run-root", type=Path, default=None)
    p_obl.add_argument("--output-dir", type=Path, default=None)
    p_obl.add_argument("--output-name", type=str, default="ours_beats_lora.json")
    p_obl.add_argument("--ours-condition", type=str, default="Ours_tier_merge")
    p_obl.add_argument("--lora-condition", type=str, default="B3_lora_all")

    def _run_export_ours_beats_lora(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_export_study_sample import run_eval_repliqa_export_ours_beats_lora

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        return run_eval_repliqa_export_ours_beats_lora(ns)

    p_obl.set_defaults(fn=_run_export_ours_beats_lora)

    p_hall = sub.add_parser(
        "eval-export-hallucination-pack",
        help="Export unanswerable cases: Ours refuses/hedges, B3 invents answer (JSONL + MD)",
    )
    p_hall.add_argument("--run-root", type=Path, required=True)
    p_hall.add_argument("--b3-condition", type=str, default="B3_lora_ctx")
    p_hall.add_argument("--ours-condition", type=str, default="Ours_tier_ctx")
    p_hall.add_argument("--output-dir", type=Path, default=None)
    p_hall.add_argument("--max-examples", type=int, default=0)
    p_hall.add_argument("--from-predictions", action="store_true")
    p_hall.add_argument(
        "--mode",
        type=str,
        choices=("auto", "refusal", "judge_gap"),
        default="auto",
    )

    def _run_export_hallucination(ns: argparse.Namespace) -> int:
        from thesis.eval_export_hallucination_pack import run_eval_export_hallucination_pack

        return run_eval_export_hallucination_pack(ns)

    p_hall.set_defaults(fn=_run_export_hallucination)

    p_triple = sub.add_parser(
        "eval-export-triple-hallucination-pack",
        help="Export cases where Ours is correct but both B3 and B5 hallucinate",
    )
    p_triple.add_argument("--run-root", type=Path, required=True)
    p_triple.add_argument("--b3-condition", type=str, default="B3_lora_ctx")
    p_triple.add_argument("--b5-condition", type=str, default="B5_adalora_ctx")
    p_triple.add_argument("--ours-condition", type=str, default="Ours_tier_ctx")
    p_triple.add_argument("--output-dir", type=Path, default=None)
    p_triple.add_argument("--max-examples", type=int, default=25)
    p_triple.add_argument(
        "--mode",
        type=str,
        choices=("auto", "refusal", "judge_gap"),
        default="auto",
    )

    def _run_export_triple_hallucination(ns: argparse.Namespace) -> int:
        from thesis.eval_export_hallucination_pack import run_eval_export_triple_hallucination_pack

        return run_eval_export_triple_hallucination_pack(ns)

    p_triple.set_defaults(fn=_run_export_triple_hallucination)

    p_tr = sub.add_parser(
        "train-repliqa-lora",
        help="B3: fixed-rank LoRA on all usable synthetic Q/A (document-level val split)",
    )
    p_tr.add_argument("--baseline", type=str, default="B3", help="Baseline label (B3, B4, …) for experiment manifest.")
    p_tr.add_argument("--qa-jsonl", type=Path, required=True)
    p_tr.add_argument("--output-dir", type=Path, required=True)
    p_tr.add_argument("--splits-dir", type=Path, default=None)
    p_tr.add_argument("--skip-prepare", action="store_true")
    p_tr.add_argument("--val-ratio", type=float, default=0.1)
    p_tr.add_argument("--quality-tier", type=str, default=None)
    p_tr.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_tr.add_argument("--lora-r", type=int, default=16)
    p_tr.add_argument("--lora-alpha", type=int, default=32)
    p_tr.add_argument("--lora-dropout", type=float, default=0.05)
    p_tr.add_argument("--epochs", type=int, default=3)
    p_tr.add_argument("--lr", type=float, default=2e-4)
    p_tr.add_argument("--max-seq-length", type=int, default=4096)
    p_tr.add_argument("--batch-size", type=int, default=1)
    p_tr.add_argument("--grad-accum", type=int, default=8)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--no-bf16", action="store_true")
    p_tr.add_argument("--use-qlora-4bit", action="store_true")
    p_tr.add_argument(
        "--no-context",
        action="store_true",
        help="Train question-only -> answer (closed-book); matches no-ctx eval.",
    )
    p_tr.add_argument(
        "--peft-type",
        type=str,
        choices=("lora", "adalora"),
        default="lora",
        help="LoRA (fixed rank) or AdaLoRA (adaptive rank budget).",
    )
    p_tr.add_argument("--adalora-init-r", type=int, default=16)
    p_tr.add_argument("--adalora-target-r", type=int, default=16)
    p_tr.add_argument("--adalora-tinit-ratio", type=float, default=0.1)
    p_tr.add_argument("--adalora-tfinal-ratio", type=float, default=0.9)
    p_tr.add_argument("--adalora-delta-t", type=int, default=10)

    def _run_train_lora(ns: argparse.Namespace) -> int:
        from thesis.train_repliqa_lora import run_train_repliqa_lora

        return run_train_repliqa_lora(ns)

    p_tr.set_defaults(fn=_run_train_lora)

    p_sp = sub.add_parser(
        "prepare-repliqa-sft-splits",
        help="Document-level train/val JSONL from synthetic_qa.jsonl (skips nan)",
    )
    p_sp.add_argument("--qa-jsonl", type=Path, required=True)
    p_sp.add_argument("--out-dir", type=Path, required=True)
    p_sp.add_argument("--val-ratio", type=float, default=0.1)
    p_sp.add_argument("--seed", type=int, default=42)
    p_sp.add_argument("--quality-tier", type=str, default=None)

    def _run_prep(ns: argparse.Namespace) -> int:
        from thesis.prepare_repliqa_sft_splits import run_prepare

        return run_prepare(ns)

    p_sp.set_defaults(fn=_run_prep)

    p_ev = sub.add_parser(
        "prepare-repliqa-eval-subset",
        help="Sample fixed eval JSONL from RepLiQA 0–3 (default 2000 Q/A)",
    )
    p_ev.add_argument("--jsonl-dir", type=Path, default=None)
    p_ev.add_argument("--run-root", type=Path, default=None)
    p_ev.add_argument("--eval-dir", type=Path, default=None)
    p_ev.add_argument("--splits", nargs="+", default=None)
    p_ev.add_argument("--docs-per-split", type=int, default=100)
    p_ev.add_argument("--questions-per-doc", type=int, default=5)
    p_ev.add_argument("--seed", type=int, default=42)
    p_ev.add_argument("--exclude-train-documents", action="store_true")
    p_ev.add_argument("--train-documents-jsonl", type=Path, default=None)
    p_ev.add_argument("--output-name", type=str, default="eval_subset_2000.jsonl")
    p_ev.add_argument("--manifest-name", type=str, default="eval_subset_manifest.json")

    def _run_eval_subset(ns: argparse.Namespace) -> int:
        from thesis.paths import DEFAULT_TRAIN_SPLITS, REPLIQA_JSONL_DIR
        from thesis.prepare_repliqa_eval_subset import run_prepare_repliqa_eval_subset

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.jsonl_dir is None:
            ns.jsonl_dir = REPLIQA_JSONL_DIR
        if ns.run_root is None:
            ns.run_root = run_root
        if ns.splits is None:
            ns.splits = list(DEFAULT_TRAIN_SPLITS)
        return run_prepare_repliqa_eval_subset(ns)

    p_ev.set_defaults(fn=_run_eval_subset)

    p_cl = sub.add_parser(
        "classify-repliqa-eval-subset",
        help="Classify eval Qs (Bedrock Haiku, local 3B, or heuristics)",
    )
    p_cl.add_argument("--run-root", type=Path, default=None)
    p_cl.add_argument("--eval-dir", type=Path, default=None)
    p_cl.add_argument("--input-jsonl", type=Path, default=None)
    p_cl.add_argument("--input-name", type=str, default="eval_subset_2000.jsonl")
    p_cl.add_argument("--output-jsonl", type=Path, default=None)
    p_cl.add_argument("--output-name", type=str, default="eval_subset_2000_classified.jsonl")
    p_cl.add_argument("--summary-json", type=Path, default=None)
    p_cl.add_argument("--summary-name", type=str, default="eval_classification_summary.json")
    p_cl.add_argument("--train-documents-jsonl", type=Path, default=None)
    p_cl.add_argument("--backend", type=str, choices=("bedrock", "llm", "heuristic"), default="bedrock")
    p_cl.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_cl.add_argument("--max-rows", type=int, default=0)
    p_cl.add_argument("--max-context-chars", type=int, default=12000)
    p_cl.add_argument("--max-new-tokens", type=int, default=384)
    p_cl.add_argument("--concurrency", type=int, default=4)
    p_cl.add_argument("--request-delay-s", type=float, default=0.1)
    p_cl.add_argument("--bf16", action="store_true", default=True)
    p_cl.add_argument("--no-bf16", action="store_true")

    def _run_classify_eval(ns: argparse.Namespace) -> int:
        from thesis.classify_repliqa_eval_subset import run_classify_repliqa_eval_subset

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        if ns.eval_dir is None:
            ns.eval_dir = run_root / "eval"
        if ns.no_bf16:
            ns.bf16 = False
        return run_classify_repliqa_eval_subset(ns)

    p_cl.set_defaults(fn=_run_classify_eval)

    p_eg = sub.add_parser(
        "eval-repliqa-generate",
        help="Generate eval answers (greedy by default; --temperature > 0 to sample)",
    )
    p_eg.add_argument("--condition", type=str, default=None)
    p_eg.add_argument("--list-conditions", action="store_true")
    p_eg.add_argument("--run-root", type=Path, default=None)
    p_eg.add_argument("--eval-dir", type=Path, default=None)
    p_eg.add_argument("--eval-jsonl", type=Path, default=None)
    p_eg.add_argument("--eval-input-name", type=str, default="eval_subset_2000.jsonl")
    p_eg.add_argument("--output-dir", type=Path, default=None)
    p_eg.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_eg.add_argument("--max-rows", type=int, default=0)
    p_eg.add_argument("--row-start", type=int, default=0)
    p_eg.add_argument("--row-end", type=int, default=0)
    p_eg.add_argument("--max-seq-length", type=int, default=4096)
    p_eg.add_argument("--max-new-tokens", type=int, default=512)
    p_eg.add_argument("--no-context", action="store_true", help="Question only (no passage in prompt)")
    p_eg.add_argument(
        "--context-fraction",
        type=float,
        default=1.0,
        help="Use first fraction of context chars (e.g. 0.5 = 50%% truncation)",
    )
    p_eg.add_argument(
        "--condition-id",
        type=str,
        default=None,
        help="Output folder name under eval/predictions/ (default: auto from flags)",
    )
    p_eg.add_argument("--bf16", action="store_true", default=True)
    p_eg.add_argument("--no-bf16", action="store_true")
    p_eg.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    p_eg.add_argument("--vllm-base-url", type=str, default=None)
    p_eg.add_argument("--vllm-model", type=str, default=None)
    p_eg.add_argument("--concurrency", type=int, default=4)
    p_eg.add_argument("--temperature", type=float, default=0.0)
    p_eg.add_argument("--top-p", type=float, default=0.95)
    p_eg.add_argument("--seed", type=int, default=None)

    def _run_eval_gen(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_generate import run_eval_repliqa_generate

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        if ns.eval_dir is None:
            ns.eval_dir = Path(ns.run_root) / "eval"
        if ns.no_bf16:
            ns.bf16 = False
        return run_eval_repliqa_generate(ns)

    p_eg.set_defaults(fn=_run_eval_gen)

    p_es = sub.add_parser(
        "eval-repliqa-score",
        help="Score eval predictions: EM, token F1, pred↔gold cosine; rank baselines",
    )
    p_es.add_argument("--run-root", type=Path, default=None)
    p_es.add_argument("--predictions-dir", type=Path, default=None)
    p_es.add_argument("--predictions-index", type=Path, default=None)
    p_es.add_argument("--predictions-jsonl", type=Path, default=None)
    p_es.add_argument("--metrics-dir", type=Path, default=None)
    p_es.add_argument("--no-embed", action="store_true")
    p_es.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_es.add_argument("--embed-device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    p_es.add_argument("--embed-batch-size", type=int, default=64)
    p_es.add_argument(
        "--rank-by",
        type=str,
        default="token_f1",
        choices=("token_f1", "exact_match", "pred_gold_cosine"),
    )
    p_es.add_argument("--write-scored", action="store_true")

    def _run_eval_score(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_score import run_eval_repliqa_score

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        if ns.run_root is None:
            ns.run_root = run_root
        else:
            ns.run_root = Path(ns.run_root).expanduser().resolve()
        eval_dir = ns.run_root / "eval"
        if ns.predictions_dir is None and ns.predictions_jsonl is None and ns.predictions_index is None:
            ns.predictions_dir = eval_dir / "predictions"
        if ns.metrics_dir is None:
            ns.metrics_dir = eval_dir / "metrics"
        return run_eval_repliqa_score(ns)

    p_es.set_defaults(fn=_run_eval_score)

    p_mg = sub.add_parser(
        "merge-qs-lora",
        help="Dense merge: weighted ΔW sum of high+medium+low strat adapters only (no B3).",
    )
    p_mg.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_mg.add_argument(
        "--high-adapter",
        type=Path,
        default=None,
        help="QS_strat_high_lora_r32 directory",
    )
    p_mg.add_argument("--medium-adapter", type=Path, default=None)
    p_mg.add_argument("--low-adapter", type=Path, default=None)
    p_mg.add_argument("--output-dir", type=Path, default=None)
    p_mg.add_argument(
        "--weight-preset",
        type=str,
        choices=[
            "custom",
            "equal",
            "tier",
            "frequency",
            "high_med",
            "low_heavy",
            "inverted",
            "equal_rank_tier",
        ],
        default="custom",
        help=(
            "equal | tier (0.6/0.3/0.1) | frequency | high_med | low_heavy | inverted | "
            "equal_rank_tier | custom"
        ),
    )
    p_mg.add_argument("--qs-dir", type=Path, default=None)
    p_mg.add_argument("--judge-summary", type=Path, default=None)
    p_mg.add_argument("--weight-high", type=float, default=1.0)
    p_mg.add_argument("--weight-medium", type=float, default=1.0)
    p_mg.add_argument("--weight-low", type=float, default=1.0)
    p_mg.add_argument("--bf16", action="store_true", default=True)
    p_mg.add_argument("--no-bf16", action="store_true")
    p_mg.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Run dir for merge timing index + pipeline log append",
    )
    p_mg.add_argument(
        "--use-gpu-merge",
        action="store_true",
        help="Shard large base across GPUs during merge (70B+).",
    )
    p_mg.add_argument(
        "--stream-merge",
        action="store_true",
        help="Merge one LoRA module at a time (low CPU RAM; default with --use-gpu-merge).",
    )

    def _run_merge(ns: argparse.Namespace) -> int:
        from thesis.merge_qs_lora import run_merge_qs_lora

        run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
        qs = run_root / "baselines/qs_strat"
        if ns.high_adapter is None:
            ns.high_adapter = qs / "QS_strat_high_lora_r32"
        if ns.medium_adapter is None:
            ns.medium_adapter = qs / "QS_strat_medium_lora_r16"
        if ns.low_adapter is None:
            ns.low_adapter = qs / "QS_strat_low_lora_r8"
        if ns.qs_dir is None:
            ns.qs_dir = qs
        if ns.output_dir is None and ns.weight_preset == "custom":
            ns.output_dir = qs / "QS_merged_strat_dense"
        if ns.judge_summary is None:
            ns.judge_summary = run_root / "train/synthetic_qa_haiku_judge_summary.json"
        if ns.run_root is None:
            ns.run_root = run_root
        if ns.no_bf16:
            ns.bf16 = False
        return run_merge_qs_lora(ns)

    p_mg.set_defaults(fn=_run_merge)

    _JUDGE_FILTER_RUN = (
        Path(__file__).resolve().parent / "experiments/judge_filter/runs/baseline_v1"
    )
    _REPLIQA_RUN = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
    _OHIO_RUN = Path(__file__).resolve().parent / "experiments/ohioline/runs/20260528T163714Z"

    p_jprep = sub.add_parser(
        "prepare-judge-filter-sft",
        help="RepLiQA train/val + OhioLine OOD test for judge-filter distillation (v2)",
    )
    p_jprep.add_argument(
        "--train-judged-jsonl",
        type=Path,
        default=_REPLIQA_RUN / "train/synthetic_qa_haiku_judge.jsonl",
    )
    p_jprep.add_argument(
        "--test-judged-jsonl",
        type=Path,
        default=None,
        help="Fixed OOD test (v1 baseline). Default for v1: OhioLine bedrock_judge.jsonl.",
    )
    p_jprep.add_argument(
        "--extra-judged-jsonl",
        type=Path,
        default=None,
        help="Merge into train/val (legacy single extra).",
    )
    p_jprep.add_argument(
        "--extra-judged-jsonls",
        type=Path,
        nargs="*",
        default=[],
        help="Multiple extra judged JSONL files merged into train/val.",
    )
    p_jprep.add_argument(
        "--extra-labels",
        type=str,
        nargs="*",
        default=[],
        help="Labels parallel to --extra-judged-jsonls.",
    )
    p_jprep.add_argument(
        "--ood-extra-label",
        type=str,
        default="ohioline",
        help="Which extra source val holdout is OOD test.",
    )
    p_jprep.add_argument("--extra-label", type=str, default="ohioline")
    p_jprep.add_argument("--out-dir", type=Path, default=_JUDGE_FILTER_RUN / "splits")
    p_jprep.add_argument("--val-ratio", type=float, default=0.1)
    p_jprep.add_argument("--seed", type=int, default=42)
    p_jprep.add_argument("--max-context-chars", type=int, default=8000)

    def _run_jprep(ns: argparse.Namespace) -> int:
        from thesis.prepare_judge_filter_sft import run_prepare

        if ns.test_judged_jsonl is None and ns.extra_judged_jsonl is None:
            ns.test_judged_jsonl = _OHIO_RUN / "train/bedrock_judge.jsonl"
        return run_prepare(ns)

    p_jprep.set_defaults(fn=_run_jprep)

    p_jtr = sub.add_parser(
        "train-judge-filter",
        help="LoRA SFT: Llama-3.2-3B training-filter judge (qa_judge_rubric/v2)",
    )
    p_jtr.add_argument("--splits-dir", type=Path, default=_JUDGE_FILTER_RUN / "splits")
    p_jtr.add_argument("--output-dir", type=Path, default=_JUDGE_FILTER_RUN / "model")
    p_jtr.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_jtr.add_argument("--lora-r", type=int, default=16)
    p_jtr.add_argument("--lora-alpha", type=int, default=32)
    p_jtr.add_argument("--lora-dropout", type=float, default=0.05)
    p_jtr.add_argument("--epochs", type=int, default=3)
    p_jtr.add_argument("--lr", type=float, default=2e-4)
    p_jtr.add_argument("--max-seq-length", type=int, default=4096)
    p_jtr.add_argument("--batch-size", type=int, default=1)
    p_jtr.add_argument("--grad-accum", type=int, default=8)
    p_jtr.add_argument("--seed", type=int, default=42)
    p_jtr.add_argument("--no-bf16", action="store_true")
    p_jtr.add_argument("--use-qlora-4bit", action="store_true")

    def _run_jtr(ns: argparse.Namespace) -> int:
        from thesis.train_judge_filter import run_train_judge_filter

        return run_train_judge_filter(ns)

    p_jtr.set_defaults(fn=_run_jtr)

    p_jev = sub.add_parser(
        "eval-judge-filter",
        help="Score OhioLine OOD test with distilled judge; compare to teacher labels",
    )
    p_jev.add_argument("--adapter-dir", type=Path, default=_JUDGE_FILTER_RUN / "model")
    p_jev.add_argument(
        "--test-jsonl",
        type=Path,
        default=_JUDGE_FILTER_RUN / "splits/test_ohioline.jsonl",
    )
    p_jev.add_argument("--out-dir", type=Path, default=None)
    p_jev.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p_jev.add_argument("--max-rows", type=int, default=0)
    p_jev.add_argument("--max-context-chars", type=int, default=8000)
    p_jev.add_argument("--max-new-tokens", type=int, default=384)
    p_jev.add_argument("--no-bf16", action="store_true")

    def _run_jev(ns: argparse.Namespace) -> int:
        from thesis.eval_judge_filter import run_eval_judge_filter

        return run_eval_judge_filter(ns)

    p_jev.set_defaults(fn=_run_jev)

    p_dgen = sub.add_parser(
        "eval-drop-generate",
        help="Generate DROP/Quoref/SQuAD eval answers (greedy by default; --temperature > 0 to sample)",
    )
    p_dgen.add_argument("--condition", type=str, default=None)
    p_dgen.add_argument("--list-conditions", action="store_true")
    p_dgen.add_argument("--run-root", type=Path, default=None)
    p_dgen.add_argument("--eval-dir", type=Path, default=None)
    p_dgen.add_argument("--eval-jsonl", type=Path, default=None)
    p_dgen.add_argument("--output-dir", type=Path, default=None)
    p_dgen.add_argument("--condition-id", type=str, default=None)
    p_dgen.add_argument("--base-model", type=str, default=None)
    p_dgen.add_argument("--max-rows", type=int, default=0)
    p_dgen.add_argument("--row-start", type=int, default=0)
    p_dgen.add_argument("--row-end", type=int, default=0)
    p_dgen.add_argument("--max-seq-length", type=int, default=4096)
    p_dgen.add_argument("--max-new-tokens", type=int, default=128)
    p_dgen.add_argument("--no-context", action="store_true")
    p_dgen.add_argument("--bf16", action="store_true", default=True)
    p_dgen.add_argument("--no-bf16", action="store_true")
    p_dgen.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    p_dgen.add_argument("--vllm-base-url", type=str, default=None)
    p_dgen.add_argument("--vllm-model", type=str, default=None)
    p_dgen.add_argument("--concurrency", type=int, default=4)
    p_dgen.add_argument("--temperature", type=float, default=0.0)
    p_dgen.add_argument("--top-p", type=float, default=0.95)
    p_dgen.add_argument("--seed", type=int, default=None)
    p_dgen.add_argument(
        "--tier-matrix",
        action="store_true",
        help="Use baselines/tier_matrix adapter paths (OhioLine naming).",
    )
    p_dgen.add_argument(
        "--baselines-subdir",
        type=str,
        default=None,
        help="Override baselines subdir under run-root (default: qs_strat).",
    )

    def _run_dgen(ns: argparse.Namespace) -> int:
        from thesis.eval_drop_generate import run_eval_drop_generate
        from thesis.paths import DROP_RUN_CPT, DROP_RUN_QA

        if ns.run_root is None:
            ns.run_root = DROP_RUN_QA
        if ns.base_model is None:
            ns.base_model = str(DROP_RUN_CPT / "merged_base")
        if ns.no_bf16:
            ns.bf16 = False
        return run_eval_drop_generate(ns)

    p_dgen.set_defaults(fn=_run_dgen)

    p_dsc = sub.add_parser(
        "eval-drop-score",
        help="Score DROP validation predictions (max EM/F1 over gold answers)",
    )
    p_dsc.add_argument("--run-root", type=Path, default=None)
    p_dsc.add_argument("--eval-dir", type=Path, default=None)
    p_dsc.add_argument("--predictions-dir", type=Path, default=None)
    p_dsc.add_argument("--predictions-jsonl", type=Path, default=None)
    p_dsc.add_argument("--leaderboard-json", type=Path, default=None)
    p_dsc.add_argument("--write-scored", action="store_true")

    def _run_dsc(ns: argparse.Namespace) -> int:
        from thesis.eval_drop_score import run_eval_drop_score
        from thesis.paths import DROP_RUN_QA

        if ns.run_root is None:
            ns.run_root = DROP_RUN_QA
        if ns.predictions_dir is None and ns.predictions_jsonl is None:
            root = Path(ns.run_root)
            ns.predictions_dir = root / "eval" / "predictions"
        return run_eval_drop_score(ns)

    p_dsc.set_defaults(fn=_run_dsc)

    p_dbj = sub.add_parser(
        "eval-drop-bedrock-judge",
        help="Haiku/Bedrock external judge on DROP val predictions (pred vs gold + context, v3)",
    )
    p_dbj.add_argument("--run-root", type=Path, default=None)
    p_dbj.add_argument("--predictions-dir", type=Path, default=None)
    p_dbj.add_argument("--predictions-index", type=Path, default=None)
    p_dbj.add_argument("--predictions-jsonl", type=Path, default=None)
    p_dbj.add_argument("--eval-jsonl", type=Path, default=None)
    p_dbj.add_argument("--judged-dir", type=Path, default=None)
    p_dbj.add_argument("--model", type=str, default=None)
    p_dbj.add_argument("--region", type=str, default=None)
    p_dbj.add_argument("--max-rows", type=int, default=0)
    p_dbj.add_argument("--max-context-chars", type=int, default=8000)
    p_dbj.add_argument("--max-tokens", type=int, default=512)
    p_dbj.add_argument("--temperature", type=float, default=0.0)
    p_dbj.add_argument("--concurrency", type=int, default=4)
    p_dbj.add_argument("--request-delay-s", type=float, default=0.05)
    p_dbj.add_argument("--dry-run", action="store_true")
    p_dbj.add_argument("--skip-existing", action="store_true", default=True)
    p_dbj.add_argument("--force", action="store_true")
    p_dbj.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful judged rows; Bedrock only for failed/missing rows.",
    )
    p_dbj.add_argument(
        "--rank-by",
        type=str,
        default="mean_gold_alignment",
        choices=("mean_gold_alignment", "mean_overall", "mean_grounding"),
    )
    p_dbj.add_argument("--leaderboard-only", action="store_true")
    p_dbj.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Only judge these prediction subdir names.",
    )
    p_dbj.add_argument(
        "--position-swap-debias",
        action="store_true",
        help="Average gold-first and pred-first eval prompts (default output: judged_debias/).",
    )

    def _run_dbj(ns: argparse.Namespace) -> int:
        from thesis.eval_repliqa_bedrock_judge import run_eval_repliqa_bedrock_judge
        from thesis.paths import DROP_JSONL_DIR, DROP_RUN_QA

        if ns.run_root is None:
            ns.run_root = DROP_RUN_QA
        if ns.eval_jsonl is None:
            ns.eval_jsonl = DROP_JSONL_DIR / "validation.jsonl"
        if ns.predictions_dir is None and ns.predictions_jsonl is None:
            ns.predictions_dir = Path(ns.run_root) / "eval" / "predictions"
        if ns.judged_dir is None:
            ns.judged_dir = Path(ns.run_root) / "eval" / (
                "judged_debias" if getattr(ns, "position_swap_debias", False) else "judged"
            )
        return run_eval_repliqa_bedrock_judge(ns)

    p_dbj.set_defaults(fn=_run_dbj)

    p_rjf = sub.add_parser(
        "eval-rejudge-failures",
        help="Re-judge Ours failure rows; compare GA before vs after (optional position-swap debias).",
    )
    p_rjf.add_argument("--run-root", type=Path, required=True)
    p_rjf.add_argument("--ours-condition", type=str, required=True)
    p_rjf.add_argument("--judged-dir", type=Path, default=None)
    p_rjf.add_argument("--output-dir", type=Path, default=None)
    p_rjf.add_argument("--eval-jsonl", type=Path, default=None)
    p_rjf.add_argument("--b3-condition", type=str, default=None)
    p_rjf.add_argument("--b5-condition", type=str, default=None)
    p_rjf.add_argument("--baseline-condition", type=str, default=None)
    p_rjf.add_argument(
        "--mode",
        type=str,
        default="loses_to_best",
        choices=("low_ga", "loses_to_baseline", "loses_to_b3", "loses_to_b5", "loses_to_best"),
    )
    p_rjf.add_argument("--low-ga-threshold", type=int, default=2)
    p_rjf.add_argument("--max-rows", type=int, default=0)
    p_rjf.add_argument("--position-swap-debias", action="store_true")
    p_rjf.add_argument("--model", type=str, default=None)
    p_rjf.add_argument("--region", type=str, default=None)
    p_rjf.add_argument("--max-context-chars", type=int, default=12000)
    p_rjf.add_argument("--max-tokens", type=int, default=512)
    p_rjf.add_argument("--temperature", type=float, default=0.0)
    p_rjf.add_argument("--concurrency", type=int, default=4)
    p_rjf.add_argument("--request-delay-s", type=float, default=0.05)
    p_rjf.add_argument("--dry-run", action="store_true")
    p_rjf.add_argument(
        "--apply-if-improved",
        action="store_true",
        help="Merge re-judged rows into main bedrock_judge.jsonl when full-set GA improves.",
    )
    p_rjf.add_argument("--apply-min-delta", type=float, default=0.0)

    def _run_rejudge_failures(ns: argparse.Namespace) -> int:
        from thesis.eval_rejudge_failures import run_eval_rejudge_failures

        if ns.baseline_condition and ns.mode == "loses_to_best":
            ns.mode = "loses_to_baseline"
        return run_eval_rejudge_failures(ns)

    p_rjf.set_defaults(fn=_run_rejudge_failures)

    ns = ap.parse_args()
    return int(ns.fn(ns))


if __name__ == "__main__":
    raise SystemExit(main())

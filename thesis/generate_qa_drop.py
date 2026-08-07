#!/usr/bin/env python3
"""
Generate synthetic Q/A from passage JSONL — one call per unique section_id.

Backends:
  vllm    — OpenAI-compatible vLLM server (fast; recommended on GPU nodes)
  local   — load merged generator with Transformers on GPU (serial, slow)
  bedrock — AWS Bedrock Haiku (needs source thesis/scripts/source_bedrock_env.sh)

Usage (from finetuning/, start vLLM first on the GPU node):
  python -m vllm.entrypoints.openai.api_server --model <merged-generator> \\
    --host 127.0.0.1 --port 8100 --dtype auto

  python -m thesis.cli generate-drop-synthetic \\
    --backend vllm \\
    --jsonl data/ohioline/jsonl/chunks.jsonl \\
    --exp-root thesis/experiments/ohioline \\
    --pairs-per-passage 2 --concurrency 8 --prompt-profile default
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PKG = Path(__file__).resolve().parent.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import thesis.bootstrap  # noqa: F401

from generate_qa_from_chunks import (
    _check_vllm,
    _openai_base_url,
    _vllm_chat,
    extract_json_object,
)
from paths import (
    MERGED_GENERATOR_DIR,
    QA_GEN_CONCURRENCY,
    QA_GEN_MAX_NEW_TOKENS,
    QA_GEN_TEMPERATURE,
    generator_vllm_base_url,
    generator_vllm_model_id,
)
from thesis.prompts import (
    DROP_GENERATOR_PROMPT_VERSION,
    DROP_GENERATOR_SYSTEM,
    GENERATOR_SYSTEM,
    OHIOLINE_GENERATOR_PROMPT_VERSION,
    OHIOLINE_GENERATOR_SYSTEM,
    drop_generator_user_block,
    generator_user_block,
    ohioline_generator_user_block,
)
from thesis.generate_qa_repliqa import load_jsonl
from thesis.paths import DROP_EXP_ROOT, DROP_JSONL_DIR


def _resolve_jsonl_paths(ns: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    raw = getattr(ns, "jsonl", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        paths.extend(Path(p).expanduser().resolve() for p in raw)
    else:
        paths.append(Path(raw).expanduser().resolve())
    for p in getattr(ns, "extra_jsonl", None) or []:
        paths.append(Path(p).expanduser().resolve())
    return paths


def dedupe_passages(
    jsonl_paths: Path | list[Path],
    *,
    min_context_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One record per section_id (supports multiple QA JSONL sources)."""
    if isinstance(jsonl_paths, Path):
        paths = [jsonl_paths]
    else:
        paths = list(jsonl_paths)

    by_id: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        "input_rows": 0,
        "unique_passages": 0,
        "skipped_short_context": 0,
        "skipped_missing_section_id": 0,
        "source_jsonl": [str(p) for p in paths],
    }

    for jsonl_path in paths:
        if not jsonl_path.is_file():
            raise FileNotFoundError(jsonl_path)
        for row in load_jsonl(jsonl_path):
            stats["input_rows"] += 1
            section_id = (row.get("section_id") or "").strip()
            ctx = (row.get("context") or "").strip()
            if not section_id:
                stats["skipped_missing_section_id"] += 1
                continue
            if section_id not in by_id:
                if len(ctx) < min_context_chars:
                    stats["skipped_short_context"] += 1
                    continue
                rec: dict[str, Any] = {
                    "section_id": section_id,
                    "context": ctx,
                    "human_qa_rows_in_source": 1,
                }
                did = (row.get("document_id") or "").strip()
                if did:
                    rec["document_id"] = did
                topic = row.get("document_topic")
                if topic is not None:
                    rec["document_topic"] = topic
                by_id[section_id] = rec
            else:
                by_id[section_id]["human_qa_rows_in_source"] = (
                    int(by_id[section_id].get("human_qa_rows_in_source", 0)) + 1
                )

    passages = list(by_id.values())
    stats["unique_passages"] = len(passages)
    return passages, stats


def _prompt_for_profile(
    profile: str,
    passage: str,
    *,
    pairs_per: int = 1,
) -> tuple[str, str, str]:
    if profile == "drop":
        return DROP_GENERATOR_SYSTEM, drop_generator_user_block(passage), DROP_GENERATOR_PROMPT_VERSION
    if profile == "default":
        return GENERATOR_SYSTEM, generator_user_block(passage), "default"
    if profile == "ohioline":
        return (
            OHIOLINE_GENERATOR_SYSTEM,
            ohioline_generator_user_block(passage, n_pairs=pairs_per),
            OHIOLINE_GENERATOR_PROMPT_VERSION,
        )
    raise ValueError(f"Unknown --prompt-profile {profile!r}")


def _extract_qa_pairs(raw: str, *, expected: int | None = None) -> list[tuple[str, str]]:
    """Parse one or many Q/A pairs from model output."""
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    obj: Any = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        single = extract_json_object(raw)
        if single:
            q = str(single.get("question", "")).strip()
            a = str(single.get("answer", "")).strip()
            return [(q, a)] if q and a else []
        pairs: list[tuple[str, str]] = []
        for m in re.finditer(
            r"\{[^{}]*\"question\"\s*:\s*\"((?:\\.|[^\"])*)\"\s*,\s*\"answer\"\s*:\s*\"((?:\\.|[^\"])*)\"\s*\}",
            s,
            re.DOTALL,
        ):
            q = json.loads(f'"{m.group(1)}"') if "\\" in m.group(1) else m.group(1)
            a = json.loads(f'"{m.group(2)}"') if "\\" in m.group(2) else m.group(2)
            q, a = str(q).strip(), str(a).strip()
            if q and a:
                pairs.append((q, a))
        return pairs[:expected] if expected else pairs

    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        if "pairs" in obj and isinstance(obj["pairs"], list):
            items = obj["pairs"]
        elif "question" in obj and "answer" in obj:
            items = [obj]
        else:
            items = []
    elif isinstance(obj, list):
        items = obj
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            out.append((q, a))
    if expected is not None and expected > 0:
        return out[:expected]
    return out


def _load_local_generator(model_path: Path, *, bf16: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = model_path.expanduser().resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Generator model not found: {path}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dtype = torch.bfloat16 if bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _generate_local(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    max_seq_length: int,
) -> str:
    import torch
    from transformers import GenerationConfig

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    device = next(model.parameters()).device
    if isinstance(inputs, torch.Tensor):
        gen_in = {"input_ids": inputs.to(device)}
    else:
        gen_in = {k: v.to(device) for k, v in dict(inputs).items()}
    if "attention_mask" not in gen_in:
        gen_in["attention_mask"] = torch.ones_like(
            gen_in["input_ids"], dtype=torch.long, device=device
        )
    input_len = gen_in["input_ids"].shape[-1]
    do_sample = temperature > 0
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.9 if do_sample else None,
    )
    with torch.no_grad():
        out = model.generate(**gen_in, generation_config=gen_cfg)
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


def _generate_bedrock(
    client: Any,
    *,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    from thesis.bedrock_judge_qa_score import _invoke_bedrock_claude

    return _invoke_bedrock_claude(
        client,
        model_id=model_id,
        system=system,
        user_message=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def run_generate_drop(ns: argparse.Namespace) -> int:
    if getattr(ns, "list_prompts", False):
        print("  default — scientific / generic excerpt Q/A")
        print("  drop    — DROP-style reasoning (count/compare/arithmetic)")
        return 0

    jsonl_paths = _resolve_jsonl_paths(ns)
    if not jsonl_paths:
        raise SystemExit("No --jsonl paths provided")

    exp_root = Path(ns.exp_root).expanduser().resolve()
    passages, dedupe_stats = dedupe_passages(
        jsonl_paths, min_context_chars=int(ns.min_context_chars)
    )
    passage_policy = str(getattr(ns, "passage_policy", "deploy_full_kb") or "deploy_full_kb")
    if int(ns.max_passages) > 0:
        passages = passages[: int(ns.max_passages)]

    run_name = (ns.run_name or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = exp_root / "runs" / run_name
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    passages_path = train_dir / "passages_unique.jsonl"
    with open(passages_path, "w", encoding="utf-8") as fp:
        for p in passages:
            fp.write(json.dumps(p, ensure_ascii=False) + "\n")

    if not passages:
        print("No passages after dedupe.", file=sys.stderr)
        return 1

    backend = str(ns.backend).strip().lower()
    profile = str(ns.prompt_profile)
    pairs_per = max(1, int(ns.pairs_per_passage))
    max_tok = int(ns.max_new_tokens)
    temp = float(ns.temperature)
    source_tag = str(ns.source_tag)
    max_seq = int(ns.max_seq_length)

    model_path = Path(ns.model_path).expanduser().resolve()
    bedrock_model = str(ns.bedrock_model or os.environ.get("BEDROCK_JUDGE_MODEL_ID", "")).strip()
    region = str(ns.region or os.environ.get("AWS_REGION", "us-east-1")).strip()

    local_model = None
    local_tokenizer = None
    bedrock_client = None
    vllm_client = None
    gen_lock = threading.Lock()
    vllm_model = str(getattr(ns, "vllm_model", "") or generator_vllm_model_id()).strip()
    vllm_base_url = str(getattr(ns, "vllm_base_url", "") or generator_vllm_base_url()).strip()

    if backend == "vllm":
        from openai import OpenAI

        vllm_client = OpenAI(base_url=_openai_base_url(vllm_base_url), api_key="unused")
        print(f"vLLM base: {_openai_base_url(vllm_base_url)}  model={vllm_model!r}", flush=True)
        _check_vllm(vllm_client)
        generator_label = vllm_model
        concurrency = max(1, int(ns.concurrency))
        print(f"vLLM concurrency={concurrency}", flush=True)
    elif backend == "local":
        bf16 = bool(ns.bf16) and not bool(ns.no_bf16)
        print(f"Loading local generator: {model_path} (bf16={bf16})", flush=True)
        t0 = time.perf_counter()
        local_model, local_tokenizer = _load_local_generator(model_path, bf16=bf16)
        print(f"Model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        generator_label = str(model_path)
        concurrency = 1
    elif backend == "bedrock":
        from thesis.bedrock_judge_qa_score import DEFAULT_BEDROCK_MODEL_ID, _bedrock_client, _check_aws_env

        if not bedrock_model:
            bedrock_model = DEFAULT_BEDROCK_MODEL_ID
        _check_aws_env(region)
        bedrock_client = _bedrock_client(region)
        generator_label = bedrock_model
        concurrency = max(1, int(ns.concurrency))
        print(f"Bedrock generator: {bedrock_model} region={region} concurrency={concurrency}", flush=True)
    else:
        raise SystemExit(f"Unknown --backend {backend!r} (use vllm, local, or bedrock)")

    work: list[dict[str, Any]] = []
    multi_pair_call = profile == "ohioline"
    for p in passages:
        sid = p["section_id"]
        ctx = p["context"]
        system, user, pver = _prompt_for_profile(
            profile, ctx, pairs_per=pairs_per if multi_pair_call else 1
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if multi_pair_call:
            work.append(
                {
                    "section_id": sid,
                    "context": ctx,
                    "pair_index": 0,
                    "expected_pairs": pairs_per,
                    "prompt_profile": profile,
                    "generator_prompt_version": pver,
                    "system": system,
                    "user": user,
                    "messages": messages,
                    "parent_section_id": p.get("parent_section_id"),
                    "document_id": p.get("document_id"),
                    "document_topic": p.get("document_topic"),
                }
            )
        else:
            for k in range(pairs_per):
                work.append(
                    {
                        "section_id": sid,
                        "context": ctx,
                        "pair_index": k,
                        "expected_pairs": 1,
                        "prompt_profile": profile,
                        "generator_prompt_version": pver,
                        "system": system,
                        "user": user,
                        "messages": messages,
                        "parent_section_id": p.get("parent_section_id"),
                        "document_id": p.get("document_id"),
                        "document_topic": p.get("document_topic"),
                    }
                )

    if not ns.no_length_sort and len(work) > 1:
        work.sort(key=lambda j: len(j["user"]))

    print(
        f"DROP synthetic gen: backend={backend} passages={len(passages)} "
        f"pairs_per_passage={pairs_per} jobs={len(work)} profile={profile} "
        f"multi_pair_call={multi_pair_call}",
        flush=True,
    )

    synthetic_path = train_dir / "synthetic_qa.jsonl"
    log_path = train_dir / "generation_log.jsonl"
    lock = threading.Lock()
    n_ok = 0
    n_fail = 0
    done = 0
    synthetic_lines: list[str] = []

    def one_job(job: dict[str, Any]) -> dict[str, Any]:
        nonlocal n_ok, n_fail, done
        sid = job["section_id"]
        expected = int(job.get("expected_pairs") or 1)
        log_entry: dict[str, Any] = {
            "section_id": sid,
            "pair_index": job.get("pair_index", 0),
            "expected_pairs": expected,
            "status": "ok",
            "error": None,
            "n_pairs_parsed": 0,
        }
        record_lines: list[str] = []
        raw = ""
        try:
            # For multi-pair JSON, allow more completion tokens.
            call_max_tok = max_tok
            if expected > 1:
                call_max_tok = max(max_tok, min(1024, 220 * expected + 80))

            if backend == "vllm":
                assert vllm_client is not None
                raw = _vllm_chat(
                    client=vllm_client,
                    model=vllm_model,
                    messages=job["messages"],
                    max_tokens=call_max_tok,
                    temperature=temp,
                )
            elif backend == "local":
                assert local_model is not None and local_tokenizer is not None
                with gen_lock:
                    raw = _generate_local(
                        local_model,
                        local_tokenizer,
                        job["messages"],
                        max_new_tokens=call_max_tok,
                        temperature=temp,
                        max_seq_length=max_seq,
                    )
            else:
                assert bedrock_client is not None
                raw = _generate_bedrock(
                    bedrock_client,
                    model_id=bedrock_model,
                    system=job["system"],
                    user=job["user"],
                    max_tokens=call_max_tok,
                    temperature=temp,
                )
            pairs = _extract_qa_pairs(raw, expected=expected)
            log_entry["n_pairs_parsed"] = len(pairs)
            if not pairs:
                log_entry["status"] = "parse_error"
            else:
                for k, (q, a) in enumerate(pairs):
                    record = {
                        "context": job["context"],
                        "question": q,
                        "answer": a,
                        "source": source_tag,
                        "chunk_id": f"{sid}::syn_{k:02d}",
                        "section_id": sid,
                        "synthetic_pair_index": k,
                        "generator_backend": backend,
                        "generator_model": generator_label,
                        "generator_prompt_version": job["generator_prompt_version"],
                        "prompt_profile": job["prompt_profile"],
                    }
                    if job.get("document_id"):
                        record["document_id"] = job["document_id"]
                    elif "::" in sid:
                        record["document_id"] = sid.split("::", 1)[0]
                    if job.get("document_topic") is not None:
                        record["document_topic"] = job["document_topic"]
                    if job.get("parent_section_id"):
                        record["parent_section_id"] = job["parent_section_id"]
                    record_lines.append(json.dumps(record, ensure_ascii=False))
                if len(pairs) < expected:
                    log_entry["status"] = "partial_pairs"
        except Exception as e:
            log_entry["status"] = "vllm_error" if backend == "vllm" else "error"
            log_entry["error"] = str(e)

        if raw and log_entry["status"] not in ("ok", "partial_pairs"):
            log_entry["raw_preview"] = raw[:500]

        with lock:
            done += 1
            if record_lines:
                n_ok += len(record_lines)
                synthetic_lines.extend(record_lines)
            else:
                n_fail += 1
            if done % 20 == 0 or done == len(work):
                print(
                    f"  ... {done}/{len(work)} jobs, {n_ok} pairs ok, {n_fail} failed jobs",
                    flush=True,
                )
        return log_entry

    log_entries: list[dict[str, Any]] = []
    if concurrency <= 1:
        for job in work:
            log_entries.append(one_job(job))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(one_job, j) for j in work]
            for fut in as_completed(futs):
                log_entries.append(fut.result())

    with open(synthetic_path, "w", encoding="utf-8") as fp:
        for line in synthetic_lines:
            fp.write(line + "\n")
    with open(log_path, "w", encoding="utf-8") as fp:
        for entry in log_entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    config = {
        "schema": "drop_synthetic_run/v2",
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passage_policy": passage_policy,
        "input_jsonl": [str(p) for p in jsonl_paths],
        "backend": backend,
        "generator_model": generator_label,
        "vllm_base_url": vllm_base_url if backend == "vllm" else None,
        "concurrency": concurrency,
        "prompt_profile": profile,
        "generator_prompt_version": (
            DROP_GENERATOR_PROMPT_VERSION
            if profile == "drop"
            else OHIOLINE_GENERATOR_PROMPT_VERSION
            if profile == "ohioline"
            else "default"
        ),
        "multi_pair_call": multi_pair_call,
        "pairs_per_passage": pairs_per,
        "paths": {
            "run_dir": str(run_dir),
            "passages_unique": str(passages_path),
            "synthetic_qa": str(synthetic_path),
            "generation_log": str(log_path),
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        **dedupe_stats,
        "passages_scheduled": len(passages),
        "generation_jobs": len(work),
        "synthetic_pairs_written": n_ok,
        "generation_failures": n_fail,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {synthetic_path} ({n_ok} pairs)", flush=True)
    print(f"Run: {run_dir}", flush=True)
    return 0 if n_ok > 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic Q/A from DROP passages")
    p.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        default=None,
        help="QA JSONL (repeat for train+validation deploy-Full KB)",
    )
    p.add_argument("--extra-jsonl", type=Path, action="append", default=None)
    p.add_argument(
        "--passage-policy",
        type=str,
        default="deploy_full_kb",
        help="deploy_full_kb | train_only",
    )
    p.add_argument("--exp-root", type=Path, default=DROP_EXP_ROOT)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--pairs-per-passage", type=int, default=2)
    p.add_argument("--max-passages", type=int, default=0, help="0 = all unique section_id")
    p.add_argument("--min-context-chars", type=int, default=40)
    p.add_argument(
        "--backend",
        choices=("vllm", "local", "bedrock"),
        default="vllm",
        help="vllm=OpenAI API to vLLM server (fast); local=Transformers; bedrock=Haiku",
    )
    p.add_argument("--vllm-base-url", type=str, default=generator_vllm_base_url())
    p.add_argument("--vllm-model", type=str, default=generator_vllm_model_id())
    p.add_argument(
        "--model-path",
        type=Path,
        default=MERGED_GENERATOR_DIR,
        help="Merged generator dir (local backend only)",
    )
    p.add_argument("--bedrock-model", type=str, default=None)
    p.add_argument("--region", type=str, default=None)
    p.add_argument(
        "--concurrency",
        type=int,
        default=QA_GEN_CONCURRENCY,
        help="vllm and bedrock parallel requests; local uses 1",
    )
    p.add_argument("--max-new-tokens", type=int, default=QA_GEN_MAX_NEW_TOKENS)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=QA_GEN_TEMPERATURE)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--source-tag", type=str, default="drop/synthetic/train")
    p.add_argument("--prompt-profile", choices=("drop", "default", "ohioline"), default="drop")
    p.add_argument("--list-prompts", action="store_true")
    p.add_argument("--no-length-sort", action="store_true")
    ns = p.parse_args()
    if not ns.jsonl:
        ns.jsonl = [DROP_JSONL_DIR / "train.jsonl"]
    if ns.no_bf16:
        ns.bf16 = False
    return run_generate_drop(ns)


if __name__ == "__main__":
    raise SystemExit(main())

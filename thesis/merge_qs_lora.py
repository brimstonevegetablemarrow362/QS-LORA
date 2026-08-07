"""
Dense merge of QS-LoRA stratified adapters (Option B: weighted sum of ΔW).

Merges only tier specialists (default: high r=32, medium r=16, low r=8).
Does NOT include B3 or other baselines.

  ΔW_merged = w_h·ΔW_high + w_m·ΔW_medium + w_l·ΔW_low
  W' = W_base + ΔW_merged  → full checkpoint for vLLM
"""

from __future__ import annotations

import argparse
import json
import time
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt_hms(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    s = int(round(float(seconds)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def timing_record(step: str, duration_s: float, **extra: Any) -> dict[str, Any]:
    return {
        "step": step,
        "duration_s": round(float(duration_s), 3),
        "duration_hms": fmt_hms(duration_s),
        "ts_end": utc_iso(),
        **extra,
    }


# Haiku judge tier counts (synthetic_qa_haiku_judge_summary.json, usable tiers only)
DEFAULT_TIER_COUNTS = {"high": 7668, "medium": 1274, "low": 643}

MERGE_WEIGHT_PRESETS: dict[str, dict[str, Any]] = {
    "equal": {
        "weights": (1.0, 1.0, 1.0),
        "output_dir_name": "QS_merged_strat_dense",
        "description": "Equal weight ablation",
    },
    "tier": {
        "weights": (0.6, 0.3, 0.1),
        "output_dir_name": "QS_merged_strat_dense_w60_30_10",
        "description": "Hand-set tier importance (high emphasis)",
    },
    "frequency": {
        "weights": None,  # computed from judge summary
        "output_dir_name": "QS_merged_strat_dense_freq",
        "description": "Proportional to Haiku tier counts in train pool",
    },
    "high_med": {
        "weights": (0.67, 0.33, 0.0),
        "output_dir_name": "QS_merged_strat_dense_high_med_w67_33_0",
        "description": "High+medium only (low tier weight=0)",
    },
    "low_heavy": {
        "weights": (0.4, 0.4, 0.2),
        "output_dir_name": "QS_merged_strat_dense_w40_40_20",
        "description": "Low-tier emphasis ablation",
    },
    "inverted": {
        "weights": (0.1, 0.3, 0.6),
        "output_dir_name": "QS_merged_strat_dense_w10_30_60",
        "description": "Inverted tier weights (low emphasis)",
    },
    "equal_rank_tier": {
        "weights": (0.6, 0.3, 0.1),
        "output_dir_name": "QS_merged_equal_rank_w60_30_10",
        "description": "Equal-rank tier adapters (r=16/16/16) with 0.6/0.3/0.1 merge",
    },
}


def frequency_weights_from_counts(
    counts: dict[str, int] | None = None,
) -> tuple[float, float, float, dict[str, Any]]:
    c = counts or DEFAULT_TIER_COUNTS
    hi, med, lo = int(c["high"]), int(c["medium"]), int(c["low"])
    total = hi + med + lo
    if total <= 0:
        raise ValueError("tier counts must sum to > 0")
    w_h, w_m, w_l = hi / total, med / total, lo / total
    meta = {
        "tier_counts": {"high": hi, "medium": med, "low": lo},
        "total_tier_rows": total,
        "weights": {"high": w_h, "medium": w_m, "low": w_l},
    }
    return w_h, w_m, w_l, meta


def resolve_merge_preset(
    preset: str,
    *,
    qs_dir: Path,
    judge_summary_jsonl: Path | None = None,
) -> tuple[float, float, float, Path, str, dict[str, Any]]:
    if preset not in MERGE_WEIGHT_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {list(MERGE_WEIGHT_PRESETS)}")

    info = dict(MERGE_WEIGHT_PRESETS[preset])
    extra: dict[str, Any] = {"preset": preset, "description": info["description"]}

    if preset == "frequency":
        summary_path = judge_summary_jsonl
        if summary_path is None:
            summary_path = (
                qs_dir.parent.parent / "train" / "synthetic_qa_haiku_judge_summary.json"
            )
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            counts = summary.get("stats", {}).get("tier_counts", DEFAULT_TIER_COUNTS)
            # use only merge tiers
            counts = {k: counts[k] for k in ("high", "medium", "low")}
        else:
            counts = DEFAULT_TIER_COUNTS
        w_h, w_m, w_l, freq_meta = frequency_weights_from_counts(counts)
        extra["frequency_meta"] = freq_meta
    else:
        w_h, w_m, w_l = info["weights"]

    out_dir = qs_dir / info["output_dir_name"]
    return w_h, w_m, w_l, out_dir, preset, extra


@dataclass
class AdapterSpec:
    name: str
    path: Path
    weight: float = 1.0


@dataclass
class MergeTiming:
    """Wall-clock breakdown for dense QS merge."""

    per_adapter_load_s: dict[str, float] = field(default_factory=dict)
    per_adapter_accumulate_s: dict[str, float] = field(default_factory=dict)
    combine_delta_s: float = 0.0
    load_base_s: float = 0.0
    apply_delta_s: float = 0.0
    save_model_s: float = 0.0
    save_tokenizer_s: float = 0.0
    total_wall_s: float = 0.0

    def to_manifest_dict(self) -> dict[str, Any]:
        load_sum = round(sum(self.per_adapter_load_s.values()), 3)
        accum_sum = round(sum(self.per_adapter_accumulate_s.values()), 3)
        return {
            "combine_delta_s": round(self.combine_delta_s, 3),
            "combine_delta_hms": fmt_hms(self.combine_delta_s),
            "combine_breakdown": {
                "load_adapter_deltas_s": load_sum,
                "load_adapter_deltas_hms": fmt_hms(load_sum),
                "weighted_accumulate_s": accum_sum,
                "weighted_accumulate_hms": fmt_hms(accum_sum),
                "per_adapter_load_s": {k: round(v, 3) for k, v in self.per_adapter_load_s.items()},
                "per_adapter_accumulate_s": {
                    k: round(v, 3) for k, v in self.per_adapter_accumulate_s.items()
                },
            },
            "load_base_s": round(self.load_base_s, 3),
            "load_base_hms": fmt_hms(self.load_base_s),
            "apply_delta_s": round(self.apply_delta_s, 3),
            "apply_delta_hms": fmt_hms(self.apply_delta_s),
            "save_model_s": round(self.save_model_s, 3),
            "save_model_hms": fmt_hms(self.save_model_s),
            "save_tokenizer_s": round(self.save_tokenizer_s, 3),
            "save_tokenizer_hms": fmt_hms(self.save_tokenizer_s),
            "save_total_s": round(self.save_model_s + self.save_tokenizer_s, 3),
            "save_total_hms": fmt_hms(self.save_model_s + self.save_tokenizer_s),
            "total_wall_s": round(self.total_wall_s, 3),
            "total_wall_hms": fmt_hms(self.total_wall_s),
        }


def _adapter_weights_path(adapter_dir: Path) -> Path:
    adapter_dir = adapter_dir.expanduser().resolve()
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")


def _load_lora_config(adapter_dir: Path) -> dict[str, Any]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _module_prefix_from_lora_key(key: str) -> str:
    if key.endswith(".lora_A.weight"):
        return key[: -len(".lora_A.weight")]
    if key.endswith(".lora_B.weight"):
        return key[: -len(".lora_B.weight")]
    raise ValueError(f"Not a LoRA weight key: {key}")


def _base_state_key(module_prefix: str) -> str:
    # base_model.model.model.layers... → model.layers...
    if module_prefix.startswith("base_model.model."):
        module_prefix = module_prefix[len("base_model.model.") :]
    return f"{module_prefix}.weight"


def _lora_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    alpha: int,
    r: int,
    fan_in_fan_out: bool = False,
) -> torch.Tensor:
    a = lora_a.float()
    b = lora_b.float()
    delta = b @ a
    if fan_in_fan_out:
        delta = delta.T
    return delta * (float(alpha) / float(r))


def _iter_lora_prefixes(adapter_dir: Path) -> list[str]:
    """Sorted module prefixes that have both lora_A and lora_B in an adapter."""
    weights_path = _adapter_weights_path(adapter_dir)
    lora_a: set[str] = set()
    lora_b: set[str] = set()

    if weights_path.suffix == ".safetensors":
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".lora_A.weight"):
                    lora_a.add(_module_prefix_from_lora_key(key))
                elif key.endswith(".lora_B.weight"):
                    lora_b.add(_module_prefix_from_lora_key(key))
    else:
        bin_tensors = torch.load(weights_path, map_location="cpu", weights_only=True)
        for key in bin_tensors:
            if key.endswith(".lora_A.weight"):
                lora_a.add(_module_prefix_from_lora_key(key))
            elif key.endswith(".lora_B.weight"):
                lora_b.add(_module_prefix_from_lora_key(key))

    prefixes = sorted(lora_a & lora_b)
    if not prefixes:
        raise ValueError(f"No LoRA A/B pairs in {weights_path}")
    return prefixes


def _load_lora_pair(adapter_dir: Path, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    weights_path = _adapter_weights_path(adapter_dir)
    key_a = f"{prefix}.lora_A.weight"
    key_b = f"{prefix}.lora_B.weight"

    if weights_path.suffix == ".safetensors":
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            if key_a not in f.keys() or key_b not in f.keys():
                raise KeyError(f"Missing LoRA pair {prefix} in {weights_path}")
            return f.get_tensor(key_a), f.get_tensor(key_b)

    bin_tensors = torch.load(weights_path, map_location="cpu", weights_only=True)
    if key_a not in bin_tensors or key_b not in bin_tensors:
        raise KeyError(f"Missing LoRA pair {prefix} in {weights_path}")
    return bin_tensors[key_a], bin_tensors[key_b]


def load_adapter_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Return base state_dict keys → ΔW (float32 CPU)."""
    adapter_dir = adapter_dir.expanduser().resolve()
    cfg = _load_lora_config(adapter_dir)
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg.get("fan_in_fan_out", False))

    weights_path = _adapter_weights_path(adapter_dir)
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}

    if weights_path.suffix == ".safetensors":
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
                    continue
                tensor = f.get_tensor(key)
                prefix = _module_prefix_from_lora_key(key)
                if key.endswith(".lora_A.weight"):
                    lora_a[prefix] = tensor
                else:
                    lora_b[prefix] = tensor
    else:
        bin_tensors = torch.load(weights_path, map_location="cpu", weights_only=True)
        for key, tensor in bin_tensors.items():
            if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
                continue
            prefix = _module_prefix_from_lora_key(key)
            if key.endswith(".lora_A.weight"):
                lora_a[prefix] = tensor
            else:
                lora_b[prefix] = tensor

    prefixes = sorted(set(lora_a) & set(lora_b))
    if not prefixes:
        raise ValueError(f"No LoRA A/B pairs in {weights_path}")

    deltas: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        state_key = _base_state_key(prefix)
        deltas[state_key] = _lora_delta(
            lora_a[prefix], lora_b[prefix], alpha=alpha, r=r, fan_in_fan_out=fan_in_fan_out
        )
    return deltas


def merge_adapter_deltas(
    specs: list[AdapterSpec],
    timing: MergeTiming,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not specs:
        raise ValueError("At least one adapter required")

    merged: dict[str, torch.Tensor] = {}
    manifest_adapters: list[dict[str, Any]] = []
    t_combine0 = time.perf_counter()

    for spec in specs:
        spec.path = spec.path.expanduser().resolve()
        cfg = _load_lora_config(spec.path)

        t_load0 = time.perf_counter()
        deltas = load_adapter_deltas(spec.path)
        load_s = time.perf_counter() - t_load0
        timing.per_adapter_load_s[spec.name] = load_s

        t_acc0 = time.perf_counter()
        for key, delta in deltas.items():
            weighted = spec.weight * delta
            if key not in merged:
                merged[key] = weighted
            else:
                merged[key] = merged[key] + weighted
        accum_s = time.perf_counter() - t_acc0
        timing.per_adapter_accumulate_s[spec.name] = accum_s

        manifest_adapters.append(
            {
                "name": spec.name,
                "path": str(spec.path),
                "weight": spec.weight,
                "lora_r": cfg["r"],
                "lora_alpha": cfg["lora_alpha"],
                "n_modules": len(deltas),
                "timing": {
                    "load_delta_s": round(load_s, 3),
                    "load_delta_hms": fmt_hms(load_s),
                    "weighted_accumulate_s": round(accum_s, 3),
                    "weighted_accumulate_hms": fmt_hms(accum_s),
                },
            }
        )

    timing.combine_delta_s = time.perf_counter() - t_combine0
    return merged, {"adapters": manifest_adapters, "n_merged_keys": len(merged)}


def _write_timing_artifacts(
    *,
    out_dir: Path,
    run_root: Path | None,
    manifest: dict[str, Any],
    spans: list[dict[str, Any]],
) -> None:
    manifest_path = out_dir / "qs_merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    exp_dir = out_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    spans_path = exp_dir / "merge_spans.jsonl"
    with spans_path.open("w", encoding="utf-8") as fp:
        for row in spans:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    timing_summary_path = out_dir / "qs_merge_timing.json"
    timing_summary_path.write_text(
        json.dumps(
            {
                "schema": "qs_merge_timing/v1",
                "output_dir": str(out_dir),
                "started_at": manifest.get("started_at"),
                "finished_at": manifest.get("finished_at"),
                "timing": manifest.get("timing"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if run_root is not None:
        run_root = run_root.expanduser().resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        idx_path = run_root / "qs_merge_timing_index.json"
        idx: dict[str, Any] = {}
        if idx_path.is_file():
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        idx[manifest.get("output_dir", str(out_dir))] = {
            "combine_delta_s": manifest["timing"]["combine_delta_s"],
            "combine_delta_hms": manifest["timing"]["combine_delta_hms"],
            "total_wall_s": manifest["timing"]["total_wall_s"],
            "total_wall_hms": manifest["timing"]["total_wall_hms"],
            "finished_at": manifest.get("finished_at"),
        }
        idx_path.write_text(json.dumps(idx, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        pipeline_log = run_root / "training_pipeline_log.jsonl"
        with pipeline_log.open("a", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    {
                        "ts": utc_iso(),
                        "event": "qs_merge_done",
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
                        "output_dir": str(out_dir),
                        "combine_delta_s": manifest["timing"]["combine_delta_s"],
                        "total_wall_s": manifest["timing"]["total_wall_s"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def apply_merged_delta_to_model(
    model: torch.nn.Module,
    merged_delta: dict[str, torch.Tensor],
) -> int:
    param_map = dict(model.named_parameters())
    missing: list[str] = []
    applied = 0
    with torch.no_grad():
        for key, delta in merged_delta.items():
            param = param_map.get(key)
            if param is None:
                missing.append(key)
                continue
            param.data.add_(delta.to(device=param.device, dtype=param.dtype))
            applied += 1
    if missing:
        sample = missing[:5]
        raise KeyError(
            f"{len(missing)} merged keys not in base model (e.g. {sample}). "
            "Check base model id matches adapter base_model_name_or_path."
        )
    return applied


def merge_adapters_streaming_to_model(
    model: torch.nn.Module,
    specs: list[AdapterSpec],
    timing: MergeTiming,
) -> tuple[int, dict[str, Any]]:
    """Apply weighted ΔW per module without materializing full-rank deltas for all adapters."""
    if not specs:
        raise ValueError("At least one adapter required")

    prefixes = _iter_lora_prefixes(specs[0].path)
    param_map = dict(model.named_parameters())
    manifest_adapters: list[dict[str, Any]] = []
    t_combine0 = time.perf_counter()

    for spec in specs:
        spec.path = spec.path.expanduser().resolve()
        cfg = _load_lora_config(spec.path)
        manifest_adapters.append(
            {
                "name": spec.name,
                "path": str(spec.path),
                "weight": spec.weight,
                "lora_r": cfg["r"],
                "lora_alpha": cfg["lora_alpha"],
                "n_modules": len(prefixes),
            }
        )

    n_applied = 0
    for prefix in prefixes:
        state_key = _base_state_key(prefix)
        param = param_map.get(state_key)
        if param is None:
            continue

        total_delta: torch.Tensor | None = None
        for spec in specs:
            cfg = _load_lora_config(spec.path)
            lora_a, lora_b = _load_lora_pair(spec.path, prefix)
            delta = _lora_delta(
                lora_a,
                lora_b,
                alpha=int(cfg["lora_alpha"]),
                r=int(cfg["r"]),
                fan_in_fan_out=bool(cfg.get("fan_in_fan_out", False)),
            )
            weighted = spec.weight * delta
            total_delta = weighted if total_delta is None else total_delta + weighted

        assert total_delta is not None
        with torch.no_grad():
            param.data.add_(total_delta.to(device=param.device, dtype=param.dtype))
        n_applied += 1
        del total_delta

    timing.combine_delta_s = time.perf_counter() - t_combine0
    timing.apply_delta_s = 0.0
    return n_applied, {"adapters": manifest_adapters, "n_merged_keys": len(prefixes)}


def _normalize_text_only_gemma_merge_config(out_dir: Path) -> bool:
    """Flatten Gemma 4B/12B ConditionalGeneration configs for text-only CausalLM eval."""
    cfg_path = out_dir / "config.json"
    if not cfg_path.is_file():
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("architectures") == ["Gemma3ForCausalLM"] and "text_config" not in cfg:
        return False
    if cfg.get("architectures") != ["Gemma3ForConditionalGeneration"] and "text_config" not in cfg:
        return False

    text_cfg = dict(cfg.get("text_config") or {})
    if not text_cfg:
        return False

    flat = {**text_cfg}
    flat["architectures"] = ["Gemma3ForCausalLM"]
    flat["model_type"] = "gemma3_text"
    flat["dtype"] = cfg.get("dtype") or text_cfg.get("dtype") or "bfloat16"
    if "eos_token_id" in cfg and "eos_token_id" not in flat:
        flat["eos_token_id"] = cfg["eos_token_id"]
    flat["transformers_version"] = cfg.get("transformers_version", flat.get("transformers_version"))
    cfg_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[merge] flattened Gemma text-only config: {cfg_path}", flush=True)
    return True


def _normalize_text_only_gemma_merge_weights(out_dir: Path) -> bool:
    """Rewrite multimodal Gemma merge weights for Gemma3ForCausalLM text-only eval."""
    import safetensors.torch

    changed = False
    for sf_path in sorted(out_dir.glob("*.safetensors")):
        with safe_open(sf_path, framework="pt") as f:
            keys = list(f.keys())
            if not any(k.startswith("language_model.") for k in keys):
                continue
            tensors: dict[str, torch.Tensor] = {}
            for key in keys:
                if key.startswith("language_model."):
                    tensors[key[len("language_model.") :]] = f.get_tensor(key)
                elif key.startswith("model.") or key.startswith("lm_head."):
                    tensors[key] = f.get_tensor(key)
        tmp_path = sf_path.with_suffix(".safetensors.rewrite_tmp")
        safetensors.torch.save_file(tensors, tmp_path)
        tmp_path.replace(sf_path)
        changed = True
        print(
            f"[merge] rewrote Gemma text-only weights: {sf_path} "
            f"({len(tensors)} tensors, dropped vision/projector keys)",
            flush=True,
        )
    return changed


def repair_text_only_gemma_merge(out_dir: Path) -> bool:
    """Fix config + weight key layout for Gemma 4B/12B dense merges saved from multimodal base."""
    out_dir = Path(out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        return False
    cfg_changed = _normalize_text_only_gemma_merge_config(out_dir)
    weights_changed = _normalize_text_only_gemma_merge_weights(out_dir)
    return cfg_changed or weights_changed


def run_merge_qs_lora(ns: argparse.Namespace) -> int:
    wall0 = time.perf_counter()
    started_at = utc_iso()
    run_root = Path(ns.run_root).expanduser().resolve() if ns.run_root else None

    qs_dir = Path(ns.qs_dir).expanduser().resolve() if getattr(ns, "qs_dir", None) else (
        Path(ns.high_adapter).expanduser().resolve().parent
    )
    preset = getattr(ns, "weight_preset", None) or "custom"
    preset_extra: dict[str, Any] = {}

    if preset != "custom":
        w_h, w_m, w_l, out_dir, preset_name, preset_extra = resolve_merge_preset(
            preset,
            qs_dir=qs_dir,
            judge_summary_jsonl=Path(ns.judge_summary).expanduser().resolve()
            if getattr(ns, "judge_summary", None)
            else None,
        )
        ns.weight_high, ns.weight_medium, ns.weight_low = w_h, w_m, w_l
        ns.output_dir = out_dir
        preset = preset_name
    else:
        out_dir = Path(ns.output_dir).expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        AdapterSpec("high", Path(ns.high_adapter), float(ns.weight_high)),
        AdapterSpec("medium", Path(ns.medium_adapter), float(ns.weight_medium)),
        AdapterSpec("low", Path(ns.low_adapter), float(ns.weight_low)),
    ]

    timing = MergeTiming()
    spans: list[dict[str, Any]] = []

    print("=== QS strat dense merge (high + medium + low only) ===", flush=True)
    for s in specs:
        print(f"  {s.name}: weight={s.weight} path={s.path}", flush=True)

    use_gpu_merge = bool(getattr(ns, "use_gpu_merge", False))
    stream_merge = bool(getattr(ns, "stream_merge", False)) or use_gpu_merge

    dtype = torch.bfloat16 if ns.bf16 else torch.float32
    print(f"Loading base {ns.base_model} (gpu_merge={use_gpu_merge}, stream={stream_merge}) ...", flush=True)
    t_load0 = time.perf_counter()
    if use_gpu_merge:
        import sys

        try:
            from ohioline_ft.multi_gpu import log_gpu_info, max_memory_map
            log_gpu_info()
            max_mem = max_memory_map(cpu_gib=0)
        except ImportError:
            max_mem = None
        model = AutoModelForCausalLM.from_pretrained(
            ns.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            max_memory=max_mem,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ns.base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    timing.load_base_s = time.perf_counter() - t_load0
    spans.append(timing_record("load_base_model", timing.load_base_s, base_model=ns.base_model))
    print(f"[TIME] load base: {timing.load_base_s:.1f}s ({fmt_hms(timing.load_base_s)})", flush=True)

    if stream_merge:
        n_applied, adapter_info = merge_adapters_streaming_to_model(model, specs, timing)
        timing.apply_delta_s = timing.combine_delta_s
        spans.append(
            timing_record(
                "streaming_merge_apply",
                timing.combine_delta_s,
                n_parameters=n_applied,
                stream_merge=True,
            )
        )
        print(
            f"[TIME] streaming merge+apply: {timing.combine_delta_s:.1f}s "
            f"({fmt_hms(timing.combine_delta_s)}) ({n_applied} params)",
            flush=True,
        )
    else:
        merged_delta, adapter_info = merge_adapter_deltas(specs, timing)
        spans.append(
            timing_record(
                "combine_delta",
                timing.combine_delta_s,
                n_modules=adapter_info["n_merged_keys"],
                per_adapter=timing.per_adapter_load_s,
            )
        )
        bd = timing.to_manifest_dict()["combine_breakdown"]
        print(
            f"[TIME] combine ΔW: {timing.combine_delta_s:.1f}s "
            f"(load adapters {bd['load_adapter_deltas_s']:.1f}s + "
            f"weighted sum {bd['weighted_accumulate_s']:.1f}s)",
            flush=True,
        )
        for name in timing.per_adapter_load_s:
            print(
                f"       {name}: load {timing.per_adapter_load_s[name]:.1f}s, "
                f"accumulate {timing.per_adapter_accumulate_s[name]:.1f}s",
                flush=True,
            )

        t_apply0 = time.perf_counter()
        n_applied = apply_merged_delta_to_model(model, merged_delta)
        timing.apply_delta_s = time.perf_counter() - t_apply0
        spans.append(timing_record("apply_delta_to_base", timing.apply_delta_s, n_parameters=n_applied))
        print(
            f"[TIME] apply ΔW to base: {timing.apply_delta_s:.1f}s ({fmt_hms(timing.apply_delta_s)}) "
            f"({n_applied} params)",
            flush=True,
        )

    print(f"Saving merged model to {out_dir} ...", flush=True)
    t_save0 = time.perf_counter()
    model.save_pretrained(out_dir)
    repair_text_only_gemma_merge(out_dir)
    timing.save_model_s = time.perf_counter() - t_save0
    spans.append(timing_record("save_merged_model", timing.save_model_s))

    t_tok0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(ns.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out_dir)
    timing.save_tokenizer_s = time.perf_counter() - t_tok0
    spans.append(timing_record("save_tokenizer", timing.save_tokenizer_s))

    timing.total_wall_s = time.perf_counter() - wall0
    finished_at = utc_iso()

    w_h, w_m, w_l = float(ns.weight_high), float(ns.weight_medium), float(ns.weight_low)

    manifest = {
        "schema": "qs_lora_dense_merge/v1",
        "method": "weighted_delta_sum",
        "weight_preset": preset,
        "preset_meta": preset_extra,
        "base_model": ns.base_model,
        "output_dir": str(out_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "host": socket.gethostname(),
        "env": {
            k: os.environ[k]
            for k in ("SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES")
            if os.environ.get(k)
        },
        "weights": {
            "high": w_h,
            "medium": w_m,
            "low": w_l,
        },
        "note": "B3 and other adapters are NOT included. Use separate output_dir per weight scheme for ablations.",
        "adapters": adapter_info["adapters"],
        "n_merged_modules": adapter_info["n_merged_keys"],
        "n_applied_parameters": n_applied,
        "timing": timing.to_manifest_dict(),
    }
    _write_timing_artifacts(out_dir=out_dir, run_root=run_root, manifest=manifest, spans=spans)

    print(f"Wrote {out_dir / 'qs_merge_manifest.json'}", flush=True)
    print(f"Wrote {out_dir / 'qs_merge_timing.json'}", flush=True)
    print(f"Wrote {out_dir / 'experiment' / 'merge_spans.jsonl'}", flush=True)
    if run_root:
        print(f"Appended {run_root / 'training_pipeline_log.jsonl'}", flush=True)
        print(f"Updated {run_root / 'qs_merge_timing_index.json'}", flush=True)
    print(
        f"[TIME] total: {timing.total_wall_s:.1f}s ({fmt_hms(timing.total_wall_s)}) — "
        f"combine={timing.combine_delta_s:.1f}s, "
        f"load_base={timing.load_base_s:.1f}s, "
        f"apply={timing.apply_delta_s:.1f}s, "
        f"save={timing.save_model_s + timing.save_tokenizer_s:.1f}s",
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    run_root = Path(__file__).resolve().parent / "experiments/repliqa/runs/repliqa_train_0-3"
    qs = run_root / "baselines/qs_strat"

    p = argparse.ArgumentParser(
        description="Dense merge QS strat LoRA adapters (high+medium+low ΔW sum only)."
    )
    p.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--high-adapter", type=Path, default=qs / "QS_strat_high_lora_r32")
    p.add_argument("--medium-adapter", type=Path, default=qs / "QS_strat_medium_lora_r16")
    p.add_argument("--low-adapter", type=Path, default=qs / "QS_strat_low_lora_r8")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
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
        help="equal=1,1,1 | tier=0.6,0.3,0.1 | frequency | high_med | low_heavy | inverted | equal_rank_tier",
    )
    p.add_argument("--qs-dir", type=Path, default=qs, help="baselines/qs_strat (for preset output paths)")
    p.add_argument(
        "--judge-summary",
        type=Path,
        default=run_root / "train/synthetic_qa_haiku_judge_summary.json",
        help="For frequency preset tier counts",
    )
    p.add_argument("--weight-high", type=float, default=1.0)
    p.add_argument("--weight-medium", type=float, default=1.0)
    p.add_argument("--weight-low", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", action="store_true", help="Save float32 weights.")
    p.add_argument(
        "--run-root",
        type=Path,
        default=run_root,
        help="RepLiQA run dir for training_pipeline_log.jsonl + qs_merge_timing_index.json",
    )
    p.add_argument(
        "--use-gpu-merge",
        action="store_true",
        help="Shard 70B+ base across GPUs during merge (avoids CPU RAM OOM).",
    )
    p.add_argument(
        "--stream-merge",
        action="store_true",
        help="Merge one LoRA module at a time (low CPU RAM; default with --use-gpu-merge).",
    )
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.no_bf16:
        args.bf16 = False
    raise SystemExit(run_merge_qs_lora(args))

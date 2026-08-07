"""
Effective-rank analysis of cross-model LoRA / AdaLoRA / QS-merged adapters.

Loads adapter ΔW per module (without materializing full dense deltas on large layers),
computes singular-value decay, Frobenius norms, and participation-rank metrics.

Usage:
  python -m thesis.cli analyze-adapter-effective-rank --preset crossover
  python -m thesis.cli analyze-adapter-effective-rank \\
    --run qwen25_3b:/fs/ess/.../cross_model/runs/qwen25_3b/repliqa \\
    --run qwen25_14b:/fs/ess/.../cross_model/runs/qwen25_14b/repliqa
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from safetensors import safe_open

THESIS_ROOT = Path(__file__).resolve().parent
DEFAULT_CROSS_ROOT = Path("/fs/ess/PAS2699/pratham2210/cross_model/runs")
DEFAULT_REF_REPLIQA = (
    THESIS_ROOT / "experiments" / "repliqa" / "runs" / "repliqa_train_0-3"
)
SCHEMA = "adapter_effective_rank/v1"

MERGE_WEIGHTS = (0.6, 0.3, 0.1)
TIER_DIRS = (
    ("high", "QS_strat_high_lora_r32"),
    ("medium", "QS_strat_medium_lora_r16"),
    ("low", "QS_strat_low_lora_r8"),
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_root: Path


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _adapter_weights_path(adapter_dir: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No adapter weights in {adapter_dir}")


def _load_lora_config(adapter_dir: Path) -> dict[str, Any]:
    return json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))


def _suffix_for_key(kind: str) -> str:
    return f".lora_{kind}"


def _iter_prefixes(adapter_dir: Path) -> list[str]:
    weights_path = _adapter_weights_path(adapter_dir)
    kinds: dict[str, set[str]] = {"A": set(), "B": set(), "E": set()}
    if weights_path.suffix == ".safetensors":
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                for kind in ("A", "B", "E"):
                    suf = _suffix_for_key(kind)
                    if key.endswith(suf) or key.endswith(f"{suf}.weight"):
                        prefix = key[: -len(suf)] if key.endswith(suf) else key[: -len(f"{suf}.weight")]
                        kinds[kind].add(prefix)
    else:
        tensors = torch.load(weights_path, map_location="cpu", weights_only=True)
        for key in tensors:
            for kind in ("A", "B", "E"):
                suf = _suffix_for_key(kind)
                if key.endswith(suf) or key.endswith(f"{suf}.weight"):
                    prefix = key[: -len(suf)] if key.endswith(suf) else key[: -len(f"{suf}.weight")]
                    kinds[kind].add(prefix)
    prefixes = sorted(kinds["A"] & kinds["B"])
    if not prefixes:
        raise ValueError(f"No LoRA A/B pairs in {weights_path}")
    return prefixes


def _tensor_key(prefix: str, kind: str) -> str:
    for candidate in (f"{prefix}.lora_{kind}", f"{prefix}.lora_{kind}.weight"):
        return candidate
    raise KeyError(prefix)


def _load_tensor(adapter_dir: Path, prefix: str, kind: str) -> torch.Tensor:
    weights_path = _adapter_weights_path(adapter_dir)
    keys = (_tensor_key(prefix, kind),)
    if weights_path.suffix == ".safetensors":
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            all_keys = set(f.keys())
            for key in (f"{prefix}.lora_{kind}", f"{prefix}.lora_{kind}.weight"):
                if key in all_keys:
                    return f.get_tensor(key).float()
    else:
        tensors = torch.load(weights_path, map_location="cpu", weights_only=True)
        for key in (f"{prefix}.lora_{kind}", f"{prefix}.lora_{kind}.weight"):
            if key in tensors:
                return tensors[key].float()
    raise KeyError(f"Missing {kind} for {prefix} in {weights_path}")


def _state_key(module_prefix: str) -> str:
    if module_prefix.startswith("base_model.model."):
        module_prefix = module_prefix[len("base_model.model.") :]
    return f"{module_prefix}.weight"


def _layer_index(state_key: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", state_key)
    return int(m.group(1)) if m else None


def _module_short(state_key: str) -> str:
    return state_key.split(".")[-2] if "." in state_key else state_key


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return dev


def _singular_values_product(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Singular values of lora_b @ lora_a without forming the full m×n product."""
    a = lora_a.float().to(device)
    b = lora_b.float().to(device)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected rank-2 LoRA factors, got {a.shape=} {b.shape=}")
    if a.shape[0] != b.shape[1]:
        raise ValueError(f"Inner rank mismatch: A{a.shape} B{b.shape}")
    _, rb = torch.linalg.qr(b, mode="reduced")
    _, ra = torch.linalg.qr(a.mT, mode="reduced")
    return torch.linalg.svdvals(rb @ ra.mT)


def _adalora_scale_and_rank(cfg: dict[str, Any], lora_e: torch.Tensor) -> tuple[float, float]:
    scaling = float(cfg.get("lora_alpha", 16))
    active = (lora_e.squeeze(-1).abs() > 1e-6).sum().item()
    ranknum = max(float(active), float(cfg.get("r", 1)), 1.0)
    return scaling / ranknum, ranknum


def _lora_scale(cfg: dict[str, Any]) -> float:
    r = float(cfg.get("r", 1))
    alpha = float(cfg.get("lora_alpha", r))
    if cfg.get("use_rslora"):
        return alpha / math.sqrt(r)
    return alpha / r


def module_metrics_from_adapter(
    adapter_dir: Path,
    *,
    device: torch.device,
    label: str = "",
) -> dict[str, dict[str, Any]]:
    adapter_dir = adapter_dir.expanduser().resolve()
    cfg = _load_lora_config(adapter_dir)
    peft_type = str(cfg.get("peft_type", "LORA")).upper()
    prefixes = _iter_prefixes(adapter_dir)
    out: dict[str, dict[str, Any]] = {}

    for i, prefix in enumerate(prefixes, start=1):
        if i == 1 or i % 50 == 0 or i == len(prefixes):
            print(f"  [{label}] module {i}/{len(prefixes)}", flush=True)
        a = _load_tensor(adapter_dir, prefix, "A")
        b = _load_tensor(adapter_dir, prefix, "B")
        scale = _lora_scale(cfg)
        ranknum = float(a.shape[0])
        if peft_type == "ADALORA":
            e = _load_tensor(adapter_dir, prefix, "E")
            scale, ranknum = _adalora_scale_and_rank(cfg, e)
            a = a * e
        singular = (_singular_values_product(a, b, device=device) * scale).clamp_min(0.0)
        singular_np = singular.detach().cpu().numpy()
        energy = singular_np**2
        total_energy = float(energy.sum())
        cum = np.cumsum(energy) / total_energy if total_energy > 0 else np.zeros_like(energy)
        rank_90 = int(np.searchsorted(cum, 0.90) + 1) if total_energy > 0 else 0
        eff_rank = (
            float((singular_np.sum() ** 2) / (energy.sum() + 1e-12)) if total_energy > 0 else 0.0
        )
        state_key = _state_key(prefix)
        out[state_key] = {
            "state_key": state_key,
            "layer": _layer_index(state_key),
            "module": _module_short(state_key),
            "frobenius": float(np.sqrt(total_energy)),
            "spectral_norm": float(singular_np.max()) if singular_np.size else 0.0,
            "effective_rank": eff_rank,
            "rank_90_energy": rank_90,
            "configured_rank": int(cfg.get("r", a.shape[0])),
            "active_rank": int(ranknum),
            "singular_values": [round(float(x), 6) for x in singular_np[:32]],
        }
    return out


def load_ours_merged_metrics(
    qs_dir: Path,
    *,
    device: torch.device,
    label: str = "",
) -> dict[str, dict[str, Any]]:
    """Weighted tier merge via stacked LoRA factors (rank ≤ 56, no dense ΔW)."""
    tier_paths = [(w, qs_dir / dirname) for (_, dirname), w in zip(TIER_DIRS, MERGE_WEIGHTS)]
    prefixes = _iter_prefixes(tier_paths[0][1])
    out: dict[str, dict[str, Any]] = {}

    for i, prefix in enumerate(prefixes, start=1):
        if i == 1 or i % 50 == 0 or i == len(prefixes):
            print(f"  [{label}] merged module {i}/{len(prefixes)}", flush=True)
        b_parts: list[torch.Tensor] = []
        a_parts: list[torch.Tensor] = []
        for weight, tier_dir in tier_paths:
            cfg = _load_lora_config(tier_dir)
            scale = _lora_scale(cfg)
            coeff = math.sqrt(max(weight * scale, 0.0))
            b_parts.append(coeff * _load_tensor(tier_dir, prefix, "B"))
            a_parts.append(coeff * _load_tensor(tier_dir, prefix, "A"))
        b_stack = torch.cat(b_parts, dim=1)
        a_stack = torch.cat(a_parts, dim=0)
        singular = _singular_values_product(a_stack, b_stack, device=device).clamp_min(0.0)
        singular_np = singular.detach().cpu().numpy()
        energy = singular_np**2
        total_energy = float(energy.sum())
        cum = np.cumsum(energy) / total_energy if total_energy > 0 else np.zeros_like(energy)
        rank_90 = int(np.searchsorted(cum, 0.90) + 1) if total_energy > 0 else 0
        eff_rank = (
            float((singular_np.sum() ** 2) / (energy.sum() + 1e-12)) if total_energy > 0 else 0.0
        )
        state_key = _state_key(prefix)
        out[state_key] = {
            "state_key": state_key,
            "layer": _layer_index(state_key),
            "module": _module_short(state_key),
            "frobenius": float(np.sqrt(total_energy)),
            "spectral_norm": float(singular_np.max()) if singular_np.size else 0.0,
            "effective_rank": eff_rank,
            "rank_90_energy": rank_90,
            "configured_rank": sum(int(_load_lora_config(p).get("r", 0)) for _, p in tier_paths),
            "active_rank": int((singular_np > 1e-6).sum()),
            "singular_values": [round(float(x), 6) for x in singular_np[:32]],
        }
    return out


def resolve_baselines(run_root: Path) -> dict[str, Path]:
    base = run_root / "baselines"
    qs = base / "qs_strat"
    paths = {
        "B3": base / "B3_all_lora_r16",
        "B5": base / "B5_adalora_r16",
        "QS_high": qs / "QS_strat_high_lora_r32",
        "QS_medium": qs / "QS_strat_medium_lora_r16",
        "QS_low": qs / "QS_strat_low_lora_r8",
    }
    missing = [k for k, p in paths.items() if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing adapters under {run_root}: {missing}")
    return paths


def aggregate_run_metrics(label: str, run_root: Path, *, device: torch.device) -> dict[str, Any]:
    paths = resolve_baselines(run_root)
    print(f"  loading B3", flush=True)
    b3 = module_metrics_from_adapter(paths["B3"], device=device, label=f"{label}/B3")
    print(f"  loading B5", flush=True)
    b5 = module_metrics_from_adapter(paths["B5"], device=device, label=f"{label}/B5")
    print(f"  loading Ours_merged", flush=True)
    ours = load_ours_merged_metrics(
        paths["QS_high"].parent, device=device, label=f"{label}/Ours"
    )
    conditions: dict[str, dict[str, dict[str, Any]]] = {
        "B3": b3,
        "B5": b5,
        "Ours_merged": ours,
    }
    for tier_key, path_key in (
        ("Ours_high", "QS_high"),
        ("Ours_medium", "QS_medium"),
        ("Ours_low", "QS_low"),
    ):
        print(f"  loading {tier_key}", flush=True)
        conditions[tier_key] = module_metrics_from_adapter(
            paths[path_key], device=device, label=f"{label}/{tier_key}"
        )

    summary: dict[str, Any] = {
        "label": label,
        "run_root": str(run_root),
        "conditions": {},
    }
    for cond, modules in conditions.items():
        if not modules:
            continue
        frob = [m["frobenius"] for m in modules.values()]
        eff = [m["effective_rank"] for m in modules.values()]
        r90 = [m["rank_90_energy"] for m in modules.values()]
        summary["conditions"][cond] = {
            "n_modules": len(modules),
            "mean_frobenius": float(np.mean(frob)),
            "median_frobenius": float(np.median(frob)),
            "mean_effective_rank": float(np.mean(eff)),
            "median_effective_rank": float(np.median(eff)),
            "mean_rank_90_energy": float(np.mean(r90)),
            "modules": modules,
        }
    return summary


def _mean_decay_curve(modules: dict[str, dict[str, Any]], max_k: int = 16) -> np.ndarray:
    curves: list[np.ndarray] = []
    for mod in modules.values():
        s = np.array(mod["singular_values"], dtype=float)
        if s.size == 0 or s[0] <= 0:
            continue
        s = s / s[0]
        pad = np.full(max_k, np.nan)
        pad[: min(max_k, s.size)] = s[:max_k]
        curves.append(pad)
    if not curves:
        return np.full(max_k, np.nan)
    return np.nanmean(np.stack(curves, axis=0), axis=0)


def _layer_frobenius(modules: dict[str, dict[str, Any]]) -> dict[int, float]:
    by_layer: dict[int, list[float]] = {}
    for mod in modules.values():
        layer = mod.get("layer")
        if layer is None:
            continue
        by_layer.setdefault(layer, []).append(mod["frobenius"])
    return {layer: float(np.mean(vals)) for layer, vals in sorted(by_layer.items())}


def plot_decay_curves(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), sharey=True, squeeze=False)
    colors = {"B3": "#4C72B0", "Ours_merged": "#55A868", "B5": "#C44E52"}
    for ax, run in zip(axes[0], runs):
        for cond in ("B3", "Ours_merged", "B5"):
            modules = run["conditions"].get(cond, {}).get("modules", {})
            curve = _mean_decay_curve(modules)
            xs = np.arange(1, len(curve) + 1)
            ax.plot(xs, curve, marker="o", label=cond, color=colors[cond])
        ax.set_title(run["label"])
        ax.set_xlabel("Singular value index")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("Normalized singular value")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Mean ΔW singular-value decay (RepLiQA adapters)", y=1.08)
    fig.tight_layout()
    path = out_dir / "svd_decay_by_scale.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_frobenius_by_layer(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, len(runs), figsize=(5 * len(runs), 7), sharex=True, squeeze=False)
    colors = {"B3": "#4C72B0", "Ours_merged": "#55A868", "B5": "#C44E52"}
    for col, run in enumerate(runs):
        for row, cond in enumerate(("Ours_merged", "B5")):
            modules = run["conditions"].get(cond, {}).get("modules", {})
            layer_frob = _layer_frobenius(modules)
            xs = list(layer_frob.keys())
            ys = list(layer_frob.values())
            axes[row, col].plot(xs, ys, marker=".", color=colors[cond], label=cond)
            axes[row, col].set_title(f"{run['label']} — {cond}")
            axes[row, col].set_ylabel("Mean ‖ΔW‖_F per layer")
            axes[row, col].grid(True, alpha=0.3)
        b3 = _layer_frobenius(run["conditions"].get("B3", {}).get("modules", {}))
        if b3:
            axes[0, col].plot(
                list(b3.keys()),
                list(b3.values()),
                linestyle="--",
                color=colors["B3"],
                alpha=0.7,
                label="B3",
            )
            axes[1, col].plot(
                list(b3.keys()),
                list(b3.values()),
                linestyle="--",
                color=colors["B3"],
                alpha=0.7,
                label="B3",
            )
    for ax in axes[-1]:
        ax.set_xlabel("Layer index")
    fig.suptitle("Per-layer Frobenius norm of ΔW", y=1.01)
    fig.tight_layout()
    path = out_dir / "frobenius_by_layer.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_b5_over_ours_ratio(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    labels: list[str] = []
    ratios: list[float] = []
    for run in runs:
        b5 = run["conditions"].get("B5", {})
        ours = run["conditions"].get("Ours_merged", {})
        if not b5 or not ours:
            continue
        labels.append(run["label"])
        ratios.append(b5["mean_frobenius"] / max(ours["mean_frobenius"], 1e-12))
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 4))
    ax.bar(labels, ratios, color="#C44E52")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("mean ‖ΔW‖_F(B5) / mean ‖ΔW‖_F(Ours)")
    ax.set_title("B5 vs Ours merged — global delta magnitude")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "b5_over_ours_frobenius_ratio.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _scale_meta(label: str) -> tuple[str, float]:
    """Map run label to (family, params_B) for scale-boundary plots."""
    mapping = {
        "Qwen-3B": ("Qwen", 3.0),
        "Qwen-14B": ("Qwen", 14.0),
        "Llama-3B-ref": ("Llama", 3.0),
        "Llama-8B": ("Llama", 8.0),
        "Llama-3.1-70B": ("Llama", 70.0),
    }
    if label not in mapping:
        raise KeyError(f"Unknown scale label {label!r}")
    return mapping[label]


def _draw_scale_transition(ax: plt.Axes) -> None:
    """Vertical marker at 14B — crossover point in Qwen sweep + start of large-model regime."""
    ax.axvline(14, color="#555555", linestyle=(0, (5, 4)), linewidth=1.4, zorder=1, alpha=0.85)
    ymin, ymax = ax.get_ylim()
    y_label = ymin + 0.04 * (ymax - ymin)
    ax.text(
        15.8,
        y_label,
        "Scale transition",
        fontsize=7.5,
        color="#444444",
        fontstyle="italic",
        ha="left",
        va="bottom",
        clip_on=True,
    )


def plot_scale_boundary(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    """
    Ch.5 scale-boundary figure: ‖ΔW‖_F and effective rank vs model size.
    """
    colors = {"B3": "#4C72B0", "Ours_merged": "#55A868", "B5": "#C44E52"}
    cond_labels = {
        "B3": "B3 (uniform LoRA)",
        "Ours_merged": "QS (tier merge)",
        "B5": "AdaLoRA",
    }
    families = ("Llama", "Qwen")

    by_family: dict[str, list[tuple[float, dict[str, Any]]]] = {f: [] for f in families}
    for run in runs:
        family, params_b = _scale_meta(run["label"])
        by_family[family].append((params_b, run))
    for family in families:
        by_family[family].sort(key=lambda x: x[0])

    # Fixed rows: title | meta | legend | caption | plots — no shared vertical band.
    fig = plt.figure(figsize=(10.0, 4.65))
    gs = GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[0.11, 0.06, 0.11, 0.22, 1.0],
        hspace=0.62,
        wspace=0.32,
        left=0.09,
        right=0.99,
        top=0.97,
        bottom=0.11,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.55,
        "Why QS wins at small scale and AdaLoRA wins at large scale",
        ha="center", va="center", fontsize=11, fontweight="bold",
    )

    ax_meta = fig.add_subplot(gs[1, :])
    ax_meta.axis("off")
    ax_meta.text(
        0.5, 0.5,
        "RepLiQA adapters  ·  Solid = Llama  ·  Dashed = Qwen",
        ha="center", va="center", fontsize=8, color="#555555",
    )

    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    legend_handles = [
        Line2D([0], [0], color=colors[c], lw=2, marker="o", markersize=5, label=cond_labels[c])
        for c in ("B3", "Ours_merged", "B5")
    ]
    ax_leg.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        fontsize=8,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    ax_cap_a = fig.add_subplot(gs[3, 0])
    ax_cap_a.axis("off")
    ax_cap_a.text(
        0.5, 0.92,
        "(a) Update magnitude increases with model size",
        ha="center", va="top", fontsize=9.5, fontweight="bold",
    )
    ax_cap_a.text(
        0.5, 0.08,
        "B3 → largest  ·  QS → moderate  ·  AdaLoRA → smallest",
        ha="center", va="bottom", fontsize=8, color="#333333",
    )

    ax_cap_b = fig.add_subplot(gs[3, 1])
    ax_cap_b.axis("off")
    ax_cap_b.text(
        0.5, 0.95,
        "(b) Effective update rank",
        ha="center", va="top", fontsize=9.5, fontweight="bold",
    )
    ax_cap_b.text(
        0.5, 0.52,
        "Small models: QS explores broader directions",
        ha="center", va="center", fontsize=8, color="#2d6a3e",
    )
    ax_cap_b.text(
        0.5, 0.08,
        "Large models: AdaLoRA targets high-value directions",
        ha="center", va="bottom", fontsize=8, color="#8b2e2e",
    )

    axes = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]
    metrics = (
        ("mean_frobenius", "Mean ‖ΔW‖_F per LoRA module"),
        ("mean_effective_rank", "Mean effective rank of ΔW"),
    )

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        ax.axvspan(3, 14, color="#55A868", alpha=0.05, zorder=0)
        ax.axvspan(14, 70, color="#C44E52", alpha=0.05, zorder=0)

        for family in families:
            if not by_family[family]:
                continue
            xs = [p for p, _ in by_family[family]]
            for cond in ("B3", "Ours_merged", "B5"):
                ys = [
                    r["conditions"].get(cond, {}).get(metric_key, np.nan)
                    for _, r in by_family[family]
                ]
                ax.plot(
                    xs,
                    ys,
                    linestyle="-" if family == "Llama" else "--",
                    marker="o" if family == "Llama" else "s",
                    color=colors[cond],
                    linewidth=2.0,
                    markersize=6,
                    zorder=3,
                )

        ax.set_xscale("log")
        ax.set_xticks([3, 8, 14, 70])
        ax.set_xticklabels(["3B", "8B", "14B", "70B"])
        ax.set_xlabel("Model size (parameters)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.25, zorder=0)
        ax.margins(x=0.06)

    axes[1].set_ylim(7, 33)
    for ax in axes:
        _draw_scale_transition(ax)

    llama_pts = by_family["Llama"]
    if llama_pts:
        _, run3 = llama_pts[0]
        qs_small = run3["conditions"]["Ours_merged"]["mean_effective_rank"]
        axes[1].annotate(
            "QS: broad rank spread",
            xy=(3, qs_small),
            xytext=(6.2, 24),
            fontsize=7.5,
            color="#2d6a3e",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.0, shrinkA=3),
        )
    if len(llama_pts) >= 3:
        _, run70 = llama_pts[-1]
        b3_70 = run70["conditions"]["B3"]["mean_effective_rank"]
        b5_70 = run70["conditions"]["B5"]["mean_effective_rank"]
        axes[1].annotate(
            "B3 rank collapse",
            xy=(70, b3_70),
            xytext=(24, 10),
            fontsize=7.5,
            color="#4C72B0",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.0),
        )
        axes[1].annotate(
            "AdaLoRA: high-value rank",
            xy=(70, b5_70),
            xytext=(24, 17),
            fontsize=7.5,
            color="#C44E52",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.0),
        )

    path = out_dir / "scale_boundary_ch5.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)

    # Single-panel Frobenius-only variant.
    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    ax2.axvspan(3, 14, color="#55A868", alpha=0.04, zorder=0)
    ax2.axvspan(14, 70, color="#C44E52", alpha=0.04, zorder=0)
    for family in families:
        if not by_family[family]:
            continue
        xs = [p for p, _ in by_family[family]]
        for cond in ("B3", "Ours_merged", "B5"):
            ys = [
                r["conditions"].get(cond, {}).get("mean_frobenius", np.nan)
                for _, r in by_family[family]
            ]
            ax2.plot(
                xs,
                ys,
                linestyle="-" if family == "Llama" else "--",
                marker="o" if family == "Llama" else "s",
                color=colors[cond],
                linewidth=2.2,
                markersize=7,
                label=f"{cond_labels[cond]} ({family})",
            )
    _draw_scale_transition(ax2)
    ax2.set_xscale("log")
    ax2.set_xticks([3, 8, 14, 70])
    ax2.set_xticklabels(["3B", "8B", "14B", "70B"])
    ax2.set_xlabel("Model size (parameters)")
    ax2.set_ylabel("Mean ‖ΔW‖_F per LoRA module")
    ax2.set_title(
        "Update magnitude vs model size\n"
        "B3 → largest  ·  QS → moderate  ·  AdaLoRA → smallest",
        fontsize=10,
        loc="left",
    )
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left", frameon=False)
    fig2.tight_layout()
    path2 = out_dir / "frobenius_by_scale.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    return path


def plot_effective_rank_bars(runs: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [r["label"] for r in runs]
    width = 0.25
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 4))
    for i, (cond, color) in enumerate(
        (("B3", "#4C72B0"), ("Ours_merged", "#55A868"), ("B5", "#C44E52"))
    ):
        vals = [r["conditions"].get(cond, {}).get("mean_effective_rank", np.nan) for r in runs]
        ax.bar(x + (i - 1) * width, vals, width=width, label=cond, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean participation-rank of ΔW")
    ax.set_title("Effective rank by model scale")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "effective_rank_by_scale.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def compact_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {"label": run["label"]}
        for cond in ("B3", "Ours_merged", "B5"):
            stats = run["conditions"].get(cond, {})
            row[f"{cond}_mean_frobenius"] = stats.get("mean_frobenius")
            row[f"{cond}_mean_effective_rank"] = stats.get("mean_effective_rank")
            row[f"{cond}_mean_rank_90"] = stats.get("mean_rank_90_energy")
        b5 = row.get("B5_mean_frobenius")
        ours = row.get("Ours_merged_mean_frobenius")
        if b5 is not None and ours:
            row["b5_over_ours_frobenius"] = b5 / ours
        rows.append(row)
    return {"comparison_rows": rows}


def _cross_run_root(model_slug: str, dataset: str) -> Path:
    if dataset == "repliqa":
        return DEFAULT_CROSS_ROOT / model_slug / "repliqa"
    if dataset == "quoref":
        return DEFAULT_CROSS_ROOT / model_slug / "quoref_qa_v1"
    raise ValueError(f"Unknown dataset {dataset!r}; want repliqa|quoref")


def default_presets(dataset: str = "repliqa") -> dict[str, list[RunSpec]]:
    ref_llama = (
        DEFAULT_REF_REPLIQA
        if dataset == "repliqa"
        else THESIS_ROOT / "experiments" / "quoref" / "runs" / "quoref_qa_v1"
    )
    return {
        "crossover": [
            RunSpec("Qwen-3B", _cross_run_root("qwen25_3b", dataset)),
            RunSpec("Qwen-14B", _cross_run_root("qwen25_14b", dataset)),
        ],
        "control": [
            RunSpec("Llama-3B-ref", ref_llama),
            RunSpec("Llama-8B", _cross_run_root("llama31_8b", dataset)),
        ],
        "scale70": [
            RunSpec("Qwen-14B", _cross_run_root("qwen25_14b", dataset)),
            RunSpec("Llama-3.1-70B", _cross_run_root("llama31_70b", dataset)),
        ],
        "70b": [
            RunSpec("Llama-3.1-70B", _cross_run_root("llama31_70b", dataset)),
        ],
        "full": [
            RunSpec("Qwen-3B", _cross_run_root("qwen25_3b", dataset)),
            RunSpec("Qwen-14B", _cross_run_root("qwen25_14b", dataset)),
            RunSpec("Llama-3B-ref", ref_llama),
            RunSpec("Llama-8B", _cross_run_root("llama31_8b", dataset)),
            RunSpec("Llama-3.1-70B", _cross_run_root("llama31_70b", dataset)),
        ],
    }


def parse_run_arg(value: str) -> RunSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL:RUN_ROOT")
    label, path = value.split(":", 1)
    return RunSpec(label.strip(), Path(path).expanduser().resolve())


def run_analysis(
    runs: list[RunSpec],
    output_dir: Path,
    *,
    device: torch.device,
    dataset: str = "repliqa",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}", flush=True)
    analyzed: list[dict[str, Any]] = []
    for spec in runs:
        print(f"Analyzing {spec.label} @ {spec.run_root}", flush=True)
        analyzed.append(aggregate_run_metrics(spec.label, spec.run_root, device=device))

    plots = {
        "svd_decay": str(plot_decay_curves(analyzed, output_dir)),
        "frobenius_by_layer": str(plot_frobenius_by_layer(analyzed, output_dir)),
        "b5_over_ours_ratio": str(plot_b5_over_ours_ratio(analyzed, output_dir)),
        "effective_rank": str(plot_effective_rank_bars(analyzed, output_dir)),
        "scale_boundary": str(plot_scale_boundary(analyzed, output_dir)),
        "frobenius_by_scale": str(output_dir / "frobenius_by_scale.png"),
    }

    summary = {
        "schema": SCHEMA,
        "created_at": utc_iso(),
        "device": str(device),
        "dataset": dataset,
        "runs": analyzed,
        "comparison": compact_summary(analyzed),
        "plots": plots,
    }

    # Drop per-module singular lists from saved JSON to keep file smaller.
    lean = json.loads(json.dumps(summary))
    for run in lean["runs"]:
        for cond in run.get("conditions", {}).values():
            for mod in cond.get("modules", {}).values():
                mod.pop("singular_values", None)

    out_json = output_dir / "effective_rank_summary.json"
    out_json.write_text(json.dumps(lean, indent=2), encoding="utf-8")

    full_json = output_dir / "effective_rank_full.json"
    full_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {full_json}")
    for name, path in plots.items():
        print(f"Plot {name}: {path}")
    return summary


def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "analyze-adapter-effective-rank",
        help="SVD / Frobenius analysis of B3 vs QS-merge vs AdaLoRA adapters",
    )
    p.add_argument(
        "--preset",
        choices=("crossover", "control", "scale70", "70b", "full"),
        default="full",
        help="crossover=Qwen 3B/14B; control=Llama 3B/8B; scale70=Qwen-14B+Llama-70B; 70b=70B only",
    )
    p.add_argument(
        "--dataset",
        choices=("repliqa", "quoref"),
        default="repliqa",
        help="Which cross-model run roots to use (repliqa or quoref_qa_v1)",
    )
    p.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        default=None,
        help="Override with LABEL:RUN_ROOT (repeatable)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: thesis/experiments/analysis/adapter_effective_rank/{dataset}",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="SVD compute device (auto prefers CUDA when available)",
    )

    def _run(ns: argparse.Namespace) -> int:
        presets = default_presets(ns.dataset)
        runs = ns.run if ns.run else presets[ns.preset]
        out = (
            Path(ns.output_dir).expanduser().resolve()
            if ns.output_dir
            else THESIS_ROOT / "experiments" / "analysis" / "adapter_effective_rank" / ns.dataset
        )
        device = _resolve_device(ns.device)
        run_analysis(runs, out, device=device, dataset=ns.dataset)
        return 0

    p.set_defaults(fn=_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_cli(sub)
    args = parser.parse_args()
    raise SystemExit(args.fn(args))

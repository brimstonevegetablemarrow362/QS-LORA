"""
Bootstrap confidence intervals for Bedrock judge GA and listwise win rates.

Paired GA: resample eval questions with replacement, recompute mean(Ours) - mean(B3).
Listwise: resample comparison rows, recompute win rate (lower rank wins).

Run all eval runs:
  python -m thesis.cli eval-bootstrap-ci --all
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCHEMA = "thesis_bootstrap_ci/v1"
DEFAULT_N_BOOT = 10_000
DEFAULT_CI = 0.95
DEFAULT_SEED = 42

THESIS_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = THESIS_ROOT / "experiments"

# Key paired GA comparisons (challenger, baseline) per eval judged dir parent name pattern.
# Also auto-pairs all Ours* vs B3* within each leaderboard.
EXTRA_PAIRED: dict[str, list[tuple[str, str]]] = {
    "repliqa_train_0-3": [
        ("Ours_tier_merge", "B3_lora_all"),
        ("Ours_tier_merge", "B5_adalora_all"),
        ("B5_adalora_all", "B3_lora_all"),
    ],
    "quoref_qa_v1": [
        ("Ours_tier_ctx", "B5_adalora_ctx"),
        ("B5_adalora_ctx", "B3_lora_ctx"),
        ("Ours_tier_ctx", "B1_cpt_ctx"),
        ("Ours_high_only_ctx", "Ours_tier_ctx"),
    ],
    "squad_qa_v1": [
        ("Ours_tier_ctx", "B5_adalora_ctx"),
        ("B5_adalora_ctx", "B3_lora_ctx"),
        ("Ours_tier_ctx", "B1_cpt_ctx"),
        ("Ours_high_only_ctx", "Ours_tier_ctx"),
    ],
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_ga_by_eval_id(judged_jsonl: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in _read_jsonl(judged_jsonl):
        eval_id = row.get("eval_id")
        if not eval_id:
            continue
        judge = row.get("llm_judge") or {}
        ga = judge.get("gold_alignment")
        if ga is None:
            continue
        try:
            scores[str(eval_id)] = float(ga)
        except (TypeError, ValueError):
            continue
    return scores


def bootstrap_ci(
    values: np.ndarray,
    *,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("bootstrap_ci: empty values")
    rng = np.random.default_rng(seed)
    n = values.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(values[idx])
    alpha = (1.0 - ci_level) / 2.0
    point = float(stat_fn(values))
    return {
        "n": int(n),
        "point": round(point, 4),
        "ci_low": round(float(np.quantile(boots, alpha)), 4),
        "ci_high": round(float(np.quantile(boots, 1.0 - alpha)), 4),
        "n_bootstrap": n_boot,
        "ci_level": ci_level,
        "seed": seed,
        "excludes_zero": bool((float(np.quantile(boots, alpha)) > 0) or (float(np.quantile(boots, 1.0 - alpha)) < 0)),
    }


def paired_ga_bootstrap(
    challenger_jsonl: Path,
    baseline_jsonl: Path,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    a = _load_ga_by_eval_id(challenger_jsonl)
    b = _load_ga_by_eval_id(baseline_jsonl)
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        raise ValueError(f"No overlapping eval_ids between {challenger_jsonl} and {baseline_jsonl}")
    diffs = np.array([a[i] - b[i] for i in common], dtype=np.float64)
    ci = bootstrap_ci(diffs, n_boot=n_boot, ci_level=ci_level, seed=seed)
    ci["challenger_mean"] = round(float(np.mean([a[i] for i in common])), 4)
    ci["baseline_mean"] = round(float(np.mean([b[i] for i in common])), 4)
    ci["delta_mean"] = ci["point"]
    ci["n_paired"] = len(common)
    return ci


def single_ga_bootstrap(
    judged_jsonl: Path,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    scores = _load_ga_by_eval_id(judged_jsonl)
    if not scores:
        raise ValueError(f"No GA scores in {judged_jsonl}")
    values = np.array(list(scores.values()), dtype=np.float64)
    return bootstrap_ci(values, n_boot=n_boot, ci_level=ci_level, seed=seed)


def listwise_pair_bootstrap(
    results_jsonl: Path,
    *,
    challenger: str,
    baseline: str,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    rows = _read_jsonl(results_jsonl)
    outcomes: list[float] = []
    for row in rows:
        if row.get("error"):
            continue
        ranks = row.get("ranks_by_condition") or {}
        if challenger not in ranks or baseline not in ranks:
            continue
        rc = int(ranks[challenger])
        rb = int(ranks[baseline])
        if rc < rb:
            outcomes.append(1.0)
        elif rc > rb:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)
    if not outcomes:
        raise ValueError(f"No paired listwise rows for {challenger} vs {baseline} in {results_jsonl}")
    values = np.array(outcomes, dtype=np.float64)
    ci = bootstrap_ci(values, n_boot=n_boot, ci_level=ci_level, seed=seed)
    ci["challenger"] = challenger
    ci["baseline"] = baseline
    ci["win_rate"] = ci["point"]
    ci["wins"] = int(np.sum(values == 1.0))
    ci["losses"] = int(np.sum(values == 0.0))
    ci["ties"] = int(np.sum(values == 0.5))
    return ci


def _run_name_from_eval_dir(eval_dir: Path) -> str:
    # .../runs/<run_name>/eval/judged -> run_name
    parts = eval_dir.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return eval_dir.parent.parent.name


def _discover_leaderboards(root: Path) -> list[Path]:
    return sorted(root.glob("**/eval/judged/judge_leaderboard.json"))


def _discover_listwise_results(root: Path) -> list[Path]:
    return sorted(root.glob("**/eval/listwise_rank*/listwise_rank_results.jsonl"))


def _infer_b3_baselines(conditions: list[str]) -> list[str]:
    return [c for c in conditions if c.startswith("B3_")]


def _suffix_family(name: str) -> str:
    if name.endswith("_no_ctx"):
        return "no_ctx"
    if name.endswith("_ctx"):
        return "ctx"
    if name.endswith("_merge") or name.endswith("_lora"):
        return "repliqa"
    if name.endswith("_all"):
        return "repliqa"
    return "other"


def _match_b3_for_ours(ours: str, b3_baselines: list[str]) -> str | None:
    fam = _suffix_family(ours)
    matches = [b for b in b3_baselines if _suffix_family(b) == fam]
    if matches:
        return matches[0]
    return b3_baselines[0] if len(b3_baselines) == 1 else None


def _infer_ours_challenger(conditions: list[str]) -> str | None:
    ours = [c for c in conditions if c.startswith("Ours_")]
    if not ours:
        return None
    # Prefer tier merge / tier ctx
    for pref in ("Ours_tier_merge", "Ours_tier_ctx", "Ours_tier_no_ctx"):
        if pref in ours:
            return pref
    return ours[0]


def bootstrap_leaderboard(
    leaderboard_path: Path,
    judged_dir: Path,
    *,
    n_boot: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    lb = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    run_name = _run_name_from_eval_dir(judged_dir)
    conditions = [row["condition"] for row in lb.get("leaderboard", [])]

    per_condition: dict[str, Any] = {}
    for row in lb.get("leaderboard", []):
        cond = row["condition"]
        judged_jsonl = judged_dir / cond / "bedrock_judge.jsonl"
        if not judged_jsonl.is_file():
            continue
        per_condition[cond] = single_ga_bootstrap(
            judged_jsonl, n_boot=n_boot, ci_level=ci_level, seed=seed
        )

    paired: dict[str, Any] = {}
    pairs: set[tuple[str, str]] = set()
    b3_list = _infer_b3_baselines(conditions)
    ours = _infer_ours_challenger(conditions)
    if b3_list and ours:
        b3 = _match_b3_for_ours(ours, b3_list)
        if b3:
            pairs.add((ours, b3))
    for c in conditions:
        if c.startswith("Ours_") and b3_list:
            b3 = _match_b3_for_ours(c, b3_list)
            if b3:
                pairs.add((c, b3))
    for pair in EXTRA_PAIRED.get(run_name, []):
        pairs.add(pair)

    for challenger, baseline in sorted(pairs):
        c_jsonl = judged_dir / challenger / "bedrock_judge.jsonl"
        b_jsonl = judged_dir / baseline / "bedrock_judge.jsonl"
        if not c_jsonl.is_file() or not b_jsonl.is_file():
            continue
        key = f"{challenger}_vs_{baseline}"
        try:
            paired[key] = paired_ga_bootstrap(
                c_jsonl, b_jsonl, n_boot=n_boot, ci_level=ci_level, seed=seed
            )
            paired[key]["challenger"] = challenger
            paired[key]["baseline"] = baseline
        except ValueError as e:
            paired[key] = {"error": str(e)}

    return {
        "run_name": run_name,
        "leaderboard_path": str(leaderboard_path),
        "n_conditions": len(per_condition),
        "per_condition_ga": per_condition,
        "paired_delta_ga": paired,
    }


def bootstrap_listwise_dir(
    results_jsonl: Path,
    *,
    n_boot: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    winrate_path = results_jsonl.with_name("listwise_winrate_vs_baseline.json")
    if winrate_path.is_file():
        wr = json.loads(winrate_path.read_text(encoding="utf-8"))
        baseline = wr["baseline"]
        challengers = [c for c in wr.get("pairwise", {}).keys()]
    else:
        rows = _read_jsonl(results_jsonl)
        conds: set[str] = set()
        for row in rows:
            conds.update((row.get("ranks_by_condition") or {}).keys())
        conds = sorted(conds)
        if len(conds) != 2:
            return {"results_jsonl": str(results_jsonl), "error": f"expected 2 conditions, got {conds}"}
        baseline = conds[0]
        challengers = [conds[1]]

    pairwise: dict[str, Any] = {}
    for challenger in challengers:
        try:
            pairwise[challenger] = listwise_pair_bootstrap(
                results_jsonl,
                challenger=challenger,
                baseline=baseline,
                n_boot=n_boot,
                ci_level=ci_level,
                seed=seed,
            )
        except ValueError as e:
            pairwise[challenger] = {"error": str(e)}

    return {
        "results_jsonl": str(results_jsonl),
        "baseline": baseline,
        "pairwise_win_rate": pairwise,
    }


def run_bootstrap_all(
    *,
    experiments_root: Path = EXPERIMENTS_ROOT,
    output_json: Path | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    ci_level: float = DEFAULT_CI,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_bootstrap": n_boot,
        "ci_level": ci_level,
        "seed": seed,
        "leaderboards": [],
        "listwise": [],
    }

    for lb_path in _discover_leaderboards(experiments_root):
        judged_dir = lb_path.parent
        try:
            block = bootstrap_leaderboard(
                lb_path, judged_dir, n_boot=n_boot, ci_level=ci_level, seed=seed
            )
            out["leaderboards"].append(block)
        except Exception as e:
            out["leaderboards"].append({"leaderboard_path": str(lb_path), "error": str(e)})

    for results_path in _discover_listwise_results(experiments_root):
        try:
            block = bootstrap_listwise_dir(
                results_path, n_boot=n_boot, ci_level=ci_level, seed=seed
            )
            out["listwise"].append(block)
        except Exception as e:
            out["listwise"].append({"results_jsonl": str(results_path), "error": str(e)})

    if output_json is None:
        output_json = experiments_root / "bootstrap_ci_summary.json"
    output_json = output_json.expanduser().resolve()
    output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    out["output_json"] = str(output_json)
    return out


def run_eval_bootstrap_ci(ns: argparse.Namespace) -> int:
    if ns.all_runs:
        summary = run_bootstrap_all(
            experiments_root=Path(ns.experiments_root).expanduser().resolve(),
            output_json=Path(ns.output_json).expanduser().resolve() if ns.output_json else None,
            n_boot=int(ns.n_bootstrap),
            ci_level=float(ns.ci_level),
            seed=int(ns.seed),
        )
        print(f"Wrote {summary['output_json']}", flush=True)
        print(f"Leaderboards: {len(summary['leaderboards'])}", flush=True)
        print(f"Listwise dirs: {len(summary['listwise'])}", flush=True)
        return 0

    if ns.leaderboard is None:
        print("Provide --all or --leaderboard", flush=True)
        return 1

    lb_path = Path(ns.leaderboard).expanduser().resolve()
    judged_dir = lb_path.parent
    block = bootstrap_leaderboard(
        lb_path,
        judged_dir,
        n_boot=int(ns.n_bootstrap),
        ci_level=float(ns.ci_level),
        seed=int(ns.seed),
    )
    out_path = judged_dir / "bootstrap_ci.json"
    out_path.write_text(json.dumps(block, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap CIs for judge GA and listwise win rates.")
    p.add_argument("--all", dest="all_runs", action="store_true", help="Process all experiment eval runs")
    p.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_ROOT)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--leaderboard", type=Path, default=None, help="Single judge_leaderboard.json")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--ci-level", type=float, default=DEFAULT_CI)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p


if __name__ == "__main__":
    raise SystemExit(run_eval_bootstrap_ci(build_arg_parser().parse_args()))

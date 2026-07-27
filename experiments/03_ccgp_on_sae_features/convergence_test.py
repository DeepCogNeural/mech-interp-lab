"""Experiment 03 addendum: converged SD scaling test only.

This deliberately does not import results from, write to, or regenerate the shipped
Experiment 03 artefacts.  It reuses the published stimulus construction, layer-8 SAE
arm definitions, dichotomy enumeration, and item-disjoint outer folds unchanged.
Only the probe solver differs: full-batch L-BFGS is stopped by a relative objective
change criterion, and lambda is chosen inside each outer training partition.

Run the timing gate first:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
    /Users/linghao/Github/mech-interp-lab/.venv/bin/python convergence_test.py --seeds 0

The file is resumable.  Each completed (seed, scaling, arm) row is atomically written
to convergence_results.json before the next arm begins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# These are required by the task, and must precede importing the published module.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ccgp_sae as published


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "convergence_results.json"
ARMS = ("resid", "sae", "sae_recon", "rand_exp")
SETTINGS = {
    "per_feature_zscore_inner_l2": "per_feature_zscore",
    "global_rms_inner_l2": "global_rms",
}
T_CRIT_95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
}
L2_STABILITY_FLOOR = 1e-5
L2_SELECTION_TOLERANCE_NATS = 1e-2
# Both thresholds are fixed before seeing this run.  They state what "precise and
# close" means on the bounded-accuracy scale; neither is chosen from the result.
PRECISION_HALF_WIDTH_MAX = 0.025
CLOSE_MEAN_DIFFERENCE_MAX = 0.025
DENSE_DIAGNOSTIC_MAX_DIFFERENCE = 0.010


@dataclass(frozen=True)
class Config:
    n_items: int = 96
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_folds: int = 5
    batch_size: int = 32
    selected_layer: int = 8
    probe_max_iterations: int = 500
    probe_relative_loss_tolerance: float = 1e-3
    probe_stable_iterations: int = 10
    # The grid is intentionally broad at both ends.  If a high edge is selected, the
    # selection routine appends two further orders of magnitude and refits, rather than
    # silently accepting an edge result.
    l2_grid_initial: tuple[float, ...] = (1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 10.0, 1000.0)
    max_l2_grid_expansions: int = 4


@dataclass
class ProbeFit:
    eval_accuracy: torch.Tensor
    train_accuracy: torch.Tensor
    diagnostics: dict[str, Any]


def _require_offline_cpu() -> None:
    required = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MPLBACKEND": "Agg",
    }
    mismatches = {key: (os.environ.get(key), value) for key, value in required.items()
                  if os.environ.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Offline environment requirement violated: {mismatches}")
    if published.device() != "cpu":
        raise RuntimeError("This addendum is CPU-only; refusing a non-CPU device.")


def _objective(head: nn.Linear, x: torch.Tensor, y: torch.Tensor, l2: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean logistic loss plus 0.5 * lambda * squared-weight norm, bias unpenalised."""
    bce = F.binary_cross_entropy_with_logits(head(x), y)
    # Divide by the number of simultaneous dichotomy heads so a lambda has identical
    # meaning for three main-effect validation heads and 35 SD heads.
    penalty = 0.5 * l2 * head.weight.square().sum() / head.weight.shape[0]
    return bce + penalty, bce, penalty


def fit_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    probe_seed: int,
    cfg: Config,
    l2: float,
) -> ProbeFit:
    """Converged full-batch L-BFGS logistic probe copied from the improved solver."""
    torch.manual_seed(probe_seed)
    x_train, x_eval = x_train.float().cpu(), x_eval.float().cpu()
    y_train, y_eval = y_train.float().cpu(), y_eval.float().cpu()
    head = nn.Linear(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.LBFGS(
        head.parameters(), lr=1.0, max_iter=1, max_eval=25, history_size=20,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )
    history: list[float] = []
    closure_calls = 0
    relative_change = math.inf
    converged = False
    final_objective = math.inf
    final_bce = math.inf
    final_penalty = math.inf
    for iteration in range(1, cfg.probe_max_iterations + 1):
        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            objective, _, _ = _objective(head, x_train, y_train, l2)
            objective.backward()
            return objective

        optimizer.step(closure)
        with torch.no_grad():
            objective, bce, penalty = _objective(head, x_train, y_train, l2)
            final_objective = float(objective)
            final_bce = float(bce)
            final_penalty = float(penalty)
        history.append(final_objective)
        if len(history) > cfg.probe_stable_iterations:
            reference = history[-1 - cfg.probe_stable_iterations]
            relative_change = abs(reference - final_objective) / max(abs(reference), 1e-12)
            if relative_change < cfg.probe_relative_loss_tolerance:
                converged = True
                break
    else:
        iteration = cfg.probe_max_iterations

    with torch.no_grad():
        result = ProbeFit(
            eval_accuracy=published.balanced_accuracy(head(x_eval), y_eval),
            train_accuracy=published.balanced_accuracy(head(x_train), y_train),
            diagnostics={
                "iterations": iteration,
                "closure_calls": closure_calls,
                "converged": converged,
                "relative_loss_change": relative_change,
                "objective": final_objective,
                "bce": final_bce,
                "l2_penalty": final_penalty,
                "unpenalised_evaluation_bce": float(
                    F.binary_cross_entropy_with_logits(head(x_eval), y_eval)
                ),
            },
        )
    if not converged:
        raise RuntimeError(
            f"Probe did not converge in {cfg.probe_max_iterations} L-BFGS iterations "
            f"(relative loss change={relative_change:.3g}, l2={l2:g})."
        )
    return result


def _inner_validation_split(
    outer_train: torch.Tensor,
    item_ids: torch.Tensor,
    seed: int,
    outer_fold: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact item-disjoint split used by the improved solver."""
    ids = torch.unique(item_ids[outer_train]).cpu().numpy()
    rng = np.random.default_rng(seed + 90_000 + outer_fold)
    rng.shuffle(ids)
    n_valid = max(1, int(round(0.2 * len(ids))))
    valid_ids = torch.tensor(ids[:n_valid], dtype=torch.long)
    valid = outer_train & torch.isin(item_ids, valid_ids)
    return outer_train & ~valid, valid


def _fit_l2_grid(
    tr: torch.Tensor,
    va: torch.Tensor,
    y_train: torch.Tensor,
    y_valid: torch.Tensor,
    seed: int,
    cfg: Config,
    grid: list[float],
) -> tuple[list[float], list[dict[str, Any]]]:
    scores, diagnostics = [], []
    for candidate_i, l2 in enumerate(grid):
        result = fit_probe(tr, y_train, va, y_valid, seed + candidate_i, cfg, l2)
        scores.append(-float(result.diagnostics["unpenalised_evaluation_bce"]))
        diagnostics.append(result.diagnostics)
    return scores, diagnostics


def select_l2(
    rep: torch.Tensor,
    stimuli: published.Stimuli,
    seed: int,
    cfg: Config,
    scale_mode: str,
) -> tuple[list[float], list[dict[str, Any]], list[torch.Tensor]]:
    """Nested L2 selection and fold-local zero-column masks from the improved solver."""
    selected: list[float] = []
    records: list[dict[str, Any]] = []
    fold_keeps: list[torch.Tensor] = []
    for outer_fold, (outer_train, _) in enumerate(published.folds(stimuli.item_ids, cfg.n_folds, seed)):
        # The unsupervised all-zero removal is nevertheless fit solely on outer training
        # activations.  An outer-test row has no route into the representation choice.
        keep = rep[outer_train].abs().amax(dim=0) > 0
        fold_keeps.append(keep)
        inner_train, inner_valid = _inner_validation_split(
            outer_train, stimuli.item_ids, seed, outer_fold,
        )
        tr, va = published.scale_features(
            rep[inner_train][:, keep], rep[inner_valid][:, keep], scale_mode,
        )
        grid = list(cfg.l2_grid_initial)
        attempts: list[dict[str, Any]] = []
        fallback_rule: str | None = None
        for expansion in range(cfg.max_l2_grid_expansions + 1):
            scores, diagnostics = _fit_l2_grid(
                tr, va, stimuli.factors[inner_train], stimuli.factors[inner_valid],
                seed + 30_000 + 100 * outer_fold + 1_000 * expansion, cfg, grid,
            )
            best_score = max(scores)
            eligible = [
                i for i, score in enumerate(scores)
                if grid[i] >= L2_STABILITY_FLOOR
                and score >= best_score - L2_SELECTION_TOLERANCE_NATS
            ]
            if not eligible:
                raise RuntimeError(
                    f"No L2 >= {L2_STABILITY_FLOOR:g} is within "
                    f"{L2_SELECTION_TOLERANCE_NATS:g} validation nats of the optimum."
                )
            best_i = max(eligible)  # predeclared conservative largest-near-optimum rule
            edge = best_i in (0, len(grid) - 1)
            attempts.append({
                "grid": grid.copy(),
                "negative_inner_validation_main_effect_bce": scores,
                "fit_diagnostics": diagnostics,
                "selected_l2": float(grid[best_i]),
                "selected_at_grid_edge": edge,
            })
            if not edge:
                break
            if best_i == len(grid) - 1 and expansion < cfg.max_l2_grid_expansions:
                # Widen the high end by two orders of magnitude.  Refit the full grid:
                # an old edge is not treated as a valid answer.
                grid.extend([grid[-1] * 100.0, grid[-1] * 10_000.0])
                continue
            # This is a documented last resort, never silent.  It is expected to be
            # unused with the broad grid; preserving the raw trace makes it auditable.
            fallback_rule = (
                "Maximum predeclared grid expansion reached; retained the largest "
                "eligible candidate and flagged this selection as an edge fallback."
            )
            break
        selected.append(float(grid[best_i]))
        records.append({
            "outer_fold": outer_fold,
            "selection_loss_tolerance_nats": L2_SELECTION_TOLERANCE_NATS,
            "stability_floor": L2_STABILITY_FLOOR,
            "selected_l2": float(grid[best_i]),
            "selected_at_grid_edge": bool(best_i in (0, len(grid) - 1)),
            "grid_expansions": len(attempts) - 1,
            "fallback_rule": fallback_rule,
            "outer_train_surviving_width": int(keep.sum()),
            "attempts": attempts,
        })
    return selected, records, fold_keeps


def sd_metric(
    rep: torch.Tensor,
    stimuli: published.Stimuli,
    seed: int,
    cfg: Config,
    dichotomies: list[dict[str, Any]],
    scale_mode: str,
    fold_l2: list[float],
    fold_keeps: list[torch.Tensor],
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    """All 35 balanced dichotomies as one 35-output probe over five item folds."""
    labels_by_condition = torch.stack([row["labels"] for row in dichotomies], dim=1)
    labels = labels_by_condition[stimuli.condition_ids]
    test_accs, train_accs, convergence = [], [], []
    for fold_index, (train, test) in enumerate(published.folds(stimuli.item_ids, cfg.n_folds, seed)):
        keep = fold_keeps[fold_index]
        tr, te = published.scale_features(rep[train][:, keep], rep[test][:, keep], scale_mode)
        result = fit_probe(
            tr, labels[train], te, labels[test], seed + 100 * fold_index, cfg, fold_l2[fold_index],
        )
        test_accs.append(result.eval_accuracy.numpy())
        train_accs.append(result.train_accuracy.numpy())
        convergence.append({
            "outer_fold": fold_index,
            "surviving_width": int(keep.sum()),
            **result.diagnostics,
        })
    test_mean = np.mean(test_accs, axis=0)
    train_mean = np.mean(train_accs, axis=0)
    by_type: dict[str, list[float]] = {}
    gap: dict[str, list[float]] = {}
    for index, dichotomy in enumerate(dichotomies):
        by_type.setdefault(dichotomy["type"], []).append(float(test_mean[index]))
        gap.setdefault(dichotomy["type"], []).append(float(train_mean[index] - test_mean[index]))
    by_type["overall"] = list(test_mean)
    gap["overall"] = list(train_mean - test_mean)
    return (
        {kind: float(np.mean(values)) for kind, values in by_type.items()},
        {kind: float(np.mean(values)) for kind, values in gap.items()},
        convergence,
    )


def _core_representations(
    residuals: torch.Tensor,
    sae: published.SAEWeights,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Published arm definitions, excluding the explicitly out-of-scope dense control."""
    sae_features = sae.encode(residuals)
    random_sparse, random_dense, target_l0 = published.random_expansion(residuals, sae, seed)
    # random_dense is necessary to form the published top-k random arm, but is never a
    # probe arm in this test.  Deleting it here prevents accidental dense-arm fitting.
    del random_dense
    reps = {
        "resid": residuals,
        "sae": sae_features,
        "sae_recon": sae.decode(sae_features),
        "rand_exp": random_sparse,
    }
    return reps, {
        "target_l0": target_l0,
        "mean_l0": {name: float((rep > 0).sum(1).float().mean()) for name, rep in reps.items()},
        "surviving_width": {
            name: int(published.prepare_features(rep)[0].shape[1]) for name, rep in reps.items()
        },
        "excluded_arms": ["rand_exp_dense", "rand_exp_width_matched", "sae_width_matched"],
    }


def _mean_ci(values: Iterable[float]) -> dict[str, float | int | None]:
    values_array = np.asarray(list(values), dtype=float)
    n = len(values_array)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    if n == 1:
        return {"mean": float(values_array[0]), "ci95": None, "n": 1}
    if n not in T_CRIT_95:
        raise RuntimeError(f"No Student-t 95% critical value configured for n={n}")
    return {
        "mean": float(values_array.mean()),
        "ci95": float(T_CRIT_95[n] * values_array.std(ddof=1) / math.sqrt(n)),
        "n": n,
    }


def _complete_seeds(payload: dict[str, Any], setting: str) -> list[int]:
    rows = payload["results"][setting]["per_seed_rows"]
    complete = []
    for seed in payload["config"]["seeds"]:
        if {row["arm"] for row in rows if row["seed"] == seed} == set(ARMS):
            complete.append(seed)
    return complete


def _summary_for_setting(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm and row["seed"] in seeds]
        if len(arm_rows) != len(seeds):
            continue
        arm_summary[arm] = {
            "sd": {
                kind: _mean_ci(row["sd"][kind] for row in arm_rows)
                for kind in ("main_effect", "two_way_xor", "three_way_parity", "unstructured", "overall")
            },
            "train_minus_test_gap": {
                kind: _mean_ci(row["train_minus_test_gap"][kind] for row in arm_rows)
                for kind in ("main_effect", "two_way_xor", "three_way_parity", "unstructured", "overall")
            },
            "lbfgs_iterations_sd_outer_fits": _iteration_summary(
                diagnostic for row in arm_rows for diagnostic in row["sd_convergence"]
            ),
        }
    return arm_summary


def _iteration_summary(diagnostics: Iterable[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = list(diagnostics)
    iterations = [row["iterations"] for row in diagnostics]
    closures = [row["closure_calls"] for row in diagnostics]
    return {
        "n_fits": len(diagnostics),
        "all_converged": bool(diagnostics) and all(row["converged"] for row in diagnostics),
        "iterations": {
            "minimum": min(iterations) if iterations else None,
            "maximum": max(iterations) if iterations else None,
            "mean": float(np.mean(iterations)) if iterations else None,
        },
        "closure_calls": {
            "minimum": min(closures) if closures else None,
            "maximum": max(closures) if closures else None,
            "mean": float(np.mean(closures)) if closures else None,
        },
    }


def _paired_sae_minus_random(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any] | None:
    if not seeds:
        return None
    differences: dict[str, list[float]] = {
        "main_effect": [], "two_way_xor": [], "three_way_parity": [], "unstructured": [], "overall": [],
    }
    for seed in seeds:
        sae = next(row for row in rows if row["seed"] == seed and row["arm"] == "sae")
        random = next(row for row in rows if row["seed"] == seed and row["arm"] == "rand_exp")
        for kind in differences:
            differences[kind].append(sae["sd"][kind] - random["sd"][kind])
    return {kind: _mean_ci(values) for kind, values in differences.items()}


def _l2_summary(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARMS:
        by_seed = []
        for seed in seeds:
            row = next((row for row in rows if row["seed"] == seed and row["arm"] == arm), None)
            if row is None:
                continue
            by_seed.append({
                "seed": seed,
                "selected_l2_by_outer_fold": row["selected_l2_by_outer_fold"],
                "any_grid_edge": any(trace["selected_at_grid_edge"] for trace in row["inner_validation_l2_selection"]),
                "any_fallback": any(trace["fallback_rule"] is not None for trace in row["inner_validation_l2_selection"]),
            })
        output[arm] = by_seed
    return output


def _refresh_summaries(payload: dict[str, Any]) -> None:
    summary: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    l2: dict[str, Any] = {}
    for setting in SETTINGS:
        rows = payload["results"][setting]["per_seed_rows"]
        seeds = _complete_seeds(payload, setting)
        summary[setting] = _summary_for_setting(rows, seeds)
        paired[setting] = _paired_sae_minus_random(rows, seeds)
        l2[setting] = _l2_summary(rows, seeds)
    payload["summary"] = summary
    payload["paired_sae_minus_rand_exp"] = paired
    payload["selected_l2_summary"] = l2

    common_seeds = sorted(set(_complete_seeds(payload, "per_feature_zscore_inner_l2")) &
                          set(_complete_seeds(payload, "global_rms_inner_l2")))
    if len(common_seeds) < len(payload["config"]["seeds"]):
        payload["scaling_decision"] = {
            "status": "pending",
            "completed_common_seeds": common_seeds,
            "reason": "The two-scaling decision is deliberately withheld until every planned seed is complete.",
        }
        return

    z = paired["per_feature_zscore_inner_l2"]["two_way_xor"]
    rms = paired["global_rms_inner_l2"]["two_way_xor"]
    dense_differences: dict[str, Any] = {}
    for arm in ("resid", "sae_recon"):
        differences = {}
        for kind in ("main_effect", "two_way_xor", "three_way_parity", "unstructured", "overall"):
            z_mean = summary["per_feature_zscore_inner_l2"][arm]["sd"][kind]["mean"]
            rms_mean = summary["global_rms_inner_l2"][arm]["sd"][kind]["mean"]
            differences[kind] = float(z_mean - rms_mean)
        dense_differences[arm] = differences
    dense_max = max(abs(value) for values in dense_differences.values() for value in values.values())
    dense_pass = dense_max <= DENSE_DIAGNOSTIC_MAX_DIFFERENCE
    precise = (z["ci95"] is not None and rms["ci95"] is not None and
               z["ci95"] <= PRECISION_HALF_WIDTH_MAX and rms["ci95"] <= PRECISION_HALF_WIDTH_MAX)
    close = abs(z["mean"] - rms["mean"]) <= CLOSE_MEAN_DIFFERENCE_MAX
    agrees = dense_pass and precise and close
    if not dense_pass:
        verdict = "diagnostic_failure_not_adjudicated"
        wording = "Dense-arm scaling invariance failed; the new solver is not a trustworthy adjudicator."
    elif agrees:
        verdict = "agree_precise_and_close_nonconvergence"
        wording = "Both paired estimates are individually precise and close, so the earlier swing was non-convergence."
    else:
        verdict = "disagree_or_imprecise_prior_geometry_not_adjudicated"
        wording = "The paired estimates differ or at least one remains imprecise; L2 prior geometry prevents adjudication by this probe family."
    payload["scaling_decision"] = {
        "status": "complete",
        "metric": "paired within-seed sae - rand_exp two-way-XOR shattering dimensionality",
        "comparison_rule": {
            "precise": f"each Student-t 95% half-width <= {PRECISION_HALF_WIDTH_MAX:.3f}",
            "close": f"absolute difference between scaling means <= {CLOSE_MEAN_DIFFERENCE_MAX:.3f}",
            "dense_diagnostic": f"every resid and sae_recon SD family mean changes by <= {DENSE_DIAGNOSTIC_MAX_DIFFERENCE:.3f}",
        },
        "per_feature_zscore": z,
        "global_rms": rms,
        "absolute_difference_between_scaling_means": abs(z["mean"] - rms["mean"]),
        "both_individually_precise": precise,
        "close": close,
        "dense_arm_diagnostic": {
            "pass": dense_pass,
            "maximum_absolute_difference": dense_max,
            "zscore_minus_global_rms_by_arm_and_type": dense_differences,
        },
        "verdict": verdict,
        "wording": wording,
    }


def _new_payload(cfg: Config) -> dict[str, Any]:
    return {
        "schema": "exp03-convergence-test-v1",
        "status": "partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(cfg), "seeds": list(cfg.seeds), "l2_grid_initial": list(cfg.l2_grid_initial)},
        "scope": {
            "metric": "shattering dimensionality only; CCGP omitted by design",
            "arms": list(ARMS),
            "excluded_arms": ["rand_exp_dense", "rand_exp_width_matched", "sae_width_matched"],
            "scalings": SETTINGS,
            "stimulus_and_arm_definitions": "Imported unchanged from published ccgp_sae.py.",
        },
        "solver": {
            "objective": "mean logistic loss + 0.5 * lambda * mean-over-heads squared weight norm; bias unpenalised",
            "optimizer": "full-batch L-BFGS with strong Wolfe line search",
            "convergence": "relative training-objective change below 1e-3 over ten accepted L-BFGS iterations",
            "iteration_cap": cfg.probe_max_iterations,
            "l2_selection": "inner item-disjoint main-effect held-out logistic loss; largest lambda >= 1e-5 within 1e-2 nats/example of optimum; edge selections expand the grid",
        },
        "source": {
            "published_script": "ccgp_sae.py",
            "improved_solver_source": "/tmp/scratch/ccgp_sae_improved.py",
            "selected_layer": cfg.selected_layer,
        },
        "results": {
            setting: {"scale_mode": mode, "per_seed_rows": []}
            for setting, mode in SETTINGS.items()
        },
        "per_seed_metadata": {},
        "runs": [],
    }


def _load_payload(cfg: Config) -> dict[str, Any]:
    if not RESULTS.exists():
        return _new_payload(cfg)
    payload = json.loads(RESULTS.read_text())
    if payload.get("schema") != "exp03-convergence-test-v1":
        raise RuntimeError(f"Refusing to resume unexpected schema in {RESULTS}")
    if payload["config"]["seeds"] != list(cfg.seeds):
        raise RuntimeError("Existing convergence_results.json has a different planned seed set.")
    return payload


def _write(payload: dict[str, Any]) -> None:
    _refresh_summaries(payload)
    temporary = RESULTS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(RESULTS)


def _row_exists(payload: dict[str, Any], setting: str, seed: int, arm: str) -> bool:
    return any(
        row["seed"] == seed and row["arm"] == arm
        for row in payload["results"][setting]["per_seed_rows"]
    )


def _all_seed_rows_exist(payload: dict[str, Any], seed: int) -> bool:
    return all(_row_exists(payload, setting, seed, arm) for setting in SETTINGS for arm in ARMS)


def _run_seed(
    payload: dict[str, Any],
    cfg: Config,
    model: Any,
    sae: published.SAEWeights,
    dichotomies: list[dict[str, Any]],
    seed: int,
) -> None:
    started = time.perf_counter()
    stimuli = published.build_stimuli(model.tokenizer, cfg.n_items, seed)
    residuals = published.collect_residuals(
        model, stimuli, (cfg.selected_layer,), "cpu", cfg.batch_size,
    )[cfg.selected_layer]
    reps, matching = _core_representations(residuals, sae, seed)
    payload["per_seed_metadata"][str(seed)] = {
        "stimuli": {
            "n_items": stimuli.attempted_items - stimuli.dropped_items,
            "n_sequences": int(stimuli.tokens.shape[0]),
            "attempted_items": stimuli.attempted_items,
            "dropped_items": stimuli.dropped_items,
            "readout_token_id": stimuli.readout_token_id,
        },
        "matching": matching,
    }
    _write(payload)
    for setting, scale_mode in SETTINGS.items():
        for arm in ARMS:
            if _row_exists(payload, setting, seed, arm):
                continue
            arm_started = time.perf_counter()
            selected_l2, selection_trace, fold_keeps = select_l2(
                reps[arm], stimuli, seed, cfg, scale_mode,
            )
            sd, gap, convergence = sd_metric(
                reps[arm], stimuli, seed, cfg, dichotomies, scale_mode, selected_l2, fold_keeps,
            )
            row = {
                "seed": seed,
                "arm": arm,
                "scale_mode": scale_mode,
                "sd": sd,
                "train_minus_test_gap": gap,
                "selected_l2_by_outer_fold": selected_l2,
                "inner_validation_l2_selection": selection_trace,
                "sd_convergence": convergence,
                "arm_wall_clock_seconds": time.perf_counter() - arm_started,
            }
            payload["results"][setting]["per_seed_rows"].append(row)
            _write(payload)
            print(
                f"seed={seed} setting={setting} arm={arm} "
                f"SD(two-way-XOR)={sd['two_way_xor']:.3f} "
                f"L2={','.join(f'{value:g}' for value in selected_l2)} "
                f"{row['arm_wall_clock_seconds']:.1f}s",
                flush=True,
            )
    print(f"seed={seed} complete in {time.perf_counter() - started:.1f}s", flush=True)


def _parse_seeds(raw: str | None, cfg: Config) -> tuple[int, ...]:
    if raw is None:
        return cfg.seeds
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or any(value not in cfg.seeds for value in values):
        raise ValueError(f"--seeds must be a non-empty subset of {cfg.seeds}")
    return values


def run(requested_seeds: tuple[int, ...]) -> dict[str, Any]:
    _require_offline_cpu()
    cfg = Config()
    payload = _load_payload(cfg)
    run_started = time.perf_counter()
    pending = tuple(seed for seed in requested_seeds if not _all_seed_rows_exist(payload, seed))
    if pending:
        model = published.load_model("cpu")
        sae = published.load_direct_res_jb(cfg.selected_layer)
        dichotomies = published.dichotomies()
        for seed in pending:
            _run_seed(payload, cfg, model, sae, dichotomies, seed)
    elapsed = time.perf_counter() - run_started
    payload["runs"].append({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seeds": list(requested_seeds),
        "completed_new_seeds": list(pending),
        "wall_clock_seconds": elapsed,
    })
    all_complete = all(_all_seed_rows_exist(payload, seed) for seed in cfg.seeds)
    payload["status"] = "complete" if all_complete else "partial"
    payload["cumulative_wall_clock_seconds"] = float(sum(run["wall_clock_seconds"] for run in payload["runs"]))
    _write(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", help="comma-separated planned seed subset; use --seeds 0 for the timing pilot")
    args = parser.parse_args()
    cfg = Config()
    requested_seeds = _parse_seeds(args.seeds, cfg)
    try:
        payload = run(requested_seeds)
    except BaseException as error:
        # Preserve an explicit error record without deleting rows written before it.
        if RESULTS.exists():
            payload = json.loads(RESULTS.read_text())
        else:
            payload = _new_payload(cfg)
        payload["status"] = "failed"
        payload["last_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        _write(payload)
        raise
    print(json.dumps({
        "status": payload["status"],
        "cumulative_wall_clock_seconds": payload["cumulative_wall_clock_seconds"],
        "scaling_decision": payload.get("scaling_decision"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

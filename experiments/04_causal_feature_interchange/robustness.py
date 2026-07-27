"""Amendment 2 robustness arms for Experiment 04.

Run from this directory:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
    .venv/bin/python robustness.py

This script never executes experiments 01--03.  It imports reusable Experiment 04
machinery and uses only additive residual deltas through ``PatchEngine``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import run_experiment as primary


HERE = Path(__file__).resolve().parent
PRIMARY_RESULTS = HERE / "run_results.json"
RESULTS = HERE / "robustness_results.json"
NOTES = HERE / "ROBUSTNESS_NOTES.md"
FIGURE = HERE / "figures" / "03_robustness.png"

LABEL = "specified after unblinding of the primary, before its own computation; never adjudicating"
R0_LABEL = "post-hoc, recomputable from run_results.json"
T_CRITICAL_DF4 = 2.776
INITIAL_CANDIDATES = 64
EXTENDED_CANDIDATES = 128
K_GRID_64 = (1, 2, 4, 8, 16, 32, 64)
GENERIC_TRAIN_FRACTION = 0.80


def stage(label: str, started: float, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"[{time.perf_counter() - started:8.1f}s] {label}{suffix}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite number cannot enter manifest: {value}")
    return value


def write_manifest(manifest: dict[str, Any]) -> None:
    RESULTS.write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def t_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    half_width = float(T_CRITICAL_DF4 * array.std(ddof=1) / math.sqrt(array.size))
    return {
        "values_by_seed": [float(value) for value in values],
        "mean": mean,
        "half_width_t4_95": half_width,
        "interval_t4_95": [mean - half_width, mean + half_width],
    }


def clipped_auc(rows: list[dict[str, Any]], field: str) -> float:
    x = np.log2(np.asarray([row["k"] for row in rows], dtype=float))
    y = np.asarray([row[field] for row in rows], dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def add_absolute_rows(rows: list[dict[str, Any]], resid_full_mean: float) -> list[dict[str, Any]]:
    if abs(resid_full_mean) < 1e-12:
        raise RuntimeError("Residual full anchor is zero in a robustness arm.")
    enriched = []
    for row in rows:
        item = dict(row)
        absolute_unclipped = float(item["metrics"]["mean_aligned_effect"] / resid_full_mean)
        item["absolute_unclipped_recovery_to_resid_full"] = absolute_unclipped
        item["absolute_recovery_to_resid_full"] = float(np.clip(absolute_unclipped, 0.0, 1.0))
        enriched.append(item)
    return enriched


def pair_noun(record: dict[str, Any]) -> tuple[str, str]:
    """Both lexical forms name the one subject noun in a single-flip pair."""
    return str(record["subject_singular"]), str(record["subject_plural"])


def r0_recomputations(primary_manifest: dict[str, Any]) -> dict[str, Any]:
    """All R0 quantities use recorded rows only; no model functions are called."""
    seed_rows = primary_manifest["seed_results"]
    noun_rows = []
    pooled_top: list[float] = []
    pooled_full: list[float] = []
    fullset_top: list[float] = []
    fullset_full: list[float] = []
    pooled_count = 0
    for seed_row in seed_rows:
        records = seed_row["stimulus_records_after_gate_A"]
        train_pairs = seed_row["splits"]["rank_train_pairs_after_gate_A"]
        eval_pairs = seed_row["splits"]["evaluation_pairs_after_gate_A"]
        train_nouns = {pair_noun(records[index]) for index in train_pairs}
        heldout_pairs = [index for index in eval_pairs if pair_noun(records[index]) not in train_nouns]
        qualifying_offsets = [offset for offset, pair in enumerate(eval_pairs) if pair in heldout_pairs]
        directed_offsets = [2 * offset + direction for offset in qualifying_offsets for direction in (0, 1)]
        top_row = next(row for row in seed_row["basis_results"]["sae"]["topk"] if row["k"] == 16)
        full_effect = np.asarray(seed_row["basis_results"]["sae"]["full"]["aligned_effect_by_directed_edit"], dtype=float)
        top_effect = np.asarray(top_row["metrics"]["aligned_effect_by_directed_edit"], dtype=float)
        fullset_top.extend(top_effect.tolist())
        fullset_full.extend(full_effect.tolist())
        if not directed_offsets:
            result = {
                "status": "not_run",
                "reason": "No evaluation pair had a subject noun absent from that seed's rank-training split.",
                "directed_edit_count": 0,
            }
        else:
            subset_top = top_effect[directed_offsets]
            subset_full = full_effect[directed_offsets]
            raw = float(subset_top.mean() / subset_full.mean())
            full_raw = float(top_row["unclipped_recovery"])
            result = {
                "status": "completed",
                "definition": "subject noun is the (singular, plural) lexical pair; both directed edits of each qualifying evaluation pair are retained",
                "rank_training_subject_nouns": sorted("/".join(noun) for noun in train_nouns),
                "qualifying_evaluation_pair_count": len(heldout_pairs),
                "directed_edit_count": len(directed_offsets),
                "within_basis_R_sae_16_unclipped": raw,
                "within_basis_R_sae_16_clipped": float(np.clip(raw, 0.0, 1.0)),
                "full_evaluation_R_sae_16_unclipped": full_raw,
                "full_evaluation_R_sae_16_clipped": float(top_row["recovery"]),
                "difference_subgroup_minus_full_unclipped": raw - full_raw,
            }
            pooled_top.extend(subset_top.tolist())
            pooled_full.extend(subset_full.tolist())
            pooled_count += len(directed_offsets)
        noun_rows.append({"label": R0_LABEL, "seed": seed_row["seed"], **result})
    if pooled_count:
        pooled_raw = float(np.mean(pooled_top) / np.mean(pooled_full))
        fullset_raw = float(np.mean(fullset_top) / np.mean(fullset_full))
        pooled = {
            "status": "completed",
            "directed_edit_count": pooled_count,
            "within_basis_R_sae_16_unclipped": pooled_raw,
            "within_basis_R_sae_16_clipped": float(np.clip(pooled_raw, 0.0, 1.0)),
            "full_evaluation_directed_edit_count": len(fullset_top),
            "full_evaluation_R_sae_16_unclipped": fullset_raw,
            "full_evaluation_R_sae_16_clipped": float(np.clip(fullset_raw, 0.0, 1.0)),
            "difference_subgroup_minus_full_unclipped": pooled_raw - fullset_raw,
        }
    else:
        pooled = {"status": "not_run", "reason": "No qualifying noun-disjoint directed edits across the five seeds.", "directed_edit_count": 0}

    top16_sets = [set(row["basis_results"]["sae"]["ranked_coordinates"][:16]) for row in seed_rows]
    common = set.intersection(*top16_sets)
    pairwise = []
    for left in range(len(top16_sets)):
        for right in range(left + 1, len(top16_sets)):
            overlap = len(top16_sets[left] & top16_sets[right])
            union = len(top16_sets[left] | top16_sets[right])
            pairwise.append({
                "seed_left": seed_rows[left]["seed"],
                "seed_right": seed_rows[right]["seed"],
                "overlap_count": overlap,
                "overlap_fraction_of_top16": overlap / 16.0,
                "jaccard": overlap / union,
            })
    stability = {
        "label": R0_LABEL,
        "status": "completed",
        "top16_coordinates_by_seed": {str(row["seed"]): sorted(top16_sets[index]) for index, row in enumerate(seed_rows)},
        "all_five_seed_intersection_count": len(common),
        "all_five_seed_intersection_coordinates": sorted(common),
        "mean_pairwise_overlap_count": float(np.mean([row["overlap_count"] for row in pairwise])),
        "mean_pairwise_overlap_fraction_of_top16": float(np.mean([row["overlap_fraction_of_top16"] for row in pairwise])),
        "mean_pairwise_jaccard": float(np.mean([row["jaccard"] for row in pairwise])),
        "pairwise_rows": pairwise,
    }

    sign_rows: dict[str, Any] = {}
    for basis in primary.ALL_BASES:
        per_seed = []
        correct_total = 0
        count_total = 0
        for seed_row in seed_rows:
            row16 = next(row for row in seed_row["basis_results"][basis]["topk"] if row["k"] == 16)
            aligned = np.asarray(row16["metrics"]["aligned_effect_by_directed_edit"], dtype=float)
            correct = int((aligned > 0.0).sum())
            count = int(aligned.size)
            correct_total += correct
            count_total += count
            per_seed.append({
                "seed": seed_row["seed"],
                "correct_directed_edits": correct,
                "directed_edit_count": count,
                "expected_sign_fraction": correct / count,
            })
        sign_rows[basis] = {
            "label": R0_LABEL,
            "status": "completed",
            "k": 16,
            "per_seed": per_seed,
            "pooled_correct_directed_edits": correct_total,
            "pooled_directed_edit_count": count_total,
            "pooled_expected_sign_fraction": correct_total / count_total,
        }
    return {
        "label": R0_LABEL,
        "noun_disjoint_transfer": {"per_seed": noun_rows, "pooled": pooled},
        "cross_seed_selected_latent_stability": stability,
        "sign_consistency_at_k16": sign_rows,
        "not_run": [],
    }


def reconstructed_seed_inputs(model: Any, primary_row: dict[str, Any], token_ids: dict[str, int]) -> dict[str, Any]:
    """Recreate the saved stimulus/split rows without rereading any primary gate."""
    seed = int(primary_row["seed"])
    primary.set_determinism(seed)
    all_stimuli = primary.build_stimuli(model.tokenizer, primary.REQUESTED_PAIRS, seed)
    retained_indices = [int(record["pair_index"]) for record in primary_row["stimulus_records_after_gate_A"]]
    stimuli = primary.subset_stimuli(all_stimuli, retained_indices)
    if stimuli.pair_records != primary_row["stimulus_records_after_gate_A"]:
        raise RuntimeError(f"Seed {seed}: deterministic stimulus reconstruction differs from saved primary rows.")
    clean = primary.clean_pass(model, stimuli.tokens, (primary.LAYER,))
    clean_number = primary.logit_difference(clean.logits, stimuli.lengths, token_ids[" is"], token_ids[" are"])
    train_pairs = list(primary_row["splits"]["rank_train_pairs_after_gate_A"])
    eval_pairs = list(primary_row["splits"]["evaluation_pairs_after_gate_A"])
    rank_base, rank_source, rank_signs = primary.directed_indices(len(stimuli.pair_records), train_pairs)
    eval_base, eval_source, eval_signs = primary.directed_indices(len(stimuli.pair_records), eval_pairs)
    rank_positions = primary.positions_for_kind(stimuli, rank_base, "both")
    eval_positions = primary.positions_for_kind(stimuli, eval_base, "both")
    return {
        "seed": seed,
        "stimuli": stimuli,
        "clean": clean,
        "clean_number": clean_number,
        "residuals": clean.residuals[primary.LAYER],
        "rank_base": rank_base,
        "rank_source": rank_source,
        "rank_signs": rank_signs,
        "eval_base": eval_base,
        "eval_source": eval_source,
        "eval_signs": eval_signs,
        "rank_positions": rank_positions,
        "eval_positions": eval_positions,
        "candidate_budget_primary": int(primary_row["candidate_coverage_trigger"]["candidate_budget_used"]),
        "k_grid_primary": tuple(int(k) for k in primary_row["candidate_coverage_trigger"]["k_grid"]),
    }


def residual_anchor(engine: Any, inputs: dict[str, Any], token_ids: dict[str, int], label: str) -> dict[str, Any]:
    return primary.patch_metrics(
        engine,
        stimuli=inputs["stimuli"],
        base_indices=inputs["eval_base"],
        source_indices=inputs["eval_source"],
        signs=inputs["eval_signs"],
        clean_logits=inputs["clean"].logits,
        clean_number=inputs["clean_number"],
        residuals=inputs["residuals"],
        positions=inputs["eval_positions"],
        deltas=primary.full_residual_delta(inputs["residuals"], inputs["eval_base"], inputs["eval_source"], inputs["eval_positions"]),
        token_ids=token_ids,
        label=label,
    )


def pca_spectrum(generic_pool: torch.Tensor) -> dict[str, Any]:
    centered = generic_pool - generic_pool.mean(dim=0)
    covariance = centered.T @ centered / float(generic_pool.shape[0] - 1)
    values = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    total = float(values.sum())
    cumulative = values.cumsum(dim=0) / max(total, 1e-12)
    def fraction(k: int) -> float:
        return float(cumulative[k - 1])
    def components_for(target: float) -> int | str:
        match = torch.nonzero(cumulative >= target, as_tuple=False)
        return int(match[0, 0] + 1) if match.numel() else "not reached"
    return {
        "total_variance": total,
        "explained_variance_fraction": {str(k): fraction(k) for k in (1, 16, 64, 128, 256, 512, 768)},
        "components_for_cumulative_variance": {str(target): components_for(target) for target in (0.50, 0.90, 0.95, 0.99)},
        "tail_bottom_quarter_variance_fraction": float(values[576:].sum() / max(total, 1e-12)),
        "tail_bottom_half_variance_fraction": float(values[384:].sum() / max(total, 1e-12)),
        "largest_eigenvalue": float(values[0]),
        "smallest_eigenvalue": float(values[-1]),
        "largest_to_smallest_eigenvalue_ratio": float(values[0] / values[-1].clamp_min(1e-12)),
    }


def rank_and_recover(
    *,
    engine: Any,
    inputs: dict[str, Any],
    token_ids: dict[str, int],
    basis_name: str,
    spec: Any,
    candidate_budget: int,
    k_grid: tuple[int, ...],
    resid_full_mean: float,
    label_prefix: str,
) -> dict[str, Any]:
    rank_delta, rank_source, rank_base = primary.feature_delta(
        spec.encode, inputs["residuals"], inputs["rank_base"], inputs["rank_source"], inputs["rank_positions"]
    )
    eval_delta, _, _ = primary.feature_delta(
        spec.encode, inputs["residuals"], inputs["eval_base"], inputs["eval_source"], inputs["eval_positions"]
    )
    candidates, prefilter = primary.candidate_prefilter_full(
        code_delta=rank_delta,
        source_code=rank_source,
        base_code=rank_base,
        signs=inputs["rank_signs"],
        decoder=spec.decoder,
        budget=candidate_budget,
        active_mode=spec.active_mode,
    )
    scores = primary.score_single_coordinates(
        engine,
        stimuli=inputs["stimuli"],
        base=inputs["rank_base"],
        source=inputs["rank_source"],
        signs=inputs["rank_signs"],
        clean_logits=inputs["clean"].logits,
        clean_number=inputs["clean_number"],
        residuals=inputs["residuals"],
        positions=inputs["rank_positions"],
        code_delta=rank_delta,
        decoder=spec.decoder,
        candidates=candidates,
        token_ids=token_ids,
        label=f"{label_prefix}_rank_{candidate_budget}",
    )
    ranked = candidates[scores.argsort(descending=True)]
    full = primary.patch_metrics(
        engine,
        stimuli=inputs["stimuli"],
        base_indices=inputs["eval_base"],
        source_indices=inputs["eval_source"],
        signs=inputs["eval_signs"],
        clean_logits=inputs["clean"].logits,
        clean_number=inputs["clean_number"],
        residuals=inputs["residuals"],
        positions=inputs["eval_positions"],
        deltas=primary.decoded_delta(eval_delta, spec.decoder),
        token_ids=token_ids,
        label=f"{label_prefix}_full",
    )
    selected = {k: ranked[:k] for k in k_grid}
    rows = primary.recovery_rows(
        engine,
        stimuli=inputs["stimuli"],
        base=inputs["eval_base"],
        source=inputs["eval_source"],
        signs=inputs["eval_signs"],
        clean_logits=inputs["clean"].logits,
        clean_number=inputs["clean_number"],
        residuals=inputs["residuals"],
        positions=inputs["eval_positions"],
        code_delta=eval_delta,
        decoder=spec.decoder,
        selected_by_k=selected,
        full_mean=full["mean_aligned_effect"],
        token_ids=token_ids,
        label=f"{label_prefix}_topk",
    )
    rows = add_absolute_rows(rows, resid_full_mean)
    return {
        "basis": basis_name,
        "candidate_budget": candidate_budget,
        "k_grid": list(k_grid),
        "prefilter": prefilter,
        "candidate_coordinates": candidates.tolist(),
        "single_coordinate_scores": [float(score) for score in scores],
        "ranked_coordinates": ranked.tolist(),
        "full": primary.raw_metric_row(full),
        "gate_C_E_full_over_E_resid": float(full["mean_aligned_effect"] / resid_full_mean),
        "topk": rows,
        "auc_within_basis": primary.normalised_auc(rows),
        "auc_absolute_to_resid_full": clipped_auc(rows, "absolute_recovery_to_resid_full"),
        "k50_within_basis": primary.k50(rows),
        "random_k_control": {
            "status": "not_run",
            "reason": "Amendment 2 specified the ranked recovery curve, not a new random-k comparator for this robustness arm.",
        },
    }


def r1_pca_budget(
    *,
    engine: Any,
    inputs: dict[str, Any],
    generic_pool: torch.Tensor,
    budget: int,
    token_ids: dict[str, int],
    resid_full_mean: float,
) -> dict[str, Any]:
    pca_spec, _, fit_meta = primary.fit_complete_bases(generic_pool[:budget])
    result = rank_and_recover(
        engine=engine,
        inputs=inputs,
        token_ids=token_ids,
        basis_name="pca",
        spec=pca_spec,
        candidate_budget=inputs["candidate_budget_primary"],
        k_grid=inputs["k_grid_primary"],
        resid_full_mean=resid_full_mean,
        label_prefix=f"R1_pca_{budget}",
    )
    return {
        "label": LABEL,
        "status": "completed",
        "fitting_budget_tokens": budget,
        "fit_metadata": fit_meta,
        "spectrum": pca_spectrum(generic_pool[:budget]),
        **result,
    }


def sae_coverage(sae: Any, decoder: torch.Tensor, inputs: dict[str, Any]) -> dict[str, Any]:
    base_x = primary.gather_positions(inputs["residuals"][inputs["eval_base"]], inputs["eval_positions"])
    source_x = primary.gather_positions(inputs["residuals"][inputs["eval_source"]], inputs["eval_positions"])
    active = ((sae.encode(base_x) > 0) | (sae.encode(source_x) > 0)).any(dim=(0, 1))
    columns = torch.nonzero(active).squeeze(1)
    coverage = float(primary.decoder_rows_nonzero(decoder, columns).float().mean())
    return {
        "evaluation_active_unit_count": int(columns.numel()),
        "decoder_nonzero_rows_for_evaluation_active_units": int(primary.decoder_rows_nonzero(decoder, columns).sum()),
        "evaluation_active_unit_coverage": coverage,
        "threshold": 0.95,
        "pass": bool(coverage >= 0.95),
    }


def gate_d_for_arm(topk: list[dict[str, Any]], residual_full: dict[str, Any]) -> dict[str, Any]:
    residual_number = torch.tensor(residual_full["number_effect"])
    residual_control = torch.tensor(residual_full["control_effect"])
    residual_S = float(residual_control.abs().mean() / residual_number.abs().mean().clamp_min(1e-12))
    first = next((row for row in topk if row["recovery"] >= 0.5), None)
    if first is None:
        return {
            "resid_full_S": residual_S,
            "k_star": "not reached within grid",
            "specificity_value": {"status": "not_run", "reason": "No first k with clipped within-basis R >= 0.5."},
            "cross_tense": {"status": "not_run", "reason": "No first k with clipped within-basis R >= 0.5."},
        }
    metrics = first["metrics"]
    number = torch.tensor(metrics["number_effect_by_directed_edit"])
    control = torch.tensor(metrics["control_effect_by_directed_edit"])
    tense = torch.tensor(metrics["tense_effect_by_directed_edit"])
    S = float(control.abs().mean() / number.abs().mean().clamp_min(1e-12))
    cross = float(((number * tense) > 0).float().mean())
    return {
        "resid_full_S": residual_S,
        "k_star": int(first["k"]),
        "specificity_value": S,
        "specificity_threshold_resid_full_plus_0_15": residual_S + 0.15,
        "specificity_pass": bool(S <= residual_S + 0.15),
        "cross_tense_same_sign_fraction": cross,
        "cross_tense_threshold": 0.80,
        "cross_tense_pass": bool(cross >= 0.80),
    }


def r2_variant(
    *,
    variant: str,
    decoder_fit: dict[str, Any],
    sae: Any,
    engine: Any,
    inputs: dict[str, Any],
    token_ids: dict[str, int],
    residual_full: dict[str, Any],
) -> dict[str, Any]:
    resid_full_mean = float(residual_full["mean_aligned_effect"])
    coverage = sae_coverage(sae, decoder_fit["decoder"], inputs)
    spec = primary.BasisSpec(
        name="sae_ridge",
        encode=sae.encode,
        decoder=decoder_fit["decoder"],
        active_mode="positive",
        note=f"{variant} exact dual-ridge decoder for the SAE code",
    )
    initial = rank_and_recover(
        engine=engine,
        inputs=inputs,
        token_ids=token_ids,
        basis_name="sae_ridge",
        spec=spec,
        candidate_budget=INITIAL_CANDIDATES,
        k_grid=K_GRID_64,
        resid_full_mean=resid_full_mean,
        label_prefix=f"R2_{variant}",
    )
    top64_raw = float(initial["topk"][-1]["unclipped_recovery"])
    extended = top64_raw < 0.8
    if extended:
        final = rank_and_recover(
            engine=engine,
            inputs=inputs,
            token_ids=token_ids,
            basis_name="sae_ridge",
            spec=spec,
            candidate_budget=EXTENDED_CANDIDATES,
            k_grid=K_GRID_64 + (128,),
            resid_full_mean=resid_full_mean,
            label_prefix=f"R2_{variant}",
        )
    else:
        final = initial
    gate_d = gate_d_for_arm(final["topk"], residual_full)
    fit_metadata = {key: value for key, value in decoder_fit.items() if key != "decoder"}
    return {
        "label": LABEL,
        "status": "completed",
        "variant": variant,
        "decoder_fit": fit_metadata,
        "coverage_check": coverage,
        "candidate_coverage_trigger": {
            "criterion": "extend to 128 iff this arm's E(top-64)/E(full) < 0.8",
            "top64_unclipped_recovery": top64_raw,
            "triggered": extended,
            "candidate_budget_used": final["candidate_budget"],
            "k_grid": final["k_grid"],
        },
        "gate_C": {
            "E_full_over_E_resid": final["gate_C_E_full_over_E_resid"],
            "band": [0.70, 1.30],
            "pass": bool(0.70 <= final["gate_C_E_full_over_E_resid"] <= 1.30),
        },
        "gate_D": gate_d,
        **final,
    }


def r1_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_budget = {}
    for budget in (2048, 8192):
        per_seed = [row["r1_pca"][str(budget)] for row in seed_rows]
        spectra = [row["spectrum"] for row in per_seed]
        explained = {
            component_count: {
                "values_by_seed": [float(spectrum["explained_variance_fraction"][component_count]) for spectrum in spectra],
                "mean": float(np.mean([spectrum["explained_variance_fraction"][component_count] for spectrum in spectra])),
            }
            for component_count in ("1", "16", "64", "128", "256", "512", "768")
        }
        by_budget[str(budget)] = {
            "label": LABEL,
            "auc_within_basis": t_summary([float(row["auc_within_basis"]) for row in per_seed]),
            "auc_absolute_to_resid_full": t_summary([float(row["auc_absolute_to_resid_full"]) for row in per_seed]),
            "k50_by_seed": [row["k50_within_basis"] for row in per_seed],
            "eigenvalue_spectrum_summary": {
                "explained_variance_fraction_by_leading_component_count": explained,
                "tail_bottom_quarter_variance_fraction": {
                    "values_by_seed": [float(spectrum["tail_bottom_quarter_variance_fraction"]) for spectrum in spectra],
                    "mean": float(np.mean([spectrum["tail_bottom_quarter_variance_fraction"] for spectrum in spectra])),
                },
                "tail_bottom_half_variance_fraction": {
                    "values_by_seed": [float(spectrum["tail_bottom_half_variance_fraction"]) for spectrum in spectra],
                    "mean": float(np.mean([spectrum["tail_bottom_half_variance_fraction"] for spectrum in spectra])),
                },
                "components_for_90_percent_variance_by_seed": [spectrum["components_for_cumulative_variance"]["0.9"] for spectrum in spectra],
            },
        }
    within_delta = [
        float(row["r1_pca"]["8192"]["auc_within_basis"] - row["r1_pca"]["2048"]["auc_within_basis"])
        for row in seed_rows
    ]
    absolute_delta = [
        float(row["r1_pca"]["8192"]["auc_absolute_to_resid_full"] - row["r1_pca"]["2048"]["auc_absolute_to_resid_full"])
        for row in seed_rows
    ]
    return {
        "label": LABEL,
        "by_fitting_budget_tokens": by_budget,
        "change_8192_minus_2048": {
            "within_basis": t_summary(within_delta),
            "absolute_to_resid_full": t_summary(absolute_delta),
        },
        "flatness_decision": {
            "status": "not_adjudicated",
            "reason": "Amendment 2 specifies measuring the curve but no numerical flatness threshold.",
        },
    }


def r2_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({name for row in seed_rows for name, value in row["r2_sae_ridge"].items() if value.get("status") == "completed"})
    output: dict[str, Any] = {"label": LABEL, "variants": {}}
    for variant in variants:
        sae_rows = [row["r2_sae_ridge"][variant] for row in seed_rows]
        pca_rows = [row["r1_pca"]["8192"] for row in seed_rows]
        within_diff = [float(sae_row["auc_within_basis"] - pca_row["auc_within_basis"]) for sae_row, pca_row in zip(sae_rows, pca_rows)]
        absolute_diff = [float(sae_row["auc_absolute_to_resid_full"] - pca_row["auc_absolute_to_resid_full"]) for sae_row, pca_row in zip(sae_rows, pca_rows)]
        output["variants"][variant] = {
            "label": LABEL,
            "sae_ridge_auc_within_basis": t_summary([float(row["auc_within_basis"]) for row in sae_rows]),
            "sae_ridge_auc_absolute_to_resid_full": t_summary([float(row["auc_absolute_to_resid_full"]) for row in sae_rows]),
            "sae_ridge_k50_by_seed": [row["k50_within_basis"] for row in sae_rows],
            "gate_C_E_full_over_E_resid": t_summary([float(row["gate_C"]["E_full_over_E_resid"]) for row in sae_rows]),
            "gate_D_specificity_by_seed": [row["gate_D"]["specificity_value"] for row in sae_rows],
            "auc_sae_ridge_minus_pca_8192": {
                "within_basis": t_summary(within_diff),
                "absolute_to_resid_full": t_summary(absolute_diff),
                "role": "robustness check; never adjudicating",
            },
        }
    return output


def make_figure(manifest: dict[str, Any]) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    primary_seeds = manifest["_primary_seed_results_for_figure"]
    robust_seeds = manifest["seed_results"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    axis = axes[0]
    sources = [
        ("primary sae", [row["basis_results"]["sae"]["topk"] for row in primary_seeds], "#0072B2", "-"),
        ("primary pca", [row["basis_results"]["pca"]["topk"] for row in primary_seeds], "#D55E00", "-"),
    ]
    for variant in manifest["r2_summary"]["variants"]:
        sources.append((
            f"sae_ridge ({variant})",
            [row["r2_sae_ridge"][variant]["topk"] for row in robust_seeds],
            "#009E73",
            "--" if variant != "pure_generic" else "-",
        ))
    for name, grids, color, style in sources:
        k = np.asarray(sorted({int(row["k"]) for grid in grids for row in grid}), dtype=float)
        values_by_k = [
            np.asarray([row["unclipped_recovery"] for grid in grids for row in grid if int(row["k"]) == int(current_k)], dtype=float)
            for current_k in k
        ]
        mean = np.asarray([values.mean() for values in values_by_k], dtype=float)
        half = np.asarray([
            T_CRITICAL_DF4 * values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else 0.0
            for values in values_by_k
        ], dtype=float)
        axis.plot(k, mean, marker="o", color=color, linestyle=style, label=name)
        axis.fill_between(k, mean - half, mean + half, color=color, alpha=0.13)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("edited coordinates k (log2 scale)")
    axis.set_ylabel("unclipped within-basis recovery")
    axis.set_title("R2 recovery curves (seed mean with t(4) 95% bands)")
    axis.axhline(0.5, color="#444444", linewidth=0.8, alpha=0.55)
    axis.legend(fontsize=8)

    axis = axes[1]
    x = np.asarray([2048, 8192], dtype=float)
    for normalization, color, marker in (("auc_within_basis", "#D55E00", "o"), ("auc_absolute_to_resid_full", "#7A3E9D", "s")):
        means = []
        halfs = []
        for budget in ("2048", "8192"):
            summary = manifest["r1_summary"]["by_fitting_budget_tokens"][budget][normalization]
            means.append(summary["mean"])
            halfs.append(summary["half_width_t4_95"])
        label = "within-basis AUC" if normalization == "auc_within_basis" else "absolute AUC to resid_full"
        axis.errorbar(x, means, yerr=halfs, color=color, marker=marker, capsize=4, label=label)
        values = np.asarray([
            [row["r1_pca"][str(budget)][normalization] for budget in (2048, 8192)]
            for row in robust_seeds
        ], dtype=float)
        for row in values:
            axis.plot(x, row, color=color, alpha=0.18, linewidth=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x)
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("PCA fitting tokens")
    axis.set_ylabel("PCA AUC")
    axis.set_title("R1 PCA fitting-budget comparison")
    axis.legend(fontsize=8)
    fig.suptitle(LABEL, fontsize=8)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def write_notes(manifest: dict[str, Any]) -> None:
    r0 = manifest["r0_recomputations"]
    r1 = manifest["r1_summary"]
    r2 = manifest["r2_summary"]
    noun = r0["noun_disjoint_transfer"]["pooled"]
    lines = [
        "# Experiment 04 Amendment 2 稳健性记录",
        "",
        f"- 标签：`{LABEL}`",
        f"- 墙钟：{manifest['wall_clock_seconds']:.1f} 秒。仅写入本文件、`robustness.py`、`robustness_results.json` 与 `figures/03_robustness.png`。",
        "- 本记录只描述给定坐标代码在一次加性残差差分写回下的测量；不对 SAE 是否改变模型任何计算作出表述。",
        "",
        "## R0：由原 manifest 复算",
        "",
        f"- 名词不相交 SAE k=16 合并：R={noun.get('within_basis_R_sae_16_clipped', 'not_run')}，有向编辑数={noun['directed_edit_count']}。",
        f"- 五种子 top-16 交集：{r0['cross_seed_selected_latent_stability']['all_five_seed_intersection_count']}；平均两两交集={r0['cross_seed_selected_latent_stability']['mean_pairwise_overlap_count']:.3f}。",
        "- k=16 符号一致性详见 JSON 的 `sign_consistency_at_k16`（每个基、每个种子和合并分子分母均保存）。",
        "",
        "## R1：PCA 拟合预算",
        "",
    ]
    for budget in ("2048", "8192"):
        row = r1["by_fitting_budget_tokens"][budget]
        lines.append(
            f"- {budget} tokens：within AUC={row['auc_within_basis']['mean']:.6f} ± {row['auc_within_basis']['half_width_t4_95']:.6f}；"
            f"absolute AUC={row['auc_absolute_to_resid_full']['mean']:.6f} ± {row['auc_absolute_to_resid_full']['half_width_t4_95']:.6f}。"
        )
    for norm, title in (("within_basis", "within"), ("absolute_to_resid_full", "absolute")):
        row = r1["change_8192_minus_2048"][norm]
        lines.append(f"- 8192−2048 {title} AUC：{row['mean']:+.6f} ± {row['half_width_t4_95']:.6f}。")
    spectrum_2048 = r1["by_fitting_budget_tokens"]["2048"]["eigenvalue_spectrum_summary"]
    spectrum_8192 = r1["by_fitting_budget_tokens"]["8192"]["eigenvalue_spectrum_summary"]
    lines.append(
        f"- 768 个完整 PCA 成分在两个预算下都捕获约 100% 方差；top-64 的种子均值为 "
        f"{spectrum_2048['explained_variance_fraction_by_leading_component_count']['64']['mean']:.6f}（2,048）和 "
        f"{spectrum_8192['explained_variance_fraction_by_leading_component_count']['64']['mean']:.6f}（8,192）；"
        f"底部一半特征值的方差份额为 {spectrum_2048['tail_bottom_half_variance_fraction']['mean']:.6f} 和 "
        f"{spectrum_8192['tail_bottom_half_variance_fraction']['mean']:.6f}。"
    )
    lines.extend([
        "- Amendment 2 没有定义“平坦”的数值阈值，因此只记录测量到的变化与 t(4) 区间，不作该分类裁定。",
        "",
        "## R2：sae_ridge",
        "",
    ])
    for variant, row in r2["variants"].items():
        gate_c = row["gate_C_E_full_over_E_resid"]
        within = row["sae_ridge_auc_within_basis"]
        absolute = row["sae_ridge_auc_absolute_to_resid_full"]
        diff_w = row["auc_sae_ridge_minus_pca_8192"]["within_basis"]
        diff_a = row["auc_sae_ridge_minus_pca_8192"]["absolute_to_resid_full"]
        lines.extend([
            f"- {variant}：Gate C E(full)/E(resid)={gate_c['mean']:.6f} ± {gate_c['half_width_t4_95']:.6f}；"
            f"AUC within={within['mean']:.6f} ± {within['half_width_t4_95']:.6f}；absolute={absolute['mean']:.6f} ± {absolute['half_width_t4_95']:.6f}。",
            f"- {variant} 的 sae_ridge−PCA(8192) robustness 差：within={diff_w['mean']:+.6f} ± {diff_w['half_width_t4_95']:.6f}；"
            f"absolute={diff_a['mean']:+.6f} ± {diff_a['half_width_t4_95']:.6f}。这不是裁定。",
        ])
    lines.extend([
        "",
        "## Amendment 留下的开放决定",
        "",
        "- R1 未指定把预算曲线称作平坦所需的阈值。",
        "- `run_results.json` 只保存了通用池的生成元数据、未保存激活张量；因此每个种子按已记录的固定种子重建一次 8,192-token 池，R1 的两个预算和 R2 在内存中共用该同一池。",
        "- R2 只有纯通用覆盖低于 0.95 时才运行混合版本；每个种子的实际选择和未运行项在 JSON 中显式列出。",
    ])
    NOTES.write_text("\n".join(lines) + "\n")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    primary_manifest = json.loads(PRIMARY_RESULTS.read_text())
    protected = {
        name: sha256(HERE / name)
        for name in ("DESIGN.md", "run_results.json", "run_experiment.py", "RUN_NOTES.md", "pilot.py", "pilot_results.json", "gate_c_diagnostic.py", "gate_c_diagnostic.json")
    }
    manifest: dict[str, Any] = {
        "schema": "exp04-causal-feature-interchange-robustness-v1; Amendment 2; CPU float32; additive residual deltas only; t(4)=2.776",
        "amendment_2_label": LABEL,
        "status": "running",
        "primary_manifest_reference": {
            "path": "run_results.json",
            "sha256": sha256(PRIMARY_RESULTS),
            "not_recomputed_or_adjudicated": True,
        },
        "protected_input_sha256_before": protected,
        "r0_recomputations": r0_recomputations(primary_manifest),
        "seed_results": [],
        "not_run": [],
        "wall_clock_seconds": 0.0,
    }
    write_manifest(manifest)
    stage("R0 complete", started)
    model = primary.load_model()
    sae = primary.load_direct_res_jb(primary.LAYER)
    token_ids = {key: int(value) for key, value in primary_manifest["token_ids"].items()}
    if sae.W_dec.shape != (24576, 768):
        raise RuntimeError(f"Unexpected SAE decoder shape {tuple(sae.W_dec.shape)}")

    for primary_row in primary_manifest["seed_results"]:
        seed = int(primary_row["seed"])
        stage(f"seed {seed}: reconstruct saved rows", started)
        inputs = reconstructed_seed_inputs(model, primary_row, token_ids)
        engine = primary.PatchEngine(model, start_at_layer8=True)
        residual_full = residual_anchor(engine, inputs, token_ids, "robust_resid_full")
        stage(f"seed {seed}: deterministic 8192-token pool", started)
        generic_pool, generation_meta = primary.generate_pool(model, seed)
        # ``generate_pool`` is deterministic and supplied by the Experiment 04 script;
        # this one in-memory pool is shared by both R1 budgets and R2 for this seed.
        stage(f"seed {seed}: R1 PCA 2048 and 8192", started)
        r1_2048 = r1_pca_budget(
            engine=engine, inputs=inputs, generic_pool=generic_pool, budget=2048,
            token_ids=token_ids, resid_full_mean=residual_full["mean_aligned_effect"],
        )
        r1_8192 = r1_pca_budget(
            engine=engine, inputs=inputs, generic_pool=generic_pool, budget=8192,
            token_ids=token_ids, resid_full_mean=residual_full["mean_aligned_effect"],
        )

        split = int(GENERIC_TRAIN_FRACTION * generic_pool.shape[0])
        generic_fit, generic_heldout = generic_pool[:split], generic_pool[split:]
        stage(f"seed {seed}: R2 pure-generic ridge", started)
        pure_fit = primary.fit_exact_dual_decoder(kind="sae_ridge", basis=sae, fit_x=generic_fit, heldout_x=generic_heldout)
        pure = r2_variant(
            variant="pure_generic", decoder_fit=pure_fit, sae=sae, engine=engine, inputs=inputs,
            token_ids=token_ids, residual_full=residual_full,
        )
        variants: dict[str, Any] = {"pure_generic": pure}
        mixed_fit: dict[str, Any] | None = None
        if not pure["coverage_check"]["pass"]:
            template_x = torch.cat((
                primary.gather_positions(inputs["residuals"][inputs["rank_base"]], inputs["rank_positions"]).reshape(-1, 768),
                primary.gather_positions(inputs["residuals"][inputs["rank_source"]], inputs["rank_positions"]).reshape(-1, 768),
            ), dim=0)
            mixed_fit_x = torch.cat((generic_fit, template_x), dim=0)
            stage(f"seed {seed}: R2 mixed ridge after pure coverage < 0.95", started)
            mixed_fit = primary.fit_exact_dual_decoder(kind="sae_ridge", basis=sae, fit_x=mixed_fit_x, heldout_x=generic_heldout)
            variants["mixed_generic_plus_rank_templates"] = r2_variant(
                variant="mixed_generic_plus_rank_templates", decoder_fit=mixed_fit, sae=sae, engine=engine, inputs=inputs,
                token_ids=token_ids, residual_full=residual_full,
            )
        else:
            variants["mixed_generic_plus_rank_templates"] = {
                "label": LABEL,
                "status": "not_run",
                "reason": "Pure-generic coverage met the pre-specified >= 0.95 threshold; Amendment 2 calls for the pure-generic version alone in that case.",
            }
        del pure_fit["decoder"]
        if mixed_fit is not None:
            del mixed_fit["decoder"]
        seed_result = {
            "label": LABEL,
            "seed": seed,
            "status": "completed",
            "generic_pool": {
                "reuse": "one deterministic 8192-token pool shared in memory by R1 budgets and R2",
                "metadata": generation_meta,
                "fit_rows_pure_generic": int(generic_fit.shape[0]),
                "heldout_rows_generic": int(generic_heldout.shape[0]),
            },
            "resid_full": primary.raw_metric_row(residual_full),
            "r1_pca": {"2048": r1_2048, "8192": r1_8192},
            "r2_sae_ridge": variants,
            "forward_timing_records": engine.records,
            "not_run": [
                {"item": name, "reason": value["reason"]}
                for name, value in variants.items() if value.get("status") == "not_run"
            ],
        }
        manifest["seed_results"].append(seed_result)
        manifest["wall_clock_seconds"] = time.perf_counter() - started
        write_manifest(manifest)
        stage(f"seed {seed}: complete", started, f"elapsed={manifest['wall_clock_seconds']:.1f}s")
        del generic_pool, generic_fit, generic_heldout, inputs, residual_full, engine
        gc.collect()

    manifest["r1_summary"] = r1_summary(manifest["seed_results"])
    manifest["r2_summary"] = r2_summary(manifest["seed_results"])
    manifest["protected_input_sha256_after"] = {
        name: sha256(HERE / name) for name in protected
    }
    manifest["protected_inputs_unchanged"] = manifest["protected_input_sha256_after"] == protected
    manifest["wall_clock_seconds"] = time.perf_counter() - started
    manifest["status"] = "completed"
    # Kept in-memory only while plotting; excluded before the final serialised manifest.
    manifest["_primary_seed_results_for_figure"] = primary_manifest["seed_results"]
    make_figure(manifest)
    del manifest["_primary_seed_results_for_figure"]
    write_manifest(manifest)
    write_notes(manifest)
    stage("all robustness arms complete", started, f"wall={manifest['wall_clock_seconds']:.1f}s")
    return manifest


def finalize_existing() -> dict[str, Any]:
    """Finish a completed seed set after a presentation-only failure, without model calls."""
    started = time.perf_counter()
    manifest = json.loads(RESULTS.read_text())
    if manifest.get("status") not in {"running", "completed"} or len(manifest.get("seed_results", [])) != 5:
        raise RuntimeError("--finalize-only requires exactly five already-completed seed rows.")
    primary_manifest = json.loads(PRIMARY_RESULTS.read_text())
    manifest["r1_summary"] = r1_summary(manifest["seed_results"])
    manifest["r2_summary"] = r2_summary(manifest["seed_results"])
    before = manifest["protected_input_sha256_before"]
    manifest["protected_input_sha256_after"] = {name: sha256(HERE / name) for name in before}
    manifest["protected_inputs_unchanged"] = manifest["protected_input_sha256_after"] == before
    manifest["status"] = "completed"
    manifest["_primary_seed_results_for_figure"] = primary_manifest["seed_results"]
    make_figure(manifest)
    del manifest["_primary_seed_results_for_figure"]
    write_manifest(manifest)
    write_notes(manifest)
    stage("finalised existing robustness seed set", started)
    return manifest


if __name__ == "__main__":
    finalize_existing() if "--finalize-only" in sys.argv else run()

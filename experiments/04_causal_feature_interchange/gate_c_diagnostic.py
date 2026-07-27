# Diagnostic only: this run informs a control-arm design decision, not an experiment result.
# It measures Gate C faithfulness E(full)/E_resid only; it never computes a headline statistic,
# any sae_topk/rand_topk comparison, AUC, or concentration/recovery curve.
#
# Run offline on CPU:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg .venv/bin/python gate_c_diagnostic.py

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pilot import (
    GATE_C_PAIRS,
    PILOT_PAIRS,
    RANK_TRAIN_PAIRS,
    SEED,
    PatchEngine,
    RandomBasis,
    build_stimuli,
    clean_pass,
    decoded_delta,
    decoder_rows_nonzero,
    directed_indices,
    dual_solve,
    feature_delta,
    full_residual_delta,
    gather_positions,
    generate_generic_activations,
    load_direct_res_jb,
    load_model,
    logit_difference,
    make_random_basis,
    positions_for_kind,
    r2_score,
    set_determinism,
    sparse_code_matrix,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "gate_c_diagnostic.json"

LAYER = 8
POSITION_SET = "both"
# The frozen pilot's actual matched L0 was 75.  The task's "72" is therefore
# treated as an approximate label, rather than silently changing the pilot arm.
NOMINAL_MATCHED_L0 = 72
GENERIC_BUDGETS = (2_048, 8_192, 32_768, 131_072)
L0_SWEEP_REQUESTED = (144, 288, 576, 768, 1_536)
GENERATION_POOL_BUDGET = 8_192
GENERIC_TRAIN_FRACTION = 0.80

# `fit_random_decoders` is the pilot's exact dual ridge.  Its dense Gram, system,
# and heldout-cross matrices scale quadratically in fit rows.  Above this cap the
# diagnostic records a non-run rather than risking an OOM or substituting a new
# estimator under the name "dual ridge".
MAX_EXACT_DUAL_FIT_ROWS = 8_000
TARGET_TOTAL_SECONDS = 15.0 * 60.0
HARD_TOTAL_SECONDS = 30.0 * 60.0
START_NEXT_CONFIG_GUARD_SECONDS = 60.0


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Non-finite float cannot enter JSON: {value}")
    return value


def write_results(manifest: dict[str, Any]) -> None:
    RESULTS.write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n")


def fit_rows_for_budget(generated_tokens: int, template_rows: int) -> int:
    return int(generated_tokens * GENERIC_TRAIN_FRACTION) + template_rows


def dual_memory_note(fit_rows: int, heldout_rows: int) -> str:
    gram_gib = fit_rows * fit_rows * 4 / 1024**3
    cross_gib = fit_rows * heldout_rows * 4 / 1024**3
    return (
        f"pilot exact dual ridge would materialize a {fit_rows}x{fit_rows} float32 Gram "
        f"({gram_gib:.2f} GiB) and a {heldout_rows}x{fit_rows} heldout-cross matrix "
        f"({cross_gib:.2f} GiB), before the lambda systems and solver workspace"
    )


def generic_subset(pool: torch.Tensor, requested_tokens: int) -> torch.Tensor:
    if requested_tokens > pool.shape[0]:
        raise ValueError(f"Need {requested_tokens} generic tokens; pool has {pool.shape[0]}")
    selected = pool[:requested_tokens].contiguous()
    if selected.shape[0] != requested_tokens:
        raise RuntimeError("Generic subset token count mismatch")
    return selected


def split_generic(generic_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(GENERIC_TRAIN_FRACTION * generic_x.shape[0])
    return generic_x[:split], generic_x[split:]


def relative_reconstruction_error(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float((target - prediction).square().sum() / target.square().sum().clamp_min(1e-12))


def fit_exact_dual_ridge_with_recorded_numeric_skips(
    basis: RandomBasis,
    fit_x: torch.Tensor,
    heldout_x: torch.Tensor,
) -> dict[str, Any]:
    """Pilot dual-ridge formula, with an auditable per-lambda numeric-failure record.

    The pilot's helper stops as soon as one float32 solve fails.  At a larger n,
    the tiny-lambda candidates can be below float32 factorisation resolution even
    though larger prescribed lambdas are well-conditioned.  They are therefore
    recorded as unavailable rather than silently changing lambda or the estimator.
    """
    fit_values, fit_indices = basis.sparse_topk(fit_x)
    G = sparse_code_matrix(fit_values, fit_indices)
    held_values, held_indices = basis.sparse_topk(heldout_x)
    G_held = sparse_code_matrix(held_values, held_indices)
    n = G.shape[0]
    gram = torch.sparse.mm(G, G.transpose(0, 1)).to_dense()
    held_cross = torch.sparse.mm(G_held, G.transpose(0, 1)).to_dense()
    identity = torch.eye(n, dtype=torch.float32)
    traces: list[dict[str, Any]] = []
    best: tuple[float, torch.Tensor, float, str] | None = None
    for lam in np.logspace(-4, 3, 8):
        lam_float = float(lam)
        try:
            solved, solver = dual_solve(gram + lam_float * identity, fit_x)
        except RuntimeError as exc:
            traces.append(
                {
                    "lambda": lam_float,
                    "status": "not_solved_float32",
                    "solver_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        held_r2 = r2_score(heldout_x, held_cross @ solved)
        traces.append(
            {
                "lambda": lam_float,
                "status": "solved",
                "heldout_generic_r2": held_r2,
                "solver": solver,
            }
        )
        if best is None or held_r2 > best[0]:
            best = (held_r2, solved, lam_float, solver)
    if best is None:
        raise RuntimeError("No prescribed dual-ridge lambda was numerically solvable in float32")
    selected_r2, selected_solved, selected_lambda, selected_solver = best
    W_ridge = torch.sparse.mm(G.transpose(0, 1), selected_solved)
    active = torch.unique(fit_indices).sort().values
    return {
        "ridge_decoder": W_ridge,
        "fit_rows": n,
        "fit_active_width": int(active.numel()),
        "lambda_grid": traces,
        "selected_lambda": selected_lambda,
        "selected_solver": selected_solver,
        "selected_heldout_generic_r2": selected_r2,
        "numeric_lambda_skips": sum(row["status"] != "solved" for row in traces),
    }


def faithfulness_effect(
    engine: PatchEngine,
    *,
    stimuli: Any,
    clean_d: torch.Tensor,
    base_indices: torch.Tensor,
    signs: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    deltas: torch.Tensor,
    is_id: int,
    are_id: int,
    label: str,
) -> float:
    """Gate-C E(delta): mean signed logit movement, without ranking calculations."""
    logits = engine.run(
        layer=LAYER,
        base_tokens=stimuli.tokens[base_indices],
        base_residual=residuals[base_indices],
        positions=positions,
        deltas=deltas,
        label=label,
    )
    patched_d = logit_difference(logits, stimuli.lengths[base_indices], is_id, are_id)
    effect = patched_d - clean_d[base_indices]
    return float((effect * signs).mean())


def evaluation_active_coverage(
    basis: RandomBasis,
    decoder: torch.Tensor,
    residuals: torch.Tensor,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[float, int]:
    _, source_code, base_code = feature_delta(basis.encode, residuals, base_indices, source_indices, positions)
    active = ((source_code > 0) | (base_code > 0)).any(dim=(0, 1))
    columns = torch.nonzero(active).squeeze(1)
    if columns.numel() == 0:
        raise RuntimeError("No evaluation-active random units; Gate-C coverage is undefined")
    coverage = float(decoder_rows_nonzero(decoder, columns).float().mean())
    return coverage, int(columns.numel())


def run_random_configuration(
    *,
    label: str,
    generated_tokens: int,
    basis: RandomBasis,
    generic_x: torch.Tensor,
    train_template_x: torch.Tensor,
    engine: PatchEngine,
    stimuli: Any,
    clean_d: torch.Tensor,
    gate_base: torch.Tensor,
    gate_source: torch.Tensor,
    gate_signs: torch.Tensor,
    gate_positions: torch.Tensor,
    residuals: torch.Tensor,
    is_id: int,
    are_id: int,
    resid_anchor: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    generic_fit, generic_heldout = split_generic(generic_x)
    fit_x = torch.cat((generic_fit, train_template_x), dim=0)
    decoders = fit_exact_dual_ridge_with_recorded_numeric_skips(basis, fit_x, generic_heldout)
    heldout_values, heldout_indices = basis.sparse_topk(generic_heldout)
    # This dense heldout code exists only for the diagnostic reconstruction error;
    # the dual fit itself remains the pilot's sparse-code implementation.
    heldout_code = torch.zeros((generic_heldout.shape[0], basis.R.shape[1]), dtype=torch.float32)
    heldout_code.scatter_(1, heldout_indices, heldout_values)
    heldout_prediction = heldout_code @ decoders["ridge_decoder"]
    rel_error = relative_reconstruction_error(generic_heldout, heldout_prediction)
    del heldout_code, heldout_prediction, heldout_values, heldout_indices

    random_diff, _, _ = feature_delta(basis.encode, residuals, gate_base, gate_source, gate_positions)
    random_effect = faithfulness_effect(
        engine,
        stimuli=stimuli,
        clean_d=clean_d,
        base_indices=gate_base,
        signs=gate_signs,
        residuals=residuals,
        positions=gate_positions,
        deltas=decoded_delta(random_diff, decoders["ridge_decoder"]),
        is_id=is_id,
        are_id=are_id,
        label=label,
    )
    coverage, active_width = evaluation_active_coverage(
        basis, decoders["ridge_decoder"], residuals, gate_base, gate_source, gate_positions
    )
    elapsed = time.perf_counter() - started
    result = {
        "status": "completed",
        "generated_token_budget": generated_tokens,
        "per_sample_l0": basis.target_l0,
        "fit_sample_size": int(decoders["fit_rows"]),
        "generic_fit_rows": int(generic_fit.shape[0]),
        "train_split_template_rows": int(train_template_x.shape[0]),
        "heldout_generic_rows": int(generic_heldout.shape[0]),
        "selected_lambda": float(decoders["selected_lambda"]),
        "selected_solver": decoders["selected_solver"],
        "numeric_lambda_skips": int(decoders["numeric_lambda_skips"]),
        "heldout_generic_reconstruction_r2": float(decoders["selected_heldout_generic_r2"]),
        "heldout_generic_relative_reconstruction_error": rel_error,
        "evaluation_active_unit_coverage": coverage,
        "evaluation_active_unit_count": active_width,
        "E_full_over_E_resid": random_effect / resid_anchor,
        "mean_aligned_E_full": random_effect,
        "elapsed_seconds": elapsed,
        "lambda_grid": decoders["lambda_grid"],
    }
    print(
        f"{label} | L0={basis.target_l0} fit_rows={result['fit_sample_size']} "
        f"lambda={result['selected_lambda']:.4g} R2={result['heldout_generic_reconstruction_r2']:.4f} "
        f"coverage={coverage:.4f} E(full)/E_resid={result['E_full_over_E_resid']:.4f} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    del decoders, random_diff, fit_x
    gc.collect()
    return result


def not_run_entry(*, generated_tokens: int, l0: int | None, reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "generated_token_budget": generated_tokens,
        "per_sample_l0": l0,
        "reason": reason,
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    set_determinism(SEED)
    manifest: dict[str, Any] = {
        "schema": "exp04-gate-c-diagnostic-v1; faithfulness-only; CPU float32; no headline statistics",
        "status": "running",
        "purpose": "Diagnostic for the random control-arm design decision; not an experiment result.",
        "headline_statistic": "not_computed",
        "configuration": {
            "layer": LAYER,
            "position_set": POSITION_SET,
            "pair_construction": "pilot.py single-flip pairs, both directions",
            "patching": "additive residual delta only",
            "random_encoder_seed": SEED,
            "pilot_actual_matched_l0": None,
            "nominal_task_matched_l0": NOMINAL_MATCHED_L0,
            "generic_train_fraction": GENERIC_TRAIN_FRACTION,
            "exact_dual_fit_row_cap": MAX_EXACT_DUAL_FIT_ROWS,
        },
        "sweep_A_ridge_fitting_power": {},
        "sweep_B_sparsity_ceiling": {},
        "reference_points": {},
        "not_run": [],
    }
    try:
        model = load_model()
        sae = load_direct_res_jb(LAYER)
        is_id = int(model.tokenizer(" is", add_special_tokens=False)["input_ids"][0])
        are_id = int(model.tokenizer(" are", add_special_tokens=False)["input_ids"][0])

        stimuli = build_stimuli(model.tokenizer, PILOT_PAIRS, SEED)
        clean = clean_pass(model, stimuli.tokens, (LAYER,))
        clean_d = logit_difference(clean.logits, stimuli.lengths, is_id, are_id)
        residuals = clean.residuals[LAYER]
        gate_pairs = list(range(RANK_TRAIN_PAIRS, RANK_TRAIN_PAIRS + GATE_C_PAIRS))
        gate_base, gate_source, gate_signs = directed_indices(PILOT_PAIRS, gate_pairs)
        gate_positions = positions_for_kind(stimuli, gate_base, POSITION_SET)
        engine = PatchEngine(model, start_at_layer8=True)

        resid_effect = faithfulness_effect(
            engine,
            stimuli=stimuli,
            clean_d=clean_d,
            base_indices=gate_base,
            signs=gate_signs,
            residuals=residuals,
            positions=gate_positions,
            deltas=full_residual_delta(residuals, gate_base, gate_source, gate_positions),
            is_id=is_id,
            are_id=are_id,
            label="diagnostic_resid_full",
        )
        if abs(resid_effect) < 1e-12:
            raise RuntimeError("Gate-C residual anchor is zero")

        sae_diff, sae_source_code, sae_base_code = feature_delta(sae.encode, residuals, gate_base, gate_source, gate_positions)
        sae_effect = faithfulness_effect(
            engine,
            stimuli=stimuli,
            clean_d=clean_d,
            base_indices=gate_base,
            signs=gate_signs,
            residuals=residuals,
            positions=gate_positions,
            deltas=decoded_delta(sae_diff, sae.W_dec),
            is_id=is_id,
            are_id=are_id,
            label="diagnostic_sae_full",
        )
        stimulus_sae_l0 = float(
            torch.cat((sae_source_code, sae_base_code), dim=0).gt(0).sum(dim=-1).float().mean()
        )
        manifest["reference_points"] = {
            "E_resid_mean_aligned_effect": resid_effect,
            "sae_E_full_over_E_resid": sae_effect / resid_effect,
            "sae_mean_l0_on_gate_c_stimulus_positions": stimulus_sae_l0,
            "gate_c_pair_indices": gate_pairs,
            "gate_c_directed_edit_count": int(gate_base.numel()),
        }
        print(
            f"reference | SAE E(full)/E_resid={sae_effect / resid_effect:.4f} "
            f"SAE mean L0={stimulus_sae_l0:.2f}",
            flush=True,
        )

        train_pairs = list(range(RANK_TRAIN_PAIRS))
        train_base, train_source, _ = directed_indices(PILOT_PAIRS, train_pairs)
        train_positions = positions_for_kind(stimuli, train_base, POSITION_SET)
        train_template_x = torch.cat(
            (
                gather_positions(residuals[train_base], train_positions).reshape(-1, 768),
                gather_positions(residuals[train_source], train_positions).reshape(-1, 768),
            ),
            dim=0,
        )
        manifest["configuration"]["train_split_template_rows"] = int(train_template_x.shape[0])

        # GPT-2-small admits at most 1024 positions, including the EOS prompt.  A
        # single 8k call would therefore try to generate 1024 new tokens/sequence
        # and exceed that limit by one.  Four pilot-shaped 2k chunks keep every
        # continuation at 256 tokens, preserve the exact pilot seed for chunk 0,
        # and produce a fixed 8k pool without repetition.
        chunk_tokens = GENERIC_BUDGETS[0]
        chunk_count = GENERATION_POOL_BUDGET // chunk_tokens
        print(f"generic generation | {chunk_count} chunks x {chunk_tokens} tokens", flush=True)
        generic_chunks: list[torch.Tensor] = []
        generation_chunks: list[dict[str, Any]] = []
        for chunk_index in range(chunk_count):
            chunk_seed = SEED + 10_000 * chunk_index
            generic_chunk, chunk_meta = generate_generic_activations(model, LAYER, chunk_seed, chunk_tokens)
            generic_chunks.append(generic_chunk)
            generation_chunks.append({"chunk_index": chunk_index, "seed": chunk_seed, **chunk_meta})
        generic_pool = torch.cat(generic_chunks, dim=0)
        manifest["generic_generation"] = {
            "requested_tokens": GENERATION_POOL_BUDGET,
            "actual_generated_tokens": int(generic_pool.shape[0]),
            "chunking": "four independent 2k-token GPT-2 continuations to stay below the 1024-position limit",
            "chunks": generation_chunks,
            "generation_seconds": sum(float(row["generation_seconds"]) for row in generation_chunks),
        }

        generic_2k = generic_subset(generic_pool, GENERIC_BUDGETS[0])
        generic_2k_fit, _ = split_generic(generic_2k)
        fixed_basis = make_random_basis(sae, torch.cat((generic_2k_fit, train_template_x), dim=0), SEED)
        matched_l0 = fixed_basis.target_l0
        manifest["configuration"]["pilot_actual_matched_l0"] = matched_l0

        completed_A: list[int] = []
        for budget in GENERIC_BUDGETS:
            expected_fit_rows = fit_rows_for_budget(budget, int(train_template_x.shape[0]))
            heldout_rows = budget - int(GENERIC_TRAIN_FRACTION * budget)
            if expected_fit_rows > MAX_EXACT_DUAL_FIT_ROWS:
                entry = not_run_entry(
                    generated_tokens=budget,
                    l0=matched_l0,
                    reason=(
                        f"Not run: {dual_memory_note(expected_fit_rows, heldout_rows)}; "
                        f"this exceeds the exact-dual diagnostic cap of {MAX_EXACT_DUAL_FIT_ROWS} fit rows. "
                        "No alternate/primal/approximate decoder was substituted."
                    ),
                )
                manifest["sweep_A_ridge_fitting_power"][str(budget)] = entry
                manifest["not_run"].append({"section": "A", **entry})
                continue
            if budget > GENERATION_POOL_BUDGET:
                entry = not_run_entry(
                    generated_tokens=budget,
                    l0=matched_l0,
                    reason=(
                        f"Not run: the exact dual fit is already infeasible at {expected_fit_rows} rows; "
                        "larger generic generation was intentionally not performed."
                    ),
                )
                manifest["sweep_A_ridge_fitting_power"][str(budget)] = entry
                manifest["not_run"].append({"section": "A", **entry})
                continue
            if time.perf_counter() - started > TARGET_TOTAL_SECONDS - START_NEXT_CONFIG_GUARD_SECONDS:
                entry = not_run_entry(
                    generated_tokens=budget,
                    l0=matched_l0,
                    reason="Not run: target 15-minute budget was exhausted before this configuration started.",
                )
                manifest["sweep_A_ridge_fitting_power"][str(budget)] = entry
                manifest["not_run"].append({"section": "A", **entry})
                continue
            result = run_random_configuration(
                label=f"A budget={budget}",
                generated_tokens=budget,
                basis=fixed_basis,
                generic_x=generic_subset(generic_pool, budget),
                train_template_x=train_template_x,
                engine=engine,
                stimuli=stimuli,
                clean_d=clean_d,
                gate_base=gate_base,
                gate_source=gate_source,
                gate_signs=gate_signs,
                gate_positions=gate_positions,
                residuals=residuals,
                is_id=is_id,
                are_id=are_id,
                resid_anchor=resid_effect,
            )
            manifest["sweep_A_ridge_fitting_power"][str(budget)] = result
            completed_A.append(budget)
            write_results(manifest)

        if not completed_A:
            raise RuntimeError("No ridge-fitting-power configuration completed")
        best_budget = max(completed_A)
        manifest["sweep_B_sparsity_ceiling"]["fit_budget_selection"] = {
            "selected_generated_token_budget": best_budget,
            "rule": "largest completed A fitting budget; not selected on a Gate-C outcome",
        }

        sweep_l0s = (matched_l0,) + L0_SWEEP_REQUESTED
        for l0 in sweep_l0s:
            if time.perf_counter() - started > HARD_TOTAL_SECONDS - START_NEXT_CONFIG_GUARD_SECONDS:
                entry = not_run_entry(
                    generated_tokens=best_budget,
                    l0=l0,
                    reason="Not run: starting it could breach the 30-minute hard ceiling.",
                )
                manifest["sweep_B_sparsity_ceiling"][str(l0)] = entry
                manifest["not_run"].append({"section": "B", **entry})
                continue
            basis = RandomBasis(
                R=fixed_basis.R,
                b_enc=fixed_basis.b_enc,
                b_dec=fixed_basis.b_dec,
                target_l0=l0,
                seed=fixed_basis.seed,
            )
            result = run_random_configuration(
                label=f"B L0={l0}",
                generated_tokens=best_budget,
                basis=basis,
                generic_x=generic_subset(generic_pool, best_budget),
                train_template_x=train_template_x,
                engine=engine,
                stimuli=stimuli,
                clean_d=clean_d,
                gate_base=gate_base,
                gate_source=gate_source,
                gate_signs=gate_signs,
                gate_positions=gate_positions,
                residuals=residuals,
                is_id=is_id,
                are_id=are_id,
                resid_anchor=resid_effect,
            )
            manifest["sweep_B_sparsity_ceiling"][str(l0)] = result
            write_results(manifest)

        completed_B = [
            row for key, row in manifest["sweep_B_sparsity_ceiling"].items()
            if key != "fit_budget_selection" and row["status"] == "completed"
        ]
        pass_rows = [row for row in completed_B if 0.70 <= row["E_full_over_E_resid"] <= 1.30]
        sae_ratio = manifest["reference_points"]["sae_E_full_over_E_resid"]
        manifest["sweep_B_sparsity_ceiling"]["summary"] = {
            "smallest_l0_in_sweep_passing_gate_C": min((row["per_sample_l0"] for row in pass_rows), default=None),
            "l0_closest_to_sae_reference": (
                min(completed_B, key=lambda row: abs(row["E_full_over_E_resid"] - sae_ratio))["per_sample_l0"]
                if completed_B else None
            ),
            "sae_reference_E_full_over_E_resid": sae_ratio,
        }
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        print(f"diagnostic failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["forward_records"] = (
            engine.records if "engine" in locals() else []
        )
        write_results(manifest)
    return manifest


if __name__ == "__main__":
    result = run()
    print(f"diagnostic status={result['status']}; wrote {RESULTS.name}", flush=True)

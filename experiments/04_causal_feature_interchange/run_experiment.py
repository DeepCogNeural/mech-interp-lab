"""Frozen Amendment 1 full run for Experiment 04.

Run from this directory with the task's offline CPU environment:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg .venv/bin/python run_experiment.py

This script imports only reusable definitions from ``pilot.py``.  It never imports or
executes any Experiment 01--03 executable, and every forward intervention remains the
pilot's literal additive ``resid[batch, pos] += delta`` implementation.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pilot import (
    CONTROL_WORDS,
    GATE_C_T,
    LAYERS,
    PatchEngine,
    RandomBasis,
    SAEWeights,
    Stimuli,
    build_stimuli,
    clean_pass,
    decoded_delta,
    decoder_row_norms,
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
    require_one_token,
    set_determinism,
    sparse_code_matrix,
    t_ci,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "run_results.json"
NOTES = HERE / "RUN_NOTES.md"
FIGURES = HERE / "figures"

SEEDS = (20_260_801, 20_260_802, 20_260_803, 20_260_804, 20_260_805)
LAYER = 8
REQUESTED_PAIRS = 240
RANK_TRAIN_PAIRS = 40
MAX_EVAL_PAIRS = 150
POSITION_SET_REQUESTED = "both"
INITIAL_CANDIDATES = 64
EXTENDED_CANDIDATES = 128
K_GRID_64 = (1, 2, 4, 8, 16, 32, 64)
GENERIC_TOKENS = 8_192
GENERIC_TRAIN_FRACTION = 0.80
PATCH_BATCH_LIMIT = 512
HARD_CEILING_SECONDS = 30.0 * 60.0
T_CRITICAL_DF4 = 2.776
ADJUDICATED_BASES = ("sae", "pca")
ALL_BASES = ("sae", "pca", "neuron", "rand_exp")


class GateStop(RuntimeError):
    """An expected frozen gate or hard-budget stop with an honest manifest."""

    def __init__(self, gate: str, message: str):
        super().__init__(message)
        self.gate = gate


@dataclass
class BasisSpec:
    name: str
    encode: Callable[[torch.Tensor], torch.Tensor]
    decoder: torch.Tensor
    active_mode: str
    complete_decode: Callable[[torch.Tensor], torch.Tensor] | None = None
    note: str = ""


@dataclass
class PreparedBasis:
    spec: BasisSpec
    rank_delta: torch.Tensor
    rank_source: torch.Tensor
    rank_base: torch.Tensor
    eval_delta: torch.Tensor
    prefilter: dict[str, Any]
    candidates: torch.Tensor
    scores: torch.Tensor
    ranked: torch.Tensor
    full: dict[str, Any]
    top64_raw_ratio: float


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


def write_results(manifest: dict[str, Any]) -> None:
    RESULTS.write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n")


def token_difference(logits: torch.Tensor, lengths: torch.Tensor, positive_id: int, negative_id: int) -> torch.Tensor:
    batch = torch.arange(logits.shape[0])
    final = lengths - 1
    return logits[batch, final, positive_id] - logits[batch, final, negative_id]


def subset_stimuli(stimuli: Stimuli, pair_indices: list[int]) -> Stimuli:
    """Compact a Gate-A-retained subset while preserving singular/plural pairing."""
    sequence_indices = [index for pair in pair_indices for index in (2 * pair, 2 * pair + 1)]
    records = [stimuli.pair_records[pair] for pair in pair_indices]
    return Stimuli(
        tokens=stimuli.tokens[sequence_indices].clone(),
        lengths=stimuli.lengths[sequence_indices].clone(),
        subject_positions=stimuli.subject_positions[pair_indices].clone(),
        texts=[stimuli.texts[index] for index in sequence_indices],
        pair_records=records,
        attempted=stimuli.attempted,
        rejected=stimuli.rejected,
    )


def subset_clean(clean: Any, pair_indices: list[int]) -> Any:
    sequence_indices = [index for pair in pair_indices for index in (2 * pair, 2 * pair + 1)]
    return type(clean)(
        logits=clean.logits[sequence_indices].clone(),
        residuals={layer: tensor[sequence_indices].clone() for layer, tensor in clean.residuals.items()},
    )


def all_positions_from_subject(stimuli: Stimuli, base_indices: torch.Tensor, residuals: torch.Tensor, source_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = int((stimuli.lengths[base_indices] - stimuli.subject_positions[base_indices // 2]).max())
    positions = torch.full((base_indices.numel(), width), -1, dtype=torch.long)
    deltas = torch.zeros((base_indices.numel(), width, 768), dtype=torch.float32)
    for row, (base, source) in enumerate(zip(base_indices.tolist(), source_indices.tolist())):
        start = int(stimuli.subject_positions[base // 2])
        stop = int(stimuli.lengths[base])
        chosen = torch.arange(start, stop)
        positions[row, : chosen.numel()] = chosen
        deltas[row, : chosen.numel()] = residuals[source, chosen] - residuals[base, chosen]
    return positions, deltas


def patch_metrics(
    engine: PatchEngine,
    *,
    stimuli: Stimuli,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    signs: torch.Tensor,
    clean_logits: torch.Tensor,
    clean_number: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    deltas: torch.Tensor,
    token_ids: dict[str, int],
    label: str,
    layer: int = LAYER,
) -> dict[str, Any]:
    """Run one additive patch and retain all three frozen readout contrasts."""
    logits = engine.run(
        layer=layer,
        base_tokens=stimuli.tokens[base_indices],
        base_residual=residuals[base_indices],
        positions=positions,
        deltas=deltas,
        label=label,
    )
    lengths = stimuli.lengths[base_indices]
    patched_number = token_difference(logits, lengths, token_ids[" are"], token_ids[" is"])
    patched_control = token_difference(logits, lengths, token_ids[" walking"], token_ids[" walked"])
    patched_tense = token_difference(logits, lengths, token_ids[" were"], token_ids[" was"])
    base_control = token_difference(clean_logits[base_indices], lengths, token_ids[" walking"], token_ids[" walked"])
    base_tense = token_difference(clean_logits[base_indices], lengths, token_ids[" were"], token_ids[" was"])
    number_effect = patched_number - clean_number[base_indices]
    control_effect = patched_control - base_control
    tense_effect = patched_tense - base_tense
    gap = clean_number[source_indices] - clean_number[base_indices]
    aligned = number_effect * signs
    return {
        "number_effect": number_effect,
        "control_effect": control_effect,
        "tense_effect": tense_effect,
        "aligned_effect": aligned,
        "gap": gap,
        "mean_aligned_effect": float(aligned.mean()),
        "mean_E_over_d_gap": float((number_effect / gap).mean()),
        "sign_consistency": float(((number_effect * gap) > 0).float().mean()),
    }


def raw_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_aligned_effect": metrics["mean_aligned_effect"],
        "mean_E_over_d_gap": metrics["mean_E_over_d_gap"],
        "sign_consistency": metrics["sign_consistency"],
        "number_effect_by_directed_edit": metrics["number_effect"].tolist(),
        "control_effect_by_directed_edit": metrics["control_effect"].tolist(),
        "tense_effect_by_directed_edit": metrics["tense_effect"].tolist(),
        "aligned_effect_by_directed_edit": metrics["aligned_effect"].tolist(),
    }


def sparse_sae_matrix(sae: SAEWeights, x: torch.Tensor, chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ReLU SAE code in COO form without retaining an n by 24576 dense matrix."""
    rows, columns, values = [], [], []
    for start in range(0, x.shape[0], chunk_size):
        block = sae.encode(x[start : start + chunk_size])
        local_row, column = torch.nonzero(block, as_tuple=True)
        rows.append(local_row + start)
        columns.append(column)
        values.append(block[local_row, column])
        del block
    row = torch.cat(rows)
    column = torch.cat(columns)
    value = torch.cat(values)
    matrix = torch.sparse_coo_tensor(
        torch.stack((row, column)), value, size=(x.shape[0], 24_576)
    ).coalesce()
    return matrix, torch.unique(column).sort().values


def sparse_design_matrix(kind: str, basis: RandomBasis | SAEWeights, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "rand_exp":
        assert isinstance(basis, RandomBasis)
        values, columns = basis.sparse_topk(x)
        matrix = sparse_code_matrix(values, columns)
        return matrix, torch.unique(columns).sort().values
    if kind == "sae_ridge":
        assert isinstance(basis, SAEWeights)
        return sparse_sae_matrix(basis, x)
    raise ValueError(f"Unknown sparse design kind: {kind}")


def fit_exact_dual_decoder(
    *,
    kind: str,
    basis: RandomBasis | SAEWeights,
    fit_x: torch.Tensor,
    heldout_x: torch.Tensor,
) -> dict[str, Any]:
    """The diagnostic's float32 dual-ridge estimator, including its fixed lambda grid."""
    started = time.perf_counter()
    G, active = sparse_design_matrix(kind, basis, fit_x)
    G_held, _ = sparse_design_matrix(kind, basis, heldout_x)
    n = G.shape[0]
    gram = torch.sparse.mm(G, G.transpose(0, 1)).to_dense()
    held_cross = torch.sparse.mm(G_held, G.transpose(0, 1)).to_dense()
    identity = torch.eye(n, dtype=torch.float32)
    trace: list[dict[str, Any]] = []
    best: tuple[float, torch.Tensor, float, str] | None = None
    for lam in np.logspace(-4, 3, 8):
        lam_value = float(lam)
        try:
            solved, solver = dual_solve(gram + lam_value * identity, fit_x)
        except RuntimeError as exc:
            trace.append({"lambda": lam_value, "status": "not_solved_float32", "error": f"{type(exc).__name__}: {exc}"})
            continue
        heldout_r2 = r2_score(heldout_x, held_cross @ solved)
        trace.append({"lambda": lam_value, "status": "solved", "heldout_generic_r2": heldout_r2, "solver": solver})
        if best is None or heldout_r2 > best[0]:
            best = (heldout_r2, solved, lam_value, solver)
    if best is None:
        raise GateStop("ridge_numeric", f"No prescribed float32 dual-ridge lambda solved for {kind}.")
    heldout_r2, solution, selected_lambda, selected_solver = best
    decoder = torch.sparse.mm(G.transpose(0, 1), solution)
    result = {
        "decoder": decoder,
        "fit_rows": int(n),
        "fit_active_width": int(active.numel()),
        "selected_lambda": selected_lambda,
        "selected_solver": selected_solver,
        "heldout_generic_reconstruction_r2": heldout_r2,
        "numeric_lambda_skips": int(sum(row["status"] != "solved" for row in trace)),
        "lambda_grid": trace,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del G, G_held, gram, held_cross, identity, solution
    gc.collect()
    return result


def generate_pool(model: Any, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """The diagnostic's four independent 2k chunks, giving one fixed 8192-token pool."""
    chunks, chunk_meta = [], []
    for index in range(4):
        chunk, meta = generate_generic_activations(model, LAYER, seed + 10_000 * index, 2_048)
        chunks.append(chunk)
        chunk_meta.append({"chunk_index": index, "seed": seed + 10_000 * index, **meta})
    pool = torch.cat(chunks, dim=0)
    if pool.shape != (GENERIC_TOKENS, 768):
        raise GateStop("generic_generation", f"Expected generic pool {(GENERIC_TOKENS, 768)}, got {tuple(pool.shape)}")
    return pool, {
        "requested_tokens": GENERIC_TOKENS,
        "actual_generated_tokens": int(pool.shape[0]),
        "chunking": "four independent 2048-token GPT-2 continuations",
        "chunks": chunk_meta,
        "generation_seconds": sum(float(item["generation_seconds"]) for item in chunk_meta),
    }


def fit_complete_bases(generic_pool: torch.Tensor) -> tuple[BasisSpec, BasisSpec, dict[str, Any]]:
    """Fit full PCA label-free on generic text; neuron uses the same generic mean and no fit."""
    started = time.perf_counter()
    mean = generic_pool.mean(dim=0)
    centered = generic_pool - mean
    covariance = centered.T @ centered / float(generic_pool.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    V = vectors[:, order].contiguous()
    # The float32 eigensolver returns a PCA basis whose orthogonality is accurate
    # but not always enough for the design's relative *write-back* identity check.
    # Two Newton--Schulz polar refinements preserve each ordered component up to
    # floating-point roundoff while improving V.T @ V before the frozen V.T decoder
    # is used.  The encoder/decoder remain exactly the registered PCA formulas.
    pca_identity = torch.eye(768, dtype=torch.float32)
    for _ in range(2):
        V = (0.5 * V @ (3.0 * pca_identity - V.T @ V)).contiguous()
    orthogonality_max_abs = float((V.T @ V - torch.eye(768)).abs().max())
    identity = torch.eye(768, dtype=torch.float32)
    pca = BasisSpec(
        name="pca",
        encode=lambda x: (x - mean) @ V,
        decoder=V.T,
        active_mode="nonzero",
        complete_decode=lambda x: ((x - mean) @ V) @ V.T + mean,
        note="full 768-component PCA fitted only on generic generated text",
    )
    neuron = BasisSpec(
        name="neuron",
        encode=lambda x: x - mean,
        decoder=identity,
        active_mode="nonzero",
        complete_decode=lambda x: (x - mean) + mean,
        note="residual-stream coordinates centred by the generic-text mean; zero fitting",
    )
    return pca, neuron, {
        "fit_rows": int(generic_pool.shape[0]),
        "components": 768,
        "mean_source": "same 8192-token off-template generated-text pool as the ridge",
        "orthogonality_max_abs": orthogonality_max_abs,
        "float32_polar_reorthogonalisation_steps": 2,
        "elapsed_seconds": time.perf_counter() - started,
    }


def candidate_prefilter_full(
    *,
    code_delta: torch.Tensor,
    source_code: torch.Tensor,
    base_code: torch.Tensor,
    signs: torch.Tensor,
    decoder: torch.Tensor,
    budget: int,
    active_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    aligned = code_delta * signs[:, None, None]
    mean_contribution = aligned.sum(dim=1).mean(dim=0)
    if active_mode == "positive":
        active = ((source_code > 0) | (base_code > 0)).any(dim=(0, 1))
    elif active_mode == "nonzero":
        active = ((source_code != 0) | (base_code != 0)).any(dim=(0, 1))
    else:
        raise ValueError(f"Unknown active mode: {active_mode}")
    candidates = torch.nonzero(active, as_tuple=False).squeeze(1)
    if candidates.numel() < budget:
        raise GateStop("ranking_candidate_coverage", f"{candidates.numel()} active candidates, need {budget}.")
    proxy = mean_contribution[candidates].abs() * decoder_row_norms(decoder)[candidates]
    selected = candidates[proxy.topk(budget).indices]
    return selected, {
        "active_union_count": int(candidates.numel()),
        "prefilter_budget": budget,
        "proxy_top_values": [float(item) for item in proxy.topk(budget).values],
    }


def score_single_coordinates(
    engine: PatchEngine,
    *,
    stimuli: Stimuli,
    base: torch.Tensor,
    source: torch.Tensor,
    signs: torch.Tensor,
    clean_logits: torch.Tensor,
    clean_number: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    code_delta: torch.Tensor,
    decoder: torch.Tensor,
    candidates: torch.Tensor,
    token_ids: dict[str, int],
    label: str,
) -> torch.Tensor:
    n_directions = base.numel()
    per_call = max(1, PATCH_BATCH_LIMIT // n_directions)
    scores = []
    for start in range(0, candidates.numel(), per_call):
        current = candidates[start : start + per_call]
        n_config = current.numel()
        coefficients = code_delta[:, :, current].permute(2, 0, 1)
        deltas = torch.einsum("cbm,cd->cbmd", coefficients, decoder[current]).reshape(
            n_config * n_directions, positions.shape[1], 768
        )
        metrics = patch_metrics(
            engine,
            stimuli=stimuli,
            base_indices=base.repeat(n_config),
            source_indices=source.repeat(n_config),
            signs=signs.repeat(n_config),
            clean_logits=clean_logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=positions.repeat(n_config, 1),
            deltas=deltas,
            token_ids=token_ids,
            label=label,
        )
        scores.append(metrics["aligned_effect"].reshape(n_config, n_directions).mean(dim=1))
    return torch.cat(scores)


def recovery_rows(
    engine: PatchEngine,
    *,
    stimuli: Stimuli,
    base: torch.Tensor,
    source: torch.Tensor,
    signs: torch.Tensor,
    clean_logits: torch.Tensor,
    clean_number: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    code_delta: torch.Tensor,
    decoder: torch.Tensor,
    selected_by_k: dict[int, torch.Tensor],
    full_mean: float,
    token_ids: dict[str, int],
    label: str,
) -> list[dict[str, Any]]:
    rows = []
    for k, selected in selected_by_k.items():
        metrics = patch_metrics(
            engine,
            stimuli=stimuli,
            base_indices=base,
            source_indices=source,
            signs=signs,
            clean_logits=clean_logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=positions,
            deltas=decoded_delta(code_delta, decoder, selected),
            token_ids=token_ids,
            label=label,
        )
        raw = metrics["mean_aligned_effect"] / full_mean
        rows.append({
            "k": int(k),
            "selected_coordinates": selected.tolist(),
            "unclipped_recovery": raw,
            "recovery": float(np.clip(raw, 0.0, 1.0)),
            "metrics": raw_metric_row(metrics),
        })
    return rows


def normalised_auc(rows: list[dict[str, Any]]) -> float:
    x = np.log2(np.asarray([row["k"] for row in rows], dtype=float))
    y = np.asarray([row["recovery"] for row in rows], dtype=float)
    if x.size < 2:
        raise ValueError("AUC requires at least two k values")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def k50(rows: list[dict[str, Any]]) -> int | str:
    for row in rows:
        if row["recovery"] >= 0.5:
            return int(row["k"])
    return "not reached within grid"


def run_self_tests(model: Any, sae: SAEWeights, token_ids: dict[str, int], started: float) -> dict[str, Any]:
    set_determinism(SEEDS[0] + 99)
    stimuli = build_stimuli(model.tokenizer, 2, SEEDS[0] + 99)
    clean = clean_pass(model, stimuli.tokens, (LAYER,))
    base, source, signs = directed_indices(2, range(2))
    subject_positions = positions_for_kind(stimuli, base, "subject")
    engine = PatchEngine(model, start_at_layer8=True)
    zeros = torch.zeros((base.numel(), 1, 768), dtype=torch.float32)
    zero_logits = engine.run(
        layer=LAYER,
        base_tokens=stimuli.tokens[base],
        base_residual=clean.residuals[LAYER][base],
        positions=subject_positions,
        deltas=zeros,
        label="selftest_zero_selection",
        force_full_hook=True,
    )
    zero_bitwise = bool(torch.equal(zero_logits, clean.logits[base]))
    if not zero_bitwise:
        raise GateStop("zero_selection", "Zero additive selection did not reproduce clean logits bit-for-bit.")

    feature_diff, _, _ = feature_delta(sae.encode, clean.residuals[LAYER], base, source, subject_positions)
    selection = torch.arange(16, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(SEEDS[0] + 100)
    exponents = torch.randint(-3, 4, (selection.numel(),), generator=generator)
    scales = torch.pow(torch.tensor(2.0, dtype=torch.float32), exponents)
    original = decoded_delta(feature_diff, sae.W_dec, selection)
    scaled_features = feature_diff.clone()
    scaled_features[..., selection] *= scales
    scaled_decoder = sae.W_dec.clone()
    scaled_decoder[selection] /= scales[:, None]
    rescaled = decoded_delta(scaled_features, scaled_decoder, selection)
    rescale_bitwise = bool(torch.equal(original, rescaled))
    if not rescale_bitwise:
        raise GateStop("rescale_invariance", "Power-of-two rescaling changed the written delta bit pattern.")

    start_logits = engine.run(
        layer=LAYER,
        base_tokens=stimuli.tokens[base],
        base_residual=clean.residuals[LAYER][base],
        positions=subject_positions,
        deltas=zeros,
        label="selftest_start_at_layer",
    )
    start_max = float((start_logits - clean.logits[base]).abs().max())
    start_path = start_max < 1e-4

    all_positions, all_deltas = all_positions_from_subject(stimuli, base, clean.residuals[LAYER], source)
    number = logit_difference(clean.logits, stimuli.lengths, token_ids[" is"], token_ids[" are"])
    exact = patch_metrics(
        engine,
        stimuli=stimuli,
        base_indices=base,
        source_indices=source,
        signs=signs,
        clean_logits=clean.logits,
        clean_number=number,
        residuals=clean.residuals[LAYER],
        positions=all_positions,
        deltas=all_deltas,
        token_ids=token_ids,
        label="selftest_prompt_swap_exactness",
    )
    source_number = number[source]
    patched_number = exact["number_effect"] + number[base]
    prompt_swap_max = float((patched_number - source_number).abs().max())
    prompt_swap_pass = bool(prompt_swap_max < 1e-3)
    if not prompt_swap_pass:
        raise GateStop("prompt_swap_exactness", f"Prompt swap max error {prompt_swap_max:.3g} >= 1e-3")
    stage("self-tests passed", started, f"zero={zero_bitwise}; rescale={rescale_bitwise}; prompt={prompt_swap_max:.3g}")
    return {
        "zero_selection_bitwise": zero_bitwise,
        "D_rescale_bitwise": rescale_bitwise,
        "D_rescale_exponents": exponents.tolist(),
        "start_at_layer8_max_abs": start_max,
        "start_at_layer8_path": "start_at_layer_8" if start_path else "full_forward_hook_fallback",
        "prompt_swap_exactness_max_abs": prompt_swap_max,
        "prompt_swap_all_directed_edits_below_1e-3": prompt_swap_pass,
    }


def gate_b_scan(
    engine: PatchEngine,
    *,
    stimuli: Stimuli,
    clean: Any,
    token_ids: dict[str, int],
    start_path: bool,
) -> tuple[dict[str, Any], str]:
    clean_number = logit_difference(clean.logits, stimuli.lengths, token_ids[" is"], token_ids[" are"])
    base, source, signs = directed_indices(len(stimuli.pair_records), range(len(stimuli.pair_records)))
    table: dict[str, dict[str, Any]] = {}
    selected: str | None = None
    for layer in LAYERS:
        table[str(layer)] = {}
        for kind in ("subject", "final", "both"):
            positions = positions_for_kind(stimuli, base, kind)
            metrics = patch_metrics(
                engine,
                stimuli=stimuli,
                base_indices=base,
                source_indices=source,
                signs=signs,
                clean_logits=clean.logits,
                clean_number=clean_number,
                residuals=clean.residuals[layer],
                positions=positions,
                deltas=full_residual_delta(clean.residuals[layer], base, source, positions),
                token_ids=token_ids,
                label="gate_B_scan",
                layer=layer,
            )
            table[str(layer)][kind] = {
                "mean_E_resid_over_d_gap": metrics["mean_E_over_d_gap"],
                "sign_consistency": metrics["sign_consistency"],
                "mean_aligned_E_resid": metrics["mean_aligned_effect"],
            }
            if layer == LAYER and selected is None and metrics["mean_E_over_d_gap"] >= 0.50 and metrics["sign_consistency"] >= 0.90:
                selected = kind
    if selected is None:
        raise GateStop("B_causal_handle", "No layer-8 position set passed Gate B.")
    return table, selected


def evaluate_seed(model: Any, sae: SAEWeights, token_ids: dict[str, int], seed: int, start_path: bool, run_started: float) -> dict[str, Any]:
    seed_started = time.perf_counter()
    set_determinism(seed)
    stage(f"seed {seed}: stimuli", run_started)
    all_stimuli = build_stimuli(model.tokenizer, REQUESTED_PAIRS, seed)
    all_clean = clean_pass(model, all_stimuli.tokens, LAYERS)
    all_number = logit_difference(all_clean.logits, all_stimuli.lengths, token_ids[" is"], token_ids[" are"])
    singular, plural = all_number[0::2], all_number[1::2]
    gap_pairs = plural - singular
    correct = (singular < 0) & (plural > 0)
    retained_original = torch.nonzero(correct).squeeze(1).tolist()
    gate_a = {
        "generated_pairs": REQUESTED_PAIRS,
        "both_members_signed_correct_fraction": float(correct.float().mean()),
        "retained_pairs": len(retained_original),
        "minimum_retained_pairs": 140,
        "median_d_gap_all_generated_pairs": float(gap_pairs.median()),
        "minimum_median_d_gap": 1.0,
        "passed": bool(float(correct.float().mean()) >= 0.60 and len(retained_original) >= 140 and float(gap_pairs.median()) >= 1.0),
    }
    if not gate_a["passed"]:
        raise GateStop("A_behaviour", f"Seed {seed} Gate A failed: {gate_a}")
    stimuli = subset_stimuli(all_stimuli, retained_original)
    clean = subset_clean(all_clean, retained_original)
    del all_stimuli, all_clean
    gc.collect()

    engine = PatchEngine(model, start_at_layer8=start_path)
    stage(f"seed {seed}: Gate B scan", run_started, f"retained={len(stimuli.pair_records)}")
    scan, selected_position_set = gate_b_scan(engine, stimuli=stimuli, clean=clean, token_ids=token_ids, start_path=start_path)
    if selected_position_set != POSITION_SET_REQUESTED:
        # DESIGN.md's declared selection rule is authoritative if a future model version changes this.
        position_note = f"DESIGN.md selection rule chose {selected_position_set}; requested both was not used."
    else:
        position_note = "frozen selection rule chose both, matching the requested full-run position set"
    gate_b = {
        "selected_layer": LAYER,
        "selected_position_set": selected_position_set,
        "selection_rule": "smallest layer-8 set passing mean E_resid/d_gap >=0.50 and sign consistency >=0.90; ties subject",
        "selected_values": scan[str(LAYER)][selected_position_set],
        "passed": True,
        "note": position_note,
    }

    split_rng = random.Random(seed + 701)
    retained_pairs = list(range(len(stimuli.pair_records)))
    split_rng.shuffle(retained_pairs)
    train_pairs = sorted(retained_pairs[:RANK_TRAIN_PAIRS])
    eval_pairs = sorted(retained_pairs[RANK_TRAIN_PAIRS : RANK_TRAIN_PAIRS + MAX_EVAL_PAIRS])
    if len(train_pairs) != RANK_TRAIN_PAIRS:
        raise GateStop("split", f"Seed {seed} lacks 40 rank-training pairs after Gate A.")
    if not eval_pairs:
        raise GateStop("split", f"Seed {seed} has no evaluation pairs after rank training.")

    clean_number = logit_difference(clean.logits, stimuli.lengths, token_ids[" is"], token_ids[" are"])
    rank_base, rank_source, rank_signs = directed_indices(len(stimuli.pair_records), train_pairs)
    eval_base, eval_source, eval_signs = directed_indices(len(stimuli.pair_records), eval_pairs)
    rank_positions = positions_for_kind(stimuli, rank_base, selected_position_set)
    eval_positions = positions_for_kind(stimuli, eval_base, selected_position_set)
    residuals = clean.residuals[LAYER]
    template_x = torch.cat((
        gather_positions(residuals[rank_base], rank_positions).reshape(-1, 768),
        gather_positions(residuals[rank_source], rank_positions).reshape(-1, 768),
    ), dim=0)

    stage(f"seed {seed}: generic 8192-token pool", run_started)
    generic_pool, generic_meta = generate_pool(model, seed)
    split = int(GENERIC_TRAIN_FRACTION * generic_pool.shape[0])
    generic_fit, generic_heldout = generic_pool[:split], generic_pool[split:]
    ridge_fit_x = torch.cat((generic_fit, template_x), dim=0)
    if ridge_fit_x.shape[0] >= 8_000:
        raise GateStop("ridge_fit_rows", f"Seed {seed} exact dual fit rows={ridge_fit_x.shape[0]} >= 8000 cap")

    stage(f"seed {seed}: PCA and ridge decoders", run_started)
    pca_spec, neuron_spec, pca_meta = fit_complete_bases(generic_pool)
    random_basis = make_random_basis(sae, ridge_fit_x, seed)
    random_ridge = fit_exact_dual_decoder(kind="rand_exp", basis=random_basis, fit_x=ridge_fit_x, heldout_x=generic_heldout)
    sae_ridge = fit_exact_dual_decoder(kind="sae_ridge", basis=sae, fit_x=ridge_fit_x, heldout_x=generic_heldout)
    sae_spec = BasisSpec("sae", sae.encode, sae.W_dec, "positive", note="trained res-jb decoder")
    rand_spec = BasisSpec("rand_exp", random_basis.encode, random_ridge["decoder"], "positive", note="8k generic-text dual ridge; reported, not adjudicated")
    specs = {"sae": sae_spec, "pca": pca_spec, "neuron": neuron_spec, "rand_exp": rand_spec}

    residual_full = patch_metrics(
        engine,
        stimuli=stimuli,
        base_indices=eval_base,
        source_indices=eval_source,
        signs=eval_signs,
        clean_logits=clean.logits,
        clean_number=clean_number,
        residuals=residuals,
        positions=eval_positions,
        deltas=full_residual_delta(residuals, eval_base, eval_source, eval_positions),
        token_ids=token_ids,
        label="resid_full",
    )
    resid_full_mean = residual_full["mean_aligned_effect"]
    if abs(resid_full_mean) < 1e-12:
        raise GateStop("resid_anchor", f"Seed {seed} evaluation residual anchor is zero.")

    complete_assertions: dict[str, Any] = {}
    prepared: dict[str, PreparedBasis] = {}
    stage(f"seed {seed}: 64-coordinate rankings", run_started)
    for basis_index, name in enumerate(ALL_BASES):
        spec = specs[name]
        rank_delta, rank_source_code, rank_base_code = feature_delta(spec.encode, residuals, rank_base, rank_source, rank_positions)
        eval_delta, _, _ = feature_delta(spec.encode, residuals, eval_base, eval_source, eval_positions)
        if spec.complete_decode is not None:
            eval_x = torch.cat((
                gather_positions(residuals[eval_base], eval_positions).reshape(-1, 768),
                gather_positions(residuals[eval_source], eval_positions).reshape(-1, 768),
            ), dim=0)
            reconstruction_max_abs = float((spec.complete_decode(eval_x) - eval_x).abs().max())
            input_max_abs = float(eval_x.abs().max())
            reconstruction_tolerance = 1e-5 * input_max_abs
            complete_assertions[name] = {
                "decoder_identity": "decode(c)=c@V.T+mean over all 768 PCA components" if name == "pca" else "decode(c)=c+mean over all 768 residual coordinates",
                "reconstruction_object": "base and source residual activations at the selected evaluation positions",
                "reconstruction_max_abs_decode_encode_minus_x": reconstruction_max_abs,
                "reconstruction_input_max_abs_x": input_max_abs,
                "reconstruction_relative_tolerance": reconstruction_tolerance,
                "reconstruction_pass": bool(reconstruction_max_abs <= reconstruction_tolerance),
                "effect_object": "mean aligned E(full) versus mean aligned E(resid) on evaluation directed edits",
                "effect_relative_tolerance": 1e-3 * abs(resid_full_mean),
            }
            if reconstruction_max_abs > reconstruction_tolerance:
                raise GateStop(
                    "complete_basis_reconstruction",
                    f"{name} decode(encode(x)) max abs {reconstruction_max_abs:.3g} > "
                    f"1e-5 * max|x| ({reconstruction_tolerance:.3g}) on evaluation residual activations",
                )
        candidates, prefilter = candidate_prefilter_full(
            code_delta=rank_delta,
            source_code=rank_source_code,
            base_code=rank_base_code,
            signs=rank_signs,
            decoder=spec.decoder,
            budget=INITIAL_CANDIDATES,
            active_mode=spec.active_mode,
        )
        scores = score_single_coordinates(
            engine,
            stimuli=stimuli,
            base=rank_base,
            source=rank_source,
            signs=rank_signs,
            clean_logits=clean.logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=rank_positions,
            code_delta=rank_delta,
            decoder=spec.decoder,
            candidates=candidates,
            token_ids=token_ids,
            label=f"rank_{name}_64",
        )
        ranked = candidates[scores.argsort(descending=True)]
        full = patch_metrics(
            engine,
            stimuli=stimuli,
            base_indices=eval_base,
            source_indices=eval_source,
            signs=eval_signs,
            clean_logits=clean.logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=eval_positions,
            deltas=decoded_delta(eval_delta, spec.decoder),
            token_ids=token_ids,
            label=f"{name}_full",
        )
        if spec.complete_decode is not None:
            full_difference = abs(full["mean_aligned_effect"] - resid_full_mean)
            effect_tolerance = 1e-3 * abs(resid_full_mean)
            complete_assertions[name]["max_abs_E_full_minus_E_resid"] = full_difference
            complete_assertions[name]["effect_relative_tolerance"] = effect_tolerance
            complete_assertions[name]["E_full_matches_E_resid_within_relative_1e-3"] = bool(full_difference <= effect_tolerance)
            if full_difference > effect_tolerance:
                raise GateStop(
                    "complete_basis_E_full",
                    f"{name} |E(full)-E(resid)|={full_difference:.3g} > "
                    f"1e-3 * |E(resid)| ({effect_tolerance:.3g})",
                )
        top64 = patch_metrics(
            engine,
            stimuli=stimuli,
            base_indices=eval_base,
            source_indices=eval_source,
            signs=eval_signs,
            clean_logits=clean.logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=eval_positions,
            deltas=decoded_delta(eval_delta, spec.decoder, ranked[:64]),
            token_ids=token_ids,
            label=f"{name}_top64_coverage",
        )
        top64_raw_ratio = top64["mean_aligned_effect"] / full["mean_aligned_effect"]
        prepared[name] = PreparedBasis(spec, rank_delta, rank_source_code, rank_base_code, eval_delta, prefilter, candidates, scores, ranked, full, top64_raw_ratio)

    coverage_trigger = {
        name: prepared[name].top64_raw_ratio for name in ADJUDICATED_BASES
    }
    extend = any(value < 0.8 for value in coverage_trigger.values())
    candidate_budget = EXTENDED_CANDIDATES if extend else INITIAL_CANDIDATES
    if extend:
        stage(f"seed {seed}: coverage trigger -> 128 candidates", run_started, str(coverage_trigger))
        for name in ALL_BASES:
            state = prepared[name]
            candidates, prefilter = candidate_prefilter_full(
                code_delta=state.rank_delta,
                source_code=state.rank_source,
                base_code=state.rank_base,
                signs=rank_signs,
                decoder=state.spec.decoder,
                budget=EXTENDED_CANDIDATES,
                active_mode=state.spec.active_mode,
            )
            scores = score_single_coordinates(
                engine,
                stimuli=stimuli,
                base=rank_base,
                source=rank_source,
                signs=rank_signs,
                clean_logits=clean.logits,
                clean_number=clean_number,
                residuals=residuals,
                positions=rank_positions,
                code_delta=state.rank_delta,
                decoder=state.spec.decoder,
                candidates=candidates,
                token_ids=token_ids,
                label=f"rank_{name}_128",
            )
            state.prefilter = prefilter
            state.candidates = candidates
            state.scores = scores
            state.ranked = candidates[scores.argsort(descending=True)]

    k_grid = K_GRID_64 + ((128,) if extend else ())
    stage(f"seed {seed}: recovery curves", run_started, f"k={k_grid}")
    basis_rows: dict[str, Any] = {}
    for basis_index, name in enumerate(ALL_BASES):
        state = prepared[name]
        top_selected = {k: state.ranked[:k] for k in k_grid}
        top_rows = recovery_rows(
            engine,
            stimuli=stimuli,
            base=eval_base,
            source=eval_source,
            signs=eval_signs,
            clean_logits=clean.logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=eval_positions,
            code_delta=state.eval_delta,
            decoder=state.spec.decoder,
            selected_by_k=top_selected,
            full_mean=state.full["mean_aligned_effect"],
            token_ids=token_ids,
            label=f"{name}_topk",
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 50_000 + basis_index * 1_000)
        random_selected = {k: state.candidates[torch.randperm(candidate_budget, generator=generator)[:k]] for k in k_grid}
        rand_rows = recovery_rows(
            engine,
            stimuli=stimuli,
            base=eval_base,
            source=eval_source,
            signs=eval_signs,
            clean_logits=clean.logits,
            clean_number=clean_number,
            residuals=residuals,
            positions=eval_positions,
            code_delta=state.eval_delta,
            decoder=state.spec.decoder,
            selected_by_k=random_selected,
            full_mean=state.full["mean_aligned_effect"],
            token_ids=token_ids,
            label=f"{name}_randk",
        )
        basis_rows[name] = {
            "role": "adjudicated" if name in ADJUDICATED_BASES else ("reported_not_adjudicated" if name == "rand_exp" else "secondary_not_adjudicated"),
            "note": state.spec.note,
            "candidate_budget": candidate_budget,
            "prefilter": state.prefilter,
            "candidate_coordinates": state.candidates.tolist(),
            "single_coordinate_scores": [float(item) for item in state.scores],
            "ranked_coordinates": state.ranked.tolist(),
            "full": raw_metric_row(state.full),
            "top64_unclipped_recovery_for_coverage_trigger": state.top64_raw_ratio,
            "topk": top_rows,
            "randk": rand_rows,
            "auc_topk": normalised_auc(top_rows),
            "auc_randk": normalised_auc(rand_rows),
            "k50_topk": k50(top_rows),
            "gate_C_E_full_over_E_resid": state.full["mean_aligned_effect"] / resid_full_mean,
        }

    train_delta = full_residual_delta(residuals, rank_base, rank_source, rank_positions)
    mu_direction = (train_delta * rank_signs[:, None, None]).mean(dim=(0, 1))
    mu_norm = float(mu_direction.norm())
    if mu_norm <= 1e-12:
        raise GateStop("mu_ref", f"Seed {seed} supervised mean-difference direction has zero norm.")
    mu_unit = mu_direction / mu_norm
    eval_delta = full_residual_delta(residuals, eval_base, eval_source, eval_positions)
    mu_delta = (eval_delta * mu_unit).sum(dim=-1, keepdim=True) * mu_unit
    mu_metrics = patch_metrics(
        engine,
        stimuli=stimuli,
        base_indices=eval_base,
        source_indices=eval_source,
        signs=eval_signs,
        clean_logits=clean.logits,
        clean_number=clean_number,
        residuals=residuals,
        positions=eval_positions,
        deltas=mu_delta,
        token_ids=token_ids,
        label="mu_ref",
    )
    mu_ref = {
        "role": "supervised k=1 reference; drawn only; never adjudicated",
        "direction_norm_before_unit_normalisation": mu_norm,
        "mean_aligned_effect": mu_metrics["mean_aligned_effect"],
        "unclipped_recovery_to_resid_full": mu_metrics["mean_aligned_effect"] / resid_full_mean,
        "metrics": raw_metric_row(mu_metrics),
    }

    def specificity(metrics: dict[str, Any]) -> float:
        numerator = float(metrics["control_effect"].abs().mean())
        denominator = float(metrics["number_effect"].abs().mean())
        return numerator / max(denominator, 1e-12)

    resid_specificity = specificity(residual_full)
    gate_d: dict[str, Any] = {
        "resid_full_S": resid_specificity,
        "resid_full_threshold": 0.5,
        "resid_full_pass": bool(resid_specificity <= 0.5),
        "adjudicated_bases": {},
    }
    for name in ADJUDICATED_BASES:
        rows = basis_rows[name]["topk"]
        first = next((row for row in rows if row["recovery"] >= 0.5), None)
        if first is None:
            gate_d["adjudicated_bases"][name] = {
                "k_star": "not reached within grid",
                "status": "not_run",
                "reason": "No first k with clipped R >= 0.5; Gate D positive-claim check is unavailable.",
            }
            continue
        metrics = first["metrics"]
        number_effect = torch.tensor(metrics["number_effect_by_directed_edit"])
        control_effect = torch.tensor(metrics["control_effect_by_directed_edit"])
        tense_effect = torch.tensor(metrics["tense_effect_by_directed_edit"])
        S = float(control_effect.abs().mean() / number_effect.abs().mean().clamp_min(1e-12))
        cross_tense = float(((number_effect * tense_effect) > 0).float().mean())
        specificity_pass = bool(S <= resid_specificity + 0.15)
        cross_tense_pass = bool(cross_tense >= 0.80)
        gate_d["adjudicated_bases"][name] = {
            "k_star": first["k"],
            "S_topk": S,
            "threshold_S_topk": resid_specificity + 0.15,
            "specificity_pass": specificity_pass,
            "cross_tense_same_sign_fraction": cross_tense,
            "cross_tense_threshold": 0.80,
            "cross_tense_pass": cross_tense_pass,
            "passed": bool(gate_d["resid_full_pass"] and specificity_pass and cross_tense_pass),
        }

    sae_eval_delta = prepared["sae"].eval_delta
    s8k_metrics = patch_metrics(
        engine,
        stimuli=stimuli,
        base_indices=eval_base,
        source_indices=eval_source,
        signs=eval_signs,
        clean_logits=clean.logits,
        clean_number=clean_number,
        residuals=residuals,
        positions=eval_positions,
        deltas=decoded_delta(sae_eval_delta, sae_ridge["decoder"]),
        token_ids=token_ids,
        label="sae_8k_ridge_full",
    )
    random_active = ((random_basis.encode(gather_positions(residuals[eval_base], eval_positions)) > 0) |
                     (random_basis.encode(gather_positions(residuals[eval_source], eval_positions)) > 0)).any(dim=(0, 1))
    sae_active = ((sae.encode(gather_positions(residuals[eval_base], eval_positions)) > 0) |
                  (sae.encode(gather_positions(residuals[eval_source], eval_positions)) > 0)).any(dim=(0, 1))
    random_columns = torch.nonzero(random_active).squeeze(1)
    sae_columns = torch.nonzero(sae_active).squeeze(1)
    faithfulness = {
        "resid_full_mean_aligned_effect": resid_full_mean,
        "sae_trained_decoder": basis_rows["sae"]["gate_C_E_full_over_E_resid"],
        "rand_exp_8k_dual_ridge": basis_rows["rand_exp"]["gate_C_E_full_over_E_resid"],
        "sae_8k_dual_ridge_s8k": s8k_metrics["mean_aligned_effect"] / resid_full_mean,
        "random_evaluation_active_row_coverage": float(decoder_rows_nonzero(random_ridge["decoder"], random_columns).float().mean()),
        "sae_ridge_evaluation_active_row_coverage": float(decoder_rows_nonzero(sae_ridge["decoder"], sae_columns).float().mean()),
        "random_ridge_fit": {key: value for key, value in random_ridge.items() if key != "decoder"},
        "sae_ridge_fit": {key: value for key, value in sae_ridge.items() if key != "decoder"},
    }
    gate_c = {
        "band": [0.70, 1.30],
        "per_basis": {name: {
            "E_full_over_E_resid": basis_rows[name]["gate_C_E_full_over_E_resid"],
            "pass": bool(0.70 <= basis_rows[name]["gate_C_E_full_over_E_resid"] <= 1.30),
        } for name in ALL_BASES},
        "complete_basis_assertions": complete_assertions,
        "random_basis_reported_not_adjudicated_under_rule_2": True,
    }

    stage(f"seed {seed}: complete", run_started, f"elapsed={time.perf_counter() - seed_started:.1f}s; extend={extend}")
    seed_record = {
        "seed": seed,
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - seed_started,
        "gate_A": gate_a,
        "gate_B": gate_b,
        "layer_position_scan": scan,
        "splits": {
            "rank_train_pairs_after_gate_A": train_pairs,
            "evaluation_pairs_after_gate_A": eval_pairs,
            "rank_train_pair_count": len(train_pairs),
            "evaluation_pair_count": len(eval_pairs),
            "both_edit_directions_per_pair": True,
            "evaluation_directed_edit_count": int(eval_base.numel()),
        },
        "stimulus_records_after_gate_A": stimuli.pair_records,
        "clean_number_d_by_sequence_after_gate_A": [float(value) for value in clean_number],
        "generic_generation": generic_meta,
        "pca_fit": pca_meta,
        "candidate_coverage_trigger": {
            "criterion": "extend to 128 iff E(top-64)/E(full) < 0.8 in any adjudicated basis",
            "adjudicated_top64_unclipped_recoveries": coverage_trigger,
            "triggered": extend,
            "candidate_budget_used": candidate_budget,
            "k_grid": list(k_grid),
        },
        "resid_full": raw_metric_row(residual_full),
        "basis_results": basis_rows,
        "mu_ref": mu_ref,
        "gate_C": gate_c,
        "faithfulness": faithfulness,
        "gate_D": gate_d,
        "forward_timing_records": engine.records,
        "not_run": [],
    }
    del random_ridge["decoder"], sae_ridge["decoder"], generic_pool, generic_fit, generic_heldout, ridge_fit_x
    del template_x, residuals, clean, stimuli, prepared
    gc.collect()
    return seed_record


def aggregate_results(manifest: dict[str, Any]) -> dict[str, Any]:
    seeds = manifest["seed_results"]
    headline_values = [row["basis_results"]["sae"]["auc_topk"] - row["basis_results"]["pca"]["auc_topk"] for row in seeds]
    headline_mean, headline_half_width = t_ci(headline_values)
    auc_summary: dict[str, Any] = {}
    for name in ALL_BASES:
        top_values = [row["basis_results"][name]["auc_topk"] for row in seeds]
        random_values = [row["basis_results"][name]["auc_randk"] for row in seeds]
        top_mean, top_hw = t_ci(top_values)
        random_mean, random_hw = t_ci(random_values)
        auc_summary[name] = {
            "topk_values_by_seed": top_values,
            "topk_mean": top_mean,
            "topk_half_width_t4_95": top_hw,
            "randk_values_by_seed": random_values,
            "randk_mean": random_mean,
            "randk_half_width_t4_95": random_hw,
            "k50_by_seed": [row["basis_results"][name]["k50_topk"] for row in seeds],
        }
    fallback_values = [row["basis_results"]["sae"]["auc_topk"] - row["basis_results"]["sae"]["auc_randk"] for row in seeds]
    fallback_mean, fallback_half_width = t_ci(fallback_values)
    gate_d_each_seed = []
    for row in seeds:
        per_basis = row["gate_D"]["adjudicated_bases"]
        passed = bool(row["gate_D"]["resid_full_pass"] and all(item.get("passed", False) for item in per_basis.values()))
        gate_d_each_seed.append(passed)
    sae_gate_c = all(row["gate_C"]["per_basis"]["sae"]["pass"] for row in seeds)
    if not sae_gate_c:
        branch = "inconclusive: SAE Gate C did not pass in every seed"
    elif headline_mean - headline_half_width > 0.0 and headline_half_width <= 0.05 and all(gate_d_each_seed):
        branch = "adjudicated positive"
    elif headline_mean - headline_half_width <= 0.0 <= headline_mean + headline_half_width and headline_half_width <= 0.05 and auc_summary["sae"]["topk_half_width_t4_95"] <= 0.10 and auc_summary["pca"]["topk_half_width_t4_95"] <= 0.10:
        branch = "adjudicated null"
    else:
        branch = "inconclusive"
    faithfulness: dict[str, Any] = {}
    for key in ("sae_trained_decoder", "rand_exp_8k_dual_ridge", "sae_8k_dual_ridge_s8k"):
        values = [row["faithfulness"][key] for row in seeds]
        mean, half_width = t_ci(values)
        faithfulness[key] = {"values_by_seed": values, "mean": mean, "half_width_t4_95": half_width}
    return {
        "auc_method": "trapezoidal AUC over log2(k), divided by log2(k_max)-log2(k_min); R is clipped to [0,1] only for this statistic",
        "headline_sae_minus_pca": {
            "values_by_seed": headline_values,
            "mean": headline_mean,
            "half_width_t4_95": headline_half_width,
            "interval_t4_95": [headline_mean - headline_half_width, headline_mean + headline_half_width],
        },
        "decision_branch": branch,
        "decision_rule_inputs": {
            "paired_half_width_requirement": 0.05,
            "sae_auc_half_width": auc_summary["sae"]["topk_half_width_t4_95"],
            "pca_auc_half_width": auc_summary["pca"]["topk_half_width_t4_95"],
            "per_basis_half_width_requirement": 0.10,
            "gate_D_pass_each_seed": gate_d_each_seed,
            "sae_gate_C_pass_each_seed": [row["gate_C"]["per_basis"]["sae"]["pass"] for row in seeds],
        },
        "per_basis_auc_and_k50": auc_summary,
        "sae_topk_minus_sae_randk_reported_not_adjudicated": {
            "values_by_seed": fallback_values,
            "mean": fallback_mean,
            "half_width_t4_95": fallback_half_width,
            "interval_t4_95": [fallback_mean - fallback_half_width, fallback_mean + fallback_half_width],
        },
        "faithfulness_E_full_over_E_resid": faithfulness,
    }


def make_figures(manifest: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    seeds = manifest["seed_results"]
    colors = {"sae": "#0072B2", "pca": "#D55E00", "neuron": "#009E73", "rand_exp": "#777777"}
    fig, axis = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for name in ALL_BASES:
        grids = [row["basis_results"][name]["topk"] for row in seeds]
        k_values = np.asarray([row["k"] for row in grids[0]], dtype=float)
        values = np.asarray([[row["unclipped_recovery"] for row in grid] for grid in grids], dtype=float)
        mean = values.mean(axis=0)
        half = T_CRITICAL_DF4 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
        style = "--" if name == "rand_exp" else "-"
        label = "rand_exp (reported; not adjudicated)" if name == "rand_exp" else name
        axis.plot(k_values, mean, linestyle=style, marker="o", color=colors[name], label=label)
        axis.fill_between(k_values, mean - half, mean + half, color=colors[name], alpha=0.16)
    mu_values = np.asarray([row["mu_ref"]["unclipped_recovery_to_resid_full"] for row in seeds], dtype=float)
    mu_mean = float(mu_values.mean())
    mu_half = float(T_CRITICAL_DF4 * mu_values.std(ddof=1) / math.sqrt(len(mu_values)))
    axis.axhline(mu_mean, color="#222222", linestyle=":", label="mu_ref (k=1; reference only)")
    axis.axhspan(mu_mean - mu_half, mu_mean + mu_half, color="#222222", alpha=0.08)
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted({row["k"] for seed in seeds for row in seed["basis_results"]["sae"]["topk"]}))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("edited coordinates k (log2 scale)")
    axis.set_ylabel("unclipped within-basis recovery R_b(k)")
    axis.set_title("Experiment 04 recovery curves: seed mean with t(4) 95% bands")
    axis.axhline(0.5, color="#444444", linewidth=0.8, alpha=0.55)
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(FIGURES / "01_recovery_curves.png", dpi=180)
    plt.close(fig)

    headline = manifest["summary"]["headline_sae_minus_pca"]
    values = np.asarray(headline["values_by_seed"], dtype=float)
    mean = float(headline["mean"])
    half = float(headline["half_width_t4_95"])
    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    x = np.arange(1, len(values) + 1)
    axis.scatter(x, values, color="#0072B2", zorder=3, label="per-seed AUC(sae) - AUC(pca)")
    axis.plot(x, values, color="#0072B2", alpha=0.35)
    axis.errorbar([len(values) + 1.25], [mean], yerr=[half], fmt="D", color="#000000", capsize=5, label="t(4) mean ± 95% interval")
    axis.axhline(0.0, color="#222222", linewidth=1.0, label="zero decision boundary")
    axis.axhline(0.05, color="#999999", linestyle="--", linewidth=0.8)
    axis.axhline(-0.05, color="#999999", linestyle="--", linewidth=0.8, label="±0.05 precision reference")
    axis.text(len(values) + 1.47, mean, f"{mean:+.3f} ± {half:.3f}", va="center", fontsize=9)
    axis.set_xlim(0.5, len(values) + 2.45)
    axis.set_xticks(list(x) + [len(values) + 1.25], [f"seed {i}" for i in range(1, len(values) + 1)] + ["mean"])
    axis.set_ylabel("normalised AUC difference")
    axis.set_title(f"Headline: SAE minus PCA — {manifest['summary']['decision_branch']}")
    axis.legend(fontsize=8, loc="best")
    fig.savefig(FIGURES / "02_headline.png", dpi=180)
    plt.close(fig)


def write_notes(manifest: dict[str, Any]) -> None:
    elapsed = manifest.get("wall_clock_seconds", 0.0)
    lines = [
        "# Experiment 04 全运行记录",
        "",
        f"- 状态：`{manifest['status']}`。墙钟：{elapsed:.1f} 秒。",
        "- 控制臂分支：DESIGN.md Amendment 1 的冻结 Rule 2；主比较为 `AUC(sae) - AUC(pca)`，`rand_exp` 只透明报告、未裁决。",
    ]
    if manifest.get("summary"):
        summary = manifest["summary"]
        headline = summary["headline_sae_minus_pca"]
        lines.extend([
            f"- 主区间：{headline['mean']:+.6f} ± {headline['half_width_t4_95']:.6f}（t(4) 95%）；冻结分支：`{summary['decision_branch']}`。",
            "",
            "## 每个基的 AUC 与 k50",
            "",
            "| basis | AUC(top-k), t(4) 95% | k50（五个种子） |",
            "|---|---:|---|",
        ])
        for name in ALL_BASES:
            row = summary["per_basis_auc_and_k50"][name]
            k50_values = ", ".join(str(value) for value in row["k50_by_seed"])
            lines.append(f"| {name} | {row['topk_mean']:.6f} ± {row['topk_half_width_t4_95']:.6f} | {k50_values} |")
        fallback = summary["sae_topk_minus_sae_randk_reported_not_adjudicated"]
        lines.extend([
            "",
            f"- 非裁决 SAE 排名参照：`AUC(sae_topk) - AUC(sae_randk)` = {fallback['mean']:+.6f} ± {fallback['half_width_t4_95']:.6f}。",
            "",
            "## 五种子写回信实度 E(full)/E(resid)",
            "",
            "| measurement | mean ± t(4) 95% | per-seed |",
            "|---|---:|---|",
        ])
        for name, row in summary["faithfulness_E_full_over_E_resid"].items():
            values = ", ".join(f"{value:.6f}" for value in row["values_by_seed"])
            lines.append(f"| {name} | {row['mean']:.6f} ± {row['half_width_t4_95']:.6f} | {values} |")
        lines.extend([
            "",
            "## Gate 与自测",
            "",
        ])
        for row in manifest["seed_results"]:
            gate_a = row["gate_A"]
            gate_b = row["gate_B"]["selected_values"]
            gate_d = row["gate_D"]
            lines.append(
                f"- seed {row['seed']}: Gate A retained={gate_a['retained_pairs']}, both-correct={gate_a['both_members_signed_correct_fraction']:.3f}, median d_gap={gate_a['median_d_gap_all_generated_pairs']:.3f}; "
                f"Gate B ({row['gate_B']['selected_position_set']}) E_resid/d_gap={gate_b['mean_E_resid_over_d_gap']:.3f}, sign={gate_b['sign_consistency']:.3f}; "
                f"Gate D resid S={gate_d['resid_full_S']:.3f}。"
            )
        self_tests = manifest["self_tests"]
        unavailable_k_stars = [
            (row["seed"], name)
            for row in manifest["seed_results"]
            for name, value in row["gate_D"]["adjudicated_bases"].items()
            if value.get("status") == "not_run" and value.get("k_star") == "not reached within grid"
        ]
        lines.extend([
            f"- 自测：zero-selection bitwise={self_tests['zero_selection_bitwise']}; D-rescale bitwise={self_tests['D_rescale_bitwise']}; "
            f"start_at_layer=8 max_abs={self_tests['start_at_layer8_max_abs']:.3g}; prompt-swap max_abs={self_tests['prompt_swap_exactness_max_abs']:.3g}。",
        ])
        if unavailable_k_stars:
            details = ", ".join(f"seed {seed} {name}" for seed, name in unavailable_k_stars)
            lines.append(
                f"- 设计保留的开放点：{details} 在扩展后网格内仍未达到 `R >= 0.5`，其 Gate D `k*` 为 `not reached within grid`、该项为 `not_run`。"
                "设计未规定无 `k*` 时 Gate D 应视作失败还是不适用；本次主分支已独立由 SAE Gate C 未在五个种子全部通过而确定。"
            )
        lines.extend([
            "",
            "本记录只陈述给定坐标基中的加性差分写回与坐标集中测量；它不测量、也不声称 SAE 对模型任何计算有损害、降级、移除或损失。",
        ])
    else:
        lines.extend([
            f"- 停止门：`{manifest.get('failed_gate')}`。",
            f"- 错误：{manifest.get('error', 'not recorded')}。",
            "- 未开始或未完成的测量均在 `run_results.json` 的 `not_run` 中显式列出。",
        ])
    NOTES.write_text("\n".join(lines) + "\n")


def run() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    manifest: dict[str, Any] = {
        "schema": "exp04-causal-feature-interchange-full-v1; frozen DESIGN.md Amendment 1 Rule 2 ladder; CPU float32; additive residual deltas only; t(4)=2.776",
        "status": "running",
        "failed_gate": None,
        "configuration": {
            "seeds": list(SEEDS),
            "generated_single_flip_pairs_per_seed": REQUESTED_PAIRS,
            "minimum_gate_A_retained_pairs": 140,
            "rank_training_pairs": RANK_TRAIN_PAIRS,
            "evaluation_pair_cap": MAX_EVAL_PAIRS,
            "both_edit_directions": True,
            "layer": LAYER,
            "requested_position_set": POSITION_SET_REQUESTED,
            "initial_candidates_per_basis": INITIAL_CANDIDATES,
            "coverage_extension_candidates": EXTENDED_CANDIDATES,
            "k_grid_initial": list(K_GRID_64),
            "generic_generated_text_tokens_per_seed": GENERIC_TOKENS,
            "ridge": "exact float32 dual ridge; eight lambda values logspace(1e-4,1e3); 80% generic fit plus rank-training template activations; selected on heldout generic R2",
            "pca": "full 768 orthonormal components fit label-free on the same 8192-token generic generated-text pool",
            "neuron": "x-mean / c+mean, with generic-text mean and zero fitted parameters",
            "amendment_branch": "Rule 2 structural random-arm failure already frozen by r8k < 0.70 and r8k-r2k <= 0.03; PCA is load-bearing control",
            "patching": "only resid[batch, pos] += delta where delta is a difference of reconstructions",
            "offline_env": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MPLBACKEND")},
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": "cpu"},
        "self_tests": None,
        "seed_results": [],
        "summary": None,
        "not_run": [],
        "trims": [],
    }
    try:
        set_determinism(SEEDS[0])
        stage("load model and SAE", started)
        model = load_model()
        sae = load_direct_res_jb(LAYER)
        required = (" is", " are", " was", " were") + CONTROL_WORDS
        token_ids = {text: require_one_token(model.tokenizer, text) for text in required}
        manifest["token_ids"] = {"eos": int(model.tokenizer.eos_token_id), **token_ids}
        if sae.W_dec.shape != (24_576, 768):
            raise GateStop("environment_sae_shape", f"W_dec is {tuple(sae.W_dec.shape)}, expected (24576, 768)")
        manifest["self_tests"] = run_self_tests(model, sae, token_ids, started)
        start_path = manifest["self_tests"]["start_at_layer8_path"] == "start_at_layer_8"

        for index, seed in enumerate(SEEDS):
            if time.perf_counter() - started >= HARD_CEILING_SECONDS:
                pending = list(SEEDS[index:])
                manifest["not_run"].append({"section": "remaining_seeds", "status": "not_run", "seeds": pending, "reason": "30-minute hard ceiling reached before seed start; no seeds or edit directions were silently cut."})
                raise GateStop("hard_budget_ceiling", "30-minute hard ceiling reached before a required seed started.")
            record = evaluate_seed(model, sae, token_ids, seed, start_path, started)
            manifest["seed_results"].append(record)
            if time.perf_counter() - started > HARD_CEILING_SECONDS:
                raise GateStop("hard_budget_ceiling", "30-minute hard ceiling exceeded during a required seed.")

        manifest["summary"] = aggregate_results(manifest)
        manifest["status"] = "completed"
        make_figures(manifest)
    except GateStop as exc:
        manifest["status"] = "gated_out"
        manifest["failed_gate"] = exc.gate
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["not_run"].append({"section": "headline_and_unstarted_measurements", "status": "not_run", "reason": f"Stopped at frozen gate {exc.gate}: {exc}"})
        print(f"GATED OUT at {exc.gate}: {exc}", flush=True)
    except Exception as exc:  # Keep an implementation/environment failure auditable rather than partial-success shaped.
        manifest["status"] = "gated_out"
        manifest["failed_gate"] = "implementation_or_environment"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["not_run"].append({"section": "headline_and_unstarted_measurements", "status": "not_run", "reason": f"Implementation or environment stop: {type(exc).__name__}: {exc}"})
        print(f"GATED OUT at implementation_or_environment: {type(exc).__name__}: {exc}", flush=True)
    finally:
        manifest["wall_clock_seconds"] = time.perf_counter() - started
        if manifest["status"] == "completed":
            manifest["wall_clock_seconds_per_seed"] = [row["elapsed_seconds"] for row in manifest["seed_results"]]
        write_results(manifest)
        write_notes(manifest)
    return manifest


if __name__ == "__main__":
    result = run()
    print(f"run status={result['status']}; wrote {RESULTS.name} and {NOTES.name}", flush=True)

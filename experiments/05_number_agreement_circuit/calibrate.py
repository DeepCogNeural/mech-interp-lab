"""Experiment 05 pre-freeze residual-only calibration pilot.

This file deliberately imports reusable Experiment 04 definitions but never calls
an Experiment 01--04 ``run()`` function.  It measures residual-stream edits and
wall-clock timing only; its two no-op attention hooks exist solely to price a
future run and never inspect, retain, or report an attention activation.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
import gc
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

# Importing Experiment 04 is explicitly allowed, but its published directory is
# not writable scope for this task.  Prevent Python from creating/updating a
# neighbouring ``__pycache__`` while importing its reusable definitions.
sys.dont_write_bytecode = True


HERE = Path(__file__).resolve().parent
EXP04 = HERE.parent / "04_causal_feature_interchange"
if str(EXP04) not in sys.path:
    sys.path.insert(0, str(EXP04))

from pilot import (  # noqa: E402  (deliberate local reusable-import path)
    ADJECTIVES,
    NOUNS,
    PREPOSITIONS,
    CleanPass,
    GateStop,
    PatchEngine,
    Stimuli,
    build_stimuli,
    clean_pass,
    directed_indices,
    gather_positions,
    load_direct_res_jb,
    load_model,
    logit_difference,
    positions_for_kind,
    require_one_token,
    set_determinism,
)


RESULTS = HERE / "calibration_results.json"
NOTES = HERE / "CALIBRATION_NOTES.md"

SEED = 20_260_899
LAYER = 8
REQUESTED_PAIRS = 240
BOOTSTRAP_RESAMPLES = 10_000
TIMING_REPETITIONS = 3
# Every layer is timed with attention no-op hooks.  This includes the requested
# 0/4/8/11 checkpoints and avoids silently treating a 48-head sample as a
# 144-head sweep in the runtime multiplication.
TIMING_LAYERS = tuple(range(12))
GATE_A_MIN_FRACTION = 0.60
GATE_A_MIN_RETAINED = 140
GATE_A_MIN_MEDIAN_GAP = 1.0
SANITY_MIN_E_OVER_GAP = 0.50
SANITY_MIN_SIGN = 0.90


class CalibrationStop(RuntimeError):
    """Expected stop that still produces the two required local artifacts."""

    def __init__(self, gate: str, message: str):
        super().__init__(message)
        self.gate = gate


def _ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])


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
        raise ValueError(f"Non-finite value cannot enter calibration manifest: {value}")
    return value


def build_stimuli_from_rows(
    tokenizer: Any,
    *,
    family: str,
    rows: list[dict[str, Any]],
) -> Stimuli:
    """Make an Experiment-04-compatible pair container from explicit constructions.

    The base family itself still comes directly from ``build_stimuli``.  Explicit
    construction is necessary only for the two wrong-source controls and the
    genuinely different syntactic frame for source C.
    """
    encoded: list[tuple[str, list[int], int]] = []
    records: list[dict[str, Any]] = []
    for pair_index, row in enumerate(rows):
        singular_text = row["singular_text"]
        plural_text = row["plural_text"]
        subject_index = int(row["subject_token_index"])
        singular_ids = _ids(tokenizer, singular_text)
        plural_ids = _ids(tokenizer, plural_text)
        singular_subject_id = require_one_token(tokenizer, f" {row['subject_singular']}")
        plural_subject_id = require_one_token(tokenizer, f" {row['subject_plural']}")
        if len(singular_ids) != len(plural_ids):
            raise CalibrationStop(
                "source_length_pairing",
                f"{family} pair {pair_index} changed length within singular/plural members.",
            )
        if not (
            0 <= subject_index < len(singular_ids)
            and singular_ids[subject_index] == singular_subject_id
            and plural_ids[subject_index] == plural_subject_id
        ):
            raise CalibrationStop(
                "source_subject_indexing",
                f"{family} pair {pair_index} did not retain its asserted one-token subject position.",
            )
        encoded.extend(
            ((singular_text, singular_ids, subject_index), (plural_text, plural_ids, subject_index))
        )
        records.append(
            {
                "pair_index": pair_index,
                "family": family,
                **row,
                "singular_token_ids": singular_ids,
                "plural_token_ids": plural_ids,
                "subject_singular_token_id": singular_subject_id,
                "subject_plural_token_id": plural_subject_id,
            }
        )
    max_length = max(len(ids) for _, ids, _ in encoded)
    eos = int(tokenizer.eos_token_id)
    tokens = torch.full((len(encoded), max_length), eos, dtype=torch.long)
    lengths = torch.empty(len(encoded), dtype=torch.long)
    for index, (_, ids, _) in enumerate(encoded):
        tokens[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths[index] = len(ids)
    return Stimuli(
        tokens=tokens,
        lengths=lengths,
        subject_positions=torch.tensor([int(row["subject_token_index"]) for row in rows], dtype=torch.long),
        texts=[text for text, _, _ in encoded],
        pair_records=records,
        attempted=len(rows),
        rejected=0,
    )


def make_source_a(tokenizer: Any, base: Stimuli, seed: int) -> Stimuli:
    """Same-number, different-subject-noun control in the original frame."""
    rng = random.Random(seed + 101)
    rows: list[dict[str, Any]] = []
    for record in base.pair_records:
        original = (record["subject_singular"], record["subject_plural"])
        alternatives = [pair for pair in NOUNS if pair != original]
        subject_sg, subject_pl = rng.choice(alternatives)
        prefix = f"The {record['adjective']}"
        rows.append(
            {
                "source_kind": "A_same_number_different_noun",
                "subject_singular": subject_sg,
                "subject_plural": subject_pl,
                "attractor": record["attractor"],
                "adjective": record["adjective"],
                "preposition": record["preposition"],
                "subject_token_index": len(_ids(tokenizer, prefix)),
                "singular_text": f"{prefix} {subject_sg} {record['preposition']} the {record['attractor']}",
                "plural_text": f"{prefix} {subject_pl} {record['preposition']} the {record['attractor']}",
            }
        )
    return build_stimuli_from_rows(tokenizer, family="source_A_same_number_different_noun", rows=rows)


def make_source_b(tokenizer: Any, base: Stimuli) -> Stimuli:
    """Attractor-number flip with the subject exactly preserved."""
    inflections = {form: pair for pair in NOUNS for form in pair}
    rows: list[dict[str, Any]] = []
    for record in base.pair_records:
        attr_sg, attr_pl = inflections[record["attractor"]]
        flipped = attr_pl if record["attractor"] == attr_sg else attr_sg
        prefix = f"The {record['adjective']}"
        rows.append(
            {
                "source_kind": "B_attractor_flip",
                "subject_singular": record["subject_singular"],
                "subject_plural": record["subject_plural"],
                "attractor_original": record["attractor"],
                "attractor": flipped,
                "adjective": record["adjective"],
                "preposition": record["preposition"],
                "subject_token_index": len(_ids(tokenizer, prefix)),
                "singular_text": f"{prefix} {record['subject_singular']} {record['preposition']} the {flipped}",
                "plural_text": f"{prefix} {record['subject_plural']} {record['preposition']} the {flipped}",
            }
        )
    return build_stimuli_from_rows(tokenizer, family="source_B_attractor_flip", rows=rows)


def make_source_c_relative_clause(tokenizer: Any, n_pairs: int, seed: int, *, with_adverb: bool) -> Stimuli:
    """Different syntactic family: matrix agreement across a relative clause.

    ``with_adverb=True`` is tried first because it deliberately gives source C a
    different sequence length from the base family, exercising the source-own /
    base-own indexing rule rather than merely asserting it.
    """
    rng = random.Random(seed + (307 if with_adverb else 311))
    rows: list[dict[str, Any]] = []
    verb_singular, verb_plural = "likes", "like"
    for pair_index in range(n_pairs):
        subject_sg, subject_pl = rng.choice(NOUNS)
        attr_sg, attr_pl = rng.choice(NOUNS)
        attr = attr_sg if pair_index % 2 == 0 else attr_pl
        rel_verb = verb_singular if attr == attr_sg else verb_plural
        middle = f"that the {attr}"
        if with_adverb:
            middle += " often"
        prefix = "The"
        rows.append(
            {
                "source_kind": "C_cross_template_number_matched",
                "template": "The {SUBJ} that the {ATTRACTOR} often {RELVERB}" if with_adverb else "The {SUBJ} that the {ATTRACTOR} {RELVERB}",
                "subject_singular": subject_sg,
                "subject_plural": subject_pl,
                "attractor": attr,
                "relative_verb": rel_verb,
                "subject_token_index": len(_ids(tokenizer, prefix)),
                "singular_text": f"The {subject_sg} {middle} {rel_verb}",
                "plural_text": f"The {subject_pl} {middle} {rel_verb}",
            }
        )
    return build_stimuli_from_rows(
        tokenizer,
        family="source_C_relative_clause_with_adverb" if with_adverb else "source_C_relative_clause",
        rows=rows,
    )


def gate_a(stimuli: Stimuli, clean: CleanPass, is_id: int, are_id: int) -> tuple[dict[str, Any], list[int], torch.Tensor]:
    number = logit_difference(clean.logits, stimuli.lengths, is_id, are_id)
    singular, plural = number[0::2], number[1::2]
    d_gap = plural - singular
    correct = (singular < 0) & (plural > 0)
    retained = torch.nonzero(correct).squeeze(1).tolist()
    fraction = float(correct.float().mean())
    median_gap = float(d_gap.median())
    row = {
        "generated_pairs": int(correct.numel()),
        "both_members_signed_correct_fraction": fraction,
        "retained_pairs": len(retained),
        "minimum_retained_pairs": GATE_A_MIN_RETAINED,
        "median_d_gap_all_generated_pairs": median_gap,
        "minimum_median_d_gap": GATE_A_MIN_MEDIAN_GAP,
        "passed": bool(
            fraction >= GATE_A_MIN_FRACTION
            and len(retained) >= GATE_A_MIN_RETAINED
            and median_gap >= GATE_A_MIN_MEDIAN_GAP
        ),
    }
    return row, retained, number


def source_delta(
    *,
    base_residual: torch.Tensor,
    source_residual: torch.Tensor,
    base_positions: torch.Tensor,
    source_positions: torch.Tensor,
) -> torch.Tensor:
    """Gather at each prompt's own positions, then write at base positions only."""
    if base_positions.shape != source_positions.shape:
        raise CalibrationStop(
            "cross_template_position_shape",
            f"Source and base position sets differ in shape: {tuple(source_positions.shape)} vs {tuple(base_positions.shape)}",
        )
    return gather_positions(source_residual, source_positions) - gather_positions(base_residual, base_positions)


def run_residual_interchange(
    engine: PatchEngine,
    *,
    label: str,
    base: Stimuli,
    base_clean: CleanPass,
    source: Stimuli,
    source_clean: CleanPass,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    signs: torch.Tensor,
    base_number: torch.Tensor,
    is_id: int,
    are_id: int,
) -> dict[str, Any]:
    base_positions = positions_for_kind(base, base_indices, "both")
    source_positions = positions_for_kind(source, source_indices, "both")
    deltas = source_delta(
        base_residual=base_clean.residuals[LAYER][base_indices],
        source_residual=source_clean.residuals[LAYER][source_indices],
        base_positions=base_positions,
        source_positions=source_positions,
    )
    logits = engine.run(
        layer=LAYER,
        base_tokens=base.tokens[base_indices],
        base_residual=base_clean.residuals[LAYER][base_indices],
        positions=base_positions,
        deltas=deltas,
        label=label,
    )
    patched = logit_difference(logits, base.lengths[base_indices], is_id, are_id)
    raw = patched - base_number[base_indices]
    aligned = raw * signs
    return {
        "raw": raw,
        "aligned": aligned,
        "patched": patched,
        "base_positions": base_positions,
        "source_positions": source_positions,
        "base_lengths": base.lengths[base_indices],
        "source_lengths": source.lengths[source_indices],
    }


def distribution(values: torch.Tensor) -> dict[str, float]:
    vector = values.detach().float().cpu()
    return {
        "mean": float(vector.mean()),
        "standard_deviation": float(vector.std(unbiased=True)),
        "min": float(vector.min()),
        "median": float(vector.median()),
        "max": float(vector.max()),
    }


def pair_values(values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if values.numel() % 2:
        raise CalibrationStop("paired_direction_count", f"Expected an even number of directed edits, got {values.numel()}.")
    directed = values.detach().float().cpu().numpy().reshape(-1, 2)
    return directed.mean(axis=1), np.abs(directed).mean(axis=1)


def bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise CalibrationStop("bootstrap_pairing", "Bootstrap numerator and denominator must be one value per common retained pair.")
    denominator_mean = float(denominator.mean())
    if abs(denominator_mean) < 1e-12:
        raise CalibrationStop("zero_right_reference", "True-flip residual reference mean is zero, so rho is undefined.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, numerator.size, size=(BOOTSTRAP_RESAMPLES, numerator.size))
    samples = np.abs(numerator[draws].mean(axis=1)) / np.abs(denominator[draws].mean(axis=1))
    return {
        "estimate": float(abs(numerator.mean()) / abs(denominator_mean)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "retained minimal-pair; both directed edits stay together",
        "ci_95_percentile": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
    }


def summarize_interchange(
    *,
    raw: torch.Tensor,
    aligned: torch.Tensor,
    right_pair_signed: np.ndarray,
    right_pair_abs: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    signed_pairs, abs_pairs = pair_values(aligned)
    return {
        "directed_edit_count": int(aligned.numel()),
        "retained_pair_count": int(signed_pairs.size),
        "raw_unaligned_delta_d": distribution(raw),
        "sign_aligned_delta_d": distribution(aligned),
        "per_pair_mean_sign_aligned_delta_d": {
            "mean": float(signed_pairs.mean()),
            "standard_deviation": float(signed_pairs.std(ddof=1)),
            "min": float(signed_pairs.min()),
            "median": float(np.median(signed_pairs)),
            "max": float(signed_pairs.max()),
        },
        "per_pair_mean_absolute_delta_d": {
            "mean": float(abs_pairs.mean()),
            "standard_deviation": float(abs_pairs.std(ddof=1)),
            "min": float(abs_pairs.min()),
            "median": float(np.median(abs_pairs)),
            "max": float(abs_pairs.max()),
        },
        "rho_full_bias_type": bootstrap_ratio(signed_pairs, right_pair_signed, bootstrap_seed),
        "rho_noise_type": bootstrap_ratio(abs_pairs, right_pair_abs, bootstrap_seed + 1),
    }


def run_self_tests(model: Any, token_ids: dict[str, int]) -> dict[str, Any]:
    """The three Experiment-04 self-tests requested for this residual-only pilot."""
    set_determinism(SEED + 1)
    stimuli = build_stimuli(model.tokenizer, 2, SEED + 1)
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
        raise CalibrationStop("zero_selection", "Zero additive residual edit did not reproduce clean logits bit-for-bit.")
    start_logits = engine.run(
        layer=LAYER,
        base_tokens=stimuli.tokens[base],
        base_residual=clean.residuals[LAYER][base],
        positions=subject_positions,
        deltas=zeros,
        label="selftest_start_at_layer",
    )
    start_max = float((start_logits - clean.logits[base]).abs().max())
    if not start_max < 1e-4:
        raise CalibrationStop("start_at_layer_equivalence", f"start_at_layer=8 max abs {start_max:.3g} is not <1e-4.")

    width = int((stimuli.lengths[base] - stimuli.subject_positions[base // 2]).max())
    all_positions = torch.full((base.numel(), width), -1, dtype=torch.long)
    all_deltas = torch.zeros((base.numel(), width, 768), dtype=torch.float32)
    for row, (base_i, source_i) in enumerate(zip(base.tolist(), source.tolist())):
        start = int(stimuli.subject_positions[base_i // 2])
        stop = int(stimuli.lengths[base_i])
        selected = torch.arange(start, stop)
        all_positions[row, : selected.numel()] = selected
        all_deltas[row, : selected.numel()] = clean.residuals[LAYER][source_i, selected] - clean.residuals[LAYER][base_i, selected]
    prompt_logits = engine.run(
        layer=LAYER,
        base_tokens=stimuli.tokens[base],
        base_residual=clean.residuals[LAYER][base],
        positions=all_positions,
        deltas=all_deltas,
        label="selftest_prompt_swap_exactness",
    )
    number = logit_difference(clean.logits, stimuli.lengths, token_ids[" is"], token_ids[" are"])
    prompt_number = logit_difference(prompt_logits, stimuli.lengths[base], token_ids[" is"], token_ids[" are"])
    prompt_max = float((prompt_number - number[source]).abs().max())
    prompt_pass = bool(prompt_max < 1e-3)
    if not prompt_pass:
        raise CalibrationStop("prompt_swap_exactness", f"Prompt-swap max abs {prompt_max:.3g} is not <1e-3.")
    return {
        "zero_intervention_bitwise_identity": zero_bitwise,
        "start_at_layer8_max_abs": start_max,
        "start_at_layer8_threshold": "< 1e-4",
        "start_at_layer8_pass": True,
        "prompt_swap_exactness_max_abs": prompt_max,
        "prompt_swap_threshold": "< 1e-3",
        "prompt_swap_pass": prompt_pass,
        "scope": "Residual-stream self-tests only; no attention or SAE latent values were inspected.",
    }


def timed(label: str, fn: Callable[[], Any]) -> dict[str, Any]:
    """One warm-up plus repeated CPU wall-clock trials, retaining timing only."""
    with torch.no_grad():
        fn()  # warm-up is deliberately excluded from the reported median
    samples: list[float] = []
    for _ in range(TIMING_REPETITIONS):
        started = time.perf_counter()
        with torch.no_grad():
            fn()
        samples.append(time.perf_counter() - started)
    return {
        "label": label,
        "warmup_excluded": True,
        "repetitions": TIMING_REPETITIONS,
        "seconds": samples,
        "median_seconds": float(np.median(samples)),
        "mean_seconds": float(np.mean(samples)),
    }


def runtime_timing(model: Any, base: Stimuli, clean: CleanPass, directed: torch.Tensor) -> dict[str, Any]:
    """Price paths without reading a head/attention/latent numerical value."""
    base_tokens = base.tokens[directed]

    def noop(activation: torch.Tensor, hook: Any) -> torch.Tensor:
        return activation

    rows: dict[str, Any] = {}
    rows["full_hooked_residual_forward"] = timed(
        "full_hooked_residual_forward",
        lambda: model.run_with_hooks(
            base_tokens,
            fwd_hooks=[(f"blocks.{LAYER}.hook_resid_pre", noop)],
            return_type="logits",
        ),
    )
    rows["partial_residual_forward_by_start_layer"] = {}
    rows["attention_noop_forward_by_start_layer"] = {"hook_z": {}, "hook_v": {}}
    for layer in TIMING_LAYERS:
        residual = clean.residuals[layer][directed]
        rows["partial_residual_forward_by_start_layer"][str(layer)] = timed(
            f"partial_residual_start_at_{layer}",
            lambda residual=residual, layer=layer: model(residual, start_at_layer=layer, return_type="logits"),
        )
        for hook_name in ("hook_z", "hook_v"):
            # Explicitly permitted no-op timing hook: no activation is copied,
            # retained, projected, compared, or emitted from this call.
            rows["attention_noop_forward_by_start_layer"][hook_name][str(layer)] = timed(
                f"no_op_{hook_name}_start_at_{layer}",
                lambda residual=residual, layer=layer, hook_name=hook_name: model.run_with_hooks(
                    residual,
                    start_at_layer=layer,
                    fwd_hooks=[(f"blocks.{layer}.attn.{hook_name}", noop)],
                    return_type="logits",
                ),
            )
    rows["scope"] = (
        "Attention hooks were no-op timing hooks only. They returned the incoming activation unchanged; "
        "no attention values, head effects, or head rankings were inspected or recorded."
    )
    return rows


def projection(timing: dict[str, Any], retained_pairs: int) -> dict[str, Any]:
    """Audit-friendly lower-bound projection using every design-mandated patch count."""
    full = float(timing["full_hooked_residual_forward"]["median_seconds"])
    residual8 = float(timing["partial_residual_forward_by_start_layer"]["8"]["median_seconds"])
    z = {
        layer: float(timing["attention_noop_forward_by_start_layer"]["hook_z"][str(layer)]["median_seconds"])
        for layer in TIMING_LAYERS
    }
    v = {
        layer: float(timing["attention_noop_forward_by_start_layer"]["hook_v"][str(layer)]["median_seconds"])
        for layer in TIMING_LAYERS
    }
    # Stage 1's unknown top-24 layer allocation is not a result to peek at before
    # freeze.  Two heads per layer is an explicit neutral projection assumption;
    # min/max envelopes state the remaining uncertainty rather than hiding it.
    stage1_z_seconds = sum(12 * z[layer] for layer in TIMING_LAYERS)
    stage1_v_expected_seconds = sum(2 * v[layer] for layer in TIMING_LAYERS)
    stage1_v_bounds = [24 * min(v.values()), 24 * max(v.values())]
    stage1 = stage1_z_seconds + stage1_v_expected_seconds

    sweep_one = sum(12 * z[layer] for layer in TIMING_LAYERS)
    stage2_per_seed = (
        2 * sweep_one  # true and source-A 144-head z sweeps
        + full  # one all-144 joint patch
        + 8 * full  # maximum nested-set patches
        + 8 * 2 * full  # maximum two-forward joint path patches
        + 3 * full  # mandatory A/C/B joint S* measurements (missing from prose budget list)
    )
    stage2 = 8 * stage2_per_seed

    random_draws = 100 + 7 * 20
    stage3_per_seed_nonrandom = 3 * 3 + 2 + 12
    stage3_patches = 8 * stage3_per_seed_nonrandom + random_draws
    stage3 = stage3_patches * residual8
    forward_seconds = stage1 + stage2 + stage3

    return {
        "formula": "one_time_load + sum(per_patch_forward_cost x patch_count) + per_seed_fixed_overhead x seed_blocks",
        "per_patch_cost_seconds": {
            "full_hooked_residual_proxy": full,
            "partial_residual_start_at_8": residual8,
            "no_op_hook_z_by_start_layer": z,
            "no_op_hook_v_by_start_layer": v,
        },
        "counts": {
            "stage_1": {
                "one_seed": True,
                "z_final_per_head": {str(layer): 12 for layer in TIMING_LAYERS},
                "z_final_total": 144,
                "v_subject_top24_total": 24,
                "v_subject_projection_allocation": "2 heads at each of layers 0 through 11; actual top-24 layers are intentionally unknown pre-freeze",
            },
            "stage_2": {
                "seeds": 8,
                "true_z_final_sweeps": 8 * 144,
                "source_A_z_final_sweeps": 8 * 144,
                "all_144_joint_patches": 8,
                "nested_set_patches_maximum": 8 * 8,
                "two_step_joint_path_patch_forward_equivalents_maximum": 8 * 8 * 2,
                "A_C_B_joint_S_star_patches_required_by_Q2_and_B_reporting": 8 * 3,
            },
            "stage_3": {
                "seeds": 8,
                "position_sets_times_interventions": 8 * 3 * 3,
                "alpha_scalings": 8 * 2,
                "matched_random_draws": random_draws,
                "PCA_span_patches": 8 * 12,
                "partial_residual_patch_total": stage3_patches,
            },
            "seed_blocks_for_fixed_overhead": {
                "conservative_no_cache_reuse": 16,
                "unique_seed_union_if_overlapping_20260806_to_20260808_caches_are_reused": 13,
            },
        },
        "forward_seconds": {
            "stage_1_expected": stage1,
            "stage_1_v_top24_layer_allocation_bounds": [stage1_z_seconds + stage1_v_bounds[0], stage1_z_seconds + stage1_v_bounds[1]],
            "stage_2": stage2,
            "stage_3": stage3,
            "total": forward_seconds,
        },
        "retained_pair_count_used_for_pricing": retained_pairs,
        "trim_to_100_retained_pairs": {
            "allowed_floor": 100,
            "forward_cost_scaling_assumption": "forward costs scale linearly with directed retained-pair count; fixed overhead and one-time loads do not",
            "forward_scaling_factor": 100.0 / retained_pairs,
            "forward_seconds": forward_seconds * (100.0 / retained_pairs),
        },
        "known_projection_omissions": [
            "The pre-freeze blinding constraint prevents measuring an attention-value cache, which a later z/v patch implementation will need.",
            "Stage 3 requires per-seed PCA fitting and latent-candidate preparation; pricing either would inspect latent/span material prohibited in this calibration pilot.",
        ],
    }


def write_notes(manifest: dict[str, Any]) -> None:
    lines = [
        "# Experiment 05 calibration pilot notes",
        "",
        f"- Status: `{manifest['status']}`." + (f" Failed gate: `{manifest['failed_gate']}`." if manifest.get("failed_gate") else ""),
        f"- Calibration seed: `{SEED}`. It is deliberately not one of either eight-seed adjudication set, so no measured constant is circular with a later adjudicating seed.",
        "- Scope: CPU-only float32 offline run. Only `hook_resid_pre` residual-stream edits were measured. No head-level or latent-span-level activation/effect/ranking was measured.",
    ]
    if manifest.get("environment"):
        env = manifest["environment"]
        lines.append(f"- Environment: device `{env['device']}`, torch `{env['torch']}`, platform `{env['platform']}`.")
    gate = manifest.get("gate_A")
    if gate:
        lines.extend(
            [
                "",
                "## Gate A and source constructions",
                "",
                f"- Base family: retained {gate['base']['retained_pairs']}/{gate['base']['generated_pairs']} (both-correct={gate['base']['both_members_signed_correct_fraction']:.3f}, median d_gap={gate['base']['median_d_gap_all_generated_pairs']:.3f}); pass={gate['base']['passed']}.",
                f"- Source C winning family: `{gate['source_C']['family']}`; retained {gate['source_C']['gate_A']['retained_pairs']}/{gate['source_C']['gate_A']['generated_pairs']} (both-correct={gate['source_C']['gate_A']['both_members_signed_correct_fraction']:.3f}, median d_gap={gate['source_C']['gate_A']['median_d_gap_all_generated_pairs']:.3f}); pass={gate['source_C']['gate_A']['passed']}.",
                "- Source A changes only the subject lexical item while preserving its number and the base adjective/preposition/attractor. Source B flips only the attractor number while preserving the base subject.",
                "- Source C is a relative-clause matrix-agreement frame (`The SUBJECT that the ATTRACTOR often RELVERB`), selected because it changes syntax and lexical frame while retaining a one-token subject and a final is/are decision. It was chosen before any residual effect was inspected; Gate A is its only selection criterion.",
            ]
        )
        lines.append("- Source-C template search:")
        for row in gate["source_C"]["template_attempts"]:
            lines.append(f"  - `{row['family']}`: pass={row['gate_A']['passed']}; retained={row['gate_A']['retained_pairs']}; reason={row['decision']}.")
    indexing = manifest.get("cross_template_indexing")
    if indexing:
        lines.extend(
            [
                "",
                "## Cross-template indexing",
                "",
                "- For every directed C→base edit, the layer-8 source residual was gathered at C's own `[subject, final]` token positions, the base residual at the base prompt's own `[subject, final]` positions, and the resulting two vectors were written only at the base positions.",
                f"- Sequence length differed on {indexing['different_sequence_length_directed_edits']}/{indexing['directed_edit_count']} directed edits; final-position index differed on {indexing['different_final_position_directed_edits']}/{indexing['directed_edit_count']}. This verifies that source positions were not reused as base write indices.",
            ]
        )
    constants = manifest.get("constants")
    if constants:
        lines.extend(["", "## Calibration constants", ""])
        for label in ("source_A", "source_B", "source_C"):
            row = constants[label]
            rho = row["rho_full_bias_type"]
            noise = row["rho_noise_type"]
            lines.append(
                f"- {label}: rho_full={rho['estimate']:.5f} (bootstrap 95% CI {rho['ci_95_percentile'][0]:.5f} to {rho['ci_95_percentile'][1]:.5f}); rho_noise={noise['estimate']:.5f} (95% CI {noise['ci_95_percentile'][0]:.5f} to {noise['ci_95_percentile'][1]:.5f})."
            )
        theta = manifest["theta_spec"]
        lines.append(f"- Frozen formula outputs: theta_spec^A={theta['A']:.5f}; theta_spec^C={theta['C']:.5f}. Source B is recorded but never adjudicates.")
    sanity = manifest.get("sanity")
    if sanity:
        lines.extend(["", "## Residual-handle sanity and self-tests", ""])
        lines.append(f"- True flip at layer 8/both: E_resid/d_gap={sanity['mean_E_resid_over_d_gap']:.5f} (floor 0.50), sign consistency={sanity['sign_consistency']:.5f} (floor 0.90).")
        self_tests = manifest.get("self_tests", {})
        if self_tests:
            lines.append(f"- Self-tests: zero bitwise={self_tests['zero_intervention_bitwise_identity']}; start_at_layer=8 max_abs={self_tests['start_at_layer8_max_abs']:.3g}; prompt-swap max_abs={self_tests['prompt_swap_exactness_max_abs']:.3g}.")
    runtime = manifest.get("runtime")
    if runtime:
        projection_row = runtime["projection"]
        total = projection_row.get("total_with_conservative_16_seed_blocks_seconds")
        trimmed = projection_row.get("trimmed_total_with_conservative_16_seed_blocks_seconds")
        lines.extend(["", "## Runtime projection", ""])
        lines.append("- Projection formula: sum(per-patch median wall-clock cost x enumerated patch count) + measured per-seed fixed block x seed blocks + one-time model/SAE loads.")
        lines.append("- Enumerated counts: Stage 1 = 144 z patches + 24 v patches; Stage 2 = per 8 seeds two 144-head sweeps, one all-144 joint patch, <=8 nested patches, <=8 two-forward path patches, and three A/C/B joint-set patches required by Q2/reporting; Stage 3 = 9 position/intervention patches + 2 alpha patches + 12 PCA-span patches per seed, plus 100 + 7x20 random draws.")
        if total is not None:
            lines.append(f"- Conservative no-reuse lower-bound total: {total / 60.0:.2f} CPU-minutes against the 120-minute cap; lower-bound fit={projection_row['lower_bound_fits_120_minutes']}.")
            lines.append(f"- At the permitted 100-retained-pair floor, the same lower-bound projection is {trimmed / 60.0:.2f} CPU-minutes (saves {(total-trimmed)/60.0:.2f} minutes).")
        lines.append("- Important defect: this cannot certify the full design meets the cap. The requested pre-freeze measurement cannot price future attention-value caching without head-level access, and the stated two-term formula omits Stage-3 PCA fitting and latent-candidate preparation. The reported number is therefore a lower bound, not a complete wall-clock projection.")
    defects = manifest.get("design_defects", [])
    if defects:
        lines.extend(["", "## Design defects / unmeasurable terms", ""])
        lines.extend(f"- {item}" for item in defects)
    NOTES.write_text("\n".join(lines) + "\n")


def write_artifacts(manifest: dict[str, Any], started: float) -> None:
    manifest["wall_clock_seconds"] = time.perf_counter() - started
    RESULTS.write_text(json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n")
    write_notes(manifest)


def run() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    set_determinism(SEED)
    manifest: dict[str, Any] = {
        "schema": "exp05-number-agreement-calibration-v1; pre-freeze residual-only; CPU float32; paired bootstrap-95",
        "status": "running",
        "failed_gate": None,
        "configuration": {
            "calibration_seed": SEED,
            "not_an_adjudication_seed": True,
            "layer": LAYER,
            "hook": "hook_resid_pre",
            "position_set": "both (subject + final)",
            "requested_pairs": REQUESTED_PAIRS,
            "both_flip_directions": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "device": "cpu",
            "dtype": "float32",
            "offline_env": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MPLBACKEND")},
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "device": "cpu"},
        "gate_A": None,
        "cross_template_indexing": None,
        "constants": None,
        "theta_spec": None,
        "sanity": None,
        "self_tests": None,
        "runtime": None,
        "design_defects": [
            "Stage 2's frozen Q2 rule requires direct joint-set measurements for source A and source C, and source B is required for reporting, but the runtime-budget prose enumerates only the two 144-head sweeps and does not count these three joint-set patches per seed. This calibration includes them in its lower-bound patch count.",
            "A complete later z/v implementation requires source attention-value caching. Measuring that cache now would violate the design's literal no-head-measurement blinding claim, so the requested residual-only fixed-overhead block cannot price it.",
            "Stage 3 requires per-seed PCA fitting and SAE candidate-pool preparation. The requested two-term projection does not assign either a term, and pre-freeze pricing them would inspect prohibited latent/span material. Thus the specified projection is necessarily a lower bound rather than a certifiable total against 120 CPU-minutes.",
        ],
    }
    try:
        sae_started = time.perf_counter()
        sae = load_direct_res_jb(LAYER)
        sae_load_seconds = time.perf_counter() - sae_started
        sae_shape = list(sae.W_dec.shape)
        # The calibration never uses SAE values: retaining a full 24,576 x 768
        # decoder alongside GPT-2 only raises peak RAM and could itself turn a
        # timing pilot into an environment failure.  Loading is still timed and
        # its required shape is asserted before it is released.
        if sae_shape != [24_576, 768]:
            raise CalibrationStop("environment_sae_shape", f"Unexpected SAE decoder shape {sae_shape}.")
        del sae
        gc.collect()
        model_started = time.perf_counter()
        model = load_model()
        model_load_seconds = time.perf_counter() - model_started
        token_ids = {text: require_one_token(model.tokenizer, text) for text in (" is", " are")}
        manifest["one_time_costs"] = {
            "model_load_seconds": model_load_seconds,
            "sae_load_seconds": sae_load_seconds,
            "sae_shape": sae_shape,
            "model_and_sae_load_seconds": model_load_seconds + sae_load_seconds,
        }

        overhead_started = time.perf_counter()
        base = build_stimuli(model.tokenizer, REQUESTED_PAIRS, SEED)
        # A later head sweep needs a residual starting state at every layer for
        # its partial-forward price.  Residual caching is permitted here; no
        # attention cache is requested or retained.
        base_clean = clean_pass(model, base.tokens, TIMING_LAYERS)
        base_gate, base_retained, base_number = gate_a(base, base_clean, token_ids[" is"], token_ids[" are"])
        if not base_gate["passed"]:
            raise CalibrationStop("Gate_A_base", f"Base Gate A failed: {base_gate}")
        source_a = make_source_a(model.tokenizer, base, SEED)
        source_b = make_source_b(model.tokenizer, base)
        source_a_clean = clean_pass(model, source_a.tokens, (LAYER,))
        source_b_clean = clean_pass(model, source_b.tokens, (LAYER,))

        attempts: list[dict[str, Any]] = []
        source_c: Stimuli | None = None
        source_c_clean: CleanPass | None = None
        source_c_gate: dict[str, Any] | None = None
        source_c_retained: list[int] | None = None
        for with_adverb in (True, False):
            candidate = make_source_c_relative_clause(model.tokenizer, REQUESTED_PAIRS, SEED, with_adverb=with_adverb)
            candidate_clean = clean_pass(model, candidate.tokens, (LAYER,))
            candidate_gate, candidate_retained, _ = gate_a(candidate, candidate_clean, token_ids[" is"], token_ids[" are"])
            family = candidate.pair_records[0]["family"]
            attempts.append(
                {
                    "family": family,
                    "gate_A": candidate_gate,
                    "decision": "selected: passed all unchanged Gate-A thresholds" if candidate_gate["passed"] else "rejected: failed unchanged Gate-A thresholds; next predeclared frame tried",
                }
            )
            if candidate_gate["passed"]:
                source_c, source_c_clean, source_c_gate, source_c_retained = candidate, candidate_clean, candidate_gate, candidate_retained
                break
        if source_c is None or source_c_clean is None or source_c_gate is None or source_c_retained is None:
            raise CalibrationStop("Gate_A_source_C", "Every pre-specified source-C relative-clause family failed Gate A.")
        fixed_overhead_seconds = time.perf_counter() - overhead_started
        common_pairs = sorted(set(base_retained).intersection(source_c_retained))
        if len(common_pairs) < GATE_A_MIN_RETAINED:
            raise CalibrationStop(
                "common_retained_pairs",
                f"Only {len(common_pairs)} pair indices pass Gate A in both base and source-C families; need {GATE_A_MIN_RETAINED}.",
            )
        manifest["gate_A"] = {
            "base": base_gate,
            "source_C": {
                "family": source_c.pair_records[0]["family"],
                "gate_A": source_c_gate,
                "template_attempts": attempts,
            },
            "common_retained_pair_indices_count": len(common_pairs),
            "common_retained_pair_indices": common_pairs,
        }
        manifest["per_seed_fixed_overhead"] = {
            "seconds": fixed_overhead_seconds,
            "block": "base + A/B/C stimulus construction; base/source-C Gate A; residual clean-pass caching only",
            "excludes": "model load, SAE load, all patched forwards, attention-value caching, PCA fit, and latent candidate preparation",
        }

        base_indices, right_indices, signs = directed_indices(REQUESTED_PAIRS, common_pairs)
        same_number_indices = base_indices.clone()
        engine = PatchEngine(model, start_at_layer8=True)
        right = run_residual_interchange(
            engine,
            label="calibration_right_true_subject_flip",
            base=base,
            base_clean=base_clean,
            source=base,
            source_clean=base_clean,
            base_indices=base_indices,
            source_indices=right_indices,
            signs=signs,
            base_number=base_number,
            is_id=token_ids[" is"],
            are_id=token_ids[" are"],
        )
        source_a_result = run_residual_interchange(
            engine,
            label="calibration_source_A",
            base=base,
            base_clean=base_clean,
            source=source_a,
            source_clean=source_a_clean,
            base_indices=base_indices,
            source_indices=same_number_indices,
            signs=signs,
            base_number=base_number,
            is_id=token_ids[" is"],
            are_id=token_ids[" are"],
        )
        source_b_result = run_residual_interchange(
            engine,
            label="calibration_source_B",
            base=base,
            base_clean=base_clean,
            source=source_b,
            source_clean=source_b_clean,
            base_indices=base_indices,
            source_indices=same_number_indices,
            signs=signs,
            base_number=base_number,
            is_id=token_ids[" is"],
            are_id=token_ids[" are"],
        )
        source_c_result = run_residual_interchange(
            engine,
            label="calibration_source_C",
            base=base,
            base_clean=base_clean,
            source=source_c,
            source_clean=source_c_clean,
            base_indices=base_indices,
            source_indices=same_number_indices,
            signs=signs,
            base_number=base_number,
            is_id=token_ids[" is"],
            are_id=token_ids[" are"],
        )
        right_pair_signed, right_pair_abs = pair_values(right["aligned"])
        right_summary = summarize_interchange(
            raw=right["raw"],
            aligned=right["aligned"],
            right_pair_signed=right_pair_signed,
            right_pair_abs=right_pair_abs,
            bootstrap_seed=SEED + 700,
        )
        constants = {
            "right_true_single_flip_reference": right_summary,
            "source_A": summarize_interchange(
                raw=source_a_result["raw"], aligned=source_a_result["aligned"], right_pair_signed=right_pair_signed, right_pair_abs=right_pair_abs, bootstrap_seed=SEED + 701
            ),
            "source_B": summarize_interchange(
                raw=source_b_result["raw"], aligned=source_b_result["aligned"], right_pair_signed=right_pair_signed, right_pair_abs=right_pair_abs, bootstrap_seed=SEED + 703
            ),
            "source_C": summarize_interchange(
                raw=source_c_result["raw"], aligned=source_c_result["aligned"], right_pair_signed=right_pair_signed, right_pair_abs=right_pair_abs, bootstrap_seed=SEED + 705
            ),
            "definitions": {
                "bias_type": "abs(mean over retained minimal pairs of the pair-mean sign-aligned delta_d_S) / abs(the same true-flip reference mean)",
                "noise_type": "mean over retained minimal pairs of mean(abs(sign-aligned delta_d_S) over the two directions) / the same true-flip quantity",
                "bootstrap": "10,000 pair-resamples preserving both directions within every minimal pair",
            },
        }
        manifest["constants"] = constants
        theta_a = max(0.20, 2.0 * constants["source_A"]["rho_full_bias_type"]["estimate"])
        theta_c = max(0.20, 2.0 * constants["source_C"]["rho_full_bias_type"]["estimate"])
        manifest["theta_spec"] = {"A": theta_a, "C": theta_c, "formula": "max(0.20, 2 * rho_full)"}

        gap = base_number[right_indices] - base_number[base_indices]
        sanity = {
            "mean_E_resid_over_d_gap": float((right["raw"] / gap).mean()),
            "sign_consistency": float(((right["raw"] * gap) > 0).float().mean()),
            "minimum_E_resid_over_d_gap": SANITY_MIN_E_OVER_GAP,
            "minimum_sign_consistency": SANITY_MIN_SIGN,
        }
        sanity["passed"] = bool(
            sanity["mean_E_resid_over_d_gap"] >= SANITY_MIN_E_OVER_GAP
            and sanity["sign_consistency"] >= SANITY_MIN_SIGN
        )
        manifest["sanity"] = sanity
        if not sanity["passed"]:
            raise CalibrationStop("residual_handle_sanity", f"Layer-8/both true flip failed sanity floor: {sanity}")

        manifest["cross_template_indexing"] = {
            "method": "source residual gathered at source [subject, final]; base residual gathered and delta written at base [subject, final]",
            "directed_edit_count": int(base_indices.numel()),
            "different_sequence_length_directed_edits": int((source_c_result["source_lengths"] != source_c_result["base_lengths"]).sum()),
            "different_final_position_directed_edits": int((source_c_result["source_positions"][:, 1] != source_c_result["base_positions"][:, 1]).sum()),
            "source_subject_positions_match_base_subject_positions_count": int((source_c_result["source_positions"][:, 0] == source_c_result["base_positions"][:, 0]).sum()),
        }
        manifest["self_tests"] = run_self_tests(model, token_ids)
        timing = runtime_timing(model, base, base_clean, base_indices)
        projected = projection(timing, len(common_pairs))
        one_time = manifest["one_time_costs"]["model_and_sae_load_seconds"]
        fixed = fixed_overhead_seconds
        lower_total = one_time + projected["forward_seconds"]["total"] + fixed * 16
        lower_total_reused = one_time + projected["forward_seconds"]["total"] + fixed * 13
        trimmed_total = one_time + projected["trim_to_100_retained_pairs"]["forward_seconds"] + fixed * 16
        projected.update(
            {
                "total_with_conservative_16_seed_blocks_seconds": lower_total,
                "total_with_unique_13_seed_blocks_seconds": lower_total_reused,
                "trimmed_total_with_conservative_16_seed_blocks_seconds": trimmed_total,
                "hard_cap_seconds": 120.0 * 60.0,
                "lower_bound_fits_120_minutes": bool(lower_total <= 120.0 * 60.0),
                "complete_cap_certification": "not possible from this blinding-preserving calibration; see known_projection_omissions",
            }
        )
        manifest["runtime"] = {"timing": timing, "projection": projected}
        manifest["status"] = "completed_with_lower_bound_runtime_projection"
        return manifest
    except (CalibrationStop, GateStop) as exc:
        manifest["status"] = "stopped"
        manifest["failed_gate"] = getattr(exc, "gate", "experiment04_gate")
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        return manifest
    except Exception as exc:
        manifest["status"] = "stopped"
        manifest["failed_gate"] = "implementation_or_environment"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        return manifest
    finally:
        write_artifacts(manifest, started)


if __name__ == "__main__":
    result = run()
    print(f"Calibration status={result['status']}; wrote {RESULTS.name} and {NOTES.name}", flush=True)

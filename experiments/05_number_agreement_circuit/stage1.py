"""Experiment 05 Stage 1: frozen one-seed 144-head coarse sweep.

Run from this directory:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
  ../../.venv/bin/python stage1.py

This program deliberately stops at Stage 1.  It measures the true single-flip
source through every ``hook_z`` head at the final position, the all-head joint
``hook_z`` reference, and ``hook_v`` at the subject position for the selected
top 24.  It does not construct or measure any Stage 2 or Stage 3 arm.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

# Experiment 04 imports are explicitly permitted by the frozen design.  Avoid
# making bytecode artifacts in that published directory while reusing them.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EXP04 = HERE.parent / "04_causal_feature_interchange"
if str(EXP04) not in sys.path:
    sys.path.insert(0, str(EXP04))

from calibrate import (  # noqa: E402 -- deliberate reuse of the Exp05 harness
    CalibrationStop,
    gate_a,
    jsonable,
    run_self_tests,
    source_delta,
)
from pilot import (  # noqa: E402 -- deliberate Experiment 04 reusable imports
    CleanPass,
    GateStop,
    PatchEngine,
    Stimuli,
    build_stimuli,
    directed_indices,
    logit_difference,
    positions_for_kind,
    require_one_token,
    set_determinism,
)


RESULTS = HERE / "stage1_results.json"
NOTES = HERE / "STAGE1_NOTES.md"

SEED = 20_260_801
REQUESTED_PAIRS = 240
LAYER_COUNT = 12
HEADS_PER_LAYER = 12
HOOK_Z = "hook_z"
HOOK_V = "hook_v"
# The model's full logits are [batch, position, 50,257].  Keep the exact same
# retained directed edits but score them in small deterministic microbatches so
# a clean and a patched vocabulary-sized tensor are never resident together.
PATCH_BATCH_SIZE = 32


class Stage1Stop(RuntimeError):
    """A required Stage 1 correctness condition did not hold."""

    def __init__(self, gate: str, message: str):
        super().__init__(message)
        self.gate = gate


def cached_stage1_clean_pass(model: Any, tokens: torch.Tensor) -> tuple[CleanPass, dict[int, torch.Tensor], torch.Tensor]:
    """Cache Stage-1 ``z`` values plus the layer-8 residual reference.

    ``hook_z`` is cached for every layer because the frozen Stage-1 sweep is a
    source-to-base replacement at that hook.  No source A/B/C head sweep and no
    latent activation is requested here.
    """
    z_names = {f"blocks.{layer}.attn.{HOOK_Z}" for layer in range(LAYER_COUNT)}
    residual_name = "blocks.8.hook_resid_pre"
    attn_out_name = "blocks.0.hook_attn_out"
    wanted = z_names | {residual_name, attn_out_name}
    z_parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(LAYER_COUNT)}
    residual_parts: list[torch.Tensor] = []
    attn_out_parts: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], PATCH_BATCH_SIZE):
        stop = min(start + PATCH_BATCH_SIZE, tokens.shape[0])
        with torch.no_grad():
            result, cache = model.run_with_cache(
                tokens[start:stop], names_filter=lambda name: name in wanted, return_type=None
            )
        if result is not None:
            raise Stage1Stop("clean_cache_return_type", "Expected return_type=None to avoid materialising full logits.")
        for layer in range(LAYER_COUNT):
            z_parts[layer].append(cache[f"blocks.{layer}.attn.{HOOK_Z}"].detach().float().cpu().clone())
        residual_parts.append(cache[residual_name].detach().float().cpu().clone())
        attn_out_parts.append(cache[attn_out_name].detach().float().cpu().clone())
        del cache
    z = {layer: torch.cat(z_parts[layer]) for layer in range(LAYER_COUNT)}
    residual = torch.cat(residual_parts)
    attn_out = torch.cat(attn_out_parts)
    # ``CleanPass.logits`` is not needed by any residual operation in this run.
    # Stage-1 readout values are extracted from exact model logits below in
    # microbatches; retaining a vocabulary-sized clean tensor would OOM CPU.
    return CleanPass(logits=torch.empty(0, dtype=torch.float32), residuals={8: residual}), z, attn_out


def cached_v_values(model: Any, tokens: torch.Tensor, layers: Iterable[int]) -> dict[int, torch.Tensor]:
    """Cache only the layers selected by the Stage-1 top-24 rule for ``v``."""
    selected = tuple(sorted(set(int(layer) for layer in layers)))
    names = {f"blocks.{layer}.attn.{HOOK_V}" for layer in selected}
    parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in selected}
    for start in range(0, tokens.shape[0], PATCH_BATCH_SIZE):
        stop = min(start + PATCH_BATCH_SIZE, tokens.shape[0])
        with torch.no_grad():
            result, cache = model.run_with_cache(
                tokens[start:stop], names_filter=lambda name: name in names, return_type=None
            )
        if result is not None:
            raise Stage1Stop("v_cache_return_type", "Expected return_type=None to avoid materialising full logits.")
        for layer in selected:
            parts[layer].append(cache[f"blocks.{layer}.attn.{HOOK_V}"].detach().float().cpu().clone())
        del cache
    return {layer: torch.cat(parts[layer]) for layer in selected}


def assert_hook_z_layout(model: Any, z_cache: dict[int, torch.Tensor], attn_out: torch.Tensor) -> dict[str, Any]:
    """Verify TransformerLens's pre-``W_O`` head axis against the installed model."""
    cfg = model.cfg
    expected_heads = int(cfg.n_heads)
    expected_d_head = int(cfg.d_head)
    shapes = {str(layer): list(z_cache[layer].shape) for layer in range(LAYER_COUNT)}
    layout_ok = all(
        tensor.ndim == 4
        and tensor.shape[2] == expected_heads
        and tensor.shape[3] == expected_d_head
        for tensor in z_cache.values()
    )
    if not layout_ok:
        raise Stage1Stop("hook_z_layout", f"Unexpected hook_z shapes: {shapes}")

    # The installed AbstractAttention implementation documents hook_z as
    # [batch, pos, head_index, d_head] and applies W_O [head_index, d_head,
    # d_model] after this hook.  Independently check that numerical contract on
    # layer 0 against the block's actual attention output.
    layer = 0
    attention = model.blocks[layer].attn
    w_o = attention.W_O.detach().float().cpu()
    b_o = attention.b_O.detach().float().cpu()
    if tuple(w_o.shape) != (expected_heads, expected_d_head, int(cfg.d_model)):
        raise Stage1Stop("hook_z_w_o_layout", f"Unexpected W_O shape {tuple(w_o.shape)}")
    # Keep this installed-version semantic check at one fixed execution
    # microbatch: materialising per-head d_model results for all retained edits
    # would be a large diagnostic-only tensor, while the head-axis contract is
    # identical in every microbatch.
    z_probe = z_cache[layer][:PATCH_BATCH_SIZE]
    attn_out_probe = attn_out[:PATCH_BATCH_SIZE]
    reconstructed = torch.einsum("bphd,hdm->bphm", z_probe, w_o).sum(dim=2) + b_o
    max_abs = float((reconstructed - attn_out_probe).abs().max())
    allclose = bool(torch.allclose(reconstructed, attn_out_probe, rtol=1e-5, atol=1e-5))
    if not allclose:
        raise Stage1Stop(
            "hook_z_pre_WO_reconstruction",
            f"sum(z @ W_O) did not reproduce layer-0 attention output; max_abs={max_abs:.6g}",
        )
    return {
        "installed_transformerlens_contract": "hook_z [batch, pos, head_index, d_head] before W_O [head_index, d_head, d_model]",
        "n_heads": expected_heads,
        "d_head": expected_d_head,
        "d_model": int(cfg.d_model),
        "hook_z_shapes_by_layer": shapes,
        "W_O_shape_layer_0": list(w_o.shape),
        "sum_z_matmul_W_O_vs_hook_attn_out_max_abs": max_abs,
        "sum_z_matmul_W_O_vs_hook_attn_out_allclose_rtol_1e-5_atol_1e-5": allclose,
        "sum_z_matmul_W_O_probe_batch_size": PATCH_BATCH_SIZE,
        "head_dimension_index": 2,
    }


def _source_values(
    activations: torch.Tensor,
    source_indices: torch.Tensor,
    source_positions: torch.Tensor,
    head: int | None,
) -> torch.Tensor:
    rows = torch.arange(source_indices.numel())
    if head is None:
        return activations[source_indices, source_positions, :, :].clone()
    return activations[source_indices, source_positions, head, :].clone()


def _patch_hook(
    activation: torch.Tensor,
    *,
    base_positions: torch.Tensor,
    replacement: torch.Tensor,
    head: int | None,
    expected_heads: int,
    expected_d_head: int,
) -> torch.Tensor:
    if activation.ndim != 4 or activation.shape[2] != expected_heads or activation.shape[3] != expected_d_head:
        raise Stage1Stop("patch_hook_layout", f"Hook activation has unexpected shape {tuple(activation.shape)}")
    if activation.shape[0] != replacement.shape[0]:
        raise Stage1Stop(
            "patch_hook_batch", f"Hook batch {activation.shape[0]} does not match replacement batch {replacement.shape[0]}"
        )
    rows = torch.arange(activation.shape[0], device=activation.device)
    positions = base_positions.to(activation.device)
    values = replacement.to(device=activation.device, dtype=activation.dtype)
    if head is None:
        activation[rows, positions, :, :] = values
    else:
        activation[rows, positions, head, :] = values
    return activation


class AttentionPatchRunner:
    """Run logical source-to-base patches in exact-logit microbatches."""

    def __init__(self, model: Any):
        self.model = model
        self.expected_heads = int(model.cfg.n_heads)
        self.expected_d_head = int(model.cfg.d_head)
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _slices(size: int) -> Iterable[slice]:
        for start in range(0, size, PATCH_BATCH_SIZE):
            yield slice(start, min(start + PATCH_BATCH_SIZE, size))

    def _record(
        self,
        *,
        label: str,
        hook_kind: str,
        layer: int | str,
        head: int | str,
        batch: int,
        seconds: float,
        forward_calls: int,
    ) -> None:
        self.records.append(
            {
                "label": label,
                "hook": hook_kind,
                "layer": layer,
                "head": head,
                "batch": batch,
                "microbatch_size": PATCH_BATCH_SIZE,
                "forward_calls": forward_calls,
                "seconds": seconds,
            }
        )

    def run_one(
        self,
        *,
        hook_kind: str,
        layer: int,
        head: int,
        base_tokens: torch.Tensor,
        base_positions: torch.Tensor,
        replacement: torch.Tensor,
        label: str,
        lengths: torch.Tensor,
        is_id: int,
        are_id: int,
    ) -> torch.Tensor:
        started = time.perf_counter()
        readouts: list[torch.Tensor] = []
        forward_calls = 0
        for chunk in self._slices(base_tokens.shape[0]):
            chunk_positions = base_positions[chunk]
            chunk_replacement = replacement[chunk]

            def hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
                del hook
                return _patch_hook(
                    activation,
                    base_positions=chunk_positions,
                    replacement=chunk_replacement,
                    head=head,
                    expected_heads=self.expected_heads,
                    expected_d_head=self.expected_d_head,
                )

            with torch.no_grad():
                logits = self.model.run_with_hooks(
                    base_tokens[chunk],
                    fwd_hooks=[(f"blocks.{layer}.attn.{hook_kind}", hook)],
                    return_type="logits",
                )
            readouts.append(logit_difference(logits, lengths[chunk], is_id, are_id).detach().float().cpu())
            del logits
            forward_calls += 1
        elapsed = time.perf_counter() - started
        self._record(
            label=label,
            hook_kind=hook_kind,
            layer=layer,
            head=head,
            batch=int(base_tokens.shape[0]),
            seconds=elapsed,
            forward_calls=forward_calls,
        )
        return torch.cat(readouts)

    def run_all_z(
        self,
        *,
        base_tokens: torch.Tensor,
        base_positions: torch.Tensor,
        replacements_by_layer: dict[int, torch.Tensor],
        label: str,
        lengths: torch.Tensor,
        is_id: int,
        are_id: int,
    ) -> torch.Tensor:
        started = time.perf_counter()
        readouts: list[torch.Tensor] = []
        forward_calls = 0
        for chunk in self._slices(base_tokens.shape[0]):
            chunk_positions = base_positions[chunk]
            hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = []
            for layer in range(LAYER_COUNT):
                replacement = replacements_by_layer[layer][chunk]

                def hook(activation: torch.Tensor, hook: Any, replacement: torch.Tensor = replacement) -> torch.Tensor:
                    del hook
                    return _patch_hook(
                        activation,
                        base_positions=chunk_positions,
                        replacement=replacement,
                        head=None,
                        expected_heads=self.expected_heads,
                        expected_d_head=self.expected_d_head,
                    )

                hooks.append((f"blocks.{layer}.attn.{HOOK_Z}", hook))
            with torch.no_grad():
                logits = self.model.run_with_hooks(base_tokens[chunk], fwd_hooks=hooks, return_type="logits")
            readouts.append(logit_difference(logits, lengths[chunk], is_id, are_id).detach().float().cpu())
            del logits
            forward_calls += 1
        elapsed = time.perf_counter() - started
        self._record(
            label=label,
            hook_kind=HOOK_Z,
            layer="all",
            head="all",
            batch=int(base_tokens.shape[0]),
            seconds=elapsed,
            forward_calls=forward_calls,
        )
        return torch.cat(readouts)

    def reconstruction_identity_all_z(
        self,
        *,
        base_tokens: torch.Tensor,
        base_positions: torch.Tensor,
        replacements_by_layer: dict[int, torch.Tensor],
        label: str,
    ) -> dict[str, Any]:
        """Compare clean and all-head base-own full logits without retaining either batch."""
        started = time.perf_counter()
        exact = True
        max_abs = 0.0
        forward_calls = 0
        for chunk in self._slices(base_tokens.shape[0]):
            with torch.no_grad():
                clean_logits = self.model(base_tokens[chunk], return_type="logits")
            chunk_positions = base_positions[chunk]
            hooks: list[tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]] = []
            for layer in range(LAYER_COUNT):
                replacement = replacements_by_layer[layer][chunk]

                def hook(activation: torch.Tensor, hook: Any, replacement: torch.Tensor = replacement) -> torch.Tensor:
                    del hook
                    return _patch_hook(
                        activation,
                        base_positions=chunk_positions,
                        replacement=replacement,
                        head=None,
                        expected_heads=self.expected_heads,
                        expected_d_head=self.expected_d_head,
                    )

                hooks.append((f"blocks.{layer}.attn.{HOOK_Z}", hook))
            with torch.no_grad():
                patched_logits = self.model.run_with_hooks(base_tokens[chunk], fwd_hooks=hooks, return_type="logits")
            exact = exact and bool(torch.equal(clean_logits, patched_logits))
            max_abs = max(max_abs, float((clean_logits - patched_logits).abs().max()))
            del clean_logits, patched_logits
            forward_calls += 2
        self._record(
            label=label,
            hook_kind=HOOK_Z,
            layer="all",
            head="all",
            batch=int(base_tokens.shape[0]),
            seconds=time.perf_counter() - started,
            forward_calls=forward_calls,
        )
        return {"bitwise_logits_equal": exact, "max_abs_logit_difference": max_abs}


def clean_readout_microbatched(
    model: Any,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    is_id: int,
    are_id: int,
) -> torch.Tensor:
    """Obtain the exact model-logit readout without retaining a full logits tensor."""
    values: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], PATCH_BATCH_SIZE):
        stop = min(start + PATCH_BATCH_SIZE, tokens.shape[0])
        with torch.no_grad():
            logits = model(tokens[start:stop], return_type="logits")
        values.append(logit_difference(logits, lengths[start:stop], is_id, are_id).detach().float().cpu())
        del logits
    return torch.cat(values)


def summarize_effect(
    *,
    patched_d: torch.Tensor,
    clean_base_d: torch.Tensor,
    signs: torch.Tensor,
) -> dict[str, Any]:
    raw = patched_d - clean_base_d
    aligned = raw * signs
    pair_aligned = aligned.reshape(-1, 2)
    return {
        "delta_d_raw": raw,
        "delta_d_sign_aligned": aligned,
        "d_patched": patched_d,
        "E_delta_d": float(aligned.mean()),
        "directed_sign_consistency_positive_fraction": float((aligned > 0).float().mean()),
        "minimal_pair_both_directions_positive_fraction": float(((pair_aligned[:, 0] > 0) & (pair_aligned[:, 1] > 0)).float().mean()),
    }


def assert_exact_zero(effect: dict[str, Any], *, label: str) -> dict[str, Any]:
    raw = effect["delta_d_raw"]
    exact = bool(torch.equal(raw, torch.zeros_like(raw)))
    max_abs = float(raw.abs().max())
    if not exact:
        raise Stage1Stop("self_patch_noop", f"{label} had nonzero self-patch delta_d; max_abs={max_abs:.6g}")
    return {"exact_zero_delta_d": exact, "max_abs_delta_d": max_abs}


def residual_reference_microbatched(
    engine: PatchEngine,
    *,
    clean: CleanPass,
    base_tokens: torch.Tensor,
    base_lengths: torch.Tensor,
    source_local_indices: torch.Tensor,
    base_positions: torch.Tensor,
    source_positions: torch.Tensor,
    signs: torch.Tensor,
    clean_base_d: torch.Tensor,
    is_id: int,
    are_id: int,
) -> dict[str, Any]:
    """Run calibrate.py's layer-8/both residual harness without full-batch logits."""
    deltas = source_delta(
        base_residual=clean.residuals[8],
        source_residual=clean.residuals[8][source_local_indices],
        base_positions=base_positions,
        source_positions=source_positions,
    )
    patched_values: list[torch.Tensor] = []
    for start in range(0, base_tokens.shape[0], PATCH_BATCH_SIZE):
        stop = min(start + PATCH_BATCH_SIZE, base_tokens.shape[0])
        chunk = slice(start, stop)
        logits = engine.run(
            layer=8,
            base_tokens=base_tokens[chunk],
            base_residual=clean.residuals[8][chunk],
            positions=base_positions[chunk],
            deltas=deltas[chunk],
            label="stage1_E_ref_layer8_residual_both",
        )
        patched_values.append(
            logit_difference(logits, base_lengths[chunk], is_id, are_id).detach().float().cpu()
        )
        del logits
    patched_d = torch.cat(patched_values)
    raw = patched_d - clean_base_d
    aligned = raw * signs
    pair_aligned = aligned.reshape(-1, 2)
    return {
        "E_delta_d": float(aligned.mean()),
        "delta_d_raw": raw,
        "delta_d_sign_aligned": aligned,
        "directed_sign_consistency_positive_fraction": float((aligned > 0).float().mean()),
        "minimal_pair_both_directions_positive_fraction": float(
            ((pair_aligned[:, 0] > 0) & (pair_aligned[:, 1] > 0)).float().mean()
        ),
    }


def load_calibration_projection() -> dict[str, Any]:
    calibration = json.loads((HERE / "calibration_results.json").read_text())
    projection = calibration["runtime"]["projection"]
    return {
        "calibration_schema": calibration["schema"],
        "stage1_forward_projection_seconds_144_z_plus_24_v": float(projection["forward_seconds"]["stage_1_expected"]),
        "z_noop_median_seconds_by_layer": projection["per_patch_cost_seconds"]["no_op_hook_z_by_start_layer"],
        "v_noop_median_seconds_by_layer": projection["per_patch_cost_seconds"]["no_op_hook_v_by_start_layer"],
        "projection_scope": "calibration's forward-only 144 single-head z patches plus 24 v patches; it excludes validation forwards and does not enumerate the required E_all joint z patch",
    }


def observed_projection_for_selected_v(calibration: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    z = calibration["z_noop_median_seconds_by_layer"]
    v = calibration["v_noop_median_seconds_by_layer"]
    z_total = sum(HEADS_PER_LAYER * float(z[str(layer)]) for layer in range(LAYER_COUNT))
    v_total = sum(float(v[str(row["layer"])]) for row in selected)
    return z_total + v_total


def write_notes(manifest: dict[str, Any]) -> None:
    lines = [
        "# Experiment 05 Stage 1 notes",
        "",
        f"- Status: `{manifest['status']}`." + (f" Failed check: `{manifest['failed_gate']}`." if manifest.get("failed_gate") else ""),
        "- Scope: frozen Stage 1 only. The run contains the true single-flip `z@final` head sweep, required all-head `E_all`, `E_ref`, and top-24 `v@subject`; it contains no Stage 2 or Stage 3 quantity.",
    ]
    if manifest.get("environment"):
        env = manifest["environment"]
        lines.append(f"- Environment: device `{env['device']}`, torch `{env['torch']}`, platform `{env['platform']}`.")
    if manifest.get("gate_A"):
        gate = manifest["gate_A"]
        lines.extend(
            [
                "",
                "## Stimuli and indexing",
                "",
                f"- Seed `{SEED}`; retained {gate['retained_pairs']}/{gate['generated_pairs']} pairs (both-correct={gate['both_members_signed_correct_fraction']:.3f}, median clean d_gap={gate['median_d_gap_all_generated_pairs']:.3f}); pass={gate['passed']}.",
            ]
        )
    if manifest.get("position_indexing"):
        position = manifest["position_indexing"]
        lines.append(
            f"- Final indices were computed per sequence as `lengths - 1`, never hard-coded; {position['base_sequences_differing_from_modal_length']}/{position['base_sequence_count']} base sequences differ from the modal length `{position['base_modal_length']}`. Base and source final indices match for all {position['directed_edit_count']} directed single-flip edits."
        )
    if manifest.get("effects"):
        effects = manifest["effects"]
        lines.extend(
            [
                "",
                "## Required reference effects",
                "",
                f"- `E_all` (all 144 `hook_z@final` heads): {effects['E_all']['E_delta_d']:.6f}.",
                f"- `E_ref` (layer-8 residual interchange at `both`): {effects['E_ref']['E_delta_d']:.6f}.",
                f"- Clean `d_gap` (mean sign-aligned source-minus-base): {effects['d_gap']['mean']:.6f}.",
                f"- Additivity report, `sum_h E({{h}})`: {effects['single_head_E_sum']:.6f}; `E_all`: {effects['E_all']['E_delta_d']:.6f}.",
            ]
        )
    ranking = manifest.get("ranking")
    if ranking:
        lines.extend(["", "## `hook_z@final` signed ranking", "", "| rank | head | signed E(delta_d) | both flip directions positive |", "|---:|:---:|---:|---:|"])
        for row in ranking["signed_descending"]:
            lines.append(
                f"| {row['rank']} | L{row['layer']}H{row['head']} | {row['E_delta_d']:.6f} | {row['minimal_pair_both_directions_positive_fraction']:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Candidate suppressors (all signed-negative heads; no unregistered magnitude cutoff)",
                "",
                "| signed rank | head | signed E(delta_d) | both flip directions positive |",
                "|---:|:---:|---:|---:|",
            ]
        )
        suppressors = ranking["candidate_suppressors_all_signed_negative"]
        if suppressors:
            for row in suppressors:
                lines.append(
                    f"| {row['rank']} | L{row['layer']}H{row['head']} | {row['E_delta_d']:.6f} | {row['minimal_pair_both_directions_positive_fraction']:.3f} |"
                )
        else:
            lines.append("| — | — | none | — |")
    v_rows = manifest.get("v_subject_top24")
    if v_rows:
        lines.extend(["", "## Top-24 signed `hook_v@subject` measurements", "", "| z rank | head | signed E(delta_d) | both flip directions positive |", "|---:|:---:|---:|---:|"])
        for row in v_rows:
            lines.append(
                f"| {row['z_signed_rank']} | L{row['layer']}H{row['head']} | {row['E_delta_d']:.6f} | {row['minimal_pair_both_directions_positive_fraction']:.3f} |"
            )
    checks = manifest.get("correctness_checks")
    if checks:
        lines.extend(
            [
                "",
                "## Correctness checks",
                "",
                f"- `hook_z` layout: head axis `{checks['hook_z_layout']['head_dimension_index']}`; `sum(z @ W_O)` reproduced layer-0 `hook_attn_out` with max absolute error {checks['hook_z_layout']['sum_z_matmul_W_O_vs_hook_attn_out_max_abs']:.3g}; pass={checks['hook_z_layout']['sum_z_matmul_W_O_vs_hook_attn_out_allclose_rtol_1e-5_atol_1e-5']}.",
                f"- All-144 base-own `z@final` reconstruction: bitwise logits={checks['all_head_base_own_reconstruction']['bitwise_logits_equal']}; max absolute logit difference={checks['all_head_base_own_reconstruction']['max_abs_logit_difference']:.3g}.",
                f"- Single-head base-own self-patches: all 144 exact `delta_d == 0`={checks['self_patch_noop']['all_144_heads_exact_zero']}; maximum absolute delta={checks['self_patch_noop']['maximum_abs_delta_d_over_144_heads']:.3g}.",
                f"- Experiment-04 residual self-tests: zero bitwise={checks['experiment04_residual_self_tests']['zero_intervention_bitwise_identity']}; start-at-layer-8 max abs={checks['experiment04_residual_self_tests']['start_at_layer8_max_abs']:.3g}; prompt-swap max abs={checks['experiment04_residual_self_tests']['prompt_swap_exactness_max_abs']:.3g}.",
            ]
        )
    runtime = manifest.get("runtime")
    if runtime:
        lines.extend(
            [
                "",
                "## Runtime",
                "",
                f"- Full measured wall clock: {runtime['wall_clock_seconds']:.2f} s ({runtime['wall_clock_seconds'] / 60.0:.2f} min).",
                f"- Every logical patch covered all retained directed edits in deterministic microbatches of {runtime['logical_patch_microbatch_size']} sequences; readouts still come from exact model logits.",
                f"- Calibration's forward-only 144-z + 24-v projection: {runtime['calibration_stage1_forward_projection_seconds']:.2f} s. Actual 144-z + selected-24-v patch forwards: {runtime['actual_predeclared_144_z_plus_24_v_patch_seconds']:.2f} s.",
                f"- Required joint `E_all` patch: {runtime['E_all_joint_patch_seconds']:.2f} s; it was not enumerated in that calibration Stage-1 count. Required validation and reference forwards are additional to the calibration projection.",
            ]
        )
    defects = manifest.get("design_defects_observed", [])
    if defects:
        lines.extend(["", "## Observed frozen-design accounting defect", ""])
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
        "schema": "exp05-number-agreement-stage1-v1; frozen Stage 1 only; CPU float32; signed source-to-base head replacements",
        "status": "running",
        "failed_gate": None,
        "configuration": {
            "seed": SEED,
            "requested_pairs": REQUESTED_PAIRS,
            "retention": "Experiment 04 Gate A, base family only",
            "source": "true single-flip opposite member of each retained minimal pair",
            "both_flip_directions": True,
            "z_sweep": "all 144 heads, hook_z at final position only",
            "E_all": "joint all-144 hook_z at final position only",
            "E_ref": "layer-8 residual interchange at positions both",
            "v_sweep": "top 24 signed z heads only, hook_v at subject position only",
            "patch_microbatch_size": PATCH_BATCH_SIZE,
            "readout": "exact TransformerLens logits; vocabulary tensors are microbatched and released",
            "device": "cpu",
            "dtype": "float32",
            "offline_env": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MPLBACKEND")},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "gate_A": None,
        "position_indexing": None,
        "correctness_checks": None,
        "directed_edits": None,
        "effects": None,
        "heads": None,
        "ranking": None,
        "v_subject_top24": None,
        "runtime": None,
        "design_defects_observed": [
            "The calibration projection describes its Stage 1 count as 144 z patches plus 24 v patches, but the frozen Stage 1 requirements also require one measured all-144-head joint z patch for E_all. The calibration is already labelled a lower bound; this additional omission makes its Stage-1 forward count one joint z forward too low. This run reports that forward separately and does not alter DESIGN.md."
        ],
    }
    try:
        from pilot import load_model  # imported late so this file stays inspectable without model loading

        model_started = time.perf_counter()
        model = load_model()
        model_load_seconds = time.perf_counter() - model_started
        token_ids = {text: require_one_token(model.tokenizer, text) for text in (" is", " are")}

        stimuli = build_stimuli(model.tokenizer, REQUESTED_PAIRS, SEED)
        clean_d_all = clean_readout_microbatched(
            model, stimuli.tokens, stimuli.lengths, token_ids[" is"], token_ids[" are"]
        )
        # gate_a only consumes d = logit(are) - logit(is). Preserve that exact
        # helper while holding only its two relevant logit coordinates.
        compact_width = max(token_ids.values()) + 1
        compact_logits = torch.zeros((stimuli.tokens.shape[0], stimuli.tokens.shape[1], compact_width), dtype=torch.float32)
        compact_rows = torch.arange(stimuli.tokens.shape[0])
        compact_finals = stimuli.lengths - 1
        compact_logits[compact_rows, compact_finals, token_ids[" are"]] = clean_d_all
        gate_clean = CleanPass(logits=compact_logits, residuals={})
        gate, retained_pairs, gate_number = gate_a(stimuli, gate_clean, token_ids[" is"], token_ids[" are"])
        if not torch.equal(gate_number, clean_d_all):
            raise Stage1Stop("gate_a_readout", "Compact Gate-A logit difference did not reproduce exact clean readout.")
        del compact_logits, gate_clean
        manifest["gate_A"] = gate
        if not gate["passed"]:
            raise Stage1Stop("Gate_A_base", f"Base Gate A failed: {gate}")

        base_indices, source_indices, signs = directed_indices(REQUESTED_PAIRS, retained_pairs)
        base_final = positions_for_kind(stimuli, base_indices, "final").squeeze(1)
        source_final = positions_for_kind(stimuli, source_indices, "final").squeeze(1)
        base_subject = positions_for_kind(stimuli, base_indices, "subject").squeeze(1)
        source_subject = positions_for_kind(stimuli, source_indices, "subject").squeeze(1)
        if not torch.equal(base_final, stimuli.lengths[base_indices] - 1):
            raise Stage1Stop("final_position_formula", "Final positions were not computed as lengths - 1.")
        if not torch.equal(base_final, source_final):
            raise Stage1Stop("single_flip_final_alignment", "Within-pair source and base final positions differ.")
        if not torch.equal(base_subject, source_subject):
            raise Stage1Stop("single_flip_subject_alignment", "Within-pair source and base subject positions differ.")
        modal_length, modal_count = Counter(stimuli.lengths.tolist()).most_common(1)[0]
        manifest["position_indexing"] = {
            "final_position_formula": "per-sequence lengths - 1",
            "final_positions_hard_coded": False,
            "base_sequence_count": int(stimuli.lengths.numel()),
            "base_sequence_length_counts": {str(key): int(value) for key, value in sorted(Counter(stimuli.lengths.tolist()).items())},
            "base_modal_length": int(modal_length),
            "base_sequences_differing_from_modal_length": int(stimuli.lengths.numel() - modal_count),
            "directed_edit_count": int(base_indices.numel()),
            "source_and_base_final_positions_equal_all_directed_edits": True,
            "source_and_base_subject_positions_equal_all_directed_edits": True,
        }
        manifest["directed_edits"] = {
            "pair_indices": [int(pair) for pair in retained_pairs for _ in (0, 1)],
            "base_item_indices": base_indices.tolist(),
            "source_item_indices": source_indices.tolist(),
            "sign_alignment": signs.tolist(),
            "direction": ["singular_to_plural", "plural_to_singular"] * len(retained_pairs),
        }

        # Cache the source/base member of each directed edit in the exact same
        # microbatches used by every patch.  This preserves bitwise base-own
        # reconstruction while still keeping only 32 vocabulary-logit rows live.
        base_tokens = stimuli.tokens[base_indices]
        base_lengths = stimuli.lengths[base_indices]
        clean, z_cache, attn_out = cached_stage1_clean_pass(model, base_tokens)
        local_indices = torch.arange(base_indices.numel())
        source_local_indices = local_indices ^ 1
        if not torch.equal(local_indices // PATCH_BATCH_SIZE, source_local_indices // PATCH_BATCH_SIZE):
            raise Stage1Stop("source_microbatch_alignment", "A single-flip source was split from its base across cache/patch microbatches.")
        base_clean_d = clean_readout_microbatched(
            model, base_tokens, base_lengths, token_ids[" is"], token_ids[" are"]
        )
        local_final = base_final
        source_local_final = local_final[source_local_indices]
        local_subject = base_subject
        source_local_subject = local_subject[source_local_indices]
        hook_layout = assert_hook_z_layout(model, z_cache, attn_out)
        runner = AttentionPatchRunner(model)

        # Required reconstruction identity: replacing every final-position head
        # with its own cached values must be exactly the clean forward.
        own_all = {layer: _source_values(z_cache[layer], local_indices, local_final, None) for layer in range(LAYER_COUNT)}
        reconstruction = runner.reconstruction_identity_all_z(
            base_tokens=base_tokens,
            base_positions=local_final,
            replacements_by_layer=own_all,
            label="check_all_144_base_own_z_final_reconstruction",
        )
        reconstruction_exact = bool(reconstruction["bitwise_logits_equal"])
        reconstruction_max = float(reconstruction["max_abs_logit_difference"])
        if not reconstruction_exact:
            raise Stage1Stop(
                "all_head_base_own_reconstruction",
                f"All-head base-own z patch did not reproduce clean logits bitwise; max_abs={reconstruction_max:.6g}",
            )

        # Required self-patch no-op: execute every individual head rather than
        # infer it from the all-head identity.
        self_checks: list[dict[str, Any]] = []
        for layer in range(LAYER_COUNT):
            for head in range(HEADS_PER_LAYER):
                replacement = _source_values(z_cache[layer], local_indices, local_final, head)
                patched_d = runner.run_one(
                    hook_kind=HOOK_Z,
                    layer=layer,
                    head=head,
                    base_tokens=base_tokens,
                    base_positions=local_final,
                    replacement=replacement,
                    label="check_single_head_base_own_z_final_noop",
                    lengths=base_lengths,
                    is_id=token_ids[" is"],
                    are_id=token_ids[" are"],
                )
                effect = summarize_effect(
                    patched_d=patched_d,
                    clean_base_d=base_clean_d,
                    signs=signs,
                )
                self_checks.append({"layer": layer, "head": head, **assert_exact_zero(effect, label=f"L{layer}H{head}")})

        # The Stage-1 measurement: true single-flip source -> base through every
        # one-head z replacement at the final position.
        head_rows: list[dict[str, Any]] = []
        for layer in range(LAYER_COUNT):
            for head in range(HEADS_PER_LAYER):
                replacement = _source_values(z_cache[layer], source_local_indices, source_local_final, head)
                patched_d = runner.run_one(
                    hook_kind=HOOK_Z,
                    layer=layer,
                    head=head,
                    base_tokens=base_tokens,
                    base_positions=local_final,
                    replacement=replacement,
                    label="stage1_true_single_flip_z_final",
                    lengths=base_lengths,
                    is_id=token_ids[" is"],
                    are_id=token_ids[" are"],
                )
                effect = summarize_effect(
                    patched_d=patched_d,
                    clean_base_d=base_clean_d,
                    signs=signs,
                )
                head_rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "hook": HOOK_Z,
                        "write_position": "final",
                        "source": "true_single_flip",
                        **effect,
                    }
                )

        # Required same-family denominator: all 144 source z values at once.
        source_all = {
            layer: _source_values(z_cache[layer], source_local_indices, source_local_final, None)
            for layer in range(LAYER_COUNT)
        }
        e_all_d = runner.run_all_z(
            base_tokens=base_tokens,
            base_positions=local_final,
            replacements_by_layer=source_all,
            label="stage1_E_all_joint_144_z_final",
            lengths=base_lengths,
            is_id=token_ids[" is"],
            are_id=token_ids[" are"],
        )
        e_all = summarize_effect(
            patched_d=e_all_d,
            clean_base_d=base_clean_d,
            signs=signs,
        )

        # Required cross-family reference, reused literally from the calibrated
        # residual harness.  It is neither a head sweep nor a later-stage arm.
        residual_engine = PatchEngine(model, start_at_layer8=True)
        e_ref = residual_reference_microbatched(
            residual_engine,
            clean=clean,
            base_tokens=base_tokens,
            base_lengths=base_lengths,
            source_local_indices=source_local_indices,
            base_positions=torch.stack((local_subject, local_final), dim=1),
            source_positions=torch.stack((source_local_subject, source_local_final), dim=1),
            signs=signs,
            clean_base_d=base_clean_d,
            is_id=token_ids[" is"],
            are_id=token_ids[" are"],
        )
        clean_gap = (base_clean_d[source_local_indices] - base_clean_d) * signs

        ranked_rows = sorted(head_rows, key=lambda row: (-float(row["E_delta_d"]), int(row["layer"]), int(row["head"])))
        ranking = []
        for rank, row in enumerate(ranked_rows, start=1):
            ranking.append(
                {
                    "rank": rank,
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "E_delta_d": float(row["E_delta_d"]),
                    "directed_sign_consistency_positive_fraction": float(row["directed_sign_consistency_positive_fraction"]),
                    "minimal_pair_both_directions_positive_fraction": float(row["minimal_pair_both_directions_positive_fraction"]),
                }
            )
        selected_top24 = ranking[:24]

        # The frozen top-24 rule is applied only after all z effects have been
        # recorded.  Cache only those selected v layers, then do exactly 24 v
        # source-to-base interventions at the subject position.
        v_cache = cached_v_values(model, base_tokens, (row["layer"] for row in selected_top24))
        v_rows: list[dict[str, Any]] = []
        for z_row in selected_top24:
            layer, head = int(z_row["layer"]), int(z_row["head"])
            replacement = _source_values(v_cache[layer], source_local_indices, source_local_subject, head)
            patched_d = runner.run_one(
                hook_kind=HOOK_V,
                layer=layer,
                head=head,
                base_tokens=base_tokens,
                base_positions=local_subject,
                replacement=replacement,
                label="stage1_top24_v_subject",
                lengths=base_lengths,
                is_id=token_ids[" is"],
                are_id=token_ids[" are"],
            )
            effect = summarize_effect(
                patched_d=patched_d,
                clean_base_d=base_clean_d,
                signs=signs,
            )
            v_rows.append(
                {
                    "z_signed_rank": int(z_row["rank"]),
                    "layer": layer,
                    "head": head,
                    "hook": HOOK_V,
                    "write_position": "subject",
                    "source": "true_single_flip",
                    **effect,
                }
            )

        # The residual tests are inherited exactly from calibrate.py / Exp04.
        # Reset the experiment seed afterward for a deterministic final state;
        # no stochastic measurement follows this reset.
        residual_self_tests = run_self_tests(model, token_ids)
        set_determinism(SEED)

        calibration = load_calibration_projection()
        actual_by_label: dict[str, float] = {}
        for record in runner.records:
            actual_by_label[record["label"]] = actual_by_label.get(record["label"], 0.0) + float(record["seconds"])
        forward_calls_by_label: dict[str, int] = {}
        for record in runner.records:
            forward_calls_by_label[record["label"]] = forward_calls_by_label.get(record["label"], 0) + int(record["forward_calls"])
        expected_selected = observed_projection_for_selected_v(calibration, selected_top24)
        self_max = max(float(row["max_abs_delta_d"]) for row in self_checks)
        manifest["correctness_checks"] = {
            "hook_z_layout": hook_layout,
            "all_head_base_own_reconstruction": {
                "bitwise_logits_equal": reconstruction_exact,
                "max_abs_logit_difference": reconstruction_max,
                "scope": "all 144 hook_z heads at final position, retained directed edits",
            },
            "self_patch_noop": {
                "all_144_heads_exact_zero": bool(all(row["exact_zero_delta_d"] for row in self_checks)),
                "maximum_abs_delta_d_over_144_heads": self_max,
                "per_head": self_checks,
                "scope": "one head at a time, source equals base, final position, retained directed edits",
            },
            "experiment04_residual_self_tests": residual_self_tests,
        }
        manifest["effects"] = {
            "E_all": e_all,
            "E_ref": e_ref,
            "d_gap": {
                "mean": float(clean_gap.mean()),
                "median": float(clean_gap.median()),
                "min": float(clean_gap.min()),
                "max": float(clean_gap.max()),
                "per_directed_edit_sign_aligned": clean_gap,
            },
            "single_head_E_sum": float(sum(float(row["E_delta_d"]) for row in head_rows)),
        }
        manifest["heads"] = head_rows
        manifest["ranking"] = {
            "sort": "signed E(delta_d), descending; ties layer then head",
            "signed_descending": ranking,
            "candidate_suppressors_definition": "all heads with signed E(delta_d) < 0; exhaustive reporting avoids an unregistered magnitude cutoff",
            "candidate_suppressors_all_signed_negative": [row for row in ranking if row["E_delta_d"] < 0.0],
        }
        manifest["v_subject_top24"] = v_rows
        manifest["runtime"] = {
            "model_load_seconds": model_load_seconds,
            "wall_clock_seconds": time.perf_counter() - started,
            "calibration_stage1_forward_projection_seconds": calibration["stage1_forward_projection_seconds_144_z_plus_24_v"],
            "calibration_projection_with_actual_top24_layer_allocation_seconds": expected_selected,
            "actual_predeclared_144_z_plus_24_v_patch_seconds": actual_by_label.get("stage1_true_single_flip_z_final", 0.0)
            + actual_by_label.get("stage1_top24_v_subject", 0.0),
            "actual_true_144_single_head_z_patch_seconds": actual_by_label.get("stage1_true_single_flip_z_final", 0.0),
            "actual_top24_v_patch_seconds": actual_by_label.get("stage1_top24_v_subject", 0.0),
            "E_all_joint_patch_seconds": actual_by_label.get("stage1_E_all_joint_144_z_final", 0.0),
            "all_144_head_base_own_reconstruction_seconds": actual_by_label.get("check_all_144_base_own_z_final_reconstruction", 0.0),
            "all_144_single_head_self_patch_seconds": actual_by_label.get("check_single_head_base_own_z_final_noop", 0.0),
            "logical_patch_microbatch_size": PATCH_BATCH_SIZE,
            "attention_patch_forward_calls_by_label": forward_calls_by_label,
            "E_ref_residual_forward_calls": len(residual_engine.records),
            "calibration_projection_scope": calibration["projection_scope"],
        }
        manifest["status"] = "completed_stage1_only"
        return manifest
    except (Stage1Stop, CalibrationStop, GateStop) as exc:
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
    print(f"Stage 1 status={result['status']}; wrote {RESULTS.name} and {NOTES.name}", flush=True)

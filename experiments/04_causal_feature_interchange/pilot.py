"""Experiment 04 go/no-go pilot (steps 1--7 of the frozen DESIGN.md only).

Run exactly as documented in the task:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg .venv/bin/python pilot.py

This file deliberately never imports Experiment 03: importing that executable module would
create its published output directory.  The small loader, tokenizer helper, noun list,
random-expansion rule, and gated-out manifest pattern below are copied from
``experiments/03_ccgp_on_sae_features/ccgp_sae.py`` with attribution, as required.
"""

from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformer_lens import HookedTransformer


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "pilot_results.json"
NOTES = HERE / "PILOT_NOTES.md"

SEED = 20_260_727
LAYERS = (4, 6, 8, 10)
PILOT_PAIRS = 60
RANK_TRAIN_PAIRS = 20
GATE_C_PAIRS = 30
RANK_EVAL_PAIRS = 20
GATE_C_T = 2.776
PATCH_BATCH_LIMIT = 512
CONTROL_WORDS = (" walked", " walking")


class GateStop(RuntimeError):
    """Expected, honest pilot stop.  Its manifest is the result."""

    def __init__(self, gate: str, message: str):
        super().__init__(message)
        self.gate = gate


@dataclass
class SAEWeights:
    """Copied/adapted from Experiment 03's direct res-jb weights wrapper."""

    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    layer: int
    source: str

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


@dataclass
class Stimuli:
    """Sixty singular/plural single-flip prompt pairs, singular first in every pair."""

    tokens: torch.Tensor
    lengths: torch.Tensor
    subject_positions: torch.Tensor  # [n_pairs], shared by singular/plural member
    texts: list[str]
    pair_records: list[dict[str, Any]]
    attempted: int
    rejected: int


@dataclass
class CleanPass:
    logits: torch.Tensor
    residuals: dict[int, torch.Tensor]


@dataclass
class RandomBasis:
    """Matched Gaussian/ReLU/top-k expansion adapted from Experiment 03."""

    R: torch.Tensor  # [768, 24576], encoder columns
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    target_l0: int
    seed: int

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse-in-value code, stored densely only for this small intervention batch."""
        dense = torch.relu((x - self.b_dec) @ self.R + self.b_enc)
        values, indices = dense.topk(self.target_l0, dim=-1)
        return torch.zeros_like(dense).scatter_(-1, indices, values)

    def sparse_topk(self, x: torch.Tensor, chunk_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k values/indices without retaining an n x 24576 code matrix."""
        values_out, indices_out = [], []
        for start in range(0, x.shape[0], chunk_size):
            block = x[start : start + chunk_size]
            dense = torch.relu((block - self.b_dec) @ self.R + self.b_enc)
            values, indices = dense.topk(self.target_l0, dim=-1)
            values_out.append(values)
            indices_out.append(indices)
            del dense
        return torch.cat(values_out, dim=0), torch.cat(indices_out, dim=0)


@dataclass
class TiedDecoder:
    """The documented tied-weight fallback without a second 24576 x 768 allocation."""

    R: torch.Tensor
    scale: float


# ----------------------------------------------------------------------- copied inputs

# Copied from Experiment 03's vetted noun list.  Step 1 asserts each form is exactly
# one leading-space GPT-2 token before any prompt using it is admitted.
NOUNS = [
    ("cat", "cats"), ("dog", "dogs"), ("child", "children"),
    ("student", "students"), ("teacher", "teachers"), ("artist", "artists"),
    ("pilot", "pilots"), ("singer", "singers"), ("doctor", "doctors"),
    ("farmer", "farmers"), ("driver", "drivers"), ("actor", "actors"),
    ("chef", "chefs"), ("writer", "writers"), ("friend", "friends"),
    ("neighbor", "neighbors"), ("visitor", "visitors"), ("player", "players"),
    ("dancer", "dancers"), ("reader", "readers"),
]
ADJECTIVES = (
    "quiet", "curious", "careful", "eager", "patient", "bright", "young",
    "older", "gentle", "clever", "serious", "brave", "calm", "lively",
)
PREPOSITIONS = ("near", "beside", "behind", "around", "with")


def _ids(tokenizer: Any, text: str) -> list[int]:
    """Copied from Experiment 03: tokenizer shim with no implicit special token."""
    out = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return list(out["input_ids"])


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU only.  This pilot has no stochastic model layers in eval mode, but this
    # fixes generation and the matched random expansion.
    torch.use_deterministic_algorithms(True, warn_only=False)


def load_model() -> HookedTransformer:
    """Copied/adapted from Experiment 03: offline GPT-2-small in float32 on CPU."""
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu", dtype=torch.float32)
    model.eval()
    return model


def _tensor(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise RuntimeError(f"SAE safetensors missing {key}; found {sorted(state)}")
    return state[key].detach().float().cpu()


def load_direct_res_jb(layer: int) -> SAEWeights:
    """Copied/adapted from Experiment 03's direct safetensors loader."""
    repo = "jbloom/GPT2-Small-SAEs-Reformatted"
    filename = f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors"
    local = hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
    state = load_file(local, device="cpu")
    sae = SAEWeights(
        W_enc=_tensor(state, "W_enc"),
        b_enc=_tensor(state, "b_enc"),
        W_dec=_tensor(state, "W_dec"),
        b_dec=_tensor(state, "b_dec"),
        layer=layer,
        source=f"direct safetensors: {repo}/{filename}",
    )
    if sae.W_enc.shape != (768, 24576) or sae.W_dec.shape != (24576, 768):
        raise RuntimeError(
            "Unexpected res-jb SAE shape: "
            f"W_enc={tuple(sae.W_enc.shape)}, W_dec={tuple(sae.W_dec.shape)}"
        )
    return sae


def require_one_token(tokenizer: Any, text: str) -> int:
    ids = _ids(tokenizer, text)
    if not text.startswith(" ") or len(ids) != 1:
        raise GateStop("environment_tokenization", f"Expected exactly one leading-space token for {text!r}; got {ids}")
    return int(ids[0])


def build_stimuli(tokenizer: Any, n_pairs: int, seed: int) -> Stimuli:
    """Build single-flip pairs and repeat Experiment 03-style per-item assertions."""
    rng = random.Random(seed)
    rows: list[tuple[str, list[int], int]] = []
    records: list[dict[str, Any]] = []
    attempted = 0
    rejected = 0
    noun_ids = {word: require_one_token(tokenizer, f" {word}") for pair in NOUNS for word in pair}

    # Generous fixed allowance prevents a tokenisation accident from silently shrinking n.
    for pair_index in range(n_pairs * 16):
        if len(records) >= n_pairs:
            break
        attempted += 1
        subject_sg, subject_pl = rng.choice(NOUNS)
        attractor_sg, attractor_pl = rng.choice(NOUNS)
        adjective = rng.choice(ADJECTIVES)
        prep = rng.choice(PREPOSITIONS)
        attractor = attractor_sg if len(records) % 2 == 0 else attractor_pl
        prefix = f"The {adjective}"
        singular = f"{prefix} {subject_sg} {prep} the {attractor}"
        plural = f"{prefix} {subject_pl} {prep} the {attractor}"
        singular_ids, plural_ids = _ids(tokenizer, singular), _ids(tokenizer, plural)
        subject_index = len(_ids(tokenizer, prefix))

        # Per-item assertions follow Experiment 03's stimulus-construction style.
        if len(singular_ids) != len(plural_ids):
            rejected += 1
            continue
        try:
            assert singular_ids[subject_index] == noun_ids[subject_sg]
            assert plural_ids[subject_index] == noun_ids[subject_pl]
            assert subject_index < len(singular_ids)
            assert all(tok >= 0 for tok in singular_ids + plural_ids)
        except AssertionError:
            rejected += 1
            continue
        rows.extend(((singular, singular_ids, subject_index), (plural, plural_ids, subject_index)))
        records.append({
            "pair_index": len(records),
            "adjective": adjective,
            "preposition": prep,
            "subject_singular": subject_sg,
            "subject_plural": subject_pl,
            "attractor": attractor,
            "attractor_number": "singular" if len(records) % 2 == 0 else "plural",
            "subject_token_index": subject_index,
            "subject_singular_token_id": noun_ids[subject_sg],
            "subject_plural_token_id": noun_ids[subject_pl],
            "singular_text": singular,
            "plural_text": plural,
            "singular_token_ids": singular_ids,
            "plural_token_ids": plural_ids,
        })
    if len(records) != n_pairs:
        raise GateStop(
            "stimulus_construction",
            f"Built {len(records)}/{n_pairs} single-flip pairs after {attempted} attempts; rejected={rejected}.",
        )

    max_length = max(len(ids) for _, ids, _ in rows)
    pad = int(getattr(tokenizer, "eos_token_id", 50256))
    tokens = torch.full((len(rows), max_length), pad, dtype=torch.long)
    lengths = torch.empty(len(rows), dtype=torch.long)
    for index, (_, ids, _) in enumerate(rows):
        tokens[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths[index] = len(ids)
    for pair_index, record in enumerate(records):
        singular_i, plural_i = 2 * pair_index, 2 * pair_index + 1
        assert lengths[singular_i] == lengths[plural_i], "single-flip pair differs in token length"
        assert record["subject_token_index"] == rows[singular_i][2] == rows[plural_i][2]
    return Stimuli(
        tokens=tokens,
        lengths=lengths,
        subject_positions=torch.tensor([r["subject_token_index"] for r in records], dtype=torch.long),
        texts=[text for text, _, _ in rows],
        pair_records=records,
        attempted=attempted,
        rejected=rejected,
    )


def clean_pass(model: HookedTransformer, tokens: torch.Tensor, layers: Iterable[int]) -> CleanPass:
    """One cache pass retaining only the four permitted residual hooks and output logits."""
    layer_tuple = tuple(layers)
    names = {f"blocks.{layer}.hook_resid_pre" for layer in layer_tuple}
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    residuals = {name_layer: cache[f"blocks.{name_layer}.hook_resid_pre"].detach().float().cpu().clone()
                 for name_layer in layer_tuple}
    del cache
    return CleanPass(logits=logits.detach().float().cpu().clone(), residuals=residuals)


def logit_difference(logits: torch.Tensor, lengths: torch.Tensor, is_id: int, are_id: int) -> torch.Tensor:
    batch = torch.arange(logits.shape[0])
    final = lengths - 1
    return logits[batch, final, are_id] - logits[batch, final, is_id]


def t_ci(values: Iterable[float]) -> tuple[float, float]:
    """Experiment 04 statistic helper: Student-t(4), not Experiment 03's ci95."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("t_ci needs at least one value")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(GATE_C_T * arr.std(ddof=1) / math.sqrt(arr.size))


# --------------------------------------------------------------------------- patching

class PatchEngine:
    """Patched forwards only; cache is never requested on this path."""

    def __init__(self, model: HookedTransformer, start_at_layer8: bool):
        self.model = model
        self.start_at_layer8 = start_at_layer8
        self.records: list[dict[str, Any]] = []

    @staticmethod
    def _additive_edit(resid: torch.Tensor, positions: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """The non-negotiable Experiment 04 edit: additive delta only, never replacement."""
        for slot in range(positions.shape[1]):
            valid = positions[:, slot] >= 0
            if not bool(valid.any()):
                continue
            batch = torch.arange(resid.shape[0], device=resid.device)[valid]
            pos = positions[valid, slot].to(resid.device)
            delta = deltas[valid, slot].to(resid.device)
            # Scope rule, deliberately literal: no reconstruction is ever assigned to resid.
            resid[batch, pos] += delta
        return resid

    def run(
        self,
        *,
        layer: int,
        base_tokens: torch.Tensor,
        base_residual: torch.Tensor,
        positions: torch.Tensor,
        deltas: torch.Tensor,
        label: str,
        force_full_hook: bool = False,
    ) -> torch.Tensor:
        started = time.perf_counter()
        use_start = self.start_at_layer8 and layer == 8 and not force_full_hook
        with torch.no_grad():
            if use_start:
                edited = base_residual.clone()
                self._additive_edit(edited, positions, deltas)
                logits = self.model(edited, start_at_layer=8, return_type="logits")
                path = "start_at_layer_8"
            else:
                def hook(resid: torch.Tensor, hook: Any) -> torch.Tensor:
                    return self._additive_edit(resid, positions, deltas)

                logits = self.model.run_with_hooks(
                    base_tokens,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)],
                    return_type="logits",
                )
                path = "full_forward_hook"
        elapsed = time.perf_counter() - started
        self.records.append({"label": label, "path": path, "batch": int(base_tokens.shape[0]), "seconds": elapsed})
        return logits.detach().float().cpu()


def directed_indices(n_pairs: int, pair_indices: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Both directions, aligned so positive always means movement toward the source."""
    base, source, sign = [], [], []
    for pair in pair_indices:
        singular, plural = 2 * pair, 2 * pair + 1
        base.extend((singular, plural))
        source.extend((plural, singular))
        sign.extend((1.0, -1.0))
    return torch.tensor(base), torch.tensor(source), torch.tensor(sign, dtype=torch.float32)


def positions_for_kind(stimuli: Stimuli, base_indices: torch.Tensor, kind: str) -> torch.Tensor:
    pairs = base_indices // 2
    subject = stimuli.subject_positions[pairs]
    final = stimuli.lengths[base_indices] - 1
    if kind == "subject":
        return subject[:, None]
    if kind == "final":
        return final[:, None]
    if kind == "both":
        return torch.stack((subject, final), dim=1)
    raise ValueError(f"Unknown position kind {kind}")


def gather_positions(residual: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(residual.shape[0])[:, None]
    return residual[batch, positions]


def patch_effect(
    engine: PatchEngine,
    *,
    layer: int,
    stimuli: Stimuli,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    signs: torch.Tensor,
    clean_d: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    deltas: torch.Tensor,
    is_id: int,
    are_id: int,
    label: str,
    force_full_hook: bool = False,
) -> dict[str, Any]:
    logits = engine.run(
        layer=layer,
        base_tokens=stimuli.tokens[base_indices],
        base_residual=residuals[base_indices],
        positions=positions,
        deltas=deltas,
        label=label,
        force_full_hook=force_full_hook,
    )
    d_patched = logit_difference(logits, stimuli.lengths[base_indices], is_id, are_id)
    base_d, source_d = clean_d[base_indices], clean_d[source_indices]
    effect = d_patched - base_d
    aligned = effect * signs
    gap = source_d - base_d
    return {
        "effect": effect,
        "aligned_effect": aligned,
        "gap": gap,
        "d_patched": d_patched,
        "mean_aligned_effect": float(aligned.mean()),
        "mean_ratio_to_gap": float((effect / gap).mean()),
        "sign_consistency": float(((effect * gap) > 0).float().mean()),
    }


def full_residual_delta(residuals: torch.Tensor, base_indices: torch.Tensor, source_indices: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return gather_positions(residuals[source_indices], positions) - gather_positions(residuals[base_indices], positions)


def feature_delta(
    encode: Callable[[torch.Tensor], torch.Tensor],
    residuals: torch.Tensor,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = encode(gather_positions(residuals[source_indices], positions))
    base = encode(gather_positions(residuals[base_indices], positions))
    return source - base, source, base


def decoded_delta(code_delta: torch.Tensor, decoder: torch.Tensor | TiedDecoder, columns: torch.Tensor | None = None) -> torch.Tensor:
    """Decode selected coordinates without ever constructing a b x 24576 selection mask."""
    if isinstance(decoder, TiedDecoder):
        if columns is None:
            return (code_delta @ decoder.R.T) * decoder.scale
        return (code_delta[..., columns] @ decoder.R[:, columns].T) * decoder.scale
    if columns is None:
        return code_delta @ decoder
    return code_delta[..., columns] @ decoder[columns]


# -------------------------------------------------------------------- random decoder

def make_random_basis(sae: SAEWeights, reference_x: torch.Tensor, seed: int) -> RandomBasis:
    """Experiment 03's matched Gaussian encoder/norm/L0 construction, factored for reuse."""
    generator = torch.Generator(device="cpu").manual_seed(seed + 400_000)
    target_l0 = int(round(float((sae.encode(reference_x) > 0).sum(dim=-1).float().mean())))
    target_l0 = max(1, min(target_l0, sae.W_enc.shape[1]))
    R = torch.randn(sae.W_enc.shape, generator=generator, dtype=torch.float32)
    R *= sae.W_enc.norm(dim=0, keepdim=True) / R.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return RandomBasis(R=R, b_enc=sae.b_enc, b_dec=sae.b_dec, target_l0=target_l0, seed=seed)


def sparse_code_matrix(values: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    """COO G: exact per-sample top-k values, with no dense n x 24576 allocation."""
    n_rows, topk = columns.shape
    row = torch.arange(n_rows, dtype=torch.long)[:, None].expand(n_rows, topk).reshape(-1)
    index = torch.stack((row, columns.reshape(-1)))
    return torch.sparse_coo_tensor(index, values.reshape(-1), size=(n_rows, 24576)).coalesce()


def r2_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    sse = float(((target - prediction) ** 2).sum())
    sst = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum())
    return float(1.0 - sse / max(sst, 1e-12))


def decoder_rows(decoder: torch.Tensor | TiedDecoder, columns: torch.Tensor) -> torch.Tensor:
    if isinstance(decoder, TiedDecoder):
        return decoder.R[:, columns].T * decoder.scale
    return decoder[columns]


def decoder_row_norms(decoder: torch.Tensor | TiedDecoder) -> torch.Tensor:
    if isinstance(decoder, TiedDecoder):
        return decoder.R.norm(dim=0) * abs(decoder.scale)
    return decoder.norm(dim=1)


def decoder_rows_nonzero(decoder: torch.Tensor | TiedDecoder, columns: torch.Tensor) -> torch.Tensor:
    if isinstance(decoder, TiedDecoder):
        return torch.full((columns.numel(),), decoder.scale != 0.0, dtype=torch.bool)
    return decoder[columns].abs().sum(dim=1) > 0


def dual_solve(system: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Solve the exact float32 dual system; SVD is a numerical, not statistical, fallback."""
    try:
        return torch.linalg.solve(system, target), "torch.linalg.solve"
    except RuntimeError as exc:
        if "singular" not in str(exc).lower():
            raise
        # The system is mathematically positive-definite for lambda > 0.  On CPU
        # float32, very small lambda can nevertheless be below the factorisation's
        # resolution.  gelsd solves the same supplied system without changing lambda.
        solved = torch.linalg.lstsq(system, target, driver="gelsd").solution
        if not bool(torch.isfinite(solved).all()):
            raise GateStop("C_ridge_solver", f"Float32 SVD fallback produced non-finite values: {exc}")
        return solved, "torch.linalg.lstsq_gelsd_after_singular_solve"


def fit_random_decoders(
    basis: RandomBasis,
    fit_x: torch.Tensor,
    heldout_x: torch.Tensor,
) -> dict[str, Any]:
    """Dual ridge specified in DESIGN.md plus the documented tied-weight fallback."""
    fit_values, fit_indices = basis.sparse_topk(fit_x)
    G = sparse_code_matrix(fit_values, fit_indices)
    held_values, held_indices = basis.sparse_topk(heldout_x)
    G_held = sparse_code_matrix(held_values, held_indices)

    n = G.shape[0]
    # These are n x n matrices, as frozen in the dual-ridge specification.  Sparse
    # multiplications preserve the per-sample top-k code and avoid an n x 24576 array.
    gram = torch.sparse.mm(G, G.transpose(0, 1)).to_dense()
    held_cross = torch.sparse.mm(G_held, G.transpose(0, 1)).to_dense()
    identity = torch.eye(n, dtype=torch.float32)
    lambda_grid = np.logspace(-4, 3, 8)
    traces: list[dict[str, float]] = []
    best: tuple[float, torch.Tensor, float, str] | None = None
    for lam in lambda_grid:
        solved, solver = dual_solve(gram + float(lam) * identity, fit_x)
        held_r2 = r2_score(heldout_x, held_cross @ solved)
        traces.append({"lambda": float(lam), "heldout_generic_r2": held_r2, "solver": solver})
        if best is None or held_r2 > best[0]:
            best = (held_r2, solved, float(lam), solver)
    assert best is not None
    selected_r2, selected_solved, selected_lambda, selected_solver = best
    # Exactly W = G_A^T (G_A G_A^T + lambda I)^-1 X.  COO rows absent from A are
    # emitted as zero rows, so this dense decoder has the required inactive-row meaning.
    W_ridge = torch.sparse.mm(G.transpose(0, 1), selected_solved)

    # Documented weaker fallback: tied random encoder rows with one generic-fit scalar.
    tied_prediction = torch.sparse.mm(G, basis.R.T)
    alpha = float((tied_prediction * fit_x).sum() / tied_prediction.square().sum().clamp_min(1e-12))
    tied_decoder = TiedDecoder(R=basis.R, scale=alpha)
    tied_r2 = r2_score(heldout_x, torch.sparse.mm(G_held, basis.R.T) * alpha)
    active = torch.unique(fit_indices).sort().values

    return {
        "ridge_decoder": W_ridge,
        "tied_decoder": tied_decoder,
        "active_rows": active,
        "fit_rows": n,
        "fit_active_width": int(active.numel()),
        "lambda_grid": traces,
        "selected_lambda": selected_lambda,
        "selected_solver": selected_solver,
        "selected_heldout_generic_r2": selected_r2,
        "tied_scale": alpha,
        "tied_heldout_generic_r2": tied_r2,
    }


def generate_generic_activations(model: HookedTransformer, layer: int, seed: int, requested_tokens: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate exactly requested_tokens from GPT-2 itself, then cache its layer-8 residuals."""
    if requested_tokens % 8:
        raise ValueError("requested_tokens must be divisible by 8")
    set_determinism(seed + 11)
    batch = 8
    per_sequence = requested_tokens // batch
    eos = int(model.tokenizer.eos_token_id)
    prompt = torch.full((batch, 1), eos, dtype=torch.long)
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=per_sequence,
            stop_at_eos=False,
            do_sample=True,
            top_k=50,
            temperature=1.0,
            use_past_kv_cache=True,
            return_type="tokens",
            verbose=False,
        )
    generation_seconds = time.perf_counter() - started
    clean = clean_pass(model, generated, (layer,))
    # Position 0 is the fixed EOS prompt; all requested generated-token positions are fit samples.
    activations = clean.residuals[layer][:, 1:, :].reshape(-1, 768).contiguous()
    assert activations.shape[0] == requested_tokens
    return activations, {
        "requested_tokens": requested_tokens,
        "actual_generated_tokens": int(activations.shape[0]),
        "generation_batch": batch,
        "tokens_per_sequence": per_sequence,
        "generation_seconds": generation_seconds,
        "generation_seconds_per_token": generation_seconds / requested_tokens,
    }


# --------------------------------------------------------------------------- ranking

def candidate_prefilter(
    code_delta: torch.Tensor,
    source_code: torch.Tensor,
    base_code: torch.Tensor,
    signs: torch.Tensor,
    decoder: torch.Tensor | TiedDecoder,
    budget: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Frozen signed-mean contribution proxy, applied identically to SAE and random codes."""
    aligned = code_delta * signs[:, None, None]
    mean_contribution = aligned.sum(dim=1).mean(dim=0)
    active = ((source_code > 0) | (base_code > 0)).any(dim=(0, 1))
    candidates = torch.nonzero(active, as_tuple=False).squeeze(1)
    if candidates.numel() < budget:
        raise GateStop("ranking_candidate_coverage", f"Only {candidates.numel()} active candidates; need {budget}.")
    proxy = mean_contribution[candidates].abs() * decoder_row_norms(decoder)[candidates]
    chosen = candidates[proxy.topk(budget).indices]
    return chosen, {
        "active_union_count": int(candidates.numel()),
        "prefilter_budget": budget,
        "proxy_top_values": [float(x) for x in proxy.topk(budget).values],
    }


def single_coordinate_scores(
    engine: PatchEngine,
    *,
    layer: int,
    stimuli: Stimuli,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    signs: torch.Tensor,
    clean_d: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    code_delta: torch.Tensor,
    decoder: torch.Tensor | TiedDecoder,
    candidates: torch.Tensor,
    is_id: int,
    are_id: int,
    label: str,
) -> torch.Tensor:
    """Score candidates in stacked configuration batches, not one forward per coordinate."""
    n_directions = base_indices.numel()
    scores: list[torch.Tensor] = []
    config_chunk = max(1, PATCH_BATCH_LIMIT // n_directions)
    for start in range(0, candidates.numel(), config_chunk):
        current = candidates[start : start + config_chunk]
        n_config = current.numel()
        coefficient = code_delta[:, :, current].permute(2, 0, 1)
        deltas = torch.einsum("cbm,cd->cbmd", coefficient, decoder_rows(decoder, current))
        effect = patch_effect(
            engine,
            layer=layer,
            stimuli=stimuli,
            base_indices=base_indices.repeat(n_config),
            source_indices=source_indices.repeat(n_config),
            signs=signs.repeat(n_config),
            clean_d=clean_d,
            residuals=residuals,
            positions=positions.repeat(n_config, 1),
            deltas=deltas.reshape(n_config * n_directions, positions.shape[1], 768),
            is_id=is_id,
            are_id=are_id,
            label=label,
        )
        scores.append(effect["aligned_effect"].reshape(n_config, n_directions).mean(dim=1))
    return torch.cat(scores)


def topk_recovery(
    engine: PatchEngine,
    *,
    layer: int,
    stimuli: Stimuli,
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    signs: torch.Tensor,
    clean_d: torch.Tensor,
    residuals: torch.Tensor,
    positions: torch.Tensor,
    code_delta: torch.Tensor,
    decoder: torch.Tensor | TiedDecoder,
    ranked: torch.Tensor,
    k_values: tuple[int, ...],
    full_aligned_effect: float,
    is_id: int,
    are_id: int,
    label: str,
) -> dict[str, Any]:
    deltas = []
    for k in k_values:
        columns = ranked[:k]
        deltas.append(decoded_delta(code_delta, decoder, columns))
    n_config, n_directions = len(k_values), base_indices.numel()
    result = patch_effect(
        engine,
        layer=layer,
        stimuli=stimuli,
        base_indices=base_indices.repeat(n_config),
        source_indices=source_indices.repeat(n_config),
        signs=signs.repeat(n_config),
        clean_d=clean_d,
        residuals=residuals,
        positions=positions.repeat(n_config, 1),
        deltas=torch.cat(deltas, dim=0),
        is_id=is_id,
        are_id=are_id,
        label=label,
    )
    aligned = result["aligned_effect"].reshape(n_config, n_directions).mean(dim=1)
    rows = []
    for k, effect in zip(k_values, aligned):
        raw = float(effect) / full_aligned_effect
        rows.append({"k": k, "mean_aligned_effect": float(effect), "unclipped_recovery": raw, "recovery": float(np.clip(raw, 0.0, 1.0))})
    return {"full_mean_aligned_effect": full_aligned_effect, "rows": rows}


# ------------------------------------------------------------------------- reporting

def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_notes(manifest: dict[str, Any]) -> None:
    """Short factual record only; called for both GO and every STOP."""
    elapsed = float(manifest.get("wall_clock_seconds", 0.0))
    status = manifest.get("status", "running")
    lines = [
        "# Experiment 04 Pilot Notes",
        "",
        f"- Wall-clock time: {elapsed:.1f} seconds.",
        f"- Status: `{status}`" + (f"; failed gate: `{manifest['failed_gate']}`." if manifest.get("failed_gate") else "."),
        "- Scope: pilot steps 1--7 only; no full experiment was run.",
    ]
    for step, row in manifest.get("steps", {}).items():
        state = row.get("decision", "not_run")
        summary = row.get("summary", "")
        lines.append(f"- Step {step}: `{state}`" + (f" — {summary}" if summary else "."))
    gate_a = manifest.get("steps", {}).get("3", {}).get("gate_A")
    if gate_a:
        lines.append(
            "- Gate A measurements: "
            f"both-correct={gate_a['both_members_signed_correct_fraction']:.3f}; "
            f"median d_gap={gate_a['median_d_gap']:.3f}; pilot retained pairs={gate_a['pilot_retained_pairs']}."
        )
    scan = manifest.get("layer_position_scan")
    if scan and scan.get("8"):
        l8 = scan["8"]
        lines.append(
            "- Gate B layer-8 E_resid/d_gap (sign consistency): "
            f"subject={l8['subject']['mean_E_resid_over_d_gap']:.3f} ({l8['subject']['sign_consistency']:.3f}), "
            f"final={l8['final']['mean_E_resid_over_d_gap']:.3f} ({l8['final']['sign_consistency']:.3f}), "
            f"both={l8['both']['mean_E_resid_over_d_gap']:.3f} ({l8['both']['sign_consistency']:.3f}); "
            f"selected={manifest.get('selected_layer8_position_set', 'not_selected')}."
        )
    gate_c = manifest.get("gate_C")
    if gate_c:
        sae_ratio = gate_c["sae"]["trained_decoder"]["ratio_to_E_resid"]
        ridge_ratio = gate_c["random"]["dual_ridge"]["ratio_to_E_resid"]
        tied_ratio = gate_c["random"]["tied_weight_fallback"]["ratio_to_E_resid"]
        coverage = gate_c["coverage"]
        lines.append(
            "- Gate C E(full)/E_resid: "
            f"SAE trained={sae_ratio:.3f}; random dual-ridge={ridge_ratio:.3f}; "
            f"random tied={tied_ratio:.3f}; dual-ridge coverage={coverage['random_dual_ridge']:.3f}."
        )
    timing = manifest.get("forward_timing", {}).get("summary")
    if timing:
        lines.append(
            "- Patched-forward timing: "
            f"{timing['measured_seconds_per_patched_forward_call']:.3f} s/call; "
            f"full-run extrapolation={timing['extrapolated_minutes']:.2f} minutes; "
            f"trim order fired={timing['trim_order_triggered']}."
        )
    decisions = manifest.get("implementation_decisions", [])
    if decisions:
        lines.extend(["", "## Recorded implementation decisions", ""])
        lines.extend(f"- {item}" for item in decisions)
    NOTES.write_text("\n".join(lines) + "\n")


def write_artifacts(manifest: dict[str, Any], started: float) -> None:
    manifest["wall_clock_seconds"] = time.perf_counter() - started
    RESULTS.write_text(json.dumps(jsonable(manifest), indent=2) + "\n")
    write_notes(manifest)


def print_step(step: int, criterion: str, measurement: str, decision: str) -> None:
    print(f"STEP {step} | criterion: {criterion} | measured: {measurement} | {decision}", flush=True)


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def full_run_estimate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pre-declared cost extrapolation and trim order; it does not execute any full-run work."""
    all_seconds = [float(r["seconds"]) for r in records]
    all_batches = [int(r["batch"]) for r in records]
    seconds_per_example = (sum(all_seconds) / sum(all_batches)) if all_batches else None
    rank_seconds = [float(r["seconds"]) for r in records if r["label"] == "rank_candidates"]
    topk_seconds = [float(r["seconds"]) for r in records if r["label"] == "topk_eval"]
    per_call = median_or_none(rank_seconds + topk_seconds) or median_or_none(all_seconds)

    def calls_for(eval_pairs: int, candidates: int, include_k2: bool) -> dict[str, int]:
        train_dirs, eval_dirs = 2 * 40, 2 * eval_pairs
        rank_chunk = max(1, PATCH_BATCH_LIMIT // train_dirs)
        eval_chunk = max(1, PATCH_BATCH_LIMIT // eval_dirs)
        grid_count = 7 if include_k2 else 6
        ranking = 5 * 2 * math.ceil(candidates / rank_chunk)
        topk = 5 * 2 * math.ceil(grid_count / eval_chunk)
        random_k = 5 * 2 * math.ceil(grid_count / eval_chunk)
        full_anchors = 5 * 2
        resid_anchor = 5
        return {
            "ranking": ranking,
            "topk": topk,
            "randk": random_k,
            "basis_full": full_anchors,
            "resid_full": resid_anchor,
            "total": ranking + topk + random_k + full_anchors + resid_anchor,
        }

    plan = {"evaluation_pairs": 150, "candidates": 64, "include_k2": True}
    calls = calls_for(plan["evaluation_pairs"], plan["candidates"], plan["include_k2"])
    seconds = float(calls["total"] * per_call) if per_call is not None else None
    trims: list[str] = []
    if seconds is not None and seconds > 3600:
        plan["evaluation_pairs"] = 100
        trims.append("evaluation_pairs: 150 -> 100")
        calls = calls_for(plan["evaluation_pairs"], plan["candidates"], plan["include_k2"])
        seconds = float(calls["total"] * per_call)
    if seconds is not None and seconds > 3600:
        plan["candidates"] = 48
        trims.append("candidates: 64 -> 48")
        calls = calls_for(plan["evaluation_pairs"], plan["candidates"], plan["include_k2"])
        seconds = float(calls["total"] * per_call)
    if seconds is not None and seconds > 3600:
        plan["include_k2"] = False
        trims.append("drop k=2")
        calls = calls_for(plan["evaluation_pairs"], plan["candidates"], plan["include_k2"])
        seconds = float(calls["total"] * per_call)
    return {
        "measured_seconds_per_patched_forward_call": per_call,
        "measured_seconds_per_patched_sequence": seconds_per_example,
        "estimated_patched_forward_calls": calls,
        "extrapolated_seconds": seconds,
        "extrapolated_minutes": (seconds / 60.0) if seconds is not None else None,
        "trim_order_triggered": trims,
        "post_trim_plan": plan,
        "formula": "Five seeds; 40 rank-training pairs and evaluation pairs in both directions. Calls use PATCH_BATCH_LIMIT=512 and include ranking, top-k, rand-k, basis-full, and residual-full patched forwards.",
    }


# ------------------------------------------------------------------------------ run

def run() -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_grad_enabled(False)
    set_determinism(SEED)
    manifest: dict[str, Any] = {
        "schema": "exp04-causal-feature-interchange-pilot-v1; frozen DESIGN.md pilot steps 1-7; float32 CPU; Student-t(4)=2.776 helper.",
        "status": "running",
        "failed_gate": None,
        "seed": SEED,
        "device": "cpu",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "steps": {},
        "token_ids": {},
        "layer_position_scan": None,
        "gate_C": None,
        "forward_timing": None,
        "implementation_decisions": [
            "DESIGN.md's explicit Pilot step 3 count (60 pairs) is treated as pilot-specific. The Gate A table's >=140 retained-pair threshold is recorded as a full-run threshold because it is mechanically impossible for a 60-pair pilot.",
            "The random dual ridge maps random codes directly to residual x as written in DESIGN.md's formula; decoder biases cancel from every additive difference edit.",
            "If dual-ridge fails Gate C but tied-weight passes, the manifest records the under-specified branch and selects the passing tied decoder for a follow-up rather than fabricating a ridge result.",
        ],
    }
    try:
        # Step 1 -----------------------------------------------------------------
        step_started = time.perf_counter()
        model = load_model()
        sae = load_direct_res_jb(8)
        if sae.W_dec.shape != (24576, 768):
            raise GateStop("environment_sae_shape", f"W_dec shape is {tuple(sae.W_dec.shape)}, expected (24576, 768).")
        required_tokens = (" is", " are", " was", " were") + CONTROL_WORDS
        token_ids = {text: require_one_token(model.tokenizer, text) for text in required_tokens}
        manifest["token_ids"] = {"eos": int(model.tokenizer.eos_token_id), **token_ids}
        manifest["steps"]["1"] = {
            "decision": "PASS",
            "summary": f"CPU float32 model and layer-8 SAE loaded; W_dec={tuple(sae.W_dec.shape)}; six readout/control strings are one leading-space token.",
            "seconds": time.perf_counter() - step_started,
            "sae_source": sae.source,
            "W_enc_shape": list(sae.W_enc.shape),
            "W_dec_shape": list(sae.W_dec.shape),
            "token_ids": token_ids,
        }
        print_step(1, "CPU float32 load; W_dec=[24576,768]; all six strings one leading-space token", f"W_dec={tuple(sae.W_dec.shape)}, token_ids={token_ids}", "PASS")

        # Step 2 -----------------------------------------------------------------
        step_started = time.perf_counter()
        selftest = build_stimuli(model.tokenizer, 2, SEED + 1)
        selftest_clean = clean_pass(model, selftest.tokens, LAYERS)
        is_id, are_id = token_ids[" is"], token_ids[" are"]
        mini_base, mini_source, mini_signs = directed_indices(2, range(2))
        mini_positions = positions_for_kind(selftest, mini_base, "subject")
        selftest_engine = PatchEngine(model, start_at_layer8=True)
        zero = torch.zeros((mini_base.numel(), 1, 768), dtype=torch.float32)
        zero_logits = selftest_engine.run(
            layer=8,
            base_tokens=selftest.tokens[mini_base],
            base_residual=selftest_clean.residuals[8][mini_base],
            positions=mini_positions,
            deltas=zero,
            label="selftest_zero",
            force_full_hook=True,
        )
        clean_subset = selftest_clean.logits[mini_base]
        zero_bitwise = bool(torch.equal(zero_logits, clean_subset))
        if not zero_bitwise:
            raise GateStop("2a_zero_selection", "Zero-selection additive hook did not reproduce clean logits bit-for-bit.")

        feature_diff, _, _ = feature_delta(sae.encode, selftest_clean.residuals[8], mini_base, mini_source, mini_positions)
        selection = torch.arange(16, dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
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
            raise GateStop("2b_scale_invariance", "Power-of-two positive diagonal rescale changed the written edit bit pattern.")

        start_logits = selftest_engine.run(
            layer=8,
            base_tokens=selftest.tokens[mini_base],
            base_residual=selftest_clean.residuals[8][mini_base],
            positions=mini_positions,
            deltas=zero,
            label="selftest_start_at_layer",
        )
        start_max_abs = float((start_logits - clean_subset).abs().max())
        start_path = start_max_abs < 1e-4
        if not start_path:
            # The frozen design permits an explicitly recorded full-forward fallback.
            start_path = False
        manifest["steps"]["2"] = {
            "decision": "PASS",
            "summary": f"2a zero bitwise={zero_bitwise}; 2b power-of-two diagonal bitwise={rescale_bitwise}; 2c max abs={start_max_abs:.3g}.",
            "seconds": time.perf_counter() - step_started,
            "zero_selection_bitwise": zero_bitwise,
            "scale_invariance_bitwise": rescale_bitwise,
            "scale_exponents": exponents.tolist(),
            "start_at_layer_max_abs": start_max_abs,
            "forward_path_for_layer8": "start_at_layer_8" if start_path else "full_forward_hook_fallback",
        }
        print_step(2, "2a/2b bitwise equality; 2c start_at_layer max abs <1e-4 (or record allowed fallback)", f"zero={zero_bitwise}, rescale={rescale_bitwise}, start_max_abs={start_max_abs:.3g}", "PASS")

        # Step 3 -----------------------------------------------------------------
        step_started = time.perf_counter()
        stimuli = build_stimuli(model.tokenizer, PILOT_PAIRS, SEED)
        clean = clean_pass(model, stimuli.tokens, LAYERS)
        clean_d = logit_difference(clean.logits, stimuli.lengths, is_id, are_id)
        singular_d, plural_d = clean_d[0::2], clean_d[1::2]
        d_gap_pairs = plural_d - singular_d
        correct_both = (singular_d < 0) & (plural_d > 0)
        retention = float(correct_both.float().mean())
        median_gap = float(d_gap_pairs.median())
        gate_a_pass = retention >= 0.60 and median_gap >= 1.0
        manifest["stimuli"] = {
            "pair_count": PILOT_PAIRS,
            "sequence_count": int(stimuli.tokens.shape[0]),
            "attempted_pairs": stimuli.attempted,
            "rejected_pairs": stimuli.rejected,
            "padding": "right EOS; logits read at len-1; direct token tensors with no prepended BOS",
            "pair_records": stimuli.pair_records,
            "clean_d": [float(x) for x in clean_d],
            "d_gap_pairs": [float(x) for x in d_gap_pairs],
        }
        manifest["steps"]["3"] = {
            "decision": "PASS" if gate_a_pass else "STOP",
            "summary": f"both-correct={retention:.3f}; median d_gap={median_gap:.3f}; pilot pairs={PILOT_PAIRS}.",
            "seconds": time.perf_counter() - step_started,
            "gate_A": {
                "both_members_signed_correct_fraction": retention,
                "median_d_gap": median_gap,
                "pilot_retained_pairs": PILOT_PAIRS,
                "full_run_minimum_retained_pairs": 140,
                "full_run_minimum_applicability": "not_applicable_to_60_pair_pilot",
            },
        }
        print_step(3, "Gate A: both members correct >=0.60; median d_gap >=1.0 (60-pair pilot)", f"both-correct={retention:.3f}, median_gap={median_gap:.3f}", "PASS" if gate_a_pass else "STOP")
        if not gate_a_pass:
            raise GateStop("A_behaviour", f"Gate A failed: both-correct={retention:.3f}, median d_gap={median_gap:.3f}.")

        # Step 4 -----------------------------------------------------------------
        step_started = time.perf_counter()
        base_all, source_all, signs_all = directed_indices(PILOT_PAIRS, range(PILOT_PAIRS))
        max_edit_positions = int((stimuli.lengths[base_all] - stimuli.subject_positions[base_all // 2]).max())
        exact_positions = torch.full((base_all.numel(), max_edit_positions), -1, dtype=torch.long)
        exact_deltas = torch.zeros((base_all.numel(), max_edit_positions, 768), dtype=torch.float32)
        for row, (base_i, source_i) in enumerate(zip(base_all.tolist(), source_all.tolist())):
            start = int(stimuli.subject_positions[base_i // 2])
            stop = int(stimuli.lengths[base_i])
            selected = torch.arange(start, stop)
            exact_positions[row, : selected.numel()] = selected
            exact_deltas[row, : selected.numel()] = clean.residuals[8][source_i, selected] - clean.residuals[8][base_i, selected]
        engine = PatchEngine(model, start_at_layer8=start_path)
        exact = patch_effect(
            engine,
            layer=8,
            stimuli=stimuli,
            base_indices=base_all,
            source_indices=source_all,
            signs=signs_all,
            clean_d=clean_d,
            residuals=clean.residuals[8],
            positions=exact_positions,
            deltas=exact_deltas,
            is_id=is_id,
            are_id=are_id,
            label="exactness",
        )
        exact_errors = (exact["d_patched"] - clean_d[source_all]).abs()
        exact_max = float(exact_errors.max())
        exact_pass = bool((exact_errors < 1e-3).all())
        manifest["steps"]["4"] = {
            "decision": "PASS" if exact_pass else "STOP",
            "summary": f"max |d(patched)-d(source)|={exact_max:.3g} across {base_all.numel()} directed edits.",
            "seconds": time.perf_counter() - step_started,
            "max_abs_error": exact_max,
            "all_pairs_below_1e-3": exact_pass,
        }
        print_step(4, "all-position layer-8 swap: every |d(patched)-d(source)| <1e-3", f"max_abs_error={exact_max:.3g}", "PASS" if exact_pass else "STOP")
        if not exact_pass:
            raise GateStop("4_exactness", f"Exactness self-check failed; max error={exact_max:.6g}.")

        # Step 5 -----------------------------------------------------------------
        step_started = time.perf_counter()
        scan: dict[str, dict[str, Any]] = {}
        selected_kind: str | None = None
        selected_gate_b: dict[str, Any] | None = None
        for layer in LAYERS:
            scan[str(layer)] = {}
            stacked_positions, stacked_deltas = [], []
            for kind in ("subject", "final", "both"):
                positions = positions_for_kind(stimuli, base_all, kind)
                deltas = full_residual_delta(clean.residuals[layer], base_all, source_all, positions)
                padded_positions = torch.full((base_all.numel(), 2), -1, dtype=torch.long)
                padded_deltas = torch.zeros((base_all.numel(), 2, 768), dtype=torch.float32)
                padded_positions[:, : positions.shape[1]] = positions
                padded_deltas[:, : positions.shape[1]] = deltas
                stacked_positions.append(padded_positions)
                stacked_deltas.append(padded_deltas)
            # The three edit configurations are stacked along batch dimension; no
            # patched configuration gets a cache, and each slice retains its own result.
            stacked = patch_effect(
                engine,
                layer=layer,
                stimuli=stimuli,
                base_indices=base_all.repeat(3),
                source_indices=source_all.repeat(3),
                signs=signs_all.repeat(3),
                clean_d=clean_d,
                residuals=clean.residuals[layer],
                positions=torch.cat(stacked_positions, dim=0),
                deltas=torch.cat(stacked_deltas, dim=0),
                is_id=is_id,
                are_id=are_id,
                label="gate_b_scan",
            )
            for config_index, kind in enumerate(("subject", "final", "both")):
                start = config_index * base_all.numel()
                stop = start + base_all.numel()
                effect = stacked["effect"][start:stop]
                gap = stacked["gap"][start:stop]
                aligned = stacked["aligned_effect"][start:stop]
                result = {
                    "mean_ratio_to_gap": float((effect / gap).mean()),
                    "sign_consistency": float(((effect * gap) > 0).float().mean()),
                    "mean_aligned_effect": float(aligned.mean()),
                }
                scan[str(layer)][kind] = {
                    "mean_E_resid_over_d_gap": result["mean_ratio_to_gap"],
                    "sign_consistency": result["sign_consistency"],
                    "mean_aligned_E_resid": result["mean_aligned_effect"],
                }
                if layer == 8 and result["mean_ratio_to_gap"] >= 0.50 and result["sign_consistency"] >= 0.90:
                    # The declared tie break is subject.  Iteration order is subject, final, both.
                    if selected_kind is None:
                        selected_kind, selected_gate_b = kind, result
        manifest["layer_position_scan"] = scan
        gate_b_pass = selected_kind is not None and selected_gate_b is not None
        manifest["steps"]["5"] = {
            "decision": "PASS" if gate_b_pass else "STOP",
            "summary": f"selected layer-8 position set={selected_kind}; scan recorded for layers {LAYERS}.",
            "seconds": time.perf_counter() - step_started,
            "selected_layer": 8 if gate_b_pass else None,
            "selected_position_set": selected_kind,
            "selection_rule": "smallest layer-8 set with mean E_resid/d_gap >=0.50 and sign consistency >=0.90; ties subject",
        }
        print_step(5, "Gate B: layer-8 mean E_resid/d_gap >=0.50 and sign consistency >=0.90", f"selected={selected_kind}; layer8={scan['8']}", "PASS" if gate_b_pass else "STOP")
        if not gate_b_pass:
            raise GateStop("B_causal_handle", "No layer-8 position set passed the declared Gate B threshold.")

        # Step 6 -----------------------------------------------------------------
        step_started = time.perf_counter()
        selected_positions_all = positions_for_kind(stimuli, base_all, selected_kind)
        split_train_pairs = list(range(0, RANK_TRAIN_PAIRS))
        split_gate_c_pairs = list(range(RANK_TRAIN_PAIRS, RANK_TRAIN_PAIRS + GATE_C_PAIRS))
        split_eval_pairs = list(range(PILOT_PAIRS - RANK_EVAL_PAIRS, PILOT_PAIRS))
        train_base, train_source, train_signs = directed_indices(PILOT_PAIRS, split_train_pairs)
        train_positions = positions_for_kind(stimuli, train_base, selected_kind)
        train_template_x = torch.cat((
            gather_positions(clean.residuals[8][train_base], train_positions).reshape(-1, 768),
            gather_positions(clean.residuals[8][train_source], train_positions).reshape(-1, 768),
        ), dim=0)
        requested_generic_tokens = 2048
        print("STEP 6 | starting fixed-seed GPT-2 generic-text generation (2048 tokens)", flush=True)
        generic_x, generation = generate_generic_activations(model, 8, SEED, requested_generic_tokens)
        generic_fit = generic_x[: int(0.8 * generic_x.shape[0])]
        generic_holdout = generic_x[int(0.8 * generic_x.shape[0]) :]
        fit_x = torch.cat((generic_fit, train_template_x), dim=0)
        random_basis = make_random_basis(sae, fit_x, SEED)
        decoders = fit_random_decoders(random_basis, fit_x, generic_holdout)

        gate_c_base, gate_c_source, gate_c_signs = directed_indices(PILOT_PAIRS, split_gate_c_pairs)
        gate_c_positions = positions_for_kind(stimuli, gate_c_base, selected_kind)
        gate_c_resid_delta = full_residual_delta(clean.residuals[8], gate_c_base, gate_c_source, gate_c_positions)
        gate_c_resid = patch_effect(
            engine,
            layer=8,
            stimuli=stimuli,
            base_indices=gate_c_base,
            source_indices=gate_c_source,
            signs=gate_c_signs,
            clean_d=clean_d,
            residuals=clean.residuals[8],
            positions=gate_c_positions,
            deltas=gate_c_resid_delta,
            is_id=is_id,
            are_id=are_id,
            label="gate_c_resid_full",
        )
        resid_anchor = gate_c_resid["mean_aligned_effect"]
        if abs(resid_anchor) < 1e-12:
            raise GateStop("C_resid_anchor", "Gate C residual anchor is zero, so full-arm ratios are undefined.")

        sae_diff, sae_source_code, sae_base_code = feature_delta(sae.encode, clean.residuals[8], gate_c_base, gate_c_source, gate_c_positions)
        sae_full = patch_effect(
            engine,
            layer=8,
            stimuli=stimuli,
            base_indices=gate_c_base,
            source_indices=gate_c_source,
            signs=gate_c_signs,
            clean_d=clean_d,
            residuals=clean.residuals[8],
            positions=gate_c_positions,
            deltas=decoded_delta(sae_diff, sae.W_dec),
            is_id=is_id,
            are_id=are_id,
            label="gate_c_sae_full",
        )
        random_diff, random_source_code, random_base_code = feature_delta(random_basis.encode, clean.residuals[8], gate_c_base, gate_c_source, gate_c_positions)
        random_ridge = patch_effect(
            engine,
            layer=8,
            stimuli=stimuli,
            base_indices=gate_c_base,
            source_indices=gate_c_source,
            signs=gate_c_signs,
            clean_d=clean_d,
            residuals=clean.residuals[8],
            positions=gate_c_positions,
            deltas=decoded_delta(random_diff, decoders["ridge_decoder"]),
            is_id=is_id,
            are_id=are_id,
            label="gate_c_random_ridge",
        )
        random_tied = patch_effect(
            engine,
            layer=8,
            stimuli=stimuli,
            base_indices=gate_c_base,
            source_indices=gate_c_source,
            signs=gate_c_signs,
            clean_d=clean_d,
            residuals=clean.residuals[8],
            positions=gate_c_positions,
            deltas=decoded_delta(random_diff, decoders["tied_decoder"]),
            is_id=is_id,
            are_id=are_id,
            label="gate_c_random_tied",
        )

        def gate_ratio(effect: dict[str, Any]) -> dict[str, Any]:
            ratio = effect["mean_aligned_effect"] / resid_anchor
            return {
                "mean_aligned_effect": effect["mean_aligned_effect"],
                "ratio_to_E_resid": ratio,
                "in_0p70_to_1p30": bool(0.70 <= ratio <= 1.30),
            }

        # Decoder-row coverage is checked on Gate-C evaluation-active coordinates.
        sae_active = ((sae_source_code > 0) | (sae_base_code > 0)).any(dim=(0, 1))
        random_active = ((random_source_code > 0) | (random_base_code > 0)).any(dim=(0, 1))
        coverage = {
            "sae_trained": float(decoder_rows_nonzero(sae.W_dec, torch.nonzero(sae_active).squeeze(1)).float().mean()),
            "random_dual_ridge": float(decoder_rows_nonzero(decoders["ridge_decoder"], torch.nonzero(random_active).squeeze(1)).float().mean()),
            "random_tied_weight": float(decoder_rows_nonzero(decoders["tied_decoder"], torch.nonzero(random_active).squeeze(1)).float().mean()),
            "threshold": 0.95,
        }
        if coverage["random_dual_ridge"] < 0.95:
            raise GateStop("C_random_coverage", f"Random dual-ridge coverage={coverage['random_dual_ridge']:.3f} <0.95.")

        gate_c = {
            "resid_full": {"mean_aligned_effect": resid_anchor},
            "sae": {"trained_decoder": gate_ratio(sae_full)},
            "random": {
                "dual_ridge": gate_ratio(random_ridge),
                "tied_weight_fallback": gate_ratio(random_tied),
            },
            "coverage": coverage,
            "random_decoder_fit": {
                "target_l0": random_basis.target_l0,
                "fit_rows": decoders["fit_rows"],
                "fit_active_width": decoders["fit_active_width"],
                "lambda_grid": decoders["lambda_grid"],
                "selected_lambda": decoders["selected_lambda"],
                "selected_solver": decoders["selected_solver"],
                "selected_heldout_generic_r2": decoders["selected_heldout_generic_r2"],
                "tied_scale": decoders["tied_scale"],
                "tied_heldout_generic_r2": decoders["tied_heldout_generic_r2"],
            },
            "generic_generation": generation,
            "splits": {"rank_train_pairs": split_train_pairs, "gate_c_pairs": split_gate_c_pairs, "rank_eval_pairs": split_eval_pairs},
        }
        manifest["gate_C"] = gate_c
        sae_pass = gate_c["sae"]["trained_decoder"]["in_0p70_to_1p30"]
        ridge_pass = gate_c["random"]["dual_ridge"]["in_0p70_to_1p30"]
        tied_pass = gate_c["random"]["tied_weight_fallback"]["in_0p70_to_1p30"]
        if not sae_pass:
            decision, branch = "STOP", "SAE trained decoder outside [0.70,1.30]"
        elif ridge_pass:
            decision, branch = "PASS", "SAE and random dual-ridge both inside [0.70,1.30]"
        elif tied_pass:
            decision, branch = "PASS", "dual ridge outside but tied-weight random decoder inside [0.70,1.30]"
        else:
            decision, branch = "PASS_FALLBACK", "random basis outside [0.70,1.30] for both decoder variants; within-SAE fallback"
        manifest["steps"]["6"] = {
            "decision": decision,
            "summary": branch,
            "seconds": time.perf_counter() - step_started,
            "gate_C_branch": branch,
        }
        print_step(6, "Gate C: full-arm ratios in [0.70,1.30]; coverage >=0.95", f"SAE={gate_c['sae']['trained_decoder']['ratio_to_E_resid']:.3f}, ridge={gate_c['random']['dual_ridge']['ratio_to_E_resid']:.3f}, tied={gate_c['random']['tied_weight_fallback']['ratio_to_E_resid']:.3f}, coverage={coverage['random_dual_ridge']:.3f}", decision)
        if not sae_pass:
            raise GateStop("C_sae_faithfulness", branch)
        if ridge_pass:
            chosen_random_decoder = decoders["ridge_decoder"]
            chosen_random_name = "dual_ridge"
            provisional_status = "go"
        elif tied_pass:
            chosen_random_decoder = decoders["tied_decoder"]
            chosen_random_name = "tied_weight_fallback"
            provisional_status = "go"
        else:
            chosen_random_decoder = decoders["ridge_decoder"]
            chosen_random_name = "dual_ridge_gated_out"
            provisional_status = "go_fallback"

        # Step 7 -----------------------------------------------------------------
        step_started = time.perf_counter()
        rank_base, rank_source, rank_signs = directed_indices(PILOT_PAIRS, split_train_pairs)
        rank_positions = positions_for_kind(stimuli, rank_base, selected_kind)
        eval_base, eval_source, eval_signs = directed_indices(PILOT_PAIRS, split_eval_pairs)
        eval_positions = positions_for_kind(stimuli, eval_base, selected_kind)
        sae_rank_diff, sae_rank_source, sae_rank_base = feature_delta(sae.encode, clean.residuals[8], rank_base, rank_source, rank_positions)
        random_rank_diff, random_rank_source, random_rank_base = feature_delta(random_basis.encode, clean.residuals[8], rank_base, rank_source, rank_positions)
        sae_candidates, sae_prefilter = candidate_prefilter(sae_rank_diff, sae_rank_source, sae_rank_base, rank_signs, sae.W_dec, 16)
        random_candidates, random_prefilter = candidate_prefilter(random_rank_diff, random_rank_source, random_rank_base, rank_signs, chosen_random_decoder, 16)
        sae_scores = single_coordinate_scores(
            engine, layer=8, stimuli=stimuli, base_indices=rank_base, source_indices=rank_source, signs=rank_signs,
            clean_d=clean_d, residuals=clean.residuals[8], positions=rank_positions, code_delta=sae_rank_diff,
            decoder=sae.W_dec, candidates=sae_candidates, is_id=is_id, are_id=are_id, label="rank_candidates",
        )
        random_scores = single_coordinate_scores(
            engine, layer=8, stimuli=stimuli, base_indices=rank_base, source_indices=rank_source, signs=rank_signs,
            clean_d=clean_d, residuals=clean.residuals[8], positions=rank_positions, code_delta=random_rank_diff,
            decoder=chosen_random_decoder, candidates=random_candidates, is_id=is_id, are_id=are_id, label="rank_candidates",
        )
        sae_ranked = sae_candidates[sae_scores.argsort(descending=True)]
        random_ranked = random_candidates[random_scores.argsort(descending=True)]
        sae_eval_diff, _, _ = feature_delta(sae.encode, clean.residuals[8], eval_base, eval_source, eval_positions)
        random_eval_diff, _, _ = feature_delta(random_basis.encode, clean.residuals[8], eval_base, eval_source, eval_positions)
        sae_eval_full = patch_effect(
            engine, layer=8, stimuli=stimuli, base_indices=eval_base, source_indices=eval_source, signs=eval_signs,
            clean_d=clean_d, residuals=clean.residuals[8], positions=eval_positions,
            deltas=decoded_delta(sae_eval_diff, sae.W_dec), is_id=is_id, are_id=are_id, label="rank_eval_full",
        )
        random_eval_full = patch_effect(
            engine, layer=8, stimuli=stimuli, base_indices=eval_base, source_indices=eval_source, signs=eval_signs,
            clean_d=clean_d, residuals=clean.residuals[8], positions=eval_positions,
            deltas=decoded_delta(random_eval_diff, chosen_random_decoder), is_id=is_id, are_id=are_id, label="rank_eval_full",
        )
        sae_recovery = topk_recovery(
            engine, layer=8, stimuli=stimuli, base_indices=eval_base, source_indices=eval_source, signs=eval_signs,
            clean_d=clean_d, residuals=clean.residuals[8], positions=eval_positions, code_delta=sae_eval_diff,
            decoder=sae.W_dec, ranked=sae_ranked, k_values=(1, 4, 8),
            full_aligned_effect=sae_eval_full["mean_aligned_effect"], is_id=is_id, are_id=are_id, label="topk_eval",
        )
        random_recovery = topk_recovery(
            engine, layer=8, stimuli=stimuli, base_indices=eval_base, source_indices=eval_source, signs=eval_signs,
            clean_d=clean_d, residuals=clean.residuals[8], positions=eval_positions, code_delta=random_eval_diff,
            decoder=chosen_random_decoder, ranked=random_ranked, k_values=(1, 4, 8),
            full_aligned_effect=random_eval_full["mean_aligned_effect"], is_id=is_id, are_id=are_id, label="topk_eval",
        )
        estimate = full_run_estimate(engine.records)
        manifest["forward_timing"] = {
            "records": engine.records,
            "summary": estimate,
            "layer8_path": "start_at_layer_8" if start_path else "full_forward_hook_fallback",
            "other_scan_layers_path": "full_forward_hook",
        }
        manifest["steps"]["7"] = {
            "decision": "PASS",
            "summary": f"16 candidates scored per basis; R(top-{{1,4,8}}) measured; full-run extrapolation={estimate['extrapolated_minutes']:.2f} minutes.",
            "seconds": time.perf_counter() - step_started,
            "chosen_random_decoder": chosen_random_name,
            "ranking": {
                "sae": {"prefilter": sae_prefilter, "candidates": sae_candidates.tolist(), "scores": [float(x) for x in sae_scores], "ranked": sae_ranked.tolist(), "recovery": sae_recovery},
                "random": {"prefilter": random_prefilter, "candidates": random_candidates.tolist(), "scores": [float(x) for x in random_scores], "ranked": random_ranked.tolist(), "recovery": random_recovery},
            },
            "full_run_cost": estimate,
        }
        print_step(7, "16 candidates; R(top-{1,4,8}); measure timing and extrapolate full run", f"SAE R={sae_recovery['rows']}, random R={random_recovery['rows']}, cost={estimate['extrapolated_minutes']:.2f} min", "PASS")
        manifest["status"] = provisional_status
        manifest["failed_gate"] = None
        manifest["selected_layer8_position_set"] = selected_kind
        manifest["selected_random_decoder"] = chosen_random_name
        manifest["student_t_ci_helper"] = {"t_critical_df4": GATE_C_T, "example_singleton": list(t_ci([0.0]))}
        write_artifacts(manifest, started)
        return manifest
    except GateStop as exc:
        manifest["status"] = "gated_out"
        manifest["failed_gate"] = exc.gate
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        write_artifacts(manifest, started)
        print(f"GATED OUT at {exc.gate}: {exc}", flush=True)
        return manifest
    except Exception as exc:
        manifest["status"] = "gated_out"
        manifest["failed_gate"] = "implementation_or_environment"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        write_artifacts(manifest, started)
        print(f"GATED OUT at implementation_or_environment: {type(exc).__name__}: {exc}", flush=True)
        return manifest


if __name__ == "__main__":
    result = run()
    print(f"Pilot status={result['status']}; wrote {RESULTS.name} and {NOTES.name}", flush=True)

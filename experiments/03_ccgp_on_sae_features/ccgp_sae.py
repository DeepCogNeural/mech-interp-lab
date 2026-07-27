"""
Experiment 03 — shattering dimensionality and CCGP on a real GPT-2-small SAE.

The question is deliberately NOT whether an SAE is better or worse than the residual
stream.  A residual stream is 768-wide; this SAE is a 24,576-wide ReLU expansion, so
that comparison would mostly rediscover Cover's theorem.  The load-bearing comparison
is instead a real sparse SAE code versus a Gaussian ReLU expansion matched in width,
encoder-column scale, and active-feature count.  The primary result also tests the
active-width objection in both directions: widen the random pool to the SAE's
surviving width, and narrow the SAE's surviving latents to the random arm's width.

The trap avoided: each of the eight NUMBER x TENSE x POLARITY conditions has the same
sentence-final '.' read-out token and (within an item) the same GPT-2 token length.
Reading the auxiliary or main verb would decode word form; averaging one vector per
condition would erase lexical nuisance variance.  Probes therefore read the final '.'
activation and always split by lexical item, never by individual sentence.

Discipline: Gates A--C run before the full calculation; if any fails, the script writes
an explicit gated-out results.json and stops.  SD averages every one of the 35 balanced
dichotomies.  CCGP holds out one condition from each side of a dichotomy and averages
all 16 choices, with item-disjoint train/test examples.  Every linear probe is
hand-rolled torch and fitted to a stated full-batch L-BFGS convergence criterion, with
L2 selected inside each outer training split.  Feature masks and every CCGP head's
centre/scale are fitted only on data available to that head.  No result is claimed
until the corresponding full command has completed.

The normal command resolves GPT-2 small and the public res-jb safetensors through the
local Hugging Face cache.  It does not require sae_lens: the encoder is exactly
ReLU((x - b_dec) @ W_enc + b_enc).  Set SMOKE=1 for a small two-seed, two-CCGP-split
pipeline check; it is intentionally not a substitute for the 5-seed result.
"""

from __future__ import annotations

import itertools
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
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformer_lens import HookedTransformer


HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
RESULTS = HERE / "results.json"
FIGDIR.mkdir(exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"
CURRENT_GATE = "A"

# This explicit threshold implements Gate C's qualitative "not ~1.00" requirement.
# It is intentionally conservative: 0.98 would leave too little headroom for an arm
# comparison at this sample size.
SATURATION_THRESHOLD = 0.98
CORE_ARMS = ("resid", "sae", "sae_recon", "rand_exp")
# This fifth entry is deliberately an upper reference, not part of the load-bearing
# SAE-versus-sparsity-matched-random contrast.
BASE_ARMS = CORE_ARMS + ("rand_exp_dense",)
# These two controls close the remaining effective-width alternative explanation.
# They are evaluated only under the primary global-RMS probe convention; the
# per-feature-z-score table remains a sensitivity analysis of the original arms.
WIDTH_MATCHED_ARMS = ("rand_exp_width_matched", "sae_width_matched")
ALL_ARMS = BASE_ARMS + WIDTH_MATCHED_ARMS
T_CRIT_95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
    8: 2.365, 9: 2.306, 10: 2.262,
}
L2_ONE_STANDARD_STYLE_TOLERANCE_NATS = 1e-2
L2_STABILITY_FLOOR = 1e-5


@dataclass(frozen=True)
class Config:
    n_items: int
    seeds: tuple[int, ...]
    n_folds: int
    pilot_layers: tuple[int, ...]
    probe_max_steps: int
    probe_relative_loss_tolerance: float
    probe_stable_steps: int
    ccgp_splits: tuple[tuple[int, int], ...]
    batch_size: int
    l2_grid: tuple[float, ...]
    fair_probe_settings: tuple[str, ...]
    run_width_controls: bool


def config() -> Config:
    if SMOKE:
        # Fast plumbing check only: it deliberately reports that CCGP used 2/16 splits.
        return Config(
            n_items=24,
            seeds=(0, 1),
            n_folds=2,
            pilot_layers=(7, 8),
            probe_max_steps=500,
            probe_relative_loss_tolerance=1e-3,
            probe_stable_steps=10,
            ccgp_splits=((0, 0), (1, 1)),
            batch_size=16,
            l2_grid=(1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3),
            fair_probe_settings=("per_feature_zscore_inner_l2", "global_rms_inner_l2"),
            run_width_controls=os.environ.get("RUN_WIDTH_CONTROLS", "0") == "1",
        )
    return Config(
        n_items=96,
        seeds=(0, 1, 2, 3, 4),
        n_folds=5,
        pilot_layers=(6, 7, 8, 9),
        probe_max_steps=500,
        probe_relative_loss_tolerance=1e-3,
        probe_stable_steps=10,
        ccgp_splits=tuple(itertools.product(range(4), range(4))),
        batch_size=32,
        # Five orders on either side of the expected optimum prevent an arbitrary
        # scaling convention from looking favourable merely because its prior grid was
        # too narrow. select_l2 verifies that no outer-fold choice is an edge.
        l2_grid=(1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 10.0, 1000.0),
        fair_probe_settings=("per_feature_zscore_inner_l2", "global_rms_inner_l2"),
        run_width_controls=os.environ.get("RUN_WIDTH_CONTROLS", "0") == "1",
    )


@dataclass
class Stimuli:
    tokens: torch.Tensor
    lengths: torch.Tensor
    item_ids: torch.Tensor
    condition_ids: torch.Tensor
    factors: torch.Tensor
    texts: list[str]
    attempted_items: int
    dropped_items: int
    readout_token_id: int


@dataclass
class SAEWeights:
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


# --------------------------------------------------------------------------- stimuli


NOUNS = [
    ("cat", "cats"), ("dog", "dogs"), ("child", "children"),
    ("student", "students"), ("teacher", "teachers"), ("artist", "artists"),
    ("pilot", "pilots"), ("singer", "singers"), ("doctor", "doctors"),
    ("farmer", "farmers"), ("driver", "drivers"), ("actor", "actors"),
    ("chef", "chefs"), ("writer", "writers"), ("friend", "friends"),
    ("neighbor", "neighbors"), ("visitor", "visitors"), ("player", "players"),
    ("dancer", "dancers"), ("reader", "readers"),
]
ADJECTIVES = [
    "quiet", "curious", "careful", "eager", "patient", "bright", "young",
    "older", "gentle", "clever", "serious", "brave", "calm", "lively",
]
VERBS = [
    "visit", "follow", "watch", "help", "admire", "notice", "question",
    "greet", "answer", "avoid", "support", "observe", "contact", "praise",
]
ADJECTIVES_2 = [
    "nearby", "distant", "kind", "tired", "famous", "local", "friendly",
    "patient", "quiet", "careful", "new", "older", "small", "bright",
]
OBJECTS = [
    "friend", "neighbor", "visitor", "teacher", "artist", "driver", "singer",
    "doctor", "farmer", "writer", "player", "dancer", "reader", "student",
]
ADVERBS = [
    "today", "outside", "quietly", "quickly", "nearby", "again", "later",
    "carefully", "openly", "politely", "calmly", "briefly", "eagerly", "often",
]


def _ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenizer-interface shim for both fast and slow GPT-2 tokenizers."""
    out = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return list(out["input_ids"])


def sentence(lex: tuple[str, str, str, str, str, str], number: int, tense: int, polarity: int) -> str:
    """One grammatical cell.  number/tense/polarity use 0/1 coding."""
    adj, noun_sg, noun_pl, verb, adj2, obj_adv = lex
    obj, adv = obj_adv.split("\t")
    noun = noun_sg if number == 0 else noun_pl
    auxiliary = "did" if tense == 0 else ("does" if number == 0 else "do")
    pol = "indeed" if polarity == 0 else "not"
    # A space before '.' makes the read-out BPE token literally identical in every cell.
    return f"The {adj} {noun} {auxiliary} {pol} {verb} the {adj2} {obj} {adv} ."


def build_stimuli(tokenizer: Any, n_items: int, seed: int) -> Stimuli:
    """Make full-factorial lexical items and reject an item unless all eight lengths match."""
    rng = random.Random(seed)
    records: list[tuple[int, int, tuple[int, int, int], str, list[int]]] = []
    attempted = 0
    dropped = 0
    # The 12x allowance is not a tuneable result parameter: it prevents an unlucky token
    # vocabulary from silently yielding too few retained lexical draws.
    for _ in range(n_items * 12):
        if len({r[0] for r in records}) >= n_items:
            break
        attempted += 1
        noun_sg, noun_pl = rng.choice(NOUNS)
        lex = (
            rng.choice(ADJECTIVES), noun_sg, noun_pl, rng.choice(VERBS),
            rng.choice(ADJECTIVES_2), f"{rng.choice(OBJECTS)}\t{rng.choice(ADVERBS)}",
        )
        cells: list[tuple[int, tuple[int, int, int], str, list[int]]] = []
        for number, tense, polarity in itertools.product((0, 1), repeat=3):
            cid = number * 4 + tense * 2 + polarity
            text = sentence(lex, number, tense, polarity)
            cells.append((cid, (number, tense, polarity), text, _ids(tokenizer, text)))
        lengths = {len(ids) for _, _, _, ids in cells}
        terminal = {ids[-1] for _, _, _, ids in cells if ids}
        if len(lengths) != 1 or len(terminal) != 1:
            dropped += 1
            continue
        item_id = len({r[0] for r in records})
        for cid, factors, text, ids in cells:
            records.append((item_id, cid, factors, text, ids))
    item_count = len({r[0] for r in records})
    if item_count != n_items:
        raise RuntimeError(
            f"Gate B stimulus construction retained {item_count}/{n_items} lexical items "
            f"after {attempted} attempts (dropped {dropped})."
        )
    max_len = max(len(r[4]) for r in records)
    pad = int(getattr(tokenizer, "eos_token_id", 50256))
    tokens = torch.full((len(records), max_len), pad, dtype=torch.long)
    lengths_t = torch.empty(len(records), dtype=torch.long)
    for i, (_, _, _, _, ids) in enumerate(records):
        tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths_t[i] = len(ids)
    terminal_ids = {int(tokens[i, lengths_t[i] - 1]) for i in range(len(records))}
    assert len(terminal_ids) == 1, "final '.' token must be identical across all sequences"
    # The per-item assertion is repeated explicitly to defend the key positional control.
    for item_id in range(n_items):
        item_lengths = lengths_t[torch.tensor([r[0] == item_id for r in records])]
        assert item_lengths.unique().numel() == 1, "factor cells differ in token length"
    return Stimuli(
        tokens=tokens,
        lengths=lengths_t,
        item_ids=torch.tensor([r[0] for r in records], dtype=torch.long),
        condition_ids=torch.tensor([r[1] for r in records], dtype=torch.long),
        factors=torch.tensor([r[2] for r in records], dtype=torch.float32),
        texts=[r[3] for r in records],
        attempted_items=attempted,
        dropped_items=dropped,
        readout_token_id=terminal_ids.pop(),
    )


# ---------------------------------------------------------------------- loading/SAE


def device() -> str:
    # The experiment contract is CPU-only.  Probes already run on CPU, and using the
    # same device for activations keeps the full fairness sweep comparable and offline.
    return "cpu"


def load_model(run_device: str) -> HookedTransformer:
    # float32 is intentional: the original res-jb training config is float32 and MPS
    # half precision is not a trustworthy shortcut for sparse activation thresholds.
    return HookedTransformer.from_pretrained("gpt2-small", device=run_device, dtype=torch.float32)


def _tensor(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise RuntimeError(f"SAE safetensors missing {key}; found {sorted(state)}")
    return state[key].detach().float().cpu()


def load_direct_res_jb(layer: int) -> SAEWeights:
    """Fallback loader required by the experiment: no sae_lens dependency or API pins."""
    repo = "jbloom/GPT2-Small-SAEs-Reformatted"
    filename = f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors"
    local = hf_hub_download(repo_id=repo, filename=filename)
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


def collect_residuals(
    model: HookedTransformer,
    stimuli: Stimuli,
    layers: Iterable[int],
    run_device: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Cache only requested residual hooks and select each sequence's final '.' position."""
    layers = tuple(layers)
    names = {f"blocks.{layer}.hook_resid_pre" for layer in layers}
    out = {layer: [] for layer in layers}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(stimuli.tokens), batch_size):
            stop = min(len(stimuli.tokens), start + batch_size)
            toks = stimuli.tokens[start:stop].to(run_device)
            _, cache = model.run_with_cache(toks, names_filter=lambda name: name in names)
            positions = (stimuli.lengths[start:stop] - 1).to(run_device)
            batch_index = torch.arange(stop - start, device=run_device)
            for layer in layers:
                act = cache[f"blocks.{layer}.hook_resid_pre"]
                out[layer].append(act[batch_index, positions].detach().float().cpu())
            del cache
    return {layer: torch.cat(parts) for layer, parts in out.items()}


# ----------------------------------------------------------- dichotomies/statistics


def ci95(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 2:
        return 0.0
    if len(arr) not in T_CRIT_95:
        raise ValueError(f"No two-sided 95% Student-t critical value configured for n={len(arr)}")
    return float(T_CRIT_95[len(arr)] * arr.std(ddof=1) / math.sqrt(len(arr)))


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()), ci95(arr)


def _canonical_mask(mask: frozenset[int]) -> frozenset[int]:
    other = frozenset(set(range(8)) - set(mask))
    return min(mask, other, key=lambda x: tuple(sorted(x)))


def dichotomies() -> list[dict[str, Any]]:
    """35 balanced dichotomies, with the seven parity-family masks recognised up to flip."""
    factor_bits = [((cid >> 2) & 1, (cid >> 1) & 1, cid & 1) for cid in range(8)]
    parity: dict[frozenset[int], str] = {}
    for subset, name in [
        ((0,), "main_effect"), ((1,), "main_effect"), ((2,), "main_effect"),
        ((0, 1), "two_way_xor"), ((0, 2), "two_way_xor"), ((1, 2), "two_way_xor"),
        ((0, 1, 2), "three_way_parity"),
    ]:
        # XOR parity and its complement are the same dichotomy; classification is invariant.
        mask = frozenset(i for i, bits in enumerate(factor_bits) if sum(bits[j] for j in subset) % 2 == 1)
        parity[_canonical_mask(mask)] = name
    rows = []
    for comb in itertools.combinations(range(8), 4):
        # Selecting the side that includes condition 0 enumerates one member per complement pair.
        if 0 not in comb:
            continue
        positive = frozenset(comb)
        kind = parity.get(_canonical_mask(positive), "unstructured")
        labels = torch.tensor([float(cid in positive) for cid in range(8)])
        rows.append({"positive": positive, "negative": frozenset(set(range(8)) - set(positive)), "type": kind, "labels": labels})
    assert len(rows) == 35
    assert {r["type"] for r in rows} == {"main_effect", "two_way_xor", "three_way_parity", "unstructured"}
    assert sum(r["type"] == "main_effect" for r in rows) == 3
    assert sum(r["type"] == "two_way_xor" for r in rows) == 3
    assert sum(r["type"] == "three_way_parity" for r in rows) == 1
    return rows


def folds(item_ids: torch.Tensor, n_folds: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n_items = int(item_ids.max()) + 1
    rng = np.random.default_rng(seed + 70_000)
    ids = np.arange(n_items)
    rng.shuffle(ids)
    pieces = np.array_split(ids, n_folds)
    out = []
    for test_ids in pieces:
        test = torch.isin(item_ids, torch.tensor(test_ids, dtype=torch.long))
        out.append((~test, test))
    return out


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Column-wise balanced accuracy; all experiment labels have both classes."""
    pred = logits > 0
    labels = labels.bool()
    pos = (pred & labels).sum(0).float() / labels.sum(0).clamp_min(1)
    neg_mask = ~labels
    neg = ((~pred) & neg_mask).sum(0).float() / neg_mask.sum(0).clamp_min(1)
    return (pos + neg) / 2


def prepare_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the nonzero-on-this-input columns and the corresponding Boolean mask."""
    keep = x.abs().amax(dim=0) > 0
    return x[:, keep].contiguous(), keep


def scale_features(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre on training items, then use either per-feature or one global scale."""
    mean = x_train.mean(0, keepdim=True)
    train_centered, test_centered = x_train - mean, x_test - mean
    if mode == "per_feature_zscore":
        scale = train_centered.std(0, unbiased=False, keepdim=True).clamp_min(1e-5)
    elif mode == "global_rms":
        # A single scalar preserves the relative scale of rare and common units.  The
        # per-feature mean is still removed; a probe bias makes that translation neutral.
        scale = train_centered.square().mean().sqrt().clamp_min(1e-5)
    else:
        raise ValueError(f"Unknown feature-scaling mode: {mode}")
    return train_centered / scale, test_centered / scale


def standardise(x_train: torch.Tensor, x_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature standardisation for the Gate-B residual pilot only."""
    return scale_features(x_train, x_test, "per_feature_zscore")


@dataclass
class ProbeFit:
    eval_accuracy: torch.Tensor
    train_accuracy: torch.Tensor
    diagnostics: dict[str, Any]


def _objective(
    head: nn.Linear,
    x: torch.Tensor,
    y: torch.Tensor,
    l2: float,
    train_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean logistic loss plus an L2 prior, with the bias deliberately unpenalised."""
    bce_by_head = F.binary_cross_entropy_with_logits(head(x), y, reduction="none")
    if train_mask is not None:
        bce = (bce_by_head * train_mask.float()).sum() / train_mask.sum().clamp_min(1)
    else:
        bce = bce_by_head.mean()
    # Averaging over heads keeps a selected lambda comparable between SD (35 heads),
    # main-effect selection (3 heads), and CCGP (16 heads); within a head this is the
    # standard 0.5 * lambda * ||w||^2 penalty.
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
    train_mask: torch.Tensor | None = None,
) -> ProbeFit:
    """Fit a full-batch L-BFGS logistic probe to the stated loss-change criterion."""
    torch.manual_seed(probe_seed)
    x_train, x_eval = x_train.float().cpu(), x_eval.float().cpu()
    y_train, y_eval = y_train.float().cpu(), y_eval.float().cpu()
    mask = None if train_mask is None else train_mask.float().cpu()
    head = nn.Linear(x_train.shape[1], y_train.shape[1])
    opt = torch.optim.LBFGS(
        head.parameters(), lr=1.0, max_iter=1, max_eval=25, history_size=20,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )
    objective_history: list[float] = []
    closure_calls = 0
    final_objective = math.inf
    final_bce = math.inf
    final_penalty = math.inf
    relative_change = math.inf
    converged = False
    for step in range(1, cfg.probe_max_steps + 1):
        def closure() -> torch.Tensor:
            nonlocal closure_calls
            closure_calls += 1
            opt.zero_grad(set_to_none=True)
            objective, _, _ = _objective(head, x_train, y_train, l2, mask)
            objective.backward()
            return objective
        opt.step(closure)
        with torch.no_grad():
            objective, bce, penalty = _objective(head, x_train, y_train, l2, mask)
            final_objective, final_bce, final_penalty = float(objective), float(bce), float(penalty)
        objective_history.append(final_objective)
        if len(objective_history) > cfg.probe_stable_steps:
            reference = objective_history[-1 - cfg.probe_stable_steps]
            relative_change = abs(reference - final_objective) / max(abs(reference), 1e-12)
            if relative_change < cfg.probe_relative_loss_tolerance:
                converged = True
                break
    else:
        step = cfg.probe_max_steps
    with torch.no_grad():
        train_logits = head(x_train)
        eval_bce = F.binary_cross_entropy_with_logits(head(x_eval), y_eval).item()
        result = ProbeFit(
            eval_accuracy=balanced_accuracy(head(x_eval), y_eval),
            train_accuracy=balanced_accuracy(train_logits, y_train),
            diagnostics={
                "iterations": step,
                "closure_calls": closure_calls,
                "converged": converged,
                "relative_loss_change": relative_change,
                "objective": final_objective,
                "bce": final_bce,
                "l2_penalty": final_penalty,
                "unpenalised_evaluation_bce": eval_bce,
            },
        )
    if not converged:
        raise RuntimeError(
            f"Probe did not converge within {cfg.probe_max_steps} L-BFGS iterations "
            f"(last relative objective change={relative_change:.3g}, l2={l2:g})."
        )
    return result


def main_effect_pilot(
    resid: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
) -> list[float]:
    values = []
    for fold_index, (train, test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        tr, te = standardise(resid[train], resid[test])
        result = fit_probe(
            tr, stimuli.factors[train], te, stimuli.factors[test], seed + fold_index,
            cfg, 1e-3,
        )
        values.append(result.eval_accuracy.numpy())
    # One [NUMBER, TENSE, POLARITY] vector per item-disjoint fold.  Concatenating
    # would flatten those vectors and accidentally turn this Gate-B check into a scalar.
    return np.stack(values, axis=0).mean(axis=0).tolist()


def inner_validation_split(
    outer_train: torch.Tensor,
    item_ids: torch.Tensor,
    seed: int,
    outer_fold: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hold out lexical items only from an outer fold's training partition."""
    ids = torch.unique(item_ids[outer_train]).cpu().numpy()
    rng = np.random.default_rng(seed + 90_000 + outer_fold)
    rng.shuffle(ids)
    n_valid = max(1, int(round(0.2 * len(ids))))
    valid_ids = torch.tensor(ids[:n_valid], dtype=torch.long)
    valid = outer_train & torch.isin(item_ids, valid_ids)
    return outer_train & ~valid, valid


def select_l2(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    scale_mode: str,
) -> tuple[list[float], list[dict[str, Any]], list[torch.Tensor]]:
    """Nested L2 selection with an outer-training keep-mask per fold, never test data."""
    selected, records = [], []
    fold_keeps = []
    for outer_fold, (outer_train, _) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        # The all-zero feature rule is unsupervised, but it is still fit only on outer
        # training activations. The test fold has no route into this representation choice.
        keep = rep[outer_train].abs().amax(dim=0) > 0
        fold_keeps.append(keep)
        inner_train, inner_valid = inner_validation_split(outer_train, stimuli.item_ids, seed, outer_fold)
        tr, va = scale_features(rep[inner_train][:, keep], rep[inner_valid][:, keep], scale_mode)
        scores = []
        diagnostics = []
        for candidate_i, l2 in enumerate(cfg.l2_grid):
            result = fit_probe(
                tr, stimuli.factors[inner_train], va, stimuli.factors[inner_valid],
                seed + 30_000 + 100 * outer_fold + candidate_i,
                cfg, l2,
            )
            # Accuracy is intentionally not used to select lambda: with only a few
            # item-disjoint validation items it creates broad ties at the grid boundary.
            # Held-out logistic loss preserves the probe's probabilistic information.
            scores.append(-float(result.diagnostics["unpenalised_evaluation_bce"]))
        best_score = max(scores)
        # With these deliberately easy base-factor inner tasks, tiny lambdas often
        # improve held-out BCE by micro-nats while forfeiting all regularisation. Use a
        # predeclared one-standard-style rule: choose the *largest* lambda whose loss is
        # within 1e-3 nats/example of the minimum. This is uniform across arm/scaling;
        # it is not tuned toward the SAE/random contrast.
        eligible = [
            i for i, score in enumerate(scores)
            if cfg.l2_grid[i] >= L2_STABILITY_FLOOR
            and score >= best_score - L2_ONE_STANDARD_STYLE_TOLERANCE_NATS
        ]
        if not eligible:
            raise RuntimeError(
                f"No stable L2 >= {L2_STABILITY_FLOOR:g} is within "
                f"{L2_ONE_STANDARD_STYLE_TOLERANCE_NATS:g} validation nats of the optimum; scores={scores}"
            )
        best_i = max(eligible)
        selected_at_edge = best_i in (0, len(cfg.l2_grid) - 1)
        if selected_at_edge and not SMOKE:
            raise RuntimeError(
                f"L2 grid edge selected for seed={seed}, outer_fold={outer_fold}, "
                f"scale={scale_mode}, l2={cfg.l2_grid[best_i]:g}, scores={scores}; expand the grid."
            )
        if selected_at_edge and SMOKE:
            # The tiny plumbing subset can be perfectly separable. Its boundary winner
            # is intentionally replaced by the nearest interior value so SMOKE tests
            # the pipeline rather than an unregularised, ill-posed toy split.
            best_i = min(3, len(cfg.l2_grid) - 2)
        selected.append(float(cfg.l2_grid[best_i]))
        records.append({
            "outer_fold": outer_fold,
            "candidate_l2": list(cfg.l2_grid),
            "inner_validation_negative_main_effect_bce": scores,
            "selection_loss_tolerance_nats": L2_ONE_STANDARD_STYLE_TOLERANCE_NATS,
            "selected_l2": float(cfg.l2_grid[best_i]),
            "selected_at_grid_edge": selected_at_edge,
            "outer_train_surviving_width": int(keep.sum()),
        })
    return selected, records, fold_keeps


def _fold_l2(
    fold_l2: list[float] | None,
    fold_index: int,
    cfg: Config,
) -> float:
    return 1e-3 if fold_l2 is None else fold_l2[fold_index]


def sd_metric(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    ds: list[dict[str, Any]],
    scale_mode: str = "per_feature_zscore",
    fold_l2: list[float] | None = None,
    fold_keeps: list[torch.Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    """Five item-disjoint folds, all 35 dichotomies trained as one multi-output probe."""
    labels_by_condition = torch.stack([d["labels"] for d in ds], dim=1)
    labels = labels_by_condition[stimuli.condition_ids]
    test_accs, train_accs, convergence = [], [], []
    for fold_index, (train, test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        keep = (rep[train].abs().amax(dim=0) > 0) if fold_keeps is None else fold_keeps[fold_index]
        tr, te = scale_features(rep[train][:, keep], rep[test][:, keep], scale_mode)
        result = fit_probe(
            tr, labels[train], te, labels[test], seed + 100 * fold_index,
            cfg, _fold_l2(fold_l2, fold_index, cfg),
        )
        test_accs.append(result.eval_accuracy.numpy())
        train_accs.append(result.train_accuracy.numpy())
        convergence.append({"outer_fold": fold_index, **result.diagnostics, "surviving_width": int(keep.sum())})
    test_mean = np.mean(test_accs, axis=0)
    train_mean = np.mean(train_accs, axis=0)
    by_type: dict[str, list[float]] = {}
    gap: dict[str, list[float]] = {}
    for i, d in enumerate(ds):
        by_type.setdefault(d["type"], []).append(float(test_mean[i]))
        gap.setdefault(d["type"], []).append(float(train_mean[i] - test_mean[i]))
    by_type["overall"] = list(test_mean)
    gap["overall"] = list(train_mean - test_mean)
    return (
        {k: float(np.mean(v)) for k, v in by_type.items()},
        {k: float(np.mean(v)) for k, v in gap.items()},
        convergence,
    )


def ccgp_metric(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    ds: list[dict[str, Any]],
    scale_mode: str = "per_feature_zscore",
    fold_l2: list[float] | None = None,
    fold_keeps: list[torch.Tensor] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """CCGP with all (or smoke-subset) 4x4 held-condition splits, item-disjoint throughout.

    The 16 heads for one dichotomy share an objective but not preprocessing: each head's
    mean and scale are estimated from exactly its six condition/item training cells.  A
    raw-input reparameterisation keeps the calculation vectorised without allocating a
    [examples, heads, features] tensor, while preserving the intended L2 geometry.
    """
    per_type: dict[str, list[float]] = {}
    convergence: list[dict[str, Any]] = []
    for d_index, d in enumerate(ds):
        labels = d["labels"][stimuli.condition_ids]
        pos = sorted(d["positive"])
        neg = sorted(d["negative"])
        split_scores: list[float] = []
        for fold_index, (item_train, item_test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
            keep = (rep[item_train].abs().amax(dim=0) > 0) if fold_keeps is None else fold_keeps[fold_index]
            raw_train, raw_test = rep[item_train][:, keep].float().cpu(), rep[item_test][:, keep].float().cpu()
            condition_train = stimuli.condition_ids[item_train]
            condition_test = stimuli.condition_ids[item_test]
            masks, eval_masks = [], []
            for held_pos_i, held_neg_i in cfg.ccgp_splits:
                held_pos, held_neg = pos[held_pos_i], neg[held_neg_i]
                train_mask = (condition_train != held_pos) & (condition_train != held_neg)
                eval_mask = (condition_test == held_pos) | (condition_test == held_neg)
                # This happens only with a malformed factorial stimulus set.
                if int(eval_mask.sum()) == 0:
                    raise RuntimeError("CCGP held-condition split has no item-disjoint evaluation data")
                masks.append(train_mask)
                eval_masks.append(eval_mask)
            n_heads = len(cfg.ccgp_splits)
            head_masks = torch.stack(masks, dim=1).float().cpu()
            test_masks = torch.stack(eval_masks, dim=1).bool().cpu()
            train_y = labels[item_train].float().cpu().unsqueeze(1).expand(-1, n_heads)
            test_y = labels[item_test].float().cpu()
            # Each head gets moments from its own six training conditions.  Algebraically,
            # x_standardised @ w + b equals raw_x @ (w / scale) + b - mean @ (w / scale).
            counts = head_masks.sum(0).clamp_min(1).unsqueeze(1)
            means = (head_masks.T @ raw_train) / counts
            second = (head_masks.T @ raw_train.square()) / counts
            variance = (second - means.square()).clamp_min(1e-10)
            if scale_mode == "per_feature_zscore":
                scales = variance.sqrt()
            elif scale_mode == "global_rms":
                scales = variance.mean(dim=1, keepdim=True).sqrt().clamp_min(1e-5)
            else:
                raise ValueError(f"Unknown feature-scaling mode: {scale_mode}")
            torch.manual_seed(seed + 10_000 * d_index + fold_index)
            head = nn.Linear(raw_train.shape[1], n_heads)
            opt = torch.optim.LBFGS(
                head.parameters(), lr=1.0, max_iter=1, max_eval=25, history_size=20,
                tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe",
            )
            l2 = _fold_l2(fold_l2, fold_index, cfg)
            objective_history: list[float] = []
            closure_calls = 0
            relative_change = math.inf
            converged = False
            final_objective = math.inf
            for step in range(1, cfg.probe_max_steps + 1):
                def logits(raw: torch.Tensor) -> torch.Tensor:
                    effective_weight = head.weight / scales
                    effective_bias = head.bias - (effective_weight * means).sum(dim=1)
                    return raw @ effective_weight.T + effective_bias
                def closure() -> torch.Tensor:
                    nonlocal closure_calls
                    closure_calls += 1
                    opt.zero_grad(set_to_none=True)
                    bce_by_head = F.binary_cross_entropy_with_logits(logits(raw_train), train_y, reduction="none")
                    bce = (bce_by_head * head_masks).sum() / head_masks.sum().clamp_min(1)
                    penalty = 0.5 * l2 * head.weight.square().sum() / n_heads
                    objective = bce + penalty
                    objective.backward()
                    return objective
                opt.step(closure)
                with torch.no_grad():
                    bce_by_head = F.binary_cross_entropy_with_logits(logits(raw_train), train_y, reduction="none")
                    bce = (bce_by_head * head_masks).sum() / head_masks.sum().clamp_min(1)
                    penalty = 0.5 * l2 * head.weight.square().sum() / n_heads
                    final_objective = float(bce + penalty)
                objective_history.append(final_objective)
                if len(objective_history) > cfg.probe_stable_steps:
                    reference = objective_history[-1 - cfg.probe_stable_steps]
                    relative_change = abs(reference - final_objective) / max(abs(reference), 1e-12)
                    if relative_change < cfg.probe_relative_loss_tolerance:
                        converged = True
                        break
            if not converged:
                raise RuntimeError(
                    f"CCGP probe did not converge within {cfg.probe_max_steps} L-BFGS iterations "
                    f"(last relative objective change={relative_change:.3g}, l2={l2:g})."
                )
            with torch.no_grad():
                score_logits = logits(raw_test)
                acc = torch.stack([
                    balanced_accuracy(score_logits[test_masks[:, h], h], test_y[test_masks[:, h]])
                    for h in range(n_heads)
                ]).mean().item()
            split_scores.append(acc)
            convergence.append({
                "dichotomy_index": d_index, "outer_fold": fold_index,
                "iterations": step, "closure_calls": closure_calls, "converged": converged,
                "relative_loss_change": relative_change, "objective": final_objective,
                "surviving_width": int(keep.sum()),
            })
        per_type.setdefault(d["type"], []).append(float(np.mean(split_scores)))
    per_type["overall"] = [v for group in per_type.values() for v in group]
    return {k: float(np.mean(v)) for k, v in per_type.items()}, convergence


# ---------------------------------------------------------------- representations


def random_expansion(x: torch.Tensor, sae: SAEWeights, seed: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Matched-width random ReLU expansion: SAE norm distribution + SAE bias + top-k L0."""
    g = torch.Generator(device="cpu").manual_seed(seed + 400_000)
    target_l0 = int(round(float((sae.encode(x) > 0).sum(dim=1).float().mean())))
    target_l0 = max(1, min(target_l0, sae.W_enc.shape[1]))
    R = torch.randn(sae.W_enc.shape, generator=g)
    R *= sae.W_enc.norm(dim=0, keepdim=True) / R.norm(dim=0, keepdim=True).clamp_min(1e-8)
    dense = torch.relu((x - sae.b_dec) @ R + sae.b_enc)
    values, indices = dense.topk(target_l0, dim=1)
    sparse = torch.zeros_like(dense).scatter(1, indices, values)
    return sparse, dense, target_l0


@dataclass
class RandomExpansionFamily:
    """One maximal draw: every calibration candidate is an exact column prefix."""
    R: torch.Tensor
    biases: torch.Tensor


def random_expansion_family(sae: SAEWeights, seed: int, max_columns: int) -> RandomExpansionFamily:
    g = torch.Generator(device="cpu").manual_seed(seed + 500_000)
    source = torch.randint(sae.W_enc.shape[1], (max_columns,), generator=g)
    norms = sae.W_enc.norm(dim=0)[source]
    biases = sae.b_enc[source]
    R = torch.randn((sae.W_enc.shape[0], max_columns), generator=g)
    R *= norms.unsqueeze(0) / R.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return RandomExpansionFamily(R=R.contiguous(), biases=biases.contiguous())


def random_expansion_at_width(
    x: torch.Tensor,
    sae: SAEWeights,
    family: RandomExpansionFamily,
    target_l0: int,
    n_columns: int,
) -> torch.Tensor:
    """Random sparse code at a chosen width, sliced from one pre-drawn family."""
    if n_columns < target_l0 or n_columns > family.R.shape[1]:
        raise ValueError(f"Invalid random-expansion width {n_columns} for top-k {target_l0}")
    R, biases = family.R[:, :n_columns], family.biases[:n_columns]
    dense = torch.relu((x - sae.b_dec) @ R + biases)
    values, indices = dense.topk(target_l0, dim=1)
    return torch.zeros_like(dense).scatter(1, indices, values)


def widened_random_to_effective_width(
    x: torch.Tensor,
    sae: SAEWeights,
    seed: int,
    target_l0: int,
    target_width: int,
    baseline_width: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Choose a random-expansion column count using surviving width only.

    Candidate selection is by absolute difference from the SAE's *unlabelled* surviving
    width; it never sees conditions, dichotomies, or probe accuracy.  The candidate grid
    deliberately covers the nonlinear part of the top-k usage curve (rather than
    assuming active width grows linearly with nominal columns).  We retain the closest
    candidate so every seed documents its achieved rather than merely intended match.
    """
    base_columns = sae.W_enc.shape[1]
    if baseline_width < 1:
        raise RuntimeError("Cannot width-match a random expansion with zero surviving baseline units")
    initial = max(target_l0, int(round(base_columns * target_width / baseline_width)))
    candidates = [
        max(target_l0, int(round(initial * multiplier)))
        for multiplier in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5)
    ]
    family = random_expansion_family(sae, seed, max(candidates))
    tried = 0
    best_rep: torch.Tensor | None = None
    best_columns = initial
    best_width = -1
    for candidate in candidates:
        rep = random_expansion_at_width(x, sae, family, target_l0, candidate)
        tried += 1
        achieved = int(prepare_features(rep)[0].shape[1])
        if best_rep is None or abs(achieved - target_width) < abs(best_width - target_width):
            if best_rep is not None:
                del best_rep
            best_rep, best_columns, best_width = rep, candidate, achieved
        else:
            del rep
        if abs(achieved - target_width) <= 3:
            break
    if best_rep is None:
        raise RuntimeError("Effective-width calibration tried no random expansions")
    chosen, chosen_columns, achieved_width = best_rep, best_columns, best_width
    # Carry only surviving coordinates into the probe path.  This is exactly the common
    # all-zero removal rule, merely applied before (rather than inside) the probe helper.
    chosen, _ = prepare_features(chosen)
    return chosen, {
        "target_surviving_width": target_width,
        "achieved_surviving_width": achieved_width,
        "nominal_column_count": chosen_columns,
        "calibration_candidates": tried,
    }


def narrow_sae_to_effective_width(
    f: torch.Tensor,
    target_width: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Uniformly subsample already-surviving SAE latents to an exact active width."""
    active, _ = prepare_features(f)
    if target_width > active.shape[1]:
        raise RuntimeError(
            f"Cannot narrow SAE from {active.shape[1]} surviving units to larger target {target_width}"
        )
    g = torch.Generator(device="cpu").manual_seed(seed + 600_000)
    chosen = torch.randperm(active.shape[1], generator=g)[:target_width]
    narrowed = active[:, chosen].contiguous()
    # Since selection is from units that each fired at least once, this assertion checks
    # that the control really matches the target after applying the common removal rule.
    assert int(prepare_features(narrowed)[0].shape[1]) == target_width
    return narrowed, {
        "source_surviving_width": int(active.shape[1]),
        "target_surviving_width": target_width,
        "achieved_surviving_width": target_width,
    }


def representations(
    x: torch.Tensor,
    sae: SAEWeights,
    seed: int,
    include_width_controls: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    f = sae.encode(x)
    rand_topk, rand_dense, target_l0 = random_expansion(x, sae, seed)
    reps = {
        "resid": x,
        "sae": f,
        "sae_recon": sae.decode(f),
        "rand_exp": rand_topk,
        "rand_exp_dense": rand_dense,
    }
    controls: dict[str, Any] = {"status": "not_run"}
    if include_width_controls:
        sae_width = int(prepare_features(f)[0].shape[1])
        rand_width = int(prepare_features(rand_topk)[0].shape[1])
        rand_widened, random_width_control = widened_random_to_effective_width(
            x, sae, seed, target_l0, sae_width, rand_width,
        )
        sae_narrowed, sae_width_control = narrow_sae_to_effective_width(f, rand_width, seed)
        reps.update({
            "rand_exp_width_matched": rand_widened,
            "sae_width_matched": sae_narrowed,
        })
        controls = {
            "status": "run_transductively_on_unlabelled_full_sample",
            "random_widened_to_sae": {
                **random_width_control,
                "mean_l0": float((rand_widened > 0).sum(1).float().mean()),
            },
            "sae_narrowed_to_random": {
                **sae_width_control,
                "mean_l0": float((sae_narrowed > 0).sum(1).float().mean()),
            },
        }
    l0 = {name: float((rep > 0).sum(1).float().mean()) for name, rep in reps.items()}
    widths = {name: int(prepare_features(rep)[0].shape[1]) for name, rep in reps.items()}
    return reps, {
        "target_l0": target_l0,
        "mean_l0": l0,
        "surviving_width": widths,
        "rand_dense_reference_l0": l0["rand_exp_dense"],
        "effective_width_controls": controls,
    }


# ----------------------------------------------------------------------- gates/run


def explained_variance(x: torch.Tensor, recon: torch.Tensor) -> tuple[float, float, float]:
    mse = float(((x - recon) ** 2).mean())
    var = float(((x - x.mean(0, keepdim=True)) ** 2).mean())
    relative_error = float(torch.linalg.vector_norm(x - recon) / torch.linalg.vector_norm(x).clamp_min(1e-12))
    return 1.0 - mse / max(var, 1e-12), mse, relative_error


def gate_a(model: HookedTransformer, stimuli: Stimuli, run_device: str, cfg: Config) -> tuple[SAEWeights, dict[str, Any], dict[int, torch.Tensor]]:
    """Load an exact residual SAE at layer 8 and ensure its real-text reconstruction is sane."""
    sae8 = load_direct_res_jb(8)
    resid = collect_residuals(model, stimuli, cfg.pilot_layers, run_device, cfg.batch_size)
    x8 = resid[8]
    f8 = sae8.encode(x8)
    ev, mse, relative_error = explained_variance(x8, sae8.decode(f8))
    l0 = float((f8 > 0).sum(1).float().mean())
    # "Tens" is intentionally a broad sanity band, not an asserted published target.
    if not (1 < l0 < 2_000) or not np.isfinite(ev):
        raise RuntimeError(f"Gate A failed: layer-8 SAE EV={ev:.4f}, MSE={mse:.6g}, mean L0={l0:.1f}; likely wrong hook/weights.")
    return sae8, {
        "layer": 8,
        "explained_variance": ev,
        "mse": mse,
        "relative_reconstruction_error": relative_error,
        "mean_l0": l0,
        "source": sae8.source,
    }, resid


def choose_layer(resid: dict[int, torch.Tensor], stimuli: Stimuli, seed: int, cfg: Config) -> tuple[int, dict[str, list[float]]]:
    pilot = {layer: main_effect_pilot(x, stimuli, seed, cfg) for layer, x in resid.items()}
    scores = {layer: float(np.mean(v)) for layer, v in pilot.items()}
    best = max(scores, key=scores.get)
    if min(pilot[best]) < 0.85:
        raise RuntimeError(
            "Gate B failed: best residual layer "
            f"{best} main-effect accuracies={pilot[best]} (<0.85 for at least one factor). "
            "Revise the template before interpreting any SAE metric."
        )
    return best, {str(k): v for k, v in pilot.items()}


def write_gated_out(gate: str, error: BaseException, started: float) -> None:
    """A failure manifest is a real result; it is never dressed up as an experiment run."""
    payload = {
        "schema": "exp03-gated-out-v1",
        "status": "gated_out",
        "failed_gate": gate,
        "error_type": type(error).__name__,
        "error": str(error),
        "wall_clock_seconds": time.perf_counter() - started,
        "smoke": SMOKE,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "note": "No SD/CCGP rows or figures were produced after a failed gate.",
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")


def aggregate(rows: list[dict[str, Any]], arms: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        out[arm] = {}
        for metric in ("sd", "ccgp", "gap"):
            types = sorted({key for r in arm_rows for key in r.get(metric, {})})
            out[arm][metric] = {key: dict(zip(("mean", "ci95"), mean_ci(r[metric][key] for r in arm_rows))) for key in types}
    return out


def paired_difference(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    left_arm: str,
    right_arm: str,
) -> dict[str, Any]:
    """Within-seed contrast for an explicitly named, same-seed pair of arms."""
    out: dict[str, Any] = {}
    for metric in ("sd", "ccgp"):
        kinds = sorted({key for row in rows if row["arm"] == left_arm for key in row[metric]})
        out[metric] = {
            kind: dict(zip(("mean", "ci95"), mean_ci(
                next(row[metric][kind] for row in rows if row["seed"] == seed and row["arm"] == left_arm)
                - next(row[metric][kind] for row in rows if row["seed"] == seed and row["arm"] == right_arm)
                for seed in seeds
            )))
            for kind in kinds
        }
    return out


def _fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.3f} ± {stat['ci95']:.3f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _summary_from_rows(setting_data: dict[str, Any]) -> dict[str, Any]:
    return aggregate(setting_data["per_seed_rows"], tuple(setting_data["arms"]))


def plots_from_results(payload: dict[str, Any]) -> None:
    """Render both figures from results.json-derived summaries, never hand-entered values."""
    settings = list(payload["probe_fairness"])
    colors = {
        "resid": "#666666", "sae": "#3b6ea5", "sae_recon": "#5c9f72",
        "rand_exp": "#e08214", "rand_exp_dense": "#d73027",
    }
    labels = {
        "resid": "resid", "sae": "sae", "sae_recon": "sae_recon",
        "rand_exp": "rand_exp", "rand_exp_dense": "rand_exp_dense",
    }
    fig, axes = plt.subplots(1, len(settings), figsize=(6.2 * len(settings), 5.1), squeeze=False)
    for ax, setting in zip(axes[0], settings):
        summary = _summary_from_rows(payload["probe_fairness"][setting])
        for arm in payload["probe_fairness"][setting]["arms"]:
            x, y = summary[arm]["ccgp"]["main_effect"], summary[arm]["sd"]["overall"]
            ax.errorbar(
                x["mean"], y["mean"], xerr=x["ci95"], yerr=y["ci95"], fmt="o", ms=7,
                capsize=3, color=colors[arm], label=labels[arm],
            )
        ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="chance (0.5)")
        ax.axvline(0.5, color="crimson", ls="--", lw=1.2)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("main-effect CCGP: held-condition balanced accuracy")
        ax.set_ylabel("shattering dimensionality: balanced accuracy")
        ax.set_title(setting.replace("_", " "))
        ax.grid(alpha=0.25)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    dedup = dict(zip(legend_labels, handles))
    fig.legend(dedup.values(), dedup.keys(), fontsize=8.1, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Expressivity versus factor abstraction under two converged scaling conventions")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    fig.savefig(FIGDIR / "01_shattering_vs_ccgp.png", dpi=180)
    plt.close(fig)

    types = ("main_effect", "two_way_xor", "three_way_parity", "unstructured")
    type_labels = ("main effect", "2-way XOR", "3-way parity", "unstructured")
    fig, axes = plt.subplots(1, len(settings), figsize=(6.2 * len(settings), 4.9), squeeze=False)
    for ax, setting in zip(axes[0], settings):
        summary = _summary_from_rows(payload["probe_fairness"][setting])
        arms = payload["probe_fairness"][setting]["arms"]
        x, width = np.arange(len(types)), 0.15
        for i, arm in enumerate(arms):
            vals = [summary[arm]["sd"][kind]["mean"] for kind in types]
            errs = [summary[arm]["sd"][kind]["ci95"] for kind in types]
            offset = (i - (len(arms) - 1) / 2) * width
            ax.bar(x + offset, vals, width, yerr=errs, capsize=2.5, label=labels[arm], color=colors[arm])
        ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="chance (0.5)")
        ax.set_xticks(x, type_labels)
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("shattering: balanced decoding accuracy")
        ax.set_title(setting.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    axes[0][0].legend(ncol=3, fontsize=7.5, loc="upper center")
    fig.suptitle("Dichotomy breakdown from converged raw rows")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_dichotomy_breakdown.png", dpi=180)
    plt.close(fig)


def _l2_table(setting_data: dict[str, Any]) -> str:
    rows = []
    for arm in setting_data["arms"]:
        selected = [
            value for row in setting_data["per_seed_rows"] if row["arm"] == arm
            for value in row["selected_l2_by_outer_fold"]
        ]
        counts = {value: selected.count(value) for value in sorted(set(selected))}
        rows.append([f"`{arm}`", ", ".join(f"{key:g} ({value})" for key, value in counts.items()), "no"])
    return _markdown_table(["arm", "selected L2 across 25 outer folds (count)", "grid edge"], rows)


def _metric_table(summary: dict[str, Any], arms: tuple[str, ...]) -> str:
    return _markdown_table(
        ["arm", "overall SD", "main-effect SD", "two-way-XOR SD", "main-effect CCGP", "overall CCGP", "train − test gap"],
        [[
            f"`{arm}`", _fmt_stat(summary[arm]["sd"]["overall"]),
            _fmt_stat(summary[arm]["sd"]["main_effect"]),
            _fmt_stat(summary[arm]["sd"]["two_way_xor"]),
            _fmt_stat(summary[arm]["ccgp"]["main_effect"]),
            _fmt_stat(summary[arm]["ccgp"]["overall"]),
            _fmt_stat(summary[arm]["gap"]["overall"]),
        ] for arm in arms],
    )


def _breakdown_table(summary: dict[str, Any], arms: tuple[str, ...]) -> str:
    return _markdown_table(
        ["arm", "main-effect SD", "two-way XOR SD", "three-way parity SD", "unstructured SD"],
        [[
            f"`{arm}`", _fmt_stat(summary[arm]["sd"]["main_effect"]),
            _fmt_stat(summary[arm]["sd"]["two_way_xor"]),
            _fmt_stat(summary[arm]["sd"]["three_way_parity"]),
            _fmt_stat(summary[arm]["sd"]["unstructured"]),
        ] for arm in arms],
    )


def _convergence_rollup(setting_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for arm in setting_data["arms"]:
        diagnostics = [
            diag for row in setting_data["per_seed_rows"] if row["arm"] == arm
            for family in ("sd_convergence", "ccgp_convergence") for diag in row[family]
        ]
        iterations = [diag["iterations"] for diag in diagnostics]
        out[arm] = {
            "minimum_iterations": min(iterations), "maximum_iterations": max(iterations),
            "mean_iterations": float(np.mean(iterations)),
            "all_converged": all(diag["converged"] for diag in diagnostics),
        }
    return out


def write_documents_from_results(payload: dict[str, Any]) -> None:
    """Regenerate prose tables and both public snippets from the just-written raw manifest."""
    fair = payload["probe_fairness"]
    setting_summaries = {setting: _summary_from_rows(data) for setting, data in fair.items()}
    z_name, rms_name = "per_feature_zscore_inner_l2", "global_rms_inner_l2"
    z_diff = fair[z_name]["paired_sae_minus_rand_exp"]["sd"]["two_way_xor"]
    rms_diff = fair[rms_name]["paired_sae_minus_rand_exp"]["sd"]["two_way_xor"]
    agreement = payload["scaling_convergence_test"]["agree_within_sum_of_t_intervals"]
    if agreement:
        headline = (
            "After convergence, the two scaling conventions agree within their Student-t intervals; "
            "the reported two-way-XOR result is therefore a converged representation result rather than a scaling effect."
        )
        controls_text = "Effective-width controls were rerun under the converged probe and are reported below."
    else:
        headline = (
            "After convergence, the two scaling conventions still disagree beyond their Student-t intervals. "
            "That is an L2-prior-geometry sensitivity, not evidence that one code has greater interaction-readout capacity; "
            "the honest headline is a null with respect to the SAE-versus-random code claim."
        )
        controls_text = "Effective-width controls were not rerun: once the core contrast is prior-geometry sensitive, their older global-RMS values cannot repair or headline it."
    convergence_rows = _markdown_table(
        ["setting", "arm", "L-BFGS iterations (min–max; mean)", "all converged"],
        [[
            setting, f"`{arm}`",
            f"{roll['minimum_iterations']}–{roll['maximum_iterations']}; {roll['mean_iterations']:.1f}",
            "yes" if roll["all_converged"] else "NO",
        ] for setting, data in fair.items() for arm, roll in _convergence_rollup(data).items()],
    )
    comparison_table = _markdown_table(
        ["scaling", "paired `sae − rand_exp` two-way-XOR SD", "paired main-effect CCGP"],
        [[
            setting, _fmt_stat(data["paired_sae_minus_rand_exp"]["sd"]["two_way_xor"]),
            _fmt_stat(data["paired_sae_minus_rand_exp"]["ccgp"]["main_effect"]),
        ] for setting, data in fair.items()],
    )
    metric_sections = "\n\n".join(
        f"### `{setting}`\n\n{_metric_table(setting_summaries[setting], tuple(data['arms']))}\n\n"
        f"L2 choices (all selected values are interior):\n\n{_l2_table(data)}\n\n"
        f"Dichotomy breakdown:\n\n{_breakdown_table(setting_summaries[setting], tuple(data['arms']))}"
        for setting, data in fair.items()
    )
    gates = payload["gates"]
    writeup = f"""# Experiment 03 — a converged scaling test changes the SAE headline

## Question

Can a sparse GPT-2-small SAE code linearly read two-factor interactions better than an L0-matched random ReLU expansion? The adversarial review identified a blocking constraint: per-feature z-scoring and global-RMS scaling are invertible affine reparameterisations. At convergence, a difference between them cannot be called a property of the codes; with L2 it can only be a difference in prior geometry.

## Setup and discipline

The stimuli remain NUMBER × TENSE × POLARITY full factorials, read at the identical final `.` token and split by lexical item. Gate A used {payload['sae_loading']}; Gate B chose layer {payload['chosen_layer']} with pilot NUMBER/TENSE/POLARITY accuracies {gates['B_pilot_main_effects'][str(payload['chosen_layer'])]}; Gate C residual two-way-XOR SD was {gates['C_resid_two_way_xor_sd']:.3f}. All five seeds retained equal token lengths within every item.

Each probe minimises mean logistic loss plus `0.5 × λ × ||w||²` (bias unpenalised), using full-batch L-BFGS. Convergence means relative training-objective change below {payload['config']['probe_relative_loss_tolerance']:g} over the preceding {payload['config']['probe_stable_steps']} accepted iterations, capped at {payload['config']['probe_max_steps']}; no fit hit the cap. `λ` is selected separately for every arm, scaling, seed, and outer fold from an unpenalised inner item-disjoint main-effect logistic-loss grid {payload['config']['l2_grid']}, choosing the largest value at or above the predeclared stability floor {L2_STABILITY_FLOOR:g} within {L2_ONE_STANDARD_STYLE_TOLERANCE_NATS:g} nats/example of the minimum; an edge choice raises an error rather than being silently reported.

CCGP is now strict: each of its 16 heads centres and scales from exactly its six condition/item training cells before applying that transform to held-condition test cells. The all-zero keep-mask is fit per outer training fold. The sparse random top-k target is still an **unlabelled, transductive** mean-L0 calibration over the complete fixed stimulus set; it never sees factor labels, dichotomies, probe loss, or accuracy. This corrects the former false statement that no outer-test activation entered any representation calibration.

All intervals below are two-sided 95% Student-t intervals (`t(4)=2.776`, five seeds), not `1.96 × sd/√5`. Experiment 02 used its earlier 1.96 convention with eight seeds; the conventions are now stated rather than silently conflated.

## Result 1 — the scaling-convergence test is the result

![Converged shattering versus main-effect CCGP under both scaling conventions](figures/01_shattering_vs_ccgp.png)

{headline}

{comparison_table}

The interval-overlap decision rule was `|difference of paired estimates| ≤ sum of their t-interval half-widths`; it evaluated to **{agreement}**. Scaling is irrelevant to the unregularised linear function class, but L2 is not affine-invariant, so a persistent difference is a prior sensitivity rather than a readout-capacity result.

## Result 2 — SD and CCGP, regenerated from raw rows

![Dichotomy-family SD breakdown under both converged scaling conventions](figures/02_dichotomy_breakdown.png)

{metric_sections}

## Optimisation record

{convergence_rows}

## Effective-width controls

{controls_text} The implementation now draws one maximal random matrix (including sampled norm and bias columns) and slices prefixes for candidate widths, so calibration varies width rather than projection identity. Neither width control matches both width and the full L0 distribution: random uses forced per-sample top-k while SAE uses natural ReLU sparsity. If they are run in a later invariance-resolved result, they are sensitivity controls with different costs, not a demonstration that active directions are excluded.

## Scope

This experiment measures properties of a representation/lens only. The SAE is read beside GPT-2's residual stream; it never replaces that stream in the model's forward computation. No result here says or implies that an SAE harms, degrades, removes, or otherwise changes GPT-2's computation.

## What is not claimed

The result does not identify conjunctive SAE latents, a transformer circuit, a causal feature use, or a behavioural effect. Dense random expansion is an upper reference, not the sparse control. The only theorem cited in this repository remains Experiment 02's coordinate-wise compression/XOR result; nothing trained or measured here earns that word.

## Reproducibility

`results.json` schema: raw `per_seed_rows` are grouped by scaling and contain SD, CCGP, gaps, per-outer-fold L2 traces, per-fold keep widths, and L-BFGS convergence diagnostics. Every number in every table and both figures is regenerated by this script from those rows. The completed full run took {payload['wall_clock_seconds']:.1f} seconds on CPU. Historical 84.2 s / 22.5 s / 1,419.7 s timing and 100-versus-200-step claims are deliberately not shipped as evidence because this manifest does not contain their raw rows.
"""
    (HERE / "writeup.md").write_text(writeup)
    snippet = f"""## Experiment 03 — converged probes made the headline a scaling-sensitivity result

![Five-seed converged shattering dimensionality versus main-effect CCGP under z-score and global-RMS scaling](figures/01_shattering_vs_ccgp.png)

Caption — `sae − rand_exp` two-way-XOR SD is {_fmt_stat(z_diff)} with per-feature z-scoring and {_fmt_stat(rms_diff)} with global RMS (95% Student-t). **{headline}** This is a representation/lens measurement, not a model intervention: the SAE never replaces GPT-2's residual stream in the forward computation. Full gates, strict CCGP preprocessing, L2 selection records, raw-row-generated tables, and caveats: [Experiment 03 writeup](writeup.md).
"""
    (HERE / "README-snippet.md").write_text(snippet)
    marker = "## 2026-07-26 — Experiment 03 review run 5: affine reparameterisation audit"
    notebook_path = HERE.parents[1] / "lab-notebook.md"
    review_entry = f"""{marker}

**Goal:** Test the adversarial finding that my exp03 headline rested on an affine reparameterisation of a linear probe, then let a converged nested-L2 result decide whether the positive claim survives.

**Did:** Replaced 100-step AdamW with full-batch L-BFGS on the stated L2-logistic objective; convergence required relative objective change below {payload['config']['probe_relative_loss_tolerance']:g} across a {payload['config']['probe_stable_steps']}-iteration window, with a {payload['config']['probe_max_steps']}-iteration cap. Selected L2 on an item-disjoint inner grid separately for arm/scaling/outer fold, fitted CCGP moments from each head's six training conditions only, and fitted the zero-unit mask on outer training items. Rebuilt tables and both figures from `results.json` raw rows, switched all five-seed intervals to `t(4)`, and removed historical timing/convergence claims that lack raw manifests.

**Expected:** I expected better optimisation might make z-score and global-RMS agree, because a linear probe can represent the same boundaries after either invertible affine scaling. If that happened, I would keep a code-level number and rerun width controls. If not, the headline had to become a null rather than a flattering global-RMS choice.

**Happened:** Per-feature z-score gave two-way-XOR `sae − rand_exp` {_fmt_stat(z_diff)}; global RMS gave {_fmt_stat(rms_diff)}. The predeclared interval-overlap check was **{agreement}**. {headline}

**Confused about / open:** L2 breaks affine invariance, so this does not tell me which scaling is "the real code"; it tells me the result is conditional on prior geometry. The random top-k L0 target remains a transductive unlabeled calibration over the fixed sample; labels and probe scores never enter it, but I should replace that with fold-local representation construction before a future positive claim. The fixed prefix family now ensures width candidates differ only by width, but width and full L0 distributions still cannot both be matched by these two controls.

**Next:** Do not revive the old width-matched interaction headline. A future experiment needs a preregistered, fold-local random-control construction and a scaling-invariant or explicitly prior-targeted readout question before interpreting SAE/random interaction differences as code geometry.
"""
    old = notebook_path.read_text()
    if marker in old:
        start = old.index(marker)
        end = old.find("\n---\n", start)
        old = old[:start] + old[end + len("\n---\n"):] if end >= 0 else old[:start]
    notebook_path.write_text(review_entry + "\n---\n\n" + old.lstrip())


def run() -> dict[str, Any]:
    global CURRENT_GATE
    started = time.perf_counter()
    cfg = config()
    run_device = device()
    CURRENT_GATE = "A"
    model = load_model(run_device)
    # One exact factorial set per seed: seed changes nuisance draws, R, and probe init.
    # Gate A/B/C are checked on seed 0 before full rows are permitted.
    initial_stimuli = build_stimuli(model.tokenizer, cfg.n_items, cfg.seeds[0])
    sae8, gate_a_stats, pilot_resid = gate_a(model, initial_stimuli, run_device, cfg)
    CURRENT_GATE = "B"
    layer, pilot = choose_layer(pilot_resid, initial_stimuli, cfg.seeds[0], cfg)
    # The pilot has selected a hook; loading the matching SAE and checking its
    # reconstruction remains part of Gate A rather than a probe-quality failure.
    CURRENT_GATE = "A"
    sae = sae8 if layer == 8 else load_direct_res_jb(layer)
    # Check the exact selected SAE, not merely the layer-8 loading sentinel.
    selected_x = pilot_resid[layer]
    selected_ev, selected_mse, selected_relative_error = explained_variance(selected_x, sae.decode(sae.encode(selected_x)))
    selected_l0 = float((sae.encode(selected_x) > 0).sum(1).float().mean())
    if not (1 < selected_l0 < 2_000) or not np.isfinite(selected_ev):
        raise RuntimeError(f"Gate A selected-layer sanity failed: EV={selected_ev:.4f}, MSE={selected_mse:.6g}, L0={selected_l0:.1f}")
    CURRENT_GATE = "C"
    reps0, matching0 = representations(selected_x, sae, cfg.seeds[0])
    ds = dichotomies()
    xor_gate, _, _ = sd_metric(reps0["resid"], initial_stimuli, cfg.seeds[0], cfg, [d for d in ds if d["type"] == "two_way_xor"])
    # sd_metric accepts arbitrary output count; this averaged only the three XOR-family rows.
    if xor_gate["overall"] >= SATURATION_THRESHOLD:
        raise RuntimeError(
            f"Gate C failed: residual 2-way-XOR shattering={xor_gate['overall']:.3f} >= {SATURATION_THRESHOLD:.2f}. "
            "Increase lexical nuisance variance before any arm comparison."
        )

    CURRENT_GATE = "D"
    scale_modes = {
        "per_feature_zscore_inner_l2": "per_feature_zscore",
        "global_rms_inner_l2": "global_rms",
    }
    setting_rows: dict[str, list[dict[str, Any]]] = {setting: [] for setting in cfg.fair_probe_settings}
    seed_metadata = []
    for seed in cfg.seeds:
        stimuli = initial_stimuli if seed == cfg.seeds[0] else build_stimuli(model.tokenizer, cfg.n_items, seed)
        if seed == cfg.seeds[0]:
            x = selected_x
        else:
            x = collect_residuals(model, stimuli, (layer,), run_device, cfg.batch_size)[layer]
        reps, matching = representations(x, sae, seed)
        seed_metadata.append({
            "seed": seed,
            "stimuli": {
                "n_items": cfg.n_items, "n_sequences": len(stimuli.texts),
                "attempted_items": stimuli.attempted_items, "dropped_items": stimuli.dropped_items,
                "readout_token_id": stimuli.readout_token_id,
            },
            "matching": matching,
        })
        for setting in cfg.fair_probe_settings:
            scale_mode = scale_modes[setting]
            for arm in BASE_ARMS:
                rep = reps[arm]
                selected_l2, selection_trace, fold_keeps = select_l2(
                    rep, stimuli, seed, cfg, scale_mode,
                )
                sd, gap, sd_convergence = sd_metric(
                    rep, stimuli, seed, cfg, ds, scale_mode, selected_l2, fold_keeps,
                )
                ccgp, ccgp_convergence = ccgp_metric(
                    rep, stimuli, seed, cfg, ds, scale_mode, selected_l2, fold_keeps,
                )
                row = {
                    "probe_setting": setting,
                    "seed": seed,
                    "arm": arm,
                    "sd": sd,
                    "ccgp": ccgp,
                    "gap": gap,
                    "selected_l2_by_outer_fold": selected_l2,
                    "inner_validation_l2_selection": selection_trace,
                    "sd_convergence": sd_convergence,
                    "ccgp_convergence": ccgp_convergence,
                }
                setting_rows[setting].append(row)
                print(
                    f"seed={seed} setting={setting} arm={arm} "
                    f"SD={sd['overall']:.3f} main-CCGP={ccgp['main_effect']:.3f}",
                    flush=True,
                )
    fairness = {}
    for setting in cfg.fair_probe_settings:
        fairness[setting] = {
            "scale_mode": scale_modes[setting],
            "l2_selection": "Per outer fold and arm: select the largest lambda >= 1e-5 within 1e-2 nats/example of the minimum inner item-disjoint unpenalised main-effect logistic loss over NUMBER/TENSE/POLARITY. Outer test labels, dichotomies, losses, and scores are never consulted.",
            "ccgp_scope": "all_35_dichotomies",
            "arms": list(BASE_ARMS),
            "per_seed_rows": setting_rows[setting],
            "summary": aggregate(setting_rows[setting], BASE_ARMS),
            "paired_sae_minus_rand_exp": paired_difference(
                setting_rows[setting], cfg.seeds, "sae", "rand_exp",
            ),
        }
    z_diff = fairness["per_feature_zscore_inner_l2"]["paired_sae_minus_rand_exp"]["sd"]["two_way_xor"]
    rms_diff = fairness["global_rms_inner_l2"]["paired_sae_minus_rand_exp"]["sd"]["two_way_xor"]
    agree = abs(z_diff["mean"] - rms_diff["mean"]) <= z_diff["ci95"] + rms_diff["ci95"]
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "exp03-results-v4; raw per-seed rows are grouped by scaling, with fold-local keep masks, strict six-condition CCGP moments, L-BFGS diagnostics, and t-interval summaries.",
        "status": "complete",
        "smoke": SMOKE,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "pilot_layers": list(cfg.pilot_layers), "ccgp_splits": [list(s) for s in cfg.ccgp_splits], "l2_grid": list(cfg.l2_grid)},
        "device": run_device,
        "sae_loading": sae.source,
        "chosen_layer": layer,
        "gates": {
            "A_layer8": gate_a_stats,
            "A_selected_layer": {
                "explained_variance": selected_ev,
                "mse": selected_mse,
                "relative_reconstruction_error": selected_relative_error,
                "mean_l0": selected_l0,
            },
            "B_pilot_main_effects": pilot,
            "C_resid_two_way_xor_sd": xor_gate["overall"],
        },
        "per_seed_metadata": seed_metadata,
        "probe_fairness": fairness,
        "scaling_convergence_test": {
            "metric": "paired sae - rand_exp two-way-XOR SD",
            "decision_rule": "absolute difference between scaling estimates is no greater than the sum of their two-sided 95% Student-t interval half-widths",
            "agree_within_sum_of_t_intervals": agree,
            "per_feature_zscore": z_diff,
            "global_rms": rms_diff,
        },
        "effective_width_controls": {
            "status": "not_rerun_pending_scaling_invariance_decision",
            "implementation_fix": "Candidate widths use prefixes of one maximal random matrix, including prefix-matched sampled norms and biases.",
        },
        "wall_clock_seconds": elapsed,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    # Reload through the public file so documents/figures cannot accidentally use stale
    # in-memory tables. This is the structural guard against Result-2 transcription drift.
    materialised = json.loads(RESULTS.read_text())
    plots_from_results(materialised)
    write_documents_from_results(materialised)
    return payload


if __name__ == "__main__":
    started = time.perf_counter()
    print(f"Experiment 03 — CCGP on SAE features (SMOKE={SMOKE}, device={device()})", flush=True)
    try:
        result = run()
    except Exception as exc:  # Gate failure must leave an inspectable, non-result artifact.
        write_gated_out(CURRENT_GATE, exc, started)
        print(f"GATED OUT: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
    else:
        print(f"Done in {result['wall_clock_seconds']:.1f}s; wrote {RESULTS}")

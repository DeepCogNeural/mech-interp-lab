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
hand-rolled torch with training-split-only standardisation and strong L2.  No result is
claimed until the corresponding full command has completed.  The completed controls
also test whether per-feature z-scoring unfairly amplifies rare sparse units: L2 is
chosen inside each outer training split, and the primary probe uses one global RMS
scale rather than one scale per feature.

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


@dataclass(frozen=True)
class Config:
    n_items: int
    seeds: tuple[int, ...]
    n_folds: int
    pilot_layers: tuple[int, ...]
    probe_steps: int
    ccgp_splits: tuple[tuple[int, int], ...]
    batch_size: int
    legacy_weight_decay: float
    weight_decay_grid: tuple[float, ...]
    fair_probe_settings: tuple[str, ...]


def config() -> Config:
    if SMOKE:
        # Fast plumbing check only: it deliberately reports that CCGP used 2/16 splits.
        return Config(
            n_items=24,
            seeds=(0, 1),
            n_folds=2,
            pilot_layers=(7, 8),
            probe_steps=40,
            ccgp_splits=((0, 0), (1, 1)),
            batch_size=16,
            legacy_weight_decay=0.08,
            weight_decay_grid=(0.03, 0.08, 0.2),
            fair_probe_settings=("per_feature_zscore_inner_l2", "global_rms_inner_l2"),
        )
    return Config(
        n_items=96,
        seeds=(0, 1, 2, 3, 4),
        n_folds=5,
        pilot_layers=(6, 7, 8, 9),
        probe_steps=100,
        ccgp_splits=tuple(itertools.product(range(4), range(4))),
        batch_size=32,
        legacy_weight_decay=0.08,
        weight_decay_grid=(0.01, 0.03, 0.08, 0.2, 0.5),
        fair_probe_settings=("per_feature_zscore_inner_l2", "global_rms_inner_l2"),
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
    return float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0


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
    """Drop units exactly zero over this experiment's whole sample, as specified."""
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
    """Legacy fixed-L2 preprocessing, retained for gates and sensitivity reporting."""
    return scale_features(x_train, x_test, "per_feature_zscore")


def fit_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    probe_seed: int,
    steps: int,
    weight_decay: float,
    train_mask: torch.Tensor | None = None,
    return_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, float]:
    """One vectorised multi-output logistic probe; optional mask supports CCGP heads."""
    torch.manual_seed(probe_seed)
    # CPU is used deliberately for numerical reproducibility of the probes, even when the
    # transformer ran on MPS.  The model and SAE are the memory-heavy part of this PoC.
    x_train, x_eval = x_train.float().cpu(), x_eval.float().cpu()
    y_train, y_eval = y_train.float().cpu(), y_eval.float().cpu()
    head = nn.Linear(x_train.shape[1], y_train.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(x_train), y_train, reduction="none")
        if train_mask is not None:
            loss = (loss * train_mask.float().cpu()).sum() / train_mask.sum().clamp_min(1)
        else:
            loss = loss.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_logits = head(x_train)
        final_loss = F.binary_cross_entropy_with_logits(train_logits, y_train, reduction="none")
        if train_mask is not None:
            final_loss = (final_loss * train_mask.float().cpu()).sum() / train_mask.sum().clamp_min(1)
        else:
            final_loss = final_loss.mean()
        result = (balanced_accuracy(head(x_eval), y_eval), balanced_accuracy(train_logits, y_train))
    if return_loss:
        return (*result, float(final_loss))
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
        acc, _ = fit_probe(
            tr, stimuli.factors[train], te, stimuli.factors[test], seed + fold_index,
            cfg.probe_steps, cfg.legacy_weight_decay,
        )
        values.append(acc.numpy())
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


def select_weight_decays(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    scale_mode: str,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Nested L2 selection on the three main effects; outer test items are untouched."""
    rep, _ = prepare_features(rep)
    selected, records = [], []
    for outer_fold, (outer_train, _) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        inner_train, inner_valid = inner_validation_split(outer_train, stimuli.item_ids, seed, outer_fold)
        tr, va = scale_features(rep[inner_train], rep[inner_valid], scale_mode)
        scores = []
        for candidate_i, weight_decay in enumerate(cfg.weight_decay_grid):
            acc, _ = fit_probe(
                tr, stimuli.factors[inner_train], va, stimuli.factors[inner_valid],
                seed + 30_000 + 100 * outer_fold + candidate_i,
                cfg.probe_steps,
                weight_decay,
            )
            scores.append(float(acc.mean()))
        best_i = int(np.argmax(scores))
        selected.append(float(cfg.weight_decay_grid[best_i]))
        records.append({
            "outer_fold": outer_fold,
            "candidate_weight_decay": list(cfg.weight_decay_grid),
            "inner_validation_main_effect_accuracy": scores,
            "selected_weight_decay": float(cfg.weight_decay_grid[best_i]),
        })
    return selected, records


def _fold_weight_decay(
    fold_weight_decays: list[float] | None,
    fold_index: int,
    cfg: Config,
) -> float:
    return cfg.legacy_weight_decay if fold_weight_decays is None else fold_weight_decays[fold_index]


def sd_metric(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    ds: list[dict[str, Any]],
    scale_mode: str = "per_feature_zscore",
    fold_weight_decays: list[float] | None = None,
    steps: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Five item-disjoint folds, all 35 dichotomies trained as one multi-output probe."""
    rep, _ = prepare_features(rep)
    labels_by_condition = torch.stack([d["labels"] for d in ds], dim=1)
    labels = labels_by_condition[stimuli.condition_ids]
    test_accs, train_accs = [], []
    steps = cfg.probe_steps if steps is None else steps
    for fold_index, (train, test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        tr, te = scale_features(rep[train], rep[test], scale_mode)
        acc, train_acc = fit_probe(
            tr, labels[train], te, labels[test], seed + 100 * fold_index, steps,
            _fold_weight_decay(fold_weight_decays, fold_index, cfg),
        )
        test_accs.append(acc.numpy())
        train_accs.append(train_acc.numpy())
    test_mean = np.mean(test_accs, axis=0)
    train_mean = np.mean(train_accs, axis=0)
    by_type: dict[str, list[float]] = {}
    gap: dict[str, list[float]] = {}
    for i, d in enumerate(ds):
        by_type.setdefault(d["type"], []).append(float(test_mean[i]))
        gap.setdefault(d["type"], []).append(float(train_mean[i] - test_mean[i]))
    by_type["overall"] = list(test_mean)
    gap["overall"] = list(train_mean - test_mean)
    return ({k: float(np.mean(v)) for k, v in by_type.items()}, {k: float(np.mean(v)) for k, v in gap.items()})


def ccgp_metric(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    ds: list[dict[str, Any]],
    scale_mode: str = "per_feature_zscore",
    fold_weight_decays: list[float] | None = None,
    steps: int | None = None,
) -> dict[str, float]:
    """CCGP with all (or smoke-subset) 4x4 held-condition splits, item-disjoint throughout.

    The 16 heads for one dichotomy share an nn.Linear.  Its loss mask gives each head its
    own six training conditions; all standardisation statistics come only from training
    lexical items.  This keeps the implementation vectorised without allowing a lexical
    nuisance item into both a head's train and test examples.  The shared feature matrix
    is deliberately *not* repeated once per head: a [examples, heads] masked loss is
    exactly equivalent and avoids a 16-fold CPU/memory multiplier.
    """
    rep, _ = prepare_features(rep)
    per_type: dict[str, list[float]] = {}
    steps = cfg.probe_steps if steps is None else steps
    for d_index, d in enumerate(ds):
        labels = d["labels"][stimuli.condition_ids]
        pos = sorted(d["positive"])
        neg = sorted(d["negative"])
        split_scores: list[float] = []
        for fold_index, (item_train, item_test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
            # Standardisation uses lexical training items only, never an item evaluated by
            # this fold.  The held conditions are unlabelled for that head but still belong
            # to the training item universe, as the global item-split rule requires.
            tr_all, te_all = scale_features(rep[item_train], rep[item_test], scale_mode)
            condition_train = stimuli.condition_ids[item_train]
            condition_test = stimuli.condition_ids[item_test]
            masks, eval_blocks, eval_labels = [], [], []
            for held_pos_i, held_neg_i in cfg.ccgp_splits:
                held_pos, held_neg = pos[held_pos_i], neg[held_neg_i]
                train_mask = (condition_train != held_pos) & (condition_train != held_neg)
                eval_mask = (condition_test == held_pos) | (condition_test == held_neg)
                # This happens only with a malformed factorial stimulus set.
                if int(eval_mask.sum()) == 0:
                    raise RuntimeError("CCGP held-condition split has no item-disjoint evaluation data")
                masks.append(train_mask)
                eval_blocks.append(te_all[eval_mask])
                eval_labels.append(labels[item_test][eval_mask])
            # Heads need a common evaluation tensor.  Each factorial held pair has exactly
            # two cells per test item, so all blocks have the same length by construction.
            eval_x = torch.stack(eval_blocks, dim=1)  # [examples, heads, features]
            eval_y = torch.stack(eval_labels, dim=1)
            # Train all 16 heads together.  Every head sees the same item-disjoint
            # lexical training universe but its loss includes only its own six cells.
            n_heads = len(cfg.ccgp_splits)
            train_y = labels[item_train].unsqueeze(1).expand(-1, n_heads)
            head_masks = torch.stack(masks, dim=1)
            # A dedicated evaluation pass is simpler than abusing fit_probe's all-head
            # accuracy API; fit the parameters here with the exact masked objective.
            torch.manual_seed(seed + 10_000 * d_index + fold_index)
            head = nn.Linear(tr_all.shape[1], n_heads)
            opt = torch.optim.AdamW(
                head.parameters(), lr=0.03,
                weight_decay=_fold_weight_decay(fold_weight_decays, fold_index, cfg),
            )
            for _ in range(steps):
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(head(tr_all), train_y, reduction="none")
                loss = (loss * head_masks.float()).sum() / head_masks.sum().clamp_min(1)
                loss.backward()
                opt.step()
            with torch.no_grad():
                # The h-th block in eval_x is scored by output h only.
                scores = torch.stack([head(eval_x[:, h])[:, h] for h in range(n_heads)], dim=1)
                acc = balanced_accuracy(scores, eval_y).mean().item()
            split_scores.append(acc)
        per_type.setdefault(d["type"], []).append(float(np.mean(split_scores)))
    per_type["overall"] = [v for group in per_type.values() for v in group]
    return {k: float(np.mean(v)) for k, v in per_type.items()}


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


def random_expansion_at_width(
    x: torch.Tensor,
    sae: SAEWeights,
    seed: int,
    target_l0: int,
    n_columns: int,
) -> torch.Tensor:
    """Random sparse code at a chosen nominal width for the effective-width control.

    The widened arm keeps the original control's residual centring and top-k rule.  It
    samples SAE encoder-column norms and encoder biases with replacement, so widening
    preserves their empirical marginal distributions instead of copying a privileged
    block of SAE columns.  Width is calibrated only against the unlabeled count of
    nonzero-on-any-sample units; no probe score enters the calibration.
    """
    if n_columns < target_l0:
        raise ValueError(f"Random-expansion width {n_columns} is below top-k {target_l0}")
    g = torch.Generator(device="cpu").manual_seed(seed + 500_000)
    source = torch.randint(sae.W_enc.shape[1], (n_columns,), generator=g)
    norms = sae.W_enc.norm(dim=0)[source]
    biases = sae.b_enc[source]
    # Draw column-major: a width-N candidate is then an exact prefix of a larger-width
    # candidate under the same seed.  That makes the unlabeled width calibration stable
    # instead of changing every existing random direction when N changes shape.
    R = torch.randn((n_columns, sae.W_enc.shape[0]), generator=g).T.contiguous()
    R *= norms.unsqueeze(0) / R.norm(dim=0, keepdim=True).clamp_min(1e-8)
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
    tried = 0
    best_rep: torch.Tensor | None = None
    best_columns = initial
    best_width = -1
    for multiplier in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 5.25, 5.5, 5.75, 6.0, 6.5):
        candidate = max(target_l0, int(round(initial * multiplier)))
        rep = random_expansion_at_width(x, sae, seed, target_l0, candidate)
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


def representations(x: torch.Tensor, sae: SAEWeights, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    f = sae.encode(x)
    rand_topk, rand_dense, target_l0 = random_expansion(x, sae, seed)
    sae_width = int(prepare_features(f)[0].shape[1])
    rand_width = int(prepare_features(rand_topk)[0].shape[1])
    rand_widened, random_width_control = widened_random_to_effective_width(
        x, sae, seed, target_l0, sae_width, rand_width,
    )
    sae_narrowed, sae_width_control = narrow_sae_to_effective_width(f, rand_width, seed)
    reps = {
        "resid": x,
        "sae": f,
        "sae_recon": sae.decode(f),
        "rand_exp": rand_topk,
        "rand_exp_dense": rand_dense,
        "rand_exp_width_matched": rand_widened,
        "sae_width_matched": sae_narrowed,
    }
    l0 = {name: float((rep > 0).sum(1).float().mean()) for name, rep in reps.items()}
    widths = {name: int(prepare_features(rep)[0].shape[1]) for name, rep in reps.items()}
    return reps, {
        "target_l0": target_l0,
        "mean_l0": l0,
        "surviving_width": widths,
        "rand_dense_reference_l0": l0["rand_exp_dense"],
        "effective_width_controls": {
            "random_widened_to_sae": {
                **random_width_control,
                "mean_l0": l0["rand_exp_width_matched"],
            },
            "sae_narrowed_to_random": {
                **sae_width_control,
                "mean_l0": l0["sae_width_matched"],
            },
        },
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


def convergence_check(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    scale_mode: str,
    selected_weight_decay: float,
) -> dict[str, float]:
    """Compare 100 and 200 optimizer steps on an inner validation split, never test items."""
    rep, _ = prepare_features(rep)
    outer_train, _ = folds(stimuli.item_ids, cfg.n_folds, seed)[0]
    inner_train, inner_valid = inner_validation_split(outer_train, stimuli.item_ids, seed, 0)
    tr, va = scale_features(rep[inner_train], rep[inner_valid], scale_mode)
    acc_100, _, loss_100 = fit_probe(
        tr, stimuli.factors[inner_train], va, stimuli.factors[inner_valid], seed + 60_000,
        cfg.probe_steps, selected_weight_decay, return_loss=True,
    )
    acc_200, _, loss_200 = fit_probe(
        tr, stimuli.factors[inner_train], va, stimuli.factors[inner_valid], seed + 60_000,
        2 * cfg.probe_steps, selected_weight_decay, return_loss=True,
    )
    return {
        "inner_validation_main_effect_100_steps": float(acc_100.mean()),
        "inner_validation_main_effect_200_steps": float(acc_200.mean()),
        "train_bce_100_steps": loss_100,
        "train_bce_200_steps": loss_200,
    }


def summarise_convergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        arm: {
            key: dict(zip(("mean", "ci95"), mean_ci(
                row["convergence"][key] for row in rows if row["arm"] == arm
            )))
            for key in rows[0]["convergence"]
        }
        for arm in sorted({row["arm"] for row in rows})
    }


def legacy_fixed_standardise_reference() -> dict[str, Any] | None:
    """Carry the actual pre-control full run forward as a labelled sensitivity baseline."""
    if not RESULTS.exists():
        return None
    previous = json.loads(RESULTS.read_text())
    if previous.get("schema", "").startswith("exp03-results-v1") and previous.get("status") == "complete":
        return {
            "description": "Completed pre-control run: per-feature z-score, fixed AdamW weight decay 0.08, 100 steps.",
            "per_seed_rows": previous["per_seed_rows"],
            "summary": previous["summary"],
            "paired_sae_minus_rand_exp": previous["paired_sae_minus_rand_exp"],
            "wall_clock_seconds": previous["wall_clock_seconds"],
        }
    return previous.get("legacy_fixed_standardise")


def plots(summary: dict[str, Any], smoke: bool) -> None:
    arms = ALL_ARMS
    colors = {
        "resid": "#666666", "sae": "#3b6ea5", "sae_recon": "#5c9f72",
        "rand_exp": "#e08214", "rand_exp_dense": "#d73027",
        "rand_exp_width_matched": "#9b59b6", "sae_width_matched": "#168aad",
    }
    labels = {
        "resid": "resid", "sae": "sae", "sae_recon": "sae_recon",
        "rand_exp": "rand_exp", "rand_exp_dense": "rand_exp_dense",
        "rand_exp_width_matched": "rand_exp (width-matched)",
        "sae_width_matched": "sae (width-matched)",
    }
    # The full panel preserves the chance lines; the zoomed panel makes the observed
    # 0.80--0.94 separation readable instead of visually collapsing it near the corner.
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(12.2, 5.25), gridspec_kw={"width_ratios": (1.0, 1.12)})
    for arm in arms:
        x = summary[arm]["ccgp"]["main_effect"]
        y = summary[arm]["sd"]["overall"]
        for panel in (ax, zoom):
            panel.errorbar(
                x["mean"], y["mean"], xerr=x["ci95"], yerr=y["ci95"], fmt="o", ms=7,
                capsize=3, color=colors[arm], label=labels[arm],
            )
    for panel in (ax, zoom):
        panel.axhline(0.5, color="crimson", ls="--", lw=1.2, label="chance (0.5)")
        panel.axvline(0.5, color="crimson", ls="--", lw=1.2)
        panel.grid(alpha=0.25)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    zoom.set_xlim(0.89, 0.97)
    zoom.set_ylim(0.68, 0.96)
    ax.set_xlabel("main-effect CCGP: held-condition balanced accuracy")
    ax.set_ylabel("shattering dimensionality: balanced accuracy")
    zoom.set_xlabel("main-effect CCGP: zoomed")
    zoom.set_ylabel("shattering: zoomed")
    suffix = "SMOKE subset" if smoke else "5 seeds; 95% CI"
    ax.set_title("Full scale (chance retained)")
    zoom.set_title("Observed-result zoom")
    fig.suptitle(f"Expressivity versus factor abstraction: SAE and matched random expansion\n({suffix}; global-RMS probe)")
    handles, legend_labels = zoom.get_legend_handles_labels()
    dedup = dict(zip(legend_labels, handles))
    fig.legend(dedup.values(), dedup.keys(), fontsize=8.1, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.91))
    fig.savefig(FIGDIR / "01_shattering_vs_ccgp.png", dpi=180)
    plt.close(fig)

    types = ("main_effect", "two_way_xor", "three_way_parity", "unstructured")
    type_labels = ("main effect", "2-way XOR", "3-way parity", "unstructured")
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    x = np.arange(len(types))
    width = 0.115
    for i, arm in enumerate(arms):
        vals = [summary[arm]["sd"][kind]["mean"] for kind in types]
        errs = [summary[arm]["sd"][kind]["ci95"] for kind in types]
        offset = (i - (len(arms) - 1) / 2) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=2.5, label=labels[arm], color=colors[arm])
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="chance (0.5)")
    ax.set_xticks(x, type_labels)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("shattering: balanced decoding accuracy")
    ax.set_title("Dichotomy breakdown: base-factor enumeration versus parity-family interactions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_dichotomy_breakdown.png", dpi=180)
    plt.close(fig)


def run() -> dict[str, Any]:
    global CURRENT_GATE
    started = time.perf_counter()
    legacy_reference = legacy_fixed_standardise_reference()
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
    xor_gate, _ = sd_metric(reps0["resid"], initial_stimuli, cfg.seeds[0], cfg, [d for d in ds if d["type"] == "two_way_xor"])
    # sd_metric accepts arbitrary output count; this averaged only the three XOR-family rows.
    if xor_gate["overall"] >= SATURATION_THRESHOLD:
        raise RuntimeError(
            f"Gate C failed: residual 2-way-XOR shattering={xor_gate['overall']:.3f} >= {SATURATION_THRESHOLD:.2f}. "
            "Increase lexical nuisance variance before any arm comparison."
        )

    CURRENT_GATE = "D"
    main_effect_ds = [d for d in ds if d["type"] == "main_effect"]
    primary_setting = "global_rms_inner_l2"
    scale_modes = {
        "per_feature_zscore_inner_l2": "per_feature_zscore",
        "global_rms_inner_l2": "global_rms",
    }
    if primary_setting not in cfg.fair_probe_settings:
        raise RuntimeError(f"Primary setting {primary_setting} is absent from config")
    setting_rows: dict[str, list[dict[str, Any]]] = {setting: [] for setting in cfg.fair_probe_settings}
    convergence_rows: list[dict[str, Any]] = []
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
            # Full CCGP is needed for the primary headline and its table.  The z-score
            # sensitivity run evaluates the required main-effect CCGP only; this keeps
            # the added fairness check inside the original CPU runtime budget.
            ccgp_ds = ds if setting == primary_setting else main_effect_ds
            # The width controls answer a distinct primary-result objection.  They do
            # not belong in the z-score sensitivity table, whose job is only to show
            # the original scaling reversal.  Keeping that scope fixed also avoids
            # multiplying a non-headline sensitivity run by two extra controls.
            arms = ALL_ARMS if setting == primary_setting else BASE_ARMS
            for arm in arms:
                rep = reps[arm]
                selected_decay, selection_trace = select_weight_decays(
                    rep, stimuli, seed, cfg, scale_mode,
                )
                sd, gap = sd_metric(
                    rep, stimuli, seed, cfg, ds, scale_mode, selected_decay,
                )
                ccgp = ccgp_metric(
                    rep, stimuli, seed, cfg, ccgp_ds, scale_mode, selected_decay,
                )
                row = {
                    "probe_setting": setting,
                    "seed": seed,
                    "arm": arm,
                    "sd": sd,
                    "ccgp": ccgp,
                    "gap": gap,
                    "selected_weight_decay_by_outer_fold": selected_decay,
                    "inner_validation_l2_selection": selection_trace,
                }
                setting_rows[setting].append(row)
                if setting == primary_setting:
                    convergence_rows.append({
                        "seed": seed,
                        "arm": arm,
                        "convergence": convergence_check(
                            rep, stimuli, seed, cfg, scale_mode, selected_decay[0],
                        ),
                    })
                print(
                    f"seed={seed} setting={setting} arm={arm} "
                    f"SD={sd['overall']:.3f} main-CCGP={ccgp['main_effect']:.3f}",
                    flush=True,
                )
    fairness = {
        setting: {
            "scale_mode": scale_modes[setting],
            "l2_selection": "Per outer fold and arm: select from the grid by inner item-disjoint validation mean accuracy over NUMBER/TENSE/POLARITY; outer test items are never consulted.",
            "ccgp_scope": "all_35_dichotomies" if setting == primary_setting else "main_effect_dichotomies_only",
            "per_seed_rows": setting_rows[setting],
            "summary": aggregate(
                setting_rows[setting], ALL_ARMS if setting == primary_setting else BASE_ARMS,
            ),
            "paired_sae_minus_rand_exp": paired_difference(
                setting_rows[setting], cfg.seeds, "sae", "rand_exp",
            ),
        }
        for setting in cfg.fair_probe_settings
    }
    fairness[primary_setting]["convergence_100_vs_200_steps"] = summarise_convergence(convergence_rows)
    summary = fairness[primary_setting]["summary"]
    sae_minus_rand = fairness[primary_setting]["paired_sae_minus_rand_exp"]
    width_matched_controls = {
        "random_widened_to_sae_width": {
            "comparison": "sae - rand_exp_width_matched",
            "matching_rule": "Calibrate random nominal columns only to the SAE surviving width after the common all-zero removal; preserve SAE encoder-norm and encoder-bias marginal distributions, residual b_dec centring, and the SAE-derived top-k L0.",
            "paired_difference": paired_difference(
                setting_rows[primary_setting], cfg.seeds, "sae", "rand_exp_width_matched",
            ),
        },
        "sae_narrowed_to_random_width": {
            "comparison": "sae_width_matched - rand_exp",
            "matching_rule": "Uniformly sample from SAE latents that survived the common all-zero removal until their count equals rand_exp's surviving width for that seed.",
            "paired_difference": paired_difference(
                setting_rows[primary_setting], cfg.seeds, "sae_width_matched", "rand_exp",
            ),
        },
    }
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "exp03-results-v3; effective_width_controls add a widened random arm and narrowed SAE arm under the global-RMS primary. Per-seed rows map dichotomy type to balanced accuracy; ci95=1.96*sample_sd/sqrt(n).",
        "status": "complete",
        "smoke": SMOKE,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "pilot_layers": list(cfg.pilot_layers), "ccgp_splits": [list(s) for s in cfg.ccgp_splits]},
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
        "legacy_fixed_standardise": legacy_reference,
        "probe_fairness": fairness,
        "primary_probe_setting": primary_setting,
        "per_seed_rows": setting_rows[primary_setting],
        "summary": summary,
        "paired_sae_minus_rand_exp": sae_minus_rand,
        "effective_width_controls": width_matched_controls,
        "wall_clock_seconds": elapsed,
    }
    RESULTS.write_text(json.dumps(payload, indent=2) + "\n")
    plots(summary, SMOKE)
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

"""
Experiment 03 — shattering dimensionality and CCGP on a real GPT-2-small SAE.

The question is deliberately NOT whether an SAE is better or worse than the residual
stream.  A residual stream is 768-wide; this SAE is a 24,576-wide ReLU expansion, so
that comparison would mostly rediscover Cover's theorem.  The load-bearing comparison
is instead a real sparse SAE code versus a Gaussian ReLU expansion matched in width,
encoder-column scale, and active-feature count.

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
claimed until the corresponding full command has completed.

The normal command downloads GPT-2 small and the public res-jb safetensors on first
use.  It does not require sae_lens: the encoder is exactly
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
ALL_ARMS = CORE_ARMS + ("rand_exp_dense",)


@dataclass(frozen=True)
class Config:
    n_items: int
    seeds: tuple[int, ...]
    n_folds: int
    pilot_layers: tuple[int, ...]
    probe_steps: int
    ccgp_splits: tuple[tuple[int, int], ...]
    batch_size: int
    weight_decay: float


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
            weight_decay=0.08,
        )
    return Config(
        n_items=96,
        seeds=(0, 1, 2, 3, 4),
        n_folds=5,
        pilot_layers=(6, 7, 8, 9),
        probe_steps=100,
        ccgp_splits=tuple(itertools.product(range(4), range(4))),
        batch_size=32,
        weight_decay=0.08,
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
    return "mps" if torch.backends.mps.is_available() else "cpu"


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


def standardise(x_train: torch.Tensor, x_test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, unbiased=False, keepdim=True).clamp_min(1e-5)
    return (x_train - mean) / std, (x_test - mean) / std


def fit_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    probe_seed: int,
    steps: int,
    weight_decay: float,
    train_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        return balanced_accuracy(head(x_eval), y_eval), balanced_accuracy(head(x_train), y_train)


def main_effect_pilot(
    resid: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
) -> list[float]:
    values = []
    for fold_index, (train, test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        tr, te = standardise(resid[train], resid[test])
        acc, _ = fit_probe(tr, stimuli.factors[train], te, stimuli.factors[test], seed + fold_index, cfg.probe_steps, cfg.weight_decay)
        values.append(acc.numpy())
    return np.concatenate(values, axis=0).mean(axis=0).tolist()


def sd_metric(
    rep: torch.Tensor,
    stimuli: Stimuli,
    seed: int,
    cfg: Config,
    ds: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Five item-disjoint folds, all 35 dichotomies trained as one multi-output probe."""
    rep, _ = prepare_features(rep)
    labels_by_condition = torch.stack([d["labels"] for d in ds], dim=1)
    labels = labels_by_condition[stimuli.condition_ids]
    test_accs, train_accs = [], []
    for fold_index, (train, test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
        tr, te = standardise(rep[train], rep[test])
        acc, train_acc = fit_probe(tr, labels[train], te, labels[test], seed + 100 * fold_index, cfg.probe_steps, cfg.weight_decay)
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
) -> dict[str, float]:
    """CCGP with all (or smoke-subset) 4x4 held-condition splits, item-disjoint throughout.

    The 16 heads for one dichotomy share an nn.Linear.  Its loss mask gives each head its
    own six training conditions; all standardisation statistics come only from training
    lexical items.  This keeps the implementation vectorised without allowing a lexical
    nuisance item into both a head's train and test examples.
    """
    rep, _ = prepare_features(rep)
    per_type: dict[str, list[float]] = {}
    for d_index, d in enumerate(ds):
        labels = d["labels"][stimuli.condition_ids]
        pos = sorted(d["positive"])
        neg = sorted(d["negative"])
        split_scores: list[float] = []
        for fold_index, (item_train, item_test) in enumerate(folds(stimuli.item_ids, cfg.n_folds, seed)):
            # Standardisation uses lexical training items only, never an item evaluated by
            # this fold.  The held conditions are unlabelled for that head but still belong
            # to the training item universe, as the global item-split rule requires.
            tr_all, te_all = standardise(rep[item_train], rep[item_test])
            condition_train = stimuli.condition_ids[item_train]
            condition_test = stimuli.condition_ids[item_test]
            ys, masks, eval_blocks, eval_labels = [], [], [], []
            for held_pos_i, held_neg_i in cfg.ccgp_splits:
                held_pos, held_neg = pos[held_pos_i], neg[held_neg_i]
                train_mask = (condition_train != held_pos) & (condition_train != held_neg)
                eval_mask = (condition_test == held_pos) | (condition_test == held_neg)
                # This happens only with a malformed factorial stimulus set.
                if int(eval_mask.sum()) == 0:
                    raise RuntimeError("CCGP held-condition split has no item-disjoint evaluation data")
                ys.append(labels[item_train])
                masks.append(train_mask)
                eval_blocks.append(te_all[eval_mask])
                eval_labels.append(labels[item_test][eval_mask])
            # Heads need a common evaluation tensor.  Each factorial held pair has exactly
            # two cells per test item, so all blocks have the same length by construction.
            eval_x = torch.stack(eval_blocks, dim=1)  # [examples, heads, features]
            eval_y = torch.stack(eval_labels, dim=1)
            # Train all 16 heads together.  Repeating x is modest here and makes the
            # masking logic auditable; no random split is introduced.
            n_heads = len(cfg.ccgp_splits)
            x_repeated = tr_all.unsqueeze(1).expand(-1, n_heads, -1).reshape(-1, tr_all.shape[1])
            # `repeat_interleave` keeps all H copies of lexical item i adjacent, just
            # like x_repeated.  Only the diagonal loss entry of each copy is enabled.
            y_repeated = labels[item_train].unsqueeze(1).expand(-1, n_heads).repeat_interleave(n_heads, dim=0)
            mask_repeated = torch.stack(masks, dim=1).repeat_interleave(n_heads, dim=0)
            # To keep a single head per split, rows corresponding to other head copies have
            # zero loss.  This is equivalent to separate probes, but vectorised parameters.
            diagonal_mask = torch.zeros((x_repeated.shape[0], n_heads), dtype=torch.bool)
            row_head = torch.arange(x_repeated.shape[0]) % n_heads
            diagonal_mask[torch.arange(x_repeated.shape[0]), row_head] = True
            head_masks = mask_repeated & diagonal_mask
            x_eval_flat = eval_x.permute(1, 0, 2).reshape(-1, tr_all.shape[1])
            y_eval_flat = torch.zeros((x_eval_flat.shape[0], n_heads), dtype=torch.float32)
            # A dedicated evaluation pass is simpler than abusing fit_probe's all-head
            # accuracy API; fit the parameters here with the exact masked objective.
            torch.manual_seed(seed + 10_000 * d_index + fold_index)
            head = nn.Linear(tr_all.shape[1], n_heads)
            opt = torch.optim.AdamW(head.parameters(), lr=0.03, weight_decay=cfg.weight_decay)
            for _ in range(cfg.probe_steps):
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(head(x_repeated), y_repeated, reduction="none")
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


def representations(x: torch.Tensor, sae: SAEWeights, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    f = sae.encode(x)
    rand_topk, rand_dense, target_l0 = random_expansion(x, sae, seed)
    reps = {
        "resid": x,
        "sae": f,
        "sae_recon": sae.decode(f),
        "rand_exp": rand_topk,
        "rand_exp_dense": rand_dense,
    }
    l0 = {name: float((rep > 0).sum(1).float().mean()) for name, rep in reps.items()}
    widths = {name: int(prepare_features(rep)[0].shape[1]) for name, rep in reps.items()}
    return reps, {
        "target_l0": target_l0,
        "mean_l0": l0,
        "surviving_width": widths,
        "rand_dense_reference_l0": l0["rand_exp_dense"],
    }


# ----------------------------------------------------------------------- gates/run


def explained_variance(x: torch.Tensor, recon: torch.Tensor) -> tuple[float, float]:
    mse = float(((x - recon) ** 2).mean())
    var = float(((x - x.mean(0, keepdim=True)) ** 2).mean())
    return 1.0 - mse / max(var, 1e-12), mse


def gate_a(model: HookedTransformer, stimuli: Stimuli, run_device: str, cfg: Config) -> tuple[SAEWeights, dict[str, Any], dict[int, torch.Tensor]]:
    """Load an exact residual SAE at layer 8 and ensure its real-text reconstruction is sane."""
    sae8 = load_direct_res_jb(8)
    resid = collect_residuals(model, stimuli, cfg.pilot_layers, run_device, cfg.batch_size)
    x8 = resid[8]
    f8 = sae8.encode(x8)
    ev, mse = explained_variance(x8, sae8.decode(f8))
    l0 = float((f8 > 0).sum(1).float().mean())
    # "Tens" is intentionally a broad sanity band, not an asserted published target.
    if not (1 < l0 < 2_000) or not np.isfinite(ev):
        raise RuntimeError(f"Gate A failed: layer-8 SAE EV={ev:.4f}, MSE={mse:.6g}, mean L0={l0:.1f}; likely wrong hook/weights.")
    return sae8, {"layer": 8, "explained_variance": ev, "mse": mse, "mean_l0": l0, "source": sae8.source}, resid


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


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ALL_ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        out[arm] = {}
        for metric in ("sd", "ccgp", "gap"):
            types = sorted({key for r in arm_rows for key in r.get(metric, {})})
            out[arm][metric] = {key: dict(zip(("mean", "ci95"), mean_ci(r[metric][key] for r in arm_rows))) for key in types}
    return out


def plots(summary: dict[str, Any], smoke: bool) -> None:
    arms = ALL_ARMS
    colors = {
        "resid": "#666666", "sae": "#3b6ea5", "sae_recon": "#5c9f72",
        "rand_exp": "#e08214", "rand_exp_dense": "#d73027",
    }
    fig, ax = plt.subplots(figsize=(7.1, 5.4))
    for arm in arms:
        x = summary[arm]["ccgp"]["overall"]
        y = summary[arm]["sd"]["overall"]
        ax.errorbar(x["mean"], y["mean"], xerr=x["ci95"], yerr=y["ci95"], fmt="o", ms=8, capsize=3, color=colors[arm], label=arm)
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="chance (0.5)")
    ax.axvline(0.5, color="crimson", ls="--", lw=1.2)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("CCGP: held-condition balanced accuracy")
    ax.set_ylabel("shattering dimensionality: balanced accuracy")
    suffix = "SMOKE subset" if smoke else "5 seeds; 95% CI"
    ax.set_title(f"Where the SAE code lands relative to its matched random expansion\n({suffix})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(FIGDIR / "01_shattering_vs_ccgp.png", dpi=180)
    plt.close(fig)

    types = ("main_effect", "two_way_xor", "three_way_parity", "unstructured")
    labels = ("main effect", "2-way XOR", "3-way parity", "unstructured")
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    x = np.arange(len(types))
    width = 0.19
    for i, arm in enumerate(arms):
        vals = [summary[arm]["sd"][kind]["mean"] for kind in types]
        errs = [summary[arm]["sd"][kind]["ci95"] for kind in types]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=2.5, label=arm, color=colors[arm])
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label="chance (0.5)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("shattering: balanced decoding accuracy")
    ax.set_title("Dichotomy breakdown: base-factor enumeration versus parity-family interactions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_dichotomy_breakdown.png", dpi=180)
    plt.close(fig)


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
    selected_ev, selected_mse = explained_variance(selected_x, sae.decode(sae.encode(selected_x)))
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
    rows: list[dict[str, Any]] = []
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
        for arm, rep in reps.items():
            sd, gap = sd_metric(rep, stimuli, seed, cfg, ds)
            ccgp = ccgp_metric(rep, stimuli, seed, cfg, ds)
            rows.append({"seed": seed, "arm": arm, "sd": sd, "ccgp": ccgp, "gap": gap})
            print(f"seed={seed} arm={arm} SD={sd['overall']:.3f} CCGP={ccgp['overall']:.3f}", flush=True)
    summary = aggregate(rows)
    sae_minus_rand = {
        metric: {
            kind: dict(zip(("mean", "ci95"), mean_ci(
                next(r[metric][kind] for r in rows if r["seed"] == seed and r["arm"] == "sae") -
                next(r[metric][kind] for r in rows if r["seed"] == seed and r["arm"] == "rand_exp")
                for seed in cfg.seeds
            )))
            for kind in summary["sae"][metric]
        }
        for metric in ("sd", "ccgp")
    }
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "exp03-results-v1; per_seed_rows: one row per (seed, arm), sd/ccgp/gap map dichotomy type to balanced accuracy; ci95=1.96*sample_sd/sqrt(n)",
        "status": "complete",
        "smoke": SMOKE,
        "config": {**cfg.__dict__, "seeds": list(cfg.seeds), "pilot_layers": list(cfg.pilot_layers), "ccgp_splits": [list(s) for s in cfg.ccgp_splits]},
        "device": run_device,
        "sae_loading": sae.source,
        "chosen_layer": layer,
        "gates": {
            "A_layer8": gate_a_stats,
            "A_selected_layer": {"explained_variance": selected_ev, "mse": selected_mse, "mean_l0": selected_l0},
            "B_pilot_main_effects": pilot,
            "C_resid_two_way_xor_sd": xor_gate["overall"],
        },
        "per_seed_metadata": seed_metadata,
        "per_seed_rows": rows,
        "summary": summary,
        "paired_sae_minus_rand_exp": sae_minus_rand,
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

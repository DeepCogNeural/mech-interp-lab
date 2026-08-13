"""
Experiment 02 — Mixed (superposed) coding under the shipped linear-probe estimator.

exp1 (Toy Models of Superposition) gives a STORAGE reason for superposition: sparse
features + a bottleneck force non-orthogonal packing. This experiment asks an
orthogonal COMPUTATION question, and it is the part a comp-neuro background makes natural.

The empirical contrast (motivated by Rigotti et al. 2013's "mixed selectivity") is:
under this fixed BCE-trained logistic estimator, mixed codes make XOR much easier to
read than the coordinate-wise control. This is estimator-specific evidence, not a
universal theorem that mixing is necessary for above-chance XOR accuracy.

Why the fixed nonlinearity is informative, and what it does NOT prove:
exp1's encoder is linear (h = W x). A linear probe on a linear projection is still
additive in x, so it cannot perfectly separate XOR for ANY geometry. It is not thereby
forced to chance: on the exact balanced quadrant distribution used here, the rule
"predict 1 iff ReLU(x_i) + ReLU(x_j) > 0" gets three quadrants right and reaches
0.75 accuracy.
We read from r = ReLU(W x), holding that nonlinearity CONSTANT across all three arms;
only W changes. The design therefore tests how geometry changes the behaviour of the
configured estimator. It does not turn the coordinate-wise arm into a chance theorem.

Math anchor (the deliberately narrow theorem): any coordinate-wise code gives an
additive linear score. Fix any positive represented magnitudes p and q. The four
coordinate-wise ReLU states obey s(0,q)+s(p,0)=s(0,0)+s(p,q), so the two XOR-positive
states cannot both score above a threshold while both negative states score below it.
Perfect separation over the continuous support is impossible because it would have to
separate this four-point subset for every p,q>0. The identity says nothing about a chance
ceiling; the constructive 0.75 witness above is the counterexample. Every XOR pair is
still drawn from represented features (indices 0..m-1), so missing coverage is not the
explanation for the empirical 0.494 returned by the configured BCE-trained logistic probe.

Three geometry arms at fixed (n, m), differing ONLY in W:
  monosemantic   selection matrix (feature k -> axis k). Coordinate-wise control.
  random         Gaussian, Frobenius-norm-matched to the learned W. A capacity ruler.
  superposition  the frozen W that exp1's storage objective trains at sparsity S.

Discipline (the over-claiming traps we explicitly avoid):
  - 8 seeds; two-sided Student-t 95% CIs over independent seeds; within-seed PAIRED
    (superposition - random) differences.
  - Any pooled contrast first averages the completed-run S values within each seed, then forms
    its interval across the 8 seed means. The 40 (seed,S) rows are never treated as 40
    independent replicates.
  - Output metric is task accuracy on a FIXED, class-balanced eval distribution,
    identical across every sparsity and geometry. S only changes the frozen W. We never
    compare reconstruction loss across distributions.
  - Chance line (0.5) on every accuracy plot. Probe train-vs-test gap reported.
  - "provably" only for coordinate-wise nonseparability, never for chance accuracy or a
    trained W. No equivalence claim is made because no margin was specified before analysis.
  - We verify the "superposition" arm is actually in superposition at each S
    (features represented > m, sum of per-feature dimensionality ~ m), so the label is earned.
  - m is not a knife-edge: repeated at m in {5, 8, 12}.

CPU only. ~10-15 min for the full run; SMOKE=1 gives a ~40s subset.
"""

import hashlib
import json
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
REAGGREGATE_ONLY = os.environ.get("REAGGREGATE_ONLY", "0") == "1"


def _output_root(here, smoke):
    return os.path.join(here, "smoke_output") if smoke else here


OUTPUT_ROOT = _output_root(HERE, SMOKE)
FIGDIR = os.path.join(OUTPUT_ROOT, "figures")
RESULT_PATH = os.path.join(OUTPUT_ROOT, "results.json")
os.makedirs(FIGDIR, exist_ok=True)
HEADLINE_FIGURE = "01_configured_probe_xor_accuracy.png"
SHIPPED_RAW_SHA256 = {
    "xor": "42f12943b1649bc1f8a78d7041a9709d9f9b0378c4e67e8d0b4966249de0d514",
    "enum": "91144e72f8dd60f6cf777bf5c68300c280180098f5baaba322b0ae40845c39c8",
    "status": "40e539944eab6cd62e0c361dae71e93a9f99fd706280b3f6678ca4472bab69cb",
    "combined": "d31965b7a01bb3760ae5a16169bb429a63eab9d9753835c2f2768b06fc40d253",
}


# ---------------------------------------------------------------- exp1 model (replicated so this experiment is self-contained)


class ToyModel(nn.Module):
    def __init__(self, n_features, n_hidden, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_hidden, n_features), generator=g))
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        return torch.relu((x @ self.W.T) @ self.W + self.b)


def sample_batch(bs, n, sparsity, gen):
    vals = torch.rand(bs, n, generator=gen)
    mask = torch.rand(bs, n, generator=gen) >= sparsity
    return vals * mask


def train_ae(n, m, sparsity, seed, steps):
    model = ToyModel(n, m, seed)
    importance = torch.ones(n)  # uniform: every feature must be mixed, none irrelevant
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed + 10_000)
    for _ in range(steps):
        x = sample_batch(1024, n, sparsity, gen)
        loss = (importance * (x - model(x)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.W.detach().clone()


def feature_dimensionality(W):
    Wn = W.numpy() if isinstance(W, torch.Tensor) else W
    norms = np.linalg.norm(Wn, axis=0)
    safe = np.where(norms > 1e-8, norms, 1.0)
    Wh = Wn / safe
    denom = ((Wh.T @ Wn) ** 2).sum(axis=1)
    denom = np.where(denom > 1e-10, denom, 1.0)
    return (norms**2) / denom


def super_status(W):
    """Is this W actually in superposition? Return (n_represented, sum_D, mean|off-diag Gram|)."""
    Wn = W.numpy()
    n_rep = int((np.linalg.norm(Wn, axis=0) > 0.1).sum())
    sumD = float(feature_dimensionality(Wn).sum())
    gram = Wn.T @ Wn
    off = gram[~np.eye(gram.shape[0], dtype=bool)]
    return n_rep, sumD, float(np.abs(off).mean())


# ---------------------------------------------------------------- the three geometries


def geometries(learned_W, m, n, seed):
    g = torch.Generator().manual_seed(seed + 20_000)
    rand = torch.randn(m, n, generator=g)
    rand *= learned_W.norm() / rand.norm()  # match total weight energy
    mono = torch.zeros(m, n)
    for k in range(m):
        mono[k, k] = 1.0  # feature k -> axis k; represents features 0..m-1 monosemantically
    return {"monosemantic": mono, "random": rand, "superposition": learned_W}


def code(W, x):
    return torch.relu(x @ W.T)  # fixed nonlinearity, identical for all three arms


# ---------------------------------------------------------------- fixed evaluation distribution


def balanced_pair_batch(B, n, pair, distractor_p, gen):
    i, j = pair
    q = B // 4
    B = q * 4
    ai = torch.cat([torch.zeros(q), torch.zeros(q), torch.ones(q), torch.ones(q)])
    aj = torch.cat([torch.zeros(q), torch.ones(q), torch.zeros(q), torch.ones(q)])
    x = torch.rand(B, n, generator=gen) * (torch.rand(B, n, generator=gen) >= (1 - distractor_p))
    x[:, i] = ai * torch.rand(B, generator=gen).clamp_min(1e-3)
    x[:, j] = aj * torch.rand(B, generator=gen).clamp_min(1e-3)
    y = (ai.bool() ^ aj.bool()).float()
    perm = torch.randperm(B, generator=gen)
    return x[perm], y[perm]


def _balanced_acc(pred, y):
    accs = []
    for c in (0.0, 1.0):
        mask = y == c
        if mask.any():
            accs.append((pred[mask] == c).float().mean().item())
    return float(np.mean(accs))


def probe(r_tr, y_tr, r_te, y_te, seed):
    """Linear logistic probe (torch). Returns (test_balanced_acc, train_balanced_acc)."""
    torch.manual_seed(seed)
    p = nn.Linear(r_tr.shape[1], 1)
    opt = torch.optim.Adam(p.parameters(), lr=0.02, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad()
        lossf(p(r_tr).squeeze(1), y_tr).backward()
        opt.step()
    with torch.no_grad():
        te = _balanced_acc((p(r_te).squeeze(1) > 0).float(), y_te)
        tr = _balanced_acc((p(r_tr).squeeze(1) > 0).float(), y_tr)
    return te, tr


def xor_acc(W, pairs, n, dp, seed):
    gen = torch.Generator().manual_seed(seed + 30_000)
    te, tr = [], []
    for pair in pairs:
        xtr, ytr = balanced_pair_batch(4000, n, pair, dp, gen)
        xte, yte = balanced_pair_batch(4000, n, pair, dp, gen)
        a, b = probe(code(W, xtr), ytr, code(W, xte), yte, seed)
        te.append(a)
        tr.append(b)
    return float(np.mean(te)), float(np.mean(tr))


def enum_acc(W, n, dp, seed):
    # NOTE: only meaningful for dp > 0. At dp=0 the inputs are all-zero (no feature is ever
    # active), so every per-feature label is the single class 0 and balanced accuracy is a
    # degenerate 1.0. The prose reports enumeration only for dp > 0.
    gen = torch.Generator().manual_seed(seed + 40_000)
    xtr = torch.rand(4000, n, generator=gen) * (torch.rand(4000, n, generator=gen) >= (1 - dp))
    xte = torch.rand(4000, n, generator=gen) * (torch.rand(4000, n, generator=gen) >= (1 - dp))
    rtr, rte = code(W, xtr), code(W, xte)
    atr, ate = (xtr > 0).float(), (xte > 0).float()
    return float(np.mean([probe(rtr, atr[:, i], rte, ate[:, i], seed)[0] for i in range(n)]))


# ---------------------------------------------------------------- run


_T975 = {
    # The experiment uses n=3 in smoke mode and n=8 in the full protocol. Keeping the
    # supported sample sizes explicit prevents silently falling back to a normal CI.
    2: 4.302652730,
    7: 2.364624252,
}


def ci95(v):
    """Two-sided 95% Student-t half-width over independent observations."""
    v = np.asarray(v, float)
    if len(v) <= 1:
        return 0.0
    df = len(v) - 1
    if df not in _T975:
        raise ValueError(f"no supported t critical value for n={len(v)}")
    return _T975[df] * v.std(ddof=1) / np.sqrt(len(v))


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_shipped_raw_rows(xor_rows, enum_rows, status_rows):
    """Fail closed before deriving claims from the one shipped full-run grid.

    ``REAGGREGATE_ONLY`` is intentionally narrower than a general result loader:
    it may re-present the original completed run, but it may not silently accept
    a missing, duplicate, reordered, or edited scientific row. A fresh experiment
    run has a different lifecycle and must not enter through this path.
    """

    m_values = (5, 8, 12)
    sparsities = (0.0, 0.7, 0.9, 0.97, 0.99)
    seeds = tuple(range(8))
    dps = (0.0, 0.05, 0.1)
    arms = ("monosemantic", "random", "superposition")

    expected_xor = {
        (m, S, seed, dp, arm)
        for m in m_values
        for S in sparsities
        for seed in seeds
        for dp in (dps if m == 8 else (0.0,))
        for arm in arms
    }
    expected_enum = {
        (S, seed, dp, arm)
        for S in sparsities
        for seed in seeds
        for dp in dps
        for arm in arms
    }
    expected_status = {(m, S, seed) for m in m_values for S in sparsities for seed in seeds}

    def finite_number(value, label):
        if isinstance(value, bool):
            raise ValueError(f"{label} is boolean, not numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} is not numeric: {value!r}") from exc
        if not np.isfinite(number):
            raise ValueError(f"{label} is not finite: {value!r}")
        return number

    observed_xor = set()
    for index, row in enumerate(xor_rows):
        if not isinstance(row, (list, tuple)) or len(row) != 7:
            raise ValueError(f"xor row {index} must contain seven fields")
        m, S, seed, dp, arm, test_acc, train_acc = row
        if not isinstance(m, int) or isinstance(m, bool) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"xor row {index} has a non-integer m or seed")
        key = (m, finite_number(S, f"xor row {index} sparsity"), seed, finite_number(dp, f"xor row {index} distractor_p"), arm)
        if not isinstance(arm, str) or key in observed_xor:
            raise ValueError(f"xor row {index} has an invalid or duplicate key: {key!r}")
        observed_xor.add(key)
        for label, value in (("test accuracy", test_acc), ("train accuracy", train_acc)):
            number = finite_number(value, f"xor row {index} {label}")
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"xor row {index} {label} is outside [0, 1]")
    if observed_xor != expected_xor:
        raise ValueError(
            f"xor grid drift: missing={len(expected_xor - observed_xor)}, "
            f"unexpected={len(observed_xor - expected_xor)}"
        )

    observed_enum = set()
    for index, row in enumerate(enum_rows):
        if not isinstance(row, (list, tuple)) or len(row) != 5:
            raise ValueError(f"enum row {index} must contain five fields")
        S, seed, dp, arm, accuracy = row
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"enum row {index} has a non-integer seed")
        key = (finite_number(S, f"enum row {index} sparsity"), seed, finite_number(dp, f"enum row {index} distractor_p"), arm)
        if not isinstance(arm, str) or key in observed_enum:
            raise ValueError(f"enum row {index} has an invalid or duplicate key: {key!r}")
        observed_enum.add(key)
        value = finite_number(accuracy, f"enum row {index} accuracy")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"enum row {index} accuracy is outside [0, 1]")
    if observed_enum != expected_enum:
        raise ValueError(
            f"enum grid drift: missing={len(expected_enum - observed_enum)}, "
            f"unexpected={len(observed_enum - expected_enum)}"
        )

    observed_status = set()
    for index, row in enumerate(status_rows):
        if not isinstance(row, (list, tuple)) or len(row) != 6:
            raise ValueError(f"status row {index} must contain six fields")
        m, S, seed, represented, sum_d, gram_off = row
        if not isinstance(m, int) or isinstance(m, bool) or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"status row {index} has a non-integer m or seed")
        key = (m, finite_number(S, f"status row {index} sparsity"), seed)
        if key in observed_status:
            raise ValueError(f"status row {index} has a duplicate key: {key!r}")
        observed_status.add(key)
        represented_value = finite_number(represented, f"status row {index} represented count")
        if represented_value != int(represented_value) or not 0 <= represented_value <= 20:
            raise ValueError(f"status row {index} represented count is invalid")
        finite_number(sum_d, f"status row {index} dimensionality sum")
        finite_number(gram_off, f"status row {index} Gram statistic")
    if observed_status != expected_status:
        raise ValueError(
            f"status grid drift: missing={len(expected_status - observed_status)}, "
            f"unexpected={len(observed_status - expected_status)}"
        )

    payloads = {"xor": xor_rows, "enum": enum_rows, "status": status_rows}
    for label, rows in payloads.items():
        observed_hash = _canonical_sha256(rows)
        if observed_hash != SHIPPED_RAW_SHA256[label]:
            raise ValueError(f"{label} shipped-row SHA-256 drift: {observed_hash}")
    combined_hash = _canonical_sha256(payloads)
    if combined_hash != SHIPPED_RAW_SHA256["combined"]:
        raise ValueError(f"combined shipped-row SHA-256 drift: {combined_hash}")


def result_payload(xor_rows, enum_rows, status_rows, m_main, dps, artifact_status):
    analysis = analysis_summary(xor_rows, m_main, dps)
    if SMOKE:
        analysis["supported_conclusion"] = "SMOKE_ONLY_NO_SCIENTIFIC_CONCLUSION"
        analysis["scope"] = "three-seed execution smoke; not a public or adjudicable artifact"
    return {
        "schema": "exp02-results-v2",
        "analysis": analysis,
        "artifact_status": artifact_status,
        "xor": xor_rows,
        "enum": enum_rows,
        "status": status_rows,
    }


def reaggregate_existing():
    """Regenerate derived metadata and figures from shipped raw rows; fit no model or probe."""
    if SMOKE:
        raise RuntimeError("SMOKE=1 cannot be combined with REAGGREGATE_ONLY=1")
    result_path = os.path.join(HERE, "results.json")
    with open(result_path) as f:
        existing = json.load(f)
    xor_rows = [tuple(r) for r in existing["xor"]]
    enum_rows = [tuple(r) for r in existing["enum"]]
    status_rows = [tuple(r) for r in existing["status"]]
    validate_shipped_raw_rows(xor_rows, enum_rows, status_rows)

    n = 20
    m_main = 8
    m_values = sorted({r[0] for r in xor_rows})
    sparsities = sorted({r[1] for r in xor_rows if r[0] == m_main})
    seeds = sorted({r[2] for r in xor_rows})
    dps = sorted({r[3] for r in xor_rows if r[0] == m_main})
    preferred_arms = ["monosemantic", "random", "superposition"]
    observed_arms = {r[4] for r in xor_rows}
    arms = [arm for arm in preferred_arms if arm in observed_arms]

    payload = result_payload(
        xor_rows,
        enum_rows,
        status_rows,
        m_main,
        dps,
        {
            "raw_rows": "loaded unchanged from the existing completed run",
            "analysis": "recomputed from shipped xor rows",
            "figures": "regenerated from shipped rows without fitting a model or probe",
            "mode": "CHECKED_IN_REAGGREGATION",
            "public_artifact": True,
            "output_root": "canonical experiment directory",
        },
    )
    _publish_payload_and_figures(
        payload,
        result_path,
        xor_rows,
        enum_rows,
        status_rows,
        n,
        m_main,
        m_values,
        sparsities,
        seeds,
        dps,
        arms,
    )
    return xor_rows


def run():
    if SMOKE and REAGGREGATE_ONLY:
        raise RuntimeError("SMOKE=1 cannot be combined with REAGGREGATE_ONLY=1")
    if REAGGREGATE_ONLY:
        return reaggregate_existing()

    n = 20
    m_values = [8] if SMOKE else [5, 8, 12]
    sparsities = [0.0, 0.9, 0.99] if SMOKE else [0.0, 0.7, 0.9, 0.97, 0.99]
    seeds = list(range(3)) if SMOKE else list(range(8))
    steps = 1500 if SMOKE else 4000
    dps = [0.0, 0.05, 0.1]
    arms = ["monosemantic", "random", "superposition"]
    m_main = 8

    xor_rows = []   # (m, S, seed, dp, arm, xor_test, xor_train)
    enum_rows = []  # (S, seed, dp, arm, enum)  -- m_main only, for prose nuance
    status_rows = []  # (m, S, seed, n_rep, sumD, gram_off) for the learned (super) W

    for m in m_values:
        npf = min(m, 8)
        pairs = [(i, j) for i in range(npf) for j in range(i + 1, npf)]
        for S in sparsities:
            for seed in seeds:
                W_learned = train_ae(n, m, S, seed, steps)
                status_rows.append((m, S, seed, *super_status(W_learned)))
                geos = geometries(W_learned, m, n, seed)
                dps_here = dps if m == m_main else [0.0]
                for dp in dps_here:
                    for arm in arms:
                        te, tr = xor_acc(geos[arm], pairs, n, dp, seed)
                        xor_rows.append((m, S, seed, dp, arm, te, tr))
                        if m == m_main:
                            enum_rows.append((S, seed, dp, arm, enum_acc(geos[arm], n, dp, seed)))
                    print(f"  m={m} S={S:<5} seed={seed} dp={dp} done")

    payload = result_payload(
        xor_rows,
        enum_rows,
        status_rows,
        m_main,
        dps,
        {
            "raw_rows": "generated by this invocation",
            "figures": "generated by this invocation under the v2 analysis",
            "mode": "SMOKE_ONLY" if SMOKE else "FULL_CONFIGURED_GRID",
            "public_artifact": not SMOKE,
            "output_root": "smoke_output" if SMOKE else "canonical experiment directory",
        },
    )
    _publish_payload_and_figures(
        payload,
        RESULT_PATH,
        xor_rows,
        enum_rows,
        status_rows,
        n,
        m_main,
        m_values,
        sparsities,
        seeds,
        dps,
        arms,
    )
    return xor_rows


def xor_mean_ci(rows, m, S, dp, arm):
    v = [r[5] for r in rows if r[0] == m and r[1] == S and r[3] == dp and r[4] == arm]
    return (np.mean(v), ci95(v)) if v else (np.nan, 0.0)


def paired_diff(rows, m, dp):
    """Within-seed superposition-random rows as (S, seed, difference)."""
    out = []
    Ss = sorted({r[1] for r in rows if r[0] == m and r[3] == dp})
    seeds = sorted({r[2] for r in rows if r[0] == m and r[3] == dp})
    for S in Ss:
        for seed in seeds:
            sup = [r[5] for r in rows if r[:5] == (m, S, seed, dp, "superposition")]
            ran = [r[5] for r in rows if r[:5] == (m, S, seed, dp, "random")]
            if sup and ran:
                out.append((S, seed, sup[0] - ran[0]))
    return out


def pooled_seed_diffs(rows, m, dp):
    """Average the completed-run S contrasts within seed; return one value per seed."""
    diffs = paired_diff(rows, m, dp)
    seeds = sorted({seed for _, seed, _ in diffs})
    return [np.mean([d for _, s, d in diffs if s == seed]) for seed in seeds]


def pooled_seed_arm_diffs(rows, m, dp, arm, baseline):
    """Average an arbitrary paired arm-baseline contrast across completed-run S within seed."""
    seeds = sorted({r[2] for r in rows if r[0] == m and r[3] == dp})
    sparsities = sorted({r[1] for r in rows if r[0] == m and r[3] == dp})
    out = []
    for seed in seeds:
        per_s = []
        for S in sparsities:
            arm_value = [r[5] for r in rows if r[:5] == (m, S, seed, dp, arm)]
            baseline_value = [r[5] for r in rows if r[:5] == (m, S, seed, dp, baseline)]
            if arm_value and baseline_value:
                per_s.append(arm_value[0] - baseline_value[0])
        if per_s:
            out.append(np.mean(per_s))
    return out


def analysis_summary(rows, m_main, dps):
    mono = [
        r[5]
        for r in rows
        if r[0] == m_main and r[1] == 0.0 and r[3] == 0.0 and r[4] == "monosemantic"
    ]
    pooled = []
    for dp in dps:
        values = pooled_seed_diffs(rows, m_main, dp)
        pooled.append({
            "distractor_p": dp,
            "n_seeds": len(values),
            "mean": float(np.mean(values)),
            "ci95_t_half_width": float(ci95(values)),
        })
    mixed_minus_coordinatewise = []
    for arm in ("random", "superposition"):
        values = pooled_seed_arm_diffs(rows, m_main, 0.0, arm, "monosemantic")
        mixed_minus_coordinatewise.append({
            "arm": arm,
            "baseline": "monosemantic",
            "distractor_p": 0.0,
            "n_seeds": len(values),
            "mean": float(np.mean(values)),
            "ci95_t_half_width": float(ci95(values)),
        })
    return {
        "independent_unit": "seed",
        "confidence_interval": "two-sided Student-t over seed-level values",
        "pooled_contrast": "average completed-run S values within seed, then summarize across seeds",
        "equivalence_margin": None,
        "supported_conclusion": "no reliable learned-v-random advantage detected in this 8-seed sample",
        "coordinatewise_math": {
            "theorem": "additive linear scores cannot perfectly separate XOR",
            "chance_ceiling": False,
            "constructive_witness_accuracy": 0.75,
            "witness": "predict 1 iff ReLU(x_i)+ReLU(x_j)>0 on the balanced four-quadrant distribution",
        },
        "configured_bce_logistic_probe": {
            "condition": {"m": m_main, "S": 0.0, "distractor_p": 0.0},
            "n_seeds": len(mono),
            "monosemantic_test_accuracy_mean": float(np.mean(mono)),
            "monosemantic_test_accuracy_ci95_t_half_width": float(ci95(mono)),
            "interpretation": "empirical estimator result; not a theorem-imposed chance ceiling",
            "pooled_mixed_minus_coordinatewise": mixed_minus_coordinatewise,
        },
        "pooled_superposition_minus_random": pooled,
    }


def _publish_payload_and_figures(
    payload,
    result_path,
    xor_rows,
    enum_rows,
    status_rows,
    n,
    m_main,
    m_values,
    sparsities,
    seeds,
    dps,
    arms,
):
    """Stage both figures and publish the result manifest last."""

    result_parent = os.path.dirname(result_path)
    os.makedirs(result_parent, exist_ok=True)
    stage_root = tempfile.mkdtemp(prefix=".exp02-stage.", dir=result_parent)
    stage_figures = os.path.join(stage_root, "figures")
    os.makedirs(stage_figures, exist_ok=True)
    stage_result = os.path.join(stage_root, "results.json")
    try:
        plot_and_summarize(
            xor_rows,
            enum_rows,
            status_rows,
            n,
            m_main,
            m_values,
            sparsities,
            seeds,
            dps,
            arms,
            figure_dir=stage_figures,
        )
        with open(stage_result, "x") as handle:
            json.dump(payload, handle, indent=1, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.makedirs(FIGDIR, exist_ok=True)
        for name in (HEADLINE_FIGURE, "02_superposition_vs_random_paired.png"):
            os.replace(os.path.join(stage_figures, name), os.path.join(FIGDIR, name))
        os.replace(stage_result, result_path)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def plot_and_summarize(
    xor_rows,
    enum_rows,
    status_rows,
    n,
    m_main,
    m_values,
    sparsities,
    seeds,
    dps,
    arms,
    *,
    figure_dir=None,
):
    figure_dir = FIGDIR if figure_dir is None else figure_dir
    colors = {"monosemantic": "#8a8a8a", "random": "#e08214", "superposition": "#3b6ea5"}

    # ---- figure 1: headline. Three arms, dp=0, XOR accuracy vs sparsity ----
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for arm in arms:
        y = [xor_mean_ci(xor_rows, m_main, S, 0.0, arm)[0] for S in sparsities]
        e = [xor_mean_ci(xor_rows, m_main, S, 0.0, arm)[1] for S in sparsities]
        ax.errorbar(sparsities, y, yerr=e, marker="o", lw=2, capsize=3, color=colors[arm], label=arm)
    ax.axhline(0.5, ls="--", color="crimson", lw=1.3, label="chance (0.5)")
    ax.axhline(0.75, ls=":", color="#555555", lw=1.3,
               label="constructive coordinate-wise witness (0.75)")
    ax.set_xlabel("sparsity S the code was trained at")
    ax.set_ylabel("balanced XOR readout accuracy\n(LINEAR probe on r = ReLU(Wx))")
    ax.set_title(
        f"The shipped BCE probe favours mixed codes; chance is not a theorem for the control\n"
        f"(n={n}, m={m_main}; identical fixed nonlinearity; balanced isolated-pair XOR)",
        fontsize=10,
    )
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = os.path.join(figure_dir, HEADLINE_FIGURE)
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  saved {p1}")

    # ---- figure 2: the honest small effect. paired (super - random) vs S, per dp ----
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    dpc = {0.0: "#c2a5cf", 0.05: "#9970ab", 0.1: "#5e3c99"}
    for dp in dps:
        diffs = paired_diff(xor_rows, m_main, dp)
        Ss = sorted({s for s, _, _ in diffs})
        y = [np.mean([d for s, _, d in diffs if s == S]) for S in Ss]
        e = [ci95([d for s, _, d in diffs if s == S]) for S in Ss]
        ax.errorbar(Ss, y, yerr=e, marker="s", lw=1.8, capsize=3, color=dpc[dp],
                    label=f"background activity dp={dp}")
    ax.axhline(0.0, ls="--", color="black", lw=1.1, label="no difference")
    ax.set_xlabel("sparsity S the code was trained at")
    ax.set_ylabel("XOR accuracy: superposition − random\n(within-seed paired difference)")
    ax.set_title(
        f"No reliable learned-v-random advantage detected in this sample\n"
        f"(n={n}, m={m_main}, {len(seeds)} seeds; per-S Student-t intervals over paired seed differences)",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = os.path.join(figure_dir, "02_superposition_vs_random_paired.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  saved {p2}")

    # ---- printed summary ----
    print("\n================  HEADLINE (m=8, dp=0, isolated pair)  ================")
    for arm in arms:
        for S in sparsities:
            mu, c = xor_mean_ci(xor_rows, m_main, S, 0.0, arm)
            print(f"  {arm:<13} S={S:<5} XOR={mu:.3f}±{c:.3f}")
    # probe overfit check
    gap_seed_means = []
    for seed in seeds:
        gaps = [
            r[6] - r[5]
            for r in xor_rows
            if r[0] == m_main and r[2] == seed and r[4] in ("random", "superposition")
        ]
        gap_seed_means.append(np.mean(gaps))
    print(
        f"\n  probe train-minus-test gap (mixed arms, S/dp averaged within seed): "
        f"mean={np.mean(gap_seed_means):+.3f} ± {ci95(gap_seed_means):.3f}"
    )

    print("\n================  SUPERPOSITION-VS-RANDOM (paired, m=8)  ================")
    for dp in dps:
        dv = pooled_seed_diffs(xor_rows, m_main, dp)
        print(
            f"  dp={dp}: mean(super-random)={np.mean(dv):+.3f} ± {ci95(dv):.3f} "
            f"(n_seeds={len(dv)}; S averaged within seed)"
        )
        for S in sparsities:
            ds = [d for s, _, d in paired_diff(xor_rows, m_main, dp) if s == S]
            if ds:
                print(f"       S={S:<5} {np.mean(ds):+.3f} ± {ci95(ds):.3f}")

    print("\n================  ROBUSTNESS ACROSS m (dp=0, paired super-random)  ================")
    for m in m_values:
        dv = pooled_seed_diffs(xor_rows, m, 0.0)
        print(f"  m={m:<3} mean={np.mean(dv):+.3f} ± {ci95(dv):.3f}")

    print("\n================  IS 'super' ACTUALLY SUPERPOSITION? (learned W, m=8)  ================")
    for S in sparsities:
        rr = [r for r in status_rows if r[0] == m_main and r[1] == S]
        nr = np.mean([r[3] for r in rr])
        sd = np.mean([r[4] for r in rr])
        go = np.mean([r[5] for r in rr])
        tag = "superposition" if nr > m_main + 0.5 else "~orthogonal (no superposition)"
        print(f"  S={S:<5} n_represented={nr:.1f}/{n}  sum_D={sd:.2f} (<=m={m_main})  gram_off={go:.3f}  -> {tag}")

    print("\n================  ENUMERATION nuance (m=8, mean per-feature decodability, dp>0 only)  ================")
    for dp in [d for d in dps if d > 0]:  # dp=0 is degenerate (all-zero inputs) -- see enum_acc note
        for arm in arms:
            v = []
            for seed in seeds:
                per_seed = [r[4] for r in enum_rows if r[1] == seed and r[2] == dp and r[3] == arm]
                if per_seed:
                    v.append(np.mean(per_seed))
            if v:
                print(f"  dp={dp} {arm:<13} enum={np.mean(v):.3f}±{ci95(v):.3f}")


if __name__ == "__main__":
    print(
        f"Experiment 02 — superposition and readout  "
        f"(SMOKE={SMOKE}, REAGGREGATE_ONLY={REAGGREGATE_ONLY})"
    )
    run()
    print("\nDone.")

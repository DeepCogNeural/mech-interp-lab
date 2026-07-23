"""
Experiment 02 — Mixed (superposed) coding has a computation reason, not just a storage reason.

exp1 (Toy Models of Superposition) gives a STORAGE reason for superposition: sparse
features + a bottleneck force non-orthogonal packing. This experiment gives an
orthogonal COMPUTATION reason, and it is the part a comp-neuro background makes natural.

The claim (this is Rigotti et al. 2013's "mixed selectivity", with ground truth):
a mixed, superposed code keeps a *downstream linear readout* able to read feature
*interactions*; a purely monosemantic code — one feature per unit, the ideal endpoint
an SAE pushes toward — provably cannot.

Why a nonlinearity is unavoidable, and why that is NOT the trivial part:
exp1's encoder is linear (h = W x). A linear probe on a linear projection is still
linear in x, so it can never represent XOR, which needs the interaction term x_i*x_j —
for ANY geometry. So we read from a fixed nonlinearity r = ReLU(W x). The nonlinearity
is held CONSTANT across all three arms; only the representation geometry W changes.
Result is therefore not "a nonlinearity lets you do XOR" (trivial) but: under the same
fixed nonlinearity, r = ReLU(single feature) is still additive and reads no interaction,
whereas r = ReLU(mixture) manufactures the equivalent cross-terms. Geometry decides it.

Math anchor (a theorem, not a trained result — the one place "provably" is earned):
any monosemantic code r_k = f_k(x_{i_k}) cannot represent XOR(a_i, a_j) via a linear
readout, because XOR(a,b) = a + b - 2ab needs the product ab, and ab is not in the span
of {1, f(a), g(b)}. The strong form of the anchor — the one we test — is that this holds
EVEN when the monosemantic code perfectly encodes both features. So every XOR pair here
is drawn from the features the monosemantic arm actually represents (indices 0..m-1);
its chance-level XOR is the theorem, not a coverage artifact.

Three geometry arms at fixed (n, m), differing ONLY in W:
  monosemantic   selection matrix (feature k -> axis k). The theorem-backed anchor.
  random         Gaussian, Frobenius-norm-matched to the learned W. A capacity ruler.
  superposition  the frozen W that exp1's storage objective trains at sparsity S.

Discipline (the over-claiming traps we explicitly avoid):
  - 8 seeds; mean +/- 95% CI; within-seed PAIRED (superposition - random) differences.
  - Output metric is task accuracy on a FIXED, class-balanced eval distribution,
    identical across every sparsity and geometry. S only changes the frozen W. We never
    compare reconstruction loss across distributions.
  - Chance line (0.5) on every accuracy plot. Probe train-vs-test gap reported.
  - "provably" only for the monosemantic theorem, never for a trained W.
  - We verify the "superposition" arm is actually in superposition at each S
    (features represented > m, sum of per-feature dimensionality ~ m), so the label is earned.
  - m is not a knife-edge: repeated at m in {5, 8, 12}.

CPU only. ~10-15 min for the full run; SMOKE=1 gives a ~40s subset.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"


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


def ci95(v):
    v = np.asarray(v, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def run():
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

    json.dump({"xor": xor_rows, "enum": enum_rows, "status": status_rows},
              open(os.path.join(HERE, "results.json"), "w"), indent=1)
    plot_and_summarize(xor_rows, enum_rows, status_rows, n, m_main, m_values, sparsities, seeds, dps, arms)
    return xor_rows


def xor_mean_ci(rows, m, S, dp, arm):
    v = [r[5] for r in rows if r[0] == m and r[1] == S and r[3] == dp and r[4] == arm]
    return (np.mean(v), ci95(v)) if v else (np.nan, 0.0)


def paired_diff(rows, m, dp):
    """within-seed (superposition - random) per (S, seed) at given m, dp -> list of (S, diff)."""
    out = []
    Ss = sorted({r[1] for r in rows if r[0] == m and r[3] == dp})
    seeds = sorted({r[2] for r in rows if r[0] == m and r[3] == dp})
    for S in Ss:
        for seed in seeds:
            sup = [r[5] for r in rows if r[:5] == (m, S, seed, dp, "superposition")]
            ran = [r[5] for r in rows if r[:5] == (m, S, seed, dp, "random")]
            if sup and ran:
                out.append((S, sup[0] - ran[0]))
    return out


def plot_and_summarize(xor_rows, enum_rows, status_rows, n, m_main, m_values, sparsities, seeds, dps, arms):
    colors = {"monosemantic": "#8a8a8a", "random": "#e08214", "superposition": "#3b6ea5"}

    # ---- figure 1: headline. Three arms, dp=0, XOR accuracy vs sparsity ----
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for arm in arms:
        y = [xor_mean_ci(xor_rows, m_main, S, 0.0, arm)[0] for S in sparsities]
        e = [xor_mean_ci(xor_rows, m_main, S, 0.0, arm)[1] for S in sparsities]
        ax.errorbar(sparsities, y, yerr=e, marker="o", lw=2, capsize=3, color=colors[arm], label=arm)
    ax.axhline(0.5, ls="--", color="crimson", lw=1.3, label="chance (0.5)")
    ax.set_xlabel("sparsity S the code was trained at")
    ax.set_ylabel("balanced XOR readout accuracy\n(LINEAR probe on r = ReLU(Wx))")
    ax.set_title(
        f"A monosemantic code can't linearly read a feature interaction; a mixed code can\n"
        f"(n={n}, m={m_main}; identical fixed nonlinearity across arms; XOR of an isolated feature pair)",
        fontsize=10,
    )
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = os.path.join(FIGDIR, "01_monosemantic_cannot_read_xor.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  saved {p1}")

    # ---- figure 2: the honest small effect. paired (super - random) vs S, per dp ----
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    dpc = {0.0: "#c2a5cf", 0.05: "#9970ab", 0.1: "#5e3c99"}
    for dp in dps:
        diffs = paired_diff(xor_rows, m_main, dp)
        Ss = sorted({s for s, _ in diffs})
        y = [np.mean([d for s, d in diffs if s == S]) for S in Ss]
        e = [ci95([d for s, d in diffs if s == S]) for S in Ss]
        ax.errorbar(Ss, y, yerr=e, marker="s", lw=1.8, capsize=3, color=dpc[dp],
                    label=f"background activity dp={dp}")
    ax.axhline(0.0, ls="--", color="black", lw=1.1, label="no difference")
    ax.set_xlabel("sparsity S the code was trained at")
    ax.set_ylabel("XOR accuracy: superposition − random\n(within-seed paired difference)")
    ax.set_title(
        f"Storage-learned superposition gives no reliable readout edge over random mixing\n"
        f"(n={n}, m={m_main}, {len(seeds)} seeds; within-seed paired differences hug zero across sparsity and background)",
        fontsize=10,
    )
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = os.path.join(FIGDIR, "02_superposition_vs_random_paired.png")
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
    gaps = [r[6] - r[5] for r in xor_rows if r[0] == m_main and r[4] in ("random", "superposition")]
    print(f"\n  probe train-minus-test gap (mixed arms): mean={np.mean(gaps):+.3f} (small => not overfit)")

    print("\n================  SUPERPOSITION-VS-RANDOM (paired, m=8)  ================")
    for dp in dps:
        dv = [d for _, d in paired_diff(xor_rows, m_main, dp)]
        print(f"  dp={dp}: mean(super-random)={np.mean(dv):+.3f} ± {ci95(dv):.3f} (n={len(dv)})")
        for S in sparsities:
            ds = [d for s, d in paired_diff(xor_rows, m_main, dp) if s == S]
            if ds:
                print(f"       S={S:<5} {np.mean(ds):+.3f} ± {ci95(ds):.3f}")

    print("\n================  ROBUSTNESS ACROSS m (dp=0, paired super-random)  ================")
    for m in m_values:
        dv = [d for _, d in paired_diff(xor_rows, m, 0.0)]
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
            v = [r[4] for r in enum_rows if r[2] == dp and r[3] == arm]
            if v:
                print(f"  dp={dp} {arm:<13} enum={np.mean(v):.3f}±{ci95(v):.3f}")


if __name__ == "__main__":
    print(f"Experiment 02 — superposition and readout  (SMOKE={SMOKE})")
    run()
    print("\nDone.")

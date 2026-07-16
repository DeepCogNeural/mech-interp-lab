"""
Minimal replication of Anthropic's "Toy Models of Superposition" (Elhage et al., 2022).
https://transformer-circuits.pub/2022/toy_model/index.html

The setup, in one paragraph:
  We have n sparse features we want to squeeze through an m-dimensional bottleneck, m < n.
  The model is a linear encoder + a ReLU decoder that shares the same weight matrix:
      h  = W x            (n -> m, compress)
      x' = ReLU(W^T h + b)  (m -> n, reconstruct)
  If W's columns were forced orthogonal, only m features could survive. They aren't forced.
  When features are sparse, the model instead packs MORE than m features into m dimensions
  using non-orthogonal directions, accepting interference between features that rarely
  co-occur. That's superposition.

Two experiments:
  1. n=5, m=2, sparsity sweep. The iconic figure: you can literally see the feature
     directions rotate from an orthogonal pair (dense) into a pentagon (sparse).
  2. n=20, m=5, sparsity sweep. Gram matrices + per-feature dimensionality, showing
     the "1/(1-S) features per dimension" scaling and the discrete geometric phases.

CPU only. Runs in ~1-2 minutes.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ---------------------------------------------------------------- model


class ToyModel(nn.Module):
    """h = Wx ; x' = ReLU(W^T h + b). Tied weights, as in the paper."""

    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_hidden, n_features))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        h = x @ self.W.T
        return torch.relu(h @ self.W + self.b)


def sample_batch(batch_size: int, n_features: int, sparsity: float) -> torch.Tensor:
    """Each feature is 0 with prob `sparsity`, else Uniform[0, 1). Features independent."""
    vals = torch.rand(batch_size, n_features)
    mask = torch.rand(batch_size, n_features) >= sparsity
    return vals * mask


def train(n_features, n_hidden, sparsity, importance, steps=10_000, batch_size=1024, lr=1e-3):
    model = ToyModel(n_features, n_hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    for _ in range(steps):
        x = sample_batch(batch_size, n_features, sparsity)
        loss = (importance * (x - model(x)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    return model, loss.item()


# ---------------------------------------------------------------- metrics


def feature_dimensionality(W: np.ndarray) -> np.ndarray:
    """
    Per-feature dimensionality from the paper:
        D_i = ||W_i||^2 / sum_j (W_hat_i . W_j)^2
    Reads as "how many of the m dimensions does feature i get to itself".
    D_i = 1  -> feature i owns a dimension outright (no superposition)
    D_i = 0  -> feature i is not represented at all
    D_i = 1/2, 2/3, ... -> feature i shares its dimensions; the specific fractions
                           correspond to specific geometric arrangements
                           (1/2 = antipodal pair, 2/3 = triangle, ...)
    """
    norms = np.linalg.norm(W, axis=0)  # (n_features,)
    safe = np.where(norms > 1e-8, norms, 1.0)
    W_hat = W / safe  # unit vectors
    interference = (W_hat.T @ W) ** 2  # (n_features, n_features)
    denom = interference.sum(axis=1)
    denom = np.where(denom > 1e-10, denom, 1.0)
    return (norms**2) / denom


# ---------------------------------------------------------------- experiment 1


def experiment_1():
    """
    n=5 features into m=2 dimensions. Sweep sparsity, watch the geometry.

    Two importance regimes, because the contrast is the lesson:
      - decaying importance (0.7^i): the model triages. It protects the important
        features and pairs the rest up antipodally.
      - uniform importance: no feature is special, so at high sparsity the model
        finds the symmetric solution -- a regular pentagon. This is the iconic
        figure from the paper.
    """
    n_features, n_hidden = 5, 2
    sparsities = [0.0, 0.6, 0.8, 0.9, 0.95, 0.99]
    regimes = [
        ("decaying importance $0.7^i$", torch.tensor([0.7**i for i in range(n_features)])),
        ("uniform importance", torch.ones(n_features)),
    ]

    fig, axes = plt.subplots(
        len(regimes), len(sparsities), figsize=(2.7 * len(sparsities), 3.1 * len(regimes))
    )
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_features))
    results = []

    for row, (regime_name, importance) in enumerate(regimes):
        for col, S in enumerate(sparsities):
            ax = axes[row, col]
            model, loss = train(n_features, n_hidden, S, importance)
            W = model.W.detach().numpy()  # (2, 5)
            n_alive = int((np.linalg.norm(W, axis=0) > 0.1).sum())
            results.append((regime_name, S, loss, n_alive))
            print(f"  [exp1] {regime_name:<26} S={S:<5} loss={loss:.5f}  represented={n_alive}/5")

            for i in range(n_features):
                ax.arrow(
                    0, 0, W[0, i], W[1, i],
                    head_width=0.05, head_length=0.07, fc=colors[i], ec=colors[i],
                    length_includes_head=True, linewidth=2.0,
                )
            lim = 1.6
            ax.add_artist(
                plt.Circle((0, 0), 1.0, fill=False, ls="--", lw=0.7, color="gray", alpha=0.6)
            )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"S = {S}", fontsize=11)
            ax.text(
                0.5, 0.02, f"{n_alive}/5 represented", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color="dimgray",
            )
            if col == 0:
                ax.set_ylabel(regime_name, fontsize=10)

    fig.suptitle(
        "Toy Models of Superposition — 5 features, 2 hidden dimensions\n"
        "Each arrow is one feature's direction (a column of $W$). Color = feature index (dark = first).",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.91])
    out = os.path.join(FIGDIR, "01_feature_geometry_5x2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return results


# ---------------------------------------------------------------- experiment 2


def experiment_2():
    """n=20 features into m=5 dimensions. Gram matrices + dimensionality."""
    n_features, n_hidden = 20, 5
    importance = torch.tensor([0.9**i for i in range(n_features)])
    sparsities = [0.0, 0.7, 0.9, 0.97, 0.99, 0.997]

    fig, axes = plt.subplots(2, len(sparsities), figsize=(2.7 * len(sparsities), 6.2))
    summary = []

    for col, S in enumerate(sparsities):
        model, loss = train(n_features, n_hidden, S, importance)
        W = model.W.detach().numpy()  # (5, 20)
        gram = W.T @ W  # (20, 20)
        D = feature_dimensionality(W)
        n_alive = int((np.linalg.norm(W, axis=0) > 0.1).sum())
        summary.append((S, loss, n_alive, D.sum()))
        print(
            f"  [exp2] S={S:<6} loss={loss:.5f}  represented={n_alive:>2}/20  "
            f"sum(D_i)={D.sum():.2f} (~n_hidden={n_hidden} if the budget is fully used)"
        )

        vmax = np.abs(gram).max()
        ax = axes[0, col]
        im = ax.imshow(gram, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"S = {S}\n{n_alive}/20 represented", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("$W^T W$\n(diag = norm, off-diag = interference)", fontsize=9)

        ax = axes[1, col]
        ax.bar(range(n_features), D, color="#3b6ea5")
        for frac, label in [(1.0, "1"), (2 / 3, "2/3"), (0.5, "1/2")]:
            ax.axhline(frac, ls="--", lw=0.7, color="gray", alpha=0.7)
            if col == 0:
                ax.text(20.4, frac, label, fontsize=7, va="center", color="gray")
        ax.set_ylim(0, 1.15)
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("dimensionality $D_i$\nper feature", fontsize=9)
        ax.set_xlabel("feature (by importance)", fontsize=8)

    fig.suptitle(
        "20 features, 5 hidden dimensions — sparsity drives superposition\n"
        "Top: Gram matrix of $W$. Off-diagonal color = features sharing directions.   "
        "Bottom: $D_i$ = fraction of a dimension feature $i$ owns.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.0, 0.94, 0.90])
    cax = fig.add_axes([0.955, 0.55, 0.010, 0.30])
    fig.colorbar(im, cax=cax)
    out = os.path.join(FIGDIR, "02_superposition_20x5.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return summary


# ---------------------------------------------------------------- experiment 3


def experiment_3():
    """The headline scaling claim: features represented vs 1/(1-S)."""
    n_features, n_hidden = 20, 5
    importance = torch.tensor([0.9**i for i in range(n_features)])
    sparsities = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.999]

    xs, n_rep, total_D = [], [], []
    for S in sparsities:
        model, loss = train(n_features, n_hidden, S, importance, steps=12_000)
        W = model.W.detach().numpy()
        D = feature_dimensionality(W)
        alive = int((np.linalg.norm(W, axis=0) > 0.1).sum())
        xs.append(1.0 / (1.0 - S))
        n_rep.append(alive)
        total_D.append(D.sum())
        print(f"  [exp3] S={S:<6} 1/(1-S)={1/(1-S):<7.1f} represented={alive:>2}/20  sum(D_i)={D.sum():.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xs, n_rep, "o-", color="#3b6ea5", lw=2, label="features represented (of 20)")
    ax.axhline(n_hidden, ls="--", color="crimson", lw=1.4,
               label=f"n_hidden = {n_hidden} (the naive orthogonal limit)")
    ax.axhline(n_features, ls=":", color="gray", lw=1.0, label="all 20 features")
    ax.set_xscale("log")
    ax.set_xlabel("1 / (1 - sparsity)   —   roughly, how rarely a feature fires")
    ax.set_ylabel("number of features represented")
    ax.set_title(
        "Superposition breaks the dimension budget\n"
        "Sparse enough, and 5 dimensions carry far more than 5 features",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "03_capacity_vs_sparsity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return list(zip(sparsities, n_rep, total_D))


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    print("Experiment 1: 5 features -> 2 dimensions, sparsity sweep")
    experiment_1()
    print("\nExperiment 2: 20 features -> 5 dimensions, geometry + dimensionality")
    experiment_2()
    print("\nExperiment 3: capacity vs sparsity")
    experiment_3()
    print(f"\nDone. Figures in {FIGDIR}")

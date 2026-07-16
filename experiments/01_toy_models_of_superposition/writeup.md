# Experiment 01 — Toy Models of Superposition

**Paper:** Elhage et al., *Toy Models of Superposition* (Anthropic, 2022) — https://transformer-circuits.pub/2022/toy_model/index.html
**Code:** `toy_models.py` (CPU, ~90s). **Figures:** `figures/`.

## The question

A network has fewer neurons than there are things worth representing. So what does it do — pick a privileged subset of features and drop the rest, or somehow pack more features than it has dimensions? The paper's answer: **when features are sparse, the network packs more features than dimensions into the same space**, storing them along non-orthogonal directions and tolerating interference between features that rarely co-occur. That's *superposition*.

This is the cleanest possible demonstration, and it's why it's my first experiment: no pretrained model, no GPU, pure synthetic data, and the central concept of the whole field falls out of a 2-parameter linear-ish model.

## Setup

`n` sparse features pushed through an `m`-dimensional bottleneck (`m < n`), reconstructed with a tied-weight ReLU decoder:

```
h  = W x                 # (n -> m)  compress
x' = ReLU(Wᵀ h + b)      # (m -> n)  reconstruct
```

Each input feature is zero with probability `S` (the sparsity), else uniform on [0, 1). Loss is importance-weighted MSE. The only thing that varies across runs is `S`. The columns of `W` **are** the learned feature directions — read the geometry straight off the weights.

## What I ran and what came out

**1. Five features, two dimensions (`figures/01_feature_geometry_5x2.png`).** Two importance regimes side by side.

- *Dense (S=0):* only 2 of 5 features survive. With no sparsity there is no free lunch — the model keeps the 2 most important features on orthogonal axes and discards the other 3. This is the "no superposition" baseline, and it's exactly the naive dimension-counting prediction.
- *Sparse, uniform importance:* at S=0.9–0.99 all five features appear as a **regular pentagon** — five equal vectors at 72°. This is the paper's iconic figure, reproduced. Five features living in two dimensions, none privileged, interference spread symmetrically.
- *Sparse, decaying importance:* the pentagon breaks. The model triages — protects the top features and pairs the rest into antipodal (180°) pairs. Same mechanism, different symmetry, driven entirely by the importance weighting.

The pentagon-vs-pairs contrast is the whole lesson in one image: **the geometry the model chooses is a function of the loss landscape's symmetry, not a fixed rule.**

**2. Twenty features, five dimensions (`figures/02_superposition_20x5.png`).** Gram matrix `WᵀW` (top) and per-feature dimensionality `Dᵢ` (bottom) across the sparsity sweep.

- Dense: 7/20 features represented, sharp diagonal Gram matrix (near-orthogonal), and `Dᵢ ≈ 1` for the survivors — each owns a full dimension. No superposition.
- Sparse (S=0.99+): **all 20/20 features represented**, dense off-diagonal Gram structure (everyone interfering with everyone), and `Dᵢ ≈ 0.25` uniformly — 20 features sharing 5 dimensions, each owning a quarter.
- Throughout, `Σ Dᵢ ≈ 5 = m` at every sparsity. **The dimension budget is conserved; superposition changes how it's divided, not how much there is.** That invariant is the part I found most satisfying — it's a conservation law, and it held to two decimals across every run.

**3. Capacity vs sparsity (`figures/03_capacity_vs_sparsity.png`).** Features represented against `1/(1−S)`. Clean monotone climb from 5 (= `m`, the orthogonal limit, dashed red) at S=0 up to the full 20 as features get sparse. The dimension count is a floor, not a ceiling.

## Numbers (last run, 2026-07-16)

| S | 5→2: uniform | 20→5: represented | 20→5: Σ Dᵢ |
|---|---|---|---|
| 0.0 | 5/5 (but orthogonal pairs, low fidelity) | 7/20 | 5.00 |
| 0.9 | 5/5 pentagon | 11/20 | 5.00 |
| 0.99 | 5/5 pentagon | 19/20 | 4.98 |
| 0.997 | — | 20/20 | 4.99 |

(Small run-to-run wobble in exact counts near the `>0.1` alive-threshold; the trend is stable.)

## The bridge — why a comp-neuro person should care

This is **mixed selectivity** with the ground truth handed to you.

In primate cortex we constantly find single neurons that respond to combinations of task variables rather than one clean tuning dimension. In V1 the textbook story is one neuron ≈ one oriented Gabor, but the population reality is messier — cells multiplex, and normalization reshapes their joint tuning. The persistent question is whether that mixing is a nuisance (noise, incomplete sampling) or a computational choice (Rigotti et al. 2013 argue mixed selectivity buys the dimensionality a linear readout needs).

Toy Models is the same phenomenon in a system where you can settle the argument:

- The "neurons" (hidden units) are provably polysemantic — I can read the exact feature directions off `W` and watch them share axes.
- It's a computational choice: it appears **only** under sparsity, and it strictly improves the importance-weighted loss. Not an artifact — an optimum.
- The controlling variable is feature **sparsity**, i.e. how often each latent cause is actually present. That maps directly onto natural-scene statistics — the sparsity of active latent causes in natural images is exactly the regime Olshausen & Field 1996 built sparse coding of V1 around. Same math object, two fields, thirty years apart.

**Two framings to keep in tension** (carried into `learning-roadmap.md`): interp treats superposition as an *obstacle* — the thing SAEs exist to undo. Rigotti treats mixed selectivity as a *feature* — the thing that makes readout easy. A model where the ground truth is fully observable is the right place to ask which framing V1 actually lives in. That's the first real research thread this repo hands me.

## Caveats (honesty about what this is not)

- Toy model, not a transformer. It shows superposition *can* happen and *why*; it doesn't show GPT-2 does it this way. That evidence is the induction-head and IOI work later in the roadmap.
- "Features represented" uses a hard norm threshold (0.1); near the threshold the integer count jitters between runs. The dimensionality `Dᵢ` and the geometry are the robust readouts, not the exact count.
- No seed sweep yet. Single seed (0). A natural next step is to check the pentagon is the reliable attractor and not one lucky basin.

## Next

- Finer sparsity grid + multi-seed to map the phase boundaries (paper's phase-diagram figures).
- Push to `m=2, n=6..8` and watch which regular polygons appear — the discrete geometric phases (`1/2, 2/3, ...` dimensionality plateaus) are the deep part of the paper.
- Then leave toy land: Week 1 of the roadmap, write a transformer forward pass and start on real activations.

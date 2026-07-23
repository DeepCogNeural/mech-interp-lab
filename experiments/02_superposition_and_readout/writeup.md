# Experiment 02 — Mixed (superposed) coding has a computation reason, not just a storage reason

**Code:** `readout.py` (CPU, ~10–15 min; `SMOKE=1` for a ~40s subset). **Figures:** `figures/`. **Raw numbers:** `results.json`.

`results.json` schema: every `xor` row is `(m, S, seed, dp, arm, test_acc, train_acc)`.

## The question

Experiment 01 gives a *storage* reason for superposition: sparse features plus a bottleneck force the network to pack more features than it has dimensions. This experiment asks whether superposition also has a *computation* reason — and it is the question my neuroscience background makes natural.

Two literatures describe the same geometry (many features sharing few non-orthogonal directions) with opposite value judgements:

- **Interpretability, SAE lineage:** superposition is something to *undo*. You train a sparse autoencoder to disentangle the superposed directions so you can enumerate monosemantic features.
- **Systems neuroscience, mixed selectivity (Rigotti et al. 2013):** the same kind of geometry is *functional* — mixing is what buys a linear readout the dimensionality it needs to compute nonlinear functions of the inputs.

Both can be true, and the interpretability literature is not blind to the second point — Elhage et al. explicitly study *computation in superposition*. What a toy model adds is ground truth: I can build the geometries by hand and *measure* which framing the readout actually rewards.

## Why a nonlinearity is unavoidable — and why that is not the trivial part

Experiment 01's encoder is linear (`h = W x`). A linear probe on a linear projection is still linear in `x`, so it can **never** represent XOR, which needs the interaction term `x_i · x_j` — for *any* geometry. So a fair test needs a nonlinearity in the readout. I use `r = ReLU(W x)`.

The important move: the nonlinearity is held **constant across all three arms**. Only the geometry `W` changes. So the result is not "a nonlinearity lets you do XOR" (trivial). It is: under the *same* fixed nonlinearity, `ReLU(a single feature)` is still additive and reads out no interaction, whereas `ReLU(a mixture)` manufactures the equivalent cross-terms. **Geometry decides whether a linear readout can see the interaction.**

## The math anchor (the one place "provably" is earned)

Any *monosemantic* code — one where each unit is a function of a single input feature, `r_k = f_k(x_{i_k})` — cannot represent `XOR(a_i, a_j)` through a linear readout. `XOR(a, b) = a + b − 2ab` requires the product `ab`, and `ab` is not in the span of `{1, f(a), g(b)}`. This is a theorem, not a trained result — the contrast with the (deleted) over-claim in exp1 that a trained `W` was "an optimum" is deliberate.

The **strong** form of the anchor, which is what I test, is that this holds *even when the monosemantic code perfectly encodes both features*. So every XOR pair here is drawn from the features the monosemantic arm actually represents (indices `0..m-1`). Its chance-level XOR is the theorem, not a coverage artifact.

## Setup

Three geometry arms at fixed `(n=20, m=8)`, differing **only** in `W`:

| arm | `W` | role |
|---|---|---|
| monosemantic | selection matrix (feature `k` → axis `k`) | theorem-backed negative-control anchor |
| random | Gaussian, Frobenius-norm-matched to the learned `W` | a capacity ruler |
| superposition | the frozen `W` that exp1's storage objective trains at sparsity `S` | the research object |

Task: for a feature pair `(i, j)`, `a_i = 1[x_i > 0]`, label `y = a_i XOR a_j`, read with a **linear** logistic probe on `r = ReLU(W x)`. The eval distribution is **fixed** and class-balanced (the pair forced into the four quadrants `{00,01,10,11}`, every other feature a fixed `Bernoulli(distractor_p)·Uniform`), identical across every sparsity and geometry — `S` only changes the frozen `W`. 8 seeds; balanced accuracy; mean ± 95% CI; within-seed paired (superposition − random) differences.

## Result 1 (headline) — monosemantic can't, mixed can

Isolated pair (`distractor_p = 0`), `m = 8`, 8 seeds:

![Monosemantic XOR readout remains at chance while random and learned mixed codes reach about 0.80 accuracy](figures/01_monosemantic_cannot_read_xor.png)

Caption — with the same ReLU nonlinearity in every arm, only the mixed geometries make XOR linearly readable.

| arm | XOR readout accuracy (across `S`) |
|---|---|
| monosemantic | **0.494 ± 0.005** at every sparsity — flat on the chance line (flat by construction — this arm's W does not depend on S; the finding is the level, not the flatness) |
| random | 0.79 → 0.81 |
| superposition | 0.78 → 0.81 (0.815 peak at S=0.9; a dip to 0.74 at S=0.7 — see the note below) |

The monosemantic anchor is exactly at chance regardless of sparsity, as the theorem says. Both mixed codes read the interaction at ~0.80. The probe train-minus-test gap on the mixed arms is **+0.002**, so the 0.80 is not overfitting.

The interpretation is the one that matters for interpretability: **an exactly monosemantic
representation carries a computational cost.** One feature per unit maximizes enumerability and, in
that exact limit, leaves a linear readout unable to see feature interactions at all. Mixing is not just
a storage compromise; it is also what keeps a nonlinear interaction linearly legible downstream. That
is Rigotti's mixed-selectivity argument, here with fully observable ground truth. This is a claim about
representations, not about SAEs as a tool — see Caveats below.

## Result 2 (honest secondary) — the *learned* geometry has no reliable edge over random

Does the specific storage-optimized geometry beat random mixing of the same norm? Within-seed paired differences:

![Paired differences between learned superposition and equal-norm random mixing remain centered near zero](figures/02_superposition_vs_random_paired.png)

Caption — the learned geometry has no reliable readout advantage over equal-norm random mixing.

| | superposition − random |
|---|---|
| `distractor_p = 0.0` (pooled) | −0.015 ± 0.017 |
| `distractor_p = 0.05` (pooled) | +0.004 ± 0.008 |
| `distractor_p = 0.1` (pooled) | +0.002 ± 0.005 |
| robustness `m = 5 / 8 / 12` (dp=0) | −0.040 / −0.015 / +0.001 |

The differences hug zero; the 95% CIs include zero in essentially every cell. A 4-seed pilot had hinted at a small positive effect at high sparsity with background activity; at 8 seeds it washes out. **The honest conclusion is a null: the readout capacity comes from mixing-plus-nonlinearity itself, not from the particular geometry exp1 learns.** The headline (Result 1) does not depend on exp1 at all — random mixing suffices; exp1 simply supplies one storage-trained, ground-truth-known instance of a mixed code.

(The `S = 0.7`, `dp = 0` dip is a seed effect, and a broader one than a single outlier: the eight seeds
split bimodally, five at 0.63–0.73 and three at 0.82. Those `S = 0.7` codes are also the least mixed in
the sweep (16–19 of 20 features represented, the lowest off-diagonal Gram energy), but the two do not
track each other seed by seed, so I report the split without claiming a cause. Learned geometries carry
training variance; random and monosemantic, being constructed, do not.)

## Controls and checks

- **Strong anchor.** XOR pairs are drawn only from features the monosemantic arm represents (`0..m-1`), so its chance accuracy is the theorem, not missing coverage.
- **The "superposition" arm really is in superposition.** For the learned `W` at `m = 8`: features represented `= 16–20 > m = 8`, `Σ Dᵢ` is an effective-dimension upper bound that gets close to `m` only when the bottleneck is well used (7.9–8.0 here), and off-diagonal Gram energy goes from `0.085` at `S = 0` (near-orthogonal) to `0.252` at `S = 0.99`
  (dense interference), though not monotonically — `S = 0.7` is the least mixed point in the sweep, at
  `0.051`. The label is earned; sparsity moves interference across the range, but it is not a clean
  monotone knob.
- **Not overfit.** Probe train−test gap `+0.002` on the mixed arms.
- **Nonlinearity held constant** across all three arms — the manipulated variable is geometry alone.

## Enumeration nuance (why the tension is the hard one, not the easy one)

A tempting story is "mixing buys computation by sacrificing enumeration." The data do **not** support that clean tradeoff. Mean per-feature linear decodability (can a probe read individual feature indicators off `r`?) at `distractor_p = 0.1`: monosemantic `0.614`, random `0.596`, superposition `0.621` — the mixed superposed code is about as enumerable as the monosemantic one here. So the tension is not "mixing destroys enumeration." It is the sharper one: **mixing is necessary for linearly reading an interaction, and the most-enumerable limit (pure monosemanticity) provably cannot read one at all.**

## Caveats (what this is and is not)

- Toy model, not a transformer. It shows the mechanism cleanly; it does not show any real model represents or reads features this way.
- This result says that if a system's representation is truly monosemantic, a linear readout cannot see feature interactions; it does **not** say that an SAE harms a model's computation. An SAE is a readout lens alongside the residual stream: downstream model components still read the mixed residual stream, the SAE does not replace the model's representation with a monosemantic basis, and this experiment does not measure that question.
- XOR is one interaction. It is the minimal nonlinear-readout probe, not a claim about all computation.
- This is **compression** (`n → m`, `m < n`). Rigotti's mixed selectivity is **expansion** (few task variables → many mixed neurons). The geometries are cousins (non-orthogonal mixing) but the dimension direction is opposite; I test whether compression-style superposition stays computationally legible, not Rigotti's expansion regime.
- XOR readout accuracy is a task-specific proxy. The native mixed-selectivity metric is shattering dimensionality / CCGP (Bernardi et al. 2020) — a cleaner, task-agnostic measure. That is the next step.

## Next

- Shattering dimensionality / CCGP on these three geometries — the task-agnostic version of Result 1.
- A mechanism check for any residual superposition-vs-random effect: correlate each pair's `super − random` margin with the Gram interference among the active features.
- Leave toy land: ask the same enumeration-vs-computation question of real SAE features on a small transformer.

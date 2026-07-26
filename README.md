# mech-interp-lab

Self-directed mechanistic-interpretability research, run from a computational-neuroscience starting
point (probabilistic inference, stochastic modeling, primate V1 electrophysiology). Everything here is
a toy model with fully observable ground truth — CPU-only, reproducible end to end — chosen so that a
question can be *settled* instead of argued about. The through-line is a question systems neuroscience
has asked for a decade and interpretability now asks in its own vocabulary: **when many features share
few non-orthogonal directions, is that a problem to undo or a computation to explain?**

Two experiments so far. One replicates the *storage* account of superposition; the other tests whether
the same geometry also has a *computation* account, and finds that it does — under a nonlinearity held
fixed across arms, a monosemantic code reads a feature interaction at `0.494 ± 0.005`, exactly chance
and provably so, while mixed codes reach `0.81`. It also produced a null, reported below in full.

---

## Experiment 02 — a monosemantic code cannot linearly read a feature interaction; a mixed one can

[Full writeup](experiments/02_superposition_and_readout/writeup.md) · [code](experiments/02_superposition_and_readout/readout.py) · [per-seed results](experiments/02_superposition_and_readout/results.json)

The SAE lineage in interpretability treats superposed coding as something to undo — disentangle the
mixed directions so features can be enumerated. The mixed-selectivity literature in systems
neuroscience (Rigotti et al. 2013) treats the same geometry as *functional*: mixing is what buys a
linear readout the dimensionality it needs to compute nonlinear functions. Both can be true, and
interpretability is not blind to the second point — Elhage et al. study computation in superposition
directly. What a toy model adds is ground truth: I can build the geometries by hand and measure which
framing the readout actually rewards.

Three geometry arms at fixed `n = 20, m = 8`, differing **only** in the encoder `W` — monosemantic (a
selection matrix), random (Gaussian, Frobenius-norm-matched), and superposition (the frozen `W` that
experiment 01's storage objective learns). Each is read by a **linear** logistic probe on
`r = ReLU(W x)`, asked for the XOR of a feature pair. 8 seeds, balanced accuracy, a fixed
class-balanced eval distribution shared across every arm and sparsity, within-seed paired statistics.

![A linear readout of a monosemantic code sits exactly on the chance line at every sparsity, while the random and learned mixed codes both reach about 0.80 XOR accuracy](experiments/02_superposition_and_readout/figures/01_monosemantic_cannot_read_xor.png)

| arm | XOR readout accuracy (across sparsity `S`) |
|---|---|
| monosemantic | **0.494 ± 0.005** at every sparsity — exactly on the chance line |
| random | 0.79 → 0.81 |
| superposition | 0.78 → 0.81 (dip to 0.74 at `S = 0.7`; the eight seeds split bimodally there) |

The monosemantic arm is a **theorem**, not a trained result: `XOR(a, b) = a + b − 2ab` needs the
product term `ab`, and `ab` is not in the span of `{1, f(a), g(b)}`, so no code with one feature per
unit can carry an interaction to a linear readout. I test the *strong* form of it — every XOR pair is
drawn only from the features the monosemantic arm actually represents, so its chance accuracy is the
theorem and not missing coverage. What the experiment adds on top of the theorem is the quantitative
ruler (how much mixing buys, against an equal-norm random control) and the fact that the nonlinearity
is held fixed across all three arms, so the manipulated variable is geometry alone. `ReLU(a single
feature)` is still additive and reads out no interaction; `ReLU(a mixture)` manufactures the
cross-terms. The probe's train-minus-test gap on the mixed arms is `+0.002`, so 0.80 is not
overfitting.

The interpretability-relevant reading: **an exactly monosemantic representation carries a
computational cost.** One feature per unit maximizes enumerability and, in that exact limit, leaves a
linear readout unable to see feature interactions at all. And the tradeoff is *not* the tidy one you
might expect — mean per-feature decodability is 0.621 for the superposed code versus 0.614 for the
monosemantic one, so mixing did not destroy enumerability here. The tension is the sharper version:
mixing is necessary to linearly read an interaction, and the most-enumerable limit provably cannot
read one at all.

### The null, stated plainly

Does the *storage-learned* geometry beat equal-norm random mixing? **No.** Within-seed paired
differences (superposition − random) are `−0.015 ± 0.017` at `distractor_p = 0`, `+0.004 ± 0.008` and
`+0.002 ± 0.005` with background activity; the 95% CIs include zero in essentially every cell. A
4-seed pilot had hinted at a small positive effect at high sparsity; **at 8 seeds it washed out, and I
am reporting the null rather than the pilot.**

That is a discriminating answer, not a failed run. The readout capacity comes from
mixing-plus-nonlinearity itself, not from the particular geometry experiment 01 learns: the storage
objective shapes *which* mixed geometry appears, but the computational benefit is a generic property of
mixing rather than something that geometry is optimized for. The headline above does not depend on this
— random mixing suffices; experiment 01 just supplies one storage-trained instance of a mixed code
whose ground truth I know.

### The probe design trap

My first instinct was the obvious one: run a linear probe directly on experiment 01's encoder,
`h = W x`. That design is **identically at chance for any geometry whatsoever** — a linear readout of a
linear projection is still linear in `x`, and XOR needs the interaction term. I would have gotten a
real null for an entirely boring reason, and the geometry question would have been untouched.

The fix is the actual content of the design: a nonlinearity is unavoidable, *and* it has to be held
constant across the arms. With `r = ReLU(W x)` fixed everywhere and only `W` varying, the result cannot
be "a nonlinearity lets you do XOR" (trivially true). It becomes the claim I wanted to test —
**geometry decides whether a linear readout can see the interaction.** The negative control (a
theorem-backed arm that must land at chance) is what makes the positive result readable at all.

### Scope — what this does not say

This is a claim about **representations**, not about SAEs as a tool. It does **not** say an SAE harms a
model's computation. An SAE is a readout lens alongside the residual stream: downstream components
still read the mixed residual stream, the SAE does not replace the model's representation with a
monosemantic basis, and this experiment does not measure that question at all. Also: a toy model, not a
transformer; XOR is one minimal interaction, not all computation; and this is compression
(`n → m`, `m < n`) whereas Rigotti's mixed selectivity is expansion — cousin geometries, opposite
dimension direction.

---

## Experiment 01 — the storage account, replicated

[Full writeup](experiments/01_toy_models_of_superposition/writeup.md) · [code](experiments/01_toy_models_of_superposition/toy_models.py)

A replication of Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html):
when features are sparse, a tied-weight ReLU autoencoder packs more features than it has dimensions,
storing them along non-orthogonal directions and tolerating interference between features that rarely
co-occur. Five features in two dimensions give
[the pentagon](experiments/01_toy_models_of_superposition/figures/01_feature_geometry_5x2.png) under
uniform importance and antipodal pairs under decaying importance — the geometry is set by the loss
landscape's symmetry, not by a fixed rule. Twenty features in five dimensions go from 7/20 represented
when dense to all 20/20 by `S ≈ 0.997`.

![Number of represented features climbing from the five-dimensional orthogonal limit up to all 20 as feature sparsity increases](experiments/01_toy_models_of_superposition/figures/03_capacity_vs_sparsity.png)

The dimension count is a floor, not a ceiling. Per-feature dimensionality `Dᵢ` sums to an
effective-dimension upper bound that approaches `m` only when training uses the bottleneck fully:
superposition redistributes the available dimension budget rather than manufacturing new dimensions.
Single seed, and the integer "features represented" count jitters near the norm threshold — the
geometry and `Dᵢ` are the robust readouts, not the exact count.

This experiment supplies the object experiment 02 tests, and the bridge that motivates it: same
non-orthogonal mixing that systems neuroscience calls mixed selectivity, with the ground truth handed
to you.

---

## Next

- **Shattering dimensionality / CCGP** (Bernardi et al. 2020) on these three geometries — the
  task-agnostic version of the headline. XOR accuracy is a task-specific proxy; CCGP is the native
  mixed-selectivity measure and does not require picking an interaction.
- Then leave toy land: ask the same enumeration-versus-computation question of **real SAE features on a
  small transformer**, with shattering dimensionality / CCGP as the metric.
- A mechanism check on any residual superposition-versus-random effect: correlate each pair's margin
  with the Gram interference among its active features.

## Reproduce

Python 3.11 (torch has no 3.14 wheels yet). CPU-only, no model downloads.

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
(cd experiments/01_toy_models_of_superposition && python toy_models.py)   # ~1-2 min
```

```bash
(cd experiments/02_superposition_and_readout && python readout.py)        # ~10-15 min; SMOKE=1 for a ~40s subset
```

Each `experiments/NN_*/writeup.md` is self-contained: setup, results, controls, and what the result is
not. [`lab-notebook.md`](lab-notebook.md) is the dated process record, including the traps I nearly
fell into and the pilot-versus-full-run story behind the null.
[`learning-roadmap.md`](learning-roadmap.md) is where this is going next.

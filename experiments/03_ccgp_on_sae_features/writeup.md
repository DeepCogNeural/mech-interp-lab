# Experiment 03 — Shattering dimensionality × CCGP on real GPT-2-small SAE features

**Code:** `ccgp_sae.py` (offline CPU; five seeds; `SMOKE=1` for a plumbing subset). **Raw result:** `results.json`. **Figures:** `figures/`.

`results.json` uses schema `exp03-results-v3`. Each `per_seed_rows` entry is `{seed, arm, sd, ccgp, gap, ...}`, where each metric maps a dichotomy type to balanced accuracy; `probe_fairness` and `legacy_fixed_standardise` hold the same rows under the other two probe settings. **Every number below is recomputed from those raw rows** as a two-sided 95% Student-t interval, `t(4) = 2.776` on five seeds. (Experiments 01–02 used a `1.96 × sd/√n` normal approximation with 8 seeds; at 5 seeds that convention is about 29% too narrow, so this experiment uses t and flags the inconsistency rather than hiding it.)

## The question

Experiment 02 showed that a monosemantic code — one unit per feature — provably cannot let a linear readout see `XOR(a_i, a_j)`, while mixed codes reach ~0.80. Two of its caveats named this experiment: XOR accuracy is a task-specific proxy where shattering dimensionality / CCGP (Bernardi et al. 2020) is the task-agnostic measure, and the whole thing was a toy model.

The naive extrapolation — "an SAE is monosemantic, so SAE features should fail the same way" — **is wrong, and saying why is part of the point.** Experiment 02's monosemantic arm was coordinate-wise *and* a compression (20 → 8). A real SAE encoder is an over-complete nonlinear *expansion* (768 → 24,576 with a ReLU), which by Cover's theorem raises linear separability on its own. The direction reverses.

So the informative comparison is not `sae` versus `resid` — that mostly rediscovers Cover. It is **a real sparse SAE code versus a random ReLU expansion matched in width, encoder column-norm distribution, and mean active-feature count (L0)**. Two forces oppose there: the sparsity objective pushes each latent toward a clean single-concept detector, which would lower shattering; SAEs are also known to learn conjunctive latents, which hand a linear readout the product term directly and would raise it. No direction was pre-registered.

## Scope: this is a lens, not a model intervention

An SAE is a read-out lens hung beside the residual stream. GPT-2's downstream components keep reading the original mixed residual stream. This experiment never substitutes SAE features into the forward computation, never ablates them, and never measures a change in model behaviour. Every score below is a property of **a code under a probe** — not evidence that an SAE harms, degrades, removes, or loses any computation the model uses.

## Setup

| arm | representation | role |
|---|---|---|
| `resid` | GPT-2 residual activation `x` at layer 8 | the model's own mixed code |
| `sae` | `ReLU((x − b_dec) @ W_enc + b_enc)` | the research object |
| `sae_recon` | `f @ W_dec + b_dec` | round-trip control |
| `rand_exp` | random ReLU expansion, per-sample top-k | **the load-bearing control**: width/norm/L0-matched |
| `rand_exp_dense` | the same expansion without top-k | Cover's-theorem upper reference, not the control |
| `rand_exp_width_matched` | random expansion widened to the SAE's *surviving* width | effective-width sensitivity |
| `sae_width_matched` | SAE latents subsampled to `rand_exp`'s surviving width | the reciprocal sensitivity |

Each lexical item instantiates NUMBER (singular/plural) × TENSE (past/present) × POLARITY (affirmative/negated), giving the eight conditions a shattering/CCGP analysis needs:

```
The {ADJ} {NOUN} {AUX} {POL} {VERB} the {ADJ2} {OBJ} {ADV} .
```

**The read-out token is the sentence-final `.`, identical across all eight cells.** Do-support keeps the main verb bare so tense never marks it, and `indeed`/`not` fill the polarity slot so negation costs no extra token. Probing the auxiliary or the verb would decode word form rather than a carried representation — every arm would inflate, and the SAE most of all, since it has token-identity latents. Every retained item was additionally asserted to tokenise to equal length with an identical final-token id across its eight cells, so the read-out position index cannot leak a factor through positional embeddings.

96 items × 8 cells = 768 sequences per seed; five seeds varying the lexical draw, the random projection, and probe init; five item-disjoint folds, so a lexical nuisance draw never appears in both train and test. All 35 balanced dichotomies (`C(8,4)/2`), and CCGP over all 16 held-condition splits. SAE weights are loaded directly from the published `jbloom/GPT2-Small-SAEs-Reformatted` res-jb safetensors; `sae_lens` was not installed or used.

## Gates

| gate | measured | decision |
|---|---|---|
| A: loading | layer-8 SAE relative reconstruction error 0.286, MSE 1.086, mean L0 71.87 | pass |
| B: stimuli and base factors | 96/96 items retained, 0 dropped, every seed; pilot NUMBER/TENSE/POLARITY 0.993–1.000 at layers 6–9 | pass; layer 8 selected |
| C: headroom | residual two-way-XOR SD 0.827 — not saturated, so there is room to measure a difference | pass |

The across-stimulus-centered explained variance is **−0.660**, which looks alarming and is not the standard SAE metric. Its denominator contains only variance *across this deliberately narrow stimulus set*, which is tiny; the directly measured relative reconstruction error of 0.286 is the meaningful figure. It is reported so the value in `results.json` is not mistaken for a broken SAE.

## Result 1 (headline) — the comparison is not adjudicated, because it is scaling-sensitive

Paired within-seed `sae − rand_exp`, under three probe settings that differ only in how features are scaled and how L2 is chosen:

| probe setting | main-effect SD | **two-way-XOR SD** | overall SD |
|---|---:|---:|---:|
| fixed L2, per-feature z-score | −0.073 ± 0.026 | **+0.009 ± 0.022** | −0.015 ± 0.024 |
| nested-L2, per-feature z-score | −0.070 ± 0.027 | **+0.011 ± 0.022** | −0.013 ± 0.025 |
| nested-L2, global RMS | +0.010 ± 0.002 | **+0.121 ± 0.015** | +0.079 ± 0.009 |

Under global-RMS scaling the SAE reads two-way interactions substantially better than the matched random expansion. Under per-feature z-scoring it does not — the interval covers zero.

**That difference cannot be a property of the codes.** For a *linear* probe, per-feature z-scoring is an invertible affine reparameterisation of the feature space: the set of achievable decision boundaries is identical. What is *not* invariant is the L2 penalty and the optimiser's path through that geometry. So the swing measures the probe's inductive bias, not the representation. **The honest headline is that this experiment does not adjudicate whether a real SAE code has an interaction-readout advantage over matched random mixing.**

![Shattering dimensionality against main-effect CCGP for all seven arms, full scale with chance lines plus a zoomed panel](figures/01_shattering_vs_ccgp.png)

Caption — arm positions under the global-RMS setting only; read this as one setting's geometry, not as an adjudicated comparison. The x-axis is **main-effect CCGP**, not CCGP pooled over all 35 dichotomies: pooled CCGP is 0.500 ± 0.000 for every arm by construction, because 28 of the 35 dichotomies have no abstract structure to transfer and the parity-family ones sit at or below chance.

## Result 2 — the sensitivity is localised, and that is diagnostic

The scaling does not perturb every arm. Overall SD under the two scalings:

| arm | nested-L2 global RMS | nested-L2 z-score | shift |
|---|---:|---:|---|
| `resid` (768, dense) | 0.902 ± 0.009 | 0.901 ± 0.009 | none |
| `sae_recon` (768, dense) | 0.877 ± 0.009 | 0.877 ± 0.010 | none |
| `sae` (~800 surviving, sparse) | 0.888 ± 0.006 | 0.718 ± 0.020 | −0.170 |
| `rand_exp` (~500 surviving, sparse) | 0.809 ± 0.008 | 0.731 ± 0.014 | −0.078 |
| `rand_exp_dense` (~20,000) | 0.936 ± 0.007 | 0.814 ± 0.003 | −0.122 |

The two dense 768-dimensional arms are **completely insensitive** to the scaling; every sparse or very wide arm moves a lot. That is exactly what the affine-reparameterisation argument predicts: the reachable function class never changes, but the L2 geometry and the conditioning of the optimisation change most where per-feature scales are most heterogeneous — and a sparse code with heavy-tailed activation frequencies is the extreme case. A latent firing on 1% of stimuli is rescaled by z-scoring into a large-magnitude direction the probe can overfit, which shows up in the train−test gaps: +0.092 ± 0.009 for `resid` under both scalings, but +0.111 ± 0.006 (global RMS) against +0.282 ± 0.020 (z-score) for `sae`.

So the finding is not "the numbers were unstable." It is: **probe preprocessing that is harmless for dense representations is decisive for sparse over-complete ones, and any SAE-versus-baseline decoding comparison that does not state its scaling convention is under-determined.** That is a methodological result about how to measure SAE codes, and it is the part of this experiment I would defend.

![Shattering dimensionality by dichotomy type across the seven arms](figures/02_dichotomy_breakdown.png)

Caption — global-RMS setting. The dense random arm has the most shattering, exactly as Cover's theorem predicts when ~12,000 coordinates fire per sample; that is why the L0-matched arm, not the dense one, is the control the argument would rest on.

| arm | main effect | two-way XOR | three-way parity | unstructured |
|---|---:|---:|---:|---:|
| `resid` | 0.998 ± 0.001 | 0.849 ± 0.023 | 0.628 ± 0.019 | 0.907 ± 0.009 |
| `sae` | 0.979 ± 0.002 | 0.845 ± 0.013 | 0.660 ± 0.009 | 0.890 ± 0.006 |
| `sae_recon` | 0.984 ± 0.004 | 0.819 ± 0.018 | 0.622 ± 0.018 | 0.881 ± 0.009 |
| `rand_exp` | 0.969 ± 0.002 | 0.723 ± 0.010 | 0.571 ± 0.018 | 0.809 ± 0.009 |
| `rand_exp_dense` | 0.996 ± 0.003 | 0.918 ± 0.012 | 0.741 ± 0.033 | 0.939 ± 0.007 |
| `rand_exp_width_matched` | 0.972 ± 0.004 | 0.763 ± 0.012 | 0.576 ± 0.014 | 0.834 ± 0.005 |
| `sae_width_matched` | 0.968 ± 0.008 | 0.798 ± 0.008 | 0.613 ± 0.011 | 0.857 ± 0.007 |

`sae_recon` tracks `resid` closely on every family. That is the round-trip control doing its job: the information survives the SAE encode/decode, so whatever differs between `sae` and `rand_exp` lives in the sparse feature space rather than in information the lens discarded.

## The effective-width sensitivity (global-RMS only — not evidence about the codes)

The un-matched arms differ in the one quantity most likely to drive shattering: `sae` has ~790 latents that ever fire on this stimulus set while `rand_exp` has ~500, even though per-sample L0 is matched at ~72. Two controls tested that, both under the global-RMS setting:

| control | main-effect SD | two-way-XOR SD |
|---|---:|---:|
| widen random to the SAE's surviving width | +0.007 ± 0.005 | +0.081 ± 0.008 |
| narrow the SAE to `rand_exp`'s surviving width | −0.001 ± 0.008 | +0.075 ± 0.016 |

Both shrink the un-matched +0.121 without erasing it, and the narrowed-SAE control matches base-factor decodability exactly (−0.001 ± 0.008) while *handicapping* the SAE on L0, which drops to ~45 against 72. Read in isolation that is suggestive.

**It does not rescue the headline.** These ran only under the one scaling that produces an effect, so they inherit the open question rather than answering it. They also do not match cleanly: neither control matches width and L0 simultaneously, and the random arm's per-sample L0 is a forced top-k while the SAE's is a natural ReLU, so their L0 *distributions* differ even when the means agree. They are two sensitivity analyses with different costs that happen to agree in direction — not a demonstration that an active-direction account is excluded.

## Controls and matching

| quantity | across the five seeds |
|---|---|
| retained / dropped lexical items | 96 / 0 every seed |
| token-length equality across the 8 cells | asserted per item; identical final-token id |
| SAE mean L0 | 70.55–72.35 |
| random top-k achieved L0 | exactly the rounded per-seed SAE L0 (71 or 72) |
| surviving width, `sae` | 784–803 |
| surviving width, `rand_exp` | 464–559 |
| surviving width, `resid` / `sae_recon` | 768 / 768 |
| train − test gap, `resid` | +0.092 ± 0.009 |
| train − test gap, `sae` / `rand_exp` (global RMS) | +0.111 ± 0.006 / +0.189 ± 0.007 |

Both expansions are nominally 24,576-wide, but only ~800 and ~500 units respectively ever fire here, so the effective comparison is never 24,576 against 24,576. That is a property of a narrow templated stimulus distribution.

**Two leakage caveats, stated precisely.** The all-samples-zero keep-mask and the effective-width calibration are computed over the whole dataset before the fold split. Neither uses labels, so no label information crosses the split — but an earlier draft's claim that "outer test items are never used for selection" was too strong and is withdrawn. Separately, CCGP's held-out conditions are excluded from each head's loss but are inside the data used to fit that head's centring and scale, so this is a slightly weaker hold-out than the strict definition.

Both are known defects with known fixes, and neither is fixed in the shipped `ccgp_sae.py`. That is deliberate: **the script in this directory is exactly the revision that produced the shipped `results.json`**, so the code, the raw rows, and every number in this writeup are one self-consistent artifact. Applying the fixes would invalidate the shipped numbers until a full run completes, and that run is listed under Next rather than claimed here.

## What actually ran

The complete five-seed run behind every number above took **1,190.8 seconds (19 min 51 s)** on CPU (Apple M1 Pro; `torch.backends.mps.is_available()` is false under the execution sandbox, so this is CPU-only). Nothing was reduced: 96 items, 5 seeds, 5 folds, all 35 dichotomies, all 16 CCGP splits, three probe settings.

An earlier attempt was blocked at Gate A because the execution sandbox had no DNS for HuggingFace; the assets were cached locally and the run repeated. A later attempt to fit every probe to a strict convergence criterion **did not complete** — inner-L2 selection found no stable candidate on the dense-random arm — and produced no shippable manifest. **No number in this document comes from that attempt.**

## Caveats

- **The headline comparison is not adjudicated.** What would settle it: a five-seed run where every headline arm uses a probe fitted to a stated convergence criterion, with L2 selected item-disjointly per arm *and* per scaling on an interior grid, and both scaling estimates individually precise. Interval overlap would not be enough — two estimates agree when both are precise and close, not when one is precise and the other is too noisy to exclude anything.
- One English template, five lexical draws, one layer, one model, one SAE. Nothing here generalises to GPT-2's representations at large.
- Shattering dimensionality and CCGP describe code geometry under linear probes. They identify no circuit, no causal feature use, and no behavioural effect.
- The only theorem this project cites is experiment 02's coordinate-wise compression result. Nothing measured here earns that word.
- Gemma-2 / GemmaScope and additional stimulus families are future work, not substitutes.

## Next

- Finish the convergence test. It is the single thing standing between this and an adjudicated result. It should land together with four fixes already identified and not yet run: probes fitted to a stated convergence criterion rather than a fixed step count; CCGP centring/scale fitted on each head's six training conditions only; the keep-mask and width calibration fitted per outer fold; and the random-expansion width sweep taking column prefixes of a single draw, so width is the only thing that varies across candidates.
- If it adjudicates positively, the mechanism question opens: are conjunctive latents responsible? That needs a preregistered feature-level analysis, not a post-hoc hunt.
- Independent template families, to see whether any of this survives a change of stimulus distribution.
- A causal design — patching SAE features and measuring model behaviour — is a different experiment with a different scope statement, and this one licenses no claims about it.

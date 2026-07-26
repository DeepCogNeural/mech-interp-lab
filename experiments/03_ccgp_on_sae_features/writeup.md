# Experiment 03 — Shattering dimensionality × CCGP on GPT-2-small SAE features

**Code:** `ccgp_sae.py` (offline CPU; default five-seed fairness run). **Raw result:** `results.json`. **Figures:** `figures/`.

`results.json` uses schema `exp03-results-v2`. It preserves the completed pre-control fixed-L2 run and records two nested-L2 probe settings. Every new per-seed row has `{probe_setting, seed, arm, sd, ccgp, gap, selected_weight_decay_by_outer_fold, inner_validation_l2_selection}`; `ci95 = 1.96 × sample_sd / sqrt(n)` across five seeds. The primary `summary` is the global-RMS setting described below.

## The question

Experiment 02's monosemantic arm was coordinate-wise *and* a compression. Its theorem-backed XOR failure therefore does not extrapolate to a real sparse autoencoder. Here GPT-2's residual stream is 768-wide while the SAE encoder is a 24,576-wide ReLU expansion: width alone can improve linear separability.

The informative comparison is consequently not `sae` versus `resid`. It is a sparse SAE code versus a random ReLU expansion matched in encoder width, column-norm distribution, and mean active-feature count (L0). Sparse-factor pressure could favour clean factor coding; conjunctive SAE features could instead favour interaction readout. A null is as informative as an edge.

## Scope: this is a lens, not a model intervention

An SAE is a read-out lens hung beside the residual stream. GPT-2's downstream components continue to read the original mixed residual stream: this experiment never substitutes SAE features into the forward computation, ablates them, or measures a change in model behaviour. Therefore every score below is a property of a code under these probes, not evidence that an SAE harms, degrades, removes, or loses computation used by the model.

## Setup

| arm | representation | role |
|---|---|---|
| `resid` | GPT-2 residual activation `x` | mixed-code reference |
| `sae` | `ReLU((x − b_dec) @ W_enc + b_enc)` | research object |
| `sae_recon` | `f @ W_dec + b_dec` | round-trip control |
| `rand_exp` | random ReLU expansion with per-sample top-k | width/norm/L0-matched control |
| `rand_exp_dense` | the same random expansion without top-k | dense upper reference, not the primary comparison |

Each lexical item instantiates NUMBER (singular/plural) × TENSE (past/present) × POLARITY (affirmative/negated):

```
The {ADJ} {NOUN} {AUX} {POL} {VERB} the {ADJ2} {OBJ} {ADV} .
```

The sentence-final `.` is the only read-out token. Do-support keeps the main verb bare, and `indeed`/`not` occupy the polarity slot. I probe neither the auxiliary nor the verb: that would read word form rather than a carried representation. Every retained item had equal GPT-2 token length and the identical final-token id across its eight cells; probes use item-disjoint splits, so a lexical draw cannot be in both train and test.

The full run used 96 items × 8 cells = 768 sequences per seed, five lexical/random/probe seeds, five item-disjoint folds, all 35 balanced dichotomies, and all 16 CCGP held-condition splits. The primary probe centres on each training split then divides every surviving unit by one **global training-RMS scalar**. For each arm and outer fold it selects AdamW L2 from `{0.01, 0.03, 0.08, 0.2, 0.5}` using an item-disjoint inner validation subset of the outer training items, scored on the three main effects; the outer test items are never used for selection.

## Gates A–C

The run used `/Users/linghao/Github/mech-interp-lab/.venv/bin/python` on CPU with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `MPLBACKEND=Agg`. GPT-2 small resolved from the local Hugging Face cache. The SAE was loaded directly from the published `jbloom/GPT2-Small-SAEs-Reformatted` res-jb safetensors; `sae_lens` was not installed or used.

| gate | measured result | decision |
|---|---|---|
| A: loading | layer-8 SAE: across-stimulus-centered EV **−0.660** *(not the standard SAE EV; see note)*, MSE 1.086, relative reconstruction error 0.286, mean L0 71.87 | pass |
| B: stimuli and base factors | all five seeds retained 96/96 items; 0 dropped; pilot NUMBER / TENSE / POLARITY accuracy at layers 6–9 was 0.993–1.000 | pass; layer 8 selected |
| C: headroom | residual two-way-XOR SD = 0.827, below the 0.98 saturation threshold | pass |

The across-stimulus-centered denominator contains only variation across this deliberately narrow stimulus set, so its value is negative even though the directly measured relative reconstruction error is 0.286. It is not the standard SAE reconstruction-EV metric. Layer-8 pilot main-effect accuracies were NUMBER 1.000, TENSE 1.000, and POLARITY 0.996; the other pilot layers were layer 6: 0.993 / 0.999 / 1.000; layer 7: 0.999 / 1.000 / 0.995; and layer 9: 0.999 / 1.000 / 0.995.

## Result 1 — the original headline axis was degenerate; the fair primary does not put SAE in the low-expressivity/high-abstraction corner

![Main-effect CCGP versus overall shattering dimensionality for the five arms](figures/01_shattering_vs_ccgp.png)

Caption — the x-axis is **main-effect CCGP**, not pooled overall CCGP. Under the global-RMS, nested-L2 probe, the SAE is above the matched sparse random expansion on overall SD (**+0.079 ± 0.006**) and has no reliable main-effect CCGP edge (**+0.009 ± 0.013**). It therefore does **not** occupy a low-expressivity/high-abstraction corner on these stimuli; nor is this a trade-off story.

| arm | overall SD | main-effect SD | main-effect CCGP | overall train − test gap |
|---|---:|---:|---:|---:|
| `resid` | 0.902 ± 0.006 | 0.998 ± 0.001 | 0.953 ± 0.006 | +0.092 ± 0.006 |
| `sae` | 0.888 ± 0.004 | 0.979 ± 0.001 | 0.918 ± 0.006 | +0.111 ± 0.004 |
| `sae_recon` | 0.877 ± 0.007 | 0.984 ± 0.003 | 0.952 ± 0.005 | +0.084 ± 0.005 |
| `rand_exp` | 0.809 ± 0.006 | 0.969 ± 0.001 | 0.909 ± 0.008 | +0.189 ± 0.005 |
| `rand_exp_dense` | 0.936 ± 0.005 | 0.996 ± 0.002 | 0.929 ± 0.005 | +0.063 ± 0.005 |

The rebuilt figure's coordinates are therefore `(main-effect CCGP, overall SD)`: `resid` `(0.953, 0.902)`, `sae` `(0.918, 0.888)`, `sae_recon` `(0.952, 0.877)`, `rand_exp` `(0.909, 0.809)`, and `rand_exp_dense` `(0.929, 0.936)`. Bars are 95% CIs on both axes; the horizontal and vertical chance lines remain at 0.5.

Overall CCGP belongs in a table, not this figure:

| arm | CCGP across all 35 dichotomies |
|---|---:|
| `resid` | 0.500 ± 0.000 |
| `sae` | 0.500 ± 0.000 |
| `sae_recon` | 0.500 ± 0.000 |
| `rand_exp` | 0.500 ± 0.000 |
| `rand_exp_dense` | 0.500 ± 0.000 |

This pooled average is ~0.5 by construction: 28 of 35 dichotomies have no abstract structure to transfer, and the parity-family dichotomies are at or below chance (for example, primary-probe residual three-way-parity CCGP is 0.038 ± 0.005), so pooling them erases the factor-abstraction signal.

## Result 2 — SD breakdown under the fair primary

![SD breakdown separates readily enumerable factors from two-way XOR, three-way parity, and unstructured dichotomies](figures/02_dichotomy_breakdown.png)

Caption — after global scaling, SAE is above the matched sparse random expansion on every displayed SD family. This is a result about the selected probe/code pair, not a conclusion about GPT-2's downstream computation.

| arm | main-effect SD | two-way XOR SD | three-way parity SD | unstructured SD |
|---|---:|---:|---:|---:|
| `resid` | 0.998 ± 0.001 | 0.849 ± 0.016 | 0.625 ± 0.013 | 0.907 ± 0.007 |
| `sae` | 0.979 ± 0.001 | 0.845 ± 0.009 | 0.660 ± 0.006 | 0.889 ± 0.004 |
| `sae_recon` | 0.984 ± 0.003 | 0.819 ± 0.012 | 0.620 ± 0.014 | 0.877 ± 0.007 |
| `rand_exp` | 0.969 ± 0.001 | 0.723 ± 0.007 | 0.571 ± 0.012 | 0.807 ± 0.006 |
| `rand_exp_dense` | 0.996 ± 0.002 | 0.918 ± 0.009 | 0.740 ± 0.024 | 0.941 ± 0.005 |

Within-seed `sae − rand_exp` differences under the primary probe are **+0.079 ± 0.006** overall SD, **+0.010 ± 0.002** main-effect SD, and **+0.121 ± 0.011** two-way-XOR SD. The original `+0.009 ± 0.016` two-way-XOR null therefore does *not* survive the fairer scaling control. The interaction comparison is still not cleanly factor-matched: the small main-effect SD difference is reliably nonzero, even though its main-effect CCGP difference is only `+0.009 ± 0.013`.

## Is this a probe artifact?

The original SAE-versus-random headline is probe-sensitive. The hypothesis was that per-feature standardisation upweights rare SAE latents and distorts the sparse arms unequally. The control changes the relevant arm ordering materially, so global-RMS scaling with nested L2 selection is the primary result; the old fixed-0.08 numbers are sensitivity analysis, not the headline. This supports the artifact concern, but does not isolate activation frequency from every other consequence of rescaling and therefore does not prove that exact mechanism.

| probe setting | arm | overall SD | main-effect SD | main-effect CCGP | overall train − test gap |
|---|---|---:|---:|---:|---:|
| fixed L2 0.08, per-feature z-score (legacy) | `resid` | 0.901 ± 0.007 | 0.998 ± 0.001 | 0.947 ± 0.004 | +0.093 ± 0.006 |
|  | `sae` | 0.716 ± 0.013 | 0.787 ± 0.014 | 0.731 ± 0.009 | +0.284 ± 0.013 |
|  | `sae_recon` | 0.876 ± 0.007 | 0.983 ± 0.003 | 0.946 ± 0.005 | +0.092 ± 0.005 |
|  | `rand_exp` | 0.731 ± 0.010 | 0.860 ± 0.009 | 0.803 ± 0.011 | +0.268 ± 0.010 |
|  | `rand_exp_dense` | 0.810 ± 0.003 | 0.937 ± 0.006 | 0.859 ± 0.007 | +0.190 ± 0.003 |
| nested L2, per-feature z-score | `resid` | 0.901 ± 0.006 | 0.998 ± 0.001 | 0.955 ± 0.007 | +0.093 ± 0.006 |
|  | `sae` | 0.718 ± 0.014 | 0.790 ± 0.015 | 0.735 ± 0.007 | +0.282 ± 0.014 |
|  | `sae_recon` | 0.877 ± 0.007 | 0.984 ± 0.004 | 0.951 ± 0.005 | +0.087 ± 0.006 |
|  | `rand_exp` | 0.731 ± 0.010 | 0.860 ± 0.010 | 0.805 ± 0.011 | +0.267 ± 0.010 |
|  | `rand_exp_dense` | 0.814 ± 0.002 | 0.940 ± 0.005 | 0.864 ± 0.005 | +0.186 ± 0.002 |
| nested L2, **global RMS** (primary) | `resid` | 0.902 ± 0.006 | 0.998 ± 0.001 | 0.953 ± 0.006 | +0.092 ± 0.006 |
|  | `sae` | 0.888 ± 0.004 | 0.979 ± 0.001 | 0.918 ± 0.006 | +0.111 ± 0.004 |
|  | `sae_recon` | 0.877 ± 0.007 | 0.984 ± 0.003 | 0.952 ± 0.005 | +0.084 ± 0.005 |
|  | `rand_exp` | 0.809 ± 0.006 | 0.969 ± 0.001 | 0.909 ± 0.008 | +0.189 ± 0.005 |
|  | `rand_exp_dense` | 0.936 ± 0.005 | 0.996 ± 0.002 | 0.929 ± 0.005 | +0.063 ± 0.005 |

The nested-L2 z-score setting reproduces the fixed-0.08 ordering, so selecting L2 alone does not rescue it. Switching to one global scale does. The primary 100-step versus 200-step diagnostic changes inner-validation main-effect accuracy by at most 0.0033 for any arm (SAE: 0.976 ± 0.007 at 100 versus 0.973 ± 0.007 at 200; random sparse: 0.966 ± 0.010 versus 0.963 ± 0.010), while training BCE continues to fall slightly. Thus 100 steps is adequate for the reported validation accuracy at this precision, though not literally at zero optimizer loss.

## Controls and matching

| quantity | result across the five seeds |
|---|---|
| retained / dropped lexical items | 96 / 0 per seed |
| SAE L0 | mean 71.80 (per-seed range 70.55–72.35) |
| random top-k target and achieved L0 | target 71 or 72 per seed; achieved exactly 71 or 72; mean 71.80 |
| dense random reference L0 | mean 12,085.70 (range 11,997.10–12,155.37) |
| surviving width: `resid` / `sae_recon` | 768 / 768 every seed |
| surviving width: `sae` | 784–803 |
| surviving width: `rand_exp` | 464–559 |
| surviving width: `rand_exp_dense` | 19,986–20,248 |

Both sparse expansions are nominally 24,576-wide, but this narrow templated stimulus distribution leaves only about **800 SAE** versus **470 random sparse** units active after all-samples-zero removal. The effective comparison is therefore not 24,576 active units against 24,576; that difference is a property of this stimulus family and remains a caveat even after the scaling control.

## What actually ran

The new five-seed fairness run completed in **1,419.7 seconds (23 minutes 39.7 seconds)** on CPU. It ran global-RMS nested-L2 SD and all-35-dichotomy CCGP for the primary result, plus the required nested-L2 z-score SD and main-effect CCGP sensitivity analysis. The completed pre-control full run remains in `results.json` as a labelled fixed-L2 sensitivity baseline (685.5 seconds). `SMOKE=1` completed in 22.5 seconds using its documented two-seed / 24-item / two-fold / 2-of-16-split subset; it is a plumbing check, not a result.

No package was installed and no script that writes into experiments 01 or 02 was run. I therefore did not re-run Experiment 02's requested smoke check: the present task explicitly forbids writes there.

## Caveats

- Global RMS avoids rare-unit amplification, but it is one defensible probe convention, not a proof that every alternative is unfair. The strong sensitivity to this convention is itself the main result of the control.
- The SAE and sparse random arm still differ by +0.010 ± 0.002 on main-effect SD, so the primary two-way-XOR edge is not a clean factor-matched causal result.
- This is a property of five lexical draws from one controlled English template, not a claim about all GPT-2 representations or language understanding.
- CCGP and shattering describe code geometry under linear probes. They do not identify a transformer circuit, causal feature use, or a behavioural effect.
- The only theorem cited by this project remains Experiment 02's coordinate-wise compression/XOR result. Nothing trained or measured here earns that label.
- Gemma-2/GemmaScope, more stimulus families, and factor-balanced matching are future work, not substitutions for this GPT-2-small result.

## Next

- Match the random sparse arm's effective active width or reweight it until main-effect SD is indistinguishable, then retest the two-way-XOR edge.
- Add diverse templates and stimulus families, then test whether the global-RMS placement survives lexical and syntactic changes.
- Use SAE features for a circuit-level or intervention study only with an explicit causal design; this representation-only experiment does not answer that question.

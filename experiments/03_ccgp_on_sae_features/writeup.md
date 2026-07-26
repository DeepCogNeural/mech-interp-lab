# Experiment 03 — Shattering dimensionality × CCGP on GPT-2-small SAE features

**Code:** `ccgp_sae.py` (CPU; default five-seed run). **Raw result:** `results.json`. **Figures:** `figures/`.

`results.json` uses schema `exp03-results-v1`: each `per_seed_rows` entry is `{seed, arm, sd, ccgp, gap}`, where each metric maps a dichotomy type to balanced accuracy. Summary intervals use `ci95 = 1.96 × sample_sd / sqrt(n)` across the five seeds. The current file is the completed non-smoke result.

## The question

Experiment 02's monosemantic arm was coordinate-wise *and* a compression. Its theorem-backed XOR failure therefore does not extrapolate to a real sparse autoencoder. Here the GPT-2 residual stream is 768-wide while the SAE encoder is a 24,576-wide ReLU expansion: width alone can improve linear separability.

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

The full run used 96 items × 8 cells = 768 sequences per seed, five lexical/random/probe seeds, five item-disjoint folds, all 35 balanced dichotomies, and all 16 CCGP held-condition splits. Probes are vectorised torch logistic regressions with train-split-only standardisation, 100 AdamW steps, and L2 weight decay 0.08. SD is cross-validated balanced decoding accuracy over the 35 dichotomies. CCGP trains on three conditions per dichotomy side and tests on the held-out one-per-side pair.

## Gates A–C

The run used `/Users/linghao/Github/mech-interp-lab/.venv/bin/python` on CPU with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `MPLBACKEND=Agg`. GPT-2 small resolved from the local Hugging Face cache. The SAE was loaded directly from the published `jbloom/GPT2-Small-SAEs-Reformatted` res-jb safetensors; `sae_lens` was not installed or used.

| gate | measured result | decision |
|---|---|---|
| A: loading | layer-8 SAE: centered explained variance −0.660, MSE 1.086, relative reconstruction error 0.286, mean L0 71.87 | pass |
| B: stimuli and base factors | all five seeds retained 96/96 items; 0 dropped; pilot NUMBER / TENSE / POLARITY accuracy at layers 6–9 was 0.993–1.000 | pass; layer 8 selected |
| C: headroom | residual two-way-XOR SD = 0.827, below the 0.98 saturation threshold | pass |

The centered explained-variance denominator contains only variation across this deliberately narrow stimulus set, so its value is negative even though the directly measured relative reconstruction error is 0.286. I report both rather than treating the centered value as a reconstruction-fidelity headline. Layer-8 pilot main-effect accuracies were NUMBER 1.000, TENSE 1.000, and POLARITY 0.996; the other pilot layers were, respectively, layer 6: 0.993 / 0.999 / 1.000; layer 7: 0.999 / 1.000 / 0.995; and layer 9: 0.999 / 1.000 / 0.995.

## Result 1 — the matched sparse comparison is a null on overall SD

![CCGP lies near chance for all arms, while residual and reconstruction have higher shattering than the sparse codes](figures/01_shattering_vs_ccgp.png)

Caption — against the width-, scale-, and L0-matched random sparse expansion, the SAE has no reliable overall shattering advantage: its within-seed difference is −0.015 ± 0.017. CCGP over all 35 dichotomies is near chance for every arm; the main-effect-specific CCGP check below is more informative about factor retention.

| arm | SD | CCGP, all 35 | CCGP, main effects | train − test gap |
|---|---:|---:|---:|---:|
| `resid` | 0.901 ± 0.007 | 0.500 ± 0.000 | 0.947 ± 0.004 | +0.093 ± 0.006 |
| `sae` | 0.716 ± 0.013 | 0.500 ± 0.001 | 0.731 ± 0.009 | +0.284 ± 0.013 |
| `sae_recon` | 0.876 ± 0.007 | 0.500 ± 0.000 | 0.946 ± 0.005 | +0.092 ± 0.005 |
| `rand_exp` | 0.731 ± 0.010 | 0.500 ± 0.001 | 0.803 ± 0.011 | +0.268 ± 0.010 |
| `rand_exp_dense` | 0.810 ± 0.003 | 0.500 ± 0.001 | 0.859 ± 0.007 | +0.190 ± 0.003 |

The SAE and sparse random arm agree on overall SD within their paired interval. Their two-way-XOR SD difference is also null (+0.009 ± 0.016). However, base-factor retention is not matched: SAE main-effect SD is 0.787 ± 0.014 versus 0.860 ± 0.009 for `rand_exp`, and main-effect CCGP is 0.731 ± 0.009 versus 0.803 ± 0.011. Thus an interaction-only comparison is confounded to that extent; it cannot establish that SAE geometry itself alters interaction readout after holding factor decodability fixed.

`sae_recon` stays much closer to `resid` than `sae` does on SD and main-effect CCGP. That is a round-trip representation control, not a claim about the model's computation.

## Result 2 — dichotomy type matters

![SD breakdown separates readily enumerable factors from two-way XOR, three-way parity, and unstructured dichotomies](figures/02_dichotomy_breakdown.png)

Caption — all arms read main effects above chance, but the sparse SAE and sparse random expansion are lower than the residual/reconstruction reference. The SAE–random two-way-XOR contrast itself is indistinguishable from zero at five seeds.

| arm | main effect SD | two-way XOR SD | three-way parity SD | unstructured SD |
|---|---:|---:|---:|---:|
| `resid` | 0.998 ± 0.001 | 0.845 ± 0.016 | 0.634 ± 0.014 | 0.906 ± 0.007 |
| `sae` | 0.787 ± 0.014 | 0.676 ± 0.013 | 0.592 ± 0.013 | 0.717 ± 0.014 |
| `sae_recon` | 0.983 ± 0.003 | 0.818 ± 0.011 | 0.620 ± 0.014 | 0.880 ± 0.007 |
| `rand_exp` | 0.860 ± 0.009 | 0.667 ± 0.009 | 0.545 ± 0.025 | 0.730 ± 0.010 |
| `rand_exp_dense` | 0.937 ± 0.006 | 0.759 ± 0.003 | 0.634 ± 0.012 | 0.808 ± 0.004 |

Within-seed `sae − rand_exp` differences:

| metric | all / main effect | two-way XOR | three-way parity | unstructured |
|---|---:|---:|---:|---:|
| SD | −0.015 ± 0.017 / −0.073 ± 0.018 | +0.009 ± 0.016 | +0.048 ± 0.024 | −0.014 ± 0.017 |
| CCGP | +0.001 ± 0.001 / −0.072 ± 0.015 | +0.037 ± 0.005 | +0.020 ± 0.018 | +0.004 ± 0.001 |

Overall CCGP pools 35 dichotomies, most of which are unstructured and therefore near chance in this stimulus family. Main-effect CCGP is the relevant abstraction check here. The positive two-way-XOR CCGP difference is measured, but it should not be promoted to an SAE-specific interaction result because the two arms already differ materially on main-effect decoding.

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

The sparse random top-k arm exactly matches the rounded per-seed SAE L0 by construction. The dense arm is intentionally not L0-matched and is only an upper reference. The all-samples-zero removal rule was applied separately to every arm; its considerably smaller surviving widths make the reported train-minus-test gaps important rather than cosmetic.

## What actually ran

The full default run completed in **685.5 seconds (11 minutes 25.5 seconds)** on CPU. No configuration knob was reduced: this is 96 items, five seeds, five folds, all 16 CCGP held-condition splits, and 100 probe steps. `SMOKE=1` also completed in 18.7 seconds using its documented two-seed / 24-item / two-fold / 2-of-16-split subset; it is a plumbing check, not a result.

Run 1 on 2026-07-26 was blocked at Gate A because this sandbox then had neither cached assets nor network DNS. The subsequent run used the pre-populated cache and did not install a package or run any script that writes into experiments 01 or 02.

## Caveats

- This is a property of five lexical draws from one controlled English template, not a claim about all GPT-2 representations or language understanding.
- The SAE and sparse random arms differ in main-effect decodability. The observed null in overall SD and the small two-way-XOR difference should therefore not be read as a cleanly factor-matched causal comparison.
- CCGP and shattering describe code geometry under linear probes. They do not identify a transformer circuit, causal feature use, or a behavioural effect.
- The centered reconstruction EV is sensitive to the low across-stimulus variance of this set; relative reconstruction error and MSE are reported alongside it.
- The only theorem cited by this project remains Experiment 02's coordinate-wise compression/XOR result. Nothing trained or measured here earns that label.
- Gemma-2/GemmaScope, more stimulus families, and factor-balanced matching are future work, not substitutions for this GPT-2-small result.

## Next

- Match or reweight the sparse random expansion to main-effect decodability before interpreting any residual interaction difference.
- Add diverse templates and stimulus families, then test whether the same SD/CCGP placement survives lexical and syntactic changes.
- Use SAE features for a circuit-level or intervention study only with an explicit causal design; this representation-only experiment does not answer that question.

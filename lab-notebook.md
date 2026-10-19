# Research log

Dated corrections, results, and limitations. Detailed measurements remain in each experiment's writeup and result files.
Historical local checks are not independently verified execution records.

## 2026-08-13 — Provenance and pre-run corrections

Experiment 02's XOR proof now distinguishes nonseparability from a chance ceiling. Its empirical probe/grid is configured, not preregistered. Smoke outputs are isolated from published results.
Experiment 05's bridge CSV hashes are bound to the historical provenance record. Compact reaggregation does not revalidate the missing raw result's Gate-A, identity, Git, or runtime fields. The bridge figure describes fixed-span rescue rather than implying mediation.
Experiment 06 binds exact model/tokenizer/SAE files, runtime versions, and before/after state fingerprints. Non-finite values are rejected and public packets are staged before atomic publication.
Historical local model-free checks reported 7/7, 27/27, and 13/13 passes for Experiments 02, 05, and 06, respectively; all 28 then-current Exp05 checksum entries passed. These reports are not independently authenticated. No model experiment ran in this correction pass.
The exact Q4 raw JSON required by Exp06 is not checked in. Exp06 remains unrun and not independently verified.

## 2026-08-10 — Experiment 06 frozen design

The protocol fixes GPT-2-small, the res-jb SAE, L7H4, `resid_pre8` timing, the 12-row span, and 100 ordinal-matched rank-12 spans. Eight seeds (`20260822–20260829`) use the calibration-exposed relative-clause family.
The complete 8/8 evidence set is required. The comparator is the second-largest raw signed-effect edge over Q4's fixed sets, not Q4's normalized `R`. A global no-fixed-point cyclic source-A map removes seed-dependent noun selection, and the same 6/8 seeds must jointly have positive target effect and matched advantage.
Only Gate-A population failure yields `NON_ESTIMABLE`. Binding, rank, token/position, identity, finiteness, runtime, or provenance failure yields `STOPPED` without a scientific verdict.
The source-A construction also differs from the earlier bridge, so this is not a one-factor template-only contrast. It is a mechanism-held-out evaluation on a calibration-exposed family, not unseen-template validation.

## 2026-08-10 — Experiment 02 correction

Additive scores cannot perfectly separate XOR, but `predict 1 iff ReLU(x_i) + ReLU(x_j) > 0` correctly classifies `00`, `01`, and `10`, reaching `0.75` and missing only `11`.
The shipped `0.494 ± 0.006` is an empirical result of the configured BCE-trained logistic probe. It is not a universal accuracy bound.
Corrected learned-minus-random contrasts average the five completed-run sparsities within each seed, then use a two-sided Student-t interval across eight seed means: `−0.015 ± 0.032`, `+0.004 ± 0.019`, and `+0.002 ± 0.011` for distractor probabilities `0`, `0.05`, and `0.10`.
The mixed arms reach about `0.80`; mean train-test gap is `+0.002`. No equivalence margin was specified, so intervals containing zero establish neither equality nor a generic mixing property. No model experiment was rerun for this correction.

## 2026-08-10 — Experiment 05 reader-clamp correction

The bridge overwrites L8H5 `blocks.8.attn.hook_z` at the final query. This is the complete per-head output after attention aggregation, jointly determined by pattern and values; it is not value-only.
The L7H4 final-only intervention changes no non-final `resid_pre8` position. Causal masking prevents earlier L8H5 query outputs from seeing it, so an all-position-clamp factorial adds no identifying contrast and was withdrawn.
The large rescue under the clamp argues against dominant dependence on the tested final-position L8H5 output, but does not identify another route or establish that L8H5 is irrelevant. The bridge is not independently verified; raw-dependent regeneration is not repository-only.

## 2026-08-09 — Exploratory bridge

The historical packet reports eight fresh seeds (`20260814–20260821`), 150 pairs per seed, and 100 target-excluded matched rank-12 spans per seed (800 total), with `L7_ONLY_RESID_PRE8` timing.
Target rescue was `R_target=0.678639`, t(7) CI `[0.673841, 0.683437]`, exceeding all 100 matched spans in every seed (8/8). Complement rescue was `0.305339`, CI `[0.301260, 0.309418]`; aggregate matched-span mean was `0.065910`, CI `[0.049408, 0.082411]`.
With complete L8H5 `hook_z@final` clamped, target rescue remained `0.674047`, CI `[0.669586, 0.678508]`. The natural reader projection coefficient was descriptive (`0.027–0.124`).
Target-plus-complement near-closure is a separate nonlinear arm, not linear attribution. This is exploratory causal-subspace evidence, not preregistered mediation or a complete-path result.

## 2026-08-08 — Stage-3/Q4 span comparison

The candidate-independent comparison used item-disjoint rank-training/evaluation roles and 100 matched draws per seed. Historical runtime was 353,120 logical forward-equivalents and 1,265.99 seconds.
The frozen target exceeded the second-largest matched edge in 8/8 seeds. `R_span=0.893525`, t(7) CI `[0.890486, 0.896565]`; `R_comp=0.084783`, CI `[0.082268, 0.087299]`.
The geometric squared-norm fraction was about `0.525–0.544`, so the result concerns concentration of a directed logit effect, not 89% activation reconstruction. Generic-text PCA span/both effect was `0.027400` raw logit units, not 2.74% recovery.
The result does not establish natural or monosemantic rows, individual-latent causality, necessity, sufficiency, mediation, a complete circuit, or cross-model/task generalisation.

## 2026-08-08 — Stage-2 intervention

Both 144-head selection sweeps used one unchanged model snapshot. The eight-candidate set was frozen before eight adjudication seeds. Runtime was 2,528 logical forward-equivalents and 6,272 seconds; an earlier hook-signature failure stopped after 688 seconds without scientific output and was restarted.
The minimum tested set was `[L7H4, L8H5]`. One head recovered `0.240–0.251` and failed every seed; the pair recovered `0.521–0.546` and passed all eight, with both members above the source-A floor.
The largest source-A/true ratio was `0.0052` against `0.20`; source-C/true was `0.1950` against `0.26983`. Q2 passed 8/8. The frozen-attention value-path effect was positive in every seed (`D_path=0.528–0.574`, smallest bootstrap lower bound `0.504`); Q3 passed 8/8.
Internal recomputation covered 16 execution cells and the bound 1,100,148-row CSV. This is not independent expert validation. Frozen-pattern interventions do not establish the heads' role in unconstrained endogenous computation.

## 2026-08-08 — Protocol corrections before adjudication

Candidate selection requires true/source-A sweeps in one invocation, one in-memory model, an immutable clean-base cache, and complete before/after state fingerprints. The older Stage-1 ranking is only a historical cross-check. The selection cost is 291 logical forward-equivalents.
Q3 uses an explicit clamped-`z` kernel. Q2 uses the intersection of complete base/source-C pairs; its global source-C Gate-A flag is diagnostic. Q4 separates rank-training from evaluation, retaining the 100-draw second-largest edge.
Self-hashes alone cannot authenticate edited scientific records. Stage-2 and Q4 adjudicating runtime cells must be completed in a fresh invocation; old checkpoints are diagnostic only. Complete artifacts bind their pair/draw records, and failures invalidate older successful-looking outputs.
Amendment 2's rules are not uniformly stricter than their predecessors; Amendment 3 withdraws that claim. `E_all` and `d_gap` are the same estimand, but float32 records differ on 2 of 472 directed edits, with maximum absolute difference `5.7220458984375e-06`.

## 2026-08-02 — Calibration and scientific corrections

The residual-only calibration took 101.8 seconds. No attention-head effect was measured at the freeze; earlier Exp04 PCA interventions were already span-level measurements.
Same-number source A moved the aligned readout by `−0.0078` against a true flip of `4.46`. Cross-template source C gave `−0.60`, or 13.5% of a full flip: negative here reinforces the number both sentences share. Q2's bound is `0.270`, above the `0.20` floor.
The attractor flip mean was `+0.0071` with the largest item spread. Subject-flip alignment cannot determine whether attractor effects are consistently directed. That direction remains unmeasured.
The disclosure trigger for `max(0.20, 2ρ)` is `ρ > 0.10`, not `0.20`. The measured `0.135` requires disclosure. The corrected guaranteed component is `0.365·E_all`, not `0.30·E_all`.
The runtime estimate is a lower bound of 34.7 CPU-minutes against a 120-minute cap, not a verified total; selection, PCA, candidate preparation, and attention-value caching costs constrain extrapolation.
Other repairs separate selection and adjudication seeds, define the matched order statistic and family error rate, specify small-pool behavior, and replace Q3's unsupported certification bar with a source-A comparison. See the dated amendments in Experiment 05's design.

## 2026-07-27 — Experiment 04 result and limitations

Pilot, diagnostic, main, and robustness runs took 42, 647, 1,760, and 1,344 seconds. Recorded commit order is not proof of read order. The frozen result is **inconclusive**.
Sixteen SAE coordinates recover half of that basis's own effect in every seed. PCA needs 64 to 128 and does not reach half in one seed. Paired gaps are `+0.304 ± 0.023` within-basis and `+0.146 ± 0.022` absolute.
The trained decoder writes back `0.694 ± 0.014` against Gate C `0.70`, passing only 2/5 seeds. The floor was not moved. A post-unblinding ridge comparison (`+0.348` within-basis) is non-adjudicating; seed `20260803` also has an independent Gate-D obstruction.
Positive-diagonal code/decoder rescaling leaves the edit unchanged; the power-of-two self-test checks one instance bitwise. Decoder bias cancels, but reconstruction-error differences do not: `W_dec(f_src − f_base) = (x_src − x_base) − (e_src − e_base)`.
A ridge reaching `0.887` used 4.7% template fitting rows. The pure-generic control reaches `0.414`, below the trained decoder. A supervised direction recovers `0.549`, versus `0.072` for the best SAE latent among the 128 scored candidates; no claim covers unscored latents.
PCA uses 8,192 tokens for 768 dimensions. Quadrupling the budget shifts AUC by `+0.018 ± 0.025`; this is not a convergence bound. The SAE training-corpus estimate of approximately `10^8` tokens was not verified against its model card.
The initial absolute `1e-4` reconstruction assertion tested float32 round-off at the wrong scale. Dated tolerance repairs preserve the distinction between a numerical assertion and a scientific gate.

## 2026-07-26 — From decoding to intervention

Experiments 01–03 measured representational geometry and linear readout, not causal model behavior. The proposed sample-size sweep would still depend on regularisation geometry when features outnumber samples. Experiment 04 instead uses `W_dec(f_src − f_base)`, invariant to positive-diagonal code/decoder rescaling.
Ranking-free full anchors and random-coordinate controls address dependence on the coordinate-ranking rule. The later quantity measured is SAE basis quality; superposition versus mixed selectivity remains background, not an established organizing result.

## 2026-07-26 — Experiment 03 convergence addendum

Full-batch L-BFGS used relative objective change below `1e-3` over ten accepted iterations, a 500-iteration cap, five item-disjoint folds, 35 dichotomies, both scalings, and four arms. CCGP and dense-random expansion were omitted. Zero-unit masks were fit within each outer training fold. Seed 0 took 30.1 seconds against a 40-minute budget.
Dense-arm SD-family shifts were at most `0.0058`. SAE-minus-random two-way-XOR was `+0.0580 ± 0.0414` under z-score and `+0.1151 ± 0.0192` under global RMS; means differ by `0.0571`. The comparison remains **not adjudicated**.
One z-score SAE inner split reached the high L2 edge after four predeclared expansions and used the documented `1e19` fallback. The comparison remains conditional on the L2 prior geometry.

## 2026-07-26 — Experiment 03 scaling correction

The seven-arm study used GPT-2-small layer 8, res-jb SAE, NUMBER × TENSE × POLARITY, 35 dichotomies, 16 CCGP splits, five seeds, and item-disjoint folds at an identical sentence-final period.
Per-feature z-scoring amplified rare SAE latents; train-test gaps were `+0.28` versus `+0.09` for residuals. Changing to global RMS reversed the ordering. Width controls gave `+0.081` and `+0.075`, but ran only under the favorable convention.
An invertible affine reparameterisation preserves a linear probe's function class. The large contrast shift therefore measures regularisation/optimization sensitivity, not an adjudicated code property. Dense arms remain stable (0.902 vs 0.901; 0.877 vs 0.877).
Five-seed intervals require t(4) `2.776`, rather than `1.96`, about 29% wider. Twelve stale table cells were corrected without changing conclusions. An earlier unavailable-model run produced a `gated_out` manifest. A dense-random inner-L2 convergence attempt did not complete; the later scoped addendum above is distinct.

## 2026-07-20 — Experiment 02 readout

Three geometries at fixed `(n,m)` were compared using XOR probes on `ReLU(Wx)`, eight seeds, a balanced evaluation distribution, and within-seed pairing. A four-seed pilot suggested about `+0.02` learned-over-random advantage at high sparsity; it was not reliable at eight seeds. The corrected interpretation is recorded on 2026-08-10.

## 2026-07-16 — Toy models

At 5 features and 2 dimensions, uniform sparse importance gives a pentagon; decaying importance gives antipodal pairs. Loss symmetry determines the geometry.
At 20 features and 5 dimensions, representation rises from 7/20 in the dense case to 19/20 at `S=0.99` and 20/20 by `S≈0.997`, with `D_i≈0.25` and total effective dimension near 5 when the bottleneck is well used. Capacity rises from 5 to 20 with sparsity.
The effective-dimension sum is an upper bound, not added capacity. Feature counts can vary near the norm threshold; these initial results used a single seed, without a seed sweep.

# Lab Notebook

Dated log. Newest at the top. Each entry: what I did, what I expected, what actually happened, what confused me. The confusion is the most valuable part — don't sand it off.

Template for a new entry:

```
## YYYY-MM-DD — <title>
**Goal:**
**Did:**
**Expected:**
**Happened:**
**Confused about / open:**
**Next:**
```

---

## 2026-07-26 — Experiment 03 run 4: effective width narrows the edge but does not erase it

**Goal:** Test the reviewer’s strongest alternative explanation for the global-RMS result: `sae` might beat L0-matched `rand_exp` on two-way XOR merely because about 794 SAE directions, versus about 503 random directions, fire somewhere in the stimulus set.

**Did:** Added two primary-only, item-split-preserving controls to `experiments/03_ccgp_on_sae_features/ccgp_sae.py` and reran all five seeds offline on CPU. First, widened the random top-k expansion and selected nominal column count using only the unlabelled surviving-width statistic; it retained the same per-sample L0 (71 or 72). Second, uniformly sampled from SAE latents that survived the common all-zero removal until each seed exactly matched its `rand_exp` surviving width; I reported the resulting L0 drop. Rebuilt Figure 1 as a chance-line overview plus a zoom panel, and updated all Experiment 03 artifacts. `SMOKE=1` completed in 84.2 s; the full run completed in 1,190.8 s.

**Expected:** I expected the original `sae − rand_exp` two-way-XOR difference (+0.121±0.011) to shrink if raw active width supplied some of the advantage. I did not know whether it would collapse to zero. A collapse would have made width the right headline; persistence in both matching directions would make the learned-code reading stronger.

**Happened:** The original number was too large to keep as the headline. The widened random arm reached 740–794 surviving units against SAE’s 784–803 (mean 772.6 versus 794.2) while keeping L0 71.80, and its two-way-XOR contrast was `sae − widened-random = +0.081±0.006`. The reciprocal control matched width exactly every seed (464/489/513/559/489) and reduced SAE L0 to 42.23/44.65/48.01/48.70/41.25; it still gave `narrowed-SAE − rand_exp = +0.075±0.011` on two-way XOR. Crucially, its main-effect SD difference was −0.001±0.006, so the surviving interaction edge is not accompanied by a visible base-factor advantage in that exact match. Width explains part of the old effect, not all of it.

**Confused about / open:**
- The widened-random grid was deliberately selected only on surviving width, but it is only near-matched on three seeds (especially seed 1: 740 versus 802). It corroborates the exact narrowed-SAE result rather than replacing it.
- Narrowing SAE width necessarily lowers L0 by about 27; that direction answers the active-direction objection but not a hypothetical code with both exact width and exact L0. The two controls agree in direction, but they are not interchangeable manipulations.
- The result still does not directly demonstrate conjunctive latents. It says a pure count-of-directions account is insufficient on this template family; identifying which SAE structure is responsible requires a separate, preregistered feature analysis.
- Dense random remains highest on shattering (overall SD 0.936, two-way XOR 0.918) because it activates roughly 12,000 coordinates per sample. That is a Cover-theorem upper reference, not a reason to treat dense random as the control.

**Next:** Freeze this width-matched conclusion for the current template. Before looking at individual latents, add independent template families; then preregister a feature-family analysis or a causal circuit question rather than promoting this code-geometry result into a claim about GPT-2’s downstream computation.

---

## 2026-07-26 — Experiment 03: a real SAE code needs a matched random expansion *and* a fair sparse probe

**Goal:** Leave toy land and locate a GPT-2-small residual-stream SAE code on the shattering-dimensionality × CCGP plane, with the right comparison: sparse SAE features versus a width-, scale-, and L0-matched random ReLU expansion.

**Did:** Wrote and ran `experiments/03_ccgp_on_sae_features/ccgp_sae.py`: full factorial NUMBER × TENSE × POLARITY stimuli; final-`.` read-out; per-item token-length assertion; residual layer pilot; direct loading of the published res-jb safetensors without `sae_lens`; four principal arms plus a dense-random reference; item-disjoint multi-output torch probes; all 35 dichotomies and all 16 CCGP held-condition splits. After review, added a probe-fairness control: inner item-disjoint L2 selection per arm and outer fold, per-feature z-score versus one global training-RMS scale, and a 100-versus-200-step convergence check. The control run used 96 items × 8 cells, 5 seeds, and 5 folds; it completed in 1,419.7 s. `SMOKE=1` completed in 22.5 s.

**Expected:** Before running, I expected the SAE/random comparison to be genuinely open. A sparse code might have lower shattering and higher CCGP because it factorises variables cleanly; or conjunctive features might give it more interaction-readout capacity than matched random expansion. I did *not* expect `sae > resid` to be scientifically interesting, since SAE width itself makes that mostly a Cover-theorem comparison.

**Happened:** Run 1 really did stop at Gate A: at that point the sandbox had neither cached assets nor DNS for Hugging Face, so I left a `gated_out` manifest rather than inventing a result. After the assets were cached locally, Gate A loaded GPT-2 small and the res-jb layer-8 SAE directly from safetensors: reconstruction MSE 1.086, relative error 0.286, L0 71.87 (the across-stimulus-centered EV was −0.660 on this narrow stimulus set). All five seeds retained 96/96 equal-length items, no drops. Layer 8 won the pilot (NUMBER / TENSE / POLARITY = 1.000 / 1.000 / 0.996), and residual two-way-XOR SD was 0.827, leaving headroom.

The original fixed-L2/per-feature-z-score result was SAE SD 0.716±0.013 versus matched sparse random 0.731±0.010, with the main-effect mismatch SAE/random 0.787/0.860. The fairness control changes that conclusion materially. Under global RMS plus nested L2, SAE overall SD is 0.888±0.004 versus random 0.809±0.006 (paired +0.079±0.006); main-effect CCGP is 0.918±0.006 versus 0.909±0.008 (paired +0.009±0.013). The SAE is not at a low-expressivity/high-abstraction corner. Its two-way-XOR SD is also higher (paired +0.121±0.011), but main-effect SD still differs by +0.010±0.002, so that is not a clean interaction claim. This is geometry of a lens, not a test of what GPT-2 uses downstream.

**Confused about / open:**
- The first conceptual trap remains the naive exp02 extrapolation: “SAE = monosemantic arm, therefore no interaction.” Wrong direction. Exp02 compressed coordinate-wise; a real SAE expands nonlinearly from 768 to 24,576. The only comparison that can answer the intended question is SAE versus a matched random expansion.
- I put **overall** CCGP on the original headline x-axis even though it pooled 3 structured and 28 unstructured dichotomies. It was ~0.500 for every arm, so the plotted points formed a useless vertical line. I had already written that main-effect CCGP was informative and failed to let the figure follow that reasoning. The rebuilt axis is main-effect CCGP; pooled CCGP is table-only.
- The quiet implementation trap remains word-form leakage. If the probe reads `did`, `does`, `do`, `not`, or the main verb rather than the sentence-final `.`, it measures vocabulary identity, not a carried representation — and token-identity SAE latents make that look especially convincing. The second quiet trap is eight condition means: it deletes lexical nuisance variance and lets every code saturate. The run enforced equal token lengths within item and always split by lexical item.
- The probe-fairness hypothesis was supported at the level of the headline: selecting L2 alone leaves the z-score ordering almost unchanged, while changing to a single global RMS scale reverses it. That is consistent with rare-unit amplification, but the control does **not** isolate activation frequency from every other consequence of rescaling, so it does not prove that mechanism. The active-width mismatch also remains: about 784–803 SAE units versus 464–559 sparse-random units fire on this template family.
- CPU was fine: 23.7 minutes is below the 30-minute budget. I did not run experiment 02's write-producing script because this task explicitly forbade writes there.

**Next:** Match effective active width as well as L0, then retest the two-way-XOR edge. Repeat with more templates before treating this grammatical family as general. A causal circuit/intervention question is separate work; do not infer it from this lens experiment.

---

## 2026-07-20 — Experiment 02: does superposition help a downstream readout?

**Goal:** Turn the exp1 bridge thought into a real, falsifiable test — is mixed/superposed coding just a storage compromise, or does it also keep nonlinear computation linearly readable (Rigotti's mixed-selectivity claim), measured in a toy model with ground truth?

**Did:** Three geometry arms (monosemantic / random / frozen superposition) at fixed `(n,m)`, read XOR of a feature pair with a linear probe on `r = ReLU(Wx)`. 8 seeds, fixed balanced eval distribution, within-seed paired stats. `experiments/02_superposition_and_readout/`.

**Expected:** monosemantic at chance (theorem: linear readout can't get the `x_i·x_j` term); mixed codes above chance; and — the part I was unsure about — maybe the storage-learned geometry beats random mixing.

**Happened:**
- Headline came out clean and strong: monosemantic sits at 0.494±0.005 at every sparsity, both mixed codes at ~0.80, probe train-test gap +0.002. The theorem-backed anchor holds exactly.
- The secondary question resolved to a **null**: superposition ≈ random (paired diffs hug zero, CIs include zero across `m`, `S`, and background). The readout benefit is from mixing+nonlinearity, not the specific learned geometry.

**Confused about / open:**
- Almost walked into a trap: my first instinct was a linear probe on the exp1 encoder `h = Wx`. That's identically chance for *any* geometry, because a linear readout of a linear projection can't represent XOR — the null would have been real but for a boring reason. The fix (and the actual insight) is that a nonlinearity is required, and it has to be held constant across arms so the comparison is about geometry, not about the nonlinearity.
- A 4-seed pilot hinted superposition beat random by ~+0.02 at high sparsity with background activity. At 8 seeds it washed out. Good reminder to not stop at the seed count that flatters the hypothesis.

**Next:** shattering dimensionality / CCGP (the task-agnostic version of the headline); then ask the same enumeration-vs-computation question of real SAE features on a small transformer.

---

## 2026-07-16 — Setup + first experiment (Toy Models of Superposition)

**Goal:** Stand up the lab and run one real experiment end to end, not just read.

**Did:**
- Created the repo (`DeepCogNeural/mech-interp-lab`), wrote the README, the week-by-week `learning-roadmap.md`, and this notebook.
- Environment: system Python is 3.14, which has no torch wheels yet. Fell back to **Python 3.11** in `.venv`. Installed torch 2.13.0, transformer_lens 3.5.1, numpy, matplotlib, jupyter. Clean.
- Built and ran `experiments/01_toy_models_of_superposition/toy_models.py`. Three experiments, all on CPU, ~90s total. Real figures on disk in `figures/`.

**Expected:** Superposition appears as sparsity rises; the 5→2 case should give the pentagon; `Σ Dᵢ` should approach `m` only when training uses the bottleneck fully, consistent with its role as an effective-dimension upper bound.

**Happened:** All three confirmed.
- 5 features → 2 dims: dense keeps only 2 orthogonal features; sparse + *uniform* importance gives the textbook **pentagon**; sparse + *decaying* importance gives antipodal pairs instead. The importance weighting picks the symmetry.
- 20 → 5: goes from 7/20 features (dense, `Dᵢ≈1`) up to the full 20 at the sparsest end (19/20 at S=0.99, 20/20 by S≈0.997; `Dᵢ≈0.25`), with `Σ Dᵢ` sitting near `m = 5` once the bottleneck is well used. It is an effective-dimension upper bound, getting close to `m` only when training uses the bottleneck fully; superposition redistributes that available dimension budget rather than adding capacity.
- Capacity climbs monotonically from 5 (the orthogonal limit) to 20 as features get sparser.

**Confused about / open:**
- First runs used decaying importance for the 5→2 sweep and I got antipodal pairs, not the pentagon — briefly thought the replication had failed. It hadn't: the pentagon needs *uniform* importance so no feature is privileged. Added both regimes side by side; the contrast turned a bug into the clearest panel in the figure. Lesson logged: the geometry is set by the loss symmetry, so "which figure you reproduce" depends on the importance vector, not just the sparsity.
- The exact "features represented" integer count jitters run-to-run near the norm threshold. `Dᵢ` and the geometry are the stable readouts. Haven't done a seed sweep — single seed so far.

**Bridge thought:** this is mixed selectivity with ground truth. The controlling variable (feature sparsity) is the same natural-scene-statistics regime Olshausen & Field built V1 sparse coding on. Real open question for me: SAE-based interp often treats superposition as something to *undo* (disentangle to enumerate features), while Rigotti's mixed-selectivity argument says the same geometry can *ease* a linear readout. Both can hold — the field already studies computation in superposition, so this isn't about who noticed what. What a fully-observable model lets me do is *measure* whether the superposed geometry helps or hurts a downstream readout. That's the thread `experiments/02` takes up. Full argument in `experiments/01_.../writeup.md`.

**Next:**
- Multi-seed + finer sparsity grid to map the phase boundaries.
- Then Week 1 of the roadmap: write a transformer forward pass by hand, verify against TransformerLens.

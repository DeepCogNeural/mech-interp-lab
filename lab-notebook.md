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

## 2026-07-26 — Experiment 03: a real SAE code needs a matched random expansion, not a straw-man residual comparison

**Goal:** Leave toy land and locate a GPT-2-small residual-stream SAE code on the shattering-dimensionality × CCGP plane, with the right comparison: sparse SAE features versus a width-, scale-, and L0-matched random ReLU expansion.

**Did:** Wrote and ran `experiments/03_ccgp_on_sae_features/ccgp_sae.py`: full factorial NUMBER × TENSE × POLARITY stimuli; final-`.` read-out; per-item token-length assertion; residual layer pilot; direct loading of the published res-jb safetensors without `sae_lens`; four principal arms plus a dense-random reference; item-disjoint multi-output torch probes; all 35 dichotomies and all 16 CCGP held-condition splits. The final CPU run used 96 items × 8 cells, 5 seeds, and 5 folds; it completed in 685.5 s. `SMOKE=1` also completed as a two-seed plumbing check.

**Expected:** Before running, I expected the SAE/random comparison to be genuinely open. A sparse code might have lower shattering and higher CCGP because it factorises variables cleanly; or conjunctive features might give it more interaction-readout capacity than matched random expansion. I did *not* expect `sae > resid` to be scientifically interesting, since SAE width itself makes that mostly a Cover-theorem comparison.

**Happened:** Run 1 really did stop at Gate A: at that point the sandbox had neither cached assets nor DNS for Hugging Face, so I left a `gated_out` manifest rather than inventing a result. After the assets were cached locally, Gate A loaded GPT-2 small and the res-jb layer-8 SAE directly from safetensors: reconstruction MSE 1.086, relative error 0.286, L0 71.87 (the centered EV was −0.660 on this narrow stimulus set). All five seeds retained 96/96 equal-length items, no drops. Layer 8 won the pilot (NUMBER / TENSE / POLARITY = 1.000 / 1.000 / 0.996), and residual two-way-XOR SD was 0.827, leaving headroom.

The headline comparison was a null: SAE SD 0.716±0.013, matched sparse random SD 0.731±0.010, paired difference −0.015±0.017. Two-way XOR was likewise close (0.676 vs 0.667; paired +0.009±0.016). Overall CCGP sat near 0.500 for every arm. The result that prevents a stronger story is base-factor mismatch: SAE main-effect SD/CCGP = 0.787/0.731, random = 0.860/0.803. The reconstruction arm nearly recovered the residual's base-factor scores. This is geometry of a lens, not a test of what GPT-2 uses downstream.

**Confused about / open:**
- The first conceptual trap remains the naive exp02 extrapolation: “SAE = monosemantic arm, therefore no interaction.” Wrong direction. Exp02 compressed coordinate-wise; a real SAE expands nonlinearly from 768 to 24,576. The only comparison that can answer the intended question is SAE versus a matched random expansion.
- The quiet implementation trap remains word-form leakage. If the probe reads `did`, `does`, `do`, `not`, or the main verb rather than the sentence-final `.`, it measures vocabulary identity, not a carried representation — and token-identity SAE latents make that look especially convincing. The second quiet trap is eight condition means: it deletes lexical nuisance variance and lets every code saturate. The run enforced equal token lengths within item and always split by lexical item.
- I expected the five-seed comparison might distinguish sparse factorisation from conjunctive features. Instead the clean overall result is a null with an unclean base-factor control: why do the sparse SAE and sparse random arm retain NUMBER/TENSE/POLARITY at different levels despite exact L0 and encoder-width matching? Is that an intended SAE-code property, a consequence of active-width differences (784–803 vs 464–559), or a probe/regularisation issue? Until it is matched, the small two-way-XOR CCGP difference is not an interaction conclusion.
- CPU was fine: 11.4 minutes is below the 30-minute budget. MPS remained unavailable. I did not run experiment 02's write-producing script; no package installation was needed.

**Next:** Match the random arm to base-factor decodability, then ask whether any two-way-XOR or CCGP difference survives. Repeat with more templates before treating this one grammatical family as a general property of GPT-2's codes. A causal circuit/intervention question is separate work; do not infer it from this lens experiment.

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

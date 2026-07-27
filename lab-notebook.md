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

## 2026-07-26 — Stepping back: what three experiments actually established, and the decision to leave probes

**Goal:** Stop running and audit. Three experiments are done; decide whether the next one finishes experiment 03 or changes the question.

**Did:** Reread all three writeups against a hostile reading of the portfolio as a whole, then wrote `experiments/04_causal_feature_interchange/DESIGN.md` as a pre-registration, before any code exists.

**Expected:** That the sensible next step was to finish experiment 03 — a scaling-invariant criterion, or more samples so the effective feature count stops exceeding the sample count. I had a design half-drafted for a sample-size sweep: push `n` well past the surviving feature count and watch the two scalings converge.

**Happened:** I talked myself out of it, and the argument that did it was not about experiment 03 at all. Every measure in this repo is linear decodability of a representation. Not one result is causal — no patching, no ablation, no intervention on a forward pass. That is representational geometry, which is the analysis style I already had from V1, and it is not what mechanistic interpretability is premised on. The sample-size sweep would have been a fourth decoding experiment, and a clever one, and it would have deepened exactly the weakness. There is an irony I had not noticed until I read the roadmap back: the methodological warning I lean on hardest, Jonas & Kording on the microprocessor, is a warning about trusting decoding results.

Worse, the sweep's own logic has a hole. When features outnumber samples, two codes related by an invertible linear map are close to indistinguishable to a decoder that carries no metric — so the metric has to come from somewhere, and it comes from the regulariser. That is the sharp version of experiment 03's finding, and it says the fair scaling point I would be hunting is probably not there to find.

**Confused about / open:**
- The honest re-score is that experiment 02 is not a null — it is a theorem-anchored positive result plus a discriminating secondary null. Experiment 03 is the precarious one, because "not adjudicated" is *weaker* than a null: a null answers the question, an instrument failure says the question could not be asked with that instrument. I had been counting both as the same kind of honesty. They are not.
- The replacement design has a property I like more than its being the missing checklist item. The interchange edit `W_dec (f' − f)` is invariant to the exact rescaling that swung experiment 03 tenfold: rescale the code by `D` and the decoder by `D⁻¹` and the vector written into the residual stream is unchanged. The knob that made the old question unanswerable cannot touch the new one. First time the failure has told me something constructive about what to build next.
- Named risk: the matched random expansion needs a decoder, and getting a random basis onto equal reconstruction footing with a trained SAE is fiddly. It is gated (Gate C) and prototyped first, with a smaller within-SAE result to fall back on. Better to find that out in twenty minutes than in a week.
- Watching for the mirror of the experiment 03 mistake. There, three mutually reinforcing controls all sat downstream of one unexamined preprocessing choice. Here the analogous single point of failure is the coordinate ranking rule — so it gets the ranking-free `*_full` anchors and the `*_randk` controls around it, and both bases get the identical rule and the identical budget.

**Next:** Gate A and Gate C on a 20-pair pilot before anything else. If the random basis will not reconstruct, the experiment gets smaller rather than looser.

---

## 2026-07-26 — Experiment 03 addendum: convergence did not remove the scaling problem

**Goal:** Run the one decisive five-seed test: fit the SAE and sparse-random SD probes to a stated convergence criterion, select L2 item-disjointly for each arm and scaling, then ask whether the paired two-way-XOR difference agrees across the two affine feature scalings.

**Did:** Used full-batch L-BFGS on L2-logistic loss, stopping below `1e-3` relative objective change across ten accepted iterations (500-iteration cap). Kept five item-disjoint folds, all 35 dichotomies, both scalings, and four arms; omitted CCGP and `rand_exp_dense` exactly as scoped. The zero-unit keep-mask was fit in each outer training fold. Seed 0 took 30.1 s, so the pilot projected far below the 40-minute budget and no planned scope was cut. Full raw rows are in `experiments/03_ccgp_on_sae_features/convergence_results.json`.

**Expected:** If the old 10× swing was merely fixed-step non-convergence, the paired `sae − rand_exp` two-way-XOR estimates would both be tight and close after convergence. If the z-score result stayed noisy or the two means stayed apart, that would confirm the comparison is conditional on the L2 prior geometry rather than adjudicate a code property.

**Happened:** The dense check behaved as expected: every `resid` / `sae_recon` SD-family shift was at most 0.0058. But z-score gave `+0.0580 ± 0.0414` and global RMS `+0.1151 ± 0.0192` for the paired SAE−random two-way-XOR contrast; their means differ by 0.0571. The z-score estimate is not precise and the means are not close, so the test **does not adjudicate** the code comparison. It supports the shipped methodological conclusion instead: the apparent edge is still inseparable from the L2 geometry induced by feature scaling.

**Confused about / open:** One z-score SAE inner split selected the high L2 grid edge even after four predeclared expansions; its documented `1e19` fallback is a direct sign that the inner main-effect criterion itself can become uninformative for that sparse scaling. I did not retune it away after seeing the result. A future probe family needs either a scaling-invariant criterion or a question that explicitly treats the regularisation geometry as the object of study.

**Next:** Keep the published headline and its `Next` item unchanged. This addendum closes the convergence test as a confirmation of non-adjudication, not as a route to revive the width-control interaction claim.

## 2026-07-26 — Experiment 03: a real SAE on the shattering × CCGP plane, and a headline I had to give back

**Goal:** Leave toy land. Ask exp02's enumeration-versus-computation question of real SAE features on a real transformer, with Bernardi's task-agnostic metrics instead of one hand-picked XOR.

**Did:** GPT-2-small layer 8, res-jb SAE loaded straight from the published safetensors. Full factorial NUMBER × TENSE × POLARITY, read at a sentence-final `.` that is byte-identical across all eight cells. Seven arms, 35 dichotomies, 16 CCGP splits, 5 seeds, item-disjoint folds. Then, after review, a probe-fairness control and two effective-width controls. `experiments/03_ccgp_on_sae_features/`.

**Expected:** Genuinely open. Sparsity pressure could make the SAE a clean factorised code (low shattering, high CCGP); conjunctive latents could instead hand a linear readout the product term (high shattering). I deliberately did not pre-register a direction. The one thing I was confident about: `sae > resid` would be uninteresting, because 768 → 24,576 with a ReLU wins that by Cover's theorem alone.

**Happened:** Four separate times I thought I had the answer and did not.

1. First real numbers said the SAE was *worse* than matched random on everything, with a big base-factor confound. I flagged it as confounded and moved on.
2. Suspecting the probe, I asked for a fairness control. Per-feature z-scoring turned out to amplify rare SAE latents into exactly the directions a probe overfits — train−test gap +0.28 versus +0.09 for the residual stream. Under a single global scale the ordering *reversed*: SAE two-way-XOR went from +0.009 to +0.121 over matched random. That felt like the result.
3. The obvious attack was that the SAE simply has more distinct latents firing (~790 versus ~500) even at matched per-sample L0. Matched it in both directions. The edge shrank to +0.081 and +0.075 but survived, and the narrowed-SAE control matched base factors exactly while handicapping the SAE on L0. That felt like a *strong* result.
4. Then an adversarial review pointed out the thing I should have seen in step 2: **for a linear probe, per-feature z-scoring is an invertible affine reparameterisation.** The reachable function class is identical. So a 10× swing in the answer cannot be a property of the codes — it is the L2 prior and the optimiser's path, and my "global RMS is principled because the decoder reads raw activations" argument was a post-hoc story for a number I liked.

So the headline is now: **not adjudicated.** The convergence test that would settle it did not complete (inner-L2 selection found no stable candidate on the dense-random arm), and I am not shipping the flattering setting.

**Confused about / open:**
- The salvage is that the sensitivity is *localised*. The two dense 768-dim arms (`resid`, `sae_recon`) are completely unmoved by the scaling — 0.902 vs 0.901, 0.877 vs 0.877. Every sparse or very wide arm swings hard. That is exactly what the affine argument predicts, and it makes the real finding methodological: probe preprocessing that is harmless for dense representations is decisive for sparse over-complete ones. Any SAE-versus-baseline decoding comparison that doesn't state its scaling convention is under-determined. I believe that part.
- I still don't know whether the SAE has a genuine interaction-readout edge. I think the width controls are suggestive, but they only ever ran under the one scaling that produces an effect, so they inherit the question rather than answering it.
- Uncomfortable lesson about my own process: at step 3 I was *more* confident than at step 2, because the controls all pointed the same way — but every one of them was downstream of an unexamined preprocessing choice. Stacking controls on top of an unvalidated foundation feels like rigour and isn't. The review caught it; I didn't.
- Two smaller things I got wrong and corrected: `1.96 × sd/√5` is not a 95% CI at five seeds (needs t(4) = 2.776, ~29% wider), and 12 cells of a results table were stale numbers from an earlier run. Both harmless to the conclusions, both exactly the kind of thing that makes a reader stop trusting everything else.
- The whole run-1 stop is worth keeping too: no network in the execution sandbox, so it hit Gate A and wrote a `gated_out` manifest instead of a plausible-looking number.

**Next:** Finish the convergence test — converged probes, L2 selected per arm *and* per scaling on an interior grid, agreement judged only when both estimates are individually precise. Then, and only then, ask whether conjunctive latents are the mechanism. Independent template families before any of it generalises.


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

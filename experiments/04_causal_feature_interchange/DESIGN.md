# Experiment 04 — Design: is an SAE basis a *causal* basis?

**Status: design only, frozen before implementation. Nothing in this document has been run, and no
number below is a measurement.** Every threshold here is pre-declared so it cannot be chosen after
seeing data. When the run happens it gets the usual `writeup.md` + raw manifest beside this file, and
this document stays as the pre-registration.

Written 2026-07-26; revised 2026-07-27 after a pre-implementation review, which changed three
load-bearing things: the stimuli became single-flip, the primary statistic became within-basis
normalised, and the random basis's decoder is now fitted off-template. Each change is flagged in place
with **[rev]** and the reason it was needed, because the reasons are the interesting part.

---

## Where the project stands, and why this is next

Three experiments are done. One replicates the storage account of superposition in a toy model. One
shows, with a theorem-backed negative control, that an exactly monosemantic code cannot let a linear
readout see a feature interaction while mixed codes can — plus a discriminating null: the
*storage-learned* geometry is no better than equal-norm random mixing. One carried that question to a
real model and did not settle it.

Reviewed honestly, that record has two weaknesses and they are the same weakness twice.

**Every measure so far is linear decodability of a representation.** Not one result is causal. No
ablation, no patching, no intervention on a forward pass, no circuit. That is representational-geometry
work — much closer to the population analyses of my neuroscience background than to mechanistic
interpretability, whose premise is that structure is *causal* structure. There is an irony worth naming:
the methodological warning this project leans on hardest, Jonas & Kording's microprocessor paper, is
precisely a warning about trusting decoding results, and the portfolio is almost entirely decoding
results.

**And the only real-model experiment ended in an instrument failure, not an answer.** "Not adjudicated"
is weaker than a null. A null answers the question; experiment 03 says the question could not be asked
with that instrument — and its own analysis, that per-feature scaling is an invertible affine
reparameterisation for a linear probe, is a *structural* reason to think no fair setting of that
instrument exists. Iterating on the probe means fighting my own argument for a week to produce a number
whose meaning I already dispute.

Both point the same way: stop asking through a fitted readout. The confound is not a bad choice of
probe, it is the existence of the probe. Remove it and the confound goes with it.

## The invariance that makes this work

This is the whole reason to run experiment 04, so it goes first.

An interchange edit writes a delta into the residual stream:

```
resid' = resid_base + W_dec (f' − f_base)
```

Take the exact manipulation that swung experiment 03 by more than tenfold — rescale every latent by a
positive diagonal `D`. The code becomes `D f`, and the matching decoder is `D⁻¹ W_dec`. The product is
unchanged, so `resid'` is **bit-for-bit identical**. The residual stream is a scale-free meeting point:
the model reads activations, not features, and it does not care what basis you named them in.

So the free knob that made experiment 03 unanswerable cannot move this measure at all. That is a
property of the design, not a hope about the data — and it is asserted mechanically in pilot step 2b
rather than argued.

## Scope — what this will not be allowed to say

Read this before the results exist, because a causal design sits closer to the line than a probe does.

The model is never modified. No SAE is inserted into GPT-2's forward pass, no component is replaced by
its reconstruction, and no feature is removed. The **add-back-error** construction is what keeps this
true: the edit is `+ W_dec (f' − f_base)`, a difference of two reconstructions, so the SAE's
reconstruction error cancels exactly and never enters the forward pass. GPT-2 runs unmodified apart from
one additive vector at one hook.

The implementation is required to make this mechanical, not aspirational: the edit is always
`resid[b, pos] += delta`, never an assignment of a reconstruction, and pilot step 2a asserts that a
zero-selection edit reproduces the clean logits bit-for-bit.

Therefore a low recovery score in *either* basis says that the factor is not concentrated in few
coordinates *of that basis*. It does not say — and may not be written as saying — that an SAE harms,
degrades, removes, or loses any computation the model performs. That claim is not measured here and no
result of this experiment can support it.

## The question, precisely

GPT-2-small computes subject–verb number agreement. Does the SAE's coordinate system localise that
computation — can a small set of SAE coordinates, edited and written back, move the model's own
agreement behaviour toward the counterfactual — and does it do so with **fewer edited coordinates** than
a random expansion matched in width, encoder column-norm distribution, and per-sample L0?

Two directions are live, neither pre-registered as expected. The sparsity objective pushes latents
toward single-concept detectors, which should concentrate a factor into few coordinates. But number is
a grammatical property distributed across many lexical latents, and a matched random expansion of the
same width has coordinates too; Cover's theorem says nothing about *concentration*, which is what this
measures.

## Stimuli — single-flip pairs **[rev]**

The first draft used the Linzen-style double-flip pair (singular subject + plural attractor against
plural subject + singular attractor). That is the right construction for *measuring agreement accuracy*
and the wrong one for *intervening*, for three reasons: the two members differ at two token positions,
so "the edited position" is not well defined; a final-position patch then mixes two opposing effects,
the carried subject-number signal pushing one way and the swapped attractor's own token identity pushing
the other, which biases the denominator `E_resid` downward for a reason that has nothing to do with the
basis; and the token-length assertion gets fiddly. *(My impression is that the causal-mediation
literature intervenes on single-flip pairs for exactly this reason — recorded as an unverified
recollection, not a citation.)*

So: **single-flip**. Within a pair only the subject noun changes; the attractor is held fixed.

```
The {ADJ} {SUBJ} {PREP} the {ATTRACTOR} ___
```

e.g. *"The tired author near the paintings ___"* against *"The tired authors near the paintings ___"*.
Attractor number is counterbalanced across items so the set still contains interference in both
directions. Nouns come from experiment 03's vetted list, where each singular and plural form is a single
token with a leading space.

Three things this buys, on top of removing the confound: the edit position is unique and is the subject
token; patching every position from the subject onward becomes *exactly* a prompt swap, so
`d(patched) == d(source)` is a free machine-checkable correctness test (pilot step 4); and the
equal-length assertion becomes trivial.

Readout at the verb slot, on the model's own logits:

```
d(prompt) = logit(" are") − logit(" is")
```

Signed so plural pushes it positive. No probe, no training, no preprocessing of any representation
before the readout. The implementation asserts at load time that `" is"`, `" are"`, and every control
word tokenise to exactly one token **with** the leading space, and records their ids in the manifest —
the space-less forms are different tokens and that trap is real.

**Both directions are run.** Every pair contributes a singular→plural edit and a plural→singular edit,
aligned by sign before averaging. This nearly doubles the data for free and kills the artifact where a
generic perturbation happens to push `" are"`.

## Arms

Let `resid_base`, `resid_source` be layer-8 `hook_resid_pre` activations for a matched pair, `f` the SAE
code, `g` the matched-random code, and `E(Δ) = d(patched) − d(base)`.

| arm | edit written into the residual stream | role |
|---|---|---|
| `resid_full` | `resid_source − resid_base` at the selected positions | basis-free gold standard; defines `E_resid` |
| `sae_full` | `W_dec (f_source − f_base)` | SAE faithfulness anchor; selection-free |
| `rand_full` | `W_dec_rand (g_source − g_base)` | random-basis faithfulness anchor; the equal-footing check |
| `sae_topk(k)` | the same sum restricted to the top-`k` ranked SAE coordinates | **the research object** |
| `rand_topk(k)` | the same, in the matched random basis | **the load-bearing control** |
| `sae_randk(k)` | `k` coordinates drawn uniformly from the candidate set | within-basis control: is the *ranking* doing anything? |
| `rand_randk(k)` | the same in the random basis | its counterpart |

`k` grid: `{1, 2, 4, 8, 16, 32, 64}`, matching the 64-candidate budget. If `E(top-64)/E(full) < 0.8` in
either basis, the candidate set extends to 128 and the grid gains `k = 128` — pre-declared, so it is a
coverage repair and not a fishing licence. `*_randk` draws one subset per seed per `k`; the five seeds
supply the variability.

The `*_full` arms are ranking-free. They anchor the curves so that no choice of ranking rule can drive
the headline on its own — the lesson experiment 03 paid for.

"Patch all positions" is **not** an arm. Under single-flip pairs it is arithmetically a prompt swap, so
it is a self-test (step 4), never a measurement.

## Primary measure **[rev]**

Within-basis normalised recovery:

```
R_b(k) = E(top-k in basis b) / E(full in basis b)
```

**not** a common `E_resid` denominator. The question is *concentration* — what fraction of a basis's own
total causal effect lives in `k` coordinates — and that is what within-basis normalisation measures. A
common denominator would multiply every point of the curve by that basis's faithfulness, so if the SAE's
trained decoder writes back better than a fitted random one, identical concentration curves would still
hand the SAE a uniformly higher score. The headline would then be won by decoder quality. Within-basis
normalisation removes that to first order; faithfulness is instead policed separately by Gate C.

Computed as a **ratio of means over evaluation pairs**, not a mean of per-pair ratios, so that pairs with
a small denominator cannot explode. `R` is clipped to `[0, 1]` for the statistic — overshoot is not extra
concentration and negative recovery earns no credit — and the unclipped curve is plotted.

Statistic: normalised area under `R_b(k)` across the grid; the headline is the **within-seed paired
difference** `AUC(sae) − AUC(rand)` over five seeds, as a two-sided Student-t(4) 95% interval,
`t(4) = 2.776`. AUC is preferred to "smallest `k` reaching 50% recovery" because it is bounded, always
defined, robust to non-monotone curves, and pairs cleanly across seeds; `k50` is reported as a narrative
number with a pre-declared censoring convention ("not reached within the grid").

The absolute curves `E/E_resid` are reported as a diagnostic figure, not as the statistic.

## Decision rule, fixed now

- **Adjudicated positive** — the paired interval lies entirely above zero, its half-width is at most
  `0.05`, and Gate D holds. Reading: the SAE basis concentrates this causal factor into fewer
  coordinates than matched random mixing does.
- **Adjudicated null** — the interval contains zero *and* its half-width is at most `0.05`, *and* each
  basis's own AUC interval has half-width at most `0.10`. Reading: sparsity does not buy causal
  localisation of this factor over matched random mixing. That extends experiment 02's null into the
  causal, real-model regime, and it is a result, not a failure.
- **Inconclusive** — anything else, reported as such, in experiment 03's words rather than dressed up.
  Overlapping intervals are not agreement when one of them is uninformative.
- **Fallback rule, if Gate C rules the random basis out**: the experiment ships as the within-SAE
  comparison with statistic `AUC(sae_topk) − AUC(sae_randk)` and the identical thresholds. Smaller
  claim, still causal, still adjudicable.

`0.05` on an AUC in `[0, 1]` is roughly the difference of one sub-grid step in the `k` needed to reach a
given recovery level; anything finer is not worth calling a localisation advantage.

## Gates

| gate | check | threshold | if it fails |
|---|---|---|---|
| A — behaviour | model actually does the task | both members of a pair signed correctly in ≥ 0.60 of generated pairs; ≥ 140 pairs retained; median `d_gap = d(source) − d(base)` ≥ 1.0 | rebuild the stimulus set; do not patch |
| B — causal handle | `resid_full` moves it | mean `E_resid / d_gap` ≥ 0.50 at layer 8 on the selected positions, and per-pair sign consistency ≥ 0.90 | `gated_out`; the layer × position table is the deliverable |
| C — equal footing | both bases write back faithfully | `E(full)/E_resid` ∈ **[0.70, 1.30]** for each basis | random basis out → fallback rule; SAE out → no headline |
| D — specificity | the edit is a factor edit | `S(a) = mean|E_control(a)| / mean|E_number(a)|`; require `S(top-k*) ≤ S(resid_full) + 0.15` and `S(resid_full) ≤ 0.5`, at each basis's first `k` reaching `R ≥ 0.5` | headline withdrawn; report as a generic perturbation |

Gate C's **upper** bound matters as much as the lower one: an edit that overshoots by 30% is
off-manifold amplification, which breaks equal footing from the other side.

Gate D's control contrast is a neutral lexical/aspect pair (e.g. `" walked"` / `" walking"`), **not**
`" was"` / `" were"` — those are themselves number-marked, so a genuine number edit *should* move them.
`was/were` instead becomes a secondary positive check: cross-tense generalisation, requiring the
`is/are` and `was/were` effects to share sign on ≥ 0.80 of items.

Note a hard constraint the first draft missed: only the `blocks.8` res-jb SAE is in the local cache, so
"sweep the layer and retry" is available to the `resid` arm and **not** to the SAE arm. If layer 8 has no
causal handle, that is a `gated_out`, not a redesign mid-run.

## The ranking rule, and why it cannot smuggle in a knob

Coordinates are ranked by their **single-coordinate causal effect**: for candidate `i`, edit only that
coordinate and measure `E({i})`, averaged over a *training* split of pairs that is item-disjoint from the
evaluation split. Scale-invariant for the same reason the whole design is.

Measuring all 24,576 is unaffordable and unnecessary — a coordinate inactive in both members contributes
exactly zero. The candidate set is the union of active coordinates over training pairs, pre-filtered to
the top 64 by the cheap scale-invariant proxy `‖ mean_pairs (f_source,i − f_base,i) · W_dec[i, :] ‖₂`
(the norm of the *signed mean* contribution, so direction consistency is rewarded), then scored causally.

**The identical rule, the identical pre-filter, and the identical budget are applied to both bases.**
That is the condition under which the comparison means anything. Ranking coordinates one at a time
ignores redundancy and interaction between them; that is a known approximation, defended by the
`*_randk` controls and by both bases receiving it. Greedy forward selection is an appendix if time
allows, never the primary rule.

## The random basis's decoder **[rev]**

The matched random expansion needs a decoder. The SAE has `W_dec` because it was trained; a random
encoder does not. Alternatives considered and rejected: rotating the 24,576-dim latent space destroys
sparsity and with it the L0 match; rotating the 768-dim output space just writes rotated noise;
permuting decoder rows breaks the code–decoder correspondence and cannot pass Gate C (kept only as an
"should do nothing" sanity arm); a random *orthogonal* 768-dim basis is fit-free and exactly faithful but
is a straw man, since a fixed causal direction lands in `k` random orthogonal directions with expected
mass `k/768` and any basis with one latent roughly aligned to number wins trivially; tied weights with a
single fitted scale stay as the documented weaker fallback.

So: ridge regression from the random code back to `x`, solved in dual form (`n × n`, since `n < p`),
giving the random basis the best *linear* write-back so that whichever basis loses, loses on the basis
itself.

**But not fitted on the stimulus items.** On a stimulus distribution this narrow, number is a dominant
variance axis, so even an unsupervised reconstruction fit would pull the number direction into the
decoder rows of correlated random units — the control would absorb task structure through its decoder,
and the fitting details (`λ`, sample) would become exactly the kind of free knob this experiment exists
to eliminate. Instead the ridge is fitted on **off-template data**: activations from GPT-2's own
generated text (fixed seed, offline, on the order of 5k tokens), with `λ` chosen on held-out generic-text
reconstruction `R²` and never touching evaluation items. Both bases then have write-backs that have only
ever seen generic text — res-jb's decoder was trained on generic text too — which is the accurate meaning
of equal footing. Train-split template activations are added to the fit sample (items still disjoint
from evaluation) to cover units that generic text never activates, plus a coverage check that ≥ 95% of
evaluation-active units have a non-zero decoder row.

Why this is not experiment 03 again: there, the *fitted readout* stood between the representation and
the conclusion, and its free parameters rewrote the answer. Here the fitted object sits inside the
intervention arm, its quality is capped independently by Gate C, and the headline statistic is
within-basis normalised, which divides decoder quality out to first order. The residual asymmetry is
real and must be reported: the SAE's decoder saw on the order of 10⁸ tokens, the ridge sees 5×10³. If the
random arm loses, that asymmetry is a live competing explanation and gets stated next to the number,
together with the Gate C values.

## Pilot — the go/no-go script, ≤ 12 minutes

One script. Every step prints its criterion; any STOP writes a `gated_out` manifest instead of a number.

1. **Environment and assertions** (~1 min). Reuse experiment 03's loader for GPT-2-small and the res-jb
   SAE. Assert `W_dec` is `[24576, 768]`. Assert `" is"`, `" are"`, `" was"`, `" were"`, and both control
   words are single tokens with the leading space; record their ids. Failure = STOP, configuration error.
2. **Machine self-tests** (seconds). (a) A zero-selection edit reproduces the clean logits bit-for-bit —
   this is the executable form of the scope rule. (b) Rescale `f` by a random positive diagonal and the
   decoder rows inversely; the written edit is bit-for-bit identical — the invariance claim, asserted.
   (c) Re-running from `start_at_layer=8` on cached clean residuals matches a full forward pass to
   `< 1e-4`. Failure = STOP, code bug, not science.
3. **Stimuli and Gate A** (~1 min). 60 single-flip pairs; per-item assertions on equal length and a
   single-token subject at a fixed index. One batched clean pass caching `hook_resid_pre` at layers
   {4, 6, 8, 10} and the final-position logits. Print retention rate, median `d_gap`, the `d`
   distribution. Below threshold = STOP and rebuild stimuli.
4. **Exactness self-check** (seconds). Swap residuals at every position from the subject onward at layer
   8; require `|d(patched) − d(source)| < 1e-3` per pair. Failure = STOP, hook indexing bug.
5. **Gate B handle scan** (~2 min). `resid` only, layers {4, 6, 8, 10} × positions {subject, final,
   both}. Print the recovery matrix and sign consistency. Choose the **smallest** position set at layer 8
   reaching 0.50, ties to the subject position. Nothing reaches it = STOP with `gated_out`, and the scan
   table itself is the deliverable.
6. **Gate C decoder prototype** (~4–6 min; generation dominates). Generate ~2k tokens from GPT-2 with a
   fixed seed, add train-split activations, build the matched random expansion with experiment 03's
   function, fit the dual ridge. Print held-out `R²`, evaluation-active coverage, and
   `E(sae_full)/E_resid`, `E(rand_full)/E_resid` on 30 pairs, plus the same for the tied-weight fallback.
   Both in `[0.7, 1.3]` = GO. Random basis fails both decoders = GO on the fallback rule. **SAE fails =
   STOP the headline**, and the deliverable is the residual-handle result written in this document's
   scope language.
7. **Ranking prototype and timing** (~2 min). 16 candidates, single-coordinate scores on 20 training
   pairs, `R(top-{1,4,8})` for both bases on 20 held-out pairs. Print measured seconds per forward and
   extrapolate the full run. Over 60 minutes triggers the pre-declared trim order: evaluation pairs
   150 → 100, then candidates 64 → 48, then drop `k = 2`. **Seeds and both directions are never cut.**

Pilot previews are not results. The decision rule is the one frozen above.

## Counts and seeds

240 pairs generated → at least 140 retained → 40 for rank-training, the remainder for evaluation, capped
at 150. Five seeds, `t(4) = 2.776`. Seed varies the lexical draw, the train/eval split, the random
encoder, the ridge fitting sample, and the `randk` draws; the SAE is fixed, because it is the object of
study. Ridge `λ` over eight log-spaced points in `[1e-4, 1e3]`, selected on held-out generic-text `R²`,
label-free throughout.

Note for implementation: experiment 03's `ci95` helper uses the `1.96` normal approximation. This
experiment uses `t(4)`. Do not reuse that function.

## Budget

Costs are forward passes; there is no probe fitting, which is where experiment 03 spent most of its 20
minutes. The projection is under an hour of M1 Pro CPU. **That is arithmetic, not a measurement** — pilot
step 7 measures the real per-forward cost and the trim order fires before the full run, exactly as
experiment 03's timing pilot did.

## Failure modes

- **No causal handle at layer 8.** Caught by Gate B in about two minutes. The SAE arm cannot move layers,
  so this ends the experiment rather than relocating it.
- **The random basis will not reconstruct.** Caught by Gate C, prototyped before any sweep, with the
  within-SAE fallback already carrying its own decision rule.
- **The ranking rule does the work instead of the basis.** Mitigated by the identical rule and budget
  across bases, the ranking-free `*_full` anchors, and the `*_randk` controls.
- **The effect is real but tiny.** Then the paired interval is wide and the honest outcome is
  "inconclusive" — which is why both branches of the decision rule carry a precision requirement.
- **The pairs are too hard, not too easy.** The readout is a continuous logit difference, so there is no
  ceiling to saturate; the real risk is a floor, where the model gets the harder member wrong and `d_gap`
  is noise. That is Gate A's job.

## What a reader should be able to conclude, in each branch

Positive: the SAE's coordinates concentrate one causal factor better than matched random mixing does,
measured without a fitted readout — a statement about basis quality that experiment 03's instrument
could not make.

Null: they do not, and the readout-capacity story from experiment 02 carries into the causal regime — the
benefit lives in mixing plus nonlinearity, not in the particular learned basis.

Either way the answer does not depend on a preprocessing convention, which is the entire point of
running it this way.

---

# Amendment 1 — 2026-07-27: the control arm, decided while blind

**Blinding declaration.** This amendment is written and committed *before* the Gate C diagnostic's
output is read. At the time of writing: `gate_c_diagnostic.json` has not been opened; the step-7 timing
preview rows in `pilot_results.json` (top-`k` recovery for both bases) have not been read; no AUC, no
concentration curve, and no `sae`-versus-anything comparison has been computed by anyone. The rules
below are therefore outcome-independent by construction, not by good intentions. That is the only
methodological asset available here, and it is worth more than a faster answer.

## What the pilot found, and why an amendment is needed

The pilot passed steps 1–5 and 7 and failed Gate C on the control arm:

| basis | `E(full)/E_resid` | pre-declared band |
|---|---:|---|
| SAE, trained decoder | 0.731 | pass |
| matched random, dual ridge | 0.588 | **fail** |
| matched random, tied weights | 0.298 | **fail** |

The pre-declared fallback therefore fired: ship the within-SAE comparison
`AUC(sae_topk) − AUC(sae_randk)`. On reflection that statistic is close to a sanity check —
`randk` draws uniformly from a candidate set already pre-filtered by a causal proxy, so ranking beating
uniform sampling is not news. It keeps two pieces of real content, and they survive below: the ranking's
transfer across items (the ranking is fitted on an item-disjoint split, so item-specific causal
coordinates would collapse `topk` onto `randk` — falsifiable, if unlikely), and the shape of `R(k)`
itself, which is a descriptive measurement about a real model.

So the fallback is reported, and it is not the headline.

## The decision rule, frozen

Let `r2k` and `r8k` be the random arm's `E(full)/E_resid` at ridge fitting budgets of 2,048 and 8,192
generated tokens, from `sweep_A_ridge_fitting_power` in `gate_c_diagnostic.json`.

The diagnostic was asked for four budgets `{2k, 8k, 32k, 128k}`; the script caps an exact dual solve at
8,000 rows and its generation pool at 8,192 tokens, so the larger budgets are expected to record as
`not_run`. That cap is the right call — it refuses to substitute a different estimator and call it the
same thing — and the rule below is written to work off two points.

**Rule 1 — under-powered fit.** `r8k ≥ 0.70` → the original design stands. Set the ridge budget to the
largest exactly-solvable value, re-verify Gate C within each seed, and change nothing else. The
amendment then records one line: *instrument repair, fitting budget 2k → 8k.*

**Rule 2 — structural.** `r8k < 0.70` and `r8k − r2k ≤ 0.03` → the failure is a property of a random
sparse code at matched L0, not of the fit. A 4× increase in data moving the number by at most 0.03,
against a 0.11 gap, on a decreasing learning curve, does not get closed by two more 4× steps. The `0.03`
knife-edge is defensible anywhere in 0.02–0.05 and is frozen here precisely so it cannot be picked
afterwards. Go to the ladder below.

**Rule 3 — ambiguous.** `r8k < 0.70` but `r8k − r2k > 0.03` → run the **symmetrisation discriminator**
before choosing: fit a decoder for the *SAE's own code* with the identical 8k dual ridge, giving `s8k`.
If `s8k ≥ 0.70`, the estimator can decode a good code at this budget and the random arm's shortfall
belongs to the code → structural → the ladder, and `s8k` becomes the fairness certificate required
below. If `s8k < 0.70`, the estimator throttles every code → pre-declared switch to a primal ridge
solved on active units only, re-run sweep A once, then re-apply Rules 1 and 2.

The symmetrisation discriminator costs a minute or two and is worth running in every branch. It is the
sharpest instrument available for separating "the code is worse" from "my estimator is weak", and the
whole point of experiment 04 is not to repeat experiment 03's mistake of letting an estimator's
limitation masquerade as a property of a representation.

## The ladder — the control arm if the failure is structural

Under Rule 2 or Rule 3-structural, the matched-random family is abandoned, and the reason is
constructive rather than defeatist. Every rescue for it collapses to the same object: a linear decoder
trained on generic activations converges to the least-squares limit the diagnostic already measures, so
"random code plus a properly trained decoder" is not a new control, it is the same control at
saturation. Shuffled-SAE variants either break the encoder–decoder correspondence (already recorded as
unable to pass Gate C) or, if permuted jointly, are a renaming and therefore a no-op.

So the control changes family, to bases that pass Gate C **by construction**: a complete orthonormal
basis has `decode ∘ encode = identity`, so `E(full) = E_resid` exactly and the Gate C ratio is 1 with
nothing fitted. Sparsity is then matched where the dependent variable actually lives — the *intervention
budget*: the same 64-candidate pre-filter, the same `k` grid, the same single-coordinate causal ranking,
the same scale-invariant proxy.

| arm | basis | role |
|---|---|---|
| `pca_topk` | PCA of layer-8 residuals, fitted on the same off-template generated text, label-free | **the new load-bearing control** |
| `neuron_topk` | the residual stream's own coordinates | secondary: the model's privileged basis, zero fitting |
| `mu_ref` | the supervised mean-difference direction from the training split | a `k = 1` reference line, drawn, never adjudicated |

PCA is the right primary control because it is the one that isolates the actual question. It has seen
the same data distribution as the SAE and is equally unsupervised; the only thing it lacks is the sparse
dictionary-learning objective. So `sae` versus `pca` asks **whether dictionary learning concentrates a
causal factor better than any data-adaptive basis does** — a sharper question than the original, and one
where the SAE can genuinely lose. It also escapes this document's own earlier objection to a random
orthogonal basis: that was a straw man because a fixed causal direction lands in `k` random directions
with expected mass `k/768`, whereas PCA's alignment with the number direction is an empirical fact that
could go either way, and number is behaviourally strong at layer 8.

`neuron_topk` is the classic features-versus-neurons question, costs nothing, and its full anchor
coincides with `resid_full`, so its `R(k)` is directly an absolute recovery fraction.

**Headline under the ladder:** `AUC(sae) − AUC(pca)`, with the frozen `t(4)`, the `0.05` paired
half-width, the `0.10` per-basis half-width, and the same three branches. `neuron_topk` and `mu_ref` are
reported without adjudication.

**What is given up, stated plainly:** complete bases are dense, so ambient L0 matching is gone. That
match existed to make an over-complete *random* code well defined; the dependent variable was always
"how many coordinates must be edited", and that is preserved exactly. The SAE keeps its 24,576-unit
over-completeness against PCA's 768 — an advantage that biases toward the SAE, which is worth saying out
loud, because it makes a null result *more* credible and a positive result correspondingly weaker.

## The Gate C failure as a finding — conditions for reporting it

`0.731` against `0.588` is not yet a claim, and this document predicted the reason before the pilot ran:
the SAE's decoder saw on the order of 10⁸ tokens and the ridge saw 2×10³. The pilot supplies direct
evidence that fitting quality is a first-order term — tied weights `0.298` against dual ridge `0.588`
differ *only* in how the decoder was fitted, and that doubles the number.

It may be reported as a secondary finding if and only if all four hold: the `s8k` symmetrisation
certificate exists; the full five-seed `t(4)` interval is computed (the pilot value is one seed, thirty
pairs, and sits 0.031 from the band edge); the wording is pinned to **faithfulness of a differential
write-back** and never slides into "the SAE represents number better"; and the symmetric half is
reported in the same breath — *the trained dictionary itself only writes back 0.731, losing 27% of the
causal effect*. That last clause is the most honest sentence available here and it is not optional.

Even fully certified it stays secondary, because it is a statement about write-back faithfulness, not
about concentration, and it does not survive the obvious retort that a trained autoencoder reconstructs
better than an untrained one.

## Faithfulness-matching, considered and reduced

Raising the random arm's L0 until its faithfulness matches the SAE's was the obvious repair and it is
rejected as a headline. Two reasons. L0 is not a nuisance variable here — it is the defining property of
the object under study, so buying validity with it turns the claim into a two-axis conditional. And the
outcome is close to geometrically pre-determined: a fixed causal direction's mass spreads as the basis
widens, so a higher-L0 random arm has a flatter concentration curve by construction, and the comparison
degenerates into measuring the SAE against a backdrop already known to be flat.

It survives as one number: sweep B's smallest L0 that passes Gate C, reported as *the L0 price of
randomness*.

## Unchanged by this amendment

Gate A, Gate B, Gate D, the invariance and exactness self-tests, the single-flip stimuli, both-direction
edits, within-basis normalisation, the AUC statistic with `R` clipped to `[0, 1]`, `k50` as a narrative
number, five seeds with `t(4) = 2.776`, and every threshold already frozen. All of them apply unchanged
to the new arms.

## A by-product worth a paragraph in the writeup

Gate B's layer × position scan is reported as method validation, not sold as a discovery: the causal
handle sits entirely at the subject position at layer 4 (0.677 against 0.005 at the final position) and
has divided roughly evenly by layer 8 (0.398 against 0.406, 0.840 together, slightly super-additive),
which is a direct trace of attention moving the number signal between those layers. It is consistent
with where the agreement causal-mediation literature localises this — flagged as an unverified
recollection, per this document's convention. It comes with an honest limitation: the SAE cache pins the
intervention to layer 8, mid-transport, while layer 4's subject position alone already carries 0.68 —
and there is no SAE there.

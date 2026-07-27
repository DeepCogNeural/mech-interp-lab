# Experiment 04 — Design: is an SAE basis a *causal* basis?

**Status: design only. Nothing in this document has been run, and no number below is a measurement.**
Written 2026-07-26, before implementation, so that the decision rule cannot be chosen after seeing the
data. When the run happens it gets the usual `writeup.md` + raw manifest beside this file, and this
document stays as the pre-registration.

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
property of the design, not a hope about the data.

## Scope — what this will not be allowed to say

Read this before the results exist, because a causal design sits closer to the line than a probe does.

The model is never modified. No SAE is inserted into GPT-2's forward pass, no component is replaced by
its reconstruction, and no feature is removed. The **add-back-error** construction is what keeps this
true: the edit is `+ W_dec (f' − f_base)`, a difference of two reconstructions, so the SAE's
reconstruction error cancels exactly and never enters the forward pass. GPT-2 runs unmodified apart from
one additive vector at one hook.

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

## Stimuli and behavioural readout

Subject–verb agreement minimal pairs with an attractor, the standard Linzen/Finlayson construction:

```
The {ADJ} {SUBJ} {PREP} the {ATTRACTOR} ___
```

e.g. *"The tired author near the paintings ___"* (singular subject, plural attractor) against *"The
tired authors near the painting ___"*. Roughly 150 pairs, drawn so the base and source members differ
**only** in the number-marked tokens and tokenise to equal length, asserted per item exactly as
experiment 03 asserts it. The attractor is the point: it forces the model to use structure rather than
the nearest noun, and it is the construction the agreement literature already validates.

Readout at the verb slot, on the model's own logits:

```
d(prompt) = logit("are") − logit("is")
```

`d` is signed so that plural pushes it positive. There is no probe, no training, and no preprocessing of
any representation before the readout.

## Arms

Let `resid_base`, `resid_source` be layer-8 `hook_resid_pre` activations for a matched pair, `f` the SAE
code, `g` the matched-random code, and `E(Δ) = d(patched) − d(base)`.

| arm | edit written into the residual stream | role |
|---|---|---|
| `resid_full` | `resid_source − resid_base` | basis-free gold standard; defines the denominator `E_resid` |
| `sae_full` | `W_dec (f_source − f_base)` | SAE faithfulness anchor; selection-free |
| `rand_full` | `W_dec_rand (g_source − g_base)` | random-basis faithfulness anchor; the equal-footing check |
| `sae_topk(k)` | the same sum restricted to the top-`k` ranked SAE coordinates | **the research object** |
| `rand_topk(k)` | the same, in the matched random basis | **the load-bearing control** |
| `sae_randk(k)` | `k` coordinates drawn uniformly from the active set | within-basis control: is the *ranking* doing anything? |
| `rand_randk(k)` | the same in the random basis | its counterpart |

`k` sweeps a log-spaced grid, roughly `{1, 2, 4, 8, 16, 32, 64, 128}`, capped by the number of
coordinates that are ever active at the edited position.

The `*_full` arms are ranking-free. They anchor the curves so that no choice of ranking rule can drive
the headline on its own — the lesson experiment 03 paid for.

## The ranking rule, and why it cannot smuggle in a knob

Coordinates are ranked by their **single-coordinate causal effect**: for candidate `i`, edit only that
coordinate and measure `E({i})`, averaged over a *training* split of pairs that is item-disjoint from
the evaluation split. This is scale-invariant for the same reason the whole design is — rescaling
latent `i` by `c` rescales its decoder row by `1/c` and leaves the written edit unchanged.

Measuring it for all 24,576 coordinates is unaffordable, and unnecessary: a coordinate inactive in both
base and source contributes exactly zero. So the candidate set is the union of active coordinates across
the training pairs (on the order of a few hundred), pre-filtered to the top ~64 by the cheap
scale-invariant proxy `‖ mean_pairs (f_source,i − f_base,i) · W_dec[i, :] ‖₂`, then scored causally.

**The identical rule, the identical pre-filter, and the identical budget are applied to both bases.**
That is the condition under which the comparison means anything.

## Primary measure and pre-declared decision rule

Recovery curve, per basis: `R(k) = E(top-k edit) / E_resid`, then the normalised area under it across
the `k` grid. The statistic is the **within-seed paired difference** `AUC(sae) − AUC(rand)` across five
seeds (lexical draw, random-expansion draw, train/eval split), reported as a two-sided Student-t(4) 95%
interval — `t(4) = 2.776`, matching experiment 03's convention.

Fixed before running:

- **Adjudicated positive** — the paired interval lies entirely above zero *and* the specificity gate
  holds. Reading: the SAE basis concentrates this causal factor into fewer coordinates than matched
  random mixing does.
- **Adjudicated null** — the interval contains zero *and* its half-width is at most `0.05`. Reading:
  sparsity does not buy causal localisation of this factor over matched random mixing. This extends
  experiment 02's null into the causal, real-model regime, and it is a result, not a failure.
- **Inconclusive** — anything else, reported as such, in experiment 03's words rather than dressed up.
  Overlapping intervals are not agreement when one of them is uninformative.

Secondary, reported but not headline: `R(k)` for the `*_randk` controls, the `k` at which each basis
first exceeds 50% recovery, and the per-item sign-consistency of the intervention.

## Gates, all cheap, all before the sweep

| gate | check | if it fails |
|---|---|---|
| A — behaviour | the clean minimal pairs flip: `d(base) < 0 < d(source)`, per-item sign consistency ≥ 0.8 | rebuild the stimulus set; nothing downstream is interpretable without this |
| B — causal handle | `E_resid` recovers ≥ 50% of the clean pair's logit difference at layer 8 | sweep layer and position and re-gate; do not proceed on a weak handle |
| C — equal footing | `R(sae_full) ≥ 0.7` **and** `R(rand_full) ≥ 0.7` | the two bases are not comparable; report the within-SAE result only, and say why |
| D — specificity | number edits move `is/are` but leave an unrelated contrast (a tense or lexical minimal pair) inside a pre-set tolerance | the edit is a generic perturbation, not a factor edit; the headline is withdrawn |

Gate C is the one that decides whether the headline comparison is even available, so it is checked on a
20-pair pilot before any sweep runs.

## The hard engineering step, isolated up front

The matched random expansion needs a decoder. The SAE has `W_dec` because it was trained; a random
encoder does not.

Primary: fit `W_dec_rand` by ridge regression from `g` back to `x` on activations drawn from the
**training item split only**, solved in dual form (`n × n`, since `n < p`) so it costs seconds rather
than minutes. Fallback if that fails Gate C: tied weights with a single fitted global scale, reported as
a weaker control.

This is the step most likely to eat the budget and the one that decides Gate C, so it gets prototyped
and gated **first**, before any pair sweep. If the random basis cannot be brought onto equal footing,
that fact is reported, and the experiment ships as the within-SAE ranked-versus-random-coordinates
result — smaller, still causal, still adjudicable.

## Budget

Costs are forward passes; there is no probe fitting, which is where experiment 03 spent most of its 20
minutes. Rough order: ranking ≈ 64 candidates × 30 training pairs × 2 bases × 5 seeds, plus evaluation ≈
150 pairs × ~20 configurations × 5 seeds. Batched on short sequences this should land inside an hour of
M1 Pro CPU.

That estimate is a projection, not a measurement. A seed-0 timing pilot runs first and the `k` grid or
the pair count is cut *before* the full run if the projection misses — the same discipline experiment 03
used, where the pilot projected 2.5 minutes and the run took 145 seconds.

## Failure modes

- **No causal handle at layer 8.** Caught by Gate B in about two minutes. Low risk; GPT-2-small number
  agreement is well documented, but "well documented" is not "measured here".
- **The random basis will not reconstruct.** Caught by Gate C. Mitigated by prototyping the decoder
  first and by having a smaller, still-shippable within-SAE result to fall back on.
- **The ranking rule does the work instead of the basis.** Mitigated three ways: identical rule and
  budget across bases, the ranking-free `*_full` anchors, and the `*_randk` controls that show what a
  ranking is worth at all.
- **The effect is real but tiny.** Then the paired interval will be wide and the honest outcome is
  "inconclusive" — which is why the null branch of the decision rule carries a precision requirement
  rather than resting on an interval that merely covers zero.
- **Attractor items are not actually hard.** If the model is at ceiling the intervention has no room.
  Checked inside Gate A by reporting clean accuracy, and the item set is built to include attractor
  interference for exactly this reason.

## What a reader should be able to conclude, in each branch

Positive: the SAE's coordinates concentrate one causal factor better than matched random mixing does,
measured without a fitted readout — a statement about basis quality that experiment 03's instrument
could not make.

Null: they do not, and the readout-capacity story from experiment 02 carries into the causal regime — the
benefit lives in mixing plus nonlinearity, not in the particular learned basis.

Either way the answer does not depend on a preprocessing convention, which is the entire point of
running it this way.

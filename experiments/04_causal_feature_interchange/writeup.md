# Experiment 04 — Is an SAE basis a *causal* basis?

**Design:** [`DESIGN.md`](DESIGN.md), frozen before implementation, with three dated amendments.
**Code:** `pilot.py`, `gate_c_diagnostic.py`, `run_experiment.py`, `robustness.py` (offline CPU, five
seeds). **Raw results:** `pilot_results.json`, `gate_c_diagnostic.json`, `run_results.json`,
`robustness_results.json`. **Figures:** `figures/`.

Every number below is recomputed from those manifests as a two-sided 95% Student-t interval,
`t(4) = 2.776` on five seeds, matching experiment 03's convention.

## The question, and why it is asked this way

Experiment 03 asked whether a real SAE code lets a linear probe read feature interactions better than
matched random mixing, and could not answer: the result swung by more than tenfold between two feature
scalings that are an *invertible affine reparameterisation* of a linear probe's input. The swing measured
the L2 prior's geometry, not the codes. That is a structural problem with fitted readouts, not a bad
choice of one.

So this experiment removes the readout. It edits coordinates of a code, writes the edit back into the
residual stream, and lets **GPT-2-small's own next-token logits** report the effect. The readout has no
free parameters.

The claim that makes this worth doing:

```
resid' = resid_base + W_dec (f' − f_base)
```

Rescale every latent by a positive diagonal `D` and the matching decoder by `D⁻¹`. The product is
unchanged, so `resid'` is bit-for-bit identical. **The knob that made experiment 03 unanswerable cannot
move this measure at all** — and that is asserted in code, not argued: the D-rescale self-test compares
bitwise and passes, in every run, alongside a zero-selection edit that reproduces the clean logits
bitwise and a prompt-swap exactness check at `5.72e-06`.

## Scope — what this does not say

The model is never modified. No SAE is inserted into GPT-2's forward pass, no component is replaced by
its reconstruction, nothing is ablated. The edit is `+ W_dec (f' − f_base)`, a *difference of two
reconstructions*, so the SAE's reconstruction error cancels exactly and never enters the forward pass.
GPT-2 runs unmodified apart from one additive vector at one hook.

A low recovery score in any basis therefore says that a factor is not concentrated in few coordinates
*of that basis*. It does not say — and may not be read as saying — that an SAE harms, degrades, removes,
or loses any computation the model performs. That question is not measured here.

One further narrowing that the run forced: the edited position set includes the final token, one step
from the readout. So the language throughout is **"localises the causal signal present at layer 8 at
these positions"**, never "localises the computation".

## Setup

Single-flip minimal pairs — within a pair only the subject noun changes, the attractor is held fixed and
counterbalanced across items:

```
The {ADJ} {SUBJ} {PREP} the {ATTRACTOR} ___          d = logit(" are") − logit(" is")
```

240 pairs generated per seed, both edit directions, layer 8 `hook_resid_pre`, position set `both`
(subject and final), `k` grid `{1,2,4,8,16,32,64,128}` after the pre-declared coverage trigger fired.
Coordinates are ranked by their **single-coordinate causal effect** on an item-disjoint training split,
pre-filtered by a scale-invariant proxy — the identical rule, pre-filter, and budget for every basis.

| arm | basis | role |
|---|---|---|
| `sae` | res-jb layer-8 SAE, trained decoder | the research object |
| `pca` | 768 PCA components of layer-8 activations, fitted label-free on generated text | **the load-bearing control** |
| `neuron` | the residual stream's own coordinates | secondary, zero fitting |
| `rand_exp` | width/norm/L0-matched random expansion with a ridge decoder | reported, not adjudicated |
| `mu_ref` | the supervised rank-one mean-difference direction | a drawn reference line, never adjudicated |

## How the control arm got here

This matters more than the result, so it is not buried. The original control was `rand_exp`, matched to
the SAE in width, column norm, and per-sample L0. **It failed Gate C**: it wrote back only `0.588` of the
residual-stream effect, against a pre-declared floor of `0.70`.

Rather than treat that as fact, a diagnostic asked whether it was an under-powered ridge fit. Quadrupling
the fitting budget moved it the wrong way — `0.588` at 2,048 tokens, `0.554` at 8,192 — so under the
frozen rule the failure is structural. An L0 sweep priced it: a random expansion needs its per-sample L0
roughly **doubled**, from the SAE's measured 72.76 to 144, before it writes back as faithfully
(`0.704`), and 1,536 to reach `0.888`.

The control therefore changed family, to bases that pass Gate C **by construction** — a complete
orthonormal basis has `decode ∘ encode = identity`, so no decoder is fitted and nothing can be tuned.
Sparsity is matched where the dependent variable lives: the intervention budget. PCA became the control
because it has seen the same data distribution and is equally unsupervised; the only thing it lacks is
the sparse dictionary-learning objective. The measured PCA identity residual on the effect is `4.77e-07`
against a threshold of `4.22e-03`.

That decision, and its decision rule, were written into `DESIGN.md` and committed **before the diagnostic
output was read** — the commit history shows the order.

## Result 1 — the verdict, which is `inconclusive`

> Under the frozen decision rule this run is **inconclusive**: Gate C requires each adjudicated basis to
> write back at least `0.70` of the residual-stream effect, and the SAE's trained decoder measured
> `0.694 ± 0.014`, passing in two of five seeds (`0.7046, 0.7072, 0.6854, 0.6832, 0.6893`). The
> measurement the rule withholds judgement on is nevertheless large and stable: ranked SAE coordinates
> concentrate the number-agreement interchange effect into far fewer coordinates than ranked PCA
> components — paired AUC difference `+0.304 ± 0.023` within-basis, `+0.146 ± 0.022` under the
> conservative absolute normalisation, `k50` of 16 against 64-to-beyond-128, positive in all five seeds
> under both conventions.

The verdict is double-locked, and it was checked for exactly that. No aggregation rescues Gate C: the
pooled five-seed mean is `0.6939`, still under the floor. Independently, in seed `20260803` PCA never
reached `R ≥ 0.5`, so Gate D is unevaluable there and blocks an adjudicated positive on its own. Two
separate frozen gates would have to be re-read to overturn this.

**A Gate C relaxation was considered and refused.** The threshold was fixed before the number existed;
moving it by 0.006 after seeing a favourable effect is the failure this experiment was built to avoid.
Experiment 03 is in this repository because I already made the softer version of that mistake once.

## Result 2 — the measurement the rule withholds judgement on

![Recovery curves per basis against the number of edited coordinates](figures/01_recovery_curves.png)

| basis | AUC, within-basis | AUC, absolute | `k50` (per seed) |
|---|---:|---:|---|
| `sae` | **0.517 ± 0.002** | **0.359 ± 0.007** | 16, 16, 16, 16, 16 |
| `pca` | 0.213 ± 0.023 | 0.213 ± 0.023 | 64, 128, —, 128, 64 |
| `rand_exp` | 0.179 ± 0.024 | 0.117 ± 0.016 | never reached |
| `neuron` | 0.157 ± 0.005 | 0.157 ± 0.005 | never reached |

Sixteen SAE coordinates recover half of that basis's own causal effect, in every seed. PCA needs 64 to
128 and in one seed never gets there within the grid; the model's own residual-stream coordinates and a
matched random expansion never reach half at any `k` on the grid.

**Two normalisations, reported as a bracket rather than a choice.** Within-basis normalisation divides by
each basis's own full effect; since the SAE writes back `0.694` and PCA writes back exactly `1.000`, the
SAE gets a denominator 31% smaller. That is the mirror image of the inflation the design was written to
prevent. The absolute normalisation gives the unwritten 31% no credit at all. The two conventions bracket
the truth from opposite sides, PCA's number is identical under both, and **both endpoints are positive in
all five seeds**. The frozen statistic happens to be the larger of the two, which is why the smaller one
is stated in the same sentence.

## The obvious deflation, tested

If the ranked SAE latents were token-identity detectors for the specific nouns used, the result would be
an artifact of a 20-noun vocabulary. Three recomputations from the existing manifest, all post-hoc and
all reproducible from `run_results.json`:

- **Noun-disjoint transfer.** Restricting evaluation to pairs whose subject noun never appears as a
  subject in that seed's ranking-training split gives `R_sae(16) = 0.594` across 150 directed edits,
  against `0.588` on the full set. No difference.
- **Cross-seed stability.** The five seeds' top-16 latent sets share a 12-latent intersection, with mean
  pairwise overlap 13.3 of 16 — the ranking finds nearly the same latents from independent draws.
- **Sign consistency at `k = 16`.** `sae` 1.0000, `pca` 0.9993, `neuron` 0.9993, `rand_exp` 0.9627.

The candidate-pool objection — the SAE ranks 128 coordinates out of 24,576, PCA out of 768 — is answered
by `rand_exp`, which is that null hypothesis implemented: the same 24,576-wide pool, the same pre-filter,
the same ranking rule, L0-matched. It finishes **last** in absolute terms. Pool width alone does not buy
concentration. Neither does the SAE's pre-filter: `sae_randk`, drawing at random from the SAE's own 128
causally pre-filtered candidates, scores 0.15–0.18 — below PCA's ranked curve.

## Robustness — specified after unblinding, never adjudicating

Blinding is spent: the primary result had been read before these were specified. They are labelled that
way in `robustness_results.json` and they cannot change the verdict.

**The strongest surviving objection is that PCA is under-sampled** — 8,192 tokens for a 768-dimensional
covariance, about 10.7 samples per dimension, against an SAE trained on the order of 10⁸ tokens. Sampling
noise rotates axes inside near-degenerate eigenvalue subspaces, which would spread a direction a
converged PCA could concentrate and bias the control downward. Measured: `AUC(pca)` is `0.195 ± 0.008` at
2,048 tokens and `0.213 ± 0.023` at 8,192, a change of `+0.018 ± 0.025` whose interval crosses zero.
Four times the data moves the control by about 0.018 against a gap of 0.304. **That bounds the objection;
it does not retire it** — no flatness threshold was pre-declared, so this is not adjudicated, and the
eigenvalue spectrum is still moving (the top-64 variance share falls from 0.530 to 0.462), so the basis
is not converged. A larger fit remains the honest way to close it.

**The `sae_ridge` arm** takes the SAE's own code with a ridge decoder, which clears every gate. Its
ordering agrees: `AUC(sae_ridge) − AUC(pca)` is `+0.348 ± 0.037` within-basis and `+0.285 ± 0.031`
absolute, with `k50 = 8` in every seed. So the ordering survives on an arm that passes Gate C
comfortably — as a robustness check only.

## A correction the clean control forced

The symmetrisation certificate — the SAE's own code decoded by an 8,192-token ridge — measured
`0.887 ± 0.013`, well above the SAE's published decoder at `0.694 ± 0.014`. Read alone that says a small
ridge writes this factor back better than a trained dictionary does.

**The clean control reverses it.** That ridge's fitting rows were 4.7% train-split template activations,
added to meet a coverage requirement. Refitting on *generic text only* drops coverage to 0.931–0.945 —
below the 0.95 requirement, which is why both versions are reported — and drops faithfulness to
`0.414 ± 0.061`, far *below* the published decoder's `0.694`. The 0.887 figure depended on the fit having
seen the stimulus family.

The fair reading is therefore favourable to the SAE and narrower than the headline number suggested: a
decoder trained for global reconstruction over a broad corpus beats a generic-only linear decoder at
writing back differences on this narrow slice by a wide margin, and only loses to one that has been shown
the slice. Both figures describe one additive write-back of a code difference at a single hook; neither
measures any effect of an SAE on the model's computation. I reported the uncorrected version to myself
before running the control, which is the argument for running it.

The certificate still does its job: an 8k ridge reaches `0.887` on SAE codes and only `0.652 ± 0.039` on
the matched random codes, so the estimator is not the bottleneck and the random arm's shortfall belongs
to the code.

## The scale, stated soberly

`mu_ref` — a single **supervised** direction, the rank-one mean difference between conditions — recovers
`0.549 ± 0.017` of the residual-stream effect on its own. Against that, the SAE's single best latent
recovers `0.072 ± 0.003`, and it takes 32 SAE coordinates (`0.506 ± 0.012`) to approach one supervised
direction.

So the finding is not that an SAE puts this factor into one coordinate. **No unsupervised basis here
does.** It is that ranked SAE coordinates approach a supervised direction with roughly four to eight
times fewer coordinates than ranked PCA components do.

## Gates and controls

| gate | threshold | measured |
|---|---|---|
| A — behaviour | ≥ 0.60 both-correct, ≥ 140 retained, median `d_gap` ≥ 1.0 | 236/230/235/237/234 retained of 240; pass every seed |
| B — causal handle | mean `E_resid/d_gap` ≥ 0.50, sign consistency ≥ 0.90 | 0.815–0.837, sign consistency 1.000 every seed |
| C — write-back | `E(full)/E_resid` ∈ [0.70, 1.30] | `sae` 0.694 ± 0.014, **fails 3 of 5 seeds**; `pca`/`neuron` 1.000 by construction |
| D — specificity | `S(top-k*) ≤ S(resid_full) + 0.15`, `S(resid_full) ≤ 0.5` | `S(resid_full)` 0.080–0.087; passes 4 of 5 seeds, unevaluable in seed `20260803` |

Cross-tense generalisation (`is/are` and `was/were` sharing sign) is 1.000. Self-tests passed in every
run: zero-selection bitwise, D-rescale bitwise, `start_at_layer=8` max abs `0.0`, prompt-swap
`5.72e-06`.

## A by-product, reported as method validation

The Gate B layer × position scan is not a discovery, but it is a clean picture. At layer 4 the causal
handle sits **entirely at the subject position** (0.677 against 0.005 at the final position). By layer 8
it has divided roughly evenly (0.398 against 0.406, 0.840 together, slightly super-additive). That is a
direct trace of attention moving the number signal between those layers, and it is why the position set
is `both`. It comes with a limitation: the cached SAE pins the intervention to layer 8, mid-transport,
while layer 4's subject position alone already carries 0.68 — and there is no SAE there.

## What actually ran

Pilot 42.3 s; Gate C diagnostic 647 s; the five-seed main run 1,760.0 s; robustness 1,344.0 s. All CPU
(Apple M1 Pro), offline, float32. No trims fired in the main run and nothing is listed `not_run`.

The pilot's projection of 1.16 minutes for the main run was wrong by a factor of 25 — it counted patched
forward passes and omitted per-seed text generation, the PCA fit, and two exact dual-ridge solves. The
measurement supersedes the projection.

Two runs stopped at a numerical assertion on the complete-basis identity (`1.34e-4` against an absolute
`1e-4` bound) and refused to loosen it. The bound was mis-specified by me — an absolute tolerance on
activations whose entries run to tens is a float32 round-off test, not a test of the claim — and
Amendment 1a restates it relatively and on the effect itself. No scientific threshold moved.

## Caveats

- **Not adjudicated, by the frozen rule.** Unlike experiment 03 this is not an instrument failure: the
  measurement is precise and stable and the invariance it rests on is verified bitwise. It is a gate
  refusing to certify. What would settle it: a run where the SAE arm clears Gate C on its own decoder,
  or a re-registered design whose Gate C is defined on an arm that does.
- One template family, one layer, one position set, one model, one SAE, five lexical draws. The five-seed
  intervals do not cover the variance of choosing a different SAE.
- Single-coordinate ranking ignores redundancy and interaction between coordinates. Both adjudicated
  bases receive the identical approximation.
- PCA is fitted on 8,192 tokens of model-generated text. Bounded above, not retired.
- The `sae_ridge` and `s8k` numbers involve a fit whose rows include template activations; the clean
  refit fails the coverage requirement and both are reported.
- Number agreement is one factor. Nothing here generalises to other computations.

## Next

- Re-register a Gate C that the SAE arm can clear on its own decoder — most naturally by defining
  faithfulness per arm on the decoder that arm actually ships with, and pre-declaring the floor from a
  pilot measured on a *different* stimulus family so the threshold cannot be tuned to this one.
- Close the PCA fitting-budget objection with a fit an order of magnitude larger, or with real corpus
  text rather than model-generated text.
- A second stimulus family, before any of this is allowed to generalise beyond number agreement.
- The mechanism question the concentration result opens: what are the 12 latents that recur across all
  five seeds? That needs a pre-registered feature-level analysis, not a post-hoc hunt.

# Experiment 05 — Which heads move the number signal, and are the twelve latents on the path?

> **STATUS: DRAFT — not yet frozen.** This design becomes frozen at the commit that first contains
> it with the calibration constants filled in (§ Order of operations). Until then it may be edited
> freely. After freezing, changes happen only as dated amendments, per the convention of
> experiments 02–04.

## The question

Experiment 04's Gate B scan measured *where* a residual-stream interchange has an effect: at layer 4
the causal handle for subject–verb number agreement sits almost entirely at the subject position
(`0.662 ± 0.012` of the clean logit gap), and by layer 10 it has largely arrived at the final
position (`0.694 ± 0.012`). Something relocates that signal. The scan measured no attention pattern,
head, or path, so it says nothing about what.

This experiment asks the mechanism question directly, and a second question that experiment 04
opened and could not answer. It is structured as **four axes, each with two pre-declared verdicts,
both publishable**, so that no single failed condition can drag the others into hedging and no axis
can end `inconclusive`:

| axis | question | verdicts |
|---|---|---|
| Q1 | do few heads carry the signal at the final position? | **Localised** / **Diffuse** |
| Q2 | is what they carry *number*, or any perturbation? | **Number-specific** / **Not-number-selective** |
| Q3 | do they carry it *from the subject position*? | **Subject-transport** / **Final-position-local** |
| Q4 | do the twelve replicated latents span the transported delta? | **On-path** / **Not distinguishable from matched chance** |

The twelve latents are ids `8922, 8952, 13352, 13594, 15165, 17956, 19093, 19955, 21401, 21581,
21805, 23011` of the res-jb layer-8 SAE — the intersection of the top-16 causal rankings across all
five experiment 04 seeds. That intersection was computed *post hoc* in experiment 04 and is
promoted here to a pre-registered target: the ids above are frozen constants, fixed before any
measurement in this experiment exists.

## What this is not

- No claim about SAEs harming, degrading, removing, or losing any computation. The model is never
  modified; every intervention swaps or projects activations the model itself produced.
- Q4 asks whether twelve specific latent directions *span a causally transported delta at one
  layer, on one task family*. It cannot become "the SAE found the number circuit", and a negative
  cannot become a claim of SAE failure — twelve latents chosen by top-16 ranking are not the SAE.
- Q1's Localised verdict, on its own, is a statement about heads *writing at the final position*,
  one step from the readout. Only together with Q2 (number-specific) and Q3 (subject-transport)
  does it become a mechanism claim, and the writeup may not merge the axes.
- One model (GPT-2-small), one stimulus family (the experiment 04 templates), one readout. Nothing
  here generalises past that without new experiments.

## Process rules carried over, and one changed

Carried over from experiments 02–04: the design freezes by commit before the measurements it
governs exist; changes after freezing are dated amendments; negative and positive verdicts are
written with equal prominence; what the commit record establishes is commit order, not the order
in which anything was read, computed, or run.

Changed, and recorded in the direction review before this design was drafted: experiment 04's
Gate C imposed a binary certification threshold on a continuous quality measure its own primary
statistic had already normalised away, and the run went `inconclusive` over a `0.694` against a
`0.70` floor. Here, **adjudication rests only on effect sizes against nulls that shrink with data**
(bootstrap intervals, measured wrong-source floors, matched-dimension chance bands). Continuous
diagnostics are reported covariates. Sanity floors exist only where failure means the harness is
broken, and they are stop conditions, not verdicts. **Boundary language is pre-declared** (§ below)
so that a near-threshold outcome has its sentence written before the number exists.

This design also inherits the two corrections experiment 04 published: an interchange written
through a code is an approximation (reconstruction-error differences do not cancel) — Q4's
intervention therefore never uses the encoder, so no reconstruction term exists anywhere in it —
and commit order is all the record proves, which is why the blinding section below claims exactly
that and no more.

## Reused, unchanged, from experiment 04

- **Stimuli**: single-flip minimal pairs, `The {ADJ} {SUBJ} {PREP} the {ATTRACTOR} ___`, subject
  noun flipped within a pair, attractor fixed and counterbalanced. Generation code and the Gate A
  retention filter reused verbatim (experiment 04 `run_experiment.py`; Gate A retained 230–237 of
  240 in every seed there). Subject position is as recorded in experiment 04's stimulus records;
  the noun vocabulary is single-token by construction there, and if any variant ever yields a
  multi-token subject, the frozen convention is the **last subword**.
- **Readout**: `d = logit(" are") − logit(" is")` at the final position, from the model's own
  unembedding. No fitted readout anywhere.
- **Causal-handle sanity floor** (stop condition, not a verdict): the full residual interchange at
  layer 8, positions `both`, must reach `E_resid/d_gap ≥ 0.50` with sign consistency `≥ 0.90`
  (experiment 04 measured 0.815–0.837, sign 1.000). Failure means the harness is broken.
- **Self-tests**: zero-intervention bitwise identity; `start_at_layer` equivalence; prompt-swap
  exactness. Thresholds unchanged from experiment 04.

## Seeds and intervals

- **Stages 1–2 (Q1–Q3): eight seeds** — `20260801…20260805` (kept for direct comparability with
  experiment 04's Gate B numbers) plus `20260806…20260808`. Experiment 02's own history is the
  reason for eight rather than five: its 4-seed pilot was overturned by an 8-seed run.
- **Stage 3 (Q4): eight seeds `20260806…20260813`** — none of which is an experiment 04 seed, so
  the twelve target latents were selected on data disjoint from every seed that adjudicates them.
  Selection contamination is closed by construction, not argued about.
- Seeds control lexical draws only; model, SAE, and template family are fixed, so every interval
  below covers lexical sampling variance and nothing else. Eight-seed summaries are two-sided 95%
  Student-t intervals, `t(7) = 2.365`.
- All `≥ 6 of 8` clauses below mean: **the same six (or more) seeds satisfy every sub-condition of
  that clause jointly.** No mixing of seed subsets across sub-conditions.

## Notation

For a minimal pair, `base` and `src` are the two runs; both flip directions are always run and
sign-aligned so that successful transport of the source's number is positive. `Δd(·)` is the
per-item signed change in the readout; `E(·)` is its mean over retained pairs. Three denominators
are always reported together:

- `E_all` — the joint patch of **all 144 heads'** per-head attention outputs (`hook_z`) at the
  final position, source→base: the same-family ceiling. **Adjudication divides by `E_all`.**
- `E_ref` — experiment 04's layer-8 residual interchange at positions `both`: the cross-family
  reference.
- `d_gap` — the clean behavioural gap.

## Wrong-source constructions (the specificity instruments)

Three source types, each answering a different question, all built by the reused generator:

- **Source A — same-number, different-noun**: the source sentence's subject has the *same* number
  as the base, different noun. Carries no number flip; a lexical-sensitivity control.
- **Source B — attractor flip**: the base sentence with the *attractor's* number flipped, subject
  untouched. Asks whether the heads read number from the subject or from any noun in range — the
  classic agreement-attraction manipulation. **B never adjudicates**: models plausibly show
  attraction, and either outcome is reported as a finding about what the heads read.
- **Source C — cross-template, number-matched**: a sentence from a different template family whose
  subject number equals the base's. This is the "writes an arbitrary strong vector" control —
  failure mode 2 below is guarded by C, not by A.

**Calibration constants, measured before freezing** (residual-level only, so the head-level
blinding claim is untouched): the pre-freeze calibration pilot measures, at layer 8, positions
`both`, the full-residual ratios
`ρ_full^A = |E(Δd_A)| / |E(Δd_right)|` and `ρ_full^C = |E(Δd_C)| / |E(Δd_right)|`.
The frozen text records their values here:

> `ρ_full^A = ` **[TO BE FILLED BY CALIBRATION PILOT BEFORE FREEZE]**
> `ρ_full^C = ` **[TO BE FILLED BY CALIBRATION PILOT BEFORE FREEZE]**

Both the bias-type ratio above and the noise-type companion `mean|Δd|` version are recorded; the
bias-type adjudicates. The specificity bound for Q2 is
`θ_spec = max(0.20, 2·ρ_full^C)` for source C and `max(0.20, 2·ρ_full^A)` for source A. The `0.20`
floor is tied to the `0.50` recovery bar: a set that clears both has a number-specific component of
at least `0.30·E_all`, so the two thresholds jointly guarantee a majority-specific effect rather
than being two unrelated numbers.

## Stage 1 — coarse ranking: 144 heads, one seed

On seed `20260801` only: for every head `(L ∈ 0…11, H ∈ 0…11)`, patch that head's `hook_z` at the
**final position only**, source→base, and record per-item `Δd`. Rank heads by **signed** `E(Δd)`,
descending. Heads with large *negative* effects are reported separately as candidate suppressors
(covariate; they do not enter the sets). For the top 24 heads by signed effect, the same patch is
also run through `hook_v` at the **subject position** (used descriptively in Stage 2's covariates;
the 144-head `v` sweep is deliberately not run — 144 numbers nobody will use invite post-hoc
fishing).

Stage 1 adjudicates nothing. Its rank order mechanically determines Stage 2's candidates; no human
choice intervenes between Stage 1 and Stage 2.

## Stage 2 — Q1, Q2, Q3: eight seeds

Per seed, three sweeps/runs, all mechanical:

1. the full 144-head `z@final` sweep with the true source (per-item `Δd` retained per head);
2. the full 144-head `z@final` sweep with **source A** (the empirical no-number-signal sweep);
3. one joint all-144-head patch (defines `E_all`), plus the nested-set and path-patch runs below.

**Individually distinguishable** (per head, per seed), both conditions required:

- the two-sided 95% bootstrap interval of its mean `Δd/E_all` (10,000 resamples over pairs,
  Holm-corrected across the 144 heads) excludes zero, **and**
- its `|E(Δd)|` exceeds the **measured noise floor**: the 99th percentile of `|E(Δd_A)|` across all
  144 heads in that seed's source-A sweep.

Both components shrink toward zero as data grows and neither references rank. (The draft of this
design used the 45th–144th ranked heads as the "null distribution"; that construction is a rank
threshold, not a noise band — its edge cannot shrink with data and "exceeds the band" reduces to
"is ranked ~top-45". It is deleted, and cross-seed rank stability is reported as a covariate
instead.)

**Candidate pool and sets**: `C` = the heads among Stage 1's top 10 (signed rank) that are
individually distinguishable and sign-consistent `≥ 0.90` across flip directions in seed
`20260801`. Nested sets `S_1 ⊂ … ⊂ S_min(8,|C|)` follow Stage-1 rank order within `C`. (Filtering
before nesting means one junk head cannot poison every larger set, and set membership is fixed
mechanically before any joint number is seen.)

**Q1 — frozen rule:**

- **Localised**: there exists `n ≤ 8` such that in `≥ 6 of 8` seeds jointly: (a)
  `E(S_n)/E_all ≥ 0.50`, and (b) every head in `S_n` is individually distinguishable in that seed.
  The claimed set is the smallest such `n`.
- **Diffuse**: no such `n`. Reported as the finding "no ≤8-head set carries half the
  attention-mediated final-position effect", with the full `E(S_n)/E_all` curve shown.

**Q2 — frozen rule** (tested set `S*` = the claimed set if Localised, else `S_8`):

- **Number-specific**: in `≥ 6 of 8` seeds, `|E(Δd_C(S*))| ≤ θ_spec^C · |E(Δd_right(S*))|` and
  `|E(Δd_A(S*))| ≤ θ_spec^A · |E(Δd_right(S*))|`.
- **Not-number-selective**: otherwise, stated affirmatively: "the set moves the readout as strongly
  for a number-matched source as for a number-flipped one; it is not number-selective at this
  position." Source B's attraction profile is reported alongside, never adjudicating.

**Q3 — frozen rule** (same tested set `S*`): the **two-step path patch** replaces, for each head in
`S*`, the value stream at the *subject position* with the source's (`hook_v`, base attention
patterns preserved), recomputes that head's output, and splices **only the final-position slice**
of the recomputed output into an otherwise clean base forward.

- **Subject-transport**: in `≥ 6 of 8` seeds, the joint path-patched effect reaches `≥ 0.50` of the
  same set's direct `z@final` joint effect.
- **Final-position-local**: otherwise — the set's output at the final position carries the effect,
  but it does not arrive via what these heads read at the subject position. Scope sentence, frozen
  now: v-patching preserves base attention patterns, so a head that routes number through its
  QK pattern rather than its value stream is invisible to this probe, and Final-position-local
  cannot be read as "no transport exists", only as "value-stream transport from the subject
  position was not shown".

**Covariates (reported, never adjudicating):** additivity ratio `E(S_n) / Σ_h E({h})` — with the
frozen caveat that LayerNorm's gain renormalisation makes single-head effects non-additive by
construction, so departures from 1 are not evidence of head-to-head interaction by themselves;
per-head `v@subject` vs `z@final` ratio for the top-24 (named **subject-value dependence** — the
negation direction is the only licensed inference: a head with no `v@subject` effect is not reading
its contribution from the subject's value stream; a nonzero effect does *not* show that head itself
delivers the signal, since a v-patch propagates to every downstream reader); cross-seed rank
stability of the top 10; sign consistency per head (a head failing `0.90` in a seed fails
distinguishability there via the bootstrap in practice, and is additionally reported).

## Stage 3 — Q4: the twelve latents, on selection-clean seeds

The twelve latents' decoder rows span a 12-dimensional subspace `V ⊂ R^768` at layer-8
`hook_resid_pre`. Let `δ = x_src − x_base` — **the true activation delta; no encoder, no
reconstruction term**. Three interventions per position set: add `δ` (reference), add `P_V δ`
(span component), add `(I − P_V) δ` (complement). All three position sets from experiment 04's
scan are run — `subject`, `final`, `both` — and **`both` adjudicates**, because it is the position
set under which the twelve latents were originally selected; `subject` is the transport-relevant
secondary.

**Statistics** (per seed, position set `both`): `R_span = E(P_V δ) / E(δ)`,
`R_comp = E((I−P_V) δ) / E(δ)`.

**Matched chance band**: per seed, recompute experiment 04's causal pre-filter to obtain that
seed's 128-candidate latent pool; draw 12-latent subsets from that pool (excluding the twelve),
build the same projection, measure `R_span`. Seed `20260806` runs 100 draws to fix the band's
shape; every other seed runs 20 draws as a stability check. The band edge is the 99th percentile.
(A uniform draw from all 24,576 latents is a straw null — most latents never activate on this
family — and is reported as context only.)

**Q4 — frozen rule:**

- **On-path**: `R_span` exceeds the matched chance band's edge in `≥ 6 of 8` seeds. `R_span` itself
  is reported as a graded effect size with its `t(7)` interval — **there is no pass/fail floor on
  its magnitude.** (The draft imposed `R_span ≥ 0.50`; that is a binary certification threshold on
  a continuous quality measure — the exact Gate C construction this design exists to retire — and
  its "anchor" to experiment 04's k50 conflated two different quantities. Deleted.)
- **Not distinguishable from matched chance**: otherwise.

**Pre-declared graded sentence** for the middle case: if `R_span` beats the band but is modest, the
frozen sentence is — "the twelve latents span more of the causal delta than matched-dimension
chance, and the fraction is `X`; this is a graded statement about a subspace, not a certification."

**Linearity/saturation control** (cheap, guards the projection reading): the `δ`-addition is also
run at `α ∈ {0.5, 1.0}`; if effects are strongly sublinear in `α`, or `R_span + R_comp` departs
far from 1, the projection decomposition is reported as non-additive and Q4's fraction is
explicitly downgraded to "band comparison only". Both diagnostics are covariates with that one
pre-declared consequence for the *wording*, not for the verdict.

**Context arms (reported, never adjudicating)**: the 12-dimensional span of that seed's top-12 PCA
components (experiment 04's PCA procedure refit per seed) — context for "how much a dense basis's
best 12 directions span", kept out of adjudication so this does not regress into the basis beauty
contest the direction review closed; geometric fractions `‖P_V δ‖²/‖δ‖²` — descriptive only, since
a span can hold norm without holding effect and vice versa.

If Q1 returns Localised, one additional covariate connects the axes: for each head in the claimed
set, its Q3 path-patch effect when `δ` at the subject position is first replaced by `P_V δ` — this
was cut from adjudication (a ratio of two total effects under different interventions has no null
and defies interpretation) but is recorded in the manifest for completeness, unplotted.

## Prior-work anchor (source-verified 2026-08-02; `notes/lit-check-number-agreement-2026-08-02.md`)

**No peer-reviewed head-level causal baseline exists for subject–verb agreement in GPT-2-small.**
Finlayson et al. (ACL 2021) is neuron-level causal mediation; its appendix's head-level GPT-2-small
readings are described by the authors themselves as possibly "simply noise" and are not usable as
ground truth. The strongest citable prior is layer-level: Lepori et al. (COLM 2024) localise
GPT-2-small agreement to **layer 6's attention block**; Finlayson et al. report causal effects
concentrated in middle-to-upper layers. One non-peer-reviewed preprint (Africa 2025,
arXiv:2506.22105) reports a 12-head path-patching circuit with layer 11 head 7 strongest — used as
a consistency comparison only, labelled as unreviewed.

**Pre-registered prediction, recorded before Stage 1 exists**: the top transport heads lie in
layers 5–10, consistent with Lepori's layer-6 localisation, Finlayson's mid-to-upper profile, and
experiment 04's own Gate B scan (the handle leaves the subject position between layers 4 and 10).
Agreement or disagreement with all of the above is reported descriptively; none of it enters any
decision rule, which depend only on this experiment's own measurements.

## Order of operations, and blinding

1. This design is committed as a draft (this commit).
2. **Calibration pilot, before freeze** — restricted to residual-level interventions and harness
   timing so the head-level blinding claim stays intact: (i) build sources A/B/C and pass Gate A;
   (ii) measure `ρ_full^A`, `ρ_full^C` at layer 8; (iii) run the self-tests; (iv) measure,
   *separately*, per-patch forward cost per stage and per-seed fixed overhead (stimulus
   generation, Gate A, activation caches, model/SAE load). Experiment 04's pilot under-projected
   by 25× precisely by scaling the per-patch cost alone — 71% of its runtime was non-forward
   overhead — so the projection here is `Σ(per-patch × counts) + (per-seed overhead × seeds)`,
   both terms measured, and the projection and the eventual measured total are both published.
3. The constants are written into this file; **the freeze commit is the commit containing them.**
   Decision rules and thresholds above are already final and may not change at fill-in.
4. Stage 1 runs and is read; Stage 2's candidate pool follows mechanically.
5. Stages 2–3 run under the frozen rules. Anything specified after their numbers are read is
   labelled post-hoc, as in experiment 04.

What is honestly claimed about blinding, and no more: the experimenter has read experiment 04's
Gate B table, knows the signal moves between layers 4 and 10, has recorded the layer-5–10
prediction above, and knows the twelve latent ids. **No head-level or span-level measurement has
ever been run in this repository**, and the calibration pilot is restricted so that this remains
true at the freeze commit. What the commit record will establish is commit order, not read order.

## Runtime budget

Measured basis (experiment 04 `forward_timing_records`): a full hooked forward over a retained
pair set is ≈ 1.2–1.4 s; a `start_at_layer=8` partial forward ≈ 0.4–0.7 s. Projected forward
costs: Stage 1 z-sweep ≈ 100–140 s per seed; Stage 2's two 144-head sweeps ≈ 4–5 min per seed,
nested sets and path patches ≈ 1–2 min per seed; Stage 3 ≈ 2–3 min per seed including draws. Total
forward budget ≈ 50–70 CPU-minutes across both seed groups, before per-seed fixed overhead —
which experiment 04 measured at 71% of wall-clock and which the calibration pilot prices
separately. **Hard cap: 120 CPU-minutes of projected total.** If the calibration projection
exceeds it, the permitted trim is pair count per seed, to no fewer than 100 retained pairs, and
past that the design must be amended *before* the main run in a dated amendment stating what was
cut. The projection and the measured total both go in the writeup.

## Failure modes, named in advance

1. **Heads compose** — single-head ranking misses a head that matters only jointly. The claim is
   about joint sets and additivity is reported, but a head outside Stage 1's top 10 that acts only
   jointly is invisible to this design; if Diffuse is returned the writeup must say so.
2. **The final position is one step from the readout** — a Localised verdict alone risks being
   near-tautological ("eight late heads can push a logit they sit next to"). This is exactly why
   Q2 (source C: an arbitrary matched-number vector must *not* work) and Q3 (the effect must
   arrive from the subject position) are separate adjudicated axes, and why the writeup may not
   present Q1 alone as a mechanism.
3. **Genuinely diffuse** — a verdict with its own pre-declared threshold and its own reportable
   sentence, not a failure.
4. **Span verdict ambiguity** — `R_span` high but inside the matched band: reported as
   Not-distinguishable-from-matched-chance, a negative on Q4, however large the fraction.
5. **Saturation** — if the intervention is out of the linear regime, the α-scaling arm catches it
   and the pre-declared wording downgrade applies.
6. **Stimulus-family narrowness** — one template family; the writeup's scope section repeats
   experiment 04's boundary verbatim.

## Pre-declared boundary language

- If the best set's `E(S_n)/E_all` lands in `[0.40, 0.60)` but the `≥ 6/8` joint rule fails:
  "the top heads recover close to half the attention-mediated effect, and the pre-declared bar was
  not met; the verdict is Diffuse, and the curve is shown." No re-reading of the rule.
- If Q1 is Localised and Q2 fails: the Not-number-selective sentence above, verbatim, adjacent to
  the Localised statement — never a Localised headline with a specificity footnote.
- If `ρ_full^C` itself exceeds `0.20` at calibration: the bound `θ_spec^C = 2·ρ_full^C` is used as
  frozen, and the writeup must state that the reference intervention itself moves the readout for
  number-matched sources by that much, as context for how hard the specificity test is.

## What would make this worth having done

If Q1–Q3 come back Localised + Number-specific + Subject-transport: the first adjudicated
real-model mechanism claim in this repository, with named heads, a measured specificity floor, and
an honest composition report. Any other combination: an adjudicated, affirmatively-worded finding
about where that chain breaks — each of which contradicts or refines the folk picture imported
from IOI-style circuits. Q4 turns experiment 04's uncertified concentration measurement into an
adjudicated statement about whether those twelve directions carry a causally transported signal,
on seeds disjoint from the ones that selected them — the question experiment 04's Next section
said needed a pre-registered design rather than a post-hoc hunt.

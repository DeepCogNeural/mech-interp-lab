# Experiment 05 — Which heads move the number signal, and are the twelve latents on the path?

> **STATUS: BASE DESIGN FROZEN 2026-08-02; calibration and Stage 1 complete; Stages 2–3 not
> started; one pre-freeze correction and three dated amendments.**
>
> **Read the end of this file before the frozen text above it.** The frozen text is preserved
> unedited on purpose, and parts of it are now known to be wrong:
>
> - [Pre-freeze correction 1](#pre-freeze-correction-1--2026-08-02) — what the calibration exposed
>   at constant fill-in. Two arithmetic/prose errors of mine and three omitted runtime cost terms.
>   No decision rule and no threshold changed.
> - [Amendment 1](#amendment-1--2026-08-02--the-runtime-cap-and-why-the-permitted-trim-was-declined)
>   — the runtime cap moves from 120 to 180 CPU-minutes after Stage 1 measured the dominant head-sweep
>   costs, and the permitted trim of the data is declined. Its `≈137`-minute total is a planning proxy,
>   not a complete measured cost. It also records the exact-arithmetic relationship between `E_all`
>   and `d_gap`; Amendment 3 corrects its false bitwise claim.
> - [Amendment 2](#amendment-2--2026-08-02--corrections-and-rule-repairs-forced-by-adversarial-review)
>   — **the important one.** Adversarial review found three factual errors and five rules that are
>   not executable as written. In particular: **§ Wrong-source constructions states source C's
>   direction backwards** (the measured `−0.60` reinforces the shared number; it does not pull
>   toward the wrong verb), **§ Order of operations claims no span-level measurement has ever been
>   run here, which is false** (experiment 04's PCA arm ran exact span projections), the source B
>   non-result claims more than its statistic can carry, and Q3 contained the certification bar this
>   design opens by saying it retires. Its claim that every repair makes a positive verdict harder
>   is withdrawn in Amendment 3: the old and new Q3 rules are not ordered that way.
> - [Amendment 3](#amendment-3--2026-08-08--pre-stage-2-executability-and-claim-boundaries)
>   — **read this before implementing Stages 2–3.** It supplies the missing selection-only source-A
>   sweep, fixes the candidate, bootstrap, Q3 and Q4 conventions that two competent implementations
>   could otherwise resolve differently, corrects the runtime and bitwise claims, and narrows all
>   four public verdict labels to what their rules actually establish.
> - The prior-work section cites `notes/lit-check-…`, which is gitignored. That record is now
>   [`LITERATURE.md`](LITERATURE.md) in this directory.

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
The frozen text records their values here, measured on calibration seed `20260899` — deliberately
not one of the adjudication seeds, so no constant is circular with a seed it later governs
(`calibration_results.json`, `CALIBRATION_NOTES.md`, produced by `calibrate.py`):

> `ρ_full^A = ` **0.00174**, bootstrap 95% CI `[0.00019, 0.01527]` → **`θ_spec^A = 0.200`**
> `ρ_full^C = ` **0.13492**, bootstrap 95% CI `[0.10909, 0.16230]` → **`θ_spec^C = 0.26983`**
> (`ρ_full^B = 0.00158`, recorded, never adjudicating.)

Both the bias-type ratio above and the noise-type companion `mean|Δd|` version are recorded; the
bias-type adjudicates. The specificity bound for Q2 is
`θ_spec = max(0.20, 2·ρ_full^C)` for source C and `max(0.20, 2·ρ_full^A)` for source A.

**What these numbers mean, and the disclosure they trigger.** Source A barely moves the readout at
all: swapping in a different noun of the same number shifts the mean readout by `−0.0078` against
the true flip's `4.46`, so the `0.20` floor binds and `θ_spec^A` is the floor. Source C is a real
control: the cross-template number-matched source moves the mean readout by `−0.60`, `13.5%` of the
true flip's magnitude and *in the opposite direction* — a systematic pull toward the wrong verb, not
noise around zero. So for source C the measured term binds, not the floor, and Q2's bound is
`0.26983`.

The `0.20` floor is tied to the `0.50` recovery bar so the two are not unrelated numbers: a set
clearing both has a number-specific component of at least `(1 − θ_spec^C) · 0.50 = 0.365` of
`E_all`, i.e. at least `73%` of the set's own effect is number-specific. (The draft asserted
`0.30·E_all` here; that figure was arithmetically wrong even under the draft's own `0.20` — see
Pre-freeze correction 1. The bound is corrected, not relaxed.)

Source B is the interesting recorded non-result: flipping the attractor's number moves the mean
readout by `+0.0071` — essentially nothing — while its per-item `ρ_noise` is `0.271`, the largest of
the three. Individual items move; the movement does not point in a consistent number direction.
That is recorded here before Stage 1 exists and is reported descriptively whatever the head-level
result turns out to be.

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

**Measured at calibration, and it is a lower bound — read that word.** The pilot timed each patch
family separately from the per-seed fixed overhead, because experiment 04's pilot under-projected
its main run by 25× by scaling per-patch cost alone when 71% of the eventual wall clock was
overhead that does not scale with patches. Enumerating every patch count the design implies gives a
conservative no-reuse total of **34.7 CPU-minutes** against the 120-minute cap, and **15.9 minutes**
at the permitted 100-retained-pair floor.

**That number does not certify the design meets the cap**, and the pilot said so rather than
reporting a total. Three terms are missing and cannot honestly be priced before the freeze:
attention-value caching for the `z`/`v` implementation (pricing it needs head-level access, which
the blinding restriction forbids); Stage 3's per-seed PCA fit; and Stage 3's latent-candidate pool
preparation. The first is the largest unknown. With roughly 3.5× headroom the cap is unlikely to
bind, but "unlikely" is the claim, not "verified" — and if the real total does exceed 120
CPU-minutes, the permitted trim is pair count per seed, to no fewer than 100 retained pairs, and
past that the design must be amended *before* the main run in a dated amendment stating what was
cut. The projection, its lower-bound status, and the eventual measured total all go in the writeup.

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
- If the measured term rather than the floor binds `θ_spec^C` — which happens whenever
  `ρ_full^C > 0.10`, and which is what occurred (`ρ_full^C = 0.13492`, `θ_spec^C = 0.26983`) — the
  writeup must state that the reference intervention itself moves the readout for number-matched
  sources by that much, as context for how hard the specificity test is. (The draft wrote the
  trigger as `ρ_full^C > 0.20`, which does not match its own `max(0.20, 2ρ)` formula: that formula
  crosses over at `ρ = 0.10`. The disclosure is applied under the stricter reading — see Pre-freeze
  correction 1.)

## What would make this worth having done

If Q1–Q3 come back Localised + Number-specific + Subject-transport: the first adjudicated
real-model mechanism claim in this repository, with named heads, a measured specificity floor, and
an honest composition report. Any other combination: an adjudicated, affirmatively-worded finding
about where that chain breaks — each of which contradicts or refines the folk picture imported
from IOI-style circuits. Q4 turns experiment 04's uncertified concentration measurement into an
adjudicated statement about whether those twelve directions carry a causally transported signal,
on seeds disjoint from the ones that selected them — the question experiment 04's Next section
said needed a pre-registered design rather than a post-hoc hunt.

---

## Pre-freeze correction 1 — 2026-08-02

The calibration pilot ran against the draft above and exposed five defects in it. All five are
recorded here rather than silently edited, because the value of a pre-registration is that its
history is visible. **No decision rule changed and no threshold moved.** Three defects were found by
the implementation refusing to paper over them; two were arithmetic and prose errors of mine that
the measured constants made visible.

**1. The `θ_spec^C` disclosure trigger was written at the wrong number.** The draft's boundary
language said the disclosure fires "if `ρ_full^C` itself exceeds `0.20`", but the frozen formula is
`max(0.20, 2·ρ_full^C)`, whose crossover is at `ρ = 0.10`. The measured `ρ_full^C = 0.13492` falls
in the gap: the formula's measured term binds, so the situation the disclosure exists for is exactly
the one that occurred, while the prose's literal trigger did not fire. **Resolved by applying the
disclosure** — the stricter reading of the clause's own intent — and rewriting the trigger to match
the formula. The formula itself is untouched and `θ_spec^C = 0.26983` is what it returns.

**2. The `0.30·E_all` figure was arithmetically wrong.** The draft justified tying the `0.20` floor
to the `0.50` recovery bar by claiming a set clearing both has a number-specific component of at
least `0.30·E_all`. Under the draft's own `θ = 0.20` the correct figure is `0.80 × 0.50 = 0.40`;
under the measured `θ_spec^C = 0.26983` it is `0.365`. The draft understated its own bound. Both the
statement and the reasoning are corrected in place, and the corrected bound is *stronger* than the
one claimed, not weaker.

**3. The runtime budget omitted three cost terms, and the projection is a lower bound.** Q2's frozen
rule requires joint-set patches for sources A and C (and B for reporting) per seed, which the
draft's runtime prose did not enumerate; Stage 3 requires a per-seed PCA fit and a latent-candidate
pool preparation, which had no term. Two further terms — attention-value caching above all — cannot
be priced before the freeze without the head-level access the blinding restriction forbids. The
runtime section now reports `34.7` CPU-minutes as an explicit **lower bound** with its omissions
named, instead of a total. This is a weakening of a claim the draft made, and it is the reason the
pilot's status field reads `completed_with_lower_bound_runtime_projection` rather than `completed`.

**4. Source C's template family is now fixed and named.** `source_C_relative_clause_with_adverb`
(`The {SUBJ} that the {ATTRACTOR} often {RELVERB}`), retained 240/240 under Gate A with median
`d_gap` 6.448. It was the first family tried and it passed on the first attempt; the notes record
that, so no reader has to wonder whether a search was hidden. Its selection criterion was Gate A
alone, applied before any residual effect was inspected. Cross-template position indexing was
verified rather than assumed: sequence length and final-position index differed on 468 of 468
directed edits, confirming that source positions were never reused as base write indices.

**5. Blinding, restated precisely at the freeze.** The pilot timed forward passes with **no-op hooks
attached at `hook_z` and `hook_v`** — the hooks return the incoming activation unchanged, and no
attention value, head effect, or head ranking was copied, retained, projected, compared, or emitted.
This was specified in advance as the way to price head-level cost without acquiring head-level
information. The claim in § Order of operations therefore still holds literally at this commit: **no
head-level or span-level measurement has ever been run in this repository.** What was measured here
is wall-clock time, plus the residual-stream constants above.

Also recorded at freeze, as a fact about the stimuli rather than about heads: source B (attractor
flip) moves the mean readout by `+0.0071` — indistinguishable from nothing — while carrying the
largest per-item `ρ_noise` of the three sources at `0.271`. Individual items move and the movement
does not point in a consistent number direction. This is descriptive, it is pre-registered as
descriptive, and it will be reported whatever Stage 1 finds.

---

## Amendment 1 — 2026-08-02 — the runtime cap, and why the permitted trim was declined

**Written after Stage 1 was run and read, before Stage 2 — the main run — has started.** Stage 1
adjudicates nothing and its output feeds a mechanical selection rule, but it has been read, and this
amendment is dated accordingly rather than presented as blind.

**What changes: one number, the runtime budget. `120` CPU-minutes becomes `180`.** Nothing else in
this document changes — not a threshold, not a decision rule, not a seed set, not a pair count, not
an axis.

**Why.** Stage 1 measured what calibration could only project, and the projection was low. The
calibration priced its enumerated Stage-1 patches at `100.84 s`; the same patches actually cost
`413.76 s`, a factor of `4.1`. The cause is now understood: the calibration timed one forward per
logical patch, while the implementation splits `472` directed edits into `15` deterministic
microbatches, so a logical patch is fifteen forward calls, not one. Rebuilding the projection on
measured costs — a 144-head sweep at `358.9 s`, a joint patch at `2.3 s`, per-seed fixed overhead at
`13.0 s` — gives **`≈ 137` CPU-minutes** for the whole experiment, of which Stage 2's two 144-head
sweeps across eight seeds are `≈ 106`. That is over the frozen cap.

**Why not the permitted trim.** This document permits cutting the pair count per seed to no fewer
than 100 retained pairs, which would bring the total to `≈ 76` minutes and require no amendment at
all. **It is declined.** The trim would shrink the data underlying the adjudication *after* Stage 1's
ranking had been read, and pair count is not an inert knob here: it enters the frozen
distinguishability rule directly through the width of a bootstrap interval and through the noise
floor estimated from 144 per-head means. Cutting it would trade a scientific quantity for a
laptop-convenience one, chosen with partial knowledge of the answer. The cap, by contrast, was a
planning number I picked before any cost had been measured; it protects no inference. Raising it
changes no measured quantity and no rule. Between weakening the evidence and admitting that my
budget estimate was wrong, the second is the honest move and the cheap one.

The permitted trim remains available and unused. If it is ever exercised, that will be its own dated
amendment stating what was cut and when the decision was taken.

### Recorded observation, no rule change: `E_all` and `d_gap` coincide by construction

Stage 1 measured `E_all = 5.180971145629883` and the clean `d_gap` mean
`= 5.180971145629883` — equal in every bit. This is an identity, not a coincidence and not an
implementation artifact, and it is worth stating because § Notation presents `E_all`, `E_ref`, and
`d_gap` as three distinct denominators when two of them are the same number for this stimulus family.

The reason: a single-flip pair changes only the subject noun, so base and source share identical
tokens at every other position, including the final one. In the source's own forward pass the
final-position residual is therefore built entirely from its own attention outputs plus MLPs, and an
MLP at the final position is a function of the final-position residual alone. Replacing every head's
final-position output with the source's makes the base run's final-position residual track the
source's exactly at every layer, so the logits match exactly.

Two consequences, both favourable and neither requiring a rule change. The adjudication denominator
cannot be pathologically small, unstable across seeds, or sign-ambiguous — it is the full behavioural
gap. And Q1's `E(S_n)/E_all ≥ 0.50` bar reads, in plain terms, as *"eight or fewer heads carry at
least half the entire behavioural effect"*, which is what it was intended to mean. The writeup must
state this identity rather than presenting `E_all` as an independently measured ceiling.

---

## Amendment 2 — 2026-08-02 — corrections and rule repairs forced by adversarial review

**Written after Stage 1 was run and read, before Stage 2 — the main run — has started.** A
multi-agent adversarial review of the freeze commit returned twenty findings that survived an
independent attempt to refute each one. Three were factual errors about measured constants or about
this repository's own history; five were decision rules that are not executable as written; the rest
were accounting and wording. All are handled here rather than by editing the frozen text.

**The check a reader needs first: every rule change below makes a positive verdict harder to reach
or leaves the difficulty unchanged. None makes one easier.** What was known when they were written:
Stage 1's 144-head ranking, its top heads, `E_all`, `E_ref`, and the calibration constants. No Stage
2 or Stage 3 quantity exists.

### A. Factual — source C's direction was stated backwards

§ Wrong-source constructions says the cross-template source shows "a systematic pull toward the
wrong verb". **That is wrong, and it inverts a measured constant.** The sign convention aligns every
source to the axis on which a *true* flip is positive, and sources A, B and C are all built with the
source's subject number equal to the base's. A negative aligned mean therefore means the intervention
pushed the readout *toward the number both sentences share* — reinforcing the verb that agrees with
the base's own subject, not the disagreeing one.

The corrected reading of `E(Δd_C) = −0.6018` against the true flip's `+4.4604`: **a number-matched
sentence from a different syntactic frame pushes the readout 13.5% of a full flip further in the
direction it was already going.** It is same-number reinforcement, not a pull toward error.

Nothing downstream changes. Q2's rule compares magnitudes, `|E(Δd_C(S*))| ≤ θ_spec^C ·
|E(Δd_right(S*))|`, so `θ_spec^C = 0.26983` stands and the test is unaffected. What changes is what
the writeup is allowed to say the number means. The same inversion appears in this directory's
`README.md` and in `lab-notebook.md`; both are corrected there directly, since neither is frozen.

### B. Factual — the blinding claim was too broad on "span-level"

§ Order of operations claims "**no head-level or span-level measurement has ever been run in this
repository**". The head-level half is true. The span-level half is false as a universal statement:
experiment 04's adjudicated PCA arm edits the top `k` coordinates of a complete orthonormal basis,
which *is* an exact orthogonal projection onto a `k`-dimensional span, and its recovery curves are
published.

**Narrowed to what is true and still load-bearing:** no measurement of any attention head's causal
effect, and no measurement of the twelve latents' span, had been run in this repository before the
freeze commit. That is the claim this design needs, and it is the only one it may make. The broader
sentence is withdrawn. `lab-notebook.md` propagated the broad version and is corrected there.

This is the fourth time an over-broad claim in this repository has had to be narrowed after the fact,
and the pattern is the same each time: the *evidence* was fine and the *sentence* reached past it.

### C. Factual — the source B non-result claims more than the statistic can carry

§ Wrong-source constructions and Pre-freeze correction 1 both say of the attractor flip that
"individual items move and the movement does not point in a consistent number direction". **The
second clause is not measured.** Every source is aligned to the *subject*-flip axis, and an effect
locked to the attractor rather than the subject yields a mean near zero on that axis by construction,
however perfectly consistent it is in its own direction. The observed signature cannot distinguish
"inconsistent" from "consistent about something this axis does not resolve".

**Corrected statement, frozen here:** on the subject-flip alignment the attractor flip moves the mean
readout by `+0.0071`, which that alignment forces toward zero for any attractor-locked effect, while
per-item movement is the largest of the three sources (`ρ_noise = 0.271`). Whether that movement
points consistently toward the flipped attractor's number **was not measured and is not claimed.**

### D. Rule repair — Q3 had the banned certification bar in it

Q3 read: Subject-transport if the joint path-patched effect reaches `≥ 0.50` of the same set's direct
`z@final` effect. That is a fixed bar on a continuous ratio with no null and no boundary language —
**the exact construction this design opens by saying it exists to retire.** It survived my own
review, the advisor's, and the freeze.

**Replaced, to the shape Q4 already uses.** Per seed, the two-step path patch is *also* run driven by
**source A**, which carries no number flip, giving a measured transport noise floor `F_path`.

- **Subject-transport**: in `≥ 6 of 8` seeds, the source-driven path-patched joint effect exceeds
  that seed's `F_path`, with a bootstrap 95% interval on the difference excluding zero.
- **Final-position-local**: otherwise, with the frozen scope sentence already in § Stage 2 (a head
  routing number through its QK pattern is invisible to a v-patch) unchanged.
- The **fraction** of the direct effect that the path patch recovers is reported as a graded effect
  size with its `t(7)` interval, and **has no threshold**. Pre-declared wording for the middle case:
  *"the set's effect is partly carried by what these heads read at the subject position, and the
  recovered fraction is `X`; this is a graded statement, not a certification."*

Adding a null makes Subject-transport strictly harder than the `0.50` bar would have been at the
Stage-1 magnitudes now known.

### E. Rule repair — the selection seed was also adjudicating

Stage 2's candidate pool `C` is fixed from seed `20260801`, and `20260801` was also one of the eight
Q1–Q3 adjudication seeds. On that seed Q1's condition (b) — every member individually
distinguishable — is true by construction, since distinguishability there is what put the head in `C`;
and condition (a) carries a winner's-curse upward bias. A `≥ 6/8` rule with one guaranteed seed is a
`≥ 5/7` rule wearing a different number.

**Repaired: `20260801` is removed from the Q1–Q3 adjudication set and `20260809` replaces it.** The
adjudication seeds become `20260802 … 20260809`, eight seeds, none of which selected the candidate
pool. Seed `20260801` keeps its Stage 1 role and its results are reported, but it adjudicates
nothing. The cost is that one fewer adjudication seed overlaps experiment 04's Gate B seeds; four
still do.

### F. Rule repair — Stage 3's band edge was undefined at 20 draws

The design sets the matched-chance band edge at "the 99th percentile" while running 100 draws on one
seed and 20 on the rest. At 20 draws that percentile is the sample maximum, it estimates a different
tail than at 100, its variance is large, and the text never says which seed's edge adjudicates.

**Repaired: 100 draws in every Stage-3 seed, and the edge is the second-largest of that seed's 100
draws** — an explicit order statistic, identical in construction across seeds, each seed adjudicated
against its own. Runtime cost is about seven CPU-minutes, which Amendment 1's cap absorbs.

### G. Rule repair — the distinguishability test was not executable as written

"The two-sided 95% bootstrap interval of its mean `Δd/E_all` … Holm-corrected across the 144 heads"
names an interval and a step-down correction without a family error rate, a p-value construction, or
a resample count adequate to resolve Holm's strictest level (`0.05/144 = 3.5e-4`, which 10,000
resamples resolve only to the last digit).

**Frozen specification:** family error rate `0.05`; per-head two-sided bootstrap p-value from the
pair-resample distribution of the head's **raw mean `Δd`** (not the ratio — dividing by `E_all` does
not change whether a mean differs from zero, and it removes the question of whether the denominator
is resampled); Holm step-down across all 144 heads; **100,000 resamples**, unit = retained minimal
pair with both directions kept together. `E(Δd_h)/E_all` continues to be reported as the effect size.

### H. Rule repair — the fallback tested set could be undefined

Q2 and Q3 test `S* = the claimed set if Localised, else S_8`, but nested sets only reach
`S_min(8,|C|)`. **Repaired: `S* = S_min(8,|C|)` when Q1 returns Diffuse.** If `|C| = 0`, the frozen
wording is: *"no candidate head was individually distinguishable in the selection seed, so no set
exists to test; Q1 returns Diffuse and Q2 and Q3 return their pre-declared negatives with the
144-head curve shown."*

### I. Accounting — Pre-freeze correction 1 miscounts its own contents

Three defects in the correction record itself, which is the one document where that matters most:

1. It says the calibration "exposed five defects" and splits them "three found by the implementation
   … two … of mine", but its five numbered items do not partition that way. The implementation's own
   `design_defects` list has exactly three entries, **all runtime**, and all of them sit inside item 3.
   Items 4 and 5 are not defects: item 4 records a calibration fill-in the design had already
   planned, and item 5 states the blinding claim *held*. The honest count is **two errors of mine
   plus three runtime omissions**, with items 4 and 5 relabelled as records, not defects. The freeze
   commit message repeats the miscount.
2. Item 3 says "Two further terms — attention-value caching above all — cannot be priced", but names
   only one, and attaches the head-level restriction to terms blocked by latent/span access instead.
   Corrected: **one** further term (attention-value caching) needs head-level access; the PCA fit and
   the candidate-pool preparation need latent/span access.
3. Item 5 says the no-op-hook timing method "was specified in advance", and item 4 says of the
   source-C search that "no reader has to wonder whether a search was hidden". Neither is provable
   from the record — the method first appears in the freeze commit alongside its implementation, and
   a commit can show one recorded attempt but not the absence of unrecorded ones. Both are withdrawn,
   for exactly the reason § Order of operations already gives: the record establishes commit order
   and commit content, nothing about what happened between commits.

### J. Housekeeping — the literature record is now in the repository

§ Prior-work anchor cites `notes/lit-check-number-agreement-2026-08-02.md`, and `notes/` is
gitignored, so a frozen portfolio document pointed at a file no reader could open. The note is moved
to `experiments/05_number_agreement_circuit/LITERATURE.md` and committed. A document cited by a
pre-registration is part of the portfolio.

---

## Amendment 3 — 2026-08-08 — pre-Stage-2 executability and claim boundaries

**Written after Stage 1 was run and read, before either Stage 2 or Stage 3 has run.** This amendment
therefore has no access to an adjudicating Q1–Q4 result. It does have access to the Stage-1 head
ranking and per-edit records, including the fact that one unresolved convention changes at least one
top-10 candidate's eligibility. That fact is disclosed below rather than hidden behind the word
"clarification."

This amendment has two jobs. First, it fixes choices that the frozen text and Amendment 2 left
underspecified, such that two competent implementations could return different verdicts. Second, it
withdraws language that reaches past the rules. Except where a missing estimator is explicitly
defined, the numerical thresholds, seed sets, retained-pair floor and `≥6/8` aggregation rules remain
unchanged. None of the new statistical conventions is claimed to be uniformly stricter than every
reasonable alternative; they are frozen for reproducibility.

### A. Amendment 2's monotonicity claim is withdrawn

Amendment 2 says every one of its rule repairs makes a positive verdict harder or leaves the
difficulty unchanged. That is false for Q3. For example, a true-source path effect of `0.30`, a
source-A path effect of `0.05`, and a bootstrap interval on their difference of `[0.20, 0.30]` fail
the old `0.50`-of-direct-effect rule and pass the replacement control comparison. Conversely, a large
but noisy effect can pass the old ratio and fail the new interval rule. The rules are not ordered.

The reason for replacing the old rule remains: a fixed recovery ratio without a control distribution
was the certification shape this experiment set out to avoid. That is a design rationale, not a claim
that the replacement is always harder.

### B. Stage 1 is complete only for its shipped scope; candidate selection is not

The shipped Stage-1 artifact contains the true-source 144-head `z@final` sweep, the joint `E_all`,
`E_ref`, and the top-24 `v@subject` measurements. It contains no source-A head sweep and no Stage-2 or
Stage-3 quantity. Because candidate pool `C` requires the selection seed's source-A noise floor, `C`
cannot yet be constructed under the frozen rule.

Before any Stage-2 run, seed `20260801` must receive one **selection-only full 144-head source-A
`z@final` sweep**, using the same retained base minimal pairs, directional alignment, positions,
microbatching and per-edit manifest fields as the true-source sweep. It adjudicates nothing and is
reported unconditionally. Its only role is to complete the already-specified filter for `C`. Dropping
that filter after seeing the true-source ranking is not permitted.

This supplement is new cost. The existing `≈137` and `≈76` CPU-minute totals did not price it, and
Amendment 2's increase from 240 to 800 Stage-3 matched-span draws was not priced either. They must be
re-estimated before an experiment run; the 180-minute cap is a planning limit, not a certified total.

### C. Candidate-set and distinguishability conventions

The following conventions apply to the selection supplement and every Stage-2 seed.

1. **Ranking ties.** Signed true-source mean effects rank descending; exact ties break by `(layer,
   head)` ascending. Stage 1 had no exact effect ties, so this convention does not alter its shipped
   order.
2. **Sign consistency.** The unit is the retained minimal pair, not the directed edit. A pair counts
   as consistent only when **both** sign-aligned directed effects are strictly positive; zero counts
   as inconsistent. The score is
   `minimal_pair_both_directions_positive_fraction`, and `C` requires it to be `≥0.90`.
3. **Known consequence.** For Stage-1 rank 7, L5H2, the directed-edit fraction is `0.9237` while the
   adopted pair-level fraction is `0.8475`; it therefore fails this component of `C`. This choice was
   made after those records existed. It is adopted because it matches the resampling unit, not because
   its effect on the final verdict is known to be conservative.
4. **Source-A noise edge.** For the 144 absolute source-A head means, the "99th percentile" means
   `numpy.percentile(values, 99, method="linear")`. No other interpolation or order statistic may be
   substituted. The source-A edge and the Holm-adjusted true-source test are both required.

All bootstrap resampling introduced before Stage 2 uses retained minimal pairs, with the two directed
edits kept together. Per-head distinguishability keeps Amendment 2's `100,000` draws and family-wise
Holm step-down at `α=0.05`; other Stage-2 bootstrap intervals use `10,000` draws. Percentile intervals
use the `2.5` and `97.5` percentiles with `method="linear"`.

For a per-head raw-mean test with `B=100,000`, the two-sided bootstrap p-value is
`min(1, 2 × min((n_{μ_b≤0}+1)/(B+1), (n_{μ_b≥0}+1)/(B+1)))`. The 144-head family is the only Holm
family within each seed. Dividing by `E_all` remains an effect-size report and does not enter that
p-value.

Every resample uses `numpy.random.default_rng(experiment_seed × 1000 + test_id)`. The constant test-id
table is part of the implementation and the result manifest: `1…144` are flat head ids
`12×layer+head+1`; `301` is the Q3 true-minus-source-A interval; `401` is the Q4 matched-subset draw.
No iteration-order-derived seed is allowed.

### D. Q3's `F_path` and decision statistic

The phrase "transport noise floor `F_path`" in Amendment 2 did not define an estimator. It is now the
**signed source-A control mean**, not an absolute-value floor:

- for the same tested set `S*` and seed, run the same two-step path patch once with the true number-flip
  source and once with source A;
- retain per-edit sign-aligned raw logit-difference changes for both arms on their common retained
  minimal pairs;
- `F_path = mean(Δd_A^path)` in raw logit-difference units;
- `D_path = mean(Δd_right^path) − mean(Δd_A^path)`;
- resample common minimal pairs, retaining both directions and both arms together, for `10,000`
  draws under `test_id=301`.

A seed supports **Subject-value transport shown** exactly when `D_path > 0` and the percentile 95%
interval of `D_path` has lower bound `>0`. The Q3 positive verdict still requires this in `≥6/8`
adjudication seeds. The direct-effect recovery fraction is reported with its across-seed interval but
has no pass/fail threshold. A negative source-A mean can increase `D_path`; the signed control was
chosen because the estimand is directional, and is not described as uniformly more conservative than
an absolute control.

The negative verdict is **Subject-value transport not shown**, not Final-position-local. It means the
registered `v@subject` probe did not meet the control-comparison rule. It does not establish that the
effect is generated locally at the final position, and it remains blind to transport through a head's
QK pattern.

### E. Q4 projector, matched subsets and claim label

The twelve target decoder rows form a matrix `D ∈ R^{12×768}`. Construct the projector in float64
with a thin SVD. Let `V_r` be the retained right-singular rows under
`tol = max(D.shape) × eps_float64 × s_max`; then `P_V δ = V_r.T @ (V_r @ δ)`. Cast the projected delta
to the model activation dtype only at injection. The target must have numerical rank 12. If it does
not, Q4 is blocked and requires a dated design amendment rather than silently changing the matched
dimension.

For each Stage-3 seed, sort the recomputed 128-candidate latent ids, exclude every target latent id,
and draw 12 distinct ids uniformly without replacement within a subset. The 100 subsets are
independent draws and may overlap across draws. Use `default_rng(experiment_seed × 1000 + 401)` and
write every drawn id set to the manifest. A rank-deficient sampled span is rejected and redrawn; if
100 full-rank subsets cannot be obtained within 10,000 attempts, that seed is blocked rather than
reusing another seed's edge. Fewer than 12 eligible candidate ids is also a block.

Amendment 2's edge remains the second-largest of the 100 per-seed matched `R_span` values. The public
verdict labels are now **Above matched-span chance** and **Not above matched-span chance**. A positive
verdict means only that, at layer-8 and the adjudicating position set `both`, the twelve decoder-row
span's `R_span` exceeds this matched-dimension empirical edge in `≥6/8` seeds. It does not establish
necessity, mediation, an attention route, or that the native model uses these coordinates as its
mechanism. `subject` remains a non-adjudicating transport-relevant diagnostic.

### F. `E_all`, `d_gap`, runtime and provenance corrections

Amendment 1's statement that `E_all` and clean `d_gap` are "equal in every bit" is withdrawn. Their
reported Stage-1 means are both `5.180971145629883`, and the equality follows in exact arithmetic for
this single-flip construction, but the shipped float32 records differ on 2 of 472 directed edits with
maximum absolute difference `5.7220458984375e-06`. The licensed statement is **same mathematical
estimand and same reported mean up to numerical error**, not bitwise identity. `E_all` remains a
separately measured denominator.

The reproducible Stage-2 subtotal behind `≈106` minutes is
`(16 × 358.863859 + 224 × 2.258879 + 8 × 12.982374) / 60 = 105.861` minutes: sixteen 144-head sweeps,
224 joint-patch equivalents and eight fixed-overhead blocks. Amendment 1's prose calling that only
"two 144-head sweeps across eight seeds" was incomplete, though the subtotal was not. The `≈137`
whole-experiment and `≈76` trimmed numbers additionally contain estimates not fully enumerated in the
published runtime manifest and omit newly required work. They are planning proxies, not measured
totals and not proof that the 180-minute cap will hold.

The committed calibration record proves commit content and order: it records one source-C family that
passed Gate A and its measured constants. It cannot prove when residual effects were inspected or that
no unrecorded family was tried. Any stronger timing or exhaustive-search claim is withdrawn.

### G. Method framing and public verdict language

The frozen claim that adjudication rests only on nulls that shrink with data is withdrawn. Sampling
uncertainty in the bootstrap and source-A estimates can narrow with more retained pairs; Q2 instead
uses fixed calibration-derived specificity ratios, and Q4 compares against a finite matched-span
reference distribution. These are different forms of evidence and are reported as such.

The numerical rules are accompanied by these controlling public labels and negative boundaries:

- **Q1 — Qualifying ≤8-head set / No qualifying ≤8-head set.** A negative says that no tested set
  simultaneously met the half-`E_all` and all-member distinguishability conditions in the same
  `≥6/8` seeds. It does not license a global claim that the mechanism is diffuse.
- **Q2 — Number-specific under registered controls / Specificity bound not met.** A negative says at
  least one registered source-A/source-C bound failed the `≥6/8` rule. It does not license "the
  matched-number source was as strong as the true source."
- **Q3 — Subject-value transport shown / Subject-value transport not shown.** The negative boundary
  is the one in §D above; neither label speaks to QK-mediated routing.
- **Q4 — Above matched-span chance / Not above matched-span chance.** The exact scope is the one in
  §E above; neither label is "On-path."

These labels supersede the broader frozen labels and sentences wherever the experiment is summarized
or written up. The legacy terms remain visible in the preserved base text so the amendment history is
auditable.

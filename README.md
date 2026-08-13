# mech-interp-lab

Self-directed mechanistic-interpretability research, run from a computational-neuroscience starting
point (probabilistic inference, stochastic modeling, primate V1 electrophysiology). Runs are CPU-only.
Checked-in compact packets support static reaggregation; raw-dependent repackaging and full model
reruns require off-Git source artifacts and cached assets, so the repository alone is not end-to-end
reproducible.

The through-line is not a thesis but an **escalation of method**, and every step was forced by the
previous step's failure rather than planned in advance: *toy models where the ground truth is fully
observable → a real model read correlationally → the same question asked causally → the mechanism
itself.*

Five completed experiments, with experiment 05 now carrying this repository's first adjudicated
real-model mechanism result and a positive matched-span representation result, plus a result-blind
sixth-experiment protocol reviewed interactively by an AI advisor, implemented but unrun; its
final-SHA receipt is still pending. The first replicates the *storage* account of
superposition. The second asks whether the same geometry changes interaction readout under a fixed
estimator. Its shipped BCE-trained logistic probe scores the coordinate-wise control at
`0.494 ± 0.006` and the mixed codes near `0.81`. The mathematics proves only that an additive
coordinate-wise score cannot *perfectly separate* XOR; it does not impose a chance ceiling. Indeed,
`predict 1 iff ReLU(x_i) + ReLU(x_j) > 0` reaches `0.75` on the experiment's balanced four quadrants. The result is
therefore estimator-specific evidence that these mixed codes make XOR easier for this probe, not a
proof that mixing is necessary for above-chance interaction readout. The third takes that question to real SAE features
on GPT-2-small and **fails to settle it** — the answer moved by more than tenfold under a rescaling a
fitted probe is not invariant to, and that is reported as unsettled rather than dressed up. The fourth
removes the failed instrument entirely, intervening causally so that the rescaling cannot move the
measure at all — and then **declines to certify what it measured**, because a faithfulness threshold
frozen in an earlier commit than any of this experiment's output was missed by 0.006. Most of these
produced results I did not want; all of them are reported in full.

[Experiment 05](experiments/05_number_agreement_circuit/writeup.md) asks the mechanism question the
first four could not: which attention heads move the number signal, whether that effect is
number-specific, and whether it arrives from the subject position. Across eight registered seeds,
**L7H4 and L8H5 were the smallest tested set that consistently recovered more than half of the frozen
direct effect** (`0.521–0.546`). The pair passed both specificity controls and produced a positive
frozen-attention-pattern value-path transport effect in every seed. Independent pair-level
recomputation, artifact-hash review, and a separate interactive AI-advisor review (not external
expert validation) all accepted the result.

That is an intervention-specific result, not a claim that these heads are necessary, sufficient, the
complete circuit, or the model's unconstrained native path. The independent Stage-3/Q4 test is now
positive as well: across eight seeds, projecting the registered layer-8 `both` intervention delta into
the frozen 12-row SAE decoder span retained `R_span = 0.8935` (t(7) CI `[0.8905, 0.8966]`), exceeded
the frozen second-largest edge of 100 matched spans in every seed, and left `R_comp = 0.0848` in the
complementary subspace. The geometric squared-norm fraction was only about `0.525–0.544`, so this is a
directed logit-effect ratio, not 89% activation reconstruction. The generic-text PCA comparator was a
raw logit effect (`0.0274`), not 2.74% recovery. The [frozen design](experiments/05_number_agreement_circuit/DESIGN.md)
and [compact public evidence](experiments/05_number_agreement_circuit/results/RESULTS.md) keep these claim
levels separate.

The Q4 result does not establish natural or monosemantic latent activations, individual-latent
causality, necessity or sufficiency, a natural head→span path or mediation, a complete circuit, or
generalisation across models or tasks. A single high-information exploratory bridge has now run on
fresh held-out seeds. The fixed 12-row span recovered `0.6786` (t(7) CI `[0.6738, 0.6834]`) of the
directed L7H4→`resid_pre8` effect and beat the maximum of 100 target-excluded matched rank-12 spans
in all eight seeds; its complement retained `0.3053` (CI `[0.3013, 0.3094]`). With L8H5's complete
`hook_z` output at the final query position overwritten by the source-A baseline, the target remained
`0.6740` (CI `[0.6696, 0.6785]`). This clamp occurs after attention-pattern-weighted value aggregation;
it is not a value-only intervention. The result does not support dominant dependence on L8H5's tested
final-position head output, and it does not establish mediation, native latent semantics, necessity,
or a complete circuit. The original raw-dependent bridge packet reported clean source `0d7c4db`,
fresh seeds `20260814–20260821`, 150 pairs per seed, and 800 frozen matched draws. The current compact
reaggregation attributes those fields through a hash-pinned historical receipt but did not revalidate
the missing raw input. That raw result stays outside Git (SHA-256 prefix `9d8446…`); no bridge-specific
independent review receipt is checked into the public packet. The
[short research brief](experiments/05_number_agreement_circuit/RESEARCH_BRIEF.md) is the short public
entry point.

![Experiment 05 claim ladder: eight-head selection, eight-seed Q1–Q3 evidence, and the positive Stage-3/Q4 matched-span result](experiments/05_number_agreement_circuit/results/figure_exp05_main.png)

The bridge follow-up is shown in [`figure_bridge_rescue.svg`](experiments/05_number_agreement_circuit/results/figure_bridge_rescue.svg)
([PNG](experiments/05_number_agreement_circuit/results/figure_bridge_rescue.png)); its compact
seed-level and matched-span evidence is in [`bridge_result_summary.json`](experiments/05_number_agreement_circuit/results/bridge_result_summary.json).

**A note on the framing this repository started with.** Experiments 01 and 02 were built around
superposition versus mixed selectivity — whether many features sharing few non-orthogonal directions is
a problem to undo or a computation to explain. That question is background here, not the organising
story, and the record is why: by experiment 03 the quantity actually being measured had become *SAE
basis quality*, a different and far more crowded question. Keeping the original banner would have been
framing rather than navigation.

---

## Experiment 04 — the causal version, and a gate that refused to certify a result I liked

[Full writeup](experiments/04_causal_feature_interchange/writeup.md) · [pre-registration, plus three dated amendments and one correction — the last amendment written after unblinding and non-adjudicating, the correction retracting three claims in the frozen text](experiments/04_causal_feature_interchange/DESIGN.md) · [code](experiments/04_causal_feature_interchange/run_experiment.py) · [raw results](experiments/04_causal_feature_interchange/run_results.json)

Experiment 03 stalled because a linear probe's L2 penalty is not invariant to rescaling its inputs, so
the answer moved with a preprocessing choice. The fix is not a better probe. It is no probe: edit
coordinates of a code, write the edit back into GPT-2-small's residual stream, and let **the model's own
next-token logits** report the effect. Rescale the code by a positive diagonal and the decoder inversely
and the written edit is algebraically unchanged — the knob that made experiment 03 unanswerable is
exactly zero here. The implementation asserts one instance of that with `torch.equal`, bit-for-bit, in
the pilot and in the main run.

The task is subject–verb number agreement on single-flip minimal pairs. The control is **PCA** of the
same activations — equally unsupervised, same kind of data, without the sparse dictionary-learning
objective. It is not otherwise matched, and every difference favours the SAE: 768 components against
24,576 latents, and one 8,192-token fit per seed against a dictionary trained on a corpus larger by many
orders of magnitude.
(The originally planned control, a width/norm/L0-matched random expansion, failed its own faithfulness
gate; a diagnostic quadrupled its fitting budget and the number moved the wrong way, which is the
pre-declared trigger for calling the failure structural rather than a fitting artifact — a rule's label
for a two-point outcome, not a proof about where the curve asymptotes — and the control changed family. All of that is in the amendments, each committed before the commit containing the output it
governs — commit order, which is not the same as proof of read order, and the writeup says so.)

![Causal recovery against the number of edited coordinates, for SAE, PCA, neuron and random-expansion bases](experiments/04_causal_feature_interchange/figures/01_recovery_curves.png)

| basis | AUC, within-basis | AUC, absolute | `k50` per seed |
|---|---:|---:|---|
| `sae` | **0.517 ± 0.002** | **0.359 ± 0.007** | 16, 16, 16, 16, 16 |
| `pca` | 0.213 ± 0.023 | 0.213 ± 0.023 | 64, 128, —, 128, 64 |
| `rand_exp` | 0.179 ± 0.024 | 0.117 ± 0.016 | never reached |
| `neuron` | 0.157 ± 0.005 | 0.157 ± 0.005 | never reached |

**The frozen rule returns `inconclusive`, so nothing in that table is a claimed result.** Gate C requires
each adjudicated basis to write back at least `0.70` of the residual-stream effect; the SAE's trained
decoder measured `0.694 ± 0.014`, passing in two seeds of five, and the design says a basis that fails
Gate C yields no headline. I did not move the threshold. Checking afterwards, moving it would not have
worked anyway — a second gate blocks independently in one seed.

What the table *is*: an uncertified measurement, reported in full because hiding it would be worse.
Sixteen SAE coordinates reach half of that basis's causal effect in every seed while PCA needs 64 to 128
and once never gets there; the paired gap is `+0.304 ± 0.023` within-basis and `+0.146 ± 0.022` absolute,
positive in all five seeds under both denominators.

The obvious deflations are tested rather than waved off, all post-hoc and recomputable from the manifest:
restricting to pairs whose subject noun never appears in the ranking-training split moves the number by
`+0.0054` (`0.5936` against `0.5882`, on 150 edits, with no equivalence test), and the five seeds' top-16
latents share a 12-latent intersection. The matched random expansion — same nominal 24,576-wide pool,
same pre-filter, same ranking rule — finishes *last*, which points away from a pure pool-width story
without settling it, since that arm failed Gate C too. The 32× width advantage stays a live alternative
reading. What keeps the whole thing honest-sized: a single **supervised** direction recovers `0.549` on
its own while the SAE's best single latent recovers `0.072`. Among the 128 candidates scored in each
unsupervised basis, none puts this factor in one coordinate.

### The by-product that became the next experiment

The design's causal-handle check swept layers × positions, and the picture is the cleanest thing in the
run. `E_resid/d_gap`, five seeds:

| layer | subject position | final position |
|---|---:|---:|
| 4 | 0.662 ± 0.012 | 0.011 ± 0.002 |
| 8 | 0.366 ± 0.015 | 0.423 ± 0.006 |
| 10 | 0.260 ± 0.014 | 0.694 ± 0.012 |

At layer 4 the causal handle for number sits almost entirely on the subject; by layer 10 it has largely
arrived at the readout position. **What is measured is where an interchange has an effect — no attention
pattern, head, or path was measured**, so this is consistent with the signal being relocated and is not
evidence about what relocates it. That gap is exactly what [experiment
05](experiments/05_number_agreement_circuit/DESIGN.md) is for.

### Scope — what this does not say

The model is never modified: the base residual is left exactly as GPT-2 produced it and one vector is
added to it, so no reconstruction is ever substituted for the model's own state. (The *difference* of
reconstruction errors does not cancel — an earlier draft said it did, and that was wrong; had it
cancelled, Gate C's ratio would be 1, and it is 0.694.) Low recovery in a basis means a factor is not concentrated
in few coordinates of that basis — never that an SAE harms a model's computation.

---

## Experiment 03 — a real SAE on the shattering × CCGP plane, and a result I had to give back

[Full writeup](experiments/03_ccgp_on_sae_features/writeup.md) · [code](experiments/03_ccgp_on_sae_features/ccgp_sae.py) · [per-seed results](experiments/03_ccgp_on_sae_features/results.json)

Experiment 02's caveats named this one twice: XOR accuracy is a task-specific proxy where shattering
dimensionality / CCGP (Bernardi et al. 2020) is the task-agnostic measure, and the whole thing was a toy
model. So this one is measured on a real model, with SAE weights I did not train: GPT-2-small layer 8 and
the published res-jb SAE from `jbloom/GPT2-Small-SAEs-Reformatted`, loaded straight from safetensors. Then
a full factorial NUMBER × TENSE × POLARITY over 96 lexical items, read at a sentence-final `.` that is
byte-identical across all eight conditions. The comparison that matters is
not SAE versus residual stream — a 768 → 24,576 ReLU expansion wins that by Cover's theorem — but **SAE
versus a random expansion matched in width, column norm, and L0.**

![Shattering dimensionality against main-effect CCGP for seven arms, full scale with chance lines and a zoomed panel](experiments/03_ccgp_on_sae_features/figures/01_shattering_vs_ccgp.png)

Under one probe convention the SAE reads two-way interactions much better than matched random mixing
(`+0.121 ± 0.015`); under another it does not (`+0.011 ± 0.022`). Those two conventions are an **invertible
affine reparameterisation** of a linear probe's input, so the swing measures the probe's inductive bias,
not the codes. **I report that as not adjudicated** rather than shipping the flattering setting — and the
width-matched follow-ups that survived (`+0.081 ± 0.008`, `+0.075 ± 0.016`) inherit the same question,
because they only ever ran under the convention that produces an effect.

Per arm, in overall shattering dimensionality (five seeds, 95% Student-t):

| arm | per-feature z-score | global RMS |
|---|---:|---:|
| `sae` (sparse; ~790 latents ever fire here) | 0.718 ± 0.020 | 0.888 ± 0.006 |
| `rand_exp` (matched random; ~500) | 0.731 ± 0.014 | 0.809 ± 0.008 |
| `resid` (768, dense) | 0.901 ± 0.009 | 0.902 ± 0.009 |
| `sae_recon` (768, dense) | 0.877 ± 0.010 | 0.877 ± 0.009 |

What does survive is a methodological result, and it is the part I'd defend: the sensitivity is
**localised**. The two dense 768-dimensional arms are unmoved by the scaling (0.902 vs 0.901; 0.877 vs
0.877) while every sparse or very wide arm swings hard — so preprocessing that is harmless for dense
representations is decisive for sparse over-complete ones, and any SAE-versus-baseline decoding comparison
that doesn't state its scaling convention is under-determined.

This is a fallback report, and the writeup says so in full. The run that would settle the comparison in
full — every headline arm and both metrics fitted to a stated convergence criterion, with L2 selected per
arm *and* per scaling — did not complete in time for the shipped manifest, so what ships is the earlier
completed run. The question counts as adjudicated only when both scaling estimates are individually
precise *and* agree, not merely when their intervals overlap.

### Final state — the convergence test ran, and the answer is still no

A follow-up [convergence test](experiments/03_ccgp_on_sae_features/convergence_test.py) did run that
criterion for shattering dimensionality on the four shared arms: full-batch L-BFGS with strong-Wolfe line
search, stopped on a stated relative-objective criterion rather than a step count, with L2 selected
item-disjointly per arm, per scaling, per seed, and per fold
([raw rows](experiments/03_ccgp_on_sae_features/convergence_results.json); 145 s of CPU). It closed about
half the discrepancy and did not close it — `+0.058 ± 0.041` under per-feature z-scoring against
`+0.115 ± 0.019` under global RMS. The predeclared rule asked for both estimates to be individually
precise *and* close; neither held, and a small overlap between a precise interval and an uninformative one
is not agreement. The dense-arm diagnostic passed (largest shift between scalings 0.0058), so the solver
is sound and what remains is the L2 prior's geometry, not a convergence artifact.

So experiment 03 stands as **an honest non-result with a defensible methodological finding beside it.**
Whether a real SAE code reads two-way interactions better than matched random mixing is a question this
probe family cannot answer, because the answer moves with a preprocessing choice that a linear probe
should be indifferent to. The lesson carried into the next design is that looking for the "fair" scaling
inside this family is most likely looking for a point that does not exist — the way forward is a question
setting where regularisation geometry is the object of study or is absent from the estimator, not one more
attempt to neutralise it.

### Scope — what this does not say

An SAE is a read-out lens beside the residual stream. Nothing here substitutes SAE features into GPT-2's
forward pass, ablates them, or measures model behaviour — so none of it says, or implies, that an SAE
harms a model's computation. These are properties of a code under a probe.

---

## Experiment 02 — mixed codes make XOR easier for the shipped probe; coordinate-wise scores are nonseparable

[Full writeup](experiments/02_superposition_and_readout/writeup.md) · [code](experiments/02_superposition_and_readout/readout.py) · [per-seed results](experiments/02_superposition_and_readout/results.json)

The SAE lineage in interpretability treats superposed coding as something to undo — disentangle the
mixed directions so features can be enumerated. The mixed-selectivity literature in systems
neuroscience (Rigotti et al. 2013) treats mixed selectivity as *functional*: it can expand the set of
nonlinear functions accessible to a linear readout. Both perspectives can be useful, and
interpretability is not blind to the second point — Elhage et al. study computation in superposition
directly. What a toy model adds is ground truth: I can build the geometries by hand and measure which
framing the readout actually rewards.

Three geometry arms at fixed `n = 20, m = 8`, differing **only** in the encoder `W` — monosemantic (a
selection matrix), random (Gaussian, Frobenius-norm-matched), and superposition (the frozen `W` that
experiment 01's storage objective learns). Each is read by a **linear** logistic probe on
`r = ReLU(W x)`, asked for the XOR of a feature pair. 8 seeds, balanced accuracy, a fixed
class-balanced eval distribution shared across every arm and sparsity, within-seed paired statistics,
and two-sided Student-t intervals across independent seed-level values.

![The shipped BCE-trained logistic probe scores the coordinate-wise arm near 0.50 while random and learned mixed codes reach about 0.80 XOR accuracy](experiments/02_superposition_and_readout/figures/01_configured_probe_xor_accuracy.png)

| arm | XOR readout accuracy (across sparsity `S`) |
|---|---|
| monosemantic | **0.494 ± 0.006** at every sparsity under the shipped BCE-trained probe — an empirical estimator result, not a theorem-imposed ceiling |
| random | 0.79 → 0.81 |
| superposition | 0.78 → 0.81 (dip to 0.74 at `S = 0.7`; the eight seeds split bimodally there) |

The theorem is narrower. A coordinate-wise code followed by a linear readout has an additive score
`s(a,b) = c + u(a) + v(b)`. Its four XOR corners obey
`s(0,q) + s(p,0) = s(0,0) + s(p,q)` for any fixed `p,q > 0`, which makes perfect separation impossible: the two positive
corners cannot both lie above a threshold while both negative corners lie below it. But this is
**nonseparability, not a chance ceiling**. On the exact balanced distribution here, the linear rule
`predict 1 iff ReLU(x_i) + ReLU(x_j) > 0` gets `00`, `01`, and `10` right and only `11` wrong, for `0.75`
accuracy. Drawing every XOR pair from features the coordinate-wise arm represents removes missing
coverage as an explanation; it does not convert the configured probe's empirical `0.494` into a
universal theorem. The small average train-minus-test gap (`+0.002`) is a check on this fitted probe,
not a proof against every form of overfitting. Averaging the five completed-run sparsities within each
seed, the paired mixed-minus-coordinate-wise gaps are `+0.313 ± 0.021` for random mixing and
`+0.297 ± 0.020` for learned superposition (Student-t intervals over eight seed means).

The interpretability-relevant reading is deliberately limited: a fully coordinate-wise representation
cannot make XOR perfectly linearly separable, while the tested mixed representations make it much easier
for this configured logistic estimator. Mean per-feature decodability is `0.621` for the superposed code
versus `0.614` for the coordinate-wise one, so this toy run also does not show a clean
enumeration-versus-readout tradeoff. It does **not** establish that mixing is necessary for above-chance
accuracy or that every monosemantic representation carries the same downstream cost.

### The learned-versus-random comparison, stated plainly

Does the *storage-learned* geometry beat equal-norm random mixing? **No reliable advantage was detected
in this eight-seed sample.** For each seed, first averaging the five completed-run sparsities and then
forming a Student-t interval across the eight seed means gives superposition − random differences of
`−0.015 ± 0.032` at `distractor_p = 0`, `+0.004 ± 0.019` at `0.05`, and `+0.002 ± 0.011` at `0.10`.
All three intervals include zero. A four-seed pilot had hinted at a small positive effect at high
sparsity; the shipped eight-seed sample did not detect it reliably.

No equivalence margin was specified before this analysis, so these intervals do **not** establish equality, practical
equivalence, or a generic property of mixing. They support only the narrower sentence above. Random
mixing supplies a second mixed code that also reaches about `0.80`; experiment 01 supplies one
storage-trained instance whose ground truth is known.

### The probe design trap

My first instinct was to run a linear probe directly on experiment 01's encoder, `h = W x`. My original
reason for rejecting it was too strong. A linear readout of a linear projection remains additive and
cannot *perfectly separate* XOR, but it is not forced to chance: the `0.75` witness above already shows
that. What the design would miss is the geometry-dependent piecewise-linear feature map introduced by
the shared ReLU.

With `r = ReLU(W x)` fixed everywhere and only `W` varying, the experiment asks whether geometry changes
the configured probe's ability to fit XOR. The coordinate-wise control establishes the empirical
baseline for that estimator; the theorem establishes only that its additive score cannot be perfect.

### Scope — what this does not say

This is a claim about **representations**, not about SAEs as a tool. It does **not** say an SAE harms a
model's computation. An SAE is a readout lens alongside the residual stream: downstream components
still read the mixed residual stream, the SAE does not replace the model's representation with a
monosemantic basis, and this experiment does not measure that question at all. Also: a toy model, not a
transformer; XOR is one minimal interaction, not all computation; above-chance additive classification
remains possible; and this is compression
(`n → m`, `m < n`) whereas Rigotti's mixed selectivity is expansion — cousin geometries, opposite
dimension direction.

---

## Experiment 01 — the storage account, replicated

[Full writeup](experiments/01_toy_models_of_superposition/writeup.md) · [code](experiments/01_toy_models_of_superposition/toy_models.py)

A replication of Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html):
when features are sparse, a tied-weight ReLU autoencoder packs more features than it has dimensions,
storing them along non-orthogonal directions and tolerating interference between features that rarely
co-occur. Five features in two dimensions give
[the pentagon](experiments/01_toy_models_of_superposition/figures/01_feature_geometry_5x2.png) under
uniform importance and antipodal pairs under decaying importance — the geometry is set by the loss
landscape's symmetry, not by a fixed rule. Twenty features in five dimensions go from 7/20 represented
when dense to all 20/20 by `S ≈ 0.997`. (This is the one experiment that ships figures but no
machine-readable results file, so these two counts cannot be re-derived without rerunning it — a gap
noted here rather than papered over.)

![Number of represented features climbing from the five-dimensional orthogonal limit up to all 20 as feature sparsity increases](experiments/01_toy_models_of_superposition/figures/03_capacity_vs_sparsity.png)

The dimension count is a floor, not a ceiling. Per-feature dimensionality `Dᵢ` sums to an
effective-dimension upper bound that approaches `m` only when training uses the bottleneck fully:
superposition redistributes the available dimension budget rather than manufacturing new dimensions.
Single seed, and the integer "features represented" count jitters near the norm threshold — the
geometry and `Dᵢ` are the robust readouts, not the exact count.

This experiment supplies the object experiment 02 tests, and the bridge that motivates it: same
non-orthogonal mixing that systems neuroscience calls mixed selectivity, with the ground truth handed
to you.

---

## Next

- **[Experiment 05 — the number-agreement circuit](experiments/05_number_agreement_circuit/DESIGN.md)**,
  with independently recomputed Stage-2 and Stage-3/Q4 results plus one exploratory bridge. L7H4 and
  L8H5 form the minimum tested compact set, pass both number-specificity controls, and transport
  subject-number information under the registered frozen-pattern value-path intervention in all eight
  seeds. Q4's frozen 12-row layer-8 span retains `0.8935` of the directed logit effect, beats the
  frozen matched-span edge in every seed, and leaves `0.0848` in the complement. The fresh bridge then
  recovers `0.6786` of the L7H4→`resid_pre8` effect (final-position L8H5-`z` clamp arm `0.6740`) and beats all 100
  matched spans per seed. It is exploratory evidence for a reproducible causal subspace, not a
  dominant L8H5 mediation or complete-circuit claim.
- **[Experiment 06 — fixed-object cross-template bridge](experiments/06_cross_template_bridge/DESIGN.md)**
  has a result-blind protocol and implementation reviewed interactively by an AI advisor, but remains
  unrun and has no final-SHA receipt. It holds exact GPT-2/SAE revisions and file hashes, L7H4, the 12-row
  span, and the 100 matched spans fixed while moving to the relative-clause family already used in
  calibration/source-C. It first asks whether the L7H4 true-versus-fixed-source-A handle transfers;
  only then does it compare the target with the second-largest raw-effect edge among the 100 frozen Q4
  matched latent sets. It reuses Q4's sets and tail-order statistic, not Q4's normalized `R` estimand. The protocol
  requires all 8/8 seeds for a directional verdict, uses a globally fixed no-fixed-point source-A
  lemma map, and distinguishes Gate-A `NON_ESTIMABLE` from implementation/provenance `STOPPED`. This is
  a mechanism-held-out evaluation on a calibration-exposed template family, not an unseen-template
  result, and no Experiment 06 result is claimed. Because the Experiment 05 bridge used seed-drawn
  source-A nouns, Experiment 06 is a second prompt/control construction, not a one-factor
  template-only contrast.
- Re-register experiment 04's Gate C on a decoder each arm actually ships with, with the floor
  pre-declared from a pilot on a *different* stimulus family so the threshold cannot be tuned to this
  one. Necessary but **not** sufficient: a rerun still has to clear every other frozen gate, and Gate D
  blocked independently in one seed.
- Close the PCA fitting-budget objection with a fit an order of magnitude larger, or with real corpus
  text instead of model-generated text. If experiment 04's comparison is wrong, this is where.
- A second stimulus family, before anything in experiment 04 generalises beyond number agreement — and
  independent template families for experiment 03 on the same principle.
- Shattering dimensionality / CCGP on experiment 02's three *toy* geometries — the same metric where
  the ground truth is fully known, which would calibrate what these numbers mean.
- A mechanism check on any residual superposition-versus-random effect: correlate each pair's margin
  with the Gram interference among its active features.

## Reproduce

Python 3.11 (torch has no 3.14 wheels yet). CPU-only. Experiments 01–02 need no downloads; experiments
03–05 need GPT-2-small and one ~150 MB res-jb SAE. Experiment 03 pulls them from the HuggingFace hub
on first run; experiments 04 and 05 reuse that cache rather than fetching it themselves.

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
(cd experiments/01_toy_models_of_superposition && python toy_models.py)   # ~1-2 min
```

```bash
(cd experiments/02_superposition_and_readout && python readout.py)        # ~10-15 min; SMOKE=1 for a ~40s subset
(cd experiments/02_superposition_and_readout && REAGGREGATE_ONLY=1 python readout.py)  # reuse shipped raw rows; fit no model/probe
```

```bash
(cd experiments/03_ccgp_on_sae_features && python ccgp_sae.py)           # 19m51s measured on an M1 Pro CPU; SMOKE=1 gives a fast plumbing subset
```

```bash
(cd experiments/04_causal_feature_interchange && python pilot.py && python gate_c_diagnostic.py && python run_experiment.py && python robustness.py)
```

Experiment 04 is four stages in the order they were actually run — go/no-go pilot, the Gate C diagnostic
that redirected the control arm, the five-seed main run, and the post-unblinding robustness arms.
Measured on an M1 Pro CPU: 42.3 s + 647 s + 1,760 s + 1,344 s, **63.2 minutes total.**

Experiment 05's calibration, fresh same-snapshot selection, Stage 2, Stage-3/Q4, and one exploratory
bridge follow-up are complete. The compact claim inputs, per-seed statistics, checksums, and generation code live in
[`results/`](experiments/05_number_agreement_circuit/results/); the raw execution artifacts stay outside
Git and are identified by SHA-256. The original raw-dependent packets reported 2,528 logical
forward-equivalents and 6,272 seconds for Stage 2, and 353,120 logical forward-equivalents and about
1,266 seconds for Stage 3/Q4; they also reported an eight-seed bridge. The current checked-in compact
reaggregation did not revalidate those raw run-integrity or runtime fields. The public packet reports
the bounded Q4 result and the exploratory bridge with their non-claims; it does not copy the raw model
outputs.

```bash
./.venv/bin/python -m unittest discover \
  -s experiments/05_number_agreement_circuit/tests -p 'test_*.py'
```

```bash
(cd experiments/05_number_agreement_circuit && python calibrate.py)     # 101.8 s measured on an M1 Pro CPU
```

```bash
(cd experiments/05_number_agreement_circuit && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg python stage1.py)
```

The Stage 1 command writes [`stage1_results.json`](experiments/05_number_agreement_circuit/stage1_results.json)
and [`STAGE1_NOTES.md`](experiments/05_number_agreement_circuit/STAGE1_NOTES.md); it stops before Stages 2–3.

Each `experiments/NN_*/writeup.md` is self-contained: setup, results, controls, and what the result is
not. Experiment 04 additionally ships its
[pre-registration](experiments/04_causal_feature_interchange/DESIGN.md) with three dated amendments and a
dated correction that retracts three claims made in the frozen text. Each amendment commit precedes the
commit containing the output it governs — commit order, which is weaker than proof of when anything was
read, and the writeup says so.
[`lab-notebook.md`](lab-notebook.md) is the dated process record, including the traps I nearly
fell into and the pilot-versus-full-run story behind the secondary comparison.
[`learning-roadmap.md`](learning-roadmap.md) is where this is going next.

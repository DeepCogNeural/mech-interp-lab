# Experiment 05 — the number-agreement circuit

**Status: design frozen 2026-08-02 and amended through 2026-08-08 (Amendments 1–9);
calibration, Stage 1, fresh same-snapshot selection, Stage 2, and Stage 3/Q4 are complete. Q1–Q4 are
positive in 8/8 registered seeds.**
[`DESIGN.md`](DESIGN.md) is the pre-registration, including [Pre-freeze correction 1](DESIGN.md#pre-freeze-correction-1--2026-08-02)
and nine dated amendments. [Amendment 2](DESIGN.md#amendment-2--2026-08-02--corrections-and-rule-repairs-forced-by-adversarial-review)
corrects three factual errors and repairs five rules that adversarial review found unexecutable;
[Amendment 3](DESIGN.md#amendment-3--2026-08-08--pre-stage-2-executability-and-claim-boundaries)
finishes the missing candidate/bootstrap/Q3/Q4 conventions and narrows the claim labels before
Stages 2–3. [Amendment 4](DESIGN.md#amendment-4--2026-08-08--q4-matched-pool-data-role-separation)
separates Q4 rank-training and evaluation items; [Amendment 5](DESIGN.md#amendment-5--2026-08-08--clamped-z-q3-kernel-q2-item-intersection-q4-denominator-and-reporting-boundary)
freezes the clamped-`z` Q3 kernel, Q2 pair intersection, Q4 denominator guard, and seed accounting.
[`Amendment 6`](DESIGN.md#amendment-6--2026-08-08--fail-closed-denominators-seed-slots-and-gate-a-boundaries)
withdraws Q3's across-seed direct-recovery interval, fixes the eight-slot descriptive summary,
freezes diagnostic-only source-C Gate-A handling, adds the float64 Q1 `E_all` guard, and separates
scientific unresolved states from execution incompleteness. A complete base global Gate-A failure
skips Q1-specific cells without suppressing independent Q2/Q3 caches; Q3's zero-retained-item case
and Q1's deterministic tested-set fallback are recorded with explicit status codes. [Amendment 7](DESIGN.md#amendment-7--2026-08-08--same-snapshot-selection-provenance)
requires same-invocation fresh true/source-A sweeps for `C`, demotes shipped Stage 1 to a
non-blocking cross-check, and freezes the fresh selection cost at 291 logical FE (historical
cross-check: 0 FE).
[Amendment 8](DESIGN.md#amendment-8--2026-08-08--stage-2-single-invocation-provenance)
makes Stage 2 a single-invocation eight-seed experiment: interrupted checkpoints are diagnostic
only, never resumable scientific inputs, and a later authorised attempt must restart from zero.
[Amendment 9](DESIGN.md#amendment-9--2026-08-08--stage-3-q4-single-invocation-runtime-provenance)
applies the same rule to the Q4 runtime while preserving independently reviewed preparation inputs;
accepted draws must bind exactly across attempts, CSV, checkpoint, and final manifest.
[`protocol_v1.json`](protocol_v1.json) is the result-free machine-readable transcription.
The completed calibration pilot ([`calibrate.py`](calibrate.py) →
[`calibration_results.json`](calibration_results.json),
[`CALIBRATION_NOTES.md`](CALIBRATION_NOTES.md)) measured the two specificity constants and produced a
**lower-bound** runtime projection — 34.7 CPU-minutes, with three cost terms it could not price before
the freeze. Stage 1 then completed ([`stage1.py`](stage1.py) →
[`stage1_results.json`](stage1_results.json), [`STAGE1_NOTES.md`](STAGE1_NOTES.md)): it ran the true
single-flip `z@final` head sweep, the required all-head `E_all`, residual `E_ref`, and top-24 `v@subject`
sweep. Stage 1 itself adjudicates none of Q1–Q4. The later fresh selection chose eight candidates;
the completed Stage-2 run selected `[L7H4, L8H5]` as the minimum tested qualifying set and returned
positive Q1, Q2, and Q3 decisions in all eight seeds. The completed Stage-3/Q4 run returned positive
matched-span decisions in all eight seeds. See the [public result](writeup.md) and [compact evidence
packet](results/RESULTS.md).

The Stage-1-informed **incomplete proxy/projection** of about 137 CPU-minutes was based on the measured
head-sweep, joint-patch, and per-seed fixed-overhead costs. It does not price Stage 3's per-seed PCA
fitting or latent-candidate preparation, and it predates Amendment 7's fresh same-snapshot
true/source-A selection. It is not a verified total runtime. Stage 1 touched no latent span, by construction, so the
design was frozen on evidence about the stimuli, not about the answer. The actual fresh selection used
291 logical forward-equivalents and 966 seconds; the completed Stage-2 invocation used 2,528 logical
forward-equivalents and 6,272 seconds. The Stage-3/Q4 invocation used 353,120 logical
forward-equivalents and 1,265.99 seconds; its raw result remains outside Git and its compact evidence is
under `results/`.

| lifecycle gate | current state |
|---|---|
| frozen design and machine protocol | present; no result fields or protocol hash |
| calibration | complete |
| shipped Stage 1 scope | complete; descriptive only |
| fresh same-snapshot selection | complete; eight-head candidate set frozen |
| Stage 2 | complete; Q1/Q2/Q3 each 8/8 positive; independently recomputed and Advisor-accepted |
| Stage 3 preparation | complete; all eight Gate-A cells pass and have 40 rank-training / 150 evaluation pairs |
| Stage 3 / Q4 | complete; Q4 above matched-span chance in 8/8; mean `R_span=0.8935`, `R_comp=0.0848` |
| offline contract tests | 20/20 pass; protocol, splits, selection fingerprints, the TransformerLens joint-patch hook contract, candidate/Q4/bridge helpers, portable preparation binding, target-excluded matched pools, artifact guards, and the empty-`C` terminal path |

The implementation files are [`selection_source_a.py`](selection_source_a.py),
[`freeze_candidate.py`](freeze_candidate.py), [`exp05_core.py`](exp05_core.py),
[`stage2.py`](stage2.py), and [`stage3.py`](stage3.py). The single post-Q4 exploratory follow-up is
isolated in [`bridge_rescue.py`](bridge_rescue.py). Stage 2 consumed the completed, self-hashed selection artifact as an explicit input
(`--selection selection_source_a.json`) together with the protocol, calibration, and candidate
artifacts; it no longer takes `--stage1`, because shipped Stage 1 is only a descriptive cross-check
inside selection. The raw Stage-2 JSON is 1.4 GB and stays outside Git; its SHA-256 and the compact
claim inputs are recorded under [`results/`](results/). Stage 3 consumed only the independently prepared
cache, split, review, and harness receipts—not the head candidate or Stage-2 result; its compact
seed-level metrics and matched-span draws are now shipped under `results/`. The offline suite is
model-free and can be reproduced from the repository root with:

```bash
./.venv/bin/python -m unittest discover \
  -s experiments/05_number_agreement_circuit/tests -p 'test_*.py'
```

## What it asks

Experiment 04 measured *where* a causal handle on subject–verb number agreement lives at each layer:
almost entirely on the subject at layer 4, largely at the readout position by layer 10. It measured
no attention pattern, head, or path, so it cannot say what relocates the signal. This experiment
asks that, and one question experiment 04 opened and could not close.

Four axes each retain a pre-declared positive and negative scientific verdict. Amendments 5–6 also
register fail-closed non-verdict states: execution failure blocks the axis, while scientifically
unresolved seeds can force `INCONCLUSIVE_UNRESOLVED_SEEDS` under the frozen lower/upper-bound rule;
Q3's direct-recovery summary is descriptive only and has no across-seed inferential interval.

| axis | question | verdicts |
|---|---|---|
| Q1 | do few heads carry the signal at the final position? | Qualifying ≤8-head set / No qualifying ≤8-head set |
| Q2 | is what they carry *number*, or any perturbation? | Number-specific under registered controls / Specificity bound not met |
| Q3 | does it arrive *from the subject position*? | Subject-value transport shown / Subject-value transport not shown |
| Q4 | do experiment 04's twelve recurring SAE latents span the delta? | Above matched-span chance / Not above matched-span chance |

These labels do not prove necessity, mediation, or a native attention path. Q4 is only the layer-8,
`both`-position decoder-row span compared with matched-dimension empirical chance. Stage 1 is a coarse
historical ranking and adjudicates none of these four axes; the completed Stage-2 artifact adjudicates
Q1–Q3 and the completed Stage-3/Q4 artifact adjudicates Q4.

**Q4 result (8/8 positive).** Across eight seeds, projecting the registered layer-8 intervention delta
into the frozen 12-row SAE span retained about `0.89` of its causal effect, exceeded the frozen
matched-span edge in every seed, and left about `0.085` in the complementary subspace, without implying
equivalent geometric energy capture or full mediation. The exact mean `R_span` is `0.893525` (t(7) CI
`[0.890486, 0.896565]`) and `R_comp` is `0.084783` (CI `[0.082268, 0.087299]`). The geometric
squared-norm fraction is only about `0.525–0.544`, and the generic-text PCA span/both comparator is a
raw logit effect of `0.027400`, not 2.74% recovery.

Q4 does not establish natural or monosemantic latent activations, individual-latent causality,
necessity or sufficiency, a natural head→span path or mediation, a complete circuit, or
generalisation across models or tasks. The next step is one fresh held-out exploratory bridge—L7H4 →
`resid_pre8` target span → natural L8H5/readout, with an L8H5 clamp. Its executable design is in
[`bridge_rescue.py`](bridge_rescue.py); it has not yet run.

Q1 alone is close to tautological — eight late heads can push a logit they sit one step away from —
which is exactly why Q2 and Q3 are separate adjudicated axes and why the writeup may not present Q1
as a mechanism on its own.

## Three design decisions worth reading

**Nulls and method boundaries.** Experiment 04's `inconclusive` came from a binary certification gate
on a continuous quality measure its own primary statistic had already normalised away. Here,
bootstrap intervals and the source-A noise floor are quantities that can be refined as the sample grows.
Q2 instead applies fixed calibration-derived specificity ratios, while Q4 compares against an empirical
matched-span chance band. Continuous diagnostics are reported covariates. Boundary language for
near-threshold outcomes is written into the design *before* the numbers exist.

**No encoder in the latent question.** Q4 projects the *true* activation delta `x_src − x_base` onto
the twelve latents' decoder span. Nothing is encoded and nothing is reconstructed, so the
reconstruction-error term that experiment 04 had to publish a correction about does not exist here
at all.

**Selection-clean seeds, on both axes where selection happens.** The twelve latents were selected
post hoc in experiment 04, so Q4 is adjudicated on eight seeds none of which is an experiment 04
seed. The same problem turned up again in Q1 — the head candidate pool is chosen on seed `20260801`,
which was also in the adjudication set, making one of the eight seeds pass by construction — and
Amendment 2 removes it from adjudication. Selection contamination is closed by construction in both
places rather than argued about.

## What calibration already established

Measured on seed `20260899`, which is not one of the adjudication seeds:

| source | what it changes | mean readout shift | as fraction of the true flip |
|---|---|---:|---:|
| true flip | subject number | `4.46` | `1.000` |
| A — same number, different noun | subject lexical item | `−0.0078` | `0.00174` |
| B — attractor flip | the attractor's number | `+0.0071` | `0.00158` |
| C — cross-template, number-matched | the whole frame | `−0.60` | `0.13492` |

Source A is the reassuring one: changing which noun, without changing its number, does essentially
nothing to the readout. Source C is the demanding one — a number-matched sentence from a different
syntactic frame moves the readout by 13.5% of a full flip, in the direction that *reinforces* the
number both sentences already share. It is a systematic same-number push rather than noise around
zero, and it is why the specificity bound for C ends up at `0.270` rather than the `0.20` floor.

Source B is the one to watch. Flipping the attractor's number leaves the average readout essentially
where it was, while carrying the largest per-item spread of the three sources. Note what that can and
cannot say: every source is aligned to the *subject*-flip axis, which forces a mean near zero for any
effect locked to the attractor no matter how consistent it is. So items move, and **whether their
movement points consistently toward the flipped attractor's number was not measured here.** All of it
is recorded before any head has been measured.

## Prior work

No peer-reviewed head-level causal baseline exists for number agreement in GPT-2-small. Finlayson
et al. (ACL 2021) is neuron-level; the authors themselves describe their appendix's head-level
readings as possibly noise. Lepori et al. (COLM 2024) localise to layer 6's attention block —
layer-level, and the strongest citable prior. The design records a pre-registered layer-range
prediction against it. The full source-verified check, with quotes and URLs, is
[`LITERATURE.md`](LITERATURE.md).

# Experiment 05 — the number-agreement circuit

**Status: designed, not yet run.** [`DESIGN.md`](DESIGN.md) is the pre-registration. There are no
results in this directory and no code yet; that is the point of the ordering.

## What it asks

Experiment 04 measured *where* a causal handle on subject–verb number agreement lives at each layer:
almost entirely on the subject at layer 4, largely at the readout position by layer 10. It measured
no attention pattern, head, or path, so it cannot say what relocates the signal. This experiment
asks that, and one question experiment 04 opened and could not close.

Four axes, each with two pre-declared verdicts — both publishable, so no axis can end
`inconclusive`:

| axis | question | verdicts |
|---|---|---|
| Q1 | do few heads carry the signal at the final position? | Localised / Diffuse |
| Q2 | is what they carry *number*, or any perturbation? | Number-specific / Not-number-selective |
| Q3 | does it arrive *from the subject position*? | Subject-transport / Final-position-local |
| Q4 | do experiment 04's twelve recurring SAE latents span the delta? | On-path / Not distinguishable from matched chance |

Q1 alone is close to tautological — eight late heads can push a logit they sit one step away from —
which is exactly why Q2 and Q3 are separate adjudicated axes and why the writeup may not present Q1
as a mechanism on its own.

## Three design decisions worth reading

**Nulls that shrink with data.** Experiment 04's `inconclusive` came from a binary certification
gate on a continuous quality measure its own primary statistic had already normalised away. Here
adjudication rests only on effect sizes against nulls that shrink as data grows: Holm-corrected
bootstrap intervals, an empirically measured wrong-source noise floor, and matched-dimension chance
bands. Continuous diagnostics are reported covariates. Boundary language for near-threshold
outcomes is written into the design *before* the numbers exist.

**No encoder in the latent question.** Q4 projects the *true* activation delta `x_src − x_base` onto
the twelve latents' decoder span. Nothing is encoded and nothing is reconstructed, so the
reconstruction-error term that experiment 04 had to publish a correction about does not exist here
at all.

**Selection-clean seeds.** The twelve latents were selected post hoc in experiment 04, from the
intersection of five seeds' top-16 rankings. Q4 is adjudicated on eight *different* seeds, so the
selection cannot contaminate the test — closed by construction rather than argued about.

## Prior work

No peer-reviewed head-level causal baseline exists for number agreement in GPT-2-small. Finlayson
et al. (ACL 2021) is neuron-level; the authors themselves describe their appendix's head-level
readings as possibly noise. Lepori et al. (COLM 2024) localise to layer 6's attention block —
layer-level, and the strongest citable prior. The design records a pre-registered layer-range
prediction against it. Sources are checked and cited in `DESIGN.md`.

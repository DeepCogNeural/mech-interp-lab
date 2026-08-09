# Learning Roadmap

Canonical path. Ordered so that each week's reading is immediately cashed out in code. Reading without running the code does not count as done.

**Calibration note:** I'm not new to math, modeling, or reverse-engineering a black-box nonlinear system from its responses. I'm new to *transformers* and to *this field's tooling and vocabulary*. So the roadmap front-loads mechanics (what a transformer actually computes, how to hook it) and moves fast through conceptual material where the neuroscience analogue already exists.

Rough pace: one section per week, ~8-12 hrs/week. Slip freely — the notebook records reality, not the plan.

**Numbering deviation (2026-07-26):** `experiments/03_ccgp_on_sae_features/` was completed before the planned IOI replication and before the planned SAE exercise; directory numbers record work actually done, so this roadmap is not renumbered.

---

## Current plan (updated 2026-08-08)

Four experiments are done; the audits are in `lab-notebook.md`. The finding that reordered this roadmap on 2026-07-26 was that **every result up to that point was linear decodability of a representation, and none of it was causal.** Week 3 below is the part that had been skipped and it was the part that mattered most.

`experiments/04_causal_feature_interchange/` closed part of that gap: a causal interchange intervention on GPT-2-small, readout on the model's own logits, run under a frozen pre-registration. Its rule returned **inconclusive** — a faithfulness gate committed before any commit containing this experiment's output was missed by 0.006 — so experiment 04 remained a causal *basis comparison* rather than a mechanism result. Experiment 05 has now crossed that boundary: its fresh selection and Stage-2 intervention ran, and all three registered head-level axes passed in eight of eight seeds.

That distinction sets what comes next.

1. **The number-agreement circuit — [Stage 2 complete and independently accepted; Q4 prepared, not run](experiments/05_number_agreement_circuit/writeup.md).** Fresh true-source and source-A sweeps selected eight candidate heads from one unchanged model snapshot. Across eight adjudication seeds, the minimum tested prefix `[L7H4, L8H5]` recovered `0.521–0.546` of the registered direct effect, both heads were individually distinguishable, both specificity ratios cleared their frozen limits, and the frozen-pattern value-path effect was positive with every bootstrap interval above zero. The licensed conclusion is a compact, number-selective causal contribution under this intervention—not necessity, sufficiency, complete mediation, or the native unconstrained path. The immediate next experiment is the independent Q4 run: test whether experiment 04's recurring twelve-row layer-8 decoder span exceeds matched 12-dimensional spans on held-out items. Its cache and role split are complete; no Q4 verdict exists yet.
2. **Week 3 proper — activation patching and IOI**, on whatever tooling that experiment forces me to build, so the machinery arrives attached to a question rather than as an exercise.
3. **Week 5–6 — train an SAE.** Everything so far uses a published one. Training a small one is the only way to separate an SAE's objective from its particular training run.
4. Then the deferred items: re-registering experiment 04's Gate C on a decoder each arm actually ships with, a larger PCA fit, a second stimulus family, shattering/CCGP on experiment 02's toy geometries, and the Gram-interference mechanism check.

Weeks 1–2 material (transformer internals, TransformerLens) is being picked up in service of these rather than as a separate pass.

**Resolved 2026-08-02:** the repository's original through-line — superposition versus mixed selectivity — is now background rather than the organising story, and the README says so explicitly. The reason is in the record: across experiments 03 and 04 the quantity actually being measured drifted into SAE basis quality, a different and far more crowded question. What replaces it is the escalation of method the work actually followed — toy models with observable ground truth, then a real model read correlationally, then causally, then the mechanism. The tools built under the old framing all stay; only the banner changed.

**Longer arc, after the mechanism work:** the place where a V1 background is a genuine edge rather than an analogy is **divisive normalization versus LayerNorm** (bridge question 2 below). It is deliberately queued *after* experiment 05 — the argument for doing it is much stronger from someone who has already reverse-engineered a mechanism in a real model with this field's own methods than from someone importing a familiar tool first.

---

## Week 0 — Orientation (before anything else)

| Item | Link | What you get from it |
|---|---|---|
| Neel Nanda, *A Comprehensive Mechanistic Interpretability Explainer & Glossary* | https://www.neelnanda.io/mechanistic-interpretability/glossary | The field's vocabulary in one pass. Read it as a dictionary, not a textbook — you'll return to it constantly. Resolves "what does 'the residual stream' / 'OV circuit' / 'logit lens' actually mean." |
| Chris Olah et al., *Zoom In: An Introduction to Circuits* (Distill) | https://distill.pub/2020/circuits/zoom-in/ | The founding argument: networks have understandable features, features connect into circuits, and both are universal. This is the field's thesis statement. Written for a vision audience — closest entry point to your background. |
| Bereska & Gavves, *Mechanistic Interpretability for AI Safety — A Review* | https://arxiv.org/abs/2404.14082 | A map of the whole field, so you know what exists before you commit. Skim; don't study. |

**Do:** run `experiments/01_toy_models_of_superposition/` (already done — see notebook).

---

## Week 1 — What a transformer actually computes

| Item | Link | What you get from it |
|---|---|---|
| ARENA, Chapter 0: Fundamentals | https://arena-chapter0-fundamentals.streamlit.app/ | Build the pieces from scratch in PyTorch. Skip the parts you already know (backprop, optimizers); do the einops/tensor-manipulation drills — you will use einops constantly and fighting it later is a tax. |
| ARENA, Chapter 1.1: Transformer from Scratch | https://arena-chapter1-transformer-interp.streamlit.app/ | Write a GPT-2 forward pass yourself. Non-negotiable. You cannot interpret a computation you can't write down. |
| Elhage et al., *A Mathematical Framework for Transformer Circuits* | https://transformer-circuits.pub/2021/framework/index.html | **The single most important paper for you.** Rewrites the transformer as a sum of interpretable paths through a linear residual stream. QK/OV circuit decomposition, virtual attention heads, the induction-head story. This is the paper your math background pays for immediately. Read it after you can write the forward pass, not before. |

**Do:** implement attention-only 1L and 2L models; verify your forward pass matches TransformerLens's output to numerical precision.

---

## Week 2 — Tooling: TransformerLens

| Item | Link | What you get from it |
|---|---|---|
| TransformerLens repo + docs | https://github.com/TransformerLensOrg/TransformerLens | The instrument. Hooks, caching, ablation, patching. |
| TransformerLens *Main Demo* notebook | https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb | Guided tour of the API. Run it, don't read it. |
| ARENA, Chapter 1.2: Intro to Mech Interp | https://arena-chapter1-transformer-interp.streamlit.app/ | Induction heads found in a real model, hands-on. First time you see a circuit fall out of the data. |
| Olsson et al., *In-context Learning and Induction Heads* | https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html | The field's flagship result: a specific circuit, found mechanistically, that explains a macroscopic capability. Note the methodology — this is what "evidence for a circuit" looks like in practice. |

**Do:** find induction heads in a real model yourself. Log the head indices. `experiments/02_*`.

---

## Week 3 — Causal methods

| Item | Link | What you get from it |
|---|---|---|
| ARENA, Chapter 1.3: Indirect Object Identification | https://arena-chapter1-transformer-interp.streamlit.app/ | Activation patching / causal tracing, worked end to end. |
| Wang et al., *Interpretability in the Wild: a Circuit for IOI in GPT-2 Small* | https://arxiv.org/abs/2211.00593 | The canonical full circuit reverse-engineering. Read critically: what would falsify their claim? What did they not check? |
| TransformerLens *Exploratory Analysis Demo* | https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb | Practical workflow for attacking a behavior you don't understand yet. |

**Bridge note:** activation patching is optogenetics with a perfect ground truth — you can silence any "neuron" at any timestep with no off-target effects, no viral expression variability, and rerun the exact same trial. Every confound that makes causal claims hard in V1 is simply absent. The interesting question is what remains hard *anyway*, and that's the part worth your attention.

**Do:** replicate a small piece of IOI on GPT-2 small. `experiments/03_*`.

---

## Week 4 — Superposition and features

| Item | Link | What you get from it |
|---|---|---|
| Elhage et al., *Toy Models of Superposition* | https://transformer-circuits.pub/2022/toy_model/index.html | Why individual neurons aren't features. Phase transitions, feature geometry, the sparsity/interference tradeoff. (Already replicated in `experiments/01`; reread the later sections — the geometry and phase-change material is deeper than the intro.) |
| Olshausen & Field, *Emergence of simple-cell receptive field properties by learning a sparse code* | https://www.nature.com/articles/381607a0 | Your side of the bridge, 1996. Sparse overcomplete codes, in V1, thirty years ago. The mathematical object under Toy Models and under SAEs is the same one. Reread it with interp eyes. |
| Rigotti et al., *The importance of mixed selectivity in complex cognitive tasks* | https://www.nature.com/articles/nature12160 | The neuroscience formulation of polysemanticity, and — importantly — the argument that mixed selectivity is *functional* (it buys the dimensionality a linear readout needs), not a nuisance. SAE-based interp often treats superposition as something to *undo* for feature enumeration, though the same literature also studies computation in superposition. Holding both is the point. |

**Do:** extend `experiments/01` — sweep sparsity finer, reproduce the phase diagram, try >2 hidden dims and inspect the polytope geometry.

---

## Week 5-6 — Sparse autoencoders

| Item | Link | What you get from it |
|---|---|---|
| Bricken et al., *Towards Monosemanticity* | https://transformer-circuits.pub/2023/monosemantic-features/index.html | SAEs as the answer to superposition. Long. Read the main narrative fully; the appendices are a reference to return to. |
| Templeton et al., *Scaling Monosemanticity* (Claude 3 Sonnet) | https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html | Same technique, frontier model. Features that are abstract, multilingual, multimodal, and causally steerable. Also: the honest limitations section — read it twice. |
| ARENA, Chapter 1.4: Superposition & SAEs | https://arena-chapter1-transformer-interp.streamlit.app/ | Train your own SAE. |
| SAELens | https://github.com/jbloomAus/SAELens | Pretrained SAEs + tooling, so you can analyze without training from scratch. |
| Neuronpedia | https://www.neuronpedia.org/ | Browse real features in real models. Good for building intuition about what features actually look like. |

**Do:** train an SAE on a small model's activations; find and characterize 3 features you can describe in a sentence each. `experiments/04_*`.

---

## Week 7-8 — Frontier: attribution graphs

| Item | Link | What you get from it |
|---|---|---|
| Ameisen et al., *Circuit Tracing: Revealing Computational Graphs in Language Models* | https://transformer-circuits.pub/2025/attribution-graphs/methods.html | The method: replacement models, attribution graphs. |
| Lindsey et al., *On the Biology of a Large Language Model* | https://transformer-circuits.pub/2025/attribution-graphs/biology.html | The results: multi-step reasoning, planning in poems, refusal circuits, a model's stated reasoning diverging from its actual computation. This is where the field is now. The "biology" framing is deliberate and will feel familiar — it's the same epistemics as systems neuroscience. |

**Do:** pick one phenomenon from *Biology* and try to reproduce a piece of it on an open model.

---

## Ongoing — Finding your own problem

| Item | Link | What you get from it |
|---|---|---|
| Neel Nanda, *200 Concrete Open Problems in Mechanistic Interpretability* | https://www.alignmentforum.org/s/yivyHaCAmMJ3CqSyj | A ranked, annotated problem list with difficulty ratings. The single best source of a first real project. Read once early for the shape of the field; return in Week 4+ to actually pick something. |
| Neel Nanda, *How to Get Started with Mechanistic Interpretability* | https://www.neelnanda.io/mechanistic-interpretability/getting-started | Meta-advice on the field's on-ramp. Short. |
| Transformer Circuits thread (all of it) | https://transformer-circuits.pub/ | The primary literature. New entries appear regularly. |
| Distill Circuits thread (vision) | https://distill.pub/2020/circuits/ | Curve detectors, high-low frequency detectors, equivariance. Vision circuits map onto your V1 knowledge more directly than anything in the language work. If you want a project where your existing expertise is a genuine edge rather than an analogy, it's likely here. |

---

## Bridge reading — neuroscience ↔ interpretability

Not optional, and not a detour. This is where a neuroscience background meets interp most directly.

| Item | Link | What you get from it |
|---|---|---|
| Jonas & Kording, *Could a Neuroscientist Understand a Microprocessor?* | https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268 | Standard neuroscience analysis methods applied to a system with known ground truth — and they mostly fail to recover it. The most important methodological warning in both fields. Read this before you trust any correlational result, yours included. |
| Barrett, Morcos & Macke, *Analyzing biological and artificial neural networks: challenges with opportunities for synergy?* | https://arxiv.org/abs/1810.13373 | Explicit two-way mapping of methods between the fields. Written for exactly your position. |
| Cao & Yamins, *Explanatory models in neuroscience: Part 1 — taking mechanistic abstraction seriously* | https://arxiv.org/abs/2104.01489 | What "mechanistic explanation" means philosophically, and whether the interp usage survives scrutiny. Sharpens what you're actually claiming when you claim to have found a circuit. |
| Lindsay, *Convolutional Neural Networks as a Model of the Visual System* | https://arxiv.org/abs/2001.07092 | The past and present of the CNN↔visual-cortex correspondence. Where the analogy earns its keep and where it's been oversold. |

**The bridge questions to keep live** (add answers to the notebook as they develop):

1. Superposition vs. mixed selectivity — same phenomenon, or convergent solutions to different pressures? Toy Models says superposition arises from *sparse features + limited dimensions*. Rigotti says mixed selectivity buys *dimensionality for linear readout*. These emphasize different pressures on the same geometry. Which one describes V1, and can a ground-truth toy model tell them apart? (`experiments/02`.)
2. Divisive normalization is everywhere in cortex and is a strong contrast-gain and decorrelation mechanism. LayerNorm sits in the same structural slot in a transformer. Is that a real correspondence or a coincidence of form? (Suspect coincidence — but the interp field's habit of "folding LayerNorm in" and ignoring it is worth poking at.)
3. Interp has perfect observability, perfect intervention, and unlimited trials. Neuroscience has none of these. So: which of the hard problems *survive* — the ones that are hard even with a perfect experimental setup? Those are the deep ones, and they're where a population-coding background is most likely to add something.
4. What does interp lack that neuroscience developed out of necessity? (Guess: the statistics of comparing across individuals, and the discipline of trial-to-trial variability as signal rather than noise.)

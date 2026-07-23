# mech-interp-lab

A public, reproducible lab notebook for testing mechanistic-interpretability ideas in small models with fully observable ground truth.

![A linear readout of a monosemantic code stays at chance on XOR, while two mixed codes reach about 0.80 accuracy](experiments/02_superposition_and_readout/figures/01_monosemantic_cannot_read_xor.png)

Caption — monosemantic: 0.494 ± 0.005, at chance; the two mixed arms: ~0.80; 8 seeds; the nonlinearity is fixed and only geometry changes.

Learned geometry versus equal-norm random mixing: −0.015 ± 0.017 (95% CI crosses zero). **I report that null.**

The predictable question is: “the theorem already says a monosemantic code cannot linearly read XOR, so what did the experiment add?” It adds the strong form (even a monosemantic code that perfectly represents both features cannot do it, ruling out coverage), a quantitative ruler (how much mixing buys relative to equal-norm random mixing), and the null above (the gain comes from mixing plus the fixed nonlinearity, not exp01's particular learned geometry).

[Read Experiment 02: mixed coding and downstream readout.](experiments/02_superposition_and_readout/writeup.md)

## Experiment 01 — reproduce the storage story

![As feature sparsity rises, the toy model represents more features than its five-dimensional bottleneck](experiments/01_toy_models_of_superposition/figures/03_capacity_vs_sparsity.png)

Caption — sparse inputs let a five-dimensional bottleneck represent all 20 features by tolerating controlled interference.

This is a reproduction of the core toy-model result: sparse features can share non-orthogonal directions and exceed the orthogonal limit. `Σ Dᵢ` is an effective-dimension upper bound; it gets close to `m` only when training uses the bottleneck fully.

[Read Experiment 01: toy models of superposition.](experiments/01_toy_models_of_superposition/writeup.md)

## Why this repo exists

I come from computational neuroscience: probabilistic inference, stochastic modeling, primate V1. Mechanistic interpretability asks a related question — what computation a system implements and how it is encoded in its units — in systems where the internals are directly observable and experiments are reproducible. This repository is a training ground for translating those questions to toy models; it is not a claim to have reverse-engineered a transformer.

## Goals

1. **Fluency with the tools.** TransformerLens, hooks, activation patching, SAEs.
2. **Fluency with the concepts.** Superposition, features, circuits, polysemanticity, attention head roles.
3. **A record of the process.** Every experiment gets a writeup with what I expected, what happened, and what I got wrong.
4. **Find the bridge.** Where do neuroscience methods (mixed selectivity, dimensionality, normalization, noise correlations, causal perturbation) transfer to interp — and where do they break?

[Read the lab notebook](lab-notebook.md) — the process record, including the traps I almost fell into.

## Structure

```
README.md                 # you are here
LICENSE                   # MIT
learning-roadmap.md       # canonical reading/doing path, week by week
lab-notebook.md           # dated log: what I did, what I learned, what confused me
requirements.txt          # environment
experiments/
  01_toy_models_of_superposition/
    toy_models.py         # runnable script
    writeup.md            # results + interpretation
    figures/              # generated plots
  02_superposition_and_readout/
    readout.py            # runnable script
    results.json          # raw per-seed results
    writeup.md            # results + interpretation
    figures/              # generated plots
```

## Environment

Python 3.11 (torch has no 3.14 wheels yet).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the experiments:

```bash
cd experiments/01_toy_models_of_superposition && python toy_models.py   # ~1-2 min
cd experiments/02_superposition_and_readout   && python readout.py      # ~10-15 min (SMOKE=1 for a subset)
```

CPU-only. No model downloads.

## Navigation

- New here? Read `learning-roadmap.md` for the path.
- Want results? Each `experiments/NN_*/writeup.md` is self-contained.

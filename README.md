# mech-interp-lab

A personal lab for learning **mechanistic interpretability** (mech interp) and running experiments — from a computational neuroscience background.

## Why this repo exists

I come from computational neuroscience: probabilistic inference, stochastic modeling, primate V1. Mechanistic interpretability asks a structurally identical question to the one I've been asking about brains — *what computation is this system actually implementing, and how is it encoded in the units?* — but with a system you can read out perfectly, intervene on losslessly, and rerun deterministically.

This repo is where I build that translation: from "reverse-engineering V1" to "reverse-engineering transformers."

## Goals

1. **Fluency with the tools.** TransformerLens, hooks, activation patching, SAEs.
2. **Fluency with the concepts.** Superposition, features, circuits, polysemanticity, attention head roles.
3. **A record of the process.** Every experiment gets a writeup with what I expected, what happened, and what I got wrong. The log is the point.
4. **Find the bridge.** Where do neuroscience methods (mixed selectivity, dimensionality, normalization, noise correlations, causal perturbation) transfer to interp — and where do they break?

## Structure

```
README.md                 # you are here
learning-roadmap.md       # canonical reading/doing path, week by week
lab-notebook.md           # dated log: what I did, what I learned, what confused me
requirements.txt          # environment
experiments/
  01_toy_models_of_superposition/
    toy_models.py         # runnable script
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

Run the first experiment:

```bash
cd experiments/01_toy_models_of_superposition
python toy_models.py
```

CPU-only, runs in seconds. No model downloads.

## Navigation

- New here? Read `learning-roadmap.md` for the path.
- Want the current state? Read `lab-notebook.md` — newest entry at the top.
- Want results? Each `experiments/NN_*/writeup.md` is self-contained.

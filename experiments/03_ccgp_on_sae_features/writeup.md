# Experiment 03 — Shattering dimensionality × CCGP on GPT-2-small SAE features

**Status:** **gated out at Gate A — no SD, CCGP, or figure result exists yet.**
**Code:** `ccgp_sae.py`. **Raw execution record:** `results.json`. **Intended figures:** `figures/` (intentionally empty after the failed gate).

`results.json` schema for the execution actually recorded here: `exp03-gated-out-v1`, with `status`, `failed_gate`, `error_type`, `error`, `wall_clock_seconds`, environment fields, and an explicit statement that no per-seed SD/CCGP rows exist. On a complete run the script writes `exp03-results-v1`; each `per_seed_rows` entry is `{seed, arm, sd, ccgp, gap}`, where each metric maps dichotomy type to balanced accuracy and `ci95 = 1.96 · sample_sd / sqrt(n)`.

## The question

Experiment 02 fixed a toy compression and asked whether a linear readout could see one XOR. Its monosemantic arm was coordinate-wise *and* narrow (`768 → 8` in the analogue): in that special setting, the exp02 theorem says a linear readout has no product term for XOR. It would be a mistake to extrapolate that result to a real sparse autoencoder.

A real SAE is an over-complete nonlinear expansion: here, residual stream `x ∈ R^768` maps to 24,576 ReLU features. Width alone can make more dichotomies linearly separable. Thus `sae` versus `resid` is a reference/control, not the research comparison. The load-bearing question is where the sparse SAE code lands on Bernardi et al.'s shattering-dimensionality × cross-condition generalisation performance (CCGP) plane **relative to a random ReLU expansion matched in width, encoder-column scale, and mean L0**.

There is no pre-registered direction. Sparse/monosemantic pressure might make a clean factor code more abstract and less able to shatter arbitrary dichotomies; conjunctive SAE features could instead hand a linear probe interactions directly. A null between SAE and the matched random expansion would also be a result. The experiment is designed to adjudicate those opposing mechanisms, not to decorate a preferred answer.

## Scope: this is a lens, not a model intervention

An SAE is a read-out lens hung beside the residual stream. GPT-2's downstream components still read the original mixed residual stream; this experiment never swaps the SAE code into the forward computation, ablates it, or measures a change in model behaviour. Any eventual lower or higher score is a property of a *code/lens* under these probes — for example, a sparse basis might trade nonlinear expressivity for abstraction or enumerability as a code. It would not show that an SAE harms, degrades, removes, or loses computation the model uses.

## Pre-specified setup

| arm | representation | role |
|---|---|---|
| `resid` | `x` at `blocks.{L}.hook_resid_pre`, read at final `.` | GPT-2's mixed-code reference |
| `sae` | `ReLU((x − b_dec) @ W_enc + b_enc)` | research object |
| `sae_recon` | `f @ W_dec + b_dec` | round-trip control |
| `rand_exp` | Gaussian ReLU expansion, width/norm/L0-matched to SAE | load-bearing control |
| `rand_exp_dense` | same random expansion without per-sample top-k | reported upper reference, not the primary comparison |

Each lexical item has all eight cells of NUMBER (singular/plural) × TENSE (past/present) × POLARITY (affirmative/negated):

```
The {ADJ} {NOUN} {AUX} {POL} {VERB} the {ADJ2} {OBJ} {ADV} .
```

The final `.` is the sole read-out token. Do-support keeps the main verb bare; `indeed` versus `not` length-matches the polarity slot. The script tokenises all eight cells per item, rejects any item with unequal token length or a non-identical final-token id, and asserts the retained equality. It uses item-disjoint folds: a lexical draw cannot occur in both a probe's train and test data.

The layer pilot is `6, 7, 8, 9`; it selects the layer with the strongest mean residual main-effect decodability and requires every main effect to reach 0.85 balanced accuracy. With eight conditions there are 35 balanced dichotomies. SD is item-cross-validated mean balanced accuracy over all 35. CCGP holds out one condition from each dichotomy side, uses all 16 `4 × 4` choices in a full run, and keeps lexical items disjoint even across those held-condition tests. The code trains each family of dichotomies as vectorised multi-output torch probes with train-split-only standardisation and L2 weight decay 0.08.

## What actually ran

I first inspected `pip install sae-lens --dry-run` in the existing `.venv`. It could not resolve PyPI, so it made no environment change. I did **not** create `.venv-exp03`; exp03 therefore still targets the existing Python 3.11 environment (`torch 2.13.0`, `transformer-lens 3.5.1`, `transformers 5.14.1`). `sae_lens` is absent, so the implementation uses the permitted direct-safetensors fallback: `hf_hub_download` from `jbloom/GPT2-Small-SAEs-Reformatted`, then manual `W_enc`, `b_enc`, `W_dec`, `b_dec` encode/decode.

`SMOKE=1 python ccgp_sae.py` was executed on 2026-07-26. It failed at Gate A before model loading: the local Hugging Face cache contains neither GPT-2 nor the requested `blocks.8.hook_resid_pre/sae_weights.safetensors`, and terminal DNS could not resolve `huggingface.co`. The script's recorded Gate-A interval was **1.025 s** (the shell invocation also includes import/start-up time). A direct `hf_hub_download` check for the SAE independently confirmed both facts: `local_files_only=True` reported no cache entry, and the network attempt failed on name resolution.

That is a blocked load, not a negative experiment result. Per the staged rule, I stopped rather than substitute synthetic features, condition means, a different SAE, or invented metric values.

## Gates and result table

| gate | required measurement | actual outcome |
|---|---|---|
| A — loading | GPT-2 + exact res-jb SAE; reconstruction EV/MSE and mean L0 | **failed before either model or SAE weight loaded**; EV/MSE/L0 unavailable |
| B — stimuli | equal token length; residual main effects ≥ 0.85 | not run after Gate A |
| C — headroom | residual two-way-XOR SD below saturation | not run after Gate A |
| D — 5-seed run | SD, CCGP, figures, paired CIs | not run after Gate A |

Consequently there is no headline table, no dichotomy-type breakdown, no `sae − rand_exp` paired difference, and no full-run wall-clock time to report. Calling these unavailable fields a null would be misleading: they were never measured.

## Controls the completed run will report

- token-length equality and retained/dropped lexical-item counts;
- residual main-effect decodability for every pilot layer and every arm at the chosen layer;
- SAE reconstruction explained variance/MSE and mean L0, plus random sparse target-versus-achieved L0 and dense-reference L0;
- surviving feature widths after the all-samples-zero rule;
- train-minus-test gap by arm;
- all SD dichotomy types (main effect, two-way XOR, three-way parity, unstructured), overall CCGP, and main-effect CCGP;
- within-seed paired `sae − rand_exp` CIs.

The code does not let an arm comparison proceed if the residual XOR family is already at or above 0.98, because there would be no discriminating headroom.

## Verification performed

- `python -m py_compile ccgp_sae.py` passed.
- A synthetic unit check imported the module, enumerated exactly 35 dichotomies with the intended `3 / 3 / 1 / 28` type split, and exercised the SD and CCGP probe paths on item-labelled mock data. It is a plumbing check only, not evidence about GPT-2 or an SAE.
- `SMOKE=1 python ccgp_sae.py` reached the documented Gate-A stop and wrote the failure manifest above.
- The existing `.venv` was not changed. I nevertheless attempted the required exp02 smoke regression after this work: it began normally and completed the first three `m=8, S=0.0, seed=0` cells, but this sandbox terminated the foreground command after about 28.5 seconds before its normal completion marker. A background retry was also terminated immediately by the sandbox. Thus exp02 completion is **not verified in this session**; the worktree stayed clean, but that is not a substitute for a completed regression.

## Caveats

- No real model activations were available in this execution. The script design is reviewable; its scientific outcome is not yet known.
- `SMOKE=1` deliberately uses two seeds and 2/16 CCGP held-condition splits. It is a pipeline subset only; only the default command is the specified five-seed, all-16-split run.
- Shattering and CCGP describe geometry of the probed code under this controlled stimulus family. They do not identify a transformer circuit or establish causal use of any feature.
- The word **theorem** belongs only to the exp02 coordinate-wise compression/XOR anchor cited above. Nothing trained or measured by this experiment earns that language.
- Gemma-2/GemmaScope is future work, not a substitute for the specified GPT-2-small res-jb test.

## Next

Restore access to cached or downloadable `gpt2` and `jbloom/GPT2-Small-SAEs-Reformatted` files, then rerun the script from this directory. The gates will restart at A; no manual layer selection or result-file editing is needed. Only after a complete five-seed command should the intended figures and a README-facing claim be produced.

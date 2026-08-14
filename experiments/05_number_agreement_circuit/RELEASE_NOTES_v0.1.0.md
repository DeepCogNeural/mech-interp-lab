# Exp05 causal subspace bridge in GPT-2-small (exploratory)

## Highlights

- Across eight registered seeds, L7H4 + L8H5 were the smallest tested set that passed the frozen
  number-specificity and transport controls in every seed (`0.521–0.546` of the frozen direct effect).
- The fixed 12-row layer-8 SAE decoder span retained `R_span = 0.8935` (t(7) CI
  `[0.8905, 0.8966]`) and beat the frozen second-largest edge of 100 matched spans in every seed.
- A fresh exploratory bridge retained `R_target = 0.6786` (t(7) CI `[0.6738, 0.6834]`) and beat
  the maximum of 100 target-excluded matched spans in all eight seeds. The complete L8H5 `hook_z@final`
  clamp arm remained `0.6740` (CI `[0.6696, 0.6785]`).

## Evidence and provenance

- [Research brief](https://github.com/DeepCogNeural/mech-interp-lab/blob/exp05-causal-subspace-v0.1.0/experiments/05_number_agreement_circuit/RESEARCH_BRIEF.md)
- [Compact results packet](https://github.com/DeepCogNeural/mech-interp-lab/blob/exp05-causal-subspace-v0.1.0/experiments/05_number_agreement_circuit/results/RESULTS.md)
- [Full writeup](https://github.com/DeepCogNeural/mech-interp-lab/blob/exp05-causal-subspace-v0.1.0/experiments/05_number_agreement_circuit/writeup.md)
- [Bridge figure](https://github.com/DeepCogNeural/mech-interp-lab/blob/exp05-causal-subspace-v0.1.0/experiments/05_number_agreement_circuit/results/figure_bridge_rescue.svg)
- The model-backed raw bridge result remains outside Git. The checked-in packet is a hash-bound
  reaggregation of compact CSV rows and the historical provenance receipt; it does not revalidate the
  missing raw result in the current invocation.

## Claim boundary

The reported values are directed logit-effect ratios, not activation-reconstruction percentages. This
exploratory bridge has no preregistered verdict. The release does not claim natural or monosemantic latent
semantics, individual-latent causality, necessity, sufficiency, mediation, a complete circuit, or
generalisation across models or tasks.

## Reaggregation

The compact presentation can be regenerated without model execution:

```bash
python3 experiments/05_number_agreement_circuit/make_bridge_summary.py --reaggregate-checked-in
```

That command validates the frozen compact CSV hashes, recomputes the seed-level summaries, and records
that the off-Git raw result was not reopened.

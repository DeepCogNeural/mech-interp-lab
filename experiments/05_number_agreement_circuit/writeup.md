# Exp05 public result: a stable two-head set and subject-value transport

## Question

Which heads carry the number signal at the final position, and do they carry
subject-number information to that position?

## Evidence

The public packet in [`results/RESULTS.md`](results/RESULTS.md) is generated from
the fresh selection artifact, frozen candidate, completed Stage-2 artifact, and
the Stage-3 preparation manifests.  It contains no copy of the 1.4 GB Stage-2
JSON or the 73 MB pair-level CSV.

- **Selection (Panel A):** [`results/selection_candidates.csv`](results/selection_candidates.csv)
  records the eight-head candidate order, true single-head effects, source-A
  noise edge, and eligibility fields extracted from `candidate.json`.
- **Q1–Q3 (Panel B):** [`results/stage2_seed_metrics.csv`](results/stage2_seed_metrics.csv)
  records one compact row per Stage-2 seed, including the two-head recovery,
  both specificity-control ratios, and the Q3 path statistics with their
  per-seed bootstrap intervals.
- **Stage 3/Q4 (Panel C):** [`results/stage3_seed_metrics.csv`](results/stage3_seed_metrics.csv)
  records one compact row per Q4 seed, while [`results/stage3_matched_ratios.csv`](results/stage3_matched_ratios.csv)
  retains all 800 matched-span ratios and decoder-row IDs. The source-bound adjudication is summarized in
  [`results/stage3_result_summary.json`](results/stage3_result_summary.json); the Gate-A preparation and
  frozen 40/150 rank-training/evaluation split remain visible in [`results/stage3_preparation.csv`](results/stage3_preparation.csv)
  and [`results/stage3/`](results/stage3/). Raw result files remain outside Git.

![Experiment 05 public evidence: selection, eight-seed Q1–Q3 results, and the positive Stage-3/Q4 matched-span result](results/figure_exp05_main.png)

## Finding

> Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal compact
> set, passed preregistered specificity controls, and causally transported
> subject-number information under a frozen-pattern value-path intervention.

The machine-readable summary reports positive registered seed cells for Q1, Q2,
and Q3, with the exact values and thresholds in
[`results/stage2_public_summary.json`](results/stage2_public_summary.json).
The main figure is [`results/figure_exp05_main.svg`](results/figure_exp05_main.svg)
and its high-resolution PNG counterpart.

For Q4, across eight seeds, projecting the registered layer-8 intervention delta into the frozen
12-row SAE span retained about `0.89` of its causal effect, exceeded the frozen matched-span edge in
every seed, and left about `0.085` in the complementary subspace, without implying equivalent
geometric energy capture or full mediation. The mean `R_span` is `0.893525` (t(7) CI
`[0.890486, 0.896565]`) and the mean `R_comp` is `0.084783` (CI `[0.082268, 0.087299]`). The
geometric squared-norm fraction is about `0.525–0.544`; the generic-text PCA span/both comparator is
a raw logit effect of `0.027400`, not 2.74% recovery.

## Not claimed

These results do not establish that the two heads are required, that they alone
account for all behavior, that every causal route is explained by this pair, or
that the pair is an exhaustive circuit or a native attention route. Q3's
direct-recovery values are descriptive only and do not adjudicate the Q3 result.
The Q4 span is not an activation reconstruction claim and does not establish natural or monosemantic
latent activations, individual-latent causality, necessity or sufficiency, a natural head→span path or
mediation, a complete circuit, or generalisation across models or tasks.

The next step is one fresh held-out exploratory bridge chosen with the Advisor: L7H4 → `resid_pre8`
target span → natural L8H5/readout, with an L8H5-clamp arm. Its executable design is
[`bridge_rescue.py`](bridge_rescue.py); it has not yet run.

The prior failed run `30d941c` appears only in
[`results/execution_ledger.csv`](results/execution_ledger.csv) with
`science_eligible=false`.

## Reproduce the package

Run the model-free packager against the six source artifacts:

```bash
python3 experiments/05_number_agreement_circuit/make_claim_ladder.py \
  --selection-source-a /tmp/exp05/selection_source_a.json \
  --candidate /tmp/exp05/candidate.json \
  --stage2 /tmp/exp05/stage2_results.json \
  --stage3-cache /tmp/exp05-stage3/stage3_gate_a_cache.jsonl \
  --stage3-split /tmp/exp05-stage3/stage3_split_manifest.json \
  --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json \
  --stage3-results /tmp/exp05-stage3/stage3_results.json \
  --stage3-draws /tmp/exp05-stage3/stage3_draws.csv
```

The script extracts only compact tables and generates the SVG/PNG figure; it
does not run the model or Stage 3/Q4. Verify a checked-out packet with:

```bash
(cd experiments/05_number_agreement_circuit/results && sha256sum -c checksums.sha256)
```

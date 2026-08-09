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
- **Stage 3 (Panel C):** [`results/stage3_preparation.csv`](results/stage3_preparation.csv)
  records the eight Gate-A-passed preparation cells and their frozen 40/150
  rank-training/evaluation split.  Exact byte-identical preparation inputs are
  copied under [`results/stage3/`](results/stage3/).  This is preparation
  evidence only.

![Experiment 05 public evidence: selection, eight-seed Q1–Q3 results, and Stage-3 pending](results/figure_exp05_main.png)

## Finding

> Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal compact
> set, passed preregistered specificity controls, and causally transported
> subject-number information under a frozen-pattern value-path intervention.

The machine-readable summary reports positive registered seed cells for Q1, Q2,
and Q3, with the exact values and thresholds in
[`results/stage2_public_summary.json`](results/stage2_public_summary.json).
The main figure is [`results/figure_exp05_main.svg`](results/figure_exp05_main.svg)
and its high-resolution PNG counterpart.

## Not claimed

This result does not establish that the two heads are required, that they alone
account for all behavior, that every causal route is explained by this pair, or
that the pair is an exhaustive circuit or a native attention route.  Q3's
direct-recovery values are descriptive only and do not adjudicate the Q3 result.
Stage 3 / Q4 is still pending and has no scientific verdict.

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
  --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json
```

The script extracts only compact tables and generates the SVG/PNG figure; it
does not run the model or Stage 3.  Verify a checked-out packet with:

```bash
(cd experiments/05_number_agreement_circuit/results && sha256sum -c checksums.sha256)
```

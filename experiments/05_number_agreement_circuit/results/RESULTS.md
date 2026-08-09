# Exp05 public results

This directory is a compact, source-bound evidence packet.  It answers the
question, shows the registered Stage-2 evidence, and keeps the Stage-3 boundary
visible without copying the raw JSON or pair-level CSV.

## Question → evidence → finding

**Question.** Which heads carry the number signal, and do they transport
subject-number information?

**Evidence.** The fresh same-snapshot selection and frozen candidate are in
[`selection_candidates.csv`](selection_candidates.csv).  The complete Stage-2
eight-seed cells are reduced to [`stage2_seed_metrics.csv`](stage2_seed_metrics.csv).
The per-seed Stage-3 Gate-A and rank/evaluation split preparation is in
[`stage3_preparation.csv`](stage3_preparation.csv).
The exact preparation inputs are copied under [`stage3/`](stage3/), with their
source hashes recorded in [`stage3_preparation_summary.json`](stage3_preparation_summary.json).

**Finding.** Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal
compact set, passed preregistered specificity controls, and causally transported
subject-number information under a frozen-pattern value-path intervention.
The machine-readable summary records Q1=8/8,
Q2=8/8, and
Q3=8/8 positive registered seed cells.

The visual overview is [`figure_exp05_main.svg`](figure_exp05_main.svg) (with a
high-resolution [`PNG`](figure_exp05_main.png)).  Panel A is selection, Panel B
is the eight-seed Q1–Q3 evidence, and Panel C is an explicit Stage-3 pending
placeholder.

## Not claimed

These results do not establish that the two heads are required or alone
sufficient, that every causal route is explained by this pair, or that the pair
constitutes an exhaustive circuit or native attention route.  Q3 direct-recovery
values are descriptive only.  Stage 3 / Q4 has no scientific verdict yet.

The historical failed run `30d941c` is retained only in
[`execution_ledger.csv`](execution_ledger.csv) with `science_eligible=false`.

## Reproduce

From the repository root, run the model-free packaging script against the six
source artifacts (the raw files stay outside Git):

```bash
python3 experiments/05_number_agreement_circuit/make_claim_ladder.py \
  --selection-source-a /tmp/exp05/selection_source_a.json \
  --candidate /tmp/exp05/candidate.json \
  --stage2 /tmp/exp05/stage2_results.json \
  --stage3-cache /tmp/exp05-stage3/stage3_gate_a_cache.jsonl \
  --stage3-split /tmp/exp05-stage3/stage3_split_manifest.json \
  --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json
```

[`index.json`](index.json) and [`artifact_index.json`](artifact_index.json)
list generated files and source receipts.  Verify the checked-in packet with
`(cd experiments/05_number_agreement_circuit/results && sha256sum -c checksums.sha256)`.

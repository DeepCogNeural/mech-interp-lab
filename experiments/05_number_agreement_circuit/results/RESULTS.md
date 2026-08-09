# Exp05 public results

This directory is a compact, source-bound evidence packet.  It answers the
question, shows the registered Stage-2 evidence, and keeps the Stage-3 inputs
and claim boundary visible without copying the raw model result.

## Question → evidence → finding

**Question.** Which heads carry the number signal, and do they transport
subject-number information?

**Evidence.** The fresh same-snapshot selection and frozen candidate are in
[`selection_candidates.csv`](selection_candidates.csv).  The complete Stage-2
eight-seed cells are reduced to [`stage2_seed_metrics.csv`](stage2_seed_metrics.csv).
The per-seed Stage-3 Gate-A and rank/evaluation split preparation is in
[`stage3_preparation.csv`](stage3_preparation.csv).
The Gate-A cache, split manifest, and role CSV are exact copies under
[`stage3/`](stage3/).  The prepare manifest is a portable copy whose two path
fields are basenames; raw and portable hashes are both recorded in
[`stage3_preparation_summary.json`](stage3_preparation_summary.json).

**Finding.** Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal
compact set, passed preregistered specificity controls, and causally transported
subject-number information under a frozen-pattern value-path intervention.
The machine-readable summary records Q1=8/8,
Q2=8/8, and
Q3=8/8 positive registered seed cells.

The completed Q4 result is reduced to [`stage3_seed_metrics.csv`](stage3_seed_metrics.csv),
with all 800 matched-span ratios and latent IDs in [`stage3_matched_ratios.csv`](stage3_matched_ratios.csv).
The source-bound adjudication summary is [`stage3_result_summary.json`](stage3_result_summary.json);
the accepted Advisor and harness receipts are copied under
[`stage3/`](stage3/) and bound by self-hash plus the raw result's receipt hashes.
The preparation manifest is also copied there in portable form: only its two
split-input path fields are rewritten to basenames; its raw source receipt and
the Q4-declared raw manifest SHA remain recorded separately.
The 41 MB raw result and raw draw CSV remain outside Git.

**Q4 finding.** Above matched-span chance, the target 12-decoder-row span recovered a mean of 89.35% of the directed logit effect and beat the frozen second-largest of 100 matched spans in all 8 seeds.  The mean R_span is 0.8935 (95% t(7) CI
[0.8905, 0.8966]); the frozen second-largest matched edge has mean 0.4756
(95% t(7) CI [0.4013, 0.5498]).  The generic PCA comparator is reported as a
raw logit effect (mean E=0.0274), not as a recovery percentage.

The visual overview is [`figure_exp05_main.svg`](figure_exp05_main.svg) (with a
high-resolution [`PNG`](figure_exp05_main.png)).  Panel A is selection, Panel B
is the eight-seed Q1–Q3 evidence, and Panel C is the eight-seed Q4 matched-span comparison.

## Not claimed

These results do not establish that the two heads are required or alone
sufficient, that every causal route is explained by this pair, or that the pair
constitutes an exhaustive circuit or native attention route.  Q3 direct-recovery
values are descriptive only.
- This matched-span result does not establish that the encoder features are natural, necessary, or sufficient.
- It does not establish mediation, an attention route, or a native model path.
- R_span is a directed-logit effect ratio, not an 89% geometric reconstruction claim.
- The generic PCA value (0.0274 mean for PCA_span/both) is a raw logit effect, not 2.74% recovery.

The historical failed run `30d941c` is retained only in
[`execution_ledger.csv`](execution_ledger.csv) with `science_eligible=false`.

## Reproduce

From the repository root, run the model-free packaging script against the source
artifacts (the raw files stay outside Git):

```bash
python3 experiments/05_number_agreement_circuit/make_claim_ladder.py \
  --selection-source-a /tmp/exp05/selection_source_a.json \
  --candidate /tmp/exp05/candidate.json \
  --stage2 /tmp/exp05/stage2_results.json \
  --stage3-cache /tmp/exp05-stage3/stage3_gate_a_cache.jsonl \
  --stage3-split /tmp/exp05-stage3/stage3_split_manifest.json \
  --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json \
  --stage3-results /tmp/exp05-stage3/stage3_results.json \
  --stage3-draws /tmp/exp05-stage3/stage3_draws.csv \
  --stage3-review-receipt /tmp/exp05-stage3/stage3_prepare_review.json \
  --stage3-harness-receipt /tmp/exp05-stage3/stage3_harness_receipt.json
```

[`index.json`](index.json) and [`artifact_index.json`](artifact_index.json)
list generated files and source receipts.  Verify the checked-in packet with
`(cd experiments/05_number_agreement_circuit/results && sha256sum -c checksums.sha256)`.

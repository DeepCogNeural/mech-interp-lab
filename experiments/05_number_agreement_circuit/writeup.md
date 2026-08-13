# Exp05 public result: a stable two-head set, span concentration, and an exploratory bridge

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

The post-Q4 bridge is summarized in [`figure_bridge_rescue.svg`](results/figure_bridge_rescue.svg)
([PNG](results/figure_bridge_rescue.png)); its compact rows and claim boundary are in
[`bridge_result_summary.json`](results/bridge_result_summary.json).

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

The single follow-up then tested a narrower causal bridge on eight fresh held-out seeds (`20260814–21`):
L7H4 was intervened, the resulting delta was measured at `resid_pre8`, and the fixed 12-row span was
rescued against 100 target-excluded matched rank-12 spans per seed. The target recovered `67.9%` of
the directed effect (mean `R_target=0.678639`, 95% t(7) CI `[0.673841, 0.683437]`) and beat the
actual maximum matched span in all eight seeds. The complementary ratio was `0.305339` (CI
`[0.301260, 0.309418]`); the matched-span mean was `0.065910` (CI `[0.049408, 0.082411]`). When the
complete L8H5 `hook_z` output at the final query position was overwritten by the source-A baseline, the
target remained `0.674047` (CI `[0.669586, 0.678508]`), while the natural reader projection coefficient
was only descriptive (`0.027–0.124`). Because `hook_z` is the per-head output after
attention-pattern-weighted value aggregation, this is not a value-only clamp. The result supports a
reproducible causal subspace under the tested intervention, not dominant dependence on L8H5's tested
final-position output or a mediation path. The bridge is exploratory and has no preregistered verdict.

## Not claimed

These results do not establish that the two heads are required, that they alone
account for all behavior, that every causal route is explained by this pair, or
that the pair is an exhaustive circuit or a native attention route. Q3's
direct-recovery values are descriptive only and do not adjudicate the Q3 result.
The Q4 span is not an activation reconstruction claim and does not establish natural or monosemantic
latent activations, individual-latent causality, necessity or sufficiency, a natural head→span path or
mediation, a complete circuit, or generalisation across models or tasks.

The near-closure of target plus complement is a separate nonlinear arm and is not a linear attribution.
The clamp does not show that L8H5 is irrelevant: parallel downstream routes and other L8H5 query
positions were not tested. At the final query it overwrites the output produced jointly by the
attention pattern and value stream; it does not separately identify either contribution. The bridge does not establish native latent semantics, individual-latent
causality, necessity, sufficiency, a complete circuit, or generalisation across models or tasks. Its
code is [`bridge_rescue.py`](bridge_rescue.py), and the compact evidence is linked above. The proposed
all-position-clamp factorial is non-identifying because the final-only upstream intervention changes no
non-final `resid_pre8` position and causal masking prevents earlier query positions from seeing that
edit. The [result-blind, interactive AI-advisor-reviewed Experiment 06 design](../06_cross_template_bridge/DESIGN.md) instead fixes
the model, L7H4, span, and matched controls while moving to a mechanism-held-out evaluation on the
calibration-exposed relative-clause family. It is implemented but has not run.

The prior failed run `30d941c` appears only in
[`results/execution_ledger.csv`](results/execution_ledger.csv) with
`science_eligible=false`.

## Reproduce the package

Run the model-free packager against the listed source artifacts:

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

The script extracts only compact tables and generates the SVG/PNG figure; it does not run the model or
Stage 3/Q4. Those source artifacts, and the bridge raw result, are outside Git. Checked-in CSV/JSON rows
support static reaggregation, but full packet regeneration requires the hash-bound off-Git inputs and a
full model rerun additionally requires cached model/SAE assets. Verify a checked-out packet with:

```bash
(cd experiments/05_number_agreement_circuit/results && sha256sum -c checksums.sha256)
```

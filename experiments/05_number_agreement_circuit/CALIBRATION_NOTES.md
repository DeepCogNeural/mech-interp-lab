# Experiment 05 calibration pilot notes

- Status: `completed_with_lower_bound_runtime_projection`.
- Calibration seed: `20260899`. It is deliberately not one of either eight-seed adjudication set, so no measured constant is circular with a later adjudicating seed.
- Scope: CPU-only float32 offline run. Only `hook_resid_pre` residual-stream edits were measured. No head-level or latent-span-level activation/effect/ranking was measured.
- Environment: device `cpu`, torch `2.13.0`, platform `macOS-26.6-arm64-arm-64bit`.

## Gate A and source constructions

- Base family: retained 234/240 (both-correct=0.975, median d_gap=5.107); pass=True.
- Source C winning family: `source_C_relative_clause_with_adverb`; retained 240/240 (both-correct=1.000, median d_gap=6.448); pass=True.
- Source A changes only the subject lexical item while preserving its number and the base adjective/preposition/attractor. Source B flips only the attractor number while preserving the base subject.
- Source C is a relative-clause matrix-agreement frame (`The SUBJECT that the ATTRACTOR often RELVERB`), recorded because it changes syntax and lexical frame while retaining a one-token subject and a final is/are decision. The committed record establishes that this family was recorded and passed Gate A; it does not establish when residual effects were inspected or exclude unrecorded attempts. Gate A is the only selection criterion visible in the committed record.
- Source-C template search:
  - `source_C_relative_clause_with_adverb`: pass=True; retained=240; reason=selected: passed all unchanged Gate-A thresholds.

## Cross-template indexing

- For every directed C→base edit, the layer-8 source residual was gathered at C's own `[subject, final]` token positions, the base residual at the base prompt's own `[subject, final]` positions, and the resulting two vectors were written only at the base positions.
- Sequence length differed on 468/468 directed edits; final-position index differed on 468/468. This verifies that source positions were not reused as base write indices.

## Calibration constants

- source_A: rho_full=0.00174 (bootstrap 95% CI 0.00019 to 0.01527); rho_noise=0.08488 (95% CI 0.07571 to 0.09491).
- source_B: rho_full=0.00158 (bootstrap 95% CI 0.00047 to 0.03233); rho_noise=0.27129 (95% CI 0.25615 to 0.28812).
- source_C: rho_full=0.13492 (bootstrap 95% CI 0.10909 to 0.16230); rho_noise=0.22002 (95% CI 0.19871 to 0.24235).
- Frozen formula outputs: theta_spec^A=0.20000; theta_spec^C=0.26983. Source B is recorded but never adjudicates.

## Residual-handle sanity and self-tests

- True flip at layer 8/both: E_resid/d_gap=0.83463 (floor 0.50), sign consistency=1.00000 (floor 0.90).
- Self-tests: zero bitwise=True; start_at_layer=8 max_abs=0; prompt-swap max_abs=5.72e-06.

## Runtime projection

- Projection formula: sum(per-patch median wall-clock cost x enumerated patch count) + measured per-seed fixed block x seed blocks + one-time model/SAE loads.
- Enumerated counts: Stage 1 = 144 z patches + 24 v patches; Stage 2 = per 8 seeds two 144-head sweeps, one all-144 joint patch, <=8 nested patches, <=8 two-forward path patches, and three A/C/B joint-set patches required by Q2/reporting; Stage 3 = 9 position/intervention patches + 2 alpha patches + 12 PCA-span patches per seed, plus 100 + 7x20 random draws.
- Conservative no-reuse lower-bound total: 34.71 CPU-minutes against the 120-minute cap; lower-bound fit=True.
- At the permitted 100-retained-pair floor, the same lower-bound projection is 15.89 CPU-minutes (saves 18.82 minutes).
- Important defect: this cannot certify the full design meets the cap. The requested pre-freeze measurement cannot price future attention-value caching without head-level access, and the stated two-term formula omits Stage-3 PCA fitting and latent-candidate preparation. The reported number is therefore a lower bound, not a complete wall-clock projection.

## Design defects / unmeasurable terms

- Stage 2's frozen Q2 rule requires direct joint-set measurements for source A and source C, and source B is required for reporting, but the runtime-budget prose enumerates only the two 144-head sweeps and does not count these three joint-set patches per seed. This calibration includes them in its lower-bound patch count.
- A complete later z/v implementation requires source attention-value caching. Measuring that cache now would violate the design's literal no-head-measurement blinding claim, so the requested residual-only fixed-overhead block cannot price it.
- Stage 3 requires per-seed PCA fitting and SAE candidate-pool preparation. The requested two-term projection does not assign either a term, and pre-freeze pricing them would inspect prohibited latent/span material. Thus the specified projection is necessarily a lower bound rather than a certifiable total against 120 CPU-minutes.

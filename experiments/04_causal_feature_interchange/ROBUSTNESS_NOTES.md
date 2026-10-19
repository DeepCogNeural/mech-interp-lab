# Experiment 04 Amendment 2 robustness record

- Label: `specified after unblinding of the primary, before its own computation; never adjudicating`
- Wall time: 1344.0 seconds.
- This record measures coordinate codes under additive residual-difference write-back; it makes no claim about whether the SAE changes model computation.

## R0: recomputed from the original manifest

- Noun-disjoint SAE k=16 pooled: R=0.5936296064503066, directed edits=150.
- Five-seed top-16 intersection: 12; mean pairwise intersection=13.300.
- For k=16 sign consistency, see JSON `sign_consistency_at_k16`; numerators and denominators are retained per basis, per seed, and pooled.

## R1: PCA fitting budget

- 2048 tokens: within AUC=0.195118 ± 0.007897; absolute AUC=0.195118 ± 0.007897.
- 8192 tokens: within AUC=0.213229 ± 0.022783; absolute AUC=0.213229 ± 0.022783.
- 8192−2048 within AUC: +0.018111 ± 0.025066.
- 8192−2048 absolute AUC: +0.018111 ± 0.025066.
- All 768 PCA components capture approximately 100% variance at both budgets; top-64 seed means are 0.529544 (2,048) and 0.461720 (8,192); variance fractions in the bottom half of eigenvalues are 0.071230 and 0.127718.
- Amendment 2 defines no numerical threshold for a flat curve; only measured changes and t(4) intervals are reported, without that classification.

## R2: sae_ridge

- mixed_generic_plus_rank_templates: Gate C E(full)/E(resid)=0.887382 ± 0.013085; AUC within=0.561495 ± 0.019843; absolute=0.498168 ± 0.013676.
- mixed_generic_plus_rank_templates sae_ridge−PCA(8192) robustness difference: within=+0.348266 ± 0.037148; absolute=+0.284939 ± 0.030798. This is non-adjudicating.
- pure_generic: Gate C E(full)/E(resid)=0.413714 ± 0.060752; AUC within=0.601839 ± 0.066887; absolute=0.250348 ± 0.057485.
- pure_generic sae_ridge−PCA(8192) robustness difference: within=+0.388609 ± 0.081917; absolute=+0.037119 ± 0.075616. This is non-adjudicating.

## Open decisions in the amendment

- R1 specifies no threshold for calling the budget curve flat.
- `run_results.json` retains generic-pool generation metadata, not activations; each recorded seed reconstructs one 8,192-token pool shared in memory by both R1 budgets and R2.
- R2 runs the mixed version only if pure-generic coverage is below 0.95; JSON explicitly records each seed choice and unrun items.

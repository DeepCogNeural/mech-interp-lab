# Experiment 04 full-run record

- Status: `completed`. Wall time: 1760.0 seconds.
- Control branch: frozen Rule 2 in DESIGN.md Amendment 1; the primary comparison is `AUC(sae) - AUC(pca)`; `rand_exp` is reported but not adjudicated.
- Primary interval: +0.303897 ± 0.023284 (t(4) 95%); frozen branch: `inconclusive: SAE Gate C did not pass in every seed`.

## AUC and k50 per basis

| basis | AUC(top-k), t(4) 95% | k50 (five seeds) |
|---|---:|---|
| sae | 0.517126 ± 0.002031 | 16, 16, 16, 16, 16 |
| pca | 0.213229 ± 0.022783 | 64, 128, not reached within grid, 128, 64 |
| neuron | 0.156661 ± 0.005198 | not reached within grid, not reached within grid, not reached within grid, not reached within grid, not reached within grid |
| rand_exp | 0.178859 ± 0.023904 | not reached within grid, not reached within grid, not reached within grid, not reached within grid, not reached within grid |

- Non-adjudicating SAE ranking reference: `AUC(sae_topk) - AUC(sae_randk)` = +0.349701 ± 0.015546.

## Five-seed write-back faithfulness E(full)/E(resid)

| measurement | mean ± t(4) 95% | per-seed |
|---|---:|---|
| sae_trained_decoder | 0.693941 ± 0.013875 | 0.704633, 0.707189, 0.685405, 0.683218, 0.689257 |
| rand_exp_8k_dual_ridge | 0.652333 ± 0.039436 | 0.641887, 0.659670, 0.678702, 0.602575, 0.678831 |
| sae_8k_dual_ridge_s8k | 0.887382 ± 0.013085 | 0.893155, 0.899886, 0.872440, 0.882119, 0.889311 |

## Gates and self-tests

- seed 20260801: Gate A retained=236, both-correct=0.983, median d_gap=5.019; Gate B (both) E_resid/d_gap=0.826, sign=1.000; Gate D resid S=0.081.
- seed 20260802: Gate A retained=230, both-correct=0.958, median d_gap=5.101; Gate B (both) E_resid/d_gap=0.837, sign=1.000; Gate D resid S=0.087.
- seed 20260803: Gate A retained=235, both-correct=0.979, median d_gap=5.180; Gate B (both) E_resid/d_gap=0.826, sign=1.000; Gate D resid S=0.083.
- seed 20260804: Gate A retained=237, both-correct=0.988, median d_gap=5.011; Gate B (both) E_resid/d_gap=0.831, sign=1.000; Gate D resid S=0.085.
- seed 20260805: Gate A retained=234, both-correct=0.975, median d_gap=4.962; Gate B (both) E_resid/d_gap=0.815, sign=1.000; Gate D resid S=0.080.
- Self-tests: zero-selection bitwise=True; D-rescale bitwise=True; start_at_layer=8 max_abs=0; prompt-swap max_abs=5.72e-06.

This record measures additive-difference write-back and coordinate concentration in a given basis; it neither measures nor claims damage, degradation, removal, or loss of model computation caused by the SAE.

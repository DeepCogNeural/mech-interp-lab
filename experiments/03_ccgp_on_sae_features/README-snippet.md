## Experiment 03 — SAE features on the shattering × CCGP plane

![Five-seed overall shattering dimensionality versus main-effect CCGP comparison of GPT-2 residual, SAE, reconstruction, sparse random-expansion, and effective-width controls](figures/01_shattering_vs_ccgp.png)

Caption — the full-scale panel preserves 0.5 chance lines and the zoom panel makes the observed arm separation legible. The original sparse comparison gave `sae − rand_exp` two-way-XOR SD **+0.121 ± 0.011**, but SAE had more surviving directions. Effective-width matching shrinks rather than erases that result: widened random gives **+0.081 ± 0.006** and an exactly narrowed SAE gives **+0.075 ± 0.011**. The latter has no main-effect edge (**−0.001 ± 0.006**) but pays an L0 cost (44.97 versus 71.80).

Dense random expansion has the most shattering, exactly as Cover's theorem suggests when roughly 12,000 coordinates fire per sample; it is an upper reference, not the sparse control. Global-RMS scaling is primary because raw relative latent magnitudes are part of the SAE code; per-feature z-scoring manufactures a code the decoder never reads. This is a representation/lens measurement, not a model intervention: the SAE never replaces GPT-2's residual stream in the forward computation. Full gates, matching table, dichotomy breakdown, sensitivity analysis, and caveats: [Experiment 03 writeup](writeup.md).

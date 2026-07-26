## Experiment 03 — SAE features on the shattering × CCGP plane

![Five-seed overall shattering dimensionality versus main-effect CCGP comparison of GPT-2 residual, SAE, reconstruction, and matched random-expansion codes](figures/01_shattering_vs_ccgp.png)

Caption — overall CCGP was degenerate at ~0.5, so the x-axis is main-effect CCGP. Under the global-RMS, nested-L2 probe, SAE has overall SD **0.888 ± 0.004** versus **0.809 ± 0.006** for matched sparse random expansion (paired **+0.079 ± 0.006**); main-effect CCGP is **0.918 ± 0.006** versus **0.909 ± 0.008** (paired **+0.009 ± 0.013**).

The old per-feature z-score ordering reverses under global scaling, so it is sensitivity analysis rather than the headline. A small main-effect SD mismatch remains, so the two-way-XOR edge is not a clean causal comparison. This is a representation/lens measurement, not a model intervention: the SAE never replaces GPT-2's residual stream in the forward computation. Full gates, dichotomy breakdown, controls, and caveats: [Experiment 03 writeup](writeup.md).

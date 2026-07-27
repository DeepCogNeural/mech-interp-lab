# Experiment 04 全运行记录

- 状态：`completed`。墙钟：1760.0 秒。
- 控制臂分支：DESIGN.md Amendment 1 的冻结 Rule 2；主比较为 `AUC(sae) - AUC(pca)`，`rand_exp` 只透明报告、未裁决。
- 主区间：+0.303897 ± 0.023284（t(4) 95%）；冻结分支：`inconclusive: SAE Gate C did not pass in every seed`。

## 每个基的 AUC 与 k50

| basis | AUC(top-k), t(4) 95% | k50（五个种子） |
|---|---:|---|
| sae | 0.517126 ± 0.002031 | 16, 16, 16, 16, 16 |
| pca | 0.213229 ± 0.022783 | 64, 128, not reached within grid, 128, 64 |
| neuron | 0.156661 ± 0.005198 | not reached within grid, not reached within grid, not reached within grid, not reached within grid, not reached within grid |
| rand_exp | 0.178859 ± 0.023904 | not reached within grid, not reached within grid, not reached within grid, not reached within grid, not reached within grid |

- 非裁决 SAE 排名参照：`AUC(sae_topk) - AUC(sae_randk)` = +0.349701 ± 0.015546。

## 五种子写回信实度 E(full)/E(resid)

| measurement | mean ± t(4) 95% | per-seed |
|---|---:|---|
| sae_trained_decoder | 0.693941 ± 0.013875 | 0.704633, 0.707189, 0.685405, 0.683218, 0.689257 |
| rand_exp_8k_dual_ridge | 0.652333 ± 0.039436 | 0.641887, 0.659670, 0.678702, 0.602575, 0.678831 |
| sae_8k_dual_ridge_s8k | 0.887382 ± 0.013085 | 0.893155, 0.899886, 0.872440, 0.882119, 0.889311 |

## Gate 与自测

- seed 20260801: Gate A retained=236, both-correct=0.983, median d_gap=5.019; Gate B (both) E_resid/d_gap=0.826, sign=1.000; Gate D resid S=0.081。
- seed 20260802: Gate A retained=230, both-correct=0.958, median d_gap=5.101; Gate B (both) E_resid/d_gap=0.837, sign=1.000; Gate D resid S=0.087。
- seed 20260803: Gate A retained=235, both-correct=0.979, median d_gap=5.180; Gate B (both) E_resid/d_gap=0.826, sign=1.000; Gate D resid S=0.083。
- seed 20260804: Gate A retained=237, both-correct=0.988, median d_gap=5.011; Gate B (both) E_resid/d_gap=0.831, sign=1.000; Gate D resid S=0.085。
- seed 20260805: Gate A retained=234, both-correct=0.975, median d_gap=4.962; Gate B (both) E_resid/d_gap=0.815, sign=1.000; Gate D resid S=0.080。
- 自测：zero-selection bitwise=True; D-rescale bitwise=True; start_at_layer=8 max_abs=0; prompt-swap max_abs=5.72e-06。

本记录只陈述给定坐标基中的加性差分写回与坐标集中测量；它不测量、也不声称 SAE 对模型任何计算有损害、降级、移除或损失。

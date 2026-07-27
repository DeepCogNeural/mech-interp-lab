# Experiment 04 Amendment 2 稳健性记录

- 标签：`specified after unblinding of the primary, before its own computation; never adjudicating`
- 墙钟：1344.0 秒。仅写入本文件、`robustness.py`、`robustness_results.json` 与 `figures/03_robustness.png`。
- 本记录只描述给定坐标代码在一次加性残差差分写回下的测量；不对 SAE 是否改变模型任何计算作出表述。

## R0：由原 manifest 复算

- 名词不相交 SAE k=16 合并：R=0.5936296064503066，有向编辑数=150。
- 五种子 top-16 交集：12；平均两两交集=13.300。
- k=16 符号一致性详见 JSON 的 `sign_consistency_at_k16`（每个基、每个种子和合并分子分母均保存）。

## R1：PCA 拟合预算

- 2048 tokens：within AUC=0.195118 ± 0.007897；absolute AUC=0.195118 ± 0.007897。
- 8192 tokens：within AUC=0.213229 ± 0.022783；absolute AUC=0.213229 ± 0.022783。
- 8192−2048 within AUC：+0.018111 ± 0.025066。
- 8192−2048 absolute AUC：+0.018111 ± 0.025066。
- 768 个完整 PCA 成分在两个预算下都捕获约 100% 方差；top-64 的种子均值为 0.529544（2,048）和 0.461720（8,192）；底部一半特征值的方差份额为 0.071230 和 0.127718。
- Amendment 2 没有定义“平坦”的数值阈值，因此只记录测量到的变化与 t(4) 区间，不作该分类裁定。

## R2：sae_ridge

- mixed_generic_plus_rank_templates：Gate C E(full)/E(resid)=0.887382 ± 0.013085；AUC within=0.561495 ± 0.019843；absolute=0.498168 ± 0.013676。
- mixed_generic_plus_rank_templates 的 sae_ridge−PCA(8192) robustness 差：within=+0.348266 ± 0.037148；absolute=+0.284939 ± 0.030798。这不是裁定。
- pure_generic：Gate C E(full)/E(resid)=0.413714 ± 0.060752；AUC within=0.601839 ± 0.066887；absolute=0.250348 ± 0.057485。
- pure_generic 的 sae_ridge−PCA(8192) robustness 差：within=+0.388609 ± 0.081917；absolute=+0.037119 ± 0.075616。这不是裁定。

## Amendment 留下的开放决定

- R1 未指定把预算曲线称作平坦所需的阈值。
- `run_results.json` 只保存了通用池的生成元数据、未保存激活张量；因此每个种子按已记录的固定种子重建一次 8,192-token 池，R1 的两个预算和 R2 在内存中共用该同一池。
- R2 只有纯通用覆盖低于 0.95 时才运行混合版本；每个种子的实际选择和未运行项在 JSON 中显式列出。

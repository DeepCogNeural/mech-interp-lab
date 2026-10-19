# Compact evidence packet

Start with the CSV tables. JSON files retain the per-seed/group statistics, rules, and random-set summaries behind them. No activations are included. H11 includes losslessly compressed raw prompts and outputs; earlier experiments retain compact summaries. `h7_invalid.json` preserves the small set of failed selected-head outputs needed to explain the validity failure.

Rates are fractions in the data and percentages in figures. Counts and denominators remain explicit. All synthetic conditions have 64 items per seed/direction. The four real-filing groups also have 64 clean-correct items per direction under H10b. Seeds/groups are equally weighted. Invalid answers count as failures. Read-out attention is a share of all attended tokens, not a share conditional on the displayed spans.

| Question / figure | Readable table | Additional evidence |
|---|---|---|
| H5 behaviour | [h5_per_seed.csv](h5_per_seed.csv) | [h5.json](h5.json), [27B behaviour only](behaviour_27b.json) |
| (a) Full-residual scan | [residual_per_seed.csv](residual_per_seed.csv) | [bounds and per-seed rates](residual_scan.json) |
| (b) Single heads | [heads_top20.csv](heads_top20.csv) | [all 768 head averages](head_ranking.json), [all seed/direction effects](head_scan.json) |
| Joint-k / frozen eight | [joint_k_per_seed.csv](joint_k_per_seed.csv) | [criteria](joint_k.json), [frozen selection](frozen_selection.json) |
| (c) H7 | [h7_per_seed.csv](h7_per_seed.csv) | [verdicts](h7.json), `h7_sets_<seed>.json` (101 sets each), [invalid outputs](h7_invalid.json) |
| Necessity | [necessity_per_seed.csv](necessity_per_seed.csv) | [all 100 random-set mean drops](necessity.json) |
| (d) Both synthetic controls | [controls_per_seed.csv](controls_per_seed.csv) | [control summaries](controls.json) |
| H9 forecast transfer | [h9_per_seed.csv](h9_per_seed.csv) | [controls / validity / rule](h9.json), [post-hoc first-integer counts](h9_secondary.json), `h9_sets_<seed>.json` (101 sets each) |
| Original H10 gate | — | [gate, formats and group manifest](h10.json) |
| (c,d) Real 10-K H10b | [h10b_per_group.csv](h10b_per_group.csv), [controls](h10b_controls.csv) | [new/old parsers](h10b.json), [all 101 sets × 4 groups](h10b_all_sets.json) |
| Fresh-seed replication of H7 and necessity | — | [counts, intervals, invalid outputs and raw hashes](replication.json) |
| (g) H11 role versus position | [both parsers and controls](h11_per_seed.csv) | [gate and decision](h11.json), `h11_sets_<seed>.json`, `h11_raw_*.json.gz` |
| (e) Attention and head read-outs | [attention_per_seed.csv](attention_per_seed.csv) | [question/number attention and single-head effects](attention_readouts.json), [direct logit attribution](direct_readouts.json) |
| Exploratory 8B | [exploratory8b_per_seed.csv](exploratory8b_per_seed.csv) | [dev scans and all 21 held-out sets](exploratory_8b.json) |
| (f) Cross-model overlap | [overlap_per_head.csv](overlap_per_head.csv) | [identity limits, overlap and correlation](overlap_4b_8b.json), [distillation source](distillation_citation.json) |

**Logit metric.** `d = logit(first digit of T) − logit(first digit of N)`; `Δd = patched d − clean d`. Donor-oriented effects use `−Δd` for T→N and `+Δd` for N→T. Head ranking averages these effects across four dev seeds and both directions; it does not measure complete-answer flips for individual heads.

**Scoring boundaries.** H7 and H9 retain the original unique-integer rule. H7 fails; H9 meets its separate flip rule but has poor output validity. H10 stops at its failed gate. H10b accepts the first decimal/comma-formatted number and does not alter earlier verdicts. `frozen_unique_integer` fields use that old parser's clean-correct denominator; `frozen_on_h10b_denominator` fields use all 64 H10b items. These are descriptive comparisons. H11 retains the unique-integer parser for its decision and pre-registers the first-number parse as secondary. Both use all 64 items per seed/direction, with invalid answers retained. H11 does not revise earlier verdicts.

**Overlap reference.** Six shared heads out of eight gives a hypergeometric tail of `200881 / 1429702652400 = 1.40505439829e-7`, fixing eight marked heads in a pool of 128 and drawing eight uniformly. This is a uniform reference, not a layer-matched permutation test or evidence that trained models are independent. Spearman correlation uses average ranks for ties. Donor-oriented effects reverse T→N signs before averaging; mixing raw signed directions gives a different correlation, retained in the JSON.

**Provenance.** [checksums.sha256](checksums.sha256) contains executable local packet/figure hashes and comment lines binding each summary to its raw server-file hashes. [provenance.json](provenance.json) gives source sizes, paths relative to the `exp07/` data root, and CSV parent summaries. `sibling/` labels refer to the data root's parent. H11 gzip files decompress to the exact raw JSON bound in `h11.json:sources`; these include every selected/random/control output, clean answers, rendered stimuli and gate. Source hashes were measured at export; existing raw-hash bindings were checked. Historical raw-result hashes remain unchanged when public path strings are redacted. Source-file entries for public copies bind their current bytes. The 27B evidence in this repository is its saved behaviour summary; its original request outputs are not bundled or re-opened here.

The corresponding cleaned FREEZE snapshot was at development commit `ef30672576004dbc511bf05e4b89f3067d0ebcf2` (private history; see the repository README). The original private FREEZE had SHA-256 `7bf5698fcd4b501c1f130c7f860e04b99a8c093dc631f1556a37671c72698ee5`; those original bytes are not distributed in this copy. Public copies redact non-scientific text, including machine-specific paths and workflow wording; scientific choices and numerical results are otherwise unchanged. The public FREEZE and `frozen_selection.json` now bind the public source/design bytes; `frozen_selection.json` also binds the public FREEZE. Historical output records retain the original input hashes; these are not hashes of the public copies. It is not a byte-identical replacement for historical runner input. No frozen scientific choices or numerical results were changed. The exploratory runner keeps the fixed compute window of the original run and its 90-minute minimum.

From the repository root:

```bash
python experiments/07_numeric_selection/post/make_figures.py
python experiments/07_numeric_selection/post/make_tables.py
(cd experiments/07_numeric_selection/results && shasum -a 256 -c checksums.sha256)
```

With read access to the original server, `post/export_packet.py --host HOST --remote-root 'workspace/exp07'` rebuilds the JSON packet through read-only SSH. For H11, copy `stage6/h11/packet/` over SSH, run `post/import_h11_packet.py PACKET_DIRECTORY`, then run `make_tables.py`. The H11 importer verifies decompressed source hashes and preserves the earlier provenance map. Full model reruns need the historical source snapshot, matching off-Git data/dev means, cached model revisions, and the environment used by the registered scripts. The packet reproduces tables and figures; it does not make an end-to-end GPU replication claim.

H11 execution uses `post/run_h11.sh` after the registered addendum and implementation are committed. Its historical execution requires the original private FREEZE bytes, which are not in the cleaned history. With those historical inputs, it evaluates all clean seeds before intervention, and runs at most four GPU workers. Figure (g) uses `h11.json` only.

# Graph Report - .  (2026-07-26)

## Corpus Check
- Corpus is ~11,322 words - fits in a single context window. You may not need a graph.

## Summary
- 106 nodes · 174 edges · 9 communities (8 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Probe Evaluation|Probe Evaluation]]
- [[_COMMUNITY_Experiment Runner|Experiment Runner]]
- [[_COMMUNITY_SAE and Random Codes|SAE and Random Codes]]
- [[_COMMUNITY_Dichotomy Breakdown|Dichotomy Breakdown]]
- [[_COMMUNITY_Stimuli and Residuals|Stimuli and Residuals]]
- [[_COMMUNITY_Headline Figure|Headline Figure]]
- [[_COMMUNITY_Gates and Pilot|Gates and Pilot]]
- [[_COMMUNITY_Result Artifacts|Result Artifacts]]
- [[_COMMUNITY_Representation Scope|Representation Scope]]

## God Nodes (most connected - your core abstractions)
1. `run()` - 23 edges
2. `Dichotomy breakdown chart` - 12 edges
3. `scale_features()` - 9 edges
4. `select_weight_decays()` - 9 edges
5. `sd_metric()` - 9 edges
6. `ccgp_metric()` - 9 edges
7. `gate_a()` - 9 edges
8. `Expressivity versus factor abstraction: SAE and matched random expansion` - 9 edges
9. `convergence_check()` - 8 edges
10. `build_stimuli()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `paired_sae_minus_rand()` --implements--> `SAE minus random two-way-XOR comparison`  [EXTRACTED]
  ccgp_sae.py → /Users/linghao/Github/mech-interp-lab/.claude/worktrees/serene-wing-ec635d/experiments/03_ccgp_on_sae_features/writeup.md
- `scale_features()` --implements--> `Per-feature z-score sensitivity analysis`  [EXTRACTED]
  ccgp_sae.py → /Users/linghao/Github/mech-interp-lab/.claude/worktrees/serene-wing-ec635d/experiments/03_ccgp_on_sae_features/writeup.md
- `representations()` --shares_data_with--> `Effective active-width caveat`  [EXTRACTED]
  ccgp_sae.py → /Users/linghao/Github/mech-interp-lab/.claude/worktrees/serene-wing-ec635d/experiments/03_ccgp_on_sae_features/writeup.md
- `gate_a()` --implements--> `Pre-execution gates A through C`  [EXTRACTED]
  ccgp_sae.py → /Users/linghao/Github/mech-interp-lab/.claude/worktrees/serene-wing-ec635d/experiments/03_ccgp_on_sae_features/ccgp_sae.py
- `folds()` --implements--> `Factorial final-token and item-disjoint probe control`  [EXTRACTED]
  ccgp_sae.py → /Users/linghao/Github/mech-interp-lab/.claude/worktrees/serene-wing-ec635d/experiments/03_ccgp_on_sae_features/ccgp_sae.py

## Hyperedges (group relationships)
- **Factorial item-disjoint probe protocol** — exp03_factorial_final_token_control, 03_ccgp_on_sae_features_ccgp_sae_build_stimuli, 03_ccgp_on_sae_features_ccgp_sae_folds, 03_ccgp_on_sae_features_ccgp_sae_ccgp_metric [EXTRACTED 1.00]
- **SAE versus sparse random-control comparison** — exp03_sparse_sae_code, exp03_matched_sparse_random_control, exp03_effective_active_width_caveat, exp03_two_way_xor_comparison [EXTRACTED 1.00]
- **Fair primary probe pipeline** — exp03_global_rms_primary_probe, exp03_nested_l2_selection, 03_ccgp_on_sae_features_ccgp_sae_sd_metric, 03_ccgp_on_sae_features_ccgp_sae_ccgp_metric [EXTRACTED 1.00]
- **Displayed representation arms** — fig01_arm_resid, fig01_arm_sae, fig01_arm_sae_recon, fig01_arm_rand_exp, fig01_arm_rand_exp_dense [EXTRACTED 1.00]

## Communities (9 total, 1 thin omitted)

### Community 0 - "Probe Evaluation"
Cohesion: 0.12
Nodes (25): balanced_accuracy(), ccgp_metric(), convergence_check(), fit_probe(), _fold_weight_decay(), folds(), inner_validation_split(), prepare_features() (+17 more)

### Community 1 - "Experiment Runner"
Cohesion: 0.17
Nodes (19): aggregate(), _canonical_mask(), ci95(), Config, device(), dichotomies(), legacy_fixed_standardise_reference(), load_model() (+11 more)

### Community 2 - "SAE and Random Codes"
Cohesion: 0.15
Nodes (15): explained_variance(), gate_a(), load_direct_res_jb(), random_expansion(), Fallback loader required by the experiment: no sae_lens dependency or API pins., Matched-width random ReLU expansion: SAE norm distribution + SAE bias + top-k L0, Load an exact residual SAE at layer 8 and ensure its real-text reconstruction is, representations() (+7 more)

### Community 3 - "Dichotomy Breakdown"
Cohesion: 0.15
Nodes (13): base-factor enumeration versus parity-family interactions, chance (0.5), Dichotomy breakdown chart, main effect, rand_exp, rand_exp_dense, resid, sae (+5 more)

### Community 4 - "Stimuli and Residuals"
Cohesion: 0.2
Nodes (10): build_stimuli(), collect_residuals(), _ids(), Tokenizer-interface shim for both fast and slow GPT-2 tokenizers., One grammatical cell.  number/tense/polarity use 0/1 coding., Make full-factorial lexical items and reject an item unless all eight lengths ma, Cache only requested residual hooks and select each sequence's final '.' positio, sentence() (+2 more)

### Community 5 - "Headline Figure"
Cohesion: 0.2
Nodes (10): rand_exp, rand_exp_dense, resid, sae, sae_recon, chance (0.5), 5 seeds; 95% CI; global-RMS probe, Expressivity versus factor abstraction: SAE and matched random expansion (+2 more)

### Community 6 - "Gates and Pilot"
Cohesion: 0.4
Nodes (5): choose_layer(), main_effect_pilot(), Legacy fixed-L2 preprocessing, retained for gates and sensitivity reporting., standardise(), Pre-execution gates A through C

### Community 7 - "Result Artifacts"
Cohesion: 0.4
Nodes (5): Dichotomy-family SD breakdown figure, Shattering versus main-effect CCGP figure, Experiment 03 README snippet, Experiment 03 results JSON, Experiment 03 writeup

## Knowledge Gaps
- **53 isolated node(s):** `Experiment 03 — shattering dimensionality and CCGP on a real GPT-2-small SAE.  T`, `Tokenizer-interface shim for both fast and slow GPT-2 tokenizers.`, `One grammatical cell.  number/tense/polarity use 0/1 coding.`, `Make full-factorial lexical items and reject an item unless all eight lengths ma`, `Fallback loader required by the experiment: no sae_lens dependency or API pins.` (+48 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run()` connect `Experiment Runner` to `Probe Evaluation`, `SAE and Random Codes`, `Stimuli and Residuals`, `Gates and Pilot`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `scale_features()` connect `Probe Evaluation` to `Experiment Runner`, `Gates and Pilot`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `random_expansion()` connect `SAE and Random Codes` to `Experiment Runner`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **What connects `Experiment 03 — shattering dimensionality and CCGP on a real GPT-2-small SAE.  T`, `Tokenizer-interface shim for both fast and slow GPT-2 tokenizers.`, `One grammatical cell.  number/tense/polarity use 0/1 coding.` to the rest of the system?**
  _53 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Probe Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
# H11 pre-registration: role versus number position

Date: 2026-10-18 (EDT). Written after the H7, H9 and H10b results were seen. Commit this addendum before any H11 model call. This new test does not revise any earlier verdict.

## Stimuli and clean gate

Use all 64 items of each held-out seed 20261201–20261208: pool B names and numbers, wording 2, paraphrase 2. Keep each item's company, N, T and questions unchanged. Reverse the two context sentences so that T comes first and N second:

`Management said it expects next-quarter revenue of about ${T} million. {Company} reported revenue of ${N} million last quarter.`

Use the original tokenizer, final `$` prefill, rendering assertions, Qwen/Qwen3-4B fp16 eager-attention harness, greedy decoding and eight-new-token limit. Save the original and swapped rendered stimuli and their hashes. Require clean Q_N accuracy >= 90% and clean Q_T accuracy >= 90%, each pooled over all eight seeds using the frozen unique-integer parser. Evaluate the complete pooled gate before any patch call. If either fails, report it and stop H11.

## Cross-order test

Patch the same eight frozen head outputs at the prefill position, before the output projection. The donor is the original-order prompt and the base is the swapped-order prompt of the same item. No head, seed or item selection is allowed.

| Direction | Donor | Swapped base | Role prediction | Position prediction |
|---|---|---|---|---|
| R1 | Q_N: N, first | Q_T: T, first | Switch to N, now second | Stay T, first |
| R2 | Q_T: T, second | Q_N: N, second | Switch to T, now first | Stay N, second |

For each direction, use the frozen eight and the same 100 frozen layer-matched random sets. Also apply the existing matched-norm random-vector construction: per selected head, add isotropic Gaussian noise with the norm of that original-donor minus swapped-base head delta; use seed + 20261105 and the existing head/item iteration order. Report validity and the base-value stay rate alongside the switch rate.

## Metrics and decision

The primary metric is role-consistent switch count / 64 for each seed and direction. Keep all items, including invalid answers, in the denominator. Score with the unchanged unique-integer parser. Report the first-number parse beside it: take the first signed numeric sequence; accept it only if it is a plain integer, ignoring later sequences. This secondary parse is specified here before H11, does not determine the verdict, and does not change H7 or H9 scoring. Validity means the parsed answer equals N or T. Also report clean-correct conditional rates with their denominators, descriptively.

- **role:** mean role-consistent switch rate >= 50% in both directions, and every seed in each direction strictly exceeds its own maximum over the 100 random sets.
- **position:** mean role-consistent switch rate <= 10% in both directions.
- **mixed:** all other completed outcomes.

Means weight the eight seeds equally. The gate-failed or incomplete state receives no role/position/mixed verdict.

## Same-order control and execution

Repeat the selected-eight intervention with both donor and base on the swapped contexts: swapped Q_N into swapped Q_T, and swapped Q_T into swapped Q_N. Both hypotheses predict a switch. Report the same metrics for this control; it checks whether these heads still work on the swapped template and does not alter the decision rule. The 100 random sets and matched-norm vector are required for the cross-order test; the same-order control uses the frozen selected eight.

Run on four server GPUs, at most one worker per GPU, with seeds in parallel in two waves. Use the existing finite-value checks and no-cache intervention semantics. Save raw outputs, including decoded strings and token IDs, and save a compact packet with per-seed/all-set summaries and hashes. No silent retries or changes after results. Record partial work if computation stops before completion.

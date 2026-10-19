# H10b pre-registration addendum

Date: 2026-10-18 (EDT). Written after the H10 behaviour gate failure and before any H10 intervention, and committed before any H10b model call.

H10 failed its frozen unique-integer behaviour gate: all 242 plain-integer questions were correct, while all 270 decimal or comma targets were invalid. No H10 selected-head intervention, random set, or control had run. H10b changes the full-answer parser to test whether the frozen head set transfers to the same real-filing stimuli. The H10 verdict remains unchanged.

- Stimuli: exactly the existing H10 manifest (SHA-256 `4b49b2aaaa4716d203880ef48a5fa8a03c9a26685d3263c1635d1bd0d3019eb3`), all 256 items, and the same four 64-item groups. Reuse the saved token IDs and prompts byte for byte; no extraction, item filtering, or reselection.
- Model and intervention: Qwen/Qwen3-4B, fp16, the same frozen eight heads, the same 100 frozen layer-matched random sets, and the same prefill head-output patches in both directions. Use the unchanged H10 generation (greedy, at most eight new tokens), batching, mismatched-donor permutation and matched-norm random-vector controls.
- Parser: take the first numeric token sequence: digits with optional correctly grouped thousands commas and an optional decimal part. Strip commas and compare using decimal arithmetic, exactly against each target at its stated decimal precision. Trailing zero differences are immaterial; do not truncate decimals or round a different number into a match. Ignore units and later numeric sequences. Validity means the parsed answer equals either candidate value. Store the original frozen unique-integer parse beside every new parse and summarize both.
- Candidate metric d is unchanged: logit(T first digit) minus logit(N first digit).
- Gate: clean accuracy >= 80% separately on both questions, each pooled over all four groups. If either fails, stop H10b interventions.
- Test denominator: each group/direction uses all items whose clean answer is correct under the H10b parser. Invalid intervention outputs count as failures.
- Pass: mean group flip >= 50% in both directions, and every group/direction strictly exceeds the maximum of its same 100 random sets.
- Controls: report mismatched-donor flip to the base own alternative separately from injection of either donor candidate; report matched-norm random-vector flip and validity. Keep seeds and construction identical to H10.
- The frozen unique-integer parse is descriptive alongside H10b. Report its own clean-correct denominator and also comparison on the H10b denominator, without changing any H7/H9/H10 output or verdict.
- All new outputs go under `workspace/exp07/stage4/`.

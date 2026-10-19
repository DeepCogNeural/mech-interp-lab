# H10 pre-registration addendum

Date: 2026-10-18 (EDT). Written after the H7/H9 held-out results were seen, and committed before any H10 model call.

**H10 — transfer of the frozen head set to period selection in real 10-K text.**
- **Stimuli.**
  - Use DocFinQA **train** contexts only, copied from compute-host `workspace/data/docfinqa/`; scp the train parquet/JSON or a filtered extract to compute-host.
  - Extract real sentences that state one line item with values for two different fiscal years. Example: "net revenues were $X million in 2018 compared with $Y million in 2017"; table rows rendered as one sentence also count.
  - Keep only items where:
    - both values parse as numbers with different leading digits and the same digit count;
    - the line-item name appears;
    - the context is ≤ 3 sentences (verbatim from the filing, no edits except whitespace).
  - Deterministic extraction rule, seed 20261018. Target 256 items, split into 4 groups of 64 by a hash of the company and document, so no document spans two groups. Hash the manifest.
- **Prompts.**
  - The user message is the context sentence(s), two newlines, then "What was {item} in {yearA}? Answer with the number only." or the same question with {yearB}.
  - Assistant prefill "$". Same rendering assert as DESIGN.md: the first token after "$" must be a single digit; drop and count items where it is not.
- **Behaviour gate.** Clean accuracy ≥ 80% on both questions, pooled. If it fails, report H10 as "not testable on real text with this model" and stop H10.
- **Test.**
  - Patch the frozen 8 heads' prefill outputs from the other-year question (same context) into the base, in both directions.
  - Primary flip rate per group, with the denominator = all items whose clean answer was correct. Invalid outputs count as failures.
  - Run the same 100 frozen random sets, the mismatched-donor control (a different item's other-year question; report base-alternative flip vs donor-number injection) and the matched-norm random-vector control.
- **Pass rule.** Mean flip ≥ 50% in both directions, and every group above the maximum of its 100 random sets.
- **Secondary parse (post hoc, labelled post hoc).** For H7, H9 and H10, also report a "first integer" parse alongside the frozen unique-integer parse, using the saved outputs only. It never changes a verdict.

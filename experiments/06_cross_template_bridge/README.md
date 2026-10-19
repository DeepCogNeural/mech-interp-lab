# Experiment 06 — fixed-object cross-template bridge

**Status: result-blind protocol and implementation; unrun and not independently verified. No Experiment 06 scientific result exists.**

Experiment 06 asks whether the narrow causal object isolated in Experiment 05 appears again on a
relative-clause template. It deliberately keeps the model, SAE, upstream head, intervention timing,
target span, and matched-span controls fixed. This is not a one-factor prompt-only contrast: Experiment
05's exploratory bridge used seed-drawn different-noun source-A controls, while Experiment 06 uses the
globally fixed cyclic source-A map. A result can therefore support reappearance under
a second prompt/control construction, but cannot isolate template-family change as the sole cause.

The registered template is:

```text
The SUBJECT that the ATTRACTOR often RELVERB ___
```

This family was already used by Experiment 05 as a calibration/source-C control. It is therefore a
**mechanism-held-out evaluation on a calibration-exposed template family**, not an unseen-template test
and not evidence of task-wide or model-wide generalisation.

The outcome is staged:

1. establish that the fixed L7H4 frozen-pattern subject-value intervention supplies a usable
   true-minus-fixed-source-A causal handle at `blocks.8.hook_resid_pre` on this family;
2. only if that within-family mechanism gate passes, compare the fixed 12-row SAE span with the 100 frozen,
   target-excluded, rank-12 matched sets using the second-largest raw-effect edge;
3. report a mechanism-negative, span-negative, positive, or non-estimable result without changing the
   registered objects or selecting seeds from their outcomes.

The [design](DESIGN.md) states the claim boundary and the [machine-readable protocol](protocol_v1.json)
becomes the frozen pre-run contract when first included in a clean source commit; the result-blind
[runner](run_experiment.py) enforces it. A separate
[model-free packet generator](make_public_results.py) can only package a complete, hash-bound raw
result. Its internal recomputation rebuilds the matched grid, seed estimands, t intervals, decision branch, and
provenance checks, and it requires the original Q4 JSON to compare every target/matched latent ID
against the frozen hash-bound sets; any inconsistency stops packaging. It cannot change the verdict and refuses to
overwrite an existing packet directory. The runner cannot execute until the protocol is in a clean
committed source tree. Any eventual model run must bind the off-Git Q4
input by SHA-256, write outside the source tree, and preserve all compact seed and matched rows needed
for model-free reaggregation.

The runner accepts only this directory's tracked `protocol_v1.json` and exact-compares the entire parsed
contract with a second frozen literal in the runner. An external path or any registered-field drift is a
fail-closed protocol error.

The protocol also freezes the exact GPT-2 and SAE repository revisions, every model/tokenizer/SAE
file hash used by the loaders, and the relevant runtime versions. The runner copies the exact
hash-validated cached bytes into a private staging directory before loading, records model-state and
SAE-decoder fingerprints before and after the run, and stops if either state changes. The public
packager performs an internal consistency check on that receipt and publishes only a fully staged packet.

The [design decisions](DESIGN_DECISIONS.md) record the scientific choices and their reasons.

No Experiment 06 model run or scientific experiment has been executed. Model-free contract checks do
not substitute for a model result or independent validation.

Exp06 reuses Q4's fixed target/matched latent sets and second-largest tail-order statistic. Its verdict
uses raw signed-logit effects, not Q4's normalized `R` estimand.

The public protocol removes internal review metadata and uses the matching neutral contract ID in the runner; scientific values, thresholds, and conclusions are unchanged.

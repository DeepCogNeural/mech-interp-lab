# Experiment 06 result-blind protocol — fixed-object cross-template bridge

**State on 2026-08-13:** result-blind design reviewed interactively by an AI advisor (not external
expert validation) after two required revisions and one final implementation-boundary correction.
No Experiment 06 model output has been observed. The first clean source commit containing these files
freezes the pre-run protocol; until then they are the reviewed result-blind candidate. A separate
final-SHA review receipt is still required before the runner may execute it.

## 1. Question

Experiment 05 found that, under one registered intervention on GPT-2-small, a fixed 12-dimensional
layer-8 SAE decoder span carried a large part of an L7H4-induced `resid_pre8` effect and beat frozen
matched spans. Experiment 06 asks a narrower transfer question:

> On a relative-clause agreement template with a globally fixed cyclic source-A control, does the
> already-fixed L7H4 intervention produce a usable causal handle, and, if it does, does the already-fixed
> 12-row span retain an advantage over the already-fixed matched edge?

This is not a new search. It cannot select a different head, layer, hook, position set, latent set,
matched set, seed, threshold, or template after seeing an Experiment 06 outcome.

## 2. Prior exposure and claim boundary

The registered family comes from `make_source_c_relative_clause(..., with_adverb=True)` in Experiment
05 calibration:

```text
The SUBJECT that the ATTRACTOR often RELVERB ___
```

Because this family already appeared in calibration/source-C, the evaluation must be called a
**mechanism-held-out evaluation on a calibration-exposed template family**. It may not be called
transfer to a fully unseen distribution, independent external validation, or broad syntactic
generalisation. The prior source-C use did not run the L7H4-to-`resid_pre8` bridge or select the
target/matched spans on this family, but it did expose the family during mechanism development.

No outcome licenses claims about natural latent semantics, monosemanticity, individual-latent
causality, necessity, sufficiency, mediation, a complete circuit, other models, other tasks, or broad
syntactic generalisation.

This is also not a one-factor template-only contrast. The Experiment 05 exploratory bridge used a
seed-drawn different-noun source-A construction; the Advisor required Experiment 06 to replace that
degree of freedom with one global cyclic lemma map. Thus a positive result would show that the fixed
objects reappear under a second prompt/control construction. It would not isolate the template-family
change from the changed source-A construction.

## 3. Fixed scientific objects

- model: GPT-2-small at repository revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`,
  with every required weight/config/tokenizer file hash frozen in `protocol_v1.json`;
- SAE: the res-jb layer-8 direct decoder at repository revision
  `57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9`, with weight/config file hashes frozen in
  `protocol_v1.json`, decoder width 768;
- upstream handle: L7H4, zero-indexed layer 7 and head 4;
- intervention: the Q3 frozen-attention-pattern subject-value construction at L7H4's final query;
- capture hook: `blocks.8.hook_resid_pre`;
- edited positions at `resid_pre8`: matrix-subject and final positions;
- target: the same 12 decoder rows contained in the bound Q4 raw result;
- empirical null: the same 100 target-excluded matched rank-12 spans per ordinal Q4 seed;
- ordinal binding: sorted Q4 source-seed order maps one-to-one to new registered-seed order;
- new registered seeds: `20260822` through `20260829` inclusive;
- 240 generated pairs per seed, Gate A, then the first at most 150 retained pair ids.

The Q4 raw input must have byte SHA-256
`81a917da187a103a1e76d79ce86f672347d01110cd50c4a93a241302237671ac`. A hash mismatch,
schema mismatch, non-complete Q4 status, changed target ids, non-rank-12 set, overlap with the target,
missing draw, duplicate draw, or changed ordinal binding stops the run without a scientific verdict.

## 4. Stimuli and controls

For each seed, generate 240 paired prompts with the existing relative-clause-with-adverb builder.
The matrix subject is singular/plural. The attractor alternates number, and the relative-clause verb
agrees with that attractor. The blank asks GPT-2-small to continue with ` is` or ` are`.

Gate A is unchanged from the bridge:

- at least 0.60 of pairs must have both members signed correctly;
- at least 140 pairs must be retained;
- the median clean plural-minus-singular logit gap over all 240 pairs must be at least 1.0.

If Gate A passes, evaluate the first at most 150 retained pair ids in ascending order. Both directions
are included for every retained pair.

For each directed base item:

- **true source:** the opposite-number form of the same matrix-subject lemma;
- **source A:** the next lemma under the fixed cyclic mapping below, with the same grammatical number
  as the base;
- **held constant:** attractor, relative verb, adverb, template, token positions, and every non-subject
  token.

The mapping is the cyclic successor in this exact vetted order, including the final wraparound:

```text
cat → dog → child → student → teacher → artist → pilot → singer → doctor → farmer
→ driver → actor → chef → writer → friend → neighbor → visitor → player → dancer
→ reader → cat
```

The same no-fixed-point mapping applies to both number directions and all eight seeds. It is
outcome-independent and admits no implementation-time lemma choice. All subject forms must be single
tokens. Base, true, and source-A sequences must have equal lengths, and token/position assertions must
establish that only the intended matrix-subject token changed. Source A is a fixed same-number lexical
control for the inherited true-vs-A handle; it does not isolate a pure number effect or remove all
lexical-identity confounding.

## 5. Intervention

For each base item, use the base prompt's frozen L7H4 attention coefficient from the final query to the
matrix-subject position. Combine that coefficient with source-own `hook_v` at the matrix subject to
construct two L7H4 `hook_z@final` replacements:

- the true-source arm;
- the same-template source-A arm.

Run both arms on the unchanged base prompt and capture their complete states at
`blocks.8.hook_resid_pre`. Define `delta = resid_true - resid_A`. Causal masking implies this final-query
L7H4 edit must not alter non-final `resid_pre8` positions; the runner checks that identity and stops on
failure.

At the matrix-subject and final positions of the source-A `resid_pre8` state, add:

- full `delta`;
- the orthogonal projection of `delta` into the fixed 12-row decoder span;
- the orthogonal complement;
- each of the 100 frozen target-excluded matched rank-12 projections.

The full edit must reproduce the true arm's final logits within the registered numerical tolerance.
Experiment 06 does not clamp L8H5 or claim to test reader mediation.

## 6. Registered estimands

Let `d = logit(" are") - logit(" is")`. Let `sign=+1` for a singular-base to plural-true direction and
`sign=-1` for a plural-base to singular-true direction. For each technically valid seed `s`:

- `D_s = mean_i sign_i * (d_true_i - d_A_i)`: direct L7H4 true-minus-A handle;
- `T_s = mean_i sign_i * (d_target_i - d_A_i)`: fixed-target raw signed effect;
- `M_sj = mean_i sign_i * (d_matched_ij - d_A_i)`: raw signed effect for matched draw `j`;
- `E_s = second-largest_j(M_sj)`: the second-largest raw-effect edge among the 100 frozen Q4 matched
  latent sets;
- `A_s = T_s - E_s`: target advantage over that registered edge;
- complement and target-plus-complement closure: descriptive only. Closure is reported as the target
  plus complement raw-effect sum, its difference from the full raw effect, and their ratio when the
  full-effect denominator is safely nonzero.

Ratios such as `T_s / D_s` may be reported descriptively when their denominator is finite and safely
nonzero. No ratio enters an Experiment 06 verdict. Exp06 reuses Q4's fixed latent sets and
second-largest tail-order statistic; it does not reuse Q4's normalized `R` estimand.

## 7. Technical validity and outcome branches

A seed is scientifically adjudicable only if Gate A passes. No seed is included or excluded because of
the sign or magnitude of `D_s`, `T_s`, `M_sj`, or `A_s`.

All eight registered seeds must execute completely and pass Gate A. A Gate-A population failure in any
seed yields **NON_ESTIMABLE** and forbids a directional mechanism-negative or span-negative conclusion.
In contrast, any input-binding, artifact-rank, token/position, finiteness, timing-identity, or
full-rescue-identity guard failure yields **STOPPED** with no scientific verdict. Missing execution,
runtime-cap termination, or an unexpected exception is also `STOPPED`, not scientific negative
evidence. Only a complete, guard-valid, Gate-A-valid eight-seed set is adjudicated; every interval is a
two-sided 95% Student-t interval with `df=7`.

The first two result-blind Advisor passes returned `REVISE`; after the matched-edge, source-A,
eight-seed-coverage, joint-seed, and STOPPED-versus-NON_ESTIMABLE corrections, the final pass returned
`SECTION_3_PREREG_FINAL: APPROVE_DESIGN`. The accepted rule is:

1. **MECHANISM_TRANSFER:** the lower 95% t bound of `mean(D_s)` is above zero and at least 6 of the 8
   registered seeds have `D_s >= 0.05`. The absolute floor was chosen before this template was run and
   is less than half the minimum direct handle in the prior original-template bridge
   (`0.104946173...`).
2. If technical coverage passes but the mechanism gate fails, report **MECHANISM_NEGATIVE** and do not
   adjudicate the span.
3. Conditional on mechanism transfer, **POSITIVE** requires both:
   - the lower 95% t bound of `mean(T_s)` is above zero;
   - the lower 95% t bound of `mean(A_s)` is above zero;
   - at least 6/8 of the same registered seeds jointly have `T_s>0` and `A_s>0`.
4. If mechanism transfer passes but either span condition fails, report **SPAN_NEGATIVE**.

All eight seeds enter the span intervals, including any whose direct handle is below the absolute
floor. Complement, closure, normalized ratios, the maximum matched draw, and attractive individual
examples are descriptive and cannot change an outcome branch.

## 8. Artifacts and provenance

The model-backed runner must:

- require a clean source tree and an exact expected commit;
- accept only the tracked `experiments/06_cross_template_bridge/protocol_v1.json`, never an external
  protocol path, and compare its entire parsed object with an independent in-code frozen contract;
- require offline model and SAE loading;
- require the exact model/SAE repository revisions, file byte hashes, and runtime versions in the
  machine-readable asset contract; load from private copies of the exact hash-validated byte buffers;
- record and require equality of before/after model-state and SAE-decoder fingerprints;
- bind the Q4 raw artifact and this protocol by hash;
- emit `NON_ESTIMABLE` only for the registered Gate-A population failure; every implementation,
  binding, identity, finiteness, or execution failure must be `STOPPED` without a scientific verdict;
- write outside the source tree during the run;
- atomically reserve a never-before-used output path and refuse to overwrite any existing path,
  including a prior `COMPLETE`, `STOPPED`, or in-progress artifact;
- write a compact raw result containing per-seed metrics and all 800 matched rows for an adjudicable
  eight-seed run, but no activations;
- record the source commit, source-tree digest, input hashes, runtime, environment flags, retained pair
  ids, target ids, and ordinal source-Q4-seed binding;
- recheck source provenance at the end of the run.

A failed run replaces only its own newly created `RUNNING` tombstone with `STOPPED` and retains every
input hash, git fact, environment flag, completed seed row, and completed matched-row count known at the
failure point. A new attempt must use a new output path.

Any public packet must be generated model-free from the bound compact result and fully staged before
one atomic directory publication. The compact result,
not a plot or prose summary, is the source for reaggregation. A future review receipt must name the
exact source commit and artifact hashes it reviewed; absence of such a receipt must remain explicit.
Packet generation also requires the original hash-bound Q4 JSON and compares every target and matched
latent ID by ordinal and draw before emitting a public claim.
The current interactive Advisor decision has no checked-in cryptographic receipt. Before a model run,
the final-SHA review must name the clean commit and the canonical protocol hash recorded by the runner.

## 9. Outcome-specific public claims

- **NON_ESTIMABLE:** technical coverage was insufficient; no mechanism or span transfer conclusion.
- **MECHANISM_NEGATIVE:** under the registered relative-clause intervention, the fixed L7H4 handle did
  not meet the registered transfer criterion; the fixed span was not adjudicated.
- **SPAN_NEGATIVE:** the L7H4 handle transferred, but the fixed span did not meet the registered
  positive-effect and frozen-matched-edge criteria.
- **POSITIVE:** on a calibration-exposed relative-clause family not previously used for bridge
  adjudication, the fixed L7H4 true-vs-source-A handle retained a nontrivial effect across fresh lexical
  seeds and the fixed 12-row span stably exceeded the second-largest raw-effect edge among the 100
  frozen Q4 matched latent sets.

Even the positive branch is an intervention-level transfer claim for one model and one previously seen
control family. It is not a complete mechanism or broad generalisation claim.

The interactive AI-advisor-reviewed comparator wording is:

> Exp06 compares the target span's signed logit effect with the second-largest raw-effect edge among
> the 100 frozen Q4 matched latent sets; it reuses Q4's fixed sets and tail-order statistic, not Q4's
> normalized `R` estimand.

## 10. Deviations

Any change after the first model-backed Experiment 06 output exists must be appended as a dated,
non-adjudicating amendment. It may improve diagnostics or implementation correctness, but it cannot
retroactively change the registered outcome.

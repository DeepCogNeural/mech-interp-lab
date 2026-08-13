# Experiment 06 Advisor decision log

This is a durable summary of interactive AI-advisor design decisions, not external expert validation.
It is **not** a cryptographic review
receipt, and `protocol_v1.json` therefore keeps `advisor_review.receipt = null`. A separate final-SHA
review must name the committed source SHA and canonical protocol hash before a model-backed run.

No Experiment 06 output existed during any decision below.

## Direction

The interactive AI advisor accepted a fixed-object cross-template bridge as the next question after Experiment 05:
hold GPT-2-small, the SAE, L7H4, the fixed 12-row span, and the frozen matched latent sets constant;
move only to the relative-clause family previously used for calibration/source-C. The permitted label is
“mechanism-held-out evaluation on a calibration-exposed template family.”

That was the initial direction, not the final one-factor interpretation. The later AI-advisor-requested
source-A revision replaced Experiment 05's seed-drawn different-noun control with a global cyclic map.
The final protocol therefore evaluates a second prompt/control construction and cannot attribute any
difference solely to the template family.

## Preregistration pass 1 — `REVISE`

The first result-blind pass required four changes:

1. require a complete eight-seed set before any directional negative or positive verdict;
2. use the second-largest matched tail-order statistic rather than the maximum;
3. freeze one deterministic, no-fixed-point source-A lemma mapping for both directions and all seeds;
4. require the same at least 6/8 seeds jointly to have positive target effect and positive matched-edge
   advantage.

It accepted the mechanism gate of `LCB95(mean D_s) > 0` plus at least 6/8 `D_s >= 0.05`, provided the
prior-template basis for the floor was disclosed.

## Preregistration pass 2 — `REVISE`

The second pass accepted the matched edge, source-A mapping, joint span gate, and public claim boundary,
but required a sharper failure taxonomy:

- only the registered Gate-A population failure may yield `NON_ESTIMABLE`;
- missing execution or any binding, rank, token/position, finiteness, timing-identity,
  full-rescue-identity, runtime, or provenance failure must yield `STOPPED` with no scientific verdict.

## Final preregistration decision — `APPROVE_DESIGN`

After that split was implemented, the Advisor returned:

```text
SECTION_3_PREREG_FINAL: APPROVE_DESIGN
IMPLEMENTATION_SPLIT: ACCEPT
REQUIRED_CHANGE: NONE
```

## Comparator clarification — `APPROVE_CLARIFICATION`

An independent static audit then noted that Experiment 05 Q4's effect metric was normalized `R`, while
the accepted Experiment 06 rule uses raw signed-logit effects. The Advisor explicitly accepted the raw
rule and approved this exact wording:

> Exp06 compares the target span's signed logit effect with the second-largest raw-effect edge among
> the 100 frozen Q4 matched latent sets; it reuses Q4's fixed sets and tail-order statistic, not Q4's
> normalized `R` estimand.

## Implementation-contract review — `APPROVE_DESIGN`

After the runner and public-packet generator were drafted, an independent read-only adversarial audit
first identified gaps in full-protocol binding, Q4 latent-ID verification, output reservation, and
descriptive-field recomputation. Those gaps were repaired before any command or model run. The audit's
final response was:

```text
EXP06_STATIC_FINAL: ACCEPT
REMAINING: NONE
```

The Advisor then reviewed the complete implementation-contract summary and returned:

```text
SECTION_3_IMPLEMENTATION_VERDICT: APPROVE_DESIGN
SCIENCE_CONTRACT: ACCEPT
PROVENANCE: ACCEPT
PACKET_INTEGRITY: ACCEPT
CLAIM_BOUNDARY: ACCEPT
HARD_BLOCKER: NONE
```

This is an interactive AI-advisor design-level decision about the frozen contract and fail-closed implementation. It is not evidence
that the code has executed, that the packet generator accepts a real result, or that any scientific
outcome has been observed.

## Remaining review gate

Design approval is not a final artifact or final-SHA receipt. The remaining sequence is deliberately
ordered:

1. obtain explicit authorization and run the model-free contract, corruption, import, and synthetic
   packet checks plus the required static-artifact regenerations;
2. commit the passing frozen state, obtain separate push authorization, and return the pushed exact
   source SHA, parent, canonical protocol hash, changed paths, and check results to the Advisor;
3. only after that final-SHA receipt and explicit model-run authorization, execute one complete Exp06
   run and independently package its raw artifact;
4. return the actual raw/public hashes, eight-seed primitives, matched-grid coverage, runtime and final
   public claim for result-level review.

# Experiment 06 design decisions

## Direction

The fixed-object cross-template bridge follows Experiment 05:
hold GPT-2-small, the SAE, L7H4, the fixed 12-row span, and the frozen matched latent sets constant;
move only to the relative-clause family previously used for calibration/source-C. The permitted label is
“mechanism-held-out evaluation on a calibration-exposed template family.”

That was the initial direction, not the final one-factor interpretation. The later
source-A revision replaced Experiment 05's seed-drawn different-noun control with a global cyclic map.
The final protocol therefore evaluates a second prompt/control construction and cannot attribute any
difference solely to the template family.

## Coverage and comparator

The result-blind protocol fixes four choices:

1. require a complete eight-seed set before any directional negative or positive verdict;
2. use the second-largest matched tail-order statistic rather than the maximum;
3. freeze one deterministic, no-fixed-point source-A lemma mapping for both directions and all seeds;
4. require the same at least 6/8 seeds jointly to have positive target effect and positive matched-edge
   advantage.

The mechanism gate is `LCB95(mean D_s) > 0` plus at least 6/8 `D_s >= 0.05`, with the
prior-template basis for the floor disclosed.

## Failure taxonomy

Population failure and execution failure have different meanings:

- only the registered Gate-A population failure may yield `NON_ESTIMABLE`;
- missing execution or any binding, rank, token/position, finiteness, timing-identity,
  full-rescue-identity, runtime, or provenance failure must yield `STOPPED` with no scientific verdict.

## Raw-effect comparator

Exp06 compares the target span's signed logit effect with the second-largest raw-effect edge among
the 100 frozen Q4 matched latent sets. It reuses Q4's fixed sets and tail-order statistic, not Q4's
normalized `R` estimand.

## Implementation boundaries

The implementation binds the full protocol, verifies the Q4 latent IDs, reserves outputs atomically,
and recomputes descriptive fields. These guards address input drift and inconsistent packets.
No Experiment 06 output existed during these decisions. The design and implementation are not
independently verified, and no scientific outcome has been observed.

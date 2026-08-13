# Research brief — from an interpretable span to a causal subspace

This is the short public account of Experiment 05. It is about a deliberately narrow question in
GPT-2-small, not a claim that a 12-dimensional SAE span is the model's ontology or that two heads are
the whole circuit.

## Question

Can a representation-level object survive a causal test that connects an upstream mechanism to a
downstream state and the model's own readout?

The motivating tension is familiar in interpretability: a sparse feature dictionary can make a state
easy to name, but naming a direction is not the same as showing that the model uses it. I therefore
separated three claims that are often collapsed: a head can move the behavior; a fixed subspace can
carry an induced effect; and that subspace is a native, necessary, or monosemantic representation.

## Evidence ladder

1. **Mechanism handle.** Across eight seeds, L7H4 and L8H5 were the minimum tested compact set. They
   passed the preregistered number-specificity controls and the frozen-pattern subject-value transport
   test in all 8/8 seeds.
2. **Span-level concentration.** The fixed 12-row layer-8 SAE decoder span retained
   `R_span = 0.8935` of the directed two-head logit effect (t(7) CI `[0.8905, 0.8966]`) and beat the
   frozen matched-span edge in every seed. The complementary ratio was `R_comp = 0.0848`. The
   geometric squared-norm fraction was only about `0.525–0.544`, so this is directed-effect
   concentration, not 89% activation reconstruction.
3. **Causal bridge.** On eight fresh held-out seeds (`20260814–20260821`), the same span recovered
   `R_target = 0.678639` (95% t(7) CI `[0.673841, 0.683437]`) of the directed L7H4→`resid_pre8`
   effect and exceeded the actual maximum of 100 target-excluded matched rank-12 spans on every seed.
   The complement was `0.305339` (CI `[0.301260, 0.309418]`), while the aggregate matched-span mean
   was `0.065910` (CI `[0.049408, 0.082411]`). With L8H5's complete `hook_z` output at the final
   query position overwritten by the natural source-A baseline, the target remained `0.674047` (CI
   `[0.669586, 0.678508]`). This is a post-attention-aggregation head-output clamp, not value-only.

## What changed my mind

Q4 by itself could have been a stable, task-aligned subspace selected by the later intervention. The
bridge changed the useful claim: the span is not merely present at the endpoint; under this tested
intervention it carries a reproducible fraction of an upstream head-induced effect. The fact that the
result stays near 67% under the complete L8H5 `hook_z@final` clamp weakens a simple
dominant-L8H5-final-output story. It does not prove a head→span→readout mediation path.

The important number is therefore not “89%.” It is the combination of a fresh bridge, matched-span
control, and a clamp that leaves most of the effect intact while making the remaining uncertainty
explicit.

## Strongest alternative and honest null

The strongest alternative is that the decoder span is a reproducible task-aligned causal subspace,
without its rows being native semantic variables. The matched spans are a useful empirical null and
lose in all eight seeds, but they do not turn the target into a privileged ontology. The reader
projection coefficient is only descriptive (`0.027–0.124` across seeds). At the final query, the clamp
overwrites the complete result of the QK/pattern-weighted value aggregation; it does not isolate QK from
V. Other L8H5 query positions and parallel downstream routes remain untested. The near-closure of target
plus complement (`≈0.984`) belongs to a separate nonlinear arm and is not a linear attribution.

Accordingly, this brief does **not** claim natural or monosemantic latent semantics, individual-latent
causality, necessity, sufficiency, dominant L8H5 mediation, a complete circuit, or generalisation
across models or tasks. The bridge is exploratory and has no preregistered verdict.

## Why this matters for interpretable representations

It is possible for a representation to be causally useful without being a list of independent,
human-named features. That distinction matters for safety-facing interpretability: a compact subspace
can be a good intervention handle while still being a poor explanation of what the model computes.
The experiment makes that boundary measurable rather than rhetorical—effect concentration, matched
nulls, and route controls answer different questions.

The research quality signal is the willingness to let those questions separate. A positive bridge earns
a narrower causal statement; it does not upgrade itself into a complete circuit. A failed clamp or a
negative matched comparison would have changed the story. The checked-in compact rows keep the reported
summaries reaggregable; reproducing the full packet still requires the hash-bound raw result outside Git.

## Reproduce and inspect

- [Main public evidence packet](results/RESULTS.md), including Q1–Q4 compact tables and checksums.
- [Bridge summary](results/bridge_result_summary.json), [seed metrics](results/bridge_seed_metrics.csv),
  and [800 matched-span rows](results/bridge_matched_ratios.csv).
- [Main evidence figure](results/figure_exp05_main.svg) and [bridge figure](results/figure_bridge_rescue.svg)
  ([PNG](results/figure_bridge_rescue.png)).
- [Bridge implementation](bridge_rescue.py) and [compact-packet generator](make_bridge_summary.py).
- From the repository root, the model-free contract suite is:

  ```bash
  ./.venv/bin/python -m unittest discover \
    -s experiments/05_number_agreement_circuit/tests -p 'test_*.py'
  ```

The original raw-dependent bridge packet reported clean source `0d7c4db`, 150 pairs per seed, and 800
frozen matched draws. The current compact reaggregation attributes those fields through a hash-pinned
historical receipt but did not revalidate the missing raw input. The raw model output remains outside
Git; its SHA-256 begins `9d8446`. The compact packet contains no bridge-specific independent review
receipt.

## Next frozen question

The proposed all-position-clamp factorial was withdrawn as non-identifying. The upstream L7H4
intervention changes only the final position at `resid_pre8`; causal masking means earlier L8H5 query
outputs cannot reveal a route from that edit, so extending the same clamp across positions supplies no
new identifying contrast.

The next experiment instead keeps GPT-2-small, the SAE, L7H4, the frozen 12-row span, and the 100
target-excluded matched spans fixed, then evaluates them on the relative-clause family already used in
calibration/source-C control under a globally fixed cyclic source-A map. Because the earlier bridge used
seed-drawn source-A nouns, this is not a one-factor template-only contrast. It first asks whether the
L7H4 causal handle meets its within-family gate; only if it does
will the fixed span be compared with the second-largest raw-effect edge among the 100 frozen Q4 matched
latent sets. This reuses Q4's sets and tail-order statistic, not its normalized `R` estimand. The
[Experiment 06 design](../06_cross_template_bridge/DESIGN.md) was reviewed interactively by an AI
advisor (not an external expert), is implemented, and remains unrun. It is a mechanism-held-out evaluation on a calibration-exposed template family, not a completely
unseen-template claim, and no additional result is claimed here.

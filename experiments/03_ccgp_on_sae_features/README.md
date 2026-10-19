# Experiment 03 — SAE decoding and probe scaling

See the [writeup](writeup.md), [probe code](ccgp_sae.py), [results](results.json), and
[convergence follow-up](convergence_results.json).

GPT-2-small layer 8 and the published res-jb SAE are compared with a random expansion matched
in width, column norm, and L0, using NUMBER × TENSE × POLARITY at an identical sentence-final period.
The SAE-minus-random two-way interaction contrast is `+0.121 ± 0.015` under one probe convention
and `+0.011 ± 0.022` under another. These conventions are invertible affine reparameterisations,
so this comparison remains **not adjudicated**. The width controls (`+0.081 ± 0.008`, `+0.075 ± 0.016`)
ran only under the convention producing an effect and inherit that limitation.

The two dense 768-dimensional arms are stable (0.902 vs 0.901; 0.877 vs 0.877), while sparse or wide
arms are sensitive to scaling. No intervention on the model's forward pass is performed here.

Public copies redact machine-specific paths; no scientific values, thresholds, or conclusions changed.
The temporary solver-source location was omitted because that source is not a reproducible repository file.

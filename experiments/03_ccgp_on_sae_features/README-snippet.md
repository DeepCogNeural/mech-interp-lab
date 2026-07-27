## Experiment 03 — a real SAE on the shattering × CCGP plane, and a result I had to give back

[Full writeup](experiments/03_ccgp_on_sae_features/writeup.md) · [code](experiments/03_ccgp_on_sae_features/ccgp_sae.py) · [per-seed results](experiments/03_ccgp_on_sae_features/results.json)

Experiment 02's caveats named this one twice: XOR accuracy is a task-specific proxy where shattering
dimensionality / CCGP (Bernardi et al. 2020) is the task-agnostic measure, and the whole thing was a toy
model. So: GPT-2-small layer 8, the published res-jb SAE, a full factorial NUMBER × TENSE × POLARITY read
at a sentence-final `.` that is byte-identical across all eight conditions. The comparison that matters is
not SAE versus residual stream — a 768 → 24,576 ReLU expansion wins that by Cover's theorem — but **SAE
versus a random expansion matched in width, column norm, and L0.**

![Shattering dimensionality against main-effect CCGP for seven arms, full scale with chance lines and a zoomed panel](experiments/03_ccgp_on_sae_features/figures/01_shattering_vs_ccgp.png)

Under one probe convention the SAE reads two-way interactions much better than matched random mixing
(`+0.121 ± 0.015`); under another it does not (`+0.011 ± 0.022`). Those two conventions are an **invertible
affine reparameterisation** of a linear probe's input, so the swing measures the probe's inductive bias,
not the codes. **I report that as not adjudicated** rather than shipping the flattering setting — and the
width-matched follow-ups that survived (`+0.081 ± 0.008`, `+0.075 ± 0.016`) inherit the same question,
because they only ever ran under the convention that produces an effect.

What does survive is a methodological result, and it is the part I'd defend: the sensitivity is
**localised**. The two dense 768-dimensional arms are unmoved by the scaling (0.902 vs 0.901; 0.877 vs
0.877) while every sparse or very wide arm swings hard — so preprocessing that is harmless for dense
representations is decisive for sparse over-complete ones, and any SAE-versus-baseline decoding comparison
that doesn't state its scaling convention is under-determined.

**Scope:** an SAE is a read-out lens beside the residual stream. Nothing here substitutes SAE features into
the forward pass, ablates them, or measures model behaviour — so none of it says an SAE harms a model's
computation.

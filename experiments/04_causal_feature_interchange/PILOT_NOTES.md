# Experiment 04 Pilot Notes

- Wall-clock time: 42.3 seconds.
- Status: `go_fallback`.
- Scope: pilot steps 1--7 only; no full experiment was run.
- Step 1: `PASS` — CPU float32 model and layer-8 SAE loaded; W_dec=(24576, 768); six readout/control strings are one leading-space token.
- Step 2: `PASS` — 2a zero bitwise=True; 2b power-of-two diagonal bitwise=True; 2c max abs=0.
- Step 3: `PASS` — both-correct=0.967; median d_gap=5.312; pilot pairs=60.
- Step 4: `PASS` — max |d(patched)-d(source)|=1.24e-05 across 120 directed edits.
- Step 5: `PASS` — selected layer-8 position set=both; scan recorded for layers (4, 6, 8, 10).
- Step 6: `PASS_FALLBACK` — random basis outside [0.70,1.30] for both decoder variants; within-SAE fallback
- Step 7: `PASS` — 16 candidates scored per basis; R(top-{1,4,8}) measured; full-run extrapolation=1.16 minutes.
- Gate A measurements: both-correct=0.967; median d_gap=5.312; pilot retained pairs=60.
- Gate B layer-8 E_resid/d_gap (sign consistency): subject=0.398 (1.000), final=0.406 (0.992), both=0.840 (1.000); selected=both.
- Gate C E(full)/E_resid: SAE trained=0.731; random dual-ridge=0.588; random tied=0.298; dual-ridge coverage=0.972.
- Patched-forward timing: 0.264 s/call; full-run extrapolation=1.16 minutes; trim order fired=[].

## Recorded implementation decisions

- DESIGN.md's explicit Pilot step 3 count (60 pairs) is treated as pilot-specific. The Gate A table's >=140 retained-pair threshold is recorded as a full-run threshold because it is mechanically impossible for a 60-pair pilot.
- The random dual ridge maps random codes directly to residual x as written in DESIGN.md's formula; decoder biases cancel from every additive difference edit.
- If dual-ridge fails Gate C but tied-weight passes, the manifest records the under-specified branch and selects the passing tied decoder for a follow-up rather than fabricating a ridge result.

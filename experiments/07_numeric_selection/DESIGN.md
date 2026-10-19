# Numeric selection: design draft

Status: frozen design (2026-10-18, see the final section). No discovery or held-out model run had occurred at the freeze.

## Question and scope

Which attention heads cause Qwen3-4B to choose one of two numbers supplied in a context? The context contains last-quarter revenue N and next-quarter guidance T. Changing the question changes the correct number. We will replace selected internal outputs between the two question conditions, then test the frozen head set on new names, numbers, and wording.

Primary model: `Qwen/Qwen3-4B`, fp16, eager attention, one TITAN V per worker, plain PyTorch/Hugging Face hooks, inference mode, `use_cache=False`. Qwen3-1.7B is used only for harness validation. No Base model, Qwen3-8B, probes, or head/site discovery runs belong to stage 1. Qwen3.8-27B-UD-Q4_K_M is **behaviour only**. No circuit claim transfers to it.

## Exact stimuli and split

`stimuli.py` is the executable specification. Every seed creates 64 contexts and 192 prompts. Python `random.Random(seed)` determines the shuffled pair order, number suffixes, and company choices.

For leading digits, enumerate the 36 unordered pairs from 1–9, omit (1,2), (3,4), (5,6), (7,8), then include both orientations of each remaining pair. Shuffle these 64 ordered pairs. Each leading digit occurs seven or eight times in each role; exact equal counts are impossible with 64 items. Digit 9 occurs eight times. Exactly 32 items have N < T. N and T always have different leading digits.

Pool A uses suffixes 00–49; pool B uses 50–99, each sampled uniformly with replacement. Thus the two pools have disjoint integer values, not merely disjoint number pairs. Names are disjoint and have the suffix ` Industries`.

- A names: Alder, Beacon, Cedar, Delta, Elm, Falcon, Granite, Harbor, Iris, Juniper, Keystone, Laurel, Maple, Northstar, Oak, Pioneer.
- B names: Quartz, Redwood, Summit, Timber, Union, Valley, Willow, Zephyr, Aspen, Birch, Canyon, Driftwood, Evergreen, Flint, Grove, Horizon.
- Stage-1 dev seeds: 20261101 and 20261102.
- Full development set, after the design freeze: 20261101, 20261102, 20261103, 20261104. Only the first two have model outputs now.
- Held-out seeds: 20261201 through 20261208. Their text/token files have been generated and hashed; none has been sent through a model.

Context, wording 1:

`{Company} reported revenue of ${N} million last quarter. Management guided next-quarter revenue to ${T} million.`

Context, wording 2:

`{Company} reported revenue of ${N} million last quarter. Management said it expects next-quarter revenue of about ${T} million.`

The user message is the context, two newlines, then the question. Every question ends with ` Answer with the number only.`

| Type | Dev question | Held-out question |
|---|---|---|
| Q_N | What was last quarter's revenue? | How much revenue did the company report for the previous quarter? |
| Q_T | What is management's guidance for next-quarter revenue? | How much revenue does management expect for the coming quarter? |
| Q_F | What is the best estimate of next-quarter revenue? | What revenue should be expected for the coming quarter? |

The manifest is `workspace/exp07/stimuli/manifest.json`. It contains a SHA-256 for each exact JSON file, including rendered text, token IDs, prefill position, question-span positions, N/T token positions, and candidate digit IDs. Stage 2 verifies these hashes. Pool B always uses wording 2 and paraphrase 2. No choice is made from held-out outputs.

Stage 1 uses two seeds and 64 items per seed across three question types: 2 × 64 × 3 = **384** identical prompts for each behaviour model.

## Rendering, answers, and candidate metric

Render with `tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True, enable_thinking=False)` and a final assistant message containing `$`. Require the rendered string to end in `</think>\n\n$`. Extending the string with N or T must preserve the prompt token prefix, and the immediately following token must decode to the corresponding single digit. The dollar token and question span are recorded before padding; the harness converts these to left-padded batch coordinates.

Greedy decoding uses at most eight new tokens. Parse exactly one unambiguous signed integer from the decoded text. Decimal/comma-formatted tokens, multiple numeric strings, and no numeric string return “other”. Trailing units or prose do not change an integer answer. This implements integer extraction, not format compliance. Separately report numeric-only compliance. The initial pilot scorer incorrectly required an integer-only string; saved text was rescored with `scoring.py`, without another model run. Raw text/token files remain available.

Candidate metric: d = logit(first digit of T) − logit(first digit of N), taken at the prefill position before generation. Agreement is measured only among answers equal to N or T. Positive d predicts T, negative d predicts N, and an exact tie counts as disagreement. An empty eligible denominator is not a pass. “Other” counts stay visible. Wilson 95% intervals describe item-level proportions; they do not substitute for the eight-seed inference below.

## Harness and validated boundaries

`patch.py` hooks decoder-block output, `self_attn` output, and `mlp` output. Captures store only requested prefill or question positions. Replacements use the same layer/component/span in the donor. Unequal question-span lengths are rejected; no implicit token alignment occurs. Question-span replacement is supported by the harness but is not part of the amended discovery plan.

Per-head patches replace the selected 128-dimensional slices of `self_attn.o_proj` input at the prefill position. Qwen3-4B has 36 decoder layers and 32 query heads per layer. This is a query-head output intervention, not a KV-head intervention. Clean and donor activations are cached once per batch per phase. During a no-cache greedy rollout, the same original prefill intervention is reapplied each forward; later generated positions are not directly patched.

Every clean/patched forward checks all decoder-block, attention-output, MLP-output tensors and returned logits for finiteness. Any non-finite fp16 result stops execution. There is no automatic precision fallback.

Stage-1 Qwen3-1.7B validation used two same-context Q_N/Q_T pairs from the first dev seed. Same-prompt layer-0 residual patch: bit-identical logits, maximum absolute difference 0. Last-layer full-residual patch from a different question: donor logits reproduced, maximum difference 0, against atol 0.02 / rtol 0.002. The 4B timing-only head patch used the same prompt, layer 0/head 0, and was also bit-identical. This identity timing is not head discovery.

## Stage-1 behavioural evidence

Each dev seed had 64/64 correct Q_N answers, 64/64 correct Q_T answers, and 64/64 Q_F answers equal to T. Pooled agreement was 384/384, and the first generated token was a single digit for 384/384 prompts. No integer answer was outside {N,T}. All 384 outputs included extra text such as `million`, so numeric-only format compliance was 0/384.

Pooled Q_F T share was 100% (Wilson 95%: 97.09%–100%); N and other shares were each 0% (0%–2.91%). H5 passes on the two pilot seeds. The numeric task remains primary. The rescue trigger is inactive on these seeds. The full four-seed dev gate will be evaluated before discovery; a failure there still invokes the same pre-registered fallback.

Qwen3.8-27B behaviour uses these exact rendered Qwen3-4B strings with raw `/completion`, temperature 0, `n_predict=8`, `cache_prompt=false`, concurrency 1. Request outputs are stored only in `compute-host:workspace/runs/m1-27b/`. It completed 384/384 requests. Both Q_N and Q_T were correct on 128/128 items, and Q_F answered T on 128/128, N on 0/128, and other on 0/128. Its numeric-only format compliance was 0/384. The request phase took 186.16 seconds; this measures request execution only. Its first-digit logit agreement is not measured. Its circuit is not studied.

## Amended discovery and selection

The old 36 × 3 × 2 site grid and maximum-flip site rule are superseded. There is no selected full-residual site, and no question-span residual scan.

1. **Descriptive residual scan.** Patch the full residual only at the prefill position, layers 0–35, in both Q_T→N and Q_N→T directions. Average unconditional flip rates over four dev seeds. L* is the earliest layer where both directions reach 80%. Describe it as “the layer by which the selection is made”, with the limitation that a late full-state replacement can copy the donor answer by construction. If no L* exists, report this result and all-layer descriptive scans, but do not freeze a head set.
2. **Component scan.** For every layer 0…L*, replace attention-block output and MLP output at the prefill position. Report signed Δd, donor-oriented Δd, greedy flip rate, and validity in both directions. These components are descriptive intermediate results, not the final selected site.
3. **Required head scan.** Scan all 32 query heads in layers 0…L*. The worst case is 1,152 heads. Rank by mean donor-oriented Δd over four dev seeds and both directions: −Δd for Q_T→N and +Δd for Q_N→T. Break exact ties by layer, then head index. Individual-head ranking requires one forward, not a greedy rollout.
4. **Joint set.** Evaluate the top k heads jointly for k in {2,4,8,16}. “80% of the residual effect” means at least 80% of the positive mean donor-oriented Δd at L*, separately in both directions. Choose the smallest k that meets this rule and has validity ≥95% in every dev seed/direction. If no k qualifies, report the failure and stop before held-out tests. Do not substitute a late residual patch or choose another k.
5. **Freeze.** Write `FREEZE_<timestamp>.md` with the exact heads, k, L*, ranking rule, random seed/sets, dev means, hashes, and task. This file is committed before any held-out model call. The runner refuses held-out execution without the committed file and unchanged source/design/data.

Primary flip rate uses all 64 items per seed/direction. Also report the rate conditional on a correct clean base answer, with its denominator. Invalid outputs count as failures, never disappear from the primary denominator.

## Controls, necessity, and descriptive read-outs

- **Matched random heads:** freeze 100 distinct random sets of size k, drawn with `random.Random(20261105)`. Every set has the selected set's exact layer histogram and distinct heads within each layer. Exclude the selected set itself. Partial overlap is allowed. Reuse these same sets for sufficiency, necessity, and rescue comparisons.
- **Necessity:** replace each selected head's prefill output by its unconditional mean over all 512 dev Q_N/Q_T runs (four seeds × 64 items × two queries). Use the same dev mean for every held-out item. Repeat for all random sets. Report Q_N/Q_T accuracy drop and output validity; invalid answers count as wrong.
- **Mismatched donor:** choose a deterministic derangement of items with neither donor number equal to either base number. Patch the same frozen heads from that item's donor question. Report flips to the base's requested alternative, donor-number injection separately, and validity.
- **Random-vector control:** at each selected head add an isotropic Gaussian vector with the norm of the matched donor-minus-base vector. Its random seed is the item seed plus 20261105. This controls intervention magnitude without supplying the donor direction.
- The reverse intervention is mandatory. Mismatched-donor and random-vector results remain explicit controls; the amended H7 head-set rule replaces the former site-selection constraint. For H8 these controls must be ≤10%.
- **Read-outs:** for each frozen head, report attention mass from the prefill token to N's tokens and T's tokens, by query. Report direct logit attribution of its projected output to the gold first digit, holding the clean final RMSNorm denominator fixed. This is a linear descriptive attribution, not a separate causal test. Read-outs use batches of eight. In Q_F, the normative gold digit belongs to T.

## Hypotheses and held-out decisions

Held-out means eight seeds, pool B, wording 2, paraphrase 2, with one attempt per seed. All 512 contexts are retained. No replacement seed, prompt revision, threshold change, or head reselection follows inspection. An interrupted attempt remains a partial result and requires an explicit recovery decision; no silent retry is allowed.

| Hypothesis | Pass rule | Failure/reporting |
|---|---|---|
| H5: task/metric validity | Dev acc(Q_N) ≥90%, acc(Q_T) ≥90%, greedy–logit agreement ≥90% | Frozen table fallback; no numeric discovery after a failed full-dev gate |
| H6: forecast behaviour | Descriptive Q_F N/T/other shares and Wilson intervals, separately for 4B and 27B | No threshold and no circuit inference for 27B |
| H7: head-set sufficiency | Held-out mean flip ≥70% in both directions; each t(7) lower bound ≥0.60; every seed/direction beats the maximum of its 100 random sets; validity ≥95% in every seed/direction | Report the measured effect and controls; do not claim a localized selection mechanism |
| H7: necessity, separate decision | Selected-set ablation lowers mean Q_N/Q_T accuracy by ≥30 percentage points; each random set lowers that mean by <10 points | Sufficiency does not establish necessity; report both decisions and validity |
| H8: rescue, only if dev Q_F N share ≥20% | On held-out clean Q_F=N cases, Q_T-donor patch restores full integer T in ≥50%, while mismatched/random controls each remain ≤10%; report full-answer correctness | Null rescue result does not erase an independent H7 result |

For H7, first compute one flip rate per seed. The two-sided 95% Student lower endpoint is mean − 2.364624251 × sample_SD / sqrt(8), separately by direction. For necessity, average the two query accuracies within each seed, then average seeds equally. Compare each random set's eight-seed mean drop to 10 points. Also report every seed's drop and validity.

The H8 trigger uses all four dev seeds and is frozen before held-out work. The reverse rescue test patches Q_N into clean Q_F=T successes and reports movement to N. Use the identical frozen heads and all control families. If there are no eligible held-out cases, the conditional rescue rate is “not measured”, not zero or a pass. Numeric pilot trigger: inactive.

## Fully specified fallback

Switch only if any H5 threshold fails. `table_stimuli.py` creates 64 tables per seed, each with three line-item rows and three year columns. Use the same name/number pools, digit-pair generator, seeds, model, prefill, integer parser, metric, discovery, frozen-head controls, and eight-seed held-out structure.

Dev line items are `Revenue`, `Operating income`, `Cash flow`; dev years are 2021, 2022, 2023. Held-out line items are `Net sales`, `Gross profit`, `Capital spending`; held-out years are 2024, 2025, 2026. Each table has nine unique three-digit values, with every leading digit 1–9 represented once. N and T are the designated two target cells. Place them at two distinct cells sampled with `random.Random(seed + 700000)`, then shuffle the seven remaining leading digits into remaining cells and draw suffixes from that split's pool. The table heading gives units of $ million; every cell is dollar-prefixed.

Dev query: `What was {item} in {year}? Answer with the number only.` Held-out paraphrase: `For {year}, what value is listed for {item}? Answer with the number only.` Each table produces the N-cell and T-cell queries, retaining internal labels Q_N/Q_T. Thus each seed has 128 prompts. The same d compares their first digits. All nine cells count as plausible lookup content; only the two designated targets count toward {N,T} validity.

Generate and hash both splits before any fallback model call. Apply the same H5 gate on table dev outputs. If it also fails, stop; no third task is permitted. Forecast anchoring and H8 have no equivalent in a literal table lookup, so they remain numeric-task results and are not relabeled as table mechanisms. No fallback data or model run has been executed in stage 1.

## Compute budget

Measured batch-64 forward latency, with finiteness hooks: 0.516276 s and 0.517304 s. The same-prompt single-head patch measured 0.521474 s. Timings are one warmed observation each, not sustained estimates. Forward peak allocation was 7.855 GiB. Across the timed forward and subsequent pilot decoding, maximum allocation/reservation was 7.982/8.947 GiB. Peak counters were reset after model loading and warm-up; these are PyTorch counters, not total driver memory. See `workspace/exp07/compute_budget.json` for exact inputs and arithmetic.

At L*=35, the head scan is 36 × 32 heads × 4 seeds × 2 directional batches = 9,216 batch forwards, about 80.10 GPU-minutes from the measured single-head latency. Head ranking uses no per-head greedy decode. Residual/component/joint and held-out answer tests budget at most eight full forwards per rollout.

The complete discovery estimate is 144.52 GPU-minutes. The held-out estimate is 228.44 GPU-minutes, including selected plus 100 random sets for sufficiency and necessity. Four available GPUs give ideal wall estimates of 37.79 and 57.11 minutes; reserve 60 and 90 minutes respectively. These are extrapolations. Loading, queue time, file I/O, multiple-head hook overhead, and read-out costs can differ. If H8 activates, add roughly 28.4 ideal four-GPU minutes for its 103 sets/control conditions. No held-out or scan latency is measured.

## Exact stage-2 commands, not executed

The runner requires a committed `DESIGN.md` with the standalone `STAGE2_FROZEN` marker and an exact source SHA. A draft does not satisfy the frozen-protocol requirement.

```bash
ssh compute-host
cd workspace/mech-interp-lab
export DESIGN_COMMIT='REPLACE_WITH_FROZEN_DESIGN_COMMIT'
bash experiments/07_numeric_selection/stage2_commands.sh discover "$DESIGN_COMMIT" numeric
```

This command reuses the two saved pilot outputs, measures the other two dev seeds, applies H5, runs residual scans, calculates L*, runs component/head scans, evaluates joint k, and writes the freeze file. It then stops. Every model call uses `srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G`, offline mode and the specified environment. Four seed jobs can run concurrently.

The held-out run requires the committed freeze file and its exact source SHA:

```bash
export FROZEN_COMMIT='REPLACE_WITH_FREEZE_COMMIT'
bash experiments/07_numeric_selection/stage2_commands.sh heldout "$FROZEN_COMMIT" numeric
```

This uses two waves of four held-out seeds, then aggregates the prespecified tests. Each seed is reserved before its first model call. Outputs stay in `workspace/exp07/stage2/`; the freeze file stays in this experiment directory.

Only if the numeric H5 gate fails, the corresponding fallback commands are:

```bash
bash experiments/07_numeric_selection/stage2_commands.sh discover "$DESIGN_COMMIT" table
# Supply the committed table freeze SHA before held-out evaluation.
bash experiments/07_numeric_selection/stage2_commands.sh heldout "$FROZEN_COMMIT" table
```

Fallback output uses `workspace/exp07/table_stimuli/` and `workspace/exp07/stage2_table/`. The table generator and stage-2 runner are implemented and syntax checked, but no stage-2 path has been model-tested. All discovery and held-out quantities remain **not measured**.

## Protocol clarification (2026-10-18)

The stage-1 behaviour result changes the framing, not the design: on two dev seeds both Qwen3-4B and Qwen3.8-27B chose the guidance figure on every forecast question (Q_F → T 128/128 each), so the pilot gives no evidence of anchoring on this task. The study is therefore a localisation of **number selection**: which heads decide which of two in-context figures becomes the answer. H6 stays descriptive, and H8 runs only if the full four-seed dev gate triggers it.

One pre-registered addition, fixed before any discovery run:

- **H9 — transfer to the inference question (descriptive, with a threshold).** The frozen head set is discovered only on Q_N/Q_T, the literal questions. On the eight held-out seeds, patch the frozen heads' prefill outputs from the Q_N donor into the Q_F base. Same context and seed; Q_F's clean answer is T. Report the flip rate to N with the same 100 layer-matched random sets, mismatched-donor and random-vector controls, and validity. **Pass:** mean flip to N ≥ 50%, with every seed above the maximum of its 100 random sets. A pass means the heads that select the literally requested figure also carry the selection when the question requires inferring which figure is the forecast. H9 is computed by a separate script, `post/qf_transfer.py`, which reads the committed FREEZE file and runs only after the H7 held-out run is complete. It cannot change H7's verdict.

All other scientific rules remain fixed.

STAGE2_FROZEN

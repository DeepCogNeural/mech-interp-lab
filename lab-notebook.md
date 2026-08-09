# Lab Notebook

Dated log. Newest at the top. Each entry: what I did, what I expected, what actually happened, what confused me. The confusion is the most valuable part — don't sand it off.

Template for a new entry:

```
## YYYY-MM-DD — <title>
**Goal:**
**Did:**
**Expected:**
**Happened:**
**Confused about / open:**
**Next:**
```

---

## 2026-08-08 — The span carries the effect; the bridge is still open

**Goal:** Close the representation-span question without turning a span-level result into a claim about
the model's native circuit, then choose one high-information follow-up.

**Did:** Ran the independently prepared, candidate-independent Stage-3/Q4 invocation on the frozen
layer-8 `both` intervention, with the item-disjoint rank-training/evaluation split and 100 matched-span
draws per seed. The run used 353,120 logical forward-equivalents and 1,265.99 seconds of wall time. A
science audit recomputed the directed arrays from the raw result, and a separate artifact audit checked
the raw/result/CSV/checkpoint bindings; both shipped the result as complete.

**Expected:** Genuinely open. The twelve decoder rows recurring in experiment 04 might fail to beat
matched 12-dimensional spans, which would leave the Stage-2 transport result without a compact
representation-level explanation.

**Happened:** The frozen target span beat the frozen second-largest matched edge in every seed (8/8).
Across seeds, `R_span = 0.893525` with t(7) CI `[0.890486, 0.896565]`, while the complementary
subspace retained `R_comp = 0.084783` with CI `[0.082268, 0.087299]`. The geometric squared-norm
fraction was only about `0.525–0.544`, so the headline is concentration of a directed logit effect,
not 89% activation reconstruction. The generic-text PCA span/both comparator was a raw logit effect
of `0.027400`, not 2.74% recovery.

**Confused about / open:** This is a positive span-level causal comparison, not evidence that the SAE
rows are natural or monosemantic, that any individual latent is causal, or that the model uses a
head→span→readout path in its unconstrained computation. It does not establish necessity, sufficiency,
mediation, a complete circuit, or cross-model/task generalisation. The remaining uncertainty is
specifically whether the observed span effect is downstream of the selected head intervention.

**Next:** Run one Advisor-chosen exploratory bridge on fresh held-out items: L7H4 → `resid_pre8` target
span → natural L8H5/readout, with an L8H5 clamp arm. It has not run; no new amendment or protocol layer
is being added.

---

## 2026-08-08 — Two heads survive the controls; the representation question stays open

**Goal:** Stop treating experiment 05 as an execution protocol and make it answer a scientific
question: is there a small, reproducible set of attention heads that carries subject-number
information under a controlled intervention?

**Did:** Reran both 144-head selection sweeps in one unchanged in-memory GPT-2-small snapshot, froze the
eight-head candidate set, and ran all eight Stage-2 seeds from zero. The completed invocation used
2,528 logical forward-equivalents and 6,272 seconds of CPU wall time. A first attempt had stopped after
688 seconds on a hook-callback signature error; it emitted no science, the callback was repaired and
regression-tested, and the adjudicating run restarted rather than resuming its scientific rows. I then
gave the final artifact and its pair-level data to two independent reviewers—one recomputed the science
without trusting runner verdicts, one checked hashes and execution coverage—and sent the compact result
to the Pro Advisor for a separate claim-boundary review.

**Expected:** Genuinely open. The Stage-1 ranking suggested a distributed set, but it did not say how
many heads would recover half of the joint effect, whether the apparent signal would survive sources
that remove or preserve number, or whether a subject-position value-path intervention would remain
positive once the attention pattern was clamped.

**Happened:** The minimum tested set was two heads, `L7H4` and `L8H5`. One head alone recovered only
`0.240–0.251` of the registered direct effect and failed in every seed; the pair recovered
`0.521–0.546` and passed in all eight, with both members individually above the source-A floor. The
largest source-A/true ratio was `0.0052` against a `0.20` limit, and the largest source-C/true ratio was
`0.1950` against `0.26983`; Q2 passed eight of eight. The clamped frozen-attention value-path effect was
positive in every seed (`D_path=0.528–0.574`, smallest 95% bootstrap lower bound `0.504`); Q3 also passed
eight of eight. Pair-level recomputation reproduced all three decisions, the artifact audit found all
16 execution cells and the bound 1,100,148-row CSV, and the Advisor returned `ACCEPT` with no major flaw.

**Confused about / open:** This is the first adjudicated real-model mechanism result in the repository,
but it is narrower than the sentence I most want to say. B-prime freezes the base attention pattern and
clamps the value-path edit, so a robust effect there may be larger than—or simply different from—the
heads' role in GPT-2's unconstrained endogenous computation. It does not prove necessity, sufficiency,
full mediation, a complete circuit, or monosemanticity. The representation question that motivated the
application is still live: the result identifies transport-related causal structure, not whether the
twelve recurring SAE decoder rows are a privileged low-dimensional basis for that structure.

**Next:** Run the already prepared, candidate-independent Q4 comparison on held-out items: the fixed
twelve-row layer-8 decoder span versus matched 12-dimensional spans. If it is positive, test one direct
head→span→readout bridge; if it is negative, test an equal-rank supervised subspace to distinguish an
SAE-basis failure from the absence of a compact transported state. One result, one alternative
explanation, one follow-up—not another amendment cycle.

---

## 2026-08-08 — Making experiment 05 executable without pretending it has run

**Goal:** Turn the frozen questions in experiment 05 into a reviewable, fail-closed execution
protocol while preserving the line between a design decision, an implementation, and evidence from a
model run.

**Did:** Asked an independent Pro Advisor to adjudicate the places where the written design still
allowed multiple scientifically different implementations, then recorded those decisions as
Amendments 5–9 and a result-free machine protocol. Implemented separate selection, candidate-freeze,
Stage-2, and Stage-3 entry points around that protocol. The implementation is still under static
adversarial review; no offline contract test or model-backed command has been run.

**Happened:** The largest correction was not a threshold. The shipped Stage-1 true-source sweep and a
new source-A sweep could not be shown to use exactly the same model snapshot. Candidate `C` therefore
cannot lawfully compare them. Amendment 7 requires both 144-head sweeps to be rerun in one invocation,
against one in-memory model and immutable clean-base cache, with complete state-dictionary fingerprints
before and after. The old Stage-1 ranking survives only as a non-blocking historical cross-check. This
costs 291 logical forward-equivalents and has not yet been authorised or executed.

The other corrections were about identifying what the interventions actually establish. Q3 now has
one explicit clamped-`z` kernel rather than an underspecified “path” patch. Q2 uses the exact intersection
of complete base and source-C item pairs; a global source-C Gate-A flag is diagnostic rather than a
second, accidental exclusion rule. Q4 keeps rank-training items separate from evaluation items and can
only support the narrow statement that the fixed layer-8 twelve-row decoder span exceeds matched
12-dimensional spans on the registered intervention. It cannot establish mediation, necessity, or the
model's native causal path.

Static review also changed the engineering standard for the experiment. A self-hash is not enough if a
plausible but edited checkpoint can be re-hashed, so resumable scientific payloads must be reconstructed
from their pair records and frozen seeds. Amendment 8 goes further for Stage 2: an earlier checkpoint is
diagnostic only, and every adjudicating cell must be rerun in one fresh invocation. Amendment 9 applies
the same evidence rule to the Q4 runtime while retaining independently reviewed preparation artifacts.
`COMPLETE` must bind its pair/draw artifacts, and a failed new invocation must invalidate any older
successful-looking output. Scientific non-estimability, a
pre-registered conditional skip, and technical execution failure now have different machine statuses.

**Confused about / open:** A long protocol can prevent silent analytic flexibility while also creating
new surface area for contradictions. The right portfolio claim is therefore not “the protocol is long,
so the result is trustworthy.” It is that each added rule answers a named failure mode, is machine-bound
where possible, and remains independently reviewable. That claim still needs contract tests and actual
artifacts before it is earned.

**Next:** Finish the static producer-to-consumer review, commit the result-free implementation, and ask
the Advisor to review the pushed commit. Only after explicit execution authorisation: run the small
offline contract suite; run the 291-FE same-snapshot selection with a declared wall-time cap; freeze and
review `C`; then run Stage 2 and the independently prepared Stage-3 pipeline. Until those steps complete,
there is no Q1–Q4 verdict and no new model finding.

---

## 2026-08-08 — Advisor review correction: experiment 05 is not ready to run

**Goal:** Record the corrections requested by the independent Advisor review of pushed docs-only commit
`07bc83c8f793cf58db79cc9667141b4e1f2cd7ae` without rewriting the 2026-08-02 historical entry.

**Did:** Reconciled that entry with experiment 05's dated amendments and added Amendment 4's Q4 data-role
gate. No Stage-2 or Stage-3 result exists, and this correction does not report a new model run.

**Happened:** Three sentences below no longer survive. Amendment 2's repairs are not uniformly ordered as
making a positive harder; Amendment 3 already withdraws that monotonicity claim, especially for Q3. The
clean `E_all` and `d_gap` are the same mathematical estimand and have the same reported mean up to numerical
error, but the shipped float32 records differ on 2 of 472 directed edits, with maximum absolute difference
`5.7220458984375e-06`; that is not bitwise identity. Finally, Stage 2 does not follow mechanically from the
shipped ranking: seed `20260801` still needs a selection-only full source-A 144-head sweep to complete
candidate set `C`, and the incomplete runtime projection must be re-estimated before any experiment run.

The review also found one separate Stage-3 gate: disjoint experiment seeds isolate the twelve target rows
from experiment 04's selection, but do not by themselves make the per-seed matched-latent pool independent
of Q4 evaluation. Amendment 4 therefore freezes an item-disjoint rank-training/evaluation split before any
Q4 run while leaving the 100-draw second-largest edge and all other numerical rules unchanged.

**Next:** Complete the selection-only source-A supplement and runtime re-estimation before Stage 2. Do not
run Stage 3 until the Amendment-4 split is implemented and its pair, pool, and subset identifiers can be
recorded. Both execution steps remain separate work that requires explicit authorisation.

---

## 2026-08-02 — Designing experiment 05, and changing the banner over the whole repository

**Goal:** Decide whether this line of work is still the right one, verify that what is already published is actually correct, and pre-register the next experiment before writing a line of its code.

**Did:** Three things in parallel, deliberately, because they check each other. An adversarial re-verification of the *current* published state (the corrections themselves had never been independently checked). A source-verified literature check on what is actually known about number agreement in GPT-2-small. And a direction review that started from "is this path right at all", not from "what should experiment 05 measure".

**Expected:** That the verification would come back clean, since experiment 04 had already been through two correction rounds; and that the design draft would need tightening rather than restructuring.

**Happened:** Both expectations were wrong in useful ways.

The verification found one surviving class of retracted claim. Correction 1 had downgraded "the threshold was fixed before the run" to "the design commit precedes the output commit", but several entry points still carried the stronger wall-clock version — including this notebook, which said a floor was "fixed hours earlier that same night, before any of these numbers existed". Commit timestamps show commit times. That is the third time the same claim has had to be walked back, which says something: the temptation is not in the numbers, it is in the sentence that frames them. Also caught: the exp03 README snippet's four links had been dead since it was written (paths written from the repo root, resolved from the file's own directory), and experiment 01 is the one experiment shipping figures with no machine-readable results file, so two of its README numbers cannot be re-derived without rerunning it. That is now stated in the README instead of left for a reader to discover.

The literature check produced the most useful negative of the day: **there is no peer-reviewed head-level causal baseline for number agreement in GPT-2-small.** Finlayson et al. (2021) is neuron-level, and its appendix's head numbers are described *by its own authors* as possibly noise — so had I used them as a comparison, I would have been citing something the authors disowned. The design note that said "these head numbers are from memory and must be checked" was right to exist. Sharper still: during the check, automated summaries returned two different and both-wrong head numbers for one of the papers surveyed (Ryu & Lewis 2021), and only reading the PDF settled it. The strongest real prior is layer-level (Lepori et al., COLM 2024, layer 6's attention block), which is now a pre-registered prediction rather than a post-hoc comparison.

The design review restructured the experiment rather than tightening it. My draft had a "noise band" defined as the 99th percentile of heads ranked 45–144 — which is a *rank threshold wearing a statistic's clothes*. Its edge does not shrink as data grows; "exceeds the band" reduces to "ranks in the top ~45"; and every head in a top-8 set passes it by arithmetic. I had rebuilt Gate C's mistake in a new costume: a criterion that cannot fail where it matters, next to a criterion that fires where it should not. It was replaced with item-bootstrap intervals and an *empirically measured* source-A floor from sweeping all 144 heads with a source that carries no number flip; those estimates can be refined with more retained pairs. The later frozen rules are not all shrinking nulls: Q2 uses fixed calibration-derived specificity ratios, and Q4 uses a finite matched-span reference. The source-A sweep also turned out to be the specificity control, which I had been treating as a separate instrument. They were always the same measurement.

Three more of my own errors, worth recording because each is the same species: comparing a head-level patch to a residual-level reference (different intervention families, so "recovers half" had no fixed meaning — now divided by an all-144-head ceiling); imposing `R_span ≥ 0.50` on the latent-span question (Gate C again, third costume — deleted, the band comparison adjudicates and the fraction is reported graded); and describing a value-stream patch at the subject position as showing that a head *reads* number there. It does not: patching a head's value at one position changes what every downstream query reads from it, so only the negative direction is licensed. The covariate is renamed and the real transport claim is now a two-step path patch.

**Confused about / open:**
- The pattern across experiments 03, 04, and this draft is not a threshold-calibration problem. It is that I keep building the *shape* of certification — a bar that a quality measure must clear — in places where the honest object is a graded effect size against a null. Naming it did not stop me doing it again in the draft. Writing the boundary sentences before the numbers exist is the only countermeasure I have found that is not just resolving to be more careful.
- The repository's stated through-line was superposition versus mixed selectivity, and by experiment 03 the measured quantity had become SAE basis quality. I had known this privately for six days while the README still advertised the old framing. A private note recording that a public document is wrong does not make the public document less wrong. The README now leads with the escalation of method the work actually followed.
- Whether four adjudicated axes is genuinely more honest than one, or whether it is a way to guarantee *something* comes back positive. I think the former, because each axis has a pre-declared negative that is itself a claim — but that is exactly what someone doing the latter would say, so it goes in the record.

**Then, same day — the calibration pilot ran and the design froze.** 101.8 s of CPU. Restricted to residual-stream interventions, with head-level cost priced by attaching hooks that return the activation unchanged and timing the forward pass: you get the wall clock without getting the number. So no attention head had been measured at the freeze. (I first wrote that as "no head-level *or span-level* measurement has ever been run in this repository", which is false on the second half — experiment 04's PCA arm edits the top k coordinates of a complete orthonormal basis, and that is exactly an orthogonal projection onto a k-dimensional span. Narrowed in Amendment 2.)

The constants: a source that changes *which* noun without changing its number moves the readout by `−0.0078` against the true flip's `4.46` — essentially nothing, which is the reassuring one. A cross-template number-matched source moves it by `−0.60`, 13.5% of a full flip — and here I got the sign story wrong on the first pass, in three files. Negative on this axis means the push *reinforces* the number both sentences already share, not that it pulls toward the wrong verb. Same-number reinforcement, which is a demanding control for a different reason than I wrote, and it sets Q2's bound at `0.270` rather than the `0.20` floor. And the attractor flip, which I had expected to be the interesting one, moves the mean by `+0.0071` while carrying the largest per-item spread of the three. I wrote that as "the movement does not point anywhere" — also wrong, and wrong in the way that should have been obvious: everything is aligned to the *subject*-flip axis, which forces a mean near zero for any attractor-locked effect no matter how consistent it is. What I can say is that items move and this axis does not resolve where. All of it is recorded before a single head has been measured, so whatever Stage 1 finds, none of it can be fitted to the answer afterwards.

Five defects surfaced at fill-in, and they are in `DESIGN.md` as Pre-freeze correction 1 rather than edited away. Two were mine and both were the same failure of care: I wrote the disclosure trigger as `ρ > 0.20` when my own formula `max(0.20, 2ρ)` crosses over at `0.10` — and the measured `0.135` landed exactly in the gap, so the clause meant to fire did not fire literally. I applied it anyway, under the stricter reading, because the alternative is letting a typo decide what gets disclosed. The other was arithmetic: I claimed clearing both bars guarantees a `0.30·E_all` specific component when it is `0.365`. I had *understated* my own bound, which is the direction that flatters nobody and still means I did not check.

Three came from the implementation refusing to paper over them. My runtime budget forgot the per-seed joint-set patches Q2's own rule requires, and Stage 3's PCA fit, and its candidate-pool prep. Worse, attention-value caching *cannot* be priced before the freeze without the head-level access blinding forbids — so the honest output is a lower bound of 34.7 CPU-minutes against a 120-minute cap, not a total. Roughly 3.5× headroom, and "unlikely to bind" is the claim rather than "verified". Experiment 04's pilot missed its main run's cost by 25× by scaling per-patch cost alone; being wrong in the safe direction this time is not the same as being right.

**Then Stage 1 ran, and then a six-lens adversarial review of the whole day's work came back with twenty findings that survived an attempt to refute each one.** Three were factual errors of mine, and the worst was a sign: I described source C's `−0.60` as a pull toward the *wrong* verb when the alignment convention makes negative mean the opposite — it reinforces the number both sentences share. A measured constant, interpreted backwards, propagated to three files, and it got past me, past the advisor, and into a frozen document. The lesson is narrow and I want it recorded narrowly: I reasoned about the sign from the words in my own design instead of from the six lines of code that produce it.

Five more were decision rules that could not actually be executed as written — a bar with no null in Q3 (the banned certification shape, third recurrence, this time inside the document that opens by saying it retires it), the candidate-selection seed also sitting in the adjudication set so one of eight seeds passed by construction, a "99th percentile" of twenty draws which is just the maximum, a Holm correction with no family error rate, and a fallback tested set that is undefined when the candidate pool is small. Amendment 2 repairs all five. I initially claimed every repair made a positive verdict harder; Amendment 3 withdraws that monotonicity claim because the old and new Q3 rules are not ordered. The legitimate basis for the changes is narrower: the original rules were not executable or did not measure the stated comparison, and the replacements were frozen before the adjudicating runs.

Stage 1 itself came out clean and gave one genuinely nice result: `E_all` and the clean `d_gap` estimate the same mathematical quantity and their reported means agree up to numerical error. They are not bitwise identical in the shipped float32 records: 2 of 472 directed edits differ, with maximum absolute discrepancy `5.72e-06`. The structural identity is still a useful harness check, but the artifact supports numerical agreement rather than exact bitwise equality.

**Next:** Run the fresh same-invocation true/source-A 144-head selection sweep required by Amendment 7, freeze candidate set `C`, and only then begin the eight-seed Stage 2 adjudication. Stage 3 remains a separate, candidate-independent Q4 run. None of those runs had started at this notebook entry's latest correction.

**Confused about / open, added at freeze:** the source-B result bothers me in a useful way. If attractor number genuinely moves individual items in inconsistent directions, that is either agreement attraction with item-dependent sign, or my attractor manipulation is doing something I have not understood. I cannot tell from a mean and a spread, and I deliberately did not go looking, because the instrument for looking is head-level and the design is now frozen. It goes in the writeup as an open question with the numbers attached.

---

## 2026-07-27 — Experiment 04: the first causal experiment, and a gate that refused to certify what it measured

**Goal:** Run the pre-registered causal interchange experiment and let the frozen rule decide, whatever it decides.

**Did:** Pilot (42 s), Gate C diagnostic (647 s), five-seed main run (1,760 s), robustness arms (1,344 s). All CPU, offline. Three dated amendments to `DESIGN.md`, each committed before the commit that carries the output it governs — which is commit order, not proof of read order, and the writeup now says so rather than claiming more.

**Expected:** Genuinely open on the science. On the process I expected the run to either adjudicate or fail a gate cleanly. I did not expect it to do both at once.

**Happened:** The cleanest measurement in this repository, and the verdict is `inconclusive`.

Sixteen SAE coordinates recover half of that basis's own causal effect on subject–verb number agreement, in every seed. PCA needs 64 to 128 and in one seed never gets there. The paired gap is `+0.304 ± 0.023` within-basis and `+0.146 ± 0.022` absolute, positive in all five seeds under both denominators. And the SAE's trained decoder writes back `0.694 ± 0.014` of the residual-stream effect, against a Gate C floor of `0.70` committed hours earlier that same night, in a commit that precedes any commit containing these numbers. It passes in two seeds of five. My own design says a basis that fails Gate C yields no headline — so the measurement is reported and nothing is claimed from it, and I am not moving the threshold by 0.006 to buy one.

**Confused about / open:**
- The best thing about this run is the part nobody would notice: **the D-rescale invariance is exact, and one instance of it is asserted bitwise.** Rescale the code by a positive diagonal and the decoder inversely and the written vector is algebraically unchanged; the self-test builds one random power-of-two diagonal and checks the written delta with `torch.equal`. The knob that made experiment 03 unanswerable is not "small" here — it is exactly zero by construction. That is what it feels like to fix a problem structurally instead of statistically.
- Three separate times the process caught me rather than the other way round. The pilot's single-seed Gate C was `0.731`, comfortably passing; five seeds gave `0.694`, and the pilot number was luck. The diagnostic I ran to check whether the random control's failure was an under-powered fit came back *worse* with four times the data, which is the answer I did not want but the one that redirected the whole control arm. And the certificate that made the SAE's *published decoder* look weak at differential write-back — a small ridge reaching `0.887` against that decoder's `0.694` — reversed completely once I ran the clean control: a *generic-only* ridge manages `0.414`, far below the trained decoder. The 0.887 depended on 4.7% of its fitting rows being template activations. None of this is about the model's computation; it is about how well a linear map writes a code difference back at one hook. I had already repeated the uncorrected version out loud before the control existed. That is the argument for running the control.
- A fourth catch, and this one was an adversarial review rather than me. I had written that the edit being "a difference of two reconstructions" makes the SAE's reconstruction error "cancel exactly". It does not: `W_dec(f_src − f_base) = (x_src − x_base) − (e_src − e_base)`. The decoder *bias* cancels; the difference of errors does not — and Gate C's `0.694` is that residual, staring at me from my own manifest. The scope guarantee I wanted (never substituting a reconstruction for the model's state) is real, but I had attached it to the wrong mechanism. A sentence can be reassuring, technically wrong, and refuted by a number in the same document.
- The refusal that cost the most to keep: I could have adjudicated by defining Gate C on the `sae_ridge` arm, which clears every gate and shows the same ordering (`+0.348` within-basis). But I had already seen the headline, so calling that pre-registered would have been false advertising. It is in the writeup labelled "specified after unblinding, never adjudicating", and the verdict stands at inconclusive. Checking afterwards, it would not even have worked — seed `20260803`'s Gate D blocks independently, so rescuing it needed two goalposts moved, not one.
- The sober number I keep coming back to: a *single supervised direction* recovers `0.549` of the effect, while the SAE's single best latent recovers `0.072`. Among the 128 candidates actually scored in each basis, none puts this factor in one coordinate — I did not search outside that set, so it is a statement about what the ranker admitted, not about the whole basis. What the measurement says is only that ranked SAE coordinates approach a supervised direction with four to eight times fewer coordinates than PCA does (per seed: 4×, 8×, more than 8×, 8×, 4×). Under my own frozen rule that is an uncertified descriptive measurement, not a result — and it is much smaller than "the SAE found the number feature".
- Still open and honestly unresolved: PCA is fitted on 8,192 tokens for a 768-dimensional covariance, against an SAE whose training corpus I am quoting from memory as ~10⁸ tokens and have not checked against the published card. Quadrupling the PCA budget moved `AUC(pca)` by `+0.018 ± 0.025` against a gap of `0.304` — an empirical difference between two budgets, not a convergence bound; the eigenvalue spectrum is still moving, so the control is not converged. If this measurement is misleading, that is where it is misleading.
- A small process note worth keeping: two runs stopped dead at a numerical assertion and refused to loosen it. The threshold was mine and it was wrong — an absolute `1e-4` bound on activations whose entries run to tens is a float32 round-off test, not a test of anything. I fixed it as a dated amendment rather than editing it in place, because the difference between "repaired a bad assertion" and "loosened a threshold when the result was inconvenient" is exactly what the record has to be able to show.

**Next:** Re-register a Gate C that the SAE arm can clear on its own decoder, with the floor pre-declared from a pilot on a *different* stimulus family so it cannot be tuned to this one. Then a second template family. The mechanism question — what are the 12 latents that recur across all five seeds — needs its own pre-registration, not a post-hoc hunt.

---

## 2026-07-26 — Stepping back: what three experiments actually established, and the decision to leave probes

**Goal:** Stop running and audit. Three experiments are done; decide whether the next one finishes experiment 03 or changes the question.

**Did:** Reread all three writeups against a hostile reading of the portfolio as a whole, then wrote `experiments/04_causal_feature_interchange/DESIGN.md` as a pre-registration, before any code exists.

**Expected:** That the sensible next step was to finish experiment 03 — a scaling-invariant criterion, or more samples so the effective feature count stops exceeding the sample count. I had a design half-drafted for a sample-size sweep: push `n` well past the surviving feature count and watch the two scalings converge.

**Happened:** I talked myself out of it, and the argument that did it was not about experiment 03 at all. Every measure in this repo is linear decodability of a representation. Not one result is causal — no patching, no ablation, no intervention on a forward pass. That is representational geometry, which is the analysis style I already had from V1, and it is not what mechanistic interpretability is premised on. The sample-size sweep would have been a fourth decoding experiment, and a clever one, and it would have deepened exactly the weakness. There is an irony I had not noticed until I read the roadmap back: the methodological warning I lean on hardest, Jonas & Kording on the microprocessor, is a warning about trusting decoding results.

Worse, the sweep's own logic has a hole. When features outnumber samples, two codes related by an invertible linear map are close to indistinguishable to a decoder that carries no metric — so the metric has to come from somewhere, and it comes from the regulariser. That is the sharp version of experiment 03's finding, and it says the fair scaling point I would be hunting is probably not there to find.

**Confused about / open:**
- The honest re-score is that experiment 02 is not a null — it is a theorem-anchored positive result plus a discriminating secondary null. Experiment 03 is the precarious one, because "not adjudicated" is *weaker* than a null: a null answers the question, an instrument failure says the question could not be asked with that instrument. I had been counting both as the same kind of honesty. They are not.
- The replacement design has a property I like more than its being the missing checklist item. The interchange edit `W_dec (f' − f)` is invariant to the exact rescaling that swung experiment 03 tenfold: rescale the code by `D` and the decoder by `D⁻¹` and the vector written into the residual stream is unchanged. The knob that made the old question unanswerable cannot touch the new one. First time the failure has told me something constructive about what to build next.
- Named risk: the matched random expansion needs a decoder, and getting a random basis onto equal reconstruction footing with a trained SAE is fiddly. It is gated (Gate C) and prototyped first, with a smaller within-SAE result to fall back on. Better to find that out in twenty minutes than in a week.
- Watching for the mirror of the experiment 03 mistake. There, three mutually reinforcing controls all sat downstream of one unexamined preprocessing choice. Here the analogous single point of failure is the coordinate ranking rule — so it gets the ranking-free `*_full` anchors and the `*_randk` controls around it, and both bases get the identical rule and the identical budget.

**Next:** Gate A and Gate C on a 20-pair pilot before anything else. If the random basis will not reconstruct, the experiment gets smaller rather than looser.

---

## 2026-07-26 — Experiment 03 addendum: convergence did not remove the scaling problem

**Goal:** Run the one decisive five-seed test: fit the SAE and sparse-random SD probes to a stated convergence criterion, select L2 item-disjointly for each arm and scaling, then ask whether the paired two-way-XOR difference agrees across the two affine feature scalings.

**Did:** Used full-batch L-BFGS on L2-logistic loss, stopping below `1e-3` relative objective change across ten accepted iterations (500-iteration cap). Kept five item-disjoint folds, all 35 dichotomies, both scalings, and four arms; omitted CCGP and `rand_exp_dense` exactly as scoped. The zero-unit keep-mask was fit in each outer training fold. Seed 0 took 30.1 s, so the pilot projected far below the 40-minute budget and no planned scope was cut. Full raw rows are in `experiments/03_ccgp_on_sae_features/convergence_results.json`.

**Expected:** If the old 10× swing was merely fixed-step non-convergence, the paired `sae − rand_exp` two-way-XOR estimates would both be tight and close after convergence. If the z-score result stayed noisy or the two means stayed apart, that would confirm the comparison is conditional on the L2 prior geometry rather than adjudicate a code property.

**Happened:** The dense check behaved as expected: every `resid` / `sae_recon` SD-family shift was at most 0.0058. But z-score gave `+0.0580 ± 0.0414` and global RMS `+0.1151 ± 0.0192` for the paired SAE−random two-way-XOR contrast; their means differ by 0.0571. The z-score estimate is not precise and the means are not close, so the test **does not adjudicate** the code comparison. It supports the shipped methodological conclusion instead: the apparent edge is still inseparable from the L2 geometry induced by feature scaling.

**Confused about / open:** One z-score SAE inner split selected the high L2 grid edge even after four predeclared expansions; its documented `1e19` fallback is a direct sign that the inner main-effect criterion itself can become uninformative for that sparse scaling. I did not retune it away after seeing the result. A future probe family needs either a scaling-invariant criterion or a question that explicitly treats the regularisation geometry as the object of study.

**Next:** Keep the published headline and its `Next` item unchanged. This addendum closes the convergence test as a confirmation of non-adjudication, not as a route to revive the width-control interaction claim.

## 2026-07-26 — Experiment 03: a real SAE on the shattering × CCGP plane, and a headline I had to give back

**Goal:** Leave toy land. Ask exp02's enumeration-versus-computation question of real SAE features on a real transformer, with Bernardi's task-agnostic metrics instead of one hand-picked XOR.

**Did:** GPT-2-small layer 8, res-jb SAE loaded straight from the published safetensors. Full factorial NUMBER × TENSE × POLARITY, read at a sentence-final `.` that is byte-identical across all eight cells. Seven arms, 35 dichotomies, 16 CCGP splits, 5 seeds, item-disjoint folds. Then, after review, a probe-fairness control and two effective-width controls. `experiments/03_ccgp_on_sae_features/`.

**Expected:** Genuinely open. Sparsity pressure could make the SAE a clean factorised code (low shattering, high CCGP); conjunctive latents could instead hand a linear readout the product term (high shattering). I deliberately did not pre-register a direction. The one thing I was confident about: `sae > resid` would be uninteresting, because 768 → 24,576 with a ReLU wins that by Cover's theorem alone.

**Happened:** Four separate times I thought I had the answer and did not.

1. First real numbers said the SAE was *worse* than matched random on everything, with a big base-factor confound. I flagged it as confounded and moved on.
2. Suspecting the probe, I asked for a fairness control. Per-feature z-scoring turned out to amplify rare SAE latents into exactly the directions a probe overfits — train−test gap +0.28 versus +0.09 for the residual stream. Under a single global scale the ordering *reversed*: SAE two-way-XOR went from +0.009 to +0.121 over matched random. That felt like the result.
3. The obvious attack was that the SAE simply has more distinct latents firing (~790 versus ~500) even at matched per-sample L0. Matched it in both directions. The edge shrank to +0.081 and +0.075 but survived, and the narrowed-SAE control matched base factors exactly while handicapping the SAE on L0. That felt like a *strong* result.
4. Then an adversarial review pointed out the thing I should have seen in step 2: **for a linear probe, per-feature z-scoring is an invertible affine reparameterisation.** The reachable function class is identical. So a 10× swing in the answer cannot be a property of the codes — it is the L2 prior and the optimiser's path, and my "global RMS is principled because the decoder reads raw activations" argument was a post-hoc story for a number I liked.

So the headline is now: **not adjudicated.** The convergence test that would settle it did not complete (inner-L2 selection found no stable candidate on the dense-random arm), and I am not shipping the flattering setting.

**Confused about / open:**
- The salvage is that the sensitivity is *localised*. The two dense 768-dim arms (`resid`, `sae_recon`) are completely unmoved by the scaling — 0.902 vs 0.901, 0.877 vs 0.877. Every sparse or very wide arm swings hard. That is exactly what the affine argument predicts, and it makes the real finding methodological: probe preprocessing that is harmless for dense representations is decisive for sparse over-complete ones. Any SAE-versus-baseline decoding comparison that doesn't state its scaling convention is under-determined. I believe that part.
- I still don't know whether the SAE has a genuine interaction-readout edge. I think the width controls are suggestive, but they only ever ran under the one scaling that produces an effect, so they inherit the question rather than answering it.
- Uncomfortable lesson about my own process: at step 3 I was *more* confident than at step 2, because the controls all pointed the same way — but every one of them was downstream of an unexamined preprocessing choice. Stacking controls on top of an unvalidated foundation feels like rigour and isn't. The review caught it; I didn't.
- Two smaller things I got wrong and corrected: `1.96 × sd/√5` is not a 95% CI at five seeds (needs t(4) = 2.776, ~29% wider), and 12 cells of a results table were stale numbers from an earlier run. Both harmless to the conclusions, both exactly the kind of thing that makes a reader stop trusting everything else.
- The whole run-1 stop is worth keeping too: no network in the execution sandbox, so it hit Gate A and wrote a `gated_out` manifest instead of a plausible-looking number.

**Next:** Finish the convergence test — converged probes, L2 selected per arm *and* per scaling on an interior grid, agreement judged only when both estimates are individually precise. Then, and only then, ask whether conjunctive latents are the mechanism. Independent template families before any of it generalises.


## 2026-07-20 — Experiment 02: does superposition help a downstream readout?

**Goal:** Turn the exp1 bridge thought into a real, falsifiable test — is mixed/superposed coding just a storage compromise, or does it also keep nonlinear computation linearly readable (Rigotti's mixed-selectivity claim), measured in a toy model with ground truth?

**Did:** Three geometry arms (monosemantic / random / frozen superposition) at fixed `(n,m)`, read XOR of a feature pair with a linear probe on `r = ReLU(Wx)`. 8 seeds, fixed balanced eval distribution, within-seed paired stats. `experiments/02_superposition_and_readout/`.

**Expected:** monosemantic at chance (theorem: linear readout can't get the `x_i·x_j` term); mixed codes above chance; and — the part I was unsure about — maybe the storage-learned geometry beats random mixing.

**Happened:**
- Headline came out clean and strong: monosemantic sits at 0.494±0.005 at every sparsity, both mixed codes at ~0.80, probe train-test gap +0.002. The theorem-backed anchor holds exactly.
- The secondary question resolved to a **null**: superposition ≈ random (paired diffs hug zero, CIs include zero across `m`, `S`, and background). The readout benefit is from mixing+nonlinearity, not the specific learned geometry.

**Confused about / open:**
- Almost walked into a trap: my first instinct was a linear probe on the exp1 encoder `h = Wx`. That's identically chance for *any* geometry, because a linear readout of a linear projection can't represent XOR — the null would have been real but for a boring reason. The fix (and the actual insight) is that a nonlinearity is required, and it has to be held constant across arms so the comparison is about geometry, not about the nonlinearity.
- A 4-seed pilot hinted superposition beat random by ~+0.02 at high sparsity with background activity. At 8 seeds it washed out. Good reminder to not stop at the seed count that flatters the hypothesis.

**Next:** shattering dimensionality / CCGP (the task-agnostic version of the headline); then ask the same enumeration-vs-computation question of real SAE features on a small transformer.

---

## 2026-07-16 — Setup + first experiment (Toy Models of Superposition)

**Goal:** Stand up the lab and run one real experiment end to end, not just read.

**Did:**
- Created the repo (`DeepCogNeural/mech-interp-lab`), wrote the README, the week-by-week `learning-roadmap.md`, and this notebook.
- Environment: system Python is 3.14, which has no torch wheels yet. Fell back to **Python 3.11** in `.venv`. Installed torch 2.13.0, transformer_lens 3.5.1, numpy, matplotlib, jupyter. Clean.
- Built and ran `experiments/01_toy_models_of_superposition/toy_models.py`. Three experiments, all on CPU, ~90s total. Real figures on disk in `figures/`.

**Expected:** Superposition appears as sparsity rises; the 5→2 case should give the pentagon; `Σ Dᵢ` should approach `m` only when training uses the bottleneck fully, consistent with its role as an effective-dimension upper bound.

**Happened:** All three confirmed.
- 5 features → 2 dims: dense keeps only 2 orthogonal features; sparse + *uniform* importance gives the textbook **pentagon**; sparse + *decaying* importance gives antipodal pairs instead. The importance weighting picks the symmetry.
- 20 → 5: goes from 7/20 features (dense, `Dᵢ≈1`) up to the full 20 at the sparsest end (19/20 at S=0.99, 20/20 by S≈0.997; `Dᵢ≈0.25`), with `Σ Dᵢ` sitting near `m = 5` once the bottleneck is well used. It is an effective-dimension upper bound, getting close to `m` only when training uses the bottleneck fully; superposition redistributes that available dimension budget rather than adding capacity.
- Capacity climbs monotonically from 5 (the orthogonal limit) to 20 as features get sparser.

**Confused about / open:**
- First runs used decaying importance for the 5→2 sweep and I got antipodal pairs, not the pentagon — briefly thought the replication had failed. It hadn't: the pentagon needs *uniform* importance so no feature is privileged. Added both regimes side by side; the contrast turned a bug into the clearest panel in the figure. Lesson logged: the geometry is set by the loss symmetry, so "which figure you reproduce" depends on the importance vector, not just the sparsity.
- The exact "features represented" integer count jitters run-to-run near the norm threshold. `Dᵢ` and the geometry are the stable readouts. Haven't done a seed sweep — single seed so far.

**Bridge thought:** this is mixed selectivity with ground truth. The controlling variable (feature sparsity) is the same natural-scene-statistics regime Olshausen & Field built V1 sparse coding on. Real open question for me: SAE-based interp often treats superposition as something to *undo* (disentangle to enumerate features), while Rigotti's mixed-selectivity argument says the same geometry can *ease* a linear readout. Both can hold — the field already studies computation in superposition, so this isn't about who noticed what. What a fully-observable model lets me do is *measure* whether the superposed geometry helps or hurts a downstream readout. That's the thread `experiments/02` takes up. Full argument in `experiments/01_.../writeup.md`.

**Next:**
- Multi-seed + finer sparsity grid to map the phase boundaries.
- Then Week 1 of the roadmap: write a transformer forward pass by hand, verify against TransformerLens.

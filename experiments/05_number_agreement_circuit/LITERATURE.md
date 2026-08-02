# Literature check: head-level ground truth for GPT-2-small subject–verb agreement

Date: 2026-08-02. Purpose: verify what published ground truth exists before pre-registering the
"which heads move subject-number information" experiment (readout: logit(" are") − logit(" is")).
Every claim below was checked against a source fetched this session; each is tagged
**[verified from source]** or **[could not verify]**. Quotes are verbatim from the fetched text
unless marked as paraphrase.

---

## 1. Finlayson et al. 2021 (ACL 2021) — the assumed baseline

Paper: "Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models",
ACL-IJCNLP 2021, pp. 1828–1843 (https://aclanthology.org/2021.acl-long.144/, https://arxiv.org/abs/2106.06087;
full text fetched via https://ar5iv.labs.arxiv.org/html/2106.06087).

- **Models** [verified from source]: DistilGPT-2, GPT-2 Small, GPT-2 Medium, GPT-2 Large, GPT-2 XL,
  plus Transformer-XL and XLNet. (https://ar5iv.labs.arxiv.org/html/2106.06087)
- **Granularity: neurons, not heads** [verified from source]. Method is causal mediation analysis with
  neuron-level natural indirect effects: "we independently analyze the individual neuron NIEs for GPT-2,
  Transformer-XL, and XLNet"; the intervention sets "a model component 𝐳 (e.g., a neuron) ... to the value
  it *would have taken* if the intervention had occurred." (https://ar5iv.labs.arxiv.org/html/2106.06087)
- **Attention heads: analyzed only in an appendix, and explicitly disclaimed** [verified from source]:
  - "we also attempt to analyze attention heads for GPT-2, though we find that they do not present
    consistent interpretable results" / "...do not present consistent interpretable results with the
    swap-number intervention".
  - Heads named descriptively for GPT-2 small: "Head 10-9 (layer 10, head 9) has negative indirect
    effects for most structures"; "Head 11-11 has the most consistently positive indirect effects across
    structures"; "Head 0-10 is always strongly implicated; since this is in the bottom layer...".
  - Critical caveat, quoted verbatim: "the sum of indirect effects across heads for most structures is
    close to 0, with many sums being a low-magnitude negative number. This indicates that these attention
    indirect effects may simply be noise."
  (all from https://ar5iv.labs.arxiv.org/html/2106.06087)
- **Where agreement is computed** [verified from source]: "both GPT-2 and Transformer-XL use two distinct
  mechanisms to accomplish subject-verb agreement, one of which is active only when the subject and verb
  are adjacent." Layer profile: for adjacent subject/verb "NIEs continually increase in higher layers";
  "for structures with subject-verb separation, NIEs peak at layer 0 and (more notably) in the
  upper-middle layers." Larger models: mechanisms "more distributed across layers."
  (https://ar5iv.labs.arxiv.org/html/2106.06087)

**Verdict on the design note's warning: the warning was right, and understated.** Finlayson et al. do
mention specific GPT-2-small heads (10.9, 11.11, 0.10), but only in an appendix that the authors
themselves flag as possibly pure noise. Any "specific head numbers remembered" from this paper cannot
be used as a comparison baseline — not because the memory might be wrong, but because the original
authors disclaim the head-level results.

---

## 2. Lakretz et al. — LSTM-only, no transformer head claims

- **Lakretz et al. 2019** (NAACL), "The emergence of number and syntax units in LSTM language models"
  [verified from source]: in an LSTM language model, "long-distance number information is largely managed
  by two 'number units'", whose behavior "is partially controlled by other units independently shown to
  track syntactic structure." LSTM only; "makes no claims about transformer attention heads."
  (https://arxiv.org/abs/1903.07435)
- **Lakretz et al. 2021** (Cognition), "Mechanisms for handling nested dependencies in neural-network
  language models and humans" [verified from source]: in an LSTM, "a very sparse set of specialized
  units ... successfully handled local and long-distance syntactic agreement for grammatical number",
  but the mechanism "does not support full recursion and fails with some long-range embedded
  dependencies"; humans stayed above chance on embedded long-range agreement where the model fell below
  chance. LSTM recurrent architecture; no transformer attention-head claims.
  (https://pubmed.ncbi.nlm.nih.gov/33941375/)

Both confirmed LSTM-only. Citable as the "sparse number-carrying units exist in LMs" precedent, not as
transformer head localization.

---

## 3. Other work localizing agreement in GPT-2-small (or nearby)

### 3a. Lepori, Serre & Pavlick — closest peer-reviewed causal localization for GPT-2-small

"Uncovering Intermediate Variables in Transformers using Circuit Probing" (https://arxiv.org/abs/2311.04354).
**Peer-review status: accepted at COLM 2024** [verified from source — listed on the COLM 2024 accepted
papers page: https://colmweb.org/2024/AcceptedPapers.html, entry links to
https://openreview.net/forum?id=gUNeyiLNxr].

- Case study on subject–verb agreement and reflexive anaphora in GPT2-Small and GPT2-Medium
  [verified from source, abstract: https://arxiv.org/abs/2311.04354].
- **Layer-level result** [verified from source]: "For both phenomena, we find that the dependency is
  computed in layer 6's attention block" (GPT-2-small). Causal support: "Ablating the circuit returned
  by circuit probing drops performance substantially for IID and OOD datasets for both phenomena, while
  ablating random subnetworks does not impact model performance."
  (https://ar5iv.labs.arxiv.org/html/2311.04354)
- **Head-level detail (appendix, secondary)** [verified from source]: "We also note that certain
  attention heads (0, 3, and 7) appear to be most important in computing syntactic number for both
  subject nouns and referents." These are heads within the layer-6 attention block per the main-text
  finding; note one figure caption reads "attention block 7", so 0- vs 1-based layer indexing should be
  re-checked against the paper's figures before quoting exact indices.
  (https://ar5iv.labs.arxiv.org/html/2311.04354)
- Granularity caveat: circuits are weight-masks at neuron level inside blocks ("mask parameters m ...
  tied across weights in a neuron", elementwise-multiplied with model weights), i.e. this is trained-mask
  circuit probing + ablation, not activation/path patching of whole heads. Comparable, not identical, to
  our planned intervention. (https://ar5iv.labs.arxiv.org/html/2311.04354)
- One fetch also reported early number-feature extraction in layer 0's MLP; I did not obtain a verbatim
  quote for that sentence — **[could not verify at quote level; re-check the paper before citing]**.

### 3b. Africa 2025 — a full 12-head GPT-2-small verb-conjugation circuit, NOT peer-reviewed

"Identifying a Circuit for Verb Conjugation in GPT-2", David Demitri Africa, arXiv:2506.22105
(June 2025). **Preprint; no peer-review venue found** [verified from source: https://arxiv.org/abs/2506.22105,
full text https://arxiv.org/html/2506.22105v1].

- Model/task: GPT-2 small; predict verb form for singular vs plural/compound subjects
  ("Alice" → walks; "Alice and Bob" → walk). Method: path patching with resample ablation; metric:
  logit difference (correct − incorrect verb) and accuracy. MLPs excluded by design: "The analysis
  focused exclusively on attention heads, omitting multi-layer perceptron (MLP) components entirely."
  (https://arxiv.org/html/2506.22105v1)
- Base circuit [verified from source, consistent across two independent fetches of the full text]:
  "This minimal circuit consists of only 12 attention heads spanning 7 layers ... This basic circuit
  achieves an accuracy of 0.65, which is close to the full model's accuracy of 0.70."
  Heads (layer, head): (11,6), (0,4), (11,4), (0,8), (11,7), (2,6), (1,0), (2,1), (1,1), (6,0), (10,0), (9,4).
  "Head 7 in layer 11 (bright yellow) exerts the most substantial influence on the model's predictions."
  Functional taxonomy: Primary Subject-Anchor (2,1),(2,6),(9,4),(10,0),(11,4),(11,6),(11,7);
  Diffuse Subject-Scanner (0,8),(1,0); Conjunction-Tracking (1,1); Invariant (0,4),(6,0).
  (https://arxiv.org/html/2506.22105v1)
- Note (0-indexed) (6,0) appears both here and in Ryu & Lewis's nsubj-specialized list (3d below), and
  layer 6 is Lepori's locus — a weak convergence worth checking in our own results.
- Caution: head list was extracted via automated summarization of the HTML twice with matching output,
  but **re-read the PDF by hand before hard-coding these 12 heads as a comparison baseline**.

### 3c. Marks et al. — SVA circuits, but Pythia/Gemma at SAE-feature granularity

"Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models",
Marks, Rager, Michaud, Belinkov, Bau, Mueller; abstract page states ICLR 2025
(https://arxiv.org/abs/2403.19647). [verified from source, full text https://arxiv.org/html/2403.19647]:
studies four SVA variants (simple "The parents" → is/are; within RC; across RC; across PP) — the same
is/are-style readout as our design — with circuits discovered in **Pythia-70M and Gemma-2-2B** over
**SAE features**, not attention heads; "does not name any GPT-2 attention heads."

### 3d. Ryu & Lewis 2021 (CMCL) — correlational GPT-2 head specialization; read from the PDF directly

"Accounting for Agreement Phenomena in Sentence Comprehension with Transformer Language Models"
(https://aclanthology.org/2021.cmcl-1.6/; PDF text extracted locally this session).

- [verified from source, PDF text]: Voita-style specialization analysis, **correlational, no
  patching/ablation**. "we obtained four syntactic heads that were found to be partly specialized for
  nsubj dependency relations: head4_3 (59%); head3_6 (51%); head6_0 (49%); head2_9 (49%)", best
  performing head4_3; notation "headn_m refers to the m-th attention head in the n-th layer".
  Reflexives: best head1_5.
- Model naming quirk [verified from source, PDF text]: they call it "the medium-sized GPT-2 which is
  constructed with 12 layers, each of which includes 12 attention heads" — 12L×12H is the 124M
  architecture standardly called GPT-2-small. Whether their indices are 0- or 1-based is not stated in
  the extracted text — **[could not verify indexing]**.
- Meta-note: two automated summaries of this paper returned two different, both-wrong head indices
  ("layer 3 head 10", "layer 8, head 10") before the PDF text settled it as head4_3 etc. This is a live
  demonstration of why the design note's "check against the original papers" rule exists.

### 3e. Checked and ruled out

- **Hanna et al. 2023 "greater-than"** (NeurIPS 2023, https://arxiv.org/abs/2305.00586)
  [verified from source]: year-comparison task only; "no mention of subject-verb agreement or grammatical
  number agreement." Useful as GPT-2-small circuit methodology precedent, not as an agreement baseline.
- **Ferrando & Costa-jussà, Findings of EMNLP 2024** (https://aclanthology.org/2024.findings-emnlp.591/)
  [verified from source]: SVA circuit "mainly driven by a particular attention head writing a 'subject
  number' signal to the last residual stream, which is read by a small set of neurons in the final MLPs"
  — but in **Gemma 2B**, not GPT-2. Valuable as a mechanistic hypothesis to test in GPT-2-small.
- **Mueller, Xia & Linzen, CoNLL 2022**, "Causal Analysis of Syntactic Agreement Neurons in Multilingual
  Language Models" (https://aclanthology.org/2022.conll-1.8/): neuron-level interventions in XGLM/mBERT-type
  models, per search-result listing — [verified only at listing level; anthology page itself not fetched].
- **Kumon & Yanaka 2026** (https://arxiv.org/html/2604.22166v1) [verified from source]: activation
  patching on syntactic phenomena, but filler-gap/NPI in Pythia/Gemma; no GPT-2-small SVA heads.
- **Finlayson search variants + Neuronpedia/blog search**: no peer-reviewed or community write-up naming
  a GPT-2-small number-agreement head circuit was found beyond the items above. Neuronpedia hosts
  GPT2-SMALL feature dashboards (https://www.neuronpedia.org/gpt2-small) but no number-agreement head
  page surfaced in searches — [absence of evidence; not exhaustively verified].

---

## 4. Bottom line for the design

**Answer: PARTIAL — no peer-reviewed head-level ground truth from activation/path patching exists for
GPT-2-small subject–verb agreement; layer-level causal ground truth does exist.**

What a pre-registration can legitimately cite as comparison points:

1. **Layer-level, peer-reviewed, causal (strongest priors):**
   - Finlayson et al. 2021 (ACL): neuron-NIE profile for GPT-2 — upper-middle-layer peak for separated
     subject–verb, monotonic late-layer increase for adjacent; predicts our patching effects should
     concentrate in upper-middle layers for long-distance items. (https://aclanthology.org/2021.acl-long.144/)
   - Lepori et al., COLM 2024: GPT-2-small SVA dependency computed in **layer 6's attention block**,
     with selective ablation evidence. This is the single most citable localization prior.
     (https://arxiv.org/abs/2311.04354)
2. **Head-level, but must be labeled non-peer-reviewed:** Africa 2025 (arXiv:2506.22105) 12-head
   path-patching circuit (dominant head layer 11 head 7; subject-anchor heads in layers 2/9/10/11).
   Usable as a "consistency check" target, not as ground truth.
3. **Head-level, peer-reviewed, but correlational only:** Ryu & Lewis 2021 (head4_3 nsubj-specialized).
4. **Must NOT be cited as head ground truth:** Finlayson's appendix heads (10.9, 11.11, 0.10) — the
   authors themselves state these "may simply be noise."

Contradictions with our design assumptions:
- If our design note's remembered head numbers trace to Finlayson, they are not merely unverified —
  they are affirmatively disclaimed by the source paper. The comparison-baseline slot should be filled
  by Lepori's layer-6 result plus Finlayson's layer profile instead.
- Positive reframe: the absence of peer-reviewed head-level patching ground truth means our experiment
  fills a real gap rather than replicating a known result; the pre-registration should say so, and
  predict against the layer-level priors (layer 6 attention; upper-middle-layer concentration).

"""Descriptive clean attention and frozen single-head effects on held-out stimuli."""
import argparse, datetime, json, re, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "post"))
from qf_transfer import verify
from stage2 import rows_for, head_patches, intervention
from patch import Harness
from stimuli import dump, HELDOUT_SEEDS

KEYWORDS = {
    "Q_N": ["previous quarter"],
    "Q_T": ["expect", "coming quarter"],
    "Q_F": ["expected", "coming quarter"],
}
LITERALS = ["last quarter", "guidance", "best estimate"]

def positions(row, tokenizer):
    enc = tokenizer(row["rendered"], add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == row["input_ids"]
    offsets = enc["offset_mapping"]
    qstart = row["rendered"].index(row["question"])
    spans = {}
    for label, phrases, question_only in [
        ("question_keywords", KEYWORDS[row["query_type"]], True),
        *[(phrase, [phrase], False) for phrase in LITERALS],
    ]:
        selected = set()
        matches = []
        text = row["question"] if question_only else row["rendered"]
        shift = qstart if question_only else 0
        for phrase in phrases:
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text, re.I):
                a, b = shift + m.start(), shift + m.end()
                matches.append([a, b])
                selected.update(i for i, (s, e) in enumerate(offsets) if s < b and e > a)
        spans[label] = dict(positions=sorted(selected), character_spans=matches)
    return spans

def attention(h, rows, heads):
    batch, pos = h.batch(rows)
    spans = [positions(row, h.tokenizer) for row in rows]
    outputs, handles = [], []
    for layer in sorted({l for l, _ in heads}):
        def hook(module, inputs, out, layer=layer):
            weights = out[1]
            assert weights is not None
            for head in [hh for ll, hh in heads if ll == layer]:
                for i, row in enumerate(rows):
                    offset = batch["input_ids"].shape[1] - len(row["input_ids"])
                    p = pos["prefill"][i][0]
                    sites = dict(question=row["question_positions"], **row["number_positions"])
                    sites.update({key: val["positions"] for key, val in spans[i].items()})
                    mass = {key: weights[i, head, p, [j + offset for j in ix]].float().sum().item()
                            if ix else 0. for key, ix in sites.items()}
                    outputs.append(dict(seed=row["seed"], item=row["item"],
                        query_type=row["query_type"], layer=layer, head=head,
                        mass=mass, token_count={k: len(v) for k, v in sites.items()}))
        handles.append(h.layers[layer].self_attn.register_forward_hook(hook))
    try:
        h.forward(batch, pos)
    finally:
        for handle in handles:
            handle.remove()
    return outputs

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--h7-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen, freeze_hash, h7_hash = verify(args)
    assert args.seed in HELDOUT_SEEDS
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (output / f"readout_{args.seed}.started").open("x") as f:
        f.write(stamp + "\n")
    result = dict(seed=args.seed, source_commit=args.expected_commit, freeze_sha256=freeze_hash,
        h7_summary_sha256=h7_hash, started_utc=stamp, attention=[], single_heads=[],
        keyword_definition=KEYWORDS, literal_phrases=LITERALS,
        note="Attention uses clean runs only. Single-head effects are descriptive interventions, with no selection or verdict.")
    path = output / f"readout_{args.seed}.json"
    dump(path, result)
    h = Harness("Qwen/Qwen3-4B")
    rows = rows_for(args, args.seed)
    for query in ["Q_N", "Q_T", "Q_F"]:
        selected = [r for r in rows if r["query_type"] == query]
        for start_idx in range(0, len(selected), 8):
            result["attention"] += attention(h, selected[start_idx:start_idx+8], frozen["heads"])
        dump(path, result)
    captures = [(l, "head", "prefill") for l in sorted({l for l, _ in frozen["heads"]})]
    data = {}
    for query in ["Q_N", "Q_T"]:
        rs = [r for r in rows if r["query_type"] == query]
        batch, pos = h.batch(rs)
        logits, cache = h.forward(batch, pos, captures=captures)
        data[query] = dict(rows=rs, batch=batch, pos=pos, cache=cache,
            outputs=[dict(d=(ll[r["digit_ids"]["T"]].float()-ll[r["digit_ids"]["N"]].float()).item())
                     for r, ll in zip(rs, logits)])
    for layer, head in frozen["heads"]:
        for b, d in [("Q_T", "Q_N"), ("Q_N", "Q_T")]:
            vals = intervention(h, data[b], head_patches(data[d]["cache"], [(layer, head)]), decode=False)
            deltas = [dict(item=r["item"], clean_d=clean["d"], patched_d=val["d"],
                           delta_d=val["d"]-clean["d"])
                      for r, clean, val in zip(data[b]["rows"], data[b]["outputs"], vals)]
            result["single_heads"].append(dict(layer=layer, head=head,
                direction=b+"->"+d[-1], items=deltas,
                mean_delta_d=sum(v["delta_d"] for v in deltas)/len(deltas)))
        dump(path, result)
    result.update(complete=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter()-start)
    dump(path, result)
    print("complete", args.seed, result["wall_seconds"], flush=True)

if __name__ == "__main__":
    main()

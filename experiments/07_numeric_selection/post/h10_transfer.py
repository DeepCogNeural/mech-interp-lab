"""H10 real-filing transfer using immutable heads and controls; no discovery."""
import argparse, collections, datetime, hashlib, json, random, subprocess, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "post"))
from qf_transfer import verify as verify_frozen
from stage2 import REPO, read, sha, head_patches, intervention
from patch import Harness
from stimuli import dump

BATCH = 8
PREREG = "ebd6f7d93274e9ee2decffc0cce08a04ed63e50f"

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify(args):
    frozen, freeze_hash, h7_hash = verify_frozen(args)
    path = HERE / "ADDENDUM_H10.md"
    committed = subprocess.check_output(["git", "show", f"{PREREG}:{path.relative_to(REPO)}"], cwd=REPO)
    assert path.read_bytes() == committed
    subprocess.run(["git", "merge-base", "--is-ancestor", PREREG, args.expected_commit], cwd=REPO, check=True)
    return frozen, freeze_hash, h7_hash

def render(item, query, tokenizer, idx):
    year = item["yearA"] if query == "Q_N" else item["yearB"]
    row = dict(item, item=idx, seed=20261018+item["group"], query_type=query,
               question=f'What was {item["item_name"]} in {year}? Answer with the number only.')
    content = row["context"]+"\n\n"+row["question"]
    rendered = tokenizer.apply_chat_template([{"role":"user", "content":content},
        {"role":"assistant","content":"$"}], tokenize=False, continue_final_message=True, enable_thinking=False)
    assert rendered.endswith("</think>\n\n$")
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    qstart = rendered.index(row["question"])
    question = [i for i,(a,b) in enumerate(offsets) if a < qstart+len(row["question"]) and b > qstart]
    assert question and tokenizer.decode([ids[-1]]) == "$"
    digits = {}
    for key in ["N", "T"]:
        display_value = row["display_values"][0 if key == "N" else 1]
        ext = tokenizer.encode(rendered+display_value, add_special_tokens=False)
        if ext[:len(ids)] != ids or tokenizer.decode([ext[len(ids)]]) != display_value[0]:
            raise ValueError("candidate single-digit rendering assertion")
        digits[key] = ext[len(ids)]
    offset = rendered.index(row["context"])
    numbers = {}
    for key,(a,b) in zip(["N","T"], row["amount_spans"]):
        numbers[key] = [i for i,(s,e) in enumerate(offsets) if s < offset+b and e > offset+a]
        assert numbers[key]
    return dict(row, rendered=rendered, input_ids=ids, prefill_position=len(ids)-1,
                question_positions=question, number_positions=numbers, digit_ids=digits)

def hash_packed_groups(items):
    """Pack whole documents in seeded hash order into four 64-item groups."""
    documents = collections.defaultdict(list)
    for item in items:
        documents[(item["company"].casefold(), item["document_sha256"])].append(item)
    remaining = sorted(documents, key=lambda k: hashlib.sha256(
        f"20261018|{k[0]}|{k[1]}".encode()).hexdigest())
    assignment = {}
    for group in range(4):
        reachable = {0: []}
        for document in remaining:
            weight = len(documents[document])
            for total, chosen in list(reachable.items()):
                new = total + weight
                if new <= 64 and new not in reachable:
                    reachable[new] = chosen + [document]
            if 64 in reachable:
                break
        assert 64 in reachable, "Insufficient document-disjoint candidates for 64-item packing"
        chosen = set(reachable[64])
        for document in chosen:
            assignment[document] = group
        remaining = [d for d in remaining if d not in chosen]
    return assignment

def prepare(args, frozen, freeze_hash, h7_hash):
    source = Path(args.candidates)
    candidates = read(source)
    assert not candidates["partial_inspection"]
    out = Path(args.output)
    assert not (out/"manifest.json").exists()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    selected, dropped = collections.defaultdict(list), []
    assignment = hash_packed_groups(candidates["candidates"])
    for original in candidates["candidates"]:
        document = (original["company"].casefold(), original["document_sha256"])
        if document not in assignment:
            continue
        item = dict(original, group=assignment[document])
        group = item["group"]
        if len(selected[group]) == 64: continue
        try:
            rows = [render(item, q, tokenizer, len(selected[group])) for q in ["Q_N", "Q_T"]]
        except ValueError as error:
            dropped.append(dict(id=item["id"], group=group, reason=str(error)))
            continue
        selected[group].append((item, rows))
    assert all(len(selected[g]) == 64 for g in range(4)), {g:len(selected[g]) for g in range(4)}
    doc_groups = {}
    groups = []
    for group in range(4):
        items, paired = zip(*selected[group])
        rows = [row for pair in paired for row in pair]
        documents = sorted({item["document_sha256"] for item in items})
        for document in documents:
            assert document not in doc_groups or doc_groups[document] == group
            doc_groups[document] = group
        name = f"group_{group}.json"
        groups.append(dict(group=group, filename=name, sha256=dump(out/name, rows),
            items=len(items), prompts=len(rows), documents=len(documents),
            companies=len({item["company"] for item in items}),
            numeric_formats=dict(collections.Counter(item["numeric_format"] for item in items)),
            min_tokens=min(len(r["input_ids"]) for r in rows),
            max_tokens=max(len(r["input_ids"]) for r in rows)))
    manifest = dict(seed=20261018, prereg_commit=PREREG, source_commit=args.expected_commit,
        freeze_sha256=freeze_hash, h7_summary_sha256=h7_hash, created_utc=now(),
        source_train_sha256=candidates["source_sha256"], candidates_sha256=sha(source),
        extraction_counts=candidates["counts"], initial_hash_mod4_candidate_counts=candidates["candidate_group_counts"],
        extraction_limitations=candidates["extraction_limitations"],
        candidate_rendering_drops=dropped, candidate_rendering_drop_count=len(dropped),
        groups=groups, document_count=len(doc_groups), document_groups=doc_groups,
        selection="Sort whole documents by SHA256(seed|casefold(company)|normalized_document_sha256). For each of four groups, scan remaining documents in this order; deterministic subset-sum keeps the first encountered document subset totaling exactly 64 candidates. Remove those documents and repeat. Order items by SHA256(seed|candidate_id). No model-output filtering.",
        direction_definition="A/N=first mentioned amount/year; B/T=second mentioned amount/year.",
        primary_parser="unchanged scoring.parse_integer", max_new_tokens=8, batch_size=BATCH)
    digest = dump(out/"manifest.json", manifest)
    (out/"manifest.sha256").write_text(digest+"\n")
    print(json.dumps(manifest, indent=2))

def manifest_rows(args):
    root = Path(args.output)
    digest = sha(root/"manifest.json")
    assert digest == (root/"manifest.sha256").read_text().strip()
    manifest = read(root/"manifest.json")
    group = next(g for g in manifest["groups"] if g["group"] == args.group)
    path = root/group["filename"]
    assert sha(path) == group["sha256"]
    return read(path), digest

def reserve(root, name):
    with (root/(name+".started")).open("x") as f: f.write(now()+"\nOne attempt.\n")

def clean(args, frozen, freeze_hash, h7_hash):
    start = time.perf_counter()
    root = Path(args.output)
    rows, digest = manifest_rows(args)
    reserve(root, f"clean_{args.group}")
    result = dict(group=args.group, manifest_sha256=digest, freeze_sha256=freeze_hash,
        source_commit=args.expected_commit, started_utc=now(), outputs={})
    path = root/f"clean_{args.group}.json"
    dump(path,result)
    h = Harness("Qwen/Qwen3-4B")
    captures = [(l,"head","prefill") for l in sorted({l for l,_ in frozen["heads"]})]
    allcache = {}
    for query in ["Q_N", "Q_T"]:
        rs = [r for r in rows if r["query_type"] == query]
        outputs, cache = [], {key:[] for key in captures}
        for idx in range(0,len(rs),BATCH):
            values, captured = h.measure(rs[idx:idx+BATCH], captures=captures)
            outputs += values
            for key in captures: cache[key] += [v.cpu() for v in captured[key]]
        result["outputs"][query] = outputs
        allcache[query] = cache
        dump(path,result)
    cache_path = root/f"clean_cache_{args.group}.pt"
    torch.save(allcache, cache_path)
    result.update(complete=True, cache_sha256=sha(cache_path), finished_utc=now(),
                  wall_seconds=time.perf_counter()-start)
    dump(path,result)

def behaviour_gate(args):
    root = Path(args.output)
    runs = [read(root/f"clean_{g}.json") for g in range(4)]
    digest = sha(root/"manifest.json")
    assert all(r.get("complete") and r["manifest_sha256"] == digest for r in runs)
    entries = []
    for query in ["Q_N","Q_T"]:
        values = [v for r in runs for v in r["outputs"][query]]
        correct = sum(v["answer"] == v[query[-1]] for v in values)
        entries.append(dict(query=query, correct=correct, total=len(values),
            accuracy=correct/len(values),
            first_generated_is_digit=sum(v["first_is_digit"] for v in values)))
    passed = all(v["accuracy"] >= .8 for v in entries)
    result = dict(manifest_sha256=digest, query_results=entries, passed=passed,
        pooled_accuracy=sum(v["correct"] for v in entries)/sum(v["total"] for v in entries),
        verdict="eligible for H10 patch test" if passed else "not testable on real text with this model",
        clean_file_sha256={str(g):sha(root/f"clean_{g}.json") for g in range(4)},
        computed_utc=now())
    dump(root/"behaviour_gate.json",result)
    print(json.dumps(result,indent=2))

def chunk_intervention(h, base, patches):
    outputs=[]
    for idx in range(0,len(base["rows"]),BATCH):
        rs=base["rows"][idx:idx+BATCH]
        batch,pos=h.batch(rs)
        subset={k:dict(heads=v["heads"], values=v["values"][idx:idx+BATCH]) for k,v in patches.items()}
        outputs += intervention(h, dict(rows=rs,batch=batch,pos=pos), subset)
    return outputs

def metric(base, values, target, donor_rows=None, order=None):
    gold=base["rows"][0]["query_type"][-1]
    eligible=[i for i,v in enumerate(base["outputs"]) if v["answer"]==v[gold]]
    denominator=len(eligible)
    flipped=sum(values[i]["answer"]==values[i][target] for i in eligible)
    valid=sum(values[i]["answer"] in [values[i]["N"],values[i]["T"]] for i in eligible)
    result=dict(denominator=denominator, flip_count=flipped,
        flip_rate=flipped/denominator if denominator else None,
        valid_count=valid, validity=valid/denominator if denominator else None,
        all_item_validity=sum(v["answer"] in [v["N"],v["T"]] for v in values)/len(values),
        mean_delta_d=sum(v["d"]-b["d"] for v,b in zip(values,base["outputs"]))/len(values),
        outputs=values)
    if donor_rows is not None:
        injected=sum(values[i]["answer"] in [donor_rows[order[i]]["N"],donor_rows[order[i]]["T"]] for i in eligible)
        result.update(donor_order=order, donor_number_injection_count=injected,
                      donor_number_injection_rate=injected/denominator if denominator else None)
    return result

def run(args,frozen,freeze_hash,h7_hash):
    started=time.perf_counter();root=Path(args.output)
    rows,digest=manifest_rows(args)
    gate=read(root/"behaviour_gate.json")
    assert gate["passed"] and gate["manifest_sha256"]==digest
    clean_run=read(root/f"clean_{args.group}.json")
    assert sha(root/f"clean_{args.group}.json")==gate["clean_file_sha256"][str(args.group)]
    assert sha(root/f"clean_cache_{args.group}.pt")==clean_run["cache_sha256"]
    reserve(root,f"patch_{args.group}")
    result=dict(group=args.group,manifest_sha256=digest,freeze_sha256=freeze_hash,
        source_commit=args.expected_commit,started_utc=now(),sets=[],controls=[],
        random_vector_seed=20261018+args.group+20261105)
    path=root/f"patch_{args.group}.json";dump(path,result)
    h=Harness("Qwen/Qwen3-4B")
    allcache=torch.load(root/f"clean_cache_{args.group}.pt",weights_only=True,map_location="cuda")
    data={q:dict(rows=[r for r in rows if r["query_type"]==q],
                outputs=clean_run["outputs"][q],cache=allcache[q]) for q in ["Q_N","Q_T"]}
    for index,heads in enumerate([frozen["heads"]]+frozen["random_sets"]):
        entry=dict(index=index,label="selected" if index==0 else "layer-matched random",directions=[])
        for b,d,target in [("Q_T","Q_N","N"),("Q_N","Q_T","T")]:
            values=chunk_intervention(h,data[b],head_patches(data[d]["cache"],heads))
            entry["directions"].append(dict(direction=b+"->"+target,**metric(data[b],values,target)))
        result["sets"].append(entry);dump(path,result)
        print("H10",args.group,"set",index,flush=True)
    for b,d,target in [("Q_T","Q_N","N"),("Q_N","Q_T","T")]:
        base,donor=data[b],data[d]
        order=list(range(len(rows)//2));rng=random.Random(result["random_vector_seed"])
        for _ in range(10000):
            rng.shuffle(order)
            if all(i!=j and not ({base["rows"][i]["N"],base["rows"][i]["T"]}&
                                  {donor["rows"][j]["N"],donor["rows"][j]["T"]}) for i,j in enumerate(order)): break
        else: raise RuntimeError("No mismatched derangement")
        cache={key:[v[j] for j in order] for key,v in donor["cache"].items()}
        values=chunk_intervention(h,base,head_patches(cache,frozen["heads"]))
        result["controls"].append(dict(kind="mismatched donor",direction=b+"->"+target,
            **metric(base,values,target,donor["rows"],order)))
        generator=torch.Generator(device="cuda").manual_seed(result["random_vector_seed"])
        cache={key:[v.clone() for v in vs] for key,vs in base["cache"].items()}
        for layer,head in frozen["heads"]:
            key=(layer,"head","prefill");sl=slice(head*h.head_dim,(head+1)*h.head_dim)
            for i in range(len(order)):
                delta=(donor["cache"][key][i][:,sl]-base["cache"][key][i][:,sl]).float()
                noise=torch.randn(delta.shape,device="cuda",generator=generator)
                noise=noise/noise.norm().clamp_min(1e-12)*delta.norm()
                cache[key][i][:,sl]+=noise.to(torch.float16)
        values=chunk_intervention(h,base,head_patches(cache,frozen["heads"]))
        result["controls"].append(dict(kind="matched-norm random delta",direction=b+"->"+target,
            **metric(base,values,target)))
        dump(path,result)
    result.update(complete=True,finished_utc=now(),wall_seconds=time.perf_counter()-started)
    dump(path,result)

def aggregate(args):
    root=Path(args.output);gate=read(root/"behaviour_gate.json")
    result=dict(behaviour_gate=gate,manifest_sha256=sha(root/"manifest.json"),per_group=[],directions=[])
    if not gate["passed"]:
        result["verdict"]=gate["verdict"];result["passes"]=None
    else:
        for group in range(4):
            run=read(root/f"patch_{group}.json")
            assert run.get("complete") and len(run["sets"])==101
            assert run["manifest_sha256"]==result["manifest_sha256"]
            for selected in run["sets"][0]["directions"]:
                direction=selected["direction"]
                randoms=[next(x for x in s["directions"] if x["direction"]==direction) for s in run["sets"][1:]]
                maximum=max(r["flip_rate"] for r in randoms)
                result["per_group"].append(dict(group=group,
                    **{k:v for k,v in selected.items() if k!="outputs"},
                    random_max=maximum, above_all_random=selected["flip_rate"]>maximum,
                    random_validity_min=min(r["validity"] for r in randoms),
                    random_validity_max=max(r["validity"] for r in randoms),
                    controls=[{k:v for k,v in c.items() if k not in ["outputs","donor_order"]}
                              for c in run["controls"] if c["direction"]==direction]))
        for direction in ["Q_T->N","Q_N->T"]:
            rows=[r for r in result["per_group"] if r["direction"]==direction]
            mean=sum(r["flip_rate"] for r in rows)/4
            result["directions"].append(dict(direction=direction,mean_flip=mean,
                passes=mean>=.5 and all(r["above_all_random"] for r in rows)))
        result["passes"]=all(r["passes"] for r in result["directions"])
        result["verdict"]="pass" if result["passes"] else "fail"
    result["computed_utc"]=now()
    dump(root/"h10_summary.json",result)
    print(json.dumps(result,indent=2))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["prepare","clean","gate","run","aggregate"])
    p.add_argument("--group",type=int,choices=range(4))
    p.add_argument("--candidates")
    p.add_argument("--freeze",required=True)
    p.add_argument("--expected-commit",required=True)
    p.add_argument("--stimuli",required=True)
    p.add_argument("--h7-output",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args();frozen,freeze_hash,h7_hash=verify(args)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    if args.phase=="prepare": prepare(args,frozen,freeze_hash,h7_hash)
    elif args.phase=="clean": clean(args,frozen,freeze_hash,h7_hash)
    elif args.phase=="gate": behaviour_gate(args)
    elif args.phase=="run": run(args,frozen,freeze_hash,h7_hash)
    else: aggregate(args)

if __name__=="__main__": main()

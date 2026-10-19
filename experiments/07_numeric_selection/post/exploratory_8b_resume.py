"""Resume the completed 8B dev discovery after a pre-heldout assembly error."""
import argparse, collections, datetime, hashlib, json, random, time
from pathlib import Path
import torch
from exploratory_8b import (ShardedHarness, data_for, now, verify, read, sha,
                           dump, DEV_SEEDS, HELDOUT_SEEDS, head_patches,
                           intervention, effects)

def discovery_hash(result):
    raw=json.dumps({k:result[k] for k in ["residual","head_scan"]},sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ["freeze","expected-commit","stimuli","h7-output","output","stage3-root"]:
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    frozen,freeze_hash,h7_hash=verify(args)
    out=Path(args.output);path=out/"exploratory_8b.json"
    result=read(path)
    assert not result.get("complete") and not result["heldout"]
    assert result["dev_seeds"]==list(DEV_SEEDS)
    assert result["heldout_seeds"]==list(HELDOUT_SEEDS[:2])
    assert not list(out.glob("heldout_8b_*.json"))
    assert not (out/"exploratory_8b_freeze.json").exists()
    assert result["unchanged_4B_freeze_sha256"]==freeze_hash
    layers=result["head_scan_layers"];lstar=result["L_star"]
    assert layers==list(range(max(0,lstar-3),lstar+1))
    directions=["Q_T->N","Q_N->T"]
    expected_residual={(seed,layer,d) for seed in DEV_SEEDS
                       for layer in range(36) for d in directions}
    assert len(result["residual"])==len(expected_residual)
    assert {(r["seed"],r["layer"],r["direction"]) for r in result["residual"]}==expected_residual
    head_ids=sorted({r["head"] for r in result["head_scan"]})
    assert head_ids==list(range(len(head_ids)))
    expected_heads={(seed,layer,head,d) for seed in DEV_SEEDS
                    for layer in layers for head in head_ids for d in directions}
    assert len(result["head_scan"])==len(expected_heads)
    assert {(r["seed"],r["layer"],r["head"],r["direction"])
            for r in result["head_scan"]}==expected_heads
    with (out/"recovery.started").open("x") as f:f.write(now()+"\n")
    started=time.perf_counter()
    discovery_digest=discovery_hash(result)
    result["recovery"]=dict(
        reason="Random-set assembly stopped with UnboundLocalError from shadowing the model variable h.",
        decision="Resume from all saved dev scan rows; do not rerun discovery. No 8B heldout call preceded this recovery.",
        discovery_source_commit=result["source_commit"],
        recovery_source_commit=args.expected_commit,
        started_utc=now(),discovery_arrays_sha256=discovery_digest)
    scores=collections.defaultdict(list)
    for r in result["head_scan"]:
        scores[(r["layer"],r["head"])].append(r["oriented_delta_d"])
    ranking=sorted(scores,key=lambda k:(-sum(scores[k])/len(scores[k]),k))
    selected=ranking[:8];histogram=collections.Counter(layer for layer,_ in selected)
    rng=random.Random(20261018);seen={tuple(sorted(selected))};random_sets=[]
    while len(random_sets)<20:
        candidate=tuple(sorted((layer,head_id) for layer,n in sorted(histogram.items())
                               for head_id in rng.sample(head_ids,n)))
        if candidate in seen:continue
        seen.add(candidate);random_sets.append(candidate)
    freeze=dict(exploratory=True,heads=selected,random_sets=random_sets,random_seed=20261018,
        L_star=lstar,head_scan_layers=layers,dev_seeds=DEV_SEEDS,heldout_seeds=HELDOUT_SEEDS[:2],
        ranking=[dict(layer=layer,head=head,mean_oriented_delta_d=sum(scores[(layer,head)])/len(scores[(layer,head)]))
                 for layer,head in ranking],
        created_before_8B_heldout_utc=now(),source_commit=args.expected_commit,
        discovery_source_commit=result["source_commit"],discovery_arrays_sha256=discovery_digest)
    result["exploratory_freeze_sha256"]=dump(out/"exploratory_8b_freeze.json",freeze)
    result["selected_heads"]=selected
    dump(path,result)
    h=ShardedHarness()
    assert h.num_heads==len(head_ids)
    assert set(h.model.hf_device_map.values())==set(range(4))
    for seed in HELDOUT_SEEDS[:2]:
        data=data_for(h,args,seed,["head"],sorted(histogram))
        entry=dict(seed=seed,source_commit=args.expected_commit,
            started_utc=now(),clean={q:v["outputs"] for q,v in data.items()},sets=[])
        for index,heads in enumerate([selected]+random_sets):
            measured=[]
            for base,donor,target in [("Q_T","Q_N","N"),("Q_N","Q_T","T")]:
                values=intervention(h,data[base],head_patches(data[donor]["cache"],heads))
                measured.append(dict(direction=base+"->"+target,outputs=values,
                    **effects(data[base],values,target)))
            entry["sets"].append(dict(index=index,directions=measured))
            dump(out/f"heldout_8b_{seed}.json",entry)
            print("8B heldout",seed,index,flush=True)
        entry.update(complete=True,finished_utc=now())
        dump(out/f"heldout_8b_{seed}.json",entry)
        result["heldout"].append(dict(seed=seed,sha256=sha(out/f"heldout_8b_{seed}.json"),
            directions=[dict(direction=d["direction"],flip_rate=d["flip_rate"],validity=d["validity"],
                random_max=max(next(x["flip_rate"] for x in s["directions"] if x["direction"]==d["direction"])
                               for s in entry["sets"][1:]),
                clean_correct_count=d["clean_correct_count"]) for d in entry["sets"][0]["directions"]]))
        dump(path,result)
    assert discovery_hash(result)==discovery_digest
    end=now()
    result["recovery"].update(finished_utc=end,wall_seconds=time.perf_counter()-started)
    result.update(complete=True,finished_utc=end,
        wall_seconds=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(result["started_utc"])).total_seconds())
    dump(path,result)

if __name__=="__main__":main()

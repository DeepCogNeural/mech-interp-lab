"""Exploratory Qwen3-8B size check. Separate discovery; no 4B verdict changes."""
import argparse, collections, datetime, json, random, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/"post"))
from qf_transfer import verify
from patch import Harness
from stage2 import rows_for, head_patches, intervention, effects, read, sha
from stimuli import DEV_SEEDS, HELDOUT_SEEDS, dump, render

class ShardedHarness(Harness):
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-8B",local_files_only=True)
        self.tokenizer.padding_side="left"
        self.model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",local_files_only=True,
            dtype=torch.float16,attn_implementation="eager",device_map="balanced",
            max_memory={i:"10GiB" for i in range(4)}).eval()
        self.layers=self.model.model.layers
        self.head_dim=self.model.config.head_dim;self.num_heads=self.model.config.num_attention_heads
        self.finite_flags=[];self.handles=[]
        for layer in self.layers:
            for module in [layer,layer.self_attn,layer.mlp]:
                self.handles.append(module.register_forward_hook(self._finite_hook))
    def _finite_hook(self,module,args,output):
        x=output[0] if isinstance(output,tuple) else output
        self.finite_flags.append(torch.isfinite(x).all().to("cuda:0"))
    @torch.inference_mode()
    def forward(self,batch,positions,captures=(),patches=None):
        self.finite_flags=[]
        with self.hooks(positions,captures=captures,patches=patches) as cache:
            position_ids=batch["attention_mask"].long().cumsum(-1)-1
            position_ids.masked_fill_(batch["attention_mask"]==0,0)
            result=self.model(**batch,position_ids=position_ids,use_cache=False,logits_to_keep=1)
            logits=result.logits[:,-1,:].to("cuda:0")
            self.finite_flags.append(torch.isfinite(logits).all())
            if not torch.stack([v.to("cuda:0") for v in self.finite_flags]).all().item():
                raise FloatingPointError("STOP: non-finite exploratory fp16 run")
        return logits,cache

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def data_for(h,args,seed,components,layers=None):
    original=rows_for(args,seed)
    rows=[render(r,h.tokenizer) for r in original if r["query_type"] in ["Q_N","Q_T"]]
    layers=range(len(h.layers)) if layers is None else layers
    captures=[(l,c,"prefill") for l in layers for c in components]
    data={}
    for q in ["Q_N","Q_T"]:
        rs=[r for r in rows if r["query_type"]==q]
        values,cache=h.measure(rs,captures=captures)
        batch,pos=h.batch(rs)
        data[q]=dict(rows=rs,outputs=values,cache=cache,batch=batch,pos=pos)
    return data

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--freeze",required=True);p.add_argument("--expected-commit",required=True)
    p.add_argument("--stimuli",required=True);p.add_argument("--h7-output",required=True)
    p.add_argument("--output",required=True);p.add_argument("--stage3-root",required=True)
    args=p.parse_args()
    frozen,freeze_hash,h7_hash=verify(args)
    root=Path(args.stage3_root)
    assert (root/"h10"/"h10_summary.json").exists()
    assert len(read(root/"circuit_summary.json")["rows"])==24
    assert any(v["hypothesis"]=="H10" for v in read(root/"secondary_parse.json")["totals"])
    deadline=datetime.datetime(2026,10,19,2,30,tzinfo=datetime.timezone.utc)
    remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert remaining>=90*60, "Optional check requires at least 90 minutes remaining"
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/"exploratory.started").open("x") as f:f.write(now()+"\n")
    start=time.perf_counter()
    result=dict(exploratory=True,model="Qwen/Qwen3-8B",source_commit=args.expected_commit,
        unchanged_4B_freeze_sha256=freeze_hash,started_utc=now(),remaining_seconds_at_start=remaining,
        dev_seeds=DEV_SEEDS,heldout_seeds=HELDOUT_SEEDS[:2],residual=[],head_scan=[],heldout=[],
        interpretation="Exploratory size check, not replication. No frozen 4B verdict is changed.")
    path=out/"exploratory_8b.json";dump(path,result)
    h=ShardedHarness()
    result["device_map"]={k:str(v) for k,v in h.model.hf_device_map.items()}
    assert set(h.model.hf_device_map.values())==set(range(4)), h.model.hf_device_map
    all_data={}
    for seed in DEV_SEEDS:
        data=data_for(h,args,seed,["residual","head"]);all_data[seed]=data
        dump(out/f"dev_clean_{seed}.json",{q:v["outputs"] for q,v in data.items()})
        for layer in range(len(h.layers)):
            key=(layer,"residual","prefill")
            for b,d,target in [("Q_T","Q_N","N"),("Q_N","Q_T","T")]:
                values=intervention(h,data[b],{key:dict(values=data[d]["cache"][key])})
                result["residual"].append(dict(seed=seed,layer=layer,direction=b+"->"+target,
                    outputs=values,**effects(data[b],values,target)))
            dump(path,result)
            print("8B residual",seed,layer,flush=True)
    rates=[];lstar=None
    for layer in range(len(h.layers)):
        directional={direction:sum(r["flip_rate"] for r in result["residual"]
            if r["layer"]==layer and r["direction"]==direction)/len(DEV_SEEDS)
            for direction in ["Q_T->N","Q_N->T"]}
        rates.append(dict(layer=layer,**directional))
        if lstar is None and min(directional.values())>=.8:lstar=layer
    result["residual_rates"]=rates;result["L_star"]=lstar;dump(path,result)
    if lstar is None:
        result.update(complete=True,stop_reason="No residual L_star met 80% in both directions",
            finished_utc=now(),wall_seconds=time.perf_counter()-start);dump(path,result);return
    layers=list(range(max(0,lstar-3),lstar+1));result["head_scan_layers"]=layers
    scores=collections.defaultdict(list)
    for layer in layers:
        for head in range(h.num_heads):
            for seed in DEV_SEEDS:
                data=all_data[seed]
                for b,d,target in [("Q_T","Q_N","N"),("Q_N","Q_T","T")]:
                    values=intervention(h,data[b],head_patches(data[d]["cache"],[(layer,head)]),decode=False)
                    measurement=effects(data[b],values,target)
                    scores[(layer,head)].append(measurement["oriented_delta_d"])
                    result["head_scan"].append(dict(seed=seed,layer=layer,head=head,
                        direction=b+"->"+target,**measurement))
            dump(path,result)
        print("8B head layer",layer,flush=True)
    ranking=sorted(scores,key=lambda k:(-sum(scores[k])/len(scores[k]),k))
    selected=ranking[:8]
    histogram=collections.Counter(l for l,_ in selected)
    rng=random.Random(20261018);seen={tuple(sorted(selected))};random_sets=[]
    while len(random_sets)<20:
        candidate=tuple(sorted((l,head_id) for l,n in sorted(histogram.items())
                               for head_id in rng.sample(range(h.num_heads),n)))
        if candidate in seen:continue
        seen.add(candidate);random_sets.append(candidate)
    exploratory_freeze=dict(exploratory=True,heads=selected,random_sets=random_sets,random_seed=20261018,
        L_star=lstar,head_scan_layers=layers,dev_seeds=DEV_SEEDS,heldout_seeds=HELDOUT_SEEDS[:2],
        ranking=[dict(layer=l,head=hh,mean_oriented_delta_d=sum(scores[(l,hh)])/len(scores[(l,hh)]))
                 for l,hh in ranking],created_before_8B_heldout_utc=now(),source_commit=args.expected_commit)
    result["exploratory_freeze_sha256"]=dump(out/"exploratory_8b_freeze.json",exploratory_freeze)
    result["selected_heads"]=selected;dump(path,result)
    del all_data
    torch.cuda.empty_cache()
    for seed in HELDOUT_SEEDS[:2]:
        data=data_for(h,args,seed,["head"],sorted(histogram))
        entry=dict(seed=seed,clean={q:v["outputs"] for q,v in data.items()},sets=[])
        for index,heads in enumerate([selected]+random_sets):
            values_by_direction=[]
            for b,d,target in [("Q_T","Q_N","N"),("Q_N","Q_T","T")]:
                values=intervention(h,data[b],head_patches(data[d]["cache"],heads))
                values_by_direction.append(dict(direction=b+"->"+target,outputs=values,
                    **effects(data[b],values,target)))
            entry["sets"].append(dict(index=index,directions=values_by_direction))
            dump(out/f"heldout_8b_{seed}.json",entry)
            print("8B heldout",seed,index,flush=True)
        entry["complete"]=True;dump(out/f"heldout_8b_{seed}.json",entry)
        result["heldout"].append(dict(seed=seed,sha256=sha(out/f"heldout_8b_{seed}.json"),
            directions=[dict(direction=d["direction"],flip_rate=d["flip_rate"],validity=d["validity"],
                random_max=max(next(x["flip_rate"] for x in s["directions"] if x["direction"]==d["direction"])
                               for s in entry["sets"][1:]),
                clean_correct_count=d["clean_correct_count"]) for d in entry["sets"][0]["directions"]]))
        dump(path,result)
    result.update(complete=True,finished_utc=now(),wall_seconds=time.perf_counter()-start)
    dump(path,result)
if __name__=="__main__":main()

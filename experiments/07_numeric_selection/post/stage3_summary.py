"""Saved-output post-hoc parse and descriptive readout aggregation; never changes verdicts."""
import argparse, collections, hashlib, json, re, statistics, time
from pathlib import Path
SEEDS=list(range(20261201,20261209))
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def first_integer(text):
    # Relax only the unique-number requirement. Do not truncate comma/decimal lexemes.
    values=re.findall(r"[+-]?\d+(?:[.,]\d+)*",text)
    if not values or re.fullmatch(r"[+-]?\d+",values[0]) is None: return None
    return int(values[0])
def stats(outputs,target,indices):
    selected=[outputs[i] for i in indices]
    original=[v["answer"] for v in selected]
    first=[first_integer(v["text"]) for v in selected]
    n=len(selected)
    return dict(n=n,unique_count=sum(a==v[target] for a,v in zip(original,selected)),
        first_count=sum(a==v[target] for a,v in zip(first,selected)),
        unique_valid=sum(a in [v["N"],v["T"]] for a,v in zip(original,selected)),
        first_valid=sum(a in [v["N"],v["T"]] for a,v in zip(first,selected)),
        changed=sum(a!=b for a,b in zip(original,first)),
        unique_rate=sum(a==v[target] for a,v in zip(original,selected))/n if n else None,
        first_rate=sum(a==v[target] for a,v in zip(first,selected))/n if n else None)

def secondary(root):
    records=[]; hashes={}
    def add(hyp,seed,condition,direction,outputs,target,indices=None,set_index=None):
        records.append(dict(hypothesis=hyp,group_or_seed=seed,condition=condition,direction=direction,
            set_index=set_index,**stats(outputs,target,list(range(len(outputs))) if indices is None else indices)))
    for seed in SEEDS:
        path=root/"stage2"/f"heldout_{seed}.json";h7=read(path);hashes[str(path)]=sha(path)
        for q in ["Q_N","Q_T"]:
            outputs=[v for v in h7["clean"] if v["query_type"]==q]
            add("H7",seed,"clean accuracy",q,outputs,q[-1])
        for entry in h7["sets"]:
            for kind in ["sufficiency","necessity"]:
                for rec in entry[kind]:
                    direction=rec.get("direction",rec["outputs"][0]["query_type"])
                    target=direction[-1] if kind=="sufficiency" else rec["outputs"][0]["query_type"][-1]
                    add("H7",seed,("selected " if entry["index"]==0 else "random ")+kind,
                        direction,rec["outputs"],target,set_index=entry["index"])
        for rec in h7["extra_controls"]:
            add("H7",seed,rec["kind"],rec["direction"],rec["outputs"],rec["direction"][-1])
        path=root/"qf_transfer"/f"qf_{seed}.json";h9=read(path);hashes[str(path)]=sha(path)
        add("H9",seed,"clean accuracy","Q_F",h9["clean_QF"],"T")
        for rec in h9["sets"]:
            add("H9",seed,"selected transfer" if rec["index"]==0 else "random transfer",
                rec["direction"],rec["outputs"],"N",set_index=rec["index"])
        for rec in h9["extra_controls"]:
            add("H9",seed,rec["kind"],rec["direction"],rec["outputs"],"N")
    h10=root/"stage3"/"h10"
    if (h10/"behaviour_gate.json").exists():
        for group in range(4):
            path=h10/f"clean_{group}.json";clean=read(path);hashes[str(path)]=sha(path)
            for q,values in clean["outputs"].items(): add("H10",group,"clean accuracy",q,values,q[-1])
            path=h10/f"patch_{group}.json"
            if not path.exists():continue
            run=read(path);hashes[str(path)]=sha(path)
            for entry in run["sets"]:
                for rec in entry["directions"]:
                    query=rec["direction"].split("->")[0]
                    ix=[i for i,v in enumerate(clean["outputs"][query]) if v["answer"]==v[query[-1]]]
                    add("H10",group,"selected transfer" if entry["index"]==0 else "random transfer",
                        rec["direction"],rec["outputs"],rec["direction"][-1],ix,entry["index"])
            for rec in run["controls"]:
                query=rec["direction"].split("->")[0]
                ix=[i for i,v in enumerate(clean["outputs"][query]) if v["answer"]==v[query[-1]]]
                add("H10",group,rec["kind"],rec["direction"],rec["outputs"],rec["direction"][-1],ix)
    groups=collections.defaultdict(list)
    for rec in records:
        groups[(rec["hypothesis"],rec["condition"],rec["direction"],rec["set_index"])].append(rec)
    totals=[]
    for (hyp,cond,direction,index),rows in groups.items():
        entry=dict(hypothesis=hyp,condition=cond,direction=direction,set_index=index)
        for key in ["n","unique_count","first_count","unique_valid","first_valid","changed"]:
            entry[key]=sum(r[key] for r in rows)
        for key in ["unique_rate","first_rate"]:
            entry["mean_"+key]=statistics.mean(r[key] for r in rows if r[key] is not None)
        totals.append(entry)
    frozen_h7=read(root/"stage2"/"heldout_summary.json")
    frozen_h9=read(root/"qf_transfer"/"qf_summary.json")
    return dict(label="post hoc; saved outputs only; no verdict is computed or altered",
        parser_definition="Take the first numeric lexeme matching [+-]?digits([.,]digits)*; accept only an unformatted signed integer. Multiple later numeric lexemes are ignored. Decimal/comma lexemes are not truncated.",
        original_H7_H9_verdicts_copied=dict(H7_sufficiency=frozen_h7["H7_sufficiency_passes"],
            H7_necessity=frozen_h7["necessity"],H9=frozen_h9["passes"]),
        source_sha256=hashes,per_seed=records,totals=totals)

def readouts(root):
    attention=collections.defaultdict(list);effects=collections.defaultdict(list);hashes={};runs=[]
    for seed in SEEDS:
        path=root/"stage3"/"readouts"/f"readout_{seed}.json"
        run=read(path);hashes[str(path)]=sha(path);assert run["complete"];runs.append(run)
        buckets=collections.defaultdict(list)
        for v in run["attention"]:buckets[(v["layer"],v["head"],v["query_type"])].append(v)
        for key,values in buckets.items():
            attention[key].append(dict(seed=seed,n=len(values),
                mass={k:statistics.mean(v["mass"][k] for v in values) for k in values[0]["mass"]},
                token_counts={k:sorted({v["token_count"][k] for v in values}) for k in values[0]["token_count"]}))
        for v in run["single_heads"]:
            effects[(v["layer"],v["head"],v["direction"])].append(dict(seed=seed,mean_delta_d=v["mean_delta_d"]))
    rows=[]
    for (l,h,q),values in attention.items():
        direction={"Q_N":"Q_N->T","Q_T":"Q_T->N","Q_F":None}[q]
        e=effects[(l,h,direction)] if direction else []
        rows.append(dict(layer=l,head=h,query=q,
            mass={k:statistics.mean(v["mass"][k] for v in values) for k in values[0]["mass"]},
            mean_delta_d=statistics.mean(v["mean_delta_d"] for v in e) if e else None,
            delta_d_range=[min(v["mean_delta_d"] for v in e),max(v["mean_delta_d"] for v in e)] if e else None,
            attention_per_seed=values,effect_per_seed=e))
    return dict(rows=rows,source_sha256=hashes,keyword_definition=runs[0]["keyword_definition"],
        literal_phrases=runs[0]["literal_phrases"],
        started_utc=min(r["started_utc"] for r in runs),finished_utc=max(r["finished_utc"] for r in runs),
        per_seed_wall_seconds={r["seed"]:r["wall_seconds"] for r in runs},
        note="All eight held-out seeds equally weighted. Clean attention only. Single-head delta d is patched minus clean, d=logit(T first digit)-logit(N first digit). No additive decomposition is assumed.")

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);args=p.parse_args()
    root=Path(args.root);start=time.perf_counter()
    sec=secondary(root);dump(root/"stage3"/"secondary_parse.json",sec)
    circuit=readouts(root);dump(root/"stage3"/"circuit_summary.json",circuit)
    for row in sec["totals"]:
        if row["set_index"] in [None,0]:print(json.dumps(row))
    for row in circuit["rows"]:
        print(json.dumps({k:v for k,v in row.items() if k not in ["attention_per_seed","effect_per_seed"]}))
    print("wall_seconds",time.perf_counter()-start)
if __name__=="__main__":main()

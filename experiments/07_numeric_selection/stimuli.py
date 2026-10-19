"""Deterministic numeric-selection stimuli; no model computation."""
import argparse, hashlib, json, random
from pathlib import Path
DEV_SEEDS = [20261101, 20261102, 20261103, 20261104]
HELDOUT_SEEDS = list(range(20261201, 20261209))
NAMES = {
 'A': ['Alder','Beacon','Cedar','Delta','Elm','Falcon','Granite','Harbor','Iris','Juniper','Keystone','Laurel','Maple','Northstar','Oak','Pioneer'],
 'B': ['Quartz','Redwood','Summit','Timber','Union','Valley','Willow','Zephyr','Aspen','Birch','Canyon','Driftwood','Evergreen','Flint','Grove','Horizon']}
QUERIES = {
 'A': {'Q_N': "What was last quarter's revenue?", 'Q_T': "What is management's guidance for next-quarter revenue?", 'Q_F': 'What is the best estimate of next-quarter revenue?'},
 'B': {'Q_N': 'How much revenue did the company report for the previous quarter?', 'Q_T': 'How much revenue does management expect for the coming quarter?', 'Q_F': 'What revenue should be expected for the coming quarter?'}}

def items(seed, pool='A'):
    rng = random.Random(seed)
    # K9 minus a four-edge matching: each digit occurs 7 or 8 times per role.
    removed = {(1,2),(3,4),(5,6),(7,8)}
    pairs = [(a,b) for a in range(1,10) for b in range(a+1,10) if (a,b) not in removed]
    pairs = pairs + [(b,a) for a,b in pairs]
    rng.shuffle(pairs)
    lo = 0 if pool == 'A' else 50
    rows = []
    for i,(a,b) in enumerate(pairs):
        n,t = a*100+rng.randrange(lo,lo+50), b*100+rng.randrange(lo,lo+50)
        company = rng.choice(NAMES[pool]) + ' Industries'
        wording = 'guided next-quarter revenue to' if pool == 'A' else 'said it expects next-quarter revenue of about'
        context = f'{company} reported revenue of ${n} million last quarter. Management {wording} ${t} million.'
        rows.append(dict(seed=seed,item=i,pool=pool,company=company,N=n,T=t,context=context))
    assert len(rows)==64 and sum(r['N']<r['T'] for r in rows)==32
    for key in ['N','T']:
        counts=[sum(str(r[key])[0]==str(d) for r in rows) for d in range(1,10)]
        assert max(counts)-min(counts)==1 and min(counts)==7
    return rows

def prompts(seed, pool='A'):
    return [dict(r,query_type=q,question=text+' Answer with the number only.')
            for r in items(seed,pool) for q,text in QUERIES[pool].items()]

def render(row, tokenizer):
    content=row['context']+'\n\n'+row['question']
    rendered=tokenizer.apply_chat_template([{'role':'user','content':content},{'role':'assistant','content':'$'}],tokenize=False,continue_final_message=True,enable_thinking=False)
    assert rendered.endswith('</think>\n\n$'), 'STOP: prefill rendering failed'
    enc=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids']; offsets=enc['offset_mapping']
    start=rendered.index(row['question']); end=start+len(row['question'])
    question=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    assert question and tokenizer.decode([ids[-1]])=='$'
    digit_ids={}
    for key in ['N','T']:
        ext=tokenizer.encode(rendered+str(row[key]),add_special_tokens=False)
        assert ext[:len(ids)]==ids, 'STOP: prefill boundary changed'
        assert tokenizer.decode([ext[len(ids)]])==str(row[key])[0], 'STOP: first number token is not a single digit'
        digit_ids[key]=ext[len(ids)]
    number_positions={}
    for key in ['N','T']:
        a=rendered.index('$'+str(row[key]))+1; b=a+3
        number_positions[key]=[i for i,(s,e) in enumerate(offsets) if s<b and e>a]
    return dict(row,rendered=rendered,input_ids=ids,prefill_position=len(ids)-1,question_positions=question,number_positions=number_positions,digit_ids=digit_ids)

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True); a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B',local_files_only=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest={}
    for seed in DEV_SEEDS+HELDOUT_SEEDS:
        pool='A' if seed in DEV_SEEDS else 'B'
        rows=[render(r,tok) for r in prompts(seed,pool)]
        name=f'{pool}_{seed}.json';manifest[name]=dict(sha256=dump(out/name,rows),items=64,prompts=len(rows),model_run_allowed_stage1=seed in DEV_SEEDS[:2])
    dump(out/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

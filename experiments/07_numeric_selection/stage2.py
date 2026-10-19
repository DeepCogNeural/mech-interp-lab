"""Gated stage-2 entry point. Not run during stage 1."""
import argparse, collections, hashlib, json, math, os, random, subprocess, time
from pathlib import Path
import torch
from patch import Harness
from scoring import summarize
from stimuli import DEV_SEEDS, HELDOUT_SEEDS, dump
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
RANDOM_SEED=20261105

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def gate(expected):
    assert os.environ.get('SLURM_JOB_ID'), 'Use the stated Slurm command'
    assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==expected
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(), 'Commit the frozen design first'
    design=HERE/'DESIGN.md';assert design.exists(), 'Stage 1 stops at DESIGN_draft.md'
    subprocess.run(['git','ls-files','--error-unmatch',str(design.relative_to(REPO))],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
    assert 'STAGE2_FROZEN' in design.read_text().splitlines(), 'design must carry the frozen-protocol marker'

def rows_for(args,seed):
    pool='A' if seed in DEV_SEEDS else 'B'
    name=f'{pool}_{seed}.json';path=Path(args.stimuli)/name
    manifest=read(Path(args.stimuli)/'manifest.json')
    assert sha(path)==manifest[name]['sha256']
    return read(path)

def caches(h,rows,components):
    result={}
    keys=[(l,c,'prefill') for l in range(len(h.layers)) for c in components]
    for query in ['Q_N','Q_T']:
        rs=[r for r in rows if r['query_type']==query]
        batch,pos=h.batch(rs);logits,cache=h.forward(batch,pos,captures=keys)
        outputs=h.greedy(batch,pos,first_logits=logits)
        for r,o,ll in zip(rs,outputs,logits):
            o.update({k:r[k] for k in ['seed','item','query_type','N','T']})
            o['d']=(ll[r['digit_ids']['T']].float()-ll[r['digit_ids']['N']].float()).item()
        result[query]=dict(rows=rs,batch=batch,pos=pos,cache=cache,outputs=outputs)
    return result

def intervention(h,base,patches,decode=True):
    logits,_=h.forward(base['batch'],base['pos'],patches=patches)
    out=h.greedy(base['batch'],base['pos'],patches=patches,first_logits=logits) if decode else [{} for _ in base['rows']]
    for r,o,ll in zip(base['rows'],out,logits):
        o.update({k:r[k] for k in ['seed','item','query_type','N','T']})
        o['d']=(ll[r['digit_ids']['T']].float()-ll[r['digit_ids']['N']].float()).item()
    return out

def effects(base,outputs,target):
    n=len(outputs); sign=1 if target=='T' else -1
    base_gold='N' if base['rows'][0]['query_type']=='Q_N' else 'T'
    return dict(n=n,mean_delta_d=sum(o['d']-b['d'] for o,b in zip(outputs,base['outputs']))/n,oriented_delta_d=sum(sign*(o['d']-b['d']) for o,b in zip(outputs,base['outputs']))/n,
                flip_rate=sum(o.get('answer')==o[target] for o in outputs)/n if 'answer' in outputs[0] else None,
                validity=sum(o.get('answer') in [o['N'],o['T']] for o in outputs)/n if 'answer' in outputs[0] else None,
                clean_correct_count=sum(b['answer']==b[base_gold] for b in base['outputs']),
                conditional_flip_rate=(sum(o.get('answer')==o[target] and b['answer']==b[base_gold] for o,b in zip(outputs,base['outputs']))/sum(b['answer']==b[base_gold] for b in base['outputs'])) if any(b['answer']==b[base_gold] for b in base['outputs']) and 'answer' in outputs[0] else None)

def head_patches(cache,heads):
    groups=collections.defaultdict(list)
    for layer,head in heads:groups[layer].append(head)
    return {(l,'head','prefill'):dict(values=cache[(l,'head','prefill')],heads=hs) for l,hs in groups.items()}

def validate_task(clean):
    s=summarize(clean)
    return s['Q_N']['accuracy']['share']>=.9 and s['Q_T']['accuracy']['share']>=.9 and s['agreement']['share'] is not None and s['agreement']['share']>=.9


def dev(args):
    rows=rows_for(args,args.seed)
    if args.task=='numeric' and args.seed in DEV_SEEDS[:2]:
        result=read(Path('workspace') / f'./exp07/pilot_{args.seed}/scored_outputs.json')
    else:
        h=Harness('Qwen/Qwen3-4B');result=[]
        for start in range(0,len(rows),64):
            vals,_=h.measure(rows[start:start+64]);result+=vals
    dump(Path(args.output)/f'dev_{args.seed}.json',dict(outputs=result,summary=summarize(result)))

def task_gate(args):
    outputs=sum([read(Path(args.output)/f'dev_{seed}.json')['outputs'] for seed in DEV_SEEDS],[])
    passed=validate_task(outputs)
    dump(Path(args.output)/'task_gate.json',dict(task=args.task,H5_passed=passed,selected_task=args.task if passed else ('table' if args.task=='numeric' else 'stop'),summary=summarize(outputs)))
    if not passed:raise RuntimeError('H5 gate failed: table fallback is the only allowed switch; failed table validity stops discovery')

def residual(args):
    h=Harness('Qwen/Qwen3-4B');data=caches(h,rows_for(args,args.seed),['residual'])
    clean=data['Q_N']['outputs']+data['Q_T']['outputs']
    out=dict(seed=args.seed,clean=clean,sites=[])
    for layer in range(len(h.layers)):
        key=(layer,'residual','prefill')
        for b,d,target in [('Q_T','Q_N','N'),('Q_N','Q_T','T')]:
            vals=intervention(h,data[b],{key:dict(values=data[d]['cache'][key])})
            out['sites'].append(dict(layer=layer,direction=b+'->'+target,outputs=vals,**effects(data[b],vals,target)))
        dump(Path(args.output)/f'residual_{args.seed}.json',out)
        print('residual',args.seed,layer,flush=True)

def bounds(args):
    rs=[read(Path(args.output)/f'residual_{s}.json') for s in DEV_SEEDS]
    earliest=None;rates=[]
    for l in range(36):
        bydir={d:sum(next(x['flip_rate'] for x in r['sites'] if x['layer']==l and x['direction']==d) for r in rs)/len(rs) for d in ['Q_T->N','Q_N->T']}
        rates.append(dict(layer=l,**bydir))
        if earliest is None and min(bydir.values())>=.8:earliest=l
    dump(Path(args.output)/'bounds.json',dict(L_star=earliest,scan_last_layer=35 if earliest is None else earliest,residual_rates=rates,interpretation='descriptive only; not a selected causal site'))

def scan(args):
    last=read(Path(args.output)/'bounds.json')['scan_last_layer']
    h=Harness('Qwen/Qwen3-4B');data=caches(h,rows_for(args,args.seed),['attention','mlp','head'])
    out=dict(seed=args.seed,scan_last_layer=last,components=[],heads=[])
    for layer in range(last+1):
        for comp in ['attention','mlp']:
            key=(layer,comp,'prefill')
            for b,d,target in [('Q_T','Q_N','N'),('Q_N','Q_T','T')]:
                vals=intervention(h,data[b],{key:dict(values=data[d]['cache'][key])})
                out['components'].append(dict(layer=layer,component=comp,direction=b+'->'+target,outputs=vals,**effects(data[b],vals,target)))
        for head in range(h.num_heads):
            for b,d,target in [('Q_T','Q_N','N'),('Q_N','Q_T','T')]:
                vals=intervention(h,data[b],head_patches(data[d]['cache'],[(layer,head)]),decode=False)
                out['heads'].append(dict(layer=layer,head=head,direction=b+'->'+target,outputs=vals,**effects(data[b],vals,target)))
        dump(Path(args.output)/f'scan_{args.seed}.json',out)
        print('component/head layer',args.seed,layer,flush=True)

def freeze(args):
    bounds=read(Path(args.output)/'bounds.json');last=bounds['scan_last_layer']
    assert bounds['L_star'] is not None, 'No L_star: descriptive discovery only; do not freeze a head set'
    scans=[read(Path(args.output)/f'scan_{s}.json') for s in DEV_SEEDS]
    scores=collections.defaultdict(list)
    for scanrow in scans:
        for r in scanrow['heads']:scores[(r['layer'],r['head'])].append(r['oriented_delta_d'])
    ranking=sorted(scores,key=lambda k:(-sum(scores[k])/len(scores[k]),k))
    residuals=[read(Path(args.output)/f'residual_{s}.json') for s in DEV_SEEDS]
    reference={d:sum(next(x['oriented_delta_d'] for x in r['sites'] if x['layer']==last and x['direction']==d) for r in residuals)/4 for d in ['Q_T->N','Q_N->T']}
    h=Harness('Qwen/Qwen3-4B');joint=[];means={};qf=[];allclean=[]
    for seed in DEV_SEEDS:
        rows=rows_for(args,seed);data=caches(h,rows,['head'])
        allclean+=data['Q_N']['outputs']+data['Q_T']['outputs']
        forecast_rows=[r for r in rows if r['query_type']=='Q_F']
        if forecast_rows:
            forecast,_=h.measure(forecast_rows);qf+=forecast
        for query in ['Q_N','Q_T']:
            for key,vals in data[query]['cache'].items():
                val=torch.cat(vals,0).float().sum(0).cpu()
                means[key]=means.get(key,torch.zeros_like(val))+val
        for k in [2,4,8,16]:
            for b,d,target in [('Q_T','Q_N','N'),('Q_N','Q_T','T')]:
                vals=intervention(h,data[b],head_patches(data[d]['cache'],ranking[:k]))
                joint.append(dict(seed=seed,k=k,direction=b+'->'+target,outputs=vals,**effects(data[b],vals,target)))
    if not validate_task(allclean+qf):raise RuntimeError('H5 failed on full dev: activate table fallback before any held-out run')
    chosen=None;criterion=[]
    for k in [2,4,8,16]:
        effects_by_dir={d:sum(r['oriented_delta_d'] for r in joint if r['k']==k and r['direction']==d)/4 for d in reference}
        validity=min(r['validity'] for r in joint if r['k']==k)
        passes=all(reference[d]>0 and effects_by_dir[d]>=.8*reference[d] for d in reference) and validity>=.95
        criterion.append(dict(k=k,effects=effects_by_dir,reference=reference,validity=validity,passes=passes))
        if chosen is None and passes:chosen=k
    dump(Path(args.output)/'joint_selection.json',dict(ranking=[dict(layer=k[0],head=k[1],mean_oriented_delta_d=sum(scores[k])/len(scores[k])) for k in ranking],joint=joint,criterion=criterion,chosen_k=chosen,dev_summary=summarize(allclean+qf)))
    if chosen is None:raise RuntimeError('No k met the pre-registered rule. Report discovery; do not run held-out or select a late residual.')
    heads=ranking[:chosen];hist=collections.Counter(l for l,h in heads);rng=random.Random(RANDOM_SEED);sets=[];seen={tuple(sorted(heads))}
    while len(sets)<100:
        candidate=tuple(sorted((l,head) for l,count in sorted(hist.items()) for head in rng.sample(range(h.num_heads),count)))
        if candidate in seen:continue
        seen.add(candidate);sets.append(candidate)
    meanpath=Path(args.output)/'dev_head_means.pt';torch.save({k:v/(len(DEV_SEEDS)*128) for k,v in means.items()},meanpath)
    manifest=dict(task=args.task,source_sha256={p.name:sha(p) for p in HERE.glob('*.py')},k=chosen,heads=heads,random_seed=RANDOM_SEED,random_sets=sets,layer_histogram=dict(hist),L_star=bounds['L_star'],scan_last_layer=last,mean_path=str(meanpath.resolve()),mean_sha256=sha(meanpath),rescue_trigger=bool(qf) and sum(r['answer']==r['N'] for r in qf)/len(qf)>=.2,dev_forecast_N_share=sum(r['answer']==r['N'] for r in qf)/len(qf) if qf else None,dev_seeds=DEV_SEEDS,heldout_seeds=HELDOUT_SEEDS,stimuli_manifest_sha256=sha(Path(args.stimuli)/'manifest.json'),design_sha256=sha(HERE/'DESIGN.md'),source_commit=args.expected_commit)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());path=HERE/f'FREEZE_{stamp}.md'
    assert not path.exists();path.write_text('# Frozen head set\n\n```json\n'+json.dumps(manifest,indent=2)+'\n```\n')
    dump(Path(args.output)/'freeze.json',dict(manifest,freeze_path=str(path),freeze_sha256=sha(path)))
    print(str(path),flush=True)

def readout(h,rows,heads):
    batch,pos=h.batch(rows);captured={};handles=[];layers=sorted(set(l for l,_ in heads))
    for layer in layers:
        def hook(module,inputs,out,layer=layer):
            weights=out[1];assert weights is not None
            for head in [hh for ll,hh in heads if ll==layer]:
                entries=[]
                for i,row in enumerate(rows):
                    offset=batch['input_ids'].shape[1]-len(row['input_ids']);p=pos['prefill'][i][0]
                    entries.append({key:weights[i,head,p,[n+offset for n in row['number_positions'][key]]].float().sum().item() for key in ['N','T']})
                captured[(layer,head)]=entries
        handles.append(h.layers[layer].self_attn.register_forward_hook(hook))
    keys=[(l,'head','prefill') for l in layers]+[(len(h.layers)-1,'residual','prefill')]
    try:_,cache=h.forward(batch,pos,captures=keys)
    finally:
        for handle in handles:handle.remove()
    outputs=[]
    for i,row in enumerate(rows):
        final=cache[(len(h.layers)-1,'residual','prefill')][i][0].float()
        denom=torch.sqrt(final.square().mean()+h.model.config.rms_norm_eps)
        gold=row['digit_ids']['N' if row['query_type']=='Q_N' else 'T']
        for layer,head in heads:
            sl=slice(head*h.head_dim,(head+1)*h.head_dim)
            z=cache[(layer,'head','prefill')][i][0,sl].float()
            contribution=h.layers[layer].self_attn.o_proj.weight[:,sl].float()@z
            # Linearized direct attribution with the clean final RMS denominator held fixed.
            dla=(h.model.lm_head.weight[gold].float()*(h.model.model.norm.weight.float()*contribution/denom)).sum().item()
            outputs.append(dict(seed=row['seed'],item=row['item'],query_type=row['query_type'],layer=layer,head=head,attention_to_N=captured[(layer,head)][i]['N'],attention_to_T=captured[(layer,head)][i]['T'],gold_first_digit_dla=dla))
    return outputs


def extra_controls(h,base,donor,heads,seed,target):
    records=[]
    # Deterministic derangement with no donor number equal to either base number.
    order=list(range(64));rng=random.Random(seed+RANDOM_SEED)
    for _ in range(10000):
        rng.shuffle(order)
        if all(i!=j and not ({base['rows'][i]['N'],base['rows'][i]['T']}&{donor['rows'][j]['N'],donor['rows'][j]['T']}) for i,j in enumerate(order)):break
    else:raise RuntimeError('No valid mismatched donor derangement')
    mismatched={key:[vals[j] for j in order] for key,vals in donor['cache'].items()}
    vals=intervention(h,base,head_patches(mismatched,heads))
    injection=sum(v['answer'] in [donor['rows'][j]['N'],donor['rows'][j]['T']] for v,j in zip(vals,order))/64
    records.append(dict(kind='mismatched donor',direction=base['rows'][0]['query_type']+'->'+target,donor_order=order,number_injection_rate=injection,outputs=vals,**effects(base,vals,target)))
    # Random isotropic intervention with the norm of the donor-minus-base head delta.
    generator=torch.Generator(device='cuda').manual_seed(seed+RANDOM_SEED)
    randcache={key:[v.clone() for v in vals] for key,vals in base['cache'].items()}
    for l,head in heads:
        key=(l,'head','prefill');sl=slice(head*h.head_dim,(head+1)*h.head_dim)
        for i in range(64):
            delta=(donor['cache'][key][i][:,sl]-base['cache'][key][i][:,sl]).float()
            noise=torch.randn(delta.shape,device='cuda',generator=generator)
            noise=noise/noise.norm().clamp_min(1e-12)*delta.norm()
            randcache[key][i][:,sl]+=noise.to(torch.float16)
    vals=intervention(h,base,head_patches(randcache,heads))
    records.append(dict(kind='matched-norm random delta',direction=base['rows'][0]['query_type']+'->'+target,outputs=vals,**effects(base,vals,target)))
    return records

def heldout(args):
    frozen=read(Path(args.output)/'freeze.json')
    assert all(sha(HERE/name)==digest for name,digest in frozen['source_sha256'].items())
    assert sha(frozen['freeze_path'])==frozen['freeze_sha256']
    subprocess.run(['git','ls-files','--error-unmatch',str(Path(frozen['freeze_path']).relative_to(REPO))],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
    assert sha(HERE/'DESIGN.md')==frozen['design_sha256']
    assert sha(Path(args.stimuli)/'manifest.json')==frozen['stimuli_manifest_sha256']
    assert sha(frozen['mean_path'])==frozen['mean_sha256']
    outpath=Path(args.output)/f'heldout_{args.seed}.json';started=Path(args.output)/f'heldout_{args.seed}.started'
    with started.open('x') as f:f.write('Reserved once; partial failure is reported, not silently retried.\n')
    h=Harness('Qwen/Qwen3-4B');rows=rows_for(args,args.seed);data=caches(h,rows,['head'])
    means=torch.load(frozen['mean_path'],weights_only=True,map_location='cpu')
    mean_cache={key:[val[None,:] for _ in range(64)] for key,val in means.items()}
    out=dict(seed=args.seed,freeze_sha256=frozen['freeze_sha256'],clean=data['Q_N']['outputs']+data['Q_T']['outputs'],sets=[],mechanism=[],extra_controls=[],rescue=[])
    sets=[frozen['heads']]+frozen['random_sets']
    for index,heads in enumerate(sets):
        record=dict(index=index,label='selected' if index==0 else 'layer-matched random',sufficiency=[],necessity=[])
        for b,d,target in [('Q_T','Q_N','N'),('Q_N','Q_T','T')]:
            vals=intervention(h,data[b],head_patches(data[d]['cache'],heads))
            record['sufficiency'].append(dict(direction=b+'->'+target,outputs=vals,**effects(data[b],vals,target)))
            vals=intervention(h,data[b],head_patches(mean_cache,heads));gold=b[-1]
            cleanacc=sum(o['answer']==o[gold] for o in data[b]['outputs'])/64
            accuracy=sum(o['answer']==o[gold] for o in vals)/64
            record['necessity'].append(dict(query_type=b,clean_accuracy=cleanacc,ablated_accuracy=accuracy,drop=cleanacc-accuracy,validity=sum(o['answer'] in [o['N'],o['T']] for o in vals)/64,outputs=vals))
        out['sets'].append(record);dump(outpath,out)
        print('heldout',args.seed,'set',index,flush=True)
    heads=frozen['heads']
    for b,d,target in [('Q_T','Q_N','N'),('Q_N','Q_T','T')]:
        out['extra_controls']+=extra_controls(h,data[b],data[d],heads,args.seed,target)
    forecast_rows=[r for r in rows if r['query_type']=='Q_F']
    if forecast_rows:
        fout,fcache=h.measure(forecast_rows,captures=[(l,'head','prefill') for l in range(len(h.layers))])
    else:fout,fcache=[],{}
    out['forecast_clean']=fout
    if frozen['rescue_trigger']:
        fb,fp=h.batch(forecast_rows);base=dict(rows=forecast_rows,batch=fb,pos=fp,outputs=fout,cache=fcache)
        for donor,target,label in [('Q_T','T','rescue'),('Q_N','N','reverse ablation')]:
            eligible=[i for i,r in enumerate(fout) if r['answer']==r['N' if target=='T' else 'T']]
            for index,hs in enumerate(sets):
                vals=intervention(h,base,head_patches(data[donor]['cache'],hs))
                out['rescue'].append(dict(label=label,set_index=index,eligible=eligible,n=len(eligible),restored_share=sum(vals[i]['answer']==vals[i][target] for i in eligible)/len(eligible) if eligible else None,outputs=vals))
            for control in extra_controls(h,base,data[donor],heads,args.seed,target):
                vals=control['outputs']
                out['rescue'].append(dict(label=label,control=control['kind'],eligible=eligible,n=len(eligible),restored_share=sum(vals[i]['answer']==vals[i][target] for i in eligible)/len(eligible) if eligible else None,outputs=vals))
    for start in range(0,len(rows),8):out['mechanism']+=readout(h,rows[start:start+8],heads)
    out['complete']=True;dump(outpath,out)

def aggregate(args):
    runs=[read(Path(args.output)/f'heldout_{s}.json') for s in HELDOUT_SEEDS]
    assert all(r.get('complete') for r in runs)
    out={'directions':{}}
    for direction in ['Q_T->N','Q_N->T']:
        rates=[];above=[];valid=[]
        for run in runs:
            values=[next(x for x in entry['sufficiency'] if x['direction']==direction) for entry in run['sets']]
            rates.append(values[0]['flip_rate']);above.append(values[0]['flip_rate']>max(v['flip_rate'] for v in values[1:]));valid.append(values[0]['validity'])
        mean=sum(rates)/8;sd=math.sqrt(sum((x-mean)**2 for x in rates)/7);lower=mean-2.364624251*sd/math.sqrt(8)
        out['directions'][direction]=dict(seed_rates=rates,mean=mean,t7_lower95=lower,above_all_random_each_seed=above,validity=valid,passes=mean>=.7 and lower>=.6 and all(above) and min(valid)>=.95)
    selected=[sum(x['drop'] for x in r['sets'][0]['necessity'])/2 for r in runs]
    random_drops=[sum(sum(x['drop'] for x in r['sets'][j]['necessity'])/2 for r in runs)/8 for j in range(1,101)]
    out['necessity']=dict(selected_mean_drop=sum(selected)/8,selected_seed_drops=selected,random_mean_drops=random_drops,passes=sum(selected)/8>=.3 and max(random_drops)<.1)
    out['H7_sufficiency_passes']=all(d['passes'] for d in out['directions'].values());dump(Path(args.output)/'heldout_summary.json',out)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['dev','task_gate','residual','bounds','scan','freeze','heldout','aggregate']);p.add_argument('--task',choices=['numeric','table'],default='numeric');p.add_argument('--seed',type=int);p.add_argument('--stimuli',required=True);p.add_argument('--output',required=True);p.add_argument('--expected-commit',required=True);a=p.parse_args()
    gate(a.expected_commit);Path(a.output).mkdir(parents=True,exist_ok=True)
    if a.phase not in ['dev','task_gate']:
        validity=read(Path(a.output)/'task_gate.json');assert validity['H5_passed'] and validity['task']==a.task
    if a.phase in ['dev','residual','scan']:assert a.seed in DEV_SEEDS
    if a.phase=='heldout':assert a.seed in HELDOUT_SEEDS
    globals()[a.phase](a)
if __name__=='__main__':main()

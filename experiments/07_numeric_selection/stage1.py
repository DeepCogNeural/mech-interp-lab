"""Stage 1 only: validation, two development seeds, and one identity-head timing."""
import argparse, hashlib, json, math, os, subprocess, time
from pathlib import Path
import torch
from stimuli import dump, render, prompts
from patch import Harness

from scoring import summarize

def provenance():
    here=Path(__file__).parent
    return dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob('*.py')},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=here,text=True).strip(),torch=torch.__version__)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validate','pilot']);p.add_argument('--seed',type=int);p.add_argument('--output',required=True);p.add_argument('--stimuli',required=True);p.add_argument('--head-timing',action='store_true');a=p.parse_args()
    assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1'
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires Slurm'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    meta=provenance();meta.update(mode=a.mode,seed=a.seed,slurm_job_id=os.environ['SLURM_JOB_ID'],start_unix=time.time())
    dump(out/'started.json',meta)
    try:
        model_id='Qwen/Qwen3-1.7B' if a.mode=='validate' else 'Qwen/Qwen3-4B'
        h=Harness(model_id);meta.update(model_id=model_id,num_layers=len(h.layers),num_heads=h.num_heads,head_dim=h.head_dim,gpu=torch.cuda.get_device_name())
        if a.mode=='validate':
            rs=[render(r,h.tokenizer) for r in prompts(20261101)[:6]]
            base=[r for r in rs if r['query_type']=='Q_T'];donor=[r for r in rs if r['query_type']=='Q_N']
            batch,pos=h.batch(base); db,dp=h.batch(donor)
            key=(0,'residual','prefill'); last=(len(h.layers)-1,'residual','prefill')
            clean,cache=h.forward(batch,pos,captures=[key]); patched,_=h.forward(batch,pos,patches={key:dict(values=cache[key])})
            identity=torch.equal(clean,patched)
            dl,dc=h.forward(db,dp,captures=[last]); pl,_=h.forward(batch,pos,patches={last:dict(values=dc[last])})
            close=torch.allclose(pl,dl,atol=0.02,rtol=0.002)
            result=dict(meta,identity_bit_identical=identity,identity_max_abs=(clean-patched).abs().max().item(),last_residual_donor_allclose=close,last_residual_max_abs=(pl-dl).abs().max().item(),atol=0.02,rtol=0.002,finite=True)
            dump(out/'validation.json',result)
            assert identity and close, 'Harness validation failed'
        else:
            assert a.seed in [20261101,20261102], 'Stage 1 cannot run other seeds'
            rows=json.loads((Path(a.stimuli)/f'A_{a.seed}.json').read_text())
            batch,pos=h.batch(rows[:64]);h.forward(batch,pos)
            torch.cuda.reset_peak_memory_stats()
            seconds,_=h.timed_forward(batch,pos)
            timing=dict(batch_size=64,forward_seconds=seconds,allocated_GiB=torch.cuda.max_memory_allocated()/2**30,reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            if a.head_timing:
                key=(0,'head','prefill');clean,cache=h.forward(batch,pos,captures=[key])
                # Same-prompt identity head patch only: no donor contrast or discovery.
                sec,patched=h.timed_forward(batch,pos,{key:dict(values=cache[key],heads=[0])})
                timing.update(single_head_forward_seconds=sec,single_head_identity=torch.equal(clean,patched),head_timing_layer=0,head_timing_head=0)
                assert timing['single_head_identity']
            outputs=[]
            for start in range(0,len(rows),64):
                vals,_=h.measure(rows[start:start+64]);outputs.extend(vals)
                dump(out/'outputs.json',outputs)
                print('completed',len(outputs),flush=True)
            result=dict(meta,timing=timing,summary=summarize(outputs),total_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,total_peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finite=True,elapsed_seconds=time.time()-meta['start_unix'])
            dump(out/'summary.json',result)
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        dump(out/'STOPPED.json',dict(error=repr(exc),stop_unix=time.time()))
        raise
if __name__=='__main__':main()

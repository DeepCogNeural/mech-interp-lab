"""Summarize complete H10b saved files; retain both parser denominators."""
import datetime, hashlib, json, subprocess
from pathlib import Path
ROOT=Path('workspace') / './exp07/stage4';RUN=ROOT/'h10b'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
gate=read(RUN/'behaviour_gate.json')
summary=read(RUN/'h10b_summary.json')
assert gate['passed']
rows=[];timings=[];hashes={};all_sets=[];clean_comparison=[]
for group in range(4):
 p=RUN/f'patch_{group}.json';run=read(p);hashes[str(p)]=sha(p)
 assert run.get('complete') and [s['index'] for s in run['sets']]==list(range(101))
 assert len(run['controls'])==4
 clean=read(RUN/f'clean_{group}.json')
 old=read(ROOT.parent/f'stage3/h10/clean_{group}.json')
 same=sum(v['generated_ids']==w['generated_ids'] for q in ['Q_N','Q_T'] for v,w in zip(clean['outputs'][q],old['outputs'][q]))
 clean_comparison.append(dict(group=group,identical_generated_ids=same,total=128))
 assert sha(RUN/f'group_{group}.json')==sha(ROOT.parent/f'stage3/h10/group_{group}.json')
 timings.append(dict(group=group,clean_started=clean['started_utc'],clean_finished=clean['finished_utc'],patch_started=run['started_utc'],patch_finished=run['finished_utc'],patch_wall_seconds=run['wall_seconds']))
 for selected in run['sets'][0]['directions']:
  d=selected['direction'];randoms=[next(t for t in s['directions'] if t['direction']==d) for s in run['sets'][1:]]
  row={k:v for k,v in selected.items() if k!='outputs'};row['group']=group
  row['random_max']=max(r['flip_rate'] for r in randoms)
  row['above_all_random']=row['flip_rate']>row['random_max']
  row['random_validity_min']=min(r['validity'] for r in randoms)
  row['random_validity_max']=max(r['validity'] for r in randoms)
  for parser in ['frozen_unique_integer','frozen_on_h10b_denominator']:
   row[parser]['random_max']=max(r[parser]['flip_rate'] for r in randoms)
   row[parser]['random_validity_min']=min(r[parser]['validity'] for r in randoms)
   row[parser]['random_validity_max']=max(r[parser]['validity'] for r in randoms)
  row['controls']=[{k:v for k,v in c.items() if k not in ['outputs','donor_order']} for c in run['controls'] if c['direction']==d]
  rows.append(row)
 for s in run['sets']:
  all_sets.append(dict(group=group,index=s['index'],directions=[{k:v for k,v in d.items() if k!='outputs'} for d in s['directions']]))
directions=[]
for d in ['Q_T->N','Q_N->T']:
 rs=[r for r in rows if r['direction']==d]
 mean=sum(r['flip_rate'] for r in rs)/4
 directions.append(dict(direction=d,mean_flip=mean,flip_count=sum(r['flip_count'] for r in rs),denominator=sum(r['denominator'] for r in rs),passes=mean>=.5 and all(r['above_all_random'] for r in rs)))
passed=all(d['passes'] for d in directions)
assert passed==summary['passes']
protected=read(ROOT/'protected_before.json')
changes=[str(p) for p,digest in protected.items() if sha(Path(p))!=digest]
assert not changes,changes
prereg=read(RUN/'protocol.json')['prereg_commit']
commit_time=subprocess.check_output(['git','show','-s','--format=%cI',prereg],cwd=Path('workspace') / './mech-interp-lab',text=True).strip()
assert datetime.datetime.fromisoformat(commit_time)<min(datetime.datetime.fromisoformat(t['clean_started']) for t in timings)
result=dict(prereg_commit_time=commit_time,clean_comparison_with_h10=clean_comparison,gate=gate,per_group=rows,directions=directions,passes=passed,verdict='pass' if passed else 'fail',timings=timings,
 protected_files=len(protected),protected_files_changed=changes,files_sha256=hashes,all_sets=all_sets,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(ROOT/'h10b_readout.json',result)
print(json.dumps({k:result[k] for k in ['directions','verdict','protected_files','protected_files_changed']},indent=2))

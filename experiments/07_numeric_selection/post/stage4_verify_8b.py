"""Read saved scan rows and cached config only; no model or forward pass."""
import collections, datetime, hashlib, json, math, struct, subprocess
from pathlib import Path
from types import SimpleNamespace

def spearmanr(x,y):
 def rank(v):
  order=sorted(range(len(v)),key=lambda i:v[i]);out=[0.]*len(v);i=0
  while i<len(v):
   j=i+1
   while j<len(v) and v[order[j]]==v[order[i]]:j+=1
   for k in range(i,j):out[order[k]]=(i+j-1)/2+1
   i=j
  return out
 a,b=rank(x),rank(y);ma=sum(a)/len(a);mb=sum(b)/len(b)
 rho=sum((u-ma)*(v-mb) for u,v in zip(a,b))/math.sqrt(sum((u-ma)**2 for u in a)*sum((v-mb)**2 for v in b))
 return SimpleNamespace(statistic=rho)
from bs4 import BeautifulSoup
ROOT=Path('workspace') / "./exp07"
OUT=ROOT/"stage4"
EXP=Path('workspace') / "./mech-interp-lab/experiments/07_numeric_selection"
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
source8=ROOT/'stage3/exploratory_8b/exploratory_8b.json'
freeze8=ROOT/'stage3/exploratory_8b/exploratory_8b_freeze.json'
r8=read(source8); f8=read(freeze8)
files4=[ROOT/f'stage2/scan_{s}.json' for s in r8['dev_seeds']]
loaded4=[read(p) for p in files4]
rows4=[dict(row,seed=run['seed']) for run in loaded4 for row in run['heads'] if row['layer'] in range(20,24)]
rows8=r8['head_scan']
expected={(s,l,h,d) for s in r8['dev_seeds'] for l in range(20,24) for h in range(32) for d in ['Q_T->N','Q_N->T']}
for rows in [rows4,rows8]:
 assert len(rows)==len(expected)
 assert {(r['seed'],r['layer'],r['head'],r['direction']) for r in rows}==expected

def scores(rows,direction=None,field='oriented_delta_d'):
 by=collections.defaultdict(list)
 for r in rows:
  if direction is None or r['direction']==direction:by[(r['layer'],r['head'])].append(r[field])
 return {k:sum(v)/len(v) for k,v in by.items()}
s4=scores(rows4);s8=scores(rows8)
rank8=sorted(s8,key=lambda k:(-s8[k],k))
assert [list(k) for k in rank8[:8]]==f8['heads']==r8['selected_heads']
assert all(abs(s8[(r['layer'],r['head'])]-r['mean_oriented_delta_d'])<1e-12 for r in f8['ranking'])
discovery_hash=hashlib.sha256(json.dumps({k:r8[k] for k in ['residual','head_scan']},sort_keys=True).encode()).hexdigest()
assert discovery_hash==f8['discovery_arrays_sha256']==r8['recovery']['discovery_arrays_sha256']
f4=read(ROOT/'stage2/freeze.json')
a=set(map(tuple,f4['heads']));b=set(map(tuple,f8['heads']));overlap=sorted(a&b)
keys=sorted(s4);assert keys==sorted(s8)
correlations=[]
for d in [None,'Q_T->N','Q_N->T']:
 x=scores(rows4,d);y=scores(rows8,d)
 r=spearmanr([x[k] for k in keys],[y[k] for k in keys])
 correlations.append(dict(direction=d or 'mean donor-oriented across both directions',rho=float(r.statistic),n=len(keys)))
# Raw delta_d is also provided so sign convention is explicit.
x=scores(rows4,field='mean_delta_d');y=scores(rows8,field='mean_delta_d')
r=spearmanr([x[k] for k in keys],[y[k] for k in keys])
correlations.append(dict(direction='mean raw signed delta_d across both directions',rho=float(r.statistic),n=len(keys)))
prep=Path('workspace') / './prep_models.json';snapshot=Path(read(prep)['Qwen/Qwen3-8B']['path'])
cfgpath=snapshot/'config.json';cfg=read(cfgpath)
assert cfg['hidden_size']==4096 and cfg['intermediate_size']==12288
# Read only serialized tensor shape headers, never tensor data.
headers=[];count=0
for p in sorted(snapshot.glob('*.safetensors')):
 with p.open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size);header=json.loads(raw)
 n=sum(math.prod(v['shape']) for k,v in header.items() if k!='__metadata__')
 count+=n;headers.append(dict(file=str(p),header_sha256=hashlib.sha256(raw).hexdigest(),parameter_count=n))
h=cfg['hidden_size'];i=cfg['intermediate_size'];hd=cfg['head_dim'];n=cfg['num_hidden_layers'];aheads=cfg['num_attention_heads'];kv=cfg['num_key_value_heads'];v=cfg['vocab_size']
formula_count=v*h*(1 if cfg['tie_word_embeddings'] else 2)+n*(2*h*aheads*hd+2*h*kv*hd+3*h*i+2*h+2*hd)+h
assert count==formula_count
ref=snapshot.parents[1]/'refs/main'
searched=[prep,Path('workspace') / './prep_finite_8B.json',Path('workspace') / './prep_finite_8B.log',source8]+list(source8.parent.glob('*.log'))
model=dict(model_id=r8['model'],snapshot_path=str(snapshot),snapshot_reference=ref.read_text().strip(),
 hidden_size=h,intermediate_size=i,parameter_count=count,parameter_count_method='saved safetensors shape headers, cross-checked with config formula; no tensor data or model loaded',
 config_parameter_formula='V*H*(1 if tied else 2) + L*(2*H*A*D + 2*H*KV*D + 3*H*I + 2*H + 2*D) + H',
 hidden_layers=n,attention_heads=aheads,head_dim=hd,config_sha256=sha(cfgpath),headers=headers,
 config_fields_saved_in_original_run=False,config_source='cached config.json read after the run; no forward pass',
 evidence_limit='Run records and source specify Qwen/Qwen3-8B; prep_models.json supplies its cached snapshot. Original exploratory logs do not contain a runtime config or parameter-count attestation.',
 metadata_files_searched=[dict(file=str(p),sha256=sha(p)) for p in searched],
 device_map=r8['device_map'])
files=[source8,freeze8,*files4,ROOT/'stage2/freeze.json',EXP/'post/exploratory_8b.py',EXP/'post/exploratory_8b_resume.py',cfgpath,prep]
result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model=model,
 independence=dict(ranking_recomputed_from='stage3/exploratory_8b/exploratory_8b.json:head_scan',
   scan_rows_per_model=len(rows8),scan_rows_match_between_models=sum(x['oriented_delta_d']==y['oriented_delta_d'] for x,y in zip(sorted(rows4,key=lambda r:(r['seed'],r['layer'],r['head'],r['direction'])),sorted(rows8,key=lambda r:(r['seed'],r['layer'],r['head'],r['direction'])))),
   ranking_matches_saved=True,discovery_arrays_sha256=discovery_hash,
   discovery_source_commit=r8['source_commit'],recovery_source_commit=r8['recovery']['recovery_source_commit']),
 heads4=f4['heads'],heads8=f8['heads'],overlap_heads=overlap,overlap_count=len(overlap),
 hypergeometric=dict(population=128,marked=8,draws=8,at_least=len(overlap),tail_probability=sum(math.comb(8,k)*math.comb(120,8-k) for k in range(len(overlap),9))/math.comb(128,8),
   exact_numerator=sum(math.comb(8,k)*math.comb(120,8-k) for k in range(len(overlap),9)),exact_denominator=math.comb(128,8)),
 rank_correlations=correlations,per_head=[dict(layer=k[0],head=k[1],mean_oriented_delta_d_4b=s4[k],mean_oriented_delta_d_8b=s8[k]) for k in keys],
 files_sha256={str(p):sha(p) for p in files})
records=[]
for commit,file in [(r8['source_commit'],'post/exploratory_8b.py'),(r8['recovery']['recovery_source_commit'],'post/exploratory_8b_resume.py')]:
 path='experiments/07_numeric_selection/'+file
 raw=subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=EXP.parents[1])
 records.append(dict(commit=commit,path=path,sha256=hashlib.sha256(raw).hexdigest(),source=raw.decode()))
result['independence']['committed_source']=records
write('verification_8b.json',result)
html=OUT/'qwen3_report.html';soup=BeautifulSoup(html.read_text(),'html.parser');section=soup.find(id='S4.SS5')
assert section is not None
paragraphs=[p.get_text(' ',strip=True) for p in section.find_all('p')]
write('distillation_citation.json',dict(url='https://huggingface.co/papers/2505.09388',arxiv='2505.09388v1',section='4.5 Strong-to-Weak Distillation',paragraphs=paragraphs,source_sha256=sha(html)))
print(json.dumps({k:result[k] for k in ['overlap_count','overlap_heads','hypergeometric','rank_correlations']},indent=2))
print('Parameters',count)

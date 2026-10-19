"""Make readable CSV tables and refresh packet checksums from compact JSON only."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results'

def read(name):
    return json.loads((ROOT / name).read_text())

def main():
    provenance = read('provenance.json')
    def table(name, rows, parents):
        rows = list(rows)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (ROOT/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
        provenance.setdefault('table_parents',{})[name]=parents
        provenance['summary_sources'][name]=sorted({s for p in parents for s in provenance['summary_sources'][p]})
    table('h5_per_seed.csv', (dict(seed=r['seed'],query=q,correct=r['summary'][q]['accuracy']['count'],n=r['summary'][q]['accuracy']['total'],forecast_T=r['summary']['Q_F']['T']['count'],forecast_n=r['summary']['Q_F']['T']['total'],agreement_correct=r['summary']['agreement']['count'],agreement_n=r['summary']['agreement']['total']) for r in read('h5.json')['per_seed'] for q in ['Q_N','Q_T']), ['h5.json'])
    table('residual_per_seed.csv',read('residual_scan.json')['per_seed'],['residual_scan.json'])
    table('heads_top20.csv',(dict(rank=i+1,layer=r['layer'],head=r['head'],oriented_delta_d=r['mean_oriented_delta_d'],T_to_N=r['directional_oriented_delta_d']['Q_T->N'],N_to_T=r['directional_oriented_delta_d']['Q_N->T'],n_per_seed_direction=64,dev_seeds=4) for i,r in enumerate(read('top20_heads.json'))),['top20_heads.json'])
    table('joint_k_per_seed.csv',read('joint_k.json')['per_seed'],['joint_k.json'])
    table('h7_per_seed.csv',(r for v in read('h7.json')['directions'].values() for r in v['per_seed']),['h7.json'])
    table('necessity_per_seed.csv',(dict(seed=r['seed'],mean_drop=r['mean_drop'],random_max_mean_drop=r['random_max_drop'],**q) for r in read('necessity.json')['per_seed'] for q in r['queries']),['necessity.json'])
    table('controls_per_seed.csv',read('controls.json'),['controls.json'])
    table('h9_per_seed.csv',({k:v for k,v in r.items() if k!='extra_controls'} for r in read('h9.json')['per_seed']),['h9.json'])
    table('h10b_per_group.csv',({k:v for k,v in r.items() if k not in ['controls','frozen_on_h10b_denominator','frozen_unique_integer']} for r in read('h10b.json')['per_group']),['h10b.json'])
    table('h10b_controls.csv',(dict(group=r['group'],**{k:v for k,v in c.items() if k not in ['frozen_on_h10b_denominator','frozen_unique_integer']}) for r in read('h10b.json')['per_group'] for c in r['controls']),['h10b.json'])
    table('attention_per_seed.csv',(dict(layer=r['layer'],head=r['head'],query=r['query'],seed=s['seed'],n=s['n'],**s['mass']) for r in read('attention_readouts.json')['rows'] for s in r['attention_per_seed']),['attention_readouts.json'])
    table('exploratory8b_per_seed.csv',(dict(seed=r['seed'],**d) for r in read('exploratory_8b.json')['heldout'] for d in r['directions']),['exploratory_8b.json'])
    table('overlap_per_head.csv',read('overlap_4b_8b.json')['per_head'],['overlap_4b_8b.json'])
    (ROOT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    lines=['# Local packet hashes. Run shasum -a 256 -c checksums.sha256 here.',
           '# Raw labels are relative to exp07/; sibling/ is relative to its parent.',
           '# Raw files are not bundled. Each summary is bound to its source bytes below.']
    paths=sorted(p for p in ROOT.iterdir() if p.is_file() and p.name!='checksums.sha256')
    for p in paths:
        for source in provenance['summary_sources'].get(p.name,[]):
            lines.append(f"# {p.name} <- {provenance['sources'][source]['sha256']}  raw:{source}")
        if p.stat().st_size>2_000_000: raise ValueError(f'File exceeds 2 MB: {p.name}')
        lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}')
    for p in sorted((ROOT.parent/'figures').glob('*')):
        if p.suffix in ['.png','.svg']:
            lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  ../figures/{p.name}')
    (ROOT/'checksums.sha256').write_text('\n'.join(lines)+'\n')
    print(f'Wrote 13 earlier CSV tables; checksums cover {len(paths)} packet files and all figure files.')

if __name__=='__main__':
    main()

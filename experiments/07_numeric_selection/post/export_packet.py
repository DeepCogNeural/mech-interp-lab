"""Read saved results on a host and export compact, hash-bound evidence locally.

No model imports, server writes, or raw activation transfers. Uses Python's
standard library. Remote file names in the packet are relative to the data root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

# This program is sent over stdin and only reads the remote filesystem.
REMOTE = r'''
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser().resolve()
files, sources, bindings = {}, {}, {}
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def label(path):
    path = pathlib.Path(path)
    try: return str(path.relative_to(root))
    except ValueError:
        try: return 'sibling/' + str(path.relative_to(root.parent))
        except ValueError: return 'external/' + path.name
def source(name):
    path = root / name
    key = label(path)
    if key not in sources:
        sources[key] = dict(sha256=digest(path), bytes=path.stat().st_size)
    return key
def read(name):
    source(name)
    return json.loads((root / name).read_text())
def sanitise(x):
    if isinstance(x, dict): return {sanitise(k):sanitise(v) for k,v in x.items()}
    if isinstance(x, list): return [sanitise(v) for v in x]
    if isinstance(x, str):
        return x.replace(str(root), 'exp07').replace(str(root.parent), 'workspace').replace(str(pathlib.Path.home()), '~')
    return x
def emit(name, obj, deps):
    files[name] = sanitise(obj)
    bindings[name] = sorted(set(source(dep) for dep in deps))
def checked_deps(mapping):
    deps = []
    for path, expected in mapping.items():
        key = source(path)
        assert sources[key]['sha256'] == expected, ('source changed', key)
        deps.append(path)
    return deps
def small(row):
    out = {k:v for k,v in row.items() if k not in ['outputs','donor_order']}
    n = row.get('n')
    if n:
        for rate, count in [('flip_rate','flip_count'),('validity','valid_count'),('number_injection_rate','donor_number_injection_count')]:
            if row.get(rate) is not None: out[count] = round(n*row[rate])
    return out
seeds = list(range(20261101,20261105))
held = list(range(20261201,20261209))
devfiles = [f'stage2/dev_{s}.json' for s in seeds]
emit('h5.json', dict(per_seed=[dict(seed=s,summary=read(p)['summary']) for s,p in zip(seeds,devfiles)], gate=read('stage2/task_gate.json')), devfiles+['stage2/task_gate.json'])
emit('behaviour_27b.json', read('behaviour_27B_summary.json'), ['behaviour_27B_summary.json'])
resfiles = [f'stage2/residual_{s}.json' for s in seeds]
emit('residual_scan.json', dict(per_seed=[dict(seed=s,**small(row)) for s,p in zip(seeds,resfiles) for row in read(p)['sites']], bounds=read('stage2/bounds.json')), resfiles+['stage2/bounds.json'])
scanfiles = [f'stage2/scan_{s}.json' for s in seeds]
headrows = [dict(seed=s,**small(row)) for s,p in zip(seeds,scanfiles) for row in read(p)['heads']]
# Keep the exhaustive per-seed scan compact enough for the 2 MB cap.
emit('head_scan.json', headrows, scanfiles)
a = read('stage2a_summary.json')
emit('top20_heads.json', a['heads'][:20], ['stage2a_summary.json']+scanfiles)
emit('head_ranking.json', a['heads'], ['stage2a_summary.json']+scanfiles)
j = read('stage2/joint_selection.json')
emit('joint_k.json', {**{k:v for k,v in j.items() if k not in ['joint','ranking']}, 'per_seed':[small(r) for r in j['joint']]}, ['stage2/joint_selection.json'])
f = read('stage2/freeze.json')
emit('frozen_selection.json', f, ['stage2/freeze.json'])
h = read('stage2b_h7_analysis.json')
hdeps = ['stage2b_h7_analysis.json'] + checked_deps(h['source_files'])
emit('h7.json', dict(directions={d:{**v,'per_seed':[small(r) for r in v['per_seed']]} for d,v in h['directions'].items()}, clean=h['clean'], sufficiency_passes=False), hdeps)
n = h['necessity']
for row in n['per_seed']:
    row['n_per_query'] = 64
    for q in row['queries']: q.update(ablated_correct=round(q['ablated_accuracy']*64),clean_correct=round(q['clean_accuracy']*64),n=64)
emit('necessity.json', n, hdeps)
emit('controls.json', [small(r) for r in h['controls']], hdeps)
emit('direct_readouts.json', h['readouts'], hdeps)
emit('h7_invalid.json', read('stage2b_invalid_selected.json'), ['stage2b_invalid_selected.json']+hdeps)
q = read('qf_transfer/qf_summary.json')
q['per_seed'] = [dict(small(r), extra_controls=[small(c) for c in r['extra_controls']]) for r in q['per_seed']]
emit('h9.json', q, ['qf_transfer/qf_summary.json']+[f'qf_transfer/qf_{s}.json' for s in held])
secondary = read('stage3/secondary_parse.json')
h9_secondary = {k:secondary[k] for k in ['label','parser_definition']}
h9_secondary['per_seed'] = [r for r in secondary['per_seed'] if r['hypothesis']=='H9' and r.get('set_index')==0]
h9_secondary['stayed_T'] = [dict(seed=s,n=64,stayed_T=sum(v['answer']==v['T'] for v in read(f'qf_transfer/qf_{s}.json')['sets'][0]['outputs'])) for s in held]
emit('h9_secondary.json', h9_secondary, ['stage3/secondary_parse.json']+[f'qf_transfer/qf_{s}.json' for s in held])
c = read('stage3/circuit_summary.json')
emit('attention_readouts.json', {k:v for k,v in c.items() if k not in ['started_utc','finished_utc','per_seed_wall_seconds','source_sha256']}, ['stage3/circuit_summary.json']+checked_deps(c['source_sha256']))
emit('h10.json', dict(summary=read('stage3/h10/h10_summary.json'), formats=read('stage3/h10/format_diagnostic.json'), manifest=read('stage3/h10/manifest.json')), ['stage3/h10/h10_summary.json','stage3/h10/format_diagnostic.json','stage3/h10/manifest.json']+[f'stage3/h10/clean_{g}.json' for g in range(4)])
b = read('stage4/h10b_readout.json')
bdeps = ['stage4/h10b_readout.json']+checked_deps(b['files_sha256'])+[f'stage4/h10b/clean_{g}.json' for g in range(4)]
emit('h10b.json', {k:v for k,v in b.items() if k in ['per_group','directions','gate','passes','verdict','clean_comparison_with_h10']}, bdeps)
emit('h10b_all_sets.json', b['all_sets'], bdeps)
e = read('stage3/exploratory_8b/exploratory_8b.json')
e8deps = ['stage3/exploratory_8b/exploratory_8b.json','stage3/exploratory_8b/exploratory_8b_freeze.json']+[f'stage3/exploratory_8b/heldout_8b_{s}.json' for s in held[:2]]
e['residual'] = [small(r) for r in e['residual']]
e['heldout_all_sets'] = [dict(seed=s,sets=[dict(row,directions=[small(d) for d in row['directions']]) for row in read(f'stage3/exploratory_8b/heldout_8b_{s}.json')['sets']]) for s in held[:2]]
for row in e['heldout']:
    for d in row['directions']: d.update(n=64,flip_count=round(64*d['flip_rate']),valid_count=round(64*d['validity']),random_sets=20)
emit('exploratory_8b.json', {k:v for k,v in e.items() if k not in ['device_map','remaining_seconds_at_start','started_utc','finished_utc','wall_seconds']}, e8deps)
v = read('stage4/verification_8b.json')
vdeps = ['stage4/verification_8b.json']+checked_deps(v['files_sha256'])
v['independence']['committed_source'] = [{k:w for k,w in r.items() if k!='source'} for r in v['independence']['committed_source']]
v['model'] = {k:w for k,w in v['model'].items() if k not in ['headers','device_map','metadata_files_searched']}
emit('overlap_4b_8b.json', {k:w for k,w in v.items() if k not in ['created_utc','files_sha256']}, vdeps)
emit('distillation_citation.json', read('stage4/distillation_citation.json'), ['stage4/distillation_citation.json','stage4/qwen3_report.html'])
# Include every random-set summary for H7/H9, so their maxima can be checked.
for s in held:
    path = f'stage2/heldout_{s}.json'
    raw = read(path)
    emit(f'h7_sets_{s}.json', [dict(r, sufficiency=[small(x) for x in r['sufficiency']], necessity=[{k:v for k,v in x.items() if k!='outputs'} for x in r['necessity']]) for r in raw['sets']], [path])
    path = f'qf_transfer/qf_{s}.json'
    raw = read(path)
    emit(f'h9_sets_{s}.json', [small(x) for x in raw['sets']], [path])
print(json.dumps(dict(files=files,sources=sources,bindings=bindings), separators=(',',':')))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--remote-root', default='workspace/exp07')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    command = 'python3 - ' + shlex.quote(args.remote_root)
    data = json.loads(subprocess.check_output(['ssh', args.host, command], input=REMOTE.encode()))
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(schema='exp07-compact-evidence-v1', source_root='exp07/',
                      sources=data['sources'], summary_sources=data['bindings'],
                      note='Hashes measured from saved server files at export; raw files are not bundled. Summary hashes bind public copies, including path redactions.')
    data['files']['provenance.json'] = provenance
    lines = ['# Local summary hashes. Run shasum -a 256 -c checksums.sha256 here.',
             '# Source labels are relative to the exp07 data root; sibling/ is relative to its parent.']
    for name, value in sorted(data['files'].items()):
        raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'))+'\n').encode()
        if len(raw) > 2_000_000:
            raise ValueError(f'{name} exceeds 2 MB: {len(raw)}')
        (args.output / name).write_bytes(raw)
        for source in data['bindings'].get(name, []):
            lines.append(f"# {name} <- {data['sources'][source]['sha256']}  raw:{source}")
        lines.append(f'{hashlib.sha256(raw).hexdigest()}  {name}')
    (args.output / 'checksums.sha256').write_text('\n'.join(lines)+'\n')
    print(f"Exported {len(data['files'])} compact files; {len(data['sources'])} hashed source files.")


if __name__ == '__main__':
    main()

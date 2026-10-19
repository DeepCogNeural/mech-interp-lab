"""Registered cross-order test using the unchanged numeric-selection harness."""
import argparse
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from patch import Harness
from scoring import parse_integer
from stage2 import gate, head_patches, intervention, read, rows_for, sha, RANDOM_SEED, REPO
from stimuli import HELDOUT_SEEDS, dump, render

PREREG = 'd054b01'
FREEZE_COMMIT = 'd8e409217735e4b8d04669f24458f0706871fdd4'
FREEZE_NAME = 'FREEZE_20261018T210631Z.md'
DIRECTIONS = [('R1', 'Q_T', 'Q_N', 'N'), ('R2', 'Q_N', 'Q_T', 'T')]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def committed(commit, name):
    return subprocess.check_output(['git', 'show', f'{commit}:experiments/07_numeric_selection/{name}'], cwd=REPO)


def verify(args):
    gate(args.expected_commit)
    assert (HERE/'ADDENDUM_H11.md').read_bytes() == committed(PREREG, 'ADDENDUM_H11.md')
    subprocess.run(['git', 'merge-base', '--is-ancestor', PREREG, args.expected_commit], cwd=REPO, check=True)
    raw = committed(FREEZE_COMMIT, FREEZE_NAME)
    frozen = json.loads(raw.decode().split('```json\n', 1)[1].split('\n```', 1)[0])
    side = read(Path(args.h7_output)/'freeze.json')
    assert hashlib.sha256(raw).hexdigest() == side['freeze_sha256']
    assert all(side[k] == v for k, v in frozen.items())
    assert all(sha(HERE/name) == digest for name, digest in frozen['source_sha256'].items())
    assert sha(HERE/'DESIGN.md') == frozen['design_sha256']
    assert sha(Path(args.stimuli)/'manifest.json') == frozen['stimuli_manifest_sha256']
    assert len(frozen['heads']) == 8 and len(frozen['random_sets']) == 100
    assert frozen['heldout_seeds'] == list(HELDOUT_SEEDS)
    return frozen, side['freeze_sha256']


def first_number(text):
    match = re.search(r'[+-]?\d+(?:[.,]\d+)*', text)
    return int(match[0]) if match and re.fullmatch(r'[+-]?\d+', match[0]) else None


def score(outputs, target, clean=None):
    metrics = {}
    for parser, field in [('unique_integer', 'answer'), ('first_number', 'first_number')]:
        for v in outputs:
            assert v['answer'] == parse_integer(v['text'])
            v['first_number'] = first_number(v['text'])
        n = len(outputs)
        switch = sum(v[field] == v[target] for v in outputs)
        stay = sum(v[field] == v['T' if target == 'N' else 'N'] for v in outputs)
        valid = sum(v[field] in [v['N'], v['T']] for v in outputs)
        item = dict(n=n, switch_count=switch, switch_rate=switch/n,
                    stay_count=stay, valid_count=valid, validity=valid/n)
        if clean is not None:
            gold = 'T' if target == 'N' else 'N'
            eligible = [i for i, v in enumerate(clean) if v['answer'] == v[gold]]
            count = sum(outputs[i][field] == outputs[i][target] for i in eligible)
            item.update(clean_correct_n=len(eligible), conditional_switch_count=count,
                        conditional_switch_rate=count/len(eligible) if eligible else None)
        metrics[parser] = item
    return metrics


def prepare(args, frozen, freeze_hash):
    root = Path(args.output)
    assert not (root/'manifest.json').exists()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', local_files_only=True)
    files = {}
    for seed in HELDOUT_SEEDS:
        original = rows_for(args, seed)
        swapped = []
        for r in original:
            row = {k:r[k] for k in ['seed','item','pool','company','N','T','query_type','question']}
            row['context'] = (f"Management said it expects next-quarter revenue of about ${r['T']} million. "
                              f"{r['company']} reported revenue of ${r['N']} million last quarter.")
            assert r['context'] == (f"{r['company']} reported revenue of ${r['N']} million last quarter. "
                                    f"Management said it expects next-quarter revenue of about ${r['T']} million.")
            assert render({**row, 'context':r['context']}, tok) == r
            swapped.append(render(row, tok))
        name = f'stimuli_{seed}.json'
        files[name] = dump(root/name, dict(original=original, swapped=swapped))
    dump(root/'manifest.json', dict(prereg_commit=PREREG, source_commit=args.expected_commit,
         freeze_sha256=freeze_hash, prepared_utc=now(), files=files,
         heads=frozen['heads'], random_sets=frozen['random_sets'],
         original_manifest_sha256=frozen['stimuli_manifest_sha256']))


def stimuli(args, seed):
    root = Path(args.output)
    manifest = read(root/'manifest.json')
    assert manifest['source_commit'] == args.expected_commit
    name = f'stimuli_{seed}.json'
    assert sha(root/name) == manifest['files'][name]
    return read(root/name)


def start(args, phase):
    root = Path(args.output)
    path = root/f'{phase}_{args.seed}.json'
    assert not path.exists()
    with (root/f'{phase}_{args.seed}.started').open('x') as stream:
        stream.write(now()+'\nOne attempt; no silent retry.\n')
    return path


def clean(args):
    path = start(args, 'clean')
    rows = stimuli(args, args.seed)['swapped']
    h = Harness('Qwen/Qwen3-4B')
    out = dict(seed=args.seed, started_utc=now(), outputs={})
    for q in ['Q_N', 'Q_T']:
        values, _ = h.measure([r for r in rows if r['query_type'] == q])
        out['outputs'][q] = values
        score(values, q[-1])
        dump(path, out)
    out.update(complete=True, finished_utc=now())
    dump(path, out)


def clean_gate(args):
    root = Path(args.output)
    runs = [read(root/f'clean_{s}.json') for s in HELDOUT_SEEDS]
    assert all(r.get('complete') for r in runs)
    pooled = {q:score([v for r in runs for v in r['outputs'][q]], q[-1]) for q in ['Q_N','Q_T']}
    passed = all(m['unique_integer']['switch_rate'] >= .9 for m in pooled.values())
    out = dict(passed=passed, pooled=pooled, computed_utc=now(),
               clean_sha256={str(s):sha(root/f'clean_{s}.json') for s in HELDOUT_SEEDS},
               per_seed=[dict(seed=r['seed'], queries={q:score(r['outputs'][q], q[-1]) for q in ['Q_N','Q_T']}) for r in runs])
    dump(root/'gate.json', out)
    print(json.dumps(out, indent=2))


def data_for(h, rows, captures):
    result = {}
    for q in ['Q_N', 'Q_T']:
        selected = [r for r in rows if r['query_type'] == q]
        outputs, cache = h.measure(selected, captures=captures)
        batch, pos = h.batch(selected)
        result[q] = dict(rows=selected, outputs=outputs, cache=cache, batch=batch, pos=pos)
    return result


def noise_patch(h, base, donor, heads, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed + RANDOM_SEED)
    cache = {key:[v.clone() for v in values] for key,values in base['cache'].items()}
    for layer, head in heads:
        key = (layer, 'head', 'prefill')
        sl = slice(head*h.head_dim, (head+1)*h.head_dim)
        for i in range(64):
            delta = (donor['cache'][key][i][:,sl] - base['cache'][key][i][:,sl]).float()
            noise = torch.randn(delta.shape, device='cuda', generator=generator)
            noise = noise/noise.norm().clamp_min(1e-12)*delta.norm()
            cache[key][i][:,sl] += noise.to(torch.float16)
    return head_patches(cache, heads)


def run(args, frozen, freeze_hash):
    root = Path(args.output)
    gate_result = read(root/'gate.json')
    assert gate_result['passed'], 'Clean gate failed; no H11 patches allowed'
    assert all(sha(root/f'clean_{s}.json') == gate_result['clean_sha256'][str(s)] for s in HELDOUT_SEEDS)
    path = start(args, 'patch')
    started = time.perf_counter()
    h = Harness('Qwen/Qwen3-4B')
    rows = stimuli(args, args.seed)
    captures = [(layer, 'head', 'prefill') for layer in sorted({l for l,_ in frozen['heads']})]
    data = {order:data_for(h, rows[order], captures) for order in ['original','swapped']}
    saved_clean = read(root/f'clean_{args.seed}.json')['outputs']
    for q in saved_clean:
        assert [(v['text'],v['generated_ids']) for v in data['swapped'][q]['outputs']] == [(v['text'],v['generated_ids']) for v in saved_clean[q]]
    out = dict(seed=args.seed, source_commit=args.expected_commit, freeze_sha256=freeze_hash,
               gate_sha256=sha(root/'gate.json'), started_utc=now(), sets=[], controls=[],
               clean={order:{q:d['outputs'] for q,d in queries.items()} for order,queries in data.items()})
    for index, heads in enumerate([frozen['heads']] + frozen['random_sets']):
        record = dict(index=index, directions=[])
        for direction, b, d, target in DIRECTIONS:
            base, donor = data['swapped'][b], data['original'][d]
            assert all((x['item'],x['N'],x['T']) == (y['item'],y['N'],y['T']) for x,y in zip(base['rows'],donor['rows']))
            outputs = intervention(h, base, head_patches(donor['cache'], heads))
            record['directions'].append(dict(direction=direction, metrics=score(outputs,target,base['outputs']), outputs=outputs))
        out['sets'].append(record)
        dump(path, out)
        print('H11', args.seed, 'set', index, flush=True)
    for direction, b, d, target in DIRECTIONS:
        base, donor = data['swapped'][b], data['original'][d]
        for kind, patches in [('matched_norm',noise_patch(h,base,donor,frozen['heads'],args.seed)),
                              ('same_order_swapped',head_patches(data['swapped'][d]['cache'],frozen['heads']))]:
            outputs = intervention(h,base,patches)
            out['controls'].append(dict(kind=kind,direction=direction,
                metrics=score(outputs,target,base['outputs']),outputs=outputs))
    out.update(complete=True, finished_utc=now(), wall_seconds=time.perf_counter()-started)
    dump(path, out)


def aggregate(args, frozen, freeze_hash):
    root = Path(args.output)
    packet = root/'packet'
    packet.mkdir(exist_ok=True)
    gate_result = read(root/'gate.json')
    out = dict(gate=gate_result, prereg_commit=PREREG, source_commit=args.expected_commit,
               freeze_sha256=freeze_hash, per_seed=[], verdict='gate_failed', sources={})
    if gate_result['passed']:
        for seed in HELDOUT_SEEDS:
            run = read(root/f'patch_{seed}.json')
            assert run.get('complete') and [r['index'] for r in run['sets']] == list(range(101))
            assert run['gate_sha256'] == sha(root/'gate.json') and run['source_commit'] == args.expected_commit
            compact = dict(seed=seed, sets=[], controls=[])
            for entry in run['sets']:
                rr = []
                for r in entry['directions']:
                    _,b,_,target = next(d for d in DIRECTIONS if d[0] == r['direction'])
                    assert r['metrics'] == score(r['outputs'],target,run['clean']['swapped'][b])
                    rr.append({k:v for k,v in r.items() if k != 'outputs'})
                compact['sets'].append(dict(index=entry['index'],directions=rr))
            for r in run['controls']:
                _,b,_,target = next(d for d in DIRECTIONS if d[0] == r['direction'])
                assert r['metrics'] == score(r['outputs'],target,run['clean']['swapped'][b])
                compact['controls'].append({k:v for k,v in r.items() if k != 'outputs'})
            dump(packet/f'h11_sets_{seed}.json',compact)
            for direction,_,_,_ in DIRECTIONS:
                row = dict(seed=seed,direction=direction)
                selected = next(r for r in compact['sets'][0]['directions'] if r['direction']==direction)
                for parser in ['unique_integer','first_number']:
                    randoms = [next(r for r in e['directions'] if r['direction']==direction)['metrics'][parser] for e in compact['sets'][1:]]
                    maximum = max(r['switch_rate'] for r in randoms)
                    row[parser] = dict(**selected['metrics'][parser],random_max=maximum,
                        random_max_count=max(r['switch_count'] for r in randoms),
                        above_all_random=selected['metrics'][parser]['switch_rate']>maximum,
                        random_validity_min=min(r['validity'] for r in randoms),
                        random_validity_max=max(r['validity'] for r in randoms))
                row['controls'] = [r for r in compact['controls'] if r['direction']==direction]
                out['per_seed'].append(row)
        out['directions'] = {d:dict(mean_switch=sum(r['unique_integer']['switch_rate'] for r in out['per_seed'] if r['direction']==d)/8,
                                   every_seed_above_random=all(r['unique_integer']['above_all_random'] for r in out['per_seed'] if r['direction']==d)) for d,_,_,_ in DIRECTIONS}
        role = all(v['mean_switch'] >= .5 and v['every_seed_above_random'] for v in out['directions'].values())
        position = all(v['mean_switch'] <= .1 for v in out['directions'].values())
        out['verdict'] = 'role' if role else 'position' if position else 'mixed'
    names = ['manifest.json','gate.json'] + [f'{phase}_{s}.json' for s in HELDOUT_SEEDS for phase in (['stimuli','clean','patch'] if gate_result['passed'] else ['stimuli','clean'])]
    for name in names:
        raw = (root/name).read_bytes()
        out['sources'][name] = dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        (packet/f'h11_raw_{name}.gz').write_bytes(gzip.compress(raw,mtime=0))
    table = []
    for r in out['per_seed']:
        for parser in ['unique_integer','first_number']:
            row = dict(seed=r['seed'],direction=r['direction'],parser=parser,**r[parser])
            for c in r['controls']:
                row.update({c['kind']+'_'+k:v for k,v in c['metrics'][parser].items() if k in ['switch_count','switch_rate','valid_count','validity']})
            table.append(row)
    if table:
        with (packet/'h11_per_seed.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(table)
    dump(packet/'h11.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['sources','gate','per_seed']},indent=2))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','clean','gate','run','aggregate'])
    p.add_argument('--seed',type=int,choices=HELDOUT_SEEDS)
    for name in ['expected-commit','stimuli','h7-output','output']:
        p.add_argument('--'+name,required=True)
    args = p.parse_args()
    Path(args.output).mkdir(parents=True,exist_ok=True)
    frozen,freeze_hash = verify(args)
    if args.phase == 'prepare': prepare(args,frozen,freeze_hash)
    elif args.phase == 'clean': clean(args)
    elif args.phase == 'gate': clean_gate(args)
    elif args.phase == 'run': run(args,frozen,freeze_hash)
    else: aggregate(args,frozen,freeze_hash)


if __name__ == '__main__':
    main()

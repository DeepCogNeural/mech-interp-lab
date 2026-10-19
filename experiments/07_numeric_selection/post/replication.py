"""Pre-registered replication of the H7 sufficiency and necessity rules on fresh seeds.

Reuses the frozen harness, head set, random sets and dev means without change.
`sanity` reruns one original held-out seed into a separate directory and
`sanity_check` compares it with the saved output; neither revises any verdict.
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from patch import Harness
from scoring import wilson
from stage2 import RANDOM_SEED, REPO, caches, effects, gate, head_patches, intervention, read, rows_for, sha
from stimuli import DEV_SEEDS, HELDOUT_SEEDS, dump, prompts, render

FREEZE_COMMIT = 'd8e409217735e4b8d04669f24458f0706871fdd4'
FREEZE_NAME = 'FREEZE_20261018T210631Z.md'
ADDENDUM = 'ADDENDUM_REPLICATION.md'
SANITY_SEED = 20261201
REPLICATION_SEEDS = list(range(20261301, 20261309))
BOOTSTRAP_SEED = 20261300
BOOTSTRAP_DRAWS = 10000
T7 = 2.364624251
DIRECTIONS = [('Q_T', 'Q_N', 'N'), ('Q_N', 'Q_T', 'T')]
assert not set(REPLICATION_SEEDS) & set(DEV_SEEDS + HELDOUT_SEEDS + [RANDOM_SEED])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def committed(commit, name):
    return subprocess.check_output(['git', 'show', f'{commit}:experiments/07_numeric_selection/{name}'], cwd=REPO)


def verify(args):
    gate(args.expected_commit)
    raw = committed(FREEZE_COMMIT, FREEZE_NAME)
    frozen = json.loads(raw.decode().split('```json\n', 1)[1].split('\n```', 1)[0])
    side = read(Path(args.h7_output)/'freeze.json')
    assert hashlib.sha256(raw).hexdigest() == side['freeze_sha256']
    assert all(side[k] == v for k, v in frozen.items())
    assert all(sha(HERE/name) == digest for name, digest in frozen['source_sha256'].items())
    assert sha(HERE/'DESIGN.md') == frozen['design_sha256']
    assert sha(frozen['mean_path']) == frozen['mean_sha256']
    assert len(frozen['heads']) == 8 and len(frozen['random_sets']) == 100
    assert frozen['dev_seeds'] == DEV_SEEDS and frozen['heldout_seeds'] == HELDOUT_SEEDS
    output = Path(args.output).resolve()
    assert Path(args.h7_output).resolve() not in [output, *output.parents], 'Never write into the original output tree'
    return frozen, side['freeze_sha256']


def verify_addendum(args):
    assert args.prereg_commit, 'New seeds require the committed pre-registration'
    text = (HERE/ADDENDUM).read_bytes()
    assert text == committed(args.prereg_commit, ADDENDUM)
    subprocess.run(['git', 'merge-base', '--is-ancestor', args.prereg_commit, args.expected_commit], cwd=REPO, check=True)
    assert all(str(seed).encode() in text for seed in REPLICATION_SEEDS)
    return hashlib.sha256(text).hexdigest()


def reserve(path):
    assert not path.exists(), 'Never overwrite a partial or completed seed'
    with path.with_suffix('.started').open('x') as stream:
        stream.write(now()+'\nOne attempt; no silent retry.\n')


def run_sets(args, frozen, freeze_hash, seed, outdir, extra):
    """Same sufficiency/necessity loop, batch composition and order as stage2.heldout."""
    path = outdir/f'heldout_{seed}.json'
    reserve(path)
    started = time.perf_counter()
    h = Harness('Qwen/Qwen3-4B')
    rows = rows_for(args, seed)
    data = caches(h, rows, ['head'])
    means = torch.load(frozen['mean_path'], weights_only=True, map_location='cpu')
    mean_cache = {key: [val[None, :] for _ in range(64)] for key, val in means.items()}
    out = dict(seed=seed, freeze_sha256=freeze_hash, source_commit=args.expected_commit, started_utc=now(),
               clean=data['Q_N']['outputs']+data['Q_T']['outputs'], sets=[], **extra)
    for index, heads in enumerate([frozen['heads']]+frozen['random_sets']):
        record = dict(index=index, label='selected' if index == 0 else 'layer-matched random', sufficiency=[], necessity=[])
        for b, d, target in DIRECTIONS:
            vals = intervention(h, data[b], head_patches(data[d]['cache'], heads))
            record['sufficiency'].append(dict(direction=b+'->'+target, outputs=vals, **effects(data[b], vals, target)))
            vals = intervention(h, data[b], head_patches(mean_cache, heads))
            gold = b[-1]
            cleanacc = sum(o['answer'] == o[gold] for o in data[b]['outputs'])/64
            accuracy = sum(o['answer'] == o[gold] for o in vals)/64
            record['necessity'].append(dict(query_type=b, clean_accuracy=cleanacc, ablated_accuracy=accuracy, drop=cleanacc-accuracy,
                                            validity=sum(o['answer'] in [o['N'], o['T']] for o in vals)/64, outputs=vals))
        out['sets'].append(record)
        dump(path, out)
        print('replication', seed, 'set', index, flush=True)
    out.update(complete=True, finished_utc=now(), wall_seconds=time.perf_counter()-started)
    dump(path, out)


def sanity(args, frozen, freeze_hash):
    assert args.seed == SANITY_SEED
    assert sha(Path(args.stimuli)/'manifest.json') == frozen['stimuli_manifest_sha256']
    outdir = Path(args.output)/'sanity'
    outdir.mkdir(parents=True, exist_ok=True)
    run_sets(args, frozen, freeze_hash, args.seed, outdir, dict(purpose='determinism check of an original held-out seed'))


def counts(run):
    """Selected-set validity and necessity counts, as defined by the frozen rules."""
    selected = run['sets'][0]
    out = dict(seed=run['seed'], sufficiency={}, necessity={})
    for r in selected['sufficiency']:
        target = r['direction'][-1]
        out['sufficiency'][r['direction']] = dict(n=len(r['outputs']), flip_count=sum(o['answer'] == o[target] for o in r['outputs']),
                                                  valid_count=sum(o['answer'] in [o['N'], o['T']] for o in r['outputs']))
    clean = {q: [o for o in run['clean'] if o['query_type'] == q] for q in ['Q_N', 'Q_T']}
    for r in selected['necessity']:
        gold = r['query_type'][-1]
        out['necessity'][r['query_type']] = dict(n=len(r['outputs']), clean_correct=sum(o['answer'] == o[gold] for o in clean[r['query_type']]),
                                                 ablated_correct=sum(o['answer'] == o[gold] for o in r['outputs']),
                                                 valid_count=sum(o['answer'] in [o['N'], o['T']] for o in r['outputs']))
    return out


def same_outputs(a, b):
    keys = ['text', 'generated_ids', 'answer']
    return [[o[k] for k in keys] for o in a] == [[o[k] for k in keys] for o in b]


def sanity_check(args, frozen, freeze_hash):
    original = read(Path(args.h7_output)/f'heldout_{SANITY_SEED}.json')
    rerun = read(Path(args.output)/'sanity'/f'heldout_{SANITY_SEED}.json')
    assert original.get('complete') and rerun.get('complete') and len(rerun['sets']) == 101
    assert original['freeze_sha256'] == rerun['freeze_sha256'] == freeze_hash
    mismatched_sets = []
    max_abs_d = 0.0
    for a, b in zip(original['sets'], rerun['sets']):
        for kind in ['sufficiency', 'necessity']:
            for x, y in zip(a[kind], b[kind]):
                if not same_outputs(x['outputs'], y['outputs']):
                    mismatched_sets.append([a['index'], kind, x.get('direction', x.get('query_type'))])
                max_abs_d = max([max_abs_d]+[abs(p['d']-q['d']) for p, q in zip(x['outputs'], y['outputs'])])
    result = dict(seed=SANITY_SEED, original=counts(original), rerun=counts(rerun),
                  counts_match=counts(original) == counts(rerun), clean_outputs_match=same_outputs(original['clean'], rerun['clean']),
                  mismatched_sets=mismatched_sets, all_101_sets_match=not mismatched_sets, max_abs_delta_d=max_abs_d,
                  original_sha256=sha(Path(args.h7_output)/f'heldout_{SANITY_SEED}.json'),
                  rerun_sha256=sha(Path(args.output)/'sanity'/f'heldout_{SANITY_SEED}.json'), computed_utc=now())
    dump(Path(args.output)/'sanity'/'sanity_check.json', result)
    print(json.dumps(result, indent=2))


def prepare(args, frozen, freeze_hash, addendum_hash):
    root = Path(args.output)/'stimuli'
    assert not (root/'manifest.json').exists()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', local_files_only=True)
    manifest = {}
    for seed in REPLICATION_SEEDS:
        rendered = [render(r, tok) for r in prompts(seed, 'B')]
        name = f'B_{seed}.json'
        manifest[name] = dict(sha256=dump(root/name, rendered), items=64, prompts=len(rendered))
    dump(root/'manifest.json', manifest)
    dump(Path(args.output)/'prepared.json', dict(prepared_utc=now(), prereg_commit=args.prereg_commit, addendum_sha256=addendum_hash,
                                                 source_commit=args.expected_commit, freeze_sha256=freeze_hash,
                                                 stimuli_manifest_sha256=sha(root/'manifest.json')))


def replicate(args, frozen, freeze_hash, addendum_hash):
    assert args.seed in REPLICATION_SEEDS
    prepared = read(Path(args.output)/'prepared.json')
    assert Path(args.stimuli).resolve() == (Path(args.output)/'stimuli').resolve()
    assert sha(Path(args.stimuli)/'manifest.json') == prepared['stimuli_manifest_sha256']
    run_sets(args, frozen, freeze_hash, args.seed, Path(args.output),
             dict(prereg_commit=args.prereg_commit, addendum_sha256=addendum_hash))


def interval_rate(k, n):
    return dict(count=k, total=n, rate=k/n, wilson95=wilson(k, n))


def bootstrap_mean(values):
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(BOOTSTRAP_DRAWS))
    return [means[int(0.025*BOOTSTRAP_DRAWS)], means[int(0.975*BOOTSTRAP_DRAWS)-1]]


def aggregate(args, frozen, freeze_hash, addendum_hash):
    root = Path(args.output)
    runs = []
    for seed in REPLICATION_SEEDS:
        run = read(root/f'heldout_{seed}.json')
        assert run.get('complete') and [s['index'] for s in run['sets']] == list(range(101))
        assert run['freeze_sha256'] == freeze_hash and run['prereg_commit'] == args.prereg_commit
        runs.append(run)
    per_seed, directions, invalid = [], {}, []
    for b, d, target in DIRECTIONS:
        name = b+'->'+target
        rows = []
        for run in runs:
            values = [next(x for x in entry['sufficiency'] if x['direction'] == name) for entry in run['sets']]
            outputs = values[0]['outputs']
            assert values[0]['validity'] == sum(o['answer'] in [o['N'], o['T']] for o in outputs)/64
            invalid += [dict(seed=run['seed'], direction=name, item=o['item'], N=o['N'], T=o['T'], text=o['text'], answer=o['answer'])
                        for o in outputs if o['answer'] not in [o['N'], o['T']]]
            rows.append(dict(seed=run['seed'], direction=name, n=64,
                             flip_count=sum(o['answer'] == o[target] for o in outputs),
                             valid_count=sum(o['answer'] in [o['N'], o['T']] for o in outputs),
                             flip_rate=values[0]['flip_rate'], validity=values[0]['validity'],
                             random_max=max(v['flip_rate'] for v in values[1:]),
                             above_all_random=values[0]['flip_rate'] > max(v['flip_rate'] for v in values[1:])))
        rates = [r['flip_rate'] for r in rows]
        mean = sum(rates)/8
        sd = math.sqrt(sum((x-mean)**2 for x in rates)/7)
        valid = sum(r['valid_count'] for r in rows)
        directions[name] = dict(mean_flip=mean, t7_lower95=mean-T7*sd/math.sqrt(8),
                                every_seed_above_random=all(r['above_all_random'] for r in rows),
                                min_seed_validity=min(r['validity'] for r in rows),
                                seeds_below_validity_bar=[r['seed'] for r in rows if r['validity'] < .95],
                                pooled_validity=interval_rate(valid, 512),
                                flips=interval_rate(sum(r['flip_count'] for r in rows), 512))
        directions[name]['validity_bar_passes'] = directions[name]['min_seed_validity'] >= .95
        directions[name]['rule_passes'] = (mean >= .7 and directions[name]['t7_lower95'] >= .6 and
                                           directions[name]['every_seed_above_random'] and directions[name]['validity_bar_passes'])
        per_seed += rows
    necessity_rows, seed_drops = [], []
    for run in runs:
        queries = []
        for r in run['sets'][0]['necessity']:
            gold = r['query_type'][-1]
            clean = [o for o in run['clean'] if o['query_type'] == r['query_type']]
            queries.append(dict(query_type=r['query_type'], n=64, clean_correct=sum(o['answer'] == o[gold] for o in clean),
                                ablated_correct=sum(o['answer'] == o[gold] for o in r['outputs']),
                                drop=r['drop'], validity=r['validity']))
        drop = sum(q['drop'] for q in queries)/2
        seed_drops.append(drop)
        necessity_rows.append(dict(seed=run['seed'], mean_drop=drop, queries=queries))
    random_drops = [sum(sum(x['drop'] for x in run['sets'][j]['necessity'])/2 for run in runs)/8 for j in range(1, 101)]
    mean_drop = sum(seed_drops)/8
    lost = sum(q['clean_correct']-q['ablated_correct'] for r in necessity_rows for q in r['queries'])
    necessity = dict(mean_drop=mean_drop, bootstrap95_seed_level=bootstrap_mean(seed_drops), lost_correct=lost, denominator=1024,
                     per_seed=necessity_rows, random_max_mean_drop=max(random_drops), random_set_mean_drops=random_drops,
                     drop_bar_passes=mean_drop >= .3)
    necessity['rule_passes'] = necessity['drop_bar_passes'] and max(random_drops) < .1
    validity_all = sum(r['valid_count'] for r in per_seed)
    out = dict(seeds=REPLICATION_SEEDS, prereg_commit=args.prereg_commit, addendum_sha256=addendum_hash,
               source_commit=args.expected_commit, freeze_sha256=freeze_hash,
               bootstrap=dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS, unit='seed', interval='percentile 2.5/97.5'),
               sufficiency=dict(directions=directions, per_seed=per_seed, pooled_validity_both=interval_rate(validity_all, 1024),
                                invalid_selected_outputs=invalid,
                                validity_bar_passes=all(v['validity_bar_passes'] for v in directions.values()),
                                rule_passes=all(v['rule_passes'] for v in directions.values())),
               necessity=necessity,
               raw_sha256={f'replication/heldout_{r["seed"]}.json': sha(root/f'heldout_{r["seed"]}.json') for r in runs})
    out['replication_passes'] = out['sufficiency']['rule_passes'] and necessity['rule_passes']
    out['computed_utc'] = now()
    dump(root/'replication.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ['sufficiency', 'necessity', 'raw_sha256']}, indent=2))
    print(json.dumps(dict(validity={k: (v['min_seed_validity'], v['pooled_validity']) for k, v in directions.items()},
                          necessity=(mean_drop, necessity['bootstrap95_seed_level'], lost)), indent=2))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['sanity', 'sanity_check', 'prepare', 'run', 'aggregate'])
    p.add_argument('--seed', type=int)
    p.add_argument('--prereg-commit')
    for name in ['expected-commit', 'stimuli', 'h7-output', 'output']:
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    frozen, freeze_hash = verify(args)
    if args.phase == 'sanity': sanity(args, frozen, freeze_hash)
    elif args.phase == 'sanity_check': sanity_check(args, frozen, freeze_hash)
    else:
        addendum_hash = verify_addendum(args)
        {'prepare': prepare, 'run': replicate, 'aggregate': aggregate}[args.phase](args, frozen, freeze_hash, addendum_hash)


if __name__ == '__main__':
    main()

"""Pre-registered fallback generator. Not executed in stage 1."""
import argparse,random
from pathlib import Path
from stimuli import DEV_SEEDS,HELDOUT_SEEDS,items,render,dump

def table_prompts(seed,pool):
    labels=['Revenue','Operating income','Cash flow'] if pool=='A' else ['Net sales','Gross profit','Capital spending']
    years=[2021,2022,2023] if pool=='A' else [2024,2025,2026]
    rng=random.Random(seed+700000);result=[];lo=0 if pool=='A' else 50
    for row in items(seed,pool):
        coords=rng.sample(range(9),2);digits=[d for d in range(1,10) if d not in [row['N']//100,row['T']//100]];rng.shuffle(digits)
        cells=[None]*9;cells[coords[0]]=row['N'];cells[coords[1]]=row['T']
        for index in range(9):
            if cells[index] is None:cells[index]=100*digits.pop()+rng.randrange(lo,lo+50)
        context=row['company']+' reported the following amounts in $ million.\n\n'
        context+='| Line item | '+' | '.join(map(str,years))+' |\n| --- | --- | --- | --- |\n'
        for i,label in enumerate(labels):context+='| '+label+' | '+' | '.join('$'+str(v) for v in cells[3*i:3*i+3])+' |\n'
        for q,cell in zip(['Q_N','Q_T'],coords):
            label=labels[cell//3];year=years[cell%3]
            question=f'What was {label} in {year}?' if pool=='A' else f'For {year}, what value is listed for {label}?'
            result.append(dict(row,context=context,query_type=q,question=question+' Answer with the number only.',table_cells=cells,target_coordinates=coords))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B',local_files_only=True);out=Path(a.output)
    assert not out.exists();out.mkdir(parents=True)
    manifest={}
    for seed in DEV_SEEDS+HELDOUT_SEEDS:
        pool='A' if seed in DEV_SEEDS else 'B';rows=[render(r,tok) for r in table_prompts(seed,pool)]
        name=f'{pool}_{seed}.json';manifest[name]=dict(sha256=dump(out/name,rows),items=64,prompts=128,model_run_allowed_stage1=False)
    dump(out/'manifest.json',manifest)
if __name__=='__main__':main()

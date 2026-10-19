"""Render figures from the compact packet. No model execution or network."""
from pathlib import Path
import argparse
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = HERE / 'results', HERE / 'figures'
plt.style.library['nature-npg'] = matplotlib.rc_params_from_file(Path(__file__).with_name('nature-npg.mplstyle'), use_default_template=False)
plt.style.use('nature-npg')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['svg.hashsalt'] = 'exp07-numeric-selection'
RED, BLUE, GRAY, BROWN, TAN = '#E64B35', '#3C5488', '#4D4D4D', '#7E6148', '#B09C85'
DIRECTIONS = ['Q_T->N', 'Q_N->T']
LABELS = ['Guidance → past revenue (T → N)', 'Past revenue → guidance (N → T)']

def read(name):
    return json.loads((RESULTS / name).read_text())

def title(fig, letter, text, note):
    fig.suptitle(f'{letter}  {text}', x=.02, y=.99, ha='left', fontsize=10, weight='bold')
    fig.text(.02, .02, note, fontsize=7, color=GRAY, va='bottom')

def save(fig, name, rect=(.02, .12, .99, .91)):
    fig.tight_layout(rect=rect, pad=.8)
    for suffix in ['png', 'svg']:
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=300,
                    metadata={'Date': None} if suffix == 'svg' else None)
        if suffix == 'svg':
            path = FIGURES / f'{name}.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)

def label_rate(ax, x, y, text, offset=(0, 6), **kwargs):
    ax.annotate(text, (x, y), xytext=offset, textcoords='offset points',
                ha='center', fontsize=7, color=GRAY, **kwargs)

def residual():
    data = read('residual_scan.json')['per_seed']
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for direction, color, offset in zip(DIRECTIONS, [BROWN, TAN], [(0,-13),(0,7)]):
        values = [100*np.mean([r['flip_rate'] for r in data if r['layer']==l and r['direction']==direction]) for l in range(36)]
        ax.plot(range(36), values, 'o-', color=color, label=direction.replace('->',' → '))
        for l in [21,22,23,35]:
            shift = {(DIRECTIONS[0],22):(-27,-15), (DIRECTIONS[0],23):(25,-15),
                     (DIRECTIONS[1],22):(-28,-13), (DIRECTIONS[1],23):(25,10)}.get((direction,l),offset)
            label_rate(ax,l,values[l],f'{values[l]:.1f}',shift,
                       arrowprops={'arrowstyle':'-','lw':.4} if l in [22,23] else None)
    ax.axhline(80, color=GRAY, ls='--', lw=.5)
    ax.text(.5,82,'80% in both directions',fontsize=7,color=GRAY)
    ax.axvline(23,color=GRAY,ls=':',lw=.7)
    ax.text(24,46,'L* = 23',fontsize=8,weight='bold')
    ax.set(xlabel='Decoder layer (zero-based)',ylabel='Complete-answer flip (%)',ylim=(-5,114),xlim=(-.5,36.5))
    ax.legend(loc='center left')
    title(fig,'a','Full-state replacement first clears 80% in both directions at layer 23',
          'Development only: 4 seeds × 64 items = 256 per direction. Invalid answers count as failures.\nFull-residual replacement is descriptive localisation; it can copy the entire donor state.')
    save(fig,'a_residual_scan')

def heads():
    rows=read('head_ranking.json'); frozen={tuple(x) for x in read('frozen_selection.json')['heads']}
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),gridspec_kw={'width_ratios':[1.65,1]})
    ax,b=axes
    for selected,color in [(False,'#A7B2C9'),(True,RED)]:
        rr=[r for r in rows if ((r['layer'],r['head']) in frozen)==selected]
        ax.scatter([r['layer']+(r['head']-15.5)/36 for r in rr],[r['mean_oriented_delta_d'] for r in rr],s=12 if selected else 5,color=color,alpha=1 if selected else .7,label='Frozen 8' if selected else 'Other 760')
    ax.annotate('L23H26: 12.92',(23+10.5/36,rows[0]['mean_oriented_delta_d']),xytext=(8,11),arrowprops={'arrowstyle':'-','lw':.5},color=GRAY)
    ax.axhline(0,color=GRAY,lw=.5)
    ax.set(xlabel='Layer (head indices spread within layer)',ylabel='Donor-oriented Δd (logits)',xlim=(-1,24.3),ylim=(-1,14.5),xticks=[0,4,8,12,16,20,23])
    ax.legend(loc='upper left',fontsize=7)
    rr=[r for r in rows if (r['layer'],r['head']) in frozen]
    y=np.arange(8)
    b.barh(y,[r['mean_oriented_delta_d'] for r in rr],color=RED,height=.58)
    b.set(yticks=y,yticklabels=[f"L{r['layer']}H{r['head']}" for r in rr],xlabel='Donor-oriented Δd (logits)',xlim=(0,15))
    b.invert_yaxis();b.tick_params(axis='y',left=False,right=False)
    for i,r in enumerate(rr): b.text(r['mean_oriented_delta_d']+.2,i,f"{r['mean_oriented_delta_d']:.2f}",va='center',fontsize=7,color=GRAY)
    title(fig,'b','One dominant single head; eight heads frozen for joint intervention',
          '768 heads in layers 0–23; mean of 4 dev seeds × 2 directions × 64 items per head.\nΔd = patched minus clean first-digit logit difference; signs point toward the donor answer. No single-head flip claim.')
    save(fig,'b_single_heads')

def heldout():
    h7=read('h7.json'); real=read('h10b.json')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.3))
    for col,d in enumerate(DIRECTIONS):
        for row in range(2):
            ax=axes[row,col]
            rr=h7['directions'][d]['per_seed'] if row==0 else [r for r in real['per_group'] if r['direction']==d]
            x=np.arange(len(rr)); y=[100*r['flip_rate'] for r in rr]; z=[100*r['random_max'] for r in rr]
            ax.plot(x,y,'o-',color=RED,label='Frozen 8'); ax.plot(x,z,'s--',color=BLUE,label='Maximum of 100 random sets')
            for i,r in enumerate(rr):
                label_rate(ax,i,y[i],f"{round(64*r['flip_rate'])}/64",(0,6))
                label_rate(ax,i,z[i],f"{round(64*r['random_max'])}/64",(0,6))
            ax.set(ylim=(-7,119),xticks=x,xticklabels=[str(r['seed'])[-2:] for r in rr] if row==0 else [str(r['group']) for r in rr],ylabel='Complete-answer flip (%)',xlabel='Held-out seed suffix (202612xx)' if row==0 else 'Document-disjoint group',title=d.replace('->',' → ') if row==0 else d.replace('Q_T->N','Second amount → first').replace('Q_N->T','First amount → second'))
            ax.tick_params(axis='x',top=False)
            if row==0 and col==0:
                ax.scatter([3],[y[3]],s=45,facecolors='none',edgecolors=GRAY,lw=.7)
                ax.annotate('59/64 valid: H7 gate missed',(3,y[3]),xytext=(.4,58),fontsize=7,arrowprops={'arrowstyle':'-','lw':.5})
    axes[0,0].legend(loc='center left',bbox_to_anchor=(0,.28),fontsize=7)
    title(fig,'c','Frozen heads strongly redirect synthetic and real-filing answers',
          'Top: synthetic H7, 8 seeds × 64; bottom: real 10-K H10b, 4 groups × 64 per direction. Labels are counts / 64.\nH7 uses the frozen unique-integer parser and fails its strict validity gate. H10b uses its separately registered decimal parser.')
    fig.text(.02,.93,'Red: frozen 8 heads    Blue: maximum of the same 100 layer-matched random sets',fontsize=7,color=GRAY)
    save(fig,'c_heldout_transfer',rect=(.02,.13,.99,.9))

def mismatch():
    rows=read('controls.json'); real=read('h10b.json')['per_group']
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8))
    for ax,is_real in zip(axes,[False,True]):
        den=256 if is_real else 512
        for i,d in enumerate(DIRECTIONS):
            rr=[c for r in real if r['direction']==d for c in r['controls'] if c['kind']=='mismatched donor'] if is_real else [r for r in rows if r['direction']==d and r['kind']=='mismatched donor']
            flip=sum(round(64*r['flip_rate']) for r in rr)
            inj=sum(r.get('donor_number_injection_count',0) for r in rr)
            for pos,count,color in [(i-.17,flip,RED),(i+.17,inj,BLUE)]:
                val=100*count/den
                ax.bar(pos,val,width=.3,color=color)
                if count == 0: ax.plot(pos,0,'s',color=color,clip_on=False)
                label_rate(ax,pos,val,f'{val:.1f}%\n{count}/{den}',(0,5))
        ax.set(xticks=[0,1],xticklabels=['T → N','N → T'] if not is_real else ['Second → first','First → second'],ylabel='Outputs (%)',ylim=(0,125),title='Real 10-K: H10b' if is_real else 'Synthetic: H7')
        ax.tick_params(axis='x',top=False)
    title(fig,'d','Different-number donors redirect answers to the base alternatives',
          'Red: base’s own alternative. Blue: scored donor-value matches (0 in all conditions). Donor and base values are disjoint.\nAll items retained: synthetic 512 and real 256 per direction. Parsers follow H7 and H10b respectively; invalid counts as failure.')
    save(fig,'d_mismatched_donor')

def attention():
    rows=read('attention_readouts.json')['rows']; frozen=read('frozen_selection.json')['heads']
    fig,axes=plt.subplots(1,3,figsize=(7.2,4.5),sharey=True)
    for ax,q in zip(axes,['Q_N','Q_T','Q_F']):
        for i,(l,h) in enumerate(frozen):
            r=next(r for r in rows if (r['layer'],r['head'],r['query'])==(l,h,q))
            for shift,value,color in [(-.17,100*r['mass']['question'],BROWN),(.17,100*(r['mass']['N']+r['mass']['T']),TAN)]:
                ax.barh(i+shift,value,height=.3,color=color)
                ax.text(value+1.2,i+shift,f'{value:.1f}',va='center',fontsize=7,color=GRAY)
        ax.set(xlim=(0,90),xlabel='Attention mass (%)',title=q,yticks=range(8))
        ax.tick_params(axis='y',left=False,right=False)
    axes[0].set_yticklabels([f'L{l}H{h}' for l,h in frozen]);axes[0].invert_yaxis()
    title(fig,'e','L23H26 attends to the question; L23H4 attends to the numbers',
          'Dark brown: question tokens. Light brown: both number spans. Other tokens account for the remaining mass.\nClean held-out prompts: 8 seeds × 64 = 512 per query, equally weighted. Attention describes allocation, not causal information flow.')
    save(fig,'e_attention_split')

def overlap():
    data=read('overlap_4b_8b.json'); rows=data['per_head']; shared={tuple(h) for h in data['overlap_heads']}
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.9),gridspec_kw={'width_ratios':[1,1.2]})
    for ax in axes:
        for chosen,color in [(False,'#A7B2C9'),(True,RED)]:
            rr=[r for r in rows if ((r['layer'],r['head']) in shared)==chosen]
            ax.scatter([r['mean_oriented_delta_d_4b'] for r in rr],[r['mean_oriented_delta_d_8b'] for r in rr],s=18 if chosen else 9,color=color)
        ax.set(xlabel='4B donor-oriented Δd (logits)',ylabel='8B donor-oriented Δd (logits)')
        ax.axhline(0,color=GRAY,lw=.5);ax.axvline(0,color=GRAY,lw=.5)
    axes[0].set_title('All 128 heads in layers 20–23')
    axes[0].text(.06,.84,f"Spearman ρ = {data['rank_correlations'][0]['rho']:.2f}\nShared top-8 indices: 6/8",transform=axes[0].transAxes,color=GRAY)
    big=next(r for r in rows if (r['layer'],r['head'])==(23,26))
    axes[0].annotate(f"L23H26\n({big['mean_oriented_delta_d_4b']:.2f}, {big['mean_oriented_delta_d_8b']:.2f})",(big['mean_oriented_delta_d_4b'],big['mean_oriented_delta_d_8b']),xytext=(-75,-35),textcoords='offset points',fontsize=7,arrowprops={'arrowstyle':'-','lw':.5})
    axes[1].set(xlim=(-.08,1.65),ylim=(-.06,1.9),title='Detail of smaller effects')
    offsets={(21,11):(.82,1.20),(21,19):(1.37,.94),(21,25):(.49,.88),(22,10):(1.35,1.65),(23,4):(.88,1.75)}
    for r in rows:
        key=(r['layer'],r['head'])
        if key in offsets:
            axes[1].annotate(f'L{key[0]}H{key[1]}',(r['mean_oriented_delta_d_4b'],r['mean_oriented_delta_d_8b']),xytext=offsets[key],textcoords='data',fontsize=7,color=GRAY,arrowprops={'arrowstyle':'-','lw':.5})
    title(fig,'f','Single-head effects correlate across 4B and exploratory 8B scans',
          'Each point averages 4 dev seeds × 2 directions × 64 items. Red marks the six shared top-8 indices.\n8B discovery used dev seeds only. Shared training or distillation is a possible explanation, not an established cause.')
    save(fig,'f_4b_8b_overlap')

def role_position():
    data = read('h11.json')
    if not data['gate']['passed']:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    notes = []
    for ax, direction, target in zip(axes, ['R1','R2'], ['N (now second)','T (now first)']):
        rows = [r for r in data['per_seed'] if r['direction']==direction]
        x = np.arange(len(rows))
        selected = [100*r['unique_integer']['switch_rate'] for r in rows]
        randoms = [100*r['unique_integer']['random_max'] for r in rows]
        ax.plot(x, selected, 'o-', color=RED, label='Frozen 8')
        ax.plot(x, randoms, 's--', color=BLUE, label='Maximum of 100 random sets')
        for i, r in enumerate(rows):
            label_rate(ax, i, selected[i], f"{r['unique_integer']['switch_count']}/64", (0,6))
            label_rate(ax, i, randoms[i], f"{r['unique_integer']['random_max_count']}/64", (0,-12))
        ax.axhline(50, color=GRAY, ls=':', lw=.5)
        ax.set(ylim=(-18,119), xlim=(-.55,7.55), xticks=x, xticklabels=[str(r['seed'])[-2:] for r in rows],
               xlabel='Held-out seed suffix (202612xx)', ylabel='Role-consistent switch (%)',
               title=f'{direction}: original donor → swapped base\nRole target: {target}')
        ax.tick_params(axis='x', top=False)
        same = sum(next(c for c in r['controls'] if c['kind']=='same_order_swapped')['metrics']['unique_integer']['switch_count'] for r in rows)
        noise = sum(next(c for c in r['controls'] if c['kind']=='matched_norm')['metrics']['unique_integer']['switch_count'] for r in rows)
        first = sum(r['first_number']['switch_count'] for r in rows)
        notes.append(f'{direction}: swapped same-order {same}/512; norm-matched noise {noise}/512; secondary first-number {first}/512.')
    axes[0].legend(loc='upper left', fontsize=7)
    title(fig, 'g', f"Cross-order transfer: registered role-versus-position verdict {data['verdict']}",
          'Primary: frozen unique-integer parser; all 64 items retained per seed. Dotted line: 50% role threshold.\n'+'\n'.join(notes))
    save(fig, 'g_role_position', rect=(.02,.19,.99,.91))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', nargs='+', choices=list('abcdefg'))
    args = parser.parse_args()
    start=time.perf_counter();FIGURES.mkdir(exist_ok=True)
    renderers = dict(a=residual,b=heads,c=heldout,d=mismatch,e=attention,f=overlap)
    if (RESULTS/'h11.json').exists(): renderers['g'] = role_position
    chosen = args.panels or list(renderers)
    for panel in chosen: renderers[panel]()
    print(f'Rendered {len(chosen)} panels in {time.perf_counter()-start:.2f} seconds.')

"""Plain HF hooks. All interventions are at explicitly indexed prompt positions."""
from contextlib import contextmanager
import re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


from scoring import parse_integer

class Harness:
    def __init__(self, model_id):
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
        self.tokenizer.padding_side='left'
        self.model=AutoModelForCausalLM.from_pretrained(model_id,local_files_only=True,dtype=torch.float16,attn_implementation='eager').to('cuda').eval()
        self.layers=self.model.model.layers
        self.head_dim=self.model.config.head_dim
        self.num_heads=self.model.config.num_attention_heads
        self.finite_flags=[]
        self.handles=[]
        for layer in self.layers:
            for module in [layer,layer.self_attn,layer.mlp]:
                self.handles.append(module.register_forward_hook(self._finite_hook))

    def _finite_hook(self,module,args,output):
        x=output[0] if isinstance(output,tuple) else output
        self.finite_flags.append(torch.isfinite(x).all())

    def batch(self, rows):
        width=max(len(r['input_ids']) for r in rows)
        pad=self.tokenizer.pad_token_id
        ids=[];masks=[];positions={'prefill':[],'question':[]}
        for r in rows:
            left=width-len(r['input_ids'])
            ids.append([pad]*left+r['input_ids']);masks.append([0]*left+[1]*len(r['input_ids']))
            positions['prefill'].append([left+r['prefill_position']])
            positions['question'].append([left+p for p in r['question_positions']])
        return dict(input_ids=torch.tensor(ids,device='cuda'),attention_mask=torch.tensor(masks,device='cuda')),positions

    @staticmethod
    def gather(x, positions):
        return [x[i,p,:].clone() for i,p in enumerate(positions)]

    @contextmanager
    def hooks(self, positions, captures=(), cache=None, patches=None):
        cache={} if cache is None else cache
        patches={} if patches is None else patches
        handles=[]
        def transform(x, key):
            # key = (layer, component, span); head patches carry an explicit head list.
            if key in patches:
                spec=patches[key]; values=spec['values']; x=x.clone()
                assert len(values)==x.shape[0]
                for i,pos in enumerate(positions[key[2]]):
                    value=values[i].to(device=x.device,dtype=x.dtype)
                    assert value.shape[0]==len(pos), 'Question spans must have equal lengths; no token re-alignment is implicit.'
                    if key[1]=='head':
                        heads=spec['heads']
                        for head in heads:
                            assert 0<=head<self.num_heads
                            sl=slice(head*self.head_dim,(head+1)*self.head_dim)
                            x[i,pos,sl]=value[:,sl]
                    else:
                        assert value.shape[-1]==x.shape[-1]
                        x[i,pos,:]=value
                self.finite_flags.append(torch.isfinite(x).all())
            if key in captures:
                cache[key]=self.gather(x,positions[key[2]])
            return x
        for key in set(captures)|set(patches):
            layer,component,span=key
            assert span in ('prefill','question')
            block=self.layers[layer]
            if component=='head':
                def prehook(module,args,key=key):return (transform(args[0],key),)+args[1:]
                handles.append(block.self_attn.o_proj.register_forward_pre_hook(prehook))
            else:
                module={'residual':block,'attention':block.self_attn,'mlp':block.mlp}[component]
                def hook(module,args,out,key=key):
                    x=out[0] if isinstance(out,tuple) else out
                    y=transform(x,key)
                    return (y,)+out[1:] if isinstance(out,tuple) else y
                handles.append(module.register_forward_hook(hook))
        try:yield cache
        finally:
            for h in handles:h.remove()

    @torch.inference_mode()
    def forward(self,batch,positions,captures=(),patches=None):
        self.finite_flags=[]
        with self.hooks(positions,captures=captures,patches=patches) as cache:
            position_ids=batch['attention_mask'].long().cumsum(-1)-1
            position_ids.masked_fill_(batch['attention_mask']==0,0)
            result=self.model(**batch,position_ids=position_ids,use_cache=False,logits_to_keep=1)
            logits=result.logits[:,-1,:]
            self.finite_flags.append(torch.isfinite(logits).all())
            if not torch.stack(self.finite_flags).all().item():
                raise FloatingPointError('STOP: non-finite fp16 clean or patched run')
        return logits,cache

    @torch.inference_mode()
    def greedy(self,batch,positions,patches=None,first_logits=None,max_new_tokens=8):
        current={k:v.clone() for k,v in batch.items()}
        finished=torch.zeros(batch['input_ids'].shape[0],device='cuda',dtype=torch.bool)
        generated=[]
        eos=self.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        for step in range(max_new_tokens):
            logits=first_logits if step==0 and first_logits is not None else self.forward(current,positions,patches=patches)[0]
            nxt=logits.argmax(-1)
            nxt=torch.where(finished,self.tokenizer.pad_token_id,nxt)
            generated.append(nxt)
            finished |= torch.stack([nxt==e for e in eos]).any(0)
            current['input_ids']=torch.cat([current['input_ids'],nxt[:,None]],-1)
            current['attention_mask']=torch.cat([current['attention_mask'],torch.ones_like(nxt[:,None])],-1)
            if finished.all().item():break
        tokens=torch.stack(generated,1).cpu().tolist()
        texts=self.tokenizer.batch_decode(tokens,skip_special_tokens=True)
        return [dict(text=t,answer=parse_integer(t),generated_ids=ids,first_token=self.tokenizer.decode(ids[:1]),first_is_digit=bool(re.fullmatch('[0-9]',self.tokenizer.decode(ids[:1])))) for t,ids in zip(texts,tokens)]

    def measure(self,rows,patches=None,captures=()):
        batch,pos=self.batch(rows)
        logits,cache=self.forward(batch,pos,captures=captures,patches=patches)
        outputs=self.greedy(batch,pos,patches=patches,first_logits=logits)
        for row,out,logit in zip(rows,outputs,logits):
            out.update(seed=row['seed'],item=row['item'],query_type=row['query_type'],N=row['N'],T=row['T'])
            out['d']=(logit[row['digit_ids']['T']].float()-logit[row['digit_ids']['N']].float()).item()
        return outputs,cache

    def timed_forward(self,batch,pos,patches=None):
        torch.cuda.synchronize();t=time.perf_counter()
        logits,cache=self.forward(batch,pos,patches=patches)
        torch.cuda.synchronize()
        return time.perf_counter()-t,logits

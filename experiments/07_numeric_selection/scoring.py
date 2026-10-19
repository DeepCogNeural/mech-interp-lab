"""Parse one unambiguous integer; units or prose do not change the integer."""
import math,re

def parse_integer(text):
    numbers=re.findall(r'[+-]?\d+(?:[.,]\d+)*',text)
    if len(numbers)!=1 or re.fullmatch(r'[+-]?\d+',numbers[0]) is None:return None
    return int(numbers[0])

def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n)
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,c-h),min(1.,c+h)]

def rate(k,n):return dict(count=k,total=n,share=k/n if n else None,wilson95=wilson(k,n))

def summarize(rows):
    out={}
    for q in ['Q_N','Q_T','Q_F']:
        rs=[r for r in rows if r['query_type']==q];n=len(rs)
        out[q]={key:rate(sum(r['answer']==r[key] for r in rs),n) for key in ['N','T']}
        out[q]['other']=rate(sum(r['answer'] not in (r['N'],r['T']) for r in rs),n)
        if q!='Q_F':out[q]['accuracy']=out[q][q[-1]]
    valid=[r for r in rows if r['answer'] in (r['N'],r['T'])]
    measured=[r for r in valid if r.get('d') is not None]
    out['agreement']=rate(sum((r['d']>0 and r['answer']==r['T']) or (r['d']<0 and r['answer']==r['N']) for r in measured),len(measured))
    out['first_digit']=rate(sum(r.get('first_is_digit',False) for r in rows),len(rows)) if all('first_is_digit' in r for r in rows) else 'not measured'
    out['number_only_compliance']=rate(sum(bool(re.fullmatch(r'\s*\$?[+-]?\d+\s*',r['text'])) for r in rows),len(rows))
    out['other_count']=len(rows)-len(valid)
    return out

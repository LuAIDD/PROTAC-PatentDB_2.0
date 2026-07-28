import numpy as np, re, pandas as pd
def _sci(m): return repr(float(m.group(1))*10**int(m.group(2)))
def normalize(s):
    s=str(s).replace('×','x').replace('X','x').replace('−','-').replace('–','-').replace('≥','>=').replace('≤','<=')
    return re.sub(r'([0-9.]+)\s*x\s*10\s*\^?\s*(-?\d+)',_sci,s)
def _num(t):
    try: return float(t.strip())
    except: return np.nan
def parse_value(raw,kind='conc'):
    if pd.isna(raw): return (np.nan,'')
    s=normalize(raw).strip()
    if s=='' : return (np.nan,'')
    parts=[p for p in s.split('/') if p.strip()!='']
    vals=[]; cens=set()
    for p in parts:
        p=p.strip(); c=''
        mr=re.match(r'^([0-9.]+)\s*-\s*([0-9.]+)$',p)
        if p.startswith('>='): v=_num(p[2:]); c='>='
        elif p.startswith('<='): v=_num(p[2:]); c='<='
        elif p.startswith('>'): v=_num(p[1:]); c='>'
        elif p.startswith('<'): v=_num(p[1:]); c='<'
        elif mr:
            a=_num(mr.group(1)); b=_num(mr.group(2)); v=np.sqrt(a*b) if (a>0 and b>0) else np.nanmean([a,b]); c='range'
        else: v=_num(p)
        if not np.isnan(v): vals.append(v)
        if c: cens.add(c)
    if not vals: return (np.nan,'')
    if kind=='conc' and all(v>0 for v in vals): agg=float(np.exp(np.mean(np.log(vals))))
    else: agg=float(np.mean(vals))
    return (agg,('dual;' if len(parts)>1 else '')+';'.join(sorted(cens)))

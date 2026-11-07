import json,glob,re,sys,collections
keys={}
for f in glob.glob('subjects/nlp/bank/gen_w*.json'):
    for q in json.load(open(f,encoding='utf-8')): keys[q['id']]=q
def parse(fns):
    r={}
    for fn in fns:
        try: txt=open(fn,encoding='utf-8').read()
        except FileNotFoundError: continue
        for ln in txt.split('\n'):
            m=re.match(r'\s*(n\d+-L\d-(?:F)?\d+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(.*)',ln)
            if m: r[m.group(1)]=dict(ans=m.group(2),conf=m.group(3),fit=m.group(4),why=m.group(5))
    return r
S=parse(['mock/solve_bank_sonnet_a.md','mock/solve_bank_sonnet_b.md'])
C=parse(['mock/solve_bank_codex.md'])
print('sonnet parsed',len(S),'codex parsed',len(C))
for name,R in [('sonnet',S),('codex',C)]:
    if not R: continue
    bad=[]; fit=collections.Counter()
    for i,q in keys.items():
        if i not in R: continue
        fit[(q['level'],R[i]['fit'])]+=1
        if q['kind']=='mcq' and R[i]['ans'][:1].upper()!=q['ans']: bad.append((i,q['ans'],R[i]['ans'][:3]))
    print(name,'mcq mismatches',len(bad),bad)
    print(name,'fit',dict(sorted(fit.items())))

import json,glob,sys
patches=json.load(open(sys.argv[1],encoding='utf-8'))
files={f:json.load(open(f,encoding='utf-8')) for f in glob.glob('subjects/nlp/bank/gen*_w*.json')}
done=set()
for f,items in files.items():
    for q in items:
        p=patches.get(q['id'])
        if not p: continue
        for k,v in p.items():
            if k=='opts': q['opts'].update(v)
            else: q[k]=v
        done.add(q['id'])
    json.dump(items,open(f,'w',encoding='utf-8'),ensure_ascii=False,indent=1)
print('patched',len(done),'missing',set(patches)-done)

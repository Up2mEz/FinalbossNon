import re,sys
key,sol=sys.argv[1],sys.argv[2]
k={}
for l in open(key,encoding='utf-8'):
    m=re.match(r'\s*[-*]*\s*\*{0,2}([QF]\d+)\*{0,2}\s*(?:→|->|:|\|)\s*\*{0,2}\s*([A-D])\b',l)
    if m: k[m.group(1)]=m.group(2)
s={}
for l in open(sol,encoding='utf-8'):
    m=re.match(r'\s*([Q]\d+)\s*\|\s*([A-D])\b',l)
    if m: s[m.group(1)]=m.group(2)
bad=[q for q in k if q in s and k[q]!=s[q]]
print('keyed',len(k),'solved',len(s),'mismatch',len(bad),bad)

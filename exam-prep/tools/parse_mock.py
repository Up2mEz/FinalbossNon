import re, json, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, 'mock')
def clean(t):
    t = t.replace(' — ', ' - ').replace('—', '-').replace('–', '-')
    return t.strip()
def parse_questions(txt):
    items = []
    parts = re.split(r'(?m)^(?=\*\*[QF]\d+\*\*)', txt)
    for p in parts:
        m = re.match(r'\*\*([QF])(\d+)\*\*\s*((?:\[[^\]]*\]\s*)*)\s*\n?(.*)', p, re.S)
        if not m: continue
        kind, num, tags, body = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        tg = re.findall(r'\[([^\]]*)\]', tags)
        week = int(re.search(r'W(\d+)', tg[0]).group(1)) if tg else 0
        topic = tg[1] if len(tg) > 1 else ''
        diff = ''
        for t in tg[2:]:
            mm = re.search(r'difficulty:\s*(\w+)', t)
            if mm: diff = mm.group(1)
        body = re.split(r'(?m)^#{1,3} ', body)[0]
        if kind == 'Q':
            lines = body.strip().split('\n')
            stem, opts, cur = [], {}, None
            for ln in lines:
                mo = re.match(r'\s*([A-D])[\)\.]\s*(.*)', ln)
                if mo and (cur or len(opts) < 4):
                    cur = mo.group(1); opts[cur] = mo.group(2).strip()
                elif cur and ln.strip():
                    opts[cur] += ' ' + ln.strip()
                elif not cur:
                    stem.append(ln.rstrip())
            if len(opts) != 4: print('WARN options', kind, num, len(opts), file=sys.stderr)
            items.append(dict(kind='mcq', num=num, week=week, topic=clean(topic), diff=diff,
                              stem=clean('\n'.join(stem)), opts={k: clean(v) for k, v in opts.items()}))
        else:
            items.append(dict(kind='fill', num=num, week=week, topic=clean(topic), diff=diff,
                              stem=clean(body)))
    return items
def parse_keys(txt, name):
    keys = {}
    for ln in txt.split('\n'):
        s = ln.strip()
        m = re.match(r'[-*\s]*\|?\s*\**([QF])(\d+)\**\s*(?:→|->|:|\||：)?\s*(.*)', s)
        if not m:
            m2 = re.match(r'\|\s*([QF])?(\d+)\s*\|\s*(.*)', s)
            if not m2: continue
            kind = m2.group(1) or 'Q'; num = int(m2.group(2)); rest = m2.group(3)
            if not (s.startswith('| Q') or s.startswith('| F') or re.match(r'\|\s*\d', s)): continue
        else:
            kind, num, rest = m.group(1), int(m.group(2)), m.group(3)
        rest = rest.strip()
        if kind == 'Q':
            ma = re.match(r'[\s:|*：-]*\**\s*([A-D])\b\**\s*[|:)\-—.]*\s*(.*)', rest)
            if not ma: continue
            ans, expl = ma.group(1), ma.group(2)
            keys['Q%d' % num] = dict(ans=ans, expl=expl)
        else:
            keys['F%d' % num] = dict(raw=rest)
    return keys
def finalize(items, keys):
    out = []
    for it in items:
        k = ('Q' if it['kind'] == 'mcq' else 'F') + str(it['num'])
        kk = keys.get(k)
        if not kk: print('NO KEY', k, file=sys.stderr); continue
        if it['kind'] == 'mcq':
            e = clean(kk['expl']); src = ''
            ms = re.search(r'(?:แหล่ง:|แหล่งที่มา:|Source:)\s*(.*)$', e)
            if ms: src = ms.group(1).strip(); e = e[:ms.start()].strip()
            if not src and ' | ' in e:
                e, src = e.rsplit(' | ', 1); src = src.strip(' |')
            it['ans'] = kk['ans']; it['expl'] = e.strip(' |'); it['src'] = src.replace('`','').strip()
        else:
            raw = clean(kk['raw']); src = ''
            ms = re.search(r'(?:แหล่ง:|Source:)\s*(.*)$', raw)
            if ms: src = ms.group(1).strip(); raw = raw[:ms.start()].strip()
            raw = re.sub(r'^[*:\s|]+', '', raw)
            if '**' in raw: a, e = raw.split('**', 1)
            elif ' | ' in raw: a, e = raw.split(' | ', 1)
            elif ' - ' in raw: a, e = raw.split(' - ', 1)
            else: a, e = raw, ''
            it['key'] = a.strip(' |*'); it['note'] = e.strip(' -:|*'); it['src'] = src.replace('`','').strip()
        out.append(it)
    return out
data = {}
for n in ['codex', 'sonnet', 'gemini']:
    q = parse_questions(open(f'{M}/q_{n}.md', encoding='utf-8').read())
    k = parse_keys(open(f'{M}/key_{n}.md', encoding='utf-8').read(), n)
    d = finalize(q, k)
    for it in d: it['set'] = n; it['id'] = f"{n[0]}{it['kind'][0]}{it['num']}"
    data[n] = d
    print(n, 'mcq', sum(i['kind'] == 'mcq' for i in d), 'fill', sum(i['kind'] == 'fill' for i in d), file=sys.stderr)
json.dump(data, open(os.path.join(ROOT, 'tools', 'mock_data.json'), 'w', encoding='utf-8'), ensure_ascii=False)

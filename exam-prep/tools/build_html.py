"""Build mock_exam.html from every subject under subjects/<id>/ .
A subject = subject.json + any number of question files (questions*.json and bank/*.json, each a JSON array).
Usage: python tools/build_html.py            (all subjects)
       python tools/build_html.py nlp        (only these subjects)"""
import os, sys, json, glob
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
want = sys.argv[1:]
subjects = []
for d in sorted(glob.glob(f'{R}/subjects/*/subject.json')):
    sid = os.path.basename(os.path.dirname(d))
    if want and sid not in want: continue
    cfg = json.load(open(d, encoding='utf-8'))
    qs = []
    for f in sorted(glob.glob(f'{R}/subjects/{sid}/questions*.json') + glob.glob(f'{R}/subjects/{sid}/bank/*.json')):
        qs += json.load(open(f, encoding='utf-8'))
    seen, uniq = set(), []
    for q in qs:
        if q['id'] in seen: print('duplicate id skipped', q['id']); continue
        seen.add(q['id']); q.setdefault('set', 'bank'); uniq.append(q)
    unit_ids = {w['id'] for w in cfg['weeks']}; set_ids = {s['id'] for s in cfg['sets']}
    bad = [q['id'] for q in uniq if q['week'] not in unit_ids or q['set'] not in set_ids or q['level'] not in (1, 2, 3, 4)]
    if bad: print('WARNING', sid, 'questions with unknown unit/set/level:', bad[:10])
    cfg['questions'] = uniq
    subjects.append(cfg)
    from collections import Counter
    print(sid, len(uniq), 'questions; mcq by level', dict(sorted(Counter(q['level'] for q in uniq if q['kind'] == 'mcq').items())))
import re
NOSHUF = re.compile(r'ข้อ\s*[A-D](?![A-Za-z])|(?:option|choice|ตัวเลือก)\s*[A-D](?![A-Za-z])|(?<![A-Za-z0-9_(])[A-D](?:\s*[,/และกับ]\s*[A-D])+(?![A-Za-z0-9])|(?<![A-Za-z0-9_])[A-D]\s*(?:ผิด|ถูก|เป็นคำตอบ|คือคำตอบ|ตัดทิ้ง)|(?:ทั้ง|ทุก|ไม่มี)\s*(?:ข้อ|ตัวเลือก)|ข้อข้างต้น|\([A-D]\s*=\s*[A-Z]|[A-D]\s*=\s*(?:Sequential|Routing|Supervisor|Debate)|\([A-D]\)|\b[A-D]\)')
for sj in subjects:
    n = 0
    for q in sj['questions']:
        if q['kind'] == 'mcq':
            blob = ' '.join([q['stem']] + list(q['opts'].values()) + [q.get('expl', '')])
            if NOSHUF.search(blob): q['ns'] = 1; n += 1
    print(sj['id'], 'options not shuffled (refer to letters):', n)
css = open(f'{R}/tools/app.css', encoding='utf-8').read()
html = open(f'{R}/tools/app.html', encoding='utf-8').read()
data = json.dumps({'subjects': subjects}, ensure_ascii=False).replace('</', '<' + chr(92) + '/')
html = html.replace('/*__CSS__*/', css).replace('/*__DATA__*/', data)

V = f'{R}/tools/vendor'
if os.path.isdir(V) and '--online' not in sys.argv:
    import base64, re
    def b64(fn, mime): return 'data:' + mime + ';base64,' + base64.b64encode(open(fn, 'rb').read()).decode()
    fonts = open(V + '/fonts.css', encoding='utf-8').read()
    fonts = re.sub(r'FONTFILE:([\w\-\.]+)', lambda m: b64(V + '/fonts/' + m.group(1), 'font/woff2'), fonts)
    kcss = open(V + '/katex.min.css', encoding='utf-8').read()
    kcss = re.sub(r'url\(fonts/([\w\-]+)\.woff2\)\s*format\("woff2"\),\s*url\(fonts/[\w\-]+\.woff\)\s*format\("woff"\),\s*url\(fonts/[\w\-]+\.ttf\)\s*format\("truetype"\)',
                  lambda m: 'url(' + b64(V + '/katex_fonts/' + m.group(1) + '.woff2', 'font/woff2') + ') format("woff2")', kcss)
    kjs = open(V + '/katex.min.js', encoding='utf-8').read(); ajs = open(V + '/auto-render.min.js', encoding='utf-8').read()
    html = re.sub(r'<link rel="preconnect"[^>]*>\s*', '', html)
    html = re.sub(r'<link rel="stylesheet" href="https://fonts.googleapis.com[^>]*>', lambda m: '<style>' + fonts + '</style>', html)
    html = re.sub(r'<link rel="stylesheet" href="https://cdnjs[^>]*katex[^>]*>', lambda m: '<style>' + kcss + '</style>', html)
    html = re.sub(r'<script src="https://cdnjs[^>]*katex.min.js"[^>]*></script>', lambda m: '<script>' + kjs + '</script>', html)
    html = re.sub(r'<script src="https://cdnjs[^>]*auto-render.min.js"[^>]*></script>', lambda m: '<script>' + ajs + '</script>', html)
    print('offline bundle embedded (fonts + KaTeX)')
open(f'{R}/mock_exam.html', 'w', encoding='utf-8').write(html)
print('wrote mock_exam.html', len(html) // 1024, 'KB')

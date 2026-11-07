"""Validate a question bank JSON file for the mock-exam system.
Usage: python tools/validate_bank.py path/to/file.json [--expect-per-level 6 --expect-fill-per-level 1]
Exit code 0 = valid. Prints every problem."""
import json, sys, re, argparse, collections
ap = argparse.ArgumentParser()
ap.add_argument('file'); ap.add_argument('--expect-per-level', type=int, default=0); ap.add_argument('--expect-fill-per-level', type=int, default=0)
a = ap.parse_args()
errs = []
try:
    data = json.load(open(a.file, encoding='utf-8'))
except Exception as e:
    print('INVALID JSON:', e); sys.exit(1)
if not isinstance(data, list): print('top level must be a JSON array'); sys.exit(1)
ids = set(); cnt = collections.Counter(); fills = collections.Counter(); letters = collections.Counter()
for i, q in enumerate(data):
    p = f'item {i} ({q.get("id")})'
    for k in ('id', 'kind', 'week', 'level', 'topic', 'stem', 'src'):
        if k not in q or q[k] in ('', None): errs.append(f'{p}: missing {k}')
    if q.get('id') in ids: errs.append(f'{p}: duplicate id')
    ids.add(q.get('id'))
    if q.get('level') not in (1, 2, 3, 4): errs.append(f'{p}: level must be 1..4')
    kind = q.get('kind')
    if kind == 'mcq':
        o = q.get('opts') or {}
        if sorted(o.keys()) != ['A', 'B', 'C', 'D']: errs.append(f'{p}: opts must be A,B,C,D')
        if q.get('ans') not in ('A', 'B', 'C', 'D'): errs.append(f'{p}: ans must be A-D')
        if not q.get('expl'): errs.append(f'{p}: missing expl')
        vals = [str(v).strip() for v in o.values()]
        if len(set(vals)) != len(vals): errs.append(f'{p}: duplicate option text')
        cnt[q.get('level')] += 1; letters[q.get('ans')] += 1
    elif kind == 'fill':
        if not q.get('key'): errs.append(f'{p}: fill needs key')
        fills[q.get('level')] += 1
    else: errs.append(f'{p}: kind must be mcq or fill')
    blob = json.dumps(q, ensure_ascii=False)
    if '\u2014' in blob or '\u2013' in blob: errs.append(f'{p}: contains em/en dash, use a hyphen')
if a.expect_per_level:
    for L in (1, 2, 3, 4):
        if cnt[L] != a.expect_per_level: errs.append(f'level {L}: expected {a.expect_per_level} mcq, got {cnt[L]}')
if a.expect_fill_per_level:
    for L in (1, 2, 3, 4):
        if fills[L] != a.expect_fill_per_level: errs.append(f'level {L}: expected {a.expect_fill_per_level} fill, got {fills[L]}')
tot = sum(letters.values())
if tot >= 8 and max(letters.values()) > 0.45 * tot: errs.append(f'answer letters unbalanced: {dict(letters)}')
print('items', len(data), 'mcq per level', dict(cnt), 'fill per level', dict(fills), 'letters', dict(letters))
if errs:
    print('PROBLEMS:'); [print(' -', e) for e in errs]; sys.exit(1)
print('OK')

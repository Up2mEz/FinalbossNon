"""Create a new subject pack for the mock-exam app.
Usage: python tools/new_subject.py <id> "<Title>" --units 6 --unit-label "บท" --mcq 10 --fill 1 [--exam-date 2026-12-20T09:00:00+07:00]
Then: put the course material in subjects/<id>/source/, generate questions (see subjects/<id>/GENERATE_PROMPT.md), run tools/validate_bank.py, then tools/build_html.py"""
import argparse, json, os, shutil
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('id'); ap.add_argument('title')
ap.add_argument('--units', type=int, default=6); ap.add_argument('--unit-label', default='บท'); ap.add_argument('--unit-short', default='U')
ap.add_argument('--mcq', type=int, default=10); ap.add_argument('--fill', type=int, default=1)
ap.add_argument('--exam-date', default=''); ap.add_argument('--minutes', type=int, default=180)
a = ap.parse_args()
d = f'{R}/subjects/{a.id}'
if os.path.exists(d): raise SystemExit(f'{d} already exists')
os.makedirs(f'{d}/bank'); os.makedirs(f'{d}/source')
hues = [25, 75, 148, 185, 225, 265, 305, 345]
weeks = [{"id": i + 1, "name": f"{a.unit_label} {i + 1} (ใส่ชื่อหัวข้อ)", "hue": hues[i % len(hues)], "mcq": a.mcq, "fill": a.fill} for i in range(a.units)]
cfg = {"id": a.id, "title": a.title, "short": a.id.upper(), "mark": a.title[:1].upper(),
 "heroTitle": f"ซ้อมสอบ {a.title} ให้เหมือนวันจริง",
 "lead": "ข้อสอบจำลองจากเอกสารในคลาส เลือกระดับความยากและโหมดได้",
 "exam": {"date": a.exam_date, "label": "สอบจริง", "minutes": a.minutes, "secPerQuestion": round(a.minutes * 60 / max(1, a.units * (a.mcq + a.fill)))},
 "units": {"label": a.unit_label, "short": a.unit_short, "range": f"{a.unit_label} 1 ถึง {a.units}"},
 "notesPath": "", "mixLevels": {"1": 2, "2": 4, "3": 3, "4": 1}, "weeks": weeks,
 "sets": [{"id": "bank", "name": "คลังตามระดับ", "desc": "ออกใหม่จากเอกสารในคลาส ครบทุกระดับ"}],
 "levels": [{"id": 1, "name": "ง่าย", "desc": "จำข้อเท็จจริง 1 ขั้น"}, {"id": 2, "name": "ปานกลาง", "desc": "แยกแนวคิดคล้ายกัน หรือใช้สูตร 1 ครั้ง"},
            {"id": 3, "name": "ยาก", "desc": "หลายขั้น เลือกวิธีให้เหมาะกับสถานการณ์"}, {"id": 4, "name": "ยากมาก", "desc": "รวมหลายแนวคิด กรณีขอบ หรือคำนวณที่มีกับดัก"}]}
json.dump(cfg, open(f'{d}/subject.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
shutil.copy(f'{R}/subjects/nlp/LEVELS.md', f'{d}/LEVELS.md')
open(f'{d}/GENERATE_PROMPT.md', 'w', encoding='utf-8').write(f"""# Prompt template: generate a leveled bank for subject `{a.id}`

Give this to one agent per unit (replace UNIT, UNIT_NAME, SOURCE_FILES). Each agent writes `subjects/{a.id}/bank/gen_uNN.json`.

---
Write a leveled question bank for the exam of the course "{a.title}", for ONE unit: UNIT (UNIT_NAME).
Workspace: the exam-prep folder. Read `subjects/{a.id}/LEVELS.md` (4-level rubric, follow it strictly) and the unit's source material: SOURCE_FILES. Questions must be answerable from the in-class material only. Skip anything the teacher said is not on the exam.
Write exactly 24 MCQ (6 per level 1..4) and 4 fill-in-the-blank (1 per level) as a JSON array. Item schema:
 mcq: {{"id":"uNN-L<level>-<nn>","kind":"mcq","set":"bank","week":UNIT,"level":1-4,"topic":"...","stem":"...","opts":{{"A":"..","B":"..","C":"..","D":".."}},"ans":"A-D","expl":"why right, why the tempting ones are wrong; show arithmetic","src":"where a student can check it"}}
 fill: {{"id":"uNN-L<level>-F<n>","kind":"fill","set":"bank","week":UNIT,"level":1-4,"topic":"...","stem":"... ___ ...","key":"answer(s) separated by ;","note":"accepted variants","src":"..."}}
 (the data field is called `week` for every course; it holds the unit id from subject.json)
Rules: exactly one correct option; plausible distractors from real misconceptions; spread correct letters; recompute every number with python; no em-dash or en-dash; level 4 must be genuinely harder reasoning, level 1 one-step recall.
Validate: python tools/validate_bank.py subjects/{a.id}/bank/gen_uNN.json --expect-per-level 6 --expect-fill-per-level 1
Then check answers with a second, independent solver (blind) before building: python tools/build_html.py
---
""")
print('created', d)

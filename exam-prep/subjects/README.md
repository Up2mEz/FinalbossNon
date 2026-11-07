# Subjects

Each folder here is one course. The app (`mock_exam.html`) embeds every subject it finds, and a drop-down in the header switches between them.

```
subjects/<id>/
  subject.json          course config: title, units (weeks/chapters) with colour hue and MCQ/fill counts per unit,
                        question sets, difficulty levels, default level mix, exam date and minutes
  questions*.json       question files (JSON arrays), any number
  bank/*.json           more question files, e.g. generated per unit
  LEVELS.md             the 4-level difficulty rubric used when generating questions
  GENERATE_PROMPT.md    prompt template for agents that write the questions
  source/               put the course slides or notes here
```

## Add a new subject

```bash
python tools/new_subject.py algo "Algorithms" --units 6 --unit-label "บท" --unit-short "Ch" --mcq 10 --fill 1 --exam-date 2026-12-20T09:00:00+07:00
```

1. Edit `subjects/algo/subject.json`: unit names, hues if you want other colours, exam date and minutes.
2. Put the slides or notes in `subjects/algo/source/` (convert PDFs with `pdftotext -layout`).
3. Generate questions with `subjects/algo/GENERATE_PROMPT.md` (one agent per unit). Validate each file:
   `python tools/validate_bank.py subjects/algo/bank/gen_u01.json --expect-per-level 6 --expect-fill-per-level 1`
4. Have a second, independent solver answer the questions blind and compare with your keys. Fix or drop disagreements.
5. `python tools/build_html.py` and open `mock_exam.html` (run `python tools/vendor_fetch.py` once first if you want the offline bundle).

## Question item format

MCQ: `id, kind:"mcq", set, week (the unit id), level 1-4, topic, stem, opts{A,B,C,D}, ans, expl, src`
Fill: `id, kind:"fill", set, week, level, topic, stem, key, note, src`
Optional on any item: `hinted:true` for things the teacher said will be on the exam. `set` must be one of the ids in `subject.json` `sets`.

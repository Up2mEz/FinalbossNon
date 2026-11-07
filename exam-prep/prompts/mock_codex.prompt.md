You are a university NLP exam writer. Write ONE complete mock final exam for a KMITL NLP course (weeks 8-14), in Thai (keep English technical terms).

SOURCES (read them from disk; paths relative to D:\KMITL\NLP\NLP-Year3\team-project\exam-prep\):
- notes/week_08_notes.md ... notes/week_14_notes.md  (study notes per week; read ALL seven fully)
- text/slides_wNN.txt (lecture slides = source of truth), text/demo_wNN.txt (in-class demo)
- source/nlp_exam_hints.md (teacher's exam hints, signals only)
Every question must be answerable from these sources. Do not use facts that contradict the slides. Where notes flag a number as unreliable / computed by hand, avoid making a question hinge on it.

EXAM BLUEPRINT (fixed): per week, 10 multiple-choice (4 options A-D, exactly one correct) plus fill-in-the-blank: wk8 1, wk9 1, wk10 1, wk11 1, wk12 2, wk13 2, wk14 2. Total 70 MCQ + 10 fill-in. Number MCQ 1-70 (week 8 first ... week 14 last), fill-in F1-F10. Within a week, give more questions to topics that have more slide content / teacher emphasis (see each notes file section 1 and the hints). Include the teacher's hinted topics (BART noising, T5 sentinel, SBERT 2304, cross vs bi-encoder cost, triplet loss, GB vs GiB, quantization absmax, BM25 saturation k1+1, multi-agent topologies, MoE speeds up but does not save memory, etc.). Some MCQ should be easy arithmetic ("add and multiply") and some demo-only facts (1-2 total).

STYLE FOCUS FOR YOU: calculation-heavy and numerically precise: about 40% of MCQ and most fill-ins need a computation (VRAM/KV-cache, token counts, loss values, BM25/RRF, quantization, dimension counts, cost ratios), with decimal/power-of-10 traps. Verify every number with python before keying.

OUTPUT FORMAT (strict markdown, write the whole exam in one file):
# Mock exam — Codex set
## Part A — MCQ
For each: `**Q<n>** [W<week>] [topic] [difficulty: easy|medium|hard]` then question, options `A) .. B) .. C) .. D) ..`.
## Part B — Fill-in
`**F<n>** [W<week>] [topic]` question with ___ blanks.
## Answer key
Table or list: Q<n> -> letter, one-line explanation, source (notes file + slide page or demo). For calculations show the arithmetic. Fill-ins: the exact expected answer(s) plus accepted variants.
Make distractors plausible (common misconceptions from the notes' 'MCQ traps' sections). Do not make the correct option systematically the longest or in a fixed position; spread A-D roughly evenly. For every numeric question, recompute the answer carefully (use a calculator/python if available) before writing the key. Do not ask anything from Lab notebooks. Output only the exam file content.

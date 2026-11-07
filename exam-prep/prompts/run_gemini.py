import re, subprocess, sys, tempfile, os
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILL = {8:1,9:1,10:1,11:1,12:2,13:2,14:2}
TITLE = {8:"Encoder-Decoder, BART, T5",9:"Sentence Embeddings, Semantic Search",10:"Hardware-Aware NLP, PEFT",11:"Prompting, Post-Training, Reasoning",12:"Hybrid RAG, Multimodal, GraphRAG",13:"Agents, Tool Calling, Constrained Decoding",14:"Data Engines, Evaluation"}
def sections(txt):
    parts = re.split(r'(?m)^(?=## )', txt)
    d = {}
    for p in parts:
        m = re.match(r'## (\d+)\.', p)
        d[int(m.group(1)) if m else 0] = p
    return d
def build(w):
    s = sections(open(f'{ROOT}/notes/week_{w:02d}_notes.md', encoding='utf-8').read())
    budget = int(os.environ.get("BUDGET","22500"))
    order = [1,3,4,6,7,2]
    body, used = [], 0
    for k in order:
        t = s.get(k, '')
        t = re.sub(r'\n{2,}', '\n', t)
        if used + len(t) > budget:
            t = t[:max(0, budget-used)]
        body.append(t); used += len(t)
    q0 = (w-8)*10 + 1
    nf = FILL[w]
    return f"""You are a university NLP exam writer. Write the Week {w} part ({TITLE[w]}) of a mock final exam, in Thai (keep English technical terms). You cannot read files; the study notes for this week (source of truth, derived from the slides) are below.
Write exactly 10 multiple-choice questions numbered Q{q0}-Q{q0+9} (4 options A-D, exactly one correct) and {nf} fill-in-the-blank question(s) numbered F{w-7 if w<=11 else 4+2*(w-12)+1}{'' if nf==1 else '-F'+str(4+2*(w-12)+2)}. More questions on topics with more emphasis (see section 1). STYLE: conceptual and scenario-based ('which is correct / what happens if / which design fits'), compare-and-contrast, misconception traps; at most 2 of the 10 MCQ numeric. Use only facts in the notes; avoid items the notes flag as unreliable. Distractors must be plausible; spread correct letters across A-D.
FORMAT: `**Q<n>** [W{w}] [topic] [difficulty: easy|medium|hard]` then the question and options on separate lines; fill-ins `**F<n>** [W{w}] [topic]` with ___ blanks. Then `### Answer key` listing each answer with a one-line explanation (show arithmetic for numeric ones) and a pointer to the notes section. Output only the exam content, no preamble.
=== STUDY NOTES WEEK {w} ===
""" + "\n".join(body)
def run(w):
    p = build(w)
    out = f'{ROOT}/mock/gemini_w{w:02d}.md'
    d = tempfile.mkdtemp()
    r = subprocess.run(["agy","-p",p,"--model","gemini-3.8-flash-high","--sandbox","--disable-slash-commands"],cwd=d,capture_output=True,text=True,encoding='utf-8',timeout=600)
    open(out,'w',encoding='utf-8').write(r.stdout)
    return w, len(p), len(r.stdout), r.stderr[:200]
if __name__ == '__main__':
    with ThreadPoolExecutor(4) as ex:
        def safe(w):
            try: return run(w)
            except Exception as e: return (w,'ERR',repr(e)[:100])
        ws=[int(a) for a in sys.argv[1:]] or range(8,15)
        for res in ex.map(safe, ws): print(res, flush=True)

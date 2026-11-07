# Week 13 — Agents, Tool Calling และ Constrained Decoding (Study Notes)

แหล่งข้อมูล: `text/slides_w13.txt` (ความจริงหลัก), `text/demo_w13.txt`, `source/nlp_exam_hints.md`
การอ้างอิง: "S12" = เลขสไลด์ที่พิมพ์มุมขวาล่าง (ไม่ใช่เลขหน้า form-feed เพราะ txt มี 91 segment จาก 54 สไลด์ที่มี animation build)

> **ข้อจำกัดของแหล่งข้อมูล (อ่านก่อน):** ตาราง/ภาพบางอัน **ไม่อยู่ใน txt** (เป็นรูป): ตาราง numeric ของ context growth (S12), ตาราง Short-term vs Long-term (S30), ตาราง Topology Comparison (S41), ตัว pseudocode ของ ReAct loop (S11) และ Supervisor loop (S39), ตาราง Format Compliance (S26), ภาพ Levels of Autonomy (S4). ที่ต้องใช้ผมเติมจากข้อความรอบ ๆ + hints และติดป้ายไว้ ควรเปิดสไลด์จริงตรวจสอบ 5 จุดนี้

---

## 1. Exam weight และลำดับหัวข้อ

**สัดส่วน:** 10 MCQ + 2 fill-in-the-blank (จาก 70 MCQ + 10 fill-in ทั้งฉบับ แต่ละสัปดาห์ได้ส่วนเท่ากัน)

**Hints (สัญญาณอ่อน):** Week 13 ไม่มีคำว่า "ออกสอบ" ชัด ๆ แต่ hints ระบุ: ตัวเลข Table 1 ของ ReAct (HotpotQA 27.4 vs CoT 29.4; FEVER 60.9 vs 56.3), `max_steps` เป็น safeguard กัน loop, tool definition (JSON schema), constrained decoding, memory, multi-agent topologies (4 มิติ: latency / complexity / robustness / best use — "ดูใน 4 มิตินี้ ข้อสอบก็ออก")

**ลำดับหัวข้อที่น่าจะออก (ตามปริมาณสไลด์ x น้ำหนักที่ครูเน้น):**

| อันดับ | หัวข้อ | สไลด์ | ประมาณข้อ (MCQ) | เหตุผล |
|---|---|---|---|---|
| 1 | ReAct loop + Tool definition/protocol + ผลเปรียบเทียบ (Table 1/2) | S3-S18 (16 สไลด์) | 3-4 | เนื้อหามากสุด, hints เจาะจงตัวเลข + max_steps |
| 2 | Multi-agent topologies (Sequential/Routing/Supervisor/Debate) + protocol | S32-S43 (12 สไลด์) | 2-3 | hints: เปรียบเทียบ 4 มิติ; สูตร y_i = f_i(y_{i-1}), Σ T_i |
| 3 | Grammar/Logit-level constrained decoding | S19-S26 (8 สไลด์) + demo หนัก | 2-3 | มีสูตร, ตัวเลข 69/50,257, demo 0% vs 100% |
| 4 | Memory architectures (taxonomy 4 แบบ, Buffer/Summary/Retrieval) | S27-S31 (5 สไลด์) + demo | 1-2 | big-O, ตัวเลข demo |
| 5 | Synthesis: Reliability stack, Guardrails, Evaluation, Code agent | S44-S52 | 1 | list-style, ออกง่ายแบบ "ข้อใดไม่ใช่" |

**Fill-in (2 ข้อ) น่าจะมาจาก:** ตัวเลข ReAct Table 1, รูปแบบ −∞ logit masking, ชื่อ memory 4 แบบ, สูตร topology, max_steps, ตัวเลข 69 tokens (ดูหัวข้อ 6)

---

## 2. Core Concepts

### 2.1 Agent คืออะไร (S3, S5, S6)

- **Definition (S3):** autonomous agent = ระบบที่ **LLM ควบคุม execution flow** — ตัดสินใจว่าจะทำอะไรต่อ, เรียก tool, และวนซ้ำจนงานเสร็จ [MEMORIZE]
- **Formal view (S3):** agent = **feedback control loop**. ถ้า state เก็บ interaction history ครบ transition เป็น **Markov** บน augmented state; ถ้า observation ถูกบีบ/ย่อ จะเป็น partially observable
- **Analogy (S3):** LLM = Calculator (พิมพ์แล้วตอบ) / Agent = Researcher (ถามคำถาม, รันการทดลอง, ปรับสมมติฐาน)
- **4 Core Components (S5)** [MEMORIZE]:
  1. **LLM Brain (Controller)** — reasoning engine
  2. **Tool Set (Actuators)** — search, calculator, code execution, APIs
  3. **Memory (State)** — short-term (context window) + long-term (persistent storage)
  4. **Planning Module** — task decomposition + strategy selection
  - Data flow: User Query -> LLM Brain -> (Plan -> Tool Call -> Observation) x N -> Final Answer
- **ทำไมต้อง Agent (S6):** สิ่งที่มีแล้ว = Reflection (W11: critic รู้เท่าที่ writer รู้) และ RAG (W12: ข้อมูลสด แต่ **pipeline คงที่ retrieve ครั้งเดียวแล้วตอบ**)
  สิ่งที่ยังขาด 4 ข้อ: (1) **Grounded verification** (ตรวจกับโลกจริง ไม่ใช่ตรวจกับตัวเอง), (2) **Live per-request state** (ราคาหุ้นตอนนี้, สถานะ order — index ล่วงหน้าไม่ได้), (3) **Model-chosen control flow** (เลือกเองว่าจะ retrieve/act หรือไม่, กี่ครั้ง), (4) **Side effects** (รันโค้ด, เรียก API, เขียน DB)
- **Key insight (S6):** LLM **ไม่ได้ฉลาดขึ้น** — แต่ "capable ขึ้นผ่านการใช้ tools" [MEMORIZE]
- Levels of Autonomy (S4): เป็นรูปจากบทความ Feng, McDonald, Zhang (2025) arXiv:2506.12469 — ไม่มีข้อความ ไม่น่าออกละเอียด
- In-class activity (S7): GDP ต่อหัวของไทย -> หา GDP (tool: search), หา population (tool: search), หาร GDP/Pop (calculator), จัดรูปแบบคำตอบ (LLM ทำเองได้) — คือหลักคิด "ขั้นไหน LLM ทำเองได้ ขั้นไหนต้องใช้ tool"

### 2.2 ReAct (S8-S15)

- Paper: Yao et al. (2023) ICLR 2023 "ReAct: Synergizing Reasoning and Acting in Language Models"
- **Core innovation (S8):** สลับ (interleave) **reasoning traces (Thought)** กับ **actions (tool calls)** ใน generation เดียว
- **Loop (S9):** [MEMORIZE]
  - **Thought:** โมเดลตัดสินใจว่าต้องทำอะไร
  - **Action:** โมเดลส่ง tool call แบบมีโครงสร้าง
  - **Observation:** environment ส่งผลลัพธ์กลับ
  - **Key point:** **observation (ไม่ใช่ความจำของโมเดล) เป็นตัวนำทางขั้นต่อไป**
- **Formal state machine (S10):**
  - State: s_t = (q, h_{1:t-1}) — q = original query, h_{1:t-1} = history ของ (thought, action, observation) ทุกรอบก่อนหน้า
  - Transition: s_t --LLM--> (thought_t, a_t) --env--> o_t --> s_{t+1}
  - Markov: P(s_{t+1} | s_t) = P(s_{t+1} | s_t, s_{t-1}, ..., s_1) (เพราะ s_t บรรจุ history ครบแล้ว)
  - Caveat: ถ้า truncate/summarize history observation จะขาดข้อมูล -> memory management สำคัญ
  - Implication: context window ต้องบรรจุ trajectory ทั้งหมด
- **Pseudocode (S11):** loop ที่เรียก `thought, action = llm_generate(prompt)`; **`max_steps` = safety guard**, **`build_prompt` ต้องรวม full history** (demo แสดงโค้ดนี้ตรง ๆ)
  - "A loop alone is not sufficient" — พฤติกรรม agentic มาจาก **policy ที่เลือก action จาก observation/tool feedback** ไม่ใช่แค่มี loop (pipeline ก็มี loop ได้)
- **Context growth (S12):** |s_t| = |system| + |q| + Σ_{i=1}^{t-1} (|thought_i| + |a_i| + |o_i|) — โตแบบ **linear O(n)**; ตัวอย่าง GPT-4o 128K context; ถ้าเฉลี่ย ~2,110 tokens/step "ทะลุ 128,000 ที่ step 64" (ดู Section 3 เรื่องความไม่ลงรอยของเลข) และต้องหยุดก่อนนั้นอีกเพราะต้องเหลือที่ให้ตอบ
- **4 prompting strategies บนคำถามเดียวกัน (S13, Figure 1 ของ paper)** — HotpotQA (คำถาม Apple Remote) และ AlfWorld:
  (a) **Standard** (ไม่มี thought/action), (b) **CoT = Reason-only**, (c) **Act-only**, (d) **ReAct = Reason + Act**
- **Benchmark (S14, Table 1, PaLM-540B)** [SHEET]:

| Method | HotpotQA (EM) | FEVER (Acc) |
|---|---|---|
| Standard | 28.7 | 57.1 |
| CoT (Wei et al. 2022) | **29.4** | 56.3 |
| CoT-SC (Wang et al. 2022) | 33.4 | 60.4 |
| Act | 25.7 | 58.9 |
| **ReAct** | **27.4** | **60.9** |
| CoT-SC -> ReAct | 34.2 | **64.6** |
| ReAct -> CoT-SC | **35.1** | 62.0 |
| Supervised SoTA | 67.5 | 89.5 |

  - **Ranking flips by task:** HotpotQA: ReAct (27.4) < CoT (29.4) แต่ FEVER: ReAct (60.9) > CoT (56.3) [MEMORIZE]
  - ReAct > Act ทั้งสอง task (27.4>25.7; 60.9>58.9)
  - **Combo ดีที่สุด:** HotpotQA = ReAct -> CoT-SC (35.1); FEVER = CoT-SC -> ReAct (64.6). Self-consistency เป็นตัวขับ gain ของ combo (CoT-SC เดี่ยว ๆ = 33.4)
  - CoT-SC ใน paper = sample 21 CoT trajectories, temperature 0.7, เลือก majority answer (เนื้อหาจากเชิงอรรถของ paper ที่ติดมาในสไลด์)
- **Error & hallucination analysis (S15, Table 2, 200 trajectories ที่คนวิเคราะห์ = ReAct 100 + CoT 100)** [SHEET]:

| Type | ReAct | CoT |
|---|---|---|
| Success: True positive (reasoning+facts ถูก) | 94% | 86% |
| Success: False positive (hallucinated trace/facts) | 6% | 14% |
| Failure: Reasoning error (รวมวนซ้ำแก้ไม่ได้) | 47% | 16% |
| Failure: Search result error (search ว่าง/ไม่มีประโยชน์) | 23% | - |
| Failure: **Hallucination** | **0%** | **56%** |
| Failure: Label ambiguity | 29% | 28% |

  - ข้อสรุป: ReAct **hallucinate น้อยกว่ามาก** (grounded ด้วย search) แต่ **reasoning error สูงกว่า** (47% vs 16%, มักติด loop ซ้ำ) และมี failure ใหม่ = search ไม่ได้ผล 23% [MEMORIZE]
  - ตัวเลข Failure เป็น % ของ "failure cases" (นอกสไลด์: นิยามตามปกติของ paper) ส่วน Success แบ่ง TP/FP

### 2.3 Tool Definition และ Tool Call Protocol (S16-S18)

- **JSON Schema = ฟอร์มเปล่า** (กล่องอะไรบ้าง, ชนิดอะไร, ช่องไหน required); ข้อมูลจริง = ฟอร์มที่กรอกแล้ว
- **Tool definition** = ฟอร์มที่ "แสดงให้ LLM เห็น" ประกอบด้วย **name**, **description** (เมื่อไรควรใช้), **parameters** (schema ของ input) [SHEET]
- การแมป definition -> call (S16):

| ใน definition | ใน call |
|---|---|
| name | name |
| parameters (รูปร่าง) | arguments (ค่า) |
| "type": "object" | arguments ต้องเป็น {...} |
| properties.query: string | "query": "..." |
| required: ["query"] | ห้ามขาด query |

  - ตัวอย่าง call: `{"name": "search_web", "arguments": {"query": "GDP Thailand 2024"}}` ; `arguments` ต้อง validate กับ schema `parameters`
- **Protocol 5 ขั้น (S17)** [SHEET]: (1) LLM generate tool call -> (2) **Parser** validate JSON กับ tool schema -> (3) **Dispatcher** route ไป function ที่ถูก -> (4) Tool execute คืน **string** -> (5) ผลถูก append เป็น **Observation** แล้ว loop ต่อ (LLM เห็นใน prompt รอบหน้า)
  - เฉพาะ `{"query": ...}` ข้างในมาจาก schema ของเรา; key ชั้นนอกเป็นของ protocol: OpenAI = `name`/`arguments`, Anthropic = `name`/`input`, demo = `Action: name[argument]` (เรียนรู้จาก fine-tuning หรือ prompt)
- **Failure modes (S17)** [SHEET]:

| Failure | ผล |
|---|---|
| Malformed JSON | Parser error -> retry หรือ fail |
| Hallucinated tool name | Tool not found -> error message |
| Infinite loop | `max_steps` guard |

- **Reliability problem (S18):** LLM generate ทีละ token ไม่มีอะไรการันตี JSON/code valid; วิธีเดิม (brittle) = regex parsing, retry loops, prompt engineering; **วิธีที่ดีกว่า = constrain ที่ระดับ logit** (Willard & Louf 2023)

### 2.4 Grammar-Constrained Decoding (S19-S26)

- **ปัญหา (S20):** vocabulary ~**50,000 (GPT-2) ถึงมากกว่า 250,000 (Gemma)** tokens ต่อ step; ไม่มีอะไรบังคับ syntax. JSON valid ต้อง: เริ่มด้วย `{` (สำหรับ tool schema), key เป็น quoted string, value ชนิดถูก, bracket ซ้อน/ปิดถูก. **Failure rate ขึ้นกับ benchmark/โมเดล/prompt/schema — อย่าสมมติ % สากล**
- **Logit-level masking (S21)** [SHEET]:
  - logit'_i = logit_i ถ้า token i ∈ V_valid ; = **−∞** ถ้าไม่ใช่ (ทำ **ก่อน softmax**)
  - P(token_i) = e^{logit'_i} / Σ_j e^{logit'_j}
  - invalid: e^{−∞} = 0 -> P = 0 ; valid tokens **แบ่ง 100% ของ probability mass**
  - Caveat: การันตี syntax เต็มรูปแบบต้อง parser state ถูก, จัดการ token-prefix, และ termination ถูก
- **Worked example (S22):** state = `{` ใน JSON; grammar อนุญาต 3 แบบ: เปิด key ด้วย `"`, ปิด `}` (empty object), whitespace
  - mask ทำที่ **token id ไม่ใช่ตัวอักษร** — GPT-2 vocab = **50,257 BPE tokens**
  - 3 continuations = **69 tokens** = 62 (เปิด key: `"`, ` "`, `":`, `","` ...) + 5 whitespace (`\t`, `\n`, `\r`, ` `, `\n\n`) + 2 รูป `}` (`}`, ` }`)
  - masked = **50,188** (**99.86%** ของ vocab)
  - ตรงข้าม: อยู่ใน string (`{"name": "A`) เกือบทุกอย่าง legal -> valid **50,146**, masked แค่ **111**. => **ความแรงของ mask ขึ้นกับ parser state**
- **CFG สำหรับ JSON (S23):** JSON อธิบายด้วย Context-Free Grammar (BNF). Parser state = rule ปัจจุบัน + **stack** ของ object/array ที่ยังไม่ปิด. **Stack คือสิ่งที่ regex/DFA ธรรมดาทำไม่ได้ -> เหตุผลที่ nesting ต้องใช้ CFG**. ในแต่ละ state CFG กำหนดเซต valid next token ที่ finite (แต่ชุด continuation เต็มอาจ infinite)
- **CFG-guided decoding 5 ขั้น (S24)** [SHEET]: (1) maintain parser state (ตำแหน่งใน rule + stack วงเล็บ), (2) compute valid next tokens, (3) apply logit mask (−∞), (4) **sample** จาก masked distribution, (5) update parser state ด้วย token ที่เลือก
  - ตัวอย่าง stack: `{"a": {"b": 1` ลึก 2 ชั้น -> `}` แรกปิด inner, `}` ที่สองยัง legal
  - Library: **`outlines`** (Willard & Louf) ; **Naive complexity: scan ทั้ง vocab ทุก step = O(|V|)**; precomputed state->token index ช่วยให้ lookup เร็ว แต่ apply mask + softmax ยังสเกลตาม |V| (ทั้ง decoding step **ไม่ใช่ O(1)**)
- **Regex constraint (S25):** ตัวอย่าง `\d{4}-\d{2}-\d{2}` (เช่น 2024-01-15). วิธี: (1) compile regex -> **DFA**, (2) ใช้ DFA state หา valid next tokens, (3) mask −∞ เหมือนกัน. ข้อดี: transition 1 ตัวอักษร = constant time. **ข้อจำกัด: ทำ balanced nesting ความลึกไม่จำกัดไม่ได้ (ไม่ใช่ regular language)** — แต่ JSON ที่จำกัดความลึกสูงสุดเป็น regular ได้ (state ระเบิด); ลึกไม่จำกัด => ต้อง CFG/pushdown automaton [MEMORIZE]
- **Performance (S26):** latency overhead ขึ้นกับความซับซ้อนของ grammar, tokenizer indexing, batch size, backend, cache -> ต้อง benchmark เอง. **Trade-off:** แลก flexibility/compute บางส่วนกับ syntactic validity; **semantic และ schema validation ยังจำเป็น**. ใช้ใน vLLM, TensorRT-LLM, llama.cpp (ตารางตัวเลข compliance เป็นรูปไม่มีใน txt)

### 2.5 Memory (S27-S31)

- ความท้าทาย: context window finite แต่ agent ต้องจำข้ามหลาย turn/session. อ้างอิง Park et al. 2023 (Generative Agents: **memory stream** -> retrieve -> action/plan/reflection) และ Sumers et al. 2024 (Cognitive Architectures for Language Agents)
- **Memory taxonomy 4 แบบ (S28)** [SHEET]:

| Type | คือ | ตัวอย่างสไลด์ | อุปมามนุษย์ |
|---|---|---|---|
| **Working (short-term)** | เนื้อหา context window ตอนนี้ — เร็ว จำกัด | | สิ่งที่กำลังคิดอยู่ |
| **Episodic** | บันทึกการโต้ตอบเฉพาะในอดีต | "what happened in conversation #42?" | บันทึก diary |
| **Semantic** | ความรู้/ข้อเท็จจริงที่ compile แล้ว | "Thailand's capital is Bangkok" | ตำรา |
| **Procedural** | ทักษะ/แพทเทิร์นที่เรียนรู้ | "how to search then summarize" | ปั่นจักรยาน |

- **Ingestion-time vs Runtime compilation (S29, Karpathy "LLM OS" 2023):** Ingestion (offline): raw docs -> extract key facts -> compile เป็น structured summaries -> store. Runtime (online): โหลด **เฉพาะ summary ที่เกี่ยว** เข้า context -> ตอบ. อุปมา: อ่านตำราทำ cheat sheet (ingestion) / เอาแค่ cheat sheet เข้าห้องสอบ (runtime). **"อย่ายัด raw docs เข้า prompt — pre-digest ก่อน"** [MEMORIZE]
- **Short vs Long term (S30):** Short = งานปัจจุบัน, tool results ล่าสุด, active plan. Long = บทสนทนาเก่า, knowledge ที่ compile แล้ว, user preferences. (ตารางเปรียบเทียบเป็นรูป)
- **3 implementations (S31)** [SHEET]:

| Memory | วิธี | ขนาด prompt | ข้อดี | ข้อเสีย |
|---|---|---|---|---|
| **Buffer** | append ทุกอย่าง | **O(n)** | ง่าย ไม่สูญเสีย | ล้น window ในที่สุด |
| **Summary** | เมื่อ buffer เกิน token budget, LLM บีบทุกอย่างยกเว้น **last k turns** เป็น summary เดียว | **O(k x msg + summary)** bounded | bounded | **lossy** |
| **Retrieval** | embed + vector search **top-k** | **O(k x \|msg\|)** | relevant | อาจพลาดของสำคัญเชิงเวลา; **store ยังเก็บ n turns (storage O(n))**, brute-force search O(n) (ANN ทำให้ sublinear) |

  - **Hybrid (default จริง):** summarize เทิร์นเก่า + retrieve episodes ที่เกี่ยวข้องทับ; ถ้า ranker 2 ตัวไม่ตรงกัน ใช้ **RRF (Week 12)** รวมอันดับ
  - สิ่งที่ "bounded" คือ **prompt ไม่ใช่ memory/storage**

### 2.6 Multi-Agent Topologies (S32-S43)

- **ทำไม (S33):** single agent ยาว ๆ context ล้น/error ตอนต้นลามต่อ; **Specialization** (prompt/tool เฉพาะ role), **Decomposition**, **Parallelism** (ถ้า coordination cost ต่ำ). **Trade-off:** messaging, consistency, latency, cost เพิ่ม. ใช้เมื่องาน decomposable จริง ไม่ใช่เพราะมี agent ให้ใช้
- **4 topologies** [SHEET]:

| Topology | Structure / Data flow | Formal | Pros | Cons |
|---|---|---|---|---|
| **1 Sequential Pipeline** (S34-35) | เรียงเส้นตรง output A = input B | y_i = f_i(y_{i-1}); Latency = **Σ_{i=1}^{N} T_i** (ไม่ได้ speedup) | simple, predictable, debug ง่าย | ช้า, rigid, **error cascade** (A พลาด B พลาด) |
| **2 Routing Network / Dispatcher** (S36-37) | router กลางส่งงานให้ specialist ตาม query (เช่น Billing/Tech/Sales) | r(q) -> agent_k | efficient, แยกโดเมน, ได้ผู้เชี่ยวชาญ | **Router = single point of failure** |
| **3 Supervisor-Led / Hierarchical** (S38-39) | tree: Supervisor plan -> assign -> sub-agents execute -> supervisor evaluate; สั่งแก้/reassign ได้ | meta-reasoning | flexible, รับงานกำกวม/ซับซ้อน, self-correcting | **latency สูงมาก, token cost สูง, implement ยาก** |
| **4 Debate / Consensus** (S40) | หลาย agent เสนอคำตอบ **อิสระ** (parallel) -> แลก argument/โหวต -> Judge สรุป | (Du et al. ICML 2024) | เหมาะ fact-checking, code review, brainstorming, complex reasoning | (นอกสไลด์: cost = หลาย agent x หลายรอบ) |

- **Sequential example (S35):** Research (tools: Search + Web Scraper -> `facts_list`) -> Writer (ไม่มี tool -> `draft_text`) -> Editor (tool: Dictionary -> `final_post`); `final = Editor(Writer(Research(query)))`. **Error cascade:** Research พลาด -> Writer เขียนร้อยแก้วลื่นไหลรอบ error -> Editor ขัดเกลา -> post สุดท้าย **"มั่นใจแต่ผิด"**
- **Router implementation 3 แบบ (S37)** [SHEET]:
  1. **LLM-based classification** (prompt ให้ output JSON category) — ช้าแต่ฉลาด
  2. **Embedding similarity:** k = argmax_i cos(E(q), E(desc_i)) — เร็ว/ถูกกว่า
  3. **Rule-based** (regex/keyword) — เร็วสุด แต่ rigid
  - **Key metric = Routing Accuracy** (router พลาด -> specialist พลาด)
- **Topology Comparison (S41):** "No single topology wins every scenario; choose by task requirements". ตารางจริงเป็นรูป — **ตารางสรุป 4 มิติตามที่ hints บอก (latency / complexity / robustness / best use) ที่ผมสรุปจากสไลด์แต่ละ topology:**

| | Latency | Complexity | Robustness | Best use |
|---|---|---|---|---|
| Sequential | สูง (Σ T_i) | ต่ำ | ต่ำ (cascade) | งาน step-by-step ชัดเจน (write -> check) |
| Routing | ต่ำ (router 1 ครั้ง + 1 specialist) | ปานกลาง | ต่ำ-กลาง (SPOF ที่ router) | หลายโดเมนแยกกัน (customer service) |
| Supervisor | สูงมาก | สูง | สูง (self-correcting) | งานกำกวม/ซับซ้อน |
| Debate | สูง (หลายรอบ) | สูง | สูง (cross-check) | fact-check, code review, reasoning |

  (ค่า Routing/Debate เป็นการอนุมานจากข้อความสไลด์ ตรวจกับสไลด์ S41 อีกที)
- **Communication protocols (S42):** 3 แบบ: **Shared Scratchpad** (state ร่วมง่าย แต่เสี่ยง context bloat + แก้ไขชนกัน), **Message Passing** (ส่งข้อความ bounded ถึง recipient ที่ระบุ เช่น AutoGen `send`/`receive`), **Structured Payloads** (JSON Schema สำหรับ field ที่ตรวจด้วยเครื่องได้)
  - Guidelines: เลือกตาม **task coupling, message volume, reliability needs**; **Design rule: นิยาม ownership, expected output, และ termination conditions (กัน infinite loop)**
  - AutoGen (Wu et al. 2024): `ConversableAgent` มี interface `send`, `receive`, `generate_reply`; agent ในตัว: **AssistantAgent**, **UserProxyAgent** (`human_input_mode="NEVER"/"ALWAYS"`), **GroupChatManager**; `initiate_chat(...)`; conversation-driven control flow

### 2.7 Synthesis, Reliability, Guardrails, Evaluation (S44-S52)

- **Single Agent = Planning + Memory + LLM + Constraints + Tools (S45)** [SHEET] ; ลำดับ: 1 Planning แตก sub-tasks, 2 Memory ดึง context/state, 3 LLM รัน ReAct loop (Thought->Action), 4 **Constraints** บังคับ grammar/JSON schema ตอน generate, 5 Tools ทำงานจริง -> append observation
  - **Multi-Agent System = Topology (x) {Agent1, Agent2, ...} + Protocol** (macro-level; topology: Routing, Supervisor, Debate)
- **Reliability Stack — Defense in Depth (S46)** [SHEET]: **L1 Constrain** (generate เฉพาะ grammar/schema ที่ valid) -> **L2 Validate** (types, ranges, permissions, business rules) -> **L3 Recover** (retry เฉพาะ failure ที่ bounded/recoverable พร้อม error feedback) -> **L4 Escalate** (ขออนุมัติมนุษย์สำหรับ action เสี่ยง/ทำลาย). "No single layer catches every failure"
- **Guardrails (S47):** Infinite loops -> `max_steps`, time, token budgets; Unauthorized actions -> **least privilege + allowlist**; **Prompt injection -> ถือ tool content เป็น untrusted data ไม่ใช่คำสั่ง**; Code execution -> isolation, resource limits; High-impact actions -> human confirmation + audit log
- **Evaluation (S48):** accuracy อย่างเดียวไม่พอ. Metrics: **Task Success Rate, Tool Call Efficiency (เรียกซ้ำซ้อนไหม), Cost & Latency per task**. Benchmarks: **WebArena** (เว็บจำลอง), **SWE-bench** (GitHub issues จริง), **GAIA** (multi-modal, multi-step). ตัวเลข: GAIA เปิดตัว 2023 คนได้ **92%** vs GPT-4+plugins **15%**; มี.ค. 2026 multi-model agent ~**92.4%**; SWE-bench Verified Pass@1 ~**79.2%** (จากหลักหน่วย 2023). Benchmark อิ่มตัวเร็ว -> วัด "scaffold" พอ ๆ กับโมเดล + ต้องตรวจ contamination (Week 14)
- **Frontier (S49):** ultimate tool = **Python code execution** (agent เขียน script รัน อ่าน output แทน predefined API); **Computer use / GUI agents** รับ screenshot/accessibility state ส่ง mouse/keyboard action; ต้องมี permissions, sandboxing, confirmation, monitoring
- **Ethics (S50):** accountability (ใครรับผิดชอบ), transparency (ต้องเปิดเผยว่าเป็นบอทไหม), consent (ทำแทนผู้ใช้โดยไม่ขออนุญาตรายงานได้ไหม) — เป็นประเด็นอภิปราย ไม่มีคำตอบตายตัว
- **Summary (S51):** `while` loop + ReAct; constrained decoding = logit masking (CFG/regex) **+ validate schema/semantic แยก**; Memory: Buffer vs Summary vs Retrieval; Topologies: Sequential / Routing / Supervisor / Debate; Guardrails: token limits, validation, human-in-the-loop
- **Where agents fit (S52):** W11 prompting/reasoning, W12 RAG, **W13 tool use + agent loops**, W14 data engines + evaluation (BERTScore, RAGAS, LLM-as-Judge)

---

## 3. Worked Calculations

### 3.1 Logit masking -> softmax (demo + S21) — ตัวเลขจริงจาก demo
logits = [1.0, 2.0, 3.0, 0.5], allowed ids = [1, 3] -> masked = [−∞, 2.0, −∞, 0.5]
- e^{2.0} = 7.389 ; e^{0.5} = 1.6487 ; e^{−∞} = 0
- ผลรวม = 7.389 + 1.6487 = 9.0377
- P(id1) = 7.389 / 9.0377 = **0.8176** ; P(id3) = 1.6487 / 9.0377 = **0.1824** ; (id0, id2 = 0.0000)
- ตรวจซ้ำทางลัด: P = 1/(1+e^{−(2.0−0.5)}) = 1/(1+e^{−1.5}) = 1/(1+0.2231) = 0.8176 ✓ ; รวม 0.8176+0.1824 = 1.0000 ✓
- ข้อสังเกต: token id2 (logit 3.0 สูงสุดเดิม) **ถูกตัดทิ้ง** แม้เดิมมีโอกาสสูงสุด; ลำดับ valid tokens ไม่เปลี่ยน (1 ยังชนะ 3) — mask ไม่เปลี่ยนอัตราส่วนระหว่าง valid tokens

### 3.2 จำนวน token ที่ถูก mask (S22)
- valid = 62 + 5 + 2 = **69** ✓
- masked = 50,257 − 69 = **50,188** ✓
- สัดส่วน = 50,188 / 50,257 = 0.99863 = **99.86%** ✓ (เช็ค: 69/50,257 = 0.00137 = 0.14%; 99.86+0.14 = 100)
- อยู่ใน string: masked = 50,257 − 50,146 = **111** ✓ (111/50,257 = 0.22%)
- Demo: P(valid) ก่อน mask = 0.4020 -> โอกาส emit สิ่งผิดกฎ = 100 x (1 − 0.4020) = **59.8%** (demo print ถูกตัดท้ายใน txt; ตัวเลข 59.8% คำนวณเอง); หลัง mask P(valid) = 1.0000

### 3.3 Context growth (S12 + demo + lab)
สูตร |s_t| = |system| + |q| + Σ_{i=1}^{t-1}(|thought_i| + |a_i| + |o_i|)
- **จาก demo (ReAct Tokyo / 1000):** cumulative tokens = [100, 141, 173] -> step เพิ่ม = 41 และ 32 tokens. (100 = system + "Question: ..." ตอน history ว่าง)
- **ตัวอย่างสมมติของผม (นอกสไลด์) — window 4,096 ตามโจทย์ lab:** base = 100, เพิ่ม 150 tokens/step
  - หลัง n step: 100 + 150n ; n = 26 -> 100 + 3,900 = **4,000 ≤ 4,096** (ยังอยู่) ; n = 27 -> 100 + 4,050 = **4,150 > 4,096** (ล้น)
  - ตรวจ: (4,096 − 100)/150 = 26.64 -> ล้นเมื่อ n ≥ 27 ✓
- **สไลด์ (128K, ~2,110 tokens/step):** 128,000 / 2,110 = **60.66** -> หลัง 60 step = 126,600 (ยังไม่เกิน), หลัง 61 step = 128,710 (เกิน). สไลด์เขียนว่า "step 64" ซึ่ง **ไม่ตรงกับการหารตรง ๆ** (ตารางจริงเป็นรูป; ถ้า 128K = 131,072 ก็ได้ 62.1 — ยังไม่ใช่ 64). **ถ้าข้อสอบให้เลขมา ให้คำนวณตามโจทย์ ไม่ต้องยึด 64**; ประเด็นที่ต้องจำคือ "linear O(n) + ต้องเหลือ headroom ให้โมเดลตอบ"

### 3.4 Sequential latency (S34)
Total latency = Σ T_i. สมมติ (นอกสไลด์) Research 3 s, Writer 10 s, Editor 4 s -> 3 + 10 + 4 = **17 s** (ไม่ใช่ max = 10 s, เพราะต้องรอทีละตัว). ถ้า Research พลาด -> ผลสุดท้ายผิดแต่ latency เท่าเดิม

### 3.5 Router ด้วย embedding (S37)
k = argmax_i cos(E(q), E(desc_i)). สมมติ cos = [Billing 0.31, Tech 0.82, Sales 0.45] -> k = Tech (ค่าสูงสุด 0.82). ใช้ argmax **ของ cosine similarity** ไม่ใช่ argmin

### 3.6 ReAct Table 1 — ผลต่าง
- HotpotQA: ReAct − CoT = 27.4 − 29.4 = **−2.0** (ReAct แพ้)
- FEVER: ReAct − CoT = 60.9 − 56.3 = **+4.6** (ReAct ชนะ)
- ReAct − Act: HotpotQA 27.4 − 25.7 = +1.7 ; FEVER 60.9 − 58.9 = +2.0
- CoT-SC -> ReAct vs CoT-SC เดี่ยว: HotpotQA 34.2 − 33.4 = +0.8 ; FEVER 64.6 − 60.4 = +4.2
- เทียบ Supervised SoTA: HotpotQA 67.5 − 35.1 = 32.4 (ยังห่างมาก)

### 3.7 Memory token usage (demo, ตอน turn 20 ของ 20 turns)
- Buffer: เพิ่ม **46 tokens/turn** ([736, 782, 828, 874, 920] ต่างกันทีละ 46 ✓), ที่ turn 20 = 46 x 20 = **920** ✓
- Retrieval (top_k=3): คงที่ **54** (flat O(k))
- Summary (max_tokens=200, keep_recent=3): [171, 107, 153, 199, 125] — **sawtooth** (พุ่งขึ้นแล้วตกเมื่อ compress)
- ลดลงเทียบ Buffer: Summary 125/920 = 0.136 -> ลด **86.4%** ; Retrieval 54/920 = 0.0587 -> ลด **94.1%**

### 3.8 Compliance rate (demo)
Unconstrained 0/20 = **0%** ; Constrained 20/20 = **100%** (N_TRIALS = 20)

---

## 4. Demo-only Content (ไม่อยู่บนสไลด์ แต่ออกได้ 1-2 ข้อ)

**Setup:** ไม่ใช้ LangChain/CrewAI/AutoGen/Outlines — สร้างจาก `torch`/`transformers` ล้วน; token counting ด้วย `tiktoken` **`cl100k_base`** (BPE ตระกูล GPT-4); ReAct section ใช้ **mock LLM** (scripted) เพื่อ reproducible; Section 2 ใช้ **`distilgpt2` (82M params), vocab 50,257** (ไม่ใช่ GPT-2 เต็ม แต่ใช้ tokenizer/vocab เดียวกัน)

**Section 1 — ReAct จากศูนย์**
- Tools = **Python functions ที่ string-in/string-out**; LLM **ไม่เรียกตรง** — มันสร้าง text ระบุ tool+input แล้ว **dispatcher** เรียกฟังก์ชันจริง [MEMORIZE]
- 3 tools: `search_wikipedia`, `calculator`, `get_weather`; calculator จำกัดตัวอักษรด้วย regex `[0-9\.\+\-\*\/\(\)\s]+` + `eval` ที่ `__builtins__={}` (teaching tool ไม่ใช่ sandbox จริง)
- Prompt format: `Thought: ...` / `Action: <tool_name>[<argument>]`; จบด้วย `Action: finish[<final answer>]`; "Then stop. Wait for an Observation"
- Parser: `re.search(r'Action:\s*(\w+)\[(.*?)\]', ..., re.DOTALL)` คืน `Action(name, argument)` (namedtuple) หรือ `None`
- `build_prompt(query, history)` **สร้าง prompt ใหม่ทั้งหมดทุก step** (= augmented state s_t = (q, h_{1:t-1}), ไม่ทิ้งอะไร)
- `react_agent(query, tools, llm_generate, max_steps=5)`: loop `for step in range(max_steps)`; ถ้า parse ไม่ได้ -> Observation = "Error: could not parse an Action..." แล้ว `continue`; ถ้า `action.name == 'finish'` -> return; ถ้า tool ไม่มี -> Observation "Error: tool ... does not exist. Available: [...]"; หมด loop -> `'Max steps reached — agent failed.'`
- ตัวอย่างรัน: "population of Tokyo divided by 1000" -> search_wikipedia[population of Tokyo] -> 14,000,000 -> calculator[14000000 / 1000] -> **14000.0** -> finish. (เกิดทั้งหมด 3 steps: step 0, 1, 2-FINAL)
- **Error recovery (1.6):** malformed action (ลืม bracket) ถูกแปลงเป็น Observation แล้ว **model แก้เองรอบหน้า**; control experiment: ถ้า **ไม่ feed error กลับ** โมเดลไม่เคย recover (loop จน max_steps) -> บทเรียน: **ต้องส่ง error เป็น observation, ไม่ใช่ crash** [MEMORIZE]

**Section 2 — Constrained decoding บน distilgpt2**
- Unconstrained: prompt `{"name": "` -> ได้ JSON ซ้ำ/พัง (เช่น `"text": "Mike McCready," "text": ...`) — โมเดลเล็กไม่ถูกฝึก JSON
- `apply_token_mask(logits, allowed_token_ids)`: สร้าง `mask = torch.full_like(logits, -inf)`; `mask[:, allowed] = 0`; **return logits + mask**. logits shape `[batch, vocab_size]`; next-token logits = `model(input_ids).logits[:, -1, :]` -> shape `[1, 50257]`
- หลัง `{`: valid tokens = **69 จาก 50,257** (ตรงสไลด์ S22); ต้องใช้ parser-aware test ไม่ใช่ `startswith('"')` ง่าย ๆ (ผิดทั้งสองทาง: ทิ้ง whitespace token, ยอมรับ `"""`); top-5 ก่อน mask: ` "` (logit −37.582), `\n` (−39.646)...; logits เป็นลบทั้งหมดได้ — ไม่เป็นไร softmax ใช้ค่าสัมพัทธ์
- P(valid) ก่อน mask = **0.4020**; หลัง mask = 1.0 (illegal "unreachable ไม่ใช่แค่ unlikely")
- **Regex-constrained:** `\d{4}-\d{2}-\d{2}` -> **10 transitions**: `\d \d \d \d - \d \d - \d \d`; precomputed index: `\d` = **10 tokens**, `-` = **1 token** -> **2 vocabulary scans สำหรับ 10 steps** (ผลของ Willard & Louf: index ครั้งเดียวต่อ character class)
- Greedy (argmax) ได้ `1111-11-11`; แบบ sample ได้ค่าอย่าง `3194-81-59` (เดือน 81 วัน 59 = format ถูก แต่ **semantics ผิด**) => **masking การันตี syntax ไม่ใช่ semantics -> ต้องมี schema/business validation (S26)** [MEMORIZE]
- Benchmark 20 trials: unconstrained 0/20 (0%) vs constrained 20/20 (100%) (ตัว unconstrained ได้เปรียบแล้ว เพราะใช้ `.search()` ส่วน constrained ใช้ `.fullmatch()`)
- Recap: ต้นทุน masking ~1 vocab scan ต่อ character class ใหม่

**Section 3 — Memory**
- BufferMemory: 2 turns = 20 tokens
- SummaryMemory: เมื่อ `token_count() > max_tokens` เรียก `_compress()` เก็บ `keep_recent` turns ล่าสุด; **summary ถูกเขียนใหม่จาก running totals (n_summarized, topics) ไม่ใช่ append** — ถ้า append summary จะโตเป็น O(n) อีก; `MAX_TOPICS = 20` จำกัด; `max_tokens` เป็น **trigger ไม่ใช่ hard ceiling**; lossy by design. ตัวอย่าง: `max_tokens=60, keep_recent=2` -> "Summary of 10 earlier turns" + 2 turns ล่าสุด = 50 tokens
- RetrievalMemory: ใช้ **TF-IDF + cosine similarity** (ไม่ใช่ dense encoder เพื่อไม่ต้องโหลดโมเดล; กลไก score-แล้ว-top-k เหมือนกัน; เปลี่ยนเป็น SBERT เปลี่ยนคะแนน ไม่เปลี่ยน architecture). Query 'Tell me about topic3' -> ได้ topic3 และ **topic5** (เพราะคำว่า "about") = จุดอ่อน "may miss/mis-rank"; TF-IDF fit ใหม่ทั้ง history ทุก query
- 20-turn simulation: Buffer linear (O(n)) / Summary sawtooth bounded / Retrieval flat O(k)

**Section 4 — Multi-agent (mock)**
- Sequential `blog_pipeline = editor_agent(writer_agent(research_agent(topic)))`; หัวข้อที่ไม่รู้ (Chiang Mai) -> Research คืน "No article found..." -> Writer เขียน "Fun fact: No article found for ..." = **error cascade**; Writer แยก fact กับ error string ไม่ได้, Editor ไม่ตรวจ fact
- Routing แบบ **rule-based keyword, first rule that hits wins**; default desk = tech; query 3 "cancel my premium plan because charged" -> ถูกส่งไป **sales** (keyword 'premium'/'plan' ชนก่อน 'charged') ทั้งที่เป็น **billing dispute** => router พลาด specialist แก้ไม่ได้ (SPOF); **ลำดับ rule มีผล**

**Lab (ไม่ออกสอบ):** แก้บรรทัดเดียวใน `react_step()`; context growth ด้วย tiktoken; 4,096-token window

---

## 5. Tagging สรุป

**[MEMORIZE] (ต้องอยู่ในหัว)**
- Agent = LLM คุมการไหลของงาน; ไม่ได้ฉลาดขึ้น แต่ capable ขึ้นด้วย tool
- 4 components: LLM brain, Tools, Memory, Planning
- ReAct = Thought -> Action -> Observation; observation นำทางขั้นต่อไป; s_t = (q, h_{1:t-1})
- ReAct vs CoT: HotpotQA ReAct แพ้ (27.4 vs 29.4), FEVER ReAct ชนะ (60.9 vs 56.3); ReAct hallucinate 0% vs CoT 56%; ReAct reasoning error 47% vs 16%
- `max_steps` กัน infinite loop; build_prompt ต้องรวม history ทั้งหมด -> context โต linear
- Masking = ตั้ง logit = −∞ ก่อน softmax -> P=0; การันตี syntax ไม่ใช่ semantics
- Regex/DFA ทำ nested ลึกไม่จำกัดไม่ได้ -> ต้อง CFG (stack)
- Memory 4 ประเภท; Buffer O(n) / Summary bounded lossy / Retrieval bounded relevant (store ยัง O(n))
- 4 topologies + จุดอ่อน: Sequential=error cascade, Routing=SPOF, Supervisor=latency/cost สูง, Debate=หลาย agent
- Reliability stack: Constrain -> Validate -> Recover -> Escalate
- Prompt injection: tool content = untrusted data

**[SHEET] candidates (เรียงตามความสำคัญ):**
1. ReAct Table 1: Std 28.7/57.1, CoT 29.4/56.3, CoT-SC 33.4/60.4, Act 25.7/58.9, **ReAct 27.4/60.9**, CoT-SC->ReAct 34.2/64.6, ReAct->CoT-SC 35.1/62.0 (HotpotQA EM / FEVER Acc)
2. Table 2: ReAct vs CoT = TP 94/86, FP 6/14, reasoning err 47/16, search err 23/-, halluc 0/56, label amb 29/28
3. Masking: logit' = logit if valid else −∞; P = e^{l'_i}/Σ e^{l'_j}
4. GPT-2 vocab 50,257; หลัง `{` valid 69 = 62+5+2, masked 50,188 (99.86%); ใน string valid 50,146, masked 111
5. s_t = (q, h_{1:t-1}); |s_t| = |sys|+|q|+Σ(|thought|+|a|+|obs|)
6. Topology 4 มิติ: Seq = latency สูง/ซับซ้อนต่ำ/robust ต่ำ; Routing = SPOF; Supervisor = latency+cost สูงมาก/self-correct; Debate = parallel + judge
7. Sequential: y_i = f_i(y_{i-1}), Latency = Σ T_i ; Router: r(q)->agent_k, k = argmax cos(E(q), E(desc_i))
8. Memory big-O: Buffer O(n); Summary O(k x msg + summary); Retrieval O(k x |msg|) prompt, storage/search O(n)
9. Tool def: name + description + parameters (JSON Schema: type object, properties, required); call: name + arguments (OpenAI), name + input (Anthropic)
10. Tool call 5 ขั้น: generate -> parser validate -> dispatcher -> execute (string) -> append Observation; failures: malformed JSON / hallucinated tool name / infinite loop (`max_steps`)
11. Reliability stack 4 ชั้น; Guardrails: max_steps/time/token, least privilege, injection, sandbox, human confirm
12. Demo: regex `\d{4}-\d{2}-\d{2}` = 10 transitions, 2 vocab scans; compliance 0/20 vs 20/20; buffer +46 tokens/turn
13. GB/GiB และ bytes อื่น ๆ (นอกสัปดาห์นี้ — ไม่ใส่)

---

## 6. Fill-in-the-blank candidates (10-15)

1. "ReAct สลับ ______ (reasoning trace) กับ ______ (tool call) ใน generation เดียว" -> **Thought / Action** (S8)
2. "ใน ReAct ทุกรอบ ______ ที่ได้จาก tool เป็นตัวนำทางขั้นต่อไป" -> **Observation** (S9)
3. "ReAct บน HotpotQA ได้ EM ______ ขณะที่ CoT ได้ ______; บน FEVER ReAct ได้ ______ vs CoT ______" -> **27.4 / 29.4 / 60.9 / 56.3** (S14)
4. "ต้องมี ______ เพื่อป้องกัน agent วน loop ไม่รู้จบ" -> **`max_steps`** (S11, S17)
5. "State ของ ReAct คือ s_t = (______, ______)" -> **q (query), h_{1:t-1} (full history)** (S10)
6. "Context length ของ ReAct agent เพิ่มแบบ ______ ตามจำนวน step (Big-O ______)" -> **linear / O(n)** (S12, S31)
7. "Constrained decoding ตั้งค่า logit ของ token ที่ invalid เป็น ______ ก่อน softmax ทำให้ความน่าจะเป็น = ______" -> **−∞ / 0** (S21)
8. "หลัง `{` ใน JSON, GPT-2 vocab 50,257 มี token ที่ valid ______ ตัว (mask ______ ตัว, ______%)" -> **69 / 50,188 / 99.86%** (S22)
9. "JSON ที่ nest ลึกไม่จำกัดต้องใช้ ______ (มี stack) ไม่ใช่ regex/______" -> **CFG (pushdown automaton) / DFA** (S23, S25)
10. "Memory 4 ประเภท: ______, Episodic, ______, Procedural" -> **Working (short-term) / Semantic** (S28)
11. "Summary memory ใช้ prompt ขนาด O(______) ส่วน Retrieval memory ใช้ O(______)" -> **k x msg + summary / k x |msg|** (S31)
12. "Sequential pipeline มี latency รวม = ______ และจุดอ่อนคือ ______" -> **Σ T_i / error cascade** (S34)
13. "Routing network ที่ router พลาด -> ______ ; metric สำคัญคือ ______" -> **single point of failure / routing accuracy** (S36-37)
14. "Reliability stack 4 ชั้น: Constrain, ______, ______, Escalate" -> **Validate, Recover** (S46)
15. "Single Agent = Planning + ______ + LLM + ______ + Tools" -> **Memory, Constraints** (S45)
16. Tool definition ประกอบด้วย ______, ______, ______ -> **name, description, parameters** (S16)

---

## 7. Common MCQ traps / misconceptions

1. **"Agent = LLM ที่ฉลาดขึ้น / ถูก train ใหม่"** ผิด -> LLM เท่าเดิม; capability มาจาก tooling + loop (S6)
2. **"มี loop = agent"** ผิด -> ต้องมี policy เลือก action จาก observation/tool feedback; pipeline ก็มี loop ได้ (S11)
3. **ReAct ชนะ CoT ทุก benchmark** ผิด -> HotpotQA ReAct (27.4) < CoT (29.4); FEVER ReAct > CoT (60.9 vs 56.3). Combo ReAct+CoT-SC ดีที่สุด
4. **ReAct ไม่ hallucinate เลย/ไม่มี error** ผิด -> hallucination 0% ใน failure แต่ reasoning error 47% (วน loop ซ้ำ) + search error 23% (S15). Distractor สลับตัวเลข ReAct/CoT (hallucination 56% เป็นของ **CoT**)
5. **State ของ ReAct ไม่ Markov เพราะต้องจำ history** ผิด -> Markov **บน augmented state** (ที่รวม history แล้ว); ถ้าบีบ history อาจ partially observable (S3, S10)
6. **Agent ที่ memory เป็น Retrieval จึงใช้ storage O(k)** ผิด -> prompt O(k x |msg|) แต่ store ยังเก็บ n turns, search O(n) brute-force (S31)
7. **Summary memory ไม่สูญเสียข้อมูล / bounded เสมอ** ผิด -> lossy; bounded ต่อเมื่อจำกัดขนาด summary (demo MAX_TOPICS)
8. **Logit masking เป็นการลดน้ำหนัก (soft penalty)** ผิด -> **hard constraint**: −∞ -> P = 0 เป๊ะ; invalid "unreachable ไม่ใช่แค่ unlikely"
9. **Masking การันตีค่าถูกต้อง (เช่น วันที่จริง)** ผิด -> การันตี **syntax** เท่านั้น; `3194-81-59` ผ่าน regex; ต้อง validate schema/semantic แยก (S26)
10. **Constrained decoding เป็น O(1) ต่อ step** ผิด -> lookup เร็วได้ แต่ mask + softmax ยังสเกลตาม |V|; naive scan = O(|V|) (S24)
11. **Regex (DFA) ใช้บังคับ JSON ซ้อนลึกไม่จำกัดได้** ผิด -> ต้อง CFG + stack; (nuance: JSON ลึกจำกัดเป็น regular ได้ แต่ state ระเบิด) (S25)
12. **mask ทำที่ตัวอักษร** ผิด -> ทำที่ **token id** (BPE หลาย token สะกดรูปเดียวกันได้ เช่น `"` และ ` "`) (S22)
13. **Mask ตัด token เท่ากันทุก state** ผิด -> หลัง `{` ตัด 99.86% แต่ใน string ตัดแค่ 111 ตัว
14. **Sequential pipeline เร็วเพราะ agent ทำงานคู่ขนาน** ผิด -> latency = Σ T_i ไม่มี parallelism speedup; error cascade
15. **Routing network = robust เพราะมีหลาย specialist** ผิด -> router เป็น SPOF; specialist ที่ถูกส่งผิดตอบผิดคำถามอย่างมั่นใจ
16. **Supervisor เหมาะกับงานเร็ว/ถูก** ผิด -> flexible + self-correcting แต่ latency สูงมาก, token cost สูง, implement ยาก
17. **Debate = agent เรียงต่อกัน** ผิด -> เสนอ **อิสระแบบ parallel** แล้วแลก argument/โหวต + Judge สรุป
18. **มี agent เยอะ = ดีกว่าเสมอ** ผิด -> เพิ่ม messaging/consistency/latency/cost; ใช้เมื่อ decomposable จริง; "No single topology wins"
19. **Tool = LLM รันโค้ดเอง** ผิด -> LLM แค่ **output text ระบุ tool + args**; dispatcher เรียกจริง, ส่งผลกลับเป็น string observation
20. **Malformed output -> ควร crash/ทิ้ง** ผิด -> ส่ง error กลับเป็น Observation ให้โมเดลแก้เอง (demo 1.6) และ Recover เฉพาะ failure ที่ bounded
21. **ชั้นเดียวพอ (constrain อย่างเดียว)** ผิด -> Defense in depth 4 ชั้น: Constrain, Validate, Recover, Escalate
22. **เนื้อหาใน tool result เชื่อได้เหมือน system prompt** ผิด -> prompt injection: ถือเป็น untrusted data (S47)
23. **Router ใช้ argmin cos / LLM router เร็วสุด** ผิด -> embedding = argmax cos, เร็ว/ถูก; **rule-based เร็วสุดแต่ rigid; LLM-based ช้าแต่ฉลาด** (S37)
24. **Ingestion-time compilation = โหลด raw docs ลง prompt ตอน runtime** ผิด -> ทำ offline แล้วโหลดเฉพาะ summary ที่เกี่ยวข้องตอน runtime
25. ผิดสัปดาห์: Hints ระบุ "bookend reordering" และ CLIP/multimodal RAG — **ไม่อยู่ในสไลด์ W13** (เป็นเรื่อง RAG ของ Week 12); ไม่ต้องอ่านเพิ่มสำหรับสัปดาห์นี้

---

## 8. Flags: สไลด์ vs hints / ความไม่ลงรอย

- **ชื่อ topology:** hints (ดู Week 14 `00:56:14`) ใช้ "sequential / routing / **parallel** / hierarchical"; **สไลด์ W13 ใช้ Sequential / Routing / Supervisor-Led (Hierarchical) / Debate-Consensus** (Debate = parallel generation + judge). ตามสไลด์; map: parallel ≈ Debate (ผมเดา), hierarchical = Supervisor
- **"Step 64" (S12)** ไม่ตรงกับ 128,000/2,110 ≈ 60.7 — ตรวจตารางในสไลด์จริง
- ตารางที่เป็นรูป (S12, S26, S30, S41) และ pseudocode (S11, S39) ไม่อยู่ใน txt -> ค่าในตาราง topology comparison ของผมเป็นการสังเคราะห์
- "Week 13 `01:58:45` ReAct Table 1: ReAct 27.4 vs CoT 29.4; FEVER 60.9 vs 56.3" ยืนยันตรงกับสไลด์ S14 ✓; `max_steps` ✓ (S11, S17, S47)

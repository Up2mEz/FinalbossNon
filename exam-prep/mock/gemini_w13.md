**Q51** [W13] [ReAct Framework & Agent Architecture] [difficulty: medium]
ข้อใดอธิบายสถาปัตยกรรมและหลักการทำงานของ LLM Agent ในรูปแบบ ReAct ได้ถูกต้องที่สุดตามความเป็นจริงของระบบ?
A. LLM Agent คือโมเดลภาษาที่ได้รับการ fine-tune เพิ่มเติมจนสามารถ execute คำสั่งภายนอกและเรียกใช้ฟังก์ชันภาษา Python ได้โดยตรงโดยไม่ต้องอาศัยตัวกลาง
B. การที่ระบบมี Feedback loop หรือโครงสร้างการวนซ้ำการประมวลผล เพียงพอที่จะจัดประเภทเป็น Agent ได้ทันทีแม้จะไม่มี policy ในการเลือก action จาก observation
C. LLM ไม่ได้เรียกใช้ Tool โดยตรง แต่ทำหน้าที่สร้าง text ระบุชื่อ Tool และ Argument แล้วมี Dispatcher ภายนอกนำไปประมวลผลจริงก่อนส่ง Observation กลับเข้ามา
D. ReAct agent ทำงานโดยรวม Thought และ Action เข้าเป็นขั้นตอนเดียวกัน แต่จะหยุดรอให้ผู้ใช้ป้อน Observation ใหม่เข้ามาเสมอจึงจะสามารถตัดสินใจต่อไปได้

**Q52** [W13] [ReAct Performance & Benchmarks] [difficulty: medium]
จากการทดลองเปรียบเทียบประสิทธิภาพของ ReAct เทียบกับ Chain-of-Thought (CoT) ในตารางผลการทดลอง (Table 1) และการวิเคราะห์ข้อผิดพลาด ข้อใดกล่าวถูกต้อง?
A. บน HotpotQA ReAct ได้ผลลัพธ์ต่ำกว่า CoT (27.4 vs 29.4) แต่บน FEVER ReAct ทำคะแนนได้สูงกว่า CoT (60.9 vs 56.3)
B. ReAct มีประสิทธิภาพเหนือกว่า CoT ในทุกชุดข้อมูลเนื่องจากมี external knowledge ช่วยกำจัดทั้ง hallucination และ reasoning error ได้อย่างสมบูรณ์
C. บน HotpotQA ReAct ได้คะแนนสูงกว่า CoT อย่างมีนัยสำคัญ แต่บน FEVER ได้คะแนนต่ำกว่าเนื่องจากข้อจำกัดด้าน context window
D. สาเหตุหลักของความล้มเหลว (failure) ใน ReAct เกิดจากการเกิด hallucination สูงถึง 56% ซึ่งเป็นอัตราเดียวกันกับข้อผิดพลาดที่พบใน CoT แบบเดี่ยว

**Q53** [W13] [Tool Calling & Error Handling] [difficulty: hard]
ในการสร้าง ReAct Agent จากศูนย์ หาก LLM สร้างคำสั่งที่มี Action syntax ผิดพลาด (malformed action เช่น ลืมใส่วงเล็บปิด `Action: calculator[10 + 20`) ระบบควรจัดการอย่างไรตามผลการทดลอง Error Recovery เพื่อให้ Agent สามารถกู้คืนและทำงานต่อไปได้สำเร็จ?
A. สั่ง throw exception และ crash ระบบทันทีเพื่อป้องกันไม่ให้เสีย token ไปกับ context ที่ผิดพลาด
B. ทำการ auto-correct สัญลักษณ์ที่ขาดหายไปแบบ hard-coded ใน parser ทันทีโดยไม่ต้องแจ้งให้ LLM รับทราบ
C. สั่ง reset ประวัติการสนทนา (history) ทั้งหมดแล้วเริ่มต้นรัน query ใหม่เพื่อล้าง bad generation ทิ้ง
D. ดักจับข้อผิดพลาดแล้วแปลงเป็นข้อความส่งกลับเข้าไปเป็น Observation (เช่น 'Error: could not parse an Action...') ใน history ของรอบถัดไป

**Q54** [W13] [Logit-level Constrained Decoding] [difficulty: medium]
กำหนดให้ Vocabulary มี 4 tokens ด้วยค่า Logits ก่อน mask ดังนี้: `id0: 1.0`, `id1: 2.0`, `id2: 3.0`, `id3: 0.5` หากโครงสร้างไวยากรณ์กำหนดให้ allowed token IDs ใน step นี้มีเพียง `[1, 3]` (โดย token อื่นเป็น invalid) จงคำนวณความน่าจะเป็น P(id1) ภายหลังการทำ Logit Masking และผ่าน Softmax layer (กำหนดให้ $e^{2.0} \approx 7.389$, $e^{0.5} \approx 1.6487$)
A. 0.2447
B. 0.8176
C. 0.5000
D. 0.6558

**Q55** [W13] [Constrained Decoding: Capabilities & Limitations] [difficulty: medium]
ในการนำ Regex-constrained decoding ไปใช้บังคับโครงสร้างวันที่ในรูปแบบ `\d{4}-\d{2}-\d{2}` ระบบสุ่มได้สตริง `3194-81-59` ออกมา ปรากฏการณ์นี้สะท้อนข้อจำกัดใด และหากต้องการบังคับโครงสร้าง JSON ที่สามารถ nest ลึกได้ไม่จำกัดจำเป็นต้องใช้ทฤษฎี/กลไกใด?
A. การทำ Masking รับประกันเฉพาะความถูกต้องทางไวยากรณ์ (Syntax) แต่ไม่ได้รับประกันความถูกต้องเชิงความหมาย (Semantics) และ JSON ที่ nest ลึกไม่จำกัดต้องใช้ Context-Free Grammar (Pushdown Automaton)
B. Logit masking เป็นเพียง Soft penalty จึงทำให้ token ผิดกฎหลุดรอดออกมาได้ และ JSON ที่ nest ลึกสามารถควบคุมได้อย่างสมบูรณ์ด้วย Regular Expression (DFA)
C. ปัญหาเกิดจาก Vocabulary scan ไม่ครอบคลุม และโครงสร้าง JSON ที่ nest ลึกไม่จำกัดสามารถประมวลผลได้ด้วย Finite Automata ธรรมดาโดยไม่ต้องใช้ stack
D. เกิดจากความผิดพลาดในการทำ Argmax greedy decoding และการควบคุมโครงสร้าง JSON ที่ซับซ้อนต้องอาศัยการ Fine-tune โมเดลร่วมกับ Few-shot prompt เท่านั้น

**Q56** [W13] [Memory Architectures & Token Trade-offs] [difficulty: hard]
จากการทดลองจำลองระบบสนทนาจำนวน 20 turns ตาม Study Notes ข้อใดอธิบายพฤติกรรมการใช้ token ใน Prompt ของสถาปัตยกรรม Memory ทั้ง 3 รูปแบบ (Buffer, Summary, Retrieval) ได้อย่างถูกต้อง?
A. Buffer มีอัตราการใช้ token คงที่แบบ O(1) ขณะที่ Summary โตแบบ Exponential และ Retrieval โตแบบ Linear ตามขนาด history ทั้งหมด
B. Buffer โตแบบ Linear O(n), Summary มีการใช้ token เป็นเส้นตรงคงที่ O(1) โดยไม่สูญเสียข้อมูล, และ Retrieval มี token เป็นรูป Sawtooth
C. Buffer มีจำนวน token เพิ่มขึ้นแบบ Linear O(n), Summary มีลักษณะเป็นฟันปลา (Sawtooth) ซึ่งจะพุ่งขึ้นแล้วลดลงเมื่อถูกบีบอัด, และ Retrieval มีขนาดคงที่ O(k) ตามจำนวนเอกสารที่ดึงมา
D. ทั้งสามรูปแบบมีอัตราการเติบโตของ context ใน Prompt เท่ากัน แต่ Retrieval Memory ช่วยลดขนาดพื้นที่จัดเก็บบน Database ให้เหลือ O(k) เสมอ

**Q57** [W13] [Multi-agent Topology: Sequential Pipeline] [difficulty: easy]
ระบบ Multi-agent แบบ Sequential Pipeline ประกอบด้วย 3 agents ทำงานต่อเนื่องกัน: Research Agent (ใช้เวลา $T_1 = 3$ วินาที) $\rightarrow$ Writer Agent (ใช้เวลา $T_2 = 10$ วินาที) $\rightarrow$ Editor Agent (ใช้เวลา $T_3 = 4$ วินาที) จงหา Total Latency ของระบบตามสูตรสถาปัตยกรรม และระบุความเสี่ยงเชิงโครงสร้างที่สำคัญที่สุดของ Pipeline นี้
A. Total Latency = 10 วินาที และมีความเสี่ยงแบบ Single Point of Failure ที่ Router
B. Total Latency = 14 วินาที และมีความเสี่ยงที่ Agent จะติด Deadlock ในระหว่างการส่งต่อ State
C. Total Latency = 17 วินาที และระบบสามารถกำจัดข้อผิดพลาดได้สมบูรณ์เนื่องจากมี Editor Agent คอย Fact-check เนื้อหา
D. Total Latency = 17 วินาที (คำนวณจาก $\sum T_i$) และมีความเสี่ยงแบบ Error cascade ที่ข้อผิดพลาดจาก Agent ต้นทางจะถูกส่งต่อและขยายผลไปยัง Agent ปลายทาง

**Q58** [W13] [Multi-agent Topology: Routing] [difficulty: medium]
ระบบ Customer Support ใช้ Routing Agent แบบ Rule-based keyword matching (first rule that hits wins) โดยเรียงลำดับกฎดังนี้: 1) Sales ('premium', 'plan') 2) Billing ('charged', 'invoice') และ Default คือ Tech Support หากผู้ใช้ส่งข้อความว่า `"cancel my premium plan because charged unfairly"` จะเกิดผลลัพธ์อย่างไร และสะท้อนจุดอ่อนใดของ Topology นี้?
A. ข้อความถูกส่งไป Billing เพราะมีคำว่า 'charged' และปัญหาการเงินมีความสำคัญกว่า ทำให้ระบบทำงานได้ถูกต้อง
B. ข้อความถูกส่งไป Sales อย่างผิดพลาดเนื่องจากตรวจพบคำว่า 'premium'/'plan' ก่อน และสะท้อนว่า Router เป็น Single Point of Failure (SPOF)
C. ข้อความถูกส่งไป Tech Support โดยอัตโนมัติเนื่องจากมี keyword ชนกันสองแผนก ทำให้ Router สรุปว่าเป็นข้อผิดพลาด
D. ข้อความถูกส่งแบบ Broadcast ไปทั้ง Sales และ Billing เพื่อรอผลสรุปแบบ Debate Topology

**Q59** [W13] [Agent Context Growth & State Taxonomy] [difficulty: hard]
ข้อใดอธิบายคุณสมบัติของ Agent State และการจัดหมวดหมู่ Memory ตาม Taxonomy 4 ประเภทใน Study Notes ได้อย่างถูกต้อง?
A. ReAct Agent ละเมิดคุณสมบัติ Markov Property โดยสิ้นเชิงเนื่องจากต้องใช้ประวัติในอดีต จึงไม่สามารถนิยามให้อยู่ในรูป Markov state ได้
B. Taxonomy ของ Agent Memory แบ่งออกเป็น 4 ประเภท ได้แก่ Short-term, Mid-term, Long-term, และ Permanent Memory
C. State ของ ReAct นิยามเป็น Augmented State $s_t = (q, h_{1:t-1})$ ซึ่งทำให้ระบบคงคุณสมบัติ Markov Property บน state ดังกล่าว และ Memory 4 ประเภทคือ Working, Episodic, Semantic, และ Procedural
D. Context length ของ ReAct Agent มีอัตราการเติบโตแบบ Exponential ตามจำนวน Step เนื่องจากการทำซ้ำ Thought และ Action ในประวัติ

**Q60** [W13] [Agent Reliability & Safeguards] [difficulty: easy]
ในการออกแบบระบบ Agentic AI สำหรับ Production สถาปัตยกรรม 'Reliability Stack' มีการจัดลำดับ 4 ชั้นอย่างไรเพื่อควบคุมความถูกต้อง และ safeguard พื้นฐานที่สุดที่จำเป็นต้องมีใน ReAct Loop เพื่อป้องกันการทำงานวนซ้ำไม่สิ้นสุดคืออะไร?
A. Reliability Stack ประกอบด้วย Constrain $\rightarrow$ Validate $\rightarrow$ Recover $\rightarrow$ Escalate และ safeguard พื้นฐานคือการกำหนด `max_steps`
B. Reliability Stack ประกอบด้วย Plan $\rightarrow$ Execute $\rightarrow$ Evaluate $\rightarrow$ Terminate และ safeguard พื้นฐานคือการปรับ Sampling Temperature ให้เป็น 0
C. Reliability Stack ประกอบด้วย Prompt $\rightarrow$ Fine-tune $\rightarrow$ RAG $\rightarrow$ RLHF และ safeguard พื้นฐานคือการเปลี่ยนไปใช้ Retrieval Memory
D. Reliability Stack ประกอบด้วย Filter $\rightarrow$ Block $\rightarrow$ Alert $\rightarrow$ Retry และ safeguard พื้นฐานคือการจำกัดคำตอบให้มีความยาวไม่เกิน 69 tokens

---

**F7** [W13] [Logit-level Constrained Decoding]
ในกระบวนการ Logit-level Constrained Decoding ค่า logit ของ token ที่ invalid ตามโครงสร้างไวยากรณ์จะถูกแทนที่ด้วยค่า ___ ก่อนส่งเข้าคำนวณใน Softmax layer ส่งผลให้ความน่าจะเป็น (Probability) ของ token ดังกล่าวกลายเป็น ___ พอดี

**F8** [W13] [ReAct vs CoT Benchmark Evaluation]
จากผลการประเมินใน Table 1 ของเปเปอร์ ReAct บนชุดข้อมูล HotpotQA โมเดล ReAct ทำคะแนน Exact Match (EM) ได้เท่ากับ ___ (ขณะที่ CoT ทำได้ 29.4) ส่วนบนชุดข้อมูล FEVER โมเดล ReAct สามารถทำคะแนนได้เท่ากับ ___ (ขณะที่ CoT ทำได้ 56.3)

---

### Answer key
- **Q51**: **C** — LLM ไม่ได้เรียกฟังก์ชันตรงๆ แต่ generate ข้อความระบุชื่อ tool/args เพื่อให้ dispatcher นำไปประมวลผลจริง และ agent ต้องมี policy เลือก action จาก observation ไม่ใช่แค่มี loop [Section 4 Section 1, Section 7 Trap 1-2]
- **Q52**: **A** — ReAct ไม่ได้ชนะทุกชุดข้อมูล โดยแพ้ CoT บน HotpotQA (27.4 vs 29.4) แต่ชนะบน FEVER (60.9 vs 56.3) และ failure ของ ReAct มาจาก reasoning/search error ไม่ใช่ hallucination 56% ซึ่งเป็นของ CoT [Section 1 Hints, Section 3.6, Section 7 Trap 3-4]
- **Q53**: **D** — การทดลอง Error recovery ชี้ชัดว่าต้องส่ง error message กลับไปเป็น Observation เพื่อให้โมเดลเรียนรู้และแก้ไขเองในรอบถัดไป หากสั่ง crash โมเดลจะไม่สามารถกู้คืนสถานะได้ [Section 4 Section 1.6]
- **Q54**: **B** — คำนวณจาก Logits หลัง mask: $e^{2.0} = 7.389$, $e^{0.5} = 1.6487$; ผลรวม $7.389 + 1.6487 = 9.0377$; จะได้ $P(\text{id1}) = 7.389 / 9.0377 = 0.8176$ [Section 3.1]
- **Q55**: **A** — Masking บังคับได้เฉพาะ Syntax ไม่ได้รับประกัน Semantics จึงเกิดวันที่ที่เป็นไปไม่ได้ในทางความหมาย และการ parse โครงสร้าง nest ลึกไม่จำกัดต้องใช้ CFG (Pushdown Automaton) ไม่ใช่ Regex (DFA) [Section 4 Section 2, Section 6 item 9, Section 7 Trap 8]
- **Q56**: **C** — Buffer เพิ่ม token แบบ Linear $O(n)$, Summary มีพฤติกรรมเป็นฟันปลา (Sawtooth) เพราะขนาดลดลงเมื่อเกิดการบีบอัด, และ Retrieval มีขนาดใน prompt คงที่ $O(k)$ ตามจำนวนท็อปปิกที่ดึงมา [Section 3.7, Section 4 Section 3, Section 7 Trap 6]
- **Q57**: **D** — Sequential Latency คำนวณจาก $\sum T_i = 3 + 10 + 4 = 17$ วินาที และมีจุดอ่อนสำคัญคือ Error cascade เมื่อ agent ตัวแรกคืน error string ตัวถัดไปจะนำไปประมวลผลต่อจนผิดพลาดทั้งหมด [Section 3.4, Section 4 Section 4, Section 6 item 12]
- **Q58**: **B** — Rule-based matching ตรวจสอบตามลำดับกฎ เมื่อพบคำว่า 'premium'/'plan' จึงจับคู่กับ Sales ทันที ทำให้ส่งผิดแผนก และหาก Router ตัดสินใจผิด Specialist จะไม่สามารถแก้ไขได้ ถือเป็น Single Point of Failure [Section 4 Section 4, Section 6 item 13]
- **Q59**: **C** — State ของ ReAct ถือเป็น Markov บน Augmented State $s_t = (q, h_{1:t-1})$ และ Taxonomy 4 แบบของ Memory ได้แก่ Working, Episodic, Semantic, และ Procedural [Section 6 item 5 & 10, Section 7 Trap 5]
- **Q60**: **A** — Reliability Stack 4 ชั้นเรียงลำดับจาก Constrain $\rightarrow$ Validate $\rightarrow$ Recover $\rightarrow$ Escalate และ safeguard พื้นฐานกัน infinite loop ใน ReAct คือตัวแปร `max_steps` [Section 1 Hints, Section 6 item 4 & 14]
- **F7**: **$-\infty$** (หรือ **-inf**), **0** (หรือ **0%** / **0.0**) — การเซ็ต logit เป็น $-\infty$ ทำให้ $e^{-\infty} = 0$ ส่งผลให้ probability ของ token ที่ผิดกฎกลายเป็น 0 พอดี [Section 3.1, Section 6 item 7]
- **F8**: **27.4**, **60.9** — ผลการทดลอง Table 1 ของ ReAct บน HotpotQA ได้ 27.4 (แพ้ CoT 29.4) และบน FEVER ได้ 60.9 (ชนะ CoT 56.3) [Section 1 Hints, Section 3.6, Section 6 item 3]

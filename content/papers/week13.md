# คลัง Paper สัปดาห์ 13: Agents, ReAct, Constrained Decoding, Memory, Multi-Agent

รวมเปเปอร์ที่สไลด์ NLP-DL-13 อ้างอิงหรือแสดงไว้ทั้งหมด **14 เรื่อง** แบ่งเป็น 9 เรื่องที่สไลด์อ้างตรง ๆ (ส่วน A) และ 5 เรื่องที่ปรากฏในหน้าเปเปอร์ ReAct ซึ่งสไลด์นำมาแสดง (ส่วน B)

**วิธีตรวจสอบ:** ทุกเรื่องผมเปิดหน้า arXiv (รูปแบบ `https://arxiv.org/abs/<id>`) เมื่อ 2026-11-07 เพื่อยืนยันชื่อเรื่อง รายชื่อผู้เขียน และวันที่ส่ง ส่วน "ใจความ" มาจาก abstract ในหน้านั้นและตัวเลขที่สไลด์ระบุ ผมไม่ได้ดาวน์โหลดไฟล์ PDF และไม่ได้อ่านเนื้อเปเปอร์ฉบับเต็ม ถ้าหน้า arXiv ไม่บอกสถานที่ตีพิมพ์ ผมเขียนว่า "ตามสไลด์" หรือ "หน้า arXiv ไม่ระบุ" แทนการเดา

| # | เปเปอร์ | ปี | สไลด์ |
|---|---|---|---|
| A1 | ReAct | 2022 (arXiv), 2023 (ICLR) | 8, 9, 11, 13, 14, 15, 54 |
| A2 | Efficient Guided Generation for Large Language Models | 2023 | 19, 24, 54 |
| A3 | Generative Agents | 2023 | 27, 54 |
| A4 | Cognitive Architectures for Language Agents | 2023 (arXiv), 2024 (TMLR) | 28 |
| A5 | AutoGen | 2023 | 33, 42 |
| A6 | Improving Factuality and Reasoning ... through Multiagent Debate | 2023 | 40 |
| A7 | Identifying the Risks of LM Agents with an LM-Emulated Sandbox | 2023 | 47 |
| A8 | A Survey on Large Language Model based Autonomous Agents | 2023 | 54 |
| A9 | Levels of Autonomy for AI Agents | 2025 | 4 |
| B1 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 2022 | 9, 13, 14 |
| B2 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 2022 | 14 |
| B3 | WebGPT | 2021 | 9, 13, 14 |
| B4 | STaR | 2022 | 14 |
| B5 | HotpotQA | 2018 | 9, 13, 14, 15 |

---

## ส่วน A: เปเปอร์ที่สไลด์อ้างตรง ๆ

### ReAct: Synergizing Reasoning and Acting in Language Models

- **ผู้เขียน:** Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- **ที่ตีพิมพ์:** ICLR 2023 (หน้า arXiv ระบุว่า v3 คือฉบับ ICLR camera ready) ส่งขึ้น arXiv 6 ต.ค. 2022 แก้ล่าสุด 10 มี.ค. 2023
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2210.03629
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 8 (เปเปอร์หลัก), 9 และ 13 (หน้าเปเปอร์และ Figure 1), 11 (อ้างที่หัวสไลด์โค้ด), 14 (Table 1), 15 (Table 2), 54 (รายการอ่านเพิ่ม)

**ใจความ**

- ก่อนหน้านี้การให้เหตุผล (เช่น chain-of-thought) กับการลงมือทำ (สร้างแผน action) ถูกศึกษาแยกกัน ReAct ให้โมเดล **สลับ reasoning trace กับ action เฉพาะงาน** ใน generation เดียว
- reasoning ช่วยให้วางแผน ปรับแผน และรับมือข้อยกเว้น ส่วน action ช่วยดึงข้อมูลจากแหล่งภายนอก
- บน HotpotQA และ FEVER ใช้ Wikipedia API แบบง่าย ลด **hallucination และ error propagation** และ trace อ่านตรวจสอบได้ง่ายกว่าวิธีที่ไม่มี reasoning
- บน ALFWorld และ WebShop ชนะวิธี imitation learning และ reinforcement learning ด้วยอัตราสำเร็จสัมบูรณ์ **34% และ 10%** ใช้ตัวอย่างใน prompt เพียง 1-2 ตัวอย่าง (จาก abstract)
- ตามสไลด์ 14 (PaLM-540B, Table 1): HotpotQA EM ReAct 27.4 เทียบ CoT 29.4 และ FEVER ReAct 60.9 เทียบ CoT 56.3 ส่วนการผสม CoT-SC→ReAct ได้ 34.2 / 64.6 และ ReAct→CoT-SC ได้ 35.1 / 62.0
- ตามสไลด์ 15 (Table 2, มนุษย์วิเคราะห์ 200 trajectory): hallucination ใน failure ของ ReAct 0% เทียบ CoT 56%, reasoning error 47% เทียบ 16%, search result error 23% (CoT ไม่มี)
- ตามสไลด์ 14 เปเปอร์ใช้ trajectory 3,000 ตัวอย่างที่ ReAct ตอบถูกมา fine-tune แบบ bootstrapping (คล้าย STaR ในส่วน B4)

**เชื่อมกับข้อสอบ:** ครอบคลุมบทสัปดาห์ 13 หัวข้อ 2.1-2.7 ฮินต์ข้อ 1 (ตัวเลข Table 1) และข้อ 2 (`max_steps`) มาจากเปเปอร์นี้โดยตรง ดูตารางผลต่างในหัวข้อ 2.6 และตารางตรวจ "ตัวเลขนี้เป็นของใคร" ในหัวข้อ 2.7

### Efficient Guided Generation for Large Language Models

- **ผู้เขียน:** Brandon T. Willard, Rémi Louf
- **ที่ตีพิมพ์:** arXiv preprint (หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์) ส่ง 19 ก.ค. 2023 เวอร์ชันล่าสุด v4 วันที่ 19 ส.ค. 2023 ตามที่สไลด์อ้างคือ "Preprint"
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2307.09702
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 19 (เปเปอร์หลักของหัวข้อ structured output), 24 (ท้ายสไลด์ CFG-guided decoding), 54 (รายการอ่านเพิ่ม)

**ใจความ**

- มองการสร้างข้อความของโมเดลภาษาใหม่ว่าเป็น **การเปลี่ยนสถานะของ finite-state machine**
- จากมุมมองนี้สร้าง **ดัชนีบนคลังคำศัพท์ของโมเดล** ทำให้ควบคุมการสร้างด้วย **regular expression และ context-free grammar** ได้
- ใช้ได้กับโมเดลใดก็ได้ บังคับข้อจำกัดเฉพาะโดเมนได้ และรับประกันโครงสร้างของข้อความที่สร้าง
- ผู้เขียนรายงานว่า overhead เล็กน้อยและดีกว่าวิธีที่มีอยู่
- มีโค้ดเปิดเป็นไลบรารี Python ชื่อ **Outlines** (สไลด์ 24 อ้างไลบรารีนี้)

**เชื่อมกับข้อสอบ:** หัวข้อ 4 ทั้งหมด (logit masking −∞, regex เป็น DFA, CFG ต้องมี stack, ดัชนีล่วงหน้าช่วยให้ค้น token ที่ valid ได้เร็ว) เดโมสัปดาห์ 13 ใช้ดัชนีตามแนวคิดนี้ (`\d` 10 token กับ `-` 1 token ใช้ 2 vocabulary scans สำหรับ 10 ขั้น ดูหัวข้อ 9.5) และตัวอย่างคำนวณ 3 และ 4 ท้ายบท

### Generative Agents: Interactive Simulacra of Human Behavior

- **ผู้เขียน:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein
- **ที่ตีพิมพ์:** ACM UIST 2023 ตามสไลด์ (หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์) ส่งขึ้น arXiv 7 เม.ย. 2023 เวอร์ชันล่าสุด v2 วันที่ 6 ส.ค. 2023
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2304.03442
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 27 (ความท้าทายของ memory), 54 (รายการอ่านเพิ่ม)

**ใจความ**

- เสนอ **generative agents** คือ agent ซอฟต์แวร์ที่จำลองพฤติกรรมมนุษย์ให้น่าเชื่อ
- ขยายโมเดลภาษาด้วยสถาปัตยกรรมที่ **บันทึกประสบการณ์ของ agent เป็นภาษาธรรมชาติ** แล้วสังเคราะห์เป็น **reflection** ระดับสูงขึ้น จากนั้นดึงกลับมาใช้วางแผนพฤติกรรม (ตามสไลด์ 27 ที่เรียกที่เก็บนี้ว่า memory stream และบอกว่ามีคำอธิบาย เวลาสร้าง และเวลาเข้าถึงล่าสุด)
- ทดลองกับ agent 25 ตัวใน sandbox แบบเกม The Sims
- ตัวอย่างใน abstract: เป้าหมายเดียวที่ผู้ใช้ระบุคือจัดปาร์ตี้วาเลนไทน์ นำไปสู่การเชิญ การรู้จักคนใหม่ การชวนไปเดต และการมาร่วมงานอย่างประสานกัน
- การทำ ablation พบว่า **observation, planning และ reflection** แต่ละส่วนช่วยให้พฤติกรรมน่าเชื่อถือขึ้น

**เชื่อมกับข้อสอบ:** หัวข้อ 5.1 (ปัญหา context ไม่พอและแนวคิด memory stream) ที่เป็นพื้นของ Retrieval memory และตารางเลือก memory ในหัวข้อ 5.5

### Cognitive Architectures for Language Agents

- **ผู้เขียน:** Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, Thomas L. Griffiths
- **ที่ตีพิมพ์:** Transactions on Machine Learning Research (TMLR) 2024 (หน้า arXiv ระบุว่า v3 คือฉบับ TMLR camera ready) ส่งขึ้น arXiv 5 ก.ย. 2023 v3 วันที่ 15 มี.ค. 2024
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2309.02427
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 28 (อนุกรมวิธานของ memory)

**ใจความ**

- งานที่ต่อโมเดลภาษากับทรัพยากรภายนอก (เช่นอินเทอร์เน็ต) หรือ control flow ภายใน (เช่น prompt chaining) ก่อให้เกิด "language agents" แต่ยังไม่มีกรอบที่จัดระเบียบ
- ผู้เขียนยืมแนวคิดจากวิทยาศาสตร์การรู้คิดและ symbolic AI เสนอกรอบ **CoALA**
- CoALA อธิบาย agent ด้วย **memory แบบแยกโมดูล, action space ที่มีโครงสร้างทั้งภายใน (memory) และภายนอก (environment) และกระบวนการตัดสินใจทั่วไป**
- ใช้กรอบนี้สำรวจงานที่มีอยู่และชี้ทิศทางสำหรับ agent ที่เก่งขึ้น
- **หมายเหตุ:** abstract ไม่ได้ระบุชื่อ memory 4 ชนิด (working, episodic, semantic, procedural) ชื่อเหล่านี้ผมยืนยันได้จากสไลด์ 28 เท่านั้น ยังไม่ได้เปิดตัวเปเปอร์ฉบับเต็ม

**เชื่อมกับข้อสอบ:** หัวข้อ 5.2 (ตารางอนุกรมวิธาน memory 4 ชนิดพร้อมตัวอย่าง) ข้อเติมคำเรื่องชื่อ memory 4 ชนิดและข้อสอบที่ถามว่า "ความรู้ที่คอมไพล์แล้วคือ memory ชนิดใด" (Semantic)

### AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

- **ผู้เขียน:** Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang (14 คน)
- **ที่ตีพิมพ์:** ตามสไลด์ 33: ICLR 2024 Workshop on Large Language Model (LLM) Agents หน้า arXiv ระบุเพียงว่าเป็น preprint (43 หน้า) ส่ง 16 ส.ค. 2023 เวอร์ชันล่าสุด v2 วันที่ 3 ต.ค. 2023
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2308.08155
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 33 (เหตุผลที่ใช้ multi-agent) และสไลด์ 42 (รูป ConversableAgent จากเปเปอร์)

**ใจความ**

- เฟรมเวิร์กโอเพนซอร์สสำหรับสร้างแอปบนโมเดลภาษา โดยให้ **หลาย agent คุยกัน** เพื่อทำงานให้สำเร็จ
- agent ปรับแต่งได้ คุยกันได้ และทำงานได้ในโหมดที่ผสมโมเดลภาษา การป้อนข้อมูลของมนุษย์ และ tool
- นักพัฒนากำหนดรูปแบบการโต้ตอบได้ทั้งด้วยภาษาธรรมชาติและโค้ด (ตามสไลด์ 42 เรียกแนวคิดนี้ว่า conversation programming และรูปแสดง interface เดียวกันคือ `send`, `receive`, `generate_reply`)
- รายงานผลบนงานคณิตศาสตร์ เขียนโค้ด ถามตอบ operations research การตัดสินใจออนไลน์ และความบันเทิง

**เชื่อมกับข้อสอบ:** หัวข้อ 6.1 (ทำไมต้อง multi-agent และข้อแลกเปลี่ยน) และ 6.7 (โปรโตคอลการสื่อสาร: shared scratchpad, message passing, structured payloads รวมถึงกฎ ownership และ termination)

### Improving Factuality and Reasoning in Language Models through Multiagent Debate

- **ผู้เขียน:** Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch
- **ที่ตีพิมพ์:** ตามสไลด์ 40: ICML 2024 (Proceedings of the 41st ICML, vol. 235) หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์ ส่ง 23 พ.ค. 2023 มีเวอร์ชันเดียว
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2305.14325
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 40 (topology แบบ Debate)

**ใจความ**

- ให้หลาย instance ของโมเดล **เสนอคำตอบและเหตุผลของตนเอง** แล้ว **โต้เถียงกันหลายรอบ** จนได้คำตอบร่วมสุดท้าย
- ผู้เขียนรายงานว่าช่วยการให้เหตุผลทางคณิตศาสตร์และเชิงกลยุทธ์ในหลายงาน และเพิ่ม **ความถูกต้องของข้อเท็จจริง** พร้อมลด hallucination
- ใช้ได้กับโมเดลแบบ black box และใช้ขั้นตอนกับ prompt ชุดเดียวกันกับทุกงาน
- ผู้เขียนเรียกแนวคิดว่า "society of minds"

**เชื่อมกับข้อสอบ:** หัวข้อ 6.5 (Debate: เสนอแบบขนานอิสระ แลก argument และมี Judge) แถว Debate ในตาราง topology (Latency Medium, Complexity Medium, Fault Tolerance High, ใช้กับ fact-checking) และสูตร latency ของ Debate (max บวก judge) ในตัวอย่าง 9 ท้ายบท

### Identifying the Risks of LM Agents with an LM-Emulated Sandbox

- **ผู้เขียน:** Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris J. Maddison, Tatsunori Hashimoto
- **ที่ตีพิมพ์:** ตามสไลด์ 47: ICLR 2024 หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์ ส่ง 25 ก.ย. 2023 เวอร์ชันล่าสุด v2 วันที่ 17 พ.ค. 2024
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2309.15817
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 47 (guardrails และความปลอดภัย)

**ใจความ**

- agent ที่ใช้ tool มีความเสี่ยง เช่น **รั่วข้อมูลส่วนตัวหรือก่อความเสียหายทางการเงิน** การหาความเสี่ยงด้วยมือเสียแรงมากเพราะต้องสร้าง tool และสภาพแวดล้อมต่อสถานการณ์
- เสนอ **ToolEmu** ให้โมเดลภาษา **จำลองการทำงานของ tool** เพื่อทดสอบ agent ได้หลาย tool หลายสถานการณ์โดยไม่ต้องตั้งค่าเอง
- มีตัวประเมินความปลอดภัยอัตโนมัติที่ใช้โมเดลภาษา วิเคราะห์ความล้มเหลวและประเมินความเสี่ยง
- ผู้ประเมินที่เป็นมนุษย์พบว่า **68.8%** ของความล้มเหลวที่ ToolEmu ระบุเป็นความล้มเหลวจริงได้ในโลกจริง
- เบนช์มาร์กเริ่มต้นมี tool ความเสี่ยงสูง 36 ตัวและกรณีทดสอบ 144 กรณี และ agent ที่ปลอดภัยที่สุดที่ทดสอบก็ยังล้มเหลวแบบนี้ **23.9%** ของเวลา (ตามตัวประเมินของเปเปอร์)

**เชื่อมกับข้อสอบ:** หัวข้อ 7.3 (ตารางความเสี่ยงกับมาตรการ: least privilege, allowlist, sandbox, human confirm, prompt injection) และตัวอย่างวิเคราะห์ช่องว่าง reliability stack ในหัวข้อ 7.2 ตัวเลข 68.8%, 144 และ 23.9% เป็นข้อมูลจากเปเปอร์ ไม่ได้อยู่ในสไลด์

### A Survey on Large Language Model based Autonomous Agents

- **ผู้เขียน:** Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (13 คน)
- **ที่ตีพิมพ์:** Frontiers of Computer Science (FCS) 2024 ตามสไลด์ 54 หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์แต่ลิงก์ไปยัง DOI 10.1007/s11704-024-40231-1 (ผมไม่ได้เปิดหน้า DOI) ส่งขึ้น arXiv 22 ส.ค. 2023 เวอร์ชันล่าสุด v7 วันที่ 2 มี.ค. 2025 (35 หน้า)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2308.11432
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 54 (รายการอ่านเพิ่มข้อ 4)

**ใจความ**

- สำรวจงานวิจัยของ autonomous agent ที่สร้างบนโมเดลภาษา เหตุผลคือ agent รุ่นก่อนที่ฝึกในสภาพแวดล้อมแคบมีความรู้จำกัด ต่างจากการเรียนรู้ของมนุษย์ ส่วนโมเดลภาษาที่เรียนจากข้อมูลเว็บจำนวนมากทำให้เกิดงานวิจัยกลุ่มนี้
- **เสนอกรอบรวมสำหรับสร้าง agent** ที่ครอบคลุมงานก่อนหน้าส่วนใหญ่
- สำรวจการใช้งานในสังคมศาสตร์ วิทยาศาสตร์ธรรมชาติ และวิศวกรรม พร้อมทบทวนกลยุทธ์การประเมิน
- ปิดท้ายด้วยความท้าทายและทิศทางในอนาคต ผู้เขียนดูแลรายการอ้างอิงใน GitHub

**เชื่อมกับข้อสอบ:** อยู่ในรายการอ่านเพิ่มเท่านั้น (หัวข้อ 9.4) ไม่ใช่เนื้อหาหลักของข้อสอบ ใช้เป็นภาพรวมถ้าอยากเห็นกรอบของ agent ที่กว้างกว่าสไลด์

### Levels of Autonomy for AI Agents

- **ผู้เขียน:** K. J. Kevin Feng, David W. McDonald, Amy X. Zhang
- **ที่ตีพิมพ์:** ตีพิมพ์ในชุดบทความ "AI and Democratic Freedoms" ของ Knight First Amendment Institute (ตามหน้า arXiv) ส่งขึ้น arXiv 14 มิ.ย. 2025 เวอร์ชันล่าสุด v2 วันที่ 28 ก.ค. 2025 สไลด์อ้างว่าเป็น arXiv preprint ปี 2025
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2506.12469
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 4 (Levels of Autonomy for AI Agents)

**ใจความ**

- มองว่า autonomy เป็นดาบสองคม: ให้ประโยชน์มากแต่มีความเสี่ยงสูง
- เสนอว่า **ระดับ autonomy ควรเป็นการตัดสินใจเชิงออกแบบ** แยกจากความสามารถของ agent และสภาพแวดล้อมที่ใช้งาน
- นิยาม 5 ระดับตามบทบาทของผู้ใช้: **operator, collaborator, consultant, approver, observer** (ตรงกับ L1 ถึง L5 บนสไลด์)
- แต่ละระดับอธิบายว่าผู้ใช้ควบคุม agent อย่างไรและมีคำถามออกแบบอะไรเหลือ
- เสนอแนวคิด "AI autonomy certificates" สำหรับกำกับระบบ agent เดี่ยวและหลาย agent พร้อมแนวคิดเบื้องต้นในการประเมิน autonomy

**เชื่อมกับข้อสอบ:** หัวข้อ 1.2 (ตารางสองแถบ L1-L5 และวิธีจำ "การมีส่วนร่วมของผู้ใช้ลด ความเป็นอิสระของ agent เพิ่ม") ข้อสอบอาจถามว่าระดับไหนตรงกับบทบาทใด

---

## ส่วน B: เปเปอร์ที่ปรากฏในหน้าเปเปอร์ ReAct ที่สไลด์แสดง

เปเปอร์เหล่านี้สไลด์ไม่ได้อ้างเป็นหัวข้อของตัวเอง แต่ปรากฏในข้อความหรือตารางของเปเปอร์ ReAct ที่แสดงบนสไลด์ 9, 13, 14, 15

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- **ผู้เขียน:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- **ที่ตีพิมพ์:** หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์ ส่งขึ้น arXiv 28 ม.ค. 2022 เวอร์ชันล่าสุด v6 วันที่ 10 ม.ค. 2023
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2201.11903
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 9, 13 (ข้อความ "Wei et al." ในหน้าเปเปอร์ ReAct) และสไลด์ 14 (แถว CoT (Wei et al., 2022) ในตาราง 1: 29.4 / 56.3)

**ใจความ**

- การให้โมเดลสร้างลำดับขั้นการให้เหตุผลระหว่างทาง (chain of thought) ช่วยการให้เหตุผลที่ซับซ้อนได้มาก
- วิธีทำคือใส่ตัวอย่างที่มีขั้นการคิดไว้ใน prompt (few-shot)
- ทดสอบกับโมเดลภาษาใหญ่ 3 ตัว ได้ผลดีขึ้นบนงานเลขคณิต สามัญสำนึก และสัญลักษณ์
- ผลเด่น: prompt โมเดล 540B พารามิเตอร์ด้วยตัวอย่าง chain of thought เพียง 8 ตัวอย่างได้ความแม่นยำระดับ state of the art บน GSM8K ชนะแม้กระทั่ง GPT-3 ที่ fine-tune พร้อม verifier

**เชื่อมกับข้อสอบ:** เป็นฐานของ CoT ในตารางเปรียบเทียบ (หัวข้อ 2.5-2.6) ตัวเลข CoT 29.4 และ 56.3 ที่ใช้เทียบกับ ReAct และเนื้อหา CoT ละเอียดอยู่ในสัปดาห์ 11

### Self-Consistency Improves Chain of Thought Reasoning in Language Models

- **ผู้เขียน:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
- **ที่ตีพิมพ์:** ICLR 2023 (หน้า arXiv ระบุว่า v4 คือฉบับ camera ready ที่ ICLR 2023) ส่งขึ้น arXiv 21 มี.ค. 2022 เวอร์ชันล่าสุด v4 วันที่ 7 มี.ค. 2023
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2203.11171
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 14 (แถว CoT-SC (Wang et al., 2022a) ในตาราง 1: 33.4 / 60.4 โดยสุ่ม 21 trajectory)

**ใจความ**

- แทนที่ greedy decoding ใน chain-of-thought ด้วยการ **สุ่มเส้นทางการให้เหตุผลหลากหลายเส้น** แล้วเลือกคำตอบสุดท้ายที่ **สอดคล้องกันมากที่สุด** (เสียงข้างมาก)
- แนวคิด: ปัญหายากมักแก้ได้ด้วยเหตุผลหลายเส้นที่ลงเอยที่คำตอบถูกตัวเดียวกัน
- ตัวเลขใน abstract: ดีขึ้น **+17.9% บน GSM8K, +11.0% บน SVAMP, +12.2% บน AQuA, +6.4% บน StrategyQA และ +3.9% บน ARC-challenge**

**เชื่อมกับข้อสอบ:** อธิบายคำว่า CoT-SC ในตาราง 1 (หัวข้อ 2.6) และข้อสังเกต "ReAct ผสม CoT-SC ใช้ 3-5 samples เทียบกับ 21" เนื้อหา self-consistency ละเอียดอยู่ในสัปดาห์ 11

### WebGPT: Browser-assisted question-answering with human feedback

- **ผู้เขียน:** Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, John Schulman (18 คน)
- **ที่ตีพิมพ์:** หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์ ส่งขึ้น arXiv 17 ธ.ค. 2021 เวอร์ชันล่าสุด v3 วันที่ 1 มิ.ย. 2022 (32 หน้า)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2112.09332
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 9 และ 13 (รายการงานเดิมในหน้าเปเปอร์ ReAct: Nakano et al., 2021) และสไลด์ 14 (ข้อความที่เทียบว่าคล้ายวิธีที่ WebGPT โต้ตอบกับสภาพแวดล้อม)

**ใจความ**

- fine-tune GPT-3 ให้ตอบคำถามแบบยาวใน **สภาพแวดล้อมเบราว์เซอร์ที่เป็นข้อความ** ซึ่งให้โมเดลค้นหาและนำทางเว็บได้
- เพราะมนุษย์ทำงานเดียวกันได้ จึงฝึกด้วย imitation learning แล้วปรับคุณภาพคำตอบด้วย human feedback
- โมเดลต้อง **เก็บอ้างอิงขณะท่องเว็บ** เพื่อให้มนุษย์ตรวจข้อเท็จจริงได้
- ใช้ ELI5 (คำถามจากผู้ใช้ Reddit) ในการฝึกและประเมิน
- โมเดลที่ดีที่สุด (behavior cloning ร่วมกับ rejection sampling ต่อ reward model) ได้ **56%** ของครั้งที่มนุษย์ชอบคำตอบมากกว่าคำตอบของผู้สาธิต และ **69%** ที่ชอบมากกว่าคำตอบที่ได้โหวตสูงสุดบน Reddit

**เชื่อมกับข้อสอบ:** เป็นตัวอย่างงานก่อนหน้าที่ agent ใช้ tool ค้นเว็บ ช่วยให้เข้าใจที่มาของแนวคิด Act ใน ReAct ไม่ใช่หัวข้อสอบโดยตรง

### STaR: Bootstrapping Reasoning With Reasoning

- **ผู้เขียน:** Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
- **ที่ตีพิมพ์:** หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์ ส่งขึ้น arXiv 28 มี.ค. 2022 เวอร์ชันล่าสุด v2 วันที่ 20 พ.ค. 2022
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2203.14465
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 14 (ข้อความในหน้าเปเปอร์ ReAct ที่บอกว่าใช้แนว bootstrapping คล้าย Zelikman et al. 2022 กับ trajectory 3,000 ตัวอย่าง)

**ใจความ**

- เริ่มจากตัวอย่างเหตุผลไม่กี่ตัวอย่างกับชุดข้อมูลไม่มีป้ายกำกับขนาดใหญ่ เรียกวิธีว่า **Self-Taught Reasoner (STaR)**
- โมเดลสร้างเหตุผลกับคำตอบ ถ้าคำตอบผิด จะสร้างเหตุผลใหม่โดยรู้คำตอบที่ถูก แล้ว fine-tune บนเหตุผลที่นำไปสู่คำตอบถูก ทำซ้ำเป็นวงรอบ
- ดีกว่าโมเดลที่ fine-tune ให้ทำนายคำตอบโดยตรงในหลายชุดข้อมูล และบน CommonsenseQA ให้ผลใกล้เคียงกับโมเดลที่ใหญ่กว่า 30 เท่าที่ fine-tune
- ข้อสรุป: โมเดลพัฒนาตัวเองได้จากเหตุผลที่ตัวเองสร้าง

**เชื่อมกับข้อสอบ:** อยู่ที่หัวข้อ 2.6 ส่วนกลยุทธ์ผสมและการ fine-tune (trajectory 3,000 ตัวอย่างที่ ReAct ตอบถูก) ไม่ใช่หัวข้อสอบหลัก

### HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

- **ผู้เขียน:** Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning
- **ที่ตีพิมพ์:** EMNLP 2018 (long paper ตามหน้า arXiv) ส่งขึ้น arXiv 25 ก.ย. 2018 มีเวอร์ชันเดียว
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1809.09600
- **อยู่ในสไลด์:** สัปดาห์ 13 สไลด์ 9 และ 13 (Figure 1 ของ ReAct ใช้คำถามจาก HotpotQA) สไลด์ 14 และ 15 (คอลัมน์ HotpotQA ใน Table 1 และ 2)

**ใจความ**

- ชุดข้อมูลถามตอบ **113,000 คู่** จาก Wikipedia
- คำถามต้อง **หาและให้เหตุผลข้ามหลายเอกสาร** (multi-hop)
- คำถามหลากหลาย ไม่จำกัดกับฐานความรู้หรือ schema ที่มีอยู่แล้ว
- มี **sentence-level supporting facts** ให้ระบบเรียนรู้และอธิบายคำตอบได้
- มีคำถามชนิดเปรียบเทียบแบบใหม่ที่ทดสอบการดึงข้อเท็จจริงมาเทียบกัน
- ผู้เขียนรายงานว่าท้าทายสำหรับระบบถามตอบในขณะนั้น

**เชื่อมกับข้อสอบ:** เป็นชุดข้อมูลของคอลัมน์ HotpotQA ในตาราง 1 และ 2 (EM ReAct 27.4 เทียบ CoT 29.4) เหตุผลที่ ReAct แพ้ CoT บนชุดนี้คือเป็นงาน multi-hop reasoning ตามที่สไลด์ 14 อธิบาย ชุดข้อมูล FEVER (คอลัมน์ที่สองของตาราง) ไม่พบการอ้างอิงเปเปอร์ในข้อความสไลด์ ผมจึงไม่ได้ใส่เป็นรายการ

---

## ส่วน C: ลิงก์และเอกสารที่สไลด์ให้ไว้ซึ่งไม่ใช่เปเปอร์

ผมไม่ได้เปิดหน้าเหล่านี้ รายการนี้คัดลอกตามที่ปรากฏในข้อความสไลด์ ไม่ได้ตรวจความถูกต้องของ URL

| สไลด์ | รายการ | ที่ปรากฏในสไลด์ |
|---|---|---|
| 16 | เอกสาร function calling ของ OpenAI | https://developers.openai.com/api/docs/guides/function-calling |
| 29 | กรอบ "LLM OS" ของ Andrej Karpathy (2023) | อ้างเป็นแนวคิด ไม่มี URL ในข้อความสไลด์ |
| 38 | เอกสาร LangGraph supervisor | https://reference.langchain.com/python/langgraph-supervisor |
| 48 | Leaderboard ของ GAIA, SWE-bench และ WebArena (สไลด์เอ่ยชื่อเบนช์มาร์กโดยไม่อ้างเปเปอร์) | https://leaderboard.steel.dev/leaderboards/gaia/ , https://www.swebench.com/ , https://huggingface.co/spaces/gaia-benchmark/leaderboard , https://webarena.dev/ |

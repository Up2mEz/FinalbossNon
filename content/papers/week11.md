# คลัง paper สัปดาห์ 11: Prompting, Post-Training และ Reasoning

รวบรวมทุกเปเปอร์และแหล่งอ้างอิงที่สไลด์ NLP-DL-11 อ้างถึงหรือแสดงหน้ากระดาษไว้ (รวม 26 รายการ) ตรวจจากข้อความในสไลด์ทั้ง 56 หน้า

**วิธีตรวจ:** เปิดหน้า abstract บน arXiv ของแต่ละเปเปอร์เพื่อยืนยันชื่อเรื่อง ผู้เขียน และปีที่ส่ง (ตรวจเมื่อ 7 พ.ย. 2026) ไม่ได้ดาวน์โหลด PDF และสรุปใจความจาก abstract เท่านั้น ส่วนที่ระบุว่า "จากสไลด์" คือข้อมูลที่สไลด์เขียนไว้ ถ้าหน้า arXiv ไม่แสดงงานประชุมหรือวารสาร ผมใช้ชื่องานตามที่สไลด์อ้าง และบอกไว้ว่ามาจากสไลด์

**ตัวชี้ไปยังบทเรียน:** เลขหัวข้ออ้างถึง week11.md

---

## Part 1: Prompt Engineering

### CO-STAR: How I Won Singapore's GPT-4 Prompt Engineering Competition (บล็อก)

- **ผู้เขียน/แหล่ง:** Teo, S. (29 ธ.ค. 2023) Towards Data Science (ตามสไลด์)
- **ลิงก์:** https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41 (ยืนยันจากผลค้นหาว่า URL และชื่อบทความตรงกัน ส่วนการเปิดหน้าตรง ๆ ถูกบล็อก HTTP 403 จึงไม่ได้อ่านเนื้อหาเต็ม; ผลค้นหาระบุว่าบทความอธิบาย CO-STAR ว่ามาจากทีม Data Science & AI ของ GovTech Singapore)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 5
- **ใจความ (จากสไลด์ 5 เท่านั้น):**
  - เป็นบล็อก ไม่ใช่เปเปอร์วิชาการ
  - เป็นที่มาของกรอบ CO-STAR: Context, Objective, Style, Tone, Audience, Response format
  - Style คือลักษณะการเขียน (เช่น formal) ส่วน Tone คืออารมณ์ (เช่น empathetic)
  - Response format คือรูปแบบผลลัพธ์ เช่น JSON, Markdown, ตาราง
  - ข้อปฏิบัติร่วมในสไลด์: เจาะจง ใส่ข้อจำกัด และพูดว่า "ทำ X" แทน "อย่าทำ Y"
- **เชื่อมกับข้อสอบ:** หัวข้อ 3 อาจารย์บอกว่าสอนแค่พื้นฐาน prompt (hints: Week 11 `15-19-48`) ข้อฝึกมีทั้งจับคู่ Tone กับ Style และเติมคำ "Audience"

### Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4

- **ผู้เขียน:** Sondos Mahmoud Bsharat, Aidar Myrzakhan, Zhiqiang Shen
- **ที่มา:** arXiv ธ.ค. 2023 (ส่งครั้งแรก 26 ธ.ค. 2023; ฉบับแก้ไข 18 ม.ค. 2024) ไม่มีงานประชุมระบุบนหน้า arXiv
- **ลิงก์:** https://arxiv.org/abs/2312.16171 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 5 (ในเชิงอรรถ ใช้เป็นแหล่งของข้อปฏิบัติ "เลี่ยงประโยคปฏิเสธ")
- **ใจความ:**
  - เสนอ **26 หลักการ** เพื่อให้เขียนคำถามและ prompt ถึง LLM ได้ง่ายขึ้น
  - ต้องการอธิบายว่าการออกแบบ prompt สัมพันธ์กับขนาดโมเดลอย่างไร
  - ทดสอบกับ LLaMA-1/2 ขนาด 7B, 13B, 70B และ GPT-3.5, GPT-4
  - abstract ไม่ได้ให้ตัวเลขผลลัพธ์ มีแต่จำนวนหลักการและขนาดโมเดล
- **เชื่อมกับข้อสอบ:** หัวข้อ 3 ข้อปฏิบัติ "Do X ไม่ใช่ Don't Y" ตัวเลข 26 ไม่เห็นว่าอาจารย์เน้น

### Language Models are Few-Shot Learners (GPT-3)

- **ผู้เขียน:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal และผู้เขียนอื่น (รวม 31 คน)
- **ที่มา:** arXiv 28 พ.ค. 2020; NeurIPS 2020 (ตามสไลด์)
- **ลิงก์:** https://arxiv.org/abs/2005.14165 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 6 (และปรากฏเป็นแหล่งอ้างอิงในหน้า self-consistency สไลด์ 11)
- **ใจความ:**
  - สร้าง GPT-3 โมเดล autoregressive **175 พันล้านพารามิเตอร์** ใหญ่กว่าโมเดลแบบไม่เบาบางก่อนหน้า 10 เท่า
  - ประเมินแบบ few-shot **โดยไม่อัปเดต gradient และไม่ fine-tune**
  - ทำได้ดีในงานแปล ถามตอบ cloze และงานที่ต้องให้เหตุผลทันที เช่น เรียงตัวอักษรที่สลับ ใช้คำใหม่ในประโยค และบวกเลข 3 หลัก
  - ยังอ่อนบางชุดข้อมูล และมีปัญหาเชิงระเบียบวิธีจากคอร์ปัสเว็บขนาดใหญ่
  - ผู้ประเมินที่เป็นมนุษย์แยกข่าวที่ GPT-3 เขียนออกจากข่าวที่คนเขียนได้ยาก
- **เชื่อมกับข้อสอบ:** หัวข้อ 4 in-context learning (ICL) ไม่อัปเดต weight ซึ่งออกสอบทั้งแบบตัวเลือกและเติมคำ

### Thinking, Fast and Slow (หนังสือ)

- **ผู้เขียน:** Daniel Kahneman (2011) Farrar, Straus and Giroux (ตามสไลด์)
- **ลิงก์:** https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow (เป็นหนังสือ ไม่มีหน้า paper; ผลค้นหายืนยันว่า Farrar, Straus and Giroux พิมพ์ปี 2011 แต่ผมไม่ได้เปิดหน้านี้ และวันวางจำหน่ายที่แน่นอนต่างกันตามแหล่ง)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 7
- **ใจความ (จากสไลด์):**
  - System 1: เร็ว สัญชาตญาณ อัตโนมัติ สไลด์ว่าเป็นโหมดปกติของ LLM
  - System 2: ช้า วิเคราะห์ ทีละขั้น
  - LLM พลาดโจทย์เหตุผลเพราะทำนาย token ถัดไปทันที ไม่มี scratchpad ภายใน
  - สไลด์ 43-44 กลับมาใช้ภาพนี้อธิบายโมเดล reasoning แบบ System 2
- **เชื่อมกับข้อสอบ:** หัวข้อ 5 และ 33 ใช้เป็นกรอบแนวคิด ไม่มีสูตร

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- **ผู้เขียน:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- **ที่มา:** arXiv 28 ม.ค. 2022; NeurIPS 2022 (ตามสไลด์)
- **ลิงก์:** https://arxiv.org/abs/2201.11903 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 8 (แสดง abstract และรูปที่ 1 ทั้งหน้า)
- **ใจความ:**
  - ให้ตัวอย่างที่มี **ขั้นตอนการให้เหตุผลกลางทาง** (chain of thought) ใน prompt
  - ทดสอบกับโมเดลภาษาขนาดใหญ่ 3 ตัว บนงานเลขคณิต สามัญสำนึก และสัญลักษณ์
  - ความสามารถนี้ **เกิดขึ้นในโมเดลที่ใหญ่พอ** (emergent)
  - **PaLM 540B + 8 exemplars** ได้ระดับ state of the art บน GSM8K เหนือ GPT-3 ที่ fine-tune พร้อม verifier
  - รูปที่ 1 ของเปเปอร์คือโจทย์ Roger (ลูกเทนนิส) และโจทย์แอปเปิลที่ standard prompting ตอบ 27 แต่ CoT ตอบ 9
- **เชื่อมกับข้อสอบ:** หัวข้อ 6 key insight "compute โตตามจำนวน token ที่สร้าง" และกับดัก "CoT ใช้ได้กับทุกขนาด" (ไม่จริง)

### Efficient Memory Management for Large Language Model Serving with PagedAttention

- **ผู้เขียน:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- **ที่มา:** arXiv 12 ก.ย. 2023; SOSP 2023
- **ลิงก์:** https://arxiv.org/abs/2309.06180 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 9 (อ้างเป็นมาตรการลด memory fragmentation ตอน rollout ยาว)
- **ใจความ:**
  - KV cache ของแต่ละคำขอใหญ่และขนาดเปลี่ยนตลอด ทำให้เสียพื้นที่จาก fragmentation และข้อมูลซ้ำ จนจำกัดขนาด batch
  - เสนอ **PagedAttention** แรงบันดาลใจจาก virtual memory และ paging ของระบบปฏิบัติการ
  - สร้างระบบ **vLLM** บนแนวคิดนี้ ลดของเสียใน KV cache เหลือเกือบศูนย์ และแชร์ KV cache ภายในและข้ามคำขอได้
  - throughput เพิ่ม **2-4 เท่า** ที่ latency เท่าเดิม เมื่อเทียบกับ FasterTransformer และ Orca
  - ได้ผลมากขึ้นเมื่อลำดับยาวขึ้น โมเดลใหญ่ขึ้น และการ decode ซับซ้อนขึ้น
- **เชื่อมกับข้อสอบ:** หัวข้อ 7 ฝั่ง "วิธีแก้" ของ trace ยาว (คู่กับ MQA/GQA) รายละเอียดเต็มอยู่ใน week10.md

### Large Language Models are Zero-Shot Reasoners

- **ผู้เขียน:** Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa
- **ที่มา:** arXiv 24 พ.ค. 2022; NeurIPS 2022 (ตามหน้า arXiv "Accepted to NeurIPS2022")
- **ลิงก์:** https://arxiv.org/abs/2205.11916 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 10 (แสดงหน้าเปเปอร์)
- **ใจความ:**
  - แค่ต่อท้ายคำถามว่า **"Let's think step by step"** โดยไม่ให้ตัวอย่าง ก็ทำให้ LLM ให้เหตุผลแบบ zero-shot ได้
  - ใช้แม่แบบเดียวกับงานเลขคณิต สัญลักษณ์ และตรรกะ ได้ผลดีกว่า zero-shot ปกติมาก
  - InstructGPT (text-davinci-002): MultiArith **17.7% → 78.7%** และ GSM8K **10.4% → 40.7%**
  - PaLM 540B ได้ผลในทำนองเดียวกัน
  - ผู้เขียนเสนอเป็น baseline ที่ง่ายและแข็งสำหรับ zero-shot ก่อนจะสร้างชุด fine-tune หรือ exemplar
- **เชื่อมกับข้อสอบ:** หัวข้อ 8 สไลด์ปัดตัวเลขเป็น 17% → 78% ข้อเติมคำถามวลีและเปอร์เซ็นต์ปลายทาง (78)

### Self-Consistency Improves Chain of Thought Reasoning in Language Models

- **ผู้เขียน:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou
- **ที่มา:** arXiv 21 มี.ค. 2022; ICLR 2023
- **ลิงก์:** https://arxiv.org/abs/2203.11171 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 11 (แสดงหน้าเปเปอร์และรูปที่ 1)
- **ใจความ:**
  - แทนที่ greedy decoding ของ CoT ด้วยการ **สุ่มหลายเส้นทาง** แล้วเลือกคำตอบที่ **สอดคล้องกันที่สุด**
  - แนวคิด: โจทย์ยากมีหลายวิธีคิดที่ลงเอยที่คำตอบถูกเดียวกัน
  - ผลดีกว่า CoT ปกติ: GSM8K **+17.9%**, SVAMP **+11.0%**, AQuA **+12.2%**, StrategyQA **+6.4%**, ARC-challenge **+3.9%** (จุดร้อยละสัมบูรณ์)
  - ตามข้อความในหน้าเปเปอร์ที่สไลด์แสดง วิธีนี้ **unsupervised** ไม่ต้องมี verifier ไม่ต้อง label เพิ่ม และเป็น "self-ensemble" บนโมเดลเดียว
  - รูปที่ 1 ใช้โจทย์เป็ดของ Janet: greedy ได้ $14 (ผิด) แต่สุ่มได้ $18, $26, $18 จึงโหวตได้ $18
- **เชื่อมกับข้อสอบ:** หัวข้อ 9 ข้อสอบวนเรื่อง temperature > 0, ไม่ต้อง verifier, ต้นทุน k เท่า และนับคะแนนโหวต

### PAL: Program-aided Language Models

- **ผู้เขียน:** Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig
- **ที่มา:** arXiv 18 พ.ย. 2022; ICML 2023 (ตามสไลด์ หน้า arXiv ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2211.10435 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 12
- **ใจความ:**
  - LLM อ่านโจทย์ภาษาธรรมชาติแล้ว **เขียนโปรแกรมเป็นขั้นตอนการให้เหตุผล** จากนั้น Python interpreter รันเพื่อได้คำตอบ
  - เหตุผลที่ทำ: LLM มักแตกโจทย์ถูกแต่คำนวณพลาด
  - ทดสอบบน 13 งานเลขคณิต สัญลักษณ์ และอัลกอริทึม (จาก BIG-Bench Hard และชุดอื่น)
  - ใช้ Codex ได้ state of the art แบบ few-shot บน GSM8K เหนือ PaLM-540B + CoT **15 จุด** (top-1)
- **เชื่อมกับข้อสอบ:** หัวข้อ 10 "แยกการให้เหตุผล (LLM) ออกจากการคำนวณ (CPU)" มักถูกถามเป็นตัวเลือกจับคู่เทคนิคกับงาน

### PromptChainer: Chaining Large Language Model Prompts through Visual Programming

- **ผู้เขียน:** Tongshuang Wu, Ellen Jiang, Aaron Donsbach, Jeff Gray, Alejandra Molina, Michael Terry, Carrie J. Cai
- **ที่มา:** arXiv 13 มี.ค. 2022; CHI 2022 Extended Abstracts (หน้า arXiv ระบุ "CHI LBW 2022")
- **ลิงก์:** https://arxiv.org/abs/2203.06566 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 13-14 (สไลด์ 14 แสดงรูปที่ 2 และตารางชนิด node)
- **ใจความ:**
  - งานจริงที่ซับซ้อนมักต้องรันโมเดลหลายรอบ แล้วนำเอาต์พุตแต่ละขั้นไปเป็นอินพุตของขั้นถัดไป (chaining)
  - การต่อสายโปร่งใสและควบคุมได้กว่า แต่ยังไม่ค่อยรู้ว่าผู้ใช้เขียนสายเองอย่างไร
  - การศึกษานำร่องพบว่าต้องรองรับการแปลงเอาต์พุตกลางทางและการ debug หลายระดับ
  - จึงสร้าง **PromptChainer** อินเทอร์เฟซโปรแกรมแบบภาพ และทดลองกับผู้ใช้ 4 คน
  - ตัวอย่างในสไลด์คือ music chatbot ที่มี node ตั้งแต่ branch ความเกี่ยวกับเพลง, เรียก YouTube API ไปจนถึงตัวกรองคำตอบเป็นพิษ
- **เชื่อมกับข้อสอบ:** หัวข้อ 11 ข้อดีของ chaining (debug ง่าย ควบคุมได้) ไม่ได้เห็นอาจารย์เน้นตัวเลข

### Lost in the Middle: How Language Models Use Long Contexts

- **ผู้เขียน:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
- **ที่มา:** arXiv 6 ก.ค. 2023; TACL (หน้า arXiv ว่า accepted ปี 2023 สไลด์อ้างเป็น TACL 12, 2024)
- **ลิงก์:** https://arxiv.org/abs/2307.03172 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 13 (ใช้เทียบกับการยัด prompt มากเกินไป)
- **ใจความ:**
  - ศึกษาว่าโมเดลใช้ข้อมูลในบริบทยาวได้ดีแค่ไหน ด้วยงานถามตอบหลายเอกสารและการดึงค่า key-value
  - ความแม่นยำลดลงมากเมื่อข้อมูลที่เกี่ยวข้องย้ายตำแหน่ง
  - โมเดลทำได้ดีเมื่อข้อมูลอยู่ **ต้นหรือท้าย** ของอินพุต และแย่เมื่ออยู่ **ตรงกลาง** แม้โมเดลที่ออกแบบมาเพื่อบริบทยาว
  - เสนอโปรโตคอลประเมินใหม่สำหรับโมเดลบริบทยาว
  - abstract ไม่ให้ตัวเลข
- **เชื่อมกับข้อสอบ:** หัวข้อ 11 (สไลด์ใช้เป็นเหตุผลที่ควรแยก prompt เป็นขั้น) และสัปดาห์ 12 ที่ใช้ซ้ำเรื่องการจัดลำดับ passage (hints: Week 12-1)

### Reflexion: Language Agents with Verbal Reinforcement Learning

- **ผู้เขียน:** Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
- **ที่มา:** arXiv 20 มี.ค. 2023; NeurIPS 2023 (ตามสไลด์ หน้า arXiv ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2303.11366 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 15
- **ใจความ:**
  - ปรับปรุง agent ที่ใช้ LLM **โดยไม่อัปเดต weight**
  - agent สะท้อนผลตอบรับเป็นข้อความ (verbal reflection) แล้วเก็บใน **episodic memory** เพื่อชี้นำความพยายามรอบต่อไป
  - ผลตอบรับเป็นตัวเลขหรือข้อความอิสระก็ได้ และมาจากภายนอกหรือจำลองภายในก็ได้
  - เหนือกว่า baseline ในงานตัดสินใจ เขียนโค้ด และให้เหตุผลทางภาษา
  - HumanEval pass@1 **91%** เทียบกับ **80%** ของ GPT-4
- **เชื่อมกับข้อสอบ:** หัวข้อ 12 ลูป Generate → Critique → Refine และเหตุผลที่ได้ผล (ประเมินง่ายกว่าสร้าง)

### Tree of Thoughts: Deliberate Problem Solving with Large Language Models

- **ผู้เขียน:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan
- **ที่มา:** arXiv 17 พ.ค. 2023 (ฉบับสุดท้าย 3 ธ.ค. 2023); NeurIPS 2023
- **ลิงก์:** https://arxiv.org/abs/2305.10601 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 16
- **ใจความ:**
  - โมเดลภาษาสร้างข้อความทีละ token ซ้ายไปขวา จึงติดขัดกับงานที่ต้องสำรวจ มองล่วงหน้า หรือการตัดสินใจต้น ๆ สำคัญมาก
  - **ToT** ขยาย CoT โดยให้สำรวจหน่วยข้อความกลางทาง ชั่งน้ำหนักหลายเส้นทาง ประเมินตัวเลือกเอง มองล่วงหน้าและถอยกลับได้
  - ทดสอบกับ 3 งานที่ต้องวางแผนหรือค้นหา คือ Game of 24, Creative Writing, Mini Crosswords
  - Game of 24 ด้วย GPT-4: CoT แก้ได้ **4%** แต่ ToT แก้ได้ **74%**
- **เชื่อมกับข้อสอบ:** หัวข้อ 13 กลไก Generate → Evaluate → Prune/Search และการแยก ToT ออกจาก GoT

### Graph of Thoughts: Solving Elaborate Problems with Large Language Models

- **ผู้เขียน:** Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler
- **ที่มา:** arXiv 18 ส.ค. 2023; AAAI 2024 (หน้า arXiv ระบุ Proceedings of AAAI 2024)
- **ลิงก์:** https://arxiv.org/abs/2308.09687 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 17
- **ใจความ:**
  - แทนข้อมูลที่ LLM สร้างเป็น **กราฟโดยพลการ** โดย thought เป็นจุดยอด และความพึ่งพากันเป็นเส้นเชื่อม
  - ทำให้ **รวม** thought หลายอันเป็นผลลัพธ์ร่วม กลั่นจากเครือข่ายใหญ่ หรือปรับปรุงผ่านลูปป้อนกลับได้
  - งาน sorting: คุณภาพดีกว่า ToT **62%** และลดต้นทุนมากกว่า **31%**
  - ขยายแปลง thought แบบใหม่ ๆ ได้
- **เชื่อมกับข้อสอบ:** หัวข้อ 14 Aggregation และ Refinement (self-loop) คือจุดที่ข้อสอบใช้แยก GoT ออกจาก ToT

### Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection

- **ผู้เขียน:** Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz
- **ที่มา:** arXiv 23 ก.พ. 2023 (ฉบับแก้ไข 5 พ.ค. 2023); ACM AISec 2023 (ตามสไลด์ หน้า arXiv ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2302.12173 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 18
- **ใจความ:**
  - แอปที่ต่อ LLM ทำให้เส้นแบ่งระหว่าง **ข้อมูลกับคำสั่ง** พร่าเลือน
  - เสนอ **indirect prompt injection**: ผู้โจมตีซ่อนพรอมต์ไว้ในเนื้อหาที่ LLM มีแนวโน้มจะดึงมา จึงโจมตีจากระยะไกลได้โดยไม่ต้องเข้าถึงระบบโดยตรง
  - จัดอนุกรมวิธานความเสี่ยง เช่น การขโมยข้อมูล การแพร่แบบหนอน และการปนเปื้อนระบบข้อมูล
  - ทดสอบกับ Bing Chat (GPT-4), เครื่องมือเติมโค้ด และแอป GPT-4 สังเคราะห์
  - ผู้เขียนเตือนว่ายังขาดวิธีลดความเสี่ยงที่ได้ผลจริง
- **เชื่อมกับข้อสอบ:** หัวข้อ 15 direct เทียบ indirect และโจทย์สถานการณ์ที่ซ่อนคำสั่งไว้ในอีเมลหรือเว็บ

### Jailbroken: How Does LLM Safety Training Fail?

- **ผู้เขียน:** Alexander Wei, Nika Haghtalab, Jacob Steinhardt
- **ที่มา:** arXiv 5 ก.ค. 2023; NeurIPS 2023 (ตามสไลด์ หน้า arXiv ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2307.02483 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 19
- **ใจความ:**
  - วิเคราะห์ว่าทำไมการโจมตีแบบ jailbreak จึงสำเร็จกับโมเดลที่ผ่านการฝึกความปลอดภัย
  - เสนอสาเหตุสองแบบ: **competing objectives** (ความสามารถขัดกับเป้าหมายความปลอดภัย) และ **mismatched generalization** (การฝึกความปลอดภัยไม่ครอบคลุมโดเมนที่โมเดลยังทำได้)
  - ใช้สองแบบนี้ออกแบบการโจมตีใหม่ ทดสอบกับ GPT-4 และ Claude v1.3
  - การโจมตีใหม่สำเร็จกับทุก prompt ในชุดคำขอไม่ปลอดภัยที่ทดสอบ และเหนือกว่า jailbreak แบบ ad hoc
  - ผู้เขียนสรุปว่าลำพังการขยายขนาดไม่แก้ปัญหา
- **เชื่อมกับข้อสอบ:** หัวข้อ 16 เทคนิค roleplay, encoding (Base64), hypothetical และภาพ "การแข่งขันระหว่างแล็บกับแฮกเกอร์"

### OWASP Top 10 for LLM Applications 2025 (เว็บไซต์)

- **ผู้เขียน/แหล่ง:** OWASP Gen AI Security Project
- **ลิงก์:** https://genai.owasp.org/llm-top-10/ (ยืนยันแล้ว หน้าชื่อ "2025 Top 10 Risk & Mitigations for LLMs and Gen AI Apps")
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 20 (ตัวสไลด์เป็นรูปภาพล้วน)
- **ใจความ (รายการ 10 ข้อตามหน้าเว็บ):**
  - LLM01 Prompt Injection, LLM02 Sensitive Information Disclosure, LLM03 Supply Chain
  - LLM04 Data and Model Poisoning, LLM05 Improper Output Handling
  - LLM06 Excessive Agency, LLM07 System Prompt Leakage
  - LLM08 Vector and Embedding Weaknesses, LLM09 Misinformation, LLM10 Unbounded Consumption
  - ข้อความของสไลด์: ต้อง **ตรวจสอบต่อเนื่อง** เพราะภัยเปลี่ยนเร็ว
- **เชื่อมกับข้อสอบ:** หัวข้อ 17 ใช้เป็นภาพรวม ไม่มีสูตร LLM08 เชื่อมกับ RAG ในสัปดาห์ 12

### Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations

- **ผู้เขียน:** Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa
- **ที่มา:** arXiv 7 ธ.ค. 2023 (ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2312.06674 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 21 (เป็นตัวอย่างของ output evaluation)
- **ใจความ:**
  - โมเดลป้องกันสำหรับกรองทั้ง **อินพุตและเอาต์พุต** ของบทสนทนา
  - ใช้อนุกรมวิธานความเสี่ยงด้านความปลอดภัยเพื่อจัดประเภททั้ง prompt ของผู้ใช้และคำตอบของ LLM
  - สร้างบน **Llama2-7b** และ instruction-tune ด้วยชุดข้อมูลเล็กแต่คุณภาพสูงที่ผู้เขียนรวบรวม
  - รายงานว่าเทียบเท่าหรือดีกว่าเครื่องมือกลั่นกรองเนื้อหาเดิมบน OpenAI Moderation Evaluation และ ToxicChat
  - ปรับอนุกรมวิธานและรูปแบบเอาต์พุตได้ (zero-shot และ few-shot) และปล่อยน้ำหนักสาธารณะ
- **เชื่อมกับข้อสอบ:** หัวข้อ 18 "output evaluation = ใช้ LLM ตัวที่สองตัดสิน"

---

## Part 2-3: Alignment และ Preference Optimization

### Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

- **ผู้เขียน:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin และอีก 14 คน (รวม 20 คน)
- **ที่มา:** arXiv 4 มี.ค. 2022; NeurIPS 2022 (ตามสไลด์)
- **ลิงก์:** https://arxiv.org/abs/2203.02155 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 26 (แสดงหน้าเปเปอร์) และสไลด์ 33 (ภาพ pipeline 3 ขั้นของ RLHF)
- **ใจความ:**
  - การขยายโมเดลไม่ได้ทำให้ทำตามเจตนาผู้ใช้เอง โมเดลอาจไม่จริง เป็นพิษ หรือไม่เป็นประโยชน์
  - fine-tune GPT-3 ด้วยฟีดแบ็กมนุษย์ ขั้นแรก supervised จากตัวอย่างของผู้ติดป้าย แล้ว reinforcement learning จากการจัดอันดับผลลัพธ์ ได้ **InstructGPT**
  - **InstructGPT 1.3B ถูกเลือกมากกว่า GPT-3 175B** ทั้งที่เล็กกว่า 100 เท่า
  - ปรับปรุงความจริงและลดข้อความเป็นพิษ โดยประสิทธิภาพบนชุดข้อมูลสาธารณะเสียไปเล็กน้อย (นี่คือที่มาของ "alignment tax" ที่สไลด์ 26 ชี้)
  - หน้าที่สไลด์แสดงยังมีตัวเลข 85 ± 3% และ 71 ± 4% (InstructGPT 175B ถูกเลือกเหนือ GPT-3 แบบ prompt ปกติและ few-shot) hallucination 21% เทียบ 41%
  - หน้าเดียวกันอ้างชุดข้อมูลอื่นไว้ภายใน เช่น RealToxicityPrompts, Winogender, CrowS-Pairs, SQuAD, DROP, HellaSwag, WMT 2015 (ไม่ได้ตรวจลิงก์ของรายการเหล่านี้)
- **เชื่อมกับข้อสอบ:** หัวข้อ 22 และ 26 SFT เป็นขั้นที่ 1 RM เป็นขั้นที่ 2 PPO เป็นขั้นที่ 3 ข้อสอบถามลำดับขั้นและสิ่งที่ SFT ใช้ (next-token บน demonstration)

### A General Language Assistant as a Laboratory for Alignment (HHH)

- **ผู้เขียน:** Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan และอีก 16 คน (รวม 22 คน)
- **ที่มา:** arXiv 1 ธ.ค. 2021 (ไม่ระบุงานประชุม; มีหลายเวอร์ชัน)
- **ลิงก์:** https://arxiv.org/abs/2112.00861 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 29 (ที่มาของเกณฑ์ Helpful, Honest, Harmless โดย Anthropic 2021)
- **ใจความ:**
  - สำรวจว่าผู้ช่วยข้อความทั่วไปจะ align ให้ **เป็นประโยชน์ ซื่อสัตย์ และไม่เป็นอันตราย** ได้หรือไม่
  - baseline ง่าย ๆ อย่าง prompting ได้ประโยชน์มากขึ้นตามขนาดโมเดลและไม่ทำให้โมเดลใหญ่แย่ลง
  - เปรียบเทียบ imitation learning, binary discrimination และ ranked preference modeling
  - **ranked preference modeling ดีกว่า imitation learning มาก** ส่วน binary discrimination ทำงานและสเกลคล้าย imitation learning
  - ศึกษาขั้นก่อนฝึก preference model เพื่อเพิ่มประสิทธิภาพของข้อมูลตอน fine-tune
- **เชื่อมกับข้อสอบ:** หัวข้อ 24 ความหมายของ HHH และ alignment tension (Helpful ขัดกับ Harmless) ข้อเติมคำถามปีและตัวอักษร H ตัวที่สอง (Honest)

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- **ผู้เขียน:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **ที่มา:** arXiv 29 พ.ค. 2023 (เวอร์ชัน 3 แก้ไข 29 ก.ค. 2024); NeurIPS 2023 (ตามสไลด์)
- **ลิงก์:** https://arxiv.org/abs/2305.18290 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 30 (เชิงอรรถ) และสไลด์ 39-42 (หัวข้อ DPO)
- **ใจความ:**
  - RLHF แบบเดิมต้อง fit reward model แล้ว optimize ด้วย RL ที่ไม่เสถียรพร้อมคุมให้ใกล้โมเดลเดิม
  - DPO **จัดรูป reward ใหม่** (reparameterize) ให้ได้ policy ที่ดีที่สุดในรูปปิด ทำให้ปัญหาเหลือเป็น **loss แบบ classification ธรรมดา**
  - ไม่ต้องสุ่มจากโมเดลระหว่าง fine-tune และแทบไม่ต้องปรับ hyperparameter
  - align กับความพึงพอใจมนุษย์ได้เท่าหรือดีกว่าวิธีเดิม: ควบคุม sentiment ดีกว่า PPO-based RLHF และเท่าหรือดีกว่าใน summarization และ single-turn dialogue
  - ง่ายกว่าในการ implement และ train
  - abstract ไม่ให้ตัวเลขผลลัพธ์
- **เชื่อมกับข้อสอบ:** หัวข้อ 31-32 สูตร loss, β, 2 โมเดล, loss = ln 2 ตอนเริ่ม ซึ่งเป็นหัวข้อที่มีโจทย์คำนวณมากที่สุดของสัปดาห์

### Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback

- **ผู้เขียน:** Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma และอีก 25 คน (รวม 31 คน)
- **ที่มา:** arXiv 12 เม.ย. 2022 (ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2204.05862 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 35 (แสดงที่มาของขั้น Reward Model)
- **ใจความ:**
  - ใช้ preference modeling และ RLHF fine-tune โมเดลให้เป็นผู้ช่วยที่ helpful และ harmless
  - การฝึก alignment ปรับปรุงผลเกือบทุกการประเมิน NLP และเข้ากันได้กับการฝึกทักษะเฉพาะ เช่น เขียนโค้ด Python และสรุปความ
  - ใช้กระบวนการฝึกแบบ **online ซ้ำ ๆ** ที่อัปเดต preference model และ policy ด้วยฟีดแบ็กมนุษย์ชุดใหม่เป็นระยะ
  - พบความสัมพันธ์ **เกือบเป็นเส้นตรง** ระหว่าง reward ของ RL กับรากที่สองของ KL ระหว่าง policy กับค่าตั้งต้น
  - วิเคราะห์เพิ่มเรื่อง calibration, เป้าหมายที่ขัดกัน และ out-of-distribution detection
- **เชื่อมกับข้อสอบ:** หัวข้อ 28 ภาพรวมของ Reward Model และหัวข้อ 29 เรื่อง KL กับ reward ซึ่งสนับสนุนเหตุผลที่ต้องมี KL penalty

---

## Part 4: Test-Time Reasoning

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

- **ผู้เขียน:** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- **ที่มา:** arXiv 6 ส.ค. 2024 (ฉบับเดียว ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2408.03314 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 45
- **ใจความ:**
  - ถามว่า LLM จะดีขึ้นแค่ไหนบน prompt ยากเมื่อให้ compute ตอน inference จำนวนคงที่แต่ไม่น้อย
  - ศึกษาสองวิธีขยาย compute ตอน inference: ค้นหาคำตอบด้วย verifier แบบ process-based และปรับการแจกแจงคำตอบต่อ prompt
  - วิธีที่ดีที่สุดขึ้นกับ **ความยากของ prompt** จึงเสนอกลยุทธ์ **compute-optimal** ที่จัดสรร compute ต่อ prompt
  - ประสิทธิภาพของ test-time compute ดีกว่า best-of-N **มากกว่า 4 เท่า**
  - เมื่อเทียบที่ FLOPs เท่ากัน โมเดลเล็กที่ทำโจทย์ได้บ้างอยู่แล้วชนะโมเดลที่ใหญ่กว่า **14 เท่า**
- **เชื่อมกับข้อสอบ:** หัวข้อ 34 training-time เทียบ inference-time scaling และข้อแลกเปลี่ยนของ Thinker

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **ผู้เขียน:** DeepSeek-AI และผู้เขียนอื่นอีก 199 คน
- **ที่มา:** arXiv 22 ม.ค. 2025 (เวอร์ชัน 2 แก้ไข 4 ม.ค. 2026); วารสาร *Nature* เล่ม 645 หน้า 633-638 (2025) ตามที่หน้า arXiv ระบุ
- **ลิงก์:** https://arxiv.org/abs/2501.12948 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 46 และ 48 (R1 และ R1-Zero)
- **ใจความ:**
  - เสนอว่า LLM พัฒนาความสามารถด้านการให้เหตุผลได้ด้วย **reinforcement learning ล้วน โดยไม่ต้องมีตัวอย่างเหตุผลที่มนุษย์ติดป้าย**
  - ชี้ว่าความก้าวหน้าก่อนหน้า รวมถึง CoT พึ่งตัวอย่างที่มนุษย์เขียนจำนวนมาก และยังติดขัดกับโจทย์ยาก
  - เกิดพฤติกรรมที่โผล่ขึ้นเอง เช่น ตรวจและแก้เหตุผลของตัวเอง และปรับกลยุทธ์กลางโจทย์
  - ผลดีบนงานที่ **ตรวจคำตอบได้** ในคณิตศาสตร์ การเขียนโปรแกรมแข่งขัน และ STEM
  - ผู้เขียนบอกว่ารูปแบบการให้เหตุผลจากโมเดลใหญ่นำไปช่วยโมเดลเล็กได้
  - abstract ไม่ให้ตัวเลขผลลัพธ์
- **เชื่อมกับข้อสอบ:** หัวข้อ 35, 37, 40 ข้อสอบวน R1-Zero ไม่ใช้ SFT, rule-based reward + format reward, `<think>` มองเห็นได้เทียบ o-series ซ่อน

### Proximal Policy Optimization Algorithms

- **ผู้เขียน:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **ที่มา:** arXiv 20 ก.ค. 2017 (เวอร์ชัน 2 แก้ไข 28 ส.ค. 2017) ไม่ระบุงานประชุม
- **ลิงก์:** https://arxiv.org/abs/1707.06347 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 49
- **ใจความ:**
  - ตระกูลวิธี policy gradient ที่สลับระหว่างเก็บข้อมูลจากสภาพแวดล้อมกับ optimize ฟังก์ชัน surrogate ด้วย stochastic gradient ascent
  - ต่างจากวิธีมาตรฐานตรงที่เป้าหมายใหม่ **ให้อัปเดตแบบ minibatch ได้หลายรอบ** ต่อข้อมูลชุดเดียว
  - เรียกว่า **PPO** คงข้อดีบางอย่างของ TRPO แต่ implement ง่ายกว่า ทั่วไปกว่า และใช้ตัวอย่างอย่างมีประสิทธิภาพกว่า
  - ทดสอบกับการเคลื่อนไหวของหุ่นยนต์จำลองและเกม Atari
  - abstract ไม่ให้ตัวเลข
- **เชื่อมกับข้อสอบ:** หัวข้อ 29, 30, 38 PPO ใน RLHF ใช้ 4 โมเดล (Actor และ Critic อัปเดต, Reward และ Reference แช่แข็ง)

### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)

- **ผู้เขียน:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **ที่มา:** arXiv 5 ก.พ. 2024 (ไม่ระบุงานประชุม)
- **ลิงก์:** https://arxiv.org/abs/2402.03300 (ยืนยันแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 11 สไลด์ 50 (อ้างเป็นที่มาของ GRPO) และใช้ต่อในสไลด์ 51
- **ใจความ:**
  - DeepSeekMath 7B ฝึกต่อจาก DeepSeek-Coder-Base-v1.5 7B ด้วยข้อมูลที่เกี่ยวกับคณิตศาสตร์ **120 พันล้าน token** จาก Common Crawl บวกข้อมูลภาษาธรรมชาติและโค้ด
  - ไม่ใช้เครื่องมือภายนอกหรือการโหวต ได้ **51.7%** บน MATH ระดับแข่งขัน ใกล้เคียง Gemini-Ultra และ GPT-4
  - self-consistency 64 ตัวอย่างดันขึ้นเป็น **60.9%**
  - ผู้เขียนอ้างปัจจัยสองข้อ: ท่อเลือกข้อมูลคณิตศาสตร์จากเว็บ และ **GRPO** ซึ่งเป็นตัวแปรของ PPO ที่เพิ่มความสามารถด้านการให้เหตุผลและ **ลดการใช้หน่วยความจำของ PPO**
- **เชื่อมกับข้อสอบ:** หัวข้อ 39 GRPO ตัด Critic ใช้ค่าเฉลี่ยกลุ่มเป็น baseline และสูตร advantage ที่ใช้ในโจทย์คำนวณหลายข้อ

---

## รายการที่ปรากฏเป็นการอ้างถึงภายในหน้าที่สไลด์แสดง (ยังไม่ได้ตรวจลิงก์)

เปเปอร์ที่สไลด์แสดงทั้งหน้ามีรายการอ้างอิงของตัวเองติดมาด้วย รายการต่อไปนี้ **ไม่ได้เป็นรายการที่สไลด์อ้างโดยตรง** และผมไม่ได้เปิดตรวจ จึงไม่ใส่ลิงก์

| ปรากฏในหน้า | รายการที่อ้างถึง |
|---|---|
| สไลด์ 11 (Self-consistency) | Cobbe et al. 2021 (verifier), Thoppilan et al. 2022 (LaMDA), Chowdhery et al. 2022 (PaLM), Tay et al. 2022 (UL2) และชุดข้อมูลเช่น GSM8K, SVAMP, AQuA, StrategyQA, ARC |
| สไลด์ 26 (InstructGPT) | Gehman et al. 2020 (RealToxicityPrompts), Rudinger et al. 2018 (Winogender), Nangia et al. 2020 (CrowS-Pairs) และชุดข้อมูลเช่น SQuAD, DROP, HellaSwag, WMT 2015 |

## สรุปการตรวจ

- รวม 26 รายการ: เปเปอร์ที่มีลิงก์ arXiv 23 รายการ **ยืนยันครบ** (เปิดหน้า abstract ชื่อเรื่อง ผู้เขียน ปีตรงกับสไลด์) เว็บ OWASP 1 รายการเปิดหน้ายืนยันแล้ว ส่วนบล็อก Teo และหนังสือ Kahneman มีลิงก์ แต่ยืนยันได้ระดับ "ผลค้นหาตรงกัน" ไม่ได้เปิดอ่านหน้าจริง (Teo ถูกบล็อก 403 และ Kahneman ผมไม่ได้เปิด)
- งานประชุมของ Greshake (AISec 2023), Wei A. et al. (NeurIPS 2023), PAL (ICML 2023), Reflexion (NeurIPS 2023), Brown (NeurIPS 2020), Wei J. (NeurIPS 2022), Ouyang (NeurIPS 2022), Rafailov (NeurIPS 2023) มาจากสไลด์ เพราะหน้า arXiv ไม่ได้ระบุ
- ไม่มีรายการใดที่ชื่อ ผู้เขียน หรือปีไม่ตรงกับที่สไลด์เขียน

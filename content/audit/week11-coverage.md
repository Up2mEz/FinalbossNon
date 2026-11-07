# ตารางความครอบคลุมสัปดาห์ 11 (สไลด์ DL-11 ทีละหน้า)

วิธีตรวจ: อ่านข้อความเต็มของสไลด์ (`slides_w11.txt`, `dl11.txt` ที่รวมหน้าเปเปอร์ที่ฝัง), เดโม (`week_11_demo.ipynb`, `demo_w11.txt`) และเปิดภาพสไลด์ที่เป็นภาพเป็นหลัก (หน้า 26, 29, 32, 33, 35, 37, 40, 41, 46, 52, 54) ด้วย pypdfium2 แล้วเทียบกับ `week11.md` คอลัมน์ "where" ใช้เลขหัวข้อในไฟล์บทเรียน

**สรุป:** สไลด์ทั้งหมด 56 หน้า; ครอบคลุมสมบูรณ์ก่อนงานนี้ 48 หน้า (อีก 8 หน้าครอบคลุมเนื้อหาข้อความแล้ว แต่ขาดรายละเอียดในภาพหรือบรรทัดชื่อเรื่อง); หลังเพิ่ม 56 หน้า; ข้ามเพราะเป็นแค่กิจกรรม 0 หน้า (แล็บหน้า 56 ถูกสรุปไว้แต่ไม่ออกสอบ)

| สไลด์ | หัวข้อ | อยู่ที่ไหนใน week11.md | สถานะ |
|---|---|---|---|
| 1 | ชื่อเรื่อง Prompting, Post-Training, Reasoning และผู้สอน | ย่อหน้าเปิดไฟล์ | added now (เดิมไม่มีบรรทัดชื่อเรื่อง) |
| 2 | Agenda 5 ส่วน | ย่อหน้าเปิดไฟล์ และตารางน้ำหนักตามหน้า | covered |
| 3 | Prompt fundamentals, ChatML, 3 role, special token | หัวข้อ 1 | covered |
| 4 | Prompt engineering เป็นวิทยาศาสตร์เชิงประจักษ์ | หัวข้อ 2 | covered |
| 5 | CO-STAR 6 ช่อง และ Do X แทน Don't do Y | หัวข้อ 3 | covered |
| 6 | Zero-shot, few-shot, ICL | หัวข้อ 4 | covered |
| 7 | ขีดจำกัดของ few-shot, System 1 และ 2 | หัวข้อ 5 | covered |
| 8 | Chain-of-Thought (Wei et al.) และ key insight | หัวข้อ 6 | covered |
| 9 | ต้นทุนฮาร์ดแวร์ของ CoT, KV cache, MQA/GQA, PagedAttention | หัวข้อ 7; ประเภท 7 ในชุดฝึก | covered |
| 10 | Zero-shot CoT (Kojima et al.) | หัวข้อ 8 | covered |
| 11 | Self-consistency | หัวข้อ 9; ประเภท 6 และ 10 | covered |
| 12 | Program-of-Thought / PAL | หัวข้อ 10 | covered |
| 13 | Prompt chaining และ workflow | หัวข้อ 11 | covered |
| 14 | PromptChainer (ภาพ + ตาราง node) | หัวข้อ 11 (ตารางชนิด node) | covered |
| 15 | Reflection และ self-critique | หัวข้อ 12; ประเภท 6 | covered |
| 16 | Tree-of-Thoughts | หัวข้อ 13 | covered |
| 17 | Graph-of-Thoughts | หัวข้อ 14 | covered |
| 18 | Prompt injection (direct, indirect) | หัวข้อ 15 | covered |
| 19 | Jailbreaking | หัวข้อ 16 | covered |
| 20 | OWASP Top 10 for LLMs 2025 (ภาพล้วน) | หัวข้อ 17 (ตาราง 10 ข้อ) | covered (ตรวจภาพแล้ว) |
| 21 | กลยุทธ์ป้องกัน 4 ชั้น | หัวข้อ 18 | covered |
| 22 | จาก GPT-3 สู่ ChatGPT (ส่วนที่ 2) | หัวข้อ 19 | covered |
| 23 | Base model เป็น next-token predictor | หัวข้อ 20 | covered |
| 24 | ทำไม next-token ไม่พอสำหรับ alignment | หัวข้อ 21 | covered |
| 25 | มีม Shoggoth และหน้ายิ้ม | หัวข้อ 21 | covered |
| 26 | SFT (ภาพ Step 1 และตัวเลข InstructGPT) | หัวข้อ 22; ประเภท 11 และ 14 | covered |
| 27 | Chat template (Llama 3) | หัวข้อ 23 | covered |
| 28 | Special tokens และ stop token | หัวข้อ 23 | covered |
| 29 | เกณฑ์ HHH (ภาพเวนน์) | หัวข้อ 24 | added now (รายละเอียดแผนภาพเวนน์) |
| 30 | ข้อจำกัดของ SFT 3 ข้อ | หัวข้อ 25 | covered |
| 31 | Exposure bias (ส่วนที่ 3 เปิด) | หัวข้อ 25 | covered |
| 32 | SFT limitation และ exposure bias (ภาพ error สะสม) | หัวข้อ 25 | added now (ป้ายในภาพ) |
| 33 | RLHF 3 ขั้น (ภาพ Figure 2) | หัวข้อ 26 | added now (รายละเอียดภาพ 3 ขั้น) |
| 34 | RLHF ขั้นที่ 1: SFT | หัวข้อ 27 | covered |
| 35 | RLHF ขั้นที่ 2: Reward Model (ภาพหน้าจอผู้ติดป้าย) | หัวข้อ 28 | added now (ภาพ HH labeling) |
| 36 | แยกส่วนประกอบของ RM loss | หัวข้อ 28; ประเภท 1 | covered |
| 37 | PPO 4 โมเดล, KL penalty, วัตถุประสงค์ RLHF | หัวข้อ 29; ประเภท 3 และ 4 | covered |
| 38 | ความท้าทายของ RLHF | หัวข้อ 30 | covered |
| 39 | DPO: ใจความ | หัวข้อ 31 | covered |
| 40 | ที่มาของสูตร DPO (reward = β log ratio + β log Z) | หัวข้อ 31 | covered |
| 41 | DPO loss และบทบาทของ β | หัวข้อ 31; ประเภท 2 | covered |
| 42 | DPO เทียบ RLHF | หัวข้อ 32 | covered |
| 43 | Test-time reasoning: ข้อจำกัดของ autoregressive | หัวข้อ 33 | covered |
| 44 | System 1 เทียบ System 2 | หัวข้อ 33 | covered |
| 45 | Training-time เทียบ inference-time scaling (Snell) | หัวข้อ 34; ประเภท 14 | covered |
| 46 | Reasoning models: o-series และ R1 (ภาพ strawberry) | หัวข้อ 35 | added now (ภาพ 12 seconds) |
| 47 | สอน reasoning ด้วย SFT: ข้อจำกัด 2 ข้อ | หัวข้อ 36 | covered |
| 48 | R1-Zero: rule-based และ format reward | หัวข้อ 37 | covered |
| 49 | คอขวดของ PPO (4 โมเดล, ~4× VRAM) | หัวข้อ 38; ประเภท 8 | covered |
| 50 | GRPO: กำจัด Critic | หัวข้อ 39 | covered |
| 51 | กลไก GRPO: μ, σ, advantage | หัวข้อ 39; ประเภท 5 | covered |
| 52 | Aha moment และ emergent reasoning (ภาพตัวอย่างจากเปเปอร์) | หัวข้อ 40 | added now (ข้อความภาพ) |
| 53 | Key takeaways 4 ข้อ | หัวข้อ 41 | covered |
| 54 | ไปป์ไลน์วิวัฒนาการ (ภาพ 3 ระยะพร้อมปี) | หัวข้อ 41; ประเภท 13 | added now (ปีและป้ายลูกศร) |
| 55 | สะพานไปสัปดาห์ 12 | หัวข้อ 42 | covered |
| 56 | แล็บ 3 งาน | หัวข้อ 42 | covered (ไม่ออกสอบ) |

## เนื้อหานอก DL-11 ที่ตรวจด้วย

| รายการ | อยู่ที่ไหนใน week11.md | สถานะ |
|---|---|---|
| เดโม: chat template และ token ID, 31 token, 75 token (zero-shot) | ข้อมูลที่มีเฉพาะในเดโม + รายละเอียดเพิ่มของเดโม | covered / added now (ID ลำดับ index 0-19) |
| เดโม: เทียบ zero-shot, few-shot, CoT; ลูป generate; max_new_tokens | ข้อมูลที่มีเฉพาะในเดโม; ประเภท 12 | covered / added now |
| เดโม: base เทียบ instruct (494.0M, top-5 token) | ข้อมูลที่มีเฉพาะในเดโม | covered |
| เดโม: DPO log prob, off-by-one, loss, toy batch, sweep β, โมเดลจริง | ข้อมูลที่มีเฉพาะในเดโม; ประเภท 2 และ 12 | covered / added now (np.linspace 50 ค่า) |
| เดโม: GRPO toy 8 คำตอบ และตาราง DPO เทียบ GRPO | ข้อมูลที่มีเฉพาะในเดโม; ประเภท 5 | covered |
| ข้อต่างสไลด์กับเดโม (β, reward model ใน GRPO, baseline ของ DPO) | ตาราง "เมื่อสไลด์กับเดโมพูดไม่ตรงกัน" | added now |
| DL-10: ตัวอย่าง quantization absmax (หน้า 44) และ block-wise 64 ค่า | ภาคผนวก; ประเภท 9 | covered / added now (เหตุผลของ 127, ตัวอย่างเพิ่ม) |
| DL-10: 5 ข้อสรุปการบีบอัด (หน้า 55) และ distillation (หน้า 51-52) | ภาคผนวก (ตาราง 5 แนวคิด + คำอธิบายว่าทำไม) | covered / added now |
| PPO ratio และ clipping (ไม่อยู่ในสไลด์/เดโม) | ประเภท 4 (ระบุ ต่อยอดจากสไลด์) | added now |
| สูตรทั้งหมดและค่าคงที่พร้อมเลขหน้า | คลังสูตรและค่าที่ต้องรู้ | added now |
| ตัวหลอกและคู่ที่สับสน | คู่ที่มักสับสนและตัวเลือกลวง | added now |

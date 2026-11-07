## Answer key

ในแหล่งอ้างอิงด้านล่าง `s.N` หมายถึงหน้าสไลด์ N และ `S.N` หมายถึงเลขสไลด์ที่พิมพ์ในชุด Week 13

### Part A

- **Q1 → B** — Deletion ลบ token โดยไม่เหลือ placeholder. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.10
- **Q2 → D** — Sentence permutation สลับลำดับประโยค ส่วน rotation หมุนเอกสารจาก pivot. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.12
- **Q3 → A** — Text infilling ใช้ mask หนึ่งตัวแทน span. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.11
- **Q4 → C** — Input \(=12-5+2=9\); target \(=5+2+1+1=9\). แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.17
- **Q5 → B** — \(N^2/(0.2N)^2=1/0.04=25\); เป็นอัตราส่วนเฉพาะ decoder self-attention. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.19
- **Q6 → D** — T5 ใช้ text-to-text และ task prefix. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.15–16
- **Q7 → A** — Relative distance ถูกจัด bucket แล้วนำ learned bias ไปบวก logits. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.20
- **Q8 → C** — Cross-attention weights มีรูป `[batch, heads, target length, source length]` = `[2, 12, 20, 50]`. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.32
- **Q9 → B** — \(200(0.15)=30\); \(200(1)=200\). แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.34
- **Q10 → D** — ในตัวอย่าง demo การไม่ใส่ prefix ทำให้ได้ output ลักษณะ translation. แหล่ง: `notes/week_08_notes.md`; `text/demo_w08.txt` Demo 5
- **Q11 → A** — \(768+768+768=2{,}304\). แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.17
- **Q12 → C** — \(1{,}000(999)/2=499{,}500\) คู่. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.7–8
- **Q13 → D** — \(1{,}000\times5\text{ ms}=5{,}000\text{ ms}=5\text{ s}\). แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.8
- **Q14 → B** — \(\max(0,0.20-0.55+0.50)=0.15\). แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.27
- **Q15 → A** — ตัด positive ของ query เองออก เหลือ \(64-1=63\) negatives. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.30
- **Q16 → C** — \((2+4)/2=3\); token ที่ mask เป็น 0 ไม่ถูกนำมาเฉลี่ย. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.15
- **Q17 → B** — เมื่อ norm ของทั้งสองเวกเตอร์เป็น 1 ตัวหารในสูตร cosine เท่ากับ 1. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.10
- **Q18 → D** — Bi-encoder ดึง candidates ได้เร็ว แล้ว cross-encoder ประเมินแต่ละคู่ร่วมกัน. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.18
- **Q19 → A** — Semi-hard negative อยู่ไกลกว่า positive แต่ยังละเมิด margin. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.28
- **Q20 → C** — \((1+1/2+1/4)/3=1.75/3\approx0.5833\). แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.37
- **Q21 → D** — \(16\times10^9/2^{30}\approx14.90\) GiB. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.5
- **Q22 → B** — \(7\times10^9\times2=14\times10^9\) bytes = 14 GB. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.4
- **Q23 → C** — \(2(2)(32)(8)(128)(4096)(1)=536{,}870{,}912\) bytes = 512 MiB. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.8
- **Q24 → A** — ขนาดใหม่เป็น \(8/32=0.25\) ของเดิม จึงลด \(75\%\). แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.14
- **Q25 → D** — \(7\times10^9\times16=112\times10^9\) bytes = 112 GB. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.15
- **Q26 → B** — \(2(4096)(16)=131{,}072\). แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.41–42
- **Q27 → C** — Router เลือก experts บางส่วนต่อ token แต่ตัวอย่าง MoE ยังต้องโหลด experts ทั้งหมด. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.25–27
- **Q28 → A** — FlashAttention เป็น exact attention ที่ลด HBM traffic ด้วย tiling และ online softmax. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.19–22
- **Q29 → D** — PagedAttention จัดการ KV cache แบบ blocks และ page table. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.32
- **Q30 → B** — \(s=3.2/127\approx0.0252\); \(2.1/0.0252\approx83.33\) ปัดเป็น 83. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.44
- **Q31 → C** — Few-shot เป็น in-context learning ซึ่งใส่ตัวอย่างใน prompt โดยไม่ปรับ weights. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.6
- **Q32 → A** — คำตอบ 12 ได้ \(3/5=0.60=60\%\). แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.11
- **Q33 → D** — SFT ใช้ demonstrations และ objective แบบ next-token prediction. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.26
- **Q34 → B** — ลำดับคือ SFT → Reward Model → PPO. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.33–37
- **Q35 → C** — \(0.1[(-12+14)-(-20+18)]=0.1(2-(-2))=0.4\). แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.40–41
- **Q36 → A** — \(\mu=0.5\), population \(\sigma=0.5\), ดังนั้น \((1-0.5)/0.5=1\). แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.50–51
- **Q37 → D** — Attention เพิ่ม \(4^2=16\) เท่า และ KV cache เพิ่ม 4 เท่า. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.9
- **Q38 → B** — คำสั่งที่ซ่อนในข้อมูลภายนอกเป็น indirect injection. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.18–19
- **Q39 → C** — DPO ใช้ preference data และ reference โดยไม่ต้องมี separate reward model หรือ PPO. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.39–42
- **Q40 → A** — LLM สร้างโปรแกรมและส่งให้ interpreter รัน. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.12
- **Q41 → D** — เพดาน TF factor คือ \(k_1+1=1.5+1=2.5\). แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.16
- **Q42 → B** — \(2(2.5)/(2+1.5)=5/3.5\approx1.429\). แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.15–16
- **Q43 → C** — Length factor \(=1-0.75+0.75(0.5)=0.625\); TF factor \(=5/(2+1.5(0.625))\approx1.702\). แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.15–17
- **Q44 → A** — \(1/61+1/63\approx0.016393+0.015873=0.032266\). แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.22–23
- **Q45 → D** — Captions อื่นใน batch มี \(16-1=15\) ตัว. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.33
- **Q46 → B** — Stride \(=512-64=448\). แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.7
- **Q47 → C** — ดึงแบบ sparse และ dense แล้ว fuse ด้วย RRF ก่อน rerank และส่งให้ LLM. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.21–27
- **Q48 → A** — Global search สังเคราะห์จาก community summaries ผ่าน map-reduce. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.42–44
- **Q49 → D** — ย้าย rank 2 ไปท้าย โดยคง rank 1 ไว้ต้น: `[1, 3, 4, 5, 2]`. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.53–54
- **Q50 → B** — \(1{,}000{,}000(384)(4)=1{,}536{,}000{,}000\) bytes = 1.536 GB. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.10
- **Q51 → C** — ReAct สลับ thought, action และ observation. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.8–9
- **Q52 → A** — Parser ตรวจ call แล้ว dispatcher ส่งต่อไปยัง function ที่ระบุ. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.17
- **Q53 → D** — \(60.9-56.3=4.6\) percentage points. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.14
- **Q54 → B** — \(50{,}257-69=50{,}188\). แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.22
- **Q55 → C** — Regex บังคับ syntax ได้ แต่ `81` เป็นเดือนที่ไม่ถูกต้อง จึงต้องตรวจ semantics เพิ่ม. แหล่ง: `notes/week_13_notes.md`; `text/demo_w13.txt` Section 2
- **Q56 → A** — ขั้นตอนรันต่อกัน จึงใช้ \(3+10+4=17\) วินาที. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.34–35
- **Q57 → D** — หาก router จัดประเภทผิด คำขออาจไปยัง specialist ที่ไม่เหมาะสม. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.36–37
- **Q58 → B** — Supervisor มอบหมาย ประเมิน และอาจสั่งแก้งานจาก sub-agents. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.38–40
- **Q59 → C** — Retrieval จำกัดจำนวนรายการใน prompt ได้ แต่ storage ยังเก็บประวัติที่เพิ่มขึ้น. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.31
- **Q60 → A** — `required` บังคับให้มี key ที่ระบุ และค่าต้องผ่านชนิดข้อมูลใน schema. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.16
- **Q61 → D** — Hash ใช้จับ exact duplicates หลัง normalization; MinHash ใช้ช่วยหา near-duplicates จาก shingle sets. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.20–24
- **Q62 → B** — \(|A\cup B|=5+4-3=6\); \(J=3/6=0.5\). แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.22
- **Q63 → C** — ค่าประมาณ \(=42/64=0.65625\). แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.23–24
- **Q64 → A** — \(1-(1-0.5^2)^2=1-0.75^2=0.4375\). แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.26–27
- **Q65 → D** — PPL วัดความสามารถทำนายภายใต้ reference model; ข้อความซ้ำอาจทำนายง่ายโดยไม่มีคุณภาพสูง. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.14–15
- **Q66 → B** — A ก่อน: \(1+0.2(10)=3\) ms; B ก่อน: \(10+0.5(1)=10.5\) ms. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.30
- **Q67 → C** — \(2(0.8)(0.6)/(0.8+0.6)=0.96/1.4\approx0.6857\). แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.34–35
- **Q68 → A** — \(1-\binom82/\binom{10}2=1-28/45=17/45\approx0.3778\). แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.44
- **Q69 → D** — การสลับลำดับช่วยตรวจว่าผลเปลี่ยนตามตำแหน่งหรือไม่. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.40–41
- **Q70 → B** — Faithfulness วัดความสอดคล้องกับ context; ความถูกต้องของ context เป็นอีกประเด็นหนึ่ง. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.52–53

### Part B

- **F1 → 16; 10** — Input \(=20-6+2=16\); target \(=6+2+1+1=10\). รับ `16 tokens; 10 tokens`. แหล่ง: `notes/week_08_notes.md`; `text/slides_w08.txt` s.17
- **F2 → 0.552** — \(\max(0,0.087-0.035+0.5)=0.552\). รับ `.552`. แหล่ง: `notes/week_09_notes.md`; `text/slides_w09.txt` s.27
- **F3 → 4 GiB** — \(2(2)(32)(8)(128)(8192)(4)=4{,}294{,}967{,}296=4(2^{30})\) bytes. รับ `4` หรือ `4.0` เมื่อหน่วย GiB ระบุไว้ในโจทย์. แหล่ง: `notes/week_10_notes.md`; `text/slides_w10.txt` p.8
- **F4 → 0.5; 1** — \(\mu=(1+1+0+0)/4=0.5\), \(\sigma=0.5\), \(A=(1-0.5)/0.5=1\). รับ `0.50; +1.0`. แหล่ง: `notes/week_11_notes.md`; `text/slides_w11.txt` p.50–51
- **F5 → \(20/11\approx1.8182\); 2.5** — TF factor \(=4(2.5)/(4+1.5)=10/5.5=20/11\); เพดาน \(=k_1+1=2.5\). รับ `1.818` หรือ `1.82` สำหรับช่องแรก. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.15–16
- **F6 → \(1/61+1/62\approx0.0325225\)** — รวมคะแนนจากสองอันดับ. รับ `0.03252` หรือ `0.0325`. แหล่ง: `notes/week_12_notes.md`; `text/slides_w12.txt` s.22–23
- **F7 → 49,566; \(-\infty\)** — \(50{,}257-691=49{,}566\); \(e^{-\infty}=0\). รับ `49566; negative infinity`. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.21–22
- **F8 → 10 วินาที** — \(2+3+5=10\). รับ `10` หรือ `10.0` เมื่อหน่วยระบุไว้ในโจทย์. แหล่ง: `notes/week_13_notes.md`; `text/slides_w13.txt` S.34
- **F9 → 8; 0.5** — Union \(=7+5-4=8\); Jaccard \(=4/8=0.5\). รับ `8; 1/2`. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.22
- **F10 → \(17/45\approx0.3778\)** — \(1-\binom82/\binom{10}2=1-28/45=17/45\). รับ `0.378` หรือ `37.78%`. แหล่ง: `notes/week_14_notes.md`; `text/slides_w14.txt` p.44
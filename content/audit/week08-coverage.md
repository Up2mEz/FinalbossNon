# Coverage สัปดาห์ 8: สไลด์ DL-08 (48 หน้า) เทียบกับ week08.md

ตรวจเมื่อ 7 พฤศจิกายน 2026 แหล่งที่ใช้: `exam-prep/text/slides_w08.txt`, ข้อความเต็มของ DL-08 (รวมหน้าเปเปอร์ที่ซ้ำ), `inclass-souce/NLP-DL-08.pdf` (เรนเดอร์ภาพสไลด์ 21, 29, 32, 34, 37, 42, 44 เพื่อตรวจหน้าที่ข้อความดึงออกไม่ชัด), `exam-prep/source/week_08_demo.ipynb` และ `exam-prep/text/demo_w08.txt`

สถานะ: **covered** = มีในบทเรียนเดิมครบ, **added now** = เติมหรือแก้ในรอบนี้, **skipped** = ไม่ใส่พร้อมเหตุผล ตัวเลขหัวข้ออ้างถึง `week08.md`

## 1. ตารางรายสไลด์

| สไลด์ | หัวข้อ | อยู่ที่ไหนใน week08.md | สถานะ |
|---|---|---|---|
| 1 | หน้าปก Paradigm 3 Encoder-Decoders | ชื่อบท | skipped (หน้าปก ไม่มีเนื้อหา) |
| 2 | Recap สามตระกูล, ตาราง attention pattern, รูป BERT/GPT/BART จากเปเปอร์ | 1.1 (ตาราง + สรุปคำบรรยายใต้รูป) | covered |
| 3 | Seq2seq imperative, ข้อจำกัดของ Encoder และ Decoder | 1.1 | covered |
| 4 | Transformer ดั้งเดิม, tensor flow, CrossAttn(Q_dec, K_enc, V_enc) | 1.2, 1.4 | covered |
| 5 | Philosophy of denoising, objective, syntax/semantics/pragmatics, lineage | 2.1, 2.3 | covered |
| 6 | Formalization: P(target ∣ previous + corrupted), CrossEntropyLoss, factorization, เทียบ GPT | 2.1, 2.2 | covered |
| 7 | Corruption function zoo, analogy table | 2.4 | covered |
| 8 | กิจกรรม: เป็นโมเดล denoise (ไทย 3 ข้อ) | 2.5 | covered (กิจกรรม ใส่คำตอบตัวอย่างที่เขียนเอง และระบุว่าไม่ใช่เฉลยทางการ) |
| 9 | BART architecture: base/large, 406M, d_model 1024, 16 heads, tensor flow 50265, BPE, strategy และ key design choice | 3.1 | covered และ **added now** (เพิ่มบรรทัดกลยุทธ์ pre-train และเหตุผลที่ target ≈ N) |
| 10 | Token Masking และ Token Deletion, tensor impact | 3.3 | covered |
| 11 | Text Infilling, Poisson λ = 3, 0-length = insertion | 3.3, 3.4, ตัวอย่างคำนวณข้อ 3 และ 10 | covered |
| 12 | Sentence Permutation และ Document Rotation | 3.3 | covered |
| 13 | ผล ablation ตารางที่ 1 และสูตรสุดท้ายของ BART-large | 3.5 | covered |
| 14 | Full reconstruction: x, x̃ (src_len 4), h, y (tgt_len 9), loss, O(N²) | 3.2, 4.4 | covered |
| 15 | T5 architecture: text-to-text, sizes, d_model 768, d_ff 3072, 12 heads, SentencePiece 32,000 | 4.1 | covered |
| 16 | T5 task prefix convention, single loss, เทียบ BERT head | 4.1 | covered |
| 17 | Span corruption: 4 ขั้นตอนและตัวอย่าง src 7, tgt 7 | 4.3, 4.4 | covered |
| 18 | Demo 1 overview | 8.2 (Demo 1) | covered |
| 19 | Compute efficiency: 0.20N, 25 เท่า, 2-3 เท่า | 4.5, ข้อ 4, ชุดฝึก ประเภท D และ E | covered |
| 20 | Relative position bias: 32 bucket, สูตร a_mn, log bucketing | 4.6, ข้อ 5, ชุดฝึก ประเภท H | covered |
| 21 | BART vs T5 side-by-side | 5 | covered และ **added now** (ตรวจแถว Best At กับภาพสไลด์ ยืนยันตรงกับตาราง แก้หมายเหตุเดิมที่เขียนว่าอ่านไม่ชัด) |
| 22 | Demo 2 overview (T5 multitask) | 8.2 (Demo 2) | covered |
| 23 | Two faces of QA (extractive เทียบ generative) | 6 | covered |
| 24 | BERT extractive QA tensor mechanics | 6, ชุดฝึก ประเภท M | covered |
| 25 | T5 generative QA: Q, K, V shapes, cache, `<pad>` start | 6, 1.3 | covered |
| 26 | T5 QA autoregressive loop และ loss | 6, ชุดฝึก ประเภท J และ L | covered |
| 27 | Demos 3 และ 4 overview | 8.2 (Demo 3, Demo 4) | covered |
| 28 | กิจกรรม: ออกแบบ prefix (NER, grammar, Thai sentiment) | 4.2 | covered (กิจกรรม ใส่คำตอบตัวอย่างที่เขียนเอง) |
| 29 | Fine-tuning BART และ mBART (รูป) | 3.6 | covered และ **added now** (แก้ป้ายตัวเลข 35% และ λ = 3.5 ซึ่งไม่อยู่บนสไลด์ ให้เป็น "นอกสไลด์ จากเปเปอร์ mBART") |
| 30 | Fine-tuning T5, proportional mixing, Flan-T5 1,836 งาน | 4.2 | covered |
| 31 | Enc-Dec เทียบ Dec-only สำหรับสรุปความ: สูตรและตัวอย่าง 1,110,000N เทียบกับ 1,210,000N | 7.4, ข้อ 7, ชุดฝึก ประเภท K | covered (ชุดฝึกเพิ่มหมายเหตุว่าสูตรสมมาตร doc/sum) |
| 32 | Attention patterns visualized (รูป heatmap + shape) | 1.3, ข้อ 6, ชุดฝึก ประเภท I | covered (ตรวจกับภาพ) |
| 33 | กิจกรรม: คำนวณความยาว target (12 token) | 5 (ท้าย), ข้อ 1 | covered |
| 34 | Paradigm matrix 3 ตระกูล | 7.1 | covered (ตรวจกับภาพตาราง) |
| 35 | Why frontier models converged: 3 เหตุผล และ scaling driver >100B | 7.2 | covered |
| 36 | Reason 1: parameter duplication, Enc-Dec penalty | 7.2, ข้อ 7 | covered |
| 37 | Reason 2: RLHF/DPO, single sequence (มีรูป) | 7.2 | covered และ **added now** (เพิ่มคำอธิบายรูป RLHF เทียบ DPO) |
| 38 | Reason 3: KV cache โตเชิงเส้น, PagedAttention, RadixAttention | 7.2 | covered |
| 39 | Scaling argument: Wang et al., ไม่มีการเทียบเกิน 100B, T5 → PaLM → Gemini | 7.3, ข้อ 7 | covered |
| 40 | Where Enc-Dec still win: NLLB, Whisper, สรุปความ | 7.5 | covered |
| 41 | Decoder-only paradox summary, meta-lesson | 7.2, 7.5 | covered |
| 42 | UL2: mode token R, S, X (รูป) | 7.6 | covered |
| 43 | กิจกรรมอภิปราย: Team Enc-Dec, Dec-only, Hybrid | 7.8 | **added now** (เดิมไม่มี) |
| 44 | Timeline และ evolutionary tree (รูป), DeepSeek MLA | 7.7 | covered และ **added now** (เพิ่มสีกิ่งของแต่ละตระกูลและข้อความจากเปเปอร์ที่ฝังในสไลด์) |
| 45 | Descendant models: mT5, Flan-T5, CodeT5+, mBART, PEGASUS | 4.7 | covered |
| 46 | Lab preview | 8.1 | covered (Lab ไม่ออกสอบ อ่านเพื่อเข้าใจ) |
| 47 | Key takeaways 5 ข้อ | 7.8 และ ## สรุปสัปดาห์ 8 | **added now** (เดิมกระจายในสรุป ไม่ได้ระบุว่าเป็น 5 ข้อของอาจารย์) |
| 48 | Bridge ไปสัปดาห์ 9 (anisotropy, SBERT) | ท้ายบท "สะพานไปสัปดาห์ 9" | covered |

## 2. เดโม (`week_08_demo.ipynb`) เทียบกับ 8.2

| เดโม | เนื้อหา | อยู่ที่ไหน | สถานะ |
|---|---|---|---|
| Demo 1 | จำลอง corruption: BART infilling กับ T5 span (seed 6, src/tgt, อัตรา 1.50) | 8.2 Demo 1, ชุดฝึก ประเภท A, B, C | covered |
| Demo 2 | T5-small text-to-text: translate, summarize, QA, sst2, cola (ผิด), `legacy=False`, mps | 8.2 Demo 2 | covered |
| Demo 3 | heatmap shapes (8, 8), (5, 5), (5, 8) และวิธีอ่านแกน | 8.2 Demo 3, ชุดฝึก ประเภท I | covered |
| Demo 4 | BERT start 11 end 21, T5 ตอบ heavy rain, zero-context | 8.2 Demo 4, ชุดฝึก ประเภท M | covered |
| Demo 5 | prefix เป็น task switch, T5 ไม่มี prefix fallback ไปแปลเป็นเยอรมัน, BART distilbart ไม่มี prefix, ค่า generate | 8.2 Demo 5 | covered |

## 3. สรุปตัวเลข

- จำนวนสไลด์ในเด็ค: 48
- ก่อนรอบนี้: covered เต็ม 40 หน้า, มีช่องว่างหรือจุดบกพร่อง 7 หน้า (9, 21, 29, 37, 43, 44, 47), skipped 1 หน้า (หน้าปก)
- หลังรอบนี้: covered 47 หน้า, skipped 1 หน้า (หน้าปก)
- สิ่งที่เพิ่มนอกเหนือจาก coverage: ## คลังสูตรและค่าที่ต้องรู้, ## ชุดฝึกคำนวณเชิงลึก, ## คู่ที่มักสับสนและตัวเลือกลวง (วางก่อน ## สรุปสัปดาห์ 8)

## 4. จุดที่สไลด์กับแหล่งอื่นไม่ตรงกัน หรือยังตัดสินไม่ได้

| ประเด็น | สิ่งที่พบ | วิธีเขียนในบทเรียน |
|---|---|---|
| สไลด์ 9: ขนาด BART-large | สไลด์เขียน ~406M ในบรรทัดสถาปัตยกรรม และ ~400M ในบรรทัด Params | ระบุทั้งสองค่า ข้อสอบน่าใช้ ~400M หรือ 406M ตามโจทย์ |
| สไลด์ 17 เทียบกับเดโมเรื่องการนับ `</s>` | สไลด์: src ไม่นับ `</s>`, tgt นับ sentinel ปิดท้ายและ `</s>`; เดโม: นับ `</s>` ทั้งสองฝั่ง และ BART นับ `<s>` `</s>` ทั้งสองฝั่ง | ระบุทั้งสองแบบ และให้ตอบตามที่โจทย์กำหนด |
| สไลด์ 31 เทียบกับสูตรของสไลด์เอง | ผลต่าง Enc-Dec กับ Dec-only เท่ากับ N·L_doc·L_sum เสมอ และสัดส่วนสูงสุดเมื่อ doc = sum (1.33) ไม่ใช่เมื่ออินพุตยาวกว่ามาก (doc 4000 / sum 100 ได้ 1.02) ขัดกับถ้อยคำ "long input short output" แบบเชิงสัดส่วน | ชุดฝึก ประเภท K หมายเหตุ: ให้ตอบตาม verdict ของสไลด์ ส่วนการคำนวณใช้สูตรตรง ๆ |
| สไลด์ 20: ตัวหาร sqrt(d_k) | สูตรบนสไลด์มี แต่หมายเหตุบนสไลด์บอกว่าโค้ดจริงของ T5 ตัดออก | ระบุทั้งสอง และให้ใช้สูตรสไลด์เมื่อโจทย์ให้ d_k |
| สไลด์ 19: 0.20N | 15% + 5% เป็นค่าประมาณ จำนวนที่นับจริง (r + k + 2) ต่างเล็กน้อย เช่น 42 เทียบกับ 40 ที่ N = 200 | ชุดฝึก ประเภท A ข้อ A4 อธิบายความต่าง |
| แถว "Best At" ของสไลด์ 21 | ไฟล์ข้อความอ่านไม่ชัด | ตรวจกับภาพสไลด์แล้ว ยืนยันว่า BART = Abstractive Summarization และ Text Generation, T5 = Multi-task, Translation และ QA |
| ตัวเลข mBART 35% และ λ = 3.5 | ไม่อยู่บนสไลด์ 29 (อยู่ในเปเปอร์) | ติดป้าย นอกสไลด์ |
| ค่า N = 6 และ d_model = 512 ของ Transformer ดั้งเดิม | สไลด์ 4 มีเฉพาะรูปจากเปเปอร์ ไม่มีตัวเลขเป็นตัวอักษร | ติดป้าย จากเปเปอร์ Vaswani ในคลังสูตร |
| 80/10/10 ของ MLM และสไลด์ DL-07 (LayerNorm, RMSNorm, ตัวเลข 7-64% เทียบกับ 7-16%) | ไม่อยู่ใน DL-08 และไม่มีข้อความสไลด์ DL-07 ให้ตรวจซ้ำในรอบนี้ (ยกจากบทเรียนเดิม) | คงเดิมตามที่เขียนไว้ และติดป้ายที่มา |

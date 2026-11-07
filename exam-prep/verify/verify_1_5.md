# ตรวจตัวเลขจากสไลด์ (ดูภาพหน้าสไลด์ที่ render จริง) - รายการ 1-5

หมายเหตุ: PDF page index ตรงกับเลขสไลด์ที่พิมพ์บนหน้า (W9 = NLP-DL-09.pdf, W8 = NLP-DL-08.pdf)

## #1 W9 s.8 (+ s.7): bi-encoder vs cross-encoder, n = 10,000
- PDF page: W9 หน้า 8 (bi-encoder) และหน้า 7 (cross-encoder)
- สไลด์ 8 เขียนว่า: "For n = 10,000: ~50 seconds for encoding + <1 second for all comparisons."
- สไลด์ 7 เขียนว่า: n(n-1)/2 = 10,000 x 9,999 / 2 ≈ 50 million forward passes; "At 5ms per inference: ~69 hours on a single GPU. Impractical."
- ตัวเลข 49,995,000 pairs: สไลด์เขียนเป็น "≈ 50 million" เท่านั้น (49,995,000 เป็นค่าที่คำนวณเอง ไม่ได้อยู่บนสไลด์)
- Verdict: **MATCH** (bi-encoder ~50 s ไม่ใช่ ~5 s; cross-encoder ~69 h ที่ 5 ms/pass)

## #2 W9 s.39: ตาราง ANN
- PDF page: W9 หน้า 39
- ตารางบนสไลด์ (Index / Method / Query Time / Recall / Best For):
  - Flat: Brute-force (exact) / O(n·d) / 100% / Small corpus (<100K)
  - IVF: Cluster partitioning / ≈ O(√n · d) / ~95% / Large corpus, low memory
  - HNSW: Graph-based / ≈ O(log n · d) / ~99% / Most production systems
  - IVF+PQ: Cluster + compression / "Very fast, low RAM" (ไม่มีสูตร O) / ~90% / Billion-scale datasets
- ข้อความรอบตาราง: ANN "high recall (>95%)", HNSW sublinear ~O(log n · d)
- Verdict: **MATCH** (recall 100 / ~95 / ~99 / ~90 ตรง) ข้อสังเกต: query time ของ IVF = ≈O(√n·d), IVF+PQ ระบุเป็นคำว่า "Very fast, low RAM" ไม่ใช่สูตร

## #3 W9 s.4, 19, 24: คะแนน STS
- PDF page: W9 หน้า 4, 19, 24
- สไลด์ 4 (ตาราง "Method / STS Benchmark", อ้าง Reimers & Gurevych 2019):
  - GloVe average = 58.02
  - BERT [CLS] (no fine-tuning) = 16.50
  - BERT mean pooling (no fine-tuning) = 46.35
  - SBERT-NLI-base = 77.03
- สไลด์ 19: "In Numbers (STS Benchmark, Spearman ρ): GloVe 58.2 → BERT mean-pool 46.4 → SBERT 77.03" (ปัดเป็นทศนิยม 1 ตำแหน่งสำหรับ 2 ค่าแรก; benchmark = STS Benchmark, Spearman ρ)
- สไลด์ 24 (SimCSE): "Achieved ~76.3 on STS ... already above SBERT-base's 74.89"; Supervised SimCSE "~81.6, beating SBERT-base's 74.89" สไลด์ไม่ระบุ benchmark ของ 74.89 (บอกแค่ "on STS")
- ประเด็นภายในสไลด์เอง: SBERT-base ถูกให้ 77.03 (s.4, s.19) แต่ 74.89 (s.24) ไม่สอดคล้องกัน สไลด์ไม่ได้อธิบาย (น่าจะเป็นคนละ benchmark/คนละการวัด) ข้อสอบควรยึดตามบริบทของสไลด์ที่ถาม
- Verdict: **CORRECTION (เล็กน้อยด้านการระบุโมเดล)**: 58.02 = GloVe average, 16.50 = BERT [CLS], 46.35 = BERT mean pooling, 77.03 = **SBERT-NLI-base** (ไม่ใช่ "SBERT-base" เฉยๆ) ทั้งหมดเป็น STS Benchmark; "SBERT-base 74.89" อยู่เฉพาะในสไลด์ SimCSE (s.24) ต้องจำแยกจาก 77.03

## #4 W8 s.9, 21: พารามิเตอร์ BART
- PDF page: W8 หน้า 9 และ 21
- สไลด์ 9: "BART-base (6 Enc + 6 Dec, ~140M) | BART-large (12 Enc + 12 Dec, ~406M, d_model=1024, 16 heads)" และบรรทัดถัดไป "Params: ~400M (BART-large). d_model = 1024, n_heads = 16."
- สไลด์ 21 (ตารางสรุป): "Model Sizes: 140M (base), 400M (large)"
- Verdict: **MATCH** (base ~140M; large ~406M บนสไลด์ 9 และปัดเป็น ~400M ในบรรทัด Params ของสไลด์ 9 และตารางสไลด์ 21)

## #5 W8 s.14, 17, 19, 21, 33: การนับ target token BART vs T5
- PDF page: W8 หน้า 14 (BART), 17, 19, 21, 33
- BART (สไลด์ 14): Decoder Target = ["<s>","The","cat","sat","on","the","mat","happily","</s>"], tgt_len = 9 -> นับทั้ง <s> และ </s>; (ต้นฉบับ 7 คำ + 2 = 9) สไลด์ 17 ระบุ "BART 10" สำหรับประโยค 8 คำ (8 + <s> + </s> = 10 สอดคล้องกัน)
- T5 (สไลด์ 17): Original "The cute cat sat on the fuzzy mat" (8 tokens); Input "The <extra_id_0> cat sat on the <extra_id_1>" src_len = 7 (ไม่มี </s> ในการนับ src); Target "<extra_id_0> cute <extra_id_1> fuzzy mat <extra_id_2> </s>" tgt_len = 7 -> **นับ </s> และ sentinel ตัวสุดท้าย <extra_id_2>** ; ไม่มี <s> นำหน้า
- สไลด์ 19: "Target: ≈ 0.20N tokens (15% corrupted + 5% sentinels)" ใช้เครื่องหมาย ≈ จริง; speedup ≈ 25x (1/0.2)^2 ; overall ≈ 2-3x
- สไลด์ 21: "Target Length: ≈N (full length) | ≈0.20N (compact)" ใช้ ≈ เช่นกัน
- สไลด์ 33 (แบบฝึกหัด 12 tokens "The quick brown fox jumps over the lazy dog in the park", span "quick brown" และ "lazy dog"): มีเฉพาะโจทย์ Q1-Q5 **ไม่มีเฉลยบนสไลด์** (สไลด์ 34 เป็นตาราง Paradigm Matrix)
- เฉลยที่คำนวณตามแบบแผนของสไลด์ 14/17 (ไม่ใช่ค่าจากสไลด์): Q1 BART src = 12 - 4 + 2 <mask> = 10; Q2 BART tgt = 12 + <s> + </s> = 14; Q3 T5 src = 10 (สอง span ถูกแทนด้วย sentinel 2 ตัว, ไม่นับ </s>); Q4 T5 tgt = <extra_id_0> quick brown <extra_id_1> lazy dog <extra_id_2> </s> = 8; Q5 = 14/8 = 1.75 (ถ้า BART ไม่นับ <s>,</s> จะเป็น 12/8 = 1.5)
- Verdict: **MATCH** สำหรับการนับ (BART นับ <s> และ </s>; T5 target นับ </s> และ sentinel ตัวสุดท้าย; 0.2N ระบุเป็นค่าประมาณ ≈ ทุกที่) ส่วนเฉลยแบบฝึกหัดสไลด์ 33 = **ไม่มีบนสไลด์ (UNREADABLE ในแง่ไม่มีคำตอบ)** ค่าที่ให้ด้านบนเป็นการคำนวณเอง

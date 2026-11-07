# ตารางครอบคลุมสไลด์สัปดาห์ 14 (NLP-DL-14) เทียบกับ week14.md

ตรวจจากข้อความสไลด์ (`exam-prep/text/slides_w14.txt` และ `dl14.txt`), เดโม (`week14_demo.ipynb`, `demo_w14.txt`) และภาพสไลด์ของหน้าที่เป็นภาพ ตาราง หรือกราฟเป็นหลัก (สไลด์ 4, 6, 8, 9, 10, 13, 15, 17, 18, 19, 20, 22, 24, 28, 30, 35, 36, 38, 40, 43, 44, 47, 53 เรนเดอร์จากไฟล์ PDF ด้วย python แล้วอ่านตัวเลขเทียบ)

สถานะ: covered = มีอยู่ก่อนแล้ว, added now = เติมครั้งนี้, skipped = ไม่ใส่พร้อมเหตุผล

| สไลด์ | หัวข้อ | อยู่ที่ไหนใน week14.md | สถานะ |
|---|---|---|---|
| 1 | ชื่อเรื่องและชื่ออาจารย์ | – | skipped: หน้าปก ไม่มีเนื้อหาสอบ |
| 2 | Recap สัปดาห์ 13 และ agenda สองภาค | ย่อหน้าต้นบท, §1 | covered |
| 3 | บทบาทใหม่: Source, Filter, Deduplicate, Evaluate, Audit; อุปมาสูตรอาหาร | §1 | covered |
| 4 | Scaling law L(N, D), Chinchilla 20 token/พารามิเตอร์, 175B → 3.5T, ข้อควรระวัง | §2; ชุดฝึกประเภท 2 | covered |
| 5 | Common Crawl, WARC/WET, S3, warcio | §3 | covered |
| 6 | Data Contraction Funnel ตาราง 6 ขั้น, r_i เทียบอัตราสะสม | §4; ชุดฝึกประเภท 1 | covered |
| 7 | fastText lid.176, 176 ภาษา, threshold 0.65, < 50 ตัวอักษร | §5; ประเภท 4 | covered |
| 8 | Script penalty, 3 ไบต์/ตัวอักษรไทย, ตาราง GPT-2 71/10 และ 57/114 | §6; ประเภท 5 | covered |
| 9 | TPC, ρ, ตาราง GPT-2 กับ BERT, Llama 3 128k | §6; ประเภท 5 | covered |
| 10 | Pipeline Tier 1/Tier 2 (0.1 ms, 10 ms), ตำแหน่ง dedup, กฎ c_A + p_A c_B, สถาปัตยกรรม FineWeb | §7; ประเภท 6 | covered |
| 11 | กฎ Gopher ทั้งเจ็ดกลุ่ม และข้อควรระวังภาษาไทย | §8; ประเภท 3 | covered |
| 12 | กฎ C4 (บรรทัด หน้า dedup ภาษา) | §9 | covered |
| 13 | ตัวอย่างเอกสารผ่านด่าน (alpha 0.58 ตก) | §10 | covered |
| 14 | CCNet: สูตร PPL, 3 ถัง, KenLM, ตัดออก 84% | §11; ประเภท 7, 8 | covered |
| 15 | กับดัก perplexity ภาพ 2×2 (correct keep, false negative, false positive, correctly deleted), Kreutzer, Gururangan | §11 | added now: ตาราง 2×2 ของภาพ พร้อมเหตุผลและความต่าง Doc 5 PPL สไลด์ ≈ 4.0 กับเดโม 4.1 |
| 16 | Neural classifier: คะแนน 0-5, Llama-3-70B, ~460k, ~110M, threshold ≥ 3, 1.3T/15T | §12 | covered |
| 17 | FineWeb และ FineWeb-Edu: 15T, 96 snapshot, ablation, MMLU 33→37, ARC 46→57, กราฟ ablation | §12; ประเภท 25 | added now: กราฟ ablation 4 เส้น (FineWeb, C4 Filters, Individual minhash, Base filtering) และเหตุผลของ ablation |
| 18 | Dolma: ~3T, ตาราง Table 1, toolkit, OLMo | §13; ประเภท 25 | added now: ตารางเต็ม 4 คอลัมน์ (GB, เอกสาร, คำ, token) และข้อสังเกตว่าคอลัมน์ไบต์บวกได้ 11,498.6 ไม่ตรง 11,519 |
| 19 | FineWeb เทียบ Dolma ตารางเปรียบเทียบ | §13 | covered |
| 20 | Dedup: ความเสี่ยง memorization/bias/compute, exact/near, กราฟขนาดกลุ่มสำเนาใน C4 | §14; ประเภท 26 | added now: ตัวเลขกราฟ Lee et al. ทั้ง 11 แท่ง สัดส่วน 99.29% และหางยาว 280 กลุ่ม |
| 21 | SHA-256: 256 บิต, 32 ไบต์, 64 hex, ความซับซ้อน, โอกาสชน, 1.28 TB | §14; ประเภท 9 | covered |
| 22 | Shingling, N − k + 1, k เท่าของเอกสาร, Jaccard, ทำไมไม่คำนวณตรง ๆ, ตัวอย่าง "The cat sat on the mat" | §15; ประเภท 10, 11 | added now: ความหมายแบบเซต (หน้าต่างซ้ำรวมกัน) พร้อมตัวอย่าง k = 1, 2 และ Jaccard ตามค่า k จากเดโม |
| 23 | MinHash: sig, x = md5 mod p, h = (a x + b) mod p, P[match] = J, J_hat, SE, ต้นทุน | §15; ประเภท 12 | covered |
| 24 | ภาพ MinHash: sliding window, 64 กฎ, 42/64 = 0.656, "kept as a SET" | §15; ประเภท 12 | added now: หมายเหตุเซต ไม่ใช่ลิสต์ (repeats collapse) ส่วนที่เหลือ covered |
| 25 | SimHash: v_t, TF-IDF, E[d_H/d] = θ/π, d = 64, Manku d_H ≤ 3 | §16; ประเภท 15 | covered |
| 26 | LSH banding: O(N²) 8 × 10^20, H = b × r, P = 1 − (1 − s^r)^b, s*, ตัวอย่าง 64/16/4 และ 256/64/4 | §17; ประเภท 13 | covered |
| 27 | ภาพตู้เอกสาร: 16 แบนด์, 0.185, 0.96, N × b, ค่า 2 หลักเป็นตัวแทน | §17; ประเภท 13 | added now: หมายเหตุว่าตัวเลข 2 หลักเป็นตัวแทนของจำนวนเต็ม 61 บิต และชื่อชั้น = (แบนด์, ค่า) |
| 28 | ต้นทุน dedup สองแกน: ตาราง s², H, N·b; N² s² ไม่ใช่ N³; 17,689 → 64 | §18; ประเภท 14 | covered |
| 29 | ตัวอย่าง dedup เต็ม: 0.662, 0.656, SE 0.059, LSH 66 คู่, FineWeb 14 × 8 → 11% | §19; ประเภท 11, 12, 13 | covered |
| 30 | กิจกรรมออกแบบ filter funnel: ตาราง 6 ตัวกรอง และคำถาม 3 ข้อ | §20; ประเภท 6 | covered (กิจกรรมถามในห้อง บทเรียนให้แนวตอบพร้อมตัวเลข) |
| 31 | สรุป Part 1: ลำดับ pipeline, 3 บทเรียน, lab pointer | §20 | covered |
| 32 | Part 2: สามยุคของการประเมิน (Era 1, 2, 3) | §21 | covered |
| 33 | BLEU (สูตร, BP), ROUGE-L, METEOR/chrF, ตัวอย่างเดโม 0.17/0.83 | §22; ประเภท 16, 17 | covered |
| 34 | BERTScore: embedding C, R, ความคล้าย S_ij | §23; ประเภท 18 | covered |
| 35 | BERTScore: greedy P/R/F, IDF, rescale, ภาพเมทริกซ์ "the weather is cold today" | §23; ประเภท 18 | added now: เมทริกซ์ 5 × 4 และ idf จากภาพ พร้อม P, R, F, R ถ่วง IDF และตัวอย่าง rescale |
| 36 | BERTScore ตัวอย่าง paraphrase เทียบ word salad (0.79/0.67) | §23 | covered |
| 37 | Chatbot Arena: blind pairwise, Elo/Bradley-Terry, จุดแข็งจุดอ่อน | §24; ประเภท 19 | covered (สูตร Elo/BT นอกสไลด์ ติดป้ายต่อยอด) |
| 38 | MT-Bench: 80 ข้อ 8 หมวด, pairwise/single, judge prompt template, ภาพตัวอย่างคำตอบ | §25 | added now: คำอธิบายภาพตัวอย่าง (คำถาม Federal Reserve สองเทิร์น เหตุผลของ GPT-4) |
| 39 | Length bias: 91.3/91.3/8.7% (23 คำตอบ) | §26; ประเภท 20 | covered |
| 40 | Positional bias: ตาราง Table 2, swap trick, few-shot 65.0 → 77.5, inconsistent | §26; ประเภท 20 | covered |
| 41 | วิธีลด bias 5 ข้อ | §26 | covered |
| 42 | MMLU: 57 วิชา, 15,908, 14,079, 5-shot, template รายงาน | §27; ประเภท 23 | covered |
| 43 | GSM8K: 8,500, ####, CoT, ตารางคะแนน, อิ่มตัว | §27 | covered |
| 44 | HumanEval และ pass@k: 164, 7.7, สูตร, n = 200 c = 2, milestones, ภาพโจทย์และสคริปต์ | §27; ประเภท 21, 23 | added now: สคริปต์เสถียร (รูปคูณ) และ Figure 2 (0.9, 0.17, 0.005) |
| 45 | Saturation และ contamination: overlap, perplexity ต่ำ, protection, reporting, อ้างอิง Golchin | §28; ประเภท 22 | added now: แหล่งอ้างอิง Golchin & Surdeanu และเชื่อมกับ dedup |
| 46 | ความล้มเหลว 3 แบบ และหลักออกแบบ benchmark สมัยใหม่ 4 ข้อ | §29 | covered |
| 47 | ตารางชุดเสริมกัน: ใครครอบคลุมอะไร | §29 | covered |
| 48 | LiveBench: รายเดือน 6 โดเมน | §29 | covered |
| 49 | AIME: 15 ข้อ, 000-999, 6.7/13.3 จุด | §29; ประเภท 23 | covered |
| 50 | SWE-bench: resolved, Verified/Lite, กฎเทียบ | §29; ประเภท 23 | covered |
| 51 | HarmBench: 510 = 400 + 110, 7 หมวด, 18 วิธี, 33 โมเดล, ASR | §29; ประเภท 23 | covered |
| 52 | RAGAS และ DeepEval: CI, เมตริกของ Es et al. กับของไลบรารีรุ่นหลัง | §30 | covered |
| 53 | RAGAS Metrics Math: faithfulness 0.75, answer relevance, context recall 0.60, precision 0.92 | §30; ประเภท 24 | covered |
| 54 | กิจกรรมตรวจหาข้อบกพร่อง Exhibit A, B, C | §31; ประเภท 22 | covered |
| 55 | Key takeaways 5 ข้อ | §32 | covered |
| 56 | Good luck | – | skipped: หน้าปิด ไม่มีเนื้อหาสอบ |

## สรุปจำนวน

| รายการ | จำนวน |
|---|---|
| สไลด์ในเด็ค | 56 |
| สไลด์ที่ไม่มีเนื้อหาสอบ (skipped) | 2 (สไลด์ 1 และ 56) |
| สไลด์เนื้อหาที่ตรวจ | 54 |
| ครอบคลุมครบก่อนแก้ (covered) | 43 |
| เติมครั้งนี้ (added now) | 11 (สไลด์ 15, 17, 18, 20, 22, 24, 27, 35, 38, 44, 45) |
| ครอบคลุมครบหลังแก้ | 54 จาก 54 |

ส่วนที่เติมเป็นภาพ กราฟ หรือตารางที่ข้อความสไลด์ไม่ได้ให้ตัวเลข (สไลด์ 15, 17, 18, 20, 35, 38, 44) และรายละเอียดย่อยของสไลด์ 22, 24, 27, 45 ส่วนเนื้อหาเฉพาะเดโมอยู่ครบแล้วที่ §34 ของ week14.md

## จุดที่สไลด์กับเดโมต่างกัน หรือตัวเลขที่ตรวจแล้วไม่ตรง

- C4: สไลด์ 12 ใช้ ≥ 3 คำต่อบรรทัด และ ≥ 5 ประโยคต่อหน้า; เดโมปรับเป็น ≥ 5 คำ และ ≥ 3 ประโยคให้เหมาะกับ corpus 12 เอกสาร ข้อสอบมักใช้ตามสไลด์
- Doc 5 (cookie banner) PPL: สไลด์ 15 เขียน ≈ 4.0; เอาต์พุตเดโมพิมพ์ 4.1
- ตาราง Dolma (สไลด์ 18): ผมบวกคอลัมน์ไบต์ได้ 11,498.6 GB ต่างจากผลรวมที่พิมพ์ 11,519 GB ราว 20 GB (คอลัมน์คำและ token บวกตรง คอลัมน์เอกสารต่างเศษปัด) ไม่ทราบที่มา ให้ใช้ตัวเลขที่พิมพ์
- FineWeb-Edu: สไลด์ 16 ว่า ~460k ตัวอย่าง; การ์ดของ classifier บน HuggingFace ว่า 450k (ข้อสอบใช้ตามสไลด์)
- ลำดับ pipeline: สไลด์ 31 (ทั่วไป) มีขั้น perplexity และ dedup หลัง ส่วน FineWeb ตามสไลด์ 10/17 ไม่มีขั้น perplexity และ dedup มาก่อน C4 (บันทึกไว้แล้วที่ §7)
- Elo/Bradley-Terry: สไลด์ 37 ระบุเพียงชื่อ ไม่มีสูตรในสไลด์หรือเดโม สูตรทั้งหมดที่ใช้ในชุดฝึกติดป้าย (ต่อยอดจากสไลด์)

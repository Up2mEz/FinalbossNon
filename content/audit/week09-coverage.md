# Coverage matrix สัปดาห์ 9 (สไลด์ NLP-DL-09 และ notebook เดโม เทียบกับ week09.md)

แหล่งที่ตรวจ: `exam-prep/text/slides_w09.txt`, ข้อความเต็มของ NLP-DL-09 (รวมหน้าเปเปอร์ซ้ำ), `exam-prep/source/week_09_demo.ipynb` กับ `exam-prep/text/demo_w09.txt` และภาพของสไลด์ที่เป็นรูปเป็นหลักจาก `inclass-souce/NLP-DL-09.pdf` (เรนเดอร์ด้วย PyMuPDF: สไลด์ 5, 14, 15, 17, 18, 19, 24, 30, 37, 39, 45) เลขสไลด์ = ลำดับหน้าใน PDF ซึ่งมี 49 หน้า

สรุปตัวเลข: สไลด์ทั้งหมด 49; ครอบคลุมแล้วก่อนรอบนี้ 44; หลังรอบนี้ครอบคลุม 48 และข้าม 1 (หน้าปก); เพิ่มเนื้อหาใหม่ให้สไลด์ที่ขาดทั้งหมด 4 หน้า (2, 25, 45, 49) และเติมรายละเอียดให้สไลด์ที่ครอบคลุมบางส่วน 13 หน้า (3, 5, 7, 8, 10, 11, 13, 14, 17, 39, 40, 47, 48)

คำอธิบายสถานะ: covered = มีอยู่เดิมและครบ; covered + detail = มีอยู่เดิม แต่เติมรายละเอียดรอบนี้; added now = ไม่มีเดิม เพิ่มรอบนี้; skipped = ไม่ต้องใส่ พร้อมเหตุผล

| สไลด์ | หัวข้อ | อยู่ที่ไหนใน week09.md | สถานะ |
|---|---|---|---|
| 1 | หน้าปก Vector Representation & Semantic Search (The Math Behind RAG) | ชื่อบทเรียนและย่อหน้าเปิดของ week09.md | skipped: หน้าปก ไม่มีเนื้อหาให้ออกสอบ |
| 2 | ทบทวนสามกระบวนทัศน์ (Encoder, Decoder, Encoder-Decoder) และช่องว่างที่ต้องปิด | หัวข้อ 1 ย่อหน้า "ทบทวนจากสัปดาห์ 8 (สไลด์ 2)" | added now |
| 3 | ประโยชน์ 5 อย่างของ sentence embedding, ฟังก์ชัน f, คุณสมบัติที่ต้องการ | หัวข้อ 1 (ย่อหน้าแรกและ "คุณสมบัติที่ต้องการ (สไลด์ 3)") | covered + detail (เพิ่มสูตร sim ≈ semantic_similarity) |
| 4 | Naive approach: mean pooling เทียบ CLS และตาราง STS | หัวข้อ 1 ตารางและ "อธิบายละเอียดว่าสองวิธีทำอะไร" | covered |
| 5 | กราฟ anisotropy ของ Ethayarajh (BERT, ELMo, GPT-2 รายชั้น) | หัวข้อ 2 (ผลเสียเชิงปฏิบัติ และวิธีอ่านกราฟสไลด์ 5) | covered + detail (เพิ่มวิธีอ่านแกนและค่าโดยประมาณ ติดป้ายว่าอ่านจากกราฟ) |
| 6 | สาเหตุและผล: มิติที่สัมพันธ์กับความถี่ของคำ, mean pooling สืบทอด, CLS ฝึกด้วย NSP | หัวข้อ 2 | covered |
| 7 | คอขวด cross-encoder: `[CLS] A [SEP] B [SEP]`, O(L²) ต่อคู่, n(n−1)/2, 69 ชั่วโมง | หัวข้อ 4 "ความซับซ้อนสองระดับ" และตัวอย่างคำนวณ 2 | covered + detail (เพิ่ม O(L²) ต่อคู่) |
| 8 | bi-encoder: n forward + O(n²·d) dot, ~50 วินาที, < 1 วินาที | หัวข้อ 4 | covered + detail (เพิ่ม O(n²·d) ชัดเจน) |
| 9 | สามทางแก้ และข้อสรุป "objective ไม่ใช่สถาปัตยกรรม" | หัวข้อ 2 ตาราง "แก้ได้ 3 ทาง" | covered |
| 10 | Cosine, cosine distance, L2, dot, normalize แล้วอันดับเท่ากัน | หัวข้อ 3 | covered + detail (เพิ่มความหมายของ cos = 1, 0, −1) |
| 11 | กิจกรรม: cosine ของเวกเตอร์ 3 มิติ s1, s2, s3 | หัวข้อ 2 "ตัวอย่างสไลด์ 11" และตัวอย่างคำนวณ 8 | covered + detail (ใส่ประโยคของแต่ละเวกเตอร์) |
| 12 | เดโม anisotropy: [CLS] 0.86 เทียบ 0.79 อันดับกลับด้าน | หัวข้อ 2 และ 13 | covered |
| 13 | Siamese network: น้ำหนักร่วม, sim = cos หรือ ‖e_A − e_B‖, ต้นกำเนิดลายเซ็น | หัวข้อ 5 | covered + detail (เพิ่ม sim แบบ cos หรือระยะ) |
| 14 | สถาปัตยกรรม SBERT, สามโครงแบบ, รูป classification และ regression | หัวข้อ 5 ตารางสามโครงแบบ และหมายเหตุรูป | covered + detail (อธิบายรูปทั้งสอง) |
| 15 | Pooling 3 แบบ และตาราง ablation (Table 6) | หัวข้อ 5 | covered |
| 16 | เดโม manual mean pooling (ตัวอย่าง activity ของเดโม) | หัวข้อ 13 (เดโมหัวข้อ 3) และชุดฝึกประเภท 8 | covered (สไลด์สำหรับเดโม) |
| 17 | Tensor walkthrough: [1,7], [1,8], [1,7,768], c ∈ [B,2304], W ∈ 2304×3 | หัวข้อ 5 "Tensor walkthrough" และตัวอย่างคำนวณ 1 | covered + detail (เพิ่มหมายเหตุสะกด passta และ attention_mask ต่อประโยค) |
| 18 | ตารางเทียบ bi-encoder กับ cross-encoder, Retrieve-then-Rerank | หัวข้อ 4 | covered |
| 19 | SBERT แก้ anisotropy: attract + repel, alignment และ uniformity, ช่วง cosine ก่อนหลัง, ตัวเลข STS | หัวข้อ 2 และ 5 ("Alignment และ uniformity") | covered |
| 20 | ไลบรารี sentence-transformers: `.encode()` ทำอะไรตามลำดับ, ตารางโมเดล | หัวข้อ 5 | covered |
| 21 | SBERT ด้วยมือใน PyTorch 5 ขั้น และ `F.normalize` | หัวข้อ 5 และ 13 | covered |
| 22 | NLI pipeline: SNLI + MultiNLI, 3 label, สูตร softmax(W·c), เชื่อมกับสัปดาห์ 6 | หัวข้อ 5 | covered |
| 23 | Regression pipeline: STSb ~8,500 คู่, label 0-5, MSE, สูตรสองเฟส | หัวข้อ 5 | covered |
| 24 | SimCSE unsupervised และ supervised พร้อมตัวเลข 76.3, 81.6, 74.89 | หัวข้อ 7 | covered |
| 25 | กิจกรรม: ออกแบบ bi-encoder pipeline ให้ตั๋วภาษาไทย 100,000 ใบ | หัวข้อ 10 "กิจกรรมสไลด์ 25" และชุดฝึกประเภท 5.5 | added now |
| 26 | กรอบ contrastive learning: anchor, positive, negative | หัวข้อ 6 | covered |
| 27 | Triplet loss: สูตร, α, zero-loss, ระยะที่ SBERT ใช้ | หัวข้อ 6 | covered |
| 28 | Hard negative mining: easy, hard, semi-hard | หัวข้อ 6 | covered |
| 29 | Margin ranking loss และความเท่ากันเมื่อ d = 1 − sim | หัวข้อ 6 | covered |
| 30 | MNRL / InfoNCE: สูตร, τ, N = 256 ได้ 255 negative | หัวข้อ 6 | covered |
| 31 | ตารางเทียบ loss ทั้งหกแบบ | หัวข้อ 6 "ตารางเทียบ loss" | covered |
| 32 | ตัวอย่างคำนวณ triplet (a, p, n, α = 0.5) | หัวข้อ 6 และตัวอย่างคำนวณ 6a | covered |
| 33 | ลูปการเทรน SBERT: ไฮเปอร์พารามิเตอร์, เวลา 20 + 5 นาที, backprop ทุกชั้น | หัวข้อ 5 "ลูปการเทรน" | covered |
| 34 | Modern training: ไทม์ไลน์ E5, BGE, instruction-based | หัวข้อ 8 | covered |
| 35 | แหล่งข้อมูลฝึก: ตารางขนาดและชนิด, gold/silver/weak | หัวข้อ 8 | covered |
| 36 | Vector DB: n × d × 4, 7.7 วินาที, ANN, ระบบ vector DB | หัวข้อ 9 | covered |
| 37 | Recall@k และ MRR | หัวข้อ 11 | covered |
| 38 | ทบทวน: ทำไมใช้ IndexFlatIP | หัวข้อ 10 | covered |
| 39 | Exact เทียบ ANN, ตาราง index, เกณฑ์ < 1M และ > 1B | หัวข้อ 9 | covered + detail (เพิ่ม FAISS รองรับ GPU และทุก index, exact ไม่เหมาะเมื่อ n > 100K) |
| 40 | HNSW: หลายชั้น, greedy walk, sublinear แบบ empirical | หัวข้อ 9 | covered + detail (เพิ่ม HNSW เป็นค่าตั้งต้นของ Qdrant, Weaviate, Pinecone) |
| 41 | FAISS 5 ขั้น | หัวข้อ 10 | covered |
| 42 | เลือก metric ตามงาน | หัวข้อ 10 | covered |
| 43 | Pipeline ออฟไลน์/ออนไลน์ และ latency | หัวข้อ 10 | covered |
| 44 | กิจกรรม: เลือกการตั้งค่า index (5M เอกสาร, 32 GB, < 50 ms) | หัวข้อ 10 "กิจกรรมสไลด์ 44" | covered |
| 45 | แผนภาพสถาปัตยกรรมครบวงจร (รูป) | หัวข้อ 10 "สไลด์ 45" | added now |
| 46 | Lab preview 4 งาน | หัวข้อ 12 | covered (ไม่ออกสอบ ตามอาจารย์) |
| 47 | Key takeaways 5 ข้อ | หัวข้อ 12 "สไลด์ 47-49" และ "สรุปสัปดาห์ 9" | covered + detail (เขียนห้าข้อครบตามสไลด์) |
| 48 | สะพานไปสัปดาห์ 12 (RAG 4 ขั้น) และสัปดาห์ 10-11 | หัวข้อ 12 "สไลด์ 47-49" และ "สรุปสัปดาห์ 9" | covered + detail (ใส่ 4 ขั้นของ RAG และสูตรภาพรวม) |
| 49 | อ่านเพิ่ม: SBERT หัวข้อ 1-4, SimCSE หัวข้อ 1-3, เก็บ FAISS index | หัวข้อ 12 "สไลด์ 47-49" | added now |

## เดโม notebook (week_09_demo.ipynb)

| ส่วนของเดโม | อยู่ที่ไหนใน week09.md | สถานะ |
|---|---|---|
| 1 Setup (โหลด bert-base-uncased, คำเตือน UNEXPECTED) | หัวข้อ 13 (ข้อความ "UNEXPECTED" ใน FAISS) | covered |
| 2 Anisotropy: 0.8610 เทียบ 0.7942 | หัวข้อ 2 และ 13 | covered |
| 2b Cone 66 คู่ (mean, min, max, range) | หัวข้อ 13 และ คลังสูตรส่วน ข | covered |
| 3 Manual mean pooling, norm ของ [PAD] 14.90 เทียบ 9.67, clamp 1e-9 | หัวข้อ 13 และคลังสูตรส่วน ก | covered |
| Verification: allclose atol 1e-4 | หัวข้อ 13 | covered |
| 4 sentence-transformers: 0.7035 และ −0.1050, shape (3, 384) | หัวข้อ 13 | covered |
| 5 Pipeline 5 ขั้นไม่ใช่ตัว SBERT, norm 10.21, 8.63, 10.50 | หัวข้อ 13 | covered |
| 6 Triplet และ Margin ranking, Euclidean ε = 1, hard negative 0.5088 | หัวข้อ 6, 13 และชุดฝึกประเภท 9-11 | covered |
| 7 FAISS: ผล k = 3, HNSW ต้องระบุ M และ METRIC_INNER_PRODUCT | หัวข้อ 10, 13 | covered |
| Summary และข้อความ Lab ของเดโม | หัวข้อ 12 | covered |

## หมายเหตุ

- สไลด์ที่เป็นรูปล้วนหรือเกือบล้วน: 5 (กราฟ), 14 (รูปสถาปัตยกรรมสองแบบ), 45 (แผนภาพ) ตรวจจากภาพแล้ว ส่วนสไลด์ 10, 13, 17, 19, 24, 30 มีรูปประกอบที่ไม่มีข้อมูลเพิ่มนอกเหนือจากข้อความ
- ตัวเลขที่สไลด์ขัดกันเองหรือกับเดโม (50 วินาทีเทียบ 5 วินาที, 77.03 เทียบ 74.89, ช่วง cosine ของ SBERT ในสไลด์ 19 เทียบเดโม 2b, สไลด์ 36 เทียบ 43 สำหรับเวลา exact search) อธิบายไว้ใน "ความขัดแย้งและข้อควรระวังระหว่างแหล่งข้อมูล" และในกิจกรรมสไลด์ 25 ของ week09.md
- สามส่วนใหม่ (คลังสูตร, ชุดฝึกคำนวณ 22 ประเภท 100 ข้อ, คู่ที่สับสน) ไม่ได้แทนที่ส่วนเดิม วางไว้ก่อน "สรุปสัปดาห์ 9"

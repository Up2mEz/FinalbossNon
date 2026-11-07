# ตารางครอบคลุมสไลด์สัปดาห์ 12 (NLP-DL-12) เทียบกับ week12.md

ตรวจจากข้อความสไลด์ (`exam-prep/text/slides_w12.txt` และ `dl12.txt`), เดโม (`week12_demo.ipynb`, `demo_w12.txt`) และภาพสไลด์ของหน้าที่เป็นรูปเป็นหลัก (9, 13, 17, 26, 27, 35, 36, 38, 39, 40, 42, 46, 49, 53, 55) เทียบกับตัวเลขในบทเรียน

สถานะ: covered = มีอยู่ก่อนแล้ว, added now = เติมครั้งนี้, skipped = ไม่ใส่พร้อมเหตุผล

| สไลด์ | หัวข้อ | อยู่ที่ไหนใน week12.md | สถานะ |
|---|---|---|---|
| 1 | ชื่อเรื่องและชื่ออาจารย์ | – | skipped: หน้าปก ไม่มีเนื้อหาสอบ |
| 2 | Recap: parametric memory, knowledge cutoff, private data, hallucination, naive fix 2 แบบ, RAG คือทางแก้ | §1 | covered |
| 3 | RAG ของ Lewis: Retriever (MIPS, non-parametric), Generator (parametric), marginalize, end-to-end backprop | §2; สูตรในคลังสูตร ก. | covered |
| 4 | Seven failure points (Barnett 2024), index/query process | §3 | covered |
| 5 | Roadmap 4 Act และเอกสารอ้างอิง (Lewis, Gao survey, Johnson FAISS) | ย่อหน้าต้นบท | added now: รายการอ้างอิง Gao และ Johnson (Act 1 ถึง 4 มีอยู่แล้ว) |
| 6 | Document pipeline 5 ขั้น และมิติ embedding 384/768/1536/3072 | §4.1; คลังสูตร ก. | covered |
| 7 | Naive chunking 512/64, 3 failure modes | §4.2, §4.2.1; ชุดฝึกประเภท 11 | covered |
| 8 | Structural chunking, split priority, token rule, ตาราง | §4.3 | covered |
| 9 | Semantic shift detection: dist = 1 − cos, breakpoint, P95, รูป SBERT และกราฟ 17 chunks | §4.4, §4.4.1; ประเภท 12, 13 | added now: คำอธิบายรูป SBERT และกราฟ 17 chunks |
| 10 | ข้อจำกัด encoder (256 token), N×d×4 byte, ตารางโมเดล | §4.5; ประเภท 14 | covered |
| 11 | Hubness และ anisotropy | §4.6; ประเภท 27 | covered |
| 12 | กิจกรรม Map Your Classmates as Vectors | §4.7 | covered (กิจกรรมถามในห้อง บทเรียนอธิบายเจตนาของคำถาม) |
| 13 | IR 70 ปี ตารางไทม์ไลน์ | §5.1; คลังสูตร ค. | covered |
| 14 | Sparse เทียบ Dense | §5.2 | covered |
| 15 | BM25 สูตร, IDF, stopword, Lucene, rank_bm25 floor, BM25 เทียบ TF-IDF | §5.3, §5.3.4; ประเภท 1, 2 | covered |
| 16 | TF saturation, เพดาน k1+1, ตาราง diminishing returns, half-saturation | §5.3, §5.3.1, §5.3.2; ประเภท 3 ถึง 5 | covered |
| 17 | Length normalization L(d), b, ตาราง 0.625/1/4 | §5.3, §5.3.3; ประเภท 6 | covered |
| 18 | BM25 ตัวอย่างเต็ม 4 กรณี | §5.3; W5; ประเภท 7, 8 | covered |
| 19 | InfoNCE, in-batch negatives, ANCE, รูปแบบข้อมูลเทรน | §5.4; ประเภท 20 | covered |
| 20 | ทำไม sparse หรือ dense อย่างเดียวไม่พอ, BEIR | §5.5 | covered |
| 21 | Hybrid architecture, score incompatibility, Min-Max, RRF | §5.6; ประเภท 10 | covered |
| 22 | RRF สูตรและตัวอย่าง | §5.7, §5.7.1; ประเภท 9 | covered |
| 23 | ทำไม k = 60, ผลของ k, Cormack | §5.7, §5.7.1; ประเภท 9 | covered |
| 24 | Recall-precision gap, ทำไมต้อง rerank | §5.8, §5.8.1; ประเภท 16 | covered |
| 25 | Cross-encoder architecture, [CLS], raw logit, O((lq+ld)^2) | §5.9; ประเภท 25 | covered |
| 26 | Bi-encoder เทียบ cross-encoder, ColBERT | §5.9; ประเภท 18, 25 | covered |
| 27 | Two-stage pipeline ตัวเลข, latency 211 ms | §5.10; W14; ประเภท 17, 18 | added now: แก้หมายเหตุ dedup 200 → 80 (union ของสองลิสต์ 100 ไม่มีทางต่ำกว่า 100) |
| 28 | กรณีออกแบบสำนักงานกฎหมาย (กิจกรรมถามในห้อง) | §5.11; ประเภท 14 | covered (กิจกรรม บทเรียนให้แนวตอบ) |
| 29 | Demo 1 (Sections 1 ถึง 15) | §5.12, §5.12.1; ประเภท 26 | added now: เหตุผลที่ query ได้คะแนนแบบนั้น (token attack, ติดจุลภาค, treatment ไม่อยู่ในคลัง) |
| 30 | Multimodal challenge, CLIP overview | §6.1 | covered |
| 31 | CLIP architecture 3 ขั้น | §6.2 | covered |
| 32 | Vision Transformer | §6.3; ประเภท 21 | covered |
| 33 | CLIP symmetric InfoNCE, N = 32,768 | §6.4; ประเภท 19 | covered |
| 34 | CLIP zero-shot, 76.2%, prompt ensembling, multimodal RAG | §6.5 | covered |
| 35 | Projection adapter LLaVA [49,768] to [49,4096] | §6.6; ประเภท 21 | added now: ตัวอย่าง LLM (Vicuna-13B, Mistral), คำสั่งตัวอย่าง, ข้อสรุปท้ายสไลด์ |
| 36 | LLaVA 2 stage, 595K, 158K, LLaVA-1.5 | §6.6; ประเภท 21 | covered |
| 37 | CLIP mental model: relation, attribute, negation | §6.7 | covered |
| 38 | Production multimodal RAG architecture | §6.8 | added now: Image Match Top-M (เดิมเขียน Top-K), ตัวอย่างคำตอบ 2M/+15%/34%, text-embedding-3 |
| 39 | Vector RAG fails: global summarization ตาราง 5 ชนิดคำถาม | §6.9 | covered |
| 40 | Knowledge graph: node, edge, triple, 2-hop, รูป Einstein | §6.10; ประเภท 24 | added now: ชนิดโหนด (PERSON, LOCATION, CONCEPT, AWARD), 8 โหนด 8 ขอบ |
| 41 | การดึง triple: NER, RE, entity linking, LLM extraction | §6.11 | covered |
| 42 | GraphRAG architecture: indexing time เทียบ query time | §6.12 | covered |
| 43 | Leiden, modularity, hierarchy, community report | §6.12; ประเภท 24 | covered |
| 44 | Local เทียบ global search, map-reduce | §6.12; ประเภท 24 | covered |
| 45 | Multi-hop reasoning, Cypher 3 ขั้น | §6.13; ประเภท 24 | covered |
| 46 | GraphRAG เทียบ Vector RAG, hybrid route, LLM Wiki | §6.14 | covered |
| 47 | Long context ไม่ทดแทน retrieval, O(N^2), lost in the middle | §7.1; ประเภท 22 | covered |
| 48 | Sliding Window Attention O(N·w), KV cache คงที่ | §7.2; ประเภท 22 | added now: อ้างอิง Longformer (Beltagy 2020) |
| 49 | Information propagation tree, receptive field L×w, Mistral 7B | §7.2; ประเภท 22 | covered |
| 50 | Linear interpolation, s = L_target/L_train, 8000 to 2000 | §7.3; ประเภท 22 | covered |
| 51 | NTK-aware, YaRN | §7.4 | covered |
| 52 | Lost in the Middle, U-shape | §7.5; ประเภท 22 | covered |
| 53 | 4 กฎ, bookend prompt structure | §7.6; ประเภท 23, 15 | covered |
| 54 | Demo 2: bookend_order | §7.6.1; ประเภท 23 | covered |
| 55 | Production RAG architecture, 3 routes, benchmark | §8; ประเภท 18 | covered |
| 56 | RAG เทียบ Fine-tune เทียบ Long-context | §8, §8.1 | covered |
| 57 | Summary และ Lab preview | §9 | covered (แล็บไม่ออกสอบ) |

## สรุป

- สไลด์ทั้งหมด 57 หน้า
- ก่อนแก้: ครอบคลุมครบ 49 หน้า, ครอบคลุมบางส่วนหรือมีข้อผิดพลาด 7 หน้า (5, 9, 27, 35, 38, 40, 48), ข้าม 1 หน้า (หน้าปก)
- หลังแก้: ครอบคลุม 56 หน้า, ข้าม 1 หน้า (หน้าปก)
- เพิ่มส่วนใหม่ 3 ส่วนก่อนหัวข้อสรุป: คลังสูตรและค่าที่ต้องรู้, ชุดฝึกคำนวณเชิงลึก (27 ประเภท 119 ข้อ), คู่ที่มักสับสนและตัวเลือกลวง

## ข้อผิดพลาดที่พบและแก้ในบทเรียนเดิม

- §5.10 เขียนว่า 200 รายการ dedup เหลือ 80 แสดงว่าซ้ำ 120 รายการ ซึ่งเป็นไปไม่ได้ เพราะลิสต์ละ 100 รายการ union ต้องไม่ต่ำกว่า 100 แก้เป็นอธิบายว่าต้องมีการตัดเหลือ 80 หรือเป็นตัวเลขภาพประกอบ
- §6.8 เขียน Image Match เป็น Top-K แต่สไลด์ 38 เขียน Top-M แก้แล้ว

## สิ่งที่ยังไม่ยืนยันได้

- เลข 400 step ของ YaRN (สไลด์ 51) ไม่ตรงกับเปเปอร์ทุกตัวอักษร (บันทึกไว้แล้วใน §7.4)
- จำนวนโหนดและขอบของกราฟสไลด์ 40 และตัวเลขกราฟสไลด์ 52 นับหรืออ่านจากรูป ไม่ได้พิมพ์บนสไลด์
- rank_bm25 ไม่ได้ติดตั้งในเครื่องนี้ จึงยืนยันคะแนนเดโมด้วยการเขียนสูตร Okapi เองซึ่งตรงกับค่าที่เดโมพิมพ์ทั้ง 5 ค่า

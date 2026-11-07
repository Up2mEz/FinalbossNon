# ตรวจเลขจากภาพสไลด์ (รายการ #6-#13)

วิธีตรวจ: render หน้า PDF เป็น PNG (90 dpi) แล้วเปิดดูภาพจริง ไม่ได้อ่านจาก text extraction
ดัชนีหน้า PDF = เลขสไลด์ที่พิมพ์บนสไลด์ (ตรงกันทุกหน้า)

## #6 W10 สไลด์ 44-45 (PDF p.44; p.45 เป็นกราฟ outlier ไม่มีตัวเลข absmax) : MATCH
สไลด์ "Quantization Math & Worked Example" แสดง:
- q = round(x/s) + z ; z = 0 สำหรับ symmetric absmax
- FP16 range [-3.2, 3.2] -> INT8 range [-127, 127]
- Scale s = 3.2/127 = 0.0252
- Value 2.1 -> round(2.1/0.0252) = 83
- Dequantize: 83 x 0.0252 = 2.092 (Error introduced!)
- Block-wise Quantization: scale ต่างกันในแต่ละ block เล็ก (e.g., 64 values) เพื่อลด error
ตัวเลขทั้งหมดตรง (block size 64 ระบุเป็น "e.g., 64 values")
หมายเหตุ: p.45 เป็นกราฟ LLM.int8() (x = 125M..175B, outlier emerge ที่ 6.7B), 8-bit baseline accuracy ตกเหลือ ~0.45 ที่ 6.7B และ ~0.31-0.33 ที่ 13B ขึ้นไป

## #7 W12 สไลด์ 16 (PDF p.16) : MATCH (แต่ตารางไม่ใช่ f=1..100)
ตั้งค่า k1 = 1.5, b = 0 (สไลด์ระบุชัดว่าปิด length normalization)
สูตร TF_BM25 = f*(k1+1)/(f+k1); asymptote = k1+1 = 2.50
ตาราง "Diminishing Returns Table" ที่สไลด์แสดงจริง:

| f(t,d) | 0 | 1 | 2 | 5 | 10 | 1,000 (->inf) |
|---|---|---|---|---|---|---|
| TF_BM25 | 0.00 | 1.00 | 1.43 | 1.92 | 2.17 | 2.50 (Max Ceiling) |
| Gain (delta) | - | +1.00 | +0.43 | +0.49 | +0.25 | -> 0 |

- คอลัมน์คือ f = 0,1,2,5,10,1000 (ไม่มี 20/50/100 และไม่มีคอลัมน์ % ของ ceiling)
- เปอร์เซ็นต์ที่สไลด์พูดถึงมีอย่างเดียว: เมื่อ f = k1 คะแนน = 50% ของ ceiling สูงสุด (1.25)
- ข้อสังเกต: Gain ที่ f=5 (+0.49) มากกว่า f=2 (+0.43) เพราะ step กระโดด 2->5 (3 หน่วย) ไม่ใช่ +1 ต่อหน่วย
ถ้า notes ระบุ "f = 1..100" หรือมี % ต่อคอลัมน์ ให้แก้เป็นตารางข้างบน

## #8 W12 สไลด์ 18 (PDF p.18) : MATCH
Setup: query "deep learning", t1="deep" (IDF=1.50), t2="learning" (IDF=2.50); avgdl = 100 words, k1 = 1.50, b = 0.75

| Scenario | Length | Word counts (deep, learn) | Length factor L(d) | Saturated TF (deep, learn) | Final BM25 | Insight |
|---|---|---|---|---|---|---|
| Case 1: Concise & Focused | 50 | 2, 2 | 0.625 | 1.70, 1.70 | 6.81 (ดาว, Winner) | short doc, high density ได้ length boost |
| Case 2: Keyword Spammer | 100 | 30, 30 | 1.000 | 2.38, 2.38 | 9.52 | กล่าวถึงคำ 15x มากกว่า Case 1 แต่คะแนนเพิ่มแค่ 1.4x (Saturation cap!) |
| Case 3: Diluted Long Doc | 500 | 2, 2 | 4.000 | 0.62, 0.62 | 2.50 (ลูกศรลง) | match เหมือน Case 1 แต่คะแนนตก 63% เพราะ length penalty |
| Case 4: Single Common Match | 100 | 1, 0 | 1.000 | 1.00, 0.00 | 1.50 | match แค่คำ IDF ต่ำ ("deep") |

(L(d) = 1 - b + b*len/avgdl; ตรวจแล้วตรงกับสูตร)

## #9 W12 สไลด์ 22 (PDF p.22) : MATCH
สูตร RRF(d) = sum_r 1/(k + rank_r(d)), k = 60 (typical). Worked Example (k=60):

| Doc | BM25 rank | Dense rank | RRF score |
|---|---|---|---|
| A | 1 | 3 | 1/61 + 1/63 = 0.03227 |
| B | 5 | 1 | 1/65 + 1/61 = 0.03178 |
| C | 2 | 50 | 1/62 + 1/110 = 0.025 |
| D | 50 | 2 | 1/110 + 1/62 = 0.025 |

Key property: doc ที่อยู่สูงในหลาย list ถูก boost; outlier ใน list เดียวถูกลดทอนด้วย k

## #10 W12 สไลด์ 26 (PDF p.26) : CORRECTION (ตารางเป็น Bi vs Cross เท่านั้น ColBERT อยู่ในข้อความด้านล่าง ไม่มีเลข ~100x ตรงๆ)
ตาราง "Bi-Encoder vs. Cross-Encoder - The Complete Tradeoff" (ไม่มีคอลัมน์ ColBERT):

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Architecture | 2 independent towers | 1 joint tower |
| Cross-term attention | None | Full bidirectional |
| Pre-computation | check (offline) | X (online per pair) |
| Latency | Sub-ms (ANN lookup) | 10-100ms per pair |
| Scale | Millions of docs | Hundreds of docs |
| Precision | Medium | High |
| Role in pipeline | Stage 1: Retrieve | Stage 2: Rerank |

ข้อความใต้ตาราง: "ColBERT (2020): The Middle Ground - Late interaction: independent encoders but token-level matching at inference time via MaxSim operator. ~2 orders of magnitude cheaper than a Cross-Encoder at query time; quality between Bi- and Cross-Encoder."
- แก้: ตัวเลขบนสไลด์คือ "~2 orders of magnitude cheaper" (ประมาณ 100x) เทียบกับ Cross-Encoder; ไม่ได้เขียน "100x" ตรงๆ และ ColBERT ไม่ได้อยู่ในตาราง คุณภาพระบุแค่ "between Bi- and Cross-Encoder"

## #11 W12 สไลด์ 27 (PDF p.27) : MATCH
สไลด์ระบุเป็น "Illustrative Pipeline Example (measure these values on your own hardware and corpus)":

| Stage | Method | In | Out | Latency |
|---|---|---|---|---|
| 1a | BM25 retrieval | 1M | 100 | ~5ms |
| 1b | FAISS dense retrieval | 1M | 100 | ~10ms |
| 2 | RRF fusion | 200 -> deduplicated | 80 | ~1ms |
| 3 | Cross-encoder rerank | 80 | 10 | ~200ms |
| 4 | LLM generation | 10 chunks (~2K tokens) | Answer | ~1-5s |

รวม: ถ้า 1a และ 1b รันขนาน wall-clock = max(5,10) + 1 + 200 = 211 ms (ตรง; ไม่รวม LLM generation) Tip: ถ้าไวต่อ latency ให้ข้าม Stage 3 แล้วจูน k ของ Stage 2

## #12 W12 สไลด์ 39 และ 46 (PDF p.39, p.46) : MATCH (ถ้า notes ยึดตามตารางด้านล่าง)
ข้อสังเกต: ไม่มีตารางเดียวที่เทียบ Vector RAG vs GraphRAG ทุกมิติ มี 2 ตาราง

### s.39 "When Vector-based RAG Fails - The Global Summarization Problem"
| Query type | Example | Vector RAG status | Primary failure mode |
|---|---|---|---|
| Specific Fact | "What is the boiling point of water?" | check Excellent | None (Exact local match) |
| Local Fact | "What did Person A say about Topic B?" | check Good | None (Local context suffices) |
| Global Summary | "What are the main themes across all 500 papers?" | X Fails | Retrieves disconnected fragments, not a synthesis |
| Multi-Hop | "Who influenced Author A, and what did they write?" | X Poor | Cannot chain relations (A->B->C) across docs |
| Comparative | "How does Policy A differ from Policy B across all docs?" | X Poor | Lacks holistic corpus-level comparative structure |

Root cause: vector search มอง chunk เป็นจุดโดดๆ ใน embedding space บอดต่อ entity relationships; Missing structure -> Knowledge Graphs

### s.46 "GraphRAG vs. Vector RAG - Decision Framework"
| | Vector RAG | GraphRAG |
|---|---|---|
| Best for | Specific facts, FAQs, short docs | Global themes, multi-hop, long narratives |
| Latency | Usually lower; ANN + one/few calls | Often higher; graph assembly and map-reduce may add calls |
| Index cost | Embeddings + vector index | Extraction + entity resolution + graph + summaries |
| Query cost | Depends on retrieved context and generator | Depends strongly on local/global mode and community count |

Hybrid Strategy: route ด้วย lightweight classifier: specific factual -> Vector RAG; thematic/relational -> GraphRAG
เพิ่มเติม: "LLM Wiki" pattern (2026) compile corpus เป็น Markdown wiki; practitioner proposal ไม่มี published evaluation; target ~100-10,000 docs
สังเกต: ตาราง s.46 เป็นคำอธิบายเชิงคุณภาพ ไม่มีตัวเลข (เช่น ไม่มี "Usually lower" เป็น ms)

## #13 W12 สไลด์ 53 (PDF p.53) : MATCH (เชิงโครงสร้าง; สไลด์ไม่ได้เขียนลิสต์ [1,3,4,5,2] ตรงๆ)
สไลด์ "Optimal Bookend Prompt Structure" แสดงเป็น template:
```
[TOP-1 CHUNK]      <- Position 1 (Primacy Zone: Strongest Attention)
[CHUNKS 3 ... K]   <- Middle Positions (Lower Attention Zone)
[TOP-2 CHUNK]      <- Last Position (Recency Zone: Next to Query)
Question: {query}
```
ดังนั้นสำหรับ top-5 (K=5) ลำดับ = [1, 3, 4, 5, 2] ถูกต้อง (อนุมานจาก template: อันดับ 1 ต้นสุด, อันดับ 2 ท้ายสุดติด query, อันดับ 3..K ตรงกลาง) ถ้าจะอ้างว่า "สไลด์แสดง [1,3,4,5,2]" ควรพูดว่า "template แสดง TOP-1, CHUNKS 3..K, TOP-2"

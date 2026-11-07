**Q41** [W12] [BM25 TF Saturation] [difficulty: medium]
ระบบ Search Engine เดิมใช้ TF-IDF ต่อมาเปลี่ยนมาใช้ BM25 โดยตั้งค่า $k_1 = 1.5$ และ $b = 0$ หากวิศวกรทำการทดสอบโดยใส่คำค้นซ้ำในเอกสาร (Keyword Stuffing) จากเดิม 1 ครั้ง เพิ่มเป็น 1,000 ครั้ง ข้อใดอธิบายผลลัพธ์ของคะแนนในส่วน Term Frequency (TF component) ใน BM25 ได้ถูกต้องที่สุด?
A. คะแนน TF component จะเพิ่มขึ้น 1,000 เท่าเช่นเดียวกับ TF-IDF เนื่องจาก $b = 0$ หมายถึงการปิดระบบ Length Normalization
B. คะแนน TF component จะถูกจำกัดด้วยเพดานสูงสุด (Ceiling asymptote) เท่ากับ $k_1 + 1 = 2.50$ ทำให้คะแนนเพิ่มขึ้นเพียงประมาณ 2.5 เท่าของกรณี $f = 1$
C. คะแนน TF component จะลดลงจนเป็น 0 เนื่องจากระบบมองว่าคำที่มีความถี่สูงผิดปกติจัดเป็น Stopword ทันที
D. คะแนน TF component จะถูกจำกัดด้วยเพดานสูงสุดที่เท่ากับค่า $k_1 = 1.50$ พอดี และอิ่มตัวสมบูรณ์ตั้งแต่ความถี่ $f = 10$ ครั้ง

**Q42** [W12] [BM25 Length Normalization Calculation] [difficulty: hard]
กำหนดให้คลังเอกสารมีค่าเฉลี่ยความยาว $avgdl = 100$ คำ ระบบใช้ BM25 โดยตั้งค่า $k_1 = 1.5$ และ $b = 0.75$ พิจารณาเอกสาร 2 ฉบับที่มีคำค้น $t$ ปรากฏด้วยความถี่เท่ากันคือ $f(t, d) = 3$ ครั้ง
- เอกสาร A มีความยาว $|d| = 50$ คำ
- เอกสาร B มีความยาว $|d| = 100$ คำ
คะแนนในส่วนของ Term Frequency component ($\frac{f \cdot (k_1 + 1)}{f + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$) ของเอกสารทั้งสองมีค่าเท่าใด และเอกสารใดได้คะแนนสูงกว่า?
A. เอกสาร A ได้ 1.667 และเอกสาร B ได้ 1.905 โดยเอกสาร B ได้คะแนนสูงกว่าเนื่องจากมีความยาวมากกว่า
B. เอกสาร A ได้ 2.500 และเอกสาร B ได้ 2.500 เท่ากัน เพราะคำค้นปรากฏ $f = 3$ ครั้งเท่ากันทั้งสองเอกสาร
C. เอกสาร A ได้ 1.905 และเอกสาร B ได้ 1.667 โดยเอกสาร A ได้คะแนนสูงกว่าเอกสาร B เนื่องจากเอกสารสั้นกว่ามีความหนาแน่นของคำค้นสูงกว่า
D. เอกสาร A ได้ 0.833 และเอกสาร B ได้ 1.667 โดยเอกสาร B ได้คะแนนสูงกว่าเนื่องจากถูกหัก Length Penalty น้อยกว่า

**Q43** [W12] [Hybrid RAG & RRF Calculation] [difficulty: medium]
ในระบบ Hybrid RAG ที่ผสานผลการค้นหาจาก BM25 (Sparse) และ Dense Retrieval เข้าด้วยกันโดยใช้สูตร Reciprocal Rank Fusion $RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$ กำหนดค่าคงที่ $k = 60$ (โดย $r_m(d)$ คือลำดับที่เริ่มต้นจาก 1) พิจารณาเอกสาร 2 ฉบับ:
- เอกสาร X: ติดอันดับที่ 1 ใน BM25 และติดอันดับที่ 3 ใน Dense
- เอกสาร Y: ติดอันดับที่ 2 ใน BM25 เพียงระบบเดียว (ไม่ติดในผลลัพธ์ของ Dense)
คะแนน RRF ของเอกสารทั้งสองมีค่าเท่าใด และเอกสารใดจะได้อันดับที่ดีกว่าหลังการ Fuse?
A. เอกสาร X ได้คะแนนประมาณ 0.0323 ส่วนเอกสาร Y ได้ประมาณ 0.0161 ทำให้เอกสาร X ชนะเอกสาร Y (สอดคล้องกับหลักการ Intersection > Union)
B. เอกสาร X ได้คะแนนประมาณ 0.0161 ส่วนเอกสาร Y ได้ประมาณ 0.0323 ทำให้เอกสาร Y ชนะเอกสาร X เพราะอันดับ 2 ในระบบเดี่ยวมีคะแนนดิบสูงกว่า
C. เอกสาร X และเอกสาร Y ได้คะแนน 0.0328 เท่ากัน เนื่องจากผลรวมของลำดับเท่ากับ 4 ทั้งคู่
D. เอกสาร X ได้คะแนนประมาณ 0.0250 ส่วนเอกสาร Y ได้ประมาณ 0.0167 โดยเอกสาร X ชนะเพราะ $k = 60$ บังคับให้คิดเฉพาะผลรวมคะแนนดิบ

**Q44** [W12] [Two-Stage Pipeline & Cross-Encoder] [difficulty: medium]
ในการออกแบบ Two-Stage Retrieval Pipeline สำหรับคลังเอกสารขนาดใหญ่ ข้อใดเปรียบเทียบการทำงานและข้อจำกัดระหว่าง Bi-encoder (Stage 1) และ Cross-encoder (Stage 2) ได้ถูกต้องตามความเป็นจริงที่สุด?
A. สามารถนำ Cross-encoder มาใช้เป็น Stage 1 ค้นหากับคลังเอกสาร 1 ล้านชิ้นได้โดยตรง โดย Pre-compute เวกเตอร์ของเอกสารล่วงหน้าเพื่อลด Latency
B. Stage 2 (Cross-encoder) มีหน้าที่เพิ่มค่า Recall ให้ครอบคลุมที่สุด โดยส่งคืนคะแนนความเกี่ยวข้องเป็นค่าความน่าจะเป็นในช่วง [0, 1] เสมอ
C. ใน Stage 1 ต้องรัน BM25 และ Dense แบบเรียงลำดับ (Sequential) จากนั้นนำคะแนนดิบมารวมกันตรง ๆ ก่อนส่งให้ Cross-encoder
D. Stage 1 (Bi-encoder/BM25) มุ่งเน้น High Recall ด้วยความเร็วสูง ขณะที่ Stage 2 (Cross-encoder) มุ่งเน้น High Precision กับ Candidate จำนวนน้อย (เช่น 20–100 ชิ้น) เนื่องจาก Cross-encoder ไม่สามารถ Pre-compute เวกเตอร์แยกได้และมีต้นทุนการคำนวณสูงจนเป็นคอขวดของ Latency

**Q45** [W12] [CLIP & Multimodal RAG] [difficulty: medium]
ในการนำโมเดล CLIP (Contrastive Language-Image Pre-training) มาประยุกต์ใช้ในระบบ Multimodal RAG ข้อใดกล่าวถึงสถาปัตยกรรม การฝึกฝน และพฤติกรรมของ CLIP ได้**ถูกต้อง**?
A. CLIP ได้รับการฝึกด้วย Asymmetric InfoNCE Loss โดยคำนวณ Cosine Similarity จากทิศทางข้อความไปยังรูปภาพเพียงฝั่งเดียว
B. CLIP ฝึกด้วย Symmetric InfoNCE Loss บนคู่ภาพ-ข้อความ 400M คู่ ดึงเวกเตอร์คู่ที่ตรงกัน (Diagonal) เข้าหากัน และผลักคู่อื่นใน Batch ออกจากกัน พร้อมทำ Zero-shot Classification ได้โดยไม่ต้อง Fine-tune บน Dataset ปลายทาง
C. CLIP มีจุดเด่นสำคัญในการทำความเข้าใจ Compositionality และ Negation ได้อย่างสมบูรณ์แบบ เช่น แยกแยะความแตกต่างระหว่าง "a cube on a sphere" กับ "a sphere on a cube" ได้อย่างแม่นยำ
D. ในการทำ Zero-shot Classification ด้วย CLIP จำเป็นต้องนำ Linear Classifier มา Fine-tune บน ImageNet ก่อนเสมอ

**Q46** [W12] [GraphRAG vs Vector RAG] [difficulty: hard]
ผู้บริหารต้องการสอบถามคำถามระดับภาพรวมขององค์กร (Global Sensemaking Query) เช่น "ประเด็นปัญหาสำคัญและธีมหลักที่เกิดขึ้นตลอด 5 ปีที่ผ่านมามีอะไรบ้าง?" ข้อใดเปรียบเทียบขีดความสามารถระหว่าง Standard Vector RAG กับ GraphRAG ในสถานการณ์นี้ได้ถูกต้อง?
A. Vector RAG มักล้มเหลวเพราะ Chunk เอกสารแยกเป็นจุด ๆ ไม่สามารถเชื่อมโยง Multi-hop ข้ามเอกสารได้ ขณะที่ GraphRAG ตอบได้ดีผ่าน Global Search ซึ่งทำ Map-Reduce สรุปข้อมูลจาก Community Summaries ที่สร้างไว้ล่วงหน้าด้วยอัลกอริทึม Leiden
B. Vector RAG ตอบคำถามนี้ได้แม่นยำเพียงแค่เพิ่มค่า Top-K ให้มากขึ้น เนื่องจาก Dense Embeddings สามารถจับกลุ่มประเด็นภาพรวมระดับคลังเอกสารได้เองโดยอัตโนมัติ
C. GraphRAG ใช้ทรัพยากรน้อยกว่าและรันเร็วกว่า Vector RAG ในคำถาม Global Search เพราะ GraphRAG เรียกใช้ LLM เพียง 1 ครั้งในขั้นตอน Query Time
D. GraphRAG ต้องประมวลผลสร้าง Community Summaries ทั้งหมดแบบ Real-time ณ ตอนรับ Query เพื่อให้ข้อมูลอัปเดตที่สุด จึงไม่สามารถทำ Indexing ล่วงหน้าได้

**Q47** [W12] [Long-Context & Bookend Reordering] [difficulty: medium]
จากปรากฏการณ์ "Lost in the Middle" ซึ่งพบว่า LLM มี Primacy และ Recency Bias ทำให้ดึงข้อมูลจากส่วนต้นและส่วนท้ายของ Context ได้ดีกว่าตรงกลาง หากขั้นตอน Reranking คืนผลลัพธ์เอกสาร 5 ชิ้นเรียงตามลำดับความเกี่ยวข้องคือ `[Doc_1, Doc_2, Doc_3, Doc_4, Doc_5]` การจัดเรียงลำดับใหม่ด้วยเทคนิค Bookend Reordering ก่อนส่งเข้า Prompt จะได้ลำดับใด?
A. `[Doc_1, Doc_2, Doc_3, Doc_4, Doc_5]`
B. `[Doc_5, Doc_4, Doc_3, Doc_2, Doc_1]`
C. `[Doc_1, Doc_3, Doc_4, Doc_5, Doc_2]`
D. `[Doc_1, Doc_2, Doc_5, Doc_4, Doc_3]`

**Q48** [W12] [Embedding Pathologies & Chunking Limits] [difficulty: hard]
ทีมงานตรวจสอบระบบ Vector Search ที่ใช้โมเดล `all-MiniLM-L6-v2` แล้วพบพฤติกรรมผิดปกติ 2 ประการ:
1. เอกสารบางชิ้นที่มีข้อความเป็นคำบรรยายกว้าง ๆ (Generic content) ติดอันดับ Top-K ซ้ำ ๆ ในเกือบทุก Query ที่ไม่เกี่ยวข้องกัน
2. ค่าเฉลี่ย Pairwise Cosine Similarity ระหว่างเวกเตอร์ที่ไม่เกี่ยวข้องกันมีค่าสูงผิดปกติ (Inflated similarities)
ข้อใดระบุชื่อปรากฏการณ์ทางสถิติทั้งสอง และขีดจำกัดของโมเดล `all-MiniLM-L6-v2` ได้ถูกต้อง?
A. อาการที่ 1 คือ Anisotropy, อาการที่ 2 คือ Hubness, และ MiniLM จะส่งคืน Error ทันทีเมื่อ Chunk มีขนาดเกิน 256 Tokens
B. อาการที่ 1 คือ Context Fragmentation, อาการที่ 2 คือ Mid-Table Cut, และ MiniLM รองรับความยาวสูงสุดได้ 3,072 Tokens
C. อาการที่ 1 คือ Vocabulary Mismatch, อาการที่ 2 คือ Hallucinated Relevance, และ MiniLM จะลดมิติเวกเตอร์ลงอัตโนมัติ
D. อาการที่ 1 คือ Hubness, อาการที่ 2 คือ Anisotropy, และ MiniLM จะตัดข้อความส่วนเกินทิ้ง (Truncate) อย่างเงียบ ๆ ที่ 256 Tokens โดยไม่แจ้งเตือน Error

**Q49** [W12] [Sparse vs Dense Retrieval Failure Modes] [difficulty: medium]
พิจารณาข้อสอบถาม 2 ข้อความในระบบสืบค้นข้อมูลทางการแพทย์:
- Query 1: `"heart attack treatment"` แต่ในคลังข้อมูลมีเอกสารที่ระบุคำว่า `"myocardial infarction"`
- Query 2: ค้นหารหัสกำกับใบเสร็จเฉพาะเจาะจง เช่น `"invoice #INV-2024-7832"`
โมเดลใดจะดึงข้อมูลที่ตรงกับความต้องการได้ดีกว่าในแต่ละ Query และเป็นเพราะเหตุใด?
A. BM25 ชนะทั้งสอง Query เพราะ BM25 เป็น Baseline ที่มีประสิทธิภาพเหนือกว่า Dense Retrieval เสมอในทุกเงื่อนไขการค้นหา
B. Query 1 โมเดล Dense ชนะเพราะเข้าใจความหมายเชิงอรรถศาสตร์ข้ามคำพ้องความหมาย (แก้ปัญหา Vocabulary Mismatch ของ BM25) ขณะที่ Query 2 BM25 ชนะเพราะจับ Exact Match และ Rare Terms ได้แม่นยำ (ป้องกันปัญหา Hallucinated Relevance ของ Dense)
C. Dense ชนะทั้งสอง Query เพราะพัฒนามาจาก Deep Learning จึงสามารถทดแทนการทำงานของ Sparse Retrieval ได้อย่างสมบูรณ์ในทุกกรณี
D. Query 1 โมเดล BM25 ชนะเพราะคำค้นหาสั้นและตรงไปตรงมา ขณะที่ Query 2 โมเดล Dense ชนะเพราะสามารถเดารหัสที่ตัวเลขใกล้เคียงกันได้

**Q50** [W12] [RAG Decision Framework & Long-Context Scaling] [difficulty: hard]
หน่วยงานแห่งหนึ่งมีเอกสารระเบียบปฏิบัติและคำพิพากษาใหม่ ๆ อัปเดตทุกสัปดาห์รวมหลายแสนหน้า วิศวกรคนหนึ่งเสนอว่า *"เนื่องจาก LLM รุ่นปัจจุบันมี Context Window ขนาดใหญ่ถึง 1 ล้าน Tokens แล้ว เราจึงไม่จำเป็นต้องทำ RAG ให้ยุ่งยาก โดยให้นำเอกสารทั้งหมดใส่ใน System Prompt ทุกครั้ง หรือไม่ก็นำเอกสารทั้งหมดไป Fine-tune โมเดล"* ข้อใดประเมินข้อเสนอนี้ตามหลักสถาปัตยกรรม RAG ได้ถูกต้องที่สุด?
A. ข้อเสนอนี้มีข้อบกพร่องทั้งสองแนวทาง: การใส่เอกสารทั้งหมดใน Long Context มี Attention Cost สูงระดับ $O(N^2)$ และเผชิญปัญหา Lost in the Middle ส่วนการ Fine-tune เหมาะสำหรับปรับ Style/Format แต่ไม่เหมาะกับการอัปเดตข้อเท็จจริงที่เปลี่ยนบ่อยเพราะมีค่าใช้จ่ายสูงและสร้าง Knowledge Cutoff ใหม่อยู่ดี
B. ข้อเสนอนี้ถูกต้องในส่วนของ Long Context เพราะการส่งข้อความยาว 1 ล้าน Tokens มีค่าใช้จ่ายคงที่ระดับ $O(1)$ เสมอ และโมเดลสามารถดึงข้อมูลทุกตำแหน่งได้แม่นยำเท่ากัน
C. ข้อเสนอนี้ถูกต้องในส่วนของ Fine-tune เพราะการ Fine-tune เป็นแนวทางที่เหมาะสมที่สุดในการอัปเดตข้อมูลไดนามิกประจำสัปดาห์โดยไม่ต้องพึ่งพาคลังข้อมูลภายนอก
D. ทั้งสองแนวทางที่เสนอมาสามารถทดแทนระบบ RAG ได้อย่างสมบูรณ์แบบโดยไม่ส่งผลกระทบต่อทั้ง Latency, ความถูกต้อง และต้นทุนการประมวลผล

**F5** [W12] [Hybrid RAG / Reciprocal Rank Fusion]
ในการผสานผลลัพธ์การค้นหาด้วยวิธี Reciprocal Rank Fusion (RRF) ตามงานวิจัยของ Cormack et al. (2009) ค่าคงที่ Smoothing Factor $k$ ที่นิยมใช้เป็นค่ามาตรฐานโดยไม่ต้องทำการ Tuning มีค่าเท่ากับ ___

**F6** [W12] [GraphRAG / Community Detection]
ในกระบวนการจัดทำดัชนีของ GraphRAG ขั้นตอนการตรวจจับกลุ่มชุมชนของเอนทิตี (Hierarchical Community Detection) เพื่อนำไปสร้าง Community Summaries จะใช้อัลกอริทึมชื่อ ___ ซึ่งได้รับการพัฒนาขึ้นมาแก้ไขจุดบกพร่องของ Louvain Algorithm โดยรับประกันว่าชุมชนย่อยที่ได้จะเชื่อมต่อกันอย่างสมบูรณ์ (Well-connected)

### Answer key
**Q41**: B — เพดานของ TF ใน BM25 คือ $k_1 + 1 = 2.50$ (เมื่อ $k_1 = 1.5$) ทำให้เมื่อ $f = 1,000$ คะแนน TF จะอยู่ที่ $2500/1001.5 \approx 2.496$ ซึ่งเพิ่มขึ้นประมาณ 2.5 เท่าจาก $f=1$ (คะแนน 1.000) ไม่ใช่ 1,000 เท่า (Notes Section 3.1, 7.1, 7.2)
**Q42**: C — Doc A ($|d|=50$): $L = 0.25 + 0.75(50/100) = 0.625$, TF component $= (3 \times 2.5)/(3 + 1.5 \times 0.625) = 7.5/3.9375 \approx 1.905$; Doc B ($|d|=100$): $L = 1.0$, TF component $= 7.5/(3 + 1.5) = 7.5/4.5 \approx 1.667$ ทำให้ Doc A ชนะ (Notes Section 3.2)
**Q43**: A — Doc X ได้ $1/(60+1) + 1/(60+3) = 1/61 + 1/63 \approx 0.016393 + 0.015873 = 0.032266$; Doc Y ได้ $1/(60+2) = 1/62 \approx 0.016129$ ทำให้ Doc X ชนะตามหลัก Intersection > Union (Notes Section 3.5, 4)
**Q44**: D — Bi-encoder เน้น High Recall และคำนวณเร็ว ส่วน Cross-encoder ให้ High Precision แต่คำนวณแบบ Full Joint Attention $O((|q|+|d|)^2)$ จึง Pre-compute แยกไม่ได้และเป็นคอขวดของ Latency (Notes Section 2.2, 4, 7.8, 7.9)
**Q45**: B — CLIP ใช้ Symmetric InfoNCE Loss บน 400M คู่ ดึง Diagonal และผลัก Off-diagonal ใน Batch ขนาด 32,768 และรองรับ Zero-shot แต่ล้มเหลวเรื่อง Compositionality/Negation (Notes Section 2, 6.9, 7.15, 7.16)
**Q46**: A — Vector RAG ล้มเหลวใน Global Queries ข้ามเอกสาร ขณะที่ GraphRAG ตอบได้ผ่าน Global Search แบบ Map-Reduce เหนือ Community Summaries ที่ทำไว้ล่วงหน้าด้วย Leiden (Notes Section 2, 7.18, 7.19, 7.21)
**Q47**: C — Bookend Reordering จัดอันดับ $[1, 2, 3, 4, 5]$ เป็น $[1, 3, 4, 5, 2]$ โดยวาง Rank 1 ไว้หน้าสุด และ Rank 2 ไว้ท้ายสุด เพื่อรับมือกับ Primacy และ Recency Bias (Notes Section 4, 7.24)
**Q48**: D — การที่ Generic Vector ติด Top-K เสมอเรียกว่า Hubness ส่วนค่า Cosine สูงผิดปกติเรียกว่า Anisotropy และ `all-MiniLM-L6-v2` ตัด Input ที่ 256 Tokens อย่างเงียบ ๆ โดยไม่แจ้ง Error (Notes Section 2.1, 7.12, 7.13)
**Q49**: B — Dense Retrieval แก้ปัญหา Vocabulary Mismatch ข้ามคำที่มีความหมายเหมือนกันได้ดี ขณะที่ BM25 เหมาะกับ Exact Match / Rare Terms และป้องกัน Hallucinated Relevance ของ Dense (Notes Section 2.2, 4, 7.5)
**Q50**: A — Long Context มี Attention Cost $O(N^2)$ และเสี่ยง Lost in the Middle ขณะที่ Fine-tuning มีไว้ปรับ Style/Format ไม่เหมาะกับข้อมูลความรู้ที่อัปเดตบ่อยเพราะสร้าง Knowledge Cutoff ใหม่ (Notes Section 2.0, 7.22, 7.27)
**F5**: 60 — Smoothing Constant $k$ ในสูตร RRF มาตรฐานตาม Cormack et al. (2009) ถูกกำหนดตายตัวไว้ที่ 60 โดยไม่ต้องจูน (Notes Section 1, 3.5, 6.1)
**F6**: Leiden — GraphRAG ใช้อัลกอริทึม Leiden ในการแบ่ง Hierarchical Communities โดยปรับปรุงจาก Louvain เพื่อการันตี Well-connected Communities (Notes Section 6.12, 7.20)

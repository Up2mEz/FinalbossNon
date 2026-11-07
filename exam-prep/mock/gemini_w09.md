**Q11** [W9] [Anisotropy in raw BERT] [difficulty: medium]
เมื่อนำ pre-trained BERT (`bert-base-uncased`) ที่ยังไม่ผ่านการ fine-tune สำหรับ sentence embeddings มาสกัดเวกเตอร์เพื่อวัดความคล้ายคลึงของประโยค (Semantic Textual Similarity) ข้อใดอธิบายปรากฏการณ์ Anisotropy และผลกระทบต่อการวัดผลได้**ถูกต้องที่สุด**ตามผลการทดลอง
A. เวกเตอร์ของประโยคกระจุกตัวอยู่ในทรงกรวยแคบๆ (narrow cone) ทำให้ค่า Cosine Similarity ของทุกคู่มีค่า floor ต่ำสุดไม่เคยต่ำกว่า ~0.45 แม้จะเป็นประโยคที่ไม่เกี่ยวข้องกัน และอาจเกิดการสลับลำดับความคล้ายคลึง (ranking inversion)
B. เวกเตอร์ของประโยคกระจายตัวแบบสุ่มสมบูรณ์ (isotropic) ส่งผลให้ค่า Cosine Similarity ของทุกคู่ประโยคมีค่าเข้าใกล้ 0 จนไม่สามารถแยกความแตกต่างได้
C. การเฉลี่ยแบบ mean pooling ได้รับผลกระทบจาก token ความถี่ต่ำมากเกินไป ส่งผลให้การใช้เวกเตอร์จาก `[CLS]` token ให้ผลการวัด semantic similarity ที่ดีกว่า mean pooling เสมอ
D. โมเดลบีบอัดเวกเตอร์ทุกประโยคจนกลายเป็นเวกเตอร์เดียวกันในทุกมิติ ทำให้ค่า Cosine Similarity ของคู่ประโยคทดสอบทั้งหมดมีค่าเท่ากับ 1.0000

**Q12** [W9] [Bi-encoder vs Cross-encoder] [difficulty: medium]
ทีมวิศวกรต้องการสร้างระบบ Semantic Search สำหรับค้นหาเอกสารจากคลังข้อความขนาด 10,000 ฉบับแบบตอบสนองทันที (real-time query) หากเวลาเฉลี่ยของ forward pass ของโมเดล BERT ต่อหนึ่งอินพุตคือ 5 ms ข้อใดเปรียบเทียบสถาปัตยกรรมและเลือกแนวทางการออกแบบระบบได้**ถูกต้อง**?
A. ควรใช้ Cross-encoder ในการค้นหาเอกสารทั้งหมดโดยตรง เพราะประมวลผลการจับคู่ทั้งหมด 10,000 ฉบับได้ในเวลาเพียง 50 วินาที
B. Bi-encoder มีความซับซ้อนระดับ $O(n^2)$ เนื่องจากต้องคำนวณ Cross-attention ระหว่าง query กับคลังเอกสารพร้อมกัน จึงไม่สามารถนำมาใช้ในงาน real-time ได้
C. Cross-encoder มีความซับซ้อน $O(n^2)$ ซึ่งการเปรียบเทียบทุกคู่ของ 10,000 ฉบับต้องใช้เวลากว่า 69 ชั่วโมง จึงควรใช้ระบบ Retrieve-then-rerank โดยใช้ Bi-encoder ดึงเอกสาร top-k อย่างรวดเร็ว แล้วใช้ Cross-encoder ทำการ rerank
D. Bi-encoder ไม่สามารถคำนวณเวกเตอร์เอกสารแบบ offline ล่วงหน้าได้ จึงต้องทำการ encode เอกสารทั้ง 10,000 ฉบับใหม่ทุกครั้งเมื่อมี query เข้ามา

**Q13** [W9] [SBERT Architecture] [difficulty: easy]
ในการฝึก Sentence-BERT (SBERT) ด้วยงาน Natural Language Inference (NLI) บนสถาปัตยกรรมโมเดล `bert-base` ($d=768$) สถาปัตยกรรมจะนำ sentence embeddings $u$ และ $v$ มาประกอบกันเป็นเวกเตอร์ $c$ ก่อนส่งเข้าสู่ Classification Head ข้อใดระบุรูปแบบของเวกเตอร์ $c$ และขนาดมิติ (dimension) ที่เข้าสู่ softmax layer ได้**ถูกต้อง**?
A. $c = [u; v]$ มีขนาดมิติเท่ากับ 1,536
B. $c = [u; v; |u - v|]$ มีขนาดมิติเท่ากับ 2,304
C. $c = [u; v; u \cdot v]$ มีขนาดมิติเท่ากับ 1,537
D. $c = [|u - v|]$ มีขนาดมิติเท่ากับ 768

**Q14** [W9] [Mean Pooling Implementation] [difficulty: hard]
ในการแปลง Token Embeddings ขนาด $[B, L, D]$ ให้เป็น Sentence Embedding ด้วยวิธี Mean Pooling หากโปรแกรมเมอร์ใช้คำสั่งหาค่าเฉลี่ย `torch.mean(token_embeddings, dim=1)` โดยตรงโดย**ไม่ได้**นำ `attention_mask` มาคูณคำนวณ จะเกิดผลกระทบใดขึ้นตามหลักการและโค้ดสาธิตของ Mean Pooling?
A. โปรแกรมจะเกิด runtime error ทันที เนื่องจากขนาดมิติของ token_embeddings ไม่สอดคล้องกับขนาดของ batch
B. เวกเตอร์ผลลัพธ์ของประโยคจะมีค่าเท่ากับเวกเตอร์ของ `[CLS]` token เสมอ
C. เวกเตอร์ของ positional embeddings จะถูกลบล้าง ทำให้โมเดลสูญเสียข้อมูลลำดับคำไปทั้งหมด
D. เวกเตอร์ของ `[PAD]` tokens ท้ายประโยคจะถูกนำมาร่วมหาค่าเฉลี่ยด้วย ทำให้เวกเตอร์ประโยคถูกเจือปนและบิดเบือนไปจากความหมายจริง

**Q15** [W9] [Contrastive Losses: Triplet Loss] [difficulty: medium]
พิจารณาฟังก์ชัน Triplet Loss:
$$L = \max(0, d(a, p) - d(a, n) + \alpha)$$
เมื่อ $a$ คือ anchor, $p$ คือ positive, $n$ คือ negative และ $\alpha > 0$ คือ margin ค่า loss จะกลายเป็น 0 (Zero-loss condition) โดยไม่มี gradient เกิดขึ้นเพื่ออัปเดตโมเดล เมื่อตัวอย่าง negative สอดคล้องกับเงื่อนไขใด?
A. $d(a, n) \ge d(a, p) + \alpha$ (จัดเป็น Easy negative)
B. $d(a, p) < d(a, n) < d(a, p) + \alpha$ (จัดเป็น Semi-hard negative)
C. $d(a, n) < d(a, p)$ (จัดเป็น Hard negative)
D. $d(a, p) - d(a, n) = \alpha$ พอดีเท่านั้น

**Q16** [W9] [Loss Functions Comparison] [difficulty: hard]
เมื่อเปรียบเทียบการทำงานระหว่าง Margin Ranking Loss กับ Triplet Loss ในการฝึก Sentence Embeddings ข้อใดสรุปความสัมพันธ์ของ Loss ทั้งสองได้**ถูกต้อง**ตามผลการทดลอง?
A. Margin Ranking Loss และ Triplet Loss จะให้ค่า slack เท่ากันเสมอในทุกกรณี ไม่ว่าจะเลือกระยะห่างแบบ Euclidean หรือ Cosine distance
B. หากนิยาม Triplet Loss โดยใช้ Cosine Distance ($d = 1 - \cos$) จะได้ค่า loss เท่ากับ Margin Ranking Loss ที่ใช้ Cosine Similarity พอดี แต่หากเปลี่ยนไปใช้ Euclidean Distance ค่า slack จะเปลี่ยนไป แสดงว่า metric เป็นส่วนหนึ่งของพฤติกรรม loss
C. Margin Ranking Loss จะให้ค่า loss เป็น 0 ได้ก็ต่อเมื่อ $\text{sim}(a, p) = 1.0$ และ $\text{sim}(a, n) = 0.0$ เท่านั้น
D. Triplet Loss ที่ใช้ Euclidean Distance ให้ผลการจัดอันดับเทียบเท่ากับ Inner Product เสมอแม้ไม่ได้ทำ Normalization

**Q17** [W9] [Vector Normalization & Metric Equivalence] [difficulty: medium]
ในการทำ Vector Search ข้อใดอธิบายความสัมพันธ์ทางคณิตศาสตร์ระหว่าง Inner Product (IP), Cosine Similarity และ Euclidean Distance ($L_2$) ได้**ถูกต้องที่สุด**?
A. เวกเตอร์สองตัวที่มีทิศทางชี้ไปทางเดียวกัน ($\cos = 1$) จะมีระยะห่างแบบ $L_2$ เท่ากับ 0 เสมอ แม้ว่าจะไม่ได้ผ่านการทำ Normalization ก็ตาม
B. การค้นหาด้วย Cosine Similarity จะให้ผลการจัดอันดับแตกต่างจากการค้นหาด้วย $L_2$ เสมอ ไม่ว่าเวกเตอร์จะถูก Normalize หรือไม่ก็ตาม
C. การใช้ FAISS IndexFlatIP สามารถใช้แทน Cosine Similarity ได้โดยตรงกับเวกเตอร์ดิบใดๆ โดยไม่จำเป็นต้องผ่าน $L_2$-normalization ล่วงหน้า
D. เมื่อเวกเตอร์ทั้งหมดผ่านการทำ $L_2$-normalization เป็น unit vectors แล้ว การจัดอันดับด้วยระยะห่าง $L_2$ จะให้ผลลัพธ์เหมือนกับการจัดอันดับด้วย Inner Product และ Cosine Similarity เนื่องจาก $\|u - v\|^2 = 2 - 2\cos(u, v)$

**Q18** [W9] [Vector Database Memory] [difficulty: medium]
หากต้องการจัดเก็บ sentence embeddings จำนวน 5,000,000 (5 ล้าน) เวกเตอร์ ซึ่งสร้างจากโมเดล BERT-base ($d = 768$) โดยแต่ละมิติจัดเก็บด้วยชนิดข้อมูล 32-bit floating point (`float32`) จะต้องใช้พื้นที่หน่วยความจำ RAM สำหรับเก็บตัวเวกเตอร์ดิบอย่างเดียว (ไม่รวม overhead ของ index) กี่กิกะไบต์ (GB โดยอิงฐานสิบ $10^9$ ไบต์)?
A. 3.84 GB
B. 7.68 GB
C. 15.36 GB
D. 30.72 GB

**Q19** [W9] [Information Retrieval Metrics] [difficulty: hard]
ในการประเมินประสิทธิภาพของระบบ Semantic Search ด้วยชุดคำถามทดสอบจำนวน 3 queries โดยอันดับ (rank) ของเอกสารที่ตรงกับความต้องการ (relevant document) ฉบับแรกที่ระบบค้นพบในแต่ละ query มีดังนี้:
- Query ที่ 1: พบคำตอบที่ถูกต้องฉบับแรกอยู่ที่อันดับ 1
- Query ที่ 2: พบคำตอบที่ถูกต้องฉบับแรกอยู่ที่อันดับ 2
- Query ที่ 3: พบคำตอบที่ถูกต้องฉบับแรกอยู่ที่อันดับ 4
ค่า Mean Reciprocal Rank (MRR) ของระบบนี้มีค่าเท่ากับเท่าใด?
A. 0.4286
B. 0.5833
C. 0.7500
D. 0.8333

**Q20** [W9] [Multiple Negatives Ranking Loss & In-batch Negatives] [difficulty: medium]
ในการฝึกโมเดล Bi-encoder ด้วยเทคนิค Multiple Negatives Ranking Loss (MNRL หรือ InfoNCE) และ In-batch Negatives ข้อใดอธิบายบทบาทของขนาด Batch Size และ Temperature Parameter ($\tau$) ได้**ถูกต้อง**?
A. การเพิ่มขนาด Batch Size ช่วยเพิ่มจำนวนตัวอย่าง Negative ให้กับแต่ละ Query ได้โดยตรงโดยไม่ต้องสุ่มตัวอย่างเพิ่ม และการลดค่า $\tau$ ให้เล็กลงจะทำให้การกระจายความน่าจะเป็นชันขึ้นและขับเน้นให้โมเดลแยก positive ออกจาก negative ได้เด็ดขาดยิ่งขึ้น
B. In-batch Negatives กำหนดให้ตัวอย่างทั้งหมดใน batch เดียวกันต้องเป็นประโยคที่มีความหมายคล้ายคลึงกัน จึงจำเป็นต้องใช้ค่า $\tau$ ที่สูงมากเพื่อควบคุมขนาดของ gradient
C. การลดค่า $\tau$ ให้ต่ำลงมากๆ จะทำให้โมเดลคำนวณ Cross-entropy loss ได้ค่าสูงขึ้นเสมอ ไม่ว่าค่า similarity ของ positive จะสูงกว่า negative หรือไม่
D. Multiple Negatives Ranking Loss สามารถนำไปใช้ได้เฉพาะกับโมเดลสถาปัตยกรรม Cross-encoder เท่านั้น ไม่สามารถใช้ร่วมกับ Bi-encoder ได้

**F2** [W9] [Contrastive Losses: Triplet Loss Calculation]
ในการฝึกโมเดล Sentence Embedding ด้วย Triplet Loss ตามสูตร $L = \max(0, d(a, p) - d(a, n) + \alpha)$
กำหนดให้ระยะห่างแบบ Euclidean ระหว่าง anchor ($a$) กับ positive ($p$) มีค่า $d(a, p) = 0.087$ และมีตัวอย่าง negative สองรูปแบบดังนี้:
1. ตัวอย่าง $n_1$ เป็น easy negative มีระยะห่าง $d(a, n_1) = 0.812$ กำหนดค่า margin $\alpha = 0.5$ ค่า Triplet Loss ที่คำนวณได้จะมีค่าเท่ากับ ___
2. ตัวอย่าง $n_2$ เป็น hard negative มีระยะห่าง $d(a, n_2) = 0.035$ โดยใช้ margin $\alpha = 0.5$ ค่า Triplet Loss ที่คำนวณได้จะมีค่าเท่ากับ ___

### Answer key
- **Q11**: A — เวกเตอร์ของ raw BERT กระจุกใน narrow cone ทำให้ค่า cosine similarity มี floor สูง (min ~0.457) และ [CLS] แย่กว่า mean pooling จนเกิด ranking inversion [Section 4 Demo 2/2b]
- **Q12**: C — Cross-encoder 10,000 ฉบับมีคู่คำนวณ $10,000 \times 9,999 / 2 \approx 5 \times 10^7$ คู่ ใช้เวลา $5 \times 10^7 \times 5\text{ ms} \approx 69.4\text{ ชม.}$ จึงต้องใช้ Bi-encoder ในการ retrieve แล้วใช้ Cross-encoder rerank [Section 1, Section 3.2]
- **Q13**: B — SBERT รวมเวกเตอร์แบบ $[u; v; |u - v|]$ ทำให้ขนาดมิติของ BERT-base ($d=768$) เข้าสู่ classification head เป็น $768 \times 3 = 2,304$ [Section 1, Section 3.1]
- **Q14**: D — หากไม่นำ `attention_mask` มาคูณเพื่อคัดกรอง เวกเตอร์ของ `[PAD]` tokens จะถูกนำมาร่วมหาค่าเฉลี่ยในมิติความยาวประโยคทำให้ผลลัพธ์ผิดเพี้ยน [Section 3.10, Section 4 Demo 3]
- **Q15**: A — เงื่อนไข Zero-loss condition เกิดขึ้นเมื่อ $d(a, p) - d(a, n) + \alpha \le 0 \iff d(a, n) \ge d(a, p) + \alpha$ ซึ่งนิยามเป็น Easy negative [Section 1, Section 3.5]
- **Q16**: B — การใช้ Triplet Loss ร่วมกับ Cosine Distance ($d = 1 - \cos$) จะได้ค่าเท่ากับ Margin Ranking Loss บน Cosine Similarity พอดี แต่ Euclidean Distance จะให้ค่า slack ที่ต่างออกไป แสดงว่า metric เป็นส่วนหนึ่งของ loss [Section 3.6]
- **Q17**: D — เมื่อทำ $L_2$-normalization บนเวกเตอร์หนึ่งหน่วย $\|u - v\|^2 = 2 - 2\cos(u, v)$ ส่งผลให้อันดับของ Euclidean distance สอดคล้องกับ Cosine Similarity และ Inner Product โดยตรง [Section 3.9]
- **Q18**: C — คำนวณจาก $5,000,000 \times 768 \times 4\text{ bytes} = 15.36 \times 10^9\text{ bytes} = 15.36\text{ GB}$ [Section 3.3]
- **Q19**: B — คำนวณ Reciprocal Rank ได้ $1/1 = 1$, $1/2 = 0.5$, $1/4 = 0.25$ หาค่าเฉลี่ย $\text{MRR} = (1 + 0.5 + 0.25) / 3 = 1.75 / 3 \approx 0.5833$ [Section 3.8]
- **Q20**: A — Batch size ที่ใหญ่ขึ้นช่วยเพิ่มจำนวน in-batch negatives โดยไม่ต้องสร้างตัวอย่างเพิ่ม และค่า $\tau$ ที่ต่ำจะขยายสเกล logit ช่วยให้โมเดลแยก positive ได้เด็ดขาดขึ้น [Section 1, Section 3.7]
- **F2**: 0 (หรือ 0.0), 0.552 — ช่องแรก: $\max(0, 0.087 - 0.812 + 0.5) = \max(0, -0.225) = 0$; ช่องสอง: $\max(0, 0.087 - 0.035 + 0.5) = 0.552$ [Section 3.5]

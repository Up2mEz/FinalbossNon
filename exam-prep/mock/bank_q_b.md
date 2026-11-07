### n11-L3-03  [unit 11] [level 3] [mcq]  topic: GRPO advantage calculation
GRPO กลุ่ม G = 4 ได้ rewards [2, 4, 6, 8] ใช้ A_i = (r_i - mu) / sigma โดย sigma = sqrt( (1/G) x sum (r_i - mu)^2 ) กำหนด sqrt(5) = 2.236 ค่า advantage ของคำตอบที่ได้ reward 8 ใกล้เคียงข้อใดที่สุด
A) 1.162
B) 0.600
C) 3.000
D) 1.342

### n11-L3-04  [unit 11] [level 3] [mcq]  topic: Choosing prompting technique per task
มี 3 งาน (X) จัดประเภท sentiment ให้ได้ label รูปแบบเดียวกันทุกครั้ง (Y) โจทย์หลายขั้นที่ถามตรง ๆ แล้วตอบผิด เช่น 23 - 20 + 6 ในเรื่องแอปเปิล (Z) คูณเลขหลายหลักที่ต้องได้ผลลัพธ์ exact การจับคู่งานกับเทคนิคข้อใดเหมาะสมที่สุดตามสไลด์
A) X: Few-shot, Y: Chain-of-Thought, Z: Program-of-Thought
B) X: Chain-of-Thought, Y: Few-shot, Z: Program-of-Thought
C) X: Program-of-Thought, Y: Chain-of-Thought, Z: Few-shot
D) X: Few-shot, Y: Program-of-Thought, Z: Chain-of-Thought

### n11-L3-05  [unit 11] [level 3] [mcq]  topic: Hardware cost of long CoT
ระหว่างให้บริการโมเดล reasoning ความยาว reasoning trace เพิ่มจาก 1,000 เป็น 3,000 tokens โดยที่ KV cache ตอน 1,000 tokens เท่ากับ 5 GB ใช้โมเดลตามสไลด์ (attention compute รวมโตแบบ quadratic, KV cache โตแบบ linear) ข้อใดถูกต้อง
A) KV cache 45 GB, attention compute x9, decoding เป็น memory-bandwidth bound
B) KV cache 15 GB, attention compute x9, decoding เป็น memory-bandwidth bound
C) KV cache 15 GB, attention compute x3, decoding เป็น memory-bandwidth bound
D) KV cache 15 GB, attention compute x9, decoding เป็น compute bound

### n11-L3-06  [unit 11] [level 3] [mcq]  topic: Limitations of SFT and exposure bias
ข้อใดอธิบายข้อจำกัดของ SFT และ exposure bias ผิด
A) Costly and slow: demonstrations คุณภาพสูง เช่น โค้ดหรือคำตอบทางการแพทย์ มีต้นทุนสูงเมื่อต้องทำในปริมาณมาก
B) No negative feedback: SFT บอกโมเดลว่าควรทำอะไร แต่ไม่ได้ลงโทษสิ่งที่ไม่ควรทำอย่างชัดเจน
C) Exposure bias: ตอน train โมเดลป้อน token ที่ตัวเองสร้างกลับเข้าไป แต่ตอน inference ถูกป้อน token จริงจากมนุษย์ (teacher forcing)
D) Distribution shift: SFT เห็นแต่คำตอบที่สมบูรณ์ จึงไม่เคยเห็นการแก้ไขหลังพลาด error จึงสะสมระหว่าง generation

### n11-L4-01  [unit 11] [level 4] [mcq]  topic: DPO relative margin edge case
ระหว่างฝึก DPO ด้วย beta = 0.1 พบว่า log pi(yw|x) ของ chosen ลดลงจาก log pi_ref = -20 เป็น -21 และ log pi(yl|x) ของ rejected ลดลงจาก log pi_ref = -30 เป็น -37 กำหนด e^-0.6 = 0.5488 ข้อใดถูกต้อง
A) loss ประมาณ 0.4375 ต่ำกว่า ln 2 แม้ prob ของ chosen ลดลง เพราะ DPO ดูส่วนต่างสัมพัทธ์ (-1) - (-7) = +6
B) loss ประมาณ 1.0375 ซึ่งสูงกว่า ln 2 เพราะ prob ของ chosen ลดลงจาก ref จึงถูกลงโทษโดยตรง
C) loss เท่ากับ ln 2 = 0.6931 เพราะ chosen และ rejected ลดลงทั้งคู่ ผลจึงหักล้างกันเป็นศูนย์
D) loss ประมาณ 0.6457 ซึ่งเท่ากับ sigma(0.6) ที่ได้จาก beta x 6 ต่ำกว่า ln 2 เล็กน้อย

### n11-L4-02  [unit 11] [level 4] [mcq]  topic: Model memory PPO vs GRPO vs DPO
ทีมมี GPU 80 GB (GB = 10^9 bytes) โมเดลทุกตัวขนาด 13B parameters เก็บแบบ FP16 (2 bytes ต่อ parameter) นับเฉพาะ weights ของโมเดลที่ต้องโหลดพร้อมกัน ไม่รวม optimizer states และ activations GRPO ที่ใช้ตัดเฉพาะ Critic ออกจากชุดของ PPO และใช้ neural reward model ข้อใดสรุปได้ถูกต้อง
A) PPO 104 GB, GRPO 52 GB, DPO 52 GB จึงมีเพียง PPO ที่เกิน 80 GB
B) PPO 104 GB, GRPO 104 GB, DPO 52 GB จึงมีเพียง DPO ที่โหลดได้
C) PPO 104 GB, GRPO 78 GB, DPO 52 GB จึงมีเพียง PPO ที่เกิน 80 GB
D) PPO 104 GB, GRPO 78 GB, DPO 26 GB จึงโหลดได้ทั้ง GRPO และ DPO

### n11-L4-03  [unit 11] [level 4] [mcq]  topic: Self-consistency temperature and cost
ทีม A ใช้ self-consistency ตั้ง temperature > 0 sample 7 paths ได้ final answers 18, 14, 18, 26, 18, 14, 26 ทีม B ทำ self-consistency 7 paths เช่นกันแต่ตั้ง temperature = 0 (greedy) แต่ละ path สร้างประมาณ 300 tokens ข้อใดถูกต้อง
A) ทีม A เลือกคำตอบไม่ได้ เพราะไม่มีคำตอบใดได้เสียงเกินครึ่งคือ 4 จาก 7 จึงต้องสุ่มเพิ่มใหม่ทั้งหมด
B) ทีม B ให้ผลดีกว่า เพราะ 7 paths ตรงกันทั้งหมดแสดงว่ามั่นใจสูง และจ่าย token น้อยกว่าทีม A
C) ทีม A ต้องฝึก verifier model เพิ่มเพื่อเลือก 18 จึงจะถือว่าเป็น self-consistency ที่ถูกต้อง
D) ทีม A ได้ 18 (3 จาก 7 เสียง) ส่วนทีม B ไม่มีความหลากหลายให้โหวต แต่ยังจ่าย 7 x 300 = 2,100 tokens เทียบ 300 ของ CoT เดียว

### n11-L4-04  [unit 11] [level 4] [mcq]  topic: Self-consistency vs reflection latency
วิธี S: self-consistency 5 paths (ไม่นับเวลา vote) วิธี R: reflection ที่สร้างร่างแรก 1 ครั้ง แล้วทำ 2 รอบของ critique ตามด้วย refine แต่ละ LLM call ใช้ 10 วินาที และฮาร์ดแวร์รองรับ 5 requests พร้อมกันได้ ข้อใดถูกต้อง
A) S ใช้ 5 calls ประมาณ 50 วินาที เพราะต้องรอทุก path ส่วน R ใช้ 5 calls ประมาณ 10 วินาที เพราะ refine เร็วกว่า
B) ทั้งสองใช้ 5 calls โดย S ใช้ประมาณ 10 วินาที (paths อิสระ ขนานได้) ส่วน R ประมาณ 50 วินาที (แต่ละ call รอผลก่อนหน้า)
C) ทั้งสองใช้ 5 calls และใช้เวลาประมาณ 50 วินาทีเท่ากัน เพราะ call ละ 10 วินาทีและต้องรันตามลำดับเหมือนกัน
D) S ใช้ 5 calls ประมาณ 10 วินาที ส่วน R ใช้ 3 calls (generate, critique, refine) ประมาณ 30 วินาที เพราะทำ refine รอบเดียว

### n11-L4-05  [unit 11] [level 4] [mcq]  topic: GRPO scale invariance of advantage
GRPO กลุ่ม G = 4 โดย rule checker ให้ reward 0 หรือ 1 แล้วมีคำตอบถูกเพียงตัวเดียว (rewards [1, 0, 0, 0]) ต่อมาทีมเปลี่ยน checker ให้ reward 0 หรือ 6 (rewards [0, 0, 0, 6]) ใช้ A_i = (r_i - mu)/sigma โดย sigma หารด้วย G (sqrt(3) = 1.732) advantage ของคำตอบที่ถูกตัวนั้นหลังเปลี่ยนเป็นเท่าใด
A) 10.39
B) 4.50
C) 1.732
D) 0.289

### n11-L4-06  [unit 11] [level 4] [mcq]  topic: R1-Zero style RL limits and thinker trade-off
ทีมอยากใช้สูตรแบบ DeepSeek R1-Zero (GRPO, ข้าม SFT, rule-based reward ที่เทียบคำตอบสุดท้ายกับ ground truth บวก format reward) ฝึกบอทตอบอีเมลลูกค้าให้สุภาพและมีประโยชน์ ข้อใดวิเคราะห์ถูกต้องที่สุด
A) อีเมลไม่มี ground truth ที่ rule ตรวจได้ จึงขาด verifiable reward (สไลด์ยืนยันแค่ math/code) และ Thinker ใช้ token กับ TTFT สูง งานง่ายควรใช้ Helper
B) ทำไม่ได้เพราะ GRPO ต้องมี Critic model เพื่อประมาณ baseline ของแต่ละคำตอบ ซึ่งงานอีเมลสร้างไม่ได้ในทางปฏิบัติเลย
C) ต้องเก็บ human step-by-step demonstrations ด้วย SFT ก่อนเสมอ เพราะ R1-Zero เริ่มจาก aligned model ที่ผ่าน SFT มาแล้วเท่านั้น
D) ทำได้เท่ากันทุกงาน เพราะ format reward ของ <think> tag ก็เพียงพอจะเป็น reward ของงานภาษาธรรมชาติทุกแบบที่ต้องการ

### n11-L1-F1  [unit 11] [level 1] [fill]  topic: In-context learning
การที่ LLM เรียนงานใหม่ตอน inference จากตัวอย่างใน prompt โดยไม่อัปเดต weights เรียกว่า ___ (ตัวย่อ ___)

### n11-L2-F2  [unit 11] [level 2] [fill]  topic: Reflection and self-critique
Reflection loop มีขั้นตอน Generate, ___, Refine และได้ผลเพราะการ ___ text ง่ายกว่าการสร้าง text ที่สมบูรณ์จากศูนย์

### n11-L3-F3  [unit 11] [level 3] [fill]  topic: DPO loss calculation
DPO ใช้ beta = 0.1 log ratio ของ chosen = +3 และของ rejected = -2 ได้ logit ที่ใส่ใน sigma เท่ากับ ___ และเมื่อกำหนด e^-0.5 = 0.6065 ค่า loss = -ln sigma(logit) เท่ากับ ___ (ทศนิยม 4 ตำแหน่ง)

### n11-L4-F4  [unit 11] [level 4] [fill]  topic: GRPO advantage with G = 5
GRPO กลุ่ม G = 5 ได้ rewards [1, 1, 1, 1, 0] ใช้ sigma แบบหารด้วย G ได้ sigma = ___ advantage ของคำตอบที่ผิด (reward 0) = ___ และ advantage ของคำตอบที่ถูก (reward 1) = ___

### n12-L1-01  [unit 12] [level 1] [mcq]  topic: BM25 parameter b
ในสูตร BM25 พารามิเตอร์ตัวใดควบคุมความแรงของ length normalization?
A) k1 (TF saturation)
B) b
C) n_t (จำนวน doc ที่มีคำนั้น)
D) epsilon (ε) ของ rank_bm25

### n12-L1-02  [unit 12] [level 1] [mcq]  topic: Sparse vs Dense
จุดอ่อนหลักของ BM25 (sparse retrieval) ตามสไลด์คืออะไร?
A) ให้คะแนนต่ำกับคำหายากและรหัสสินค้า
B) ช้ามากเพราะต้องทำ ANN search ทั้ง corpus
C) Vocabulary mismatch: คำต่างกันแต่ความหมายเดียวกันจะ match ไม่ได้
D) Hallucinated relevance: ดึงเอกสารที่ใกล้เคียงเชิงความหมายแต่ผิดข้อเท็จจริง

### n12-L1-03  [unit 12] [level 1] [mcq]  topic: Cross-encoder score
Scoring head ของ Cross-encoder (ดึง [CLS] embedding แล้วผ่าน linear layer) ส่งออกคะแนนชนิดใด?
A) raw scalar logit
B) ความน่าจะเป็นในช่วง 0 ถึง 1 เสมอ
C) cosine similarity ในช่วง -1 ถึง 1
D) อันดับ (rank) ที่เป็นจำนวนเต็ม

### n12-L1-04  [unit 12] [level 1] [mcq]  topic: GraphRAG triple
Triple ที่ใช้สร้าง Knowledge Graph ในขั้น extraction มีรูปแบบใด?
A) (Query, Document, Score)
B) (Entity, Community, Summary)
C) (Chunk, Embedding, Index)
D) (Subject, Relation, Object)

### n12-L1-05  [unit 12] [level 1] [mcq]  topic: Lost in the Middle
ตามผลของ Lost in the Middle (Liu et al.) เมื่อย้ายเอกสารที่เกี่ยวข้องไปตำแหน่งต่าง ๆ ใน context ยาว accuracy ต่ำที่สุดเมื่อเอกสารอยู่ที่ใด?
A) ต้น context
B) กลาง context
C) ท้าย context
D) ไม่มีรูปแบบ accuracy เท่ากันทุกตำแหน่ง

### n12-L1-06  [unit 12] [level 1] [mcq]  topic: Semantic chunking
Semantic shift detection chunking ตัดข้อความเป็น chunk ที่ตำแหน่งใด?
A) ทุกจำนวนตัวอักษรคงที่ตาม chunk_size
B) ทุก heading หรือ section ที่พบในเอกสารต้นฉบับ
C) จุดที่ 1 - cosine ของประโยคติดกันพุ่งเกิน τ
D) จุดที่ BM25 score ของประโยคสูงสุดเทียบกับ query

### n12-L2-01  [unit 12] [level 2] [mcq]  topic: BM25 TF ceiling
ปรับ BM25 จาก (k1 = 1.5, b = 0.75) เป็น (k1 = 3.0, b = 0) ส่วน TF มี ceiling (asymptote เมื่อ f เพิ่มไม่จำกัด) เป็นเท่าใด?
A) 2.50
B) 3.00
C) 4.50
D) 4.00

### n12-L2-02  [unit 12] [level 2] [mcq]  topic: BM25 TF diminishing returns
BM25 (k1 = 1.5, b = 0) ถ้าคำหนึ่งปรากฏ f = 1 ครั้ง แล้วเพิ่มเป็น f = 2 ครั้ง ส่วน TF เพิ่มขึ้นประมาณกี่เปอร์เซ็นต์?
A) +42.9%
B) +100%
C) +57.1%
D) +14.3%

### n12-L2-03  [unit 12] [level 2] [mcq]  topic: RRF k effect
ใช้ RRF ที่ k = 60 คะแนนของเอกสารเมื่อขยับจากอันดับ 1 ไปอันดับ 2 ลดลงประมาณกี่เปอร์เซ็นต์ (เทียบกับคะแนนอันดับ 1)?
A) 50%
B) 1.6%
C) 16%
D) 0.016%

### n12-L2-04  [unit 12] [level 2] [mcq]  topic: ColBERT
ข้อใดอธิบาย ColBERT ได้ถูกต้องตามสไลด์?
A) encode query กับ document ร่วมกันใน tower เดียว จึงแม่นสูงสุดเทียบเท่า cross-encoder
B) ใช้ inverted index แบบ BM25 ต่อกับ dense จึงเร็วกว่า bi-encoder ประมาณ 100 เท่า
C) Late interaction ด้วย MaxSim: ถูกกว่า cross-encoder ~100x, คุณภาพอยู่ระหว่างกลาง
D) ทำ cross-attention เต็มรูปแบบทุก token จึงไม่ถูกกว่า cross-encoder

### n12-L2-05  [unit 12] [level 2] [mcq]  topic: GraphRAG local vs global
คำถามใดเหมาะกับ GraphRAG Global search (map-reduce ข้าม community summaries) มากที่สุด?
A) What are the key achievements of Sam Altman?
B) เอกสารฉบับใดมีเลข invoice #INV-2024-7832
C) จุดเดือดของน้ำที่ความดันบรรยากาศมาตรฐานคือเท่าใด
D) What are the top themes across all 500 reports?

### n12-L2-06  [unit 12] [level 2] [mcq]  topic: Bookend ordering
ผล rerank ได้ chunk อันดับ 1 ถึง 6 (C1..C6) ตาม Optimal Bookend Prompt Structure ลำดับใน prompt (ก่อนถึง Question) ควรเป็นข้อใด?
A) C1, C3, C4, C5, C6, C2
B) C1, C2, C3, C4, C5, C6
C) C2, C3, C4, C5, C6, C1
D) C1, C2, C6, C5, C4, C3

### n12-L3-01  [unit 12] [level 3] [mcq]  topic: BM25 length normalization
BM25 ตั้ง k1 = 1.2, b = 0.75, avgdl = 200 คำค้นปรากฏ f = 4 ครั้งในเอกสาร A (ยาว 100 คำ) และเอกสาร B (ยาว 400 คำ) ส่วน TF ของ A เป็นกี่เท่าของ B (ประมาณ)?
A) 4.00
B) 2.80
C) 1.28
D) 0.78

### n12-L3-02  [unit 12] [level 3] [mcq]  topic: RRF ranking
ใช้ RRF ที่ k = 60 เอกสาร X อยู่อันดับ 1 ใน BM25 เท่านั้น, Y อยู่อันดับ 4 ใน BM25 และอันดับ 6 ใน dense, Z อยู่อันดับ 3 ในทั้งสองระบบ ลำดับหลัง fuse จากมากไปน้อยคือข้อใด?
A) X, Z, Y
B) Z, Y, X
C) X, Y, Z
D) Y, Z, X

### n12-L3-03  [unit 12] [level 3] [mcq]  topic: Two-stage pipeline latency
ตามตัวอย่างในสไลด์: BM25 ~5 ms ขนานกับ dense ~10 ms, RRF ~1 ms, cross-encoder rerank 80 candidates ~200 ms. ถ้าเวลา cross-encoder แปรตามจำนวน pair เป็นเส้นตรง แล้วลด candidates เหลือ 20 pairs เวลารวมของ Stage 1 ถึง 3 เป็นเท่าใด?
A) 211 ms
B) 66 ms
C) 56 ms
D) 61 ms

### n12-L3-04  [unit 12] [level 3] [mcq]  topic: CLIP failure modes
ทีมใช้ CLIP ทำ multimodal RAG แล้วพบ: query 'broken phone screen' ได้โทรศัพท์ปกติ, 'bird without wings' ได้นกปกติ, 'dog chasing cat' ได้รูปหมากับแมวที่ยืนห่างกัน ข้อใดอธิบายและแนะนำถูกต้อง?
A) เป็น failure mode (attribute, negation, relation) ต้องทดสอบ compositional queries
B) เกิดจาก image encoder เป็น ViT ซึ่งแบ่งภาพเป็น patch จึงต้องเปลี่ยนมาใช้ ResNet แทน
C) แก้ได้เสมอโดยเปลี่ยน prompt template เป็น 'A photo of a {object}.' อย่างเดียว
D) เกิดจาก batch size N = 32,768 ใหญ่เกินไปจน negatives มากเกินไป

### n12-L3-05  [unit 12] [level 3] [mcq]  topic: GraphRAG multi-hop
ถาม 'บริษัทใดที่อดีตผู้บริหาร OpenAI ไปก่อตั้งภายหลัง' แล้ว vector RAG ตอบพลาด ข้อใดอธิบายสาเหตุและทางแก้ได้ถูกต้อง?
A) embedding dimension น้อยเกินไป จึงต้องเพิ่มเป็น 3072 มิติเพื่อให้จับความสัมพันธ์ได้
B) BM25 IDF ของคำว่า OpenAI ติดลบ จึงต้องเปลี่ยนเป็นสูตร ln(1 + ...) ของ Lucene
C) cosine ดึง chunk แยกจุด ต่อ chain A->B->C ไม่ได้ ต้องใช้ Text-to-Cypher เดิน graph
D) เกิดจาก lost in the middle ของ context ยาว จึงต้องใช้ bookend จัดลำดับ chunk ใหม่

### n12-L3-06  [unit 12] [level 3] [mcq]  topic: Lost in the Middle design rules
ข้อใดขัดกับ 4 production rules ที่สไลด์สรุปจาก Lost in the Middle?
A) rerank ด้วย cross-encoder ก่อนจัดลำดับ chunk
B) เพิ่ม K จาก 5 เป็น 50 เพราะ chunk ยิ่งมากยิ่งตอบดีขึ้น
C) วางหลักฐานสำคัญที่สุดไว้ที่ต้นและท้าย prompt
D) สรุป chunk ยาวก่อน synthesis สำหรับคำถามซับซ้อน (context distillation)

### n12-L4-01  [unit 12] [level 4] [mcq]  topic: BM25 keyword stuffing
ใช้ setup ของสไลด์: query 'deep learning', IDF(deep) = 1.50, IDF(learning) = 2.50, avgdl = 100, k1 = 1.5, b = 0.75. เอกสารยาว 100 คำที่มี deep และ learning อย่างละ 30 ครั้งได้ราว 9.52 ถ้าผู้เขียนอัดคำเป็นอย่างละ 300 ครั้ง (ยังยาว 100 คำ) คะแนนเป็นเท่าใด?
A) ประมาณ 9.95 (+4.5%)
B) ประมาณ 95.2 (+900%)
C) ประมาณ 10.47 (+10%)
D) ประมาณ 9.52 (+0%)

### n12-L4-02  [unit 12] [level 4] [mcq]  topic: BM25 effect of b
avgdl = 100, k1 = 1.5, คำหนึ่งปรากฏ f = 3 ครั้งในเอกสาร 3 ฉบับยาว 50, 100 และ 500 คำ ถ้าเปลี่ยน b จาก 0 เป็น 1 ส่วน TF ของแต่ละฉบับเปลี่ยนอย่างไร?
A) ทั้งสามฉบับยังได้ 1.667 เท่าเดิม เพราะ b ไม่กระทบ TF เมื่อ f คงที่
B) ฉบับ 100 คำเพิ่มจาก 1.667 เป็น 2.500 (ถึงเพดาน k1 + 1)
C) ฉบับ 50 คำลดเหลือ 1.429 และฉบับ 500 คำเพิ่มเป็น 2.000
D) 100 คำคงที่ 1.667, 50 คำเพิ่มเป็น 2.000, 500 คำลดเหลือ 0.714

### n12-L4-03  [unit 12] [level 4] [mcq]  topic: RRF k edge case
เอกสาร P อยู่อันดับ 1 ในลิสต์ BM25 ลิสต์เดียว ส่วนเอกสาร Q อยู่อันดับ 3 ทั้งในลิสต์ BM25 และ dense เมื่อเปลี่ยนค่า k ของ RRF จาก 60 เป็น 0 การจัดอันดับ P กับ Q เป็นอย่างไร?
A) Q ชนะทั้งสองกรณี เพราะติดสองลิสต์ย่อมได้คะแนนรวมสูงกว่าเสมอ
B) P ชนะทั้งสองกรณี เพราะอันดับ 1 ได้คะแนนสูงสุดไม่ว่า k เป็นเท่าใด
C) ที่ k = 60 Q ชนะ (0.0317 เทียบ 0.0164) ที่ k = 0 P ชนะ (1.000 เทียบ 0.667)
D) ที่ k = 60 P ชนะ (0.0164 เทียบ 0.0159) ที่ k = 0 Q ชนะ (1.000 เทียบ 0.667)

### n12-L4-04  [unit 12] [level 4] [mcq]  topic: Reranker candidate pool
ระบบ: BM25 + dense -> RRF -> cross-encoder rerank เฉพาะ top K = 50 -> LLM รับ top 5. เอกสารที่มีคำตอบอยู่อันดับ 70 หลัง RRF จึงไม่ถึง LLM ถ้าเปลี่ยนเป็น cross-encoder รุ่นที่แม่นกว่า (ขนาดและ K เท่าเดิม) จะช่วยหรือไม่?
A) ช่วย เพราะ cross-encoder ที่แม่นกว่าจะค้นเจอเอกสารที่หลุดออกจากชุด K ได้เอง
B) ไม่ช่วย เพราะอันดับ 70 ไม่เคยเข้า reranker ต้องเพิ่ม K (แลก latency)
C) ช่วย เพราะ reranker ทำหน้าที่เพิ่ม recall ของ Stage 1
D) ไม่ช่วย เพราะ cross-encoder precompute ไม่ได้ จึงต้องเปลี่ยนไปใช้ bi-encoder แทนในขั้น rerank

### n12-L4-05  [unit 12] [level 4] [mcq]  topic: RAG system design
ระบบ QA ของสำนักงานกฎหมาย (~2M chunks) รับคำถาม 3 แบบ: (1) 'termination clause section 12.3', (2) 'ภาระของเราเมื่อลูกค้าผิดนัดคืออะไร', (3) 'สัญญากับบริษัท A ต่างจากบริษัท B ด้านความรับผิดอย่างไร' ออกแบบข้อใดสอดคล้องกับสไลด์มากที่สุด?
A) ใช้ dense retrieval อย่างเดียว เพราะข้อ (2) เป็นเชิงความหมาย ส่วนข้อ (3) ใช้ k ใหญ่ ๆ แทน
B) ใช้ BM25 อย่างเดียว เพราะเร็วที่สุดและ BEIR ชี้ว่าชนะ DPR ในทุก dataset
C) ยัดสัญญาทั้งหมดเข้า long context 1M tokens แล้วถามตรง ๆ ทุกข้อ
D) hybrid + rerank สำหรับข้อ (1)-(2) และ route ข้อ (3) ไป GraphRAG

### n12-L4-06  [unit 12] [level 4] [mcq]  topic: CLIP temperature
ใน batch N = 4 รูป i มี similarity กับ caption ที่ถูก 0.30 และกับ caption อื่นอีก 3 ตัว 0.20 เท่ากัน loss ฝั่ง image->text: -ln( exp(s_ii/τ) / Σ_j exp(s_ij/τ) ). เมื่อลด τ จาก 0.1 เป็น 0.01 loss ของรูป i เปลี่ยนจากประมาณเท่าใดเป็นเท่าใด?
A) 0.744 เป็น 0.000136
B) 0.744 เป็น 7.44
C) 1.386 คงที่ ไม่ขึ้นกับ τ
D) 0.475 เป็น 0.99986

### n12-L1-F1  [unit 12] [level 1] [fill]  topic: BM25 IDF
IDF(t) = ln((N - n_t + 0.5)/(n_t + 0.5)) ของ BM25 มีค่าเท่ากับ ___ พอดีเมื่อ n_t = ___

### n12-L2-F1  [unit 12] [level 2] [fill]  topic: Retriever vs reranker
ในสถาปัตยกรรมสองขั้น retriever (Stage 1) มุ่งเน้นค่า ___ ส่วน reranker มุ่งเน้นค่า ___

### n12-L3-F1  [unit 12] [level 3] [fill]  topic: BM25 TF calculation
BM25 ตั้ง k1 = 1.2 และ b = 0 คำหนึ่งปรากฏ f = 3 ครั้ง ส่วน TF = ___ (ทศนิยม 3 ตำแหน่ง) และเพดานของ TF เมื่อ f เพิ่มไม่จำกัดคือ ___

### n12-L4-F1  [unit 12] [level 4] [fill]  topic: RRF rank threshold
ใช้ RRF ที่ k = 60 เอกสารอันดับ 1 ในลิสต์เดียวได้ 1/61 เอกสารที่อยู่อันดับ r เท่ากันในทั้งสองลิสต์จะได้คะแนนมากกว่า (strictly) เมื่อ r มากที่สุดเท่ากับ ___ (จำนวนเต็ม) และที่ r = ___ คะแนนเท่ากันพอดี

### n13-L1-01  [unit 13] [level 1] [mcq]  topic: ReAct max_steps
ในโค้ด ReAct loop ตัวแปร `max_steps` ถูกใส่ไว้เพื่อวัตถุประสงค์ใด?
A) เพิ่มความแม่นยำของ Thought ในแต่ละรอบ
B) จำกัดความยาวของ Observation ที่ tool ส่งกลับมา
C) เป็น safety guard กัน agent วน loop ไม่รู้จบ
D) กำหนดจำนวน tool ทั้งหมดที่ agent เรียกใช้ได้ในระบบ

### n13-L1-02  [unit 13] [level 1] [mcq]  topic: JSON Schema analogy
สไลด์เปรียบ JSON Schema ของ tool กับอะไร?
A) ฟอร์มเปล่าที่บอกว่ามีช่องอะไร ชนิดใด และช่องไหน required
B) ฟอร์มที่กรอกแล้ว ซึ่งเก็บค่า argument จริงที่ LLM ส่งให้ tool ไปเรียกใช้
C) พจนานุกรมของ token ทั้งหมดใน vocabulary ที่ LLM เลือกได้ในแต่ละ step
D) บันทึกประวัติ Observation ของทุกรอบที่ผ่านมา

### n13-L1-03  [unit 13] [level 1] [mcq]  topic: Regex constraint DFA
ใน regex-constrained decoding เช่น `\d{4}-\d{2}-\d{2}` ขั้นแรกคือ compile regex ให้เป็นอะไรเพื่อหา valid next tokens?
A) Context-Free Grammar ที่มี stack ของวงเล็บ
B) โมเดล Transformer ขนาดเล็กอีกตัวหนึ่ง
C) Hash table ของทุก string ที่ regex รับได้
D) Deterministic Finite Automaton (DFA)

### n13-L1-04  [unit 13] [level 1] [mcq]  topic: Sequential pipeline weakness
จุดอ่อนหลักของ Sequential Pipeline (Research -> Writer -> Editor) ที่สไลด์ยกมาคือข้อใด?
A) Router ตัวเดียวเป็น single point of failure
B) Error cascade: ถ้า agent ต้นทางพลาด agent ถัดไปพลาดตาม
C) ต้องให้หลาย agent เถียงกันก่อนสรุป จึงมี token cost สูงมาก
D) ทุก agent ทำงานพร้อมกันจึงเกิดการแก้ state ชนกัน

### n13-L1-05  [unit 13] [level 1] [mcq]  topic: Memory taxonomy
Memory ประเภทใดเก็บความรู้หรือข้อเท็จจริงที่ compile แล้ว เช่น "Thailand's capital is Bangkok"?
A) Semantic memory
B) Episodic memory
C) Procedural memory
D) Working memory

### n13-L1-06  [unit 13] [level 1] [mcq]  topic: Reliability stack
ชั้นใดของ Reliability Stack (Defense in Depth) ที่ให้มนุษย์อนุมัติก่อนทำ action ที่เสี่ยงหรือทำลายข้อมูล?
A) Layer 1: Constrain
B) Layer 2: Validate
C) Layer 3: Recover
D) Layer 4: Escalate

### n13-L2-01  [unit 13] [level 2] [mcq]  topic: ReAct Table 1 deltas
จาก Table 1 (PaLM-540B) เมื่อเปลี่ยนจาก CoT (HotpotQA EM 29.4, FEVER Acc 56.3) มาเป็น ReAct (27.4, 60.9) คะแนนเปลี่ยนไปเท่าใด (HotpotQA / FEVER)?
A) +2.0 / -4.6
B) -2.0 / +4.6
C) -2.0 / +3.6
D) +1.7 / +2.0

### n13-L2-02  [unit 13] [level 2] [mcq]  topic: Tool call failure modes
LLM สร้าง tool call ชื่อ `web_lookup` ซึ่งไม่มีอยู่ใน tool registry ตามตาราง Failure Modes ของสไลด์ผลที่ถูกต้องคืออะไร?
A) Parser error แล้ว retry หรือ fail เหมือนกรณี malformed JSON
B) Tool not found แล้วส่ง error message กลับมา
C) ใช้ `max_steps` guard ตัด loop ทันที
D) Dispatcher เลือก tool ที่ชื่อใกล้เคียงที่สุดมารันแทน

### n13-L2-03  [unit 13] [level 2] [mcq]  topic: Topology comparison table S41
ตาราง Topology Comparison (S41) มีคอลัมน์ Latency / Complexity / Fault Tolerance / Best use topology ใดมี Latency = Low, Complexity = Low, Fault Tolerance = Medium และเหมาะกับ customer support / triage?
A) Sequential
B) Supervisor
C) Routing
D) Debate

### n13-L2-04  [unit 13] [level 2] [mcq]  topic: Mask strength depends on parser state
vocabulary 50,257 token: หลัง `{` ใน JSON มี token ที่ valid 69 ตัว (mask 99.86%) แต่ในสตริง `{"name": "A` มี valid 50,146 ตัว ข้อสรุปใดถูกต้อง?
A) mask ตัด token จำนวนเท่ากันทุก state เพราะ vocabulary มีขนาดคงที่
B) ในสตริงไม่มี token ถูก mask เลยเพราะ grammar อนุญาตเกือบทุกอย่าง
C) จำนวน token ที่ถูก mask ขึ้นกับ parser state ณ ขณะนั้น
D) จำนวน token ที่ valid ขึ้นกับขนาดของโมเดล ไม่ใช่ grammar

### n13-L2-05  [unit 13] [level 2] [mcq]  topic: Tool definition to call mapping
tool definition มี `parameters` = {type: object, properties: {query: string}, required: [query]} ใน tool call ที่ LLM สร้าง ส่วนของ definition นี้ตรงกับฟิลด์ใดและต้องมีลักษณะอย่างไร?
A) `arguments` ที่ต้องเป็น object โดยมี `query` เป็น string
B) `name` ที่ต้องมีค่าเป็นสตริง "query" ตรงกับชื่อ property ใน schema
C) `arguments` ที่ต้องคัดลอก type/properties/required ทั้งก้อนมาใส่
D) `description` ที่ต้องใส่ค่าของ query ลงไป

### n13-L2-06  [unit 13] [level 2] [mcq]  topic: Retrieval memory weakness
ข้อใดเป็นจุดอ่อนที่สไลด์ระบุไว้สำหรับ Retrieval memory (embed + vector search top-k)?
A) Lossy เพราะ LLM บีบข้อความเก่าทั้งหมดเป็น summary เดียว
B) ล้น context window ในที่สุดเพราะ append ทุกอย่างลง prompt
C) prompt โตเป็น O(n) ตามจำนวน turn
D) อาจพลาดข้อความที่สำคัญเชิงเวลา (temporally important)

### n13-L3-01  [unit 13] [level 3] [mcq]  topic: ReAct vs CoT error analysis Table 2
ข้อใดเกี่ยวกับ Table 2 (HotpotQA, วิเคราะห์ ReAct 100 + CoT 100 trajectory) ไม่ถูกต้อง?
A) ใน failure ReAct มี hallucination 0% ส่วน CoT มี 56%
B) reasoning error ของ ReAct (47%) สูงกว่าของ CoT (16%)
C) search result error 23% เป็น failure ที่ ReAct และ CoT พบในสัดส่วนใกล้เคียงกัน
D) success แบบ false positive (hallucinated trace) ของ ReAct 6% ต่ำกว่า CoT 14%

### n13-L3-02  [unit 13] [level 3] [mcq]  topic: Why agents: missing capability
ระบบ RAG ที่ index เอกสารไว้ล่วงหน้าตอบคำถาม "order #8812 ของฉันตอนนี้อยู่สถานะอะไร" ไม่ได้ ตามสไลด์ Why Agents ช่องว่างข้อใดตรงกับปัญหานี้มากที่สุดซึ่ง agent เติมได้?
A) Side effects: ต้องเขียนลง database
B) Model-chosen control flow: เลือกเองว่าจะ retrieve กี่ครั้ง
C) Grounded verification: ให้ critic ตรวจร่างคำตอบของตัวเอง
D) Live per-request state: ข้อมูลปัจจุบันที่ pre-index ไม่ได้

### n13-L3-03  [unit 13] [level 3] [mcq]  topic: Mask counts ratio
vocabulary 50,257 token: ที่ state หลัง `{` มี token ที่ valid 69 ตัว ส่วนใน string มี valid 50,146 ตัว จำนวน token ที่ถูก mask ที่ state หลัง `{` เป็นกี่เท่าของที่ถูก mask ในสตริง (โดยประมาณ)?
A) ประมาณ 45 เท่า
B) ประมาณ 452 เท่า
C) ประมาณ 4,520 เท่า
D) ประมาณ 0.0022 เท่า

### n13-L3-04  [unit 13] [level 3] [mcq]  topic: Choose topology from S41 table
ทีมสร้างระบบ fact-checking ข่าว ต้องการ Fault Tolerance สูง แต่ไม่ยอมรับ Latency ระดับ Very High และรับ Complexity ระดับ Medium ได้ ตามตาราง S41 ควรเลือก topology ใด?
A) Debate / Consensus
B) Supervisor-Led (Hierarchical)
C) Routing Network
D) Sequential Pipeline

### n13-L3-05  [unit 13] [level 3] [mcq]  topic: Hybrid memory scenario
แชตบอตคุยกับผู้ใช้หลายหมื่น turn ต้องการ (1) prompt ไม่โตตามจำนวน turn (2) ดึงรายละเอียดของ turn เก่าที่เกี่ยวกับคำถามปัจจุบันได้ (3) ยังรักษาภาพรวมของบทสนทนาโดยไม่พึ่ง similarity ของคำถามเพียงอย่างเดียว ตามสไลด์ควรเลือกแบบใด?
A) Buffer memory อย่างเดียว (append ทุก turn ลงใน prompt)
B) Summary memory อย่างเดียว (บีบทุกอย่างยกเว้น last k turns)
C) Retrieval memory อย่างเดียว (top-k จาก vector store)
D) Hybrid: summarize turn เก่า + retrieve episode ที่เกี่ยวข้อง

### n13-L3-06  [unit 13] [level 3] [mcq]  topic: Semantic validity vs masking, reliability layers
regex `\d{4}-\d{2}-\d{2}` กับ logit masking สร้าง `3194-81-59` (เดือน 81) ออกมา ชั้นใดของ Reliability Stack ควรจับปัญหานี้และเพราะเหตุใด?
A) Layer 2 Validate เพราะ masking การันตีแค่ syntax ไม่ใช่ semantics
B) Layer 3 Recover เพราะ retry จะทำให้ regex แก้ค่าเดือนเอง
C) Layer 4 Escalate เพราะต้องให้มนุษย์อนุมัติทุกวันที่
D) Layer 1 Constrain เพราะ logit masking ตรวจค่าเดือนได้อยู่แล้ว

### n13-L4-01  [unit 13] [level 4] [mcq]  topic: Context growth with headroom
ตาราง context growth ของสไลด์: cumulative ที่ step 10 = 15,220 tokens และสมมติแต่ละ step หลังจากนั้นเพิ่มเฉลี่ย 2,110 tokens ถ้า window = 128,000 และต้องเหลือที่ว่างอย่างน้อย 4,000 tokens ไว้เขียนคำตอบ (cumulative ต้องไม่เกิน 124,000) step สุดท้ายที่ทำให้เสร็จได้คือ step ใด?
A) Step 62
B) Step 61
C) Step 63
D) Step 64

### n13-L4-02  [unit 13] [level 4] [mcq]  topic: ReAct Table 1 vs Table 2 reasoning
ข้อใดอธิบายได้ถูกต้องที่สุดว่าทำไม ReAct จึงแพ้ CoT บน HotpotQA EM (27.4 vs 29.4) ทั้งที่ hallucination ใน failure ต่ำกว่ามาก (0% vs 56%)?
A) เพราะ Table 2 วัดเฉพาะ success ของ trajectory จึงไม่เกี่ยวข้องกับคะแนน EM ใน Table 1
B) เพราะ ReAct ไม่ได้เรียก search ตอนตอบ HotpotQA จึงไม่ได้ประโยชน์จาก external knowledge
C) เพราะ ReAct มี failure ที่ CoT ไม่มี เช่น reasoning error 47% (CoT 16%) และ search ว่าง 23%
D) เพราะ baseline ของ CoT ในตารางคือ CoT-SC ซึ่งผสม ReAct อยู่แล้ว จึงได้เปรียบ

### n13-L4-03  [unit 13] [level 4] [mcq]  topic: CFG parser stack
CFG-guided decoding สร้างมาถึง `{"a": {"b": [1` (stack ของ parser คือ object, object, array) ลำดับ token ปิดวงเล็บใดที่ valid ตลอดทางเมื่อต้องปิดโครงสร้างให้ครบ?
A) ]  }  }
B) }  ]  }
C) }  }  ]
D) ]  ]  }

### n13-L4-04  [unit 13] [level 4] [mcq]  topic: Regular vs CFG nuance
ข้อใดเกี่ยวกับ regex/DFA กับ CFG ในการบังคับโครงสร้าง JSON ถูกต้องที่สุดตามสไลด์?
A) JSON ทุกแบบ แม้จำกัดความลึกสูงสุด ก็ compile เป็น DFA ไม่ได้เพราะต้องนับวงเล็บ
B) เมื่อใช้ CFG ทั้ง decoding step เป็น O(1) เพราะมี precomputed index ของ state-to-token
C) DFA รองรับการซ้อนลึกไม่จำกัดได้ ถ้ามีจำนวน state เพียงพอ
D) JSON ที่จำกัดความลึกสูงสุดเป็น regular ทำ DFA ได้แต่ state บาน ส่วนลึกไม่จำกัดต้องใช้ CFG

### n13-L4-05  [unit 13] [level 4] [mcq]  topic: Markov state vs summary memory
ReAct นิยาม state s_t = (q, h_{1:t-1}) และเป็น Markov บน augmented state นี้ หากเปลี่ยนจากเก็บ history เต็มเป็นบีบ/สรุป history ด้วย Summary memory ข้อใดถูกต้องที่สุด?
A) ยังเป็น Markov เหมือนเดิม เพราะ summary มีข้อมูลเท่ากับ history เต็ม
B) ไม่การันตี Markov (อาจ partially observable) เพราะ summary อาจทิ้งข้อมูลที่เกี่ยวข้อง
C) ReAct ใช้ Summary memory ไม่ได้เลย เพราะ loop ต้องการ full history ทุกครั้งโดยสมบูรณ์
D) ยิ่ง summary สั้น state ยิ่งเป็น Markov มากขึ้น เพราะตัวแปรน้อยลง

### n13-L4-06  [unit 13] [level 4] [mcq]  topic: Latency: Debate (parallel) vs Sequential
Debate: agent 3 ตัวเสนอคำตอบอิสระแบบขนาน ใช้เวลา 6, 9 และ 7 วินาที แล้ว Judge สรุปอีก 4 วินาที (ไม่นับรอบแลก argument) ส่วน Sequential pipeline ที่ใช้ agent 4 ตัวเดิมต่อกัน (6, 9, 7, 4 วินาที) latency ของ Sequential มากกว่า Debate กี่วินาที?
A) 9 วินาที
B) 17 วินาที
C) 13 วินาที
D) 26 วินาที

### n13-L1-F1  [unit 13] [level 1] [fill]  topic: Tool definition parts
Tool definition ที่แสดงให้ LLM เห็นประกอบด้วย name, ___ (บอกว่าเมื่อไรควรใช้ tool) และ ___ (schema ของ input)

### n13-L2-F1  [unit 13] [level 2] [fill]  topic: ReAct vs CoT ranking flip
บน task ___ ReAct ได้ EM 27.4 ต่ำกว่า CoT (29.4) แต่บน task ___ ReAct ได้ Accuracy 60.9 สูงกว่า CoT (56.3)

### n13-L3-F1  [unit 13] [level 3] [fill]  topic: Valid token count after {
หลัง `{` ใน JSON ของ GPT-2 (vocab 50,257) token ที่ valid 69 ตัว = 62 (เปิด key) + ___ (whitespace) + ___ (รูป `}`) ดังนั้นถูก mask ___ ตัว

### n13-L4-F1  [unit 13] [level 4] [fill]  topic: Memory prompt size formulas
บทสนทนา n = 60 turn แต่ละ message ยาว 50 token; Summary memory เก็บ last k = 4 turn + summary 300 token; Retrieval memory ดึง top-k = 5 message ขนาด prompt ของ Buffer = ___ token, Summary = ___ token, Retrieval = ___ token

### n14-L1-01  [unit 14] [level 1] [mcq]  topic: Gopher rules: alpha-word ratio
ตามกฎ Gopher (heuristic filter) เอกสารต้องมี alpha-word ratio (สัดส่วนคำที่มีตัวอักษร) อย่างน้อยเท่าใดจึงจะผ่าน?
A) 0.5
B) 0.8
C) 0.95
D) 0.2

### n14-L1-02  [unit 14] [level 1] [mcq]  topic: LSH banding: H = b x r
MinHash signature ยาว H = 100 ตำแหน่ง ถูกแบ่งเป็น b = 20 bands ใน LSH banding แต่ละ band มีกี่ rows (r)?
A) 20
B) 80
C) 5
D) 2,000

### n14-L1-03  [unit 14] [level 1] [mcq]  topic: Three eras of evaluation
BERTScore, Chatbot Arena และ MT-Bench จัดอยู่ในยุค (era) ใดของการประเมินผลตามสไลด์?
A) Era 2 (2020-2024): semantic matching + human preference
B) Era 1 (ก่อน 2020): surface string matching
C) Era 3 (2024-2026): dynamic, capability-specific, adversarial
D) ไม่จัดอยู่ใน era ใดเลย

### n14-L1-04  [unit 14] [level 1] [mcq]  topic: MMLU facts
MMLU (Massive Multitask Language Understanding) ครอบคลุมกี่วิชา (subjects)?
A) 164
B) 80
C) 510
D) 57

### n14-L1-05  [unit 14] [level 1] [mcq]  topic: LLM judge: positional bias
LLM judge มักเลือกคำตอบที่ถูกวางเป็นอันดับแรก แม้จะสลับลำดับของคู่คำตอบเดิมแล้วก็ตาม อคติแบบนี้เรียกว่าอะไร?
A) Length (verbosity) bias
B) Data contamination
C) Positional bias
D) Benchmark saturation

### n14-L1-06  [unit 14] [level 1] [mcq]  topic: LiveBench refresh
LiveBench รีเฟรชคำถามใหม่บ่อยเพียงใด?
A) รายวัน
B) รายเดือน
C) รายปี
D) ไม่เปลี่ยนเลย (static)

### n14-L2-01  [unit 14] [level 2] [mcq]  topic: RAGAS faithfulness calculation
คำตอบของระบบ RAG ถูกแตกเป็น atomic claims ได้ 10 claims และ context สนับสนุน 7 claims ค่า RAGAS faithfulness เท่าใด?
A) 0.30
B) 1.43
C) 0.07
D) 0.70

### n14-L2-02  [unit 14] [level 2] [mcq]  topic: Exact vs near deduplication
ข่าวฉบับเดียวกันถูกเผยแพร่ซ้ำสองเว็บ โดยเปลี่ยนเฉพาะพาดหัว เนื้อหาส่วนที่เหลือเหมือนกันเกือบทั้งหมด ต้องการจับว่าเป็นสำเนาใกล้เคียงกัน วิธีใดเหมาะที่สุด?
A) แปลงเป็น shingles แล้วใช้ MinHash + LSH หา candidate จากนั้นตรวจ Jaccard จริง
B) normalise ข้อความ (lowercase, ตัดช่องว่าง) แล้วเทียบ SHA-256 เพราะ hash ของเนื้อหาที่ต่างกันเล็กน้อยจะมีค่าใกล้เคียงกัน
C) ใช้ fastText lid.176 ตรวจว่าทั้งสองหน้าได้ language label เดียวกัน
D) ใช้ Gopher alpha-word ratio >= 0.8 กับทั้งสองหน้าแล้วเทียบผลผ่าน/ไม่ผ่าน

### n14-L2-03  [unit 14] [level 2] [mcq]  topic: FineWeb vs Dolma
ทีมวิจัยต้องการคลังข้อมูลที่ผสมโค้ด เปเปอร์ และ Wikipedia รวมกับ web text และต้องย้อนตรวจ log ของ tag/filter ทุกเอกสารเพื่อ audit ควรเลือกชุดใด?
A) FineWeb เพราะใหญ่ที่สุด (15T tokens) และออกแบบ filter แบบ ablation-driven
B) FineWeb-Edu เพราะเก็บเฉพาะเอกสารที่ classifier ให้คะแนน 3 ขึ้นไป
C) Dolma เพราะผสมหลายแหล่งและมี tagger log รายเอกสาร
D) Common Crawl WET เพราะเป็น plain text ที่สกัดแล้วจึงตรวจสอบง่าย

### n14-L2-04  [unit 14] [level 2] [mcq]  topic: BERTScore recall from similarity matrix
candidate มี 3 tokens (แถว c1-c3) และ reference มี 2 tokens (คอลัมน์ r1, r2) ตาราง cosine similarity S เป็น [[0.6, 0.9], [0.8, 0.7], [0.3, 0.4]] BERTScore Recall (R) เท่าใด?
A) 0.70
B) 0.85
C) 0.65
D) 0.62

### n14-L2-05  [unit 14] [level 2] [mcq]  topic: pass@k calculation
โจทย์ HumanEval ข้อหนึ่ง generate n = 20 samples มี c = 1 sample ที่ผ่านทุก test ค่า pass@5 = 1 - C(n-c, k)/C(n, k) เท่าใด?
A) 0.05
B) 0.75
C) 0.95
D) 0.25

### n14-L2-06  [unit 14] [level 2] [mcq]  topic: HarmBench ASR interpretation
โมเดล X ได้ ASR = 0% บน HarmBench เพราะปฏิเสธทุกคำขอ รวมถึงคำถามปกติที่ไม่เป็นอันตรายด้วย ข้อสรุปใดถูกต้อง?
A) ASR ต่ำ = ทนต่อการโจมตีเหล่านี้ แต่ไม่ได้แปลว่าใช้งานได้ ต้องรายงานผลงานปกติด้วย
B) ASR 0% แปลว่าโมเดลปลอดภัยและดีที่สุดแล้ว จึงไม่จำเป็นต้องวัดผลงานด้านอื่นของโมเดลอีกต่อไป
C) ASR 0% น่าสงสัย เพราะ ASR ตัดสินโดยผู้เชี่ยวชาญที่เป็นมนุษย์ซึ่งน่าจะเจอการโจมตีที่สำเร็จอย่างน้อยบางกรณี
D) ASR เป็นสัดส่วนของ attack methods ทั้ง 18 วิธีที่ jailbreak สำเร็จ จึงเป็น 0% ไม่ได้

### n14-L3-01  [unit 14] [level 3] [mcq]  topic: Filter ordering by expected cost
Filter A: ต้นทุน c_A = 2 ms/doc เก็บเอกสารไว้ p_A = 0.9 ส่วน Filter B: c_B = 3 ms/doc เก็บไว้ p_B = 0.2 (p คือสัดส่วนที่รอด) ต้นทุนเฉลี่ยต่อเอกสารต่ำสุดได้จากข้อใด?
A) ทำ B ก่อน A เฉลี่ย 3.4 ms/doc
B) ทำ A ก่อน B เฉลี่ย 4.7 ms/doc เพราะ A ถูกกว่า
C) ทำ B ก่อน A เฉลี่ย 4.7 ms/doc
D) ลำดับไหนก็ได้ เฉลี่ย 5.0 ms/doc เท่ากัน

### n14-L3-02  [unit 14] [level 3] [mcq]  topic: Dedup cost scaling: brute force vs MinHash
Brute-force เทียบ shingle sets ทุกคู่ มีต้นทุนรวม N^2 x s^2 (N = จำนวนเอกสาร, s = shingles ต่อเอกสาร) ถ้าจำนวนเอกสารเพิ่มเป็น 2 เท่า และแต่ละเอกสารยาวขึ้นจน s เพิ่ม 3 เท่า ข้อใดถูก?
A) Brute-force เพิ่ม 6 เท่า และการเทียบ MinHash ต่อคู่เพิ่ม 3 เท่า ตามความยาวเอกสาร
B) Brute-force เพิ่ม 72 เท่า เพราะต้นทุนขึ้นกับ N^3
C) Brute-force เพิ่ม 36 เท่า และการเทียบ MinHash ต่อคู่เพิ่ม 3 เท่า ตามความยาวเอกสาร
D) Brute-force เพิ่ม 36 เท่า ส่วนการเทียบ MinHash ต่อคู่คงเดิม (H ตำแหน่ง)

### n14-L3-03  [unit 14] [level 3] [mcq]  topic: Benchmark contamination reasoning
โมเดล Gamma ได้ 94.2% บน MMLU และพบว่า 12% ของข้อ MMLU ปรากฏ verbatim ใน pre-training corpus ข้อใดถูกต้อง?
A) คะแนนที่ไม่ปนเปื้อนคือ 94.2 - 12 = 82.2% เพราะถือว่าข้อที่รั่วตอบถูกทั้งหมดทุกข้อ
B) คำนวณจากสองตัวเลขนี้ไม่ได้ ต้องรันซ้ำบนข้อ clean 88% หรือชุดข้อสอบใหม่
C) คะแนนที่ไม่ปนเปื้อนคือ 94.2 x 0.88 = 82.9% เพราะตัดข้อที่รั่วออกแล้วคูณสัดส่วนที่เหลือ
D) ไม่ต้องกังวล เพราะ MMLU เป็นข้อสอบแบบ 4 ตัวเลือก การรั่วจึงไม่มีผลต่อคะแนนรวมของโมเดล

### n14-L3-04  [unit 14] [level 3] [mcq]  topic: Testing length bias in an LLM judge
Model Alpha ชนะ pairwise ทุกครั้ง โดยตอบยาวเฉลี่ย 850 tokens เทียบกับ Beta 320 tokens ต้องการทดสอบว่า length bias อธิบายผลนี้หรือไม่ วิธีใดสะอาดที่สุด?
A) ดู correlation ระหว่างความยาวกับคะแนนข้ามหลายโมเดล ถ้าสูงก็ถือว่ายืนยัน length bias ได้แล้ว
B) หารคะแนนของแต่ละโมเดลด้วยจำนวน token ของคำตอบ แล้วเทียบคะแนนที่ได้ของทั้งสองโมเดล
C) ใช้เนื้อหาเดิมแต่ขยายความโดยพูดซ้ำ ไม่เพิ่มข้อมูล แล้วดูว่า judge เลือกฉบับยาวหรือไม่
D) เปลี่ยนไปใช้ judge ที่ใหญ่ขึ้นหนึ่งตัว แล้วถือว่า length bias หายไปแล้วทั้งหมด

### n14-L3-05  [unit 14] [level 3] [mcq]  topic: BLEU vs BERTScore: find the wrong statement
จากตัวอย่าง reference 'The cat sat on the mat' เทียบ paraphrase 'A feline rested upon the rug' และ word salad 'mat the on is cat the' ข้อใด ไม่ถูกต้อง?
A) BLEU-1 ให้ word salad 0.83 สูงกว่า paraphrase ที่ได้ 0.17 เพราะนับแค่คำที่ตรงกัน
B) BERTScore F1 0.79 ของ paraphrase เทียบกับ BLEU-1 0.17 ได้ตรง ๆ จึงดีกว่าราว 4.6 เท่า
C) BERTScore F1 ให้ paraphrase 0.79 สูงกว่า word salad 0.67 แต่ยังตรวจข้อเท็จจริงไม่ได้
D) BERTScore ใช้ cosine similarity ระหว่าง token embeddings จึงจับ paraphrase ได้ดีกว่า

### n14-L3-06  [unit 14] [level 3] [mcq]  topic: RAGAS Context Precision@K
ระบบ RAG ดึง 5 chunks ความเกี่ยวข้องตามอันดับคือ rel = [0, 1, 1, 0, 1] (1 = เกี่ยวข้อง) ใช้ Context Precision@K = (sum_k Precision@k x v_k) / (จำนวน item ที่เกี่ยวข้องใน top K) โดย v_k = rel_k ค่า Context Precision@5 ประมาณเท่าใด?
A) 0.35
B) 0.45
C) 0.59
D) 0.60

### n14-L4-01  [unit 14] [level 4] [mcq]  topic: LSH configuration and S-curve ramp
ใช้ MinHash signature H = 128 ตำแหน่ง ต้องการให้ช่วงเปลี่ยนของ S-curve อยู่แถว s ประมาณ 0.7 โดยใช้ s* ~ (1/b)^(1/r) และ P(candidate) = 1 - (1 - s^r)^b ข้อใดถูกต้อง?
A) b = 32, r = 4: s* ~ 0.42 จึงจับคู่ J = 0.7 เป็น candidate ได้เกือบแน่นอน (P ~ 1.00) ตรงกับเป้าหมาย
B) b = 16, r = 8: s* ~ 0.71 และที่ J = 0.7 โอกาสเป็น candidate คือ 0.50 พอดี เพราะ s* คือจุด 50%
C) b = 8, r = 16: s* ~ 0.88 จึงเป็นค่าที่ใกล้ 0.7 ที่สุดในบรรดาทางเลือกทั้งหมด
D) b = 16, r = 8: s* ~ 0.71 แต่ที่ J = 0.7 P ราว 0.61 จึงพลาดราว 39% ต้อง verify ด้วย Jaccard จริง

### n14-L4-02  [unit 14] [level 4] [mcq]  topic: Chinchilla ratio vs available corpora
ต้องการเทรนโมเดล 175B parameters ให้ได้จำนวน token ตามอัตรา Chinchilla (~20 tokens/parameter) โดยไม่วนข้อมูลซ้ำ มีคลัง FineWeb-Edu 1.3T tokens และ FineWeb เต็ม 15T tokens ข้อใดถูก?
A) ต้องการ 3.5T tokens FineWeb-Edu มีราว 37% ของที่ต้องใช้ (วนซ้ำราว 2.7 รอบ) FineWeb เต็มมีราว 4.3 เท่า
B) ต้องการ 350B tokens FineWeb-Edu 1.3T มีมากเกินพอราว 3.7 เท่า ไม่ต้องใช้ FineWeb เต็มเลย
C) ต้องการ 35T tokens ดังนั้นต่อให้ใช้ FineWeb เต็ม 15T ก็ยังไม่พอ ต้องหาข้อมูลเพิ่มจากแหล่งอื่น
D) ต้องการ 3.5T tokens แต่ FineWeb-Edu 1.3T ยังเพียงพอ เพราะ Edu คุณภาพสูงจึงใช้ token น้อยลง 10 เท่าได้เลย

### n14-L4-03  [unit 14] [level 4] [mcq]  topic: Aggregating pass@k over problems
benchmark มี 3 โจทย์ แต่ละข้อ generate n = 10 samples มี c ที่ผ่านเป็น 0, 10, 3 ตามลำดับ คะแนน pass@2 ของ benchmark = ค่าเฉลี่ยของ pass@2 รายข้อ โดย pass@k = 1 - C(n-c, k)/C(n, k) (ถ้า n - c < k ให้ pass@k = 1) ได้ค่าเท่าใด?
A) 0.43
B) 0.64
C) 0.67
D) 0.51

### n14-L4-04  [unit 14] [level 4] [mcq]  topic: Swap test results and aggregation policy
judge ตัดสิน 100 คู่คำตอบทั้งสองลำดับ (A-B และ B-A) ผล: Alpha ชนะทั้งสองลำดับ 54 คู่, Beta ชนะทั้งสองลำดับ 16 คู่, ผลไม่ตรงกัน (inconsistent) 30 คู่ ข้อใดถูกต้อง?
A) tie policy: Alpha = 69%; ตัดออก: 54/70 = 77.1% ทั้งคู่เป็น policy ที่ต้องระบุ และ 30% inconsistent บ่งชี้ positional bias
B) Alpha = 54 + 30 = 84% เพราะคู่ที่ไม่ตรงกันถือว่า Alpha เป็นฝ่ายชนะ เนื่องจากถูกวางเป็นลำดับแรกในทุกคู่ที่ตัดสินทั้งหมด 100 คู่
C) ป้ายที่ถูกของคู่ inconsistent คือ tie ซึ่งเป็นผลการวัด Alpha = 69% จึงไม่ใช่การเลือก policy ของผู้ประเมินแต่อย่างใดทั้งสิ้น
D) ถ้าตัดคู่ inconsistent ออก Alpha = 54/100 = 54% และไม่จำเป็นต้องรายงานจำนวนคู่ที่ถูกตัดออกไปให้ผู้อ่านทราบ

### n14-L4-05  [unit 14] [level 4] [mcq]  topic: Tokenizer cost vs context budget
เอกสารภาษาไทย 40,000 ตัวอักษร ถูก tokenize ด้วย GPT-2 ซึ่งมี TPC ภาษาไทย = 2.00 tokens/ตัวอักษร (ภาษาอังกฤษ 0.14) ใส่ใน context ที่มี system+query อีก 500 tokens ถ้าแต่ละ step ของ agent เพิ่ม 2,110 tokens และรวมกันต้องไม่เกิน 128,000 tokens agent รันได้เต็ม ๆ อย่างมากกี่ step?
A) 23
B) 22
C) 41
D) 57

### n14-L4-06  [unit 14] [level 4] [mcq]  topic: Diagnosing a self-repeating page through the pipeline
หน้าเว็บภาษาอังกฤษยาว 200 คำ ซึ่งประโยคเดียวถูกพิมพ์ซ้ำ 20 ครั้ง ผ่านทั้ง Gopher กฎพื้นฐาน (word count, mean word length, alpha ratio, stop-word), perplexity filter และ MinHash dedup ข้อใดอธิบายและแก้ได้ถูกต้อง?
A) MinHash + LSH ควรจับได้ เพราะ shingle ซ้ำกันเองมากจึงมี Jaccard สูง ให้ลดจำนวน b ลงเพื่อให้จับหน้าแบบนี้ได้
B) ให้เก็บเฉพาะ bucket head ของ perplexity เพราะ PPL ต่ำหมายถึงคุณภาพสูงและช่วยตัดหน้าแบบนี้ออกได้
C) หน้านี้ผ่านสถิติพื้นฐาน, PPL ต่ำจึงถูกเก็บ, MinHash เทียบข้ามเอกสาร จึงต้องใช้ Gopher repetition rules
D) ให้เพิ่ม alpha-word ratio threshold เป็น 0.95 เพื่อตัดหน้าที่ซ้ำออกจากชุดข้อมูลทั้งหมด

### n14-L1-F1  [unit 14] [level 1] [fill]  topic: fastText language ID facts
fastText lid.176.bin รองรับ ___ ภาษา และ FineWeb เก็บข้อความภาษาอังกฤษเมื่อ confidence score >= ___

### n14-L2-F1  [unit 14] [level 2] [fill]  topic: BERTScore F1
BERTScore ได้ Precision P = 0.9 และ Recall R = 0.6 ค่า F1 = 2PR/(P+R) เท่ากับ ___ (ทศนิยม 2 ตำแหน่ง)

### n14-L3-F1  [unit 14] [level 3] [fill]  topic: LSH banding parameters
MinHash signature ยาว H = 128 แบ่งเป็น b = 32 bands จะได้ r = ___ rows ต่อ band และ threshold ประมาณ s* = (1/b)^(1/r) = ___ (ทศนิยม 2 ตำแหน่ง)

### n14-L4-F1  [unit 14] [level 4] [fill]  topic: MinHash standard error
MinHash ใช้ H = 256 hash positions standard error สูงสุดของค่าประมาณ Jaccard = 1/(2 sqrt(H)) = ___ และถ้าต้องการให้ standard error สูงสุดลดลงเหลือครึ่งหนึ่ง ต้องเพิ่ม H เป็น ___

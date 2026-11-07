### n08-L1-01  [unit 8] [level 1] [mcq]  topic: BART Document Rotation
ใน BART วิธี noising แบบ Document Rotation ทำอย่างไร?
A) แบ่งเอกสารเป็นประโยคแล้วสลับลำดับประโยคแบบสุ่ม ให้โมเดลเรียงกลับ
B) สุ่มเลือก token เป็น pivot แล้วหมุนเอกสารให้เริ่มที่ pivot โมเดลต้องหาจุดเริ่มต้นจริง
C) ลบ token แบบสุ่มออกโดยไม่เหลือ placeholder ใด ๆ ไว้ในตำแหน่งเดิม แล้วให้โมเดลเดาว่าอะไรหาย
D) แทน span ของ token ที่ติดกันด้วย [MASK] ตัวเดียว โดยความยาว span สุ่มจากการแจกแจง Poisson

### n08-L1-02  [unit 8] [level 1] [mcq]  topic: Original Transformer decoder stack
ใน Decoder stack ของ Transformer ดั้งเดิม (Vaswani et al., 2017) ลำดับ sub-layer ภายในแต่ละ layer คือข้อใด?
A) Multi-Head Self-Attention (ไม่มี mask) -> FFN -> Cross-Attention
B) Cross-Attention -> Masked Multi-Head Self-Attention -> FFN
C) Masked Multi-Head Self-Attention -> FFN เท่านั้น เพราะ Cross-Attention อยู่ที่ฝั่ง Encoder
D) Masked Multi-Head Self-Attention -> Cross-Attention -> FFN

### n08-L1-03  [unit 8] [level 1] [mcq]  topic: T5 task prefix (summarization)
ใน T5 ถ้าต้องการให้โมเดลสรุปบทความ ต้องนำข้อความใดไปขึ้นต้น input?
A) summarize: ตามด้วยบทความ
B) sst2 sentence: ตามด้วยบทความ
C) translate English to German: ตามด้วยบทความ
D) question: ตามด้วยบทความ context: ตามด้วยบทความ

### n08-L1-04  [unit 8] [level 1] [mcq]  topic: BART-large architecture
ตามสไลด์ BART-large มีโครงสร้างตามข้อใด?
A) 6 Encoder layers + 6 Decoder layers, d_model = 1024, 16 heads
B) 12 Encoder layers + 12 Decoder layers, d_model = 768, 12 heads
C) 12 Encoder layers + 12 Decoder layers, d_model = 1024, 16 heads
D) 24 Decoder layers เท่านั้น (ไม่มี Encoder), d_model = 1024, 16 heads

### n08-L1-05  [unit 8] [level 1] [mcq]  topic: T5 position encoding
T5 ใช้การเข้ารหัสตำแหน่งแบบใด?
A) Sinusoidal absolute positional encoding แบบเดียวกับ Transformer ดั้งเดิมของ Vaswani
B) Relative Position Bias: learned bias ตามระยะสัมพัทธ์ บวกเข้า attention logits
C) Learned absolute position embedding ที่บวกเข้ากับ token embedding ก่อนเข้า Encoder
D) ไม่ใช้ข้อมูลตำแหน่งเลย เพราะ sentinel token บอกตำแหน่งอยู่แล้ว

### n08-L1-06  [unit 8] [level 1] [mcq]  topic: Denoising objective
Denoising Autoencoder (ที่ BART และ T5 ใช้ pre-train) มี objective ตามสไลด์คือข้อใด?
A) max log P(x | x~) คือ reconstruct x จากข้อความที่ corrupt แล้ว x~
B) max log P(x~ | x) คือทำนายข้อความที่ถูก corrupt จากต้นฉบับ
C) max log P(next word | previous words) โดยไม่ใช้ข้อความที่ถูก corrupt
D) จำแนกด้วย binary classifier ว่า input นี้ถูก corrupt มาหรือไม่

### n08-L2-01  [unit 8] [level 2] [mcq]  topic: Masking vs Deletion src_len
ประโยคต้นฉบับมี 10 tokens (ไม่นับ <s>, </s>) ถูกทำ noising กับ 2 token ได้แก่ กรณี (ก) Token Masking แทน 2 token ด้วย [MASK] ตัวละหนึ่ง กรณี (ข) Token Deletion ลบ 2 token ทิ้ง ข้อใดให้ (src_len ของ ก, src_len ของ ข) ตามลำดับ โดย tgt_len ของทั้งสองกรณีคือ 10?
A) (8, 10)
B) (10, 10)
C) (10, 8)
D) (8, 8)

### n08-L2-02  [unit 8] [level 2] [mcq]  topic: T5 target ending
T5 span corruption ได้ encoder input เป็น `A <extra_id_0> B C <extra_id_1> D <extra_id_2> E` (3 sentinel) ตามรูปแบบในสไลด์ token สุดท้าย 2 ตัวของ decoder target เรียงตามลำดับคือข้อใด?
A) `<extra_id_2>` แล้วตามด้วย `</s>`
B) `</s>` แล้วตามด้วย `<extra_id_3>`
C) token ของ span สุดท้ายที่ถูกตัด แล้วตามด้วย `</s>` (ไม่มี sentinel ปิดท้าย)
D) `<extra_id_3>` แล้วตามด้วย `</s>`

### n08-L2-03  [unit 8] [level 2] [mcq]  topic: Decoder self-attn cost ratio
สมมติ decoder target ของ BART ยาว N ส่วน target ของ T5 ยาว 0.25N (ไม่ใช่ 0.20N ในสไลด์) และต้นทุน decoder self-attention แปรตามกำลังสองของความยาว target ต้นทุนของ T5 ถูกกว่า BART กี่เท่า?
A) 16 เท่า
B) 4 เท่า
C) 8 เท่า
D) 0.0625 เท่า

### n08-L2-04  [unit 8] [level 2] [mcq]  topic: Relative position bias logit
ตามสูตรในสไลด์ a_mn = (q_m . k_n) / sqrt(d_k) + b_(m-n) ถ้า q_m . k_n = 20, d_k = 64 และ learned bias b_(m-n) = -1.5 ค่า attention logit ก่อน softmax เท่ากับเท่าใด?
A) 18.5
B) 1.0
C) 2.5
D) -1.19

### n08-L2-05  [unit 8] [level 2] [mcq]  topic: BART fine-tuning for classification
ตามสไลด์ การ fine-tune BART สำหรับ sentiment classification ทำอย่างไร?
A) ป้อน input เข้า Encoder อย่างเดียวแล้วใช้ hidden state ของ token [CLS] เข้า linear classifier เหมือน BERT
B) ใส่ prefix 'sst2 sentence:' แล้วให้ Decoder generate คำว่า positive/negative โดยไม่มี classifier head ใด ๆ เหมือน T5
C) ป้อน input เข้าทั้ง Encoder และ Decoder แล้วนำ hidden state ของ final Decoder token เข้า linear classifier
D) ใช้ Decoder อย่างเดียวและตัด Encoder ทิ้ง เพราะ classification ไม่ต้องอ่านแบบสองทิศทาง

### n08-L2-06  [unit 8] [level 2] [mcq]  topic: Parameter duplication in Enc-Dec
โมเดล Encoder-Decoder และ Decoder-only มีพารามิเตอร์รวมเท่ากัน P = 400M ตามสไลด์ s.36 (Parameter Duplication Problem) พารามิเตอร์ที่ทุ่มให้กับการ generate ข้อความของแต่ละแบบเป็นข้อใด?
A) Enc-Dec ทั้ง 400M ส่วน Dec-only ประมาณ 200M
B) ทั้งสองแบบประมาณ 400M เท่ากัน เพราะ Encoder ก็ช่วย generate ด้วย
C) ทั้งสองแบบประมาณ 200M เท่ากัน เพราะต้องกันครึ่งหนึ่งไว้ให้ attention
D) Enc-Dec ประมาณ 200M (ฝั่ง Decoder) ส่วน Dec-only ทั้ง 400M

### n08-L3-01  [unit 8] [level 3] [mcq]  topic: T5 token counting
ประโยคต้นฉบับ 30 tokens ถูก T5 ตัด 3 span ที่ไม่ติดกัน ยาว 3, 2 และ 4 tokens แทนแต่ละ span ด้วย sentinel หนึ่งตัวใน encoder input ใช้การนับตามสไลด์ s.17 (src_len ไม่นับ </s>; tgt_len นับ sentinel ตัวสุดท้ายที่ปิดท้ายและนับ </s>) ข้อใดให้ (src_len, tgt_len) ถูกต้อง?
A) (24, 14)
B) (24, 12)
C) (24, 13)
D) (21, 14)

### n08-L3-02  [unit 8] [level 3] [mcq]  topic: BART noising: what each teaches
ข้อใดจับคู่ 'วิธี noising ของ BART -> สิ่งที่โมเดลต้องเรียนรู้' ผิด?
A) Token Masking -> เรียนรู้ vocabulary และ local context เหมือนการทำข้อสอบเติมคำ
B) Sentence Permutation -> เรียนรู้ discourse และความต่อเนื่อง (coherence) ระดับเอกสาร
C) Document Rotation -> ต้องตรวจว่า token ใดหายไปและหายไปตรงไหน เพราะไม่เหลือ placeholder
D) Text Infilling -> เรียนรู้การ generate ข้อความหลาย token และการประมาณความยาวของส่วนที่หายไป

### n08-L3-03  [unit 8] [level 3] [mcq]  topic: Scaling argument evidence
ข้อใดสรุปหลักฐานเรื่อง Decoder-only vs Encoder-Decoder ตามสไลด์ s.39 ผิด?
A) ในงานของ Wang et al. (2022) หลัง pre-train อย่างเดียว Dec-only (4.8B) + full LM ชนะ zero-shot ต่อ Enc-Dec (11B) ที่ใช้ span corruption
B) ลำดับ T5 (2020) -> PaLM (2022) -> Gemini (2023) เป็นผลเปรียบเทียบ head-to-head ที่พิสูจน์ว่า Dec-only ดีกว่า
C) Enc-Dec + span corruption ชนะเฉพาะหลัง multitask finetuning เท่านั้น ข้อได้เปรียบจึงมีเงื่อนไข ไม่ใช่ของฟรี
D) ไม่มีการเปรียบเทียบแบบควบคุมที่ขนาดเกิน 100B parameters เพราะไม่มีใครทุนฝั่ง Enc-Dec

### n08-L3-04  [unit 8] [level 3] [mcq]  topic: T5 relative position bias details
ข้อใดอธิบาย Relative Position Bias ของ T5 ผิด?
A) ใช้ตาราง trainable 32 buckets ต่อ head แบ่งเป็น 16 forward และ 16 backward
B) offset 0-7 ได้ bias แยกแต่ละค่า ส่วน offset ไกลขึ้นไปจะใช้ bucket ร่วมกันเพื่อประหยัด parameter
C) bias ที่ดึงมาเป็น learned scalar ที่ถูกบวกเข้า attention logits โดยตรง
D) bias ที่ดึงจากตารางตาม offset (m-n) ถูกคูณเข้ากับ attention weights หลัง softmax

### n08-L3-05  [unit 8] [level 3] [mcq]  topic: Cross-attention weight count
T5-base มี 12 heads ป้อน batch = 8, source length = 64 และ target length = 16 cross-attention weights ของหนึ่ง layer ตามรูปร่างในสไลด์ s.32 มีจำนวนค่า (elements) ทั้งหมดเท่าใด?
A) 393,216
B) 24,576
C) 98,304
D) 12,288

### n08-L3-06  [unit 8] [level 3] [mcq]  topic: Extractive vs generative QA scenario
ระบบ Q&A ด้านกฎหมายต้องการคำตอบที่ย้อนตรวจกลับไปยังข้อความต้นฉบับได้ทุกคำ และห้ามมีเนื้อหาที่ไม่อยู่ในเอกสาร ควรเลือกแนวทางใดและต้องยอมรับข้อจำกัดอะไร?
A) Extractive QA (BERT start/end pointers) โดยยอมรับว่าสังเคราะห์คำตอบใหม่ไม่ได้ ต้องเป็น substring จริง
B) Generative QA (T5) เพราะ Decoder ถูกบังคับให้ copy จาก context เท่านั้น จึงไม่มีทาง hallucinate และ paraphrase ได้อิสระ
C) Generative QA (T5) เพราะเมื่อ fine-tune ดีแล้วจะออกนอก context ไม่ได้เชิงโครงสร้าง
D) Extractive QA (BERT) โดยยอมรับว่า pointer ชี้ออกนอก context ได้ จึงมีความเสี่ยง hallucinate

### n08-L4-01  [unit 8] [level 4] [mcq]  topic: Text infilling zero-length spans
ประโยคต้นฉบับ 10 tokens ทำ BART text infilling 3 span (ไม่ซ้อนกัน) ความยาว span ที่สุ่มได้คือ 0, 0 และ 1 token (span ยาว 0 คือการแทรก [MASK] โดยไม่ลบ token) ไม่นับ <s>, </s> ข้อใดถูกต้อง?
A) src_len = 9 ซึ่งน้อยกว่า tgt_len = 10
B) src_len = 10 เท่ากับ tgt_len เพราะ [MASK] หนึ่งตัวแทนหนึ่ง token
C) src_len = 7 ซึ่งน้อยกว่า tgt_len = 10
D) src_len = 12 ซึ่งมากกว่า tgt_len = 10

### n08-L4-02  [unit 8] [level 4] [mcq]  topic: Diagnose a T5 pipeline
ทีมหนึ่งสร้าง pipeline 'T5 span corruption' โดย encoder input มี sentinel `<extra_id_k>` แทน span ถูกต้อง แต่ใช้ decoder target เป็นประโยคต้นฉบับเต็มทุก token ผลวัดว่าเวลาเทรนต่อ step แทบไม่ต่างจาก BART baseline ทีมสงสัยว่า sentinel ไม่ทำงาน ข้อใดวินิจฉัยถูกต้อง?
A) T5 ประหยัดที่ฝั่ง Encoder จึงต้องเพิ่มจำนวน sentinel ใน input ให้มากขึ้นเพื่อให้ input สั้นลงมาก ๆ และลดต้นทุนรวมของทั้งโมเดล
B) ปัญหาอยู่ที่ target: ต้องมีเฉพาะ span ที่หาย + sentinel (~0.20N) ถ้ายาว N decoder ยังเป็น O(N^2) ไม่ประหยัด 25x
C) sentinel ต้องใช้ตัวเดียวกันทุก span (เหมือน [MASK] ของ BART) เพื่อให้ decoder สั้นลงและเร็วขึ้น
D) ต้องเปลี่ยน loss จาก cross-entropy เป็น loss ชนิดอื่นที่ไม่บังคับให้ decoder ต้องผลิตทุก token ในประโยคต้นฉบับ

### n08-L4-03  [unit 8] [level 4] [mcq]  topic: Overall speedup (Amdahl)
สมมติในการเทรน BART ต้นทุนฝั่ง encoder = 0.5 หน่วย และต้นทุน decoder self-attention = 1 หน่วย (ไม่นับต้นทุนอื่น) เมื่อเปลี่ยนเป็น T5 span corruption ที่ target ยาว 0.20N encoder cost คงเดิม ส่วน decoder self-attention แปรตามกำลังสองของความยาว target ความเร็วเทรนโดยรวมของ T5 เพิ่มขึ้นประมาณกี่เท่า?
A) 2.78 เท่า
B) 25 เท่า
C) 2.14 เท่า
D) 3.00 เท่า

### n08-L4-04  [unit 8] [level 4] [mcq]  topic: Equal-parameter assumption in FLOPs comparison
สไลด์ s.31 (Doc = 1000, Summary = 100) ได้ Enc-Dec = N x [1000^2 + 100^2 + 100x1000] = 1,110,000N และ Dec-only = 2N x [(1100)^2/2] = 1,210,000N โดยสมมติ parameter เท่ากัน (Enc-Dec N layers, Dec-only 2N layers) ถ้านักศึกษาเข้าใจผิดว่า Dec-only ใช้ N layers เท่า Enc-Dec ผลจะเป็นอย่างไร?
A) Dec-only = 605,000N แต่ยังแพงกว่า Enc-Dec ประมาณ 1.83 เท่า เพราะ Enc-Dec มี cross-attention น้อยกว่าจึงเร็วกว่า
B) Dec-only = 605,000N ถูกกว่า Enc-Dec ประมาณ 1.83 เท่า ข้อสรุปสไลด์จึงขึ้นกับสมมติฐาน param เท่ากัน
C) Dec-only ยังเป็น 1,210,000N เท่าเดิม เพราะต้นทุนขึ้นกับความยาว sequence เท่านั้น ไม่ขึ้นกับจำนวน layer
D) Dec-only = 2,420,000N เพราะ causal mask ไม่ช่วยลด FLOPs เมื่อใช้ N layers ตามที่นักศึกษาเข้าใจ

### n08-L4-05  [unit 8] [level 4] [mcq]  topic: Signal density: MLM vs T5 target
เอกสารยาว 400 tokens BERT MLM เลือก 15% ของ token เป็นเป้าหมายโดยใช้กฎ 80/10/10 (80% แทนด้วย [MASK], 10% แทนด้วย token สุ่ม, 10% คงเดิม) ส่วน T5 มี decoder target ยาวประมาณ 0.20N ข้อใดให้ (จำนวน token ที่ถูกแทนด้วย [MASK] จริงใน BERT, จำนวน token ที่ decoder ของ T5 ต้อง generate) ถูกต้อง?
A) (60, 80)
B) (48, 400)
C) (48, 80)
D) (48, 60)

### n08-L4-06  [unit 8] [level 4] [mcq]  topic: Cost of not caching cross-attention K,V
ตามสไลด์ s.25 K, V ของ cross-attention จาก Encoder ถูกคำนวณครั้งเดียวแล้ว cache ถ้า Decoder รัน 20 steps และ source ยาว 50 tokens การ project K และ V ของตำแหน่ง Encoder (ดูเฉพาะ 1 layer) ถ้า 'ไม่ cache' จะต้องทำมากกว่าแบบ cache กี่เท่า?
A) 50 เท่า
B) 40 เท่า
C) 1,000 เท่า
D) 20 เท่า

### n08-L1-F1  [unit 8] [level 1] [fill]  topic: Flan-T5 and mT5
Flan-T5 ถูก fine-tune ด้วย instructions บน ___ tasks ส่วน mT5 (Multilingual T5) รองรับ ___ ภาษา

### n08-L2-F2  [unit 8] [level 2] [fill]  topic: BART text infilling
ใน BART วิธี Text Infilling ความยาวของแต่ละ span สุ่มจากการแจกแจง ___ (lambda = ___) แล้วแทนทั้ง span ด้วย [MASK] ตัวเดียว และ span ที่ยาว 0 หมายถึงการ ___ [MASK] โดยไม่ลบ token ใด

### n08-L3-F3  [unit 8] [level 3] [fill]  topic: Enc-Dec vs Dec-only FLOPs
สรุปความด้วย Doc = 500 tokens, Summary = 50 tokens สมมติ parameter เท่ากันตามสไลด์ s.31 Enc-Dec = N x [500^2 + 50^2 + (50 x 500)] = ___ N และ Dec-only (2N layers) = 2N x [(550)^2 / 2] = ___ N

### n08-L4-F4  [unit 8] [level 4] [fill]  topic: BART vs T5 target length ratio
ประโยคต้นฉบับ 50 tokens BART มี decoder target = ข้อความเต็ม + <s> + </s> ส่วน T5 ตัด 3 span รวม 10 tokens และ target นับ sentinel ที่ปิดท้ายกับ </s> ด้วย (ตามสไลด์ s.17) tgt_len_BART / tgt_len_T5 = ___ (ปัดเป็นทศนิยม 2 ตำแหน่ง)

### n09-L1-01  [unit 9] [level 1] [mcq]  topic: SBERT classification head dimension
SBERT แบบ classification (BERT-base, d = 768) นำเวกเตอร์ประโยค u และ v มาต่อกันเป็น c = [u; v; |u - v|] ก่อนเข้า softmax classifier เวกเตอร์ c มีกี่มิติ
A) 768
B) 1,536
C) 2,304
D) 3,072

### n09-L1-02  [unit 9] [level 1] [mcq]  topic: Contrastive learning goal
เป้าหมายหลักของ contrastive learning ในการฝึก sentence embedding คือข้อใด
A) ดึง positive pairs เข้าใกล้กัน และผลัก negative pairs ให้ห่างกัน
B) ทำให้ทุกประโยคในชุดข้อมูลมีเวกเตอร์ใกล้เคียงกันมากที่สุดเพื่อให้ cosine สูง
C) ทำให้ทุกคู่ประโยค รวมทั้งคู่ที่คล้ายกัน อยู่ห่างกันให้มากที่สุด
D) ลดจำนวนมิติของเวกเตอร์ประโยคให้เล็กลงโดยไม่สนใจความหมาย

### n09-L1-03  [unit 9] [level 1] [mcq]  topic: Siamese network
Siamese network ที่ใช้ใน SBERT มีลักษณะตามข้อใด
A) สองสาย BERT ที่มีสถาปัตยกรรมต่างกัน สายหนึ่งสำหรับประโยค A อีกสายสำหรับประโยค B
B) สองสาย BERT สถาปัตยกรรมเดียวกัน แต่แต่ละสายมี weights แยกกันและอัปเดตอิสระ
C) BERT หนึ่งสายสำหรับประโยคแรก และ classifier อีกชุดสำหรับประโยคที่สอง
D) สองสายที่ใช้สถาปัตยกรรมและ weights ชุดเดียวกัน (tied weights)

### n09-L1-04  [unit 9] [level 1] [mcq]  topic: SBERT pooling strategy
pooling แบบใดคือค่า default ของ SBERT และให้ผลดีที่สุดใน ablation ของสไลด์
A) CLS pooling คือใช้เวกเตอร์ของ token [CLS] อย่างเดียว
B) Mean pooling ที่ใช้ attention mask ตัด [PAD]
C) Max pooling คือเลือกค่าสูงสุดของแต่ละมิติจากทุก token ในประโยค
D) Sum pooling รวมทุก token รวมถึง [PAD] โดยไม่หาร

### n09-L1-05  [unit 9] [level 1] [mcq]  topic: FAISS IndexFlatIP score
เมื่อเวกเตอร์ถูก L2-normalize แล้วและใช้ faiss.IndexFlatIP คะแนนที่ index.search คืนมา (D) คืออะไร
A) squared L2 distance ซึ่งค่าต่ำหมายถึงคล้ายกันมากที่สุด
B) cosine distance (1 - cos) ซึ่งค่าต่ำหมายถึงคล้ายกันมากที่สุด
C) cosine similarity อยู่ในช่วง [-1, 1] ซึ่งค่าสูงหมายถึงคล้ายกัน
D) dot product ที่ไม่มีขอบเขตและไม่เกี่ยวข้องกับ cosine similarity ในทุกกรณี แม้ normalize แล้ว

### n09-L1-06  [unit 9] [level 1] [mcq]  topic: MRR reciprocal rank
สำหรับ query หนึ่ง เอกสารที่ถูกต้องอันแรกอยู่ที่อันดับ 4 ค่า reciprocal rank (1/rank) ที่ใช้คำนวณ MRR คือเท่าใด
A) 0.25
B) 4
C) 0.75
D) 0.4

### n09-L2-01  [unit 9] [level 2] [mcq]  topic: Bi-encoder vs cross-encoder complexity
ต้องการเทียบความคล้ายของทุกคู่ในประโยค n ประโยค ข้อใดอธิบายต้นทุนของ bi-encoder และ cross-encoder ได้ถูกต้อง
A) ทั้งสองแบบต้องทำ forward pass ระดับ O(n^2) ต่างกันแค่ขนาดของโมเดลที่ใช้
B) Bi-encoder: n forward + dot product O(n^2 d); cross-encoder: O(n^2) forward passes
C) Bi-encoder ใช้ O(n^2) forward passes ส่วน cross-encoder ใช้ n forward passes แล้วทำ dot product
D) ทั้งสองแบบใช้ n forward passes ต่างกันที่ cross-encoder มี attention ข้ามประโยคเท่านั้น

### n09-L2-02  [unit 9] [level 2] [mcq]  topic: Triplet loss calculation
Triplet loss = max(0, d(a,p) - d(a,n) + margin) ถ้า d(a,p) = 0.4, d(a,n) = 0.7 และ margin = 0.5 ค่า loss คือเท่าใด
A) 0
B) 0.6
C) 0.8
D) 0.2

### n09-L2-03  [unit 9] [level 2] [mcq]  topic: Cause of anisotropy
ตามสไลด์ สาเหตุที่แท้จริงที่ raw BERT ให้ sentence embedding ที่อยู่ใน narrow cone (anisotropy) คืออะไร
A) training objective แบบ MLM ไม่เคยกระตุ้นให้ geometry ระดับประโยคดี
B) สถาปัตยกรรม self-attention ของ BERT ไม่สามารถสร้างเวกเตอร์ประโยคได้
C) มิติ 768 เล็กเกินไปสำหรับเก็บความหมายของประโยค
D) ข้อมูลที่ใช้ pre-train มีจำนวนน้อยเกินไป

### n09-L2-04  [unit 9] [level 2] [mcq]  topic: Margin ranking vs triplet loss
ถ้าใช้ cosine distance d = 1 - sim ใน Triplet loss max(0, d(a,p) - d(a,n) + margin) จะเทียบเท่ากับสูตรใดของ Margin Ranking Loss
A) max(0, sim(a,p) - sim(a,n) + margin)
B) max(0, sim(a,p) + sim(a,n) - margin)
C) max(0, sim(a,n) - sim(a,p) + margin)
D) max(0, sim(a,n) - sim(a,p) - margin)

### n09-L2-05  [unit 9] [level 2] [mcq]  topic: Vector memory estimate
คลังเอกสาร 2,000,000 เอกสาร ใช้โมเดล all-MiniLM-L6-v2 (384 มิติ) เก็บเวกเตอร์เป็น float32 (4 bytes ต่อค่า) ต้องใช้หน่วยความจำของเวกเตอร์ประมาณเท่าใด (1 GB = 10^9 bytes)
A) 0.768 GB
B) 30.72 GB
C) 6.144 GB
D) 3.072 GB

### n09-L2-06  [unit 9] [level 2] [mcq]  topic: MNRL in-batch negatives
ฝึกด้วย Multiple Negatives Ranking Loss (MNRL) ด้วย batch ที่มี N = 128 คู่ (q_i, p_i) แต่ละ query มี in-batch negatives กี่ตัว
A) 127
B) 128
C) 256
D) 1

### n09-L3-01  [unit 9] [level 3] [mcq]  topic: Cross-encoder all-pairs time
ต้องการเทียบทุกคู่ของ 2,000 ประโยคด้วย cross-encoder ที่ใช้เวลา 4 ms ต่อคู่ (GPU เดียว) ใช้เวลาประมาณเท่าใด
A) 4.44 ชั่วโมง
B) 0.22 ชั่วโมง
C) 2.22 ชั่วโมง
D) 22.2 ชั่วโมง

### n09-L3-02  [unit 9] [level 3] [mcq]  topic: Semi-hard negative and loss
Triplet loss มี d(a,p) = 0.3 และ margin = 0.5 มี negative 3 ตัว n1, n2, n3 ที่อยู่ห่างจาก anchor d(a,n) = 0.2, 0.5, 0.9 ตามลำดับ ข้อใดถูกต้อง
A) n1 เป็น semi-hard negative และมี loss = 0.6
B) n2 เป็น semi-hard negative และมี loss = 0.3
C) n3 เป็น hard negative และมี loss = 0
D) n2 เป็น easy negative จึงมี loss = 0

### n09-L3-03  [unit 9] [level 3] [mcq]  topic: Spot the wrong statement about SBERT
ข้อใดกล่าวเกี่ยวกับ SBERT และ bi-encoder ไม่ถูกต้อง
A) เวกเตอร์ของเอกสารจาก bi-encoder สามารถคำนวณล่วงหน้าและเก็บไว้ใช้ซ้ำได้เสมอ
B) Cross-encoder ให้คุณภาพสูงเพราะมี cross-attention เต็ม แต่ต้องรันใหม่ทุกคู่
C) SBERT แบบ classification ป้อน [u; v; |u-v|] เข้า softmax classifier
D) freeze BERT แล้วฝึกเฉพาะ pooling กับ classifier ก็แก้ anisotropy ได้

### n09-L3-04  [unit 9] [level 3] [mcq]  topic: ANN speed claim needs Recall@k
บทความหนึ่งวัดว่า HNSW เร็วกว่า IndexFlatIP ประมาณ 100 เท่า แล้วสรุปว่า HNSW ดีกว่า Flat ในทุกด้าน ควรตอบอย่างไรตามสไลด์
A) ไม่ครบ เพราะ ANN แลก accuracy กับ latency ต้องรายงาน Recall@k ด้วย
B) สรุปถูกต้อง เพราะ HNSW มี recall 100% เท่ากับ Flat ทุกกรณีจึงเปรียบเทียบความเร็วได้เลย
C) สรุปถูกต้อง เพราะความเร็วสำคัญกว่าความแม่นยำเสมอในระบบ production ทุกระบบที่มีผู้ใช้จำนวนมาก
D) สรุปผิด เพราะ Flat เร็วกว่า HNSW เสมอเมื่อจำนวนเอกสาร n มากกว่า 100,000 รายการขึ้นไป

### n09-L3-05  [unit 9] [level 3] [mcq]  topic: Why normalize before IP search
query q = [3, 4] และเอกสาร d1 = [1.5, 2] (ทิศเดียวกับ q แต่สั้นครึ่งหนึ่ง) กับ d2 = [4, 3] ถ้าไม่ได้ L2-normalize เอกสารใดอันดับ 1 เมื่อใช้ cosine, dot product และ L2 distance ตามลำดับ
A) d1, d1, d1
B) d1, d2, d2
C) d2, d1, d1
D) d1, d1, d2

### n09-L3-06  [unit 9] [level 3] [mcq]  topic: Masked mean pooling and pair vector
ประโยค A มี H_A = [[2,4],[4,8],[100,100]] และประโยค B มี H_B = [[1,3],[7,5],[40,40]] (d = 2) โดยแถวที่ 3 ของทั้งสองเป็น [PAD] attention mask = [1,1,0] ใช้ mean pooling แล้วสร้าง c = [u; v; |u-v|] ได้เวกเตอร์ใด
A) [3, 6, 4, 4, -1, 2]
B) [35.3, 37.3, 16, 16, 19.3, 21.3]
C) [3, 6, 4, 4, 1, 2]
D) [4, 8, 7, 5, 3, 3]

### n09-L4-01  [unit 9] [level 4] [mcq]  topic: Unsupervised SimCSE without dropout
ทีมงานฝึก unsupervised SimCSE แต่เผลอปิด dropout (โมเดลอยู่ใน eval mode) ทั้งสองครั้งที่ป้อนประโยคเดียวกัน ผลที่ตามมาคือข้อใด
A) ต้องกลับไปใช้ entailment label จาก NLI เป็น positive แทน จึงจะฝึกต่อได้ตามปกติและไม่มีปัญหาอื่น
B) negatives ภายใน batch กลายเป็นเวกเตอร์เดียวกันหมดทุกตัว ทำให้ loss เป็นศูนย์ทันทีในทุก batch
C) ไม่มีผลต่างจากเดิมเลย เพราะ positive ของแต่ละประโยคมาจากประโยคอื่นภายใน batch เดียวกัน
D) สอง view เหมือนกันเป๊ะ ไม่มี noise จาก dropout เป็น augmentation positive จึงไม่ให้สัญญาณเรียนรู้

### n09-L4-02  [unit 9] [level 4] [mcq]  topic: Same triplet under different losses
a = [1, 0], p = [0.8, 0.6], n = [0, 1] คำนวณ (i) Triplet loss แบบ cosine distance (d = 1 - cos) ด้วย margin 0.5, (ii) Margin Ranking Loss ด้วย sim = cos และ margin 0.5, (iii) Triplet loss แบบ Euclidean ด้วย margin 1 ผลคือข้อใด
A) (i) = 0, (ii) = 0, (iii) ประมาณ 0.218
B) (i) = 0, (ii) = 0, (iii) = 0
C) (i) = 0.3, (ii) = 0.3, (iii) ประมาณ 0.218
D) (i) = 0, (ii) = 1.3, (iii) ประมาณ 0.218

### n09-L4-03  [unit 9] [level 4] [mcq]  topic: Retrieve-then-rerank speedup
คลังมี 1,000,000 เอกสาร cross-encoder ใช้ 5 ms ต่อคู่ (query, doc) ระบบ A ใช้ cross-encoder กับทุกเอกสารต่อ 1 query ระบบ B ใช้ bi-encoder (เวกเตอร์เอกสาร encode ไว้ล่วงหน้าแล้ว) encode query 5 ms + FAISS search 2 ms แล้ว rerank top-100 ด้วย cross-encoder ต่อ 1 query ระบบ B เร็วกว่า A ประมาณกี่เท่า
A) ประมาณ 986 เท่า
B) ประมาณ 9,860 เท่า
C) ประมาณ 98,600 เท่า
D) ประมาณ 100 เท่า

### n09-L4-04  [unit 9] [level 4] [mcq]  topic: SBERT head size when changing encoder
ทีมเปลี่ยน encoder จาก BERT-base (d = 768) เป็น BAAI/bge-large-en-v1.5 (d = 1024) แต่ยังใช้ SBERT classification objective แบบ NLI 3 labels (softmax(W c), c = [u; v; |u-v|]) จำนวนน้ำหนักใน W (ไม่นับ bias) เพิ่มขึ้นกี่ตัว
A) 9,216
B) 3,072
C) 2,304
D) 6,912

### n09-L4-05  [unit 9] [level 4] [mcq]  topic: Anisotropy breaks thresholding
ทีมใช้ raw BERT (mean pooling) ตรวจประโยคซ้ำ โดยตั้งกฎ cosine >= 0.40 จึงถือว่าซ้ำ ในสไลด์ raw BERT มี cosine ของคู่ประโยคอยู่ในช่วง [0.46, 0.85] (เฉลี่ย 0.58) ส่วน SBERT อยู่ในช่วง [0.1, 0.99] ผลของกฎนี้กับ raw BERT เป็นอย่างไร
A) แทบไม่มีคู่ใดถูกตัดสินว่าซ้ำ เพราะ cosine ของ BERT มักต่ำกว่า 0.40 อยู่แล้ว
B) ประมาณครึ่งหนึ่งของคู่ถูกตัดสินว่าซ้ำ เพราะค่าเฉลี่ย 0.58 อยู่เหนือเส้นแบ่งพอดี
C) ใช้ได้ดี เพราะ 0.40 ต่ำกว่าค่าเฉลี่ย 0.58 และช่วยจับคู่ที่คล้ายกันจริงได้ครบ
D) ทุกคู่ถูกตัดสินว่าซ้ำ เพราะค่าต่ำสุด 0.46 ยังสูงกว่า 0.40

### n09-L4-06  [unit 9] [level 4] [mcq]  topic: Index choice under RAM limit
คลังมี 20,000,000 เอกสาร เวกเตอร์ 768 มิติ float32 เซิร์ฟเวอร์มี RAM 32 GB (1 GB = 10^9 bytes) ข้อสรุปใดสมเหตุสมผลที่สุดตามสไลด์
A) ใช้ HNSW เก็บเวกเตอร์แบบ float32 ครบทั้งหมดได้ใน RAM 32 GB เพราะ HNSW มี recall ประมาณ 99%
B) ต้องใช้ 61.44 GB เกิน RAM ควรใช้ index แบบบีบอัด เช่น IVF+PQ (recall ~90%) แล้ววัด Recall@k
C) ใช้ IndexFlatIP ได้เลย เพราะเวกเตอร์ใช้เพียงประมาณ 15.4 GB จึงพอดีกับ RAM ขนาด 32 GB
D) เปลี่ยนจาก IndexFlatIP เป็น IndexFlatL2 จะประหยัดหน่วยความจำจนพอดีกับ RAM ขนาด 32 GB ได้

### n09-L1-F1  [unit 9] [level 1] [fill]  topic: SBERT classification head dimension
SBERT แบบ classification ต่อเวกเตอร์เป็น c = [u; v; |u-v|] สำหรับ BERT-base (d = 768) เวกเตอร์ c มี ___ มิติ แล้วส่งเข้า linear layer และ softmax

### n09-L2-F1  [unit 9] [level 2] [fill]  topic: Cross-encoder all-pairs cost
ใช้ cross-encoder เทียบทุกคู่ของ n = 1,000 ประโยค ต้องทำ forward pass ___ ครั้ง ถ้าแต่ละครั้งใช้ 5 ms จะใช้เวลารวม ___ วินาที

### n09-L3-F1  [unit 9] [level 3] [fill]  topic: Triplet loss with Euclidean distance
Triplet loss ใช้ระยะ Euclidean a = (0, 0), p = (0.3, 0.4), n = (0.6, 0.8) และ margin = 0.8 จะได้ d(a,p) = ___ , d(a,n) = ___ , loss = ___ และ negative ตัวนี้เป็นแบบ ___ (easy, hard หรือ semi-hard)

### n09-L4-F1  [unit 9] [level 4] [fill]  topic: Margin ranking loss with cosine
Margin Ranking Loss = max(0, sim(a,n) - sim(a,p) + margin) ใช้ cosine เป็น sim ให้ a = [3, 4], p = [4, 3], n = [5, 12] และ margin = 0.5 จะได้ sim(a,p) = ___ , sim(a,n) = ___ (ทศนิยม 4 ตำแหน่ง) และ loss = ___ (ทศนิยม 4 ตำแหน่ง)

### n10-L1-01  [unit 10] [level 1] [mcq]  topic: Weight memory N x bytes
โมเดลมี 13 พันล้าน (13B) parameters เก็บน้ำหนักแบบ FP16 (2 bytes ต่อ parameter) น้ำหนักอย่างเดียวใช้หน่วยความจำประมาณเท่าใด (decimal GB)
A) 13 GB
B) 26 GB
C) 52 GB
D) 6.5 GB

### n10-L1-02  [unit 10] [level 1] [mcq]  topic: GB vs GiB
1 GiB เท่ากับกี่ bytes
A) 1,000,000,000 bytes
B) 1,048,576 bytes
C) 1,073,741,824 bytes
D) 1,024,000,000 bytes

### n10-L1-03  [unit 10] [level 1] [mcq]  topic: TTFT
เมื่อ prompt ยาวมาก เวลาก่อนผู้ใช้เห็น token แรก (TTFT) ส่วนใหญ่ถูกครอบงำโดยส่วนใด
A) Prefill time
B) Network latency
C) Decode time ของ token ที่ 2 เป็นต้นไป
D) เวลา quantize weights

### n10-L1-04  [unit 10] [level 1] [mcq]  topic: Distillation temperature
ใน knowledge distillation การใช้ temperature T > 1 กับ softmax ของ teacher ทำให้การแจกแจงความน่าจะเป็นเป็นอย่างไร
A) แหลมขึ้นจนมี class เดียวได้เกือบ 1.0 เสมอ
B) ไม่เปลี่ยนแปลง เพราะ softmax รวมเป็น 1 อยู่แล้ว
C) ผลรวมความน่าจะเป็นมากกว่า 1 อย่างชัดเจน
D) แบนขึ้น class ที่ความน่าจะเป็นต่ำเห็นชัดขึ้น

### n10-L1-05  [unit 10] [level 1] [mcq]  topic: QLoRA Paged Optimizers
ใน QLoRA ตัว Paged Optimizers ย้าย optimizer states ที่ทำให้ GPU memory พุ่งสูงไปพักไว้ที่ใด
A) Disk SSD (NVMe)
B) CPU RAM (unified memory)
C) SRAM ภายใน GPU
D) VRAM ของ GPU ใบอื่นเสมอ

### n10-L1-06  [unit 10] [level 1] [mcq]  topic: Speculative decoding
ใน speculative decoding โมเดลใดเป็นผู้ตรวจ (verify) candidate tokens ที่ draft สร้างไว้
A) draft model ขนาดเล็กตรวจ token ของตัวเองซ้ำอีกรอบ
B) ผู้ใช้เป็นผู้เลือก token ที่ถูกต้องจากตัวเลือก
C) target model ขนาดใหญ่ ใน forward pass เดียว
D) page table ของ vLLM เป็นผู้ตัดสินให้

### n10-L2-01  [unit 10] [level 2] [mcq]  topic: GB vs GiB conversion
tensor ตัวหนึ่งมีขนาด 3,221,225,472 bytes (= 3 x 2^30) ข้อใดแปลงเป็น GB (10^9 bytes) และ GiB (2^30 bytes) ได้ถูกต้อง
A) GB = 3.46, GiB = 3.22
B) GB = 3.00, GiB = 3.22
C) GB = 3.22, GiB = 3.46
D) GB = 3.22, GiB = 3.00

### n10-L2-02  [unit 10] [level 2] [mcq]  topic: GQA KV cache reduction
LLaMA-3 8B ใช้ GQA โดยมี 32 query heads และ 8 KV heads เมื่อเทียบกับ MHA ที่มี 32 KV heads (ค่าอื่นเท่ากัน) ข้อใดถูกต้อง
A) KV cache เหลือ 1/4 (ลด 75%) และ 4 query heads ใช้ KV head ชุดเดียวกัน
B) KV cache เหลือ 1/8 และ 8 query heads ใช้ KV head ชุดเดียวกัน
C) KV cache เหลือ 1/4 แต่จำนวน query heads ลดเหลือ 8
D) KV cache เหลือ 3/4 (ลด 25%) และ 4 query heads ใช้ KV head ชุดเดียวกัน

### n10-L2-03  [unit 10] [level 2] [mcq]  topic: Serving techniques matching
ข้อใดจับคู่เทคนิคกับปัญหาหลักที่เทคนิคนั้นแก้ได้ถูกต้องทั้งสามคู่
A) PagedAttention แก้ N x N ที่ต้องเขียนลง HBM; FlashAttention แก้ fragmentation ของ KV cache; Continuous batching แก้ GPU ว่างระหว่างรอ
B) FlashAttention แก้ N x N ที่ต้องเขียนลง HBM; PagedAttention แก้ fragmentation ของ KV cache; Continuous batching แก้ GPU ว่างระหว่างรอ
C) Continuous batching แก้ N x N ที่ต้องเขียนลง HBM; PagedAttention แก้ GPU ว่างระหว่างรอ; FlashAttention แก้ fragmentation ของ KV cache
D) FlashAttention แก้ GPU ว่างระหว่างรอ; Continuous batching แก้ fragmentation ของ KV cache; PagedAttention แก้ N x N ที่ต้องเขียนลง HBM

### n10-L2-04  [unit 10] [level 2] [mcq]  topic: LoRA parameter count non-square
LoRA แทน W (ขนาด d x k) ด้วย B x A โดย B เป็น d x r และ A เป็น r x k เมื่อ W_K ใน LLaMA-3 8B มีขนาด 1024 x 4096 (d = 1024, k = 4096) และ r = 16 จำนวน trainable parameters ของเมทริกซ์นี้เท่าใด
A) 131,072
B) 81,920
C) 65,536
D) 4,194,304

### n10-L2-05  [unit 10] [level 2] [mcq]  topic: MoE memory vs compute
Mixtral 8x7B มีพารามิเตอร์รวมประมาณ 47B และ active ประมาณ 13B ต่อ token (top-2 จาก 8 experts) หากโหลดแบบ FP16 ข้อใดประมาณ VRAM สำหรับ weights และ compute ต่อ token ได้ถูกต้อง
A) weights ประมาณ 94 GB; compute ใกล้เคียงโมเดล dense 13B
B) weights ประมาณ 26 GB; compute ใกล้เคียงโมเดล dense 13B
C) weights ประมาณ 94 GB; compute ใกล้เคียงโมเดล dense 47B
D) weights ประมาณ 26 GB; compute ใกล้เคียงโมเดล dense 47B

### n10-L2-06  [unit 10] [level 2] [mcq]  topic: DeepSeekMoE shared vs routed experts
ใน DeepSeekMoE ที่แบ่งเป็น shared experts และ routed micro-experts สองส่วนนี้ต่างกันอย่างไร
A) shared: router เลือก Top-K ต่อ token; routed: active กับทุก token
B) shared: clone มาจาก MLP ของ dense model ตอน upcycling; routed: เทรนใหม่
C) shared: ใช้เฉพาะตอนเทรน; routed: ใช้เฉพาะตอน inference เท่านั้น
D) shared: active ทุก token (ความรู้ทั่วไป); routed: router เลือกบางตัว

### n10-L3-01  [unit 10] [level 3] [mcq]  topic: MoE INT4 fit on 24 GB
ทีมต้องการรัน Mixtral 8x7B (47B params รวม, 13B active) แบบ INT4 (0.5 byte ต่อ parameter) บน RTX 4090 ขนาด 24 GB สมมติ KV cache 0.54 GB และ runtime overhead 1.46 GB เท่ากับ audit ของ LLaMA-3 8B ข้อใดถูกต้อง
A) พอ ใช้ประมาณ 8.5 GB เพราะโหลดเฉพาะ active experts 13B
B) ไม่พอ เพราะ weights ต้อง 47 x 2 = 94 GB เกิน 24 GB มาก
C) ไม่พอ เพราะ weights 23.5 + KV 0.54 + overhead 1.46 = 25.5 GB > 24 GB
D) พอพอดี เพราะ weights 23.5 GB น้อยกว่า 24 GB (ไม่ต้องนับ KV และ overhead)

### n10-L3-02  [unit 10] [level 3] [mcq]  topic: Concurrent requests capacity
GPU 24 GB รัน LLaMA-3 8B FP16 (weights 16 GB, runtime overhead 1.46 GB, ไม่นับ activations) KV cache คิดจากสูตร 2 x Precision x n_layers x (n_kv x d_head) x n_ctx x B โดย n_layers = 32, n_kv = 8, d_head = 128, n_ctx = 4096 รองรับ request พร้อมกันได้มากที่สุดกี่ request (นับเฉพาะจำนวนเต็มที่ใส่ได้จริง)
A) 12
B) 13
C) 14
D) 3

### n10-L3-03  [unit 10] [level 3] [mcq]  topic: Full fine-tuning 16N
Full fine-tuning โมเดล 13B ด้วย Adam ตามการประมาณ 16N (FP16 weights + Adam ไม่ shard ไม่นับ activations) ข้อใดระบุ optimizer states, model states รวม และจำนวน H100 80 GB ขั้นต่ำได้ถูกต้อง
A) optimizer 156 GB; รวม 208 GB; 2 ใบ
B) optimizer 26 GB; รวม 208 GB; 3 ใบ
C) optimizer 156 GB; รวม 182 GB; 3 ใบ
D) optimizer 156 GB; รวม 208 GB; 3 ใบ

### n10-L3-04  [unit 10] [level 3] [mcq]  topic: QLoRA forward/backward mechanics
ข้อใดเกี่ยวกับ QLoRA เป็นข้อความที่ผิด
A) Forward: dequantize block NF4 เป็น 16-bit ใน SRAM แล้วทิ้งหลัง matmul
B) Backward: อัปเดต NF4 base weights ทุก step เพื่อชดเชย quantization error
C) Parameter gradients มีเฉพาะ LoRA adapters (A และ B)
D) Double Quantization quantize ตัว scale factors ไม่ใช่ตัว weights

### n10-L3-05  [unit 10] [level 3] [mcq]  topic: Absmax INT8 round trip
ใช้ symmetric absmax INT8 โดย FP16 range [-3.2, 3.2] map ไป [-127, 127] ด้วย scale s = 3.2/127 = 0.0252 ค่า x = 1.5 เมื่อ quantize แล้ว dequantize กลับ จะได้ค่า q (จำนวนเต็ม) ค่าหลัง dequantize และ error เท่าใด
A) q = 59, dequant = 1.487, error = 0.013
B) q = 60, dequant = 1.512, error = 0.012
C) q = 60, dequant = 1.512, error = 0.12
D) q = 60, dequant = 1.512, error = 0.0012

### n10-L3-06  [unit 10] [level 3] [mcq]  topic: Attention matrix size scaling
สไลด์ระบุว่า standard attention ที่ N = 32,000 เก็บ S = QK^T ประมาณ 2 GB ต่อ head (64 GB สำหรับ 32 heads, FP16) หากลดความยาวลงครึ่งหนึ่งเป็น N = 16,000 โดยยังใช้ FP16 และ 32 heads ที่ standard attention ต้องเก็บ S รวมกี่ GB (decimal)
A) ประมาณ 8.2 GB
B) ประมาณ 32.8 GB
C) ประมาณ 16.4 GB
D) ประมาณ 65.5 GB

### n10-L4-01  [unit 10] [level 4] [mcq]  topic: GB vs GiB trap in KV cache fit
LLaMA-3 8B ใช้ GQA (8 KV heads) เมื่อ seq = 4096 และ batch = 16 KV cache มีขนาด 8,589,934,592 bytes (สไลด์เขียนเป็น 8 GiB หรือ 8.6 GB) torch รายงานหน่วยความจำว่างของ GPU เป็น 8.2 GiB (นับแบบ 2^30) ข้อใดสรุปได้ถูกต้อง
A) ไม่พอ ขาดประมาณ 0.39 GB (เทียบ 8.59 กับ 8.2 ตรงๆ)
B) ไม่พอ ขาดประมาณ 0.95 GB
C) พอ เหลือประมาณ 0.8 GB
D) พอ เหลือประมาณ 0.2 GiB (ประมาณ 0.21 GB)

### n10-L4-02  [unit 10] [level 4] [mcq]  topic: LoRA params with GQA dims and ratio
LLaMA-3 8B (32 layers, d_model = 4096, K/V projection ออกขนาด 1024 เพราะ GQA) ใช้ LoRA rank r = 8 เฉพาะ W_Q (4096 x 4096) และ W_V (1024 x 4096) ทุก layer โดยเมทริกซ์ d x k ใช้ r(d + k) parameters ข้อใดระบุจำนวน trainable params รวม และสัดส่วนต่อ 8B ได้ถูกต้อง
A) ประมาณ 3.41M; ประมาณ 0.043%
B) ประมาณ 4.19M; ประมาณ 0.052%
C) ประมาณ 3.41M; ประมาณ 0.43%
D) ประมาณ 106,496; ประมาณ 0.0013%

### n10-L4-03  [unit 10] [level 4] [mcq]  topic: LoRA OOM diagnosis, QLoRA fix
ทีมใช้ LoRA (r = 16 ที่ Q/K/V/O) fine-tune LLaMA-3 8B บน T4 16 GB ตามสไลด์ประเมินว่า base FP16 (frozen) 16 GB + LoRA weights/grads/Adam 0.22 GB + activations และ overhead ประมาณ 2.0 GB (small batch + checkpointing) = 18.2 GB แล้วเกิด OOM การแก้ข้อใดตรงกับเลขนี้และได้ผลที่สุด
A) ลด rank จาก 16 เหลือ 4 เพราะ LoRA states คือส่วนที่ใหญ่ที่สุดของ 18.2 GB
B) เปิด gradient checkpointing เพิ่ม เพราะ activations คือส่วนที่ใหญ่ที่สุดของ 18.2 GB
C) เปิด Paged Optimizers อย่างเดียว เพราะปัญหาหลักคือ optimizer states ล้น
D) ใช้ NF4 เก็บ base (QLoRA) เพราะ base 16 GB ใหญ่สุด ส่วนอื่นรวม 2.2 GB

### n10-L4-04  [unit 10] [level 4] [mcq]  topic: Roofline and decode bottleneck
Roofline: Attainable FLOP/s = min(Peak FLOP/s, Peak memory BW x AI) โดย AI = FLOPs ต่อ byte ที่ดึง ในช่วง decode (GEMV) กำหนด AI = 1.5 FLOP/byte เลือกจาก GPU ที่ (peak, BW) = P (900 TFLOP/s, 2 TB/s), Q (300 TFLOP/s, 3 TB/s), R (600 TFLOP/s, 2.5 TB/s), S (1200 TFLOP/s, 1.5 TB/s) ข้อใดถูกต้องสำหรับความเร็ว decode ที่ดีที่สุด
A) เลือก P เพราะ peak สูงสุดที่ 900 TFLOP/s ในกลุ่ม P, Q, R
B) เลือก Q ได้ 4.5 TFLOP/s เพราะ BW x AI ต่ำกว่า peak
C) เลือก R ได้ 3.75 TFLOP/s
D) เลือก S ได้ 2.25 TFLOP/s

### n10-L4-05  [unit 10] [level 4] [mcq]  topic: Per-tensor vs block-wise absmax with outlier
เทนเซอร์มี 256 ค่า แบ่งเป็น 4 block ละ 64 ค่า โดย block 1 มี outlier 12.7 ส่วนค่าอื่นทุกตัวใน block 2 ถึง 4 มีขนาดไม่เกิน 1.27 quantize เป็น INT8 แบบ absmax ([-127, 127]) พิจารณาค่า w = 0.04 หนึ่งตัวใน block 3 และอีกหนึ่งตัวใน block 1 (ข้าง outlier) ข้อใดถูกต้อง
A) per-tensor: s = 0.1 ทำให้ 0.04 เป็น 0 ทั้งสองค่า; block-wise: ค่าใน block 3 ได้ q = 4 (s = 0.01) แต่ค่าใน block 1 ยังเป็น 0
B) per-tensor: s = 0.1 ทำให้ 0.04 เป็น 0; block-wise: ทั้งสองค่าได้ q = 4 เพราะทุก block ใช้ s = 0.01 เหมือนกัน
C) per-tensor: s = 0.01 ทำให้ 0.04 ได้ q = 4 อยู่แล้ว ส่วน block-wise ก็ให้ผลไม่ต่างจาก per-tensor เลย
D) per-tensor และ block-wise ให้ผลเหมือนกัน เพราะ absmax ใช้ค่าสูงสุดของทั้งเทนเซอร์เสมอ ไม่ว่าจะแบ่ง block หรือไม่

### n10-L4-06  [unit 10] [level 4] [mcq]  topic: PagedAttention utilization and capacity
KV pool ของ GPU ขนาด 12 GB แต่ละ request ใช้ KV จริงเฉลี่ย 0.3 GB ระบบแบบ static allocation ใช้ KV memory ที่จองไว้ได้จริงเพียง 30% ส่วน PagedAttention ใช้ได้ประมาณ 95% (ตัวเลขจากสไลด์) แต่ละระบบรองรับกี่ request พร้อมกัน และเพิ่มขึ้นกี่เท่า
A) static 12, paged 40 (ประมาณ 3.3 เท่า)
B) static 36, paged 38 (ประมาณ 1.06 เท่า)
C) static 12, paged 38 (ประมาณ 3.2 เท่า)
D) static 4, paged 38 (ประมาณ 9.5 เท่า)

### n10-L1-F1  [unit 10] [level 1] [fill]  topic: 16N training memory
Full fine-tuning ด้วย Adam ตามสไลด์ใช้ 16N bytes = weights 2N + gradients 2N + optimizer states ___N

### n10-L2-F1  [unit 10] [level 2] [fill]  topic: GB to GiB conversion
โมเดล 70B parameters เก็บแบบ FP16 ใช้ 140 GB (decimal) เมื่ออ่านด้วย torch.cuda.memory_allocated แล้วแปลงเป็น GiB จะได้ประมาณ ___ GiB (ทศนิยม 1 ตำแหน่ง)

### n10-L3-F1  [unit 10] [level 3] [fill]  topic: Absmax INT8 calculation
symmetric absmax INT8 ([-127, 127]) กับ weights ที่ max|w| = 2.54 ได้ scale s = ___ ค่า x = 0.904 ถูกเก็บเป็นจำนวนเต็ม q = ___ และเมื่อ dequantize ได้ ___

### n10-L4-F1  [unit 10] [level 4] [fill]  topic: KV cache per token and long context
LLaMA-3 8B (n_layers = 32, n_kv = 8, d_head = 128, FP16) KV cache ต่อ 1 token = ___ bytes ที่ context 128,000 tokens batch 1 ได้ ___ GB (decimal ทศนิยม 2 ตำแหน่ง) และถ้าเป็น MHA ที่มี 32 KV heads จะได้ ___ GB (decimal ทศนิยม 2 ตำแหน่ง)

### n11-L1-01  [unit 11] [level 1] [mcq]  topic: ChatML roles
ใน ChatML (Chat Markup Language) role ใดทำหน้าที่กำหนดพฤติกรรมโดยรวม persona และ constraints ของโมเดล
A) Assistant
B) User
C) System
D) Critic

### n11-L1-02  [unit 11] [level 1] [mcq]  topic: apply_chat_template (demo)
ใน demo เรียก tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) ค่า add_generation_prompt=True ทำอะไร
A) เติมส่วนหัวของ assistant turn ไว้ท้าย prompt เพื่อให้โมเดลเริ่มสร้างคำตอบต่อจากนั้น
B) แปลงข้อความทั้งหมดเป็น token ID ทันทีโดยไม่ต้องเรียก tokenizer อีก
C) เติม stop token ท้ายคำตอบของโมเดลเพื่อจบ generation loop ให้เรียบร้อย
D) เพิ่ม temperature ให้โมเดลสุ่มคำตอบหลากหลายขึ้นในรอบ generation แรก

### n11-L1-03  [unit 11] [level 1] [mcq]  topic: RLHF reward model architecture
ใน RLHF Step 2 Reward Model ถูกสร้างโดย initialize จาก SFT model แล้วเปลี่ยน LM head เป็นอะไร
A) softmax head ขนาดเท่า vocabulary
B) binary classification head ตอบแค่ winner หรือ loser
C) embedding lookup head สำหรับ special tokens
D) scalar regression head ที่ให้ค่ารางวัลหนึ่งค่า

### n11-L1-04  [unit 11] [level 1] [mcq]  topic: Reflection and self-critique
วงจร Reflection and Self-Critique ตามสไลด์มีลำดับขั้นตอนใด
A) Critique, Generate, Refine
B) Generate, Critique, Refine
C) Generate, Refine, Critique
D) Refine, Generate, Critique

### n11-L1-05  [unit 11] [level 1] [mcq]  topic: Sandwich defense
Sandwich defense ใน prompt security คือวิธีใด
A) วาง user input ไว้ระหว่าง system reminder ด้านบน (Header) และด้านล่าง (Footer)
B) ใช้ LLM ตัวที่สองตัดสินว่าคำตอบของ LLM ตัวแรกปลอดภัยหรือไม่
C) กรอง keyword อันตรายออกจาก prompt ของผู้ใช้ก่อนส่งเข้า LLM (input filtering)
D) จำกัดสิทธิ์ของ tools และ APIs ที่ LLM เรียกใช้ได้ และให้คนยืนยันก่อนทำสิ่งเสี่ยง

### n11-L1-06  [unit 11] [level 1] [mcq]  topic: Few-shot prompting
Prompt: 'Bad -> Negative / Good -> Positive / Amazing ->' จัดเป็นเทคนิค prompting แบบใด
A) Zero-shot prompting
B) Chain-of-Thought prompting
C) Program-of-Thought prompting
D) Few-shot prompting (k = 2)

### n11-L2-01  [unit 11] [level 2] [mcq]  topic: DPO loss calculation
DPO ใช้ beta = 0.5 ตัวอย่างหนึ่งมี log ratio ของ chosen = log pi(yw|x) - log pi_ref(yw|x) = +1.5 และ log ratio ของ rejected = -0.5 กำหนด e^-1 = 0.3679 ค่า loss = -ln sigma( beta x (ratio_w - ratio_l) ) ใกล้เคียงข้อใดที่สุด
A) 1.3133
B) 0.3133
C) 0.1269
D) 0.6931

### n11-L2-02  [unit 11] [level 2] [mcq]  topic: CO-STAR components
Prompt ตามกรอบ CO-STAR มี 2 บรรทัดคือ (1) 'ใช้น้ำเสียงเห็นอกเห็นใจ (empathetic)' และ (2) 'เขียนแบบทางการ (formal)' ตามลำดับ เป็นองค์ประกอบใดของ CO-STAR
A) Tone และ Style
B) Style และ Tone
C) Audience และ Context
D) Objective และ Response format

### n11-L2-03  [unit 11] [level 2] [mcq]  topic: Zero-shot CoT vs few-shot CoT
ข้อใดอธิบายความแตกต่างระหว่าง Zero-shot CoT กับ Few-shot CoT (แบบ Wei et al.) ได้ถูกต้อง
A) Zero-shot CoT ต้อง fine-tune weights ด้วยโจทย์เลข ส่วน Few-shot CoT ไม่ต้องอัปเดต weights
B) Zero-shot CoT ใส่ตัวอย่างพร้อมขั้นตอนเหตุผลหลายตัวอย่าง ส่วน Few-shot CoT ใช้วลีกระตุ้นแทนตัวอย่าง
C) ทั้งสองแบบต้องใช้ตัวอย่างพร้อมเฉลยเท่ากัน ต่างกันเพียงค่า temperature ที่ตั้งตอนสร้างคำตอบ
D) Zero-shot CoT ไม่ใส่ตัวอย่างแต่ต่อท้ายด้วย Let's think step by step ส่วน Few-shot CoT ใส่ตัวอย่างที่มีเหตุผลเป็นขั้นตอน

### n11-L2-04  [unit 11] [level 2] [mcq]  topic: Reward model Bradley-Terry loss
Reward Model ให้ r(x,yw) = 3.0 และ r(x,yl) = 1.0 กำหนด e^-2 = 0.1353 ค่า L_RM = -log sigma( r(x,yw) - r(x,yl) ) (ln ธรรมชาติ) ใกล้เคียงข้อใดที่สุด
A) 0.8808
B) 2.1269
C) 0.1269
D) 0.3133

### n11-L2-05  [unit 11] [level 2] [mcq]  topic: Base model behavior
ป้อน 'The capital of France is' ให้ base model แล้วได้ 'Paris. The capital of Italy is Rome.' ต่อไปเรื่อย ๆ ไม่หยุด สาเหตุหลักตามสไลด์คือข้อใด
A) base model ฝึกด้วย next-token prediction บนข้อความเว็บ จึงแค่ continue เอกสารและไม่รู้ว่าต้องหยุดเมื่อตอบเสร็จ
B) base model ถูก SFT มากเกินไปจน overfit ตัวอย่างถาม-ตอบสั้น ๆ ทำให้เพิ่มประโยคเองต่อท้ายไม่หยุด
C) base model เกิด reward hacking เพราะ KL penalty ต่ำเกินไปในขั้น PPO ของ RLHF
D) base model โดน prompt injection จากข้อมูลภายนอกที่ฝังอยู่ใน prompt ที่ป้อนเข้ามา

### n11-L2-06  [unit 11] [level 2] [mcq]  topic: Chat template and special tokens
ข้อใดอธิบายบทบาทของ chat template และ special tokens ในโมเดลที่ผ่าน SFT ได้ถูกต้อง
A) เป็นเพียงรูปแบบแสดงผลภายนอก จึงส่งข้อความดิบโดยไม่ใช้ template ก็ได้ผลเท่าเดิมกับโมเดลที่ผ่าน SFT แล้ว
B) ทุก model family ใช้ stop token ตัวเดียวกัน คือ <|im_end|> ไม่ว่าจะเป็น ChatML หรือ Llama 3
C) ใน SFT โมเดลเรียนว่าเห็น assistant header ต้องตอบ และจบด้วย stop token ของ family นั้น จึงต้องใช้ template ตรงกันตอน inference
D) template ทำให้จำนวน parameters ของโมเดลเพิ่มขึ้นเพื่อรองรับ role ทั้ง 3 ของบทสนทนา

### n11-L3-01  [unit 11] [level 3] [mcq]  topic: DPO loss from per-token log probs
คู่ preference หนึ่งคู่ ใช้ beta = 0.1 และ log prob ของคำตอบ = ผลรวม log prob ทุก token ข้อมูลต่อ token ดังนี้ chosen ยาว 30 tokens: policy เฉลี่ย -2.0 ต่อ token, ref เฉลี่ย -2.1 ต่อ token และ rejected ยาว 20 tokens: policy เฉลี่ย -3.2 ต่อ token, ref เฉลี่ย -3.0 ต่อ token กำหนด e^-0.7 = 0.4966 ค่า DPO loss ใกล้เคียงข้อใดที่สุด
A) 0.6783
B) 0.6931
C) 1.1032
D) 0.4032

### n11-L3-02  [unit 11] [level 3] [mcq]  topic: ToT vs GoT scenario
ทีมแก้ปัญหาที่ต้องแตกเป็นหลาย sub-solution จาก path ที่แยกกัน แล้วรวมหลาย thought เข้าเป็น thought เดียวที่แข็งแรงกว่า และวนกลับไปปรับปรุง thought เดิมได้ โครงสร้าง reasoning แบบใดตรงกับงานนี้ที่สุด
A) Chain-of-Thought เพราะเป็นเส้นตรงจึงควบคุมง่าย
B) Graph-of-Thoughts เพราะมี aggregation และ refinement (self-loop)
C) Tree-of-Thoughts เพราะ prune branch ที่แย่แล้วขุดต่อ branch ที่ดี
D) Self-consistency เพราะ sample หลาย path แล้วโหวต

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

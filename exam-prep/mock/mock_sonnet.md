# Mock exam — Sonnet set

## Part A — MCQ

**Q1** [W8] [BART noising] [difficulty: easy]
ข้อใด **ไม่ใช่** หนึ่งในวิธี noising 5 แบบของ BART ตามสไลด์?
A) Token Deletion
B) Document Rotation
C) Span Corruption ที่ใช้ sentinel token แบบ `<extra_id_N>`
D) Sentence Permutation

**Q2** [W8] [BART text infilling] [difficulty: medium]
ใน Text Infilling ของ BART ความยาวของ span ที่ถูกตัดสุ่มจากการแจกแจงใด และ span หนึ่งถูกแทนด้วยอะไร?
A) Poisson (lambda = 1) และแทน token ละ 1 `[MASK]`
B) Poisson (lambda = 3) และแทนทั้ง span ด้วย `[MASK]` ตัวเดียว
C) Poisson (lambda = 3) และแทนด้วย sentinel ที่มีเลขลำดับไม่ซ้ำกัน
D) Poisson (lambda = 5) และแทนทั้ง span ด้วย `[MASK]` ตัวเดียว

**Q3** [W8] [BART token deletion] [difficulty: medium]
ข้อใดอธิบาย Token Deletion ของ BART ได้ถูกต้อง?
A) ลบ token ทิ้งโดยไม่เหลือ placeholder ทำให้โมเดลต้องตรวจเองว่าอะไรหายและหายที่ไหน (src_len < tgt_len)
B) แทน token ที่ถูกลบด้วย `[MASK]` ตัวเดียว ทำให้ src_len = tgt_len
C) สลับลำดับประโยคแบบสุ่มแล้วให้เรียงกลับ
D) แทน span ด้วย sentinel `<extra_id_N>` ที่ไม่ซ้ำกัน

**Q4** [W8] [T5 span corruption] [difficulty: medium]
ข้อความต้นฉบับ "The cute cat sat on the fuzzy mat" ถูกตัดเป็น encoder input `The <extra_id_0> cat sat on the <extra_id_1>` Decoder target ของ T5 คือข้อใด?
A) `cute fuzzy mat </s>`
B) `<extra_id_0> cute <extra_id_1> fuzzy mat </s>`
C) `The cute cat sat on the fuzzy mat </s>`
D) `<extra_id_0> cute <extra_id_1> fuzzy mat <extra_id_2> </s>`

**Q5** [W8] [BART vs T5 target length] [difficulty: medium]
ประโยค 12 token ถูกตัด 2 span (span ละ 2 token รวม 4 token ที่หายไป) ถ้า T5 ใช้ sentinel แทนแต่ละ span อยู่แล้ว target ของ T5 ยาวกี่ token (ไม่นับ `</s>`)?
A) 4
B) 7
C) 8
D) 12

**Q6** [W8] [T5 compute efficiency] [difficulty: medium]
ตามสไลด์ target ของ T5 ยาวประมาณ 0.20N (15% corrupted + 5% sentinels) เมื่อเทียบกับ target ยาว N ของ BART decoder self-attention ของ T5 ถูกกว่าประมาณกี่เท่า?
A) 5 เท่า
B) 2-3 เท่า
C) 25 เท่า
D) 50 เท่า

**Q7** [W8] [T5 relative position bias] [difficulty: medium]
ข้อใดอธิบาย position encoding ของ T5 ได้ถูกต้อง?
A) ใช้ตาราง learned bias 32 buckets ต่อ head บวกเข้า attention logits โดย offset 0-7 มี bucket ของตัวเอง ส่วน offset ไกลกว่านั้นแชร์ bucket
B) ใช้ sinusoidal absolute positional encoding แบบเดียวกับ Transformer ดั้งเดิม
C) ใช้ learned scalar คูณกับ attention weights หลัง softmax
D) ใช้ RoPE หมุน Q และ K ตามตำแหน่ง

**Q8** [W8] [MLM vs CLM signal] [difficulty: easy]
ข้อความยาว 100 token: Encoder-only แบบ MLM (mask 15%) กับ Decoder-only แบบ CLM ใช้ token เป็น training signal กี่ token ตามลำดับ?
A) 100 และ 100
B) 15 และ 15
C) 100 และ 15
D) 15 และ 100

**Q9** [W8] [Why decoder-only won] [difficulty: medium]
ข้อใดเป็นหนึ่งในเหตุผลเชิงโครงสร้างที่สไลด์ใช้อธิบายว่าทำไม frontier model ส่วนใหญ่เป็น Decoder-only?
A) Self-attention ของ Decoder-only มีต้นทุนเป็น O(N) ขณะที่ Enc-Dec เป็น O(N^2)
B) KV cache โตเชิงเส้นอย่างคาดเดาได้ (cache เดียว) จึงรองรับ PagedAttention, chunked prefill และ prefix caching ได้ง่าย
C) Enc-Dec มีพารามิเตอร์น้อยกว่าจึงเรียนรู้ได้ช้ากว่า
D) มีการเปรียบเทียบแบบควบคุมที่ขนาดเกิน 100B ซึ่งพิสูจน์ว่า Decoder-only ชนะ

**Q10** [W8] [T5 prefix, demo-only] [difficulty: medium]
ใน demo ของสัปดาห์ที่ 8 เมื่อส่งประโยครีวิวภาษาอังกฤษให้ T5-small โดย **ไม่ใส่ task prefix** ผลเป็นอย่างไร?
A) คืนประโยคเดิมกลับมาเหมือนเดิม
B) เกิด error เพราะไม่มี prefix
C) ได้ข้อความภาษาเยอรมัน (fallback ไปทำ translation)
D) ตอบ "positive"

**Q11** [W9] [SBERT classification head] [difficulty: easy]
SBERT (BERT-base, d = 768) ในโหมด classification (NLI) ต่อเวกเตอร์ [u; v; |u-v|] ก่อนเข้า softmax layer เวกเตอร์ที่ได้มีกี่มิติ?
A) 768
B) 1,536
C) 3,072
D) 2,304

**Q12** [W9] [Triplet loss] [difficulty: medium]
Triplet loss L = max(0, d(a,p) - d(a,n) + alpha) จะมีค่าเป็น 0 เมื่อใด?
A) d(a,p) + alpha <= d(a,n) คือ positive ใกล้ anchor กว่า negative อย่างน้อยเท่า margin
B) d(a,p) > d(a,n)
C) d(a,p) = d(a,n)
D) เมื่อ alpha = 0 เท่านั้น

**Q13** [W9] [Triplet loss calc] [difficulty: easy]
กำหนด d(a,p) = 0.1, d(a,n) = 0.2 และ margin alpha = 0.5 ค่า triplet loss เท่าใด?
A) 0
B) 0.2
C) 0.4
D) 0.8

**Q14** [W9] [Cross-encoder cost] [difficulty: easy]
ต้องการเทียบความคล้ายของ 1,000 ประโยคทุกคู่ด้วย Cross-encoder ต้องรัน forward pass กี่ครั้ง?
A) 1,000
B) 499,500
C) 999,000
D) 1,000,000

**Q15** [W9] [Bi-encoder vs cross-encoder O()] [difficulty: medium]
ข้อใดอธิบายต้นทุนของการเทียบทุกคู่ของ n ประโยคได้ถูกต้อง?
A) ทั้ง Bi-encoder และ Cross-encoder ต้องรัน BERT forward pass จำนวน n^2 ครั้ง
B) Bi-encoder ต้อง forward n^2 ครั้ง ส่วน Cross-encoder forward n ครั้ง
C) Bi-encoder ไม่ต้อง forward เลยเพราะ cache เวกเตอร์ไว้ล่วงหน้าได้
D) Cross-encoder ต้อง forward ประมาณ n^2 ครั้ง ส่วน Bi-encoder forward n ครั้งแล้วคำนวณ dot product n^2 ครั้ง (ราคาถูกมาก)

**Q16** [W9] [Raw BERT on STS] [difficulty: medium]
บน STS Benchmark (Spearman x100) BERT [CLS] ที่ไม่ได้ fine-tune ได้คะแนนเท่าใดตามสไลด์?
A) 16.50
B) 46.35
C) 58.02
D) 77.03

**Q17** [W9] [SBERT pooling] [difficulty: easy]
Pooling ที่เป็นค่า default และให้ผลดีที่สุดใน SBERT คืออะไร?
A) ใช้ [CLS] token ตัวแรก
B) Mean pooling โดยใช้ attention mask ตัด [PAD] ออก
C) Max pooling ต่อ dimension
D) ใช้ hidden state ของ token สุดท้าย

**Q18** [W9] [MNRL in-batch negatives] [difficulty: easy]
ใน MNRL / InfoNCE ที่ batch มี N = 256 คู่ (q_i, p_i) แต่ละ query มี in-batch negatives กี่ตัว?
A) 256
B) 128
C) 255
D) 65,535

**Q19** [W9] [FAISS index] [difficulty: medium]
หลัง L2-normalize เวกเตอร์เองแล้ว index ใดของ FAISS ให้คะแนนเป็น cosine similarity ที่ค่ามาก = ดี?
A) `IndexFlatIP`
B) `IndexFlatL2`
C) `IndexHNSWFlat` ที่ไม่ระบุ metric
D) IVF+PQ ที่ใช้ L2

**Q20** [W9] [MRR] [difficulty: easy]
สามคำถามพบเอกสารที่ถูกต้องอันแรกที่อันดับ 1, 2 และ 4 ตามลำดับ ค่า MRR เท่าใด?
A) 2.33
B) 0.25
C) 0.667
D) 0.583

**Q21** [W10] [GB vs GiB] [difficulty: easy]
โมเดลที่ใช้หน่วยความจำ 16 GB (decimal, 10^9 byte) เท่ากับประมาณกี่ GiB (2^30 byte)?
A) 17.2 GiB
B) 14.9 GiB
C) 16.0 GiB
D) 13.6 GiB

**Q22** [W10] [Weight memory] [difficulty: easy]
โมเดล 70B parameter เก็บ weight แบบ INT4 (0.5 byte ต่อ parameter) ใช้หน่วยความจำเท่าใด (decimal GB)?
A) 140 GB
B) 70 GB
C) 17.5 GB
D) 35 GB

**Q23** [W10] [GQA KV cache] [difficulty: medium]
โมเดลมี 64 query heads และใช้ GQA แบ่งเป็น 8 กลุ่ม KV ถ้าแบบ MHA ต้องใช้ KV cache 160 GB แบบ GQA จะใช้ KV cache เท่าใด?
A) 20 GB
B) 80 GB
C) 40 GB
D) 2.5 GB

**Q24** [W10] [PagedAttention] [difficulty: medium]
PagedAttention (vLLM) แก้ปัญหาใดและทำอย่างไร?
A) ลด memory ของ attention matrix ด้วย tiling ใน SRAM
B) จัดตาราง request ระดับ iteration เพื่อไม่ให้ GPU ว่าง
C) แบ่ง KV cache เป็น block ขนาดคงที่ที่ไม่ต้องอยู่ติดกัน แล้วใช้ page table ชี้ตำแหน่ง เพื่อลด fragmentation
D) quantize KV cache เป็น 4-bit

**Q25** [W10] [LoRA initialization] [difficulty: easy]
ใน LoRA (delta W = B·A) การ initialize ตอนเริ่มเทรนเป็นแบบใด?
A) A = 0 และ B สุ่มแบบ Gaussian
B) B = 0 และ A สุ่มแบบ Gaussian ทำให้ delta W = 0 ตอนเริ่ม
C) ทั้ง A และ B เป็น 0
D) ทั้ง A และ B สุ่มแบบ Gaussian

**Q26** [W10] [LoRA parameter count] [difficulty: easy]
เมทริกซ์ weight ขนาด 4096 x 4096 ใช้ LoRA rank r = 16 จะมี trainable parameter ต่อเมทริกซ์เท่าใด?
A) 131,072
B) 65,536
C) 16,777,216
D) 262,144

**Q27** [W10] [QLoRA innovations] [difficulty: easy]
สามนวัตกรรมหลักของ QLoRA คือข้อใด?
A) NF4, GPTQ, FlashAttention
B) NF4, LoRA+, Gradient checkpointing
C) INT8, Double Quantization, Paged KV cache
D) 4-bit NormalFloat (NF4), Double Quantization, Paged Optimizers

**Q28** [W10] [MoE] [difficulty: medium]
Mixtral 8x7B มีพารามิเตอร์รวมประมาณ 47B และ active ประมาณ 13B ต่อ token ข้อใดถูกต้อง?
A) ต้องใช้ VRAM เพียงประมาณ 13B x 2 byte
B) MoE ช่วยประหยัด memory เพราะโหลดเฉพาะ expert ที่ใช้
C) MoE ลด compute ต่อ token แต่ต้องโหลด expert ทั้งหมดไว้ใน VRAM จึงไม่ประหยัด memory
D) router เลือก top-1 expert เสมอ

**Q29** [W10] [FlashAttention] [difficulty: medium]
ข้อใดอธิบาย FlashAttention ได้ถูกต้อง?
A) ประมาณค่า attention ด้วย sparse pattern จึงเร็วขึ้นแต่คุณภาพลดลงเล็กน้อย
B) คำนวณ exact attention ด้วย tiling ใน SRAM และ online softmax โดยไม่เขียนเมทริกซ์ N x N ลง HBM (extra memory O(N))
C) ลดจำนวน FLOPs ของ attention ลงอย่างมากด้วยการข้ามบาง token
D) แบ่ง KV cache เป็น page เพื่อลด fragmentation

**Q30** [W10] [Absmax quantization] [difficulty: medium]
Quantize FP16 เป็น INT8 แบบ absmax ช่วง [-3.2, 3.2] map ไป [-127, 127] (scale = 3.2/127 = 0.0252) ค่า 2.1 จะถูกเก็บเป็นจำนวนเต็มใด?
A) 83
B) 84
C) 67
D) 127

**Q31** [W11] [CO-STAR] [difficulty: easy]
CO-STAR framework ประกอบด้วยองค์ประกอบใดตามลำดับ?
A) Context, Objective, Style, Task, Audience, Rules
B) Context, Output, Style, Tone, Action, Reasoning
C) Context, Objective, Style, Tone, Audience, Response format
D) Constraint, Objective, Steps, Tone, Audience, Response

**Q32** [W11] [Self-consistency] [difficulty: medium]
Self-consistency ทำงานอย่างไร?
A) ใช้ greedy decoding (temperature 0) ครั้งเดียวแล้วให้ผู้ใช้ตรวจ
B) ฝึก verifier model แยกเพื่อเลือก reasoning path ที่ดีที่สุด
C) ให้โมเดลวิจารณ์และแก้คำตอบตัวเองวนซ้ำ
D) ตั้ง temperature > 0 สุ่มหลาย independent CoT path (เช่น 5-10 ครั้ง) แล้ว majority vote คำตอบสุดท้าย

**Q33** [W11] [ToT vs GoT] [difficulty: medium]
Graph-of-Thoughts (GoT) ต่างจาก Tree-of-Thoughts (ToT) อย่างไร?
A) GoT เป็น search แบบ BFS/DFS ส่วน ToT เป็น linear chain
B) GoT เป็น arbitrary graph ที่ทำ aggregation (รวมหลาย thought) และ refinement (loop กลับ) ได้ ส่วน ToT เป็น tree ที่ branch แยกกัน
C) GoT ใช้โค้ด Python คำนวณ ส่วน ToT ใช้ majority vote
D) GoT ไม่มีขั้นประเมิน (evaluate) thought

**Q34** [W11] [Program-of-Thought] [difficulty: easy]
Program-of-Thought / PAL แก้จุดอ่อนเรื่อง raw arithmetic ของ LLM อย่างไร?
A) ให้ LLM เขียน Python script แล้วส่งให้ interpreter ภายนอกคำนวณ แยกการ reasoning ออกจากการคำนวณ
B) ให้ LLM ลองคำนวณซ้ำหลายครั้งแล้ว majority vote
C) fine-tune LLM บนโจทย์เลขเพิ่ม
D) ให้ LLM คำนวณทีละหลักใน scratchpad

**Q35** [W11] [RLHF PPO models] [difficulty: medium]
ใน PPO ขั้นที่ 3 ของ RLHF ซึ่งมี 4 model ข้อใดคือ model ที่ถูกอัปเดตระหว่างเทรน?
A) Reward Model และ Reference Model
B) ทั้ง 4 model
C) Actor (Policy) และ Critic (Value)
D) Actor เพียงตัวเดียว

**Q36** [W11] [DPO loss at init] [difficulty: medium]
ตอนเริ่มเทรน DPO ที่ policy = reference model ค่า DPO loss เท่าใด?
A) 0
B) ln 2 ≈ 0.693
C) 1.0
D) beta (เช่น 0.1)

**Q37** [W11] [DPO loss calc] [difficulty: hard]
log ratio ของ chosen = +2, ของ rejected = -2, beta = 0.1 ค่า DPO loss = -ln sigma(beta x (2 - (-2))) เท่าใด?
A) 0.693
B) 0.598
C) 0.127
D) 0.513

**Q38** [W11] [GRPO advantage] [difficulty: hard]
GRPO กลุ่ม G = 4 ได้ reward [1, 0, 0, 0] ใช้ A_i = (r_i - mean)/std โดย std หารด้วย G advantage ของคำตอบที่ได้ reward 1 ประมาณเท่าใด?
A) 1.73
B) 0.75
C) 3.0
D) 1.0

**Q39** [W11] [GRPO vs PPO] [difficulty: easy]
GRPO ตัด model ใดออกจากชุด 4 model ของ PPO?
A) Reference model
B) Reward model
C) Critic (value) model
D) KL penalty

**Q40** [W11] [Base vs instruct, demo-only] [difficulty: medium]
ใน demo สัปดาห์ที่ 11 โมเดล Qwen2.5-0.5B (base) กับ Qwen2.5-0.5B-Instruct มีจำนวนพารามิเตอร์ต่างกันอย่างไร?
A) Instruct มากกว่าเพราะมี reward head เพิ่ม
B) Base มากกว่าเพราะ Instruct ถูก prune
C) Instruct มี embedding เพิ่มสำหรับ chat token
D) เท่ากัน (494.0M) ต่างกันที่วิธีเทรน ไม่ใช่สถาปัตยกรรม

**Q41** [W12] [BM25 TF ceiling] [difficulty: medium]
ส่วน TF ของ BM25 f(k1+1)/(f + k1·L) เมื่อ f เพิ่มขึ้นไม่จำกัด ค่าจะเข้าใกล้เพดานเท่าใด?
A) k1 + 1
B) k1
C) infinity (เพิ่มเรื่อย ๆ เหมือน TF-IDF)
D) 1

**Q42** [W12] [BM25 calc] [difficulty: easy]
กำหนด k1 = 1.5, b = 0 และคำหนึ่งปรากฏ f = 3 ครั้งในเอกสาร ส่วน TF ของ BM25 เท่าใด (ไม่รวม IDF)?
A) 1.25
B) 2.50
C) 1.667
D) 3.00

**Q43** [W12] [BM25 length normalization] [difficulty: medium]
เอกสารสองฉบับมีคำค้นปรากฏจำนวนครั้งเท่ากัน ฉบับแรกสั้นกว่าค่าเฉลี่ย (avgdl) ฉบับที่สองยาวกว่า (b > 0) ข้อใดถูกต้อง?
A) ฉบับยาวได้คะแนนสูงกว่าเพราะมีเนื้อหามากกว่า
B) ฉบับสั้นได้คะแนนสูงกว่า และ b เปลี่ยนเพียงความเร็วที่เข้าใกล้เพดาน k1+1 ไม่ได้เปลี่ยนเพดาน
C) b ที่มากขึ้นทำให้เพดานของ TF เพิ่มขึ้น
D) คะแนนเท่ากันเพราะ f เท่ากัน

**Q44** [W12] [RRF calc] [difficulty: medium]
RRF ที่ k = 60 เอกสาร A อยู่อันดับ 1 ใน BM25 และอันดับ 3 ใน dense retrieval คะแนน RRF ของ A เท่าใด?
A) 0.0164
B) 1.333
C) 0.0500
D) 0.0323

**Q45** [W12] [Why RRF] [difficulty: medium]
เหตุผลหลักที่ใช้ RRF รวมผล BM25 กับ dense retrieval คืออะไร?
A) คะแนนดิบอยู่คนละสเกล (BM25 ไม่มีขอบบน, cosine อยู่ใน [-1,1]) RRF ใช้เฉพาะอันดับจึงรวมกันได้โดยไม่ต้อง normalize
B) RRF ถ่วงน้ำหนักคะแนนดิบตามความแม่นยำของแต่ละระบบ
C) RRF ทำให้ BM25 และ dense ให้คะแนนสเกลเดียวกันโดยอัตโนมัติ
D) RRF เร็วกว่า cross-encoder เพราะไม่ต้องมี retriever

**Q46** [W12] [Cross-encoder reranking] [difficulty: medium]
ทำไม Cross-encoder จึงใช้เป็นขั้น rerank ของ top-K (เช่น 20-100) ไม่ใช่ค้นทั้ง corpus?
A) เพราะ Cross-encoder แม่นน้อยกว่า Bi-encoder
B) เพราะต้อง encode query กับเอกสารร่วมกัน (joint attention) จึง precompute ล่วงหน้าไม่ได้และแพงต่อคู่ แต่ให้ precision สูง
C) เพราะ Cross-encoder ไม่รองรับ query ภาษาธรรมชาติ
D) เพราะ Cross-encoder ใช้ได้กับ sparse index เท่านั้น

**Q47** [W12] [CLIP zero-shot] [difficulty: medium]
CLIP ทำ zero-shot image classification อย่างไร?
A) fine-tune classification head บน ImageNet label
B) ใช้ OCR อ่านข้อความในรูปแล้วจับคู่ชื่อคลาส
C) ใส่ชื่อคลาสลงใน prompt template เช่น "A photo of a {object}." ผ่าน text encoder แล้วเลือกคลาสที่ similarity กับ image embedding สูงสุด
D) ใช้ image encoder สร้าง caption แล้วค้นด้วย BM25

**Q48** [W12] [GraphRAG global search] [difficulty: medium]
GraphRAG ตอบคำถามเชิงภาพรวมของทั้ง corpus (global search) อย่างไร?
A) เดิน graph จาก entity เดียวแล้วเรียก LLM 1 ครั้ง
B) สร้าง community summary ใหม่ทุกครั้งที่ผู้ใช้ถามคำถาม
C) ดึง chunk top-k จาก vector index แล้วสรุปตรง ๆ
D) ทำ Map-Reduce ข้าม community summary ที่ pre-generate ไว้ตอน indexing (Leiden communities)

**Q49** [W12] [Bookend ordering, demo] [difficulty: medium]
Bookend reordering ใน demo นำเอกสาร rank 1-5 มาจัดเป็น [rank 1, rank 3, rank 4, rank 5, rank 2] เพื่อแก้ปัญหาอะไรและให้ผลอย่างไร?
A) แก้ Lost in the Middle โดยวาง rank 1 ไว้ต้นและ rank 2 ไว้ท้าย (โซนที่ LLM ให้ความสนใจสูง)
B) เรียง chunk ตามความยาวจากสั้นไปยาว
C) สลับให้ rank 1 อยู่ท้ายสุดเพื่อให้ LLM จำได้ดี
D) วางเอกสารที่สำคัญที่สุดไว้ตรงกลางเพราะ attention สูงสุดที่กลาง

**Q50** [W12] [Embedding storage] [difficulty: easy]
เก็บ embedding 1,000,000 chunk ที่ d = 384 แบบ float32 ใช้พื้นที่ประมาณเท่าใด?
A) 0.38 GB
B) 1.5 GB
C) 3.0 GB
D) 12.3 GB

**Q51** [W13] [ReAct loop] [difficulty: easy]
หนึ่งรอบของ ReAct loop เรียงลำดับอย่างไร?
A) Action -> Thought -> Observation
B) Observation -> Action -> Thought
C) Thought -> Observation -> Action
D) Thought -> Action -> Observation แล้ว observation ถูกนำเข้า prompt รอบถัดไป

**Q52** [W13] [ReAct Table 1] [difficulty: medium]
จากตาราง PaLM-540B ในสไลด์ ข้อใดถูกต้อง?
A) ReAct ชนะ CoT ทั้ง HotpotQA และ FEVER
B) CoT ชนะ ReAct ทั้ง HotpotQA และ FEVER
C) HotpotQA: CoT (29.4) สูงกว่า ReAct (27.4) แต่ FEVER: ReAct (60.9) สูงกว่า CoT (56.3)
D) HotpotQA: ReAct (27.4) สูงกว่า CoT (29.4) แต่ FEVER: CoT (60.9) สูงกว่า ReAct (56.3)

**Q53** [W13] [ReAct error analysis] [difficulty: hard]
จากการวิเคราะห์ error (Table 2) ข้อใดถูกต้อง?
A) ใน failure case, CoT มี hallucination 56% ส่วน ReAct 0% แต่ ReAct มี reasoning error 47% สูงกว่า CoT 16%
B) ReAct มี hallucination 56% ส่วน CoT 0%
C) ReAct ไม่มี error ใด ๆ เพราะ grounded ด้วย search
D) ทั้งสองมี reasoning error เท่ากัน

**Q54** [W13] [Logit masking] [difficulty: medium]
Grammar-constrained decoding ทำงานที่ระดับ logit อย่างไร?
A) ลดค่า logit ของ token ที่ผิดกฎลงเล็กน้อยเป็น soft penalty
B) ตั้ง logit ของ token ที่ไม่ valid เป็น -infinity ก่อน softmax ทำให้ P = 0 และ token ที่ valid แบ่ง probability mass 100%
C) ตัดผลลัพธ์ที่ผิดทิ้งหลังสร้างเสร็จแล้วสร้างใหม่
D) เพิ่ม logit ของ token ที่ valid เป็น +infinity หลัง softmax

**Q55** [W13] [Mask count] [difficulty: medium]
ที่ state หลัง `{` ใน JSON มี token ที่ valid 69 ตัวจาก GPT-2 vocab 50,257 token จำนวน token ที่ถูก mask คือเท่าใด?
A) 69
B) 50,257
C) 50,146
D) 50,188

**Q56** [W13] [CFG vs regex] [difficulty: medium]
ทำไม JSON ที่ซ้อนกันลึกไม่จำกัดต้องใช้ CFG แทน regex/DFA?
A) regex ช้ากว่า CFG เสมอ
B) regex ไม่รองรับ string ที่มีเครื่องหมายคำพูด
C) ต้องมี stack จำวงเล็บที่ยังไม่ปิด ซึ่ง DFA ไม่มี
D) CFG ทำงานด้วยเวลา O(1) ต่อ step

**Q57** [W13] [Agent memory] [difficulty: medium]
ข้อใดเกี่ยวกับ Buffer / Summary / Retrieval memory ถูกต้อง?
A) Retrieval memory ใช้ storage เพียง O(k)
B) Summary memory ทำให้ prompt bounded แต่ lossy ส่วน Buffer memory โตเป็น O(n) จนล้น context
C) Buffer memory เป็น lossy เพราะบีบข้อความเก่า
D) Summary memory ไม่สูญเสียข้อมูลเลย

**Q58** [W13] [Sequential topology latency] [difficulty: easy]
Sequential pipeline มี 3 agent: Research ใช้ 3 วินาที, Writer 10 วินาที, Editor 4 วินาที latency รวมเท่าใด?
A) 17 วินาที
B) 10 วินาที
C) 7 วินาที
D) 34 วินาที

**Q59** [W13] [Routing topology] [difficulty: medium]
จุดอ่อนหลักของ Routing Network (Dispatcher) คือข้อใด?
A) latency สูงมากเพราะต้องผ่านหลายรอบ
B) error cascade เพราะ output ต่อกันเป็นสายตรง
C) ต้องมีหลาย agent ถกเถียงกันก่อนสรุปจึงแพงที่สุด
D) router เป็น single point of failure (ถ้า route ผิด specialist ก็ตอบผิด) และ routing accuracy เป็น metric สำคัญ

**Q60** [W13] [Supervisor topology] [difficulty: medium]
เมื่อเทียบ 4 มิติ (latency / complexity / fault tolerance / best use) Supervisor-Led (Hierarchical) topology ตรงกับข้อใด?
A) latency ต่ำ, complexity ต่ำ, robustness ต่ำ, เหมาะงาน step-by-step ชัดเจน
B) latency ต่ำ, complexity ปานกลาง, เหมาะหลายโดเมนที่แยกกัน
C) latency และ token cost สูงมาก, implement ยาก แต่ self-correcting จึงเหมาะงานกำกวมหรือซับซ้อน
D) รันคู่ขนานแล้วให้ judge สรุป เหมาะ fact-checking

**Q61** [W14] [MinHash estimate] [difficulty: easy]
Signature MinHash ของสองเอกสารมี H = 64 ตำแหน่ง ตรงกัน 42 ตำแหน่ง ค่าประมาณ Jaccard เท่าใด?
A) 0.42
B) 0.656
C) 1.52
D) 0.50

**Q62** [W14] [MinHash vs SimHash] [difficulty: medium]
ข้อใดถูกต้อง?
A) MinHash ประมาณ Jaccard similarity (set overlap) ส่วน SimHash รักษา cosine similarity ผ่าน bit string สั้น ๆ และ Hamming distance
B) MinHash รักษา cosine similarity ส่วน SimHash ประมาณ Jaccard
C) ทั้งสองประมาณ Jaccard เหมือนกัน ต่างกันที่จำนวน hash
D) ทั้งสองจับได้เฉพาะ exact duplicate

**Q63** [W14] [MinHash vs LSH roles] [difficulty: medium]
ในการทำ near-dedup ระดับ 40 พันล้านเอกสาร MinHash กับ LSH banding แก้ปัญหาต่างกันอย่างไร?
A) LSH ลดต้นทุนต่อคู่ ส่วน MinHash ลดจำนวนคู่ที่ต้องเทียบ
B) ทั้งสองลดต้นทุนต่อคู่เหมือนกันจึงใช้อย่างใดอย่างหนึ่งก็พอ
C) LSH แทนที่การตรวจ exact Jaccard จึงไม่ต้อง verify candidate
D) MinHash ลดต้นทุนต่อคู่ (shingle set ใหญ่ -> H ตัวเลข) ส่วน LSH ลดจำนวนคู่ที่ต้องเทียบ ทั้งสองใช้ซ้อนกัน

**Q64** [W14] [Chinchilla] [difficulty: easy]
ตาม Chinchilla (~20 token ต่อ parameter) โมเดล 70B parameter ควรเทรนด้วยประมาณกี่ token?
A) 350B
B) 3.5T
C) 1.4T
D) 14T

**Q65** [W14] [pass@k calc] [difficulty: medium]
สร้างโค้ด n = 200 ตัวอย่างต่อโจทย์ มี c = 1 ตัวที่ผ่าน unit test ทั้งหมด pass@100 = 1 - C(n-c,k)/C(n,k) เท่าใด?
A) 0.005
B) 0.5
C) 1.0
D) 0.99

**Q66** [W14] [BLEU vs BERTScore] [difficulty: medium]
ในตัวอย่าง "The cat sat on the mat" เทียบกับ paraphrase "A feline rested upon the rug" และ word salad "mat the on is cat the" ข้อสรุปใดถูกต้อง?
A) BLEU-1 ให้ paraphrase สูงกว่า word salad
B) BERTScore ตรวจข้อเท็จจริงของประโยคได้
C) BERTScore ของ paraphrase (0.79) สูงกว่า BLEU-1 (0.17) แสดงว่า BERTScore เป็นระบบที่ดีกว่าในเชิงตัวเลขดิบ
D) BLEU-1 ให้ word salad (0.83) สูงกว่า paraphrase (0.17) แต่ BERTScore จัดอันดับ paraphrase สูงกว่า (เทียบ ranking ไม่ใช่เลขดิบข้ามเมตริก)

**Q67** [W14] [LLM-judge positional bias] [difficulty: medium]
วิธีลด positional bias ของ LLM-as-a-judge ที่สไลด์เสนอคือข้อใด?
A) Position swap: ตัดสินทั้ง A-B และ B-A และประกาศผู้ชนะเมื่อชนะทั้งสองลำดับ
B) หารคะแนนด้วยจำนวน token ของคำตอบ
C) ใช้โมเดลเดียวกับผู้ตอบเป็น judge
D) ตั้ง temperature = 0 อย่างเดียวก็เพียงพอ

**Q68** [W14] [SWE-bench] [difficulty: medium]
ใน SWE-bench issue ถือว่า resolved เมื่อใด?
A) patch ที่สร้างคอมไพล์ผ่าน
B) LLM judge ให้คะแนนว่า patch สมเหตุสมผล
C) tests ที่ควรพลิกจาก fail เป็น pass ผ่านแล้ว และ tests ที่เคยผ่านยังผ่านอยู่
D) patch เหมือน patch ของมนุษย์ทุกบรรทัด

**Q69** [W14] [Gopher rules] [difficulty: medium]
ข้อใดถูกต้องเกี่ยวกับ Gopher heuristic filters?
A) เอกสารผ่านถ้าผ่านเกณฑ์ข้อใดข้อหนึ่ง
B) เอกสารต้องผ่านทุกข้อ เช่น จำนวนคำ 50-100,000 และ alpha-word ratio >= 0.8
C) เอกสารต้องมีอย่างน้อย 5 ประโยค และไม่มีเครื่องหมาย `{`
D) ใช้ perplexity จาก KenLM แบ่งเป็น head/middle/tail

**Q70** [W14] [Perplexity filter trap] [difficulty: medium]
ข้อควรระวังของ perplexity filter (CCNet) คืออะไร?
A) ข้อความ spam/ซ้ำ ๆ อาจมี perplexity ต่ำจึงถูกเก็บ ส่วนเนื้อหาเทคนิคที่หายากอาจ perplexity สูงและถูกทิ้ง
B) perplexity สูงหมายถึงคุณภาพสูงเสมอ
C) CCNet ตัดเอกสารด้วย threshold perplexity คงที่ค่าเดียวทุกภาษา
D) perplexity filter ใช้ได้เฉพาะภาษาอังกฤษ

## Part B — Fill-in

**F1** [W8] [BART vs T5 tokenizer]
T5 ใช้ tokenizer ชนิด ______ (vocab ______ ตัว) ส่วน BART ใช้ BPE แบบ GPT-2 ที่มี output vocab ______ ตัว

**F2** [W9] [Triplet loss numeric]
กำหนด d(a,p) = 0.1, d(a,n) = 0.9 และ margin = 0.5 ค่า triplet loss = max(0, d(a,p) - d(a,n) + margin) = ______

**F3** [W10] [Full fine-tuning memory, 16N]
Full fine-tuning โมเดล 70B ด้วย Adam แบบ mixed precision ใช้ 16 byte ต่อ parameter คิดเป็น ______ GB (decimal) ซึ่งต้องใช้ GPU H100 80 GB อย่างน้อย ______ ตัว (ยังไม่รวม activations)

**F4** [W11] [Zero-shot CoT]
Zero-shot CoT ต่อท้าย prompt ด้วยวลี "______" และบน MultiArith (175B InstructGPT) accuracy เพิ่มจาก 17% เป็น ______%

**F5** [W12] [RRF]
ค่าคงที่ smoothing k ของ RRF ที่นิยมใช้คือ ______ ดังนั้นเอกสารอันดับ 1 ในลิสต์เดียวได้คะแนน 1/(k+1) = ______ (ปัดทศนิยม 4 ตำแหน่ง)

**F6** [W12] [BM25 saturation]
ส่วน TF ของ BM25 จะไม่เกินเพดาน ______ (เขียนเป็นสูตรของ k1) ซึ่งเท่ากับ ______ เมื่อ k1 = 1.5

**F7** [W13] [ReAct Table 1, FEVER]
บน FEVER (accuracy) ReAct ได้ ______ ส่วน CoT ได้ ______ (PaLM-540B ตามสไลด์)

**F8** [W13] [Reliability stack]
Reliability stack แบบ defense in depth มี 4 ชั้นคือ Constrain, ______, ______, Escalate

**F9** [W14] [FineWeb-Edu]
FineWeb-Edu เก็บเอกสารที่ educational classifier ให้คะแนน (0-5) >= ______ ได้ประมาณ ______ T token จาก FineWeb 15T token

**F10** [W14] [SHA-256]
SHA-256 ให้ digest ขนาด 256 bit = ______ byte = ______ ตัวอักษร hex


## Answer key

หมายเหตุ: "notes wNN" = notes/week_NN_notes.md; "s.N" = หน้าสไลด์; "demo" = text/demo_wNN.txt

| Q | ตอบ | คำอธิบายสั้น | แหล่งที่มา |
|---|---|---|---|
| Q1 | C | 5 วิธีของ BART = masking, deletion, text infilling, sentence permutation, document rotation; span corruption + sentinel เป็นของ T5 | notes w08 §2.5/§7 trap 6; s.5, 7, 10-12 |
| Q2 | B | Span ~ Poisson(lambda=3) แทนด้วย `[MASK]` ตัวเดียว (0-length = insertion) | notes w08 §2.5; s.11 |
| Q3 | A | Deletion ไม่เหลือ placeholder จึงต้องตรวจว่าอะไรหาย/ที่ไหน src < tgt; B คือ masking, C คือ permutation, D คือ T5 | notes w08 §2.5; s.10 |
| Q4 | D | Target = เฉพาะ span นำหน้าด้วย sentinel ที่ตรงกัน และจบด้วย sentinel ถัดไป `<extra_id_2>` แล้วตามด้วย `</s>` | notes w08 §2.6; s.17 |
| Q5 | B | 1 (e0) + 2 (quick brown) + 1 (e1) + 2 (lazy dog) + 1 (e2 สุดท้าย) = 7; ตัวเลือก 8 คือกรณีนับ `</s>`, 12 คือ BART | notes w08 §3.1; s.33 |
| Q6 | C | (0.2N)^2 = 0.04 N^2, 1/0.04 = 25 เท่า (เฉพาะ decoder self-attn); ทั้งโมเดลเร็วขึ้น ~2-3x (กับดัก B) | notes w08 §3.4; s.19 |
| Q7 | A | 32 buckets ต่อ head (16 fwd + 16 bwd), offset 0-7 exact, ไกลกว่าแชร์ (log), บวกเข้า logits ก่อน softmax | notes w08 §2.6; s.20 |
| Q8 | D | MLM เรียนจาก 15 token, CLM จาก 100 token | notes w08 §3.5; s.34, hint W8 |
| Q9 | B | 3 เหตุผล: serving simplicity/param efficiency, RLHF multi-turn, KV cache linear (PagedAttention, chunked prefill, prefix caching); D ผิดเพราะสไลด์ระบุว่าไม่มีการเทียบควบคุมที่ >100B | notes w08 §2.11; s.36-39 |
| Q10 | C | ไม่ใส่ prefix T5 fallback ไปทำ translation ใน demo | demo w08 Demo 5; notes w08 §4 |
| Q11 | D | c = [u;v;abs(u-v)] = 768 x 3 = 2,304 | notes w09 §3.1; s.17 |
| Q12 | A | L = 0 เมื่อ d(a,p) + alpha <= d(a,n) | notes w09 §2.13; s.27 |
| Q13 | C | 0.1 - 0.2 + 0.5 = 0.4 (ตัวหลอก 0.8 = ใส่เครื่องหมายผิด 0.1+0.2+0.5) | notes w09 §3.5; s.27 |
| Q14 | B | n(n-1)/2 = 1,000 x 999 / 2 = 499,500 | notes w09 §3.2; s.7-8 |
| Q15 | D | Cross-encoder O(n^2) forward passes (ระดับ ms); Bi-encoder n forward แล้ว n^2 dot products (ระดับ ns) | notes w09 §2.3/§2.8; s.7-8, 18; hint W9 |
| Q16 | A | GloVe avg 58.02, BERT [CLS] 16.50, BERT mean 46.35, SBERT-NLI 77.03 | notes w09 §2.1; s.4 |
| Q17 | B | SBERT default = mean pooling โดย mask ตัด [PAD] | notes w09 §2.6; s.15 |
| Q18 | C | batch N -> N-1 negatives = 255 | notes w09 §2.13; s.30 |
| Q19 | A | IndexFlatIP หลัง normalize = cosine (สูง=ดี); IndexFlatL2/HNSW default คืน squared L2 (ต่ำ=ดี) | notes w09 §2.15; s.38, 41 |
| Q20 | D | (1/1 + 1/2 + 1/4)/3 = 1.75/3 = 0.5833 | notes w09 §3.8; s.37 |
| Q21 | B | 16 x 10^9 / 1.0737 x 10^9 = 14.90 GiB (GiB เล็กกว่า GB) | notes w10 §3.1; s.5; hint W10 |
| Q22 | D | 70 x 10^9 x 0.5 = 35 x 10^9 B = 35 GB | notes w10 §3.1; s.3-4 |
| Q23 | A | 64/8 = 8x; 160/8 = 20 GB | notes w10 §3.2(ฉ); hint W7/W10 |
| Q24 | C | PagedAttention = fixed-size blocks non-contiguous + page table ลด fragmentation; A คือ FlashAttention, B คือ continuous batching | notes w10 §2.11; s.32 |
| Q25 | B | A ~ Gaussian, B = 0 ดังนั้น delta W = 0 ตอนเริ่ม | notes w10 §2.12; s.40 |
| Q26 | A | 2dr = 2 x 4096 x 16 = 131,072 (d^2 = 16,777,216) | notes w10 §3.5; s.42 |
| Q27 | D | NF4, Double Quantization (quantize scale factors), Paged Optimizers | notes w10 §2.13; s.46 |
| Q28 | C | MoE ลด compute ต่อ token แต่ต้องโหลดทั้ง 47B ใน VRAM ไม่ประหยัด memory | notes w10 §2.10; s.25-26; hint W10 |
| Q29 | B | FlashAttention = exact, tiling ใน SRAM + online softmax, extra memory O(N), ไม่เขียน N x N ลง HBM | notes w10 §2.8; s.19-22 |
| Q30 | A | 2.1 / 0.0252 = 83.33 -> round = 83 (dequantize 83 x 0.0252 = 2.092) | notes w10 §3.6; s.44; hint W11 |
| Q31 | C | CO-STAR = Context, Objective, Style, Tone, Audience, Response format | notes w11 §2.2; s.5 |
| Q32 | D | Temperature > 0, sample 5-10 CoT paths, majority vote | notes w11 §2.7; s.11 |
| Q33 | B | ToT = tree/search (branch แยกกัน); GoT = arbitrary graph + aggregation + refinement (self-loop) | notes w11 §2.11; s.16-17 |
| Q34 | A | PoT/PAL: LLM เขียนโค้ด, external interpreter คำนวณ (decouple reasoning จาก computation) | notes w11 §2.8; s.12 |
| Q35 | C | PPO 4 models: Actor + Critic ถูกอัปเดต; Reward + Reference frozen | notes w11 §2.18; s.37 |
| Q36 | B | policy = ref -> logits = 0 -> sigma = 0.5 -> loss = ln 2 = 0.6931 | notes w11 §3.0; s.41 |
| Q37 | D | logits = 2 - (-2) = 4; x beta 0.1 = 0.4; sigma(0.4) = 0.5987; -ln(0.5987) = 0.513 (0.598 คือค่า sigma เอง, 0.127 คือกรณี scaled = 2.0) | notes w11 §3.1; s.41 |
| Q38 | A | mean = 0.25; std = sqrt((0.75^2 + 3 x 0.25^2)/4) = sqrt(0.1875) = 0.4330; A = 0.75/0.4330 = 1.732 | notes w11 §3.4; s.51 |
| Q39 | C | GRPO ตัด Critic ใช้ group mean เป็น baseline | notes w11 §2.22; s.50 |
| Q40 | D | Demo: ทั้งสองโมเดล 494.0M params | demo w11 Setup; notes w11 §4 |
| Q41 | A | b=0: f(k1+1)/(f+k1) -> k1+1 เมื่อ f -> infinity (ไม่ขึ้นกับ b) | notes w12 §2.2; s.16; hint W12 |
| Q42 | C | 3 x 2.5 / (3 + 1.5) = 7.5 / 4.5 = 1.667 | notes w12 §3.1; s.16 |
| Q43 | B | doc สั้น (L<1) ได้ TF part สูงกว่า; b ไม่เปลี่ยนเพดาน k1+1 แค่ความเร็วเข้าหา | notes w12 §2.2/§3.2; s.17; hint W12 |
| Q44 | D | 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266 ≈ 0.0323 (A = ลิสต์เดียว) | notes w12 §3.5; s.22-23 |
| Q45 | A | BM25 (0, infinity) กับ cosine [-1,1] คนละสเกล RRF ใช้อันดับอย่างเดียว | notes w12 §2.2; s.21-22 |
| Q46 | B | Cross-encoder joint attention precompute ไม่ได้ O((abs(q)+abs(d))^2) ต่อคู่ จึง rerank top K = 20-100 | notes w12 §2.2; s.24-25 |
| Q47 | C | Zero-shot: prompt template "A photo of a {object}." -> text encoder -> เลือกคลาส similarity สูงสุด | notes w12 §2.3; s.31, 34 |
| Q48 | D | Community summaries สร้างตอน indexing (offline); Global search = Map-Reduce ข้าม summaries; Local = entity-focused ~1 LLM call | notes w12 §2.3; s.42-44 |
| Q49 | A | demo: [1,2,3,4,5] -> [1,3,4,5,2] (rank 1 ต้น, rank 2 ท้าย) แก้ Lost in the Middle (U-shaped) | demo w12 Part 2; notes w12 §4; s.52-53 |
| Q50 | B | 1,000,000 x 384 x 4 = 1.536 x 10^9 B = 1.5 GB (D = d 3072 = 12.3 GB) | notes w12 §3.7; s.10 |
| Q51 | D | Thought -> Action -> Observation; observation นำทางรอบถัดไป | notes w13 §2.2; s.8-9 |
| Q52 | C | HotpotQA: CoT 29.4 > ReAct 27.4; FEVER: ReAct 60.9 > CoT 56.3 | notes w13 §2.2; s.14; hint W13 |
| Q53 | A | Failure: hallucination CoT 56% vs ReAct 0%; reasoning error ReAct 47% vs CoT 16% | notes w13 §2.2; s.15 |
| Q54 | B | logit' = -infinity ถ้าไม่ valid ก่อน softmax -> P = 0 | notes w13 §2.4; s.21 |
| Q55 | D | 50,257 - 69 = 50,188 (99.86%) | notes w13 §3.2; s.22 |
| Q56 | C | JSON ซ้อนลึกไม่จำกัดต้องมี stack = CFG (pushdown), ไม่ใช่ regular language | notes w13 §2.4; s.23, 25 |
| Q57 | B | Buffer O(n) ล้น; Summary bounded แต่ lossy; Retrieval ลด prompt แต่ store ยังเก็บ n turns | notes w13 §2.5; s.31 |
| Q58 | A | 3 + 10 + 4 = 17 s (latency = sum T_i ไม่ใช่ max) | notes w13 §3.4; s.34 |
| Q59 | D | Routing: router = SPOF, key metric = routing accuracy | notes w13 §2.6; s.36-37 |
| Q60 | C | Supervisor: latency/token cost สูงมาก, implement ยาก, self-correcting, เหมาะงานกำกวม/ซับซ้อน (A=Sequential, B=Routing, D=Debate) | notes w13 §2.6; s.38-41; hint W14 (4 มิติ) |
| Q61 | B | 42/64 = 0.65625 ≈ 0.656 | notes w14 §3.2; s.23-24, 29 |
| Q62 | A | MinHash -> Jaccard (H ตัวเลข); SimHash -> cosine (bit string, Hamming) | notes w14 §2.16; s.25 |
| Q63 | D | MinHash ลดต้นทุนต่อคู่ (17,689 -> 64), LSH ลดจำนวนคู่ (8 x 10^20); ซ้อนกัน; ต้อง verify candidate | notes w14 §2.18; s.26-28 |
| Q64 | C | 70 x 10^9 x 20 = 1.4 x 10^12 = 1.4T | notes w14 §3.4; s.4 |
| Q65 | B | c = 1 -> pass@k = k/n = 100/200 = 0.5; ตรวจ 1 - C(199,100)/C(200,100) = 1 - 100/200 | notes w14 §3.1; s.44 |
| Q66 | D | BLEU-1: salad 0.83 > paraphrase 0.17; BERTScore F1: paraphrase 0.79 > salad 0.67; เทียบ ranking ไม่ใช่เลขดิบ | notes w14 §2.22-2.23; s.33-36 |
| Q67 | A | Position swap (A-B และ B-A ชนะทั้งสองลำดับ); B คือกับดัก length-control | notes w14 §2.27-2.28; s.40-41 |
| Q68 | C | Resolved = fail->pass tests ผ่านและ pass->pass tests ยังผ่าน | notes w14 §2.33; s.50 |
| Q69 | B | Gopher ต้องผ่านทุกข้อ (50-100,000 คำ, alpha >= 0.8 ...); C คือ C4; D คือ CCNet | notes w14 §2.7; s.11 |
| Q70 | A | Perplexity trap: spam ซ้ำ PPL ต่ำ ถูกเก็บ; เนื้อหาเทคนิคหายาก PPL สูง ถูกทิ้ง; CCNet แบ่ง 3 buckets ไม่มี cutoff คงที่ | notes w14 §2.10; s.14-15 |

### Fill-in

| F | คำตอบที่คาดหวัง | คำตอบที่ยอมรับ | คำอธิบาย / แหล่งที่มา |
|---|---|---|---|
| F1 | SentencePiece; 32,000; 50265 | 32000 / 32k; 50,265 / ~50k | T5 = SentencePiece vocab 32,000; BART = BPE (GPT-2) vocab 50265; notes w08 §2.5-2.7; s.21 |
| F2 | 0 | 0.0 (ค่าก่อน max = -0.3 ติดลบจึงเป็น 0) | 0.1 - 0.9 + 0.5 = -0.3 -> max(0, -0.3) = 0 (easy negative); notes w09 §3.5; s.27, 32 |
| F3 | 1,120 GB; 14 | 1.12 TB; 14 หรือ 14+ | 70 x 10^9 x 16 = 1,120 x 10^9 B = 1,120 GB; 1,120/80 = 14; notes w10 §3.4; s.39 |
| F4 | Let's think step by step; 78 | "think step by step"; 78% | Kojima et al. 2022: MultiArith 17% -> 78% บน 175B InstructGPT; notes w11 §2.6; s.10 |
| F5 | 60; 0.0164 | 60; 1/61 = 0.01639 (0.0164 / 0.016) | k = 60 (Cormack 2009); 1/61 = 0.016393; notes w12 §3.5; s.22-23 |
| F6 | k1 + 1; 2.5 (2.50) | k1+1; 2.5 | f(k1+1)/(f+k1) เพดาน k1+1 ไม่ขึ้นกับ b; 1.5 + 1 = 2.5; notes w12 §2.2; s.16 |
| F7 | 60.9; 56.3 | 60.9%; 56.3% | Table 1 FEVER Acc: ReAct 60.9, CoT 56.3; notes w13 §2.2; s.14; hint W13 |
| F8 | Validate; Recover | Validate, Recover (ลำดับนี้) | L1 Constrain -> L2 Validate -> L3 Recover -> L4 Escalate; notes w13 §2.7; s.46 |
| F9 | 3; 1.3 | score >= 3 หรือ 3 ขึ้นไป; 1.3T (~9%) | FineWeb-Edu threshold >= 3 -> 1.3T จาก 15T tokens (~9%); notes w14 §2.11; s.17 |
| F10 | 32; 64 | 32 bytes; 64 hex chars | 256 / 8 = 32 byte; 1 byte = 2 hex -> 64; notes w14 §2.14; s.21 |

### Answer distribution (MCQ)
A = 18 (Q3, 7, 12, 16, 19, 23, 26, 30, 34, 38, 41, 45, 49, 53, 58, 62, 67, 70); B = 18 (Q2, 5, 9, 14, 17, 21, 25, 29, 33, 36, 43, 46, 50, 54, 57, 61, 65, 69); C = 17 (Q1, 6, 10, 13, 18, 24, 28, 31, 35, 39, 42, 47, 52, 56, 60, 64, 68); D = 17 (Q4, 8, 11, 15, 20, 22, 27, 32, 37, 40, 44, 48, 51, 55, 59, 63, 66)

### Source conflicts noted
- Bi-encoder เวลา 10,000 ประโยค: hint W9 บอก ~5 วินาที แต่สไลด์ W9 s.8 ระบุ ~50 วินาที (ข้อสอบนี้ไม่ใช้ตัวเลขเวลา; Q14-15 ถามจำนวน pass และ big-O เท่านั้น)
- Hint "Week 11 quantization (0.0252, 83)" ตรงกับเนื้อหาสไลด์ W10 (s.44) ไม่ใช่ W11 จึงจัดเป็น Q30 ของ W10
- Hint ชื่อ topology (sequential/routing/parallel/hierarchical) ต่างจากสไลด์ W13 (Sequential/Routing/Supervisor-Led/Debate) ข้อสอบนี้ใช้ชื่อตามสไลด์ W13
- Hint W13 ระบุ CLIP/bookend แต่เนื้อหาอยู่ในสไลด์ W12 จึงจัดไว้ W12 (Q47, Q49)

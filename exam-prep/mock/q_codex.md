# Mock exam — Codex set

## Part A — MCQ

เลือกคำตอบที่ถูกที่สุดเพียงข้อเดียวในแต่ละข้อ กำหนดให้ GB = \(10^9\) bytes และ GiB = \(2^{30}\) bytes เว้นแต่โจทย์ระบุเป็นอย่างอื่น

**Q1** [W8] [BART noising] [difficulty: easy]  
วิธีใดลบ token ออกจาก input โดยไม่เหลือ placeholder ไว้?

A) Token masking  
B) Token deletion  
C) Text infilling  
D) Sentence permutation

**Q2** [W8] [BART noising] [difficulty: medium]  
ข้อใดจับคู่การเปลี่ยนแปลงของเอกสารกับวิธี noising ได้ถูกต้อง?

A) แทน span ที่ติดกันด้วย mask หนึ่งตัว → token deletion  
B) เลือก pivot แล้วนำข้อความตั้งแต่ pivot ไปไว้ต้นเอกสาร → sentence permutation  
C) สลับลำดับประโยค → document rotation  
D) สลับลำดับประโยค → sentence permutation

**Q3** [W8] [BART text infilling] [difficulty: medium]  
ใน BART text infilling การแทน span ที่มีหลาย token ทำอย่างไร?

A) ใช้ mask หนึ่งตัวแทนทั้ง span  
B) ใช้ sentinel คนละตัวแทนทุก token  
C) ลบ span โดยไม่มี placeholder เสมอ  
D) ใช้ mask เท่าจำนวน token ใน span เสมอ

**Q4** [W8] [T5 sentinel] [difficulty: hard]  
ข้อความต้นฉบับมี 12 tokens และ T5 ตัดออก 2 spans รวม 5 tokens โดยแทนแต่ละ span ใน encoder input ด้วย sentinel หนึ่งตัว ใน target ให้มี sentinel นำหน้าแต่ละ span, sentinel ปิดท้ายอีกหนึ่งตัว และ `</s>` อีกหนึ่งตัว จงหา `(input length, target length)` โดยนับ sentinel และ `</s>` ตามที่ระบุ และไม่มี special token อื่น

A) (7, 5)  
B) (9, 7)  
C) (9, 9)  
D) (10, 10)

**Q5** [W8] [BART vs T5 compute] [difficulty: medium]  
สมมติ decoder target ของ BART ยาว \(N\) และของ T5 ยาว \(0.2N\) หากต้นทุน decoder self-attention แปรตามกำลังสองของความยาว target ต้นทุนส่วนนี้ของ BART เป็นกี่เท่าของ T5?

A) 5 เท่า  
B) 25 เท่า  
C) 2.5 เท่า  
D) 50 เท่า

**Q6** [W8] [T5 text-to-text] [difficulty: easy]  
แนวคิดหลักของ T5 ในการทำหลาย NLP tasks คือข้อใด?

A) ใช้ classifier head ใหม่เสมอสำหรับทุก task  
B) เปลี่ยน encoder เป็นภาพสำหรับ classification  
C) ลบ decoder ทิ้งเมื่อทำ translation  
D) กำหนด task ด้วยข้อความ เช่น prefix และสร้างคำตอบเป็นข้อความ

**Q7** [W8] [relative position bias] [difficulty: medium]  
T5 relative position bias ทำงานอย่างไร?

A) นำ learned bias ตาม bucket ของระยะสัมพัทธ์ไปบวกกับ attention logits  
B) คูณ absolute position เข้ากับ output หลัง softmax เสมอ  
C) ใช้ bucket แยกหนึ่ง bucket สำหรับทุกระยะที่เป็นไปได้  
D) ลบข้อมูลตำแหน่งออกจาก attention ทั้งหมด

**Q8** [W8] [cross-attention shape] [difficulty: medium]  
กำหนด batch = 2, จำนวน heads = 12, source length = 50 และ target length = 20 รูปร่างของ cross-attention weights เป็นข้อใด?

A) `[2, 12, 50, 50]`  
B) `[2, 12, 50, 20]`  
C) `[2, 12, 20, 50]`  
D) `[2, 20, 12, 20]`

**Q9** [W8] [MLM vs CLM] [difficulty: easy]  
หากมี 200 tokens และสมมติว่า MLM เลือก 15% ของ tokens เป็นเป้าหมาย ส่วน CLM ใช้ 100% ตามแบบจำลองในสไลด์ จะมี target tokens เท่าใดตามลำดับ?

A) 15 และ 200  
B) 30 และ 200  
C) 30 และ 100  
D) 170 และ 200

**Q10** [W8] [T5 demo] [difficulty: easy]  
ในตัวอย่าง sentiment ของ T5-small ใน demo เมื่อไม่ใส่ task prefix โมเดลทำอะไรกับข้อความตัวอย่าง?

A) หยุดโดยไม่สร้างข้อความ  
B) คืน label `positive` เสมอ  
C) สร้าง `[MASK]` แทนทุกคำ  
D) สร้างข้อความลักษณะ translation

**Q11** [W9] [SBERT classification head] [difficulty: easy]  
ถ้า sentence embeddings \(u,v\) มีขนาด 768 มิติ และ classification head รับ \([u;v;|u-v|]\) เวกเตอร์ที่ต่อกันมีขนาดเท่าใด?

A) 2,304  
B) 768  
C) 1,536  
D) 3,072

**Q12** [W9] [cross-encoder cost] [difficulty: medium]  
ต้องการเทียบประโยค 1,000 ประโยคทุกคู่แบบไม่ซ้ำและไม่เทียบประโยคกับตัวเอง โดยใช้ cross-encoder หนึ่ง forward pass ต่อคู่ ต้องใช้กี่ forward passes?

A) 1,000  
B) 500,000  
C) 499,500  
D) 999,000

**Q13** [W9] [bi-encoder cost] [difficulty: easy]  
Bi-encoder ใช้เวลา encode ประโยคละ 5 ms สำหรับ 1,000 ประโยค หากประมวลผลทีละประโยค เวลารวมเฉพาะการ encode เท่าใด?

A) 0.5 วินาที  
B) 50 วินาที  
C) 500 วินาที  
D) 5 วินาที

**Q14** [W9] [triplet loss] [difficulty: medium]  
กำหนด \(L=\max(0,d(a,p)-d(a,n)+\alpha)\), \(d(a,p)=0.20\), \(d(a,n)=0.55\), \(\alpha=0.50\) ค่า loss เท่าใด?

A) 0  
B) 0.15  
C) 0.35  
D) 1.25

**Q15** [W9] [in-batch negatives] [difficulty: easy]  
ใน MNRL batch มี positive pairs 64 คู่ และใช้ positive ของ query อื่นเป็น in-batch negatives แต่ละ query มี negatives กี่ตัว?

A) 63  
B) 64  
C) 65  
D) 128

**Q16** [W9] [masked mean pooling] [difficulty: medium]  
ค่าใน embedding มิติหนึ่งของสาม tokens เป็น 2, 4 และ 100 ตามลำดับ โดย attention mask เป็น `[1, 1, 0]` ค่า masked mean ของมิตินี้เท่าใด?

A) 2  
B) 4  
C) 3  
D) \(106/3\)

**Q17** [W9] [vector similarity] [difficulty: medium]  
เมื่อใด inner product ของเวกเตอร์สองตัวจึงเท่ากับ cosine similarity ของทั้งคู่?

A) เมื่อเวกเตอร์ทั้งคู่มีมิติเท่ากันเท่านั้น  
B) เมื่อเวกเตอร์ทั้งคู่ถูก L2-normalize ให้มี norm เท่ากับ 1  
C) เมื่อใช้เวกเตอร์จาก BERT เสมอ  
D) เมื่อ inner product มีค่าเป็นบวก

**Q18** [W9] [bi-encoder vs cross-encoder] [difficulty: medium]  
ข้อใดอธิบาย retrieve-then-rerank ได้เหมาะสมที่สุด?

A) Cross-encoder encode เอกสารล่วงหน้าทั้ง corpus แล้ว bi-encoder ตรวจทุกคู่ร่วมกัน  
B) ใช้ cross-encoder เฉพาะสร้าง document vectors ถาวร  
C) Bi-encoder มี cross-attention ระหว่าง query กับ document ทุกชั้น  
D) Bi-encoder ค้น candidates แล้ว cross-encoder ให้คะแนนคู่ query-document อีกครั้ง

**Q19** [W9] [negative mining] [difficulty: medium]  
ข้อใดเป็นเงื่อนไขของ semi-hard negative สำหรับ triplet loss ที่มี margin \(\alpha\)?

A) \(d(a,p)<d(a,n)<d(a,p)+\alpha\)  
B) \(d(a,n)<d(a,p)\)  
C) \(d(a,n)\ge d(a,p)+\alpha\)  
D) \(d(a,n)=0\) เสมอ

**Q20** [W9] [MRR] [difficulty: medium]  
สำหรับ 3 queries ตำแหน่งของ relevant result ตัวแรกคือ 1, 2 และ 4 ตามลำดับ ค่า MRR ใกล้เคียงเท่าใด?

A) 0.2500  
B) 0.4375  
C) 0.5833  
D) 0.7500

**Q21** [W10] [GB vs GiB] [difficulty: easy]  
หน่วยความจำขนาด 16 GB เท่ากับประมาณกี่ GiB?

A) 16.00  
B) 17.18  
C) 15.63  
D) 14.90

**Q22** [W10] [weight memory] [difficulty: easy]  
โมเดลมี 7 พันล้าน parameters เก็บทุก parameter ด้วย FP16 อย่างละ 2 bytes น้ำหนักโมเดลเพียงอย่างเดียวใช้หน่วยความจำเท่าใด?

A) 3.5 GB  
B) 14 GB  
C) 7 GB  
D) 28 GB

**Q23** [W10] [KV cache] [difficulty: hard]  
ให้ \(M_{KV}=2\times b\times L\times H_{KV}\times d_h\times S\times B\) bytes โดย \(b=2\), \(L=32\), \(H_{KV}=8\), \(d_h=128\), \(S=4096\), \(B=1\) KV cache มีขนาดเท่าใด?

A) 128 MiB  
B) 256 MiB  
C) 512 MiB  
D) 1,024 MiB

**Q24** [W10] [GQA] [difficulty: medium]  
หากเปลี่ยนจาก MHA ที่มี 32 KV heads เป็น GQA ที่มี 8 KV heads โดยค่าที่เหลือในสูตร KV cache คงเดิม ขนาด KV cache ลดลงกี่เปอร์เซ็นต์?

A) 75%  
B) 25%  
C) 80%  
D) 400%

**Q25** [W10] [training memory] [difficulty: medium]  
ใช้การประมาณ full fine-tuning ด้วย Adam ที่ 16 bytes ต่อ parameter โดยยังไม่รวม activations โมเดล 7 พันล้าน parameters ต้องใช้ model-state memory เท่าใด?

A) 14 GB  
B) 56 GB  
C) 96 GB  
D) 112 GB

**Q26** [W10] [LoRA] [difficulty: medium]  
LoRA สำหรับเมทริกซ์จัตุรัสขนาด \(d\times d\) ใช้ trainable parameters \(2dr\) หาก \(d=4096\), \(r=16\) จะมี trainable parameters เท่าใด?

A) 65,536  
B) 131,072  
C) 262,144  
D) 16,777,216

**Q27** [W10] [MoE] [difficulty: medium]  
ข้อใดอธิบายข้อแลกเปลี่ยนของ MoE ตามสไลด์ได้ถูกต้อง?

A) ทุก token ต้องผ่านทุก expert จึงใช้ compute เท่า dense เสมอ  
B) ใช้เฉพาะ active experts จึงต้องโหลดเฉพาะ expert เหล่านั้นในตัวอย่างการใช้งานของสไลด์  
C) เลือกใช้ experts บางส่วนต่อ token เพื่อลด compute แต่ยังต้องมี memory สำหรับ experts ทั้งหมด  
D) ลดทั้งจำนวน parameters ที่เก็บและจำนวน experts ทั้งหมดเหลือศูนย์

**Q28** [W10] [FlashAttention] [difficulty: medium]  
FlashAttention ลดการใช้ memory และการถ่ายโอนข้อมูลอย่างไร?

A) ใช้ tiling และ online softmax เพื่อคำนวณ exact attention โดยไม่เขียน attention matrix เต็มลง HBM  
B) ตัด attention ระหว่าง tokens ที่อยู่ไกลออกทั้งหมด  
C) แทน softmax ด้วยการเลือก token แบบสุ่ม  
D) เก็บ attention matrix เต็มไว้ใน HBM ทุกขั้น

**Q29** [W10] [serving] [difficulty: medium]  
PagedAttention มีหน้าที่หลักข้อใด?

A) เลือก token โดยใช้ temperature สูง  
B) เปลี่ยน full attention เป็น sliding-window attention  
C) ฝึก reward model แทน PPO  
D) จัดสรร KV cache เป็น blocks และจัดการ mapping เพื่อลด fragmentation

**Q30** [W10] [quantization absmax] [difficulty: medium]  
กำหนด symmetric INT8 scale \(s=\max|w|/127\), \(\max|w|=3.2\) และใช้ \(s\approx0.0252\) ค่า \(w=2.1\) จะถูก quantize เป็นจำนวนเต็มใด โดยปัดเป็นจำนวนเต็มใกล้สุด?

A) 8  
B) 83  
C) 127  
D) 833

**Q31** [W11] [in-context learning] [difficulty: easy]  
Few-shot prompting เปลี่ยนแปลงอะไรระหว่าง inference?

A) อัปเดต weights จากตัวอย่างทุกครั้ง  
B) เพิ่ม optimizer states ให้โมเดล  
C) เพิ่มตัวอย่างใน prompt โดยไม่อัปเดต weights  
D) ลบ pre-training ของโมเดล

**Q32** [W11] [self-consistency] [difficulty: easy]  
สุ่ม reasoning paths อิสระ 5 ครั้ง ได้คำตอบสุดท้าย 12, 12, 9, 12, 15 หากใช้ majority vote คำตอบที่เลือกและสัดส่วนเสียงเป็นเท่าใด?

A) 12 และ 60%  
B) 12 และ 40%  
C) 9 และ 20%  
D) 15 และ 60%

**Q33** [W11] [SFT] [difficulty: medium]  
ข้อใดอธิบาย supervised fine-tuning ได้ถูกต้อง?

A) ฝึกจาก ranking ของคำตอบเท่านั้นโดยไม่มี demonstrations  
B) ใช้ PPO แทน next-token prediction เสมอ  
C) ไม่ต้องใช้ตัวอย่างคำตอบที่ต้องการ  
D) ฝึก next-token prediction บน demonstrations ของพฤติกรรมที่ต้องการ

**Q34** [W11] [RLHF] [difficulty: medium]  
ลำดับหลักของ RLHF ตามสไลด์คือข้อใด?

A) PPO → Reward Model → SFT  
B) SFT → Reward Model → PPO  
C) Reward Model → pre-training → tokenizer  
D) DPO → retrieval → SFT

**Q35** [W11] [DPO] [difficulty: hard]  
กำหนด DPO logit เป็น  
\(\beta[(\log\pi_w-\log\pi_{\text{ref},w})-(\log\pi_l-\log\pi_{\text{ref},l})]\)  
โดย \(\log\pi_w=-12\), \(\log\pi_{\text{ref},w}=-14\), \(\log\pi_l=-20\), \(\log\pi_{\text{ref},l}=-18\) และ \(\beta=0.1\) ค่า logit เท่าใด?

A) -0.4  
B) 0.2  
C) 0.4  
D) 4.0

**Q36** [W11] [GRPO] [difficulty: medium]  
กลุ่ม rewards คือ `[1, 1, 0, 0]` ใช้ population standard deviation และ \(A_i=(r_i-\mu)/\sigma\) ค่า advantage ของคำตอบที่ได้ reward 1 เท่าใด?

A) 1  
B) 0.5  
C) -1  
D) 2

**Q37** [W11] [reasoning cost] [difficulty: medium]  
ภายใต้แบบจำลองในสไลด์ หากความยาว reasoning trace เพิ่มเป็น 4 เท่า attention compute รวมซึ่งแปรตามกำลังสอง และ KV cache ซึ่งแปรเชิงเส้น จะเพิ่มเป็นกี่เท่าตามลำดับ?

A) 4 และ 16  
B) 8 และ 4  
C) 16 และ 16  
D) 16 และ 4

**Q38** [W11] [prompt injection] [difficulty: easy]  
คำสั่งที่ผู้โจมตีซ่อนไว้ในเว็บเพจซึ่ง agent ไปดึงมาอ่าน เป็นตัวอย่างของข้อใด?

A) Direct prompt injection โดยผู้ใช้ใน turn ปัจจุบัน  
B) Indirect prompt injection ผ่านข้อมูลภายนอก  
C) การทำ L2 normalization  
D) การฝึก SFT

**Q39** [W11] [DPO vs RLHF] [difficulty: medium]  
ข้อใดเป็นลักษณะของ DPO ตามสไลด์?

A) ต้องฝึก separate reward model และใช้ PPO เสมอ  
B) ต้องมี critic model เพื่อคำนวณทุก preference pair  
C) ใช้ preference data ฝึก policy เทียบกับ reference โดยไม่ต้องฝึก separate reward model หรือใช้ PPO  
D) ใช้เพียงข้อมูลที่ไม่มี preference และไม่มี reference model

**Q40** [W11] [Program-of-Thought] [difficulty: easy]  
Program-of-Thought แบ่งหน้าที่ระหว่าง LLM กับเครื่องมืออย่างไร?

A) LLM เขียนโปรแกรม แล้ว interpreter รันเพื่อคำนวณ  
B) Interpreter เขียน prompt แล้ว LLM แปลงเป็น weights  
C) LLM ส่งคำตอบแบบสุ่มให้ interpreter โหวต  
D) Interpreter ฝึกโมเดลใหม่สำหรับทุกคำถาม

**Q41** [W12] [BM25 saturation] [difficulty: easy]  
เมื่อ \(k_1=1.5\) ค่าเพดานที่ TF factor ของ BM25 เข้าใกล้เมื่อ term frequency เพิ่มมากขึ้นคือเท่าใด?

A) 1.0  
B) 1.5  
C) 2.0  
D) 2.5

**Q42** [W12] [BM25 calculation] [difficulty: medium]  
เมื่อ \(b=0\), \(k_1=1.5\), \(f=2\) และใช้เฉพาะ TF factor  
\(\frac{f(k_1+1)}{f+k_1}\) ค่าที่ได้ใกล้เคียงข้อใด?

A) 0.571  
B) 1.429  
C) 2.000  
D) 2.500

**Q43** [W12] [BM25 length normalization] [difficulty: hard]  
ให้ \(k_1=1.5\), \(b=0.75\), \(f=2\), \(|d|/\text{avgdl}=0.5\) จงหา TF factor  
\(\frac{f(k_1+1)}{f+k_1(1-b+b|d|/\text{avgdl})}\)

A) 0.851  
B) 1.429  
C) ประมาณ 1.702  
D) 2.500

**Q44** [W12] [RRF] [difficulty: medium]  
ใช้ \(RRF(d)=\sum_r 1/(k+\operatorname{rank}_r(d))\), \(k=60\) และอันดับเริ่มที่ 1 ถ้าเอกสารอยู่ rank 1 และ rank 3 ในสองระบบ คะแนนประมาณเท่าใด?

A) 0.03227  
B) 0.01639  
C) 0.03125  
D) 0.32266

**Q45** [W12] [CLIP contrastive learning] [difficulty: easy]  
ใน batch ที่มี 16 คู่ image-text หากพิจารณา image หนึ่งภาพและใช้ captions ของภาพอื่นเป็น negatives จะมี negative captions กี่ตัว?

A) 1  
B) 8  
C) 16  
D) 15

**Q46** [W12] [chunking] [difficulty: easy]  
Fixed-size chunking ใช้ `chunk_size = 512` และ `chunk_overlap = 64` ระยะ stride ระหว่างจุดเริ่มต้นของ chunks เท่าใด?

A) 64  
B) 448  
C) 512  
D) 576

**Q47** [W12] [hybrid retrieval] [difficulty: medium]  
ลำดับใดตรงกับ pipeline ในสไลด์?

A) LLM → cross-encoder → BM25 และ dense → RRF  
B) RRF → LLM → BM25 → cross-encoder  
C) BM25 และ dense retrieval → RRF → cross-encoder rerank → LLM  
D) Cross-encoder → LLM → RRF → dense retrieval

**Q48** [W12] [GraphRAG] [difficulty: medium]  
GraphRAG global search ใช้ประโยชน์จาก community summaries อย่างไร?

A) ใช้ map-reduce เพื่อสังเคราะห์ภาพรวมจาก summaries หลายส่วน  
B) ลบความสัมพันธ์ระหว่าง entities ก่อนค้นหาเสมอ  
C) แทน community summaries ด้วย exact keyword match เพียงคำเดียว  
D) สร้างคำตอบโดยไม่อ่านข้อมูลที่ index ไว้

**Q49** [W12] [lost in the middle] [difficulty: medium]  
หากใช้ bookend ตามตัวอย่างในสไลด์และ demo โดยวาง rank 1 ไว้ต้น prompt และ rank 2 ไว้ท้าย prompt ลำดับ `[1, 2, 3, 4, 5]` จะเป็นข้อใด?

A) `[2, 3, 4, 5, 1]`  
B) `[1, 2, 5, 4, 3]`  
C) `[5, 4, 3, 2, 1]`  
D) `[1, 3, 4, 5, 2]`

**Q50** [W12] [embedding storage] [difficulty: medium]  
เก็บ embeddings จำนวน 1,000,000 vectors แต่ละ vector มี 384 มิติ และแต่ละมิติใช้ float32 ขนาด 4 bytes ต้องใช้พื้นที่ข้อมูลเวกเตอร์อย่างเดียวเท่าใด?

A) 0.1536 GB  
B) 1.536 GB  
C) 3.840 GB  
D) 15.36 GB

**Q51** [W13] [ReAct] [difficulty: easy]  
วงจรหลักของ ReAct คือข้อใด?

A) Train → quantize → deploy  
B) Retrieve → freeze → delete  
C) Thought → Action → Observation  
D) Tokenize → pool → classify

**Q52** [W13] [tool calling] [difficulty: medium]  
หลัง LLM สร้าง tool call แล้ว องค์ประกอบใดทำหน้าที่ส่งต่อการเรียกไปยัง function ที่ระบุ หลังผ่านการตรวจรูปแบบ?

A) Dispatcher  
B) Tokenizer  
C) Optimizer  
D) Embedding pooler

**Q53** [W13] [ReAct benchmark] [difficulty: medium]  
ในตารางผลการทดลอง FEVER ที่สไลด์ยกมา ReAct ได้ 60.9 และ CoT ได้ 56.3 คะแนน ReAct สูงกว่ากี่ percentage points?

A) 2.0  
B) 3.6  
C) 4.0  
D) 4.6

**Q54** [W13] [constrained decoding] [difficulty: medium]  
ในตัวอย่าง vocab มี 50,257 tokens และ parser state ปัจจุบันอนุญาต 69 tokens ต้อง mask tokens ที่ไม่อนุญาตกี่ตัว?

A) 49,188  
B) 50,188  
C) 50,257  
D) 50,326

**Q55** [W13] [constrained decoding demo] [difficulty: medium]  
ใน demo การบังคับ regex รูปแบบวันที่ `YYYY-MM-DD` อาจสร้าง `3194-81-59` ได้ ข้อสรุปใดเหมาะสมที่สุด?

A) Regex รับประกันว่าวันที่มีอยู่จริง  
B) ต้องปิด logit masking จึงจะตรวจรูปแบบได้  
C) รูปแบบผ่าน regex ได้ แต่ยังต้องตรวจความหมายและช่วงค่าของวันที่  
D) วันที่ดังกล่าวเป็น valid date เพราะมีตัวเลขครบสิบตัว

**Q56** [W13] [sequential topology] [difficulty: easy]  
Sequential pipeline มี Research ใช้ 3 วินาที, Writer ใช้ 10 วินาที และ Editor ใช้ 4 วินาที หากแต่ละขั้นเริ่มได้เมื่อขั้นก่อนหน้าจบ และไม่มี overhead รวม latency เท่าใด?

A) 17 วินาที  
B) 10 วินาที  
C) 14 วินาที  
D) 120 วินาที

**Q57** [W13] [routing topology] [difficulty: medium]  
ความเสี่ยงเด่นของ routing network คือข้อใด?

A) ทุก specialist ต้องทำงานทุกคำถามเสมอ  
B) ไม่สามารถแยก specialist ตามโดเมนได้  
C) ต้องเรียกทุก agent ตามลำดับตายตัว  
D) Router ส่งงานผิด specialist ได้ จึงเป็นจุดล้มเหลวสำคัญ

**Q58** [W13] [multi-agent topologies] [difficulty: medium]  
ข้อใดจับคู่ topology กับลักษณะได้ถูกต้อง?

A) Sequential: หลาย agents เสนอคำตอบอิสระพร้อมกันและมี judge  
B) Supervisor-led: ผู้ควบคุมมอบหมายงานและประเมินผลจาก sub-agents  
C) Routing: output ทุก agent ต้องต่อเป็นสายเดียวเสมอ  
D) Debate: มี specialist เพียงตัวเดียวและไม่มีการเปรียบเทียบคำตอบ

**Q59** [W13] [agent memory] [difficulty: medium]  
Retrieval memory เลือก top-\(k\) รายการเข้า prompt ข้อใดถูกต้อง?

A) Storage ทั้งหมดจึงต้องมีขนาด \(O(k)\) เสมอ  
B) ไม่อาจพลาดเหตุการณ์สำคัญในอดีตได้  
C) ขนาด context ที่ดึงมาอาจจำกัดด้วย \(k\) แม้ storage ยังคงเติบโตตามจำนวนรายการ  
D) จำเป็นต้องใส่ประวัติทั้งหมดเข้า prompt ทุกครั้ง

**Q60** [W13] [JSON schema] [difficulty: easy]  
หาก tool schema กำหนด `required: ["query"]` tool arguments ที่ถูกต้องตามข้อกำหนดนี้ต้องมีอะไร?

A) key `query` พร้อมค่าที่ตรงชนิดข้อมูลใน schema  
B) key `response` เท่านั้น  
C) string เปล่าแทน JSON object เสมอ  
D) key `query` อยู่ใน description โดยไม่ต้องอยู่ใน arguments

**Q61** [W14] [deduplication] [difficulty: medium]  
ข้อใดเปรียบเทียบ exact deduplication กับ near-deduplication ได้ถูกต้อง?

A) SHA-256 ใช้ประมาณ cosine similarity ของเอกสาร  
B) SHA-256 รับประกันจับเอกสารที่แก้คำเล็กน้อยว่าเหมือนกัน  
C) MinHash ใช้ได้เฉพาะเอกสารที่เหมือนกันทุก byte  
D) Hash จับ exact duplicates หลัง normalization ส่วน shingles และ MinHash ช่วยหา near-duplicates

**Q62** [W14] [Jaccard similarity] [difficulty: medium]  
Shingle sets \(A\) และ \(B\) มีขนาด 5 และ 4 ตามลำดับ และมีสมาชิกร่วมกัน 3 ตัว ค่า Jaccard similarity เท่าใด?

A) 0.3333  
B) 0.5  
C) 0.6  
D) 0.75

**Q63** [W14] [MinHash] [difficulty: easy]  
MinHash signatures สองชุดตรงกัน 42 ตำแหน่งจาก 64 ตำแหน่ง ค่าประมาณ Jaccard เท่าใด?

A) 0.065625  
B) 0.420000  
C) 0.65625  
D) 0.76190

**Q64** [W14] [LSH banding] [difficulty: hard]  
ให้โอกาสเป็น candidate ของ LSH เท่ากับ \(1-(1-s^r)^b\) ถ้า \(s=0.5\), \(r=2\), \(b=2\) โอกาสนี้เท่าใด?

A) 0.4375  
B) 0.2500  
C) 0.5000  
D) 0.7500

**Q65** [W14] [perplexity filtering] [difficulty: medium]  
เหตุใดการใช้ perplexity ต่ำเพียงอย่างเดียวจึงไม่รับประกันว่าเอกสารมีคุณภาพสูง?

A) Perplexity วัดได้เฉพาะจำนวนตัวอักษร  
B) ข้อความซ้ำคำนวณ perplexity ไม่ได้  
C) เอกสารเชิงเทคนิคทุกฉบับต้องมี perplexity ต่ำ  
D) ข้อความซ้ำอาจทำนายง่าย ขณะที่เนื้อหาคุณภาพเฉพาะทางอาจทำนายยาก

**Q66** [W14] [filter ordering] [difficulty: hard]  
Filter A มีต้นทุน 1 ms/doc และเก็บ 20% ส่วน B มีต้นทุน 10 ms/doc และเก็บ 50% สมมติ retention ของแต่ละ filter คงเดิมไม่ว่าลำดับใด ลำดับใดมี expected cost ต่อเอกสารเริ่มต้นต่ำกว่า?

A) B → A มีต้นทุน 3 ms  
B) A → B มีต้นทุน 3 ms  
C) A → B มีต้นทุน 11 ms  
D) B → A มีต้นทุน 0.7 ms

**Q67** [W14] [BERTScore] [difficulty: medium]  
หาก BERTScore precision \(P=0.8\) และ recall \(R=0.6\) ค่า \(F_1=2PR/(P+R)\) ใกล้เคียงเท่าใด?

A) 0.4800  
B) 0.6000  
C) 0.6857  
D) 0.8000

**Q68** [W14] [pass@k] [difficulty: hard]  
สำหรับโจทย์เขียนโปรแกรมหนึ่งข้อ มี samples \(n=10\) ตัว โดยผ่าน tests \(c=2\) ตัว ใช้  
\(\mathrm{pass@}k=1-\binom{n-c}{k}/\binom{n}{k}\)  
จงหา pass@2

A) ประมาณ 0.3778  
B) 0.2000  
C) ประมาณ 0.6222  
D) 0.8000

**Q69** [W14] [LLM-as-a-judge] [difficulty: medium]  
หากต้องการตรวจ positional bias ของ judge ในการเปรียบเทียบคำตอบ A และ B วิธีใดเหมาะสม?

A) ประเมินเฉพาะลำดับ A แล้ว B หนึ่งครั้ง  
B) เลือกคำตอบที่ยาวกว่าเป็นผู้ชนะเสมอ  
C) ตัด rubric ออกเพื่อลดอิทธิพลของคำสั่ง  
D) ทดลองสลับลำดับ A–B และ B–A แล้วเปรียบเทียบผล

**Q70** [W14] [RAG evaluation] [difficulty: medium]  
คำตอบมี faithfulness สูงเมื่อเทียบกับ context ที่ดึงมา ข้อสรุปใดถูกต้อง?

A) คำตอบต้องเป็นความจริงเสมอ แม้ context ผิด  
B) คำตอบสอดคล้องกับ context ตามเกณฑ์ประเมิน แต่ context อาจผิดหรือไม่ครบ  
C) Retriever ต้องค้นเอกสารที่เกี่ยวข้องครบทุกฉบับ  
D) คำตอบต้องเกี่ยวข้องกับคำถามเสมอ

## Part B — Fill-in

**F1** [W8] [T5 token counts]  
ข้อความต้นฉบับมี 20 tokens ตัดออก 2 spans รวม 6 tokens แทนแต่ละ span ด้วย sentinel หนึ่งตัวใน encoder input ส่วน decoder target มี sentinel นำหน้าแต่ละ span, tokens ที่ถูกตัด, sentinel ปิดท้ายหนึ่งตัว และ `</s>` หนึ่งตัว โดยไม่มี special token อื่น Encoder input ยาว ___ tokens และ decoder target ยาว ___ tokens

**F2** [W9] [triplet loss]  
กำหนด \(L=\max(0,d(a,p)-d(a,n)+\alpha)\), \(d(a,p)=0.087\), \(d(a,n)=0.035\), \(\alpha=0.5\) ค่า loss เท่ากับ ___

**F3** [W10] [KV cache]  
ใช้ \(M_{KV}=2\times b\times L\times H_{KV}\times d_h\times S\times B\) bytes โดย \(b=2\), \(L=32\), \(H_{KV}=8\), \(d_h=128\), \(S=8192\), \(B=4\) ขนาด KV cache เท่ากับ ___ GiB

**F4** [W11] [GRPO advantage]  
สำหรับ rewards `[1, 1, 0, 0]` ใช้ population standard deviation และ \(A_i=(r_i-\mu)/\sigma\) ค่า mean เท่ากับ ___ และ advantage ของคำตอบที่ได้ reward 1 เท่ากับ ___

**F5** [W12] [BM25]  
เมื่อ \(k_1=1.5\), \(b=0\), \(f=4\) ค่า TF factor \(\frac{f(k_1+1)}{f+k_1}\) เท่ากับ ___ และเพดานที่ factor นี้เข้าใกล้เมื่อ \(f\) เพิ่มมากขึ้นเท่ากับ ___

**F6** [W12] [RRF]  
ใช้ \(k=60\) และอันดับเริ่มที่ 1 เอกสารหนึ่งอยู่ rank 1 ในระบบแรกและ rank 2 ในระบบที่สอง คะแนน RRF เท่ากับ ___

**F7** [W13] [logit masking]  
Vocabulary มี 50,257 tokens และ parser อนุญาต 691 tokens ใน state หนึ่ง ต้อง mask ___ tokens โดยตั้ง logits ของ tokens เหล่านั้นเป็น ___ ก่อน softmax

**F8** [W13] [sequential topology]  
Sequential pipeline มี 3 ขั้น ใช้เวลา 2, 3 และ 5 วินาทีตามลำดับ โดยไม่มี overhead latency รวมเท่ากับ ___ วินาที

**F9** [W14] [Jaccard similarity]  
Shingle sets สองชุดมีขนาด 7 และ 5 และมีสมาชิกร่วมกัน 4 ตัว ขนาด union เท่ากับ ___ และ Jaccard similarity เท่ากับ ___

**F10** [W14] [pass@k]  
กำหนด \(n=10\), \(c=2\), \(k=2\) และ \(\mathrm{pass@}k=1-\binom{n-c}{k}/\binom{n}{k}\) ค่า pass@2 เท่ากับ ___


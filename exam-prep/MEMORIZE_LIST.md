# MEMORIZE LIST — ต้องอยู่ในหัว (ไม่ลอกลง A4)

> หลักการ: สิ่งที่นี่คือ **เหตุผล / สัญชาตญาณ / นิยาม / ลำดับขั้น / mnemonic** ที่ตอบ MCQ แบบ "ทำไม / ข้อใดถูก / ข้อใดไม่ใช่" ได้เร็ว. ตัวเลขและสูตรอยู่ใน `CHEATSHEET_A4_handwrite.md`.
> รูปแบบ: ข้อเท็จจริง -> `ถาม:` self-test 1 บรรทัด (ปิดคำตอบแล้วลองตอบออกเสียง). mnemonic ที่ระบุ "(ตั้งเอง)" ผู้เขียนคิดขึ้นเพื่อช่วยจำ ไม่ได้มาจากสไลด์.
> แหล่งข้อมูลทุกข้อ: `notes/week_NN_notes.md` + สไลด์/ hint. ข้อที่ไม่ได้อยู่ในโน้ตติดป้าย (นอกโน้ต).
> ที่มาของความถี่: [H] = ครูพูดว่าออก, [M] = ปรากฏใน mock ทั้ง 3 ชุด (`mock/`), [W] = สไลด์เยอะ/น้ำหนักมากในสัปดาห์.

---

## W7 carry-over (hint เท่านั้น — โน้ตสัปดาห์ 8-14 ไม่ครอบคลุม) — 6 บรรทัด
1. [H] Original Transformer = **Post-Norm** (Norm หลัง attention/FFN); โมเดลสมัยใหม่ = **Pre-Norm + RMSNorm** (norm ก่อน attention) — ครูบอก "ต้องเข้าใจทั้งสองยุค". ถาม: LLaMA วาง norm ก่อนหรือหลัง attention?
2. [H] MLM mask 15% ของ token: 80% -> [MASK], 10% -> random token, 10% -> เหมือนเดิม. CLM เรียนจาก 100% token. ถาม: 100 token, MLM ใช้เทรนกี่ตัว? (15)
3. [H] GQA: โจทย์จะ "ให้" จำนวน Q head และจำนวนกลุ่ม KV มา; KV ลด = n_heads / n_kv. ถาม: 64 Q head, 8 กลุ่ม -> KV cache เล็กลงกี่เท่า? (8x; 160 GB -> 20 GB)
4. MHA vs MQA vs GQA: MHA = KV ทุกหัว, GQA = KV หลายกลุ่ม, MQA = KV หัวเดียว (นิยาม MQA = นอกโน้ต; โน้ตมีแต่ MHA/GQA). ถาม: GQA เป็นจุดกลางระหว่างอะไรกับอะไร?
5. [H] ครูย้ำ: ข้อสอบเขียนระวัง **ทศนิยม / เลขยกกำลัง 10** (10.7 เขียนผิดเป็น 0.107 เสียคะแนน) และบางข้อ "แค่บวกคูณ". ถาม: 0.0075 รากที่สองประมาณเท่าไร? (0.0866 ไม่ใช่ 0.866)
6. [H] **Lab ไม่ออก**; demo-only อาจออก 1-2 ข้อทั้งฉบับ; แต่ละสัปดาห์ได้ส่วนแบ่งเท่ากัน (~10 MCQ + 1-2 ช่อง); ในสัปดาห์: หัวข้อที่ครูพูดนาน + สไลด์เยอะ = ออกเยอะ.

---

## W8 Encoder-Decoder (BART / T5) — 20 บรรทัด
1. **ทำไมต้อง Enc-Dec**: encoder อ่านสองทางแต่ generate ไม่ได้; decoder generate ได้แต่อ่าน causal -> Enc-Dec อ่าน input เต็มก่อน (bidirectional) แล้ว generate ผ่าน cross-attention. ถาม: encoder-only และ decoder-only แต่ละแบบขาดอะไร?
2. **Denoising**: corrupt x -> x~ แล้ว reconstruct x; **ชนิด corruption กำหนดสิ่งที่โมเดลเรียน**; MLM = denoising แบบอ่อน. ถาม: P(x | x~) ต่างจาก GPT P(next | prev) อย่างไร?
3. **BART 5 noising [H "ออกแน่นอน"]** mnemonic (ตั้งเอง): **M-D-I-S-R** = Mask, Delete, Infill, Sentence-permute, Rotate. ถาม: เรียง 5 วิธี + บอกว่า span corruption ไม่ใช่ของ BART
4. **Infilling = signature ของ BART**: span ~ Poisson(lambda=3) -> [MASK] ตัวเดียว; 1 [MASK] = 0 ขึ้นไปหลาย token, ความยาว 0 = แทรก [MASK] -> โมเดลต้องเดา **จำนวน token ที่หาย**. ถาม: ต่างจาก token masking ของ BERT อย่างไร?
5. **Deletion**: ลบแล้วไม่เหลือ placeholder -> โมเดลต้องตรวจ "อะไรหาย **และหายที่ไหน**", src < tgt; Masking src = tgt. ถาม: วิธีไหนที่ src_len < tgt_len?
6. **Sentence permutation / Rotation**: สอน discourse / จุดเริ่มต้นจริงของเอกสาร (global structure ไม่ใช่แค่ local). ถาม: rotation ให้โมเดลระบุอะไร?
7. **BART decoder reconstruct ข้อความเต็ม** (loss ทุก target token, self-attn O(N^2)) vs **T5 decoder สร้างเฉพาะ span ที่หาย**. ถาม: target ของใครสั้นกว่า และทำไมถูกกว่า?
8. **T5 sentinel** [H "ออกแน่ๆ"]: unique เรียงเลข `<extra_id_0..>` ; target = sentinel นำหน้า span แต่ละอัน + span, **จบด้วย sentinel ถัดไป แล้ว `</s>`**; sentinel สุดท้ายไม่ใช่ EOS (= "ไม่มี span หายอีกแล้ว"). ถาม: target ของ input `The <e0> cat sat <e1>` ลงท้ายด้วยอะไร?
9. **ที่มา 25x**: target ~0.2N (15% corrupt + 5% sentinel) -> (0.2)^2 = 0.04 -> 1/0.04 = 25 เฉพาะ decoder self-attn; encoder เท่าเดิม -> ทั้งโมเดล 2-3x. ถาม: ทำไมไม่ใช่ 25x ทั้งโมเดล? (ระวัง 0.04 ไม่ใช่ 0.4)
10. **T5 text-to-text** [H]: ทุก task = text in -> text out; prefix = **task switch**; single loss + single decoding; **ไม่มี task head** (BERT ต้องมี head ต่อ task). ไม่ใส่ prefix -> demo: fallback ไปทำ **translation**. ถาม: sst2 sentence: ใส่/ไม่ใส่ prefix ต่างกันอย่างไร?
11. **Relative position bias** [H "น่าจะออก"]: ระยะใกล้ = bucket ละตัว, ไกลแชร์ bucket (log), learned scalar **บวกเข้า attention logits ก่อน softmax**; สูตร log **ไม่ออก**. ถาม: bias บวกก่อนหรือหลัง softmax?
12. **Cross-attention**: Q จาก decoder, K/V จาก encoder (คำนวณครั้งเดียว cache), ไม่ต้อง mask; weights = [target x source]. Encoder self = full; decoder self = causal. ถาม: Q, K, V มาจากไหน?
13. **นับ token** (ขั้นตอน): (1) นับ n, r (token ที่ตัด), k (จำนวน span) (2) src = n - r + k (3) T5 tgt = r + k + 1 (sentinel ปิด) (4) บวก `</s>` / `<s>` ตามที่โจทย์บอก. ถาม: 12 token ตัด 2 span รวม 4 token, T5 src/tgt = ? (10 / 7 หรือ 8)
14. **MLM 15% vs CLM 100%** = signal density; BERT ทำนายคำที่ mask **อิสระต่อกัน** -> generate ไม่สะดวก. ถาม: ทำไม CLM "ได้สัญญาณมากกว่า" ต่อ sequence?
15. **ทำไม decoder-only ชนะ** (3 ข้อ): serving ง่าย + ไม่ซ้ำ param (Enc-Dec = P/2+P/2, สอง routing pass) / RLHF multi-turn (Enc-Dec ต้อง re-encode ทุกเทิร์น) / KV cache เส้นตรง (Enc-Dec มีสอง cache). ถาม: เลือกได้ 3 เหตุผลและอย่าตอบ "attention ถูกกว่า"
16. **หลักฐาน**: ไม่มีการเทียบควบคุมที่ >100B; Wang 2022: dec-only ชนะ zero-shot หลัง pretrain อย่างเดียว, enc-dec ชนะ **หลัง multitask finetune**. ถาม: "พิสูจน์แล้วว่า dec-only ดีกว่าที่ >100B" จริงไหม?
17. **Enc-Dec ยังชนะ**: input ยาว output สั้น (FLOPs ถูกกว่า), MT (NLLB), speech-to-text (Whisper), summarization -> **specialist vs generalist**. ถาม: ยกตัวอย่าง 2 งานที่ Enc-Dec ยังดี
18. **Extractive vs Generative QA**: BERT = pointer start/end (verbatim, ออกนอก context ไม่ได้, ไม่มี context -> ไฮไลต์คำถามตัวเอง) / T5 = generate (hallucinate ได้ แต่ไม่จำเป็นต้องออกนอก context). ถาม: ความต่างอยู่ที่ผลลัพธ์หรือ mechanism?
19. **ปรัชญา**: BART = maximize generation quality (heads หรือ generation); T5 = efficiency + generality (always text-to-text). UL2: **objective สำคัญพอๆ กับ architecture**. ถาม: ใครใช้ `<s>` ใครใช้ `<pad>` เป็น decoder start?
20. **demo-only (อาจออก)**: T5 ใน demo ใช้ span_len คงที่ 3 (จริง Poisson); ratio demo BART/T5 = 1.50; cola = acceptability (ไม่ใช่ correction) และ t5-small ตอบผิด. ถาม: CoLA ตัดสินอะไร?

---

## W9 Sentence Embeddings — 20 บรรทัด
1. **เป้าหมาย**: f: ประโยค -> R^d ที่ sim(f(a), f(b)) ~ ความคล้ายเชิงความหมาย; ใช้ใน search, dedup, cluster, recsys, RAG. ถาม: ต่างจาก [CLS]+linear head ของ W6 อย่างไร?
2. **Raw BERT แย่กว่า GloVe** (CLS 16.50 < mean 46.35 < GloVe 58.02): สาเหตุคือ **MLM objective ไม่ปั้น geometry ระดับประโยค** ไม่ใช่ architecture; [CLS] ถูกฝึก NSP. ถาม: จะแก้ด้วยการเพิ่ม layer ได้ไหม? ทำไม?
3. **Anisotropy**: embedding อยู่ใน narrow cone -> ไม่มีคู่ไหนได้ cosine ต่ำ -> cosine จัดอันดับไม่ได้; mean pooling สืบทอดปัญหา; 3 วิธีแก้: post-process / **contrastive FT** / SimCSE. ถาม: pooling อย่างเดียวแก้ cone ไหม?
4. **Siamese** = architecture เดียว + **weights เดียว (shared)**, encode แต่ละประโยคอิสระ -> cache vector ได้. ถาม: ถ้าสองสายคนละ weights จะเทียบ vector ได้ไหม?
5. **Bi vs Cross** [H]: bi = n forward แล้ว n^2 dot (ระดับ **ns**), cache ได้, ใช้ **recall**; cross = n^2 forward (ระดับ **ms**), full cross-attn, ใช้ **rerank precision**. ต้องรู้ว่า O() **นับอะไร** (forward pass vs dot product). mnemonic (ตั้งเอง): **Bi = Broad, Cross = Careful**. ถาม: O(n^2) ของ cross-encoder นับอะไร?
6. **Retrieve-then-rerank**: bi-encoder ดึง top-k จากหลายล้าน -> cross-encoder ให้คะแนนใหม่. ถาม: ทำไมไม่ใช้ cross-encoder กับทั้ง corpus?
7. **SBERT 3 config**: Classification (NLI, softmax บน [u;v;|u-v|]) / Regression (STS, MSE บน cos) / Triplet. ถาม: ทำไม concat ถึงมี |u-v| ไม่ใช่แค่ u,v? (เหตุผลที่ว่า head เห็น "ระยะ" ตรงๆ = นอกโน้ต; ที่แน่ใจ: 2304 ไม่ใช่ 1536)
8. **Pooling**: mean ที่ **mask [PAD]** ดีสุด; PAD ไม่ใช่เวกเตอร์ศูนย์ (norm 9.67) -> เฉลี่ยรวมจะ corrupt. ถาม: ทำไมต้องคูณ attention mask ก่อนเฉลี่ย?
9. **Contrastive = attract + repel**: positive ใกล้ negative ไกล -> **Alignment + Uniformity** (Wang & Isola). ถาม: ทำไมจึงแก้ cone ได้?
10. **Triplet** [H written part]: L = 0 เมื่อ positive ใกล้กว่า negative อย่างน้อย margin; margin กัน lazy learning, ใหญ่ = เข้มกว่า/ฝึกยากกว่า. ถาม: margin ใหญ่ขึ้น negative ง่ายเดิมยังให้ loss 0 ไหม?
11. **Mining**: easy (L=0 เรียนไม่ได้) / hard (gradient ใหญ่แต่ไม่เสถียร) / **semi-hard = สมดุล (FaceNet)**. ถาม: ทำไมไม่เลือก hard ล้วน?
12. **Triplet vs Margin Ranking**: triplet ใช้ **distance** (min d_pos), margin ranking ใช้ **similarity** (max sim_pos); เทียบเท่ากันเมื่อ d = 1 - sim เท่านั้น. ถาม: Euclidean triplet เท่ากับ margin ranking ไหม?
13. **MNRL / InfoNCE**: negative = ตัวอื่นใน batch ฟรี (N-1), ต้องการแค่คู่ (q,p); **tau ต่ำ = คม/เข้ม**; batch ใหญ่ = negative มากขึ้น. ถาม: tau สูงหรือต่ำที่ boundary คมกว่า?
14. **NLI -> STS สองเฟส**: NLI สอน "คล้าย=ใกล้ ต่าง=ไกล" โดยไม่ต้องมีเป้า cosine; entailment ดึงใกล้, contradiction ผลัก, neutral ไม่ใช่ positive; STS ปรับละเอียด (score 0-5 -> 0-1). ถาม: label ไหนคือ hard negative ของ SimCSE sup?
15. **SimCSE unsup**: ประโยคเดิมผ่าน BERT 2 ครั้ง **dropout ต่างกัน** = positive, ไม่ใช้ label. ถาม: positive ของ SimCSE unsup มาจากไหน?
16. **Normalization**: unit vector -> dot = cosine = L2 ranking เท่ากัน; ข้ามขั้นนี้ `IndexFlatIP` จะจัดอันดับตาม**ความยาว** ไม่ใช่ความหมาย; `.encode()` ไม่ normalize ให้ (default off). ถาม: ทำไม "เลือก IndexFlatIP" หลัง normalize?
17. **ANN** = accuracy-latency trade-off; HNSW = multi-layer graph (top sparse -> greedy walk -> drop down -> layer 0); **เทียบ speed โดยไม่รายงาน Recall@k ถือว่าไม่ถูก**. ถาม: ANN รับประกัน exact ไหม?
18. **Metrics**: Recall@k = coverage (relevant ที่เจอใน top-k / relevant ทั้งหมด); MRR = ตำแหน่งแรกที่ถูก (rank2 = 0.5 ไม่ใช่ 2). ถาม: MRR ของ rank 1, 2, 4?
19. **วิวัฒนาการ**: SBERT (2019) -> SimCSE (2021) -> E5 (2022, weak web-scale) -> BGE (2023, + instruction) -> instruction-based embeddings; recipe: pre-train weak ขนาดใหญ่ แล้ว fine-tune gold. ถาม: ทำไมโมเดลใหม่ดีกว่า SBERT?
20. **ระวัง**: Week 6 = [CLS]+head vs Week 9 = [u;v;|u-v|]; head 2304 ไม่ใช่ 768/1536/3072; ข้อเขียนระวังทศนิยม (0.0075 vs 0.075). ถาม: เปลี่ยนเป็น BGE-large (1024) head กี่มิติ? (3072)

---

## W10 Hardware-aware NLP / PEFT — 22 บรรทัด
1. **Memory hierarchy**: SRAM (เร็ว เล็ก) > VRAM/HBM > system RAM (ช้า ใหญ่); เป้าหมาย = เก็บไว้ใน VRAM. ถาม: FlashAttention ทำงานที่ชั้นไหน?
2. **Static vs dynamic**: weights = static; KV cache + activations = dynamic -> **OOM กลาง generation** มักมาจาก KV cache โต. ถาม: ทำไมโหลดได้แต่ crash ที่ token ที่ 500?
3. **KV formula เหตุผล**: ตัว 2 = K และ V; ใช้ **n_kv** (ไม่ใช่ n_heads) ตัวคูณ batch และ ctx; ลืมตัว 2 / ใช้ n_heads แทน n_kv = ผิดที่พบบ่อย. ถาม: GQA 8 KV head เทียบ MHA 32 ลดกี่ %? (75%)
4. **GB vs GiB** [H เคยออกกลางภาค]: GiB เป็นตัวเลขที่ **เล็กกว่า** GB (หาร 1.0737 ไม่ใช่คูณ); nvidia-smi รายงาน MiB. ถาม: 16 GB = กี่ GiB? (14.9)
5. **16N**: weights 2N + grads 2N + optimizer 12N (FP32 master 4 + momentum 4 + variance 4) -> optimizer 75%; ไม่รวม activations; full FT ~ 8x inference. ถาม: ทำไม optimizer ถึงใหญ่สุด?
6. **BF16 vs FP16**: BF16 = range (exp 8 bit) แลก precision (mantissa 7); FP16 overflow ง่าย (65,504); T4 ไม่มี BF16. ถาม: BF16 แม่นกว่า FP16 ไหม? (ไม่ ช่วงกว้างกว่า)
7. **Prefill vs Decode**: prefill = GEMM compute-bound (parallel); decode = GEMV **memory-bandwidth-bound** (stream weights ต่อ 1 token, AI ต่ำ). ถาม: TTFT ถูกกำหนดโดยอะไร?
8. **TTFT vs TPS**: speculative decoding -> TPS; chunked prefill -> TPS ของ request เดิม แต่ TTFT ของ request ใหม่แย่ลง. ถาม: เทคนิคไหนช่วย TTFT, ไหนช่วย TPS?
9. **FlashAttention**: exact attention, **tiling ใน SRAM + online softmax**, extra memory O(N), recompute ใน backward; **ลด memory/IO ไม่ลด FLOPs**; ไม่ต้องจำสูตร [H]. ถาม: เป็น approximation ไหม?
10. **MoE** [H weak]: ต้องการความรู้มากโดยไม่เพิ่ม compute; router เลือก top-k; **เร็วขึ้นแต่ไม่ประหยัด memory** (โหลดทุก expert); aux loss กัน router ส่งทุก token ไป expert เดียว; DeepSeek = shared + fine-grained routed. ถาม: Mixtral 13B active ใช้ VRAM แค่ 13B ไหม?
11. **Fragmentation**: internal (จองเกินไม่ใช้) vs external (ช่องว่างระหว่าง request); naive ใช้ KV ที่จองได้ 20-40%. ถาม: ต่างกันอย่างไร?
12. **3 ตัวที่ชอบสลับ** mnemonic (ตั้งเอง): **Flash = fast kernel (exact), Paged = pages (OS-style KV blocks), Continuous = calendar (iteration-level scheduling)**. ถาม: ตัวไหนจัดการ fragmentation?
13. **Speculative decoding**: draft เล็ก -> verify ใหญ่ใน 1 forward; **distribution เท่า target เป๊ะ**; เหมาะ single-user/local (concurrency สูง -> vLLM). ถาม: คุณภาพลดไหม?
14. **TP vs PP**: TP แบ่ง matrix (ทุก GPU ทำทุก layer, ต้อง all-reduce เร็ว -> intra-node NVLink); PP แบ่ง layer (bubble, inter-node); 3D = TP+PP+DP. ถาม: ใครมี bubble?
15. **LoRA เหตุผล**: intrinsic rank ต่ำ -> ΔW ~ BA; **B=0, A~Gaussian** -> ΔW=0 ตอนเริ่ม (ถ้าทั้งคู่ 0 gradient ตาย — นอกโน้ต/นอกสไลด์); merge ได้ -> ไม่เพิ่ม latency; LoRA ลดแค่ trainable params (ไม่ลด memory ของ base, ไม่ลด inference compute). ถาม: ทำไม init B เป็นศูนย์?
16. **QLoRA**: base 4-bit **frozen**; forward = dequant เป็น 16-bit on-the-fly + LoRA path; gradient เฉพาะ A,B; 3 = NF4 / Double Quant (quantize **scale**) / Paged optimizer (-> CPU RAM). ถาม: Double Quant quantize อะไร?
17. **Quantization**: absmax scale; outlier ทำให้ scale หยาบ -> **block-wise (64)** ลด error; LLM.int8 outlier features เมื่อโมเดลใหญ่; NF4 = sweet spot (4x เล็ก คุณภาพเกือบเท่า 16-bit ต้องวัดเอง). ถาม: block-wise ช่วยอะไร?
18. **Distillation**: soft targets / dark knowledge (cat 0.9, tiger 0.08 บอก "ความคล้าย"); T>1 ทำ softmax แบน; T^2 ชดเชย gradient 1/T^2. ถาม: ทำไมใช้ soft ไม่ใช้ one-hot?
19. **3 แกน compression**: quant = bits, LoRA = trainable params, **distill = layers (แกนเดียวที่ลด FLOPs)**; quant 12 layer ยังเดิน 12 layer. ถาม: อะไรลด compute จริง?
20. **Decision tree**: พอใน FP16 -> FP16+Flash+vLLM; ตึง -> quantize; fine-tune GPU เดียว -> LoRA/QLoRA; concurrency สูง -> vLLM; ผู้ใช้เดียว -> speculative; compute คือกำแพง -> distill. ถาม: GPU เดียว fine-tune 8B ทำอย่างไร?
21. **Adapters**: MAD-X = Language adapter + Task adapter (L+T ไม่ใช่ LxT); O-LoRA = orthogonal subspace **ลด** forgetting ไม่หมด. ถาม: Thai NER โดยไม่มี Thai NER data ทำอย่างไร?
22. **Gradient checkpointing**: แลก compute กับ memory (O(L) -> O(sqrt L), +~30% เวลา). ถาม: ลดทั้ง memory และ compute ไหม? (ไม่)

---

## W11 Prompting / Post-training / Reasoning — 20 บรรทัด
1. **Base vs Aligned**: base แค่ continue เอกสาร (ไม่ตอบ, ไม่รู้จักหยุด); SFT สอบรูปแบบ; param เท่ากัน (demo 494.0M ทั้งสอง). ถาม: instruct ต่างจาก base ที่ param ไหม?
2. **ICL** = เรียนตอน inference **ไม่อัปเดต weights**; few-shot ดีที่ format/edge case ล้มเหลวที่ multi-step reasoning. ถาม: few-shot อัปเดต weights ไหม?
3. **CoT**: token ที่สร้างเป็น external scratchpad = compute เพิ่มตามจำนวน token; ได้ผลในโมเดลใหญ่พอ; ต้นทุน: attention quadratic, KV linear, decode memory-bound. ถาม: CoT ยาวขึ้น 2x attention/KV เพิ่มกี่เท่า? (4x / 2x)
4. **CO-STAR** [H weak] ตัวย่อ = Context Objective Style Tone Audience Response-format; best practice: "Do X" ไม่ใช่ "Don't Y". ถาม: A ตัวสุดท้ายคืออะไร?
5. **Self-consistency ลำดับ**: temp > 0 -> sample หลาย CoT (5-10) -> ดึง final answer -> majority vote; ไม่ต้อง train verifier. ถาม: temp = 0 ใช้ได้ไหม?
6. **PoT/PAL**: LLM เขียนโค้ด, interpreter คำนวณ -> แยก reasoning ออกจาก computation. ถาม: ใครคำนวณตัวเลข?
7. **ToT vs GoT**: CoT = เส้นตรง; ToT = tree + search (generate-evaluate-prune, backtrack, branch แยกขาด); GoT = graph (aggregate รวม thought + refine วนกลับ). ถาม: อันไหน merge thoughts ได้?
8. **Reflection** (generate -> critique -> refine): ได้ผลเพราะ **ประเมินง่ายกว่าสร้าง**; ระดับ prompt ไม่ต้องเทรน. ถาม: ทำไม critique ถึงมีประโยชน์?
9. **Prompt injection vs jailbreak**: injection = คำสั่งปนกับ data ช่องภาษาธรรมชาติเดียวกัน (direct = user พิมพ์, **indirect** = ซ่อนในเว็บ/อีเมลที่ LLM ดึง); jailbreak = ล่อ aligned model (DAN, encoding, hypothetical). ป้องกัน = defense-in-depth 4 ชั้น (ไม่มี silver bullet). ถาม: indirect ซ่อนที่ไหน?
10. **SFT**: behavior cloning ด้วย demos หมื่นระดับ, objective เดิม next-token; ข้อจำกัด: แพง, **exposure bias** (teacher forcing ตอน train vs autoregressive ตอน infer), ไม่มี negative feedback. ถาม: exposure bias มาจากอะไร?
11. **HHH**: Helpful / Honest / Harmless (Anthropic 2021) มี tension helpful vs harmless. ถาม: H ที่สองคืออะไร?
12. **RLHF 3 ขั้น**: SFT (RL ไม่เสถียร ต้องเริ่มจากโมเดลดี) -> Reward Model (Bradley-Terry: เพิ่มส่วนต่าง r_w - r_l; init จาก SFT, head scalar) -> PPO. ถาม: ทำไม SFT ก่อน RL?
13. **PPO 4 models**: Actor + Critic (อัปเดต) | Reward + Reference (frozen); KL penalty = "leash" กัน **reward hacking**; ปัญหา: กิน memory, ไม่เสถียร, pipeline ซับซ้อน. ถาม: โมเดลไหน frozen?
14. **DPO**: จัดสมการ RLHF ใหม่ -> เทรนตรงจาก preference ด้วย classification loss; ไม่ต้อง RM/PPO; 2 models; **beta ใหญ่ = อยู่ใกล้ ref มากขึ้น**; เริ่ม (policy = ref) loss = ln2 = 0.693; ข้อเสีย overfit เร็ว. ถาม: beta ใหญ่ -> ใกล้หรือไกล ref?
15. **GRPO**: ทำเพราะ PPO ต้อง 4 models (~4x VRAM); ตัด **Critic**, baseline = mean ของกลุ่ม; A = z-score (หาร G); A>0 เพิ่ม prob; ผลรวม A = 0; rewards เท่ากันหมด -> A = 0. ถาม: GRPO ตัดโมเดลอะไร? (Critic ไม่ใช่ Reward/Ref)
16. **R1-Zero**: pure RL ไม่มี SFT; rule-based reward (ตอบตรง) + format reward `<think>`; เกิด self-correction / "Aha" / ยาวขึ้นเอง; ได้ผลเฉพาะ **verifiable** (math, code). ถาม: ทำไมต้อง verifiable?
17. **R1 vs o-series**: R1 trace มองเห็น `<think>`; o-series ซ่อน. Test-time compute: คิดนานขึ้น = ดีขึ้น แต่ **diminishing returns**, TTFT สูง -> งานง่ายใช้ Helper. ถาม: คิดนานขึ้นดีขึ้นเสมอไหม?
18. **System 1 vs 2**: LLM ปกติ = System 1 (เร็ว); reasoner = System 2 (ช้า วิเคราะห์ backtrack). ถาม: pipeline Scholar -> Helper -> Thinker คืออะไร? (pretrain -> SFT+DPO -> GRPO)
19. **Chat template**: roles System/User/Assistant; special tokens ต่าง family (ChatML `<|im_end|>`, Llama3 `<|eot_id|>`); ใช้ SFT model โดยไม่ใช้ template ตรงกัน = performance ตก; `add_generation_prompt=True`. ถาม: ทำไม template ไม่ใช่แค่ convention?
20. **demo-only**: few-shot ไม่ได้ดีกว่า zero-shot เสมอ (0.5B ตอบ negative); `logits[prompt_len-1 : prompt_len+resp_len-1]` (off-by-one); beta default 0.1; log-prob รวม = ผลรวมต่อ token. ถาม: ทำไม slice เริ่มที่ prompt_len-1?

---

## W12 Hybrid RAG / CLIP / GraphRAG — 22 บรรทัด
1. **ทำไม RAG**: parametric memory มี knowledge cutoff + private data -> LLM confabulate อย่างมั่นใจ; ยัด prompt พัง (O(N^2)), fine-tune แพงและสร้าง cutoff ใหม่. ถาม: ทำไมไม่ยัดทุกอย่างใน prompt?
2. **Lewis RAG**: retriever = non-parametric, generator = parametric, marginalize top-K docs, end-to-end. ถาม: อะไรคือ non-parametric memory?
3. **Ingestion 5 ขั้น** (mnemonic ตั้งเอง: "L-P-C-E-I"): Load -> Parse -> Chunk -> Embed -> Index. ถาม: เรียง 5 ขั้นได้ไหม?
4. **Chunking**: fixed-size 3 failure modes (context fragmentation, mid-table cut, cross-section bleed); structural: Chapter > Section > Paragraph > Sentence > Token และเก็บ header ของตาราง; semantic: dist = 1 - cos, ตัดที่ spike (95th percentile). ถาม: 3 failure modes?
5. **Embedding pathologies**: hubness = vector บางตัวโผล่เป็น neighbor ของ query ไม่เกี่ยวหลายตัว; anisotropy = กระจุกในพื้นที่แคบ cosine พอง; MiniLM truncate เงียบที่ 256 tok. ถาม: hubness ต่างจาก anisotropy อย่างไร?
6. **Sparse vs Dense เสริมกัน**: sparse พังที่ vocabulary mismatch ("heart attack" vs "myocardial infarction"), ชนะ id/code/version; dense พังที่ hallucinated relevance (python 3.10 vs 3.11), ชนะ paraphrase. ถาม: invoice #INV-2024-7832 ใครชนะ? (BM25)
7. **BM25 ทำไมดีกว่า TF-IDF** [H]: TF-IDF linear -> keyword stuffing; BM25 มี **saturation (k1)** + **length norm (b)**; ceiling = k1+1 ไม่ขึ้นกับ b; b แค่เปลี่ยน "ความเร็วเข้าหา ceiling"; doc สั้นคะแนนสูงกว่า (f เท่ากัน). ถาม: ซ้ำ keyword 1,000 ครั้งได้ 1,000x ไหม? (ไม่ ~2.5x ของ f=1)
8. **b ทิศทาง** (ตัวลวง): b = 0 ไม่สนความยาว, b = 1 penalty เต็ม, 0.75 มาตรฐาน. ถาม: b=1 หมายถึงอะไร?
9. **IDF**: ลบเมื่อ n > N/2, 0 ที่ N/2; Lucene ใช้ ln(1+...) (บวกเสมอ); rank_bm25 floor. ถาม: stopword มี IDF เท่าไร?
10. **ทำไมต้อง RRF**: BM25 (0,inf) กับ cosine [-1,1] คนละสเกล -> ใช้อันดับ; min-max ไวต่อ outlier; k = ตัวทำให้เส้นโค้งแบน (k=0: rank1->2 ลด 50%; k=60: ลด ~1.6%); **อยู่ 2 list > โด่ง list เดียว** ("Intersection > Union"); k=60 จาก Cormack 2009 ไม่ tune ใหม่. ถาม: k ใหญ่ขึ้น เส้นชันขึ้นหรือแบนขึ้น?
11. **Retriever = recall, Reranker = precision**; cross-encoder joint attention จับ negation แต่ precompute ไม่ได้ -> rerank เฉพาะ top K=20-100; score = raw logit (ติดลบได้ ไม่ใช่ probability); ColBERT = late interaction MaxSim อยู่ระหว่างกลาง. ถาม: Reranker เพิ่ม recall ไหม?
12. **Pipeline ลำดับ** [H weak]: BM25 || Dense -> RRF -> Cross-encoder -> LLM; ถ้า latency-sensitive ข้าม stage 3 แล้ว tune k. ถาม: ใครรันขนาน? (BM25 กับ Dense)
13. **CLIP**: contrastive image+text encoder (ViT/ResNet + Transformer) 400M คู่; เพิ่ม sim คู่ทแยง ลดนอกทแยง (symmetric: image->text + text->image, หาร 2N); zero-shot = template "A photo of a {x}" -> เลือก sim สูงสุด; **พลาด relation / attribute / negation**. ถาม: CLIP loss ต้อง symmetric เพราะอะไร?
14. **ViT**: แบ่งรูปเป็น patch -> linear projection + [class] token + position embedding -> Transformer; CLIP แทน MLP head ด้วย projection เข้า embedding space ร่วม. ถาม: ViT มอง "รูป" เป็นอะไร? (sequence ของ patch)
15. **Vector RAG พังตรงไหน**: global summarization / multi-hop เพราะ chunk เป็นจุดอิสระ ไม่เห็นความสัมพันธ์ -> Knowledge Graph. ถาม: ทำไม vector RAG ตอบ "themes ทั้ง corpus" ไม่ได้?
16. **GraphRAG** (Microsoft): offline = chunk -> entities/relations (triples) -> graph -> Leiden communities (modularity, ดีกว่า Louvain, hierarchical) -> community summaries; online = Map-Reduce บน summaries; **Local = entity traversal ~1 LLM call, Global = Map-Reduce แพงกว่ามาก**. ถาม: Global search ทำงานแบบไหน?
17. **Multi-hop 3 ขั้น**: query decomposition (Text-to-Cypher) -> deterministic traversal -> LLM synthesis; triple = (S, R, O); NER -> RE -> entity linking -> validation. ถาม: ทำไม cosine ทำ chain A->B->C ไม่ได้?
18. **Long-context trap**: แม้ 1M token ก็ต้อง RAG (cost O(N^2) + **Lost in the Middle** = U-curve); 4 กฎ: rerank first / bookend / tune K (more != better) / context distillation. ถาม: ตำแหน่งไหนที่ LLM ใช้ข้อมูลแย่สุด?
19. **Bookend**: rank1 หน้าสุด, rank2 ท้ายสุด, ที่เหลือกลาง; primacy + recency bias. ถาม: [1,2,3,4,5] bookend เป็นอะไร? ([1,3,4,5,2])
20. **ขยาย context**: SWA O(N^2) -> O(N w), KV rolling buffer คงที่; RoPE linear interpolation m' = m/s **บีบทุก frequency เท่ากัน -> คำใกล้ๆ เบลอ**; NTK-aware บีบเฉพาะ low freq คง high freq; YaRN 4K -> 128K ~400 steps. ถาม: ทำไม linear interpolation เสียลำดับคำใกล้เคียง?
21. **RAG vs FT vs Long-context** decision: knowledge เปลี่ยนบ่อย -> RAG; เปลี่ยน style/format -> FT; เอกสารเล็กคงที่ -> long-context; corpus ใหญ่ + relational -> GraphRAG (+ route ด้วย classifier). ถาม: ต้องการเปลี่ยน output style ใช้อะไร?
22. **demo-only**: BM25 จัด doc "myocardial infarction" ที่ 5, dense ที่ 1; BM25 score 0-1.893 vs dense 0.136-0.541; fused = union 14 docs; cross-encoder ดัน doc 11 กลับที่ 1 (logit -11.2..2.2); rank ใน `rrf_merge` 0-indexed เลยบวก 1. ถาม: ทำไม top-5 ของ BM25 ได้ "heart failure" นำ?

---

## W13 Agents / Constrained Decoding / Memory / Multi-agent — 20 บรรทัด
1. **Agent นิยาม**: LLM ควบคุม execution flow; LLM **ไม่ได้ฉลาดขึ้น** แต่ capable ขึ้นผ่าน tool. ถาม: agent กับ pipeline ต่างกันตรงไหน?
2. **4 components**: LLM brain / Tools / Memory / Planning (mnemonic ตั้งเอง: **B-T-M-P**). ถาม: เรียง 4 อย่าง
3. **ทำไมเกิน RAG/Reflection**: 4 ช่องว่าง = grounded verification, live state, model-chosen control flow, side effects. ถาม: RAG ขาดอะไร? (pipeline คงที่ retrieve ครั้งเดียว)
4. **ReAct loop** [H]: Thought -> Action -> Observation; **observation (ไม่ใช่ความจำโมเดล) นำทางขั้นต่อไป**; `max_steps` กัน loop; "มี loop ≠ agent" (ต้องมี policy เลือก action จาก feedback). ถาม: observation มาจากไหน?
5. **State**: s_t = (q, h_1..t-1) Markov บน augmented state; context โต linear -> ต้องเหลือ headroom ให้ตอบ; ถ้าบีบ history อาจ partially observable. ถาม: ทำไม context ต้องเก็บ history ทั้งหมด?
6. **ReAct vs CoT story** [H]: **ผลสลับตาม task** (HotpotQA ReAct แพ้, FEVER ชนะ); ReAct hallucinate 0% แต่ reasoning error สูง (47% วนซ้ำ) + search error; combo ReAct+CoT-SC ดีที่สุด. ถาม: ใครชนะบน FEVER? ทำไม ReAct hallucinate น้อย? (grounded ด้วย search)
7. **Tool ไม่ใช่ LLM รัน**: LLM output ข้อความ (name+args) -> parser validate -> dispatcher เรียกฟังก์ชัน -> ผลกลับเป็น **string Observation**; definition = ฟอร์มเปล่า (name, description, parameters), call = ฟอร์มที่กรอก. ถาม: ใครเรียกฟังก์ชันจริง?
8. **Error -> Observation ไม่ใช่ crash**: demo: ส่ง error กลับ โมเดลแก้เองรอบหน้า; ไม่ส่ง = วนจนหมด max_steps. ถาม: malformed action ควรทำอย่างไร?
9. **ทำไม constrain ที่ logit**: LLM generate ทีละ token ไม่มีอะไรการันตี JSON valid; regex parse/retry เปราะ; masking = **hard constraint** (-inf -> P=0 เป๊ะ ไม่ใช่ลดน้ำหนัก), ทำที่ **token id** ไม่ใช่ตัวอักษร, valid token ถูก renormalize. ถาม: mask คือ soft penalty หรือ hard?
10. **ความแรง mask ขึ้นกับ state**: หลัง `{` ตัด 99.86%, ใน string ตัดแค่ 111. ถาม: ทำไมต่างกัน?
11. **Syntax ≠ semantics**: `3194-81-59` ผ่าน regex `\d{4}-\d{2}-\d{2}` แต่เดือน 81; ต้อง validate schema/business แยก. ถาม: masking การันตีค่าถูกต้องไหม?
12. **Regex/DFA vs CFG**: nesting ลึกไม่จำกัด = ไม่ใช่ regular language ต้อง stack (CFG/pushdown); regex 1 transition = constant time (1 vocab scan ต่อ character class); 5 ขั้น CFG: state -> valid tokens -> mask -> sample -> update. ถาม: ทำไม JSON ซ้อนลึกต้อง CFG?
13. **Memory 4 ชนิด** (อุปมา): Working = สิ่งที่คิดอยู่ / Episodic = diary / Semantic = ตำรา / Procedural = ปั่นจักรยาน. ถาม: "Bangkok is capital" คือชนิดไหน?
14. **Ingestion vs runtime compilation**: pre-digest ตอน offline (cheat sheet) แล้วโหลดเฉพาะที่เกี่ยวตอน runtime — ไม่ยัด raw docs. ถาม: ทำไม "ทำ cheat sheet ก่อนเข้าห้องสอบ" ตรงกับ ingestion-time?
15. **Buffer / Summary / Retrieval**: O(n) ล้น / bounded แต่ lossy (เก็บ last k) / prompt O(k) แต่ **store ยัง O(n)**, อาจพลาดของสำคัญ; hybrid + RRF. ถาม: ใคร lossy?
16. **ทำไม multi-agent**: specialization / decomposition / parallelism แต่ trade-off messaging, latency, cost; ใช้เมื่อ decomposable จริง. ถาม: multi-agent ดีกว่าเสมอไหม?
17. **4 topologies** [H 4 มิติ: latency/complexity/robustness/best use]: Sequential (Σ T_i ไม่มี speedup, error cascade -> "มั่นใจแต่ผิด"), Routing (router = SPOF, metric = routing accuracy; LLM ฉลาดช้า / embedding argmax cos / rule เร็วสุด rigid), Supervisor (plan-assign-evaluate, self-correct, latency + token สูงมาก), Debate (parallel อิสระ + judge). mnemonic (ตั้งเอง): **S-R-S-D**. ถาม: จุดอ่อนของแต่ละอัน?
18. **ชื่อ topology ต่าง hint**: hint ใช้ sequential/routing/parallel/hierarchical; สไลด์ใช้ Sequential/Routing/Supervisor(hierarchical)/Debate (debate ≈ parallel). ถาม: "hierarchical" = อันไหนในสไลด์?
19. **Protocols 3 แบบ + design rule**: scratchpad (context bloat) / message passing (bounded) / structured payload; ต้องกำหนด ownership, expected output, **termination**. ถาม: กันวน loop ใน multi-agent ทำอย่างไร?
20. **Reliability stack + guardrails**: Constrain -> Validate -> Recover (retry เฉพาะ bounded + error feedback) -> Escalate (คนอนุมัติ); mnemonic (ตั้งเอง): **CVRE**; tool content = untrusted data; least privilege; eval = success rate / tool efficiency / cost+latency (WebArena, SWE-bench, GAIA). ถาม: ชั้นเดียวพอไหม?

---

## W14 Data engine / Evaluation — 22 บรรทัด
1. **บทบาท Data & QA Engineer**: Source -> Filter -> Dedup -> Evaluate -> Audit (mnemonic ตั้งเอง: **S-F-D-E-A**); architecture = สูตร, data = วัตถุดิบ. ถาม: 5 หน้าที่?
2. **Chinchilla**: compute-optimal = ขยายโมเดล **และ** token ไปพร้อมกัน ~20 tok/param; E = entropy ต่ำสุดที่ไม่มีโมเดลลดได้; GPT-3 300B tok = data-limited; เป็น empirical ไม่ใช่กฎสากล. ถาม: อะไรคือ E?
3. **Filter = precision-recall trade-off**; funnel: per-stage r_i = |D_i|/|D_{i-1}| vs cumulative |D_i|/|D_0|. ถาม: ต่างกันอย่างไร?
4. **ลำดับ pipeline**: LangID -> heuristic (ถูก ~0.1 ms) -> PPL -> **dedup (กลาง)** -> neural (~10 ms); dedup หลัง heuristic ไม่ต้อง dedup ขยะ, ก่อน neural ไม่จ่ายซ้ำสำเนา; cheap-first = ฮิวริสติก ไม่ใช่กฎ (ใช้ c_A + p_A c_B vs c_B + p_B c_A). ถาม: ทำไม dedup ก่อน neural classifier?
5. **Gopher vs C4 vs CCNet**: Gopher = rules **ผ่านทุกข้อ** (words, mean len, alpha, symbol, bullet, stopword, repetition); C4 = line/page rules + `{` + lorem + dedup 3-sentence (ลบ code/LaTeX!); CCNet = perplexity buckets. ถาม: ข้อไหนที่ทำให้ LaTeX/code หาย?
6. **ภาษาไทยกับ rule อังกฤษ**: "word" = whitespace token -> ต้อง segment + retune ทุก threshold; segmentation อย่างเดียวไม่พอ (stopword list). ถาม: Thai ตก Gopher ข้อไหนใน demo?
7. **Perplexity trap**: PPL วัด "คล้าย reference LM" **ไม่ใช่คุณภาพ**; spam/ข้อความซ้ำ PPL ต่ำ (รอด), เนื้อหาเทคนิคหายาก PPL สูง (ถูกทิ้ง); CCNet **ไม่มี cut-off** แบ่ง 3 bucket. ถาม: cookie banner ซ้ำๆ PPL สูงหรือต่ำ?
8. **FineWeb-Edu**: คุณภาพ > ปริมาณ; Llama-3-70B ให้คะแนน 0-5 ~460k ตัวอย่าง -> เทรน student ~110M (regression head) -> กรองทั้ง corpus ที่ >= 3; ได้ MMLU/ARC สูงกว่าที่ budget เท่ากัน. ถาม: ใครสร้าง label?
9. **Dedup ทำไม**: memorization (privacy), bias (ซ้ำบ่อย = น้ำหนักเกิน), compute. SHA-256 จับเฉพาะ byte-for-byte หลัง normalise (เปลี่ยน 1 ตัวอักษร hash ต่างหมด). ถาม: จับ near-duplicate ได้ไหม?
10. **Shingling**: k-word window ซ้อนทับ -> set ใหญ่ ~k x เอกสาร; ปัญหาจริงคือ **storage** (เก็บ shingle set ทุก doc) ไม่ใช่การคำนวณ intersection. ถาม: ปัญหาของการเทียบ shingle set ตรงๆ?
11. **MinHash**: ต่อ hash function เก็บค่า min; P(ตำแหน่งตรงกัน) = Jaccard; **ต้องใช้ชุด hash เดียวกันทั้งสองเอกสาร**; SE ลดด้วย **H ไม่ใช่ shingle**; p เป็น prime. ถาม: จะลด SE ทำอย่างไร?
12. **MinHash vs SimHash**: MinHash -> Jaccard (set overlap, H ตัวเลข); SimHash -> cosine/มุม (bit string สั้น, Hamming). ถาม: ใครประมาณ cosine?
13. **LSH**: แบ่ง signature เป็น b bands x r rows, ชน >=1 band = candidate; S-curve; s* ไม่ใช่ cutoff/ไม่ใช่ P=0.5; candidate **ต้อง verify**. ถาม: LSH candidate คือ duplicate แน่นอนไหม?
14. **2 แกนต้นทุน**: work = (#pairs) x (cost/pair); MinHash ลด cost/pair, LSH ลด #pairs (40B docs -> 8e20) -> **ซ้อนกัน ไม่แข่งกัน**; brute force = N^2 s^2 (ไม่ใช่ N^3). ถาม: ใครลดจำนวนคู่?
15. **Dedup ไม่แก้ repetition ในเอกสารเดียว**: cookie banner ซ้ำภายใน doc รอดทุกด่าน -> ต้อง Gopher repetition rules; "ไม่มี filter เดียวพอ". ถาม: MinHash จับ doc ที่ซ้ำตัวเองไหม?
16. **Tokenizer fairness**: ไทย 1 ตัวอักษร = 3 UTF-8 bytes; TPC = tokens / chars (ไม่ใช้ fertility เพราะไทยไม่มีขอบคำ); rho = TPC_TH/TPC_EN; BERT-base -> [UNK] = coverage failure (TPC ต่ำ **ไม่ใช่** ประสิทธิภาพ); ผลกระทบ = context + ค่า API. ถาม: ทำไม TPC ต่ำของ BERT ไม่ได้แปลว่าดี?
17. **3 ยุค eval**: <2020 string match (BLEU/ROUGE) / 2020-24 semantic + human preference (BERTScore, Arena, MT-Bench) / 2024-26 dynamic+capability-specific (LiveBench, SWE-bench, HarmBench, RAGAS). ถาม: BERTScore ยุคไหน?
18. **BLEU paradox / BERTScore**: BLEU เป็น lexical n-gram precision + BP -> word salad > paraphrase; BERTScore = cos matrix + greedy match (ไม่ one-to-one), **ไม่ใช่ %**, เทียบ ranking ไม่ใช่เลขดิบ, ไม่ใช่ fact-check. ถาม: BERTScore 0.79 > BLEU 0.17 แปลว่าอะไร?
19. **LLM-as-judge flaws**: length bias (ชอบยาว; GPT-4 ต้านดีกว่า), positional bias (ชอบมาก่อน) -> swap A/B (ชนะทั้งสองลำดับ), rubric anchoring, reference, repeated runs, judge diversity; inconsistent = ป้าย ไม่ใช่ tie. ถาม: ลำดับ swap แล้วไม่ตรงกันเรียกว่าอะไร?
20. **Arena vs MT-Bench**: Arena = คนจริง blind pairwise ไม่มี fixed test set แต่ช้า/ไม่ reproducible; MT-Bench = LLM judge 80 ข้อ เร็วแต่ bias. ถาม: ข้อเสียของ Arena?
21. **Benchmark**: saturation (ใกล้ ceiling แยกไม่ออก) vs contamination (รั่วใน training); 4 หลัก: Dynamic / Capability-specific / Ground-truth verifiable / Contamination-resistant (LiveBench ไม่รับประกันถาวร); 12% รั่ว **ไม่ใช่** 94.2-12; HarmBench ASR ต่ำ ≠ safe (ปฏิเสธทุกอย่าง ASR=0). ถาม: ทำไมคำนวณคะแนนไม่ปนเปื้อนจาก 12% ไม่ได้?
22. **RAGAS / pass@k**: faithfulness = claim ที่ context สนับสนุน / claim ทั้งหมด -> สูงแต่ตอบผิดได้ (retrieval ผิด, ถามไม่ตรง); reference-free; DeepEval = pytest/CI. pass@k = โอกาสที่ลอง k ครั้งมี >=1 ผ่าน (1 - โอกาสล้มทั้ง k). ถาม: faithfulness 0.95 แต่ user ว่าผิด เป็นไปได้ไหม?

---

# TOP 25 — ข้อเท็จจริงที่น่าจะออกสูงสุด (ทุกสัปดาห์)

> จัดอันดับจาก: ครูบอก "ออก" [H] > ปรากฏซ้ำใน mock ทั้ง 3 ชุด [M] > น้ำหนักสไลด์ [W]. คอลัมน์ "บน sheet" บอกว่าตัวเลขอยู่กล่องไหน (ส่วนเหตุผลต้องอยู่ในหัว).

| # | ข้อเท็จจริง | สัญญาณ | บน sheet |
|---|---|---|---|
| 1 | BART 5 noising + Deletion (ไม่มี placeholder) vs Infilling (1 [MASK], Poisson 3) | H ("ออกครบทุกเรื่อง") M | 1A |
| 2 | T5 span corruption: sentinel + target จบด้วย sentinel ถัดไป + `</s>` | H M | 1A |
| 3 | BART vs T5: target N vs 0.2N, 25x (decoder) vs 2-3x (ทั้งโมเดล), นับ src/tgt | H M | 1A, 1D |
| 4 | MLM 15% / CLM 100% (+ 80/10/10) | H M | 1A |
| 5 | KV cache + GQA: ratio = n_heads/n_kv (64/8 = 8x); สูตรมีตัวคูณ 2, ใช้ n_kv | H M | 1D |
| 6 | GB vs GiB: GB=1e9, GiB=2^30, GiB เลขเล็กกว่า (16 GB = 14.9 GiB) | H (เคยออกกลางภาค) M | 1D |
| 7 | SBERT head [u; v; abs(u-v)] = 3d = 2,304 | H M | 1B, 1D |
| 8 | Cross-encoder n(n-1)/2 forward (ms) vs bi-encoder n forward + n^2 dot (ns); ต้องรู้ว่า O() นับอะไร | H M | 1B, 1D |
| 9 | Triplet loss = max(0, d(a,p) - d(a,n) + margin) และเงื่อนไข L=0 | H (written) M | 1B, 1D |
| 10 | Absmax INT8: s=max(abs w)/127, 2.1 -> 83, deq 2.092, block-wise 64 | H M | 1D |
| 11 | BM25: TF ceiling = k1+1, b แค่เปลี่ยนความเร็ว, doc สั้นคะแนนสูงกว่า | H ("อาจให้ตัวเลข/%/เปรียบเทียบ") M | 2B, 1D |
| 12 | RRF = sum 1/(k+rank), k=60, ทำไมไม่รวมคะแนนดิบ | H(weak) M | 2B, 1D |
| 13 | Multi-agent topologies 4 มิติ: latency / complexity / robustness / best use | H ("ข้อสอบก็ออก") M | 2C |
| 14 | PPO ใช้ 4 models (Actor, Critic อัปเดต; RM, Ref frozen) vs DPO 2 models vs GRPO ตัด Critic | W (demo+lab) M | 2A |
| 15 | DPO loss: policy = ref -> ln2; beta ใหญ่ = ใกล้ ref; คำนวณ x = beta*(Δw - Δl) | W M | 2A, 1D |
| 16 | GRPO advantage = z-score ในกลุ่ม (หาร G), ผลรวม A = 0, ไม่มี Critic | W M | 2A, 1D |
| 17 | LoRA: 2dr แทน d^2, B=0, A~Gaussian, merge ไม่เพิ่ม latency; QLoRA = NF4 + Double Quant + Paged Opt | W M | 1C, 1D |
| 18 | MoE ประหยัด compute ไม่ประหยัด memory; DeepSeek = shared + fine-grained | H(weak) M | 1C |
| 19 | T5 relative position bucket: ใกล้ = exact, ไกล = แชร์, บวกเข้า logits (ไม่ต้องจำสูตร log) | H | 1A |
| 20 | Pre-Norm vs Post-Norm (original = post, modern RMSNorm ก่อน attention) | H (W7) | หัวเท่านั้น |
| 21 | ReAct: Thought-Action-Observation, Table 1 (ReAct แพ้ CoT บน HotpotQA แต่ชนะบน FEVER), max_steps | H M | 2C |
| 22 | Logit masking (-inf ก่อน softmax = hard), syntax ไม่ใช่ semantics, JSON ซ้อนต้อง CFG ไม่ใช่ regex, 69 valid / 50,188 masked | W M | 2C |
| 23 | Lost in the Middle (U-curve) + bookend + 4 กฎ production (rerank first, bookend, tune K, context distill) | H(weak) M | 2B |
| 24 | FlashAttention = exact (tiling+online softmax, ลด memory ไม่ลด FLOPs); PagedAttention = KV fragmentation; continuous batching = scheduling | W M | 1C |
| 25 | Dedup/Eval ชุดตัวเลข: MinHash (Jaccard, SE ลดด้วย H) + LSH (candidate ต้อง verify) + pass@k formula + judge length/position bias | W (14 มีสไลด์เยอะ) M | 2D, 1D |

หมายเหตุอันดับ: #25 รวม 4 เรื่องของ W14 เพราะ W14 ไม่มีคำว่า "ออก" จากครูแต่ได้ส่วนแบ่งเท่าสัปดาห์อื่น และ mock ทั้ง 3 ชุดถามเรื่อง MinHash/LSH/pass@k/judge; hint ระบุว่า MinHash/SHA-256 รายละเอียด "น่าจะไม่ออก" (กล่าวถึงใน W7) — จึงจัดอันดับท้ายสุด

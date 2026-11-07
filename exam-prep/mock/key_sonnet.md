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

## Answer key

- **Q1: B** — Token Deletion ลบ token โดยไม่เหลือ placeholder ($src < tgt$) ขณะที่ Text Infilling แทน span สุ่มจาก Poisson ด้วย `[MASK]` ตัวเดียว และ span ยาว 0 จะเป็นการ insert `[MASK]` บังคับให้โมเดลลบทิ้ง [Notes: 2.5, 4 (Demo 1), 7 (Traps #4, #5)]
- **Q2: D** — Target ของ T5 จะบรรจุเฉพาะ corrupted spans คั่นด้วย sentinel และต้องปิดท้ายด้วย sentinel ถัดไป (`<extra_id_N+1>`) เพื่อระบุว่าหมด span ที่หาย แล้วจึงตามด้วย `</s>` [Notes: 3.1, 4 (Demo 1), 7 (Traps #1, #2)]
- **Q3: A** — Decoder self-attn ratio = $N^2 / (0.2N)^2 = 1 / 0.04 = 25$ เท่า แต่ความเร็วรวมทั้งโมเดลเร็วขึ้นเพียง ~2-3 เท่า เพราะ Encoder ยังประมวลผล input ยาวใกล้เคียงเดิม [Notes: 3.4, 7 (Traps #7, #8)]
- **Q4: C** — Enc-Dec แบ่งพารามิเตอร์ $P$ ไปที่ Encoder $\sim P/2$ และ Decoder $\sim P/2$ ทำให้ตอน generate ทำงานจริงแค่ครึ่งเดียว ขณะที่ Dec-only ใช้เต็ม $P$ และเข้ากับ multi-turn RLHF ได้ดีกว่า [Notes: 2.1, 7 (Traps #22, #23, #24)]
- **Q5: A** — ใน Demo 5 เมื่อไม่ใส่ prefix ให้ T5 โมเดลไม่ได้คัดลอกหรือคืนค่าว่าง แต่ fallback ไปทำ translation ซึ่งเป็น task ที่พบบ่อยสุดใน pre-training แสดงว่า prefix คือ task switch [Notes: 4 (Demo 5), 7 (Trap #14)]
- **Q6: C** — Cross-Attention ใช้ Query จาก Decoder ข้ามไปมอง Key และ Value จาก Encoder โดยไม่ต้องใส่ Mask และ Attention Matrix มีขนาดตามแถวและคอลัมน์คือ $[tgt\_len, src\_len]$ [Notes: 2.1, 2.3, 3.8, 7 (Traps #19, #20)]
- **Q7: D** — เมื่อไม่มี context กลไก Index Pointing ของ BERT ถูกจำกัดโครงสร้างให้ออกนอกข้อความไม่ได้จึงไฮไลต์คำถามตัวเอง ขณะที่ T5 สร้าง token อิสระจึงพยายามตอบแม้จะเกิด hallucination [Notes: 4 (Demo 4), 7 (Traps #17, #18)]
- **Q8: B** — T5 ใช้ Relative Position Bias ขนาด 32 buckets ต่อ head (0-7 exact, 8+ log) โดยนำค่า bias สเกลาร์ไปบวกเข้ากับ logits ก่อนคำนวณ Softmax [Notes: 3.7, 7 (Traps #10, #11, #12)]
- **Q9: C** — Enc-Dec FLOPs $= N \times (1000^2 + 100^2 + 1000 \times 100) = 1,110,000 N$ ขณะที่ Dec-only FLOPs $= 2N \times (1100^2 / 2) = 1,210,000 N$ คิดเป็นอัตราส่วน $1,210,000 / 1,110,000 \approx 1.09$ เท่า [Notes: 3.6, 7 (Trap #27)]
- **Q10: D** — Enc-Dec ยังคงเป็นเลิศในงานที่ Input ขนาดยาวและต้องการ bidirectional encoding ร่วมกับ Output ที่สั้นกว่า เช่น NLLB (Translation) และ Whisper (Speech-to-text) [Notes: 1.4, 6 (#12), 7 (Trap #26)]
- **F1:** (1) Transfer, (2) sentinel token, (3) 0.20 (หรือ 20% / 0.2), (4) 25, (5) Text Infilling [Notes: 2.5, 3.4, 6 (#1, #2, #3, #4)]

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

- **Q21**: **D** — Model Weights เป็น static memory ที่คงที่ แต่ KV cache โตแบบ dynamic ตาม sequence length และ batch จนทำให้เกิด OOM กลาง generation (Notes Section 2.2, 7.21).
- **Q22**: **A** — Arithmetic: $M_{KV} = 2 \times 2 \times 32 \times 8 \times 128 \times 4096 \times 16 = 8,589,934,592\text{ B} \approx 8.6\text{ GB}$ ซึ่งลดลงจาก MHA ($n_{kv}=32$) ที่ใช้ $34.4\text{ GB}$ คิดเป็นประหยัดลง $(32-8)/32 = 75\%$ (Notes Section 3.2 (ค)).
- **Q23**: **B** — Prefill ประมวลผล prompt ขนานด้วย GEMM (compute-bound กำหนด TTFT) ส่วน Decode ทำทีละ token ด้วย GEMV ต้อง stream weights ทุกรอบ (memory-bandwidth bound กำหนด TPS) (Notes Section 2.4, 7.7).
- **Q24**: **D** — FlashAttention เป็น exact attention (ไม่ลด FLOPs) ที่ใช้ Tiling และ Online Softmax บน SRAM หลีกเลี่ยงการเขียน matrix $N \times N$ ลง HBM ลด extra memory เหลือ $O(N)$ (Notes Section 2.8, 7.4, 7.5).
- **Q25**: **C** — PagedAttention จัดการ KV cache แบบ paging ลด fragmentation ทำให้ KV memory utilization พุ่งจาก ~30% เป็น ~95% (Notes Section 6.13, 7.5, 7.8).
- **Q26**: **B** — MoE ลดเฉพาะ FLOPs ต่อ token (13B) แต่ยังต้องโหลดพารามิเตอร์ทั้งหมด (~47B หรือ ~94 GB) ลงใน VRAM เพราะไม่รู้ล่วงหน้าว่า token จะ route ไป expert ใด (Notes Section 6.11, 7.3).
- **Q27**: **A** — LoRA กำหนด $\Delta W = B \cdot A$ โดยเริ่ม $B=0$ และ $A$ เป็น Gaussian เพื่อให้ $\Delta W=0$ ตอนเริ่มเทรน (Notes Section 6.15, 7.10, 7.11).
- **Q28**: **C** — สัดส่วน 16N คือ Weights 2N + Grads 2N + Adam states 12N (75%) และ Gradient Checkpointing ลด activation เหลือ $O(\sqrt{L})$ แลกกับการเพิ่มเวลาเทรน ~30% (Notes Section 2.6, 7.14, 7.15).
- **Q29**: **B** — Arithmetic: $s = 3.2 / 127 \approx 0.0252$, $q = \text{round}(2.1 / 0.0252) = \text{round}(83.33) = 83$, $\hat{x} = 83 \times 0.0252 = 2.0916 \approx 2.092$ (Notes Section 3.6, 6.17).
- **Q30**: **C** — Quantization ลดเพียง memory bandwidth แต่ยังคำนวณครบ 12 layers เท่าเดิม มีเพียง Distillation ที่ตัด layers จึงลด compute (FLOPs) ได้จริง (Notes Section 4.4, 7.16).
- **F3**: **(1) NF4 (NormalFloat4), (2) Double Quantization, (3) Paged Optimizers** — เสาหลัก 3 ประการของ QLoRA ในการลดหน่วยความจำ base model, scale factor, และ optimizer memory spike (Notes Section 6.16, 7.11).

- **Q31: B** | คำนวณ: chosen ratio = $-12 - (-14) = +2$, rejected ratio = $-20 - (-18) = -2$, logits = $2 - (-2) = 4$, scaled = $0.1 \times 4 = 0.4$, DPO loss = $-\ln \sigma(0.4) \approx 0.5130$ (Notes §3.1)
- **Q32: D** | $\beta$ ใน DPO ทำหน้าที่เป็นตัวคูณควบคุม KL divergence constraint โดยค่า $\beta$ สูงหมายถึงการบังคับให้นโยบายอยู่ใกล้ Reference model มากขึ้น และ DPO ใช้โมเดลเพียง 2 ตัวโดยไม่ต้องมี RM (Notes §2, §4, §7 trap 15, 16)
- **Q33: B** | Reward Model แปลง LM head เป็น scalar regression head และเทรนด้วย Bradley-Terry preference loss ส่วนในขั้นตอน PPO จะอัปเดตเฉพาะ Actor และ Critic ขณะที่ RM และ Reference จะถูก freeze ไว้ (Notes §1, §3.5, §7 trap 12, 13)
- **Q34: A** | คำนวณ: $\mu = 1/4 = 0.25$, $\text{Var} = [(1-0.25)^2 + 3(0-0.25)^2]/4 = (0.5625 + 0.1875)/4 = 0.1875$, $\sigma = \sqrt{0.1875} \approx 0.4330$, ดังนั้น $A_1 = (1.0 - 0.25) / 0.4330 \approx +1.732$ (Notes §3.4)
- **Q35: D** | DeepSeek-R1 เปิดเผย reasoning trace ภายในแท็ก `<think>` ต่างจาก o-series ที่ซ่อนไว้ ส่วน R1-Zero ใช้ pure RL โดยไม่ผ่าน SFT และ test-time compute มีข้อจำกัดเรื่อง diminishing returns (Notes §1, §2.4, §6 candidate 15, §7 trap 18, 19)
- **Q36: C** | PoT/PAL แยกการคิดเชิงเหตุผล (LLM เขียนโค้ด) ออกจากการคำนวณตัวเลข (ส่งให้ external Python interpreter) เพื่อแก้ปัญหา arithmetic error ของ LLM (Notes §2.8, §7 trap 4)
- **Q37: A** | Self-Consistency เป็นเทคนิค unsupervised ที่ต้องตั้ง Temperature > 0 เพื่อ sample ความคิดหลายเส้นทางอย่างอิสระ แล้วใช้ Majority vote ตัดสินคำตอบ โดยไม่ต้องเทรน verifier เพิ่ม (Notes §2.7, §7 trap 3)
- **Q38: C** | SFT ใช้ Teacher forcing ตอนเทรนแต่ใช้ Autoregressive decoding ตอน inference ก่อให้เกิด Exposure bias ส่วน Base และ Instruct model มีจำนวนพารามิเตอร์เท่ากันและต้องพึ่งพา Chat template (Notes §1, §4, §7 trap 8, 9, 10, 21)
- **Q39: D** | คำสั่งอันตรายที่แฝงมากับข้อมูลภายนอกที่โมเดลดึงมาจัดเป็น Indirect Prompt Injection ซึ่งสามารถบรรเทาได้ด้วย Sandwich Defense และ Delimiters (Notes §2.12, §7 trap 7)
- **Q40: A** | Trace ที่ยาวขึ้นทำให้ Attention compute รวมเพิ่มแบบ Quadratic ($O(N^2)$) และ KV cache เพิ่มแบบ Linear ($O(N)$) โดยสามารถจัดการ fragmentation ได้ด้วย PagedAttention (Notes §2.5, §3.6, §7 trap 20)
- **F4: Audience** | กรอบ CO-STAR ย่อมาจาก Context, Objective, Style, Tone, Audience, Response Format (Notes §2.2, §6 candidate 2)

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

- **Q51**: **C** — LLM ไม่ได้เรียกฟังก์ชันตรงๆ แต่ generate ข้อความระบุชื่อ tool/args เพื่อให้ dispatcher นำไปประมวลผลจริง และ agent ต้องมี policy เลือก action จาก observation ไม่ใช่แค่มี loop [Section 4 Section 1, Section 7 Trap 1-2]
- **Q52**: **A** — ReAct ไม่ได้ชนะทุกชุดข้อมูล โดยแพ้ CoT บน HotpotQA (27.4 vs 29.4) แต่ชนะบน FEVER (60.9 vs 56.3) และ failure ของ ReAct มาจาก reasoning/search error ไม่ใช่ hallucination 56% ซึ่งเป็นของ CoT [Section 1 Hints, Section 3.6, Section 7 Trap 3-4]
- **Q53**: **D** — การทดลอง Error recovery ชี้ชัดว่าต้องส่ง error message กลับไปเป็น Observation เพื่อให้โมเดลเรียนรู้และแก้ไขเองในรอบถัดไป หากสั่ง crash โมเดลจะไม่สามารถกู้คืนสถานะได้ [Section 4 Section 1.6]
- **Q54**: **B** — คำนวณจาก Logits หลัง mask: $e^{2.0} = 7.389$, $e^{0.5} = 1.6487$; ผลรวม $7.389 + 1.6487 = 9.0377$; จะได้ $P(\text{id1}) = 7.389 / 9.0377 = 0.8176$ [Section 3.1]
- **Q55**: **A** — Masking บังคับได้เฉพาะ Syntax ไม่ได้รับประกัน Semantics จึงเกิดวันที่ที่เป็นไปไม่ได้ในทางความหมาย และการ parse โครงสร้าง nest ลึกไม่จำกัดต้องใช้ CFG (Pushdown Automaton) ไม่ใช่ Regex (DFA) [Section 4 Section 2, Section 6 item 9, Section 7 Trap 8]
- **Q56**: **C** — Buffer เพิ่ม token แบบ Linear $O(n)$, Summary มีพฤติกรรมเป็นฟันปลา (Sawtooth) เพราะขนาดลดลงเมื่อเกิดการบีบอัด, และ Retrieval มีขนาดใน prompt คงที่ $O(k)$ ตามจำนวนท็อปปิกที่ดึงมา [Section 3.7, Section 4 Section 3, Section 7 Trap 6]
- **Q57**: **D** — Sequential Latency คำนวณจาก $\sum T_i = 3 + 10 + 4 = 17$ วินาที และมีจุดอ่อนสำคัญคือ Error cascade เมื่อ agent ตัวแรกคืน error string ตัวถัดไปจะนำไปประมวลผลต่อจนผิดพลาดทั้งหมด [Section 3.4, Section 4 Section 4, Section 6 item 12]
- **Q58**: **B** — Rule-based matching ตรวจสอบตามลำดับกฎ เมื่อพบคำว่า 'premium'/'plan' จึงจับคู่กับ Sales ทันที ทำให้ส่งผิดแผนก และหาก Router ตัดสินใจผิด Specialist จะไม่สามารถแก้ไขได้ ถือเป็น Single Point of Failure [Section 4 Section 4, Section 6 item 13]
- **Q59**: **C** — State ของ ReAct ถือเป็น Markov บน Augmented State $s_t = (q, h_{1:t-1})$ และ Taxonomy 4 แบบของ Memory ได้แก่ Working, Episodic, Semantic, และ Procedural [Section 6 item 5 & 10, Section 7 Trap 5]
- **Q60**: **A** — Reliability Stack 4 ชั้นเรียงลำดับจาก Constrain $\rightarrow$ Validate $\rightarrow$ Recover $\rightarrow$ Escalate และ safeguard พื้นฐานกัน infinite loop ใน ReAct คือตัวแปร `max_steps` [Section 1 Hints, Section 6 item 4 & 14]
- **F7**: **$-\infty$** (หรือ **-inf**), **0** (หรือ **0%** / **0.0**) — การเซ็ต logit เป็น $-\infty$ ทำให้ $e^{-\infty} = 0$ ส่งผลให้ probability ของ token ที่ผิดกฎกลายเป็น 0 พอดี [Section 3.1, Section 6 item 7]
- **F8**: **27.4**, **60.9** — ผลการทดลอง Table 1 ของ ReAct บน HotpotQA ได้ 27.4 (แพ้ CoT 29.4) และบน FEVER ได้ 60.9 (ชนะ CoT 56.3) [Section 1 Hints, Section 3.6, Section 6 item 3]

- **Q61**: C — LSH banding กรองคู่เพื่อลดการเปรียบเทียบจาก $O(N^2)$ แต่เอกสารที่ผ่านยังเป็นเพียง candidate ที่ต้อง verify ด้วย exact Jaccard อีกครั้ง (Section 3.3, 4, 7 trap 5 & 7).
- **Q62**: A — MinHash/LSH ทำงานเปรียบเทียบระหว่างคู่เอกสาร (pairwise) จึงจับการซ้ำในเอกสารเดียวไม่ได้ ต้องใช้ repetition heuristics ของ Gopher เข้ามาช่วย (Section 4 dedup pipeline, 7 trap 10).
- **Q63**: C — ข้อความซ้ำซาก/cookie spam มี PPL ต่ำมากเพราะทำนายง่าย ขณะที่ศัพท์เฉพาะทางอาจตกไปอยู่ tail และ CCNet แบ่ง 3 bucket ตามขนาดข้อมูลโดยไม่มี fixed cut-off (Section 4 perplexity demo, 7 trap 1 & 2).
- **Q64**: B — ต้นทุนรัน A ก่อน B คือ $c_A + p_A c_B = 0.1 + 0.3(10) = 3.1\text{ ms/doc}$ ซึ่งต่ำกว่ารัน B ก่อน A คือ $10 + 0.5(0.1) = 10.05\text{ ms/doc}$ (Section 3.7, 7 trap 11).
- **Q65**: B — TPC ที่ต่ำมากของ BERT เกิดจาก coverage failure ที่ไม่มีคำศัพท์ภาษาไทยใน vocab ทำให้กลายเป็น `[UNK]` ทั้งหมด ไม่ใช่การบีบอัดที่ดี (Section 3.5, 4, 7 trap 14).
- **Q66**: D — BLEU อิง n-gram surface matching จึงให้คะแนน word salad สูง ส่วน BERTScore แก้ไขโดยเทียบ contextual embeddings ผ่าน greedy matching บน cosine similarity matrix (Section 3.9, 4, 7 trap 18 & 19).
- **Q67**: C — คำนวณจากสูตรกรณีพิเศษ $c = 1$: $pass@k = k/n = 100/200 = 0.500$ (Section 3.1, 7 trap 23).
- **Q68**: A — SWE-bench ตัดสินว่า resolved เมื่อผ่านเทสต์ชุด fail-to-pass และยังคงผ่านชุด pass-to-pass ครบถ้วน (Section 4 pass@k SWE-bench, 6 item 16, 7 trap 24 & 25).
- **Q69**: D — LLM Judge มักมี length bias และ positional bias ซึ่งการทำ swap test ช่วยลดอคติลงได้แต่ไม่สามารถกำจัดให้หมดสิ้น (Section 3.10 mock judge, 7 trap 21 & 22).
- **Q70**: B — RAGAS Faithfulness เป็น reference-free metric ที่วัดการ entailment ระหว่าง claims ของคำตอบกับ context เท่านั้น โดยไม่ได้รับประกัน factual truth ของ context (Section 4 RAGAS proxy, 7 trap 27).
- **F9**: 20, 3.5T (หรือ 3.5 ล้านล้าน) — อัตราส่วน Chinchilla คือ 20 tokens ต่อ parameter โดย $175\text{B} \times 20 = 3,500\text{B} = 3.5\text{T tokens}$ (Section 3.4, 6 item 1).
- **F10**: SimHash, MinHash — SimHash รักษา Cosine similarity ผ่าน Hamming distance ส่วน MinHash ประมาณค่า Jaccard similarity (Section 6 item 10, 7 trap 4).

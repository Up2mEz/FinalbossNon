# Week 09 — Sentence Embeddings & Semantic Search (SBERT, Contrastive Loss, FAISS)

แหล่งข้อมูล: `slides_w09.txt` (49 สไลด์, เลขหน้า = เลขสไลด์ตามที่ตรวจจาก form-feed แล้ว), `demo_w09.txt`, `nlp_exam_hints.md`.
Tag: **[MEMORIZE]** = ต้องอยู่ในหัว (เหตุผล/สัญชาตญาณ) | **[SHEET]** = ควรลอกลง A4 (สูตร ค่าคงที่ ตาราง mnemonic) | `(นอกสไลด์)` = ความรู้ทั่วไปของผู้เขียนโน้ต ไม่ได้อยู่ในสไลด์

---

## 1. Exam weight และลำดับหัวข้อ

**น้ำหนัก**: brief ระบุ 10 MCQ + ข้อเติมคำสัปดาห์ละประมาณ 1-2 ข้อ (จากทั้งหมด 10 ข้อ, สัปดาห์ 7-14).
หมายเหตุ: ถ้าแบ่งเท่ากัน 8 สัปดาห์ (7-14) จะได้ MCQ ≈ 8.75/สัปดาห์; ใช้ 10 ตาม brief แต่ไม่ควรแปลกใจถ้าเหลือ 8-9.
hints ระบุว่า **"Written part" ของ Week 9 มีเรื่อง Triplet/Margin loss และตัวเลข (ระวังทศนิยม/เลขยกกำลัง 10)** และ "บางข้อเลขง่าย แค่บวกคูณ" -> ข้อเติมคำสัปดาห์นี้น่าจะเป็นตัวเลข/การคำนวณ.

**ลำดับหัวข้อตามโอกาสออก** (สไลด์เยอะ + ครูเน้น = มาก):

| อันดับ | หัวข้อ | สไลด์ | เหตุผล |
|---|---|---|---|
| 1 | SBERT architecture: Siamese, shared weights, pooling, forward pass, **head 2304 = 768x3**, NLI classification / STS regression | 13-17, 22-23, 33 | hints "strong signal" (2,304) + สไลด์เยอะ |
| 2 | **Contrastive losses**: Triplet (สูตร, margin, zero-loss condition), hard/semi-hard mining, Margin Ranking, MNRL/InfoNCE, ตารางเทียบ | 26-32 | hints (written part) + 7 สไลด์ติดกัน |
| 3 | **Bi-encoder vs Cross-encoder**: ความซับซ้อน, 69 ชม. vs 50 วินาที, retrieve-then-rerank | 7-8, 18 | hints "strong signal" (know what the O counts) |
| 4 | **Anisotropy**: raw BERT ล้มเหลวเพราะอะไร, [CLS] vs mean, ตัวเลข STS, 3 fixes | 4-6, 9, 12, 19 | สไลด์ + demo เยอะ, Key Takeaway #1 |
| 5 | **Vector search**: FAISS (Flat/IP), normalization, ANN (IVF/HNSW/PQ), vector DB, IP vs L2 | 36, 38-44 | สไลด์เยอะ (9 สไลด์) + demo section 7 |
| 6 | **Metrics**: cosine/dot/L2/cosine distance, Recall@k, MRR | 10-11, 37 | สูตรสั้น, ออกคำนวณง่าย |
| 7 | SimCSE (unsup/sup), วิวัฒนาการ E5/BGE, แหล่งข้อมูลฝึก | 24, 34-35 | ตัวเลขจำได้ (76.3, 81.6) |
| 8 | Pipeline offline/online, latency, ความเชื่อม Week 12 | 43, 48 | น่าจะออก 0-1 ข้อ |

---

## 2. Core concepts

### 2.1 โจทย์และปัญหาของ raw BERT (สไลด์ 2-7)
- ต้องการฟังก์ชัน f: sentence -> R^d (fixed-size dense vector) ที่ **sim(f(s1), f(s2)) ≈ semantic_similarity(s1,s2)** (สไลด์ 3). [MEMORIZE]
- การใช้งาน 5 แบบ (สไลด์ 3): Semantic Search, Duplicate Detection (Quora), Clustering, Recommendation, RAG.
- Week 6 ใช้ BERT + [CLS] + linear head เพื่อ classify; แต่เทียบ 2 ประโยค/ค้นหาต้องมี sentence vector ที่ดี ซึ่ง BERT ดิบให้ไม่ดี (สไลด์ 2).
- Naive 2 วิธี (สไลด์ 4): (a) mean pooling s = (1/T) sum_t H_t (H = BERT(tokens) ∈ [B,T,d], d=768); (b) ใช้ [CLS] = H_0.
- **ผลช็อก**: ทั้งสองแย่กว่า GloVe average บน STS Benchmark (Spearman rho x100): [SHEET]

| Method | STS (สไลด์ 4) |
|---|---|
| GloVe average | 58.02 |
| BERT [CLS] (no fine-tuning) | **16.50** |
| BERT mean pooling (no fine-tuning) | 46.35 |
| SBERT-NLI-base | **77.03** |

- **Anisotropy** (สไลด์ 5-6, Ethayarajh 2019): embedding ไม่กระจายทุกทิศทางสม่ำเสมอ; cosine เฉลี่ยของคำสุ่มสองคำ **ไม่เป็น 0** ในเกือบทุกชั้นของ BERT/ELMo/GPT-2; **ชั้นบนยิ่ง anisotropic กว่าชั้นล่าง**; ข้อยกเว้นคือ input layer ของ ELMo (ใช้ character-level, ไม่มี context). [MEMORIZE]
- สาเหตุ/ผลต่อเนื่อง (สไลด์ 6): มีไม่กี่ dimension ที่สัมพันธ์กับ token frequency ครอบงำ space; คำอย่าง "the, a, is" ดันทุก embedding ไปทิศเดียวกัน; **ตัด dominant dimensions ออกช่วยให้ cosine ดีขึ้น** (Mu & Viswanath 2018). Mean pooling **สืบทอด** ปัญหา -> sentence vector ก็ยุบเป็น cone; [CLS] ถูกฝึกด้วย **NSP** ไม่ใช่ semantic similarity -> แย่กว่าอีก. [MEMORIZE]
- **สาเหตุที่แท้จริง (Key Takeaway #1, สไลด์ 9/47)**: ไม่ใช่ architecture ของ BERT แต่คือ **training objective (MLM) ไม่เคยกระตุ้นให้ geometry ระดับประโยคดี**. [MEMORIZE]
- อาการ: vectors อยู่ใน "narrow cone" -> **ไม่มีคู่ไหนได้คะแนนต่ำ** -> cosine ไม่มีที่ว่างให้จัดอันดับ (สไลด์ 47).

### 2.2 สามวิธีแก้ (สไลด์ 9) [SHEET]
| วิธี | กลไก | คุณภาพ | หมายเหตุ |
|---|---|---|---|
| 1. Post-processing | ตัด dominant dims (all-but-the-top, whitening) | Moderate | เร็ว, "Band-aid, not principled" |
| 2. **Contrastive fine-tuning** | Siamese/Dual-Encoder + contrastive loss | High | **Industry standard (SBERT)** = โฟกัสสัปดาห์นี้ |
| 3. Unsupervised **SimCSE** | ใช้ dropout เป็น data augmentation | Good | ไม่ต้องมี label, strong baseline |

### 2.3 ทำไม Cross-encoder ช้าเกินไป (สไลด์ 7-8)
- Cross-encoder: ต่อ `[CLS] A [SEP] B [SEP]` -> BERT -> [CLS] -> classification head -> similarity score. คุณภาพเยี่ยม (full cross-attention ระหว่าง A,B). ความซับซ้อน **O(L^2) ต่อคู่** (L = ความยาวรวม).
- เทียบทุกคู่ของ n ประโยค = **C(n,2) = n(n-1)/2 forward passes**.
- **Bi-encoder**: encode แต่ละประโยคอิสระ s_A=f(A), s_B=f(B); sim = cos(s_A,s_B). ใช้ **n forward passes + O(n^2 · d) dot products** (ถูกมาก). [MEMORIZE]
- ตัวเลขสไลด์: n=10,000 -> 10,000x9,999/2 ≈ **50 ล้านคู่**; ที่ 5 ms/inference ≈ **~69 ชั่วโมง** (GPU เดียว); bi-encoder ≈ **~50 วินาที** encode + **<1 วินาที** เทียบทุกคู่. (ดู 3.2)

### 2.4 Cosine และ metric อื่น (สไลด์ 10) [SHEET]
- cos(u,v) = u·v / (‖u‖‖v‖) = Σ u_i v_i / (√Σu_i² √Σv_i²) ∈ [-1,1]  (1=ทิศเดียวกัน, 0=orthogonal, -1=ตรงข้าม)
- Cosine distance d_cos = 1 - cos ∈ [0,2]
- L2 distance d_L2 = ‖u-v‖₂ = √Σ(u_i - v_i)²
- Dot product u·v = Σ u_i v_i = ‖u‖‖v‖ cos(u,v)  -> **dot = cosine x ขนาด**; ถ้า **L2-normalize** (unit vectors) แล้ว **dot = cosine** และทั้ง 3 metric **จัดอันดับเท่ากัน** (ranking equivalent). [MEMORIZE]

### 2.5 Siamese Networks และสถาปัตยกรรม SBERT (สไลด์ 13-14)
- Siamese: เครือข่ายเหมือนกันสองสาย **architecture เดียวกัน weights เดียวกัน (weight sharing / tied weights)** ประมวลผล input คนละตัว; update พร้อมกันเหมือนกัน; เป้าหมายคือฉายเข้า shared embedding space ที่ระยะ/ความคล้าย "มีความหมาย". e_A=f(x_A), e_B=f(x_B), **f ตัวเดียวกัน**. ต้นกำเนิด: signature verification (Bromley 1993; Chopra 2005); ต่อมา face recognition (FaceNet), one-shot learning. [MEMORIZE]
- **SBERT** (Reimers & Gurevych, 2019 / EMNLP-IJCNLP): pre-trained BERT + Siamese fine-tuning บนข้อมูลคู่ประโยค. 3 ส่วน: (1) Shared BERT encoder (2) Pooling layer (variable-length -> fixed-size) (3) Similarity/Loss function.
- 3 Configurations (สไลด์ 14) [SHEET]:

| Config | Input | Loss / Head | ข้อมูลฝึก |
|---|---|---|---|
| Classification | (sA, sB) + label (entailment/neutral/contradiction) | **Softmax** บน [u; v; \|u-v\|] | NLI |
| Regression | (sA, sB) + similarity score | **MSE** บน cos(u,v) | STS |
| Triplet | (anchor, positive, negative) | Triplet margin loss | retrieval training |

### 2.6 Pooling (สไลด์ 15) [SHEET]
แปลง `[batch, seq_len, d_model]` -> `[batch, d_model]`
| กลยุทธ์ | สูตร | หมายเหตุ |
|---|---|---|
| CLS | s = H_0 | ฝึกด้วย NSP -> signal แย่ (RoBERTa ablation พิสูจน์ว่า NSP ไม่ช่วย) |
| **Mean (SBERT default, best)** | s = Σ_t H_t m_t / Σ_t m_t, m_t ∈ {0,1} | m_t คือ attention mask: ตัด `[PAD]` ก่อนรวม; **ผลดีสุด** ใน SBERT ablation (Table 6, Spearman) |
| Max | s_i = max_t H_{t,i} (element-wise ต่อ dimension) | จับ peak activation |

### 2.7 Forward pass แบบ tensor (สไลด์ 17) [SHEET]
ตัวอย่าง A="A man is eating food" (7 tokens), B="A man is eating pasta" (8 tokens):
1. Tokenize แยกกัน: input_ids_A [1,7], input_ids_B [1,8] (+ attention_mask)
2. BERT (shared θ รันสองรอบ): H_A [1,7,768], H_B [1,8,768]
3. Mean pool: u [1,768], v [1,768]
4. Concatenate **c = [u; v; |u-v|] ∈ [B, 2304]** (768+768+768)
5. Classify: W ∈ R^{2304x3} -> logits [1,3] -> CrossEntropy(logits, label)
**จุดสำคัญ**: ใช้ pair vector [u;v;|u-v|] แทน [CLS] เดี่ยวใน Week 6 (สไลด์ 22). ใน general: c ∈ R^{3d}.

### 2.8 Bi-Encoder vs Cross-Encoder (สไลด์ 18) [SHEET]
| Property | Bi-Encoder (SBERT) | Cross-Encoder |
|---|---|---|
| Input | encode อิสระ | concat แล้ว encode พร้อมกัน |
| Architecture | 2 BERT towers -> pool -> เทียบ vectors | BERT เดียว -> [CLS] -> score |
| Cross-sentence attention | **ไม่มี** ระหว่าง encode | **มีเต็ม** |
| Output | vectors u,v -> similarity | scalar score เดียว |
| Quality | Good | **Excellent** |
| Speed | O(n) forward + O(n^2·d) dot products | **O(n^2) forward passes** |
| Pre-compute/cache vectors | ได้ | ไม่ได้ (รันซ้ำทุกคู่) |
| Use case | **Stage 1 retrieval (recall)** | **Stage 2 re-ranking (precision)** |

**Retrieve-then-Rerank**: Stage 1 bi-encoder เอา top-k จากหลายล้านเอกสาร (เร็ว/โดยประมาณ) -> Stage 2 cross-encoder ให้คะแนนใหม่ top-k (แม่น). [MEMORIZE]

### 2.9 SBERT แก้ anisotropy อย่างไร (สไลด์ 19)
- ก่อน: cone แคบ, avg pairwise cosine **~0.58**, cos(s_i,s_j) ∈ **[0.46, 0.85]** mean 0.58 — ทุกคู่ "เท่าๆ กัน".
- หลัง: กระจายทั่ว hypersphere, cosine ∈ **[0.1, 0.99]** แยกแยะได้.
- กลไก **Attract + Repel**: contrastive loss ดึง positive pairs เข้าหากัน, ผลัก negative pairs ออก, บังคับใช้พื้นที่มุมเต็ม hypersphere.
- Wang & Isola (2020): **Alignment** (positives อยู่ใกล้) + **Uniformity** (embeddings กระจายสม่ำเสมอ/รักษาข้อมูลสูงสุด). [MEMORIZE]
- STS: GloVe 58.2 -> BERT mean-pool 46.4 -> SBERT 77.03 (สไลด์ 19 ปัดต่างจากสไลด์ 4 นิดหน่อย).

### 2.10 sentence-transformers library (สไลด์ 20-21) [SHEET]
- `SentenceTransformer('model-name').encode(sentences)` -> fixed-size dense vectors 2 บรรทัด. (Nils Reimers, UKP Lab; wrap HF transformers + pooling + (ตัวเลือก) L2-normalize.)
- `.encode()` ทำตามลำดับ: (1) Tokenize (2) transformer forward (3) **masked mean pooling** (4) L2-normalize (**off by default**) -> คืน numpy `[batch, d_model]`.
- Manual 5 ขั้น (สไลด์ 21): Load BERT -> Tokenize -> Forward -> Masked Mean Pool -> **L2 Normalize `F.normalize(embedding, p=2, dim=1)`** -> unit vectors; หลังนี้ dot = cosine.

| Model | Dim | Speed | Quality (STS) | Best for |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | **384** | Fast | Good | Prototyping, on-device |
| all-mpnet-base-v2 | **768** | Medium | High | General production |
| BAAI/bge-large-en-v1.5 | **1024** | Slower | Very High | High-precision retrieval |

### 2.11 Training data และ pipeline (สไลด์ 22-23, 33, 35)
- **NLI**: SNLI (570K) + MultiNLI (430K) ≈ **1M pairs**, 3 labels: Entailment / Neutral / Contradiction. ตัวอย่าง: "A man is eating pasta." + "A man is eating food." = Entailment (ดึงใกล้); + "A man is sleeping." = Contradiction (ผลักไกล); + "The man is in Italy." = Neutral. ทำไมได้ผล: label สอนเรื่อง "คล้าย=ใกล้, ต่าง=ไกล" **โดยไม่ต้องมีเป้า cosine ตรงๆ**.
- **STS Benchmark (STSb)**: ~**8,500** pairs, score **0-5** (เช่น "A plane is taking off."/"An air plane is taking off." = 5.0; "A man is playing a flute."/"A man is surfing." = 0.0). Regression: ŝ = cos(u,v) ∈ [-1,1]; ground truth s ∈ [0,5] **normalize เป็น [0,1]**; Loss = MSE(ŝ, s).
- **กลยุทธ์สองเฟส** (Reimers & Gurevych): Phase 1 = NLI + classification (โครงหยาบ) -> Phase 2 = STS + regression (ปรับความละเอียด) -> ผลดีสุด. [MEMORIZE]
- Training loop (สไลด์ 33): โหลด BERT-base + mean pooling -> ข้อมูล SNLI+MNLI -> forward shared BERT -> pool -> [u;v;|u-v|] -> 3-way softmax -> CE -> **backprop ผ่านทุกชั้น BERT** (ไม่ใช่แค่ pooling; ทำให้ geometry เปลี่ยน = ยา anisotropy). **Hyperparameters [SHEET]**: batch **16**, lr **2x10^-5** (Adam + warmup), **1 epoch**, max seq **128**, mean pooling. เวลา: ~**20 นาที** (NLI) + ~**5 นาที** (STS) บน **V100** เดียว. สูตรสมัยใหม่ (2026): **AllNLI triplets + MNRL**.
- ตารางข้อมูล (สไลด์ 35): [SHEET]

| Source | Size | Type |
|---|---|---|
| SNLI + MultiNLI | ~1M pairs | Entailment labels |
| STS Benchmark | ~8.5K | Similarity score 0-5 |
| MS MARCO | ~500K+ | Query -> Passage |
| Quora Duplicate | ~400K | Duplicate / Not |
| AllNLI | ~1M | SNLI + MNLI รวม |
| Web-mined (E5, BGE) | 100M-1B+ | Weak supervision |

  Quality vs Scale: **Gold** (human: NLI, STS) คุณภาพสูงสุด/เล็ก -> **Silver** (curated: MS MARCO) -> **Weak** (web-mined) คุณภาพแปรผัน/ระดับพันล้าน. Recipe ปัจจุบัน: pre-train บน weak ขนาดมหึมา แล้ว fine-tune บน gold.

### 2.12 SimCSE (สไลด์ 24) — Gao et al., 2021
- **Unsupervised**: ป้อนประโยคเดียวกันเข้า BERT **2 ครั้ง ด้วย dropout mask ต่างกัน** -> ได้คู่ positive (ประโยคเดิม, ตัวเลขต่างเล็กน้อย); ประโยคอื่นใน batch = negatives. ได้ ~**76.3** บน STS **โดยไม่ใช้ label** (สูงกว่า SBERT-base **74.89**).
- **Supervised**: ใช้ NLI dataset เดิม แต่ contrastive loss; **Positive = entailment** (x_i, x_i+); **Hard negative = contradiction** (x_i, x_i-). ได้ ~**81.6** (เหนือ SBERT-base 74.89).
- ใช้ loss เดียวกับ MNRL/InfoNCE (สไลด์ 30).

### 2.13 Contrastive Learning framework (สไลด์ 26-31)
- ปรัชญา: positive pairs ใกล้, negative pairs ไกล; เป็น **paradigm** ไม่ใช่ loss เดียว. 3 ส่วนประกอบ: **Anchor** x_a, **Positive** x_p, **Negative** x_n; encoder f. เป้า: d(f(x_a),f(x_p)) < d(f(x_a),f(x_n)).

**Triplet Loss (FaceNet, Schroff 2015; สไลด์ 27)** [SHEET]
  L = max(0, d(a,p) − d(a,n) + α)
  - α = margin (ปกติ **0.5-1.0**); d = Euclidean หรือ cosine distance. SBERT triplet experiments ใช้ **Euclidean, ε(margin) = 1**.
  - **Zero-loss**: ถ้า d(a,p) + α ≤ d(a,n) (positive ใกล้กว่า negative อย่างน้อย α) -> L=0 ไม่ update. ถ้าละเมิด -> L>0 -> gradient ดึง p เข้า ผลัก n ออก.
  - บทบาท margin: ไม่มี margin โมเดลพอใจกับช่องห่างจิ๋ว (lazy learning); α บังคับระยะห่างขั้นต่ำ; α ใหญ่ = เข้มกว่า geometry ดีกว่า แต่ฝึกยากกว่า.

**Hard Negative Mining (สไลด์ 28)** [SHEET]
  - Random negative มักง่าย (ไกล anchor อยู่แล้ว) -> L=0 -> ไม่เรียนรู้อะไร.
  - **Easy**: d(a,n) ≥ d(a,p) + α -> L=0 (ไร้ประโยชน์)
  - **Hard**: d(a,n) < d(a,p) -> gradient ใหญ่ แต่ฝึกไม่เสถียรได้
  - **Semi-hard**: d(a,p) < d(a,n) < d(a,p) + α -> **สมดุลสุด** (FaceNet ใช้)

**Margin Ranking Loss (สไลด์ 29)** [SHEET]
  L = max(0, sim(a,n) − sim(a,p) + α)  (PyTorch `MarginRankingLoss`)
  - Triplet ใช้ **distance** (min d_pos, max d_neg); Margin Ranking ใช้ **similarity** (max sim_pos, min sim_neg).
  - **เทียบเท่ากันทางคณิตศาสตร์เมื่อ d = 1 − sim** (cosine distance) — เทอม (1−x) ตัดกัน.

**MNRL / InfoNCE (สไลด์ 30)** [SHEET]
  L_i = − log [ exp(sim(q_i,p_i)/τ) / Σ_{j=1..N} exp(sim(q_i,p_j)/τ) ]
  - Batch ของ N คู่ (q_i,p_i); p_j (j≠i) เป็น **negative** ของ q_i (**in-batch negatives**, "ฟรี" ไม่ต้อง mine). ใช้ใน E5, BGE, GTE.
  - **τ (temperature) ปกติ 0.05-0.1**; τ ต่ำ = decision boundary คมขึ้น/hard.
  - Batch N=256 -> แต่ละ query มี **255** in-batch negatives.
  - SimCSE ใช้ loss เดียวกัน (positive จาก dropout).

**ตารางเทียบ Loss (สไลด์ 31)** [SHEET]
| Loss | Input | Negatives | Margin | Use |
|---|---|---|---|---|
| Softmax Cross-Entropy | (sA,sB,label) | Implicit (classes) | N/A | NLI classification |
| MSE | (sA,sB,score) | N/A | N/A | STS regression |
| Triplet | (a,p,n) | Explicit | α (อนุมานจากการสกัดข้อความ) | Retrieval |
| Margin Ranking | (a,p,n) | Explicit | α | Retrieval |
| **MNRL/InfoNCE** | (q,p) batch | **In-batch** | τ (temp) | Modern retrieval, **ไม่ต้อง explicit neg.** |
| Cosine Similarity Loss | (sA,sB,y ∈ {-1,1}) | Implicit (label) | N/A | binary similar/dissimilar |
Takeaway: MNRL เป็นที่นิยมปี 2026 เพราะ **data format ง่ายสุด (แค่ positive pairs)** และ scale ตาม batch size.

### 2.14 วิวัฒนาการ (สไลด์ 34)
2019 **SBERT** (NLI+STS supervised) -> 2021 **SimCSE** (unsupervised dropout) -> 2022 **E5** (weakly-supervised web-scale (query,passage) จาก CommonCrawl; "Embeddings from bidirectional Encoder representations") -> 2023 **BGE** (BAAI; + instruction tuning "Represent this sentence: ...") -> 2024-26 **Instruction-based embeddings** (embedding ต่างกันตาม intent). โมเดลสมัยใหม่ดีกว่า SBERT เพราะฝึกบน (query, document) pairs จากเว็บมหาศาล.

### 2.15 Vector search และ ANN (สไลด์ 36-44)
- ปัญหา scale (สไลด์ 36): **1M docs x 768 dims x 4 bytes ≈ 3 GB**; brute-force O(n·d) ต่อ query ≈ **7.7 วินาที สำหรับ 10M docs ที่ 1 GHz** -> ช้าเกินไป.
- **ANN**: ใช้ index prune search space, แลกความแม่นเล็กน้อยกับความเร็วมหาศาล: ปกติ recall **99%+** ที่เร็วขึ้น **100x**. Algorithms: **FAISS** (Facebook), **HNSW**, **Annoy** (Spotify), **ScaNN** (Google).
- Vector DB: Standalone = Pinecone, Weaviate, Qdrant, Milvus, Chroma; Embedded = pgvector (PostgreSQL), SQLite-VSS; Library = FAISS, ScaNN (คุม index/hardware เอง).
- **FAISS metric** (สไลด์ 38): `IndexFlatL2` = L2 distance (squared, **ต่ำ=ดี**); `IndexFlatIP` = Inner Product. **หลัง L2-normalize เองแล้ว** -> IP = cosine -> **เลือก `IndexFlatIP` เสมอ**: คะแนนที่ได้คือ cosine ∈[-1,1] **สูง=ดี** อ่าน/ตั้ง threshold ง่ายกว่า; dot product เป็น operation ฮาร์ดแวร์เดียว (เร็ว). (สไลด์ย้ำ: "don't assume the model did it").
- **Exact vs ANN** (สไลด์ 39) [SHEET]: Exact = brute-force กับ n vectors ทั้งหมด, 100% ถูก, O(n·d), impractical เมื่อ n>100K. ANN = recall >95%, เวลา "empirically sublinear" ~ **O(log n · d) กับ HNSW** (บน data ที่ดี). **กฎ**: ANN คือ **accuracy-latency trade-off; เทียบ speed โดยไม่รายงาน Recall@k ถือว่าไม่ถูกต้อง**.

| Index | Method | Recall | Best for |
|---|---|---|---|
| Flat | Brute-force (exact) | **100%** | corpus เล็ก (<100K) |
| IVF | Cluster partitioning | ~95% | corpus ใหญ่, memory ต่ำ |
| HNSW | Graph-based | ~**99%** | **production ส่วนใหญ่** |
| IVF+PQ | Cluster + compression | ~90% | **billion-scale**, เร็ว, RAM ต่ำ |
  (คอลัมน์ Query time ในไฟล์ txt สกัดมาเพี้ยน ไม่แน่ใจสัญลักษณ์ O(.) ของ IVF/IVF+PQ จึงไม่ยืนยัน; ที่แน่ใจ: Flat=O(n·d), HNSW≈O(log n·d))
  Guideline: corpus **<1M -> HNSW**; **>1B -> IVF+PQ** (memory-constrained). FAISS: GPU-accelerated, รองรับทุก index.
- **HNSW** (สไลด์ 40) Hierarchical Navigable Small World: multi-layer graph; **top layers = sparse entry points** กระโดดไกล; **Layer 0 = dense graph มีทุกจุด**. ค้น coarse-to-fine: เข้า top layer -> **greedy walk** ไปเพื่อนบ้านใกล้สุด -> **drop down** ชั้นล่าง ทำซ้ำจนถึง layer 0. ประสิทธิภาพเป็น empirical sublinear ขึ้นกับการสร้าง graph, `efSearch`, data, dimension -> ต้องวัด Recall@k+latency. ใช้เป็น default ใน Qdrant, Weaviate, Pinecone (recall 99%+).
- **FAISS 5 ขั้น (สไลด์ 41)** [SHEET]: (1) encode -> `[n,d]` **float32** (2) `faiss.normalize_L2(vectors)` (**critical**: ถ้าข้าม การค้นจะถูกครอบงำด้วย magnitude ไม่ใช่ความหมาย) (3) `faiss.IndexFlatIP(d)` (4) `index.add(vectors)` (5) encode+normalize query แบบเดียวกัน -> `D, I = index.search(query, k=5)`: **D = similarity scores `[1,k]`, I = document indices `[1,k]`**.
- **เลือก metric (สไลด์ 42)** [SHEET]: Semantic Search -> Cosine; Document Retrieval -> Dot Product (หลัง norm, เร็วสุด); Image Similarity -> Cosine (CLIP เป็นทิศ); **Anomaly Detection -> L2** (ระยะสัมบูรณ์สำคัญ); RAG Retrieval -> Cosine (มาตรฐาน SBERT/E5/BGE). **Common mistake**: ใช้ L2 กับ embedding ที่ไม่ normalize -> vectors ขนาดต่างกันดูไม่คล้ายแม้ชี้ทิศเดียวกัน.
- **Pipeline (สไลด์ 43)**: OFFLINE (ครั้งเดียว): load docs -> encode [n,d] -> L2-normalize -> build index (`add`) -> save to disk. ONLINE (ต่อ query): รับ query -> encode [1,d] -> normalize -> `index.search(q, k=10)` -> return docs[I] เรียงตาม D. **Latency**: query encode ~5-10 ms (GPU)/~50 ms (CPU); search <1 ms (exact, 100K) / <5 ms (ANN, 10M); รวม **<20 ms end-to-end บน GPU**.

### 2.16 Metrics การประเมิน retrieval (สไลด์ 37) [SHEET]
- **Recall@k** = |Retrieved_k ∩ Relevant| / |Relevant|  (coverage)
- **MRR** = (1/|Q|) Σ_{i=1..|Q|} 1/rank_i  (rank_i = อันดับของเอกสารถูกอันแรกของ query i; rank 1 -> 1.0, rank 2 -> 0.5)
- ใช้ตรวจว่า ANN ไม่ทำให้ความแม่นยำตก.

### 2.17 Bridge to Week 12 (สไลด์ 48)
Dense retrieval (bi-encoder + FAISS = ผลงานวันนี้) -> Sparse BM25 -> merge ด้วย **RRF (Reciprocal Rank Fusion)** -> **Cross-encoder re-rank** -> LLM generation. Weeks 10-11: PEFT/LoRA/Quantization, RLHF/DPO.

---

## 3. Worked calculations

### 3.1 [SHEET] หัว classification ของ SBERT (hints: "อาจถามแบบนี้")
c = [u; v; |u−v|] -> มิติ = 3d.
- BERT-base d=768 -> **768x3 = 2,304**. ตรวจ: 768x3 = 2,304 ✓ (700x3=2,100 + 68x3=204).
- W ∈ R^{2304x3}: พารามิเตอร์ = 2304x3 = **6,912** (+3 bias = 6,915) (นอกสไลด์).
- ถ้า d=384 (MiniLM) -> 1,152; d=1024 (BGE-large) -> 3,072 (นอกสไลด์ ใช้หลักเดียวกัน).
- ถ้าแค่ concat (u,v) = 1,536 (ระวังกับดัก: ไม่ใช่ SBERT).

### 3.2 [SHEET] Cross-encoder vs Bi-encoder เวลา
สูตร: คู่ = n(n−1)/2.
- n=10,000: 10,000 x 9,999 / 2 = 99,990,000 / 2 = **49,995,000 ≈ 5.0x10^7 (50 ล้าน)**.
- เวลา = 49,995,000 x 5 ms = 249,975,000 ms = **249,975 s** ÷ 3,600 = **69.44 ชม.** (≈ 69 ชม. ตามสไลด์) ✓ ตรวจซ้ำ: 5x10^7 x 5x10^-3 s = 2.5x10^5 s; 2.5x10^5/3600 ≈ 69.4 ✓.
- Bi-encoder: 10,000 forward x 5 ms = 50,000 ms = **50 s**; dot products 5x10^7 ครั้งในระดับ ns -> <1 s.
- ตัวอย่างเล็ก: n=1,000 -> 499,500 คู่ x 5 ms = 2,497.5 s ≈ 41.6 นาที; bi-encoder = 1,000 x 5 ms = 5 s.
- ⚠ **ขัดกับ hints**: hints (Whisper) บอก bi-encoder "~5 seconds"; **สไลด์ 8 ระบุ ~50 วินาที** (n=10,000) -> ยึดสไลด์. (5 s เป็นเลขของ n=1,000 ตามคำนวณข้างบน อาจเป็นต้นเหตุสับสน.)

### 3.3 Memory ของ vector (สไลด์ 36, 44) [SHEET]
bytes = n x d x 4 (float32).
- 1M x 768 x 4 = 768x10^6 x 4 = 3.072x10^9 B = **3.072 GB ≈ 3 GB** (≈ 2.86 GiB; 1 GiB=1.0737 GB).
- **5M x 768 x 4 (กิจกรรมสไลด์ 44)** = 3.84x10^9 x 4 = 1.536x10^10 B = **15.36 GB** (≈ 14.3 GiB) -> พอใน RAM 32 GB (แต่เหลือที่ให้ index overhead/OS น้อย; HNSW ต้องการ graph เพิ่ม — นอกสไลด์).
- 100,000 x 768 x 4 = 307.2 MB.
- Brute-force เวลา 10M x 768 = 7.68x10^9 ops / 10^9 = **7.68 s** (≈ 7.7 s ✓).
- ระวัง GB (10^9) vs GiB (2^30) ตาม hints Week 10.

### 3.4 Cosine บน toy vectors (กิจกรรม สไลด์ 11) — คำนวณเอง
s1=[0.91,0.38,0.15] ("The cat sat on the mat."), s2=[0.89,0.41,0.19] ("I love machine learning."), s3=[0.90,0.39,0.17] ("Dogs chase cats in the park.").
- ‖s1‖² = 0.8281+0.1444+0.0225 = 0.9950; ‖s2‖² = 0.7921+0.1681+0.0361 = 0.9963; ‖s3‖² = 0.8100+0.1521+0.0289 = 0.9910.
- s1·s2 = 0.8099+0.1558+0.0285 = 0.9942 -> cos = 0.9942/√(0.9950x0.9963)= 0.9942/0.99565 = **0.9985**.
- s1·s3 = 0.8190+0.1482+0.0255 = 0.9927 -> cos = 0.9927/√(0.9950x0.9910) = 0.9927/0.99300 = **0.9997**.
- s2·s3 = 0.8010+0.1599+0.0323 = 0.9932 -> cos = 0.9932/0.99365 = **0.9995**.
- Takeaway: ทั้งหมด ≈ 0.999 แม้ s1/s2 ไม่เกี่ยวกัน (0.9985 แทบเท่า s1/s3 ที่เกี่ยว 0.9997) -> cone แคบ cosine แยกไม่ได้.

### 3.5 Triplet Loss (สไลด์ 32) [SHEET] — ตรวจเลขซ้ำแล้ว
a=[0.3,0.7,0.1], p=[0.35,0.65,0.15], n=[0.8,0.2,0.5], α=0.5.
- a−p = (−0.05, 0.05, −0.05) -> d² = 0.0025x3 = 0.0075 -> **d(a,p) = 0.0866 ≈ 0.087**.
- a−n = (−0.5, 0.5, −0.4) -> d² = 0.25+0.25+0.16 = 0.66 -> **d(a,n) = 0.8124 ≈ 0.812**.
- L = max(0, 0.087 − 0.812 + 0.5) = max(0, **−0.225**) = **0** (easy negative; ไม่ update). เช็ค: 0.087+0.5 = 0.587 ≤ 0.812 ✓ zero-loss condition.
- Hard negative n'=[0.32,0.68,0.12]: a−n' = (−0.02,0.02,−0.02) -> d² = 0.0012 -> d = **0.0346 ≈ 0.035**. L = 0.087 − 0.035 + 0.5 = **0.552** > 0 -> gradient.
- **ตัวแปรที่ครูอาจเปลี่ยน**: ถ้า margin = 1.0 กับ negative ง่ายเดิม: 0.0866 − 0.8124 + 1.0 = **0.2742 > 0** (margin ใหญ่ขึ้นทำให้ negative เดิม "ไม่ง่ายพอ" อีกต่อไป). ถ้า margin=0.5: ต้อง d(a,n) ≥ 0.087+0.5 = 0.587 ถึงจะ L=0.
- ระวังทศนิยม: 0.0075 ไม่ใช่ 0.075; รากของ 0.66 ≈ 0.81 ไม่ใช่ 8.1.

### 3.6 Margin Ranking vs Triplet (demo section 6) — ตรวจซ้ำ
anchor=[1,1], positive=[1.2,0.9], negative=[3,−1], margin=0.5.
- cos(a,p) = (1.2+0.9)/(√2 · √2.25) = 2.1/(1.41421x1.5) = 2.1/2.12132 = **0.9899** ✓ (ตรงกับ demo).
- cos(a,n) = (3−1)/(√2 · √10) = 2/√20 = 2/4.4721 = **0.4472** ✓.
- Margin Ranking L = max(0, 0.4472 − 0.9899 + 0.5) = max(0, −0.0427) = **0**.
- Triplet (cosine distance): d(a,p)=1−0.9899=0.0101; d(a,n)=1−0.4472=0.5528; 0.0101−0.5528+0.5 = −0.0427 -> **0 เท่ากันเป๊ะ** (เทอม 1 ตัดกัน) ✓.
- Triplet (Euclidean, margin=1): d(a,p)=√0.05=0.2236; d(a,n)=√8=2.8284; slack = 0.2236−2.8284+1 = **−1.6048** (loss 0 เช่นกัน แต่ slack ต่างจาก cosine -0.0427 -> **metric เป็นส่วนหนึ่งของ loss**). (ค่า slack Euclidean คำนวณเอง demo พิมพ์ไม่ครบ)
- Hard negative [1.05,0.95]: cos = 2.0/(1.41421x1.41598)= 2.0/2.00249 = 0.99875 -> L = 0.99875−0.98995+0.5 = **≈ 0.509** (>0, gradient fires; ค่าเต็มถูกตัดใน demo output, นี่คือคำนวณเอง).

### 3.7 MNRL ตัวอย่างเลข (นอกสไลด์: เลขสมมติเพื่อฝึกสูตร)
q_i มี sim กับ p_i=0.8 (positive) และ in-batch negatives 0.2, 0.1 (N=3).
- τ=1: exp = 2.2255, 1.2214, 1.1052; Σ=4.5521; p=2.2255/4.5521=0.4889; L = −ln 0.4889 = **0.716**.
- τ=0.1: sim/τ = 8, 2, 1; exp = 2980.96, 7.389, 2.718; Σ=2991.07; p=0.99662; L = **0.0034** -> τ ต่ำทำให้โมเดล "มั่นใจ" มากขึ้นเมื่อ positive นำ.
- ถ้า positive ไม่นำ (sim_pos ต่ำกว่า negatives) L จะใหญ่ (−log ของ prob เล็ก).

### 3.8 Recall@k และ MRR (สไลด์ 37) — เลขสมมติ [SHEET]
- Recall@5: query มี relevant 3 ฉบับ; top-5 มี 2 ฉบับ -> 2/3 = **0.667**.
- MRR: 3 queries, rank ของคำตอบถูกอันแรก = 1, 2, 4 -> (1 + 0.5 + 0.25)/3 = 1.75/3 = **0.5833**. (ถ้า query ไหนไม่พบเลย 1/rank นับเป็น 0 — นอกสไลด์.)

### 3.9 Normalization และ metric equivalence (นอกสไลด์: ตัวอย่างฝึก)
- u=[3,4], v=[4,3]: ‖u‖=‖v‖=5; u·v = 12+12 = 24; cos = 24/25 = **0.96**; cos distance = **0.04**. หลัง normalize u'=[0.6,0.8], v'=[0.8,0.6]: u'·v' = 0.48+0.48 = **0.96** = cosine ✓.
- u=[3,4], v=[6,8] (ทิศเดียวกัน): cos = 1; dot = 50 = 5x10x1; **แต่ L2 = ‖u−v‖ = 5 ≠ 0** -> นี่คือเหตุผล common mistake "L2 บน non-normalized".
- (นอกสไลด์) บน unit vectors: ‖u−v‖² = 2 − 2cos(u,v) -> ทำไม L2 ranking = cosine ranking; ตัวอย่าง cos=0.96 -> dist²=0.08.

### 3.10 Mean pooling ด้วย mask (demo section 3) — ฝึกมือ
Token embeddings (2 dims): H = [[2,4],[4,8],[100,100]] mask m=[1,1,0] (token ที่ 3 คือ [PAD]).
- Σ H_t m_t = [2+4+0, 4+8+0] = [6,12]; Σ m_t = 2 -> s = **[3,6]**.
- ถ้าไม่ mask: เฉลี่ยทั้งสาม = [106/3, 112/3] = [35.3, 37.3] -> ถูกปน [PAD] (เลขสมมติให้เห็นผลกระทบ).

---

## 4. Demo-only content (ไม่อยู่ในสไลด์/แสดงใน notebook) — ออกได้ ~1-2 ข้อ

1. **Section 2: [CLS] ของ raw BERT กลับอันดับ** (`bert-base-uncased`). A="The cat sat on the mat." B="I love training deep neural networks." (ไม่เกี่ยว) C="Dogs are chasing cats in the park." (เกี่ยว). cos(A,B)=**0.8610** > cos(A,C)=**0.7942** -> **ranking inverted**. CLS shape [3,768]. สไลด์ 12 พูดเลขนี้เหมือนกัน (0.86 vs 0.79). บทเรียน: ดู **ลำดับ** และ **ค่าต่ำสุด (floor)** ไม่ใช่แค่ขนาดตัวเลข; 2 คู่ไม่พอ -> ต้องวัดทั้ง distribution.
2. **Section 2b: วัด cone บน 12 ประโยค = 66 คู่ (12x11/2)** [SHEET]:

| Encoder | mean | min | max | range |
|---|---|---|---|---|
| raw BERT [CLS] | +0.768 | +0.561 | +0.942 | 0.380 |
| raw BERT mean-pooled | +0.580 | +0.457 | +0.848 | 0.391 |
| SBERT (all-MiniLM-L6-v2) | +0.057 | **−0.156** | +0.556 | 0.712 |
   บทเรียน: ดูคอลัมน์ **MIN**: raw BERT ไม่เคยต่ำกว่า ≈0.457 (ไม่มีอะไร "dissimilar"); SBERT ลงถึง −0.156; raw BERT บีบ 66 คู่ในแถบกว้าง ~0.39 vs SBERT ~0.71. cone ≠ "ทุกค่า 0.99" แต่คือ "ไม่มีค่าต่ำ". (mean 0.580 ตรงกับสไลด์ 19 "mean 0.58, min ~0.46".) **[CLS] แย่กว่า mean ในแง่ mean cosine (0.768 vs 0.580)**.
3. **Section 3: Mean pooling ด้วย attention_mask**: `mask.unsqueeze(-1).expand(token_embeddings.size()).float()` -> คูณ -> `sum(dim=1)` -> หารด้วย `clamp(mask.sum(1), min=1e-9)` (กันหารด้วยศูนย์). ตัวอย่างโทเค็น: `['[CLS]','the','cat','sat','on','the','mat','.','[SEP]','[PAD]']` mask `[1,1,1,1,1,1,1,1,1,0]`; mean-pooled shape [3,768]. **ข้อเท็จจริงเด่น**: `[PAD]` output **ไม่ใช่เวกเตอร์ใกล้ศูนย์**: mean L2 norm ของ real tokens = **14.90**, ของ [PAD] = **9.67** (ยังใหญ่) เพราะ PAD ยังมี position embedding, ผ่าน FFN/residual และ attend ไปหา real tokens (padding ถูก mask เฉพาะในฐานะ key) -> เฉลี่ยรวมจะ **corrupt** เวกเตอร์ ไม่ใช่แค่ทำให้เล็กลง.
4. **Verification**: `SentenceTransformer('bert-base-uncased')` (ไม่มี pooling module -> library สร้าง mean pooling ให้อัตโนมัติ) ให้ผลตรงกับ manual mean pooling ถึง floating-point (`torch.allclose(..., atol=1e-4)`; ค่าต่าง ~1e-7, ไม่ bit-exact เพราะลำดับคำนวณต่าง). 5 dim แรก: −0.1818, −0.2662, −0.2189, 0.2109, 0.2847. "Library has no magic".
5. **Section 4: all-MiniLM-L6-v2** (dim **384**, ฝึกบน 1B+ sentence pairs): "A man is eating food." vs "A man is eating pasta." = **0.7035** (สูง); vs "The girl is carrying a baby." = **−0.1050** (ต่ำ, ติดลบได้!). output shape (3,384). `encode(..., normalize_embeddings=True)` ต้องสั่งเอง.
6. **Section 5: 5 ขั้น ≠ SBERT เอง**: "These 5 steps are SBERT's *architecture*, not SBERT itself"; pooling+normalize raw BERT **ไม่แก้ anisotropy**; **contrastive fine-tuning** ต่างหากที่แก้. Step 5 normalize: norm ก่อน = [10.21, 8.63, 10.50] -> หลัง ≈ [1.0, 1.0, 1.0]; ค่า 5 dim แรกถูกหารด้วย norm (−0.1818 -> −0.0178 หาร ≈10.21).
7. **Section 6**: `nn.MarginRankingLoss(margin)` ใช้ `loss_fn(sim_pos, sim_neg, target)` โดย **target = +1** (= x1 ควรอยู่เหนือ x2) -> สูตรภายใน max(0, −y·(x1−x2)+margin) = max(0, sim_neg − sim_pos + margin). `nn.TripletMarginLoss(margin=1.0, p=2)` = Euclidean. Cosine-triplet == Margin Ranking (พิสูจน์ด้วยการตัด 1−x); Euclidean ต่างกัน. Hard negative ทำให้ loss ≠ 0 -> "easy negative: loss 0 เรียนรู้ไม่ได้อะไร" -> เหตุผลของ hard/semi-hard mining.
8. **Section 7 (FAISS)**: corpus 8 ประโยค, query "What is a fast library for vector similarity search?" `k=3`. **all-MiniLM-L6-v2 ส่ง unit vectors อยู่แล้ว** (Normalize module ในโมเดล) — norms ก่อน/หลัง = 1.0; **แต่เป็นสมบัติของ checkpoint นี้ ไม่ใช่ของ `.encode()`**: `normalize_embeddings` default = **False**; โมเดลอย่าง BGE/E5 ผ่าน raw transformers **ไม่ normalize ให้**. raw-BERT mean-pooled norms ≈ [10.12, 9.97, 9.84, 8.94] (ยาวต่างกัน) -> `faiss.normalize_L2` -> 1.0. ถ้าข้าม: `IndexFlatIP` จะ **จัดอันดับตามความยาว ไม่ใช่ความหมาย**. **สรุป: normalize ชัดเจนเสมอ**.
9. **`IndexHNSWFlat` ไม่ใช่ drop-in แทน `IndexFlatIP`**: ต้องระบุ M (graph connectivity) เช่น `faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)`; **default เป็น L2** -> D คือ **squared distance ต่ำ=ดี** (กลับกับ cosine score ที่สูง=ดี). HNSW sublinear เชิง empirical ขึ้นกับ M, efSearch, data -> วัด Recall@k และ latency.
10. (นอกสไลด์) log "UNEXPECTED keys cls.seq_relationship / cls.predictions" ตอนโหลด `AutoModel` = head ของ NSP/MLM ที่ไม่ได้ใช้ถูกทิ้ง (ปกติ) — ไม่ใช่ error.

---

## 5. Tag สรุป — [MEMORIZE] (เข้าใจ/เหตุผล)

- ทำไม raw BERT พัง: **MLM objective** ไม่ optimize sentence geometry -> anisotropy (narrow cone, ไม่มีค่า cosine ต่ำ). ไม่ใช่ปัญหา architecture. [CLS] ถูกฝึก NSP.
- Mean pooling (ที่มี attention mask) ดีกว่า CLS; แต่ pooling อย่างเดียวไม่แก้ cone — ต้อง **contrastive fine-tuning ทั้ง BERT**.
- Siamese = **shared weights**, f ตัวเดียวกัน, encode อิสระ -> cache ได้ -> scale ได้.
- Bi-encoder = recall/เร็ว (stage 1); Cross-encoder = precision/ช้า (stage 2, rerank). O ของ cross = จำนวน **forward passes** ที่เป็น n^2; ของ bi = n forward + n^2 **dot products** (ถูกกว่าระดับ ms vs ns).
- Contrastive = attract positive + repel negative -> alignment + uniformity.
- Triplet loss = 0 เมื่อ positive ใกล้กว่า negative อย่างน้อย margin; easy negative ไร้ประโยชน์ -> semi-hard ดีสุด.
- MNRL: in-batch negatives ฟรี, batch ใหญ่ = negative เยอะ, τ ต่ำ = คม; ต้องการแค่ positive pairs.
- Normalize -> dot = cosine -> ใช้ `IndexFlatIP`; ข้าม normalize = จัดอันดับตามความยาว.
- ANN = trade-off accuracy/latency; ต้องรายงาน Recall@k คู่กับความเร็ว.
- NLI (classification) สอน geometry โดยนัย; STS (regression) ปรับละเอียด; สองเฟส = ดีสุด.

---

## 6. Fill-in-the-blank candidates (12 ข้อ)

1. SBERT classification objective ต่อ vector เป็น c = [u; v; |u−v|] มี **2,304** มิติ (768x3 สำหรับ BERT-base) แล้วส่งเข้า linear layer + **softmax**.
2. Cross-encoder เทียบทุกคู่ของ n ประโยคต้อง **n(n−1)/2** forward passes; n=10,000 ≈ **50 ล้าน** คู่ ≈ **69 ชั่วโมง** ที่ 5 ms/คู่.
3. Bi-encoder ใช้ **n** forward passes ตามด้วย **O(n²·d)** dot products; n=10,000 ≈ **50 วินาที**.
4. Siamese network ใช้ **shared (tied) weights** — โครงสร้างและพารามิเตอร์เดียวกันทั้งสองสาย.
5. SBERT pooling default คือ **mean pooling** โดยใช้ **attention mask** ตัด `[PAD]`.
6. Triplet loss: L = max(0, d(a,p) − d(a,n) + **α**) โดย α = **margin** (ปกติ 0.5-1.0); semi-hard: d(a,p) < d(a,n) < d(a,p)+α.
7. Margin Ranking Loss เทียบเท่า Triplet loss เมื่อ d = **1 − sim** (cosine distance).
8. MNRL/InfoNCE มี **temperature τ** ปกติ **0.05-0.1**; batch 256 -> **255** in-batch negatives.
9. Raw BERT [CLS] บน STS = **16.50**, BERT mean pooling = **46.35**, GloVe average = **58.02**, SBERT-NLI-base = **77.03**.
10. Unsupervised SimCSE ใช้ **dropout** เป็น data augmentation (ได้ ~76.3); supervised SimCSE ใช้ **entailment** เป็น positive และ **contradiction** เป็น hard negative (~81.6).
11. NLI = SNLI (570K) + MultiNLI (430K) ≈ **1M** pairs; STS Benchmark ≈ **8.5K** pairs, score **0-5**.
12. `IndexFlatIP` คืนคะแนน **cosine similarity** (สูง=ดี) เมื่อ vectors ถูก **L2-normalize** แล้ว; `IndexFlatL2` คืน squared distance (ต่ำ=ดี).
13. **HNSW** = Hierarchical Navigable Small World; layer 0 มี **ทุกจุด**; recall ~**99%**; (IVF+PQ ~90%, billion-scale).
14. **Recall@k** = |Retrieved_k ∩ Relevant| / |Relevant|; **MRR** = (1/|Q|) Σ 1/rank_i.
15. 1M x 768 x 4 bytes ≈ **3 GB** (3.072 GB); 5M ≈ **15.36 GB**.

---

## 7. Common MCQ traps / misconceptions

1. **"Mean pooling raw BERT ก็พอแก้ anisotropy"** ผิด: demo 2b ยัง mean=0.58, min=0.457. ต้อง contrastive fine-tuning. (distractor: "เพราะ BERT ใหญ่ไม่พอ" / "ต้อง layer มากขึ้น" — สาเหตุคือ MLM objective)
2. **"[CLS] ดีที่สุดเพราะฝึกมาเพื่อ sentence"** ผิด: [CLS] ฝึก NSP, STS แค่ 16.50 (แย่สุด); SBERT ใช้ mean.
3. **"Raw BERT ทำงานหนักกว่า GloVe"**: ในเชิงผล raw BERT (CLS 16.50, mean 46.35) **แย่กว่า** GloVe avg 58.02.
4. **"Cross-encoder ดีกว่าจึงใช้ค้นทั้ง corpus"** ผิด: O(n²) forward passes; ใช้ rerank top-k. **"Bi-encoder ไม่ต้อง forward เลย"** ผิด: ต้อง n forward (แต่ cache vector ได้).
5. **Head dimension**: distractor 768, 1536 (แค่ u,v), 1,536+..., 768x2; ที่ถูก **2,304** (u, v, |u−v|). อีกกับดัก: ใช้ u−v (ไม่มีค่าสัมบูรณ์) / u⊙v — SBERT classification ใช้ |u−v|.
6. **Siamese = สองโมเดลคนละ weights** ผิด; ต้อง weights เดียวกัน.
7. **Triplet loss**: "loss>0 เมื่อ positive ใกล้กว่า negative" ผิด; L=0 เมื่อ d(a,p)+α ≤ d(a,n). "Margin ใหญ่ = ฝึกง่ายกว่า" ผิด (stricter/harder). **"Hard negative ดีที่สุด"** — สไลด์บอก **semi-hard** สมดุลสุด (hard ทำให้ไม่เสถียร). "Random negatives ให้ gradient มาก" ผิด (มักง่าย -> loss 0).
8. **Triplet กับ Margin Ranking เหมือนกันเสมอ** — เทียบเท่าเมื่อ d=1−sim (cosine) เท่านั้น; Euclidean ต่าง. Triplet ใช้ distance, Margin Ranking ใช้ similarity (ลำดับเครื่องหมาย sim(a,n) − sim(a,p)).
9. **MNRL ต้องการ explicit hard negatives** ผิด: ใช้ in-batch negatives, ต้องการแค่ (q,p). "τ สูง = เส้นแบ่งคมกว่า" ผิด: **τ ต่ำ** = คมกว่า. Batch N -> N−1 negatives (ไม่ใช่ N).
10. **L2 distance บน embedding ไม่ normalize ใช้ได้ดี** ผิด (ขนาดต่างทำให้ดูไม่คล้าย); dot ≠ cosine ถ้าไม่ normalize. "Dot product = cosine เสมอ" ผิด.
11. **`.encode()` normalize ให้เสมอ** ผิด: default off (`normalize_embeddings=False`); MiniLM ที่ normalize อยู่แล้วเป็นสมบัติเฉพาะ checkpoint; สไลด์ 38/41 บอกให้ normalize เอง.
12. **`IndexFlatL2` คืน cosine / IP ต่ำ=ดี** ผิด: L2 คืน squared distance ต่ำ=ดี; IP หลัง norm คืน cosine สูง=ดี. `IndexHNSWFlat` default = L2.
13. **ANN รับประกัน exact / เร็วขึ้นโดยไม่เสียอะไร** ผิด: trade-off; ห้ามเทียบ speed โดยไม่ดู Recall@k. "Flat recall ~95%" ผิด (100%). IVF+PQ recall ~90% (ต่ำสุดแต่ประหยัด RAM). **HNSW = graph-based, ~99%** ไม่ใช่ cluster.
14. **Pad tokens ≈ 0 เฉลี่ยแล้วไม่เป็นไร** ผิด (demo: norm 9.67, ไม่ใช่ศูนย์ -> corrupt).
15. **Recall@k vs MRR สลับกัน**: Recall@k = coverage (สัดส่วน relevant ที่เจอใน top-k); MRR = ตำแหน่งของ relevant ตัวแรก (1/rank เฉลี่ย). MRR rank 2 = 0.5 (ไม่ใช่ 2).
16. **NLI label** = entailment ดึงใกล้, contradiction ผลักไกล, neutral ไม่ใช่ positive; STS score 0-5 (normalize เป็น 0-1), cos∈[-1,1].
17. **SimCSE unsupervised ใช้ label** ผิด; positive = ประโยคเดียวกันผ่าน dropout 2 ครั้ง. ตัวเลข 76.3 (unsup), 81.6 (sup).
18. ผสมสัปดาห์: Week 6 ใช้ [CLS]+linear head (classification) vs Week 9 ใช้ [u;v;|u−v|] — คนละ input.

---

## 8. ความขัดแย้ง/ข้อควรระวัง (สไลด์ vs hints vs ตัวเลขในสไลด์เอง)

- **Bi-encoder 10K ประโยค**: สไลด์ 8 = **~50 วินาที**; hints (Whisper) = "~5 seconds" -> **ยึดสไลด์ (50 s)**.
- **GloVe/BERT-mean ในสไลด์ 4 vs 19**: 58.02/46.35 (สไลด์ 4) vs 58.2/46.4 (สไลด์ 19, ปัด) — ใช้ค่าทศนิยมสองตำแหน่งจากสไลด์ 4 ถ้าโจทย์ระบุละเอียด.
- **SBERT-base**: 77.03 (สไลด์ 4, 19; STS Benchmark) vs **74.89** (สไลด์ 24; ในบริบท SimCSE) — คนละ eval/benchmark; จำแยกตามบริบท (นอกสไลด์: SimCSE รายงานค่าเฉลี่ยหลาย STS tasks). 
- **avg cosine ของ raw BERT**: 0.58 (mean-pooled; สไลด์ 19 + demo) vs 0.768 ([CLS]; demo เท่านั้น) — อย่าสับสน.
- ตาราง ANN สไลด์ 39: คอลัมน์ Query time สกัดมาไม่ชัด; อย่าจำสัญลักษณ์ O(.) ของ IVF/IVF+PQ จาก txt โดยไม่ดูสไลด์ต้นฉบับ.
- hints เตือน: ข้อเติมคำเขียนระวังทศนิยม/เลขยกกำลัง 10 (เช่น 10.7 ≠ 0.107); lab ไม่ออก.

---

## 9. SHEET candidates (เรียงตามความสำคัญ, <= 1 บรรทัดต่ออัน)

1. SBERT head: c=[u;v;|u−v|] = 3d = **768x3 = 2,304** -> W 2304x3 -> softmax -> CE
2. Triplet: **L = max(0, d(a,p) − d(a,n) + α)**; L=0 ถ้า d(a,p)+α ≤ d(a,n); α 0.5-1.0; SBERT: Euclid, ε=1
3. Margin Ranking: **L = max(0, sim(a,n) − sim(a,p) + α)**; ≡ Triplet เมื่อ d=1−sim
4. MNRL: **L_i = −log[ e^{sim(q_i,p_i)/τ} / Σ_j e^{sim(q_i,p_j)/τ} ]**, τ 0.05-0.1, batch N -> N−1 negatives
5. Mining: easy d(a,n)≥d(a,p)+α (L=0) | hard d(a,n)<d(a,p) | **semi-hard d(a,p)<d(a,n)<d(a,p)+α (best)**
6. Triplet ตัวอย่าง: d(a,p)=0.087, d(a,n)=0.812, α=0.5 -> L=0; hard n: 0.035 -> **L=0.552**
7. Cross: **n(n−1)/2** forward passes; n=10⁴ -> 5x10⁷ x 5 ms ≈ **69 ชม.**; Bi: n forward ≈ **50 s** + n² dot (<1 s)
8. Cosine: cos=u·v/(‖u‖‖v‖) ∈[−1,1]; d_cos=1−cos ∈[0,2]; dot=‖u‖‖v‖cos; normalized -> dot=cos
9. STS: GloVe **58.02** | BERT [CLS] **16.50** | BERT mean **46.35** | SBERT-NLI **77.03**
10. Mean pooling: s=Σ H_t m_t / Σ m_t (mask ตัด [PAD]); CLS=H_0; Max=max_t H_{t,i}; Mean ดีสุด
11. Anisotropy: raw BERT cos ∈[0.46,0.85] mean 0.58 -> SBERT [0.1,0.99]; demo SBERT min −0.156; สาเหตุ = MLM objective
12. Fixes: post-process (whitening) | **contrastive FT (SBERT)** | SimCSE (dropout; unsup 76.3, sup 81.6 vs SBERT 74.89)
13. SBERT train: NLI(1M=570K+430K; CE) -> STSb(8.5K, 0-5 -> 0-1; MSE on cos); batch 16, lr 2e-5, 1 epoch, seq 128; 20+5 min V100
14. Memory = n x d x 4 B: 1M x 768 -> 3.072 GB; **5M x 768 -> 15.36 GB**; brute 10M x 768 ≈ 7.7 s
15. FAISS: float32 -> `normalize_L2` -> `IndexFlatIP(d)` -> `add` -> `search(q,k)` -> D (scores), I (idx) [1,k]
16. IndexFlatIP = cosine (สูง=ดี) หลัง norm | IndexFlatL2 = squared dist (ต่ำ=ดี) | HNSW default L2
17. ANN: Flat 100% (<100K) | IVF ~95% | **HNSW ~99%** (<1M) | IVF+PQ ~90% (>1B); กฎ: รายงาน Recall@k เสมอ
18. HNSW: top layers sparse entry -> greedy walk -> drop down -> layer 0 (ทุกจุด); Qdrant/Weaviate/Pinecone
19. Recall@k = |Ret_k ∩ Rel|/|Rel|; MRR = (1/|Q|)Σ1/rank_i (rank1=1.0, rank2=0.5)
20. Dims: MiniLM-L6-v2 **384** (Fast) | mpnet-base-v2 **768** | bge-large-en-v1.5 **1024** (Slower)
21. Bi vs Cross: bi = stage1 recall, cacheable, no cross-attn | cross = stage2 rerank, precision, full attn, O(n²)
22. Latency: encode 5-10 ms GPU / ~50 ms CPU; search <1 ms (100K exact) / <5 ms (10M ANN); total <20 ms
23. Pipeline Wk12: Dense (bi+FAISS) + BM25 -> **RRF** -> Cross-encoder rerank -> LLM
24. `.encode()`: tokenize -> forward -> masked mean pool -> L2-norm (**off by default**)
25. Data: Gold (NLI, STS) -> Silver (MS MARCO ~500K+) -> Weak (E5/BGE, 100M-1B+); Quora ~400K

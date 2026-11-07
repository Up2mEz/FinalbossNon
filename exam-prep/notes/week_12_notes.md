# Week 12 — Hybrid RAG, Multimodal (CLIP) และ GraphRAG (Study Notes)

แหล่งข้อมูล: `text/slides_w12.txt` (เลขสไลด์ที่อ้าง = เลขท้ายหน้า = segment ที่ N ของไฟล์), `text/demo_w12.txt`, `source/nlp_exam_hints.md`
หมายเหตุสำคัญ: สไลด์หลายหน้ามี **ตารางเป็นรูปภาพ** (diminishing returns ของ BM25 s.16, 4 scenarios s.18, RRF worked example s.22, Bi vs Cross s.26, pipeline numbers s.27, Vector vs Graph s.39/46, bookend prompt s.53) ที่ text extract ไม่ได้ ตัวเลขในตารางเหล่านั้นผมคำนวณใหม่จากสูตรในสไลด์ และติดป้าย "(คำนวณเอง)" ให้ไปเทียบกับสไลด์จริงอีกครั้ง
สัญลักษณ์ที่ถูก extract พัง (เช่น ≠, Σ, θ) ผมกู้ความหมายจากบริบท และติดธง "(ตีความจากบริบท)" ที่ไม่แน่ใจ

---

## 1. น้ำหนักข้อสอบ + ลำดับหัวข้อที่น่าจะออก

- Week 12 = **10 MCQ + 2 fill-in-the-blank** (แต่ละสัปดาห์ได้ส่วนแบ่งเท่ากัน; ในสัปดาห์เดียวกัน หัวข้อที่สไลด์เยอะ + อาจารย์เน้นจะได้ข้อมากกว่า)
- Hints ที่ตรงกับ Week 12: (a) **BM25 saturation & length normalization** (strong: "อาจให้ตัวเลข / ถามกี่เปอร์เซ็นต์ / เปรียบเทียบ"), (b) RAG pipeline: BM25 + dense → RRF → cross-encoder → LLM, "lost in the middle", (c) bookend reordering (week 13 hint แต่เนื้อหาอยู่ใน slide w12)
- Lab ไม่ออก แต่ **Demo ออกได้ 1-2 ข้อ** (ดูหัวข้อ 4)

ลำดับ (ประมาณจำนวนข้อ MCQ จาก 10 + โอกาสเป็น fill-in):

| อันดับ | หัวข้อ | สไลด์ | เหตุผล | ประมาณข้อ |
|---|---|---|---|---|
| 1 | **BM25** (IDF, TF saturation k1, length norm b, คำนวณ) | 15-18 | 4 สไลด์เต็ม + hint ตรง ๆ ว่าอาจให้คำนวณ/เปรียบเทียบ | 2-3 |
| 2 | **Hybrid + RRF** (score incompatibility, สูตร RRF, k=60) | 20-23 | สูตรสั้น คำนวณง่าย (hint ทั่วไป: "บวกคูณง่าย ๆ") + demo มี rrf_merge | 1-2 |
| 3 | **Cross-encoder / Re-ranking / Two-stage pipeline** | 24-27 | เปรียบเทียบ bi vs cross, ColBERT, latency 211 ms | 1-2 |
| 4 | **GraphRAG** (triples, community, Leiden, local/global, multi-hop, vs vector RAG) | 39-46 | 8 สไลด์ แนวคิดล้วน เหมาะออก MCQ | 1-2 |
| 5 | **CLIP / Multimodal RAG** (contrastive, symmetric InfoNCE, zero-shot, ViT, failure modes) | 30-38 | 8-9 สไลด์ | 1-2 |
| 6 | **Long-context / Lost in the Middle / Bookend / SWA / RoPE interpolation** | 47-53, 56 | 7 สไลด์ + demo 2 | 1-2 |
| 7 | Chunking, embedding limits, hubness/anisotropy | 6-12 | สไลด์เยอะแต่ทำนองจำ | 0-1 |
| 8 | RAG recap / Lewis architecture / 7 failure points | 2-5 | เนื้อหาเก่า (Recap) | 0-1 |

โอกาส fill-in 2 ข้อสูงสุด: ตัวเลขเฉพาะ (k=60, k1+1, 76.2%, 32,768, 400M, 256 tokens) และคำศัพท์ (Leiden, MaxSim, Hubness, U-shaped, Map/Reduce, Reciprocal Rank Fusion)

---

## 2. Core Concepts

### 2.0 RAG recap (s.2-5)
- **Parametric memory** = ความรู้ใน weights ตอน train; ปัญหา 2 อย่าง: (1) **Knowledge cutoff** (2) **Private data** → LLM ไม่ตอบ "ไม่รู้" แต่ **confabulate อย่างมั่นใจ** (hallucination)
- Naive fix 1: ยัดทุกอย่างใน system prompt → ใช้ได้กับเอกสารเล็ก, พังเมื่อใหญ่เพราะ attention cost **O(N²)**; Naive fix 2: Fine-tune → แพง ช้า และสร้าง cutoff ใหม่
- **RAG (Lewis et al., 2020)** [MEMORIZE]
  - **Retriever p_η = non-parametric memory**: Query Encoder q(x) + **MIPS** (Maximum Inner Product Search) หา d(z) จาก Document Index
  - **Generator p_θ = parametric memory** (seq2seq): รับ query + docs → สร้าง y โดย **marginalize** over top-K docs (z = latent variable)
  - **End-to-end backprop** ผ่าน generator และ query encoder (ฝึกร่วมกันได้)
  - ใช้ได้กับ QA, Fact Verification, Jeopardy question generation
- **Seven Failure Points** (Barnett et al., 2024, s.5): สไลด์แสดงเป็นรูปภาพ ไม่มีรายการในข้อความ → ถ้าจะออก น่าเป็นระดับ "RAG ยากเพราะมีหลายจุดล้มเหลวทั้ง indexing และ query time" (รายการ 7 ข้อเป็นความรู้นอกสไลด์: missing content, missed top-ranked, not in context, not extracted, wrong format, incorrect specificity, incomplete)

### 2.1 Act 1: Ingestion pipeline (s.6-12)
Pipeline offline 5 ขั้น: **Load → Parse → Chunk → Embed → Index** (vector store: FAISS, Qdrant, Chroma) [MEMORIZE ลำดับ]
- "Garbage in → garbage retrieved": การตัดสินใจขั้น 1-3 กำหนดว่าอะไร *retrievable*
- Embedding dimension ที่ต้องรู้ (s.6): **384 (MiniLM), 768 (BERT-base), 1536 (text-embedding-3-small), 3072 (text-embedding-3-large)**; s.10 เพิ่ม **bge-m3 = 1024**

**Chunking** [MEMORIZE]

| วิธี | แนวคิด | ข้อเสีย/หมายเหตุ |
|---|---|---|
| Fixed-size character (s.7) | ตัดทุก `chunk_size` ตัวอักษร + `chunk_overlap` (ตัวอย่าง 512/64, `range(0,len,512-64)` → stride 448) | 3 failure modes: (1) **Context Fragmentation** แม้มี overlap ถ้า subject/title อยู่ไกล fact (2) **Mid-Table Cut** header อยู่ chunk A, data อยู่ B (3) **Cross-Section Bleed** บทสรุป sec 3 ติดกับบทนำ sec 4 → false positive สูง, LLM ได้ context ไม่ coherent |
| Structural / layout-aware (s.8) | หาขอบธรรมชาติก่อน: heading, section, paragraph; tools `unstructured`, `PyMuPDF`, `pdfminer.six` | ลำดับตัด **Chapter > Section > Paragraph > Sentence > Token**; ตัดที่ "หน่วยธรรมชาติใหญ่สุดที่ยังพอดี max_tokens"; ตาราง: **เก็บ header กับทุก row group**; "structure-aware ไม่ได้ดีกว่าเสมอไป ต้องวัดผล retrieval" |
| Semantic shift detection (s.9) | embed ทุกประโยคด้วย bi-encoder (SBERT) → distance ติดกัน `dist_i = 1 − cos(s_i, s_{i+1})` → ตัดที่ distance spike > threshold τ | heuristic: **τ = 95th percentile** ของ distances; ข้อเสีย: ต้อง embed ทุกประโยค → cost **เป็นเส้นตรงกับความยาวเอกสาร** |

**Bi-encoder / SentenceTransformers limits (s.10)** [MEMORIZE]
- Doc vectors precompute ครั้งเดียว; L2-normalise แล้ว **dot product = cosine**
- Limit 1: `all-MiniLM-L6-v2` **truncate ที่ 256 tokens แบบเงียบ (ไม่ error)** → ส่วนท้ายหาย = หาไม่เจอ → ต้องเช็ค chunker กับขีดจำกัดนี้
- Limit 2: dimension = ค่าจัดเก็บ: **N × d × 4 bytes** (float32); 1 ล้าน chunks = **1.5 GB ที่ d=384, 12.3 GB ที่ d=3072**
- ตารางโมเดล: all-MiniLM-L6-v2 = 384; bge-m3 = 1024 (multilingual, multiple retrieval modes); API embedding model = provider-specific

**Embedding pathologies (s.11)** [MEMORIZE]
- **Hubness**: เวกเตอร์ไม่กี่ตัวเป็น nearest neighbor ของ query ที่ไม่เกี่ยวกันจำนวนมาก → อาการใน RAG: chunk generic เดิม ๆ โผล่ใน top-k ซ้ำ
- **Anisotropy**: embeddings กระจุกอยู่ในบริเวณแคบ → cosine similarity ของเวกเตอร์ที่ไม่เกี่ยวกัน "พอง" (inflated)
- วินิจฉัย: ดู mean pairwise cosine + ความถี่ที่แต่ละ chunk ปรากฏใน top-k; แก้: retrieval-trained embeddings, normalization, whitening, hard-negative training
- (s.12 in-class activity: map classmates เป็นเวกเตอร์ 1D/2D; insight = ใน 768 มิติ นักเรียนที่ "เฉลี่ยทุกฟีเจอร์" = hub)

### 2.2 Act 2: Retrieval

**Sparse vs Dense (s.14)** [MEMORIZE + SHEET]

| | Sparse (Lexical) | Dense (Semantic) |
|---|---|---|
| Representation | high-dim sparse (ขนาด vocabulary) | low-dim dense vector (d) |
| Speed | เร็วมาก (inverted index) | ช้ากว่า (ANN search) |
| จุดแข็ง | คำหายาก, รหัสสินค้า, ชื่อเฉพาะ | paraphrase, จับ concept |
| จุดอ่อน | **Vocabulary mismatch** | **Hallucinated relevance** |
| ตัวอย่าง | BM25 | Bi-Encoder (SBERT) |

Failure modes ส่งเสริมกัน → เหตุผลของ Hybrid. ตัวอย่างสไลด์ (s.14, s.20):
- "heart attack treatment" vs doc "myocardial infarction" → BM25 miss, **Dense ชนะ**
- "invoice #INV-2024-7832" → **BM25 ชนะ** (dense อาจคืนเลข invoice ที่คล้ายกัน)
- "explain quantum entanglement simply" → **Dense ชนะ**
- "python 3.11 asyncio bug fix" → **BM25 ชนะ** (dense อาจดึง python 3.10)
- BEIR (Thakur 2021, **18 datasets**): BM25 เป็น zero-shot baseline แข็ง **ชนะ DPR ในส่วนใหญ่**; การ fuse BM25 + dense ดีกว่าอย่างใดอย่างหนึ่งในส่วนใหญ่

#### BM25 (s.15-18) — หัวข้อสำคัญที่สุด

สูตรเต็ม [SHEET]:

BM25(d,q) = Σ_{t∈q} IDF(t) · f(t,d)·(k1+1) / [ f(t,d) + k1·(1 − b + b·|d|/avgdl) ]

IDF(t) = ln( (N − n_t + 0.5) / (n_t + 0.5) )

| สัญลักษณ์ | ความหมาย |
|---|---|
| f(t,d) | จำนวนครั้งที่ term t ปรากฏใน doc d (term frequency) |
| \|d\| | ความยาว doc d (จำนวนคำ); avgdl = ความยาวเฉลี่ยของทั้ง corpus |
| N | จำนวน doc ทั้งหมด; n_t = จำนวน doc ที่มี t |
| k1 | พารามิเตอร์ TF saturation (ในสไลด์ใช้ 1.5) |
| b ∈ [0,1] | ความแรงของ length normalization |

**IDF (s.15)**
- คำหายาก/entity (`"INV-7832"`, `"myocardial"`): n_t ≪ N → IDF สูง
- stopword (n_t > N/2) → IDF **ติดลบ**; **เท่ากับ 0 พอดีที่ n = N/2**
- 2 convention แก้ติดลบ: **Lucene/Elasticsearch ใช้ ln(1 + …)** (บวกเสมอ); **`rank_bm25` (ใช้ใน demo/lab) floor ค่าติดลบที่ ε × avg_idf** → ทุก term IDF ลบเท่ากันหมด
- ทำไม BM25 ดีกว่า TF-IDF: TF-IDF เป็น linear ใน TF → **keyword stuffing** ได้ (ซ้ำ 1,000 ครั้ง = 1000× score); BM25 แก้ด้วย **TF Saturation (k1)** + **Length Normalization (b)**

**TF saturation (s.16)** [MEMORIZE + SHEET]
- ตั้ง b = 0: TF_BM25 = f(k1+1) / (f + k1)
- **Asymptote ceiling = k1 + 1** (k1=1.5 → **2.50**) ไม่ขึ้นกับความยาวเอกสาร (เปิด b แล้วก็ยัง k1+1)
- **Half-saturation**: เมื่อ f = k1 → score = **50% ของ ceiling = 1.25** (k1=1.5)
- k1 ใหญ่ → saturate ช้า (TF มีน้ำหนักนานขึ้น); k1 เล็ก → อิ่มเร็ว (ตีความจากสูตร)

**Length normalization (s.17)**
- ปัญหา: หน้า 500 คำย่อมมีคำซ้ำมากกว่าโน้ต 50 คำโดยธรรมชาติ
- L(d) = 1 − b + b·|d|/avgdl (อยู่ในตัวส่วน คู่กับ k1)
- **b = 0**: ไม่สนความยาว; **b = 1**: penalty สัดส่วนเต็ม; **b = 0.75**: ค่ามาตรฐาน TREC
- b **ไม่เปลี่ยน ceiling (k1+1)** เปลี่ยนแค่ "เข้าใกล้ ceiling เร็ว/ช้า" (ตรงกับ hint)
- doc สั้นกว่าค่าเฉลี่ย (L<1) → ตัวส่วนเล็ก → TF part สูงขึ้น; doc ยาว (L>1) → ถูกลดคะแนน

**Dense retrieval training (s.19)**
- InfoNCE: ℓ_i = −log [ exp(sim(q_i,p_i)/τ) / Σ_{j=1}^{N} exp(sim(q_i,p_j)/τ) ]; ตัวเศษ = คู่ที่ถูก, ตัวส่วน = ทุก passage ใน batch (รวมคู่ถูก); **τ = ความคม (sharpness)**
- **In-batch negatives**: batch N คู่ → p_j (j≠i) เป็น negative ฟรี; **N = 256 → 255 negatives**; cost **O(N²)** comparisons ต่อ batch
- สำหรับ retrieval: คู่ (query, passage) **asymmetric**, passage ยาว; sim(q,p) = q·p บนเวกเตอร์ L2-normalised
- **Hard negative mining (ANCE, Xiong 2021)**: in-batch negatives "ง่ายเกิน" → sample จากใกล้ decision boundary (non-relevant ที่คะแนนสูง) → representation คมขึ้น
- Training format: `{query, pos, neg:[hard negative, random negative]}`

#### Hybrid + RRF (s.21-23) [MEMORIZE]
- ดึงขนาน BM25 + dense แล้ว fuse
- **Score incompatibility**: BM25 unbounded **(0, ∞)** vs cosine bounded **[−1, 1]** → บวกตรง ๆ ไม่ได้
- Naive fix: **min-max normalize** แต่ละ list เป็น [0,1] แล้วบวก → **sensitive ต่อ outlier**
- Better: **RRF** — rank-based, distribution-free, ไม่สนคะแนนดิบ

**RRF(d) = Σ_{r∈R} 1 / (k + rank_r(d))** [SHEET] — R = ชุดระบบ retrieval, rank_r(d) = อันดับของ d ในระบบ r, **k = smoothing constant (typically 60)**
- Doc ที่ติดอันดับสูงในหลาย list ได้ boost; doc ที่โด่งใน list เดียวถูกทอนด้วย k
- **ทำไม k=60 (s.23)**: ไม่มี k → rank1 = 1.0, rank2 = 0.5 (ลด 50% จากขยับ 1 อันดับ = รุนแรงเกิน); k=60 → rank1 = 1/61 ≈ **0.01639**, rank2 = 1/62 ≈ **0.01613** (ลดแค่ ≈ **1.6%**)
- ผลต่างอันดับติดกัน: score(r) − score(r+1) = 1/(k+r) − 1/(k+r+1) = **1/((k+r)(k+r+1))**
- **k เล็ก → ตกชันตอนต้น (aggressive); k ใหญ่ → เส้นแบน (ทุกอันดับเท่ากัน)**
- ที่มา: Cormack et al. (2009) **fix k=60 ใน pilot study บน TREC และไม่เคย tune ใหม่**; ในทางปฏิบัติผลไม่ไวใน **k ≈ 40-80** → default zero-shot; practical rule: เริ่ม 60, truncate input lists ให้สม่ำเสมอ, tune เมื่อ validation data บอกว่าได้ประโยชน์

#### Re-ranking (s.24-27)
- **Retriever → recall (กว้าง เร็ว); Reranker → precision (เรียงใหม่อย่างละเอียด)**
- Bi-encoder limit: encode query/doc แยกกัน; Cross-encoder: **joint attention** จับ negation + token-level interactions
- Trade-off: K ใหญ่ → recall อาจดีขึ้น แต่ cost rerank สูงขึ้น
- Pipeline: **Retrieve K → rerank → ส่งเฉพาะ top N ให้ LLM**
- **Cross-encoder architecture (s.25)**: input `[CLS] query [SEP] document [SEP]`; ทุก query token attend ทุก doc token ในทุก layer; **score = wᵀ·h_[CLS] + b** (raw scalar logit); cost **O((|q|+|d|)²)**, precompute offline ไม่ได้ → รันออนไลน์บน **top K = 20-100** pairs จาก RRF; ตัวอย่าง `cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`
- **ColBERT (2020, s.26)**: **late interaction** — encoder แยกกัน แต่จับคู่ระดับ token ตอน inference ด้วย **MaxSim**; ถูกกว่า cross-encoder ≈ **2 orders of magnitude (~100×)** ที่ query time; คุณภาพ **อยู่ระหว่าง** bi กับ cross

| | Bi-Encoder | ColBERT | Cross-Encoder |
|---|---|---|---|
| Encode | query/doc แยก, precompute doc | แยก, token-level vectors | ร่วมกัน (joint) |
| ความแม่น | ต่ำสุด | กลาง | สูงสุด |
| Cost | ถูกสุด (n forward + dot products) | ถูกกว่า CE ~100× | แพงสุด (n pairs เต็ม forward) |
| ใช้ | Stage 1 ทั้ง corpus | กลาง | Stage 3 เฉพาะ top-K |

(ตารางสรุปเอง จากข้อความ s.24-26 + hint week 9: cross-encoder O(n²) forward passes ระดับ ms, ตัวอย่าง "69 ชั่วโมง"; bi-encoder n forward แล้ว n² dot products ระดับ ns ≈ 5 วินาที — ต้องรู้ว่า O() นับอะไร)

**Two-stage pipeline latency (s.27)** [SHEET]: Stage 1a (BM25) ≈ 5 ms ∥ Stage 1b (dense) ≈ 10 ms → ขนานกัน = max(5,10); Stage 2 RRF ≈ 1 ms; Stage 3 cross-encoder ≈ 200 ms → **รวม ≈ max(5,10) + 1 + 200 = 211 ms** ("illustrative — วัดเองบน hardware จริง"). Tip: ถ้า latency-sensitive **ข้าม Stage 3** และ tune k ของ Stage 2 แทน

### 2.3 Act 3: Multimodal (CLIP) และ GraphRAG

#### Multimodal motivation (s.30)
หลักฐานอยู่ใน chart/scan/diagram; OCR อาจได้ label แต่ **เสียโครงสร้างภาพ/trend** → ต้องการ image และ text ใน **embedding space เดียวกัน**; **CLIP** train image encoder + text encoder แบบ contrastive → text ค้นหา image ได้ และ image ค้นหา text ได้

#### CLIP (Radford et al., 2021; s.31-34) [MEMORIZE]
1. **Contrastive pre-training**: Text Encoder (Transformer) + Image Encoder (**ViT / ResNet**) บน **400M** คู่ (image, text); batch **N = 32,768**; เพิ่ม cosine similarity ของคู่บนเส้นทแยง (I_i·T_i) ลดคู่นอกทแยงในเมทริกซ์ N×N
2. **Create dataset classifier**: ใช้ prompt template เช่น "A photo of a {object}." → Text Encoder → class embeddings T_1..T_N
3. **Zero-shot prediction**: encode รูปที่ไม่เคยเห็น → I_1 → similarity กับทุก T → เลือกสูงสุด

**Symmetric InfoNCE (s.33)** [SHEET]:
L_CLIP = −(1/2N) Σ_{i=1}^{N} [ log( exp(I_i·T_i/τ) / Σ_{j=1}^{N} exp(I_i·T_j/τ) ) + log( exp(T_i·I_i/τ) / Σ_{j=1}^{N} exp(T_i·I_j/τ) ) ]
- Term 1 (**Image→Text, rows**): แต่ละรูปหา caption ที่ถูกจาก N captions; Term 2 (**Text→Image, columns**): แต่ละ caption หารูปที่ถูกจาก N images
- Scale: N=32,768 → แต่ละคู่บวกมี negative distractors (N−1 ≈ 32,767, นับเองจากสูตร) ใน forward pass เดียว; เมทริกซ์ similarity N×N

**Zero-shot (s.34)**: p(c | image) = exp(I·T_c/τ) / Σ_{k=1}^{C} exp(I·T_k/τ) (softmax เหนือ C classes)
- **Zero-shot CLIP ViT-L/14@336px = 76.2% Top-1 ImageNet** เท่า supervised ResNet-50 โดยไม่เห็น ImageNet label เลย
- Prompt engineering + ensembling ("a photo of a {label}, a type of pet") ดีขึ้น **เกือบ 5 points**
- **Multimodal RAG**: Index phase = precompute CLIP embeddings ของ figure/diagram/scan ทั้งหมดลง FAISS; **Text→Image** ("Show Q3 revenue chart" → กราฟการเงิน); **Image→Document** (อัปโหลดรูปชิ้นส่วน → คู่มือซ่อม)

**ViT (s.32)**: แบ่งรูป 2D เป็น **patches ขนาดคงที่** → flatten + **linear projection** → token embedding; prepend learnable **[class] token (0\*)** + **1D position embeddings** → Transformer encoder. ต่างกัน: **Standard ViT** ส่ง token 0\* เข้า **MLP Head** ทำนาย fixed classes (Bird, Ball, Car…); **CLIP** แทน MLP Head ด้วย **projection** ไปยัง embedding space ร่วม (ตีความจากบริบท)

**CLIP Mental Model (s.37) — failure modes** [MEMORIZE]: contrastive similarity พลาด **relations** ("A dog chasing a cat" → ได้สัตว์ที่แค่อยู่ใกล้), **attributes** ("broken phone screen" → over-rely ที่ "phone"), **negation** ("bird without wings" → ได้นกปกติ); Takeaway: ต้อง test compositional queries ไม่ assume ว่าเข้าใจความหมาย
(s.36 = รูป Visual Instruction Tuning/LLaVA, s.35/38 เป็นรูปเท่านั้น → ไม่มีข้อความให้สอบ)

#### Vector RAG ล้มเหลวเมื่อไร (s.39)
Vector RAG เก่ง **local retrieval** แต่พังกับ **macro-level / global summarization** ("themes ทั้ง corpus") เพราะ **มอง chunk เป็นจุดแยกอิสระ ไม่เห็นความสัมพันธ์ระหว่าง entity** → ต้องมีโครงสร้างเชื่อม entity ข้ามเอกสาร = **Knowledge Graph**

#### Triple extraction (s.41)
Triple = **(Subject, Relation, Object)**
- Traditional: **NER** → **Relation Extraction** → **Entity Linking** (รวม alias/pronoun เป็น canonical node) → **Triple Validation** (กรอง noise)
- Modern GraphRAG: prompt LLM ทีเดียวให้ extract **(entity1, relation, entity2, description)** จาก {chunk}; trade-off: **flexibility สูง vs token cost** ข้ามพัน chunks

#### GraphRAG (Edge et al., 2024, Microsoft; s.42-44) [MEMORIZE]
สอง phase:
- **Indexing time (offline)**: Source docs → text chunks → **entities & relationships (extract graph)** → knowledge graph → **community detection** → **community summaries (pre-generate)**
- **Query time (online)**: **Map-Reduce** ข้าม community summaries → community answers → **global answer**
- Core breakthrough: **precompute ความเข้าใจภาพรวมของ corpus ตอน offline** เลี่ยงข้อจำกัด context ตอน query
- **Leiden algorithm** (Traag 2019): จัดกลุ่มโดย optimize **Modularity Q** (ความหนาแน่น edge ภายในเทียบกับที่คาดไว้); รับประกัน **well-connected communities ไม่มีส่วนที่หลุดขาดภายใน** (ดีกว่า **Louvain**)
- **Hierarchical (multi-level)**: **coarse levels** = themes กว้างทั้ง corpus; **fine-grained levels** = sub-cluster สำหรับคำถามเฉพาะทาง
- **Community report**: LLM รับ (Entities, Relations, Source snippets) → เขียนรายงานสรุปเชิงโครงสร้างของแต่ละ community

**Local vs Global search (s.44)** [SHEET]

| | Local Search | Global Search |
|---|---|---|
| โฟกัส | Entity-focused: เดินผ่าน entities, relationships, chunks รอบ entity เป้าหมาย | Corpus-wide, ไม่มี entity anchor |
| ตัวอย่าง | "What are the key achievements of Sam Altman?" (**1 LLM call**, latency ต่ำ) | "What are the top 5 emerging technology themes in all 500 reports?" |
| กลไก | graph traversal | **Map** (fan-out ส่ง query ไปทุก community summary แบบขนาน → intermediate answers) → **Reduce** (fan-in รวม/จัดอันดับ → LLM สังเคราะห์ Global Answer) |
| Cost | ต่ำ | **สูงกว่ามาก** (LLM calls และ tokens มากกว่า) |

**Multi-hop (s.45)**: ตัวอย่าง "บริษัทไหนที่ executives ของ OpenAI ไปก่อตั้งภายหลัง?" — 3 ขั้น:
1. **Query decomposition** (LLM / **Text-to-Cypher**) → Cypher 2 hops: `MATCH (p:PERSON)-[:WORKED_AT]->(:ORG {name:"OpenAI"})` / `MATCH (p)-[:FOUNDED]->(c:COMPANY)` / `RETURN p.name, c.name`
2. **Deterministic traversal** ใน Graph DB (ตามพอยน์เตอร์ที่เก็บไว้): Hop 1 (incoming) OpenAI → {Sam Altman, Dario Amodei}; Hop 2 (outgoing) → {Worldcoin, Anthropic}
3. **Answer synthesis** (LLM): sub-graph triples → ภาษาธรรมชาติ
- Vector RAG พังเพราะ cosine ดึง chunk ที่พูดถึง "OpenAI"/"executives"/"founded" แบบแยกส่วน แต่ **chain A→B→C ข้ามเอกสารแบบ deterministic ไม่ได้**

**Decision framework (s.46, s.56)**
- Hybrid strategy: **สร้างทั้งคู่**, route ด้วย lightweight classifier: **specific factual → Vector RAG; thematic/relational → GraphRAG**
- "LLM Wiki" pattern (2026): compile corpus เป็น Markdown wiki ที่ link กัน → query wiki แทน chunks; แนวคิดเดียวกับ community reports (precompute synthesis) แต่โครงสร้างเป็น pages+links; เป็น practitioner proposal **ไม่มี published evaluation**, เหมาะ ~100–10,000 docs
- **RAG vs Fine-Tune vs Long-Context (s.56)**: (1) ความรู้เปลี่ยนบ่อย → **RAG**; (2) ต้องเปลี่ยน format/style ของ output → **Fine-Tune**; (3) reason ข้ามชุดเอกสาร **คงที่และเล็ก** → **Long-Context**; corpus ใหญ่ + คำถามเชิงความสัมพันธ์ → **GraphRAG**

### 2.4 Act 4: Long-Context Trap (s.47-53)
- ต่อให้มี 1M tokens ก็ยังต้อง RAG: (1) **Cost**: full attention = **O(N²)** pairwise scores (cost จริงขึ้นกับ architecture, kernel, caching, sparsity, hardware); (2) **Lost in the Middle** (Liu et al., 2024): LLM ใช้ข้อมูลกลาง context ได้แย่
- **Lost in the Middle (s.52)**: multi-doc QA — ย้าย relevant doc ไปตำแหน่งต่าง ๆ ใน context ยาวคงที่ → accuracy เป็น **U-shaped curve** (ดีที่ต้น และท้าย, แย่ที่กลาง); เป็นผลเชิงประจักษ์ (empirical) ใน LLM ชั้นนำหลายตัว; takeaway: **retrieval quality ไม่พอ — ลำดับ/ตำแหน่ง document มีผลตรงต่อ accuracy**
- **4 production rules (s.53)** [MEMORIZE]: (1) **Reranker first** — กรองด้วย cross-encoder ก่อนจัดลำดับ; (2) **Bookend ordering** — วางหลักฐานสำคัญที่ต้นและท้าย (high-attention zones); (3) **Tune chunk count K** — สมดุล recall vs distraction; **more chunks ≠ better answers** (ตัว ≠ ใน extract หาย แต่เนื้อความชัด ตีความจากบริบท); (4) **Context distillation** — สรุป chunk ยาวก่อน synthesis สุดท้ายสำหรับคำถามซับซ้อน
- **Sliding Window Attention (s.48; Longformer)**: แต่ละ token attend เฉพาะ **w tokens ก่อนหน้า**; complexity **O(N²) → O(N·w)** (linear); **KV cache คงที่** เก็บแค่ w tokens ล่าสุดต่อ layer แบบ **rolling buffer**; คำถามปิดสไลด์: "ถ้าแต่ละ layer เห็นแค่ w แล้วเข้าใจเอกสารยาวได้อย่างไร?" → (นอกสไลด์: stack หลาย layer, receptive field ขยายเป็น ~ L×w; s.49 เป็นรูปประกอบ)
- **Linear interpolation (s.50; Chen 2023)**: RoPE degrade ที่ตำแหน่งเกิน training context → scale ตำแหน่ง **m' = m/s**, **s = L_target / L_train**; RoPE(z, m) = z·e^{j·m·θ} → ใช้ m/s แทน m. **ตัวอย่างสไลด์: 4K → 16K, s = 4, ตำแหน่ง 8000 → 2000**; trade-off: บีบแล้ว **positional separation ลดลง**; ต้อง adapt + evaluate บนข้อมูล long-context
- **NTK-aware (s.51)**: frequency bands ต่างบทบาท — **high frequencies (θ ใหญ่)** = ตำแหน่งละเอียด/ใกล้ (คำข้างเคียง); **low frequencies (θ เล็ก)** = ตำแหน่งโดยรวม (โครงสร้างเอกสาร). Linear scaling ผิดตรงบีบทุกความถี่เท่ากัน → **ทำให้ลำดับคำใกล้ ๆ เบลอ**. NTK-aware = **non-uniform: บีบ low freq, คง high freq**. **YaRN (Peng 2023)**: ขยาย **LLaMA-2 จาก 4K → 128K ด้วย ≈400 fine-tuning steps (≈0.1% ของ pre-training tokens)**

---

## 3. Worked Calculations (ตรวจเลขสองรอบแล้ว)

### 3.1 BM25 TF saturation (k1 = 1.5, b = 0) — คำนวณเองแทนตารางภาพ s.16
TF = f·2.5/(f+1.5); ceiling = 2.5

| f | คำนวณ | TF | % ของ ceiling |
|---|---|---|---|
| 1 | 2.5/2.5 | **1.000** | 40.0% |
| 2 | 5/3.5 | **1.429** | 57.1% |
| 3 | 7.5/4.5 | **1.667** | 66.7% |
| 5 | 12.5/6.5 | **1.923** | 76.9% |
| 10 | 25/11.5 | **2.174** | 87.0% |
| 100 | 250/101.5 | **2.463** | 98.5% |
| 1000 | 2500/1001.5 | **2.496** | 99.9% |
| 1.5 (=k1) | 3.75/3.0 | **1.25** | 50% พอดี |

เพิ่มกี่ % (hint บอกอาจถามเปอร์เซ็นต์): f 1→2: 1.4286/1 = **+42.9%**; 2→3: 1.6667/1.4286 = **+16.7%**; 10→100: 2.4631/2.1739 = **+13.3%** (ทั้งที่ f เพิ่ม 10 เท่า); 1→1000: เพิ่มเป็น 2.496/1 ≈ **2.5 เท่า ไม่ใช่ 1000 เท่า** (TF-IDF จะได้ 1000×)

### 3.2 Length normalization (b = 0.75, avgdl = 100)
L(d) = 0.25 + 0.75·|d|/100
- |d| = 50: 0.25 + 0.375 = **0.625** | |d| = 100: **1.000** | |d| = 500: 0.25 + 3.75 = **4.000**
- ตัวอย่าง f = 3, k1 = 1.5 (TF part = 3·2.5 / (3 + 1.5·L)):
  - สั้น (50): 7.5 / (3 + 0.9375) = 7.5/3.9375 = **1.905**
  - เฉลี่ย (100): 7.5/4.5 = **1.667**
  - ยาว (500): 7.5 / (3 + 6) = 7.5/9 = **0.833**
  - → **คำเท่ากัน doc สั้นได้คะแนนสูงกว่า**; ceiling ยัง 2.5 ทั้งสามกรณี (f→∞ ทำให้ L ไม่สำคัญ)

### 3.3 BM25 เต็มสูตร — setup สไลด์ s.18 (deep IDF = 1.50, learning IDF = 2.50, avgdl = 100, k1 = 1.5, b = 0.75)
(ตาราง 4 scenarios ในสไลด์เป็นรูป; ตัวอย่างข้างล่างผมตั้งเอง ใช้ setup เดียวกัน) กำหนด f(deep) = 3, f(learning) = 2

**doc ยาว 100 คำ (L = 1):**
- deep: 3·2.5/(3+1.5) = 7.5/4.5 = 1.6667 → ×1.5 = **2.5000**
- learning: 2·2.5/(2+1.5) = 5/3.5 = 1.42857 → ×2.5 = **3.5714**
- รวม = **6.0714**

**doc สั้น 50 คำ (L = 0.625, k1·L = 0.9375):**
- deep: 7.5/(3+0.9375) = 7.5/3.9375 = 1.90476 → ×1.5 = **2.8571**
- learning: 5/(2+0.9375) = 5/2.9375 = 1.70213 → ×2.5 = **4.2553**
- รวม = **7.1124** (สูงกว่า doc 100 คำ ≈ +17.1%: 7.1124/6.0714 = 1.1715)

**doc ยาว 500 คำ (L = 4, k1·L = 6):**
- deep: 7.5/9 = 0.8333 → ×1.5 = **1.2500**
- learning: 5/8 = 0.625 → ×2.5 = **1.5625**
- รวม = **2.8125** (ต่ำกว่า doc 100 คำ ≈ −53.7%: 2.8125/6.0714 = 0.4632)

### 3.4 IDF — N = 100 docs
- n = 10: ln((100−10+0.5)/(10+0.5)) = ln(90.5/10.5) = ln 8.619 = **2.154**
- n = 50 (= N/2): ln(50.5/50.5) = ln 1 = **0**
- n = 90: ln(10.5/90.5) = ln 0.1160 = **−2.154** (ติดลบ; สมมาตรกับ n=10)
- Lucene ln(1 + …): n=10 → ln(9.619) = **2.264**; n=90 → ln(1.116) = **0.110** (บวกเสมอ)

### 3.5 RRF (k = 60)
- doc A: rank 1 ใน BM25, rank 3 ใน dense → 1/61 + 1/63 = 0.016393 + 0.015873 = **0.032266**
- doc B: rank 2 ใน BM25 อย่างเดียว → 1/62 = **0.016129** → A > B (ติดสองลิสต์ชนะ)
- ติด **rank 10 ในทั้งสองลิสต์** = 2/70 = **0.028571** > rank 1 ลิสต์เดียว 1/61 = 0.016393 (**Intersection > Union**)
- คะแนนสูงสุดเมื่อมี 2 ระบบ = 2/61 = **0.032787**
- เปรียบเทียบ rank 1→2: k=60: (1/61 − 1/62)/(1/61) = **1/62 = 1.61%**; k=0: (1 − 0.5)/1 = **50%**; ทั่วไป = 1/(k+2)
- **ตรวจกับ demo** (BM25 อันดับ → dense อันดับ): doc15 = rank1+rank4 = 1/61+1/64 = 0.016393+0.015625 = **0.03202** ✓; doc10 = rank3+rank2 = 1/63+1/62 = 0.015873+0.016129 = **0.03200** ✓; doc11 = rank5+rank1 = 1/65+1/61 = 0.015385+0.016393 = **0.03178** ✓

### 3.6 Min-max normalization (ทำไมอ่อนไหวต่อ outlier)
x' = (x − min)/(max − min). ตัวอย่างจาก demo: BM25 ค่า 1.241 ในช่วง [0, 1.893] → 1.241/1.893 = **0.656**; ถ้ามี outlier BM25 ตัวหนึ่ง = 10 → 1.241/10 = 0.124 (ทุกอย่างถูกบีบ) → ตัวอย่างนี้ตั้งเองเพื่ออธิบาย

### 3.7 Storage ของ embeddings (N × d × 4 bytes)
- 1,000,000 × 384 × 4 = 1.536 × 10⁹ B = **1.5 GB** (สไลด์ s.10) = 1.43 GiB (÷1.0737×10⁹)
- 1,000,000 × 3072 × 4 = 12.288 × 10⁹ B = **12.3 GB**; ≈ 11.4 GiB
- สัดส่วน 3072/384 = **8×**
- ตัวอย่างสไลด์ s.28: 2M chunks × 768 × 4 = 6.144×10⁹ = **6.1 GB** (คำนวณเอง; GB = 10⁹, GiB = 2³⁰ ตาม hint week 10)

### 3.8 Chunking stride
`chunk_size=512, overlap=64` → stride = 512−64 = **448**. ข้อความ 10,000 ตัวอักษร: จุดเริ่ม i = 0, 448, …, 9856 → ⌈10000/448⌉ = **23 chunks** (22×448 = 9856 < 10000 < 23×448)

### 3.9 Pipeline latency
- ขนาน: max(5,10) + 1 + 200 = **211 ms** (ถ้ารันอนุกรม 5+10+1+200 = 216 ms)
- ข้าม Stage 3: max(5,10)+1 = **11 ms** (คำนวณเอง)

### 3.10 CLIP softmax (ตัวอย่างตั้งเอง; τ = 0.01)
sim = [0.30, 0.25, 0.20] → logits = sim/τ = [30, 25, 20] → p(class 1) = 1/(1 + e⁻⁵ + e⁻¹⁰) = 1/(1 + 0.006738 + 0.0000454) = 1/1.006783 = **0.9933**

### 3.11 InfoNCE (ตัวอย่างตั้งเอง)
batch N = 4; logits (sim/τ): คู่ถูก = 5, negatives = 1, 1, 1 → ℓ = −ln( e⁵/(e⁵ + 3e¹) ) = ln(1 + 3e⁻⁴) = ln(1 + 0.054947) = **0.0535**

### 3.12 RoPE position interpolation
s = 16K/4K = 4 → m' = m/s: 8000 → **2000**; ตำแหน่ง 12000 → 3000; 16000 → 4000. ตัวอย่าง YaRN 4K→128K: s = 128/4 = **32** (คำนวณเอง)

### 3.13 SWA vs full attention
N = 100,000 tokens, w = 4,096: full ≈ N² = 10¹⁰ pairs; SWA ≈ N·w = 4.096×10⁸ → ถูกกว่า ≈ N/w = **24.4×** (คำนวณเอง)

### 3.14 Cross-encoder cost จาก demo
373 ms / 14 pairs = **26.6 ms/pair**; ถ้าไล่ 100 pairs ≈ 2.7 s (extrapolate เอง; สไลด์ใช้ ≈200 ms เป็นตัวอย่าง "illustrative")

---

## 4. Demo-only content (demo_w12 — ออกได้ 1-2 ข้อ)

**ภาพรวม**: ไม่ใช้ LangChain; สร้างด้วยมือ. Libraries: sentence-transformers, rank-bm25, faiss-cpu, numpy, torch, transformers, matplotlib

**Corpus & BM25**
- Corpus **30 docs, 3 topics (ML idx 0-9, Cardiac Medicine 10-19, Contract Law 20-29)**
- `BM25Okapi` จาก `rank_bm25`; tokenize ด้วย `doc.lower().split()` (whitespace ง่าย ๆ); vocabulary = **307 unique terms**
- Query **"heart attack treatment"**: BM25 ให้ **Rank 1 = Doc 15 Heart failure (1.893)**, Rank 2 = Doc 16 (1.456), Rank 3 = Doc 10 (1.421), Rank 4 = Doc 17 (1.355), **Rank 5 = Doc 11 myocardial infarction (1.241)** → doc ที่เขียนว่า "heart attack" ตรง ๆ **ได้แค่ที่ 5** เพราะ BM25 ให้รางวัล token ซ้ำ "heart" มากกว่า synonym match (= ตัวอย่าง vocabulary mismatch/lexical)
- **BM25 score range 0.000–1.893** vs **Dense 0.136–0.541** → บวกตรง ๆ ไม่ได้ (คนละสเกล)

**Dense / FAISS**
- Model `all-MiniLM-L6-v2`, dim **384**; `model.encode(..., normalize_embeddings=True)` → corpus_embeddings shape **(30, 384)**; encode corpus = **0.12 s (OFFLINE, one-time cost)**; query_emb shape **(1, 384)**
- `faiss.IndexFlatIP(d)` = **exhaustive inner-product search**; vectors normalized → **IP = cosine**; production หลายล้าน docs ใช้ **IndexIVFFlat หรือ HNSW**; `index.add(float32)`; `faiss.write_index` / `faiss.read_index` (persist แทน re-compute); `index.search(q, k)` คืน (D scores, I indices)
- Dense top-1 = **Doc 11 (0.541)**, Doc 10 (0.524), Doc 17 cardiac arrest (0.419), Doc 15 (0.393), Doc 12 (0.305). หมายเหตุ demo: **cardiac arrest ≠ heart attack ทางการแพทย์** แต่ติดอันดับสูงเพราะ semantically adjacent (dense = "hallucinated relevance"/ใกล้เคียงเชิงความหมาย)
- ผลเปรียบเทียบ top-5 (doc idx): BM25 = 15,16,10,17,11; Dense = 11,10,17,15,12

**RRF**
- `rrf_merge(results_list, k=60)`: scores[doc] += 1.0/(k + rank + 1) โดย **rank เป็น 0-indexed จึงบวก 1**; คืน list เรียง score มาก→น้อย
- RRF top: doc15 (0.03202), doc10 (0.03200), doc11 (0.03178), doc17 (0.03150), doc16 (0.03128), doc14 (0.02964)
- Theory ใน demo: "**Intersection > Union**" (doc ที่ปรากฏทั้งสอง list ถูกดัน)
- fused list = **union ของสอง top-10 = 14 docs** (ไม่ใช่ 20)

**Cross-encoder**
- `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`; input = pairs `(query, chunk)`; `reranker.predict(pairs)`
- ให้คะแนน 14 pairs ใน **0.373 s**; score range **−11.155 ถึง 2.205** → เป็น **raw logits ไม่จำกัดช่วง ติดลบได้ ไม่ใช่ probability**
- ผล rerank: Doc 11 ขึ้นจากที่ 3 → **ที่ 1** (2.205), Doc 10 คงที่ 2 (−5.854), Doc 15 ลงจากที่ 1 → ที่ 3 (−6.503), Doc 17 ที่ 4 (−6.995), Doc 12 ขึ้นจาก 7 → 5 (−9.901) → **cross-encoder แก้ผลที่ BM25 พลาด (Doc 11 กลับมาอันดับ 1)**
- Latency วัดจริง (เครื่องผู้สอน, 30 docs): BM25 **0.07 ms**, Dense (encode query + FAISS) **10.8 ms** (query encoding ครอบงำที่ corpus ขนาดเล็ก), RRF **0.039 ms**, Cross-encoder (14 pairs) **373 ms** → cross-encoder เป็นคอขวด latency

**Baseline prompt**: system prompt "Answer using ONLY the provided documents; ถ้าไม่มีให้ตอบ 'I don't have enough information.'"; context 5 docs เรียง rank 1→5, ยาว ≈ **673 ตัวอักษร**

**Bookend (Part 2)**
- Theory: LLM มี **primacy + recency bias** (Lost in the Middle)
- `bookend_order(chunks)`: ถ้า ≤ 2 คืนเดิม; ไม่งั้น `[chunks[0]] + chunks[2:] + [chunks[1]]` → **rank 1 อยู่หน้าสุด, rank 2 อยู่ท้ายสุด, rank 3..N อยู่กลาง**
- ตัวอย่าง: [1,2,3,4,5] → **[1,3,4,5,2]**
- ผลจริง: standard [Doc11, Doc10, Doc15, Doc17, Doc12] → bookend **[Doc11, Doc15, Doc17, Doc12, Doc10]** (Doc 11 ตำแหน่ง 1, Doc 10 ตำแหน่ง 5)
- (comment ใน demo อ้าง Liu et al. **2023**; สไลด์ใช้ TACL **2024** — เรื่องเดียวกัน (ปีต่างกันตาม arXiv vs. ตีพิมพ์; ตีความเอง))

---

## 5. Tagging

**[MEMORIZE]** (ต้องเข้าใจ/จำเหตุผล)
- BM25: ทำไมดีกว่า TF-IDF (saturation + length norm); ceiling = k1+1 ไม่ขึ้นกับ b; b แค่เปลี่ยนความเร็วเข้าหา ceiling; doc สั้นคะแนนสูงกว่าที่ f เท่ากัน
- IDF ลบเมื่อ n > N/2, ศูนย์ที่ N/2; Lucene ใช้ ln(1+…), rank_bm25 ใช้ floor ε·avg_idf
- Sparse vs Dense: จุดแข็ง/จุดอ่อน และเหตุผลที่ต้อง Hybrid (vocabulary mismatch vs hallucinated relevance)
- Score incompatibility → RRF (rank-based) ดีกว่า min-max (ไวต่อ outlier)
- k ใน RRF: ทำให้โค้งการตกแบนขึ้น; k เล็ก = ชัน, k ใหญ่ = แบน; k=60 มาจาก Cormack 2009 ไม่ tune ใหม่
- Retriever = recall, Reranker = precision; cross-encoder = joint attention (จับ negation) แต่ precompute ไม่ได้
- Two-stage pipeline และ latency 211 ms
- Hubness vs Anisotropy; encoder truncation เงียบที่ 256 tokens
- Chunking 3 failure modes + ลำดับ split priority
- CLIP: contrastive + symmetric InfoNCE + zero-shot ผ่าน prompt template; failure modes (relation/attribute/negation)
- ทำไม vector RAG พังที่ global/multi-hop; GraphRAG offline (index+community summaries) vs online (map-reduce)
- Local vs Global search + cost
- Lost in the Middle = U-shaped; 4 production rules; bookend; "long context ไม่ได้แทน RAG"
- RAG vs FT vs Long-Context decision rule
- Linear interpolation vs NTK-aware (บีบทุกความถี่ vs บีบเฉพาะ low freq)

**[SHEET]** (ลงกระดาษ A4) — ดูรายการลำดับความสำคัญด้านล่าง

### SHEET candidates (เรียงตามความสำคัญ, ≤ 1 บรรทัดต่อข้อ)
1. BM25 = Σ IDF · f(k1+1)/[f + k1(1−b+b|d|/avgdl)]; IDF = ln((N−n+0.5)/(n+0.5))
2. TF saturation (b=0): f(k1+1)/(f+k1); ceiling = k1+1 (=2.5 ที่ k1=1.5); f=k1 → 50% = 1.25
3. ตาราง TF (k1=1.5,b=0): f=1→1.000, 2→1.429, 3→1.667, 5→1.923, 10→2.174, 100→2.463
4. L(d) = 1−b+b|d|/avgdl; b=0.75 TREC; |d|=50,100,500 (avgdl 100) → L = 0.625, 1, 4
5. RRF = Σ 1/(k+rank); k=60; rank1=1/61=0.01639, rank2=1/62=0.01613 (−1.6%); max 2 ระบบ = 2/61 = 0.03279
6. IDF: n>N/2 → ลบ, n=N/2 → 0; Lucene ln(1+…) บวกเสมอ; rank_bm25 floor ε·avg_idf
7. Latency pipeline: max(5,10)+1+200 = 211 ms; cross-encoder rerank top K=20-100
8. Storage = N × d × 4 bytes; 1M×384 = 1.5 GB; 1M×3072 = 12.3 GB; GB=10⁹, GiB=2³⁰
9. Dim: MiniLM 384, bge-m3 1024, BERT-base 768, emb-3-small 1536, emb-3-large 3072; MiniLM truncate 256 tokens
10. Sparse (BM25: vocab mismatch) vs Dense (hallucinated relevance); BM25 ชนะ รหัส/เวอร์ชัน/ชื่อ; Dense ชนะ paraphrase
11. Bi vs Cross vs ColBERT (MaxSim, ~100× ถูกกว่า CE, คุณภาพกลาง); CE input `[CLS] q [SEP] d [SEP]`, O((|q|+|d|)²)
12. CLIP: 400M pairs, N=32,768, loss 1/(2N) symmetric InfoNCE, zero-shot ViT-L/14@336 = 76.2%, prompt ensembling +~5 pts
13. CLIP failures: relation, attribute, negation; ViT = patches + [class] token + 1D pos emb
14. GraphRAG: index (extract → graph → Leiden communities → summaries) | query (map-reduce); Local = 1 LLM call, Global = หลาย call; Leiden > Louvain (modularity Q)
15. Multi-hop = Text-to-Cypher → deterministic traversal → LLM synthesis; triple = (S, R, O); NER→RE→Entity Linking→Validation
16. Lost in Middle = U-curve; 4 rules: Reranker first, Bookend, Tune K (more ≠ better), Context distillation; bookend [1,2,3,4,5] → [1,3,4,5,2]
17. RoPE interpolation m' = m/s, s = L_target/L_train; 4K→16K: 8000→2000; NTK: บีบ low freq คง high freq; YaRN 4K→128K ≈ 400 steps (0.1%); SWA O(N·w)
18. Decision: knowledge เปลี่ยนบ่อย → RAG; style/format → FT; เอกสารเล็กคงที่ → long-context; relational/large → GraphRAG
19. Semantic chunk: dist = 1 − cos, τ = 95th percentile; split priority Chapter>Section>Paragraph>Sentence>Token; stride = size−overlap

---

## 6. Fill-in-the-blank candidates (10-15)

(2 ข้อใน week นี้ — ตัวเลขชัด ๆ และศัพท์ที่สไลด์ขีดเส้นใต้มักถูกเลือก)

1. สูตร RRF มีค่าคงที่ smoothing k โดยทั่วไป = **60** (Cormack et al., 2009)
2. TF ของ BM25 ไม่มีทางเกินเพดาน (asymptote) เท่ากับ **k1 + 1** (= 2.50 เมื่อ k1 = 1.5)
3. ที่ f(t,d) = k1 คะแนน TF จะอยู่ที่ **50%** ของเพดาน (= 1.25 เมื่อ k1 = 1.5)
4. ค่า b มาตรฐานของ TREC = **0.75**; b = **0** คือไม่มี length penalty
5. IDF ของ BM25 มีค่าเท่ากับ **0** เมื่อ n_t = **N/2** และเป็นค่าติดลบเมื่อ n_t > N/2
6. `all-MiniLM-L6-v2` truncate input ที่ **256** tokens และมี embedding dimension **384**
7. embeddings 1 ล้าน chunks ที่ d = 384 ใช้พื้นที่ ≈ **1.5 GB** (N × d × **4** bytes)
8. BM25 มี score ในช่วง **(0, ∞)** ส่วน cosine similarity อยู่ในช่วง **[−1, 1]** จึงต้องใช้ RRF
9. CLIP train ด้วย **400M** คู่ image-text, batch **N = 32,768**, ด้วย **symmetric InfoNCE** loss
10. Zero-shot CLIP ViT-L/14@336px ได้ **76.2%** Top-1 บน ImageNet (เท่า ResNet-50 supervised)
11. ColBERT ใช้ **late interaction** ด้วย **MaxSim** operator และถูกกว่า cross-encoder ≈ **2 orders of magnitude**
12. GraphRAG ใช้อัลกอริทึม **Leiden** (ปรับปรุงจาก Louvain) แบ่ง community โดย optimize **Modularity**
13. GraphRAG Global search ทำงานแบบ **Map-Reduce** เหนือ **community summaries**
14. ปรากฏการณ์ที่ LLM ใช้ข้อมูลกลาง context ได้แย่ และ accuracy เป็นรูปตัว **U** เรียกว่า **Lost in the Middle**
15. Position interpolation ขยาย 4K → 16K: s = **4**, ตำแหน่ง 8000 → **2000**; Sliding Window Attention ลด complexity จาก O(N²) เป็น **O(N·w)**
16. (bonus) ลำดับ bookend ของ [1,2,3,4,5] = **[1,3,4,5,2]**; pipeline latency ≈ **211 ms**

---

## 7. MCQ traps / misconceptions (ตัวลวงที่น่าจะเจอ)

1. **"BM25 ceiling = k1" หรือ "= k1(1−b+…)"** — ผิด: ceiling = **k1 + 1** และ **ไม่ขึ้นกับ b / ความยาว doc**. ตัวลวง: "เพิ่ม b จะทำให้ ceiling เปลี่ยน" (b เปลี่ยนแค่ความเร็วเข้าหา ceiling)
2. **"keyword ซ้ำ 1000 ครั้งได้คะแนน 1000 เท่าใน BM25"** — นั่นคือ TF-IDF; BM25 ≈ 2.5 เท่าของ f=1 ที่ k1=1.5 (saturate)
3. **"b=1 คือไม่คิดความยาว"** — กลับด้าน: b = 0 = ไม่สนความยาว, b = 1 = penalty เต็ม
4. **"IDF ไม่มีทางติดลบ"** — ติดลบได้เมื่อ n > N/2 (ในสูตรมาตรฐาน); Lucene ใช้ ln(1+…) จึงไม่ติดลบ
5. **"Dense ชนะ BM25 เสมอ"** — ผิด: BM25 ชนะ rare terms/รหัส/เวอร์ชัน; BEIR: BM25 zero-shot ชนะ DPR ส่วนใหญ่; demo: BM25 ยังได้ Doc 11 ที่ 5 แต่ dense ได้ที่ 1 (ต่างกันตาม query)
6. **"รวมคะแนน BM25 + cosine ตรง ๆ ได้ถ้า normalize"** — min-max ใช้ได้แต่ไว outlier; RRF ไม่ต้องใช้ raw scores
7. **"RRF ใช้ค่าคะแนนดิบถ่วงน้ำหนัก"** — ผิด: ใช้ **อันดับ** เท่านั้น; k ใหญ่ขึ้น → โค้งแบนขึ้น (ไม่ใช่ชันขึ้น); k=60 ถูก **fix** ใน pilot ไม่ได้ tune
8. **"Cross-encoder ใช้เป็น Stage 1 กับทั้ง corpus ได้"** — ไม่ได้ (precompute ไม่ได้, O((|q|+|d|)²) ต่อคู่); ใช้เฉพาะ top K = 20-100
9. **"Bi-encoder แม่นกว่า cross-encoder"** — กลับด้าน; bi = เร็ว/recall, cross = ช้า/precision; ColBERT อยู่กลาง
10. **"Reranker ทำเพื่อเพิ่ม recall"** — ผิด: retriever = recall, reranker = precision
11. **"Cross-encoder score เป็นความน่าจะเป็น 0-1"** — demo: เป็น raw logit −11.155…2.205
12. **"MiniLM error เมื่อ chunk เกิน 256 tokens"** — ไม่ error; truncate เงียบ ๆ ส่วนท้ายหาย (retrieve ไม่ได้)
13. **"Anisotropy = hubness"** — ต่างกัน: hubness = บาง vector เป็น NN ของ query หลายตัว; anisotropy = embeddings กระจุกในพื้นที่แคบ → cosine พอง
14. **"Structural chunking ดีกว่า fixed เสมอ"** — สไลด์เตือนให้ evaluate; overlap ไม่ได้แก้ context fragmentation ทั้งหมด
15. **"CLIP เข้าใจ composition/negation"** — ผิด: พลาด relation, attribute, negation
16. **CLIP loss ฝั่งเดียว** — ต้อง *symmetric* (image→text + text→image) และหาร 2N; ตัวลวง: "pull ทุกคู่เข้าหากัน" (จริง: คู่ทแยงเข้าหา คู่นอกทแยงผลัก)
17. **"CLIP zero-shot ต้อง fine-tune บน ImageNet"** — ไม่: ใช้ prompt template ("A photo of a {object}.") สร้าง classifier
18. **"GraphRAG Global search = 1 LLM call"** — Local ≈ 1 call; Global = map-reduce หลาย calls แพงกว่า
19. **"GraphRAG สร้าง community summary ตอน query"** — ผิด: ตอน **indexing (offline)**; query time แค่ map-reduce ข้าม summaries
20. **"Leiden = Louvain เดิม"** — Leiden รับประกัน well-connected communities (แก้ Louvain)
21. **"Vector RAG ตอบ global summary/multi-hop ได้ดีเพราะ top-k มากพอ"** — ผิด: chunk เป็นจุดแยก chain A→B→C ไม่ได้
22. **"Context ยาว (1M tokens) แทน RAG ได้"** — ผิด: cost O(N²) + lost in the middle
23. **"Lost in the Middle: ใส่ doc มากขึ้นยิ่งดี"** — more chunks ≠ better; ต้อง tune K; และ accuracy เป็น U-shape ไม่ใช่เส้นตรง
24. **Bookend สลับ**: ใน demo rank 1 = ต้น, **rank 2 = ท้าย** (ไม่ใช่ rank 3 ที่ท้าย); [1,2,3,4,5] → [1,3,4,5,2] ไม่ใช่ [1,5,4,3,2] หรือ [1,2,...]
25. **SWA เก็บ KV ทุก token** — ผิด: เก็บเฉพาะ w ล่าสุด (rolling buffer), ขนาดคงที่
26. **Linear interpolation vs NTK**: ตัวลวง "NTK บีบ high frequency" — จริง: **บีบ low, คง high** (linear บีบทุกอย่างเท่ากันและทำให้ local order เบลอ); high freq (θ ใหญ่) = ตำแหน่งใกล้; low = ตำแหน่งโดยรวม
27. **Decision rule สลับ**: "style/format ต้องเปลี่ยน → RAG" ผิด (ต้อง Fine-tune); "ความรู้เปลี่ยนบ่อย → Fine-tune" ผิด (ต้อง RAG)
28. **Lewis RAG**: retriever = non-parametric, generator = parametric (ตัวลวงสลับกัน); retrieval ใช้ **MIPS**; marginalize over top-K
29. **Dense search ใน FAISS `IndexFlatIP`** ไม่ใช่ ANN — เป็น exhaustive (แค่ corpus เล็ก); production ใช้ IVF/HNSW
30. **Hybrid "แทนที่" หนึ่งด้วยอีกหนึ่ง** — จริงคือรัน **ขนาน** แล้ว fuse (latency = max ไม่ใช่ผลรวม)

---

### ธงความขัดแย้ง / ข้อควรระวัง
- Hints กับสไลด์ **ไม่ขัดกัน** ในเรื่อง BM25 saturation (k1+1) และ pipeline BM25+dense→RRF→CE→LLM
- Hint ระบุ "bookend reordering ของ top-5 reranked passages" เป็น week 13 แต่เนื้อหาและโค้ดอยู่ใน slide/demo w12 (s.53-54, Sec 16-17) → ยังต้องอ่านเมื่อทบทวน
- สไลด์ s.39, 46 ตารางเป็นรูป: ข้อมูลที่ผมสรุปมาจากข้อความรอบข้าง ไม่ใช่ตารางต้นฉบับ → ควรเปิดสไลด์ภาพจริงดูตาราง "Query Types and RAG Suitability" และ "Decision Framework" ก่อนสอบ
- ตัวเลข "69 ชั่วโมง / 5 วินาที" (cross-encoder vs bi-encoder) อยู่ใน hint ของ week 9 ไม่ใช่สไลด์ w12

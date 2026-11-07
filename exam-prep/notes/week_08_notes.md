# Week 08 — Encoder-Decoder (BART, T5) : Study Notes

แหล่งข้อมูล: `text/slides_w08.txt` (49 หน้า, อ้าง "s.N" = สไลด์หน้า N), `text/demo_w08.txt`, `source/nlp_exam_hints.md`
ป้ายกำกับ: [MEMORIZE] = ต้องจำในหัว (เข้าใจ+เหตุผล), [SHEET] = ควรลอกลง A4, (นอกสไลด์) = ความรู้ทั่วไปของผู้เขียนโน้ต ไม่มีในสไลด์

---

## 1. Exam weight และลำดับหัวข้อ

**น้ำหนัก:** 10 MCQ (ทุกสัปดาห์ได้ส่วนแบ่งเท่ากัน) + fill-in-the-blank ประมาณ 1-2 ข้อ (ประมาณการ: 10 ข้อ / ~7 สัปดาห์; ไม่มีแหล่งยืนยันตัวเลขที่แน่นอน) ส่วนที่มีแต่ใน demo อาจออก 1-2 ข้อทั้งวิชา (ไม่ใช่ต่อสัปดาห์) ข้อสอบ Lab ไม่ออก

ภายในสัปดาห์: หัวข้อที่อาจารย์พูดเยอะ + มีเนื้อหาในสไลด์เยอะ = ออกเยอะ

| อันดับ | หัวข้อ | สไลด์ | เหตุผล / สัญญาณ |
|---|---|---|---|
| 1 | **BART 5 วิธี noising** (masking, deletion, text infilling, sentence permutation, document rotation) | s.5, 7, 10-12, 14 | Hint: "ออกแน่นอน ออกครบทุกเรื่อง" ; ~6 หน้า |
| 2 | **T5 span corruption + sentinel** (`<extra_id_N>`, target มีแต่ span ที่หาย) | s.17, 19, 21, 33 + Demo 1 | Hint: "ออกข้อสอบแน่ ๆ" ; มีโจทย์นับ token |
| 3 | **BART vs T5: ต้นทุน target/decoder** (N vs 0.2N, 25x, 2-3x) + นับ src_len/tgt_len | s.14, 17, 19, 21, 33 | Hint: "ที่ต้องจำคือ..." รวมเลขคำนวณ |
| 4 | **ทำไม decoder-only ชนะ / Enc-Dec ยังเก่งตรงไหน** (3 เหตุผล, scaling, NLLB/Whisper) | s.35-41 | 7 หน้า, ออก MCQ เชิงแนวคิดได้เยอะ |
| 5 | **T5 text-to-text + task prefix + fine-tuning (ไม่มี head)** | s.15, 16, 22, 28, 30 + Demo 2, 5 | หน้าเยอะ + demo เน้น |
| 6 | **Attention 3 แบบ / cross-attention / tensor shapes** | s.2, 4, 25, 26, 32 + Demo 3 | shape ถามได้ตรง ๆ |
| 7 | **Extractive (BERT) vs Generative (T5) QA** | s.23, 24 + Demo 4 | demo เน้น เหมาะเป็นข้อ demo-only |
| 8 | **T5 relative position bias** (32 buckets, 0-7 exact, log) | s.20 | Hint: "น่าจะออก" (แต่ไม่ออกสูตร log) |
| 9 | **MLM 15% vs CLM 100% (signal density)**, Paradigm matrix | s.2, 5, 34 | Hint: "100 tokens ใช้เทรนกี่ตัว: MLM 15, CLM 100" |
| 10 | Enc-Dec vs Dec-only FLOPs สำหรับ summarization | s.31 | มีโจทย์เลข (ตัวเลขในสไลด์) |
| 11 | BART/T5 specs (ขนาด, vocab, d_model), fine-tuning BART, mBART, UL2, Flan-T5, mT5 | s.9, 15, 21, 29, 30, 42, 45 | fill-in ได้ |

---

## 2. Core concepts

### 2.1 Recap 3 paradigms (s.2, 34) [MEMORIZE]

| | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (BART/T5) |
|---|---|---|---|
| Context | Bidirectional | Causal (left-to-right) | Bidirectional (enc) + Causal (dec) |
| Pre-training | MLM (mask 15%) | CLM (signal 100%) | Denoising (span / infilling) |
| Signal density | ~15% | 100% | ~15-100% (ขึ้นกับ corruption) |
| Generation | ไม่ native | native | native |
| Understanding | ดี (bidirectional) | อ่อนกว่า (causal) | ดี |
| Params | Encoder only | Decoder only | Encoder+Decoder (2x params) |
| Serving | ง่าย (single forward pass) | ง่าย (one model) | ซับซ้อน (two modules) |
| KV cache | N/A | ง่าย (linear growth) | ซับซ้อน (Encoder cache + Decoder cache) |
| Task flexibility | ต้องมี task-specific heads | text generation ทำทุกอย่าง | text-to-text (หรือ task heads) |

- **BERT**: คำที่ถูก mask ถูกทำนาย **อิสระต่อกัน** จึงใช้ generate ไม่สะดวก (s.2)
- **GPT**: autoregressive, condition ได้เฉพาะ context ซ้าย จึงเรียนรู้ bidirectional interaction ไม่ได้ (s.2)
- **BART/T5**: encode input แบบ bidirectional ก่อน แล้ว decode autoregressive ผ่าน **Cross-Attention** (s.3)

Attention pattern (s.2, 32):

| Encoder self-attn | Decoder self-attn | Cross-attn |
|---|---|---|
| Full bidirectional ("all ones") | Causal masked (lower-triangular) | Rectangular, **ไม่ต้อง mask**, Q จาก Dec, K/V จาก Enc |

### 2.2 ทำไมต้อง Enc-Dec (s.3) [MEMORIZE]
- งาน seq2seq: Translation, Summarization (long -> short), Generative QA
- Encoder: อ่านสองทางแต่ **generate autoregressively ไม่ได้**
- Decoder: generate ได้แต่ประมวลผล input แบบ causal (encode input เต็ม ๆ ก่อนไม่ได้)
- Enc-Dec: encode input สองทางให้ครบก่อน แล้วค่อย generate ผ่าน Cross-Attention
- Original Transformer (Vaswani 2017) **คือ** Enc-Dec (s.4)

### 2.3 Original Transformer / tensor flow (s.4) [SHEET]
- Encoder stack: Multi-Head Self-Attn -> FFN
- Decoder stack: **Masked** Multi-Head Self-Attn -> Cross-Attn -> FFN
- Encoder in/out: `[batch, src_len, d_model]`
- Decoder out: `[batch, tgt_len, d_model]` -> (linear) `[batch, tgt_len, vocab_size]`
- Cross-Attn: `CrossAttn(Q_dec, K_enc, V_enc)`

### 2.4 Denoising Autoencoder (s.5, 6) [MEMORIZE]
- ใส่ input x, corrupt ด้วย g(.) ได้ x~ = g(x), ให้โมเดล **reconstruct x จาก x~**
- Objective: **max log P(x | x~)** (PyTorch: `CrossEntropyLoss(logits, targets)`)
- ทำไมได้ผล: ต้องใช้ **Syntax** (ไวยากรณ์เติมช่อง), **Semantics** (ความหมายลึก), **Pragmatics** (world knowledge แก้ความกำกวม)
- Lineage: Vincent (2008) -> BERT MLM (2019) -> BART/T5 (2020)
- Key insight: MLM (mask 15%) คือ denoising แบบ **อ่อน ๆ**
- การแบ่งงาน: Encoder อ่าน `corrupted_text` -> `h` shape `[batch, src_len, d_model]`; Decoder อ่าน `h` + `previous_words` -> เดา next word
- เทียบ: GPT = P(next | previous); BART/T5 = P(next | previous **+ corrupted input**)
- Key distinction: Encoder input (corrupted) != Decoder target (original) -> BART/T5 ใช้ noise ได้อิสระ ไม่ต้อง aligned กัน (Figure 1c ของ BART)

### 2.5 BART (Lewis et al., Facebook AI Research) (s.9-14, 29) [MEMORIZE + SHEET specs]
- ชื่อ: **B**idirectional and **A**uto-**R**egressive **T**ransformers
- Standard Enc-Dec; pre-train ด้วย **หลาย corruption ผสมกัน** แล้ว reconstruct ข้อความเต็ม
- **Decoder เห็นและ reconstruct ข้อความต้นฉบับทั้งหมด** (full reconstruction)
- Tokenizer: **BPE (เหมือน GPT-2)**; vocab output **50265**

| | BART-base | BART-large |
|---|---|---|
| Layers | 6 Enc + 6 Dec | 12 Enc + 12 Dec |
| Params | ~140M | ~406M (สไลด์ปัดเป็น ~400M ใน s.9 / s.21) |
| d_model / heads | (ไม่ระบุ) | d_model = 1024, 16 heads |

- Tensor flow: Input `[B, src_len]` -> Encoder `[B, src_len, 1024]` -> Decoder `[B, tgt_len, 50265]`
- Training data: **160GB** (BookCorpus + Wikipedia + CC-News + OpenWebText + Stories) (s.21)

**5 วิธี corruption ของ BART** (s.5, 7, 10-12) — **ออกสอบแน่นอน**

| # | วิธี | ทำอะไร | ตัวอย่าง (สไลด์) | โมเดลเรียนรู้อะไร | src vs tgt |
|---|---|---|---|---|---|
| 1 | **Token Masking** (เหมือน BERT) | token สุ่มถูกแทนด้วย `[MASK]` | "The cat `<mask>` on the mat" -> "The cat sat on the mat" | vocabulary & local context | src_len = tgt_len |
| 2 | **Token Deletion** (**เฉพาะ BART**) | ลบ token ทิ้งเลย **ไม่เหลือ placeholder** | "The cat on the mat" -> "The cat sat on the mat" | ต้องตรวจว่า **อะไรหาย และหายที่ไหน** | src_len < tgt_len |
| 3 | **Text Infilling** (**signature ของ BART**) | span ติดกันถูกแทนด้วย `[MASK]` **ตัวเดียว** ; ความยาว span ~ **Poisson(lambda = 3)** | "The `<mask>` mat `<mask>` happily" (แทน "cat sat on the") | generation & **ประมาณความยาว** ที่หาย | src < tgt |
| 4 | **Sentence Permutation** | แบ่งเป็นประโยค แล้วสลับสุ่ม | "[S3][S1][S4][S2]" -> "[S1][S2][S3][S4]" | document structure & coherence (discourse) | |
| 5 | **Document Rotation** | เลือก token สุ่มเป็น pivot แล้วหมุนเอกสารให้เริ่มที่ pivot | "sat on the mat. The cat" -> "The cat sat on the mat." | ต้องระบุ **จุดเริ่มต้นจริง** ของเอกสาร | |

- ใน Text Infilling: `[MASK]` 1 ตัวแทน **0 หรือมากกว่า** token ; span ยาว 0 = **insertion** ของ `[MASK]` (s.11)
- Sentence Permutation + Rotation รวมกัน = สอน **global document structure** ไม่ใช่แค่ local dependency (s.12)
- หลักการออกแบบ: "ชนิดของ corruption กำหนดสิ่งที่โมเดลเรียนรู้" (s.7)
- Human analogy (s.7): Masking = ข้อสอบเติมคำ; Permutation = ประกอบเอกสารที่ถูกฉีก; Infilling/Span = เติมช่องที่ไม่รู้ความยาว

**BART Full Reconstruction flow (s.14)** [MEMORIZE]
1. Original x: "The cat sat on the mat happily" (7 tokens)
2. Corrupted x~: "happily The `<mask>` mat" (**src_len = 4**; ตัวอย่างผสม mask + หมุน)
3. Encoder output h = `[B, 4, 1024]`
4. Decoder target y = `["<s>", The, cat, sat, on, the, mat, happily, "</s>"]` (**tgt_len = 9**)
5. Loss = cross-entropy บน **ทุก** target token
- Compute signature: Encoder ประมวลผล input สั้น; **Decoder generate ข้อความเต็ม -> self-attention O(N^2)**

**Fine-tuning BART (s.29)**
- Classification: ส่ง input เข้า **ทั้ง Enc และ Dec**, ใช้ hidden state ของ **final Decoder token** -> linear classifier
- Summarization: seq2seq ตรง ๆ
- Translation (**mBART**): เติม **language tag tokens** คุมภาษา output
- BART ยังมัก **ใช้ task-specific heads** สำหรับ classification (ไม่ได้ text-to-text 100%)

### 2.6 T5 (Raffel et al., Google Research) (s.15-17, 19-21, 30) [MEMORIZE + SHEET specs]
- ชื่อ: **T**ext-**t**o-**T**ext **T**ransfer **T**ransformer
- Unifying principle: **ทุก NLP task = text in -> text out**
- สถาปัตยกรรม: Enc-Dec + **Relative Position Bias** (**ไม่มี** absolute positional encoding)
- Tokenizer: **SentencePiece, vocab 32,000**
- ขนาด: **T5-Small (60M) ถึง T5-11B**
- T5-base: **d_model = 768, d_ff = 3072, n_heads = 12** (Encoder/Decoder `[B, seq_len, 768]`) ; T5 v1.1 ใช้ d_ff = 2048 + GEGLU
- Training data: **C4 (Colossal Clean Crawled Corpus), ~750GB** (s.21)
- Task prefix:
  - Translation: `translate English to German: That is good` -> `Das ist gut`
  - Summarization: `summarize: [long article]` -> `[summary]`
  - Sentiment: `sst2 sentence: this is great` -> `positive`
  - QA: `question: What color? context: The car is red.` -> `red`
  - (รูป s.15 ยังมี `cola sentence:` -> "not acceptable" และ `stsb sentence1: ... sentence2: ...` -> ตัวเลข เช่น "3.8")
- ทำไมสำคัญ: **single loss, single decoding procedure**, multi-task = แค่ผสม dataset ที่มี prefix; BERT ต้องมี head เฉพาะต่อ task
- Fine-tuning (s.30): ทุก task ใช้ขั้นตอนเดียวกัน = prefix -> Encoder -> Decoder target text -> **Cross-Entropy**; **ไม่มี task-specific heads**
- Multi-task fine-tuning: ผสมหลาย task ใน batch เดียว; **proportional mixing** กัน dataset ใหญ่ครอบงำ
- **Flan-T5** (Chung 2022): fine-tune บน **1,836 tasks** ผ่าน instructions -> สะพานสู่ Instruction Tuning

**T5 Span Corruption (s.17)** [MEMORIZE]
1. สุ่มเลือก span ที่ **ไม่ซ้อนทับกัน** (ยาว >= 1 token)
2. แทนแต่ละ span ด้วย **sentinel เฉพาะตัว** `<extra_id_0>`, `<extra_id_1>`, ...
3. Encoder input = ข้อความที่มี sentinel
4. Decoder target = **เฉพาะ span ที่ถูกตัด** นำหน้าด้วย sentinel ที่ตรงกัน และ **จบด้วย sentinel ถัดไป + `</s>`**

ตัวอย่างสไลด์: Original "The cute cat sat on the fuzzy mat" (8 tokens)
- Input: `The <extra_id_0> cat sat on the <extra_id_1>` (src_len = 7)
- Target: `<extra_id_0> cute <extra_id_1> fuzzy mat <extra_id_2> </s>` (tgt_len = 7, vs original 8, vs BART 10)
- Sentinel สุดท้าย `<extra_id_2>` **ไม่ใช่ EOS** แปลว่า "ไม่มี span ที่หายแล้ว" ; `</s>` ตามหลัง (Demo, code 4)

**T5 vs BART: sentinel ต่างจาก [MASK]** (นัยจากสไลด์ s.11 vs s.17): BART ใช้ `[MASK]` ตัวเดียวกันทุกที่ ต้องเดาจำนวน token ที่หาย ; T5 ใช้ sentinel **unique ลำดับเลข** ทำให้ decoder รู้ว่า span ไหนคือ span ไหน

**Compute efficiency argument (s.19)** [MEMORIZE + SHEET]
- BART: target = N tokens -> decoder self-attn FLOPs O(N^2)
- T5: target ~ **0.20N** (= **15% corrupted + 5% sentinels**) -> O((0.2N)^2) = O(**0.04** N^2)
- Decoder self-attn ถูกกว่า (1/0.2)^2 = **25x**
- Overall training speedup ~ **2-3x** (เพราะ Encoder cost ยังเท่าเดิม ไม่ใช่ 25x ทั้งโมเดล)
- Takeaway: T5 เทรนบนข้อมูลมากกว่าได้ในงบ compute เท่าเดิม

**T5 Relative Position Bias (s.20)** [MEMORIZE concept; ไม่ต้องจำสูตร log]
- แทน absolute encoding ด้วย **lookup table 32 buckets ต่อ head (16 forward + 16 backward)** ที่ trainable
- Relative offset (m - n) -> ดึง learned scalar b_{m-n} **บวกตรง ๆ เข้า attention logits**
- Logarithmic bucketing: offset **0-7 = exact 1-to-1**; offset **8-128+ ใช้ bucket ร่วมกัน** (ประหยัด parameter)
- ข้อดี: parameter-efficient, ไม่ต้อง trigonometry, extrapolate ง่าย
- สูตร: a_mn = (q_m . k_n)/sqrt(d_k) + b_{m-n} ; **หมายเหตุสไลด์: ใน codebase จริงของ T5 ตัด 1/sqrt(d_k) ออก** (ใช้ initialization scaling แทน)
- Hint: สูตร log bucket **ไม่ออกสอบ** ("ยากไป") แต่ idea (ใกล้=bucket ละตัว, ไกล=แชร์) "น่าจะออก"

### 2.7 BART vs T5 สรุป (s.21) [SHEET]

| Feature | BART | T5 |
|---|---|---|
| ผู้พัฒนา | Lewis et al. (Facebook AI) | Raffel et al. (Google) |
| Architecture | Standard Enc-Dec | Enc-Dec + Relative Pos Bias |
| Corruption | Text Infilling + Sentence Perm. | Span Corruption w/ Sentinels |
| Target length | **N (เต็ม)** | **0.20N (compact)** |
| Task framing | Task heads **หรือ** generation | **ALWAYS** Text-to-Text |
| Philosophy | Maximize **generation quality** | Maximize **efficiency & generality** |
| Best at | Abstractive Summarization, Text Generation (*) | Multi-task, Translation, QA (*) |
| Sizes | 140M (base), 400M (large) | 60M ถึง 11B |
| Data | 160GB | C4 ~750GB |
| Tokenizer / Vocab | BPE (GPT-2) / 50265 | SentencePiece / 32,000 |
| Decoder start (Demo/s.14, 26) | `<s>` | `<pad>` |

(*) แถว "Best At" ในไฟล์ txt ออกมาคอลัมน์ไม่ชัด; การจับคู่ข้างบนอนุมานจาก Philosophy ในแถวก่อนหน้า ถ้าสอบแถวนี้ควรเปิดดูสไลด์ s.21 อีกครั้ง

### 2.8 QA: Extractive vs Generative (s.23-26) [MEMORIZE]

| | Extractive (BERT) | Generative (T5, Paradigm 3) |
|---|---|---|
| Output | **Start/End pointers** ใน context | Decoder **generate** ข้อความอิสระ |
| ข้อจำกัด | สังเคราะห์/paraphrase ไม่ได้ ต้องเป็น **verbatim** substring | เสี่ยง **hallucinate** |
| ข้อดี | ตอบอยู่ใน context เสมอ | สังเคราะห์คำตอบใหม่ได้ |
| Input | `[CLS] question [SEP] context [SEP]` | `question: ... context: ...` |

**BERT QA tensor flow (s.24)**
1. Input `[CLS] q [SEP] ctx [SEP]` -> Encoder -> hidden `[batch, seq_len, 768]`
2. **สอง linear heads** W_start, W_end ขนาด **768x1**
3. start logits = `(hidden @ W_start).squeeze(-1)` -> `[batch, seq_len]` ; end logits เช่นกัน
4. Answer = `tokens[argmax(start) : argmax(end)+1]`

**T5 generative QA (s.25-26)**
- Encoder: `h_enc [batch, src_len, d_model]`
- Decoder เริ่มด้วย token **`<pad>`**
- Cross-attn: Q จาก Decoder `[batch, 1, d_model]` (ต่อ step); K, V จาก Encoder `[batch, src_len, d_model]` **คำนวณครั้งเดียวและ cache**
- Loop: Step 1 input `<pad>` -> predict "red"; Step 2 input `<pad> red` -> predict `</s>`
- Loss = - sum_t log P(y_t | y_<t, h_enc)

### 2.9 Attention shapes (s.32) [SHEET]
- Encoder self-attn: `[batch, heads, src_len, src_len]` (all ones)
- Decoder masked self-attn: `[batch, heads, tgt_len, tgt_len]` (lower-triangular)
- Cross-attn: Q `[tgt_len, d_k]`, K `[src_len, d_k]`; weights `[batch, heads, tgt_len, src_len]`; output `[batch, tgt_len, d_model]`

### 2.10 Enc-Dec vs Dec-only สำหรับ summarization (s.31) [MEMORIZE verdict]
- Enc-Dec: Encoder O(L_doc^2) + Decoder O(L_sum^2 + L_sum x L_doc) (ส่วนหลังคือ cross-attention)
- Dec-only: concat เป็น sequence เดียว O((L_doc + L_sum)^2 / 2) (causal mask ลด FLOPs ครึ่งหนึ่ง)
- สมมติ param เท่ากัน: Enc-Dec N layers vs Dec-only 2N layers
- Verdict: Enc-Dec **มีประสิทธิภาพกว่าเชิงคณิตศาสตร์** เมื่อ **input ยาว output สั้น**

### 2.11 ทำไม frontier models มาเป็น Decoder-only (s.35-41) [MEMORIZE]
3 เหตุผลเชิงโครงสร้าง:
1. **Serving Simplicity & Parameter Efficiency** (s.36)
   - Enc-Dec P params = ~P/2 Encoder + ~P/2 Decoder ; Dec-only ใช้ P ทั้งหมดกับการ generate (parameter duplication problem)
   - Enc-Dec ต้อง **สอง routing pass** -> "Enc-Dec Penalty": ทำลาย continuous batching และ kernel fusion (CPU-GPU synchronization overhead) ; Dec-only = one unified loop
2. **Natural fit กับ multi-turn alignment / RLHF** (s.37)
   - Dec-only: sequence เดียว `<system>...<user>...<assistant>...` คำนวณ reward gradient ตลอด context ได้ง่าย
   - Enc-Dec: ต้อง **re-encode history ทุกเทิร์น** สัญญาณสับสน ("reward gradient ไปไหน")
3. **KV cache เติบโตเชิงเส้น คาดเดาได้** (s.38)
   - Dec-only: `[batch, seq_len, d_head]` linear ; เปิดทาง **PagedAttention (vLLM), Chunked Prefill, Prefix Caching (RadixAttention)** ซึ่งสำคัญกับ RAG
   - Enc-Dec: **สอง cache** = Static Encoder Cache (src_len) + Dynamic Decoder Cache (tgt_len โต) -> ยุ่งกับ continuous batching / memory paging

Scaling argument (s.39):
- Scale driver: ที่ **>100B params** In-Context Learning + Chain-of-Thought (single-stream) ปลดล็อก multi-step reasoning
- Wang et al. (2022): **Dec-only 4.8B vs Enc-Dec 11B** -> หลัง pre-train อย่างเดียว **Dec-only + full LM ชนะ zero-shot** ; Enc-Dec + span corruption ชนะ **หลัง multitask finetuning เท่านั้น** (ข้อได้เปรียบเป็นเงื่อนไข ไม่ฟรี)
- **ไม่มีการเปรียบเทียบควบคุมที่ >100B** (ไม่มีใครให้ทุนฝั่ง Enc-Dec)
- หลักฐานการเปลี่ยนผ่านคือ "สิ่งที่ค่ายต่าง ๆ สร้าง" ไม่ใช่ head-to-head: T5 (2020) -> PaLM (2022) -> Gemini (2023)

ที่ Enc-Dec ยังชนะ (s.40): **Machine Translation (NLLB, Meta)**, **Speech-to-Text (Whisper, OpenAI: Audio Encoder + Text Decoder)**, **Summarization** (long -> short)
- Insight: Enc-Dec ไม่ได้แพ้ แต่กลายเป็น **specialist** ; Dec-only เป็น **generalist**
- Meta-lesson (s.41): ระบบที่ simple+scalable มักชนะระบบที่ในทฤษฎีดีกว่าแต่ซับซ้อน

### 2.12 UL2 และ descendants (s.42, 45)
- **UL2** (Tay et al., 2023): ใช้ **mode tokens** ผสม objectives: **[R]-Denoising** (short span, low corruption; T5-like), **[S]-Denoising** (long prefix causal / PrefixLM; GPT-like), **[X]-Denoising** (extreme: long span / high corruption). Takeaway: **objective สำคัญพอ ๆ กับ architecture** ; Flan-UL2 ยังแข่งได้
- **mT5**: Multilingual T5 (**101 languages**) ; **Flan-T5**: instruction-tuned **1,836 tasks** ; **CodeT5+**: code ; **mBART / PEGASUS**: translation / summarization

Timeline (s.44): 2017 Transformer; 2018-19 GPT-1/2, BERT, BART; 2020 T5, GPT-3; 2022 PaLM, Whisper, UL2; 2023 LLaMA, GPT-4; 2024-26 Gemini 1.5, LLaMA-3, Qwen-2.5, Claude 3.5, DeepSeek (V2/V3/R1). DeepSeek ได้ **MLA + DeepSeekMoE** บีบ KV cache. Trend: Dec-only = แม่น้ำสายหลักที่กว้างขึ้น, Enc-Dec = สาขาย่อยเฉพาะทาง

### 2.13 BART ablation (s.13)
สไลด์ s.13 คือ Table 1 จากเปเปอร์ BART (เปรียบเทียบ pre-training objectives: Masked LM, Permuted LM, Multitask Masked LM, Masked Seq-to-Seq, + variants ของ BART บน SQuAD, MNLI, ELI5, XSum, ConvAI2, CNN/DM) แต่ข้อความ txt เละอ่านตัวเลขไม่ได้ ไม่ควรจำตัวเลขจากหน้านี้; ใจความที่ใช้ได้จากหัวข้อสไลด์ s.13/s.21 คือ BART ใช้ Text Infilling + Sentence Permutation เป็น recipe หลัก (นอกสไลด์: ในเปเปอร์ text infilling ให้ผลดีสม่ำเสมอที่สุด)

---

## 3. Worked calculations

### 3.0 สิ่งที่ต้องรู้ก่อนนับ token (ระวัง convention)
- สไลด์ s.17: `src_len` **ไม่นับ** `</s>` (7 = The, e0, cat, sat, on, the, e1); `tgt_len` **นับ** `</s>` (7) ; "BART 10" = `<s>` + 8 token + `</s>`
- Demo: นับ special tokens **ทั้งสองฝั่งเหมือนกัน** (BART: `<s>...</s>` ; T5: `...<extra_id_N> </s>`)
- โจทย์จริงน่าจะบอก convention มา ให้ตอบตามที่โจทย์กำหนด; ในโน้ตนี้แสดงทั้งสองแบบ

### 3.1 ตัวอย่างในสไลด์ s.33: นับ src_len / tgt_len [SHEET วิธีทำ]
ประโยค 12 tokens: "The quick brown fox jumps over the lazy dog in the park" (นับซ้ำ: The1 quick2 brown3 fox4 jumps5 over6 the7 lazy8 dog9 in10 the11 park12 = 12 OK). Span ที่ตัด: "quick brown" (2) และ "lazy dog" (2) = ตัด 4 token

**Scenario A: BART text infilling** (แต่ละ span -> `[MASK]` ตัวเดียว)
- Input: The [MASK] fox jumps over the [MASK] in the park
- src_len = 12 - 4 + 2 = **10** (ถ้านับ `<s>`,`</s>` = **12**)
- tgt_len = ข้อความเต็ม = **12** (ถ้านับ `<s>`,`</s>` = **14**)

**Scenario B: T5 sentinel**
- Input: The `<extra_id_0>` fox jumps over the `<extra_id_1>` in the park
- src_len = 12 - 4 + 2 = **10** (ถ้านับ `</s>` = **11**)
- Target: `<extra_id_0>` quick brown `<extra_id_1>` lazy dog `<extra_id_2>` -> 1+2+1+2+1 = **7** (ถ้านับ `</s>` = **8**)

**Q5 ratio tgt_BART / tgt_T5:**
- ไม่นับ special: 12 / 7 = **1.71** (1.714)
- นับ special ทั้งสองฝั่ง: 14 / 8 = **1.75**
- ตรวจ: 12/7 = 1.7142; 14/8 = 1.75 (ตรวจสองรอบแล้ว)

### 3.2 ตัวอย่าง Demo 1 (seed 6): ตรวจเลขที่พิมพ์ออกมา
ประโยค 13 tokens: The quick brown fox jumps over the lazy dog and runs away rapidly
- BART: `<s> The [MASK] fox jumps over the lazy dog [MASK] rapidly </s>` -> นับ 12 = src_len **12** ; target `<s>` + 13 + `</s>` = **15**
- T5 (span_len = 3 คงที่ใน demo): `The quick <extra_id_0> over the <extra_id_1> runs away rapidly </s>` = **10** ; target `<extra_id_0> brown fox jumps <extra_id_1> lazy dog and <extra_id_2> </s>` = 3+3 sentinel(3)+`</s>`(1) = **10**
- Ratio BART/T5 = 15/10 = **1.50** (ตรงกับ output)
- ข้อสังเกต: T5 span = 3+3 = 6 tokens ที่ถูกตัดของ 13 (~46% เพราะประโยคสั้นมาก ไม่ใช่ 15% จริง)

### 3.3 ตัวอย่างสไลด์ s.17 (ตรวจเลข)
- Original 8 tokens; T5 input 7; T5 target 7 (`e0 cute e1 fuzzy mat e2 </s>` = 7); BART tgt = 8 + `<s>` + `</s>` = 10 -> ratio BART/T5 = 10/7 = **1.43**

### 3.4 Compute efficiency 25x (s.19) [SHEET]
สมมติเอกสาร N = 1000 tokens
- BART target = N = 1000 -> decoder self-attn ~ N^2 = 1000^2 = **1,000,000** units (10^6)
- T5 target ~ 0.2N = **200** -> 200^2 = **40,000** units (4 x 10^4)
- อัตราส่วน = 10^6 / (4 x 10^4) = **25x** (ตรวจ: (1/0.2)^2 = 5^2 = 25)
- ดังนั้น ratio = 1/0.04 = 25 ; ระวัง **0.04 ไม่ใช่ 0.4 และไม่ใช่ 0.2**
- รวมทั้งโมเดล (Encoder เท่าเดิม) ~ **2-3x** ไม่ใช่ 25x
- ตัวอย่างนับ token: เอกสาร N = 100 -> T5 target ~ 20 (15 corrupted + 5 sentinels), 5 เท่าสั้นกว่า -> attention 25 เท่าถูกกว่า
- (นอกสไลด์) ที่มาของ "5% sentinels": noise density 15% + mean span length 3 -> 15/3 = 5 sentinels ต่อ 100 tokens

### 3.5 MLM vs CLM: "100 tokens ใช้เทรนกี่ตัว" (Hint + s.34) [SHEET]
- Dec-only CLM: ทุก token ทำนาย next -> **100** (100% signal)
- Encoder-only MLM: mask 15% -> เรียนรู้จาก **15** token (15%)
- BART/T5 decoder: เรียนจาก target ทั้งหมดที่ decoder generate: BART = N=100 (full), T5 ~ 20 (0.2N) (ช่วง ~15-100% ขึ้นกับ corruption)
- (Week 7 ที่อาจารย์พูดซ้ำ) MLM 80/10/10: จาก token ที่เลือก 15% -> 80% [MASK], 10% random token, 10% ไม่เปลี่ยน

### 3.6 Enc-Dec vs Dec-only FLOPs (s.31) [SHEET]
Doc = 1000, Sum = 100 (N layers vs 2N layers)
- Enc-Dec = N x [1000^2 + 100^2 + 100 x 1000] = N x [1,000,000 + 10,000 + 100,000] = **1,110,000 N**
- Dec-only = 2N x [(1000+100)^2 / 2] = 2N x [1,210,000 / 2] = 2N x 605,000 = **1,210,000 N**
- Dec-only แพงกว่า 1,210,000 / 1,110,000 = **1.09x** (~9% มากกว่า); ส่วนต่าง = 100,000 N = N x L_doc x L_sum
- ตรวจด้วย Doc = 2000, Sum = 200: Enc-Dec = 4,000,000 + 40,000 + 400,000 = 4,440,000 N ; Dec-only = 2 x (2200^2 / 2) = 2200^2 = 4,840,000 N ; ratio = 1.09 (เท่าเดิม)
- ระวัง: ที่มาของเลขสไลด์คือ "1100^2/2 = 605,000 แล้วคูณ 2" ไม่ใช่ 1100^2 / 2 = 605,000 เป็นคำตอบสุดท้าย

### 3.7 Attention logit กับ relative bias (s.20)
สมมติ q.k = 12.0, d_k = 64 (T5-base: 768/12 = 64) -> sqrt(64) = 8
- แบบสูตรสไลด์: a = 12.0/8 + b = 1.5 + (-0.5) = **1.0**
- แบบ T5 codebase จริง (ตัดการหาร): a = 12.0 + (-0.5) = **11.5**
- ถ้าโจทย์ให้ b_{m-n} มา **บวกเข้า logits ก่อน softmax** (ไม่ใช่คูณ ไม่ใช่บวกหลัง softmax)

### 3.8 Shape calculations [SHEET]
- Demo 3: "translate English to German: Hello world" -> src = 8 tokens (translate, English, to, German, :, Hello, world, `</s>`) ; tgt = 5 -> shapes: Enc **(8,8)**, Dec **(5,5)**, Cross **(5,8)** (rows = target, cols = source)
- ตัวอย่างสมมติ T5-base: batch 2, src_len 50, tgt_len 20, heads 12, d_model 768 -> d_k = 768/12 = 64; Cross-attn weights `[2, 12, 20, 50]`; Enc self `[2,12,50,50]`; Dec self `[2,12,20,20]`; output `[2, 20, 768]` ; logits (vocab 32,000) `[2, 20, 32000]`
- BART-large: d_k = 1024/16 = 64; logits `[B, tgt_len, 50265]`; tgt_len=9, B=1 -> 9 x 50,265 = 452,385 ค่า
- T5-base position bias: 12 heads x 32 buckets = 384 learned scalars ต่อ 1 ตาราง (นอกสไลด์: ใน T5 จริงตารางอยู่เลเยอร์แรกและแชร์ทุกเลเยอร์)

---

## 4. Demo-only content (ไม่มีหรือไม่ชัดในสไลด์)

[ตัวอย่างที่ demo มีแต่เสี่ยงออก 1-2 ข้อ: (a) ไม่มี prefix, (b) BERT ไม่มี context, (c) 0-length span]

**Demo 1: จำลอง corruption**
- BART text infilling: span length `np.random.poisson(lam=3)` ; **span ยาว 0 = INSERT `[MASK]` ที่แทน "ไม่มีอะไร"** บังคับให้โมเดลเรียนรู้ที่จะลบมันทิ้ง (ตามเปเปอร์ Lewis et al.) -- ไม่ใช่ "ข้าม"
- BART ครอบทั้ง encoder input และ target ด้วย `<s>` (BOS) และ `</s>` (EOS)
- T5 ใน demo ใช้ **fixed span_len = 3** เพื่อดูง่าย; **T5 จริงสุ่ม span length จาก Poisson(lambda = 3) เหมือน BART infilling**
- T5 ไม่เคยปล่อย sentinel ติดกัน: span คั่นด้วย token ที่เก็บไว้อย่างน้อย 1 ตัวเสมอ (non-adjacent)
- T5 จบทั้ง input และ target ด้วย `</s>` ; `<extra_id_N>` สุดท้ายใน target = "ไม่มี span หายอีกแล้ว" ไม่ใช่ EOS (ตรวจได้ `T5Tokenizer(...).input_ids[-1] == 1`)
- ผลที่พิมพ์: BART src 12 / tgt 15 ; T5 src 10 / tgt 10 ; **ratio 1.50**
- Takeaway (md 6): decoder self-attn เป็น O(N^2) ต่อ target length -> target สั้นกว่า = decoder เร็วกว่าตอน pre-training

**Demo 2: T5-small text-to-text (โมเดล/Tokenizer/decoding เดียวกันหมด ต่างแค่ prefix)**
- โหลด: `T5Tokenizer.from_pretrained('t5-small', legacy=False)`, `T5ForConditionalGeneration.from_pretrained('t5-small')`
- `generate(**inputs, max_length=50)`, `decode(..., skip_special_tokens=True)`
- ผลลัพธ์:
  - `translate English to German: The house is wonderful.` -> **"Das Haus ist wunderbar."**
  - `summarize: <article>` -> (ก๊อปประโยคกลับมาแบบ lowercase)
  - `question: ... context: ...` -> **"attention"**
  - `sst2 sentence: I absolutely loved this movie...` -> **"positive"**
  - `cola sentence: She go to school yesterday.` -> **"acceptable"** (**ผิด**; t5-small เล็กเกินไป ต้อง t5-base ขึ้นไป) ; CoLA = ตัดสิน grammatical **acceptability** (classification) ไม่ใช่ sentence **correction**
- Note ใน code: **ห้ามใช้ 'mps' (Mac GPU)** กับ T5 `generate()` เพราะ PyTorch bug ทำให้ค้าง -> ใช้ cuda/cpu (demo รันบน cpu)

**Demo 3: attention heatmap**
- `model(input_ids=..., decoder_input_ids=generated_ids, output_attentions=True)` -> ได้ `encoder_attentions`, `decoder_attentions`, `cross_attentions` (ดึง `[-1][0, 0]` = เลเยอร์สุดท้าย, batch 0, head 0)
- Shapes: **Encoder (8,8), Decoder (5,5), Cross (5,8)**
- การอ่าน heatmap: **แกน Y (แถว) = Query**, **แกน X (คอลัมน์) = Key** ; Cross-attn: Decoder (Y) ใช้ Query มองหา Encoder Keys (X)
- Encoder = full matrix (bidirectional) ; Decoder = lower-triangular (causal) ; Cross = Decoder tokens (rows) -> Encoder tokens (columns)

**Demo 4: Extractive vs Generative**
- BERT: `bert-large-uncased-whole-word-masking-finetuned-squad` + `BertForQuestionAnswering` ; โหลดแล้วเตือน `bert.pooler.*` **UNEXPECTED** = ไม่ต้องสนใจ (QA head ไม่ใช้ pooler)
- Q: "Why was the game canceled?" -> BERT start = **11**, end = **21** -> `'heavy rain that started yesterday, the field was completely flooded'` (ตัดช่วง token ตรง ๆ จาก context ; argmax ของ start_logits / end_logits)
- T5 generative: `heavy rain` (บังเอิญเป็น substring ของ context ; ไม่ได้ abstractive)
- ประเด็น (IMPORTANT ใน demo): ความต่างอยู่ที่ **MECHANISM** (pointing vs generating) ไม่ใช่ผลลัพธ์ครั้งนี้ ; BERT **ออกนอก context ไม่ได้**, T5 ออกได้แต่ไม่จำเป็นต้องออก (จึง hallucinate ได้)
- Zero-context ("What is the capital of Germany?", context ว่าง): **BERT -> `'what is the capital of germany'` (ไฮไลต์คำถามตัวเอง = ล้มเหลว)** ; **T5-small -> `'Germany'` (hallucinate เพราะ capacity เล็ก)**

**Demo 5: BART vs T5 (prefix)**
- Part A: `sst2 sentence: ...` -> "positive" ; **ไม่มี prefix -> `'Ich habe voll lieb diese Film, es war fantas fantas!'`** = T5 **ไม่ได้ "ไม่ทำอะไร"** แต่ **fallback ไปทำ translation** (prefix ที่พบบ่อยสุดใน multi-task pre-training mix) => **prefix = task switch ไม่ใช่ของตกแต่ง**
- Part B: summarization ไม่มี prefix ยัง "ดูเหมือน summary" เพราะข้อความแบบ C4 ใกล้ pre-training distribution (ตัวอย่างนี้ไม่ชัดเท่า sentiment)
- BART = **specialist**: `sshleifer/distilbart-cnn-12-6` (BartTokenizer / BartForConditionalGeneration) fine-tune เพื่อ **summarization อย่างเดียว ไม่มี prefix เลย**
- `generate` args ของ BART: `max_length=80, min_length=20, num_beams=4, length_penalty=2.0` ; T5 ใช้ `num_beams=4, max_length=80`; input `max_length=512, truncation=True`
- สรุป: T5 = generalist (prefix เลือก task), BART (มาตรฐาน) = fine-tune เฉพาะ task, ไม่มี prefix

---

## 5. SHEET candidates (เรียงตามความสำคัญ, <= 1 บรรทัด)

1. BART 5 noising: **Token Masking / Token Deletion / Text Infilling / Sentence Permutation / Document Rotation**
2. BART Text Infilling: span ~ **Poisson(lambda=3)** -> `[MASK]` **ตัวเดียว**; 0-length = insertion; ต้องเดาจำนวน token ที่หาย
3. Deletion: ไม่มี placeholder -> src < tgt; Masking: src = tgt
4. T5 span corruption: input มี `<extra_id_0..N>`; **target = เฉพาะ span + sentinel, จบด้วย `<extra_id_N+1>` `</s>`**
5. BART target = N (เต็ม) ; T5 target ~ **0.2N** (15% corrupted + 5% sentinels) -> decoder self-attn **(1/0.2)^2 = 25x** ถูกกว่า ; รวม **2-3x**
6. นับ token: infill/sentinel แทน k span -> src = n - (token ที่ตัด) + k ; T5 tgt = (token ที่ตัด) + k + 1 (sentinel จบ) (+1 ถ้านับ `</s>`)
7. **MLM 15% / CLM 100%** (100 tokens: MLM ใช้ 15, CLM ใช้ 100) ; MLM 80/10/10
8. BART: base **6+6, ~140M**; large **12+12, ~406M(~400M), d=1024, 16 heads, vocab 50265, BPE(GPT-2), 160GB**
9. T5: **60M-11B, SentencePiece 32,000, base d_model 768 / d_ff 3072 / 12 heads, C4 ~750GB, relative position bias**
10. T5 position bias: **32 buckets/head (16 fwd + 16 bwd), offset 0-7 exact, 8-128+ shared (log)**; a = qk/sqrt(d_k) + b_{m-n} (T5 จริงตัด sqrt)
11. Cross-Attn: **Q จาก Decoder; K, V จาก Encoder (cache ครั้งเดียว)**; weights `[batch, heads, tgt_len, src_len]`
12. Shapes: Enc in/out `[B, src, d_model]`; Dec logits `[B, tgt, vocab]`; BERT QA head `W_start, W_end` **768x1**; Demo: (8,8), (5,5), (5,8)
13. Enc-Dec vs Dec-only FLOPs: **N[L_doc^2 + L_sum^2 + L_sum L_doc] = 1,110,000N** vs **2N[(L_doc+L_sum)^2/2] = 1,210,000N** (Doc 1000, Sum 100)
14. Decoder-only ชนะเพราะ **(1) Serving ง่าย + param ไม่ซ้ำ (2) RLHF multi-turn (3) KV cache linear** ; Enc-Dec ยังชนะ: **MT (NLLB), ASR (Whisper), Summarization**
15. T5 decoder เริ่มด้วย **`<pad>`** ; BART decoder เริ่มด้วย `<s>`
16. T5: ไม่มี prefix = fallback ทำ translation ; ไม่มี task head ; **Flan-T5 = 1,836 tasks; mT5 = 101 languages**
17. UL2 = **[R] / [S] / [X]-Denoising**, mode tokens ; Wang 2022: **Dec-only 4.8B vs Enc-Dec 11B**, >100B ไม่มีการเทียบควบคุม
18. Extractive = start/end pointers (verbatim, ออกนอก context ไม่ได้) vs Generative = generate (hallucinate ได้)

---

## 6. Fill-in-the-blank candidates (13 ข้อ)

1. T5 ย่อมาจาก Text-to-Text ________ Transformer. (**Transfer**)
2. BART ใช้วิธี ________ ที่ span ความยาวสุ่มจาก Poisson (lambda = ____) ถูกแทนด้วย `[MASK]` ตัวเดียว. (**Text Infilling**, **3**)
3. ใน T5 span ที่ถูกตัดจะถูกแทนด้วย special token ที่เรียกว่า ________ เช่น `<extra_id_0>`. (**sentinel token**)
4. Target ของ T5 ยาวประมาณ ____ N ซึ่งทำให้ decoder self-attention ถูกกว่า BART ประมาณ ____ เท่า. (**0.20 (20%)**, **25**)
5. T5 มี target ที่ประกอบด้วย 15% corrupted tokens บวก ____% sentinels. (**5**)
6. T5 ใช้ ________ position bias แทน absolute positional encoding, โดยมี ____ buckets ต่อ head. (**Relative**, **32** (16 forward + 16 backward))
7. BART-large มี ____ Encoder layers + ____ Decoder layers และ d_model = ____ . (**12, 12, 1024**) ; BART-base = 6 + 6, ~140M
8. T5 ใช้ tokenizer ________ ขนาด vocab ____ ; BART ใช้ ________ (เหมือน GPT-2) vocab 50265. (**SentencePiece**, **32,000**, **BPE**)
9. BART วิธีที่ลบ token ออกโดยไม่เหลือ placeholder เรียกว่า ________ . (**Token Deletion**)
10. ใน Document Rotation โมเดลต้องระบุ ________ ที่แท้จริงของเอกสาร. (**จุดเริ่มต้น (start)** ; pivot = token สุ่ม)
11. MLM เรียนรู้จาก ____% ของ token ส่วน CLM เรียนรู้จาก ____% . (**15, 100**)
12. Encoder-Decoder เหมาะกับงานที่ input ________ output ________ เช่น summarization, และยังเป็น king ของ ________ (Meta) และ ________ (OpenAI, speech-to-text). (**ยาว, สั้น, NLLB, Whisper**)
13. Flan-T5 ถูก fine-tune บน ____ tasks ; mT5 รองรับ ____ ภาษา. (**1,836 ; 101**)
14. ใน Cross-Attention, Query มาจาก ________ ส่วน Key/Value มาจาก ________ . (**Decoder, Encoder**)
15. Extractive QA ของ BERT ทำนาย ________ และ ________ index ของคำตอบด้วย linear head ขนาด 768x1. (**start, end**)

---

## 7. MCQ traps / misconceptions

| # | Distractor ที่น่าจะโผล่ | ความจริง |
|---|---|---|
| 1 | "T5 decoder ต้อง reconstruct ข้อความเต็มเหมือน BART" | **ผิด** BART reconstruct เต็ม; T5 generate เฉพาะ span ที่หาย (+ sentinel) |
| 2 | "Target ของ T5 จบด้วย `<extra_id_0>`" / "จบด้วย `</s>` เลย" | จบด้วย **sentinel ถัดไป** (`<extra_id_N+1>`) **แล้วตามด้วย `</s>`**; sentinel สุดท้ายไม่ใช่ EOS |
| 3 | "BART ใช้ sentinel unique เหมือน T5" | BART ใช้ `[MASK]` ตัวเดียวกันหมด; unique sentinel เป็นของ T5 |
| 4 | "Text infilling: 1 `[MASK]` = 1 token" | 1 `[MASK]` = **0 หรือมากกว่า** token (Poisson lambda=3); โมเดลต้องเดาความยาว (นั่นต่างจาก token masking แบบ BERT) |
| 5 | "Token Deletion = แทนด้วย [MASK]" | Deletion **ไม่ทิ้ง placeholder** ; src_len < tgt_len. Masking ต่างหากที่ src = tgt |
| 6 | "Span corruption เป็นหนึ่งใน 5 วิธีของ BART" | 5 วิธีของ BART = masking, deletion, text infilling, sentence permutation, document rotation ; span corruption + sentinel = T5 |
| 7 | "T5 เร็วกว่า 25x ทั้งโมเดล" | 25x เฉพาะ **decoder self-attention** ; overall training ~ **2-3x** (Encoder เท่าเดิม) |
| 8 | "(0.2N)^2 = 0.2 N^2" หรือ "= 0.4 N^2" | = **0.04 N^2** ; ratio 1/0.04 = **25** |
| 9 | "ข้อได้เปรียบ T5 คือ encoder ถูกกว่า" | ได้เปรียบที่ **decoder/target สั้น**; ฝั่ง Encoder ใกล้เคียงกัน |
| 10 | "T5 ใช้ absolute positional encoding / sinusoidal" | T5 = **relative position bias** (learned scalar บวกเข้า attention logits) ; ไม่มี absolute |
| 11 | "Relative bias ทุก offset มี bucket ของตัวเอง" | แค่ **0-7 exact**, **8-128+ แชร์ bucket** (log) ; 32 buckets = 16 forward + 16 backward |
| 12 | "Relative bias คูณเข้ากับ logits / บวกหลัง softmax" | **บวก (additive) เข้า logits ก่อน softmax** |
| 13 | "T5 ต้องการ task-specific head ต่อ task" | **ไม่มี head**; text-in/text-out, prefix + loss เดียว. BERT ต่างหากที่ต้องมี head |
| 14 | "ไม่ใส่ prefix T5 จะเอาท์พุตเหมือน input / ไม่ทำอะไร" | จะ **fallback ไปทำ task อื่น (translation ใน demo)** ; prefix = task switch |
| 15 | "BART ต้องใส่ prefix เหมือน T5" | BART (fine-tuned) เป็น specialist ไม่มี prefix |
| 16 | "BART เป็น text-to-text 100% / ไม่ต้อง head เลย" | BART มักยังใช้ **task-specific heads** สำหรับ classification (ใช้ hidden state ของ final decoder token) |
| 17 | "Extractive QA (BERT) ตอบเรื่องนอก context ได้ถ้า fine-tune ดี" | **ทำไม่ได้เชิงโครงสร้าง** (slice index) ; ไม่มี context -> ไฮไลต์คำถามตัวเอง (demo) |
| 18 | "Generative QA ไม่มีวัน hallucinate / ตอบนอก context เสมอ" | **อาจ** hallucinate ; ไม่จำเป็นต้องออกนอก context (ผล demo `heavy rain` อยู่ใน context) ; ความต่างอยู่ที่ mechanism |
| 19 | "Cross-attn: Q, K, V มาจาก Encoder ทั้งหมด" / "Q จาก Enc K,V จาก Dec" | **Q จาก Decoder; K, V จาก Encoder** (K,V คำนวณครั้งเดียว cache) ; ไม่ต้อง mask |
| 20 | "Decoder self-attn ใน Enc-Dec เป็น full/bidirectional" | **Causal (lower-triangular)** ; Encoder self-attn เท่านั้นที่ full |
| 21 | "T5 decoder เริ่มด้วย `<s>`" | เริ่มด้วย **`<pad>`** (สไลด์ s.25-26) |
| 22 | "Decoder-only ชนะเพราะ Enc-Dec ด้อยกว่าทุกด้าน / พิสูจน์แล้วที่ >100B" | สไลด์ย้ำ: **ไม่มีการเทียบควบคุมที่ >100B**; Wang 2022: Dec-only ชนะหลัง pretrain อย่างเดียว, Enc-Dec ชนะ **หลัง multitask finetuning** ; "evidence of shift" = สิ่งที่ค่ายต่าง ๆ สร้าง ไม่ใช่ head-to-head |
| 23 | "เหตุผลที่ Dec-only ชนะคือ attention ถูกกว่า" | 3 เหตุผลคือ **serving simplicity/param efficiency, RLHF multi-turn fit, KV cache linear** |
| 24 | "Enc-Dec P params ใช้ทั้งหมด P กับการ generate" | แบ่ง ~P/2 Encoder, ~P/2 Decoder ; Dec-only ใช้ P ทั้งหมดกับ generation |
| 25 | "KV cache ของ Enc-Dec โตเชิงเส้นเหมือน Dec-only" | Enc-Dec มี **สอง cache** (static encoder + dynamic decoder) ซับซ้อนกว่า ; Dec-only linear |
| 26 | "Enc-Dec ไม่เหมาะกับงานใดเลยในปัจจุบัน" | ยังเป็น king ของ **MT (NLLB), speech-to-text (Whisper), summarization** |
| 27 | "Enc-Dec ถูกกว่า Dec-only เสมอ" | สไลด์สรุปว่าถูกกว่า **เมื่อ input ยาว output สั้น** (ตัวอย่าง 1.11M vs 1.21M N) |
| 28 | "MLM เรียนจากทุก token" | MLM เรียนจาก **~15%** ; CLM 100% |
| 29 | "UL2 เป็นโมเดลที่เปลี่ยน architecture" | UL2 = **mixture ของ denoising objectives** ผ่าน mode tokens [R]/[S]/[X] ; takeaway "objective สำคัญพอ ๆ กับ architecture" |
| 30 | "CoLA = แก้ไวยากรณ์ (correction)" | CoLA = ตัดสิน **acceptable หรือไม่** (classification) |
| 31 | "T5-small ตอบ CoLA ถูก" (demo) | t5-small ตอบ "acceptable" **ผิด** กับ "She go to school yesterday." |

---

### หมายเหตุขัดแย้ง / ความไม่ชัดเจน ระหว่างสไลด์ กับ hint
- ไม่พบจุดขัดแย้งตรง ๆ ระหว่าง hint และสไลด์ในเนื้อหาสัปดาห์นี้ (hint ย้ำ: 5 noising methods, sentinel, BART vs T5 output cost, relative position bucket ไม่เอาสูตร log)
- สไลด์ s.19 บอก T5 target ~ 0.20N เป็นค่าโดยประมาณ ตัวอย่างเล็ก ๆ (s.17: 7 vs 8) ไม่ตรง 0.2N -- โจทย์จะบอกว่าให้ใช้เลขไหน ถ้าไม่บอก ให้ใช้ 0.20N
- s.21 "Best At" ออกมาไม่ชัดในไฟล์ txt (ดูข้างบน); s.13 (ตาราง ablation) อ่านตัวเลขไม่ได้จาก txt
- ใน hint เพื่อนถอดความ: "MLM 15 / CLM 100" ตรงกับ s.34
- ไฟล์ slides txt เป็น encoding เก่า (ligature "fi/ff" หาย เช่น "Classi cation") ไม่กระทบเนื้อหา

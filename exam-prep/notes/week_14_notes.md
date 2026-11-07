# Week 14 — Data Engines, Advanced Evaluation, Pipeline Metrics (สรุปเตรียมสอบ)

แหล่งข้อมูล: `text/slides_w14.txt` (56 หน้า, "p.N" = หน้าสไลด์), `text/demo_w14.txt` (demo notebook), `source/nlp_exam_hints.md`.
หมายเหตุสำคัญ: ไฟล์ txt ของสไลด์ **ตาราง/รูปหลายหน้าหายไป** (p.6 funnel, p.8-9 tokenizer table, p.13 gauntlet, p.19 FineWeb vs Dolma, p.30 filter table, p.40 position table, p.43 GSM8K table, p.47 suite matrix, p.53 RAGAS math). ตัวเลขในหน้าเหล่านั้นที่ใช้ด้านล่างมาจาก demo / ข้อความรอบข้าง และติดป้ายไว้ ถ้าจะใช้บนแผ่น A4 ควรเปิด PDF สไลด์ตรวจซ้ำ.

---
## 1. น้ำหนักข้อสอบและลำดับหัวข้อ

**Week 14 = 10 MCQ + 2 ข้อเติมคำ** (ทุกสัปดาห์ได้ส่วนแบ่งเท่ากัน; ภายในสัปดาห์ หัวข้อที่ครูพูดมาก / สไลด์เยอะ ออกมากกว่า — hints: Week 14 `01:32:58`-`01:36:26`).

ครูไม่ได้พูด "ออกสอบ" ชัดๆ ใน Week 14 (hints บอกว่ามีน้อย) จึงจัดอันดับจากปริมาณสไลด์:

| อันดับ | หัวข้อ | สไลด์ | เหตุผล |
|---|---|---|---|
| 1 | **Deduplication**: SHA-256, shingling, Jaccard, MinHash, SimHash, LSH banding, cost 2 แกน, worked example | p.20-29 (10 หน้า) | มีสูตร+ตัวเลขเยอะ เหมาะออกเติมคำ/คำนวณ |
| 2 | **Filter pipeline**: hybrid 2-tier, Gopher, C4, CCNet perplexity, neural classifier, FineWeb/Dolma, ลำดับ cost | p.10-19, 30-31 | ตัวเลข threshold เยอะ |
| 3 | **Benchmarks**: MMLU, GSM8K, HumanEval/pass@k, contamination, LiveBench, AIME, SWE-bench, HarmBench, ชุด 2026 | p.42-51 | ตัวเลข/นิยามเยอะ, pass@k คำนวณได้ |
| 4 | **LLM-as-a-Judge**: MT-Bench, length bias, positional bias, mitigation, Chatbot Arena | p.37-41, 54 | ตัวเลข 91.3/8.7/65.0 ฯลฯ, คำถามเชิงแนวคิด |
| 5 | **BLEU/ROUGE vs BERTScore** (paradox, สูตร P/R/F, IDF) | p.33-36 | คำนวณได้ง่าย, demo ละเอียด |
| 6 | **Tokenizer fairness / TPC / ρ** | p.8-9 | ตัวเลขจาก demo (14.2x) |
| 7 | **RAGAS / DeepEval** (faithfulness ฯลฯ) | p.52-53, 54 | สไลด์สั้น แต่ demo มี |
| 8 | Scaling law / Chinchilla (20 tok/param), Common Crawl, fastText LID | p.4-7 | เรียบง่าย ออกเป็นเติมคำได้ |

**ขัดแย้ง/ข้อควรระวังจาก hints (ตามสไลด์เป็นหลัก):**
- Hints แถว Week 14 `NLP-DL-14 00:56:14` พูดถึง **multi-agent topologies (sequential/routing/parallel/hierarchical) เปรียบเทียบ 4 มิติ** แต่ **สไลด์ W14 ไม่มีเนื้อหานี้เลย** (p.2 บอกว่า multi-agent เป็น Week 13) → น่าจะเป็นวิดีโอ Week 13 ที่ตั้งชื่อไฟล์สับสน ไม่ใช่ข้อสอบ W14 อย่าเสียเวลาในโน้ตนี้.
- Hints Week 7 บอก "MinHash, SHA-256 dedup น่าจะไม่ออก (covered again in Week 14)". แต่สไลด์ W14 ให้พื้นที่ dedup มากที่สุด (10 หน้า + demo ยาว) → **ผมถือว่ายังออกได้** ควรเข้าใจ concept + สูตร Jaccard/LSH ไม่ต้องจำรายละเอียด hash เชิงลึก.
- สูตร ที่ครูบอกว่า "ไม่ต้องจำสูตร" ใน W10/W11 ไม่เกี่ยวกับ W14.

---
## 2. Core concepts

### 2.0 บทบาทใหม่: Data & QA Engineer (p.3)
5 หน้าที่: **Source** (provenance, permission) -> **Filter** (ตัด low-quality/unsafe/irrelevant) -> **Deduplicate** (ลด memorization, ประหยัด compute) -> **Evaluate** (เลือก metric ให้ตรง capability) -> **Audit** (bias, contamination, reproducibility). Analogy: architecture = สูตรอาหาร, data = วัตถุดิบ, evaluation = ชิมผล. [MEMORIZE]

### 2.1 Scaling law & Chinchilla (p.4)
- Hoffmann et al. 2022: `L(N, D) = E + A/N^alpha + B/D^beta` (สไลด์เขียนรูป E + A/N + B/D; เลขชี้กำลังหายในไฟล์ txt). N = จำนวน parameter, D = จำนวน token, **E = entropy ของ natural text (irreducible floor ที่ไม่มีโมเดลใดต่ำกว่าได้)**. [SHEET]
- Chinchilla: compute-optimal -> ขยาย **ขนาดโมเดลและจำนวน token ไปพร้อมกัน**, ratio ที่ fit ได้ **~20 tokens/parameter**. [MEMORIZE]
- ตัวอย่าง: 175B x 20 = **3.5T tokens**; GPT-3 train 300B tokens -> "data-limited" (ได้เพียง 300B/175B ~ 1.7 tok/param).
- Caveat: เป็น empirical result **ไม่ใช่กฎสากล** (data mixture, repeated epochs, architecture, optimizer, inference budget เลื่อน optimum ได้). เปรียบเทียบโมเดลต้องคุมงบ compute + คุณภาพข้อมูล; ขนาดโมเดลอย่างเดียวไม่พิสูจน์คุณภาพข้อมูล.

### 2.2 Common Crawl (p.5)
- crawl เว็บอัตโนมัติ ทุก **1-2 เดือน** ตั้งแต่ **2008**; หลายพันล้านหน้า, หลายร้อย **tebibytes** (uncompressed); อ้าง crawl index ที่ใช้.
- **WARC** = raw HTTP response (HTML markup, headers, metadata). **WET** = plain text ที่สกัดแล้ว. (บาง pipeline parse WARC เองเพื่อคุมคุณภาพ extraction)
- ฟรีบน Amazon S3 (`s3://commoncrawl/`), อ่านด้วย `warcio`. ต้นทางของ C4, FineWeb, Dolma (อย่าเดา mixture ของโมเดลที่ไม่เปิดเผย).

### 2.3 Data Contraction Funnel (p.6) และ trade-off
- เว็บมี duplicate, boilerplate, spam, หลายภาษา -> filter ลด corpus มาก. ตาราง funnel บนสไลด์เป็น "illustrative only (ไม่ใช่ตัววัดจริงของ FineWeb/Dolma)".
- แยก 2 อัตรา: **per-stage** `r_i = |D_i| / |D_{i-1}|` vs **cumulative** `|D_i| / |D_0|`. [SHEET]
- **ทุก filter = precision-recall trade-off**: เข้มงวด -> คุณภาพสูงแต่เสี่ยงทิ้งเนื้อหาการศึกษาที่หายาก. [MEMORIZE]

### 2.4 Language ID ด้วย fastText (p.7)
- `lid.176.bin` = linear classifier บน **character n-gram** features, **176 ภาษา**. สถาปัตย์: embed char n-grams -> average เป็น doc vector -> linear -> score 176 classes (n-gram range/output layer ที่แท้จริงไม่เปิดเผย).
- Output: `__label__th 0.9823` (code + confidence).
- Routing: เลือก threshold บน labeled validation; **FineWeb เก็บ English ที่ score >= 0.65**. ข้อความ low-confidence -> ส่ง fallback/review ไม่ใช่ถือว่าเป็น spam.
- ข้อจำกัด: เอกสารสั้น (**< 50 chars**) ไม่แม่น; หน้า code-heavy อาจ misclassify. [MEMORIZE: 176, 0.65, <50 chars]

### 2.5 Tokenizer fairness (p.8-9; ตัวเลขจาก demo Section 3)
- tokenizer ส่วนใหญ่ (GPT-2 BPE, Llama 2 SentencePiece) เทรนบนอังกฤษเป็นหลัก. **ไทย 1 ตัวอักษร = 3 UTF-8 bytes**; byte-level BPE ในกรณีเลวร้าย encode ทีละ byte = 1 token/byte.
- Metric: **TPC = tokens / Unicode characters** (ไม่ใช้ *fertility* = pieces/word เพราะไทยไม่มีขอบเขตคำที่เชื่อถือได้). **Rate ratio `rho = TPC_TH / TPC_EN`** (tokenizer เดียวกัน, ข้อความเทียบเท่ากัน, ต้องไม่มี [UNK]). [SHEET]
- ตัวเลข demo (1 คู่ประโยค): EN 71 chars -> 10 tokens (TPC 0.141); TH 57 chars -> 114 tokens (TPC 2.000); **rho = 14.2x** (GPT-2). ~2 tokens/ตัวอักษรไทย.
- **BERT-base-uncased**: ไทย -> 1 token = `[UNK]` (TPC 0.018, rho 0.1x) = **coverage failure ไม่ใช่ประสิทธิภาพ** ต้องเช็ก [UNK] ก่อนเทียบ TPC. [MEMORIZE]
- ผลกระทบ: token มากขึ้น = กิน context + ค่า API ตาม token แพงขึ้น. "ทำให้ model เรียนรู้แย่ลงหรือไม่" เป็นอีกข้อกล่าวอ้างที่ต้องวัดแยก. Vocab multilingual ใหญ่ (Llama 3, **128k**) ลดช่องว่าง; ตัวคูณขึ้นกับตัวอย่าง ต้องวัดบน parallel text (อ้างอิง Ahia et al. 2023).

### 2.6 Hybrid Neural-Heuristic Pipeline (p.10, 31)
- **Tier 1 Heuristic filters**: rule-based, deterministic, **~0.1 ms/doc** (Gopher, C4). **Tier 2 Learned scorers**: perplexity (CCNet) แล้ว small neural classifier **~10 ms/doc**.
- **Dedup อยู่ตรงกลาง**: หลัง heuristic ราคาถูก (ไม่ต้อง dedup ขยะ) และก่อน neural classifier (ไม่ต้องจ่ายค่า score สำเนา).
- ลำดับ Part 1 (p.31): **Language ID -> Heuristics (Gopher+C4) -> Perplexity filter -> MinHash Dedup -> Neural classifier -> Clean corpus**. [SHEET]
- สถาปัตย์ FineWeb: URL filter -> text extraction -> Language ID (fastText 0.65) -> Gopher filters -> **per-crawl MinHash dedup** -> C4 filters -> custom filters -> PII removal -> **FineWeb**; แล้ว educational classifier -> **FineWeb-Edu**.
- **Cost-ordering rule (p.10, 30)**: filter 2 ตัว A, B (cost c, retention p = fraction kept). ทำ A ก่อน ถ้า `c_A + p_A*c_B < c_B + p_B*c_A` ⇔ `c_A/(1-p_A) < c_B/(1-p_B)`. "Cheap-first = ฮิวริสติกเริ่มต้น ไม่ใช่กฎเหล็ก" (filter แพงที่คัดทิ้งเยอะมากอาจควรมาก่อน). [SHEET]

### 2.7 Gopher rules (p.11; DeepMind, MassiveText, 2021) — ต้องผ่าน **ทุกข้อ** ไม่ผ่านข้อใด = ทิ้ง
| กลุ่ม | เกณฑ์ | ตัดอะไร |
|---|---|---|
| Word count | **50 <= N <= 100,000** | ท่อนเดียว / spam ยาวระดับหนังสือ |
| Mean word length | **3 <= avg <= 10** chars | tokenizer artifact, ข้อความเละ |
| Alpha-word ratio | `r = |{w : w มีตัวอักษร}| / N >= 0.8` | ตารางสัญลักษณ์, code |
| Symbol-word ratio | `(#(#) + #(...)) / N <= 0.1` | hashtag-spam, snippet |
| Line rules | **<= 90%** ของบรรทัดขึ้นต้น bullet; **<= 30%** ลงท้าย "..." | หน้า list/feed |
| Stop-word | ต้องมี **>= 2** ของ {the, be, to, of, and, that, have, with} | ไม่ใช่ prose |
| Repetition | ~10 thresholds (duplicate line/paragraph fraction, n-gram share; เช่น dup-line 0.30) | หน้าที่ซ้ำประโยคเดียว 20 ครั้ง |
- "words" = whitespace tokens -> **ไทยต้อง segment ก่อน (เช่น PyThaiNLP) และ re-tune ทุก threshold** ค่า English ใช้ตรงๆ ไม่ได้. [MEMORIZE]
- Gopher dedup (demo comment): cut ที่ J > 0.8 บน 13-word shingles, ลบสำเนาแบบสุ่ม 1 อัน (นอกสไลด์ แต่อยู่ใน demo).

### 2.8 C4 rules (p.12; Raffel 2020 §2.2 -> T5 ใน Week 8)
- **Line-level**: เก็บเฉพาะบรรทัดที่ลงท้ายด้วย terminal punctuation (`. ! ?` หรือปิดเครื่องหมายคำพูด); ทิ้งบรรทัด **< 3 คำ**; ทิ้งบรรทัดที่มีคำ **"Javascript"**.
- **Page-level**: ทิ้งหน้าที่ **< 5 ประโยค**, มี **"lorem ipsum"**, มี **`{`**, มีคำจาก bad-word list.
- **Dedup**: เก็บสำเนาเดียวของ **span 3 ประโยค** ที่ซ้ำ.
- **Language**: `langdetect` ติดป้ายอังกฤษ **probability >= 0.99**.
- Caution: ลบ code, คณิต (`{` ใน LaTeX), กวี, non-English ที่ถูกต้อง -> ต้อง audit sample. [MEMORIZE]

### 2.9 Worked example "gauntlet" (p.13)
เพจรวมโปรโมชันท่องเที่ยวไทย (นับคำหลัง segment): **fail ที่ alpha ratio 0.58** (< 0.8) เพราะตารางราคามีตัวเลขและสัญลักษณ์ ( , %, * ). บทเรียน: heuristic ถูกบล็อกใน **< 0.1 ms** ประหยัด inference ของ neural classifier. (ตัวเลข stage อื่นเป็น illustrative.)

### 2.10 CCNet perplexity filter (p.14-15)
- แนวคิด: เทรน LM บน reference คุณภาพสูง (Wikipedia) แล้วให้คะแนนเอกสารเว็บด้วย **perplexity**. `PPL(W) = exp( -(1/N) * sum_{i=1..N} log P(w_i | w_{i-1}, ..., w_{i-n+1}) )`. **PPL ต่ำ = ทำนายง่ายสำหรับ LM ที่เทรนบน Wikipedia = คล้ายข้อความสารานุกรม.** [SHEET]
- **ไม่มี cut-off คงที่**: ต่อภาษา เรียงเอกสารตาม PPL แล้วแบ่ง **3 bucket เท่าๆ กัน (head / middle / tail)** ผู้ใช้เลือกว่าจะเก็บ bucket ไหน (demo: แบ่งตาม **จำนวนตัวอักษร** ไม่ใช่จำนวนเอกสาร; threshold = จุดที่ cumulative char mass ข้าม 1/3 และ 2/3).
- เครื่องมือ: **KenLM** = **5-gram LM (Kneser-Ney)** inference บน CPU ถูกพอจะให้คะแนนทั้ง Common Crawl. CCNet ทำที่สเกล CC เป็นครั้งแรก, pipeline ตัดเนื้อหา **84% (by size)**.
- **Perplexity Trap (p.15)**: PPL วัด "ความคล้าย reference model" **ไม่ใช่คุณภาพจริง**. spam/ข้อความซ้ำ ->  PPL ต่ำ (ถูกเก็บ); เนื้อหาเทคนิค/ความรู้หายาก -> PPL สูง (ถูกทิ้ง). ต้องใช้ร่วมกับ classifier + heuristic. [MEMORIZE]
- อ้างอิง audit: Kreutzer et al. 2022 (subset ภาษา low-resource หลายชุดติดป้ายผิด/ใช้ไม่ได้); Gururangan et al. 2022 (classifier ที่ชอบ Wikipedia/books/news เอนเอียงไปทางข้อความของคนรวย การศึกษาสูง ในเมือง).

### 2.11 Neural classifier tier / FineWeb-Edu (p.16-17)
- เป้า: ทำนายคะแนน educational value **0-5** จากข้อความ. Labels: **Llama-3-70B-Instruct** ให้คะแนน **~460k** ตัวอย่าง FineWeb ตาม rubric (0 = ไม่มีคุณค่า ... 5 = คำอธิบายแบบตำรา).
- Student: encoder เล็ก **Snowflake-arctic-embed-m (~110M params)** + **regression head** (BERT-class ใดก็ได้, Week 9).
- Threshold: **score >= 3 -> 1.3T จาก 15T tokens (~9%)**; ขอบเขตเป็นการ tune ไม่ใช่สากล. ประเมินด้วย downstream (MMLU, ARC), ภาษา, ความเร็ว, retention.
- **FineWeb** (พ.ค. 2024, HuggingFace): **15T tokens จาก 96 CC snapshots**; ออกแบบแบบ **ablation-driven** (ถอดแต่ละ filter แล้ววัด downstream) = gold standard. Public: `HuggingFaceFW/fineweb`, `fineweb-edu`, code `datatrove`.
- ผล (matched **350B-token** budget, โมเดล **1.8B**): FineWeb-Edu ชนะ FineWeb: **MMLU 33% -> 37%**, **ARC 46% -> 57%**; ถึง MMLU ของ dataset คู่แข่งด้วย token น้อยลงเกือบ **10x**. บทเรียน: **คุณภาพ > ปริมาณ**. [MEMORIZE]

### 2.12 Dolma vs FineWeb (p.18-19)
| | FineWeb | Dolma (AI2, v1.6) |
|---|---|---|
| ขนาด | 15T tokens (1.3T Edu) | ~**3T** tokens |
| แหล่ง | Common Crawl อย่างเดียว | multi-source (Table 1: Common Crawl row = Dolma CC **2,281B** + C4 **198B** tokens; + code/paper/wiki ฯลฯ) |
| จุดเด่น | web-scale อังกฤษ, pipeline ablation-driven | **โปร่งใส/reproducible ทุก tag/filter ถูก log**, audit ได้; toolkit `dolma` (taggers: language, quality, PII + mixer ผสม source ตามสัดส่วน) |
| ใช้เมื่อ | ต้องการสเกลเว็บอังกฤษสูงสุด | ต้องการ reproducibility, ความหลากหลาย code/paper/wiki, audit |
Dolma underpins **OLMo** (Allen AI, fully reproducible). Trend: multi-source + multi-signal. (ตารางหน้า 19 อ่านจากข้อความรอบข้าง)

### 2.13 Dedup: ทำไมต้อง dedup (p.20)
Syndication/copied pages -> exact + near duplicates. ความเสี่ยง: **Memorization** (recall verbatim, privacy), **Bias** (ซ้ำบ่อย = น้ำหนักเทรนมากเกิน), **Compute** (เสีย token/steps). Exact dedup = hash เอกสารที่ normalize แล้ว; Near dedup = **shingles + MinHash + LSH** เพื่อสร้าง candidates. รายงาน retention บน corpus ของตัวเอง (ไม่มี % สากล). Ref: Lee et al. 2022 (demo: 348M clusters ขนาด 1 ... 280 clusters ขนาด 5000+).

### 2.14 Exact dedup: SHA-256 (p.21)
1. normalise (strip whitespace, lowercase, ลบ HTML artefacts) 2. `.hexdigest()` = **64 ตัวอักษร hex** (หรือ `.digest()` = **32 bytes** = 256 bits) 3. หาใน hash set: มีแล้ว = duplicate ทิ้ง, ไม่มี = เพิ่มและเก็บ.
- Complexity: hashing **O(|D|)**; lookup **O(1) amortised**.
- Collision: คู่หนึ่ง `2^-256`; ทุกคู่ใน n docs `~ n(n-1) / 2^257` — เล็กมากที่ web scale.
- ข้อจำกัด: จับได้เฉพาะ **byte-for-byte เหมือนกัน** หลัง normalise (หัวข้อต่าง เนื้อเหมือน = พลาด).
- Storage: 32 bytes/doc -> **40B docs x 32 B = 1.28 TB** (lower bound; hexdigest 64 chars + overhead ของ set สูงกว่า) -> ใช้ distributed KV (Redis, DynamoDB).

### 2.15 Shingling & Jaccard (p.22)
- Shingle = **k-word sequence ซ้อนทับ**: `S(D) = {(w_i..w_{i+k-1})}`, เอกสาร N คำ -> **N - k + 1 windows**; แต่ละคำอยู่ใน <= k shingles -> shingle set ใหญ่ **~k x** เอกสารเดิม (demo: **2.8x** ที่ k=3).
- **Jaccard**: `J(A,B) = |S(A) ∩ S(B)| / |S(A) ∪ S(B)|` (สัดส่วน k-gram ที่ใช้ร่วมกัน). [SHEET]
- ถามว่า "ทำไมไม่คำนวณ intersection ตรงๆ": จริงๆ ถูก (hash set -> |A| lookups ไม่ใช่ |A|x|B|); **ปัญหาจริงคือ storage** (เก็บ shingle set ของทุกเอกสาร ขนาด k x เอกสาร) ที่ web scale เป็นไปไม่ได้. [MEMORIZE]

### 2.15b MinHash (p.23-24)
- Signature: ใช้ H hash functions: `sig(D)[i] = min_{s in S(D)} h_i(s)`. แต่ละตำแหน่งเก็บ **ค่า hash ไม่ใช่ rank** (H ค่าไม่เกี่ยวกัน ไม่เรียงลำดับ).
- `h_i` สองชั้น: `x = md5(s) mod p` (id เสถียร, shingle เดียวกัน -> x เดียวกันทุกเอกสาร) แล้ว `h_i(x) = (a_i x + b_i) mod p` (คู่ (a_i, b_i) สุ่มต่อ 1 ตำแหน่ง); **p = 2^61 - 1 เป็น prime** (ถ้า p เป็น composite ช่วงค่าจะยุบ -> Jaccard สูงเกินจริง; demo: p=12,a=4 เหลือแค่ 3 ค่า).
- **Key property**: `P[ sig(A)[i] = sig(B)[i] ] = J(A,B)` (exact เมื่อ permutation สุ่มสมบูรณ์; affine hash เป็น approximation). **ต้องใช้ (a_i, b_i) ชุดเดียวกันทั้งสองเอกสาร** (randomness ต้อง correlated): ถ้า independent sampling เลือก shingle 1 อัน/ฝั่ง จะชนด้วยโอกาส `|A∩B|/(|A||B|)` (~0.006 ในคู่ demo) เทียบกับ 0.662.
- **Estimate**: `J^ = (1/H) * sum_{i=1..H} 1[ sig(A)[i] = sig(B)[i] ]`.
- Cost: คำนวณ `H x |S(D)|` hashes **ครั้งเดียวต่อเอกสาร** แล้วเปรียบเทียบคู่ละ **H** integer comparisons ไม่ขึ้นกับความยาวเอกสาร.
- **Standard error** `= sqrt(J(1-J)/H)` , **สูงสุด `1/(2 sqrt(H))`**. ลดด้วย **H มากขึ้น ไม่ใช่ shingle มากขึ้น**; และ estimate เดี่ยวอาจพลาดเกิน 1 SE. ตัวอย่างพารามิเตอร์ในสไลด์: **k = 5, H = 128 -> SE สูงสุด ~0.044**. [SHEET]
- ภาพ p.24: Doc 9 vs Doc 10: 42/64 ตำแหน่งตรง = 0.656; (133 windows ต่อเอกสาร ที่ k=3).

### 2.16 SimHash (p.25)
- MinHash ประมาณ **Jaccard (set overlap)**, เก็บ H ตัวเลข; **SimHash รักษา cosine similarity (มุม)** เก็บ **bit string สั้นๆ ต่อเอกสาร**.
- วิธี: แต่ละ term -> random vector `v_t in {-1,+1}^64` x น้ำหนัก TF-IDF `w_t` -> รวม `sum w_t v_t` -> เก็บ **sign ของแต่ละ coordinate**.
- ความสัมพันธ์: `E[ d_H(sim(A), sim(B)) ] / d = theta / pi`, โดย `theta = arccos(cos(A,B))` (ตามสไลด์: **Hamming distance คาดหวังตามมุม ไม่ใช่ตาม cosine**); d = 64 bits (Charikar 2002).
- ปฏิบัติ (Manku et al. 2007): **64-bit fingerprints, 8B pages ของ Google crawl, d_H <= 3 = near-duplicate** (operating point ไม่ใช่นิยาม). (สไลด์พิมพ์ว่า "arccos(cos) = angle"; การหารด้วย pi เป็นนอกสไลด์ แต่เป็นรูปมาตรฐาน)

### 2.17 LSH banding (p.26-27)
- ปัญหา **O(N^2)**: 40B docs -> `C(40B, 2) ~ 8 x 10^20` คู่ — เป็นไปไม่ได้.
- แบ่ง signature ยาว H เป็น **b bands x r rows** (**H = b x r**). **candidate** ถ้าชน bucket เดียวกัน **ใน band อย่างน้อย 1 band** (ต้องตรงทั้ง r ค่าใน band นั้น).
- `P(candidate) = 1 - (1 - s^r)^b`, s = Jaccard. [SHEET]
- Threshold ประมาณ `s* ~ (1/b)^(1/r)` เป็น **ตำแหน่งช่วงเปลี่ยน (S-curve ramp) ไม่ใช่ inflection point และไม่ใช่จุด 50%**. H=64,b=16,r=4: s* = 0.500 แต่ P(candidate | s=0.5) = **0.644**, P = 0.5 ที่ s = **0.454**. อย่าถือเป็น cut-off.
- ตัวอย่างจูน: **H=256, b=64, r=4 -> s* ~ (1/64)^(1/4) ~ 0.35** (recall สูง, false positive เยอะ). **Candidate ต้องตรวจ exact Jaccard อีกที** ("candidate ไม่ใช่การพิสูจน์ duplicate").
- (ผล: เพิ่ม b -> ramp เลื่อนไปทาง s ต่ำ, เพิ่ม r -> เลื่อนไปทาง s สูง/ชันขึ้น — นอกสไลด์ แต่ตรงกับสูตร)

### 2.18 ต้นทุน dedup 2 แกน (p.28)
- Total work = (**จำนวนคู่**) x (**cost ต่อคู่**). s = shingles/doc, N = docs. **brute force = N^2 s^2 (ไม่ใช่ N^3)** เพราะ corpus ใหญ่ขึ้นไม่ได้ทำให้แต่ละเอกสารยาวขึ้น.
- **MinHash ลดคอลัมน์แรก (cost/คู่)**: 17,689 -> 64 (คู่ใน demo). **LSH ลดคอลัมน์ที่สอง (จำนวนคู่)**. ที่ N=40B: คู่ ~ 8 x 10^20 x 64 การเทียบ = MinHash อย่างเดียวยังรัน "หลายพันปี" -> **LSH ไม่ใช่ optimisation แต่คือสิ่งที่ทำให้งานจบได้**. ทั้งสองจึง **ซ้อนกัน (stack) ไม่ได้แข่งกัน**. [MEMORIZE]

### 2.19 Dedup worked example (p.29)
Doc 9 & Doc 10 (อธิบาย self-attention เหมือนกัน เปลี่ยน ~12 คำ): SHA-256 ต่างกัน -> **ไม่ถูกจับ**; shingles k=3: **exact J = 0.662**; MinHash H=64: **42/64 -> 0.656** (error **0.006**, SE = sqrt(0.662 x 0.338 / 64) = **0.059**); LSH b=16, r=4: ชนอย่างน้อย 1 band -> เป็น **candidate เดียวจาก 66 คู่** (C(12,2) = 66); verify exact; ที่ threshold J >= 0.5 ทิ้ง 1 สำเนา. ภายใต้ FineWeb (shingle **5 คำ**, 14 bands x 8 rows): J = **0.550**, โอกาสเป็น candidate แค่ **~11%** -> "likely kept ไม่ใช่ certainly" (S-curve ให้ความน่าจะเป็น ไม่ใช่การรับประกัน).

### 2.20 In-class activity: ออกแบบ filter funnel (p.30)
Q1 เรียง filter ด้วยสูตร `c_A + p_A c_B` vs `c_B + p_B c_A`; Q2 dedup ก่อน/หลัง neural classification? (คำตอบจาก p.10: **ก่อน** classifier เพราะไม่ต้องจ่ายค่า score สำเนา); Q3 filter ไหนเสี่ยงลบเนื้อหาการศึกษาหายากมากสุด (perplexity / threshold ของ classifier / C4 rules) และวิธีลดความเสี่ยง (audit sample, ดูหลาย signal, ขยับ threshold). **Part 1 สามบทเรียน (p.31)**: คุณภาพ > ปริมาณ; ไม่มี filter เดียวพอ; tokenizer fairness ต้องออกแบบตั้งแต่ต้น.

### 2.21 Part 2: 3 ยุคของ evaluation (p.32)
- ปัญหาหลัก: generative LLM มี **valid output ได้ไม่จำกัด** ต่อ input.
- **Era 1 (ก่อน 2020)**: surface string matching — BLEU, ROUGE, exact match.
- **Era 2 (2020-2024)**: semantic matching + human preference — BERTScore, Chatbot Arena, MT-Bench.
- **Era 3 (2024-2026)**: dynamic, capability-specific, adversarial — LiveBench, SWE-bench, HarmBench, RAGAS. [SHEET]

### 2.22 BLEU / ROUGE (p.33)
- **BLEU** = modified n-gram precision + brevity penalty: `BLEU = BP * exp( sum_{n=1..N} w_n log p_n )`, p_n = modified n-gram precision (clipped), BP = brevity penalty. (demo: `BP = min(1, exp(1 - |ref|/|hyp|))` เมื่อ hyp สั้นกว่า ref; = 1 ถ้ายาวเท่ากัน/มากกว่า)
- **ROUGE-L** = LCS (longest common subsequence), มัก recall-oriented. METEOR, chrF ปรับบางส่วน. ข้อจำกัด: **lexical overlap != semantic equivalence**.
- Paradox (demo): ref "The cat sat on the mat"; paraphrase "A feline rested upon the rug" -> **BLEU-1 = 0.17**; word salad "mat the on is cat the" -> **BLEU-1 = 0.83** (lower-cased). BLEU-4 ของ paraphrase = **0**. ROUGE-L: 0.167 vs 0.500.
- Reporting: ระบุ tokenization, references, smoothing, ภาษา, aggregation level.

### 2.23 BERTScore (p.34-36)
- Embed ทุก token ของ candidate `c = (c_1..c_n)` และ reference `r = (r_1..r_m)` ด้วย pre-trained BERT -> `C in R^{n x d}`, `R in R^{m x d}`.
- Similarity matrix: `S_ij = cos(c_i, r_j) = c_i . r_j / (||c_i|| ||r_j||)`, `S in R^{n x m}`, ค่า in [-1, 1].
- **Greedy matching** (ไม่ใช่ one-to-one; token ละ 1 best partner):
  - `P = (1/|c|) * sum_i max_j S_ij` , `R = (1/|r|) * sum_j max_i S_ij` , `F = 2PR / (P+R)`. [SHEET]
  - Precision = เฉลี่ย "max ตามแถว candidate"; Recall = เฉลี่ย "max ตามคอลัมน์ reference". reference token หนึ่งเป็น best match ให้ candidate หลาย token ได้.
- **IDF weighting (optional)**: `idf(w) = -log(df_w / N)` ให้ token หายากน้ำหนักมากกว่า "the", "is".
- **ไม่ใช่ percentage**: raw cosine ~0.7-0.9 แม้ประโยคไม่เกี่ยวกัน; rescale `(F-b)/(1-b)` อาจติดลบ; เทียบได้เฉพาะ setting เดียวกัน แต่ **สัมพันธ์กับมนุษย์ดีกว่า BLEU** (WMT18, Zhang Table 1).
- Demo: **F1 paraphrase 0.79 > salad 0.67** (BLEU-1 กลับกัน 0.17 < 0.83). บทเรียน: **เทียบ ranking ไม่ใช่เลขดิบ** (คนละสเกล); BERTScore วัดความคล้ายเชิงความหมาย/on-topic **ไม่ใช่ fact-check** (ประโยคที่ขัดกับข้อเท็จจริงอาจได้คะแนนสูง). [MEMORIZE]

### 2.24 Human eval: Chatbot Arena (p.37)
- LMSYS Chatbot Arena (**lmarena.ai**, เริ่ม **2023**, UC Berkeley + LMSYS). **Blind pairwise A/B**: user ใส่ prompt -> ได้ 2 คำตอบนิรนามจาก 2 โมเดลสุ่ม -> vote (หรือ tie); ชื่อโมเดลเปิดเผยหลัง vote. Output = leaderboard ด้วย **Elo / Bradley-Terry**.
- ข้อดี: คนจริง prompt จริง หัวข้อหลากหลาย **ไม่มี fixed test set ให้ leak**. ข้อเสีย: ช้า (วัน-สัปดาห์), **ไม่ reproducible**, selection bias ของ user, prompt ไม่ปลอด contamination (คนวางคำถาม benchmark/บทสนทนาที่เผยแพร่). ถ้าอ้างอันดับให้บันทึกวันที่ snapshot.

### 2.25 LLM-as-a-Judge / MT-Bench (p.38)
- **MT-Bench: 80 คำถาม multi-turn, 8 categories**, ให้ LLM ตัดสินแทน crowd -> **นาทีแทนสัปดาห์, ชุดคำถามเดิมทุกครั้ง**.
- **Pairwise mode**: judge อ่าน 2 คำตอบ เลือก 1 + เขียนเหตุผล. **Single-answer mode**: ให้คะแนน 1 คำตอบตาม rubric (ใช้เมื่อไม่มีคู่แข่ง).
- Judge prompt = **template** (คำสั่งให้เป็นผู้ตัดสิน + เกณฑ์ + รูปแบบ output ที่ parse ได้) ห่อ question(s), answer(s), และ reference answer (ถ้ามี). Zheng et al. มี **5 templates** (Figures 5-9) เลือกตาม mode / จำนวน turn / มี reference หรือไม่.
- ความตรงกับมนุษย์ **ไม่ใช่ตัวเลขคงที่** ขึ้นกับ judge model, task, protocol -> ต้องรายงานครบทั้งสาม.

### 2.26 Flaw I: Length (verbosity) bias (p.39)
- Judge ชอบคำตอบยาว โดยไม่สนคุณภาพ. ทดสอบ "repetitive list" attack (padding ด้วยการวนพูด list item เดิม ไม่เพิ่มข้อมูล), อัตราที่ judge **เลือกเวอร์ชัน padded** (Table 3, 23 answers): **Claude-v1 91.3%, GPT-3.5 91.3%, GPT-4 8.7%**. [SHEET]
- Root cause: ยาว = ดูละเอียดกว่า -> judge สับสน verbosity กับคุณภาพ; judge แข็งแรงกว่าต้านได้ดีกว่าแต่ไม่สมบูรณ์.
- ผลจริง: โมเดล "game" ได้ด้วยการเติมคำฟุ่มเฟือย ("Certainly! Great question!...").
- การตรวจจับ: correlation ของ length-vs-score ข้ามโมเดล **เป็นแค่ hint** (โมเดลเก่งก็เขียนยาว); การทดสอบที่สะอาด = **controlled: เนื้อหาเดียวกัน คำมากขึ้น**.
- Mitigation: เทียบคู่ที่ content-matched; length control เชิงสถิติ (**AlpacaEval-LC** regress ความยาวออก; "หารคะแนนด้วยจำนวน token" ไม่เท่ากัน); rubric ช่วยแต่ไม่หายขาด.

### 2.27 Flaw II: Positional bias (p.40)
- Judge ชอบคำตอบที่มาก่อน (**primacy**) แม้ตัดสินคู่เดิมสองรอบ สลับลำดับ; judge อ่อนยิ่งแย่. ("Error" = output ผิดรูป ผลรวมแถวไม่ถึง 100%)
- **Swap trick**: judge ทั้ง A-B และ B-A, **ประกาศผู้ชนะเมื่อชนะทั้งสองลำดับ**. Few-shot ยก **GPT-4 consistency 65.0% -> 77.5%** (ดีขึ้น ไม่ใช่หายขาด). (demo: Table 2 default prompt **GPT-4 consistent 65%, Claude-v1 24%**)
- ถ้าสองลำดับไม่ตรงกัน ป้ายที่ถูกคือ **"inconsistent"**; การนับเป็น tie เป็น "aggregation policy ไม่ใช่ measurement" (ต้องระบุว่าใช้อะไร).

### 2.28 Mitigating flaws (p.41) — 5 วิธี [SHEET]
**Position Swap**; **Repeated Runs** (average/aggregate ข้าม seed ที่คุม); **Rubric Anchoring** (กำหนดเกณฑ์สังเกตได้ + ตัวอย่างระดับคะแนน); **Reference Guidance** (ให้คำตอบที่เชื่อถือได้); **Judge Diversity** (หลาย judge/calibrate กับ human label). ความเสี่ยงที่เหลือ: shared bias, correlated errors ไม่หายอัตโนมัติ.

### 2.29 MMLU (p.42)
**57 วิชา, 15,908 ข้อ** multiple-choice (**4 ตัวเลือก**), **test split 14,079**; **5-shot** (5 ตัวอย่าง/วิชา) โมเดลเลือก A/B/C/D; score = accuracy. ต้องการ breadth ของความรู้โลก + แก้ปัญหา. Reporting template: model/checkpoint · MMLU version · 0/5-shot · answer extraction · tool use · score · source/date — เทียบกันได้เมื่อ protocol ตรงกัน.

### 2.30 GSM8K (p.43)
**8,500** โจทย์เลขโรงเรียนประถม multi-step; คำตอบเป็นตัวเลขเดียว ดึงหลัง marker **`####`**; scoring = **exact match** (extraction rule เป็นส่วนหนึ่งของ protocol). ปัญหา: public -> อาจอยู่ใน training data; ตรวจแค่คำตอบสุดท้าย ไม่พิสูจน์ chain ที่ถูก. **CoT** (Wei et al. 2022) คือ step-by-step ก่อนตอบ. Top models **> 95% ใกล้ ceiling** แยกโมเดล frontier ไม่ได้; ตารางคะแนนเทียบคนละ prompting protocol.

### 2.31 HumanEval & pass@k (p.44)
- **164 โจทย์ Python** (signature + docstring -> เขียน body); ต่อโจทย์มี hidden unit tests **เฉลี่ย 7.7**; ผ่านเมื่อผ่าน **ทุก test**; score = สัดส่วนโจทย์ที่แก้ได้.
- ทำไม k ครั้ง: sampling ได้โค้ดต่างกันทุกรอบ -> ถามว่า "ลอง k ครั้ง มีอย่างน้อย 1 อันผ่านโอกาสเท่าไร".
- **`pass@k = 1 - C(n-c, k) / C(n, k)`** : generate **n** samples ครั้งเดียว, **c** = จำนวนที่ผ่าน, **k** = จำนวนครั้งที่ให้ลอง; เศษส่วน = โอกาสที่ k ตัวที่สุ่ม **ล้มทั้งหมด**. เป็น unbiased estimator (สุ่มแบบไม่คืน). [SHEET]
- ต้องระบุ k เสมอ: n=200, c=2 -> pass@1 = 0.01, pass@100 ~ 0.75.
- pass@1 milestones: **GPT-3 0%** · **Codex-12B 28.8%** · **GPT-4 67.0%** · **Claude 3.5 Sonnet 92.0%**.

### 2.32 Saturation & contamination (p.45)
Saturation: ใกล้ ceiling -> แยกไม่ออก (ต้องมี uncertainty estimate จึงบอกได้ว่าสองคะแนนไม่ต่างกัน). **Contamination**: test item หรือรูปแปรผันอยู่ใน training data. ตรวจ: overlap แบบ exact/near-dup กับ training corpora; perplexity ต่ำผิดปกติ = เบาะแส **ไม่ใช่หลักฐาน**. ป้องกัน: private hold-out, ข้อสอบใหม่, ควบคุมการปล่อย. รายงาน: data cutoff, tool use, exposure ที่สงสัย.

### 2.33 ชุด benchmark 2026 (p.46-51)
- 3 failure modes ของ classic: **Static** (จำได้), **General** (ทดสอบกว้าง ไม่ลึก -> frontier saturate), **Lagging** (ไม่ครอบคลุม tool-use/long-context).
- 4 design principles: **Dynamic** (อัปเดตสม่ำเสมอ เช่น LiveBench รายเดือน), **Capability-specific**, **Ground-truth verifiable** (ตรวจด้วยโปรแกรม: math, unit tests ไม่ใช้ judge), **Contamination-resistant** (สร้างหลัง training cutoff). [MEMORIZE]
- ตาราง "Who covers what" (p.47): **มีเพียง AIME ที่ผ่านครบ 4 แกน และเฉพาะปีล่าสุด**; 15 ข้อ/paper เล็กเกินไปจะยืนเดี่ยว; RAGAS ใช้ "--" เพราะเป็น metric รันบนข้อมูลตัวเอง ไม่ใช่ชุดคำถามคงที่.

| Benchmark | ข้อเท็จจริงหลัก |
|---|---|
| **LiveBench** (p.48) | รีเฟรชคำถาม **รายเดือน**; **6 โดเมน**: math, coding, reasoning, language, instruction following, data analysis; แหล่งจากแข่งขัน/arXiv/ข่าว/datasets; ตรวจคำตอบด้วยโปรแกรมเมื่อทำได้; dynamic = กระบวนการ ไม่ใช่ประกัน contamination ถาวร. |
| **AIME** (p.49) | แข่งขัน high-school US; ปีละ 2 papers (**AIME I, II**) **15 ข้อ/paper**; คำตอบจำนวนเต็ม **000-999**; ตรวจ exact; วัด final-answer accuracy ไม่ใช่คุณภาพ reasoning ที่ซ่อน; **1 ข้อ = 6.7 pp, 2 ข้อ = 13.3 pp**; ข้อสอบเก่ามี exposure risk. |
| **SWE-bench** (p.50) | repo + issue -> สร้าง **patch**; apply แล้วรัน tests: resolved เมื่อ **tests ที่ fix ควรพลิก (fail->pass) ผ่านแล้ว และ tests ที่เคยผ่านยังผ่าน (pass->pass)**; score = % issues resolved. Splits: **Verified** (มนุษย์ตรวจ issue ชัด tests ตัดสินได้จริง), **Lite** (เล็ก ถูกกว่า); เทียบ Original/Lite/Verified ไม่ได้; ข้อมูลเป็นประวัติ GitHub สาธารณะ -> contamination risk. |
| **HarmBench** (p.51) | red-teaming มาตรฐาน; **510 behaviors (400 text + 110 multimodal), 7 categories**; **18 attack methods** (GCG, AutoDAN, PAIR, TAP...) vs **33 models/defenses**; 2 แกนแยก: สิ่งที่ขอ (behavior category) vs วิธีโจมตี. **ASR = ส่วนของ test cases (ต่อ behavior) ที่ elicit พฤติกรรมนั้น** เฉลี่ยข้าม behaviors, ตัดสินโดย **fine-tuned Llama-2-13B classifier** (ไม่ใช่คน). ASR ต่ำ = robust ต่อการโจมตีเหล่านี้ **ไม่ใช่ "safe"**; ปฏิเสธทุกอย่างได้ ASR 0 -> ต้องรายงาน normal-task performance ด้วย. |

### 2.34 RAGAS & DeepEval (p.52-53)
- ปัญหา: LLM ใน production pipeline (RAG, chatbot) จะตรวจสอบต่อเนื่องอย่าง unit test ได้อย่างไร.
- **RAGAS** (Retrieval Augmented Generation Assessment, **2023**): ประเมิน RAG **โดยไม่ต้องมี human annotation**. **DeepEval**: open-source LLM testing framework, **เชื่อมกับ pytest, รันใน CI/CD**. Paradigm shift: **LLM eval = software engineering** (assertions + CI/CD green/red).
- Es et al. paper นิยาม 3 metrics (reference-free): **faithfulness, answer relevance, context relevance**. Library รุ่นหลังเพิ่ม **context precision, context recall** (ต้องมี reference: answer / contexts / IDs แล้วแต่ variant; บาง precision variant ไม่ต้อง).
- **Faithfulness = (จำนวน claim ใน answer ที่ context สนับสนุน) / (จำนวน claim ทั้งหมดใน answer)** (นิยามจาก demo Section 20; สูตรหน้า p.53 เป็นรูปภาพ). LLM แตก answer เป็น atomic claims แล้วตรวจ entailment กับ context. **Answer relevance**: สร้างคำถามย้อนกลับ (reverse questions) จากคำตอบ แล้ววัดว่าใกล้คำถามเดิมแค่ไหน (embedding cosine).
- (นอกสไลด์) context precision = retrieved chunks ที่เกี่ยวข้องอยู่อันดับต้นหรือไม่; context recall = ข้อมูลที่ต้องใช้ตอบถูกดึงมาครบหรือไม่ — มีใน library, ไม่แน่ใจว่าสไลด์ p.53 มีสูตร.

### 2.35 Key Takeaways 5 ข้อ (p.55) [MEMORIZE]
1. คุณภาพข้อมูลคือตัวแปรที่ซ่อนอยู่ใน scaling law (Chinchilla ~20 tok/param; FineWeb-Edu = token ต้องสะอาด; Garbage In, Garbage Out).
2. ทุก filter = precision-recall trade-off; cheap heuristics ก่อน (Gopher, C4, PPL) แล้ว neural classifier แพง; PPL filter อาจลบเนื้อหาการศึกษาหายาก.
3. BERTScore จัดอันดับ paraphrase ถูกต้องกว่า BLEU (S matrix เทียบความหมาย, IDF เน้นคำสำคัญ) แต่ไม่ใช่การตรวจข้อเท็จจริง และเทียบสเกลกับ BLEU ไม่ได้.
4. LLM judge เร็วแต่ bias: length bias + positional bias; แก้ด้วย position swap, rubric anchoring, multi-judge.
5. Static benchmark ต้อง "ควบคุม" ไม่ใช่ทิ้ง: contamination checks, held-out/refreshed sets, capability-specific, uncertainty estimates, human eval.

### 2.36 In-class activity "Spot the Evaluation Flaw" (p.54)
- **Exhibit A**: Alpha ชนะ pairwise ตลอด, ความยาวเฉลี่ย **850 vs 320 tokens** -> สงสัย **length bias** (ไม่ใช่หลักฐานว่าดีกว่า).
- **Exhibit B**: Gamma ได้ 94.2% MMLU และ **12% ของข้อ MMLU อยู่ verbatim ใน pre-training** -> คำนวณ "คะแนนที่ไม่ปนเปื้อน" จากสองตัวเลขนี้ไม่ได้ (ไม่รู้ว่าโมเดลตอบข้อที่รั่วถูกกี่ %); ต้อง **re-evaluate บน clean subset** (88% ที่ไม่รั่ว) / ชุดใหม่/private held-out แล้วเทียบ.
- **Exhibit C**: RAGAS faithfulness = **0.95** แต่ user บอกตอบผิดอย่างมั่นใจ -> faithfulness บอกแค่ "ตรงกับ context ที่ดึงมา". สาเหตุอื่นที่ยังเข้ากันได้: **retrieval ผิด/ไม่ครบ (context เองผิดหรือล้าสมัย)**, **คำถามไม่ถูกตอบตรงจุด (relevance ต่ำ)**, **แหล่งข้อมูล/ความรู้ใน corpus ผิด หรือ claim extraction/judge ผิดพลาด** (ตัวอย่างสาเหตุ นอกสไลด์; สไลด์ถามให้ตั้งเอง 3 อย่าง).

---
## 3. Worked calculations (ตรวจเลขสองรอบแล้ว)

### 3.1 pass@k (สูตร p.44; ตัวเลขจาก demo Section 19)
`pass@k = 1 - C(n-c, k)/C(n, k)`. กรณีพิเศษ: **k = 1 -> pass@1 = c/n**; **c = 1 -> pass@k = k/n**.
- (ก) n=200, c=2, k=1: = 2/200 = **0.010** ✓.
- (ข) n=200, c=2, k=10: `C(198,10)/C(200,10) = (190/200)(189/199)` [เพราะ `C(n-c,k)/C(n,k) = prod_{i=0}^{c-1} (n-k-i)/(n-i)`] = 0.95 x 0.94975 = 0.90226 -> pass@10 = 1 - 0.90226 = **0.0977 ~ 0.098** ✓ (ตรง demo 0.098).
- (ค) n=200, c=2, k=100: ratio = (100/200)(99/199) = 0.5 x 0.49749 = 0.24874 -> pass@100 = **0.7513 ~ 0.751** ✓.
- (ง) n=200, c=1, k=100: = 100/200 -> **0.500** ✓. (จ) c=150, n=200, k=1: **0.750**. (ฉ) c=50, k=10: **0.948**; c=10, k=10: **0.409** (ตาราง demo).
- ตาราง demo (n=200): c=150 -> 0.750 / 1.000 / 1.000; c=50 -> 0.250 / 0.948 / 1.000; c=10 -> 0.050 / 0.409 / 0.999; c=2 -> 0.010 / 0.098 / 0.751; c=1 -> 0.005 / 0.050 / 0.500 (คอลัมน์ pass@1 / @10 / @100).
- ข้อควรระวัง: pass@k ต่าง k = ปริมาณต่างกัน ไม่ใช่ความละเอียดต่างกันของตัวเลขเดียว; n มาก -> variance ของ estimate น้อยลง (สำหรับ k เดียวกัน).

### 3.2 Jaccard และ MinHash estimate (p.22-23, 29)
- Jaccard ตัวอย่างเล็ก: A = {a,b,c,d,e}, B = {c,d,e,f}: |∩| = 3, |∪| = 6 -> J = **0.5**.
- MinHash: 42 ตำแหน่งตรงจาก 64 -> `J^ = 42/64 = 0.65625 ~ 0.656`; true J = 0.662 -> error = |0.656 - 0.662| = **0.006**.
- SE = `sqrt(J(1-J)/H)` = sqrt(0.662 x 0.338 / 64) = sqrt(0.223756/64) = sqrt(0.0034962) = **0.0591**. SE สูงสุด (J=0.5) H=64: 1/(2 x 8) = **0.0625**; H=128: 1/(2 x 11.31) = **0.0442**; ต้องการ SE ลดครึ่งต้องเพิ่ม **H เป็น 4 เท่า** (SE ∝ 1/sqrt(H)).
- Brute-force vs MinHash (demo): shingles 133 x 133 = **17,689** เทียบ vs **64** (ลด ~276 เท่า); hash-set lookup = **133**.
- (ตรวจย้อนกลับ นอกสไลด์/ไม่ได้พิมพ์ใน demo) J=0.662 กับ 133 shingles/doc สอดคล้อง |∩|=106, |∪|=160 (106/160 = 0.6625); independent sampling = 106/17,689 = 0.0060 ✓ ตรง "0.006".

### 3.3 LSH S-curve `P = 1 - (1 - s^r)^b`
- (b=16, r=4), s=0.5: s^4 = 0.0625; (1-0.0625)^16 = 0.9375^16 = 0.3560 -> P = **0.644** ✓. s* = (1/16)^(1/4) = 16^(-0.25) = 1/2 = **0.500** ✓. P = 0.5 ที่ s = 0.454 ✓ (ตรวจ: 0.454^4 = 0.04248, (0.95752)^16 = 0.4993 -> P ~ 0.5007).
- (b=16, r=4), s=0.656 (คู่ Doc 9/10): s^4 = 0.1852; (1-0.1852)^16 = 0.8148^16 = 0.0377 -> P = **0.962 ~ 0.96** ✓ (สไลด์ p.27: "at least one of 16: 0.96"; หนึ่ง band ตรงด้วยโอกาส s^4 = **0.185**).
- (b=14, r=8, FineWeb), s=0.550: s^8 = 0.008374; (1-0.008374)^14 = 0.8890 -> P = **0.111 ~ 11%** ✓.
- (b=64, r=4): s* = (1/64)^(1/4) = 1/2.828 = **0.354 ~ 0.35** ✓. (ใช้เทียบ: ที่ s = 0.35 จริง P = 0.620 — ย้ำว่า s* ไม่ใช่ 50%.)
- s ต่ำ: b=16,r=4, s=0.2 -> s^4 = 0.0016; P = 1-(0.9984)^16 = **0.025** (false candidate น้อย); s=0.8 -> P = **0.9998**.
- ตรวจ H = b x r: 16 x 4 = 64 ✓; 64 x 4 = 256 ✓; 14 x 8 = 112.

### 3.4 ขนาดการคำนวณ web-scale
- คู่เอกสาร: `C(40x10^9, 2) = 40e9 x (40e9 - 1)/2 ≈ 800e18 = 8 x 10^20` ✓ (ระวังยกกำลัง 10: 40B = 4x10^10; (4x10^10)^2/2 = 8x10^20).
- SHA-256 storage: 40 x 10^9 docs x 32 B = 1,280 x 10^9 B = **1.28 x 10^12 B = 1.28 TB** ✓.
- Chinchilla: 175 x 10^9 x 20 = 3,500 x 10^9 = **3.5 x 10^12 = 3.5T** ✓. FineWeb-Edu: 1.3/15 = 0.0867 ~ **8.7% (~9%)**.

### 3.5 TPC และ rho (p.8-9 / demo)
TPC_EN = 10/71 = 0.1408; TPC_TH = 114/57 = 2.000; **rho = 2.000/0.1408 = 14.2** ✓ (หรือ rho = (114/57)/(10/71) = (114 x 71)/(57 x 10) = 8094/570 = 14.2). BERT: TPC_TH = 1/57 = 0.0175 -> rho = 0.0175/0.1408 = 0.12 (~0.1x) แต่เป็น [UNK] -> ใช้ไม่ได้.

### 3.6 Perplexity (สูตร p.14)
`PPL = exp( -(1/N) sum log P(w_i | context) )` (log ธรรมชาติ).
- ตัวอย่าง A: LM ให้ P = 1/4 ทุก token -> `-(1/N) sum log(1/4) = ln 4 = 1.386` -> PPL = e^{1.386} = **4** (เหมือนเลือกสุ่ม 1 ใน 4 ตัวเลือก — นอกสไลด์).
- ตัวอย่าง B: ถ้า average log-prob = -3 -> PPL = e^3 = **20.1**. (demo ค่า: cookie banner PPL 4.1, code 9.2, doc3 20.0; thresholds 57.2 / 66.4 บน 5,055 chars; buckets 5/2/4 docs, share 31/20/48% ของ chars)

### 3.7 ลำดับ filter (p.10, 30) — ตัวอย่างตัวเลข (สมมติเอง ตารางหน้า 30 ไม่ปรากฏใน txt)
A = heuristic: c_A = 0.1 ms, p_A = 0.3; B = neural classifier: c_B = 10 ms, p_B = 0.5.
- A ก่อน B: `c_A + p_A c_B = 0.1 + 0.3 x 10 = 0.1 + 3 = 3.1 ms/doc`.
- B ก่อน A: `c_B + p_B c_A = 10 + 0.5 x 0.1 = 10.05 ms/doc`. -> A ก่อนชนะ (3.1 < 10.05).
- เทียบแบบ "cost/(1-p)": A: 0.1/0.7 = 0.143; B: 10/0.5 = 20 -> ตัวที่ค่าน้อยกว่าไปก่อน (สอดคล้อง). ตัวอย่างที่ filter แพงควรมาก่อน: c_A = 1, p_A = 0.95 (ทิ้งแค่ 5%) vs c_B = 2, p_B = 0.1 (ทิ้ง 90%): A ก่อน = 1 + 0.95 x 2 = 2.9; B ก่อน = 2 + 0.1 x 1 = 2.1 -> **B (แพงกว่า) มาก่อนชนะ**.

### 3.8 Funnel (demo Section 11): retention
Raw 12 -> Lang ID 10 -> Gopher 8 -> C4 6 -> PPL(head+mid) 4 -> Dedup 4.
Cumulative: 100%, 83.3%, 66.7%, 50.0%, 33.3%, 33.3%. Per-stage r: 10/12 = 0.833, 8/10 = 0.800, 6/8 = 0.750, 4/6 = 0.667, 4/4 = 1.000. (ผู้รอดชีวิต: Doc 2, 3, 5 (cookie spam!), 10)

### 3.9 BLEU-1, ROUGE-L, BERTScore (demo)
- BLEU-1 paraphrase: hyp 6 คำ, match เฉพาะ "the" (1) -> p1 = 1/6 = **0.167**; BP = 1 (ความยาวเท่า) -> BLEU-1 = 0.167.
- BLEU-1 salad: mat ✓, the ✓, on ✓, is ✗, cat ✓, the ✓ (the ใน ref มี 2 ตัว clip = 2 ✓) -> 5/6 = **0.833**.
- ถ้า hyp สั้นกว่า ref เช่น 4 คำ เทียบ ref 6: BP = exp(1 - 6/4) = e^{-0.5} = **0.607** (นอกสไลด์: สูตร BP มาตรฐานตามที่ demo ใช้).
- ROUGE-L paraphrase: LCS = 1 ("the"): R = 1/6, P = 1/6, F1 = **0.167**. Salad: LCS = 3 ("the on the" หรือ "the cat the"): R = 3/6 = 0.5, P = 3/6 = 0.5, F1 = **0.500**.
- BERTScore: `F1 = 2PR/(P+R)` ด้วย P=0.7838, R=0.7952: 2 x 0.7838 x 0.7952 = 1.2466; / 1.5790 = **0.7895** ✓. IDF-weighted: P/R/F1 = 0.7700 / 0.7817 / 0.7758 (ลดเล็กน้อย). ตัวอย่างมือ: idf = [1, 3] และ max-sim = [0.9, 0.7] -> P_idf = (1x0.9 + 3x0.7)/(1+3) = 3.0/4 = **0.75** (uniform = 0.80).
- ขนาด: candidate 7 tokens (a, fe, ##line, rested, upon, the, rug), reference 6 tokens -> `S in R^{7 x 6}`, embedding **(7, 768)** และ **(6, 768)** (DistilBERT d=768). *แต่ละ word อาจแตกหลาย wordpiece (feline -> fe + ##line) ทำให้ n != จำนวนคำ.*

### 3.10 Mock-judge arithmetic (demo S16-17) — ใช้เป็นตัวอย่างคำนวณ length bias
score = 0.6 x (จำนวนคำ) + 10 x (คำของคำถามที่ปรากฏ) + 4 (ถ้าตำแหน่ง A). Padding sweep (concise 21 คำ เป็น A): เพิ่ม 1 filler -> 30 คำ: margin B = 0.6 x (30-21) - 4 = 5.4 - 4 = **1.4** ✓; 2 fillers -> 40 คำ: 0.6 x 19 - 4 = **7.4** ✓; 3 -> 50: 0.6 x 29 - 4 = **13.4** ✓; 4 -> 61: 0.6 x 40 - 4 = **20.0** ✓. ที่ 0 filler (ข้อความเหมือนกัน) A ชนะด้วย bonus 4.0 อย่างเดียว.

### 3.11 AIME granularity
1 ข้อ = 1/15 = 6.67 pp; 2 ข้อ = 13.3 pp; ทั้งปี 2 papers = 30 ข้อ (นอกสไลด์: 1/30 = 3.3 pp ถ้ารวม).

---
## 4. Demo-only content (ไม่อยู่ในสไลด์ หรือมีรายละเอียดเพิ่ม) — โอกาสออก 1-2 ข้อ

- **Corpus 12 docs**: id 0,1,9,10 = high_edu (9 และ 10 = near-duplicate), 2,3 = mediocre, 4,5 = seo_spam (**5 = cookie banner ซ้ำในตัวเอง**), 6 = low_alpha (ตารางสัญลักษณ์), 7 = Thai, 8 = code (JS), 11 = Lorem ipsum.
- **Language ID demo**: heuristic ปลอม (non-ASCII > 25% -> th, 0.80; function words >= 3 -> en 0.92; >= 1 -> en 0.70; อื่นๆ -> unknown 0.45) + threshold 0.65 -> เก็บ 10/12 (Doc 7 ไทยตกเพราะ corpus ภาษาอังกฤษ; Doc 8 code ตกที่ unknown 0.45; ผลบางแถวถูกตัดใน output จึงอย่าท่องรายละเอียดต่อ doc).
- **Tokenizer demo**: GPT-2 ไทย 57 chars -> **114 tokens** (2.0/char), BERT-base-uncased -> **1 token [UNK]**; GPT-2 token ไทยหน้าตาเป็น byte ที่ decode ผิด (`'à¸', 'ģ', ...`).
- **Gopher demo (subset 5 rules)**: ใช้ได้ 5 จาก 7 กลุ่ม (ไม่ implement line rules + repetition rules). Output: Doc 0 passes (97 words, mean 5.61, alpha 0.97, 5 stopwords); Doc 1 (85 words); Doc 2,3 (65 words) ผ่าน; ตารางตัวเลข+โค้ดตก alpha_ratio; **Thai ตกที่ word_count และ mean_word_len** (5 "คำ" ยาว ~37 chars เพราะไม่ segment) **และ stop_words** (list ภาษาอังกฤษ) -> segmentation อย่างเดียวไม่พอ ต้องมี rule เฉพาะไทย. ผลหลัง Gopher (รวมลำดับ funnel) = 8/12 ... ตัวเลขรวมใน funnel ดู 3.8.
- **C4 demo**: ปรับ threshold ให้เหมาะกับ toy corpus (>= 5 words/line, >= 3 sentences/page; ของจริง C4 = >= 3 words, >= 5 sentences). **Collateral damage: Doc 0 (gradient descent ดี) ถูกทิ้งเพราะมี `{` ใน `theta_{t+1}`** (LaTeX). C4 ทำ "ลบบรรทัดเสีย เก็บที่เหลือ" ใน production, demo ทำแค่ keep/drop.
- **Perplexity demo**: reference LM = **DistilGPT-2** (ไม่ใช่ KenLM Wikipedia), เฉพาะเอกสาร non-Thai (11 docs). **Doc 5 (cookie banner) PPL ต่ำสุด = 4.1** (repetition ทำนายง่าย); Doc 8 = 9.2; Doc 3 = 20.0; **Doc 0, 1 (เนื้อหาคุณภาพ) ตกใน tail** เพราะคำศัพท์เทคนิคหายาก; Doc 9 กับ 10 PPL ห่างกันแค่ 0.5 แต่ตกคนละ bucket (threshold ตัดกลาง). **Terciles = convention ไม่ใช่เส้นแบ่งคุณภาพ**. KEEP = head+middle. Bucket by chars: thresholds 57.2 และ 66.4.
- **SHA-256 demo**: ไม่พบ exact duplicate ใน corpus (Doc 9/10 ไม่ซ้ำ byte-for-byte) -> `exact dedup: 12/12 retained` (สรุปจากข้อความ; แถว output ถูกตัด).
- **MinHash demo**: p = 2^61 - 1 (**Mersenne prime**); `get_shingles` lower-case + split; x = md5(shingle) mod p; ค่า sig_9[0] = sig_10[0] = 8040667816458162 (ตำแหน่ง 0 ตรงกัน), ตำแหน่ง 1 ต่าง; **42/64 ตรง**. Toy permutation: p=13, a=5, b=3 -> `h(x) = (5x+3) mod 13` เป็น permutation ของ 0..12; p=12 (composite), a=4 -> เหลือ **3 ค่า** (ยุบ). **Union-pile identities**: (1) `min over (A ∪ B) = min(sig(A)[i], sig(B)[i])` (2) `sig(A)[i] = sig(B)[i] <=> union winner ∈ A ∩ B` (ตรวจครบ 64/64) — นี่คือ proof ว่า P(match) = J. ผลลัพธ์ deterministic (seed 42, md5 คงที่) re-run ได้ 42/64 เดิมเสมอ. Cost: 64 x 133 = **8,512 hash computations** ต่อ Doc 9 (เก็บแค่ 64 ตัวเลข), จ่ายครั้งเดียว.
- **Dedup pipeline 3 steps (demo Section 10)**: (1) verify exact Jaccard >= 0.5, (2) **cluster (connected components; ถ่ายทอด A~B, B~C)**, (3) **keep-one policy** (demo เก็บ id ต่ำสุด; Gopher: J > 0.8 บน 13-word shingles ลบสุ่ม 1; C4: เก็บ 1 สำเนาไม่ระบุว่าอันไหน; Lee et al.: เก็บสำเนาใน test/validation ทิ้งใน train). ผล: 66 คู่ -> **1 candidate (Doc 9 vs 10)**, 1 confirmed, 11 clusters, ทิ้ง Doc 10, corpus 12 -> 11. **Doc 5 (cookie banner) รอด dedup เพราะซ้ำ "ภายในเอกสารเดียว" ไม่ใช่ระหว่างคู่ -> ต้องใช้ Gopher repetition rules (คนละปัญหากันที่ใช้คำว่า duplicate ร่วมกัน).**
- **ตำแหน่ง dedup เปลี่ยน "ใครรอด"**: C4 -> PPL -> dedup เก็บ Doc 10; C4 -> dedup -> PPL เก็บ Doc 9 แล้ว Doc 9 ตก PPL (ผู้รอดต่างกัน) — filter อื่นเป็น per-document จึงสลับลำดับไม่เปลี่ยนผู้รอด (เปลี่ยนแค่ต้นทุน) แต่ dedup เป็น corpus-level.
- **Funnel**: 12 -> 10 -> 8 -> 6 -> 4 -> 4. **Doc 5 ผ่านทุกด่าน (Lang, Gopher, C4, PPL, dedup)** -> "ไม่มี filter เดียวพอ" ต้องมี repetition rules/neural tier.
- **BLEU/ROUGE demo**: lower-cased ทั้งคู่; BLEU-1 0.167 vs 0.833; ROUGE-L 0.167 vs 0.500; BLEU-4 paraphrase = 0.
- **BERTScore demo**: **distilbert-base-uncased**, `add_special_tokens=False`, L2-normalise แล้ว `S = cand_norm @ ref_norm.T`. P = max ตามแกน 1 (rows=candidate) `.mean()`, R = max ตามแกน 0. **P 0.7838 / R 0.7952 / F1 0.7895**; IDF (mock idf: feline 3.8, rested 3.5, rug 3.2, cat 3.1, sat 3.0, mat 3.3, the 0.1, a 0.2, on 0.3, upon 2.1; unknown = 1.0): **0.7700 / 0.7817 / 0.7758**. Salad F1 ~ 0.67 (สไลด์ p.36). raw cosine ไม่ rescale (reference impl. rescale ด้วย random-pair baseline).
- **Mock judge**: สร้างให้ bias ตั้งใจ (score = 0.6 x words + 10 x question-keyword overlap + **POSITION_A_BONUS = 4.0**). 6 cases: **ข้อ 1-5 คำตอบยาวชนะทุกครั้ง** (margin 19.8, 40.6, 25.8, 28.8, ...) = length bias; **ข้อ 6 (คุณภาพ+ความยาวเท่ากัน) A ชนะด้วย position bonus อย่างเดียว**. Swap test: ข้อ 1-5 consistent (resp2 ชนะทั้งสองลำดับ), ข้อ 6 inconsistent -> นับเป็น tie (policy). **Swap disagreement = 1/6**. หลักการ Goodhart's law: judge ที่ให้รางวัลสถิติผิวเผินจะให้รางวัล verbosity เกินจริง.
- **pass@k implementation**: `1 - math.comb(n-c, k)/math.comb(n, k)`; edge cases: c = 0 -> 0.0; **n - c < k -> 1.0**; ต้องการ `0 <= c <= n`, `1 <= k <= n`, ทุกตัวเป็น int. ตัวอย่างข้อความ: "Even for very hard problems (c/n = 0.005), pass@100 is still non-trivial (0.5)". SWE-bench รายงาน % resolved มักจาก 1 attempt/issue.
- **RAGAS faithfulness proxy**: แยก claim ตามประโยค (> 10 chars); claim "supported" ถ้า (a) **ตัวเลขทุกตัวใน claim ต้องอยู่ใน context** และ (b) >= 50% ของ content words อยู่ใน context. Context: "The Eiffel Tower is 330 metres tall and was built in 1889 in Paris, France." Faithful answer -> 2/2 = **1.000**; Hallucinated ("500 metres", "1750 in London") -> ทั้งสอง claim UNSUPPORTED = **0.000** (สรุปจากโครงสร้าง; output ถูกตัดตรงนี้). ข้อจำกัด: ไม่เห็น negation/paraphrase ("NOT in Paris" จะผ่าน); production = LLM ตรวจ entailment แพงแต่แม่น; ต้อง calibrate กับ human/LLM judgments.
- **RAGAS answer relevance proxy**: mean-pool token embeddings (DistilBERT, max_len 128, unit norm 768-d) -> cosine ของ question vs answer: relevant **0.6891**, irrelevant (Paris) **0.5248**, delta **0.1643**. Production = LLM สร้าง reverse questions แล้ววัดความใกล้กับคำถามเดิม. (สังเกต: cosine ของ irrelevant ยังสูง ~0.52 = เหมือน BERTScore ที่ raw cosine ไม่ใช่ percentage)
- **Concept map สรุป demo**: Part 1 ต่อกัน Common Crawl (WARC/WET) -> Language ID (fastText) -> Gopher + C4 -> Perplexity (KenLM, CCNet) -> SHA-256 + MinHash/LSH -> Edu classifier (FineWeb-Edu) -> clean corpus (1.3T tokens). Part 2: Era 1 BLEU/ROUGE -> Era 2 BERTScore/Arena/Judge -> Era 3 LiveBench/SWE-bench -> Production RAGAS/DeepEval.

---
## 5. Tagging รวม (สรุป)

**[MEMORIZE] — เข้าใจให้ได้ ตอบเชิงเหตุผล**
- ทำไม perplexity filter ถึงเป็น "trap" (วัดความคล้าย reference ไม่ใช่คุณภาพ; spam ซ้ำ PPL ต่ำ, เนื้อหาเทคนิคหายาก PPL สูง).
- ทำไม dedup ต้องอยู่ "ระหว่าง" heuristic กับ neural classifier; cheap-first เป็น heuristic ไม่ใช่กฎ.
- SHA-256 จับได้แค่ exact; near-dup ต้อง shingle+MinHash+LSH; **MinHash ลดต้นทุนต่อคู่ (17,689 -> 64), LSH ลดจำนวนคู่ -> ซ้อนกัน**; ปัญหา shingle set คือ storage ไม่ใช่ intersection.
- MinHash ต้องใช้ hash function ชุดเดียวกันทั้งสองเอกสาร (correlated randomness); SE ลดด้วย H ไม่ใช่จำนวน shingle.
- LSH: s* ไม่ใช่ cutoff; candidate ต้อง verify.
- Gopher/C4 trade-off: ลบ code/LaTeX/ไทย; ไทยต้อง segment+retune.
- BLEU/ROUGE ให้ word salad สูงกว่า paraphrase; BERTScore แก้ได้แต่ไม่ใช่ fact-check, ไม่ใช่ percentage.
- LLM judge: length bias + positional bias + วิธีแก้ (swap, rubric, reference, multi-judge, repeated runs).
- Contamination vs saturation; ground-truth verifiable; contamination-resistant (LiveBench ไม่รับประกันถาวร).
- HarmBench: ASR ต่ำ != safe (ปฏิเสธทุกอย่าง = 0).
- RAGAS faithfulness สูงไม่ได้แปลว่าตอบถูก (ตรงกับ context เท่านั้น).
- Tokenizer: TPC ไม่ใช่ fertility; [UNK] ทำให้ TPC ไร้ความหมาย.

**[SHEET] candidates (เรียงลำดับความสำคัญ; ≤ 1 บรรทัด/ข้อ)**
1. `pass@k = 1 - C(n-c,k)/C(n,k)`; pass@1 = c/n; c=1 -> k/n.
2. LSH: `P = 1-(1-s^r)^b`, H=b·r, s* ≈ (1/b)^(1/r) (ไม่ใช่ cutoff); b=16,r=4: s=.5→.644, s=.656→.96.
3. Jaccard = |A∩B|/|A∪B|; MinHash Ĵ = (#ตรง)/H; SE = √(J(1−J)/H), สูงสุด 1/(2√H); 42/64 = 0.656 vs 0.662.
4. BERTScore: P = mean_i max_j S_ij, R = mean_j max_i S_ij, F = 2PR/(P+R); S_ij = cos; idf = −log(df/N).
5. Gopher: 50–100,000 words; mean 3–10; alpha ≥0.8; symbol ≤0.1; bullet ≤90%; "..." ≤30%; ≥2 stop words of 8.
6. C4: end punct, ≥3 words/line, no "Javascript", ≥5 sentences/page, no lorem ipsum / `{`, 3-sentence dedup, langdetect 0.99.
7. PPL = exp(−(1/N) Σ log P(w_i|ctx)); CCNet = 3 buckets head/middle/tail, KenLM 5-gram, ตัด 84%.
8. ลำดับ filter: A ก่อน B ถ้า c_A + p_A·c_B < c_B + p_B·c_A.
9. FineWeb 15T/96 CC; Edu ≥3 → 1.3T (~9%); MMLU 33→37, ARC 46→57 (1.8B, 350B tok); Dolma ~3T (AI2); Llama-3-70B ให้คะแนน ~460k, student 110M.
10. Chinchilla ~20 tok/param (175B→3.5T); L = E + A/N^α + B/D^β.
11. Funnel: per-stage r_i = |D_i|/|D_{i−1}| vs cumulative |D_i|/|D_0|.
12. 3 Eras: <2020 BLEU/ROUGE; 2020–24 BERTScore/Arena/MT-Bench; 2024–26 LiveBench/SWE-bench/HarmBench/RAGAS.
13. Judge bias: length padded-preferred Claude-v1 91.3 / GPT-3.5 91.3 / GPT-4 8.7; position GPT-4 65.0→77.5 w/ few-shot; swap A-B & B-A.
14. TPC = tokens/chars; ρ = TPC_TH/TPC_EN; GPT-2 TH 114 tok/57 ch = 2.0 → ρ 14.2×; BERT → [UNK].
15. Benchmarks: MMLU 57 subj/15,908 q/5-shot; GSM8K 8,500 `####`; HumanEval 164 / 7.7 tests; AIME 15/paper, 000–999; HarmBench 510 (400+110), 7 cat, 18 attacks, 33 models.
16. HumanEval pass@1: GPT-3 0 / Codex-12B 28.8 / GPT-4 67.0 / Claude 3.5 Sonnet 92.0 (%).
17. SHA-256: 256 bits = 32 B = 64 hex; collision 2^-256; 40B×32B = 1.28 TB; SimHash 64-bit, d_H ≤ 3, 8B pages.
18. fastText lid.176 (176 langs, ≥0.65 FineWeb, <50 chars weak); tier cost 0.1 ms vs ~10 ms; 40B docs → 8×10^20 pairs.
19. Mitigations: position swap, repeated runs, rubric anchoring, reference guidance, judge diversity.
20. 2026 benchmark principles: Dynamic, Capability-specific, Ground-truth verifiable, Contamination-resistant; LiveBench monthly, 6 domains.

---
## 6. Fill-in-the-blank candidates (10-15) — 2 ข้อจริงน่าจะเป็นตัวเลข/สูตรสั้นๆ

1. Chinchilla ระบุว่า compute-optimal ต้องใช้ประมาณ **20** tokens ต่อ parameter; โมเดล 175B parameters จึงควรใช้ประมาณ **3.5T** tokens.
2. FineWeb ใช้ fastText language ID เก็บภาษาอังกฤษเมื่อ confidence score >= **0.65**; โมเดล `lid.176.bin` รองรับ **176** ภาษา.
3. ตาม Gopher rules เอกสารต้องมี alpha-word ratio อย่างน้อย **0.8** (80%) และมีคำอย่างน้อย **50** คำ.
4. C4 rules เก็บบรรทัดที่ลงท้ายด้วย **terminal punctuation** และทิ้งหน้าที่มีเครื่องหมาย **`{`** (curly bracket) หรือคำว่า **lorem ipsum**.
5. CCNet ใช้ **perplexity** จาก **KenLM 5-gram** (Kneser-Ney) เทรนบน Wikipedia แล้วแบ่งเอกสารเป็น 3 bucket คือ **head / middle / tail**.
6. FineWeb-Edu เก็บเอกสารที่ classifier ให้คะแนน educational value >= **3** (จาก 0-5) ได้ **1.3T** จาก **15T** tokens (~9%).
7. เอกสารสองฉบับเหมือนกันทุกตัวอักษรหลัง normalise จะถูกจับด้วย **SHA-256** hash ซึ่งให้ digest **256** bits (= 32 bytes / 64 hex chars) แต่จับ **near-duplicate** ไม่ได้.
8. MinHash ประมาณค่า **Jaccard similarity**; โอกาสที่ signature position ตรงกันเท่ากับ **J(A,B)**; ถ้าตรง 42 จาก 64 ตำแหน่ง ค่าประมาณ = **0.656** (42/64).
9. ใน LSH banding แบ่ง signature H ตำแหน่งเป็น **b bands × r rows**; ความน่าจะเป็นที่เป็น candidate = **1 − (1 − s^r)^b**.
10. **SimHash** เก็บ fingerprint 64-bit และรักษา **cosine similarity** (มุม) ผ่าน **Hamming distance**; MinHash ประมาณ **Jaccard**.
11. ภาษาไทยหนึ่งตัวอักษรใช้ **3** UTF-8 bytes; metric **TPC** = tokens / **Unicode characters**; rate ratio ρ ของ GPT-2 ในตัวอย่าง demo ≈ **14.2** เท่า.
12. BERTScore คำนวณ **cosine similarity matrix** ระหว่าง token embeddings แล้วใช้ **greedy matching**; F = **2PR/(P+R)** (harmonic mean).
13. LLM judge มักชอบคำตอบยาว เรียกว่า **length (verbosity) bias** และชอบคำตอบที่อยู่ลำดับแรก เรียกว่า **positional bias**; แก้ด้วยการ **swap** ลำดับ A-B / B-A.
14. MT-Bench มี **80** คำถาม **multi-turn** ใน **8** categories; Chatbot Arena จัดอันดับด้วย **Elo / Bradley-Terry**.
15. pass@k = 1 − C(n−c, k)/C(n, k); ถ้า n=200, c=1, k=100 จะได้ pass@100 = **0.5**; HumanEval มี **164** problems.
16. (สำรอง) MMLU: **57** subjects, **5-shot**; SWE-bench: resolved เมื่อ tests **fail→pass** ผ่านและ **pass→pass** ยังผ่าน; HarmBench metric = **ASR** (Attack Success Rate).

---
## 7. MCQ traps / ตัวลวงที่พบบ่อย

1. **"Perplexity ต่ำ = คุณภาพสูง"** (ผิด): PPL ต่ำ = คล้าย reference model; spam/ข้อความซ้ำ PPL ต่ำมาก; PPL สูงอาจเป็นความรู้เฉพาะทางหายาก.
2. **"CCNet ตัดที่ PPL threshold คงที่"** (ผิด): ไม่มี fixed cut-off; แบ่ง head/middle/tail เท่าๆ กัน (ตามขนาดข้อมูล) แล้วผู้ใช้เลือก.
3. **"SHA-256 จับ near-duplicate ได้ถ้า hash ต่างน้อย"** (ผิด): hash ต่างโดยสิ้นเชิงแม้เปลี่ยน 1 ตัวอักษร; จับได้เฉพาะ exact. **"MinHash จับ exact duplicate เท่านั้น"** (ผิด).
4. **MinHash vs SimHash สลับกัน**: MinHash -> Jaccard (set overlap, H ตัวเลข); SimHash -> cosine/มุม (bit string สั้น, Hamming). ตัวลวง: "SimHash ประมาณ Jaccard".
5. **"LSH ทำให้ MinHash เร็วขึ้นต่อคู่"** (ผิด): MinHash ลดต้นทุนต่อคู่ (17,689 -> 64); **LSH ลดจำนวนคู่** (ไม่ต้องดูทุกคู่ N^2).
6. **"s* = (1/b)^(1/r) คือ cutoff / จุด P=0.5"** (ผิด): เป็นตำแหน่ง ramp คร่าวๆ; ที่ s* P = 0.644 (b=16,r=4), P=0.5 ที่ 0.454.
7. **"LSH candidate = duplicate ยืนยันแล้ว"** (ผิด): ต้อง verify ด้วย exact Jaccard.
8. **"เพิ่มจำนวน shingle ลด SE ของ MinHash"** (ผิด): ลดด้วยการเพิ่ม **H** (จำนวน hash functions).
9. **"brute-force dedup เป็น O(N^3)"** (ผิด): N^2 s^2.
10. **"Dedup ภายในเอกสารเดียว (cookie banner ซ้ำ) ถูกกำจัดโดย MinHash/LSH"** (ผิด): ต้องใช้ Gopher repetition rules; dedup เทียบระหว่างคู่เอกสาร.
11. **ลำดับ pipeline**: dedup ต้องอยู่หลัง heuristic ก่อน neural classifier (ตัวลวง: "classifier ก่อนเพราะแม่นสุด"). และ "cheap-first เป็นกฎตายตัว" (ผิด: เป็น heuristic เริ่มต้น; ใช้สูตร c_A + p_A c_B).
12. **Gopher: "ผ่านข้อใดข้อหนึ่งก็เก็บ"** (ผิด): ต้องผ่านทุกข้อ. **Thai ใช้ threshold อังกฤษได้** (ผิด).
13. **C4 ปฏิบัติต่อโค้ด/LaTeX ถูกต้อง** (ผิด): ลบเพราะ `{`; ลบ non-English ด้วย (langdetect 0.99).
14. **TPC = fertility** (ผิด): TPC หาร Unicode chars; fertility หารด้วยคำ (ไทยไม่มี word boundary). **TPC ต่ำของ BERT = มีประสิทธิภาพ** (ผิด: เป็น [UNK] coverage failure).
15. **Chinchilla = กฎสากล / เพิ่มแต่ขนาดโมเดล** (ผิด: เป็น empirical, ขยายโมเดลและ token ไปด้วยกัน ~20 tok/param; GPT-3 300B tok = data-limited).
16. **FineWeb-Edu ใหญ่กว่า FineWeb** (ผิด: 1.3T vs 15T แต่ผลดีกว่าที่ matched budget).
17. **FineWeb vs Dolma สลับกัน**: FineWeb = CC อย่างเดียว/ablation-driven/15T; Dolma = multi-source/3T/transparency toolkit, AI2, OLMo.
18. **BLEU สูง = แปลดี** (ลวง): word salad ได้ BLEU-1 0.83 > paraphrase 0.17. **BERTScore = ตรวจข้อเท็จจริง** (ผิด) / "BERTScore 0.79 > BLEU 0.17 แสดงว่าเก่งกว่า" (ผิด: คนละสเกล เทียบ ranking เท่านั้น); BERTScore เป็น % (ผิด: raw cosine ~0.7-0.9 แม้ไม่เกี่ยว).
19. **BERTScore matching เป็น one-to-one** (ผิด: greedy, หลาย candidate token เลือก ref token เดียวกันได้). Precision = max over **reference** ต่อ candidate token (แถว) — อย่าสลับกับ Recall.
20. **Chatbot Arena reproducible / ไม่มี contamination** (ผิด: ไม่ reproducible, user วางคำถาม benchmark ได้). **LLM judge ตรงกับคน ~80% คงที่** (ผิด: ขึ้นกับ judge/task/protocol).
21. **Length bias แก้ด้วยการหารคะแนนด้วยจำนวน token** (ผิด: ใช้ statistical length control เช่น AlpacaEval-LC / content-matched pairs). **Position swap แล้ว inconsistent = tie วัดจริง** (ผิด: เป็น aggregation policy; ป้ายที่ถูก = inconsistent). **Swap ทำให้ bias หายหมด** (ผิด: ลดแต่ไม่หมด; GPT-4 65%->77.5% ด้วย few-shot).
22. **GPT-4 ต้านการ padding ไม่ได้** (ผิด: 8.7% เท่านั้น เทียบ 91.3% ของ Claude-v1/GPT-3.5).
23. **pass@k**: "pass@10 คือค่าที่แม่นกว่า pass@1" (ผิด: คนละปริมาณ); "c = 0 ให้ pass@k = 1"(ผิด = 0); ลืมว่า pass@1 = c/n; ใช้ `C(c,k)/C(n,k)` แทน `C(n−c,k)` (ผิดสูตร: เศษส่วนคือโอกาสที่ k ตัวล้มทั้งหมด).
24. **MMLU/GSM8K สูงสุด = โมเดลดีกว่า** (ลวง: saturated >95% GSM8K; contamination; ต่าง protocol). **12% ข้อสอบรั่ว -> คะแนนจริง = 94.2 − 12 = 82.2** (ผิด: คำนวณไม่ได้โดยตรง ต้องรัน clean subset ใหม่).
25. **Dynamic benchmark (LiveBench) = ไม่มี contamination แน่นอน** (ผิด: ลดแต่ไม่ลบ). **AIME ผ่านทั้ง 4 แกนและตัดสินใจได้ลำพัง** (ผิด: ปีล่าสุดเท่านั้น, 15 ข้อ/paper เล็กเกิน). **SWE-bench Lite/Verified/Original เทียบกันได้** (ผิด).
26. **HarmBench: ASR ต่ำ = ปลอดภัย** (ผิด; ปฏิเสธทุกอย่าง ASR=0) และ **ตัดสินโดยมนุษย์** (ผิด: Llama-2-13B classifier fine-tuned).
27. **RAGAS faithfulness สูง = ตอบถูก** (ผิด: ตรงกับ context เท่านั้น; context อาจผิด/ไม่ครบ/ไม่เกี่ยวกับคำถาม). **RAGAS ต้องมี human annotation** (ผิด: reference-free สำหรับ 3 metrics ต้นฉบับ). **context precision/recall มาจาก paper ต้นฉบับ** (ผิด: เป็น library รุ่นหลัง และต้อง reference).
28. **WET = raw HTML** (ผิด: WARC = raw, WET = extracted text).
29. **Era**: ตัวลวงสลับยุค (BERTScore เป็น Era 2 ไม่ใช่ Era 1; RAGAS/SWE-bench/LiveBench/HarmBench Era 3).
30. **ผู้สร้าง label ของ FineWeb-Edu**: Llama-3-70B-Instruct (ไม่ใช่มนุษย์/GPT-4); student = encoder ~110M (ไม่ใช่ LLM 70B).

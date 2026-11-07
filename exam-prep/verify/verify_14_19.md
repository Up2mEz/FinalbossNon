# ตรวจสอบรายการ #14-#19 จากภาพสไลด์จริง (render ด้วย pdftoppm -r 90)

PDF page index ตรงกับเลขสไลด์ที่พิมพ์ในทุกหน้า (W13 p12 = S12, p41 = S41; W14 p8, 9, 18, 19, 40, 53 ตรงกับเลขพิมพ์)

## #14 W13 S41 (PDF p.41) Topology Comparison Table -- CORRECTION (ชื่อคอลัมน์)

ตารางบนสไลด์ (ถอดตรงตัว):

| Topology | Latency | Complexity | **Fault Tolerance** | Best Use Case |
|---|---|---|---|---|
| Sequential | High | Low | Low | Content pipelines (Write -> Edit) |
| Routing | Low | Low | Medium | Customer support, Triage |
| Supervisor | Very High | High | High | Software dev, Complex planning |
| Debate | Medium | Medium | High | Fact-checking, Evaluation |

Takeaway บนสไลด์: "No single topology wins every scenario. Choose based on task requirements."

- คอลัมน์ที่ 4 ชื่อ "Fault Tolerance" ไม่ใช่ "robustness"
- แถวที่ 3 เขียนว่า "Supervisor" (ไม่ใช่ "Supervisor-Led") ในตาราง
- ค่าตัวเลือกมีแค่ Low / Medium / High / Very High

## #15 W13 S12 (PDF p.12) Context growth table -- MATCH (ตัวเลขสอดคล้องกันเอง)

| Step | Thought | Action | Observation | Added this step | Cumulative |
|---|---|---|---|---|---|
| 0 | - | - | - | - | 500 (system + query) |
| 1 | 50 | 30 | 200 | 280 | 780 |
| 2 | 60 | 30 | 500 | 590 | 1,370 |
| 5 | 70 | 30 | 1,000 | 1,100 | 4,670 |
| 10 | 80 | 30 | 2,000 | 2,110 | 15,220 |

- Note บนสไลด์: rows 5 และ 10 สมมติว่าทุก step ระหว่างทางเพิ่มเท่ากับแถวนั้น
- Risk: เฉลี่ย ~2,110 tokens/step แล้วยอดรวมข้าม 128,000 ที่ **step 64** และต้องหยุดก่อนหน้านั้นเพราะต้องเหลือที่ให้โมเดลเขียนคำตอบ
- กราฟบนสไลด์ระบุ "Step 64: 129,160 Tokens (Limit Exceeded)"
- ตรวจคำนวณ: 50+30+200=280, 60+30+500=590, 70+30+1000=1100, 80+30+2000=2110 ถูกทุกแถว
  780+590=1370; 1370+3x1100=4670; 4670+5x2110=15220; step 64 = 15220 + 54x2110 = 129,160 (>128,000) ส่วน step 63 = 127,050 (<128,000) จึงข้าม 128K ที่ step 64 พอดี
- สูตรบนสไลด์: |s_t| = |system| + |q| + sum_{i=1}^{t-1} (|tau_i| + |a_i| + |o_i|)

## #16 W14 p.8-9 Tokenizer comparison -- MATCH

p.8 "Script Penalties - The Non-Latin Tax" (GPT-2 tokenizer, transformers 5.x):

| Language | Sample | Characters | Tokens | Tokens/Char |
|---|---|---|---|---|
| English | "Natural language processing enables computers to understand human text." | 71 | 10 | 0.14 |
| Thai | "การประมวลผลภาษาธรรมชาติช่วยให้คอมพิวเตอร์เข้าใจภาษามนุษย์" | 57 | 114 | 2.00 |

สไลด์เขียน "about 14x more tokens per character on Thai" (multiplier ขึ้นกับตัวอย่าง ต้องวัดบน parallel text)

p.9 "Tokenizer Fairness - Token/Character Rate" (TPC = tokens/Unicode chars; rho = TPC_TH / TPC_EN):

| Tokenizer | TPC EN | TPC TH | rho | Note |
|---|---|---|---|---|
| GPT-2 BPE (50k) | 0.14 | 2.00 | 14x | byte fallback for Thai |
| BERT-base-uncased (30k) | 0.14 | 0.02 | - | Thai -> [UNK]; TPC meaningless |

- ตรวจ: 10/71 = 0.141, 114/57 = 2.00, 2.00/0.141 = 14.2 ตรงกับ ~14x
- สไลด์ใช้ "14x" (ไม่ได้เขียน 14.2x)

## #17 W14 p.18-19 Dolma vs FineWeb -- MATCH

p.19 "FineWeb vs. Dolma - Side-by-Side Comparison":

| Dimension | FineWeb | Dolma |
|---|---|---|
| Size | 15T tokens | 3T tokens |
| Sources | Common Crawl only | CC + GitHub + Wiki + Papers |
| Filter Design | Ablation-driven | Modular taggers |
| Transparency | Data + datatrove code released; ablations published | Data + dolma toolkit + per-document tagger logs |
| Reproducibility | Pipeline re-runnable on Common Crawl | Fully reproducible, mixture ratios documented |
| Best For | Maximum web-text scale | Multi-source mixtures, research auditing |
| Associated Model | 1.8B ablation models (Llama architecture) | OLMo (fully open) |

p.18 Dolma (v1.6, ~3 trillion tokens, AI2) Table 1 (Llama tokens, billions): Common Crawl 2,479 (web pages, 9,812 GB, 3,734M docs, 1,928B words); GitHub 411 (code, 1,043 GB, 210M docs, 260B words); Reddit 89 (339 GB, 377M, 72B); Semantic Scholar 70 (268 GB, 38.8M, 50B); Project Gutenberg 6.0 (20.4 GB, 0.056M, 4.0B); Wikipedia/Wikibooks 4.3 (16.2 GB, 6.2M, 3.7B); Total 11,519 GB, 4,367M docs, 2,318B words, 3,059B Llama tokens. หมายเหตุสไลด์: แถว Common Crawl = Dolma CC (2,281B) + C4 (198B)

## #18 W14 p.40 LLM-judge positional bias -- MATCH

(Zheng et al. 2023, Table 2, default prompt; "Error" = malformed judge output จึงรวมไม่ครบ 100%)

| Judge | Consistent | Biased to first | Biased to second | Error |
|---|---|---|---|---|
| Claude-v1 | 23.8% | 75.0% | 0.0% | 1.2% |
| GPT-3.5 | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-4 | 65.0% | 30.0% | 5.0% | 0.0% |

ข้อความบนสไลด์: "Few-shot examples lift GPT-4 from 65.0% to 77.5% -- better, not fixed." (ตัวเลข 77.5% อยู่ในข้อความ ไม่อยู่ในตาราง)
Swap trick: judge ทั้ง A-B และ B-A แล้วประกาศผู้ชนะเฉพาะเมื่อชนะทั้งสองลำดับ; ถ้าสองลำดับไม่ตรงกัน label ที่ถูกต้องคือ "inconsistent" (การให้เป็น tie เป็น aggregation policy ไม่ใช่การวัด)

## #19 W14 p.53 RAGAS formulas -- MATCH (faithfulness = supported claims / total claims)

อ่านชัดเจน (ภาพทั้งหน้า) สูตรบนสไลด์:

1. Faithfulness = |claims in answer supported by context| / |total claims in answer|
   Pipeline: LLM แยกคำตอบเป็น atomic claims แล้วเช็คแต่ละ claim กับ context
   ตัวอย่าง: 3/4 = 0.75 (Claim 3 ไม่ถูกสนับสนุน)
2. Answer Relevance = (1/n) * sum_{i=1}^{n} cos(emb(q_i), emb(q))
   q_1..q_n = คำถามที่ LLM สร้างย้อนจากคำตอบ (reverse question generation); เป็น topicality check ไม่ใช่ factual check; คำตอบ = ค่าเฉลี่ย similarity
3. Context Recall = |reference statements supported by context| / |reference statements|
   (ต้องมี reference answer; "supported" = entailment ไม่ใช่ตรงตัวอักษร) ตัวอย่าง 3/5 = 0.60
4. Context Precision@K = (sum_{k=1}^{K} Precision@k * v_k) / (relevant items in top K)
   (v_k = rel_k; rank-aware) ตัวอย่าง rel = [1,1,0,1,0]: Precision@1 = 1/1 = 1.00, @2 = 2/2 = 1.00, @4 = 3/4 = 0.75; Context Precision@5 = (1.00+1.00+0.75)/3 = 0.92 (2.75/3 = 0.9167)
   สไลด์ระบุว่านับ Precision@k เฉพาะอันดับที่ item เป็น relevant แล้วเฉลี่ย

# CHEATSHEET A4 (เขียนมือ 2 หน้า) — สอบ NLP Final จันทร์ 9 พ.ย. 2026

> ใช้ลอกลงกระดาษ A4 ด้วยมือ (ห้ามปริ้นท์). เนื้อหาในกล่อง ``` คือสิ่งที่ลอก, ส่วนอื่นเป็นคำแนะนำ
> หลักการ: แผ่นนี้เก็บ **ตัวเลข / สูตร / ตารางเปรียบเทียบ / ลำดับตัวเลือก** เท่านั้น. "ทำไม/สัญชาตญาณ" อยู่ใน `MEMORIZE_LIST.md` (ไม่ลอกซ้ำ)
> สัญลักษณ์: `(!n)` = ตัวเลข/ตารางที่ยังไม่ชัวร์ → ดู "VERIFY LIST" ท้ายไฟล์ข้อ n แล้วเปิดสไลด์ PDF ตรวจก่อนลอก
> เขียนลูกศร `->` แทน →, `<=` แทน ≤, `sum` แทน Σ, `sqrt` แทน √ ได้ทั้งหมด (ลอกง่ายกว่า)
> นับ token = กลุ่มอักขระที่คั่นด้วยช่องว่างและมีตัวอักษร/ตัวเลขอย่างน้อย 1 ตัว (`[B,h,s,s]` นับ 1; ตัวคั่นเปล่า `|` `->` `=` ไม่นับ เพราะเขียนเร็วมาก). ตัวเลข `~NNN` คือจำนวนจริงที่นับจากกล่อง (ถ้านับตัวคั่นด้วย รวมประมาณ 1,830)

## งบพื้นที่ (A4 ตัวอักษรเล็ก ~1,300-1,500 token ทั้งแผ่น)

| กล่อง | หน้า | หัวข้อ | token (นับจริง) | พื้นที่โดยประมาณ |
|---|---|---|---|---|
| 1A | 1 | W8 Enc-Dec: BART/T5 | ~163 | 20% ของหน้า 1 |
| 1B | 1 | W9 SBERT / contrastive / FAISS | ~138 | 17% |
| 1C | 1 | W10 Hardware-aware / PEFT | ~154 | 25% |
| 1D | 1 | CALC RECIPES (สูตรคำนวณทุกสัปดาห์) | ~224 | 30% |
| 1S | 1 | SPARE | - | ~8% (ว่างไว้เติมข้อที่ผิดตอนทำ mock) |
| 2A | 2 | W11 Prompt / RLHF / DPO / GRPO | ~183 | 19% ของหน้า 2 |
| 2B | 2 | W12 BM25 / RRF / RAG / CLIP / GraphRAG | ~201 | 24% |
| 2C | 2 | W13 Agents / masking / memory / topologies | ~170 | 21% |
| 2D | 2 | W14 Dedup / filters / metrics / judge | ~213 | 28% |
| 2S | 2 | SPARE | - | ~8% |
| รวม | | | ~1446 | |

## Layout แนะนำ (A4 แนวตั้ง, แบ่ง 2 คอลัมน์ต่อหน้า)

```
หน้า 1                               หน้า 2
+---------------+---------------+    +---------------+---------------+
| 1A  W8        | 1C  W10       |    | 2A  W11       | 2C  W13       |
|               |               |    |               |               |
+---------------+               |    +---------------+---------------+
| 1B  W9        |               |    | 2B  W12       | 2D  W14       |
|               +---------------+    |               |               |
|               | 1S spare      |    |               +---------------+
+---------------+---------------+    |               | 2S spare      |
| 1D  CALC RECIPES (เต็มความกว้าง)|    +---------------+---------------+
+-------------------------------+
```

ลำดับความสำคัญถ้าที่ไม่พอ (ตัดจากล่างขึ้นบน): ตัด `[LOW]` ในกล่องก่อน -> ตัด 2D บรรทัดท้าย -> อย่าตัด 1D และสูตรที่ครูใบ้ (BM25, RRF, GB/GiB, KV/GQA, absmax, 2304, cross vs bi, triplet, BART/T5 token count, BART 5 noising, MLM 80/10/10, topologies)

---

# หน้า 1 (W8 - W10 + calc)

## BOX 1A — W8 Encoder-Decoder (~163 token)
```
BART noise 5: Mask(src=tgt) | Delete(no placeholder, src<tgt) | Infill(span~Poisson lam=3 -> ONE [MASK], len0=insert) | SentPerm | DocRot(pivot->start)
T5 span corruption: sentinel <extra_id_0,1,..>; tgt = ONLY removed spans, each led by its sentinel, ends <extra_id_k+1> then </s>
            BART            T5
tgt len     N (full)        ~0.2N (15% + 5% sentinel)
dec attn    N^2             (0.2N)^2=0.04N^2 -> 25x (train 2-3x)
framing     heads or gen    ALWAYS text2text, prefix, NO head
tok/vocab   BPE 50265       SentencePiece 32,000
size        6+6 ~140M | 12+12 ~406M d1024 16h (!4)     60M-11B; base d768 ff3072 12h
data        160GB           C4 ~750GB
dec start   <s>             <pad>
T5 pos: relative bias, 32 bkt/head (16f+16b), 0-7 exact, 8-128+ shared(log), ADDED to logits pre-softmax
count (n tok, k spans, r removed): src=n-r+k ; T5 tgt=r+k+1 (+1 if </s> counted) ; BART tgt=n (+2 <s></s>) (!5)
MLM 15% [80% MASK/10% rand/10% same] ; CLM 100%
shapes: enc [B,h,s,s] full | dec [B,h,t,t] causal | cross [B,h,t,s] Q<-dec K,V<-enc
Dec-only won: (1) serving+params (2) RLHF multi-turn (3) KV linear. EncDec still wins: MT(NLLB) ASR(Whisper) summ(long->short)
no prefix -> T5 translates | Flan-T5 1836 tasks | mT5 101 lang
```

## BOX 1B — W9 Sentence embeddings (~138 token)
```
SBERT cls head: [u;v;|u-v|]=3d=2304 (768x3) -> softmax. pool: MEAN (mask PAD) best > CLS(NSP) ; max
STS: GloVe 58.02 | BERT-CLS 16.50 | BERT-mean 46.35 | SBERT-NLI-base 77.03 (all = STS Benchmark, Spearman, W9 p4) ; SimCSE unsup 76.3 / sup 81.6 vs SBERT-base 74.89 (p24, no benchmark named; slides never reconcile 77.03 vs 74.89)
cause = MLM objective not sentence geometry -> anisotropy (cos mean .58, [.46,.85]); fix = contrastive FT
Bi: n fwd + n^2 dot (ns), cacheable, stage1 RECALL | Cross: n^2 fwd (ms), full attn, rerank PRECISION
Triplet L=max(0, d(a,p)-d(a,n)+alpha); =0 iff d(a,p)+alpha <= d(a,n) | MarginRank max(0,sim(a,n)-sim(a,p)+alpha) == triplet if d=1-sim
mining: easy L=0 | hard d(a,n)<d(a,p) | semi-hard between = best
MNRL -log e^{s(qi,pi)/tau}/sum_j e^{s(qi,pj)/tau}; in-batch neg N-1; tau .05-.1 low=sharp; only (q,p) needed
cos=u.v/|u||v|; d=1-cos; normalized: dot=cos, |u-v|^2=2-2cos. FlatIP=cos(high good) | FlatL2=sq dist(low good) | HNSW default L2
ANN recall: Flat 100 | IVF ~95 | HNSW ~99 | IVF+PQ ~90 (>1B) (!2) ; always report Recall@k
dims MiniLM 384 | mpnet 768 | bge-large 1024 ; NLI 1M(570K+430K) -> STSb 8.5K (0-5) ; bs16 lr2e-5 1ep
```

## BOX 1C — W10 Hardware-aware NLP (~154 token)
```
bytes/param: FP32 4 | FP16,BF16 2 | INT8 1 | INT4 .5 ; 1B=2GB ; 8B/70B/175B = 16/140/350 GB
BF16 1/8/7 max 3.4e38 | FP16 1/5/10 max 65,504 | FP32 1/8/23 ; BF16 needs Ampere+ (T4 no -> FP16)
Prefill GEMM compute-bound | Decode GEMV mem-bandwidth-bound ; TTFT = net+prefill+1st dec
FlashAttn: EXACT, tiling in SRAM + online softmax, extra mem O(N) (std N^2: N=32k -> 2GB/head), saves memory/IO not FLOPs
PagedAttn = KV blocks + page table (20-40% -> ~95%) | cont. batching = iteration-level
SpecDec: small draft -> big verify in 1 pass ; SAME dist as target ; ~2-3x ; helps TPS ; accept min(1,p/q)
TP split matrices (intra-node) | PP split layers (inter-node, bubbles) | 3D = TP+PP+DP
MoE Mixtral 8x7B: 47B total, 13B active, top-2 ; saves COMPUTE not MEMORY ; aux loss ; DeepSeek = shared + fine-grained routed
LoRA dW=BA, B=0, A~N, base frozen, merge -> no latency | QLoRA = NF4 + DoubleQuant(scales) + PagedOpt(->CPU RAM); grads only A,B
KD L=a*T^2*KL(soft)+(1-a)*CE ; T>1 flattens ; DistilBERT 12->6L, -40% size, +60% speed, 97%
compress axes: quant=bits | LoRA=trainable params | distill=layers (ONLY one cutting FLOPs) ; grad ckpt +30% time
```

## BOX 1D — CALC RECIPES (ทุกสัปดาห์) (~224 token)
```
UNITS GB=1e9 ; GiB=2^30=1.0737 GB ; GiB = GB/1.0737 (16GB=14.9GiB) ; MB/MiB same. watch 10^n !
W=N*bytes | Train(Adam)=16N=2+2+12(75%) -> 8B 128GB, 70B 1,120GB=14 H100 | KV=2*bytes*L*n_kv*d_h*ctx*B
Llama3-8B L32 n_kv8 d128: 131,072 B/tok ; 4096 -> .537GB ; 128K -> 16.8GB ; GQA x = n_heads/n_kv (32/8=4 ; 64/8=8 : 160->20GB) ; infer 16+.54+1.46=18GB
LoRA 2dr (4096,r16 -> 131,072) ; 8B QKVO: Q,O 131,072 ; K,V 81,920 ; x32 = 13.63M = .17% ; mem 16+.22+2 = 18.2GB
absmax (!6): s=max|w|/127 (3.2 -> .0252) ; q=round(x/s): 2.1 -> 83 ; deq 83*.0252 = 2.092 ; err .008 ; block 64
SBERT 768x3=2304 | pairs n(n-1)/2: 1000 -> 499,500 ; 1e4 -> 5e7 x5ms = 69h ; bi n*5ms = 50s (!1) | n*d*4B: 1M x 768 = 3.07GB
triplet d(a,p)=.087 d(a,n)=.812 al=.5 -> 0 ; hard n d=.035 -> .552 | MRR ranks 1,2,4 = .583
BM25 (b=0) TF=f(k1+1)/(f+k1) k1=1.5: f1 1.0 | 2 1.43 | 3 1.67 | 5 1.92 | 10 2.17 | 100 2.46 ceil 2.5 ; slide table f=0,1,2,5,10,1000 -> 0,1.00,1.43,1.92,2.17,2.50 ; f=k1 -> 50% of ceiling ; L=1-b+b|d|/avgdl : |d| 50,100,500 (avgdl 100, b .75) -> .625, 1, 4 ; f=3: L.625 1.905 | L1 1.667 | L4 .833 (!8)
RRF k=60 : rank1+rank3 = 1/61+1/63 = .0323 ; (1/61-1/62)/(1/61) = 1.6%
DPO: x = beta*[(lp_w-ref_w)-(lp_l-ref_l)] ; loss = ln(1+e^-x) : x=0 -> ln2 .693 ; .4 -> .513 ; 2 -> .127 | RM loss -ln sig(r_w-r_l)
GRPO G=4 r=[1,0,0,0]: mu .25 sig .433 -> A +1.73,-.577 | [1,1,0,0] -> +1,+1,-1,-1 | all same -> A=0
mask [2.0,.5] -> .818/.182 ; 50,257-69 = 50,188 | seq latency = sum T_i ; ctx = base + n*step
MinHash 42/64 = .656 ; SE .059 ; H x4 -> SE /2 | LSH b16 r4: s .5 -> .644, s .656 -> .96 | pass@k n200: c2 k10 .098, k100 .751 ; c1 -> k/n
Chinchilla 175B*20 = 3.5T | 40B docs -> 8e20 pairs ; 40B*32B = 1.28TB | BLEU BP=e^(1-|ref|/|hyp|)
```

### 1S — SPARE (หน้า 1)
เว้นว่างไว้ ~8% : เติมข้อที่ทำ mock ผิดซ้ำ / ตัวเลขที่เพิ่งตรวจจาก VERIFY LIST

---

# หน้า 2 (W11 - W14)

## BOX 2A — W11 Prompting / Post-training / Reasoning (~183 token)
```
RLHF: SFT -> RM (Bradley-Terry -log sig(r_w-r_l), scalar head init from SFT) -> PPO ; PPO 4 models: Actor,Critic (train) | RM,Ref (frozen) ; obj max E[r] - beta*KL(pi||pi_ref) (stops reward hacking)
DPO: -log sig(beta[log pi_w/ref_w - log pi_l/ref_l]) ; 2 models (policy+ref) ; NO RM / PPO ; stable but overfits ; beta big -> stay near ref ; init loss ln2
GRPO: NO Critic ; G samples/prompt ; A=(r-mu)/sigma, sigma=sqrt((1/G)sum(r-mu)^2) (divide G) ; sum A=0 ; baseline = group mean ; verifiable reward (math/code)
            RLHF/PPO     DPO          GRPO
models      4            2            Actor+Ref+RM/rule (no Critic)
data        pref -> RM   pref pairs   self-gen groups
baseline    Critic       ref model    group mean
signal      scalar RM    binary       continuous
R1-Zero = pure RL no SFT, rule reward + format <think> ; R1 trace visible, o-series hidden ; think longer = diminishing returns, high TTFT
CO-STAR = Context Objective Style Tone Audience Response-format ; ICL = no weight update ; self-consistency: temp>0, 5-10 paths, majority ; zero-shot CoT "Let's think step by step" 17->78
PoT = LLM writes code, interpreter computes | ToT = tree search (gen-eval-prune) | GoT = graph (aggregate+refine)
injection direct / indirect (external data) ; defenses: input filter, sandwich, output eval, least privilege (no silver bullet)
SFT = behavior cloning, next-token ; limits: costly, exposure bias (teacher forcing), no negative fb ; ChatML <|im_start|> ; Llama3 <|eot_id|>
```

## BOX 2B — W12 Hybrid RAG / CLIP / GraphRAG (~201 token)
```
BM25 = sum IDF * f(k1+1) / [f + k1(1-b+b|d|/avgdl)] ; IDF=ln((N-n+.5)/(n+.5)): <0 if n>N/2, =0 at N/2 (Lucene ln(1+..))
TF ceiling = k1+1 (indep of b) ; f=k1 -> 50% ; b=0 no len-norm, b=1 full, .75 std ; b changes SPEED to ceiling only ; short doc > long doc (same f)
RRF = sum 1/(k+rank), k=60 ; BM25 (0,inf) vs cos [-1,1] -> use RANK not score ; min-max sensitive to outlier ; both-list doc > one-list doc
Sparse: vocab mismatch, wins ids/codes/versions | Dense: paraphrase, "hallucinated relevance"
Bi=recall fast | Cross=precision, [CLS]q[SEP]d, O((|q|+|d|)^2), rerank top K 20-100, logit not prob | ColBERT late-interaction MaxSim ~2 orders cheaper than CE at query time, quality between Bi & CE (slide bullet) ; Bi: sub-ms, millions docs, prec Med | CE: 10-100 ms/pair, 100s docs, prec High
pipeline latency max(5,10)+1+200 = 211 ms (!11)
embed storage N*d*4B: 1M x 384 = 1.5GB ; x 3072 = 12.3GB ; MiniLM truncates 256 tok silently ; dims 384/768/1024(bge-m3)/1536/3072
CLIP: 400M pairs, N=32,768, symmetric InfoNCE (1/2N), zero-shot prompt "A photo of a {obj}" 76.2% (ViT-L/14@336) ; fails: relations, attributes, negation
            Vector RAG (!12)          GraphRAG
best        specific factual, local   thematic/global, multi-hop, relations
how         chunk embed top-k         triples -> graph -> Leiden communities -> summaries (offline)
query       1 retrieval               Local=entity traversal (1 LLM call) | Global=Map-Reduce on summaries (costly)
Lost-in-middle = U-curve ; bookend [1,2,3,4,5] -> [1,3,4,5,2] (!13) ; 4 rules: rerank first, bookend, tune K, context distill
RoPE interp m'=m/s (4K->16K: 8000->2000) ; SWA O(N*w) ; NTK squeezes low-freq only ; changing facts=RAG, style=FT, small static=long-ctx
```

## BOX 2C — W13 Agents / constrained decoding / memory / multi-agent (~170 token)
```
ReAct Thought -> Action -> Observation ; s_t=(q,h_{1:t-1}) ; max_steps guard ; ctx grows: +280,+590,+1100,+2110/step ; crosses 128K at step 64 (129,160) [slide ok]
Table1 HotpotQA EM / FEVER acc: Std 28.7/57.1 | CoT 29.4/56.3 | CoT-SC 33.4/60.4 | Act 25.7/58.9 | ReAct 27.4/60.9 | CoT-SC->ReAct 34.2/64.6 | ReAct->CoT-SC 35.1/62.0
Err ReAct/CoT: halluc 0/56 | reasoning err 47/16 | search err 23/- | true+ 94/86
tool def = name + description + parameters(JSON Schema) ; call = name + arguments ; 5 steps gen -> parse -> dispatch -> exec(string) -> Observation ; fails: bad JSON / fake tool / loop
mask: logit=-inf if invalid, BEFORE softmax -> P=0 (hard) ; syntax NOT semantics ; on token ids ; after '{' valid 69 = 62+5+2 of 50,257 -> mask 50,188 ; in string mask 111 ; regex/DFA can't nest unbounded -> CFG + stack
memory: Working/Episodic/Semantic/Procedural ; Buffer O(n) | Summary bounded, lossy | Retrieval prompt O(k), storage O(n)
topology [slide S41 verified]  latency / complexity / FAULT TOLERANCE / best use
Sequential   High  / Low  / Low    -> content pipeline (write->edit)
Routing      Low   / Low  / Medium -> customer support, triage ; router: LLM / argmax cos / rule(fastest) ; metric routing accuracy
Supervisor   V.High/ High / High   -> software dev, complex planning
Debate       Med   / Med  / High   -> fact-checking, evaluation ; parallel + judge
protocols: scratchpad / message passing / structured payload ; reliability: Constrain -> Validate -> Recover -> Escalate ; tool content = untrusted ; guard: max_steps, least privilege
```

## BOX 2D — W14 Data engine / metrics / judge (~213 token)
```
dedup: SHA-256 exact only (256b=32B=64hex) | MinHash -> Jaccard (H nums) | SimHash -> cosine (64b, Hamming<=3) ; P(sig match)=J ; SE=sqrt(J(1-J)/H) max 1/(2sqrt H); lower SE via H not shingles ; same hash fns both docs
LSH H=b*r ; P(cand)=1-(1-s^r)^b ; s*~(1/b)^(1/r) NOT cutoff ; verify cands ; MinHash cuts cost/pair, LSH cuts #pairs -> stack
pipeline: LangID(fastText 176, .65) -> Gopher+C4 -> PPL -> dedup -> neural clf ; heur ~.1ms, neural ~10ms ; A first iff c_A+p_A*c_B < c_B+p_B*c_A
Gopher ALL: 50<=words<=100k; mean wlen 3-10; alpha>=.8; symbol<=.1; bullets<=90%; "..."<=30%; >=2 stopwords ; Thai: segment + retune
C4: end punct; >=3 words/line; no "Javascript"; >=5 sent/page; no lorem / { ; 3-sent dedup ; langdetect .99
PPL=exp(-(1/N)sum log P) ; CCNet KenLM 5-gram Wiki, 3 buckets head/mid/tail (no cutoff) ; low PPL != quality (spam low, rare tech high)
FineWeb 15T/96 CC ; Edu >=3 -> 1.3T (9%) ; labels Llama-3-70B ; Dolma ~3T AI2 multi-source (!17) ; Chinchilla 20 tok/param
BLEU lexical ; BERTScore greedy: P=mean_i max_j cos, R=mean_j max_i ; not %, compare rank ; salad BLEU .83 > paraphrase .17 ; BERT .67 < .79
Judge: length bias (padded chosen Claude-v1 91.3, GPT-3.5 91.3, GPT-4 8.7) ; position bias -> swap A/B (GPT-4 65->77.5 (!18)) ; fixes swap, repeat, rubric, reference, multi-judge ; MT-Bench 80q 8cat
pass@k=1-C(n-c,k)/C(n,k) ; MMLU 57 subj 5-shot ; GSM8K 8,500 ; HumanEval 164 ; SWE-bench fail->pass + pass->pass ; HarmBench ASR low != safe ; RAGAS faith = supported claims/all (!19) != correct
TPC=tok/chars ; rho=TPC_TH/TPC_EN (GPT-2 14.2) (!16) ; BERT Thai -> [UNK] ; eras: BLEU -> BERTScore/Arena -> LiveBench/SWE-bench/RAGAS ; contamination != saturation
```

### 2S — SPARE (หน้า 2)
เว้นว่างไว้ ~8% : เติมหลังตรวจ VERIFY LIST (ที่ตรวจแล้วไม่ตรง) / ข้อที่ผิดซ้ำใน mock

---

# VERIFY LIST — ต้องเปิดสไลด์ PDF ตรวจก่อนลอก (19 รายการ)

> ที่มา: โน้ตระบุว่าตัวเลขมาจากตาราง/รูปที่ extract เป็น text ไม่ได้, คำนวณเอง, ขัดกับ hint หรือสไลด์ไม่ตรงกันเอง. "เลขหน้า" = ตามที่โน้ตอ้าง (W8/W9/W14 = หน้า "p./s.", W13 = เลขสไลด์ที่พิมพ์มุมขวาล่าง "S")

| # | หน้า PDF | อะไรที่ต้องตรวจ | ทำไมต้องตรวจ | อยู่ใน box |
|---|---|---|---|---|
| 1 | W9 s.8 | Bi-encoder 10,000 ประโยค ~50 s (+ cross-encoder ~69 h) | hint ว่า "~5 s" สไลด์ txt ว่า ~50 s (5 s = ค่าของ n=1,000) | 1D |
| 2 | W9 s.39 | ตาราง ANN: recall Flat 100 / IVF ~95 / HNSW ~99 / IVF+PQ ~90 และคอลัมน์ query time | คอลัมน์ extract เพี้ยน | 1B |
| 3 | W9 s.4, 19, 24 | STS 58.02/16.50/46.35/77.03 และ SBERT-base 74.89 (ใน SimCSE) | คนละปัดเศษ/คนละ benchmark | 1B |
| 4 | W8 s.9, 21 | BART-base ~140M, large ~406M (สไลด์เขียน ~400M) | ปัดเศษไม่ตรงกัน | 1A |
| 5 | W8 s.17, 19, 33 | นับ src/tgt: `</s>` นับเฉพาะ tgt (s.17) หรือทั้งสองฝั่ง (demo); 0.2N เป็นค่าประมาณ | convention ต่างกัน — ตอบตามที่โจทย์บอก | 1A |
| 6 | W10 p.44-45 | absmax: 3.2/127=.0252, 2.1->83, deq 2.092, block 64 | hint จัดอยู่ใต้ W11 แต่เนื้อหาอยู่สไลด์ W10 (ตัวเลขตรงกับ hint) | 1D |
| 7 | W12 s.16 | ตาราง BM25 TF (f=1..100, k1=1.5) | ตารางเป็นรูป; ตัวเลขคำนวณเองจากสูตร | 1D |
| 8 | W12 s.18 | 4 scenarios (IDF deep 1.5, learning 2.5, avgdl 100, k1 1.5, b .75) และค่า L=.625/1/4 | ตารางเป็นรูป; ตัวอย่างในโน้ตตั้งเอง | 1D |
| 9 | W12 s.22 | RRF worked example (อันดับ/คะแนน) | ตารางเป็นรูป; ตัวอย่างใน sheet คำนวณเอง | 1D |
| 10 | W12 s.26 | Bi vs ColBERT vs Cross: ~100x, คุณภาพกลาง | ตารางสรุปเอง จาก text s.24-26 | 2B |
| 11 | W12 s.27 | latency 5 / 10 / 1 / 200 ms = 211 ms | ตัวเลขใน pipeline เป็นรูป ("illustrative") | 2B |
| 12 | W12 s.39, 46 | ตารางเปรียบเทียบ Vector RAG vs GraphRAG | ตารางเป็นรูป; ตาราง sheet สังเคราะห์จาก text | 2B |
| 13 | W12 s.53 | bookend ordering [1,2,3,4,5] -> [1,3,4,5,2] | slide เป็นรูป; ลำดับนี้มาจาก demo | 2B |
| 14 | W13 S41 | Topology Comparison (latency/complexity/robustness/best use) | **ตารางเป็นรูป (ยืนยันจาก txt)** + hint (4 มิติ) ใช้ชื่อ sequential/routing/parallel/hierarchical ต่างจากสไลด์ (Supervisor/Debate); ค่า Routing/Debate เป็นการอนุมาน | 2C |
| 15 | W13 S12 | ตาราง context growth (สไลด์เขียน "step 64" แต่ 128,000/2,110 = 60.7) | ตัวเลขไม่ลงรอย; sheet ใช้สูตรแทน | 2C |
| 16 | W14 p.8-9 | tokenizer ρ = 14.2x (EN 71 ch -> 10 tok; TH 57 ch -> 114 tok) | ตารางเป็นรูป; เลขมาจาก demo | 2D |
| 17 | W14 p.18-19 | Dolma ~3T vs FineWeb 15T | ตาราง p.19 เป็นรูป | 2D |
| 18 | W14 p.40 | positional bias: GPT-4 consistency 65.0% -> 77.5% (few-shot) | ตาราง p.40 เป็นรูป | 2D |
| 19 | W14 p.53 | RAGAS faithfulness = supported claims / total claims | สูตรเป็นรูป; นิยามมาจาก demo | 2D |

**จำนวนรายการที่ต้องตรวจ = 19** (ลำดับความสำคัญสูงสุดที่ควรเปิดก่อน: #14 (ครูใบ้), #7, #1, #6, #11)

หมายเหตุ ตัวเลขอื่นที่ไม่ได้ใส่ (!) แต่ควรระวัง:
- ตัวเลขทุกตัวใน sheet เป็นของสไลด์/demo/สูตรที่โน้ตตรวจแล้ว ยกเว้นตัวอย่างใน 1D ที่ติดป้าย "ตั้งเอง" ในโน้ต (triplet .087/.812/.552 มาจากสไลด์ W9 s.32; ตัวอย่าง GRPO G=4, DPO x=.4, MinHash/LSH ฯลฯ คำนวณเอง ตรวจสองรอบ)
- ตัวเลขที่ครูบอกว่า "ไม่ออก/ไม่ต้องจำ" ไม่ได้ใส่: สูตร log bucket ของ T5, สูตร online softmax ของ FlashAttention, สมการ compression (pruning ฯลฯ), รายละเอียดเชิงลึกของ MinHash hash function (ใส่เฉพาะหลักการ), W7 data-cleaning (MinHash/SHA-256 ใส่เพราะสไลด์ W14 ให้พื้นที่มากสุด)
- ตัวเลข W7 ที่ครูใบ้และใส่ไว้ใน 1D: GQA 64/8 = 8x (160 -> 20 GB), MLM 80/10/10 (ใน 1A); Pre-Norm vs Post-Norm ไม่ใส่ (อยู่ใน MEMORIZE_LIST) เพราะนอก notes สัปดาห์ 8-14


---

# VERIFY RESULT (ตรวจกับสไลด์ PDF จริงแล้ว, รายละเอียดใน verify/verify_*.md)

- MATCH 15: #1 (bi-encoder ~50 s, cross ~69 h ที่ 5 ms/pair; hint "5 s" ผิด), #2 (ANN recall 100/95/99/90), #4, #5, #6, #8, #9, #11, #12, #13 (สไลด์เป็นแม่แบบ ไม่ใช่รายการ), #15 (step 64 ถูกต้อง = 129,160), #16, #17, #18 (77.5% อยู่ในข้อความไม่ใช่ตาราง), #19
- CORRECTION 4 (แก้ในชีทแล้ว): #3 ป้ายชื่อโมเดล/benchmark, #7 ตารางสไลด์ใช้ f=0,1,2,5,10,1000, #10 ตารางเทียบแค่ Bi vs Cross ส่วน ColBERT เป็น bullet "~2 orders cheaper", #14 คอลัมน์ที่ 4 คือ Fault Tolerance และค่าจริงตามตารางใน box 2C
- จุดที่ยังเป็นอนุมาน: #13 (ลำดับ [1,3,4,5,2] คำนวณจากแม่แบบ K=5), #5 (สไลด์หน้า 33 มีแค่คำถาม ไม่มีเฉลย; BART นับ <s> และ </s>, T5 src ไม่นับ </s>, tgt นับ sentinel สุดท้ายกับ </s>)

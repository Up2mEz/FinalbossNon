n11-L3-03 | D | high | ok | A=(8-5)/sqrt(5)=3/2.236=1.342
n11-L3-04 | A | high | ok | few-shot=format consistency, CoT=multi-step, PoT=exact arithmetic (slides w11)
n11-L3-05 | B | high | ok | KV linear x3 = 15 GB; attention quadratic x9; decoding memory-bandwidth bound
n11-L3-06 | C | high | ok | C reverses exposure bias: train uses true tokens (teacher forcing), inference uses own tokens
n11-L4-01 | A | high | ok | margin beta*((-1)-(-7))=0.6, loss=-ln sigma(0.6)=0.4375 < ln2
n11-L4-02 | C | high | ok | PPO 4x26=104, GRPO 3x26=78 (no critic), DPO policy+ref=52
n11-L4-03 | D | high | ok | plurality 18 gets 3/7; temp=0 gives identical paths; 7x300=2100 tokens
n11-L4-04 | B | high | easier | S parallel 10 s; R = draft + 2x(critique+refine) = 5 sequential calls = 50 s
n11-L4-05 | C | high | ok | GRPO advantage is scale invariant: both sets give 1.732
n11-L4-06 | A | high | ok | no verifiable ground truth for email; Thinker costs tokens/TTFT; Helper for easy tasks
n11-L1-F1 | In-context learning (ICL) | high | ok | slides w11
n11-L2-F2 | Critique (self-critique/evaluate); critiquing/evaluating | med | ok | Generate-Critique-Refine; critiquing is easier than generating from scratch
n11-L3-F3 | logit = 0.5; loss = 0.4741 | high | ok | 0.1*(3-(-2))=0.5; sigma=1/(1+0.6065)=0.6225; -ln=0.4741
n11-L4-F4 | sigma = 0.4; wrong = -2.0; correct = +0.5 | high | ok | mu=0.8, var=0.16
n12-L1-01 | B | high | ok | b controls length normalization
n12-L1-02 | C | high | ok | vocabulary mismatch
n12-L1-03 | A | high | ok | slide: Linear layer -> raw scalar logit
n12-L1-04 | D | high | ok | (Subject, Relation, Object)
n12-L1-05 | B | high | ok | U-shaped, lowest in middle
n12-L1-06 | C | high | ok | cut where 1 - cos of adjacent sentences exceeds tau
n12-L2-01 | D | high | ok | ceiling = k1+1 = 4.00; b=0 irrelevant to ceiling
n12-L2-02 | A | high | ok | 1.00 -> 1.43 = +42.9%
n12-L2-03 | B | high | ok | (1/61-1/62)/(1/61)=1/62=1.6%
n12-L2-04 | C | high | ok | late interaction MaxSim, ~2 orders cheaper than cross-encoder
n12-L2-05 | D | high | ok | cross-corpus themes = global search
n12-L2-06 | A | high | ok | bookend: top-1 first, 3..K middle, top-2 last
n12-L3-01 | C | high | easier | ratio of TF_A to TF_B = 1.28
n12-L3-02 | B | high | ok | X=0.01639, Y=0.03078, Z=0.03175 -> Z,Y,X
n12-L3-03 | D | high | ok | 10+1+200*20/80=61 ms
n12-L3-04 | A | high | ok | attribute/negation/relation failures of CLIP
n12-L3-05 | C | high | ok | multi-hop A->B->C chain; graph / Text-to-Cypher
n12-L3-06 | B | high | ok | rules oppose blindly raising K
n12-L4-01 | A | high | ok | 4*TF: 9.95 vs 9.52 (+4.5%); saturation
n12-L4-02 | D | high | ok | b=1: 50->2.000, 100->1.667, 500->0.714
n12-L4-03 | C | high | ok | k=60: Q 0.0317 > P 0.0164; k=0: P 1.0 > Q 0.667
n12-L4-04 | B | high | easier | rank 70 never enters reranker; raise K
n12-L4-05 | D | high | ok | hybrid+rerank for (1)(2), GraphRAG for comparative (3)
n12-L4-06 | A | high | ok | tau=0.1: 0.744; tau=0.01: 0.000136
n12-L1-F1 | 0 ; n_t = N/2 | high | ok | ln(1)=0 when N-n_t = n_t
n12-L2-F1 | recall ; precision | high | ok | Stage 1 recall, Stage 2 precision
n12-L3-F1 | TF = 1.571 ; ceiling = 2.2 (k1+1) | high | ok | 3*2.2/(3+1.2)
n12-L4-F1 | r = 61 ; equal at r = 62 | high | ok | 2/(60+r) > 1/61 iff r < 62
n13-L1-01 | C | high | ok | safety guard
n13-L1-02 | A | high | ok | blank form analogy
n13-L1-03 | D | high | ok | regex -> DFA
n13-L1-04 | B | high | ok | error cascade
n13-L1-05 | A | high | ok | semantic memory
n13-L1-06 | D | high | ok | Layer 4 Escalate = human approval
n13-L2-01 | B | high | ok | 27.4-29.4=-2.0; 60.9-56.3=+4.6
n13-L2-02 | B | high | ok | slide: Hallucinated tool name -> Tool not found -> error message
n13-L2-03 | C | high | ok | Routing: Low/Low/Medium
n13-L2-04 | C | high | ok | mask depends on parser state
n13-L2-05 | A | high | ok | arguments object with query string
n13-L2-06 | D | high | ok | slide: may miss temporally important items
n13-L3-01 | C | high | ok | CoT search-error entry is "-" (none), so C is false
n13-L3-02 | D | high | ok | live per-request state
n13-L3-03 | B | high | ok | masked 50,188 vs 111 -> 452x
n13-L3-04 | A | high | ok | only Debate has High FT + Medium latency + Medium complexity
n13-L3-05 | D | high | ok | hybrid practical default
n13-L3-06 | A | high | ok | validate semantics (month 81)
n13-L4-01 | B | high | ok | (124000-15220)/2110=51.56 -> step 61 (122,830; step 62 = 124,940)
n13-L4-02 | C | high | ok | ReAct reasoning error 47%, empty search 23%
n13-L4-03 | A | high | ok | stack top is array -> ] } }
n13-L4-04 | D | high | ok | bounded depth is regular, unbounded needs CFG
n13-L4-05 | B | med | ok | summary lossy, Markov not guaranteed
n13-L4-06 | C | high | ok | Sequential 26 s; Debate 9+4=13 s; diff 13
n13-L1-F1 | description ; parameters (JSON Schema/input schema) | high | ok | slides w13
n13-L2-F1 | HotpotQA ; FEVER | high | ok | table 1
n13-L3-F1 | 5 ; 2 ; masked 50,188 | high | ok | 62+5+2=69; 50,257-69
n13-L4-F1 | Buffer 3000 ; Summary 500 ; Retrieval 250 | med | ok | 60*50; 4*50+300; 5*50
n14-L1-01 | B | high | ok | alpha-word ratio >= 0.8
n14-L1-02 | C | high | ok | r=100/20=5
n14-L1-03 | A | high | ok | Era 2
n14-L1-04 | D | high | ok | 57 subjects
n14-L1-05 | C | high | ok | positional bias
n14-L1-06 | B | high | ok | monthly
n14-L2-01 | D | high | ok | 7/10
n14-L2-02 | A | high | ok | MinHash+LSH
n14-L2-03 | C | high | ok | Dolma mixed sources + tagger logs
n14-L2-04 | B | high | ok | recall: row max over ref tokens (0.8,0.9) avg 0.85
n14-L2-05 | D | high | ok | 1-C(19,5)/C(20,5)=1-15/20=0.25
n14-L2-06 | A | high | ok | low ASR can come from over-refusal
n14-L3-01 | A | high | ok | B first: 3+0.2*2=3.4 vs A first 4.7
n14-L3-02 | D | high | ok | 4*9=36x; MinHash compare cost fixed at H
n14-L3-03 | B | high | ok | cannot derive; need clean-subset rerun
n14-L3-04 | C | high | ok | controlled padding of same content
n14-L3-05 | B | high | ok | BERTScore and BLEU not comparable scales
n14-L3-06 | C | high | ok | (1/2+2/3+3/5)/3=0.589
n14-L4-01 | D | high | ok | b=16,r=8: s*=0.707; P(0.7)=0.613
n14-L4-02 | A | high | ok | 3.5T; 1.3/3.5=37%; 15/3.5=4.3x
n14-L4-03 | D | high | ok | (0+1+0.5333)/3=0.511
n14-L4-04 | A | high | ok | 54+15=69%; exclude 54/70=77.1%
n14-L4-05 | B | high | ok | 80,500+22*2110=126,920; 23 steps exceed
n14-L4-06 | C | high | ok | slide: Gopher repetition rules
n14-L1-F1 | 176 languages ; 0.65 | high | ok | slides w14
n14-L2-F1 | 0.72 | high | ok | 1.08/1.5
n14-L3-F1 | r = 4 ; s* = 0.42 | high | ok | 128/32; (1/32)^(1/4)
n14-L4-F1 | 0.03125 (~0.031) ; H = 1024 | high | ok | 1/(2*16); error ~1/sqrt(H), so 4x H

## Issues
- n11-L4-04: answer B depends on reading "2 rounds" as 2x(critique+refine) = 5 calls. Option D (3 calls) and C leak via obvious wording.
- n12-L3-01: level 3 is a single formula plug-in; closer to level 2.
- n12-L4-04: answerable by logic from option wording (K cutoff); easier than level 4.
- n12-L1-F1: second blank expected N/2, not a number; grader must accept N/2.
- n13-L4-F1: "turn" vs "message" ambiguous. Summary 500 assumes 1 turn = 1 message; Retrieval 250 ignores system prompt/current query. Alternative answers possible.
- n13-L4-05: slide support for "Markov not guaranteed under summary memory" not confirmed in sources; only inferable.
- n13-L2-02: option A (parser error) arguably overlaps with tool-not-found; B matches slide table only.
- n14-L4-F1: halving SE to 0.0156 needs H=1024 (computed, slide gives only the 1/(2 sqrt(H)) formula).
- n14-L3-02, n14-L4-04, n14-L4-06: wrong options contain strong absolute qualifiers, so the answer is guessable from wording.
- n11-L4-01 and n14-L4-01 rely on stem-supplied constants; values confirmed by computation.

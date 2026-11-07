# Solutions (independent, Sonnet) - q_gemini.md

Q1 | B | high | W8 notes: token deletion leaves no placeholder; infilling Poisson(3) -> single [MASK], 0-length = insertion
Q2 | D | high | W8 demo: target ends with <extra_id_2> then </s>
Q3 | A | high | (1/0.2)^2 = 25x decoder self-attn; overall ~2-3x (encoder unchanged)
Q4 | C | high | W8 s.37: P/2 enc + P/2 dec; Dec-only uses all P; RLHF multi-turn fit
Q5 | A | high | W8 demo: no prefix -> German translation fallback
Q6 | C | high | Q from decoder, K/V from encoder, no causal mask, [tgt_len, src_len]
Q7 | D | high | W8 demo: BERT highlights the question itself; T5-small says "Germany"
Q8 | B | high | W8 s.20: 32 buckets/head, offset 0-7 exact, rest log-shared, added to logits pre-softmax
Q9 | C | high | Enc-Dec = N(1000^2+100^2+100*1000)=1,110,000N; Dec-only = 2N*(1100^2/2)=1,210,000N; ratio 1.09
Q10 | D | high | MT (NLLB), ASR (Whisper) still Enc-Dec
F1 | (1) Transfer; (2) sentinel token; (3) 0.2N (about 20%); (4) 25; (5) Text Infilling | high | W8 notes
Q11 | A | high | W9 demo: raw BERT cos floor ~0.457 (mean 0.58), ranking inversion 0.861 vs 0.794
Q12 | C | high | 10^4*9999/2 = 49,995,000 pairs * 5 ms = 249,975 s = 69.4 h; retrieve-then-rerank
Q13 | B | high | c=[u;v;|u-v|], 3*768 = 2,304
Q14 | D | high | without attention_mask, [PAD] vectors enter the mean
Q15 | A | med | L=0 iff d(a,n) >= d(a,p)+alpha; D says "only exactly equal" (wrong), B/C give L>0. See Issues
Q16 | B | high | W9 notes: cosine-triplet == margin ranking; Euclidean slack differs (-1.6048 vs -0.0427)
Q17 | D | high | unit vectors: ||u-v||^2 = 2-2cos
Q18 | C | high | 5e6*768*4 B = 15.36e9 B = 15.36 GB
Q19 | B | high | (1+1/2+1/4)/3 = 0.5833
Q20 | A | high | more batch = more in-batch negatives; smaller tau sharpens softmax
F2 | (1) 0 ; (2) 0.552 | high | (1) max(0,0.087-0.812+0.5)=max(0,-0.225)=0; (2) 0.087-0.035+0.5=0.552
Q21 | D | high | weights static, KV cache grows with sequence length
Q22 | A | high | 2*32*8*128*2*4096*16 = 8.59e9 B ~ 8.6 GB; MHA 34.4 GB; saves 75%
Q23 | B | high | Prefill GEMM compute-bound, decode GEMV memory-bandwidth-bound
Q24 | D | high | exact attention, tiling + online softmax in SRAM, O(N^2)->O(N) extra memory
Q25 | C | high | PagedAttention page table, KV use ~30% -> ~95% (W10 s.30-38); A false (same distribution), B false (iteration-level), D false (spec. decoding suits single-user/local)
Q26 | B | high | all ~47B (~94 GB FP16) must be resident; MoE saves compute not memory
Q27 | A | high | A Gaussian init, B zero -> dW=0; QLoRA dequantizes on the fly; DQ quantizes scales; paged optimizer -> CPU RAM
Q28 | C | high | 16N = 2N+2N+12N (75% optimizer); checkpointing O(L)->O(sqrt L), ~+30% time
Q29 | B | high | s=3.2/127=0.0252; q=round(83.3)=83; xhat=83*0.025197=2.0914 ~ 2.092
Q30 | C | high | KD cuts layers/FLOPs; PTQ reduces memory/bandwidth, same layers
F3 | (1) NF4 (NormalFloat4); (2) Double Quantization; (3) Paged Optimizers | high | W10 notes
Q31 | B | high | log-ratio w = +2, l = -2; beta*(2-(-2)) = 0.4; loss = 0.5130
Q32 | D | high | higher beta = stronger KL tie to reference; DPO needs only policy+reference, no reward model
Q33 | B | high | RM = SFT with scalar head, Bradley-Terry loss; ref frozen, KL is a leash not a speedup
Q34 | A | high | mean 0.25, pop std sqrt(0.1875)=0.4330; A1 = 0.75/0.433 = 1.732
Q35 | D | high | R1 shows <think>, o-series hides trace; R1-Zero needs no human reasoning SFT
Q36 | C | high | PoT offloads arithmetic to interpreter; A describes GoT, B describes ToT
Q37 | A | high | temperature>0, sample several paths, majority vote
Q38 | C | high | exposure bias from teacher forcing vs autoregressive inference
Q39 | D | high | indirect prompt injection; sandwich defense + delimiters
Q40 | A | high | attention compute quadratic, KV cache linear; PagedAttention reduces fragmentation (W11 s.9)
F4 | Audience | high | CO-STAR = Context, Objective, Style, Tone, Audience, Response
Q41 | B | high | b=0: f(k1+1)/(f+k1) -> ceiling k1+1 = 2.5; f=1000 gives 2.496 (~2.5x of f=1)
Q42 | C | high | A(|d|=50): 7.5/(3+1.5*0.625)=1.905; B(|d|=100): 7.5/4.5=1.667; shorter doc wins
Q43 | A | high | X = 1/61+1/63 = 0.0323; Y = 1/62 = 0.0161
Q44 | D | high | Stage 1 high recall, Stage 2 high precision on 20-100 candidates; CE cannot precompute
Q45 | B | high | symmetric InfoNCE, 400M pairs, zero-shot classification
Q46 | A | high | Vector RAG weak on global queries; GraphRAG Global Search = map-reduce over Leiden community summaries
Q47 | C | high | W12 demo: bookend_order([1..5]) = [1,3,4,5,2]
Q48 | D | high | generic chunks in every top-k = hubness; inflated cosine = anisotropy; MiniLM silently truncates at 256 tokens
Q49 | B | high | Dense wins synonyms; BM25 wins exact IDs
Q50 | A | high | long context has O(N^2) cost + lost in the middle; fine-tuning is poor for frequently changing facts
F5 | 60 | high | RRF k=60 (Cormack 2009)
F6 | Leiden | high | fixes Louvain's badly-connected communities
Q51 | C | high | LLM emits text; external dispatcher executes and returns Observation
Q52 | A | high | W13 Table 1: HotpotQA ReAct 27.4 < CoT 29.4; FEVER ReAct 60.9 > CoT 56.3
Q53 | D | high | W13 demo: parse error becomes Observation "Error: could not parse an Action..." so model self-corrects
Q54 | B | high | e^2/(e^2+e^0.5) = 7.389/9.0377 = 0.8176
Q55 | A | high | masking guarantees syntax not semantics; nested JSON needs CFG / pushdown automaton
Q56 | C | high | W13 demo: Buffer linear, Summary sawtooth, Retrieval flat O(k)
Q57 | D | high | 3+10+4 = 17 s; sequential = error cascade
Q58 | B | high | first match 'premium'/'plan' -> Sales (W13 demo); routing weakness = SPOF at router (misroute)
Q59 | C | high | augmented state (q,h_{1:t-1}) keeps Markov; Working/Episodic/Semantic/Procedural
Q60 | A | high | Constrain->Validate->Recover->Escalate; max_steps
F7 | -infinity ; 0 | high | W13 S21
F8 | 27.4 ; 60.9 | high | ReAct Table 1 (W13 notes)
Q61 | C | high | LSH banding generates candidates; verify with true Jaccard
Q62 | A | high | MinHash/LSH are pairwise cross-doc; internal repetition needs Gopher repetition heuristics
Q63 | C | high | repetitive/spam gets low PPL; technical text may fall in high-PPL tail; CCNet uses percentile buckets (head/middle/tail), not a fixed global threshold
Q64 | B | high | A first: 0.1+0.3*10 = 3.10; B first: 10+0.5*0.1 = 10.05
Q65 | B | high | BERT uncased has no Thai vocab, whole text -> single [UNK] (TPC 1/57=0.0175), coverage failure not compression
Q66 | D | high | BLEU n-gram surface matching rewards word salad; BERTScore uses contextual embeddings + greedy cosine matching
Q67 | C | high | pass@100 = 1 - C(199,100)/C(200,100) = 1 - 100/200 = 0.5
Q68 | A | high | SWE-bench resolved = fail-to-pass and pass-to-pass; others false (contamination not subtractable, HarmBench uses classifier, variants not comparable)
Q69 | D | high | length + position bias; swap test reduces but does not eliminate bias
Q70 | B | high | Faithfulness = claims supported by retrieved context, reference-free; 1.0 does not mean factually true
F9 | 20 ; 3.5 trillion (3.5T) | high | Chinchilla ~20 tokens/param; 175B*20 = 3.5T
F10 | SimHash ; MinHash | high | SimHash 64-bit fingerprint + Hamming distance (cosine); MinHash approximates Jaccard of shingle sets

## Issues

- Q15: A is correct, but D ("loss = 0 only when d(a,p)-d(a,n) = alpha exactly") is also a plausible distractor; it is wrong only because of the word "only" (loss is 0 for every d(a,n) >= d(a,p)+alpha). Exactly one defensible answer (A). Low risk.
- Q9: the Dec-only figure 1,210,000N depends on the slide convention 2N x (1100^2/2). Read literally ("2N layers", full 1100^2 each) one would get 2.42M N, so answer C only holds with the slide formula. Option D shows the 605,000N trap. Answerable from slides, but ambiguous from the stem alone.
- Q5: the German fallback is demo-specific T5-small output, not a guarantee; the stem says "as found in Demo", so acceptable.
- Q3 / F1(4): "overall speedup 2-3x" is a slide estimate, not computable from the stem; fine per W8 s.19/21.
- Q25: the "~30% -> ~95%" figure is slide-given; notes also quote "naive 20-40% -> ~95%". Still consistent with C.
- Q22: decimal GB (8.59 GB) only; in GiB it would be 8.0. Unambiguous only because the stem fixes 10^9.
- Q58: B's stated weakness (Router SPOF) is the notes' label for routing; a more precise weakness is first-match keyword ambiguity. B is still the only option with the correct outcome (Sales).
- Q65 / Q66: TPC 0.0175, rho ~14.2 and the BLEU-1 values 0.833 / 0.167 come from demo output, not slides; I did not recompute them. The answers do not depend on those numbers.
- Q47: answer [1,3,4,5,2] rests on the demo's bookend_order code; the slide table for bookend is an image (flagged unreadable in W12 notes).
- No question found with 0 correct options, and none with 2+ correct options apart from the minor Q15 wording point.

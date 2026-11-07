Q1 | B | high | BART deletion leaves no placeholder (w8 s.10-12)
Q2 | D | high | permutation = shuffle sentences
Q3 | A | high | text infilling: one [MASK] per span (s.11)
Q4 | C | high | input 12-5+2=9; target 5+2+1+1=9
Q5 | B | high | (1/0.2)^2=25
Q6 | D | high | T5 text-to-text with task prefix
Q7 | A | high | learned bias per relative-distance bucket added to logits (s.20)
Q8 | C | high | [B,H,tgt,src]=[2,12,20,50]
Q9 | B | high | 15%*200=30; CLM 100%=200
Q10 | D | high | demo w08: no prefix -> German translation-like output
Q11 | A | high | 3*768=2304
Q12 | C | high | 1000*999/2=499500
Q13 | D | high | 1000*5ms=5s
Q14 | B | high | 0.20-0.55+0.50=0.15
Q15 | A | high | N-1=63 (w9 notes/slides N-1)
Q16 | C | high | (2+4)/2=3
Q17 | B | high | unit-norm vectors: dot=cos
Q18 | D | high | retrieve with bi-encoder then rerank with cross-encoder
Q19 | A | high | semi-hard: d(a,p)<d(a,n)<d(a,p)+margin
Q20 | C | high | (1+1/2+1/4)/3=0.5833
Q21 | D | high | 16e9/2^30=14.90 (w10 notes)
Q22 | B | high | 7e9*2=14 GB
Q23 | C | high | 536,870,912 B = 512 MiB
Q24 | A | high | 8/32 -> 75% reduction
Q25 | D | high | 7e9*16=112 GB
Q26 | B | high | 2*4096*16=131,072
Q27 | C | high | MoE saves compute, all experts must be in memory (w10 s., 47B load)
Q28 | A | high | tiling + online softmax, exact, no full N x N to HBM
Q29 | D | high | PagedAttention = block KV cache allocation
Q30 | B | high | 2.1/0.0252=83.3 -> 83 (slides_w10 absmax s=3.2/127)
Q31 | C | high | ICL: no weight update
Q32 | A | high | 12 appears 3/5=60%
Q33 | D | high | SFT = next-token prediction on demonstrations
Q34 | B | high | SFT -> RM -> PPO
Q35 | C | high | 0.1*(2-(-2))=0.4
Q36 | A | high | mean .5, pop std .5, A=+1
Q37 | D | high | 4^2=16 attention, 4 KV cache
Q38 | B | high | indirect prompt injection via external data
Q39 | C | high | DPO: preferences vs reference, no RM, no PPO
Q40 | A | high | PoT: LLM writes code, interpreter computes
Q41 | D | high | ceiling k1+1=2.5
Q42 | B | high | 2*2.5/3.5=1.4286
Q43 | C | high | 5/2.9375=1.702
Q44 | A | high | 1/61+1/63=0.03227
Q45 | D | high | 15 negatives
Q46 | B | high | 512-64=448
Q47 | C | high | BM25+dense -> RRF -> CE rerank -> LLM
Q48 | A | high | GraphRAG global = map-reduce over community summaries
Q49 | D | high | demo_w12 bookend: [1,3,4,5,2]
Q50 | B | high | 1e6*384*4=1.536e9 B
Q51 | C | high | Thought-Action-Observation
Q52 | A | high | parser validates, dispatcher routes (w13 S17)
Q53 | D | high | 60.9-56.3=4.6
Q54 | B | high | 50257-69=50188
Q55 | C | high | regex gives format only, not validity
Q56 | A | high | 3+10+4=17
Q57 | D | high | router = single point of failure
Q58 | B | high | supervisor-led = delegate + evaluate
Q59 | C | high | context limited by k, storage grows
Q60 | A | high | required key query with correct type
Q61 | D | high | hash for exact (after normalization); shingles+MinHash for near
Q62 | B | high | union=6, 3/6=0.5
Q63 | C | high | 42/64=0.65625
Q64 | A | high | 1-(1-0.25)^2=0.4375
Q65 | D | high | repetitive text low PPL; technical text high PPL (w14 p.15 Perplexity Trap)
Q66 | B | high | A->B: 1+0.2*10=3 ms; B->A: 10+0.5*1=10.5 ms
Q67 | C | high | 2*.48/1.4=0.6857
Q68 | A | high | 1-28/45=0.3778
Q69 | D | high | swap positions (Position Swap)
Q70 | B | high | faithfulness = grounded in context, context may be wrong/incomplete
F1 | 16 ; 10 | high | input 20-6+2=16; target 6+2+1+1=10
F2 | 0.552 | high | 0.087-0.035+0.5
F3 | 4 | high | 2*2*32*8*128*8192*4=4,294,967,296 B = 4 GiB
F4 | mean 0.5 ; advantage +1 | high | pop std 0.5
F5 | 1.818 (20/11) ; 2.5 | high | 4*2.5/5.5; ceiling k1+1
F6 | 0.03252 | high | 1/61+1/62=0.016393+0.016129
F7 | 49,566 ; -inf | high | 50257-691; logits set to -inf (w13 logit masking)
F8 | 10 | high | 2+3+5
F9 | 8 ; 0.5 | high | union 7+5-4=8; 4/8
F10 | 0.3778 (17/45) | high | 1-C(8,2)/C(10,2)=1-28/45

## Issues
- Q4 / F1: wording lists only sentinels and </s> for the target, with span tokens implied; answer assumes target also contains the removed tokens (otherwise 4 would be the target length for Q4, which is not an option). Mildly ambiguous but C is the only consistent option.
- Q9: depends on the 'slide model' (MLM 15% targets, CLM 100%); answer B follows slides_w08 table, though it is a simplification.
- Q15: assumes 'in-batch negatives = N-1 = 63' per w9 slides; wording 'positive ของ query อื่น' is consistent, no issue found otherwise.
- Q27: B and C could both look plausible to a careless reader; B contradicts slides (all experts must be loaded), so C only.
- Q30: week tag W10 matches slides_w10 (absmax 3.2/127); week_11_notes flag claims it is absent from w11 slides, which is a stale note, not a question problem.
- Q53: relies on ReAct Table 1 numbers (60.9 vs 56.3) which are confirmed in hints/notes; slide table may be an image, so verified only via notes.
- Q49: slides/demo state bookend; hint labels it week 13 but content is in w12 slides/demo (tag W12 is fine).
- Q66: distractor A says B->A costs 3 ms (actual 10.5); exactly one correct option (B).
- No question found with 0 or 2+ correct options.

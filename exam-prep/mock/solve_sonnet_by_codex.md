Q1 | C | high | BART's five corruptions exclude numbered T5 sentinels; slides_w08, noising.
Q2 | B | high | Poisson λ=3; one mask per span; slides_w08 p.11.
Q3 | A | high | Deletion leaves no placeholder; notes/week_08_notes.md §2.5.
Q4 | D | high | Two removed spans plus terminal sentinel and EOS; slides_w08 p.17.
Q5 | B | high | 4 removed tokens + 2 span sentinels + 1 final sentinel = 7; slides_w08 p.17.
Q6 | C | high | (N/0.2N)² = 25; decoder self-attention only; slides_w08 p.19.
Q7 | A | high | 32 buckets per head, nearby offsets exact, bias added to logits; notes/week_08_notes.md §2.6.
Q8 | D | high | 15 masked tokens versus approximately 100 CLM prediction positions; slides_w08.
Q9 | B | high | Single linearly growing KV cache simplifies serving; notes/week_08_notes.md §2.1.
Q10 | C | high | Demo without prefix produces German translation; demo_w08.
Q11 | D | high | 3 × 768 = 2,304; slides_w09 classification head.
Q12 | A | high | Loss is zero when d(a,p) − d(a,n) + α ≤ 0; slides_w09 triplet loss.
Q13 | C | high | max(0, 0.1 − 0.2 + 0.5) = 0.4; slides_w09.
Q14 | B | high | 1,000 × 999 / 2 = 499,500 unordered distinct pairs; slides_w09.
Q15 | D | high | Cross-encoder runs per pair; bi-encoder embeds each sentence once, then compares vectors; slides_w09.
Q16 | A | high | Raw BERT CLS = 16.50 on slide's STS table; slides_w09 p.4.
Q17 | B | high | Mean pooling excludes padding by attention mask; SBERT default/ablation, slides_w09.
Q18 | C | high | N − 1 = 255 other positives used as negatives; notes/week_09_notes.md.
Q19 | A | high | Inner product of L2-normalized vectors equals cosine; slides_w09 FAISS.
Q20 | D | high | (1 + 1/2 + 1/4)/3 = 0.58333; notes/week_09_notes.md MRR.
Q21 | B | high | 16 × 10⁹ / 2³⁰ = 14.901 GiB; notes/week_10_notes.md.
Q22 | D | high | 70 × 10⁹ × 0.5 / 10⁹ = 35 GB; slides_w10 memory.
Q23 | A | high | 160 × 8/64 = 20 GB, assuming equal head dimensions; slides_w10 GQA.
Q24 | C | high | Noncontiguous fixed-size KV blocks mapped via page table reduce fragmentation; slides_w10.
Q25 | B | high | Random A and zero B give BA=0 initially; slides_w10 LoRA.
Q26 | A | high | 16 × (4096 + 4096) = 131,072; slides_w10 LoRA.
Q27 | D | high | NF4, double quantization and paged optimizers; slides_w10 QLoRA.
Q28 | C | high | Course model assumes all experts resident; sparse activation reduces compute, not weight memory; notes/week_10_notes.md §2.10.
Q29 | B | high | Exact tiled attention and online softmax avoid materializing N×N in HBM; slides_w10 FlashAttention.
Q30 | A | high | round(2.1/(3.2/127)) = round(83.34375) = 83; slides_w10 quantization.
Q31 | C | high | Context, Objective, Style, Tone, Audience, Response; slides_w11 p.5.
Q32 | D | high | Sample multiple reasoning paths and vote on answers; slides_w11 self-consistency.
Q33 | B | high | GoT permits aggregation and refinement beyond tree branches; notes/week_11_notes.md.
Q34 | A | high | LLM generates code; external interpreter performs arithmetic; slides_w11 PoT/PAL.
Q35 | C | high | PPO updates actor and critic; reward and reference remain fixed; slides_w11 RLHF.
Q36 | B | high | Equal policy/reference makes logit zero, −ln σ(0) = ln 2; slides_w11 DPO.
Q37 | D | high | −ln σ(0.1 × 4) = 0.513015; slides_w11 DPO loss.
Q38 | A | high | Mean=0.25, population std=√0.1875; 0.75/std=1.732; slides_w11 GRPO.
Q39 | C | high | GRPO uses group-relative advantage without a critic; slides_w11.
Q40 | D | high | Demo reports both as 494.0M; demo_w11 model loading.
Q41 | A | high | As f→∞, f(k1+1)/(f+k1L)→k1+1; slides_w12 BM25.
Q42 | C | high | L=1 when b=0; 3×2.5/(3+1.5)=1.6667; slides_w12.
Q43 | B | high | For equal f, shorter document has smaller denominator; limit is k1+1; slides_w12 BM25.
Q44 | D | high | 1/61 + 1/63 = 0.0322665 ≈ 0.0323; slides_w12 RRF.
Q45 | A | high | RRF fuses ranks instead of incompatible raw scores; slides_w12.
Q46 | B | high | Joint query-document encoding costs a forward pass per pair; slides_w12 reranking.
Q47 | C | high | Embed templated class texts and choose greatest image-text similarity; slides_w12 CLIP.
Q48 | D | high | Offline community summaries; online map-reduce for global query; slides_w12 GraphRAG.
Q49 | A | high | Demo puts ranks 1 and 2 at opposite ends to address lost in the middle; demo_w12 §16.
Q50 | B | high | 1,000,000×384×4 = 1.536×10⁹ bytes ≈ 1.5 GB; slides_w12 embedding storage.
Q51 | D | high | Thought → Action → Observation; observation enters subsequent context; slides_w13 ReAct.
Q52 | C | high | Slide table: HotpotQA 29.4 vs 27.4, FEVER 56.3 vs 60.9 (CoT vs ReAct); slides_w13 S14.
Q53 | A | high | Table's failure categories: hallucination CoT 56%, ReAct 0%; reasoning CoT 16%, ReAct 47%; slides_w13 S15.
Q54 | B | high | Invalid logits are −∞ before softmax, leaving probability zero; slides_w13 S21.
Q55 | D | high | 50,257 − 69 = 50,188; slides_w13 S22.
Q56 | C | high | Unbounded matching brackets require stack memory beyond a DFA; slides_w13 CFG vs regex.
Q57 | B | high | Bounded summary loses information; full buffer grows with conversation; notes/week_13_notes.md §2.
Q58 | A | high | Sequential latency 3 + 10 + 4 = 17 seconds; slides_w13 S34.
Q59 | D | high | Wrong dispatch is a single point of failure; notes/week_13_notes.md topology.
Q60 | C | med | Supervisor can reassign and self-correct at greater latency and coordination cost; notes/week_13_notes.md S38–41.
Q61 | B | high | 42/64 = 0.65625; slides_w14 MinHash.
Q62 | A | high | MinHash estimates Jaccard; SimHash uses bit fingerprints and Hamming distance for cosine similarity; slides_w14.
Q63 | D | high | MinHash compresses per-pair comparison; LSH reduces candidate pairs; slides_w14 dedup.
Q64 | C | high | 70B × 20 = 1,400B = 1.4T tokens; slides_w14 Chinchilla.
Q65 | B | high | 1 − C(199,100)/C(200,100) = 100/200 = 0.5; slides_w14 pass@k.
Q66 | D | med | BLEU-1 is 5/6 vs 1/6; slide reports opposite BERTScore ranking; slides_w14 pp.33,36; demo_w14 §12–15.
Q67 | A | high | Judge both orders, require consistent winner; slides_w14 position swap.
Q68 | C | high | Fail-to-pass tests pass and pass-to-pass tests remain passing; slides_w14 SWE-bench.
Q69 | B | high | All Gopher rules must pass, including word count and alpha-word ratio; slides_w14.
Q70 | A | high | Low perplexity can favor repetitive spam and high perplexity can penalize rare technical content; slides_w14 CCNet.
F1 | SentencePiece; 32,000; 50,265 | high | T5 and BART tokenizer/vocabulary table; slides_w08 p.21.
F2 | 0 | high | max(0, 0.1 − 0.9 + 0.5) = 0; slides_w09 triplet loss.
F3 | 1,120 GB; 14 H100 80 GB GPUs | high | 70B×16 bytes = 1,120 GB; 1,120/80 = 14, excluding activations; slides_w10.
F4 | Let's think step by step; 78 | high | Slide reports 17%→78%; slides_w11 p.10.
F5 | 60; 0.0164 | high | 1/(60+1) = 0.0163934; slides_w12 RRF.
F6 | k1 + 1; 2.5 | high | Limit of BM25 TF as f→∞; slides_w12.
F7 | 60.9; 56.3 | high | FEVER ReAct and CoT in slide table; slides_w13 S14.
F8 | Validate; Recover | high | Constrain → Validate → Recover → Escalate; slides_w13 S46.
F9 | 3; 1.3 | high | Classifier threshold ≥3/5; 1.3T from 15T; slides_w14 FineWeb-Edu.
F10 | 32; 64 | high | 256/8 = 32 bytes; 256/4 = 64 hexadecimal digits; slides_w14 SHA-256.

## Issues

- Q60: The four-dimension topology comparison table (S41) is an image missing from the supplied slide text. The notes explicitly say their comparison table is synthesized from surrounding text and hints; C is the best-supported choice, but its exact tabular wording cannot be verified from the supplied slide text (notes/week_13_notes.md §8).
- Q66: The BLEU values can be recomputed as 5/6 and 1/6. The stated BERTScore ranking is reported in slides_w14 p.36, but the corresponding printed ranking in demo_w14 is truncated, so it could not be independently verified from the available output. The demo's code prints the asserted ranking; it is not itself a numeric result.

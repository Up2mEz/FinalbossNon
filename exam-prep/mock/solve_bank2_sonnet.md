n08-X4-01 | B | high | ok | src=60-12+3=51 both; BART tgt=62, T5 tgt=3+12+1+1=17; (51^2+62^2)/(51^2+17^2)=6445/2890=2.23
n08-X4-02 | D | high | easier | text-to-text: per-task prefix, number as text (3.8), same CE loss/decode, mixed batches; no extra heads
n08-X4-03 | A | high | easier | shuffled sentences=Sentence Permutation; deletion with no placeholder=Token Deletion; causal left-to-right=[S]-Denoising (prefix LM)
n08-X4-04 | C | high | ok | Enc-Dec 11B decoder ~P/2=5.5B > 4.8B, so Dec-only's zero-shot win after pretraining is not explained by generate-params; s.39 says Enc-Dec wins only after multitask finetune
n08-X4-05 | B | med | ok | 4 steps (3 tokens + </s>); cross=4x120x12=5760; self=(1+2+3+4)x12=120; total 5880
n08-X4-F1 | BART = 4,222,260 ; T5 = 1,088,000 | high | ok | BART tgt=42 -> 2x42x50265; T5 tgt=3+12+1+1=17 -> 2x17x32000
n09-X4-01 | B | high | ok | negatives per query = N-1; ratio 255/31 = 8.23 (same number of queries in both)
n09-X4-02 | D | high | ok | cos<=1 so labels 5,4,1 all pushed to cos=1 (errors 16+9+0), label 0 -> cos 0; sum 25/4 = 6.25; three cos equal so ranking lost
n09-X4-03 | A | high | easier | Recall@10 A = 3/4 = 0.75 (rank 12 outside); MRR = (1/3 + 1)/2 = 0.667
n09-X4-04 | C | high | ok | d(a,p)=0.6; loss = max(0, 2.1 - d(a,n)), d(a,n)<=2 so min loss 0.1 at cos=-1, always positive
n09-X4-05 | D | high | ok | 46.35-16.50=29.85; 58.02-46.35=11.67; 77.03-46.35=30.68; mean pooling alone still below GloVe
n09-X4-F1 | 76.8 ; 0.768 ; 307.2 | high | ok | 200e6x384=7.68e10 ops /1e9 = 76.8 s; /100 = 0.768 s; 200e6x384x4 B = 307.2 GB
n10-X4-01 | C | high | ok | 1.2e5 > FP16 max 65,504 but within BF16 range; T4 (Turing) has no native BF16; weights stay 2 B/param=16 GB; mantissa 7 vs 10 bit
n10-X4-02 | B | high | ok | 8e9x16 = 128 GB > 80 GB by 48 GB (excludes activations); checkpointing adds ~30% compute: 10 -> 13 h
n10-X4-03 | D | low | ok | see Issues: slide table says TP intra-node / PP inter-node (A), but model (140 GB) fits in one node (8x80), and D gives low latency (no PP bubbles) plus throughput via DP
n10-X4-04 | A | high | ok | FlashAttention: exact result, extra memory O(N), no NxN in HBM; KV cache = 2x32x8x128x2 B = 131,072 B/token x 32,000 = 4.19 GB unchanged
n10-X4-05 | B | high | ok | TTFT=60+410+40=510 ms >500; prefill is compute-bound; speculative decoding improves TPS not TTFT; chunked prefill worsens new-request TTFT
n10-X4-F1 | 54.5 ; 0.68 ; 6.9 | med | ok | per layer: Q 64x8192 + K 64x5120 + V 64x5120 + O 64x8192 = 1,703,936; x32 = 54.53M; /8B = 0.68%; 54.53M x16 B = 0.87 GB + 4 + 2 = 6.9 GB
n11-X4-01 | C | high | easier | external data in email = indirect injection; LLM cannot separate instruction/data; defense in depth (least privilege + human confirm + second LLM)
n11-X4-02 | A | high | easier | (i) Harmless, (ii) Honest, (iii) Helpful; more refusals hurts Helpful, so trade-off must be balanced
n11-X4-03 | D | high | ok | P1 pairs + 2 models -> DPO; P2 verifiable reward, no preference data -> GRPO (group mean baseline, no critic)
n11-X4-04 | B | high | ok | DPO on (chosen=admit unknown, rejected=fabricated) with SFT as reference gives negative feedback SFT lacks
n11-X4-05 | C | high | easier | Y = 0.8x300+0.2x5000 = 1,240 tokens = 12.4 s vs X 50 s; saves 37.6 s (75.2%)
n11-X4-F5 | 0.6931 ; 0.3133 ; 2 | high | ok | -ln sigma(0)=ln2=0.6931; beta*diff=0.25x4=1 -> ln(1+e^-1)=ln1.3679=0.3133; need 0.5xdiff=1 -> diff=2
n12-X4-01 | C | high | ok | starts every 448: ceil(9920/448)=23 chunks; last starts 9856 (64 chars) lying fully inside chunk 22 (9408-9920)
n12-X4-02 | A | high | ok | RRF: Q=2/80=0.02500; R=1/63+1/130=0.02357; S=1/110+1/120=0.01742; P=1/61=0.01639 -> Q,R,S,P
n12-X4-03 | D | high | ok | stages 1-3 = max(5,10)+1 = 11 ms; budget 49 ms / 2.5 ms per pair = 19.6 -> 19 pairs
n12-X4-04 | B | high | ok | image->text rows for dup captions: p=1/2 each (2x ln2); text->image for T1,T2 identical rows: min -ln p -ln(1-p) = 2 ln2; sum=4 ln2=2.773; /(2N=8)=0.347
n12-X4-05 | C | high | easier | "Sam Altman" and "Altman" become two nodes; need entity linking / canonical alias node
n12-X4-F1 | 21 ; 20 ; 2 | med | ok | embed all 21 sentences; 20 adjacent distances; 95th percentile of 20 distinct values leaves 1 distance above -> 1 cut -> 2 chunks
n13-X4-01 | B | high | ok | e^3,e^2,e^1,e^0 sum 31.19; invalid=(20.09+1)/31.19=67.6%; after mask 7.389/(7.389+2.718)=0.731
n13-X4-02 | D | high | ok | valid destructive action triggered by injected tool content; needs L2 permission validate + L4 human approval + untrusted-data handling; L1/L3 cannot catch it
n13-X4-03 | A | high | ok | time: 8 steps = 56 s, step 9 would end at 63 s > 60 so cut; tokens allow 10 steps, steps allow 12 -> time first, 8 complete
n13-X4-04 | C | high | ok | S41: Routing = Low/Low/Medium is the only row with Latency<High, Complexity<=Low, FT>Low (Sequential FT Low; Debate Complexity Medium; Supervisor Very High)
n13-X4-05 | B | high | easier | gap 92-15=77; 92.4 > 92 by 0.4 so contamination/scaffold caveat; other options wrong arithmetic or dismiss other metrics
n13-X4-F1 | 81 ; 740 | med | ok | Buffer: (8000-500-1000)/80 = 81.25 -> 81 turns; Retrieval: 500 + 6x40 = 740
n14-X4-01 | C | high | ok | 1 - C(8,3)/C(10,3) = 1 - 56/120 = 0.533
n14-X4-02 | A | high | easier | EN 7100 x 10/71 = 1,000 tokens; TH 5700 x 114/57 = 11,400; ratio 11.4
n14-X4-03 | D | high | ok | 500/0.65 = 769.2 -> 770; 500/0.775 = 645.2 -> 646
n14-X4-04 | B | high | ok | X: P(0.55)=0.111 -> ~1,110; P(0.2)=3.6e-5 -> 0.36 pairs (<1); Y: 0.7846 -> 7,846; P(0.2)=0.0253 -> 253
n14-X4-05 | C | high | ok | P1 has { -> C4 removes (Gopher passes); P2 45<50 words -> Gopher; P3 symbol ratio 12/80=0.15>0.1 -> Gopher
n14-X4-F1 | 65 ; 1.7 | high | ok | 1.3T/20 = 65B; 300B/175B = 1.714 -> 1.7 tokens/param
n08-L4-06 | D | high | ok | no cache re-projects 50 positions every step: 20x50 vs 50 once = 20x
n09-L3-01 | C | high | ok | 2000x1999/2 = 1,999,000 pairs x 4 ms = 7,996 s = 2.22 h
n09-L3-03 | D | high | ok | freezing BERT and training only pooling/classifier cannot fix anisotropy; A,B,C true
n09-L4-01 | D | med | ok | dropout off -> two identical views, trivial positive (alignment signal ~0); negatives still exist
n09-L4-05 | D | high | easier | range [0.46,0.85] all >= 0.40 so every pair flagged duplicate
n09-L2-F1 | 499,500 ; 2,497.5 | high | ok | 1000x999/2 = 499,500 passes x 5 ms = 2,497.5 s
n10-L4-04 | B | high | ok | attainable = min(peak, BWx1.5): P 3, Q 4.5, R 3.75, S 2.25 TFLOP/s -> Q best
n10-L4-05 | A | high | ok | per-tensor s=12.7/127=0.1 -> 0.04->0.4->0 for both; block-wise: block3 s=0.01 -> q=4, block1 still s=0.1 -> 0
n11-L4-04 | B | high | ok | S 5 parallel calls ~10 s; R = 1+2x2 = 5 sequential calls ~50 s
n11-L4-06 | A | high | ok | politeness has no rule-checkable ground truth (no verifiable reward); Thinker costs tokens/TTFT; easy tasks use Helper
n12-L3-01 | C | high | ok | TF_A=4x2.2/(4+1.2x0.625)=1.853; TF_B=8.8/(4+1.2x1.75)=1.443; ratio 1.28
n12-L3-05 | C | high | easier | multi-hop chain A->B->C not retrievable via chunk cosine; GraphRAG / Text-to-Cypher
n12-L4-01 | A | high | ok | tf=45: 45x2.5/46.5=2.419 x (1.5+2.5)=9.68 vs 9.52 (+1.6%), saturation
n12-L4-04 | B | high | easier | rank 70 never enters reranker (K=50); better reranker does not help, must raise K (latency trade-off)
n13-L4-02 | C | high | ok | Table 2: ReAct reasoning error 47% vs CoT 16%, search-empty 23% vs none, despite hallucination 0% vs 56%
n13-L4-F1 | 3000 ; 500 ; 250 | high | ok | Buffer 60x50=3000; Summary 4x50+300=500; Retrieval 5x50=250
n14-L4-02 | A | high | ok | 175Bx20=3.5T; Edu 1.3T = 37% (3.5/1.3=2.7 passes); FineWeb 15T = 4.3x
n14-L4-04 | A | high | ok | ties-as-half: (54+15)/100=69%; drop: 54/70=77.1%; 30% inconsistent signals position bias
n14-L4-06 | C | high | ok | repeated page passes basic Gopher stats, low PPL kept, MinHash is cross-document; fix = Gopher repetition rules

## Issues
- n10-X4-03: ambiguous between A and D. Slide (W10 p.36-37) says TP suits intra-node and PP suits inter-node, and 3D parallelism = TP within node + PP across nodes + DP, which matches option A literally. But 140 GB fits in one 8x80 GB node, PP adds bubbles (latency), and D (TP8 per node, 2 DP replicas) best meets the stated goals "low latency + many users". A student who memorised the slide table may choose A. Needs wording fix (state that the model must be split across nodes, or drop one option).
- n08-X4-05: answer assumes the loop runs 4 decoder steps (3 tokens + </s>) and that cross-attention counts 1 x 120 weights per head per step. The "t positions including first <pad>" rule is given in the stem, so it is determinable, but the 5-step reading is possible (the loop that stops at </s>); distractor 5,952 is not reproducible from any obvious misreading I could find.
- n08-X4-04: option C depends on deriving 5.5B from P/2 (stem-supplied). Option D contains "ไม่มีเงื่อนไขใด" and B/A are clearly wrong by wording, so partly guessable.
- n09-X4-01: B is the only non-round option, so guessable from option wording; correct count assumes each query sees N-1 in-batch negatives (standard MNRL, not explicitly given in the stem).
- n09-L4-01: option D hedges ("แทบไม่ให้สัญญาณ") while B/C are absolutely worded; B and C are wrong by wording alone.
- n12-X4-F1: result depends on percentile convention. With 20 distinct distances and numpy-style linear interpolation, exactly 1 distance exceeds the 95th percentile (2 chunks). Slide only says "95th percentile of distances"; no explicit interpolation method. Accept 2 only if the key uses strict ">" with interpolation.
- n13-X4-F1: Buffer "81 turns" uses floor(6500/80); Retrieval 740 assumes system prompt 500 is included and each message is 40 tokens (stem says so). Fine but no tokens for retrieved-message metadata.
- n13-X4-04: the S41 table is an image; the values (Routing Low/Low/Medium etc.) come from verify/verify_14_19.md, not from text/slides_w13.txt. The unconfirmed-in-text-source risk is only that.
- n10-X4-F1: relies on "FP32 LoRA + grads + Adam = 16 B/param" given in stem and the slide's "base NF4 = 4 GB"; answer 6.9 GB is sensitive to rounding of 0.872 GB.
- n11-X4-F5: second value uses e^-1 = 0.3679 -> ln(1.3679)=0.3133 (true value 0.31326), fine; third blank has a unique answer (2).
- n12-X4-04: answer 0.347 relies on the assumption that encoder can map the two different images with the same caption to arbitrary embeddings (best case p=1/2 on the text->image side). Mathematically the minimum is 4 ln2/8, but the stem's "negative exp terms approach 0" is the only hint of this; hard to verify from slides, which do not discuss duplicate captions.
- n14-X4-04: option A vs B differ only in "false candidate X = 36 vs <1" (true value 0.36); the 1,111 vs 1,110 difference is rounding, so A and B both contain "~1,111" and the discriminator is the 36 vs <1 detail. Acceptable but the numeric trap is subtle.
- n14-X4-05: P2 has 6 sentences but is removed earlier by Gopher word count (45<50); correct reading requires applying Gopher first, as stated.
- Level labels: n12-L3-01 and n09-L2-F1 carry "L3"/"L2" ids with mismatching header levels in places (n12-L3-01 header level 2, n12-L4-04 header level 3, n09-L4-05 header level 3); I judged fit against the header level. Several [level 4] items (n08-X4-02, n08-X4-03, n09-X4-03, n11-X4-01/02/05, n12-X4-05, n13-X4-05, n14-X4-02, n09-L4-05, n12-L4-04) are one-step recall or single arithmetic, rated "easier".
- Guessable from wording alone: n11-X4-01 (C is the only multi-layer defense), n12-X4-05 (only C names entity linking), n13-X4-05 (only B flags contamination), n14-L4-04 (A only nuanced option; B/C/D contain absolute or "ไม่จำเป็น" phrasing), n13-L4-02 (C is the only option citing Table 2 numbers).

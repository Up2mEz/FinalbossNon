n08-L1-01 | B | high | ok | slides: random pivot, rotate to start there, model finds true start
n08-L1-02 | D | high | ok | masked self-attn -> cross-attn -> FFN
n08-L1-03 | A | high | ok | "summarize: [article]" prefix on slide
n08-L1-04 | C | high | ok | BART-large 12+12, d=1024, 16 heads (slide: 400M large); not directly confirmed in extracted slide text, standard fact
n08-L1-05 | B | high | ok | 32-bucket relative position bias, no absolute encoding
n08-L1-06 | A | high | ok | denoising: reconstruct x from x~
n08-L2-01 | C | high | ok | masking keeps src_len=10 (each token -> one [MASK]); deletion gives 8
n08-L2-02 | D | high | ok | slide pattern: final sentinel (index = #spans) then </s>; 3 spans -> extra_id_3 then </s>
n08-L2-03 | A | high | ok | (1/0.25)^2 = 16
n08-L2-04 | B | high | ok | 20/8 = 2.5; 2.5 - 1.5 = 1.0
n08-L2-05 | C | med | ok | BART classification: same input to encoder and decoder, final decoder token hidden state -> linear head
n08-L2-06 | D | high | ok | Enc-Dec: ~half (decoder) generates, Dec-only: all P
n08-L3-01 | A | high | ok | src = 30-9+3 = 24; tgt = 9 + 3 sentinels + final sentinel + </s> = 14
n08-L3-02 | C | high | ok | Rotation teaches finding the true start, not detecting a missing token (that is Token Deletion); C is the wrong pairing
n08-L3-03 | B | high | ok | slide says T5->PaLM->Gemini is evidence of shift, explicitly not a head-to-head result
n08-L3-04 | D | high | ok | bias is added to logits before softmax, not multiplied after
n08-L3-05 | C | high | ok | 8*12*16*64 = 98,304
n08-L3-06 | A | high | ok | extractive QA: substring only, cannot synthesize
n08-L4-01 | D | high | ok | src = 10 - 1 + 3 [MASK] = 12 > tgt 10
n08-L4-02 | B | high | ok | target must be only dropped spans + sentinels (~0.20N), else decoder stays O(N^2)
n08-L4-03 | A | high | ok | BART 0.5+1 = 1.5; T5 0.5+0.04 = 0.54; ratio 2.78
n08-L4-04 | B | high | ok | N layers: N*1100^2/2 = 605,000N vs 1,110,000N -> 1.83x cheaper
n08-L4-05 | C | high | ok | 15% of 400 = 60, 80% of 60 = 48 [MASK]; T5 target 0.2*400 = 80
n08-L4-06 | D | high | easier | without cache K,V recomputed each of 20 steps vs once -> 20x (50 is a distractor)
n08-L1-F1 | 1,836 ; 101 | high | ok | slide: Flan-T5 1,836 tasks, mT5 101 languages
n08-L2-F2 | Poisson ; lambda = 3 ; insert (แทรก) | high | ok | slide: Poisson lambda=3, 0-length spans create token insertions
n08-L3-F3 | 277,500 ; 302,500 | high | ok | 250,000+2,500+25,000 = 277,500; 2*(550^2/2) = 302,500
n08-L4-F4 | 3.47 | med | ok | BART tgt = 52; T5 tgt = 10 span tokens + 3 sentinels + final sentinel + </s> = 15; 52/15 = 3.47
n09-L1-01 | C | high | ok | 3*768 = 2,304
n09-L1-02 | A | high | ok | attract positives, repel negatives
n09-L1-03 | D | high | ok | shared/tied weights
n09-L1-04 | B | high | ok | masked mean pooling is SBERT default
n09-L1-05 | C | high | ok | normalized IP = cosine similarity, higher is better
n09-L1-06 | A | high | ok | 1/4 = 0.25
n09-L2-01 | B | high | ok | bi-encoder n forwards + O(n^2 d) dot products; cross-encoder O(n^2) forwards
n09-L2-02 | D | high | ok | 0.4-0.7+0.5 = 0.2
n09-L2-03 | A | high | ok | slide: MLM objective never optimises sentence geometry
n09-L2-04 | C | high | ok | (1-s_ap)-(1-s_an) = s_an - s_ap
n09-L2-05 | D | high | ok | 2e6*384*4 = 3.072e9 bytes
n09-L2-06 | A | high | ok | N-1 = 127
n09-L3-01 | A | med | ok | slide uses n(n-1): 2000*1999*4ms = 15,992 s = 4.44 h
n09-L3-02 | B | high | ok | n1 hard (loss 0.6), n2 semi-hard (0.3), n3 easy (0)
n09-L3-03 | D | high | ok | freezing BERT and training only pooling/classifier does not fix anisotropy (MLM geometry stays)
n09-L3-04 | A | high | ok | ANN trades recall for latency, must report Recall@k
n09-L3-05 | B | high | ok | cosine d1; dot 12.5 vs 24 -> d2; L2 2.5 vs 1.41 -> d2
n09-L3-06 | C | high | ok | u=[3,6], v=[4,4], |u-v|=[1,2]
n09-L4-01 | D | high | ok | no dropout noise -> identical views, no learning signal
n09-L4-02 | A | high | ok | (i)=max(0,0.2-1+0.5)=0; (ii)=max(0,0-0.8+0.5)=0; (iii)=0.632-1.414+1=0.218
n09-L4-03 | B | high | ok | A = 5000 s; B = 5+2+500 = 507 ms; ratio ~9,860
n09-L4-04 | C | high | ok | W is 3 x 3d: 9,216 - 6,912 = 2,304
n09-L4-05 | D | high | easier | min cosine 0.46 > 0.40 so every pair flagged duplicate
n09-L4-06 | B | high | ok | 20e6*768*4 = 61.44 GB > 32 GB, use IVF+PQ
n09-L1-F1 | 2304 | high | ok | 3*768
n09-L2-F1 | 999,000 (about 10^6) ; 4,995 (about 5,000) | med | ok | slide uses n(n-1); 999,000*5ms = 4,995 s
n09-L3-F1 | 0.5 ; 1.0 ; 0.3 ; semi-hard | high | ok | 0.5-1.0+0.8=0.3; d(a,p)<d(a,n)<d(a,p)+margin
n09-L4-F1 | 0.9600 ; 0.9692 ; 0.5092 | high | ok | 63/65 = 0.9692; 0.9692-0.9600+0.5 = 0.5092
n10-L1-01 | B | high | ok | 13B*2 = 26 GB
n10-L1-02 | C | high | ok | 2^30
n10-L1-03 | A | high | ok | prefill dominates TTFT
n10-L1-04 | D | high | ok | T>1 flattens distribution
n10-L1-05 | B | high | ok | paged optimizers spill to CPU RAM
n10-L1-06 | C | high | ok | target model verifies in one forward pass
n10-L2-01 | D | high | ok | 3.22 GB, 3.00 GiB
n10-L2-02 | A | high | ok | 8/32 = 1/4, 4 Q heads share each KV head
n10-L2-03 | B | high | ok | Flash: HBM N x N; Paged: fragmentation; continuous batching: idle GPU
n10-L2-04 | B | high | ok | 16*(1024+4096) = 81,920
n10-L2-05 | A | high | ok | 47B*2 = 94 GB; compute ~13B active
n10-L2-06 | D | high | ok | shared always active, routed chosen by router
n10-L3-01 | C | high | ok | 23.5+0.54+1.46 = 25.5 > 24
n10-L3-02 | A | med | ok | KV/request 0.537 GB; (24-16-1.46)/0.537 = 12.2 -> 12 (decimal GB; in GiB it would be 15)
n10-L3-03 | D | high | ok | opt 12N=156, total 208 GB, 208/80=2.6 -> 3 H100
n10-L3-04 | B | high | ok | NF4 base is frozen, never updated
n10-L3-05 | B | high | ok | 1.5/0.0252 = 59.5 -> q=60; 60*0.0252=1.512; error 0.012
n10-L3-06 | C | high | ok | quadratic: 65.5/4 = 16.4 GB
n10-L4-01 | D | high | ok | 8 GiB cache vs 8.2 GiB free -> fits, 0.2 GiB (0.21 GB) left
n10-L4-02 | A | high | ok | (65,536+40,960)*32 = 3.41M; /8B = 0.043%
n10-L4-03 | D | high | ok | base 16 GB dominates; NF4 -> ~4 GB
n10-L4-04 | B | high | ok | Q: min(300, 3*1.5=4.5) = 4.5 TFLOP/s is highest (P=3, R=3.75, S=2.25)
n10-L4-05 | A | med | ok | per-tensor s=0.1 -> 0.04 rounds to 0; block 3 s~0.01 -> q=4; block 1 still s=0.1 -> 0
n10-L4-06 | C | high | ok | static 3.6/0.3=12; paged 11.4/0.3=38; 3.2x
n10-L1-F1 | 12 | high | ok | 2N+2N+12N = 16N
n10-L2-F1 | 130.4 | high | ok | 140e9/2^30 = 130.38
n10-L3-F1 | 0.02 ; 45 ; 0.90 | high | ok | s=2.54/127=0.02; 0.904/0.02=45.2->45; 45*0.02=0.90
n10-L4-F1 | 131,072 ; 16.78 ; 67.11 | high | ok | 2*2*32*8*128 = 131,072 B/token; *128,000 = 16.78 GB; MHA x4 = 67.11 GB
n11-L1-01 | C | high | ok | System role
n11-L1-02 | A | high | ok | adds assistant header so model starts answering
n11-L1-03 | D | high | ok | scalar regression head
n11-L1-04 | B | high | ok | Generate, Critique, Refine
n11-L1-05 | A | high | ok | user input between header and footer reminders
n11-L1-06 | D | high | ok | two labeled examples = few-shot
n11-L2-01 | B | high | ok | beta*(1.5+0.5)=1; -ln sigma(1) = 0.3133
n11-L2-02 | A | high | ok | slide: Tone = emotional (empathetic), Style = writing style (formal)
n11-L2-03 | D | high | ok | zero-shot CoT = "Let's think step by step"; few-shot CoT = examples with reasoning
n11-L2-04 | C | high | ok | sigma(2)=0.8808, -ln = 0.1269
n11-L2-05 | A | high | ok | base model only continues text (next-token prediction)
n11-L2-06 | C | high | ok | SFT learns template + family-specific stop token
n11-L3-01 | D | high | ok | ratio_w = -60-(-63) = +3; ratio_l = -64-(-60) = -4; beta*7 = 0.7; -ln sigma(0.7) = 0.4032
n11-L3-02 | B | high | ok | aggregation + refinement loops = Graph-of-Thoughts

## Issues
- n08-L1-04: BART-large "16 heads" and d_model=1024 not verifiable in the extracted slide text (slide table only says 140M base / 400M large); the standard BART-large config matches option C.
- n08-L2-05: BART classification fine-tuning wording (same input to encoder and decoder, final decoder token) relies on the BART paper caption partly garbled in the extracted slide text; only option C fits, but the slide text is hard to confirm.
- n08-L2-02: answer depends on the slide convention that the closing sentinel is numbered by the span count (extra_id_3 for 3 spans). Consistent with the slide example (2 spans -> extra_id_2).
- n08-L3-04 option A: "16 forward / 16 backward" is slide-supported; B (offsets 0-7 exact) also supported; only D wrong. OK.
- n08-L4-06: ratio is 20 only because the question counts per-decoder-step re-projection; option A (50) is a distractor that can be eliminated by reading, mild "easier" feel.
- n08-L4-F4: tgt_len_T5 = 15 relies on counting the final sentinel and </s> (slide s.17 convention), and BART tgt=52 includes <s> and </s>; if a student counts without the final sentinel the answer changes (52/14=3.71). Question states the convention, but fill answer must be exactly 3.47.
- n09-L3-01 and n09-L2-F1: pair count is ambiguous (n^2 vs n(n-1) vs n(n-1)/2). Slide uses n(n-1). MCQ: A (4.44 h) uses n(n-1)/n^2 and C (2.22 h) uses unordered pairs n(n-1)/2, so both options 4.44 and 2.22 are defensible without the slide convention. Fill answer 999,000 / 4,995 s only under the slide's n(n-1); "10^6 / 5,000" approximations and 499,500 / 2,497.5 would be the unordered alternative.
- n09-L3-03: option A says "เสมอ" (always); true for document vectors from a bi-encoder, but a student could argue it is too strong. D is still the only clearly wrong statement.
- n09-L4-05: answerable by pure comparison (0.46 > 0.40), no slide knowledge beyond the quoted range needed; also option wording (A "ต่ำกว่า 0.40", C "ครบ") is easily eliminated. Level 4 is too high (easier).
- n10-L3-02: result depends on decimal GB (24 GB, 6.54 GB free, 0.537 GB/request -> 12). If 24 GB is read as GiB the answer would be 15, which is not offered; 12 is the only one near 12.2. Options 12/13/14 are close, so rounding conventions matter (0.54 GB rounded gives 12.1, still 12).
- n10-L4-04: options C (R = 3.75 TFLOP/s) and D (S = 2.25 TFLOP/s) are arithmetically true statements about those GPUs; only the "best" framing makes B the unique answer. Option A is false (P = 3). Acceptable but could be read as 3 true statements if "ถูกต้อง" is read literally.
- n10-L4-05: option A assumes block 3's absmax is exactly 1.27 (s = 0.01); stem only says "ไม่เกิน 1.27", so q for 0.04 in block 3 is at least 4, not exactly 4 unless that block's max is 1.27. Also all other options are clearly wrong, so A is unique.
- n10-L3-05: depends on rounding s = 3.2/127 = 0.0252 (stem gives it). Using exact s the dequant is 1.5118 and error 0.0118, still option B.
- n10-L4-03: D is the intended answer; option wording "base 16 GB ใหญ่สุด" gives it away (answerable from option D wording alone: it is the only option whose justification matches 16 of 18.2 GB).
- n10-L4-01: relies on unit trap (8 GiB cache vs 8.2 GiB free) and slide listing 8.6 GB; answer D is unambiguous.
- n11-L2-02: depends on slide definitions Style = writing style, Tone = emotional; students could swap. Order "ตามลำดับ" makes A unique.
- n11-L3-01: stem gives e^-0.7 = 0.4966 but not ln, so student must know -ln(1/(1+0.4966)); answer 0.4032 (option B 0.6931 is the zero-diff distractor, A 0.6783 is the wrong sign of ref difference).
- n08-L1-F1 / n08-L2-F2: fill answers rely on slide facts (1,836 tasks, 101 languages, lambda=3). Confirmed in slides_w08.txt.
- n11-L1-02: demo-based (apply_chat_template add_generation_prompt); not re-verified against text/demo_w11.txt.
- File contains 98 questions in total; n11 section ends at n11-L3-02 (no n11 L3-03+ or L4 items, no n11 fill items in this file).

# Week 10 — Hardware-Aware NLP (Resource Math, Fast Inference, PEFT)

แหล่งข้อมูล: `slides_w10.txt` (57 หน้า, เลขหน้า = segment ที่คั่นด้วย form-feed; หน้าที่เป็นรูปจาก paper อ่านจาก text ได้ไม่ครบ), `demo_w10.txt`, `nlp_exam_hints.md`.
สัญลักษณ์: [MEMORIZE] = ต้องจำในหัว (ความเข้าใจ/เหตุผล), [SHEET] = ควรลอกลง A4, (นอกสไลด์) = ความรู้ทั่วไปของผม ไม่ได้อยู่ในสไลด์/demo
เลขหน้าอ้างอิงแบบ "p.N" = หน้าสไลด์ (±1 เพราะ text extraction ของรูป/ footer เพี้ยนบางหน้า)

---

## 1. น้ำหนักข้อสอบและลำดับหัวข้อ

**น้ำหนัก**: ข้อสอบ 70 MCQ + 10 เติมคำ, "แต่ละสัปดาห์ได้ส่วนแบ่งเท่าๆ กัน" (hints, Week 14). Hints ครอบคลุม Week 7-14 = 8 สัปดาห์ → โดยประมาณ Week 10 ≈ **8-10 MCQ + 1-2 ข้อเติมคำ** (70/8 = 8.75; 10/8 = 1.25). [สมมติฐาน: ถ้าหาร 8 สัปดาห์; ถ้า brief ว่า 10 MCQ ก็ใช้ตาม brief]. ข้อเติมคำ/คำนวณมักเป็น "แค่บวกคูณ" + ระวังทศนิยม/เลขยกกำลัง 10 (hints Week 9).

**ลำดับหัวข้อตามโอกาสออก (เนื้อหาในสไลด์มาก + อาจารย์เน้น)**

| อันดับ | หัวข้อ | หน้า | เหตุผล |
|---|---|---|---|
| 1 | **VRAM math**: N×bytes, KV cache formula, 16N training, GB vs GiB, full audit | 4-9, 14-16 | Hint "GB vs GiB เคยออกกลางภาค", GQA KV cache, คำนวณง่ายๆ, demo Section 1 |
| 2 | **LoRA / QLoRA**: low-rank, 2dr, ตัวอย่าง 13.6M/0.17%, NF4, Double Quant, Paged Optimizer, การ forward/backward, memory 18.2 GB | 39-49 | สไลด์เยอะที่สุดหลัง VRAM, demo Section 3 |
| 3 | **Serving stack**: fragmentation, PagedAttention, vLLM, continuous batching, speculative decoding, TP vs PP | 30-38 | ~9 หน้า, demo Section 2 |
| 4 | **FlashAttention + MoE + Hybrid attention** | 19-29 | ~10 หน้า; hint: MoE เร็วแต่ไม่ประหยัด memory, DeepSeek shared + fine-grained; FlashAttention ไม่ออกสูตร |
| 5 | **Prefill vs Decode, Roofline, TTFT vs TPS** | 11-13 | แนวคิด compute-bound vs memory-bound |
| 6 | **Distillation** (soft targets, T, T², DistilBERT, Llama 3.2) | 51-52 + demo 4.2-4.4 | demo มีเลขจริง |
| 7 | **Quantization math** (absmax, scale, block-wise) | 45, 50 | ไม่ใช่ Week 11 แต่อยู่ใน slide นี้ (hint Week 11 พูดถึงการคำนวณแบบเดียวกัน → โอกาสออก) |
| 8 | BF16 vs FP16, GPU landscape, gradient checkpointing, decision tree, MAD-X/O-LoRA | 10, 16-17, 49, 53 | ข้อสั้นๆ ข้อเดียว |

---

## 2. Core concepts

### 2.1 Memory hierarchy และขนาด 1 parameter (p.3-4) [MEMORIZE]

| ระดับ | ลักษณะ |
|---|---|
| **SRAM** (L1/L2 cache) | เร็วสุด เล็กมาก (~50 MB ตามสไลด์) |
| **VRAM** (HBM บน datacenter, GDDR บน T4/RTX) | เร็ว ความจุจำกัด |
| **System RAM** | ช้า ใหญ่ (TB) |

Goal: เก็บข้อมูลไว้ใน VRAM แล้วคำนวณ

| Precision | bits | bytes/param |
|---|---|---|
| FP32 | 32 | 4 (มาตรฐานเก่า) |
| FP16 / BF16 | 16 | **2** (default ยุคใหม่) |
| INT8 | 8 | 1 |
| INT4 | 4 | 0.5 |

- **สูตร weights: M_w = N × 2 bytes** (N = จำนวน parameter, FP16/BF16). [SHEET]
- **1B params ≈ 2 GB ใน FP16** (p.18). [MEMORIZE]
- ใช้ decimal GB: **1 GB = 10⁹ bytes** (p.5).
- **GB vs GiB** [SHEET]: 1 GiB = 2³⁰ = 1,073,741,824 B ≈ **1.074 GB**. `nvidia-smi` รายงาน MiB, `torch.cuda.memory_allocated()` คืน bytes (ทั้งคู่ binary) → โมเดล "16 GB" แสดงเป็น ~**14.9 GiB** (ช่องว่าง ~7%, เป็นเรื่องปกติ). MB vs MiB ก็แบบเดียวกัน (1 MiB = 2²⁰ B).

### 2.2 หน่วยความจำตอน inference (p.6-9)

**Total Inference VRAM = M_w + M_KV + M_act + M_overhead** [SHEET]
1. Weights: static (2N) 2. **KV Cache**: dynamic โตตาม seq length และ batch size 3. Activations: temporary tensors 4. Workspace/Overhead: CUDA kernels, OS (CUDA context & PyTorch buffers ไม่ใช่ activations)

- **Static** = จัดสรรครั้งเดียวตอน start (weights); **Dynamic** = จัดสรร/คืนซ้ำๆ (KV cache, activations).
- ปัญหา: KV cache โตจน **OOM กลาง generation** (Capacity bottleneck) — โมเดล load ได้แต่ crash ตอน token ที่ 500 ได้ (p.7). [MEMORIZE]

**KV Cache formula (exact)** (p.8) [SHEET]
- MHA: M_KV = 2 × Precision × n_layers × d_model × n_ctx × B
- exact (GQA/MQA): **M_KV = 2 × Precision × n_layers × (n_kv × d_head) × n_ctx × B**
- 2 = K และ V; Precision = bytes/element (FP16 = 2); n_ctx = sequence length; B = batch size; n_kv = จำนวน KV heads; d_head = dim ต่อ head.
- เพราะ d_model = n_heads × d_head → MHA คือกรณี n_kv = n_heads.
- Tensor shape ที่เก็บจริง (demo): `[layers, 2(K,V), kv_heads, seq_len, head_dim]` × batch.

**Full VRAM audit — LLaMA-3 8B, FP16, seq 4096, batch 1** (p.9) [SHEET]
- dims: n_layers = 32, n_kv = 8, d_head = 128
- Weights 8×10⁹×2 = **16 GB**; KV = 2×2×32×8×128×4096×1 ≈ **0.54 GB** (0.537); Overhead 1.46 GB (วัดจริง)
- รวม 16+0.54+1.46 ≈ **18 GB** → พอดี **24 GB RTX 4090**; บน **16 GB T4** ใส่ได้ก็ต่อเมื่อ quantize INT8/INT4. สไลด์ย้ำ "measure it, don't assume it".

### 2.3 BF16 vs FP16 (p.10) [SHEET ตาราง]

| Format | Sign | Exponent | Mantissa | Total |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 32 |
| FP16 | 1 | 5 | 10 | 16 |
| BF16 | 1 | 8 | 7 | 16 |

- FP16 max ≈ **65,504** → gradient spike overflow ง่าย. BF16 max ≈ **3.4×10³⁸** (เท่า FP32) → โอกาส overflow ต่ำมาก "ไม่ใช่ศูนย์".
- BF16 = แลก precision (mantissa) เพื่อ range (exponent). [MEMORIZE]
- Rule: **train ด้วย BF16**; inference ใช้ dtype เดิมของโมเดล หรือ FP16/INT8 ถ้าเทรน/validate มาในช่วงนั้น.
- Native BF16 ต้องใช้ **Ampere ขึ้นไป** (A100, RTX 30xx+, H100). **T4 (Turing) ไม่มี native BF16 → ใช้ FP16** (Colab T4 = ใช้ float16 ทั้ง demo).

### 2.4 Prefill vs Decode, Roofline, TTFT/TPS (p.11-13) [MEMORIZE]

| | **Prefill** | **Decode** |
|---|---|---|
| ทำอะไร | ประมวลผล prompt | สร้างทีละ token (autoregressive) |
| Operation | Matrix-Matrix (**GEMM**) | Matrix-Vector (**GEMV**) |
| ลักษณะ | parallel สูง ใช้ tensor cores | sequential ต้อง stream น้ำหนักทุกตัวจาก HBM (ผ่าน SRAM ทีละ tile) ต่อ 1 token |
| คอขวด | **Compute-bound** | **Memory-bandwidth bound** → TPS ต่ำ |

- Arithmetic Intensity (AI) = FLOPs / Byte (ทำคณิตกี่ครั้งต่อ byte ที่ดึง). ตัวที่ reuse weight chunk ได้มาก (prefill: reuse ข้าม token ใน prompt) → AI สูง; decode reuse ไม่ได้ → AI ต่ำ.
- **Roofline**: Attainable GFlops/s = **min(Peak FLOPs, Peak Memory BW × AI)** [SHEET]. ฝั่งซ้าย (AI ต่ำ) = memory-bound (LLM decoding); ฝั่งขวา (AI สูง) = compute-bound (LLM prefill/training). เส้นขอบฟ้า = limit ของ hardware.
- **TTFT** = Network latency + Prefill time + Decode time (token แรก). prompt ยาว → prefill ครอบงำ (+ queueing ตอน load สูง). user รู้สึก "lag" ถ้า **> 500 ms**.
- **TPS** (tokens/sec, decode rate) ถูกกำหนดโดย memory bandwidth ของ decode. คนอ่าน ≈ **4-5 words/sec**; **> 10-15 TPS** รู้สึกเหมือนทันที.
- Trade-offs: **Speculative decoding → ปรับปรุง TPS**; **chunked prefill → ปรับปรุง TPS/inter-token latency ของ request ที่กำลังรัน แต่ TTFT ของ request ใหม่แย่ลง**.

### 2.5 GQA / MQA เชิง memory (p.14) [MEMORIZE + SHEET]

- ประดิษฐ์ GQA/MQA เพื่อ **ย่อ KV cache** → batch ใหญ่ขึ้น/ผู้ใช้พร้อมกันมากขึ้น.
- LLaMA-3 8B, seq 4096, **batch 16**: MHA (32 KV heads) = **32 GiB (34 GB)**; GQA (**8 KV heads**) = **8 GiB (8.6 GB)** → ลด **75%**.
- ตัวคูณลด = n_heads / n_kv (32/8 = 4×). Hint: โจทย์จะ "ให้" จำนวนหัวและจำนวนกลุ่มมา. Llama 3 ใช้ GQA 8 KV heads (hint Week 10). (Week 7: 64 Q heads, 8 groups → 8×.)

### 2.6 Training memory (p.15-16) [SHEET]

- **"16N" accounting** (FP16 weights + Adam, ไม่ shard): **16 × Params bytes** สำหรับ model states
  - Weights **2N** + Gradients **2N** + Optimizer states **12N**
  - Optimizer (master weights, momentum, variance) = 12/16 = **75%** ของ memory.
  - (demo) 12N = FP32 master weights 4 + Adam momentum 4 + Adam variance 4; "FP32 master copy คือส่วนที่คนลืม".
- LLaMA-3 8B: 8B × 16 = **128 GB** > 1 H100 80 GB (ยังไม่นับ activations). 70B: **1,120 GB** → **14+ H100** (p.39).
- **Full fine-tuning ≈ 8× inference footprint** (16N vs 2N) (p.18).
- **Gradient Checkpointing** (p.16): backprop ต้องเก็บ activations ทุก layer (O(L)). แก้: เก็บเฉพาะ checkpoint layers (ทุก k layers; k ≈ √L; L=32 → ทุก ~6), recompute ที่ขาดจาก checkpoint ใกล้สุดตอน backward. Memory: O(L) → **O(√L)** (O(L/k + k) เมื่อ k = √L); Compute: เพิ่มเวลาเทรน **~30%**. = แลก compute กับ memory. [MEMORIZE]

### 2.7 GPU landscape 2026 (p.17) [SHEET ตาราง, ความสำคัญต่ำ]

| ชั้น | ตัวอย่าง | VRAM | ราคา |
|---|---|---|---|
| Consumer | RTX 4090 / 5090 | 24 / 32 GB | $1.5k-2.5k |
| Datacenter mid | T4, L4, A10G, L40S (**T4 = Colab**) | 16-48 GB | $2k-8k |
| Datacenter high | A100, H100 (HBM2e/HBM3) | 80 GB | $15k-30k+ |
| Next-gen | Blackwell B100/B200 (HBM3e, 2nd-gen Transformer Engine, NVLink 5) | 192 GB | $30k-70k+ |

- ข้อสรุป: **Compute scales faster than memory capacity** — ชิปเร็วแต่ที่ไม่พอใส่ parameters.
- **Act I&II summary** (p.18): (1) memory คือคอขวด ไม่ใช่ compute (โดยเฉพาะ generation) (2) 1B ≈ 2 GB FP16 (3) KV cache โตแบบ dynamic → OOM (4) full FT = 8× inference.

### 2.8 FlashAttention (p.19-22) [MEMORIZE แนวคิด; hint: ไม่ออกสูตร]

- ปัญหา: standard attention สร้าง **S = QKᵀ** ขนาด [batch, heads, seq, seq] = **N×N** → O(N²) memory. N = 32,000 → S ≈ 2 GB ต่อ head (FP16); 32 heads = **~64 GB**. ต้องเขียน N×N ลง HBM → อ่านกลับทำ Softmax → เขียนอีก → อ่านคูณ V = memory-bandwidth latency มหาศาล.
- **FlashAttention** (Dao et al., 2022): คำนวณ **exact attention** โดยไม่เขียน N×N ลง HBM.
  - **Tiling** (inner/outer loops): แบ่ง Q, K, V เป็น block ที่พอดี **SRAM**; คำนวณ partial QKᵀ, partial softmax, คูณ tile ของ V ใน SRAM; เขียนกลับ HBM เฉพาะ **output ขนาด N×d**.
  - ปัญหา: softmax ต้องใช้ global max ทั้งแถว → ใช้ **Online Softmax**: เก็บ running max **m** และ running sum **l**; เมื่อ tile ใหม่: m_new = max(m_old, max(tile)); l_new = l_old·e^(m_old−m_new) + Σ e^(x−m_new) (rescale ผลรวมเก่า). [hint: ไม่ต้องจำสูตร]
  - ผล: **exact (ไม่มี quality loss)**, extra memory **O(N)** แทน O(N²), เร็วขึ้นมาก (IO-aware).
- **Backward pass** (p.22): ทิ้ง attention matrix ตอน forward, **recompute บล็อกใน SRAM ตอน backward** → แลก compute นิดหน่อยเพื่อลด VRAM มหาศาล → long-context training ได้.

### 2.9 Hybrid / sparse attention (p.23-24)

- ตัวอย่าง Longformer, Mistral: (a) full n² attention (b) **sliding window** (เพื่อนบ้านใกล้) (c) **dilated sliding window** (ข้าม token เพื่อขยาย receptive field) (d) **global + sliding window** (บาง token global เห็นทั้งหมด).
- Mix ratio เช่น **3:1** (local 3 layers ต่อ full 1 layer) เพื่อเร่ง context 100k+. ต่างจาก FlashAttention: อันนี้ **ไม่คำนวณ exact attention ทุกที่**.

### 2.10 Mixture of Experts (p.25-28) [MEMORIZE]

- เป้าหมาย: **เพิ่ม parameters (ความรู้) โดยไม่เพิ่ม compute**. Dense (LLaMA-3 8B): ทุก token ผ่านทุก parameter ใน FFN [d, d_ff]. MoE: แทน FFN ด้วยหลาย FFN ย่อย "Experts" (เช่น 8) + **Router** ส่งแต่ละ token ไป **top-k** experts (ปกติ **k = 2**).
- **Mixtral 8×7B**: ~**47B total** params (attention ใช้ร่วม), **~13B active** ต่อ token. Config (Table 1): dim 4096, n_layers 32, head_dim 128, hidden_dim 14336, n_heads 32, n_kv_heads 8, context_len 32768, vocab 32000, num_experts 8, top_k 2.
- **Router**: linear layer **W_r ∈ ℝ^(d×E)** (E = จำนวน experts); **P(x) = Softmax(W_r x)**. Top-1 (Switch Transformers): y = P₁ · Expert₁(x). Top-2 (Mixtral): y = ผลรวมถ่วงน้ำหนักของ 2 experts (weights = gating). Output ของ gate: G(x) = Softmax(TopK(x·W_g)).
- **Load balancing**: ถ้า router ส่งทุก token ไป Expert 1, experts อื่น idle → เพิ่ม **auxiliary loss** ตอนเทรนเพื่อลงโทษ routing ไม่สมดุล.
- **True costs**: MoE **ประหยัด compute (inference เร็ว)** แต่ **ต้อง load ทั้ง 47B ใน VRAM → ไม่ประหยัด memory** (hint: "อย่าเข้าใจผิด"). Batch size trade-off (หัวข้อบนสไลด์, text อ่านไม่ครบ).
- **DeepSeekMoE** (p.27): แบ่งเป็น **160+ micro-experts** (แทน 8 ใหญ่): **Shared Experts** = active ทุก token (ความรู้ทั่วไป: grammar, formatting); **Routed Micro-Experts** = เฉพาะทาง, Router เลือก Top-K_r ที่เกี่ยวข้อง. = shared + fine-grained.
- **Sparse Upcycling** (p.28): เทรน MoE จากศูนย์แพง → (1) copy Attention/Norm จาก dense (2) clone MLP เป็น E copies (3) เพิ่ม **Router ใหม่** → ฉลาดตั้งแต่วันแรก; มีแค่ Router ที่เริ่มจากศูนย์ แล้วเทรนต่อทั้งหมด.

### 2.11 Serving stack (p.30-38)

**Fragmentation** (p.30-31) [MEMORIZE]
- Static allocation: pre-allocate KV ตาม max length (เช่น 4096) ต่อ request; generate แค่ 50 → ที่เหลือ 4,046 slots ล็อกทิ้ง.
- **Internal fragmentation** = พื้นที่ pre-allocate ที่ไม่ได้ใช้ (+ reserved); **External fragmentation** = ช่องว่างระหว่าง request ต่างๆ; request ใหม่ไม่พอดี.
- Naive serving ใช้ KV memory ที่จองไว้ได้เพียง **20-40%**. (รูป: Request A 2038 slots never used, B 507 slots.)

**PagedAttention** (Kwon et al., 2023) (p.32) [MEMORIZE]
- ยืม **virtual memory paging ของ OS**: แบ่ง KV cache เป็น **fixed-size blocks (pages)**, allocate **on demand**, **ไม่ต้องติดกัน (non-contiguous)** ใน physical GPU memory, map ด้วย **page table (block table)**.
- ผล: (1) fragmentation เกือบ 0 (KV memory ที่ใช้จริง **~30% → ~95%**) (2) batch size/throughput เพิ่ม **2-4×** (3) **block sharing** (parallel sampling, beam search ไม่ copy).

**vLLM** (p.32-33): open-source serving engine บน PagedAttention. Features: (1) PagedAttention (2) **Continuous batching** (3) **Tensor parallelism** (multi-GPU) (4) Quantization: **GPTQ, AWQ**.
- **Continuous batching** แทน static batching: **iteration-level scheduling (ORCA-style)** ที่ระดับ token step — request เข้า/ออกจาก running batch ได้กลาง generation โดยไม่ต้องรอตัวอื่นจบ → GPU ไม่ idle.

**Speculative Decoding** (Leviathan et al., 2023) (p.33-35) [MEMORIZE + SHEET]
- ปัญหา: decode sequential + memory-bound; อยากประมวลผลหลาย token พร้อมกันเหมือน prefill.
- Step 1 **Draft**: โมเดลเล็กเร็ว (เช่น 1B) สร้าง candidate tokens ทีละตัว; Step 2 **Verify**: โมเดล target ใหญ่ (เช่น 70B) ประมวลผล candidates **ทั้งหมดใน 1 parallel forward pass** (เหมือน prefill เร็วๆ); Step 3 **Accept/Reject**.
- **Guarantee**: output **มี distribution เหมือน sample จาก target model อย่างเดียว** (ไม่เสีย quality). Speedup **~2-3×**.
- Rejection sampling math (p.35): sample x ~ q(x) (draft); **accept ด้วยความน่าจะเป็น min(1, p(x)/q(x))**; ถ้า reject → sample ใหม่จาก adjusted distribution **p'(x) = norm(max(0, p(x) − q(x)))**. (p = target, q = draft). ถ้า q(x) ≤ p(x) → accept แน่นอน.
- Expected tokens ต่อ 1 รอบของ target = **(1 − α^(γ+1)) / (1 − α)**, α = acceptance rate, γ = จำนวน token ที่ draft (สไลด์แสดงสมการแต่ text เพี้ยน; สูตรนี้อ้างจาก paper) — ถ้าไม่แน่ใจ ให้ถือเป็น (นอกสไลด์) ในรายละเอียดสูตร.

**Tensor vs Pipeline Parallelism** (p.36) [SHEET ตาราง]

| | **Tensor Parallelism (TP)** (Megatron-LM) | **Pipeline Parallelism (PP)** |
|---|---|---|
| แบ่งอะไร | แบ่ง **weight matrices** ข้าม GPU (แต่ละ GPU คำนวณส่วนของ **ทุก layer**) | แบ่ง **layers** เป็นกลุ่ม (GPU0: L1-16, GPU1: L17-32) |
| Latency | **ต่ำ** (parallel) | เพิ่ม **pipeline bubbles** |
| ต้องการ | synchronization (all-reduce) ความเร็วสูง | ตัดแค่ activations ระหว่าง stage |
| เหมาะ | **intra-node** (NVLink) | **inter-node** |

- เหตุผลที่ต้องใช้: LLaMA-3 70B FP16 = 140 GB > 1 GPU.
- **Multi-node** (p.37): สูงสุด **8 GPUs ต่อ node** (เช่น 8× H100 DGX). **3D Parallelism** = TP (ใน node) + PP (ข้าม node) + **Data Parallelism** (replicate ชุดเดิมสำหรับ batch ต่างกัน). **InfiniBand** กัน Memory Wall ขยายเป็น "Network Wall".
- **Modern serving stack 2026** (p.38): User → API Gateway → **vLLM Scheduler (continuous batching)** → **PagedAttention KV manager** → **FlashAttention kernel** → [optional] Speculative Decoding → [optional] Tensor Parallelism → Streamed tokens.

### 2.12 PEFT และ LoRA (p.39-44) [MEMORIZE]

- Full FT: LLaMA-3 8B ≈ 128 GB (+ activations); 70B ≈ 1,120 GB → 14+ H100 → ไม่ practical.
- **PEFT**: **freeze base weights** (ไม่มี gradient/optimizer state), เพิ่ม parameters เทรนได้จำนวนน้อย, เทรนเฉพาะที่เพิ่ม → ลด trainable params **99%+**. Insight: ไม่ได้เทรนโมเดลใหม่ แต่ "ชี้นำ" โมเดลที่มีอยู่.
- Taxonomy 4 กลุ่ม (survey figure, p.39-40): **Additive** (adapters, soft prompts), **Selective** (BitFit, diff pruning), **Reparameterized** (LoRA = low-rank decomposition), **Hybrid**.
- **Intrinsic dimensionality** (Aghajanyan 2021, p.41): pre-trained LMs มี intrinsic dimension ต่ำ → weight update ระหว่าง adapt มี "intrinsic rank" ต่ำ → ประมาณได้ด้วย low-rank.

**LoRA** (Hu et al., 2021/2022) [SHEET]
- ΔW = **B·A**; forward: **h = W₀x + ΔWx = W₀x + BAx**
- W₀ ∈ ℝ^(d×k) **frozen**; **B ∈ ℝ^(d×r)**, **A ∈ ℝ^(r×k)**, rank **r ≪ min(d,k)**.
- **Init**: **A ~ Gaussian N(0, σ²)**, **B = 0** → ΔW = 0 ตอนเริ่ม (โมเดลเริ่มเท่า pretrained). **เทรนแค่ A และ B**.
- **Parameters: 2dr แทน d²** (สำหรับ square d×d). ตัวอย่าง d = 4096, r = 16 → **131,072** vs 16.7 M (≈ 0.78%).
- Merge ΔW เข้า W₀ ได้ตอน deploy → **ไม่เพิ่ม inference latency** (ข้อความจาก paper บนสไลด์ p.40).
- ใช้ผ่าน HF `peft`: `LoraConfig` + `get_peft_model()`. Scaling factor = α/r (นอกสไลด์ในรายละเอียด; demo ใช้ r=16, lora_alpha=32).

**Worked example LLaMA-3 8B (r=16, apply ที่ Q, K, V, O)** (p.42) [SHEET]
- Q/O: แต่ละ matrix **131k** params; K/V (GQA, out = 8×128 = 1024): แต่ละ matrix **~81k**.
- รวมทั้งโมเดล ≈ **13.6 M** trainable; ratio 13.6M / 8B = **0.17%**.
- Memory (FP16 base, FP32 LoRA): base frozen **16 GB** + LoRA weights/grads/Adam (FP32, **16 B/param**) **0.22 GB** + Activations & overhead **~2.0 GB** (small batch + checkpointing) = **18.2 GB** → **พอดี 24 GB RTX 4090, ไม่พอ 16 GB T4**.

**LoRA variants** (p.43): **DoRA** = แยก magnitude + direction (คุณภาพดีขึ้น); **LoRA+** = learning rate ต่างกันสำหรับ A และ B; **rsLoRA** = scale ด้วย α/√r แทน α/r เพื่อเสถียรเมื่อ rank สูง (สัญลักษณ์ใน text ของสไลด์หาย; อ่านจากบริบท).
**Multi-adapter serving** (vLLM native): load base model **ครั้งเดียว (16 GB)**, **สลับ LoRA adapter เล็กๆ ต่อ request** (User A → Coding LoRA, User B → Medical LoRA) → scale ได้โดยไม่ duplicate memory. (รูป PetS: task register/scheduler.)

### 2.13 Quantization และ QLoRA (p.44-48, 50)

**Quantization** (p.44): แมปค่าต่อเนื่อง → ระดับไม่ต่อเนื่อง **q = round(x/s) + z** (s = scale, z = zero-point; z = 0 สำหรับ symmetric absmax). **Absmax (FP16→INT8)**: ช่วง [−3.2, 3.2] → [−127, 127]; **s = 3.2/127 = 0.0252**; 2.1 → round(2.1/0.0252) = **83**; dequantize 83 × 0.0252 = **2.092** (มี error). **Block-wise**: ใช้ scale ต่างกันต่อ block เล็ก (เช่น **64 ค่า**) เพื่อลด error (outlier ไม่ทำลายทั้ง tensor). LLM.int8() (Dettmers 2022, p.45): outlier features เกิดตอนโมเดลใหญ่ (>~6.7B) ทำให้ quantize naive พัง.

**QLoRA** (Dettmers et al., 2023) — 3 innovations [MEMORIZE + SHEET]
1. **4-bit NormalFloat (NF4)**: data type 4-bit ที่ระดับค่า (levels) ตามการแจกแจงปกติของ weights ที่เทรนแล้ว.
2. **Double Quantization**: quantize **ตัว scale factors เอง** (FP32 → FP8) ประหยัด memory เพิ่ม.
3. **Paged Optimizers**: optimizer states **overflow ไป CPU RAM** (unified memory) กัน GPU OOM จาก memory spike.
- ผล: เทรน 8B บน consumer GPU; ประหยัด ~ ระดับ order of magnitude เทียบกับ full FP16 FT (ขึ้นกับ batch/seq/checkpointing). (paper บนสไลด์: 65B จาก >780 GB เหลือ <48 GB.)

**QLoRA forward/backward** (p.47) [MEMORIZE]
- **Storage**: base weights ล็อกเป็น **4-bit NF4** ใน VRAM.
- Forward: (1) **dequantize block NF4 → 16-bit on-the-fly ใน SRAM** (FP16 บน T4) (2) matmul แล้ว **ทิ้ง** block 16-bit (มีแค่ working copy เล็กๆ ใน VRAM) (3) บวก **LoRA path** 16-bit: **h = W_dequant·x + BA·x**.
- Backward: **parameter gradients มีเฉพาะ LoRA (A, B)**; activation gradients ยังไหลผ่าน frozen layers; NF4 weights = "frozen reference grid".
- Practical (p.48): 2 ขั้นตอน — (1) `BitsAndBytesConfig` โหลดโมเดล 4-bit (2) `LoraConfig` + `get_peft_model()` → 8B ≈ **0.17%** trainable, พอดี 16 GB T4.

**Quality vs Compression** (p.50) [SHEET ตาราง] (น้ำหนัก LLaMA-3 8B, decimal GB)

| Format | Memory | คุณภาพ |
|---|---|---|
| FP16 | 16 GB | baseline |
| INT8 | ~8 GB | near-baseline |
| **NF4 (4.13-bit)** | **~4 GB** (4× เล็กกว่า) | **เท่า 16-bit บน benchmarks** (ต้อง re-measure กับงานตัวเอง) |
| INT3 | ~3 GB | เสื่อมเร็วขึ้นชัดเจน |
| INT2 | ~2 GB | เสียหนัก ถ้าใช้ naive quantizer (ต้องใช้ เช่น AQLM) |

"Weight memory is arithmetic; quality must be measured." — **NF4 คือ sweet spot** (4× memory ลด, คุณภาพไม่เสียมาก).

### 2.14 Adapters ขั้นสูง (p.49)

- **MAD-X** (Pfeiffer 2020): **Language Adapter** (MLM บน raw text) + **Task Adapter** (เทรนด้วยภาษาอังกฤษ) แล้ว **stack** ตอน inference. ตัวอย่าง **Thai-LA + NER-TA → Thai NER โดยไม่มี Thai NER data**. L ภาษา × T task ต้องใช้แค่ **L + T adapters (ไม่ใช่ L × T)**.
- **O-LoRA** (Wang 2023): LoRA subspace ของแต่ละ task ใหม่ถูกบังคับให้ **orthogonal** กับของ task ก่อนหน้า: **L_orth = Σ_{i<j} ‖A_i A_jᵀ‖²_F**. **ลด catastrophic forgetting แต่ไม่ได้กำจัดหมด**.
- Takeaway: adapters = หน่วยความรู้แบบ modular — stack (MAD-X) หรือ steer เชิงเรขาคณิต (O-LoRA).

### 2.15 Knowledge Distillation (p.51-52) [MEMORIZE]

- **3 แกนของ compression**: Quantization ลด **bits**; LoRA ลด **trainable params**; **Distillation ลด architecture (จำนวน layers)** — เป็น **แกนเดียวที่ลด compute (FLOPs)** ไม่ใช่แค่ bytes. ข้อเสีย: ต้องมี data + retrain (แพง).
- Student เรียนจาก **soft targets** (full probability vector) ของ teacher ไม่ใช่แค่ one-hot. **Dark knowledge**: label "cat" แต่ teacher บอก cat 0.90, tiger 0.08, car 0.00001 → **ลำดับ/ความคล้าย** คือ signal พิเศษ.
- **Temperature**: p_i = exp(z_i/T) / Σ_j exp(z_j/T); **T > 1 ทำให้ softmax แบน** → class ความน่าจะเป็นต่ำมองเห็นได้.
- **Loss = α · T² · KL(p_teacher^T ‖ p_student^T) + (1 − α) · CE(y, p_student)**; **T² ชดเชย gradient ที่ถูกลดด้วย 1/T²** [SHEET].
- **Production**: **DistilBERT (2019)**: BERT **12 → 6 layers**, init จาก **ทุก layer ที่สลับ (every other)** ของ teacher; **~40% เล็กลง, ~60% เร็วขึ้น, ~97% ของ BERT บน GLUE dev**. **TinyBERT** (ตรงกับ hidden states + attention ทีละ layer), **MiniLM** (last-layer attention + value relations) — "copy the reasoning, not just the answer". LLM: **Llama 3.2 1B/3B** = pruned จาก Llama 3.1 8B แล้ว logit KD จาก teacher 8B + 70B; **Gemma 2 (2B, 9B)** เทรนด้วย KD แทน next-token loss; **DeepSeek-R1-Distill** = SFT บน **~800k** samples จาก R1 (**~600k reasoning + ~200k general**); **TinyLlama-1.1B ไม่ใช่ distilled** (pretrain from scratch ~3T tokens).
- **Decision rule**: ต้องพอดี VRAM → **quantize** (ถูก); ต้องตอบเร็วบน edge และมี teacher ดี → **distill**.

### 2.16 Decision Tree (p.53) [SHEET]

1. VRAM budget: พอใน FP16 → FP16 + FlashAttention + vLLM; ตึง → 8-bit/4-bit quantization.
2. ต้อง fine-tune? full FT ไหว → ทำ; GPU เดียว → **LoRA/QLoRA**.
3. Traffic: **concurrency สูง → vLLM (continuous batching)**; **ผู้ใช้เดียว/local → Speculative Decoding**.
4. Compute (ไม่ใช่ memory) คือกำแพง? → **Distill** เป็น student ตื้นขึ้น (quantization ลดแค่ bytes).

Bridge to Week 11 (p.56): SFT/RLHF แพง → QLoRA เป็นตัวเลือกงบน้อยมาตรฐานสำหรับ alignment.

---

## 3. Worked calculations (ตรวจเลขสองรอบแล้ว)

### 3.1 Weights (N × bytes) — decimal GB
| โมเดล | N | FP16 (×2) | INT8 (×1) | INT4 (×0.5) |
|---|---|---|---|---|
| LLaMA-3 8B | 8×10⁹ | **16 GB** | 8 GB | **4 GB** |
| LLaMA-3 70B | 70×10⁹ | **140 GB** | 70 GB | 35 GB |
| GPT-3 175B | 175×10⁹ | **350 GB** | 175 GB | 87.5 GB |

แปลงเป็น GiB (÷1.0737): 16 GB → **14.90 GiB**; 140 GB → 130.4 GiB; 350 GB → 326.0 GiB. กลับทาง: 14.9 GiB × 1.074 = 16.0 GB. (ระวัง: GiB ตัวเลข "เล็กกว่า" GB, อย่าคูณผิดทิศ). (ตัวเลข INT8/GPT-3 INT4 ผมคำนวณเอง จากสูตร N×bytes)

### 3.2 KV cache — ตัวอย่างเต็ม
**ต่อ 1 token (LLaMA-3 8B: L=32, n_kv=8, d_h=128, FP16)** = 2×2×32×8×128 = 4 × 32 = 128; ×8 = 1,024; ×128 = **131,072 B (= 128 KiB) ต่อ token**.

**(ก) seq 4096, batch 1** (สไลด์ p.9): 131,072 × 4096 = 536,870,912 B = **0.537 GB ≈ 0.54 GB**.
**(ข) 128K ctx** (demo): 131,072 × 128,000 = 16,777,216,000 B = **16.78 GB**.
**(ค) MHA vs GQA, seq 4096, batch 16** (p.14):
- MHA (n_kv = 32): 2×2×32×**32**×128×4096×16 → 4×32=128; ×32=4,096; ×128=524,288; ×4096 = 2,147,483,648; ×16 = **34,359,738,368 B = 34.4 GB = 32 GiB** ✓
- GQA (n_kv = 8): ÷4 = **8,589,934,592 B = 8.6 GB = 8 GiB** ✓ ลด **75%**
**(ง) ตัวอย่างฝึก (ผมสร้างเอง)**: L=32, n_kv=8, d_h=128, seq 8192, batch 4: ต่อ token 131,072 × 8192 = 2³⁰ B = 1.074 GB (= 1 GiB พอดี) ต่อ request; ×4 = **4.29 GB (4 GiB)**.
**(จ) กี่ request พอ (ผมสร้างเอง)**: GPU 24 GB, weights 16, overhead 1.46 → เหลือ 24 − 16 − 1.46 = 6.54 GB; ÷ 0.537 = 12.2 → **12 requests** (seq 4096). ถ้าเป็น MHA (4× ใหญ่กว่า = 2.15 GB/request) → 3 requests.
**(ฉ) หา ratio จาก heads**: 64 Q heads / 8 KV groups = **8×**; 32/8 = 4×. (ตัวแปร: "กลุ่มละกี่หัว" = n_heads/n_kv)

### 3.3 Full VRAM audit (p.9) = 16 + 0.54 + 1.46 = **18.00 GB** (demo ใช้ overhead 1.5 → 18.04). เทียบ 24 GB 4090: เหลือ ~6 GB ✓; T4 16 GB: 18 > 16 ✗ → ต้อง INT8/INT4.

### 3.4 Training memory
- 8B × 16 B = **128 GB** (weights 16 + grads 16 + optimizer 96); optimizer = 12/16 = **75%**.
- 70B × 16 = **1,120 GB** → 1,120 / 80 = **14** H100 (ขั้นต่ำ, ยังไม่รวม activations → "14+").
- Full FT / inference = 16N/2N = **8×**.
- Demo check: Qwen2.5-0.5B (0.494B) × 2 = 0.988 → **0.99 GB** (วัดจริงได้ 0.99 GB).

### 3.5 LoRA parameter count (ตรวจสไลด์ 13.6M) [SHEET]
ต่อ layer (r=16, d=4096): Q: A (16×4096) + B (4096×16) = 65,536 + 65,536 = **131,072**; O เท่ากัน **131,072**. K: A (16×4096)=65,536 + B (1024×16)=16,384 = **81,920**; V เท่ากัน **81,920**.
รวมต่อ layer = 2×131,072 + 2×81,920 = 262,144 + 163,840 = **425,984**. × 32 layers = **13,631,488 ≈ 13.6 M**. Ratio = 13.63M / 8,000M = **0.170%** (0.17%).
LoRA memory = 13.63M × 16 B = 218,103,808 B ≈ **0.22 GB**; total = 16 + 0.22 + 2.0 = **18.2 GB**.
ตัวอย่างเดี่ยว: d=k=4096, r=16: 2dr = 2×4096×16 = **131,072**; d² = 16,777,216; อัตราส่วน = 1/128 = 0.78%.
**QLoRA estimate (ผมคำนวณเอง)**: base NF4 ≈ 4 GB + LoRA 0.22 + overhead ~2 → ≈ **6.2 GB** → พอ T4 16 GB (demo วัดจริง base ~5-6 GiB เพราะ embeddings/norms เป็น FP16; 4 GB ≈ 3.73 GiB).

### 3.6 Quantization absmax (p.44) [SHEET]
s = max|w| / 127 = 3.2/127 = 0.025197 ≈ **0.0252**. x = 2.1: 2.1/0.0252 = 83.33 → **round = 83**. Dequant: 83 × 0.0252 = **2.0916 ≈ 2.092**. Error = 2.1 − 2.092 = **~0.008**. (hint Week 11 ยืนยันตัวเลขเดียวกัน; ระวังทศนิยม 0.0252 ไม่ใช่ 0.252 / 0.00252). ตัวอย่างเพิ่ม (ผมสร้าง): x = −1.0 → −1.0/0.0252 = −39.68 → **−40**; dequant = −40×0.0252 = −1.008; error 0.008.

### 3.7 Attention matrix ของ standard attention (p.19)
N = 32,000: N² = 1.024×10⁹ elements × 2 B (FP16) = **2.048 GB ≈ 2 GB ต่อ head**; × 32 heads = **65.5 GB ≈ 64 GB**. (FlashAttention: extra O(N).)

### 3.8 Speculative decoding (นอกสไลด์ในตัวเลข; สูตรจาก paper ที่ปรากฏบนสไลด์)
E[tokens] = (1 − α^(γ+1)) / (1 − α). α = 0.8, γ = 4: α⁵ = 0.32768; (1 − 0.32768)/0.2 = 0.67232/0.2 = **3.36 tokens/รอบ**. α = 0.5, γ = 3: (1 − 0.0625)/0.5 = **1.875**.

### 3.9 Fragmentation (p.30)
pre-allocate 4096 ใช้ 50 → เสีย 4,046 slots = 4046/4096 = **98.8%** ของที่จองไว้ (เสียเปล่า).

### 3.10 Roofline (นอกสไลด์ — ตัวเลขตัวอย่างของผม)
Peak = 100 TFLOPs, BW = 1 TB/s → ridge AI = 100 FLOPs/B. Decode GEMV FP16: 2 FLOPs ต่อ weight 2 B → AI ≈ 1 → attainable = min(100 T, 1 TB/s × 1) = **1 TFLOPs** (1% ของ peak → memory-bound). ขอบบน TPS ≈ BW / weight bytes = 1 TB/s / 16 GB ≈ 62 tok/s (batch 1, 8B FP16).

### 3.11 Distillation (demo)
Softmax T: logits [8, 5, 1, 0.2, −2]. T=1: cat 0.9513, tiger 0.0474, dog 0.0009, truck 0.0004, banana 0.0000 (ตรวจ: e⁰=1, e⁻³=0.0498, e⁻⁷=0.00091, Σ=1.0512 → 1/1.0512 = 0.951 ✓). T=4: 0.5346/0.2525/0.0929/0.0761/0.0439. T=8: 0.3613/0.2483/0.1506/0.1363/0.1035. **T สูง → แบน, ผลรวมยังเป็น 1**.
Depth vs latency: 12 layers → 85.1 M params, 470.0 ms; 6 layers → 42.5 M, 244.3 ms; speed-up = 470.0/244.3 = **1.92×** (ใกล้ 2×; params เป็นครึ่งหนึ่งพอดี).

---

## 4. Demo-only content (อาจออก 1-2 ข้อ) — `demo_w10`

- **VRAMCalculator class** (Section 1): `static_memory_gb(precision_bytes)` = params × bytes / 1e9; `training_memory_gb()` = params × 16 / 1e9; `kv_cache_gb(layers, kv_heads, head_dim, seq_len, batch)` = 2·L·n_kv·d_h·s·b·2 / 1e9. ผล: FP16 16.00, INT4 4.00, Full FT 128.00, KV 4K = **0.537 GB**, KV **128K ctx = 16.78 GB**; total 16.0+0.54+1.5 ≈ **18.04 GB**.
- **16 B/param breakdown (demo)**: FP16 weights 2 + FP16 grads 2 + **FP32 master weights 4** + FP32 Adam momentum 4 + FP32 Adam variance 4 = **16** — master copy แยกจาก FP16 weights ที่ใช้ forward/backward (คนมักลืม). [SHEET]
- **Empirical proof**: Qwen2.5-0.5B = **0.494B** params, float16 → predicted 0.99 GB = measured 0.99 GB (ยืนยันกฎ N×2). Parameter ในโค้ดคือ `dtype=torch.float16` (`torch_dtype` deprecated). วัดด้วย `torch.cuda.memory_allocated` (CUDA), `torch.mps.current_allocated_memory` (Mac). ต้อง `empty_cache()` และ `gc.collect()` ก่อน/หลัง.
- Print VRAM: `total_gib = total_memory/(1024**3)`; GB = GiB × 1.074. memory_allocated ปกติ **น้อยกว่า** ค่าสไลด์ ~7% เพราะ GiB.
- **Section 2 vLLM**: `LLM(model="unsloth/llama-3-8b-bnb-4bit", quantization="bitsandbytes", load_format="bitsandbytes", dtype="half", enforce_eager=True, max_model_len=1024)`; `SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)`; `llm.generate(prompts, sampling_params)`; **TPS = total_tokens / duration**. vLLM **pre-allocates KV cache** (init ~60 s บน T4); **T4 → dtype half (FP16)**. vLLM ต้องการ CUDA; บน Mac ใช้ MLX-LM แทน (sequential ไม่ใช่ continuous batching; วัดได้ 59.0 tok/s, 304 tokens, 5.16 s — ผล output ในตัวอย่างไม่สำคัญ). vLLM teardown ต้อง `destroy_model_parallel()` + `destroy_distributed_environment()` + `del llm` เพราะ CUDA context/NCCL ค้าง; ถ้ายัง >2 GiB → Restart session.
- **Section 3 QLoRA config** [SHEET]: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)` (**float16 เพราะ T4 ไม่มี BF16**). `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM")`; `get_peft_model(model, lora_config)`; `print_trainable_parameters()` → คาด **13.6M / ~0.17%**. โมเดล `unsloth/llama-3-8b-bnb-4bit` เป็น pre-quantized (ไม่ต้องส่ง quantization_config, config ฝังใน `model.config.quantization_config`). VRAM จริง ~**5-6 GiB** (สไลด์คำนวณ 4 GB decimal ≈ 3.7 GiB) เพราะ **embeddings และ LayerNorm คง FP16**.
- **4.1 O-LoRA penalty**: `penalty = ‖A1·A2ᵀ‖²_F` = `(a @ b.T).pow(2).sum()`; r=8, d=64: independent = **3547.79**; หลัง project A2 ออกจาก rowspace(A1) (QR + A2 − (A2 Q)Qᵀ) = **0.00**. สรุป: overlap ของ subspace ทำให้ task 2 overwrite task 1; orthogonality = "do not step on each other"; **reduces, not removes, forgetting** (งานจริงต้อง overlap features บ้าง).
- **4.2 Temperature**: ตารางข้างบน (3.11). ที่ T=1 banana = 0.0000 (student ไม่ได้เรียนอะไร); T=4 เห็นลำดับ cat>tiger>dog>truck>banana ชัด = dark knowledge. (code ลบ max ก่อน exp เพื่อ numerical stability)
- **4.3 KD vs CE** (student เห็นแค่ 40 จุด, teacher เทรน 4,000, 5 seeds, T=4): teacher **81.6%**; **CE only 74.8% ± 2.3**; **KD only 80.8% ± 0.8**; **KD+CE (α=0.5) 78.4% ± 2.5**. บทเรียน: soft targets ช่วยมากเมื่อข้อมูล student น้อย (ใกล้ teacher); โปรดสังเกตใน demo นี้ KD only ชนะ mix. โค้ด loss = (1−α)·CE + α·T²·KL (kl_div batchmean, `log_softmax(out/T)` vs `softmax(teacher/T)`).
- **4.4 Depth vs latency**: ตาราง 12 vs 6 layers (3.11). ข้อสรุป: quantize 12-layer เป็น 4-bit ลด **memory ~4×** และ bytes ที่ stream ต่อ token แต่ **ยังเดิน 12 layers → FLOPs ไม่เปลี่ยน**; **distillation = แกนเดียวที่ลด compute**. (ตัวอย่างเสริม จากโค้ด: BERT-base-shaped, batch 8, seq 128, d=768, 12 heads, FFN 4d.)
- Env versions printed (torch 2.13.0, transformers 5.16.1, peft 0.20.0, bitsandbytes 0.50.2, accelerate 1.14.0) — ไม่น่าออก.

---

## 5. Hints vs สไลด์ (ข้อสังเกต/ความขัดแย้ง)

- ไม่พบความขัดแย้งโดยตรงระหว่าง hints กับสไลด์ใน Week 10: GB vs GiB ✓, GQA 8 KV heads ✓, MoE ไม่ประหยัด memory ✓ (สไลด์: "must load all 47B"), DeepSeek shared+fine-grained ✓, FlashAttention ไม่ออกสูตร ✓ (สไลด์ p.21 มีสูตร แต่ให้เข้าใจแนวคิด).
- ข้อสังเกต: hint "สอบไปแล้วด้วย" ของ GQA = เคยออกกลางภาค → ถ้าออกอีกจะเป็นแนวคำนวณ.
- สไลด์ p.14 ใช้ **ผสมหน่วย** (32 GiB / 34 GB) — ถ้าโจทย์ถามเลข ต้องดูว่าให้หน่วยไหน.
- สไลด์ p.9 เขียน KV "≈ 0.5 GB" บางที่ แต่ที่ถูกคือ **0.54 GB** (0.537).
- demo overhead ใช้ 1.5 GB (→18.04) สไลด์ใช้ 1.46 (→18.0).

---

## 6. Fill-in-the-blank candidates (10-15)

1. M_w = N × ___ bytes สำหรับ FP16 → **2**.
2. 1 GiB = 2³⁰ bytes ≈ ___ GB → **1.074**; โมเดล 16 GB แสดงเป็น ≈ ___ GiB → **14.9**.
3. LLaMA-3 8B ใน FP16 ต้อง ___ GB (**16**), INT4 ___ GB (**4**), full fine-tuning (16N) ___ GB (**128**).
4. Full fine-tuning ด้วย Adam ใช้ **16N** bytes = weights **2N** + gradients **2N** + optimizer states **12N**; optimizer กิน **75%**.
5. KV cache = 2 × Precision × n_layers × (___ × d_head) × n_ctx × B → **n_kv** (จำนวน KV heads).
6. GQA (8 KV heads) เทียบ MHA (32 heads) ลด KV cache ___% → **75%** (4×).
7. FP16 max ≈ **65,504**; BF16 มี exponent **8** bits, mantissa **7** bits; BF16 ต้อง GPU ตระกูล **Ampere** ขึ้นไป (T4 = **Turing** ไม่มี).
8. Decode phase เป็น ___-bound (**memory-bandwidth**); prefill เป็น ___-bound (**compute**); decode ใช้ **GEMV** (matrix-vector), prefill ใช้ **GEMM**.
9. TTFT = network latency + ___ time + decode (1st token) → **prefill**.
10. FlashAttention ใช้ **tiling** และ **online softmax** คำนวณ attention แบบ exact ด้วย extra memory **O(N)** แทน O(N²).
11. Mixtral 8×7B มี ~**47B** params ทั้งหมด แต่ active ~**13B** ต่อ token (top-**2** of 8 experts); MoE ต้อง **load ทั้งหมด** ใน VRAM.
12. DeepSeekMoE = **shared experts** (active ทุก token) + **routed (fine-grained / micro) experts**.
13. PagedAttention แบ่ง KV cache เป็น ___ blocks mapped by **page table**; ใช้ KV memory จาก ~30% → ~**95%**; throughput เพิ่ม **2-4×**. Continuous batching = **iteration-level** scheduling.
14. Speculative decoding: **draft** model เล็กสร้าง candidates, **target** model ตรวจใน **1 forward pass**; output distribution **เหมือน target** ทุกประการ; speedup ~**2-3×**.
15. LoRA: ΔW = **B·A**, A init Gaussian, **B init zero**; params **2dr** แทน d²; LLaMA-3 8B (r=16, Q/K/V/O) ≈ **13.6M** = **0.17%**.
16. QLoRA 3 ข้อ: **NF4**, **Double Quantization** (quantize scale factors), **Paged Optimizers** (overflow ไป CPU RAM).
17. Absmax INT8: s = max|w|/**127**; 3.2/127 = **0.0252**; 2.1 → **83**; dequant = **2.092**.
18. Distillation loss = α·**T²**·KL(...) + (1−α)·CE; T > 1 ทำให้ softmax **แบน**; DistilBERT = **6** layers, ~**40%** เล็กกว่า, ~**60%** เร็วกว่า, ~**97%** ของ BERT.

---

## 7. Common MCQ traps / ตัวหลอก

1. **GB vs GiB สลับทิศ**: "16 GB = 14.9 GiB" (ไม่ใช่ 17.2 GiB). ตัวหลอกคูณ 1.074 แทนที่จะหาร. และ `nvidia-smi` รายงาน MiB (binary).
2. **ลืมคูณ 2 (K และ V)** ใน KV cache; หรือใช้ n_heads แทน n_kv ใน GQA (ได้ 4× ใหญ่เกินจริง); หรือลืม batch.
3. **"MoE ลด memory"** ผิด — ลด **compute** เท่านั้น ต้อง load ทุก expert (47B). "Active params 13B ⇒ VRAM แค่ 13B×2" = ตัวหลอก.
4. **"FlashAttention เป็น approximation"** ผิด — **exact**, ไม่มี quality loss; มันลด **memory/IO** ไม่ได้ลด FLOPs. "FlashAttention ลด compute (FLOPs) ลง" ก็ตัวหลอก. อีกข้อ: มันทำงานเพราะ tile ใน **SRAM** ไม่ใช่เขียน N×N ลง HBM.
5. **PagedAttention vs FlashAttention**: PagedAttention จัดการ **KV cache memory (fragmentation)**; FlashAttention คือ **compute kernel** ลด O(N²) memory ของ attention matrix. อย่าสลับ.
6. **Continuous batching ≠ PagedAttention**: อันแรกคือ scheduling ระดับ iteration; อันหลังคือ memory management.
7. **Decode เป็น compute-bound** ผิด — decode **memory-bandwidth-bound** (GEMV, stream weights ต่อ token); **prefill** = compute-bound (GEMM). และ **TTFT ≈ prefill** ไม่ใช่ decode.
8. **Speculative decoding ทำให้คุณภาพต่ำลง/เป็น approximation** ผิด — distribution **เหมือน target** ทุกประการ. และมันเป็น **draft เล็ก + verify ใหญ่ขนาน**, ไม่ใช่ใช้ 2 โมเดลโหวต. มันปรับปรุง **TPS** ไม่ใช่ TTFT. เหมาะ single-user/local (ไม่ใช่ high concurrency → vLLM).
9. **BF16 มี precision ดีกว่า FP16** ผิด — BF16 **range** ดีกว่า (exponent 8) **precision แย่กว่า** (mantissa 7 vs 10). T4 ใช้ BF16 ได้ผิด.
10. **LoRA init**: B = 0 (A เป็น Gaussian) — ตัวหลอก "A=0" หรือ "ทั้งคู่สุ่ม" หรือ "ทั้งคู่ 0" (ทั้งคู่ 0 → gradient เป็นศูนย์ทั้งคู่เทรนไม่ขึ้น (นอกสไลด์)). LoRA ไม่เพิ่ม inference latency หลัง merge.
11. **QLoRA เทรน base 4-bit weights** ผิด — base **frozen**; gradient เฉพาะ LoRA A, B (activation gradient ไหลผ่านได้). Forward: **dequantize on-the-fly**, ไม่ได้ compute ใน 4-bit. Double Quantization quantize **scale factors** ไม่ใช่ weights ซ้ำอีกรอบ. Paged optimizer ไป **CPU RAM** (ไม่ใช่ disk).
12. **QLoRA 3 innovations** ตัวหลอก: "GPTQ", "LoRA+", "FlashAttention", "Gradient checkpointing" (ไม่ใช่ 3 ตัวหลัก; ถึงแม้ใช้ร่วมกันได้).
13. **Trainable ratio**: 0.17% (ไม่ใช่ 1.7% หรือ 0.017%) — ระวังทศนิยม. 13.6M ≠ 136M.
14. **16N**: "2N + 2N + 12N" — ตัวหลอก "ไม่นับ optimizer"/"optimizer = 4N". Optimizer ใหญ่สุด (75%). และ 16N **ไม่รวม activations**.
15. **Gradient checkpointing** ลด memory **แต่เพิ่ม compute ~30%** (ไม่ใช่ลดทั้งคู่; ไม่ใช่ลด weights).
16. **Quantization vs Distillation**: quantization ลด **bytes/bits**, **FLOPs เท่าเดิม** (ใน demo 4.4); distillation ลด **layers/compute**; LoRA ลด **trainable params** (ไม่ได้ลด memory ของ base, ไม่ได้ลด inference compute). Distillation: ใช้ **soft targets (T>1)** ไม่ใช่ one-hot only. T² เพื่อชดเชย gradient scale 1/T².
17. **TP vs PP**: TP แบ่ง **matrices** (ทุก GPU ทำทุก layer บางส่วน), เหมาะ **intra-node**; PP แบ่ง **layers**, มี **bubbles**, เหมาะ **inter-node**. 3D parallelism = TP+PP+DP.
18. **TinyLlama-1.1B** ไม่ใช่ distilled (pretrain 3T tokens). **Llama 3.2 1B/3B** = pruned จาก 3.1 8B + logit KD.
19. **O-LoRA**: **ลด** forgetting ไม่ได้ **กำจัด**; ใช้ ‖A_i A_jᵀ‖²_F; **MAD-X** ใช้ L+T adapters ไม่ใช่ L×T.
20. **Optimization trade-offs**: chunked prefill ช่วย TPS/inter-token ของ request เดิมแต่ **TTFT ของ request ใหม่แย่ลง**.
21. **Static memory (weights) vs dynamic (KV cache, activations)**; OOM กลาง generation เกิดจาก KV cache โต, ไม่ใช่ weights.
22. NF4 = levels ตาม **normal distribution** (ไม่ใช่ uniform spacing แบบ INT4). 4.13 bits รวม overhead ของ scale.

---

## 8. SHEET candidates (เรียงตามความสำคัญ, 1 บรรทัดต่อข้อ)

1. `M_w = N × bytes` (FP32 4, FP16/BF16 2, INT8 1, INT4 0.5); 1B ≈ 2 GB FP16; 8B→16, 70B→140, 175B→350 GB.
2. `1 GB = 10⁹ B; 1 GiB = 2³⁰ B ≈ 1.074 GB; 16 GB ≈ 14.9 GiB (GiB เล็กกว่า GB)`.
3. `M_KV = 2 × bytes × L × (n_kv × d_head) × n_ctx × B` — LLaMA-3 8B: L=32, n_kv=8, d_h=128 → 131,072 B/token; 4096 tok → 0.537 GB.
4. `GQA ratio = n_heads/n_kv (32/8 = 4× = −75%; 64/8 = 8×)`; MHA b16 s4096 = 34.4 GB (32 GiB) → GQA 8.6 GB (8 GiB).
5. `Inference VRAM = M_w + M_KV + M_act + M_overhead`; 8B s4096: 16 + 0.54 + 1.46 ≈ 18 GB (พอ 24 GB, ไม่พอ T4 16 GB).
6. `Train (Adam) = 16N = 2N weights + 2N grads + 12N optimizer (75%)`; 8B = 128 GB; 70B = 1,120 GB → 14 H100; = 8× inference.
7. `LoRA: h = W₀x + BAx; A~N(0,σ²), B=0; params 2dr (vs d²); d=4096,r=16 → 131,072`; 8B r=16 QKVO → 13.6M = 0.17%; mem 16+0.22+2.0 = 18.2 GB.
8. `LoRA per layer: Q,O = 131,072 each; K,V (GQA) = 81,920 each; ×32 layers = 13,631,488`.
9. `Absmax INT8: s = max|w|/127 (3.2/127=0.0252); q=round(x/s); 2.1→83; deq 83×0.0252=2.092; err≈0.008; block-wise 64`.
10. `QLoRA = NF4 + Double Quant (scales FP32→FP8) + Paged Optimizers (→CPU RAM)`; fwd: dequant NF4→16-bit in SRAM, W·x + BAx; grads เฉพาะ A,B.
11. ตาราง bits: `FP32 1/8/23; FP16 1/5/10 (max 65,504); BF16 1/8/7 (max 3.4×10³⁸)`; T4 ไม่มี BF16, Ampere+ มี.
12. `Prefill = GEMM, compute-bound | Decode = GEMV, memory-BW-bound; Roofline: attainable = min(peak FLOPs, BW × AI)`; TTFT = net + prefill + 1st decode; >500 ms lag; >10-15 TPS ลื่น.
13. `Std attention matrix N²: N=32,000 → ~2 GB/head ×32 = ~64 GB; FlashAttention = tiling + online softmax (m,l), exact, O(N) extra, backward recompute`.
14. `MoE: Mixtral 8×7B = 47B total / 13B active, top-2; router W_r ∈ ℝ^(d×E), softmax; aux loss balance; compute saved, memory NOT saved; DeepSeek = shared + 160+ routed micro-experts; upcycling = copy dense + clone MLP + new router`.
15. `PagedAttention: fixed blocks + page table, non-contiguous; naive 20-40% → ~95% KV usage; 2-4× throughput | continuous batching = iteration-level (ORCA) | vLLM: GPTQ, AWQ, TP`.
16. `Speculative: draft (small) → verify (target, 1 pass) → accept min(1, p/q), reject → norm(max(0,p−q)); same distribution as target; ~2-3×; E[tok] = (1−α^(γ+1))/(1−α)`.
17. `TP = split matrices, intra-node, NVLink | PP = split layers, inter-node, bubbles | 3D = TP+PP+DP | max 8 GPUs/node | InfiniBand`.
18. `KD: p_i = softmax(z_i/T); Loss = α T² KL(p_T^teacher‖p_T^student) + (1−α) CE; T² undoes 1/T²; DistilBERT 12→6 layers, −40% size, +60% speed, 97% GLUE`.
19. `Gradient checkpointing: O(L)→O(√L), k≈√L (L=32 → ทุก ~6), compute +~30%`.
20. `Quantization quality: FP16 16 GB → INT8 8 GB (near) → NF4 4 GB (≈16-bit) → INT3 degrades → INT2 severe; 3 axes: quant=bits, LoRA=trainable params, distill=layers (only one cutting FLOPs)`.
21. `O-LoRA: Σ_{i<j}‖A_i A_jᵀ‖²_F (reduces forgetting); MAD-X: L+T adapters (Thai-LA + NER-TA); LoRA variants: DoRA (mag+dir), LoRA+ (LR A≠B), rsLoRA (α/√r)`.
22. `Decision tree: fits FP16 → FP16+FA+vLLM; tight → 8/4-bit; single GPU FT → LoRA/QLoRA; many users → vLLM; single user → spec decoding; compute wall → distill`.

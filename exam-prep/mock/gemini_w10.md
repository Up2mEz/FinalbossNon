**Q21** [W10] [VRAM Allocation & OOM Dynamics] [difficulty: medium]
วิศวกรทำการ Deploy โมเดล LLaMA-3 8B (น้ำหนักโมเดลแบบ FP16 ใช้พื้นที่ 16 GB) บน GPU NVIDIA RTX 4090 ที่มี VRAM ขนาด 24 GB โดยระบบสามารถโหลดโมเดลเข้า GPU และเริ่มต้นรันได้สำเร็จ ทว่าเมื่อผู้ใช้ส่ง prompt ขนาดยาวและตั้งค่าให้ generate ข้อความต่อเนื่อง 4,000 tokens โมเดลกลับหยุดทำงานกลางคันเนื่องจากเกิด CUDA Out-Of-Memory (OOM) ที่ประมาณ token ที่ 500 ปรากฏการณ์นี้อธิบายได้ดีที่สุดด้วยเหตุผลใด?
A) เกิด memory leak ในส่วนของ Model Weights ซึ่งมีขนาดขยายตัวแบบ dynamic ตามจำนวน token ที่ถูกสร้างขึ้น
B) เกิดจาก CUDA context และ PyTorch overhead มีขนาดเพิ่มขึ้นแบบ quadratic ($O(N^2)$) เมื่อ sequence length ยาวขึ้น
C) Model Weights เป็น dynamic memory แต่ Activation memory เป็น static memory ทำให้เกิด buffer overflow ภายใน VRAM
D) Model Weights เป็น static memory ที่จัดสรรคงที่ครั้งเดียว แต่ KV Cache เป็น dynamic memory ที่มีขนาดเติบโตตาม sequence length จนเกิด capacity bottleneck ทะลุความจุ VRAM ที่เหลืออยู่

**Q22** [W10] [KV Cache Memory & GQA] [difficulty: hard]
ระบบ LLM serving กำลังประมวลผลโมเดล LLaMA-3 8B (จำนวน layer $L = 32$, มิติของ head $d_{head} = 128$, ใช้ precision แบบ FP16 คือ 2 bytes/element) ให้กับผู้ใช้แบบ concurrent ด้วย batch size $B = 16$ และ sequence length รวม (prompt + output) $n_{ctx} = 4,096$ tokens หากโมเดลนี้ใช้ Grouped-Query Attention (GQA) ที่มี $n_{kv} = 8$ KV heads เมื่อเปรียบเทียบกับ Multi-Head Attention (MHA) แบบดั้งเดิมที่มี 32 heads ขนาดของ KV Cache ของ GQA จะมีค่าประมาณกี่ Gigabytes (GB, ระบบเลขฐานสิบ $10^9$ bytes) และประหยัดหน่วยความจำเมื่อเทียบกับ MHA ไปได้ร้อยละเท่าใด?
A) ประมาณ 8.6 GB และประหยัดหน่วยความจำลง 75%
B) ประมาณ 17.2 GB และประหยัดหน่วยความจำลง 50%
C) ประมาณ 34.4 GB และประหยัดหน่วยความจำลง 25%
D) ประมาณ 4.3 GB และประหยัดหน่วยความจำลง 87.5%

**Q23** [W10] [Prefill vs Decode & Roofline Model] [difficulty: medium]
ในการประมวลผล LLM Inference ข้อใดเปรียบเทียบความแตกต่างระหว่าง Prefill phase และ Decode phase ได้ถูกต้องตามหลักการทางสถาปัตยกรรมฮาร์ดแวร์และ Roofline Model?
A) Prefill phase ทำงานแบบ sequential จึงเป็น memory-bandwidth bound ในขณะที่ Decode phase ทำงานแบบ parallel สูง จึงเป็น compute-bound
B) Prefill phase ประมวลผล prompt ด้วย GEMM ซึ่งมี Arithmetic Intensity สูงและเป็น compute-bound ส่วน Decode phase สร้างทีละ token ด้วย GEMV ซึ่งเป็น memory-bandwidth bound
C) Decode phase มี Arithmetic Intensity สูงกว่า Prefill phase เนื่องจากสามารถ reuse ข้อมูล weights ข้ามแต่ละ token ได้อย่างมีประสิทธิภาพ
D) ความเร็วในการทำงานของ Prefill phase เป็นตัวกำหนด Tokens Per Second (TPS) ของระบบ ส่วน Decode phase เป็นตัวกำหนด Time to First Token (TTFT)

**Q24** [W10] [FlashAttention Architecture] [difficulty: medium]
ข้อใดอธิบายหลักการทำงานและประโยชน์ของ FlashAttention ได้ถูกต้องตามความเป็นจริง?
A) เป็นการประมาณค่า (approximation) ของ attention matrix โดยตัดค่า attention ที่ต่ำทิ้ง เพื่อลดจำนวน FLOPs ลงเป็นเส้นตรง $O(N)$
B) ช่วยลดขนาดความจุของ KV Cache ในหน่วยความจำ HBM ลงด้วยการแบ่งเป็นบล็อกและแมปผ่านระบบตารางหน้า (Page table)
C) ย้ายการคำนวณทั้งหมดจาก GPU ไปยัง System RAM ชั่วคราวเพื่อหลีกเลี่ยงข้อจำกัดด้านความจุของ VRAM
D) เป็นการคำนวณแบบ exact attention โดยใช้ Tiling และ Online Softmax บน SRAM ทำให้ไม่ต้องเขียน-อ่าน attention matrix ขนาด $N \times N$ บน HBM ช่วยลด extra memory จาก $O(N^2)$ เหลือ $O(N)$

**Q25** [W10] [Serving Stack & Speculative Decoding] [difficulty: hard]
วิศวกรระบบต้องการเพิ่ม Throughput และลด Latency ของระบบ LLM Serving สำหรับบริการที่มีผู้ใช้จำนวนมาก ข้อใดระบุคุณลักษณะและข้อจำกัดของเทคนิคใน Serving Stack ได้ถูกต้อง?
A) Speculative decoding ทำให้ output distribution เบี่ยงเบนไปจาก target model เล็กน้อย จึงต้องแลกมาด้วยการลดลงของคุณภาพข้อความที่สร้าง
B) Continuous batching จัดการคิวงานที่ระดับ request-level ทำให้ทุก sequence ใน batch ต้องรอจนกว่า sequence ที่ยาวที่สุดจะประมวลผลเสร็จสิ้น
C) PagedAttention แบ่ง KV Cache ออกเป็น blocks ย่อยที่ map ด้วย page table ช่วยลด memory fragmentation ส่งผลให้การใช้ประโยชน์จาก KV memory เพิ่มขึ้นจาก ~30% สู่ ~95%
D) Speculative decoding เหมาะสมที่สุดสำหรับระบบ serving ขนาดใหญ่ที่มี dynamic batching และ high concurrency สูงมาก มากกว่าการใช้งานแบบ local หรือ single-user

**Q26** [W10] [MoE Architecture & VRAM Footprint] [difficulty: medium]
บริษัทสตาร์ทอัพต้องการเปลี่ยนมาใช้โมเดล Mixture of Experts (MoE) เช่น Mixtral 8x7B (มีพารามิเตอร์รวม ~47B แต่มี active parameters ~13B ต่อ token โดยเลือก top-2 experts) วิศวกรคนหนึ่งเสนอว่า "เราสามารถรันโมเดลนี้บน GPU VRAM 32 GB ได้อย่างสบายในระดับ FP16 เนื่องจากในแต่ละ token มี active parameters เพียง 13B เท่านั้น (ใช้ VRAM เพียง $13 \times 2 = 26$ GB)" ข้อใดประเมินข้อเสนอนี้ได้ถูกต้องตามหลักฮาร์ดแวร์?
A) ถูกต้อง เพราะ MoE ออกแบบมาเพื่อลดทั้ง compute และ memory footprint โดย GPU จะโหลดเฉพาะ expert ที่ถูกเรียกใช้เข้ามาใน VRAM ตามความต้องการ
B) ไม่ถูกต้อง เพราะแม้ MoE จะลด compute (FLOPs) เหลือ 13B ต่อ token แต่ในทางปฏิบัติจะต้องโหลดพารามิเตอร์ทั้งหมด (~47B หรือ ~94 GB ใน FP16) ไว้ใน VRAM ตลอดเวลา เนื่องจากไม่สามารถคาดการณ์ expert ล่วงหน้าได้
C) ถูกต้อง หากเปิดใช้งาน FlashAttention ร่วมด้วย ซึ่งจะช่วยบีบอัด expert weights ที่ไม่ได้ถูกเลือกในรอบนั้นให้เหลือขนาด $O(N)$
D) ไม่ถูกต้อง เพราะ active parameters 13B ใน FP16 ต้องใช้ VRAM สำหรับ weights เริ่มต้นที่ $13 \times 16 = 208$ GB เสมอ

**Q27** [W10] [LoRA & QLoRA Mechanics] [difficulty: hard]
ในการทำ Parameter-Efficient Fine-Tuning ด้วย LoRA และ QLoRA ข้อใดกล่าวถึงกลไกทางคณิตศาสตร์และการจัดการหน่วยความจำได้ถูกต้องตรงตามความเป็นจริง?
A) LoRA กำหนดให้ $\Delta W = B \cdot A$ โดยสุ่มค่าเริ่มต้นของ $A$ ด้วยการแจกแจงแบบ Gaussian และตั้งค่าเริ่มต้นของ $B$ เป็นศูนย์ เพื่อให้เริ่มต้นเทรนด้วย $\Delta W = 0$
B) QLoRA ทำการคำนวณ matrix multiplication ใน forward pass บนข้อมูล 4-bit NF4 โดยตรง เพื่อประหยัด computation บน Tensor Cores
C) Double Quantization ใน QLoRA คือการนำค่าน้ำหนัก base weights 4-bit มาทำ quantization ซ้ำอีกครั้งเป็น 2-bit เพื่อประหยัดพื้นที่ลงครึ่งหนึ่ง
D) Paged Optimizers ใน QLoRA แก้ปัญหา Out-Of-Memory โดยการย้าย optimizer states ที่เกิด memory spike ไปพักไว้บนพื้นที่เก็บข้อมูล NVMe SSD Disk

**Q28** [W10] [Training Memory & Gradient Checkpointing] [difficulty: medium]
ในการทำ Full Fine-Tuning โมเดล LLaMA-3 8B ด้วย Mixed Precision (FP16 weights + Adam optimizer) บนคลัสเตอร์ GPU ข้อใดระบุการกระจายตัวของหน่วยความจำตามหลัก "16N" และผลกระทบของการเปิดใช้ Gradient Checkpointing ได้ถูกต้อง?
A) หน่วยความจำ 16N ประกอบด้วย Model Weights 4N, Gradients 4N, และ Optimizer States 8N โดย Gradient Checkpointing ช่วยลดหน่วยความจำของ weights ลงครึ่งหนึ่ง
B) หน่วยความจำส่วนใหญ่ใน 16N หมดไปกับ Activation memory และ Gradient Checkpointing ช่วยกำจัด optimizer states ออกไปจาก VRAM ได้ทั้งหมด
C) สัดส่วนหน่วยความจำ 16N ประกอบด้วย Weights 2N, Gradients 2N, และ Optimizer States 12N (คิดเป็น 75%) และ Gradient Checkpointing ช่วยลด Activation memory จาก $O(L)$ เหลือ $O(\sqrt{L})$ โดยแลกกับการเพิ่มเวลาเทรนขึ้นประมาณ 30%
D) สัดส่วนหน่วยความจำ 16N เป็นหน่วยความจำรวมทั้งหมดที่ครอบคลุม activations สำหรับ sequence ขนาดยาวแล้ว และการเปิดใช้ Gradient Checkpointing จะช่วยลดเวลาในการเทรนลง 30%

**Q29** [W10] [Quantization Absmax Math] [difficulty: hard]
ในการทำ Symmetric Absmax INT8 Quantization สำหรับบล็อกน้ำหนักหนึ่ง ซึ่งมีค่าสัมบูรณ์สูงสุดของสมาชิกในบล็อกเป็น $\max |w| = 3.2$ หากต้องการแปลงค่าน้ำหนัก $x = 2.1$ ให้อยู่ในรูป INT8 จงหาค่า Scale factor ($s$), ค่า Quantized integer ($q$), และค่าที่ได้จากการแปลงกลับ (Dequantized value, $\hat{x}$) ตามลำดับ (กำหนดให้ใช้ทศนิยม 4 ตำแหน่งสำหรับ scale factor และใช้การปัดเศษจำนวนเต็มแบบปกติ)
A) $s \approx 0.2520, \quad q = 8, \quad \hat{x} \approx 2.016$
B) $s \approx 0.0252, \quad q = 83, \quad \hat{x} \approx 2.092$
C) $s \approx 0.0252, \quad q = 84, \quad \hat{x} \approx 2.117$
D) $s \approx 0.0126, \quad q = 167, \quad \hat{x} \approx 2.104$

**Q30** [W10] [Knowledge Distillation vs Quantization] [difficulty: medium]
ในการพิจารณาแนวทางการบีบอัดโมเดลเพื่อเพิ่มประสิทธิภาพในการประมวลผล ข้อใดเปรียบเทียบความแตกต่างระหว่าง Knowledge Distillation (KD) และ Post-Training Quantization (PTQ) ได้ถูกต้องตามหลักการ?
A) Quantization ช่วยลดทั้งปริมาณการอ่าน-เขียน memory และลดจำนวน FLOPs ในการประมวลผลลงตามจำนวนบิต ในขณะที่ Distillation ช่วยลดเฉพาะ memory เท่านั้น
B) การใช้ Temperature $T > 1$ ใน Distillation มีวัตถุประสงค์เพื่อทำให้ Softmax มีความชันขึ้นและผลักดันความน่าจะเป็นของคลาสที่ถูกต้องให้เข้าใกล้ 1.0 เพื่อให้ Student เลียนแบบได้แม่นยำ
C) Distillation เป็นแกนหลักที่ช่วยลด Computation (FLOPs) และ Latency โดยตรงจากการลดจำนวน layers (เช่น 12 เหลือ 6 layers) ขณะที่ Quantization ลด memory bandwidth แต่ยังคงจำนวน layers และ FLOPs เท่าเดิม
D) ในฟังก์ชัน Distillation Loss พจน์ $T^2$ ที่คูณอยู่กับ KL Divergence มีจุดประสงค์เพื่อลดน้ำหนักของ soft targets เมื่อใช้ Temperature ค่าสูง

**F3** [W10] [QLoRA & PEFT Mechanics]
เทคนิค QLoRA อาศัย 3 เสาหลักในการลดการใช้หน่วยความจำ VRAM ได้แก่:
(1) การเก็บ base model weights ในรูปแบบชนิดข้อมูล 4-bit ชนิดพิเศษที่ชื่อว่า ___
(2) การทำ ___ เพื่อบีบอัด scale factors ของ quantization เพิ่มเติม ช่วยประหยัดหน่วยความจำได้อีกประมาณ 0.37 bits ต่อพารามิเตอร์
(3) การใช้ ___ เพื่อป้องกันปัญหา Out-Of-Memory จาก memory spike โดยการ offload optimizer states ชั่วคราวไปยัง CPU RAM

### Answer key
- **Q21**: **D** — Model Weights เป็น static memory ที่คงที่ แต่ KV cache โตแบบ dynamic ตาม sequence length และ batch จนทำให้เกิด OOM กลาง generation (Notes Section 2.2, 7.21).
- **Q22**: **A** — Arithmetic: $M_{KV} = 2 \times 2 \times 32 \times 8 \times 128 \times 4096 \times 16 = 8,589,934,592\text{ B} \approx 8.6\text{ GB}$ ซึ่งลดลงจาก MHA ($n_{kv}=32$) ที่ใช้ $34.4\text{ GB}$ คิดเป็นประหยัดลง $(32-8)/32 = 75\%$ (Notes Section 3.2 (ค)).
- **Q23**: **B** — Prefill ประมวลผล prompt ขนานด้วย GEMM (compute-bound กำหนด TTFT) ส่วน Decode ทำทีละ token ด้วย GEMV ต้อง stream weights ทุกรอบ (memory-bandwidth bound กำหนด TPS) (Notes Section 2.4, 7.7).
- **Q24**: **D** — FlashAttention เป็น exact attention (ไม่ลด FLOPs) ที่ใช้ Tiling และ Online Softmax บน SRAM หลีกเลี่ยงการเขียน matrix $N \times N$ ลง HBM ลด extra memory เหลือ $O(N)$ (Notes Section 2.8, 7.4, 7.5).
- **Q25**: **C** — PagedAttention จัดการ KV cache แบบ paging ลด fragmentation ทำให้ KV memory utilization พุ่งจาก ~30% เป็น ~95% (Notes Section 6.13, 7.5, 7.8).
- **Q26**: **B** — MoE ลดเฉพาะ FLOPs ต่อ token (13B) แต่ยังต้องโหลดพารามิเตอร์ทั้งหมด (~47B หรือ ~94 GB) ลงใน VRAM เพราะไม่รู้ล่วงหน้าว่า token จะ route ไป expert ใด (Notes Section 6.11, 7.3).
- **Q27**: **A** — LoRA กำหนด $\Delta W = B \cdot A$ โดยเริ่ม $B=0$ และ $A$ เป็น Gaussian เพื่อให้ $\Delta W=0$ ตอนเริ่มเทรน (Notes Section 6.15, 7.10, 7.11).
- **Q28**: **C** — สัดส่วน 16N คือ Weights 2N + Grads 2N + Adam states 12N (75%) และ Gradient Checkpointing ลด activation เหลือ $O(\sqrt{L})$ แลกกับการเพิ่มเวลาเทรน ~30% (Notes Section 2.6, 7.14, 7.15).
- **Q29**: **B** — Arithmetic: $s = 3.2 / 127 \approx 0.0252$, $q = \text{round}(2.1 / 0.0252) = \text{round}(83.33) = 83$, $\hat{x} = 83 \times 0.0252 = 2.0916 \approx 2.092$ (Notes Section 3.6, 6.17).
- **Q30**: **C** — Quantization ลดเพียง memory bandwidth แต่ยังคำนวณครบ 12 layers เท่าเดิม มีเพียง Distillation ที่ตัด layers จึงลด compute (FLOPs) ได้จริง (Notes Section 4.4, 7.16).
- **F3**: **(1) NF4 (NormalFloat4), (2) Double Quantization, (3) Paged Optimizers** — เสาหลัก 3 ประการของ QLoRA ในการลดหน่วยความจำ base model, scale factor, และ optimizer memory spike (Notes Section 6.16, 7.11).

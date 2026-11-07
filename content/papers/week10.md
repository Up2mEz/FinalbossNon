# คลัง paper สัปดาห์ 10: Hardware-Aware NLP

รวม paper ที่สไลด์ DL-10 อ้างอิงหรือแทรกหน้าไว้ เรียงตามลำดับที่ปรากฏในสไลด์ ชื่อเรื่อง ผู้เขียน และปีตรวจจากหน้า arXiv, USENIX หรือ ACL Anthology ที่เปิดอ่านเอง (ไม่ได้ดาวน์โหลด PDF) ส่วน venue ที่มาจากรายการอ้างอิงบนสไลด์แต่หน้า arXiv ไม่ระบุ จะบอกไว้ในแต่ละรายการ ตัวเลขใน "ใจความ" มาจากบทคัดย่อที่เปิดอ่าน หรือจากสไลด์ (ระบุไว้)

---

### 1. Roofline: An Insightful Visual Performance Model for Multicore Architectures

- ผู้เขียน: Samuel Williams, Andrew Waterman, David Patterson
- ที่ตีพิมพ์: Communications of the ACM 52(4), หน้า 65-76, เมษายน 2009
- ลิงก์: https://doi.org/10.1145/1498765.1498785 (DOI ยืนยันจากผลค้นหา หน้าสำนักพิมพ์เปิดอ่านตรงไม่ได้เพราะเซิร์ฟเวอร์ปฏิเสธ) ฉบับ preprint: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html (จากผลค้นหา ไม่ได้เปิดอ่าน)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 12 (กราฟ roofline และหน้าบทความที่แทรก) สไลด์ไม่มีรายการอ้างอิง การระบุว่าเป็นบทความนี้มาจากรูปกราฟและเนื้อหาที่ตรงกัน (ข้อความที่ดึงจากหน้าแทรกอ่านไม่ออกเพราะฟอนต์)
- ใจความ:
  - เสนอโมเดลที่เชื่อมประสิทธิภาพ floating-point, operational intensity และประสิทธิภาพหน่วยความจำไว้ในกราฟสองมิติเดียว
  - ขอบเขตบนของประสิทธิภาพถูกกำหนดโดยสองเส้น: เส้นนอนคือ peak floating-point ของเครื่อง และเส้นเฉียงคือ peak memory bandwidth คูณ operational intensity
  - สองเส้นนี้ตัดกันที่จุดที่ทั้ง compute และ bandwidth ถึงขีดสุดพร้อมกัน เส้นเหล่านี้เป็นของเครื่อง จึงสร้างครั้งเดียวต่อเครื่อง ไม่ใช่ต่อโปรแกรม
  - ผู้เขียนสาธิตกับคอมพิวเตอร์มัลติคอร์ 4 แบบและ kernel floating-point 4 ตัว
  - สูตรที่สไลด์ใช้: ประสิทธิภาพที่ทำได้ = min(peak FLOPs, bandwidth × AI)
- เชื่อมกับข้อสอบ: หัวข้อ 5 (Prefill เทียบ Decode, Arithmetic Intensity) ตัวอย่างคำนวณข้อ 12 และข้อ n10-L4-04 ในคลัง (decode ที่ AI ต่ำ ต้องเลือก GPU จาก bandwidth × AI ไม่ใช่ peak)

---

### 2. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- ผู้เขียน: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- ที่ตีพิมพ์: NeurIPS 2022 (ตามรายการอ้างอิงบนสไลด์ 20: Advances in Neural Information Processing Systems 35) หน้า arXiv ไม่ระบุ venue
- ลิงก์: https://arxiv.org/abs/2205.14135 (ส่ง 27 พ.ค. 2022)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 20 (อ้างอิง) และสไลด์ 57 (แหล่งอ่านเพิ่ม ส่วนที่ 3 อัลกอริทึม tiling)
- ใจความ:
  - attention แบบ exact ที่ "IO-aware" คือนับจำนวนการอ่านเขียนระหว่าง GPU HBM กับ SRAM บนชิป
  - ใช้ tiling เพื่อลดการเข้าถึง HBM เมื่อเทียบกับ attention มาตรฐาน (ซึ่งเวลาและหน่วยความจำโตกำลังสองกับความยาว)
  - BERT-large (ความยาว 512) เร็วขึ้น 15% ทั้งขั้นตอนเมื่อเทียบกับสถิติ MLPerf 1.1; GPT-2 (ความยาว 1K) เร็วขึ้น 3 เท่า; long-range arena (1K-4K) เร็วขึ้น 2.4 เท่า
  - คุณภาพดีขึ้น: perplexity ของ GPT-2 ดีขึ้น 0.7 และ long-document classification ดีขึ้น 6.4 จุด
  - ทำให้ Transformer ชุดแรกชนะการเดาสุ่มบน Path-X (ความยาว 16K, ถูก 61.4%) และ Path-256 (ความยาว 64K, ถูก 63.1%)
- เชื่อมกับข้อสอบ: หัวข้อ 9 อาจารย์ให้เข้าใจแนวคิด (ผลเท่าเดิม ใช้ memory น้อยลง เร็วขึ้นมาก) ไม่ต้องจำสูตร online softmax ดูตัวอย่างข้อ 11 และข้อ n10-X4-04 ในคลัง (FlashAttention ไม่ลด KV cache และไม่ใช่การประมาณ)

---

### 3. Longformer: The Long-Document Transformer

- ผู้เขียน: Iz Beltagy, Matthew E. Peters, Arman Cohan
- ที่ตีพิมพ์: arXiv preprint 2020 (หน้า arXiv ไม่ระบุ venue; สไลด์ไม่มีรายการอ้างอิง)
- ลิงก์: https://arxiv.org/abs/2004.05150 (ส่ง 10 เม.ย. 2020)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 23 (รูปเปรียบเทียบรูปแบบ attention (a)-(d) และข้อความที่แทรกจากบทความ)
- ใจความ:
  - attention ที่โตเชิงเส้นกับความยาว รับเอกสารยาวหลายพันโทเคนขึ้นไป
  - เป็นตัวแทนของ self-attention มาตรฐานที่ใส่แทนได้ ผสม windowed attention แบบท้องถิ่นกับ global attention ที่เลือกตามงาน
  - สไลด์ 23 แสดงรูปแบบ full n², sliding window, dilated sliding window และ global + sliding window
  - ผลลัพธ์ SOTA บน text8, enwik8 (ภาษาระดับตัวอักษร), WikiHop และ TriviaQA และชนะ RoBERTa บนงานเอกสารยาว
  - มีรุ่น Longformer-Encoder-Decoder (LED) ที่ใช้ได้ดีกับสรุปบทความ arXiv
  - จากหน้าที่แทรกในสไลด์: window ขนาด w ทำให้ complexity เป็น O(n × w) และการซ้อนหลายชั้นทำให้ชั้นบนมองเห็นอินพุตทั้งหมดคล้าย CNN
- เชื่อมกับข้อสอบ: หัวข้อ 10 (Hybrid / Sliding-window attention และ mix ratio 3:1) ใช้แยกให้ออกว่า sliding window เป็นการประมาณ ต่างจาก FlashAttention ที่เป็น exact

---

### 4. Mixtral of Experts

- ผู้เขียน: Albert Q. Jiang และคณะ รวม 26 คน (Alexandre Sablayrolles, Arthur Mensch และอื่น ๆ)
- ที่ตีพิมพ์: arXiv preprint 2024 (หน้า arXiv ไม่ระบุ venue) สไลด์ 24 ลิงก์ไปยังหน้าข่าวของ Mistral
- ลิงก์: https://arxiv.org/abs/2401.04088 (ส่ง 8 ม.ค. 2024) และ https://mistral.ai/news/mixtral-of-experts/ (ตามที่สไลด์ให้ ไม่ได้เปิดอ่าน)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 24-26 (รูป MoE layer และตารางคอนฟิก)
- ใจความ:
  - Sparse Mixture of Experts ชื่อ Mixtral 8x7B ใช้สถาปัตยกรรมฐานเดียวกับ Mistral 7B
  - แต่ละชั้นมี 8 feed-forward blocks (experts) และ router เลือก 2 experts ต่อโทเคนในแต่ละชั้น ตัวที่ถูกเลือกต่างกันได้ในแต่ละก้าวเวลา
  - แต่ละโทเคนเข้าถึง 47B พารามิเตอร์ แต่ใช้ active เพียง 13B ตอน inference
  - เทรนด้วย context 32k
  - ตรงหรือชนะ Llama 2 70B และ GPT-3.5 ในทุก benchmark ที่ประเมิน โดยเฉพาะคณิตศาสตร์ โค้ด และหลายภาษา
  - ปล่อยทั้งรุ่นฐานและรุ่น Instruct ภายใต้ Apache 2.0
- เชื่อมกับข้อสอบ: หัวข้อ 11 ที่ตัวเลข 47B/13B และ top-2 จาก 8 experts ออกสอบได้ ข้อคำใบ้ "เร็วแต่ไม่ประหยัด memory" ข้อ n10-L2-05, n10-L3-01 ในคลัง และตัวอย่างข้อ 16

---

### 5. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

- ผู้เขียน: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
- ที่ตีพิมพ์: arXiv preprint 2017 (หน้า arXiv ไม่ระบุ venue)
- ลิงก์: https://arxiv.org/abs/1701.06538 (ส่ง 23 ม.ค. 2017)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 25 เฉพาะในข้อความของหน้า Switch Transformers ที่แทรก (Shazeer et al. 2017) ไม่ได้อยู่ในรายการอ้างอิงของสไลด์
- ใจความ:
  - เสนอชั้น Sparsely-Gated Mixture-of-Experts ที่มี feed-forward sub-networks ได้ถึงหลักพัน และ gating network ที่ฝึกได้เลือกชุดย่อยแบบเบาบางต่อตัวอย่าง
  - เพิ่ม capacity ได้มากกว่า 1000 เท่า โดยเสียประสิทธิภาพการคำนวณบนคลัสเตอร์ GPU เพียงเล็กน้อย
  - ใช้ MoE ที่มีถึง 137 พันล้านพารามิเตอร์คั่นระหว่างชั้น LSTM
  - ชนะผลก่อนหน้าบน language modeling และ machine translation ด้วยต้นทุนคำนวณที่ต่ำกว่า
- เชื่อมกับข้อสอบ: เป็นต้นแบบของ router แบบ top-k ในหัวข้อ 11 (ไม่ใช่เนื้อหาที่ต้องจำรายละเอียด)

---

### 6. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- ผู้เขียน: William Fedus, Barret Zoph, Noam Shazeer
- ที่ตีพิมพ์: Journal of Machine Learning Research 23(120), 2022 (ตามรายการอ้างอิงบนสไลด์ 25 และช่อง Comments ของ arXiv)
- ลิงก์: https://arxiv.org/abs/2101.03961 (ส่ง 11 ม.ค. 2021)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 25 (อ้างอิงและหน้าที่แทรกหัวข้อ Switch Routing)
- ใจความ:
  - ทำ MoE ให้เรียบง่ายขึ้นโดยส่งแต่ละโทเคนไปยัง expert เพียงตัวเดียว (top-1) ตรงกับสูตร `y = P1 × Expert1(x)` ในสไลด์
  - โมเดลที่สร้างจาก T5-Base และ T5-Large เทรนเร็วขึ้นได้ถึง 7 เท่าเมื่อใช้ทรัพยากรการคำนวณเท่ากัน
  - ได้ผลดีขึ้นบนทั้ง 101 ภาษาเมื่อเทียบกับ mT5-Base
  - เทรนโมเดลถึงหนึ่งล้านล้านพารามิเตอร์บน C4 และเร็วกว่า T5-XXL 4 เท่า
  - เทรนโมเดลแบบเบาบางขนาดใหญ่ด้วย bfloat16 ได้ (ผู้เขียนบอกว่าเป็นครั้งแรก)
- เชื่อมกับข้อสอบ: หัวข้อ 11 (Router, Top-1 เทียบ Top-2, load balancing ด้วย auxiliary loss)

---

### 7. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- ผู้เขียน: DeepSeek-AI และคณะ (ทีมใหญ่ประมาณ 157 คนตามไฟล์ PDF)
- ที่ตีพิมพ์: arXiv preprint 2024 (หน้า arXiv ไม่ระบุ venue)
- ลิงก์: https://arxiv.org/abs/2405.04434 (ส่ง 7 พ.ค. 2024 หน้าที่เปิดเป็นเวอร์ชัน 5)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 27 (หัวข้อ DeepSeekMoE)
- ใจความ:
  - โมเดลรวม 236B พารามิเตอร์ active 21B ต่อโทเคน และ context 128K
  - สถาปัตยกรรมมี Multi-head Latent Attention (MLA) และ DeepSeekMoE
  - pretrain บน 8.1T โทเคน
  - ประหยัดต้นทุนเทรน 42.5% เมื่อเทียบกับ DeepSeek 67B
  - ลด KV cache 93.3% และ throughput การสร้างสูงสุดเพิ่ม 5.76 เท่า
  - ที่อาจารย์เน้นในสไลด์ 27: shared experts (active ทุกโทเคน) และ routed micro-experts (fine-grained)
- เชื่อมกับข้อสอบ: หัวข้อ 11 ข้อคำใบ้ "DeepSeek MoE = shared + fine-grained" และข้อ n10-L2-06 ในคลัง

---

### 8. Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints

- ผู้เขียน: Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, Neil Houlsby
- ที่ตีพิมพ์: ICLR 2023 (ตามรายการอ้างอิงบนสไลด์ 28: The Eleventh International Conference on Learning Representations) หน้า arXiv ไม่ระบุ venue
- ลิงก์: https://arxiv.org/abs/2212.05055 (ส่ง 9 ธ.ค. 2022)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 28 (MoE Upcycling)
- ใจความ:
  - การเทรน dense model ให้ลู่เข้าแพงมาก และ MoE เทรนจากศูนย์ก็แพงและกินข้อมูล
  - เสนอ "sparse upcycling": เริ่ม MoE จาก dense checkpoint ที่มีอยู่ เพื่อนำการลงทุนเดิมกลับมาใช้
  - T5 Base/Large/XL ที่ upcycle ชนะรุ่น dense บน SuperGLUE และ ViT Base/Large ชนะรุ่น dense บน ImageNet โดยใช้ราวครึ่งหนึ่งของ compute การ pretrain เดิม
  - ชนะ MoE ที่เทรนจากศูนย์แม้ใช้งบ compute เต็มของ dense
  - ขั้นตอนที่สไลด์สรุป: คัดลอกน้ำหนัก attention/norm, โคลน MLP เป็น E ชุด, ใส่ router ใหม่ แล้วเทรนต่อ
- เชื่อมกับข้อสอบ: หัวข้อ 11 (MoE upcycling เป็นเรื่องคนละอย่างกับ shared experts ของ DeepSeek ซึ่งมีตัวหลอกในข้อสอบ)

---

### 9. Efficient Memory Management for Large Language Model Serving with PagedAttention

- ผู้เขียน: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica
- ที่ตีพิมพ์: SOSP 2023 (Proceedings of the 29th Symposium on Operating Systems Principles; arXiv ระบุ "SOSP 2023")
- ลิงก์: https://arxiv.org/abs/2309.06180 (ส่ง 12 ก.ย. 2023)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 31-32 (อ้างอิงและหน้าที่แทรก) และสไลด์ 57 (แหล่งอ่านเพิ่ม)
- ใจความ:
  - เสนอ PagedAttention ที่ได้แรงบันดาลใจจาก virtual memory และ paging ของระบบปฏิบัติการ และสร้างระบบ vLLM บนฐานนั้น
  - เป้าหมายคือ "ของเสียใน KV cache เกือบเป็นศูนย์" และแชร์ KV cache ได้ทั้งภายในและข้ามคำขอ
  - throughput ดีขึ้น 2-4 เท่าเทียบระบบชั้นนำ (FasterTransformer และ Orca) ที่ latency ระดับเดียวกัน
  - ประโยชน์เพิ่มขึ้นเมื่อ sequence ยาว โมเดลใหญ่ และอัลกอริทึมถอดรหัสซับซ้อน
  - ตัวเลขที่บทเรียนยกจากหน้าที่แทรกในสไลด์: KV cache ต่อโทเคนของ OPT-13B ≈ 800 KB; ระบบเดิมใช้ KV memory ที่จองไว้ราว 20-40%; แชร์บล็อกด้วย reference counting
- เชื่อมกับข้อสอบ: หัวข้อ 12 และตัวอย่างข้อ 14 ข้อ n10-L2-03, n10-L4-06 ในคลัง (ต้องแยกให้ออกจาก FlashAttention และ continuous batching)

---

### 10. Fast Inference from Transformers via Speculative Decoding

- ผู้เขียน: Yaniv Leviathan, Matan Kalman, Yossi Matias
- ที่ตีพิมพ์: ICML 2023 (Oral; ตามช่อง Comments ของ arXiv และรายการอ้างอิงบนสไลด์ 33)
- ลิงก์: https://arxiv.org/abs/2211.17192 (ส่ง 30 พ.ย. 2022)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 33-35
- ใจความ:
  - เร่งการสุ่มจากโมเดล autoregressive เช่น Transformer โดยโมเดลเล็กร่างโทเคนและโมเดลใหญ่ตรวจแบบขนานในรอบเดียว
  - ผลลัพธ์มีการแจกแจงเดียวกับโมเดลใหญ่ ("identical outputs") ไม่ได้ลดคุณภาพ
  - บน T5-XXL เร็วขึ้น 2-3 เท่าเทียบกับการใช้งาน T5X มาตรฐาน
  - ใช้กับโมเดลสำเร็จรูปได้ ไม่ต้องเทรนใหม่และไม่ต้องเปลี่ยนสถาปัตยกรรม
  - สไลด์ 35: รับโทเคนด้วยความน่าจะเป็น min(1, p/q) และปฏิเสธแล้วสุ่มแก้จาก normalize(max(0, p − q)); จำนวนโทเคนที่คาดหวังต่อรอบ = (1 − α^(γ+1)) ÷ (1 − α)
- เชื่อมกับข้อสอบ: หัวข้อ 13, ตัวอย่างข้อ 15 ข้อ n10-L1-06 และ gm25 ในคลัง (ตัวหลอกที่พบ: คิดว่าคุณภาพลดลง หรือคิดว่าเหมาะกับ serving ที่คนใช้พร้อมกันมาก)

---

### 11. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

- ผู้เขียน: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro
- ที่ตีพิมพ์: arXiv preprint 2019 (ตามรายการอ้างอิงบนสไลด์ 36)
- ลิงก์: https://arxiv.org/abs/1909.08053 (ส่ง 17 ก.ย. 2019 หน้าที่เปิดเป็นเวอร์ชัน 4)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 36 (Tensor Parallelism)
- ใจความ:
  - วิธี model parallel ภายในชั้น (intra-layer) สำหรับ Transformer ที่มีหลายพันล้านพารามิเตอร์
  - เทรนโมเดลถึง 8.3 พันล้านพารามิเตอร์ด้วย 512 GPU ได้ประสิทธิภาพต่อเนื่อง 15.1 PetaFLOPs
  - ประสิทธิภาพการขยาย 76% เมื่อเทียบกับ baseline GPU เดียวที่ 39 TeraFLOPs (ประมาณ 30% ของ peak)
  - เทรน BERT ขนาด 3.9 พันล้านพารามิเตอร์ และชี้ว่าตำแหน่งของ layer normalization สำคัญขึ้นเมื่อโมเดลใหญ่ขึ้น
  - โมเดลแบบ GPT-2: perplexity บน WikiText103 = 10.8 (ก่อนหน้า 15.8) และ LAMBADA ถูก 66.5% (ก่อนหน้า 63.2%)
  - จากสไลด์: แบ่ง GEMM แรกของ MLP ตามคอลัมน์ เพื่อให้ GeLU คำนวณอิสระต่อ GPU ได้ แบ่งตาม head ในส่วน attention ใช้ all-reduce 2 ครั้งใน forward และ 2 ครั้งใน backward ต่อชั้น
- เชื่อมกับข้อสอบ: หัวข้อ 14 (TP แบ่ง matrices ใช้ภายใน node ที่เชื่อม NVLink ต่างจาก PP ที่แบ่งชั้น) ตัวอย่างข้อ 17 และข้อ n10-X4-03 ในคลัง

---

### 12. Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey

- ผู้เขียน: Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, Sai Qian Zhang
- ที่ตีพิมพ์: arXiv preprint 2024 (หน้า arXiv ไม่ระบุ venue)
- ลิงก์: https://arxiv.org/abs/2403.14608 (ส่ง 21 มี.ค. 2024 หน้าที่เปิดเป็นเวอร์ชัน 7)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 39 (ไดอะแกรม (a) Additive, (b) Selective, (c) Reparameterization PEFT และหน้าอนุกรมวิธาน) สไลด์ไม่มีรายการอ้างอิงของ survey นี้ การระบุว่าเป็น survey นี้อนุมานจากข้อความบนสไลด์ ("Fig. 3: Taxonomy of Parameter-Efficient Fine-Tuning Methods for Large Models", "Fig. 4: Different types of PEFT algorithms") ซึ่งตรงกับชื่อเรื่อง โดยยืนยันได้เฉพาะชื่อ ผู้เขียนและปีจากหน้า arXiv ไม่ได้ตรวจเนื้อ PDF
- ใจความ:
  - นิยาม PEFT ว่าเป็นการปรับพารามิเตอร์ของโมเดลใหญ่ที่ pretrain แล้วให้เข้ากับงานหรือโดเมนหนึ่ง โดยเพิ่มพารามิเตอร์หรือการคำนวณให้น้อยที่สุด
  - สำรวจอัลกอริทึม PEFT เปรียบเทียบประสิทธิภาพและภาระการคำนวณ
  - ทบทวนงานประยุกต์ที่สร้างด้วย PEFT แบบต่าง ๆ และเทคนิคทั่วไปในการลดต้นทุนการคำนวณ
  - ตรวจการออกแบบระบบจริงเพื่อประเมินต้นทุนการใช้งานของแต่ละวิธี
  - หน้าที่แทรกในสไลด์แบ่ง PEFT เป็น 4 ตระกูล: Additive (แทรกโมดูลหรือพารามิเตอร์ใหม่), Selective (ทำให้บางส่วนของพารามิเตอร์เดิมเทรนได้), Reparameterized (สร้างการแทนที่มิติต่ำแล้วแปลงกลับตอน inference) และ Hybrid
  - สมการ adapter ที่สไลด์แสดง: Adapter(x) = W_up · ω(W_down · x) + x โดย r คือ bottleneck dimension
- เชื่อมกับข้อสอบ: หัวข้อ 15 (อนุกรมวิธาน PEFT และสมการ adapter) ใช้แยกให้ออกว่า LoRA จัดเป็น Reparameterized ส่วน adapter จัดเป็น Additive

---

### 13. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning

- ผู้เขียน: Armen Aghajanyan, Sonal Gupta, Luke Zettlemoyer (ลำดับผู้เขียนบนหน้า arXiv ต่างจาก ACL Anthology: arXiv ลงเป็น Aghajanyan, Zettlemoyer, Gupta)
- ที่ตีพิมพ์: ACL-IJCNLP 2021 (ACL Anthology 2021.acl-long.568 หน้า 7319-7328 และได้รางวัล Outstanding Paper ตามผลค้นหา) ส่งขึ้น arXiv เดือนธันวาคม 2020 (สไลด์อ้างปี 2021)
- ลิงก์: https://arxiv.org/abs/2012.13255 และ https://aclanthology.org/2021.acl-long.568/ (ยืนยัน ACL จากผลค้นหา ไม่ได้เปิดหน้า Anthology)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 40 ("The Key Observation (Aghajanyan et al., 2021)")
- ใจความ:
  - โมเดลภาษาที่ pretrain แล้วมี intrinsic dimension ต่ำมาก การทำ reparameterization ในมิติต่ำก็ fine-tune ได้ผลเท่ากับใช้พื้นที่พารามิเตอร์เต็ม
  - เทรนเพียง 200 พารามิเตอร์ที่ฉายแบบสุ่มกลับเข้าพื้นที่เต็ม ทำให้ RoBERTa ได้ 90% ของประสิทธิภาพ full fine-tuning บน MRPC
  - การ pretrain ช่วยลด intrinsic dimension โดยนัย และโมเดลที่ใหญ่กว่ามักมี intrinsic dimension ต่ำกว่าเมื่อผ่านจำนวน update เท่ากัน
  - เสนอขอบเขต generalization ที่ขึ้นกับ intrinsic dimension ไม่ใช่จำนวนพารามิเตอร์ทั้งหมด
- เชื่อมกับข้อสอบ: หัวข้อ 16 (เหตุผลที่ ΔW เป็น low-rank ได้) ซึ่งเป็นสัญชาตญาณเบื้องหลัง LoRA

---

### 14. LoRA: Low-Rank Adaptation of Large Language Models

- ผู้เขียน: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- ที่ตีพิมพ์: ICLR 2022 (ตามรายการอ้างอิงบนสไลด์ 41 หน้า arXiv ไม่ระบุ venue)
- ลิงก์: https://arxiv.org/abs/2106.09685 (ส่ง 17 มิ.ย. 2021)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 40-42 (หน้าบทความและตัวอย่าง) และสไลด์ 57 (แหล่งอ่านเพิ่ม หัวข้อ 1-4)
- ใจความ:
  - แช่แข็งน้ำหนักของโมเดล pretrain แล้วแทรกเมทริกซ์การแยกอันดับต่ำ (rank decomposition) ที่เทรนได้เข้าไปในแต่ละชั้นของ Transformer
  - เมื่อเทียบกับ GPT-3 175B ที่ fine-tune ด้วย Adam ลดพารามิเตอร์ที่เทรนได้ 10,000 เท่า และลดหน่วยความจำ GPU 3 เท่า
  - ทดสอบกับ RoBERTa, DeBERTa, GPT-2 และ GPT-3 คุณภาพเท่าหรือดีกว่าการ fine-tune ทั้งก้อน
  - **ไม่เพิ่ม latency ตอน inference** เพราะ merge BA กลับเข้า W0 ได้
  - ตัวเลขจาก Table 4 ที่บทเรียนยกมาจากหน้าแทรกในสไลด์ 42: LoRA 4.7M พารามิเตอร์ ได้ WikiSQL 73.4 และ MNLI-m 91.7 เทียบ full fine-tune ที่ได้ 73.8 และ 89.5
  - จากสไลด์: ΔW = B × A, A สุ่ม Gaussian และ B เริ่มที่ 0 (ΔW = 0 ตอนเริ่ม)
- เชื่อมกับข้อสอบ: หัวข้อ 16, ตัวอย่างข้อ 8 และข้อ 9 ข้อ n10-L2-04, n10-L4-02 และข้อเติมคำ 13.6M / 0.17% ในคลัง (สูตร r × (d_in + d_out))

---

### 15. PetS: A Unified Framework for Parameter-Efficient Transformers Serving

- ผู้เขียน: Zhe Zhou (Peking University), Xuechao Wei (Peking University, Alibaba Group), Jiejing Zhang (Alibaba Group), Guangyu Sun (Peking University)
- ที่ตีพิมพ์: USENIX Annual Technical Conference (USENIX ATC '22), Carlsbad, CA, ปี 2022 หน้า 489-504
- ลิงก์: https://www.usenix.org/conference/atc22/presentation/zhou-zhe
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 43 (multi-adapter serving)
- ใจความ:
  - ระบุว่าเป็นเฟรมเวิร์กแรกที่ให้บริการงาน parameter-efficient Transformer หลายงานในระบบเดียว
  - รองรับงานพร้อมกันได้มากกว่าวิธีเดิมถึง 25 เท่า
  - เพิ่ม throughput 1.53 เท่าบน Desktop GPU และ 1.63 เท่าบน Server GPU
  - ประเมินบน GPU แบบ Edge, Desktop และ Server
  - องค์ประกอบ: การแทนงานแบบรวม, เอนจิน inference ที่รวมคิวรีข้ามงานเป็น batch เดียว, กลยุทธ์ batch ที่ประสานกัน และการจัดตารางตัวดำเนินการ
- เชื่อมกับข้อสอบ: หัวข้อ 16 (multi-adapter serving: โหลดโมเดลฐานครั้งเดียวแล้วสลับ adapter เล็ก ๆ ตามคำขอ) ไม่ใช่เนื้อหาที่ต้องจำตัวเลขของ paper

---

### 16. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

- ผู้เขียน: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- ที่ตีพิมพ์: NeurIPS 2022 (สไลด์ 45: Proceedings of the 36th International Conference on Neural Information Processing Systems; หน้า arXiv ระบุ NeurIPS 2022)
- ลิงก์: https://arxiv.org/abs/2208.07339 (ส่ง 15 ส.ค. 2022)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 45 (The Outlier Problem & LLM.int8()) และสไลด์ 50
- ใจความ:
  - การคูณเมทริกซ์ Int8 ใน feed-forward และ attention projection ลดหน่วยความจำตอน inference ลงราวครึ่งหนึ่ง เมื่อเทียบกับ 16 หรือ 32 บิต โดยคงประสิทธิภาพเต็ม
  - แปลงจุดตรวจ 175B เป็น Int8 แล้วใช้ได้ทันทีโดยไม่เสียประสิทธิภาพ
  - mixed-precision decomposition แยกมิติ outlier ไปคำนวณที่ 16 บิต ส่วนมากกว่า 99.9% ของค่าถูกคูณที่ 8 บิต
  - ทำให้รัน OPT-175B หรือ BLOOM บนเซิร์ฟเวอร์เครื่องเดียวที่ใช้ GPU ผู้บริโภคได้
  - จากสไลด์: outlier features เกิดอย่างเป็นระบบเมื่อโมเดลมีขนาด ≥ 6.7B และกระจุกใน 6 feature dimensions
- เชื่อมกับข้อสอบ: หัวข้อ 17 (ปัญหา outlier ทำให้ absmax ใช้ระดับ INT8 ไม่คุ้ม และ block-wise/vector-wise ช่วยจำกัดความเสียหาย) ตัวอย่างข้อ 10

---

### 17. QLoRA: Efficient Finetuning of Quantized LLMs

- ผู้เขียน: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- ที่ตีพิมพ์: NeurIPS 2023 (ตามรายการอ้างอิงบนสไลด์ 46 และ 50: Advances in Neural Information Processing Systems 36) หน้า arXiv ระบุเพียง "Extended NeurIPS submission"
- ลิงก์: https://arxiv.org/abs/2305.14314 (ส่ง 23 พ.ค. 2023)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 46-50 และสไลด์ 57 (แหล่งอ่านเพิ่ม หัวข้อ 1-3)
- ใจความ:
  - fine-tune โมเดล 65B บน GPU 48 GB ใบเดียวโดยคงประสิทธิภาพของ 16-bit fine-tuning เต็มรูปแบบ
  - Guanaco ได้ 99.3% ของ ChatGPT บน Vicuna benchmark ใช้เวลา fine-tune 24 ชั่วโมงบน GPU ใบเดียว
  - fine-tune มากกว่า 1,000 โมเดลบน 8 ชุดคำสั่งและหลายตระกูลโมเดล (LLaMA, T5) รวมขนาด 33B และ 65B
  - 3 นวัตกรรม: 4-bit NormalFloat (NF4) ที่เหมาะกับน้ำหนักแบบปกติ, Double Quantization (quantize ค่าคงที่ของ quantization), Paged Optimizers (รับมือ memory spike)
  - ที่บทเรียนคำนวณ: Double Quantization ทำให้ NF4 เฉลี่ย ≈ 4.13 บิตต่อพารามิเตอร์ (สไลด์ 50) และต้นฉบับ paper นับหน่วยความจำของโมเดล 65B ก่อนใช้ที่ >780 GB ซึ่งต่างจากการนับ 16N ของสไลด์
- เชื่อมกับข้อสอบ: หัวข้อ 18 และ 20, ตัวอย่างข้อ 9 ข้อ n10-L1-05, n10-L3-04, gf3 และ n10-X4-F1 ในคลัง (QLoRA 3 ข้อ, paged optimizer ไป CPU RAM, ฐาน NF4 ถูกแช่แข็ง)

---

### 18. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer

- ผู้เขียน: Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, Sebastian Ruder
- ที่ตีพิมพ์: EMNLP 2020 (ตามช่อง Comments ของ arXiv)
- ลิงก์: https://arxiv.org/abs/2005.00052 (ส่ง 30 เม.ย. 2020)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 49
- ใจความ:
  - โมเดลหลายภาษา (mBERT, XLM-R) แย่ที่สุดกับภาษาทรัพยากรน้อยและภาษาที่ไม่เคยเห็นตอน pretrain ผู้เขียนอธิบายว่าเป็นเพราะ capacity จำกัด
  - เฟรมเวิร์กใช้ adapter แยกเป็นโมดูล "ภาษา" และ "งาน" เพื่อถ่ายโอนไปยังงานและภาษาใหม่อย่างประหยัดพารามิเตอร์
  - เสนอสถาปัตยกรรม invertible adapter และวิธีมาตรฐานสำหรับปรับโมเดลหลายภาษาให้เข้ากับภาษาใหม่
  - ชนะ SOTA เดิมของ cross-lingual transfer บน NER และ causal commonsense reasoning (CCR) ในหลายภาษา และได้ผลสูสีบน QA
  - ปล่อยโค้ดและ adapter ผ่าน AdapterHub
  - จากสไลด์: Language Adapter (MLM บนข้อความดิบ) + Task Adapter (เทรนด้วยภาษาอังกฤษ) ซ้อนกันตอน inference เช่น Thai-LA + NER-TA
- เชื่อมกับข้อสอบ: หัวข้อ 19 (adapter รวม L + T ตัว ไม่ใช่ L × T)

---

### 19. Orthogonal Subspace Learning for Language Model Continual Learning (O-LoRA)

- ผู้เขียน: Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, Xuanjing Huang
- ที่ตีพิมพ์: Findings of EMNLP 2023 (ตามช่อง Comments ของ arXiv)
- ลิงก์: https://arxiv.org/abs/2310.14152 (ส่ง 22 ต.ค. 2023)
- อยู่ในสไลด์: สัปดาห์ 10 สไลด์ 49 (สไลด์เขียนว่า Wang et al., 2023)
- ใจความ:
  - ปัญหา: LLM ที่เรียนหลายงานต่อเนื่องจะลืมงานเก่า (catastrophic forgetting)
  - เสนอ O-LoRA: เรียนแต่ละงานใน subspace อันดับต่ำของตัวเอง และบังคับให้ subspace เหล่านี้ตั้งฉากกันเพื่อลดการรบกวนระหว่างงาน
  - เพิ่มพารามิเตอร์น้อยและไม่ต้องเก็บข้อมูลผู้ใช้เก่าไว้เล่นซ้ำ (replay)
  - รายงานว่าชนะวิธี SOTA เดิมบน benchmark continual learning และรักษาการ generalize ไปยังงานที่ไม่เคยเห็นได้ดีกว่า
  - จากสไลด์และเดโม: penalty = ‖A_i A_jᵀ‖²_F ในเดโม r = 8, d = 64 ได้ 3547.79 เมื่อไม่บังคับ และ 0.00 เมื่อตั้งฉาก แต่ **ลด forgetting ไม่ได้กำจัด**
- เชื่อมกับข้อสอบ: หัวข้อ 19 และส่วนเดโม (ตัวหลอกที่พบ: คิดว่ากำจัด forgetting ได้หมด)

---

## ถูกกล่าวถึงบนสไลด์ 51-52 (ผลงานต่อยอดของ distillation)

สไลด์ 51-52 กล่าวถึงชื่อโมเดลและปีโดยไม่มีรายการอ้างอิง ผมตรวจหน้า arXiv ให้แล้วดังนี้

| ผลงาน | ผู้เขียน และปี | ลิงก์ | ข้อมูลจากบทคัดย่อ |
|---|---|---|---|
| DistilBERT (สไลด์ 52 "DistilBERT (2019)") | Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf; 2019; NeurIPS 2019 EMC² workshop | https://arxiv.org/abs/1910.01108 | เล็กลง 40%, เร็วขึ้น 60%, คงความเข้าใจภาษา 97% ของ BERT (ตรงกับสไลด์) ใช้ loss 3 แบบ: language modeling, distillation และ cosine-distance |
| TinyBERT | Xiaoqi Jiao และคณะ; 2019; Findings of EMNLP 2020 | https://arxiv.org/abs/1909.10351 | รุ่น 4 ชั้นได้ 96.8% ของ BERT-Base บน GLUE เล็กลง 7.5 เท่า เร็วขึ้น 9.4 เท่า; รุ่น 6 ชั้นเทียบเท่า BERT-Base |
| MiniLM | Wenhui Wang และคณะ; 2020; arXiv | https://arxiv.org/abs/2002.10957 | student เลียนแบบ self-attention ของชั้นสุดท้ายของ teacher และ value relations คงความแม่นยำมากกว่า 99% บน SQuAD 2.0 และหลายงาน GLUE โดยใช้พารามิเตอร์และ compute 50% |
| Gemma 2 | Gemma Team (Morgane Riviere และคณะ); 2024; arXiv | https://arxiv.org/abs/2408.00118 | โมเดล 2B-27B; รุ่น 2B และ 9B เทรนด้วย knowledge distillation แทน next-token prediction ตามปกติ |
| TinyLlama-1.1B | Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu; 2024; arXiv | https://arxiv.org/abs/2401.02385 | ~1.1B พารามิเตอร์ pretrain จากศูนย์ ราว 1 ล้านล้านโทเคน × 3 รอบ (สไลด์เขียนรวมว่า ~3T โทเคน ซึ่งสอดคล้องกัน) ไม่ใช่โมเดลที่ distill |
| DeepSeek-R1 (ต้นทางของ DeepSeek-R1-Distill) | DeepSeek-AI และคณะ; 2025; arXiv และ Nature 645:633-638 (2025) | https://arxiv.org/abs/2501.12948 | บทคัดย่อกล่าวเพียงว่ารูปแบบการให้เหตุผลของโมเดลใหญ่ใช้ชี้นำโมเดลเล็กได้ ไม่ได้ให้ขนาดหรือตัวเลขของรุ่น distill (ตัวเลข ~800k ตัวอย่างบนสไลด์ไม่ได้ตรวจกับต้นฉบับ) |

## ชื่อที่ถูกกล่าวถึงโดยไม่ได้ตรวจลิงก์

สไลด์เอ่ยชื่อต่อไปนี้ แต่ไม่มีรายการอ้างอิงและผมยังไม่ได้ตรวจหน้าต้นฉบับ จึงไม่ใส่ลิงก์: Llama 3.2 1B/3B (สไลด์ 52), DoRA, LoRA+, rsLoRA (สไลด์ 43), AQLM, GPTQ, AWQ (สไลด์ 32 และ 50), ORCA (ปรากฏในข้อความของหน้า vLLM ที่แทรกในสไลด์ 32), ETC และ BigBird (ปรากฏในข้อความของหน้า Longformer ที่แทรกในสไลด์ 23)

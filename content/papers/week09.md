# คลัง paper สัปดาห์ 9: Vector Representation และ Semantic Search

รวม paper ที่สไลด์ DL-09 อ้างถึงหรือแสดงหน้าเปเปอร์ไว้ทั้งหมด 19 รายการ (15 paper และ 4 ชุดข้อมูล) เรียงตามลำดับที่ปรากฏในสไลด์ ทุกรายการผมตรวจชื่อเรื่อง ผู้เขียน ปี และที่ตีพิมพ์กับหน้า arXiv, ACL Anthology หรือ NeurIPS Proceedings โดยเปิดอ่านเฉพาะหน้าบทคัดย่อ ไม่ได้ดาวน์โหลด PDF ส่วนตัวเลขที่ระบุว่ามาจากสไลด์ ผมอ่านจากสไลด์ ไม่ใช่จากตัวเปเปอร์

หมายเหตุการอ่าน: บทคัดย่อที่เครื่องมือดึงมาเป็นการสรุปความ ผมจึงเก็บเฉพาะข้อเท็จจริงและตัวเลขที่ปรากฏในบทคัดย่อ ส่วนรายละเอียดภายในเนื้อเปเปอร์ (เช่น วิธีคำนวณเต็ม) ผมไม่ได้อ่านเอง จึงไม่เขียนเกินกว่าที่สไลด์แสดง

---

### 1. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (SBERT)

- ผู้เขียน: Nils Reimers, Iryna Gurevych
- ที่ตีพิมพ์: EMNLP-IJCNLP 2019 (พ.ย. 2019), หน้า 3982-3992, DOI 10.18653/v1/D19-1410
- ลิงก์ (ยืนยันแล้ว): https://aclanthology.org/D19-1410/ และ https://arxiv.org/abs/1908.10084
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 4 (ตาราง STS), 13, 14 (หน้าเปเปอร์ Figure 1-2), 15 (ตาราง ablation pooling), 23, 49 (อ่านเพิ่มเติม)

ใจความ

- ปัญหาเดิม: BERT และ RoBERTa ทำงานคู่ประโยคได้ดีมาก แต่ต้องป้อนสองประโยคเข้าเครือข่ายพร้อมกัน จึงช้าเกินไปสำหรับการค้นหาหรือ clustering ขนาดใหญ่
- วิธีแก้: ดัดแปลง BERT ที่ pre-train แล้วให้เป็นโครงสร้าง siamese และ triplet เพื่อให้ได้ sentence embedding ที่เทียบกันด้วย cosine similarity ได้
- ตัวเลขในบทคัดย่อ: การหาคู่ที่คล้ายที่สุดในประโยค 10,000 ประโยคลดจากราว 65 ชั่วโมงเหลือราว 5 วินาที โดยรักษาความแม่นยำระดับ BERT (ตัวเลขนี้ต่างจาก 69 ชั่วโมงและ 50 วินาทีของสไลด์ 7-8 ซึ่งคำนวณจากสมมติ 5 ms ต่อ forward pass)
- ผลลัพธ์: SBERT และ SRoBERTa ชนะวิธี sentence embedding อื่นบนงาน STS และ transfer learning
- สิ่งที่สไลด์ 14-15 ดึงมา: ใช้ mean pooling เป็นค่าเริ่มต้น สามฟังก์ชัน objective (classification บน `[u; v; |u-v|]`, regression ด้วย cosine และ MSE, triplet ที่ใช้ระยะ Euclidean และ ε = 1) และเวลา fine-tune ไม่ถึง 20 นาที
- ผล ablation ของ pooling ตามสไลด์ 15 (Spearman ρ): Mean 80.78 (ฝึกด้วย NLI) และ 87.44 (ฝึกด้วย STSb), CLS 79.80 และ 86.62, Max 79.07 และ 69.92

เชื่อมกับข้อสอบ: เป็นหัวใจของสัปดาห์ 9 ครอบคลุมหัว classification 2,304 มิติ (hint สัปดาห์ 9 เวลา 00:08:22), ต้นทุน cross-encoder เทียบ bi-encoder (hint 00:14:37), pooling default และตาราง STS 58.02 / 16.50 / 46.35 / 77.03 ดูบทเรียนหัวข้อ 4 และ 5

---

### 2. How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings

- ผู้เขียน: Kawin Ethayarajh
- ที่ตีพิมพ์: EMNLP-IJCNLP 2019 (พ.ย. 2019), หน้า 55-65, DOI 10.18653/v1/D19-1006
- ลิงก์ (ยืนยันแล้ว): https://aclanthology.org/D19-1006/ และ https://arxiv.org/abs/1909.00512
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 5 (แสดงหน้าเปเปอร์ Figure 1) และสไลด์ 24 (SimCSE อ้างถึง)

ใจความ

- คำถามหลัก: representation ของ ELMo, BERT และ GPT-2 เป็น context-specific จริงแค่ไหน
- พบว่า representation **ไม่ isotropic ในทุกชั้น** ของโมเดลที่ศึกษา
- คำเดียวกันในบริบทต่างกันยังคล้ายกันมากกว่าคำต่างกัน แต่ความคล้ายกันของคำเดียวกัน (self-similarity) ลดลงในชั้นบน แปลว่าชั้นบนให้ representation ที่เฉพาะเจาะจงต่อบริบทมากขึ้น
- static embedding ของคำหนึ่งอธิบายความแปรปรวนของ representation แบบ contextualized ได้เฉลี่ยไม่ถึง 5%
- ข้อความที่สไลด์ 5 แสดงจากเปเปอร์: ค่าเฉลี่ยของ cosine ระหว่างคำสุ่มสองคำไม่เป็นศูนย์เกือบทุกชั้น ชั้นสูงมักมี anisotropy มากกว่าชั้นต่ำ ข้อยกเว้นคือ input layer ของ ELMo (ใช้ character-level ไม่มี context) และ penultimate layer ของ BERT ที่ anisotropic มากกว่าชั้นสุดท้าย stopword อย่าง "the", "of", "to" มี representation ที่ context-specific ที่สุด

เชื่อมกับข้อสอบ: เป็นหลักฐานเรื่อง anisotropy ในบทเรียนหัวข้อ 2 ข้อสอบมักถามว่าสาเหตุจริงคืออะไร ซึ่งสไลด์ 9 ตอบว่าเป็น training objective แบบ MLM ไม่ใช่สถาปัตยกรรม

---

### 3. All-but-the-Top: Simple and Effective Postprocessing for Word Representations

- ผู้เขียน: Jiaqi Mu, Suma Bhat, Pramod Viswanath
- ที่ตีพิมพ์: arXiv preprint 1702.01417 (v1 ก.พ. 2017, v2 มี.ค. 2018) หน้า arXiv ไม่ระบุที่ตีพิมพ์ สไลด์ 6 ใช้ปี 2018
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1702.01417
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 6 ("Mu & Viswanath, 2018"), สไลด์ 9 (แนวทางที่ 1 post-processing)

ใจความ

- เสนอขั้นตอน post-processing ง่าย ๆ: ลบเวกเตอร์เฉลี่ยร่วม (common mean vector) และทิศทางหลักไม่กี่ทิศ (top dominant directions) ออกจากเวกเตอร์คำ
- ได้ผลดีขึ้นกับ representation สำเร็จรูป (เช่น word2vec, GloVe) ทั้งงานระดับคำและระดับประโยค (STS และ text classification)
- ทดสอบกับหลายชุดข้อมูล หลายวิธี และหลายภาษา โดยผลหลังประมวลผลดีกว่าเดิมสม่ำเสมอ
- สไลด์ 6 ใช้เป็นเหตุผลว่ามีไม่กี่มิติที่สัมพันธ์กับความถี่ของ token ครอบงำ space และสไลด์ 9 จัดวิธีนี้เป็น "band-aid ไม่ใช่วิธีที่มีหลักการ" (ดีขึ้นปานกลาง)

เชื่อมกับข้อสอบ: ข้อสอบถามว่า post-processing เทียบกับ contrastive fine-tuning ต่างกันอย่างไร คำตอบคือ post-processing เร็วแต่ได้แค่ดีขึ้นปานกลาง ส่วน contrastive fine-tuning คือมาตรฐานอุตสาหกรรม

---

### 4. Signature Verification using a "Siamese" Time Delay Neural Network

- ผู้เขียน: Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, Roopak Shah
- ที่ตีพิมพ์: Advances in Neural Information Processing Systems 6 (NIPS 1993) หน้า 737-744 (หน้าที่มาจากผลค้นเว็บ ไม่ได้ดูในหน้าบทคัดย่อ)
- ลิงก์ (ยืนยันแล้ว): https://proceedings.neurips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 13 ("Bromley et al., 1993")

ใจความ

- ตรวจสอบว่าลายเซ็นบนแท็บเล็ตปากกาเป็นของจริงหรือไม่ ด้วยเครือข่ายแบบใหม่ที่เรียกว่า "Siamese"
- โครงสร้างคือ sub-network ที่เหมือนกันสองสาย ผลลัพธ์ถูกส่งไปรวมกันที่หน่วยที่วัดระยะห่างระหว่างเวกเตอร์คุณลักษณะของสองลายเซ็น
- ตอนตรวจสอบ เทียบลักษณะของลายเซ็นกับของผู้ลงนามที่เก็บไว้ ถ้าใกล้พอตาม threshold ก็ยอมรับ ไม่เช่นนั้นปฏิเสธเป็นลายเซ็นปลอม
- แต่ละสายเป็น time delay neural network (ข้อมูลจากผลค้นเว็บซึ่งอ้างอิงตัวเปเปอร์)
- คือที่มาของแนวคิด "สองสาขา weights เดียวกัน" ที่สไลด์ 13 บอกว่า Siamese มีต้นกำเนิดจากการตรวจลายเซ็น

เชื่อมกับข้อสอบ: Siamese = สถาปัตยกรรมเดียวกัน weights เดียวกัน (tied weights) ที่มาจากงานตรวจลายเซ็น (ดูบทเรียนหัวข้อ 5)

---

### 5. Learning a Similarity Metric Discriminatively, with Application to Face Verification

- ผู้เขียน: Sumit Chopra, Raia Hadsell, Yann LeCun
- ที่ตีพิมพ์: CVPR 2005 หน้า 539-546, DOI 10.1109/CVPR.2005.202
- ลิงก์ (ยืนยันข้อมูลบรรณานุกรมแล้ว ส่วนบทคัดย่อไม่ได้อยู่ในหน้านั้น): https://mlanthology.org/cvpr/2005/chopra2005cvpr-learning/
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 13 ("Chopra et al., 2005")

ใจความ

- เรียนฟังก์ชันที่ฉายอินพุตเข้า space ซึ่งระยะห่างสะท้อนความคล้ายเชิงความหมาย
- ทดสอบกับงานยืนยันใบหน้า (face verification) ด้วย convolutional network
- เป้าหมายการเทรนคือดึงคู่ที่เป็นคนเดียวกันเข้าหากัน และผลักคู่ที่เป็นคนต่างกันออกจากกัน
- ทดสอบกับฐานข้อมูลใบหน้า Purdue/AR ที่มีความแปรผันของท่าทาง แสง สีหน้า และสิ่งบดบังมาก
- สามข้อหลังนี้มาจากผลค้นเว็บ ผมไม่ได้อ่านบทคัดย่อฉบับเต็ม

เชื่อมกับข้อสอบ: เป็นรากของ Siamese สำหรับ face recognition ที่สไลด์ 13 พูดถึง และแนวคิดดึง-ผลัก (attract + repel) ของ contrastive learning

---

### 6. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- ผู้เขียน: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- ที่ตีพิมพ์: arXiv preprint 1810.04805 (ต.ค. 2018, v2 พ.ค. 2019) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1810.04805
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 18 (แสดงหน้าเปเปอร์ Figure 4 และคำอธิบายชุดข้อมูล GLUE) และสไลด์ 24 (SimCSE ใช้ BERT เป็น encoder)

ใจความ

- pre-train representation แบบสองทิศทางจากข้อความที่ไม่มี label โดยใช้บริบททั้งซ้ายและขวาในทุกชั้น
- ใช้ output layer เพิ่มชั้นเดียวก็ fine-tune ไปทำงานอย่าง QA และ language inference ได้
- ผลในบทคัดย่อ: ได้ผลดีที่สุดใน 11 งาน NLP รวมถึง GLUE 80.5%, MultiNLI 86.7%, SQuAD v1.1 F1 93.2 และ SQuAD v2.0 F1 83.1
- หน้าที่สไลด์ 18 แสดงคือ Figure 4 (วิธี fine-tune BERT กับงานต่าง ๆ: ประโยคเดี่ยว, คู่ประโยค, QA แบบ span, ติดป้ายระดับ token) และรายการชุดข้อมูล GLUE (CoLA, STS-B, MRPC, RTE, WNLI)
- ข้อสังเกตในหน้านั้น: เขียนว่า STS-B ให้คะแนน 1 ถึง 5 ส่วนสไลด์ 23 ของอาจารย์เขียนว่า 0 ถึง 5 (ดูหัวข้อความขัดแย้งในไฟล์ audit)

เชื่อมกับข้อสอบ: Week 6 ใช้ `[CLS]` และ head เดี่ยว ส่วน Week 9 ใช้เวกเตอร์คู่ `[u; v; |u-v|]` นี่คือหน้าที่สไลด์ 18 ใช้เทียบ cross-encoder (อินพุต `[CLS] A [SEP] B [SEP]`) กับ bi-encoder

---

### 7. RoBERTa: A Robustly Optimized BERT Pretraining Approach

- ผู้เขียน: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
- ที่ตีพิมพ์: arXiv preprint 1907.11692 (ก.ค. 2019) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1907.11692
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 15 (กล่าวถึงด้วยชื่อ "RoBERTa ablation proved this" ไม่มีเอกสารอ้างอิงที่ตัวสไลด์) และสไลด์ 14, 24 (SBERT และ SimCSE ใช้ RoBERTa เป็น encoder)

ใจความ

- เป็นการศึกษาซ้ำ (replication study) ของการ pre-train BERT ที่วัดผลของไฮเปอร์พารามิเตอร์สำคัญและขนาดข้อมูลฝึก
- สรุปว่า BERT เดิม "undertrained อย่างมีนัยสำคัญ" และเวอร์ชันที่ปรับแล้วเทียบเท่าหรือเหนือกว่าโมเดลที่ออกหลังจากนั้น
- ได้ผลดีที่สุดใน GLUE, RACE และ SQuAD พร้อมปล่อยโมเดลและโค้ด
- ข้ออ้างที่สไลด์ 15 ใช้ ("NSP เป็นสัญญาณฝึกที่แย่") มาจากตัวสไลด์ ผมยืนยันได้เฉพาะบทคัดย่อ ซึ่งไม่ได้พูดถึง NSP โดยตรง

เชื่อมกับข้อสอบ: เหตุผลที่ไม่ใช้ `[CLS]` เป็น sentence vector คือถูกเทรนด้วย NSP ซึ่งไม่ใช่สัญญาณที่ดีสำหรับความหมายของประโยค (ตาราง STS: `[CLS]` ได้ 16.50)

---

### 8. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere

- ผู้เขียน: Tongzhou Wang, Phillip Isola
- ที่ตีพิมพ์: ICML 2020 (arXiv 2005.10242, v10 ส.ค. 2022)
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/2005.10242
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 19 (แสดงหน้าแรกของเปเปอร์และ Figure 1) และสไลด์ 24 (SimCSE ใช้เครื่องมือวิเคราะห์นี้)

ใจความ

- ชี้สมบัติสองอย่างของ contrastive loss: alignment (ฟีเจอร์ของคู่ positive อยู่ใกล้กัน) และ uniformity (ฟีเจอร์ที่ normalize แล้วกระจายสม่ำเสมอบน hypersphere)
- พิสูจน์ว่าในเชิง asymptotic contrastive loss ปรับสมบัติทั้งสองนี้ และวิเคราะห์ผลดีต่องานปลายทาง
- เสนอเมตริกที่วัดได้ของแต่ละสมบัติ และทดลองบนชุดข้อมูลภาพและภาษา พบว่าเมตริกสัมพันธ์สูงกับผลของงานปลายทาง
- การปรับสองเมตริกนี้โดยตรงให้ผลเทียบเท่าหรือดีกว่าการเรียนแบบ contrastive
- สไลด์ 19 สรุปเป็น "maximize alignment (positives close) + uniformity (embeddings evenly spread)"

เชื่อมกับข้อสอบ: uniformity คือสิ่งตรงข้ามกับ anisotropy (กรวยแคบ) ดูบทเรียนหัวข้อ 5 ส่วน Alignment และ uniformity

---

### 9. SimCSE: Simple Contrastive Learning of Sentence Embeddings

- ผู้เขียน: Tianyu Gao, Xingcheng Yao, Danqi Chen
- ที่ตีพิมพ์: EMNLP 2021 (พ.ย. 2021) หน้า 6894-6910, DOI 10.18653/v1/2021.emnlp-main.552
- ลิงก์ (ยืนยันแล้ว): https://aclanthology.org/2021.emnlp-main.552/ และ https://arxiv.org/abs/2104.08821
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 9, 24 (แสดงหน้าเปเปอร์ Figure 1), 30 (loss แบบเดียวกับ MNRL), 49

ใจความ

- Unsupervised SimCSE: ให้ประโยคทำนายตัวมันเองจาก in-batch negatives โดยใช้ dropout มาตรฐานเป็นสัญญาณรบกวนอย่างเดียว บทคัดย่อบอกว่า dropout ทำหน้าที่เป็น data augmentation ขั้นต่ำ และ **ถ้าเอา dropout ออกจะเกิด representation collapse**
- Supervised SimCSE: ใช้คู่ NLI โดย entailment เป็น positive และ contradiction เป็น hard negative
- ผลกับ BERT-base: ค่าเฉลี่ย Spearman บน STS คือ **76.3% (unsupervised)** และ **81.6% (supervised)** เพิ่มขึ้น 4.2 และ 2.2 จุดจากผลดีที่สุดก่อนหน้า (ตามหน้า ACL Anthology)
- วิเคราะห์ว่า contrastive objective ทำให้ space แบบ anisotropic กระจายสม่ำเสมอขึ้น (uniformity) และเมื่อมี supervision จะ align คู่ positive ได้ดีขึ้น
- สไลด์ 24 เทียบกับ SBERT-base ที่ 74.89 (คนละเกณฑ์กับ 77.03 ของสไลด์ 4)

เชื่อมกับข้อสอบ: ข้อสอบถามเงื่อนไข "ปิด dropout แล้วเกิดอะไร" คำตอบตามเปเปอร์คือสอง view เหมือนกัน ไม่มี augmentation และ representation collapse (ดูบทเรียนหัวข้อ 7)

---

### 10. FaceNet: A Unified Embedding for Face Recognition and Clustering

- ผู้เขียน: Florian Schroff, Dmitry Kalenichenko, James Philbin
- ที่ตีพิมพ์: CVPR 2015, DOI 10.1109/CVPR.2015.7298682 (arXiv 1503.03832 มี.ค. 2015)
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1503.03832
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 13 (กล่าวถึงเรื่อง face recognition), 26-27 (อ้างอิงสูตร triplet loss), 28 (กลยุทธ์ semi-hard)

ใจความ

- เรียนการแปลงภาพใบหน้าเป็นเวกเตอร์กะทัดรัดใน Euclidean space ที่ระยะห่างสะท้อนความคล้ายของใบหน้า
- ฝึก CNN โดยปรับ embedding โดยตรงด้วย triplet (ใบหน้าที่ตรงกันกับที่ไม่ตรงกัน) ที่คัดเลือกด้วยวิธี online triplet mining
- ผล: ใช้เพียง 128 ไบต์ต่อใบหน้า ได้ความแม่นยำ 99.63% บน Labeled Faces in the Wild และ 95.12% บน YouTube Faces DB ลดอัตราความผิดพลาดของผลดีที่สุดก่อนหน้าลงราว 30%
- สไลด์ 28 ดึงแนวคิดว่า FaceNet เลือก **semi-hard negative** เพื่อให้เทรนเสถียรและมีประสิทธิภาพ

เชื่อมกับข้อสอบ: สูตร `L = max(0, d(a,p) − d(a,n) + α)` และการจำแนก easy / hard / semi-hard ดูบทเรียนหัวข้อ 6

---

### 11. In Defense of the Triplet Loss for Person Re-Identification

- ผู้เขียน: Alexander Hermans, Lucas Beyer, Bastian Leibe
- ที่ตีพิมพ์: arXiv preprint 1703.07737 (มี.ค. 2017, v4 พ.ย. 2017) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1703.07737
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 28 (หัวข้อ Hard Negative Mining)

ใจความ

- ปัญหา: นักวิจัยจำนวนมากเชื่อว่า triplet loss แย่กว่าฟังก์ชัน surrogate อย่าง classification หรือ verification loss ที่ตามด้วยขั้น metric learning แยก
- ข้อเสนอ: ตัวแปรหนึ่งของ triplet loss ที่ใช้เทรน deep metric แบบ end-to-end
- ผล: ตัวแปรนั้นชนะวิธีที่ตีพิมพ์ส่วนใหญ่ด้วยระยะห่างมาก ทั้งเทรนจากศูนย์และเริ่มจากโมเดล pre-train
- สไลด์ 28 ใช้เปเปอร์นี้ประกอบหัวข้อการขุด negative (easy: loss = 0 ไร้ประโยชน์, hard: gradient ใหญ่แต่อาจไม่เสถียร, semi-hard: สมดุลที่สุด)

เชื่อมกับข้อสอบ: การเลือกระดับความยากของ negative ดูบทเรียนหัวข้อ 6

---

### 12. Representation Learning with Contrastive Predictive Coding (CPC)

- ผู้เขียน: Aaron van den Oord, Yazhe Li, Oriol Vinyals
- ที่ตีพิมพ์: arXiv preprint 1807.03748 (ก.ค. 2018, v2 ม.ค. 2019) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1807.03748
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 30 (หัวข้อ MNRL / InfoNCE)

ใจความ

- เสนอ Contrastive Predictive Coding วิธี unsupervised ทั่วไปสำหรับเรียน representation จากข้อมูลมิติสูง
- เรียนด้วยการทำนายข้อมูลอนาคตใน latent space โดยใช้ autoregressive model
- ใช้ probabilistic contrastive loss ที่มี negative sampling ทำให้เทรนได้จริง
- ได้ผลดีใน 4 โดเมน: เสียงพูด ภาพ ข้อความ และสภาพแวดล้อม reinforcement learning แบบ 3 มิติ
- ชื่อ "InfoNCE" ที่สไลด์ใช้ มาจากตัวเปเปอร์นี้ตามสไลด์ 30 ส่วนบทคัดย่อที่ผมเปิดอ่านไม่ได้เรียกชื่อนี้ตรง ๆ ผมจึงยืนยันได้เฉพาะแนวคิด contrastive loss กับ negative sampling

เชื่อมกับข้อสอบ: สูตร `Lᵢ = −log [ exp(sim(qᵢ,pᵢ)/τ) / Σⱼ exp(sim(qᵢ,pⱼ)/τ) ]` ดูบทเรียนหัวข้อ 6 ส่วน MNRL

---

### 13. Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)

- ผู้เขียน: Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei
- ที่ตีพิมพ์: arXiv preprint 2212.03533 (ธ.ค. 2022, v2 ก.พ. 2024) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- ลิงก์ (ยืนยันแล้ว; เป็นลิงก์เดียวกับที่สไลด์ 34 ใส่ไว้): https://arxiv.org/abs/2212.03533
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 34 (ไทม์ไลน์ปี 2022)

ใจความ

- เสนอ E5 ตระกูลโมเดล text embedding แบบ single-vector ใช้ได้กับ retrieval, clustering และ classification
- เทรนแบบ contrastive จากคู่ข้อความที่ weakly supervised ในชุดข้อมูลชื่อ CCPairs
- ประเมินบน 56 ชุดข้อมูลจาก BEIR และ MTEB
- zero-shot: เป็นโมเดลแรกที่ชนะ BM25 บน BEIR retrieval โดยไม่ใช้ label
- หลัง fine-tune ได้ผล MTEB ดีที่สุด ชนะโมเดล embedding ที่มีพารามิเตอร์มากกว่าราว 40 เท่า
- สไลด์ 34 ขยายชื่อ E5 เป็น "Embeddings from bidirectional Encoder representations" และสรุปว่าเทรนบนคู่ (query, passage) ที่ขุดจาก CommonCrawl

เชื่อมกับข้อสอบ: โมเดลสมัยใหม่ชนะ SBERT เพราะเทรนบนคู่ (query, document) จากเว็บมหาศาล ไม่ใช่ NLI/STS อย่างเดียว (ดูบทเรียนหัวข้อ 8)

---

### 14. C-Pack: Packed Resources For General Chinese Embeddings (ชุดทรัพยากรที่มาพร้อมโมเดล BGE)

- ผู้เขียน: Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, Jian-Yun Nie
- ที่ตีพิมพ์: SIGIR 2024 (arXiv 2309.07597, ก.ย. 2023, v5 ก.ย. 2024)
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/2309.07597
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 20 (โมเดล `BAAI/bge-large-en-v1.5`) และสไลด์ 34 (BGE ปี 2023) **สไลด์ไม่ได้อ้างเปเปอร์นี้ตรง ๆ** ผมจับคู่เพราะ BAAI ปล่อยโมเดล BGE พร้อมชุดทรัพยากรนี้ โปรดเช็กกับอาจารย์ถ้าต้องอ้างอิงจริง

ใจความ

- C-Pack ประกอบด้วย 3 ส่วน: C-MTEB (benchmark 6 งาน 35 ชุดข้อมูล), C-MTP (ชุดข้อมูลฝึก embedding ขนาดใหญ่) และ C-TEM (โมเดล embedding หลายขนาด)
- โมเดลชนะ embedding จีนก่อนหน้าบน C-MTEB สูงสุดถึง 10% ณ เวลาที่เปิดตัว
- ปล่อยข้อมูลและโมเดลภาษาอังกฤษด้วย ซึ่งได้ผลสูงสุดบน MTEB และชุดข้อมูลภาษาอังกฤษใหญ่กว่าภาษาจีนสองเท่า
- ทรัพยากรทั้งหมดเปิดสาธารณะที่ FlagOpen/FlagEmbedding
- รายละเอียดเฉพาะของ bge-large-en-v1.5 (เช่น 1024 มิติ) มาจากตารางสไลด์ 20 ไม่ได้อยู่ในบทคัดย่อ

เชื่อมกับข้อสอบ: จำมิติโมเดลสามตัวจากสไลด์ 20 (384, 768, 1024) เพื่อคำนวณขนาด head และหน่วยความจำ

---

### 15. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models

- ผู้เขียน: Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych
- ที่ตีพิมพ์: NeurIPS 2021 Datasets and Benchmarks Track (arXiv 2104.08663, เม.ย. 2021, v4 ต.ค. 2021) หมายเหตุ: ชื่อบน arXiv สะกด "Heterogenous" ส่วนสไลด์พิมพ์ "Heterogeneous"
- ลิงก์ (ยืนยันแล้ว; เป็นลิงก์เดียวกับที่สไลด์ 35 ใส่ไว้): https://arxiv.org/abs/2104.08663
- อยู่ในสไลด์: สัปดาห์ 9 สไลด์ 35 (ตารางแหล่งข้อมูลฝึก)

ใจความ

- สร้าง benchmark ที่หลากหลายจาก 18 ชุดข้อมูลสาธารณะ ครอบคลุมงานและโดเมนการค้นข้อความต่างกัน
- ประเมินระบบค้น 10 ระบบ ทั้ง lexical, sparse, dense, late-interaction และ re-ranking
- BM25 เป็น baseline ที่แข็งแรง
- โมเดล re-ranking และ late-interaction ได้ค่าเฉลี่ย zero-shot ดีที่สุด แต่ต้นทุนคำนวณสูง
- โมเดล dense และ sparse มีประสิทธิภาพกว่าแต่มักด้อยกว่า ซึ่งชี้ว่ายังมีที่ให้ปรับ generalization

เชื่อมกับข้อสอบ: สอดคล้องกับ pipeline Retrieve-then-Rerank (bi-encoder รอบแรก, cross-encoder รอบสอง) และใช้เป็น benchmark ที่คนใช้ตรวจ embedding สมัยใหม่

---

## ชุดข้อมูลที่สไลด์ระบุชื่อ (สไลด์ 22, 23, 35)

### 16. A large annotated corpus for learning natural language inference (SNLI)

- ผู้เขียน: Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning
- ที่ตีพิมพ์: EMNLP 2015 (arXiv 1508.05326)
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1508.05326
- อยู่ในสไลด์: สไลด์ 22, 35 (ชื่อ SNLI, 570K) และสไลด์ 14 (หน้าเปเปอร์ SBERT อ้างถึง)

ใจความ

- ชุดข้อมูลประโยคคู่ที่มนุษย์เขียนประมาณ 570K คู่ สร้างจากคำบรรยายภาพ
- งาน natural language inference คือจำแนก entailment, contradiction (และ neutral)
- ใหญ่กว่าทรัพยากรที่เทียบได้ราวสองลำดับขนาด
- ตัวเลข 570K ตรงกับสไลด์ 22

เชื่อมกับข้อสอบ: NLI ร่วมกับ MultiNLI ได้ราว 1M คู่ ใช้ฝึก SBERT แบบ classification

### 17. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference (MultiNLI)

- ผู้เขียน: Adina Williams, Nikita Nangia, Samuel R. Bowman
- ที่ตีพิมพ์: NAACL 2018 (arXiv 1704.05426, v4 ก.พ. 2018)
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1704.05426
- อยู่ในสไลด์: สไลด์ 22, 35 (ชื่อ MultiNLI, 430K)

ใจความ

- ชุดข้อมูล NLI 433K ตัวอย่าง จัดเป็นหนึ่งในชุดที่ใหญ่ที่สุด
- ครอบคลุมภาษาอังกฤษแบบเขียนและพูด 10 แนว (genre)
- ใช้ทดสอบการปรับโดเมนข้ามแนวได้ด้วย
- สไลด์ปัดเป็น 430K ซึ่งตรงกับ 433K ของเปเปอร์

เชื่อมกับข้อสอบ: จำ 570K + 430K ≈ 1M คู่ (AllNLI)

### 18. SemEval-2017 Task 1: Semantic Textual Similarity (แหล่งของ STS Benchmark)

- ผู้เขียน: Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, Lucia Specia
- ที่ตีพิมพ์: SemEval workshop ที่ ACL 2017 (arXiv 1708.00055, DOI 10.18653/v1/S17-2001)
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1708.00055
- อยู่ในสไลด์: สไลด์ 23, 35 (STS Benchmark ~8,500 คู่) และสไลด์ 18 (หน้าเปเปอร์ BERT อ้างถึง "Cer et al., 2017")

ใจความ

- อธิบายงานร่วมของ SemEval ปี 2017 เรื่องวัดความคล้ายเชิงความหมายของประโยค เน้นหลายภาษาและข้ามภาษา
- มีทีมเข้าร่วม 31 ทีม และ 17 ทีมร่วมทุกแทร็กภาษา
- แนะนำ **STS Benchmark** เป็นชุดฝึกและประเมินที่คัดจากข้อมูล STS ภาษาอังกฤษปี 2012-2017
- ขนาด ~8,500 คู่ และสเกลคะแนน 0-5 มาจากสไลด์ 23 ไม่ได้อยู่ในบทคัดย่อ

เชื่อมกับข้อสอบ: ใช้เป็นข้อมูลฝึกแบบ regression ของ SBERT (label 0-5 ต้อง normalize เป็น [0,1] ก่อนทำ MSE กับ cosine) และเป็นเกณฑ์ที่ใช้วัด Spearman ในตาราง STS

### 19. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset

- ผู้เขียน: Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, Tong Wang
- ที่ตีพิมพ์: arXiv preprint 1611.09268 (พ.ย. 2016, v3 ต.ค. 2018) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- ลิงก์ (ยืนยันแล้ว): https://arxiv.org/abs/1611.09268
- อยู่ในสไลด์: สไลด์ 35 (ตาราง "MS MARCO ~500K+ pairs, Query → Passage")

ใจความ

- ชุดข้อมูล machine reading comprehension ขนาดใหญ่ มีคำถามประมาณ 1 ล้านข้อที่สุ่มจาก log การค้นของ Bing พร้อมคำตอบที่มนุษย์เขียน
- มีคำตอบที่มนุษย์เขียนใหม่อีกราว 183,000 ข้อ และ passage ราว 8.8 ล้านชิ้นจากเอกสารเว็บราว 3.5 ล้านชิ้น
- เสนอ 3 งาน: ตัดสินว่าตอบได้หรือไม่และสกัดคำตอบ, สร้างคำตอบที่เรียบเรียงดี, และ **จัดอันดับ passage ตามคำถาม**
- ตัวเลข "500K+ pairs" ของสไลด์ 35 ต่างจาก "1 ล้านคำถาม" ในบทคัดย่อ ผมไม่ได้ตรวจว่าต่างเพราะนับคนละอย่างหรือคนละเวอร์ชัน ให้ใช้เลขของสไลด์ตอบข้อสอบ

เชื่อมกับข้อสอบ: ตัวอย่างข้อมูลระดับ Silver (curated) คู่ query → passage ที่เทรน retrieval

---

## ที่สไลด์พูดถึงแต่ไม่มีเปเปอร์ให้ตรวจ

- Quora Duplicate Questions (สไลด์ 35, ~400K คู่): สไลด์ไม่ได้อ้างเปเปอร์ และผมไม่ได้ค้นเพิ่ม
- FAISS, HNSW, Annoy, ScaNN, IVF+PQ (สไลด์ 36, 39-40): สไลด์ไม่ได้อ้างเปเปอร์ จึงไม่รวมในคลังนี้ ตัวเลข recall ใช้ตามสไลด์ 39
- GloVe (สไลด์ 4 ในตาราง STS): สไลด์ไม่ได้ระบุที่มา
- Arora, Yosinski, Liu และอื่น ๆ: ปรากฏเฉพาะในข้อความของเปเปอร์ Ethayarajh และ SimCSE ที่สไลด์แสดง ไม่ใช่เอกสารที่อาจารย์อ้างเอง

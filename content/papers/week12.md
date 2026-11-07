# คลัง paper สัปดาห์ 12 (NLP-DL-12)

เปเปอร์ทั้งหมดที่สไลด์สัปดาห์ 12 อ้างเป็นรายการอ้างอิง ระบุชื่อในเนื้อหา หรือฉายหน้าเปเปอร์ขึ้นจอ เรียงตามหน้าสไลด์

**วิธีตรวจ:** ผมเปิดหน้า abstract ของแต่ละเปเปอร์ (arXiv หรือหน้า DOI/คลังอ้างอิง) ในวันที่ 7 พ.ย. 2026 เพื่อยืนยันชื่อเรื่อง ผู้เขียน ปี และเนื้อหาหลัก ไม่ได้ดาวน์โหลดไฟล์ PDF ส่วน "ใจความ" มาจาก abstract และจากเนื้อหาที่สไลด์สรุปไว้ ข้อที่มาจากสไลด์ไม่ใช่ abstract ผมบอกไว้ในบรรทัดนั้น ตัวเลขที่ไม่อยู่ใน abstract ผมไม่เติมเอง

**ข้อที่ยืนยันไม่ได้หรือขัดกับสไลด์ (สรุป):**
1. สไลด์ 20 อ้าง "Ma et al., 2021" โดยไม่ให้ชื่อเรื่อง ผมระบุเป็น *A Replication Study of Dense Passage Retriever* จากการค้นหา ชื่อผู้เขียนและปีตรงกันและ abstract กล่าวถึงผล hybrid แต่สไลด์ไม่ได้ยืนยันชื่อเรื่อง จึงควรถือว่า "น่าจะใช่"
2. สไลด์ 51 ระบุว่า YaRN ขยาย LLaMA-2 ถึง 128K ด้วยการ fine-tune ประมาณ 400 step จากการอ่านหน้า ar5iv ของเปเปอร์ ตัวเลข 400 step เป็นของโมเดล s = 16 (64K) ส่วนโมเดล 128K (s = 32) เริ่มจาก checkpoint s = 16 แล้วเทรนต่ออีก 200 step ดูรายละเอียดที่รายการ YaRN
3. SBERT abstract บอกว่าหาคู่ที่คล้ายที่สุดใน 10,000 ประโยคใช้ประมาณ 50 ล้านการคำนวณหรือประมาณ 65 ชั่วโมงด้วย BERT ส่วนสไลด์สัปดาห์ 9 ใช้ 5 ms ต่อครั้งและได้ประมาณ 69 ชั่วโมง ต่างกันเพราะสมมติเวลาต่อครั้งไม่เท่ากัน (ไม่ใช่ข้อผิดพลาด)

---

### Retrieval Augmentation Reduces Hallucination in Conversation

- **ผู้เขียน:** Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, Jason Weston
- **แหล่ง/ปี:** Findings of EMNLP 2021 (ตามรายการอ้างอิงบนสไลด์) ส่ง arXiv 15 เม.ย. 2021
- **ลิงก์:** https://arxiv.org/abs/2104.07567 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 2
- **ใจความ:**
  - ปัญหา: โมเดลสนทนาชั้นนำมักตอบข้อเท็จจริงผิดและ "hallucinate" ความรู้
  - ศึกษาสถาปัตยกรรมแบบ retrieval-in-the-loop ที่ประกอบด้วย retriever, ranker และ encoder-decoder สำหรับบทสนทนาที่อิงความรู้
  - เป้าหมายคือเพิ่มความรู้ที่ถูกต้องโดยยังคงความสามารถด้านการสนทนา
  - โมเดลที่ดีที่สุดได้ผลระดับ state of the art ในสองงานสนทนาที่อิงความรู้
  - การประเมินโดยมนุษย์ชี้ว่าวิธีนี้ลดการ hallucinate ความรู้ลงอย่างมาก (abstract ไม่ให้ตัวเลข)
- **เชื่อมกับข้อสอบ:** ใช้เป็นหลักฐานใน §1 ว่า retrieval augmentation ลด hallucination (ข้อสอบมักถามเหตุผลที่ต้องใช้ RAG ไม่ได้ถามชื่อเปเปอร์)

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

- **ผู้เขียน:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
- **แหล่ง/ปี:** NeurIPS 2020 (arXiv 22 พ.ค. 2020)
- **ลิงก์:** https://arxiv.org/abs/2005.11401 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 3 (ฉายหน้าเปเปอร์)
- **ใจความ:**
  - รวมโมเดล seq2seq ที่ pre-train แล้ว (parametric memory) เข้ากับดัชนีเวกเตอร์ของ Wikipedia (non-parametric memory) ที่เข้าถึงผ่าน neural retriever
  - เปรียบเทียบสองแบบ: ใช้ passage ชุดเดียวกันตลอดทั้งลำดับ กับใช้ passage ต่างกันต่อ token (ตรงกับ per-output และ per-token ในบทเรียน §2)
  - ได้ state of the art บนงานถาม-ตอบโดเมนเปิด 3 งาน เหนือกว่าโมเดล parametric ล้วนและสถาปัตยกรรมแบบดึงแล้วสกัดคำตอบ
  - ข้อความที่สร้างมีความเฉพาะเจาะจง หลากหลาย และถูกต้องตามข้อเท็จจริงมากกว่า baseline seq2seq แบบ parametric ล้วน
  - ใช้ DPR เป็น retriever และ BART เป็น generator (รายละเอียดจากเนื้อหาเปเปอร์ที่สไลด์ฉาย ไม่ใช่จาก abstract)
- **เชื่อมกับข้อสอบ:** §2 ข้อแม่นยำที่ตัวลวงชอบสลับ: retriever เป็น non-parametric (MIPS), generator เป็น parametric (ตาราง §8.1)

### Seven Failure Points When Engineering a Retrieval Augmented Generation System

- **ผู้เขียน:** Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, Mohamed Abdelrazek
- **แหล่ง/ปี:** CAIN 2024 (IEEE/ACM 3rd International Conference on AI Engineering) ตามรายการอ้างอิงบนสไลด์ ส่ง arXiv 11 ม.ค. 2024
- **ลิงก์:** https://arxiv.org/abs/2401.05856 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 4 (ฉายหน้าเปเปอร์)
- **ใจความ:**
  - เป็นรายงานประสบการณ์จาก 3 กรณีศึกษา ในด้านงานวิจัย การศึกษา และชีวการแพทย์
  - สรุป 7 จุดล้มเหลวที่ควรคิดตอนออกแบบ RAG (รายชื่อ 7 จุดอยู่ใน §3 ของบทเรียน จากรูปในสไลด์)
  - ข้อสรุป 1: ตรวจสอบความถูกต้องของระบบ RAG ทำได้จริงเมื่อระบบทำงานอยู่ (validation is only feasible during operation)
  - ข้อสรุป 2: ความทนทานของระบบ RAG เกิดจากการพัฒนาต่อเนื่อง ไม่ใช่ออกแบบเสร็จตั้งแต่ต้น
- **เชื่อมกับข้อสอบ:** §3 ชื่อ 7 จุดล้มเหลวและตำแหน่งใน pipeline

### Retrieval-Augmented Generation for Large Language Models: A Survey

- **ผู้เขียน:** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang
- **แหล่ง/ปี:** arXiv preprint (arXiv:2312.10997) ส่ง 18 ธ.ค. 2023 หน้า arXiv ไม่ระบุสถานที่ตีพิมพ์
- **ลิงก์:** https://arxiv.org/abs/2312.10997 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 5
- **ใจความ:**
  - LLM มีปัญหา hallucination ความรู้ล้าสมัย และเหตุผลที่ตรวจสอบไม่ได้ RAG แก้ด้วยการดึงข้อมูลจากฐานข้อมูลภายนอก
  - แบ่ง RAG เป็น 3 กระบวนทัศน์: Naive, Advanced และ Modular RAG
  - วิเคราะห์ 3 องค์ประกอบหลัก: retrieval, generation และ augmentation
  - เสนอกรอบการประเมินและ benchmark พร้อมสรุปความท้าทายที่เหลือ
- **เชื่อมกับข้อสอบ:** เป็นเปเปอร์ภาพรวมของ §2 ถึง §8 ไม่มีข้อสอบถามชื่อเปเปอร์โดยตรง

### Billion-scale similarity search with GPUs

- **ผู้เขียน:** Jeff Johnson, Matthijs Douze, Hervé Jégou
- **แหล่ง/ปี:** IEEE Transactions on Big Data 7(3), 2019 (ตามรายการอ้างอิงบนสไลด์) ส่ง arXiv 28 ก.พ. 2017
- **ลิงก์:** https://arxiv.org/abs/1702.08734 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 6 (เปเปอร์ของ FAISS)
- **ใจความ:**
  - เสนอการออกแบบ k-selection บน GPU ที่ใช้ประสิทธิภาพสูงสุดตามทฤษฎีได้ถึง 55%
  - ทำให้ค้นเพื่อนบ้านใกล้สุดเร็วกว่า GPU state of the art เดิมประมาณ 8.5 เท่า
  - เพิ่มประสิทธิภาพทั้งแบบ brute-force, แบบประมาณ และแบบ product quantization
  - สร้างกราฟ k-NN ความแม่นยำสูงของภาพ Yfcc100M 95 ล้านภาพใน 35 นาที
  - สร้างกราฟเชื่อม 1 พันล้านเวกเตอร์ในไม่ถึง 12 ชั่วโมงด้วย GPU 4 ใบ
  - เปิดซอร์สโค้ด (ไลบรารี FAISS)
- **เชื่อมกับข้อสอบ:** FAISS ใน §4.1 และเดโม §5.12.1 (`IndexFlatIP` ค้นแบบครบทุกเวกเตอร์ งานใหญ่ใช้ IVF หรือ HNSW)

### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

- **ผู้เขียน:** Nils Reimers, Iryna Gurevych
- **แหล่ง/ปี:** EMNLP-IJCNLP 2019 (arXiv 27 ส.ค. 2019)
- **ลิงก์:** https://arxiv.org/abs/1908.10084 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 10 (และสัปดาห์ 9)
- **ใจความ:**
  - BERT และ RoBERTa ต้องป้อนสองประโยคเข้าไปพร้อมกัน ทำให้การหาคู่ที่คล้ายที่สุดแพงมาก
  - abstract: หาคู่ใน 10,000 ประโยคใช้ประมาณ 50 ล้านการคำนวณ ประมาณ 65 ชั่วโมงด้วย BERT
  - SBERT ใช้โครงสร้าง siamese และ triplet สร้าง embedding ที่เทียบด้วย cosine ได้ ลดเวลาจากประมาณ 65 ชั่วโมงเหลือประมาณ 5 วินาที โดยความแม่นยำใกล้เคียง BERT
  - ได้ผลดีกว่าวิธี sentence embedding อื่นในงาน STS และงาน transfer
- **เชื่อมกับข้อสอบ:** ที่มาของ bi-encoder ใน §4.5, §5.9 (ตัวเลข 69 ชั่วโมงของสไลด์สัปดาห์ 9 ต่างจาก 65 ชั่วโมงในเปเปอร์เพราะใช้เวลาต่อครั้งต่างกัน)

### Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval (ANCE)

- **ผู้เขียน:** Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk
- **แหล่ง/ปี:** ICLR 2021 (ตามที่สไลด์อ้าง "Xiong et al., 2021") ส่ง arXiv 1 ก.ค. 2020 หน้า arXiv ที่ตรวจไม่แสดงที่ตีพิมพ์
- **ลิงก์:** https://arxiv.org/abs/2007.00808 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 19 (ระบุชื่อในเนื้อหา)
- **ใจความ:**
  - ปัญหาคอขวดคือ negative ที่ใช้เทรนไม่เหมือนเอกสารที่ไม่เกี่ยวข้องที่เจอตอนใช้จริง
  - ANCE สร้าง negative จากดัชนี approximate nearest neighbor ของคลัง ซึ่งอัปเดตขนานไปกับการเทรน
  - โมเดล BERT-Siamese ที่เทรนด้วย ANCE ชนะ baseline แบบ dense และ sparse ที่แข่งขันได้
  - ใช้ dot product ในพื้นที่ที่เรียนรู้แล้ว ความแม่นยำเกือบเท่า sparse retrieval ตามด้วย BERT rerank
  - เร็วขึ้นเกือบ 100 เท่า (ตัวเลขเดียวที่ abstract ให้)
- **เชื่อมกับข้อสอบ:** §5.4 hard negative mining (in-batch negative ง่ายเกินไป จึงสุ่ม negative ใกล้ขอบการตัดสินใจ)

### BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models

- **ผู้เขียน:** Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych
- **แหล่ง/ปี:** NeurIPS 2021 Datasets and Benchmarks Track (arXiv 17 เม.ย. 2021)
- **ลิงก์:** https://arxiv.org/abs/2104.08663 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 20 (ระบุชื่อในเนื้อหา)
- **ใจความ:**
  - รวม 18 ชุดข้อมูลสาธารณะที่ครอบคลุมงานและโดเมนค้นข้อความต่างกัน เพื่อวัดการทั่วไปแบบ zero-shot
  - ประเมินระบบ 10 ระบบ ทั้ง lexical, sparse, dense, late-interaction และ re-ranking
  - BM25 เป็น baseline ที่แข็งแรง (robust)
  - re-ranking และ late-interaction ได้ผลเฉลี่ย zero-shot ดีที่สุดแต่ต้นทุนการคำนวณสูง
  - dense และ sparse มีประสิทธิภาพด้านการคำนวณดีกว่า แต่มักด้อยกว่าแบบอื่นในการทั่วไป
- **เชื่อมกับข้อสอบ:** §5.5 และ §8.1 ("Dense ชนะ BM25 เสมอ" เป็นตัวลวง) สไลด์สรุปว่า BM25 ชนะ DPR ในส่วนใหญ่ (ข้อสรุปนี้เป็นของสไลด์ abstract ไม่ได้พูดเรื่อง DPR)

### A Replication Study of Dense Passage Retriever

- **ผู้เขียน:** Xueguang Ma, Kai Sun, Ronak Pradeep, Jimmy Lin
- **แหล่ง/ปี:** arXiv 12 เม.ย. 2021 (หน้าที่ตรวจไม่ระบุที่ตีพิมพ์)
- **ลิงก์:** https://arxiv.org/abs/2104.05740 (ตรวจหน้า abstract แล้ว แต่ **การจับคู่กับสไลด์ยืนยันไม่ได้** ดูข้อ 1 ด้านบน)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 20 (ระบุเป็น "Ma et al., 2021" โดยไม่มีชื่อเรื่อง)
- **ใจความ:**
  - ทำซ้ำ DPR ของ Karpukhin et al. (2020) ด้วยการเขียนใหม่อิสระใน Pyserini และ PyGaggle เริ่มจาก checkpoint เดิม
  - ผลส่วนใหญ่ยืนยันข้ออ้างของเปเปอร์ต้นฉบับ
  - พบว่าเปเปอร์ต้นฉบับน่าจะรายงานผล baseline BM25 ต่ำเกินไป ซึ่งส่งผลต่อผล hybrid ระหว่าง dense กับ sparse ด้วย
  - ปรับปรุงงานถาม-ตอบแบบ end-to-end ด้วยการเพิ่มหลักฐานจาก retriever และวิธีให้คะแนนช่วงคำตอบที่ดีขึ้น
- **เชื่อมกับข้อสอบ:** §5.5 ข้ออ้างว่าการรวม BM25 กับ dense ดีกว่าอย่างใดอย่างหนึ่ง

### Dense Passage Retrieval for Open-Domain Question Answering (DPR)

- **ผู้เขียน:** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih
- **แหล่ง/ปี:** EMNLP 2020 (arXiv 10 เม.ย. 2020)
- **ลิงก์:** https://arxiv.org/abs/2004.04906 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 20 (ระบุชื่อในเนื้อหา) และเป็น retriever ของ RAG ในสไลด์ 3
- **ใจความ:**
  - แย้งว่า retrieval ใช้ embedding แบบ dense ล้วนได้ เรียนรู้จากคำถามและ passage จำนวนไม่มากด้วยโครงสร้าง dual-encoder แบบง่าย
  - TF-IDF และ BM25 เป็นวิธีมาตรฐานมาช้านาน
  - บนชุดข้อมูลถาม-ตอบโดเมนเปิดหลายชุด dense retriever ชนะ Lucene-BM25 ประมาณ 9 ถึง 19 จุดสัมบูรณ์ใน top-20 passage accuracy
  - ระบบ QA แบบ end-to-end ได้ state of the art บนหลาย benchmark
- **เชื่อมกับข้อสอบ:** §5.4 ถึง §5.5 (dual encoder, BM25 เป็นคู่แข่ง)

### Reciprocal rank fusion outperforms Condorcet and individual rank learning methods

- **ผู้เขียน:** Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher
- **แหล่ง/ปี:** SIGIR 2009 (Proceedings of the 32nd international ACM SIGIR conference) หน้า 758-759
- **ลิงก์:** https://doi.org/10.1145/1571941.1572114 (DOI ยืนยันจากผลค้นหา; ไฟล์ฉบับเต็มโฮสต์ที่ https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf ผมยังไม่ได้อ่านฉบับเต็ม)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 22 (ตัวอ้างอิงบนหัวสไลด์) และสไลด์ 23
- **ใจความ:**
  - เสนอ Reciprocal Rank Fusion วิธีแบบไม่ต้องมี label สำหรับรวมอันดับจากหลายระบบค้นหา
  - รายงานว่าให้ผลดีกว่าระบบเดี่ยวทุกระบบอย่างสม่ำเสมอ และดีกว่าวิธี Condorcet Fuse
  - สูตรและค่า k = 60 ที่สไลด์ 22 และ 23 ใช้ (ข้อสอบถามตัวเลข 60 และการคำนวณ) มาจากการอธิบายของสไลด์ เปเปอร์ฉบับเต็มผมยังไม่ได้เปิดยืนยัน
- **เชื่อมกับข้อสอบ:** §5.7, §5.7.1, W7 ถึง W10 และข้อเติมคำ "ค่า k ของ RRF = 60 (Cormack et al. 2009)"

### Passage Re-ranking with BERT

- **ผู้เขียน:** Rodrigo Nogueira, Kyunghyun Cho
- **แหล่ง/ปี:** arXiv 13 ม.ค. 2019 (ปรับปรุงล่าสุด 14 เม.ย. 2020) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- **ลิงก์:** https://arxiv.org/abs/1901.04085 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 25
- **ใจความ:**
  - เป็นการนำ BERT มา re-rank passage ตาม query แบบเรียบง่าย
  - ได้ state of the art บนชุดข้อมูล TREC-CAR
  - อยู่อันดับ 1 บน leaderboard ของ MS MARCO passage retrieval ณ ตอนนั้น
  - ชนะ state of the art เดิม 27% (สัมพัทธ์) ใน MRR@10
- **เชื่อมกับข้อสอบ:** เป็นต้นแบบของ cross-encoder ใน §5.9 (สไลด์ 25: `[CLS] query [SEP] document [SEP]`)

### Document Ranking with a Pretrained Sequence-to-Sequence Model

- **ผู้เขียน:** Rodrigo Nogueira, Zhiying Jiang, Jimmy Lin
- **แหล่ง/ปี:** Findings of EMNLP 2020 (ตามรายการอ้างอิงบนสไลด์) ส่ง arXiv 14 มี.ค. 2020
- **ลิงก์:** https://arxiv.org/abs/2003.06713 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 25 (อยู่ในบรรทัดอ้างอิงเดียวกับ Passage Re-ranking with BERT)
- **ใจความ:**
  - ปรับโมเดล seq2seq ที่ pre-train แล้วไปจัดอันดับเอกสาร ต่างจากแนวทางจำแนกประเภทของโมเดล encoder-only อย่าง BERT
  - เทรนให้โมเดลสร้างป้ายความเกี่ยวข้องเป็นคำเป้าหมาย แล้วใช้ logit ของคำเหล่านั้นเป็นความน่าจะเป็นของความเกี่ยวข้อง
  - บน MS MARCO passage ranking ผลเทียบเท่าโมเดลแบบจำแนกประเภทรุ่นก่อน และชนะได้เมื่อใช้โมเดลใหญ่ขึ้น
  - บน TREC 2004 Robust Track วิธี zero-shot ชนะวิธีเดิมที่ต้อง cross-validate ในชุดข้อมูลเอง
  - ชนะโมเดล encoder-only ชัดเจนเมื่อข้อมูลเทรนน้อย
- **เชื่อมกับข้อสอบ:** §5.9 (ตัวอย่างโมเดล re-ranker หลายตระกูล)

### ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

- **ผู้เขียน:** Omar Khattab, Matei Zaharia
- **แหล่ง/ปี:** SIGIR 2020 (arXiv 27 เม.ย. 2020)
- **ลิงก์:** https://arxiv.org/abs/2004.12832 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 26 (ระบุชื่อ "ColBERT (2020)" ในตาราง ไม่มีบรรทัดอ้างอิง)
- **ใจความ:**
  - ใช้ late interaction: encode query และเอกสารแยกกันด้วย BERT แล้วเทียบกันด้วยขั้น similarity ระดับละเอียด
  - คำนวณ representation ของเอกสารล่วงหน้าแบบออฟไลน์ได้ ทำให้ประมวลผล query เร็ว
  - ใช้ดัชนีเวกเตอร์ในการค้นแบบ end-to-end จากคลังใหญ่ได้
  - ประสิทธิภาพเทียบเท่าโมเดลที่ใช้ BERT และชนะทุก baseline ที่ไม่ใช่ BERT
  - เร็วกว่าประมาณ 2 ลำดับขนาด และใช้ FLOPs ต่อ query น้อยกว่าประมาณ 4 ลำดับขนาด เมื่อเทียบกับ BERT re-ranking (abstract ระบุคู่เทียบเป็น BERT re-ranking ซึ่งสไลด์ 26 เรียกว่า cross-encoder)
- **เชื่อมกับข้อสอบ:** §5.9 และข้อ ColBERT ในคลัง (late interaction ด้วย MaxSim ถูกกว่า cross-encoder ประมาณ 2 ลำดับขนาด คุณภาพอยู่ระหว่างกลาง) ชื่อ MaxSim เป็นรายละเอียดจากสไลด์ ไม่ได้อยู่ใน abstract

### Learning Transferable Visual Models From Natural Language Supervision (CLIP)

- **ผู้เขียน:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
- **แหล่ง/ปี:** ICML 2021 (PMLR) ตามรายการอ้างอิงบนสไลด์ ส่ง arXiv 26 ก.พ. 2021
- **ลิงก์:** https://arxiv.org/abs/2103.00020 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 31 ถึง 34 (ฉายหน้าเปเปอร์)
- **ใจความ:**
  - pre-train ด้วยงานทำนายว่าคำบรรยายใดตรงกับภาพใด บนคู่ (ภาพ, ข้อความ) 400 ล้านคู่จากอินเทอร์เน็ต
  - ภาษาธรรมชาติอ้างถึงแนวคิดทางภาพที่เรียนรู้ได้ จึง transfer แบบ zero-shot ไปงานปลายทางได้
  - ทดสอบบนชุดข้อมูลคอมพิวเตอร์วิชันเกิน 30 ชุด (OCR, จำแนกการกระทำในวิดีโอ, ระบุตำแหน่งทางภูมิศาสตร์, จำแนกละเอียด)
  - มักแข่งกับ baseline แบบ supervised ได้โดยไม่ฝึกเฉพาะชุดข้อมูล
  - zero-shot ได้ความแม่นยำบน ImageNet เท่า ResNet-50 ต้นฉบับ โดยไม่ใช้ตัวอย่างฝึก 1.28 ล้านตัวอย่างของ ImageNet
  - ตัวเลข 76.2% Top-1 (ViT-L/14@336px) และ batch N = 32,768 ที่บทเรียนอ้างมาจากสไลด์/เนื้อหาเปเปอร์ ไม่ใช่ abstract
- **เชื่อมกับข้อสอบ:** §6.2 ถึง §6.5, W16 (ข้อมูล 400M คู่, symmetric InfoNCE, zero-shot ด้วยเทมเพลตข้อความ)

### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)

- **ผู้เขียน:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- **แหล่ง/ปี:** ICLR 2021 (arXiv 22 ต.ค. 2020 ฉบับ camera-ready แก้ไข 3 มิ.ย. 2021)
- **ลิงก์:** https://arxiv.org/abs/2010.11929 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 32 (ฉายหน้าเปเปอร์ มีบรรทัดอ้างอิง)
- **ใจความ:**
  - Transformer ใช้เป็นมาตรฐานใน NLP แต่ใน computer vision มักใช้ร่วมกับ CNN
  - แย้งว่าไม่จำเป็นต้องพึ่ง CNN: Transformer ล้วนที่ใช้กับลำดับของ patch ภาพจำแนกภาพได้ดี
  - เมื่อ pre-train บนชุดข้อมูลใหญ่แล้ว transfer ไป ImageNet, CIFAR-100, VTAB ได้ผลยอดเยี่ยมเมื่อเทียบกับ CNN ชั้นนำ
  - ใช้ทรัพยากรเทรนน้อยกว่าอย่างมีนัยสำคัญ
  - จำนวน patch = HW/P² เป็นความยาวลำดับของ Transformer (จากเนื้อหาเปเปอร์ที่สไลด์สรุป)
- **เชื่อมกับข้อสอบ:** §6.3, W17 (patch 224×224 ที่ P = 32 ได้ 49)

### Visual Instruction Tuning (LLaVA)

- **ผู้เขียน:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- **แหล่ง/ปี:** NeurIPS 2023 (Oral) arXiv 17 เม.ย. 2023
- **ลิงก์:** https://arxiv.org/abs/2304.08485 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 36 (บรรทัดอ้างอิง) และสไลด์ 35
- **ใจความ:**
  - ใช้ GPT-4 ที่รับข้อความล้วนสร้างข้อมูลสอนตามคำสั่งแบบหลายสื่อ (ผู้เขียนเรียกว่าความพยายามครั้งแรก)
  - เสนอ LLaVA โมเดลหลายสื่อขนาดใหญ่ที่เทรนแบบ end-to-end เชื่อม vision encoder เข้ากับ LLM
  - มีความสามารถสนทนาแบบหลายสื่อที่ดี บางครั้งใกล้เคียง GPT-4 หลายสื่อบนภาพและคำสั่งที่ไม่เคยเห็น
  - ได้คะแนนสัมพัทธ์ 85.1% เทียบกับ GPT-4 บนชุดข้อมูลสังเคราะห์
  - ได้ความแม่นยำ 92.53% บน Science QA เมื่อ fine-tune (state of the art ในตอนนั้น)
  - ข้อมูล 158K ตัวอย่างและสูตรเทรน 2 ขั้นที่ §6.6 มาจากสไลด์ ไม่ใช่ abstract
- **เชื่อมกับข้อสอบ:** §6.6, W17 (adapter ที่ต่อ visual token ไว้หน้า text token)

### Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)

- **ผู้เขียน:** Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee
- **แหล่ง/ปี:** CVPR 2024 (Highlight) arXiv 5 ต.ค. 2023
- **ลิงก์:** https://arxiv.org/abs/2310.03744 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 36 (กล่าวถึงชื่อ LLaVA-1.5 ในเนื้อหา ไม่มีบรรทัดอ้างอิง)
- **ใจความ:**
  - ตัวเชื่อมแบบ fully-connected ระหว่างภาพกับภาษาของ LLaVA มีพลังและใช้ข้อมูลอย่างมีประสิทธิภาพอย่างน่าประหลาดใจ
  - การเปลี่ยนสองอย่างให้ผลดี: ใช้ CLIP-ViT-L ความละเอียด 336px พร้อม projection แบบ MLP และเพิ่มข้อมูล VQA เชิงวิชาการพร้อมพรอมป์ตจัดรูปแบบคำตอบ
  - ได้ผล state of the art ใน 11 benchmark
  - โมเดล 13B ตัวสุดท้ายใช้ข้อมูลสาธารณะเพียง 1.2 ล้านตัวอย่าง
  - เทรนเต็มรูปแบบใช้เวลาประมาณ 1 วันบนเครื่อง A100 จำนวน 8 ใบ
- **เชื่อมกับข้อสอบ:** §6.6 "อัปเกรดใน LLaVA-1.5" (MLP 2 ชั้นแทน linear และภาพ 336×336 = 576 patch)

### Knowledge Graphs

- **ผู้เขียน:** Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d'Amato, Gerard de Melo, Claudio Gutierrez, José Emilio Labra Gayo, Sabrina Kirrane, Sebastian Neumaier, Axel Polleres, Roberto Navigli, Axel-Cyrille Ngonga Ngomo, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan Sequeda, Steffen Staab, Antoine Zimmermann (18 คน)
- **แหล่ง/ปี:** ACM Computing Surveys 54(4), 2021 (arXiv 4 มี.ค. 2020 ฉบับ v6 แก้ไข 11 ก.ย. 2021)
- **ลิงก์:** https://arxiv.org/abs/2003.02320 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 40 (บรรทัดอ้างอิงเรื่องนิยามกราฟความรู้)
- **ใจความ:**
  - เป็นบทนำฉบับครอบคลุมเรื่องกราฟความรู้ ที่ได้รับความสนใจจากอุตสาหกรรมและวิชาการ
  - เปรียบเทียบโมเดลข้อมูลแบบกราฟและภาษา query
  - อภิปรายบทบาทของ schema, identity และ context
  - ครอบคลุมการแทนและสกัดความรู้ด้วยเทคนิคแบบ deductive และ inductive
  - สรุปวิธีสร้าง เพิ่มคุณค่า ประเมินคุณภาพ ปรับปรุง และเผยแพร่กราฟความรู้ พร้อมสำรวจกราฟความรู้แบบเปิดและแบบองค์กร
- **เชื่อมกับข้อสอบ:** §6.10 นิยามกราฟ (node, edge, triple) abstract ไม่มีตัวเลข

### From Local to Global: A Graph RAG Approach to Query-Focused Summarization (GraphRAG)

- **ผู้เขียน:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson
- **แหล่ง/ปี:** arXiv 24 เม.ย. 2024 (ปรับปรุง 19 ก.พ. 2025) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- **ลิงก์:** https://arxiv.org/abs/2404.16130 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 40 ถึง 44 (ฉายหน้าเปเปอร์ มีบรรทัดอ้างอิง)
- **ใจความ:**
  - RAG มาตรฐานล้มเหลวกับคำถามเชิงภาพรวมของทั้งคลัง ซึ่งผู้เขียนมองเป็นงานสรุปความที่ตอบคำถาม (query-focused summarization) ไม่ใช่ retrieval
  - วิธีสรุปความเดิมไม่ scale กับปริมาณข้อความที่ระบบ RAG ทำดัชนี
  - GraphRAG ใช้ LLM สร้างดัชนีแบบกราฟ: เริ่มจากกราฟความรู้ของเอนทิตี แล้วสร้างสรุปของ community เอนทิตีที่เกี่ยวข้องกันไว้ล่วงหน้า
  - ตอน query แต่ละสรุปของ community สร้างคำตอบบางส่วน แล้วสรุปรวมเป็นคำตอบสุดท้าย (map-reduce)
  - กับชุดข้อมูลระดับ 1 ล้าน token และคำถามเชิง sensemaking ระดับโลก GraphRAG เพิ่มความครอบคลุมและความหลากหลายของคำตอบเหนือ RAG แบบดั้งเดิมอย่างมีนัยสำคัญ (abstract ไม่ให้คะแนน)
- **เชื่อมกับข้อสอบ:** §6.12 และข้อตัวลวงที่ว่า community summary สร้างตอน indexing ไม่ใช่ตอน query

### From Louvain to Leiden: guaranteeing well-connected communities

- **ผู้เขียน:** Vincent Traag, Ludo Waltman, Nees Jan van Eck
- **แหล่ง/ปี:** Scientific Reports 9, 5233 (2019) (arXiv 19 ต.ค. 2018)
- **ลิงก์:** https://arxiv.org/abs/1810.08473 (ตรวจแล้ว; DOI ที่หน้า arXiv ระบุคือ 10.1038/s41598-019-41695-z)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 43 (บรรทัดอ้างอิง)
- **ใจความ:**
  - อัลกอริทึม Louvain อาจสร้าง community ที่เชื่อมต่อกันแย่มาก และ community อาจขาดจากกัน โดยเฉพาะเมื่อรันซ้ำหลายรอบ
  - ในการทดลองของผู้เขียน ถึง 25% ของ community เชื่อมต่อไม่ดี และถึง 16% ขาดจากกัน
  - Leiden รับประกันว่า community เชื่อมต่อกัน
  - เมื่อรันซ้ำ Leiden ลู่เข้าสู่การแบ่งที่ทุกส่วนย่อยของทุก community ถูกจัดให้อยู่ที่ที่เหมาะสมในเชิงท้องถิ่น
  - เร็วกว่า Louvain และหาการแบ่งที่ดีกว่าบนเครือข่ายทดสอบและเครือข่ายจริง
- **เชื่อมกับข้อสอบ:** §6.12 และข้อเติมคำ (Leiden แก้ Louvain ให้ community เชื่อมต่อกันดี) ตัวเลข 25% และ 16% มาจาก abstract ไม่อยู่ในสไลด์ ไม่น่าเป็นข้อสอบ

### Lost in the Middle: How Language Models Use Long Contexts

- **ผู้เขียน:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
- **แหล่ง/ปี:** Transactions of the Association for Computational Linguistics (TACL) เล่ม 12, 2024 (arXiv 6 ก.ค. 2023 หน้า arXiv ระบุ "accepted for publication in TACL, 2023")
- **ลิงก์:** https://arxiv.org/abs/2307.03172 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 47 และสไลด์ 52 (ฉายหน้าเปเปอร์ มีบรรทัดอ้างอิง)
- **ใจความ:**
  - ทดสอบความสามารถในการหาข้อมูลที่เกี่ยวข้องในอินพุตยาว ด้วยการถาม-ตอบหลายเอกสารและการค้นคู่ key-value
  - ประสิทธิภาพลดลงชัดเจนเมื่อตำแหน่งของข้อมูลที่เกี่ยวข้องเปลี่ยน
  - ความแม่นยำมักสูงสุดเมื่อข้อมูลอยู่ที่ต้นหรือท้ายอินพุต และลดลงเมื่ออยู่กลาง รวมถึงโมเดลที่ออกแบบมาสำหรับ context ยาว
  - เสนอโปรโตคอลประเมินใหม่สำหรับโมเดล long-context
  - ค่า 76%, 54-55%, 63% และ closed-book 56% ในบทเรียน อ่านจากกราฟของสไลด์ ไม่ใช่ abstract
- **เชื่อมกับข้อสอบ:** §7.5, §7.6 (U-shaped, bookend ordering, 4 กฎ) ปีในเดโมคือ 2023 และสไลด์คือ 2024 เป็นเปเปอร์เดียวกัน

### Longformer: The Long-Document Transformer

- **ผู้เขียน:** Iz Beltagy, Matthew E. Peters, Arman Cohan
- **แหล่ง/ปี:** arXiv 10 เม.ย. 2020 (ฉบับ v2 ธ.ค. 2020 เพิ่ม LED) หน้า arXiv ไม่ระบุที่ตีพิมพ์
- **ลิงก์:** https://arxiv.org/abs/2004.05150 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 48 (บรรทัดอ้างอิงของหัวข้อ Sliding Window Attention)
- **ใจความ:**
  - self-attention มาตรฐานโตตามกำลังสองของความยาวลำดับ จึงจำกัดอินพุตยาว ส่วน attention ของ Longformer โตเป็นเส้นตรงตามความยาว
  - รวม local windowed attention กับ global attention ที่เลือกตามงาน ใช้แทน self-attention เดิมได้โดยตรง
  - ได้ผล state of the art ด้านโมเดลภาษาระดับตัวอักษรบน text8 และ enwik8
  - โมเดลที่ pre-train แล้วชนะ RoBERTa ในงานเอกสารยาว และได้ผลใหม่สูงสุดบน WikiHop และ TriviaQA
  - รุ่น LED รองรับงานสร้างข้อความเอกสารยาว เช่น สรุปเปเปอร์ arXiv
- **เชื่อมกับข้อสอบ:** §7.2 (O(N²) → O(N·w) และ rolling KV buffer)

### Mistral 7B

- **ผู้เขียน:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (18 คน)
- **แหล่ง/ปี:** arXiv 10 ต.ค. 2023 (ไม่มีที่ตีพิมพ์แบบ peer-review)
- **ลิงก์:** https://arxiv.org/abs/2310.06825 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 49 (ใช้เป็นตัวอย่างในกฎการขยาย 32 × 4,096 = 131,072 โดยไม่มีบรรทัดอ้างอิง)
- **ใจความ:**
  - โมเดลภาษา 7 พันล้านพารามิเตอร์ ที่เน้นประสิทธิภาพและความเร็ว
  - อ้างว่าชนะ Llama 2 13B ในทุก benchmark ที่ประเมิน และชนะ Llama 1 34B ด้านการให้เหตุผล คณิตศาสตร์ และการเขียนโค้ด
  - ใช้ grouped-query attention เพื่ออนุมานเร็วขึ้น และ sliding window attention เพื่อรับลำดับยาวด้วยต้นทุนต่ำ
  - มีรุ่น Instruct ที่ชนะ Llama 2 13B Chat ในการประเมินโดยมนุษย์และอัตโนมัติ
  - ตัวเลข 32 ชั้นและหน้าต่าง 4,096 มาจากสไลด์ 49 ไม่ได้อยู่ใน abstract
- **เชื่อมกับข้อสอบ:** §7.2 และ W18 (receptive field ≈ L × w = 131,072 token)

### Extending Context Window of Large Language Models via Positional Interpolation

- **ผู้เขียน:** Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian
- **แหล่ง/ปี:** arXiv 27 มิ.ย. 2023 หน้า arXiv ไม่ระบุที่ตีพิมพ์
- **ลิงก์:** https://arxiv.org/abs/2306.15595 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 50 (บรรทัดอ้างอิงท้ายสไลด์)
- **ใจความ:**
  - ขยาย context ของโมเดลที่ใช้ RoPE เช่น LLaMA ไปถึง 32,768 token ด้วยการ fine-tune ไม่เกิน 1,000 step ทดสอบบน LLaMA 7B ถึง 65B
  - ได้ผลดีในงาน passkey retrieval, การสร้างแบบจำลองภาษา และการสรุปเอกสารยาว
  - รักษาประสิทธิภาพภายในหน้าต่างเดิมได้เป็นส่วนใหญ่
  - ทำโดย "ย่อดัชนีตำแหน่งของอินพุตแบบเชิงเส้น" ให้พอดีหน้าต่างเดิม (m' = m/s ตรงกับสไลด์)
  - การวิเคราะห์เชิงทฤษฎีระบุว่าขอบบนของ interpolation เล็กกว่าขอบบนของ extrapolation อย่างน้อยประมาณ 600 เท่า
  - สถาปัตยกรรมคงเดิม จึงใช้โครงสร้างพื้นฐานเดิมได้
- **เชื่อมกับข้อสอบ:** §7.3, W18 (s = L_target/L_train, ตำแหน่ง 8000 → 2000 เมื่อ s = 4) ตัวอย่าง 4K → 16K เป็นของสไลด์ ไม่ใช่ของเปเปอร์

### YaRN: Efficient Context Window Extension of Large Language Models

- **ผู้เขียน:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
- **แหล่ง/ปี:** ICLR 2024 (ตามบรรทัดอ้างอิงบนสไลด์) ส่ง arXiv 31 ส.ค. 2023
- **ลิงก์:** https://arxiv.org/abs/2309.00071 (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 12 สไลด์ 51
- **ใจความ:**
  - RoPE เข้ารหัสตำแหน่งได้ดี แต่โมเดลทั่วไปไม่ได้ผลเกินความยาวลำดับที่เทรน
  - YaRN ต้องการ token น้อยลง 10 เท่าและ step เทรนน้อยลง 2.5 เท่าเมื่อเทียบกับวิธีก่อนหน้า (ตามข้อความ abstract ฉบับปัจจุบัน)
  - ทำให้โมเดล LLaMA ใช้และ extrapolate ไปยัง context ที่ยาวกว่าที่ pre-train ไว้ได้มาก
  - ชนะวิธีขยายหน้าต่างบริบทรุ่นก่อน และ extrapolate เกินความยาวของชุด fine-tune ได้
  - **ตรวจข้ออ้างของสไลด์ 51:** สไลด์เขียนว่าขยาย LLaMA-2 จาก 4K เป็น 128K ด้วย fine-tune ประมาณ 400 step (≈ 0.1% ของ token ตอน pre-train) จากหน้า ar5iv ของเปเปอร์ (อ่านผ่านตัวสรุปอัตโนมัติ ควรเปิดเปเปอร์ยืนยันอีกรอบ) ตัวเลข ≈ 0.1% ของข้อมูล pre-train มีอยู่ในเปเปอร์จริง (หัวข้อ Introduction และ Experiments) แต่ 400 step เป็นของโมเดล s = 16 (64K) ส่วนโมเดล 128K (s = 32) เริ่มจาก checkpoint s = 16 แล้วเทรนต่ออีก 200 step ดังนั้นประโยค "128K ด้วยประมาณ 400 step" ในสไลด์ไม่ตรงกับเปเปอร์ทุกตัวอักษร
  - NTK-aware (บีบความถี่ต่ำ รักษาความถี่สูง) เป็นเนื้อหาของสไลด์ 51 ไม่ได้อยู่ใน abstract
- **เชื่อมกับข้อสอบ:** §7.4 ในข้อสอบให้จำแนวคิด (linear บีบทุกความถี่ NTK-aware บีบความถี่ต่ำ) ไม่ต้องจำจำนวน step

---

## ชื่อที่ปรากฏในสไลด์แต่ไม่ได้ตรวจเป็นรายเปเปอร์

ชื่อต่อไปนี้ปรากฏเฉพาะในข้อความของหน้าเปเปอร์ที่ฉายขึ้นจอ หรือในไทม์ไลน์ที่สไลด์ไม่ให้เปเปอร์ ผมไม่ได้ตรวจและไม่ได้เขียนเป็นรายการแยก

- **อยู่ในหน้า CLIP (สไลด์ 31 ถึง 33):** Li et al. (2017), Mahajan et al. (2018), Kolesnikov et al. (2019), Dosovitskiy et al. (2020)
- **อยู่ในหน้า ViT (สไลด์ 32):** Wang et al. (2019) และ Baevski & Auli (2019) เรื่องการวาง LayerNorm
- **อยู่ในหน้า GraphRAG (สไลด์ 42):** Es et al. (2023) คือ RAGAS และ Wang et al. (2023a)
- **อยู่ในหน้า SBERT (สไลด์ 9 ถึง 10):** Bowman et al. (2015) และ Schroff et al. (2015)
- **ไทม์ไลน์ IR (สไลด์ 13):** Okapi BM25 (1994) และ PageRank (1998) ไม่มีบรรทัดอ้างอิงในสไลด์

รวมที่ตรวจเป็นรายเปเปอร์ **26 เปเปอร์** (ผ่านหน้า abstract ทุกเปเปอร์ ยกเว้น Cormack ที่ยืนยันจากหน้าผลค้นหาและ DOI)

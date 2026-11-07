# คลัง paper สัปดาห์ 14 (Data Engines, Advanced Evaluation, Pipeline Metrics)

รวม paper ที่สไลด์ DL-14 อ้างถึงหรือแสดง ตรวจจากสไลด์ PDF ทุกหน้า (รวมตัวอักษรเล็กท้ายสไลด์) ได้ **29 รายการ**: อ้างอิงในสไลด์ 27 รายการ และอีก 2 รายการ (HarmBench, GSM8K) สไลด์แสดงชุดทดสอบแต่ไม่ได้ใส่ citation

## วิธีตรวจและข้อจำกัด

- เปิดหน้า abstract บน arXiv, ACL Anthology, หน้าโครงการ หรือหน้าข้อมูลด้วย WebFetch (ลิงก์ขึ้นต้น https://) แล้วยืนยัน ชื่อเรื่อง ผู้เขียน และปี **ไม่ได้ดาวน์โหลด PDF**
- สรุป "ใจความ" ในรายการมาจากหน้า abstract และจากสไลด์ ตัวเลขที่ปรากฏเฉพาะในเนื้อเปเปอร์ (ไม่อยู่ในหน้า abstract) จะระบุว่าไม่ได้ตรวจ
- สถานะ: **ตรวจแล้ว** = ชื่อ ผู้เขียน ปี ตรงกับหน้าต้นทาง (24 รายการ); **ตรวจบางส่วน/จากแหล่งรอง** = 5 รายการ (Lozhkov การ์ดชุดข้อมูล, Manku, Charikar, Leskovec et al., Anthropic) ระบุเหตุผลในรายการ
- ข้อมูล venue บางรายการ (เช่น Chatbot Arena, RAGAS, CCNet, T5, Ahia, CoT) หน้า arXiv ไม่ระบุ ใช้ venue ตามที่สไลด์เขียนและติดป้ายไว้ว่า "ตามสไลด์"

## ผลตรวจที่ควรรู้ (ขัดหรือยืนยันไม่ได้)

1. **จำนวนตัวอย่างที่ Llama-3-70B-Instruct ให้คะแนน (FineWeb-Edu):** การ์ดของ classifier บน HuggingFace ระบุ **450k** annotations (ขณะที่บทเรียนและสไลด์ 16 เขียน ~460k) ข้อสอบจะอิงสไลด์ แต่ควรรู้ว่าแหล่งต้นทางว่า 450k
2. **ตัวเลขของ Claude 3.5 Sonnet (GSM8K 96.4%, HumanEval 92.0%):** หน้าประกาศของ Anthropic ที่เปิดได้ ไม่แสดงตัวเลขสองตัวนี้ในข้อความ (ตัวเลขอยู่ในรูปแผนภูมิ) จึงยืนยันจากหน้าไม่ได้ และตัวเลข GSM8K 92.0% ของ GPT-4 (5-shot CoT) ก็ไม่อยู่ในหน้า abstract ของ GPT-4 Technical Report
3. **venue ไม่ตรงกับหน้า arXiv:** หน้า arXiv ของ T5, CCNet, Ahia, Chatbot Arena, RAGAS และ CoT ไม่ระบุที่ตีพิมพ์ ส่วนสไลด์ระบุ JMLR, LREC 2020, EMNLP 2023, ICML 2024, EACL 2024 System Demonstrations และ NeurIPS 2022 ตามลำดับ ผมไม่ได้ตรวจ venue เหล่านั้นกับหน้าของ venue เอง
4. **ตัวเลขที่เป็นของสไลด์ ไม่ใช่ของ abstract:** "~20 token ต่อพารามิเตอร์" (Chinchilla), "ตัดออก 84%" (CCNet), กฎเกณฑ์ Gopher ทั้งชุด, 15,908/14,079 ข้อของ MMLU ไม่ปรากฏในหน้า abstract ที่ตรวจ จึงใช้ตัวเลขของสไลด์เป็นหลัก

---

## ข้อมูลและ scaling

### Training Compute-Optimal Large Language Models (Chinchilla)
- **ผู้เขียน:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford และคณะ
- **ปี / ที่ตีพิมพ์:** 2022 (arXiv v1, 29 มี.ค. 2022) หน้า arXiv ระบุเป็น preprint
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2203.15556
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 4 (อ้างในข้อความ "Hoffmann et al. 2022" ไม่มี footer)
- **ใจความ**
  - ฝึกโมเดลมากกว่า 400 ตัว ขนาดราว 70 ล้านถึงกว่า 16 พันล้านพารามิเตอร์ บน 5 ถึง 500 พันล้าน token
  - สรุปว่าโมเดลใหญ่ในยุคนั้น "ฝึกไม่พอ" (undertrained) เพราะเน้นขยายโมเดลโดยข้อมูลคงที่ ควรขยายขนาดโมเดลและจำนวน token ไปพร้อมกัน (โมเดลใหญ่ขึ้นสองเท่า ข้อมูลก็ควรเป็นสองเท่า)
  - ทดสอบด้วย Chinchilla 70B ที่ใช้คอมพิวต์เท่า Gopher (280B) แต่ข้อมูลมากกว่า 4 เท่า ได้ผลดีกว่า Gopher, GPT-3 (175B), Jurassic-1 (178B) และ Megatron-Turing NLG (530B)
  - MMLU เฉลี่ย 67.5% สูงกว่า Gopher เกิน 7 จุดเปอร์เซ็นต์ และโมเดลเล็กกว่าจึง fine-tune/inference ถูกกว่า
- **เชื่อมกับข้อสอบ:** หัวข้อ 2 และ 37 ข้อ 2 (D ≈ 20 × N, N ≈ D / 20, GPT-3 ≈ 1.71 token/พารามิเตอร์) ตัวเลข 20 token/พารามิเตอร์ และสูตร L = E + A/N^α + B/D^β มาจากสไลด์ (หน้า abstract ไม่ระบุสองอย่างนี้)

### Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models
- **ผู้เขียน:** Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R. Mortensen, Noah A. Smith และคณะ
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 23 พ.ค. 2023) ตามสไลด์: EMNLP 2023 (หน้า arXiv ไม่ระบุ)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2305.13707
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 9
- **ใจความ**
  - โมเดลภาษาถูกขายเป็น API ที่คิดเงินตามจำนวน token แต่จำนวน token ที่ต้องใช้เพื่อแสดงเนื้อหาเดียวกันต่างกันตามภาษา
  - ทดสอบกับ API ของ OpenAI ด้วย benchmark หลายภาษา 22 ภาษาที่หลากหลายทางภาษาศาสตร์
  - พบว่าผู้ใช้ภาษาหลายภาษาจ่ายแพงกว่าแต่ได้ผลแย่กว่า และมักอยู่ในภูมิภาคที่เข้าถึง API ได้ยากกว่าอยู่แล้ว
  - เสนอให้ผู้ให้บริการเปิดเผยราคาและทำให้ราคาเป็นธรรมขึ้น (หน้า abstract ไม่ให้ตัวเลขค่าใช้จ่ายเฉพาะ)
- **เชื่อมกับข้อสอบ:** หัวข้อ 6 (TPC, ρ, ต้นทุนต่อเอกสารเทียบ ρ) และ 37 ข้อ 7 ข้อสอบใช้ตัวเลขตัวอย่างของเดโม (71/10, 57/114) ไม่ได้ใช้ตัวเลขของเปเปอร์

## Corpus และตัวกรอง

### The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
- **ผู้เขียน:** Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel และคณะ (รวม 8 คน)
- **ปี / ที่ตีพิมพ์:** 2024 (arXiv v1, 25 มิ.ย. 2024; v2 31 ต.ค. 2024) หน้า arXiv ระบุเป็น preprint
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2406.17557
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 10, 16, 17
- **ใจความ**
  - FineWeb: ชุดข้อมูล pretraining ราว 15 ล้านล้าน token จาก 96 snapshot ของ Common Crawl
  - บันทึกและทำ ablation การตัดสินใจเชิงออกแบบ รวมถึงวิธี deduplicate และกรอง
  - FineWeb-Edu: เซตย่อยราว 1.3 ล้านล้าน token ของข้อความเชิงการศึกษา ทำให้โมเดลได้ผลดีขึ้นมากบนงานที่ใช้ความรู้และเหตุผล เช่น MMLU และ ARC
  - เผยแพร่ชุดข้อมูล โค้ดสร้างข้อมูล และโมเดลจากการทดลอง ablation
- **เชื่อมกับข้อสอบ:** หัวข้อ 7, 12, 13, 35 และ 37 (15T, 96 snapshot, 1.3T ≈ 9%, MMLU 33 → 37, ARC 46 → 57 ตัวเลข MMLU/ARC อยู่ในสไลด์ ไม่ได้อยู่ในหน้า abstract)

### FineWeb-Edu: The finest collection of educational content (Lozhkov et al., การ์ดชุดข้อมูลบน HuggingFace)
- **ผู้เขียน:** Lozhkov et al. (ตามสไลด์; หน้าที่เปิดไม่แสดงรายชื่อ)
- **ปี / ที่ตีพิมพ์:** พ.ค. 2024 ตามสไลด์ เป็นการ์ดชุดข้อมูล ไม่ใช่เปเปอร์
- **ลิงก์ (ตรวจบางส่วน):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu หน้านี้ที่เปิดได้แสดงเฉพาะตัวดูข้อมูล (ฟิลด์ `int_score` ค่า 3-5 สอดคล้องกับเกณฑ์ ≥ 3) ไม่แสดงข้อความของการ์ด จึงตรวจรายละเอียดจากการ์ดของ classifier: https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 16, 17
- **ใจความ (จากการ์ด classifier)**
  - ฝึกด้วย **450k** annotations ที่ Llama3-70B-instruct ให้คะแนน (สไลด์เขียน ~460k)
  - ตัว classifier ใช้ Snowflake-arctic-embed-m เป็นฐาน ต่อหัวถดถอยเอาต์พุตเดียว (a classification head with a single regression output) ตอนฝึกตรึงเลเยอร์ embedding และ encoder
  - เกณฑ์ที่ใช้ในชุดข้อมูล: คะแนน ≥ 3 (ตรวจจากฟิลด์ `int_score` และสไลด์)
- **เชื่อมกับข้อสอบ:** หัวข้อ 12 และ 35 ตัวเลขที่สไลด์ให้คือ 460k/~110M/≥ 3/1.3T ตอบตามสไลด์ แต่ควรรู้ว่าการ์ดต้นทางว่า 450k (ดูผลตรวจข้อ 1)

### Scaling Language Models: Methods, Analysis & Insights from Training Gopher
- **ผู้เขียน:** Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song และคณะ (80 คน)
- **ปี / ที่ตีพิมพ์:** 2021 (arXiv v1, 8 ธ.ค. 2021; v2 21 ม.ค. 2022) หน้า arXiv ระบุเป็น preprint 120 หน้า
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2112.11446
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 11, 13
- **ใจความ**
  - ศึกษาโมเดล Transformer ขนาดตั้งแต่หลักสิบล้านถึง Gopher 280B พารามิเตอร์
  - ทดสอบ 152 งาน Gopher ให้ผลดีสุดในงานส่วนใหญ่
  - การขยายขนาดช่วยมากในงานอ่านจับใจความ ตรวจข้อเท็จจริง และระบุข้อความเป็นพิษ แต่ช่วยน้อยในการให้เหตุผลเชิงตรรกะและคณิตศาสตร์
  - วิเคราะห์ข้อมูลฝึกและพฤติกรรมโมเดล รวมถึงอคติและความเป็นพิษ
- **เชื่อมกับข้อสอบ:** หัวข้อ 8 กฎ Gopher (50-100,000 คำ, alpha ≥ 0.8, symbol ≤ 0.1, stop-word ≥ 2, repetition) และหัวข้อ 34 (Gopher ตัด near-duplicate ที่ J > 0.8 บน shingle 13 คำ) รายละเอียดกฎอยู่ในภาคผนวก (เดโมระบุ Appendix A) ซึ่งหน้า abstract ไม่ครอบคลุม จึงยึดสไลด์

### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5 และ C4)
- **ผู้เขียน:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena และคณะ
- **ปี / ที่ตีพิมพ์:** 2019 (arXiv v1, 23 ต.ค. 2019; v4 19 ก.ย. 2023) ตามสไลด์: JMLR 21(140) (หน้า arXiv ไม่ระบุ)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1910.10683
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 12 (และสัปดาห์ 8)
- **ใจความ**
  - ศึกษา transfer learning ใน NLP: pre-train บนงานที่ข้อมูลมาก แล้ว fine-tune
  - เสนอกรอบที่แปลงทุกงานภาษาให้เป็น text-to-text
  - เทียบ objective, สถาปัตยกรรม, ชุดข้อมูลไม่มีป้าย และวิธี transfer อย่างเป็นระบบ
  - สร้างชุดข้อมูล Colossal Clean Crawled Corpus (C4) และได้ผลระดับ state-of-the-art ในงานสรุปความ ถามตอบ และจัดประเภท
  - เผยแพร่ชุดข้อมูล โมเดล และโค้ด
- **เชื่อมกับข้อสอบ:** หัวข้อ 9 กฎ C4 (§2.2 ตามสไลด์: ลงท้ายเครื่องหมายจบ, ≥ 3 คำ/บรรทัด, ≥ 5 ประโยค/หน้า, ไม่มี `{`, lorem ipsum ฯลฯ) รายละเอียดกฎ §2.2 ไม่อยู่ในหน้า abstract

### CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data
- **ผู้เขียน:** Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin และคณะ
- **ปี / ที่ตีพิมพ์:** 2019 (arXiv v1, 1 พ.ย. 2019) ตามสไลด์: LREC 2020 (หน้า arXiv ไม่ระบุ)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1911.00359
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 14
- **ใจความ**
  - ท่อประมวลผลอัตโนมัติที่สกัดชุดข้อมูลภาษาเดียวคุณภาพสูงหลายภาษาจาก Common Crawl
  - ทำตามแนว fastText: dedup เอกสาร และระบุภาษา
  - เพิ่มขั้นกรองที่เก็บเอกสารที่คล้ายคลังคุณภาพสูงอย่าง Wikipedia
  - เหตุผล: คลังข้อมูลใหญ่ช่วยโมเดลแทนข้อความ ตราบที่คุณภาพคงอยู่
- **เชื่อมกับข้อสอบ:** หัวข้อ 11 (KenLM 5-gram, 3 ถัง head / middle / tail, ไม่มี cut-off ตายตัว, PPL trap) ตัวเลข "ตัดออก 84%" เป็นของสไลด์ ไม่ใช่ของหน้า abstract

### Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets
- **ผู้เขียน:** Julia Kreutzer, Isaac Caswell, Lisa Wang, Ahsan Wahab, Daan van Esch, Nasanbayar Ulzii-Orshikh และคณะ (52 คน)
- **ปี / ที่ตีพิมพ์:** 2021 (arXiv v1, 22 มี.ค. 2021); TACL 10 (2022) หน้า 50-72
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2103.12028
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 15 (อ้างเป็นหลักฐานการตรวจสอบ)
- **ใจความ**
  - ตรวจด้วยมือ 205 corpus เฉพาะภาษา จากชุดข้อมูลที่ขุดจากเว็บ 5 ชุด (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4)
  - พบปัญหาคุณภาพเป็นระบบใน corpus ภาษาทรัพยากรน้อย อย่างน้อย 15 corpus ไม่มีข้อความที่ใช้ได้เลย
  - หลาย corpus ติดป้ายภาษาผิดหรือใช้รหัสภาษาไม่ชัด
  - ผู้ตรวจที่ไม่พูดภาษานั้นก็จับปัญหาได้ และเสนอแนวทางประเมินกับปรับปรุง
- **เชื่อมกับข้อสอบ:** หัวข้อ 11 (ใช้เป็นหลักฐานว่าตัวกรอง/ป้ายภาษามีข้อผิดพลาด) โอกาสออกเป็นข้อสอบโดยตรงต่ำ

### Whose Language Counts as High Quality? Measuring Language Ideologies in Text Data Selection
- **ผู้เขียน:** Suchin Gururangan, Dallas Card, Sarah K. Dreier, Emily K. Gade, Leroy Z. Wang, Zeyu Wang และคณะ
- **ปี / ที่ตีพิมพ์:** 2022 (arXiv v1, 25 ม.ค. 2022); EMNLP 2022 (หน้า 2562-2580 ตามฐานข้อมูล ACL Anthology ที่ปรากฏในผลค้นหา)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2201.10474
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 15
- **ใจความ**
  - วิเคราะห์ตัวกรองคุณภาพที่ใช้กับ GPT-3 ซึ่งยึด Wikipedia/หนังสือ/ข่าวเป็นข้อความที่ "ดี"
  - ใช้ชุดบทความหนังสือพิมพ์ของโรงเรียนมัธยมในสหรัฐฯ พบว่ากรองให้ผ่านคะแนนสูงกับโรงเรียนใหญ่ในพื้นที่ที่ร่ำรวย การศึกษาสูง และเป็นเมืองมากกว่า
  - คำนิยามของ "คุณภาพ" ของตัวกรองไม่สอดคล้องกับเกณฑ์อื่นที่สมเหตุสมผล เช่น ความถูกต้องของข้อเท็จจริงหรือคุณค่าทางวรรณกรรม
  - สรุปว่าการเลือกคลังใดเป็น "คุณภาพสูง" ฝังอุดมการณ์ทางภาษาไว้ จึงต้องโปร่งใสและให้เหตุผลในการเลือกข้อมูล
- **เชื่อมกับข้อสอบ:** หัวข้อ 11 (classifier ที่ฝึกให้ชอบ Wikipedia/หนังสือ/ข่าวเอนเอียง) ข้อสอบน่าจะถามเป็นแนวคิด (bias ของตัวกรองคุณภาพ) มากกว่าตัวเลข

### Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- **ผู้เขียน:** Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur และคณะ (รวม 41 คน)
- **ปี / ที่ตีพิมพ์:** 2024 (arXiv v1, 31 ม.ค. 2024; v2 6 มิ.ย. 2024); ACL 2024
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2402.00159
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 18, 19
- **ใจความ**
  - ปัญหา: หลายโมเดลแม้แบบเปิดไม่เปิดเผยข้อมูลฝึก ทำให้ศึกษาผลของข้อมูลต่อความสามารถได้ยาก
  - Dolma: คลังภาษาอังกฤษ 3 ล้านล้าน token จากเว็บ บทความวิชาการ โค้ด หนังสือสาธารณสมบัติ โซเชียลมีเดีย และสารานุกรม
  - บันทึกหลักการออกแบบ วิธีสร้าง และเนื้อหาของคลัง พร้อมการวิเคราะห์ในเวอร์ชันกลาง
  - เปิดซอร์ส toolkit สำหรับสร้างคลัง เพื่อให้ผู้อื่นทำซ้ำและต่อยอดได้
- **เชื่อมกับข้อสอบ:** หัวข้อ 13 และ 35 (≈ 3T, AI2, หลายแหล่ง, toolkit `dolma`, log ต่อเอกสาร) ตัวเลขตารางแหล่งข้อมูล (2,479 B ฯลฯ) มาจากสไลด์/Table 1 ซึ่งไม่ได้อยู่ในหน้า abstract

## Deduplication

### Deduplicating Training Data Makes Language Models Better
- **ผู้เขียน:** Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch และคณะ
- **ปี / ที่ตีพิมพ์:** 2021 (arXiv v1, 14 ก.ค. 2021; v2 24 มี.ค. 2022); ACL 2022
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2107.06499
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 20, 21
- **ใจความ**
  - ชุดข้อมูลสร้างโมเดลภาษาทั่วไปมีตัวอย่างซ้ำใกล้เคียงและสตริงย่อยซ้ำยาวมาก
  - มากกว่า 1% ของเอาต์พุตแบบไม่มี prompt จากโมเดลที่ฝึกบนข้อมูลเหล่านั้นคัดลอกจากข้อมูลฝึกคำต่อคำ
  - สร้างเครื่องมือ dedup สองแบบ (ตัวอย่าง: ประโยค 61 คำที่ซ้ำใน C4 มากกว่า 60,000 ครั้ง)
  - โมเดลที่ฝึกบนข้อมูล dedup แล้ว ปล่อยข้อความที่จำได้น้อยลงราว 10 เท่า และใช้ขั้นตอนฝึกน้อยลงเพื่อได้ความแม่นเท่าเดิมหรือดีกว่า
  - การซ้อนทับระหว่าง train-test กระทบมากกว่า 4% ของชุด validation ของข้อมูลมาตรฐาน dedup ช่วยให้ประเมินแม่นขึ้น
- **เชื่อมกับข้อสอบ:** หัวข้อ 14 (ความเสี่ยง: memorization, bias, compute; exact vs near dedup) และหัวข้อ 34 (นโยบาย "เก็บอันไหน": Lee et al. เก็บสำเนาใน test/validation)

### Detecting Near-Duplicates for Web Crawling (Manku, Jain, Das Sarma)
- **ผู้เขียน:** Gurmeet Singh Manku, Arvind Jain, Anish Das Sarma
- **ปี / ที่ตีพิมพ์:** 2007; Proceedings of the 16th International Conference on World Wide Web (WWW 2007) หน้า 141-150
- **ลิงก์ (ตรวจบางส่วน):** https://research.google/pubs/detecting-near-duplicates-for-web-crawling/ หน้านี้ยืนยันชื่อ ผู้เขียน venue ปี และหน้า แต่ไม่แสดงข้อความ abstract
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 25
- **ใจความ (จากสไลด์ ไม่ได้ตรวจจากหน้านั้น)**
  - ใช้ fingerprint 64 บิตแบบ SimHash กับ 8 พันล้านหน้าของ crawl ของ Google
  - ถือคู่ที่ Hamming distance ≤ 3 เป็นใกล้ซ้ำ สไลด์ย้ำว่าเป็น "จุดทำงาน" ของผู้เขียน ไม่ใช่นิยาม
- **เชื่อมกับข้อสอบ:** หัวข้อ 16 และ 35 (ตัวเลข 64 บิต, d_H ≤ 3, 8 พันล้านหน้า ยังตรวจกับเนื้อเปเปอร์ไม่ได้)

### Similarity Estimation Techniques from Rounding Algorithms (Charikar)
- **ผู้เขียน:** Moses S. Charikar
- **ปี / ที่ตีพิมพ์:** 2002; Proceedings of the 34th Annual ACM Symposium on Theory of Computing (STOC 2002) หน้า 380-388
- **ลิงก์ (ตรวจจากแหล่งรอง):** DOI 10.1145/509907.509965 (หน้า ACM Digital Library ตอบ HTTP 403 เปิดไม่ได้) ยืนยันชื่อ ผู้เขียน venue ปี หน้า และ DOI จากระเบียน BibTeX: https://bibtex.github.io/STOC-2002-Charikar.html (ตรวจแล้ว)
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 25
- **ใจความ (ตามผลค้นหาและสไลด์ ไม่ได้อ่านข้อความเต็ม)**
  - มองอัลกอริทึม rounding ของ LP/SDP เป็นโครงสร้าง locality-sensitive hashing
  - เทียบกับ min-wise independent permutations ซึ่งให้โครงสร้าง LSH สำหรับ Jaccard
  - สไลด์ 25 ใช้เปเปอร์นี้ที่มาของความสัมพันธ์ E[d_H / d] = arccos(cos) / π ของ SimHash แบบ hyperplane สุ่ม (ข้อความนี้ผมไม่ได้ยืนยันจากหน้าเปเปอร์)
- **เชื่อมกับข้อสอบ:** หัวข้อ 16 (SimHash → cosine ผ่าน Hamming distance) ข้อสอบน่าจะถามเพียงว่า SimHash รักษา cosine และ MinHash ประมาณ Jaccard

### Mining of Massive Datasets, บทที่ 3 (Leskovec, Rajaraman, Ullman)
- **ผู้เขียน:** Jure Leskovec, Anand Rajaraman, Jeffrey D. Ullman
- **ปี / ที่ตีพิมพ์:** 2014 (ตามสไลด์) หนังสือ Cambridge University Press บทที่ 3 "Finding Similar Items"
- **ลิงก์ (ตรวจจากแหล่งรอง):** สไลด์ให้ www.mmds.org ซึ่งเปิดไม่สำเร็จ (ใบรับรองของเซิร์ฟเวอร์ไม่ตรงชื่อโฮสต์) ตรวจหัวข้อบทจากรายการหนังสือของ Cambridge ที่พบจากการค้นหา: https://www.cambridge.org/core/books/mining-of-massive-datasets/finding-similar-items/FDDA225F039792324F28D73D26950E89
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 26
- **ใจความ (จากคำอธิบายบทที่พบ)**
  - แปลงปัญหาความคล้ายของเอกสารให้เป็นปัญหาเซตด้วย shingling
  - minhashing บีบเซตเป็น signature ที่ยังประมาณ Jaccard ได้
  - locality-sensitive hashing โฟกัสการค้นหาไปที่คู่ที่น่าจะคล้ายกัน ไม่ต้องเทียบทุกคู่
- **เชื่อมกับข้อสอบ:** หัวข้อ 15-18 (shingling → MinHash → LSH banding) สูตร 1 − (1 − s^r)^b ในบทเรียนมาจากสไลด์และเดโม ไม่ได้ตรวจกับตำราเล่มนี้

## การประเมินผล

### A Survey on Evaluation of Large Language Models
- **ผู้เขียน:** Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu และคณะ
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 6 ก.ค. 2023) ACM Transactions on Intelligent Systems and Technology (TIST)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2307.03109
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 32
- **ใจความ**
  - จัดระเบียบงานประเมิน LLM ด้วยสามคำถาม: ประเมินอะไร ประเมินที่ไหน ประเมินอย่างไร
  - ครอบคลุมงาน NLP ทั่วไป การให้เหตุผล การแพทย์ จริยธรรม การศึกษา วิทยาศาสตร์ และ agent
  - สรุปจุดที่ LLM ทำได้และทำไม่ได้ พร้อมความท้าทายที่ยังเปิดอยู่
  - ข้อเสนอหลัก: การประเมินควรถือเป็นวินัยสำคัญที่หนุนการพัฒนา LLM
- **เชื่อมกับข้อสอบ:** หัวข้อ 21 (3 era ของการประเมิน) การแบ่ง Era 1-3 เป็นของสไลด์ หน้า abstract ไม่ได้แบ่งแบบนั้น

### BERTScore: Evaluating Text Generation with BERT
- **ผู้เขียน:** Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi
- **ปี / ที่ตีพิมพ์:** 2019 (arXiv v1, 21 เม.ย. 2019); ICLR 2020 (หน้า arXiv ระบุ "To appear in ICLR2020")
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1904.09675
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 34, 35
- **ใจความ**
  - เมตริกประเมินการสร้างข้อความที่เทียบ candidate กับ reference ทีละโทเค็นด้วย contextual embedding แทนการจับคำตรงตัว
  - ทดสอบกับเอาต์พุตของระบบแปลและบรรยายภาพ 363 ระบบ
  - ตรงกับการตัดสินของมนุษย์มากกว่าเมตริกเดิม และเลือกโมเดลที่ดีที่สุดได้ดีกว่า
  - ทนต่อตัวอย่างยากในงานตรวจ paraphrase แบบปรปักษ์ได้ดีกว่าเมตริกเดิม
- **เชื่อมกับข้อสอบ:** หัวข้อ 23 และ 37 ข้อ 10 (P = เฉลี่ย max ของแถว candidate, R = เฉลี่ย max ของคอลัมน์ reference, F = 2PR/(P+R), IDF) สูตรเหล่านี้มาจากสไลด์ 35

### Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference
- **ผู้เขียน:** Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li และคณะ
- **ปี / ที่ตีพิมพ์:** 2024 (arXiv v1, 7 มี.ค. 2024) ตามสไลด์: ICML 2024 (PMLR) หน้า arXiv ไม่ระบุ venue
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2403.04132
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 37
- **ใจความ**
  - แพลตฟอร์มเปิดที่จัดอันดับ LLM จากคะแนนโหวตความชอบของมนุษย์ แบบเปรียบเทียบคู่ โดยผู้ใช้จำนวนมาก
  - รวบรวมโหวตมากกว่า 240K ครั้งในหลายเดือน
  - ใช้วิธีสถิติเพื่อจัดอันดับให้มีประสิทธิภาพและแม่นยำ (หน้า abstract ไม่ระบุแบบจำลองที่ใช้ สไลด์บอก Elo / Bradley-Terry)
  - ผู้เขียนอ้างว่าคำถามจากฝูงชนหลากหลายและแยกแยะโมเดลได้ และโหวตสอดคล้องกับผู้ประเมินผู้เชี่ยวชาญ
- **เชื่อมกับข้อสอบ:** หัวข้อ 24 และตัวอย่าง Elo (นอกสไลด์) ก่อนหัวข้อ 25 ข้อสอบน่าจะถามจุดแข็ง/จุดอ่อน ไม่ใช่สูตร Elo

### Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **ผู้เขียน:** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang และคณะ
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 9 มิ.ย. 2023); NeurIPS 2023 Datasets and Benchmarks Track
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2306.05685
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 38-41
- **ใจความ**
  - ศึกษาการใช้ LLM แข็งแรง เช่น GPT-4 เป็นผู้ตัดสินคำตอบของแชตบอตต่อคำถามปลายเปิด
  - รายงานว่าผู้ตัดสินสอดคล้องกับความชอบของมนุษย์ทั้งแบบควบคุมและแบบฝูงชนเกิน 80% ใกล้เคียงอัตราที่มนุษย์เห็นตรงกันเอง
  - ระบุอคติ 3 แบบ: position bias, verbosity bias, self-enhancement bias พร้อมข้อจำกัดด้านการให้เหตุผล และเสนอวิธีลดบางแบบ
  - เสนอ MT-Bench (คำถามหลายเทิร์น) และ Chatbot Arena; ปล่อยคำถาม MT-Bench โหวตผู้เชี่ยวชาญ 3K และบทสนทนาที่มนุษย์ชอบ 30K
- **เชื่อมกับข้อสอบ:** หัวข้อ 25, 26, 37 ข้อ 12 (swap trick, length/positional bias) ตัวเลขใน Table 2 และ Table 3 (23.8/46.2/65.0 และ 91.3/91.3/8.7) อยู่ในสไลด์ ไม่ได้อยู่ในหน้า abstract

### Measuring Massive Multitask Language Understanding (MMLU)
- **ผู้เขียน:** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song และคณะ
- **ปี / ที่ตีพิมพ์:** 2020 (arXiv v1, 7 ก.ย. 2020); ICLR 2021
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2009.03300
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 42
- **ใจความ**
  - ชุดทดสอบ 57 วิชา เช่น คณิตศาสตร์ระดับประถม ประวัติศาสตร์สหรัฐฯ วิทยาการคอมพิวเตอร์ และกฎหมาย ต้องใช้ความรู้โลกกว้างและการแก้ปัญหา
  - โมเดลส่วนใหญ่ที่ทดสอบใกล้เคียงการเดาสุ่ม GPT-3 ตัวใหญ่สุดดีกว่าเดาสุ่มราว 20 จุดเปอร์เซ็นต์
  - โมเดลที่แรงที่สุดยังต่ำกว่าระดับผู้เชี่ยวชาญในทุกวิชา และความสามารถไม่สม่ำเสมอข้ามวิชา
  - ความแม่นยำใกล้เดาสุ่มในหัวข้อที่สำคัญต่อสังคม เช่น จริยธรรมและกฎหมาย (ณ เวลาที่เขียน)
- **เชื่อมกับข้อสอบ:** หัวข้อ 27 และ 35 (57 วิชา, 15,908 ข้อ, test 14,079, 5-shot) จำนวนข้อและ 5-shot เป็นของสไลด์ ไม่ใช่ของหน้า abstract

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **ผู้เขียน:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia และคณะ
- **ปี / ที่ตีพิมพ์:** 2022 (arXiv v1, 28 ม.ค. 2022) ตามสไลด์: NeurIPS 2022 (หน้า arXiv ไม่ระบุ)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2201.11903
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 43 (และสัปดาห์ 11)
- **ใจความ**
  - เสนอ chain-of-thought prompting: ใส่ตัวอย่างที่แสดงขั้นการให้เหตุผลระหว่างทางไว้ใน prompt
  - ให้ผลดีขึ้นในงานเลขคณิต สามัญสำนึก และสัญลักษณ์ ในโมเดลขนาดใหญ่พอ
  - ผลเด่น: โมเดล 540B พารามิเตอร์ที่ใช้ 8 ตัวอย่าง ได้ผลระดับ state-of-the-art บน GSM8K เหนือ GPT-3 ที่ fine-tune พร้อม verifier
- **เชื่อมกับข้อสอบ:** หัวข้อ 27 (ตาราง GSM8K: GPT-3 175B 15.6% → 46.9% ด้วย CoT) ตัวเลขในตารางมาจากสไลด์/เนื้อเปเปอร์ ไม่ใช่หน้า abstract

### GPT-4 Technical Report
- **ผู้เขียน:** OpenAI (รายชื่อผู้ร่วมงานจำนวนมาก)
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 15 มี.ค. 2023; v6 4 มี.ค. 2024) หน้า arXiv ระบุเป็น preprint
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2303.08774
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 43
- **ใจความ**
  - GPT-4 เป็นโมเดลมัลติโมดัลขนาดใหญ่ รับภาพและข้อความ ให้ข้อความ
  - ในบริบทจริงหลายแบบด้อยกว่ามนุษย์ แต่ถึงระดับมนุษย์บน benchmark วิชาชีพและวิชาการหลายชุด เช่น สอบเนติฯ จำลองได้ราวผู้สอบ 10% บนสุด
  - ฝึกด้วยการทำนาย token ถัดไป และปรับหลังฝึก (post-training alignment) เพื่อเพิ่มความถูกต้องของข้อเท็จจริงและการทำตามพฤติกรรมที่ต้องการ
  - สร้างโครงสร้างพื้นฐานที่ขยายแบบคาดเดาได้ ทำนายสมรรถนะบางด้านของ GPT-4 จากโมเดลที่ใช้คอมพิวต์ไม่เกิน 1/1,000
- **เชื่อมกับข้อสอบ:** หัวข้อ 27 ตาราง GSM8K แถว GPT-4 (92.0%, 5-shot CoT) ตัวเลข 92.0% ไม่ได้อยู่ในหน้า abstract จึงยืนยันจากหน้านั้นไม่ได้ ตอบตามสไลด์

### Introducing Claude 3.5 Sonnet (Anthropic, ประกาศ)
- **ผู้เขียน:** Anthropic
- **ปี / ที่ตีพิมพ์:** 21 มิ.ย. 2024 (หน้าประกาศของบริษัท ไม่ใช่เปเปอร์)
- **ลิงก์ (ตรวจบางส่วน):** https://www.anthropic.com/news/claude-3-5-sonnet หน้าเปิดได้ ยืนยันชื่อหน้าและวันที่ แต่ **ไม่พบตัวเลข 96.4% (GSM8K) และ 92.0% (HumanEval)** ในข้อความของหน้า (ตัวเลขน่าจะอยู่ในรูปแผนภูมิที่ข้อความไม่ได้แสดง) จึงยืนยันตัวเลขของสไลด์ไม่ได้
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 43
- **ใจความ**
  - หน้าประกาศบอกว่าโมเดลทำสถิติใหม่ของอุตสาหกรรมบน benchmark หลายชุดรวมถึง HumanEval (ตามผลที่หน้านี้แสดง ไม่ให้คะแนน)
- **เชื่อมกับข้อสอบ:** หัวข้อ 27 ตาราง GSM8K (96.4%) และตัวเลข pass@1 ของ HumanEval (92.0%) ตอบตามสไลด์ ข้อควรระวังในบทเรียน: 92.0% ของ GPT-4 ใช้กับ GSM8K ส่วน 92.0% ของ Claude 3.5 Sonnet ใช้กับ HumanEval

### Evaluating Large Language Models Trained on Code (Codex และ HumanEval)
- **ผู้เขียน:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan และคณะ
- **ปี / ที่ตีพิมพ์:** 2021 (arXiv v1, 7 ก.ค. 2021) หน้า arXiv ระบุเป็น preprint
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2107.03374
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 44
- **ใจความ**
  - Codex: โมเดล GPT ที่ fine-tune บนโค้ด GitHub สาธารณะ ใช้เบื้องหลัง GitHub Copilot
  - บน HumanEval (สร้างโปรแกรมจาก docstring) Codex แก้ได้ **28.8%** เทียบ GPT-3 0% และ GPT-J 11.4%
  - การสุ่มซ้ำได้ผลดี: 100 ตัวอย่างต่อโจทย์ แก้ได้ **70.2%**
  - ข้อจำกัด: ลำบากเมื่อ docstring อธิบายลำดับการดำเนินการยาว และการผูกค่ากับตัวแปร
- **เชื่อมกับข้อสอบ:** หัวข้อ 27 (pass@k, 164 ข้อ, pass@1 ของ Codex-12B 28.8% ตรงกับหน้า abstract) สูตร pass@k = 1 − C(n−c, k)/C(n, k) มาจากสไลด์ 44 และเนื้อเปเปอร์ ไม่ได้อยู่ในหน้า abstract

### Training Verifiers to Solve Math Word Problems (GSM8K)
- **ผู้เขียน:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser และคณะ
- **ปี / ที่ตีพิมพ์:** 2021 (arXiv v1, 27 ต.ค. 2021) หน้า arXiv ระบุเป็น preprint
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2110.14168
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 43 (สไลด์แสดงชุดข้อมูล GSM8K แต่ **ไม่มี citation ของเปเปอร์นี้**)
- **ใจความ**
  - แนะนำ GSM8K: โจทย์เลขปัญหาคำระดับประถม 8.5K ข้อ
  - โมเดล transformer ขนาดใหญ่สุดยังทำได้ไม่ดี ทั้งที่โจทย์ไม่ซับซ้อนเชิงแนวคิด
  - เสนอฝึก verifier ตัดสินว่าคำตอบที่โมเดลสร้างถูกหรือไม่ แล้วสร้างหลายคำตอบและเลือกอันที่ verifier ให้คะแนนสูงสุด
  - การตรวจสอบช่วยเพิ่มผลบน GSM8K ได้มากและขยายตามข้อมูลได้ดีกว่า fine-tune ธรรมดา
- **เชื่อมกับข้อสอบ:** หัวข้อ 27 (8,500 ข้อ, ดึงคำตอบหลัง `####` อยู่ในสไลด์ ไม่ได้ตรวจกับหน้า abstract)

### Time Travel in LLMs: Tracing Data Contamination in Large Language Models
- **ผู้เขียน:** Shahriar Golchin, Mihai Surdeanu
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 16 ส.ค. 2023); ICLR 2024 (Spotlight)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2308.08493
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 45
- **ใจความ**
  - เสนอวิธีตรวจว่าข้อมูลทดสอบรั่วเข้าข้อมูลฝึกของ LLM ซึ่งบิดเบือนการประเมิน
  - ขั้นแรกให้โมเดลเติมตัวอย่างที่ตัดท้าย ถ้าเอาต์พุตตรงกับส่วนที่ซ่อนมาก จะติดธงว่ารั่ว
  - ขั้นสองตัดสินทั้งพาร์ติชัน: เทียบคะแนน overlap (ROUGE-L หรือ BLEURT) กับ baseline หรือใช้ตัวจำแนก GPT-4 แบบ few-shot
  - วิธีที่ดีสุดแม่น 92-100% เทียบกับผู้เชี่ยวชาญใน 7 ชุดข้อมูล และรายงานว่า GPT-4 ดูเหมือนปนเปื้อนกับ AG News, WNLI และ XSum
- **เชื่อมกับข้อสอบ:** หัวข้อ 28 (contamination, การตรวจ overlap, บริบท "perplexity ต่ำเป็นเบาะแสไม่ใช่หลักฐาน") ข้อสอบน่าจะถามแนวคิดและการคำนวณขอบเขต (หัวข้อ 28, 37 ข้อ 14)

### LiveBench: A Challenging, Contamination-Limited LLM Benchmark
- **ผู้เขียน:** Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain และคณะ (18 คน)
- **ปี / ที่ตีพิมพ์:** 2024 (arXiv v1, 27 มิ.ย. 2024; v2 18 เม.ย. 2025); ICLR 2025 Spotlight (ตามช่องหมายเหตุของ arXiv) สไลด์ 48 อ้าง "White et al. 2024" ไม่มี footer ชัด
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2406.19314 (สไลด์ให้เว็บไซต์ https://livebench.ai/)
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 46-48
- **ใจความ**
  - Benchmark ที่ออกแบบให้ทน contamination และอคติของผู้ตัดสิน LLM และการประเมินโดยฝูงชน
  - **รีเฟรชรายเดือน** มีงานใหม่และงานเวอร์ชันยากขึ้นเพิ่มตามเวลา
  - ใช้แหล่งใหม่ เช่น การแข่งขันคณิตศาสตร์ล่าสุด เปเปอร์ arXiv ข่าว และชุดข้อมูล
  - ตรวจคำตอบอัตโนมัติด้วยค่าเฉลย; **6 หมวด**: math, coding, reasoning, language, instruction following, data analysis
  - โมเดลชั้นนำทำได้ต่ำกว่า 70%
- **เชื่อมกับข้อสอบ:** หัวข้อ 29 (รายเดือน, 6 โดเมน, ลด ไม่ขจัด contamination) ตรงกับสไลด์ 48 ทั้งหมด

### SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- **ผู้เขียน:** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press และคณะ
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 10 ต.ค. 2023); ICLR 2024
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2310.06770
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 50
- **ใจความ**
  - กรอบประเมิน 2,294 ปัญหาวิศวกรรมซอฟต์แวร์จาก issue และ pull request จริงใน 12 repository Python ยอดนิยม
  - โมเดลต้องแก้โค้ดเพื่อแก้ issue ต้องประสานการเปลี่ยนแปลงหลายฟังก์ชัน คลาส และไฟล์ ใช้บริบทยาว และโต้ตอบกับสภาพแวดล้อมรันโค้ด
  - โมเดลชั้นนำตอนนั้นแก้ได้เฉพาะ issue ง่ายที่สุด Claude 2 ดีสุด แก้ได้ 1.96%
- **เชื่อมกับข้อสอบ:** หัวข้อ 29 (resolved = เทสต์ fail → pass ผ่านและ pass → pass ยังผ่าน, Verified/Lite เทียบกันตรง ๆ ไม่ได้) เกณฑ์ resolved และชุดย่อย Verified/Lite อยู่ในสไลด์ ไม่อยู่ในหน้า abstract ที่ตรวจ

### HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal
- **ผู้เขียน:** Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu และคณะ (12 คน)
- **ปี / ที่ตีพิมพ์:** 2024 (arXiv v1, 6 ก.พ. 2024; v2 27 ก.พ. 2024) หน้า arXiv ระบุเป็น preprint
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2402.04249
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 51 (สไลด์แสดง benchmark แต่ **ไม่มี citation ของเปเปอร์นี้**)
- **ใจความ**
  - ชี้ว่า red teaming อัตโนมัติของ LLM ไม่มีมาตรฐานการประเมิน จึงเสนอ HarmBench
  - เทียบวิธี red teaming **18 วิธี** กับโมเดลเป้าหมายและกลไกป้องกัน **33** ตัว (ตรงกับสไลด์ 51)
  - เสนอวิธี adversarial training ที่ช่วยเพิ่มความทนทานต่อการโจมตีหลายแบบ
  - โค้ดเปิดซอร์ส
- **เชื่อมกับข้อสอบ:** หัวข้อ 29 (ASR ต่ำ ≠ ปลอดภัย ผู้ตัดสินคือ classifier ไม่ใช่มนุษย์) ตัวเลข 510 พฤติกรรม (400 + 110), 7 หมวด และ classifier Llama-2-13B อยู่ในสไลด์ ไม่ได้อยู่ในหน้า abstract ที่ตรวจ

### Ragas: Automated Evaluation of Retrieval Augmented Generation
- **ผู้เขียน:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **ปี / ที่ตีพิมพ์:** 2023 (arXiv v1, 26 ก.ย. 2023; v2 28 เม.ย. 2025) ตามสไลด์: EACL 2024 System Demonstrations (หน้า arXiv ไม่ระบุ venue)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2309.15217
- **อยู่ในสไลด์:** สัปดาห์ 14 สไลด์ 52, 53
- **ใจความ**
  - กรอบประเมินระบบ RAG ที่ให้คะแนนส่วน retrieval กับส่วน generation ของ LLM แยกกัน
  - ไม่ต้องมีคำตอบอ้างอิงที่มนุษย์เขียน เพื่อให้รอบการประเมินเร็วขึ้น
  - สไลด์ 52: เปเปอร์นิยาม faithfulness, answer relevance, context relevance (reference-free); context precision/recall เป็นเมตริกที่ไลบรารีรุ่นหลังเพิ่ม
- **เชื่อมกับข้อสอบ:** หัวข้อ 30, 37 ข้อ 15 (faithfulness = claims ที่ context รองรับ / claims ทั้งหมด, context recall, context precision@K) สูตรทั้งสี่ในสไลด์ 53 ไม่ได้อยู่ในหน้า abstract

---

## แหล่งที่ไม่ใช่เปเปอร์ (สไลด์เอ่ยถึง)

| รายการ | สไลด์ | สถานะ |
|---|---|---|
| Common Crawl (https://commoncrawl.org/) | 5 | หน้า https://commoncrawl.org/overview ยืนยันว่ามีข้อมูลระดับ petabyte เก็บเป็นระยะตั้งแต่ปี 2008 และให้เข้าถึงฟรีบน Amazon; หน้านั้นไม่ระบุความถี่ crawl และชื่อรูปแบบ WARC/WET ในข้อความที่ได้ ตัวเลข "ทุก 1-2 เดือน" และ WARC/WET จึงอ้างตามสไลด์ |
| AIME | 49 | การแข่งขันคณิตศาสตร์ ไม่ใช่เปเปอร์ ไม่ได้ตรวจลิงก์ |
| fastText `lid.176.bin` | 7 | สไลด์ไม่ใส่ citation ไม่ได้ตรวจ |
| DeepEval | 52 | เฟรมเวิร์กโอเพนซอร์ส สไลด์ไม่ใส่ citation ไม่ได้ตรวจ |

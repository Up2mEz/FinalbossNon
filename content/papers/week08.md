เปเปอร์ที่สไลด์ DL-08 อ้างถึงหรือฉายหน้าเปเปอร์ขึ้นจอ จัดเป็น 3 กลุ่ม: (ก) เปเปอร์หลักที่มีบรรทัดอ้างอิงท้ายสไลด์ (ข) เปเปอร์ที่สไลด์ยกหน้ามาแสดงหรือเอ่ยชื่อเป็นหัวข้อ (ค) เปเปอร์ที่ปรากฏเพียงในข้อความของหน้าเปเปอร์ที่ฝังในสไลด์

ความหมายของ "ตรวจแล้ว": เปิดหน้า abstract บน arXiv แล้วเทียบชื่อเรื่อง ผู้เขียน และวันที่ส่งครั้งแรกตรงกัน ส่วนชื่อวารสารหรือการประชุมใช้ตามบรรทัดอ้างอิงในสไลด์ (หน้า abstract ของ arXiv ไม่ระบุ) ถ้าไม่ได้มาจากสไลด์จะบอกไว้ ไม่ได้ดาวน์โหลด PDF ใด ๆ

**กลุ่ม (ก): เปเปอร์หลักที่ท้ายสไลด์อ้างอิง**

### BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

- **ผู้เขียน:** Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer
- **ที่ตีพิมพ์:** ACL 2020 (บรรทัดอ้างอิงสไลด์ 9: Proceedings of the 58th annual meeting of the ACL, 2020) ส่ง arXiv ครั้งแรก 29 ตุลาคม 2019
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1910.13461
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 2, 5, 6, 7, 9, 13, 14 (หน้าเปเปอร์ฝังอยู่หลายหน้า) และสไลด์ 29 (อ้าง Lewis et al. 2019)

**ใจความ**

- BART เป็น denoising autoencoder สำหรับ pre-train โมเดล seq2seq: ทำให้ข้อความเสียด้วยฟังก์ชัน noise แล้วเรียนกู้ข้อความต้นฉบับ
- วิธี noising ที่ให้ผลดีที่สุดคือ **สลับประโยค (sentence shuffling)** และ **text infilling** ที่แทน span ด้วย mask ตัวเดียว (ตรงกับสูตรของ BART-large ในสไลด์ 13)
- เทียบเท่า RoBERTa บน GLUE และ SQuAD ด้วยทรัพยากรเทรนใกล้เคียงกัน
- ได้ผลดีที่สุดในขณะนั้นในงานบทสนทนาแบบสรุป, QA และการสรุปความ โดยเพิ่ม ROUGE ได้สูงสุด 6
- งานแปล: ใช้เฉพาะ pre-train ฝั่งภาษาเป้าหมายก็ได้ผลดีกว่าระบบ back-translation 1.1 BLEU (ตัวเลขที่สไลด์ 2 คัดมาจากเปเปอร์)
- ส่วนที่ฝังในสไลด์: ใช้ GeLU แทน ReLU, base 6+6 ชั้น และ large 12+12 ชั้น, Decoder ทุกชั้นทำ cross-attention ต่อ Encoder, BART มีพารามิเตอร์มากกว่า BERT ขนาดเทียบเท่าราว 10%

**เชื่อมกับข้อสอบ:** แหล่งของหัวข้อ "BART noising 5 แบบ ออกแน่นอน" ดูบทเรียน 3.1 ถึง 3.6 โดยเฉพาะตาราง ablation (3.5) ซึ่งเป็นตารางที่ 1 ของเปเปอร์นี้

### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

- **ผู้เขียน:** Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
- **ที่ตีพิมพ์:** Journal of Machine Learning Research 21(140): 1-67 ปี 2020 (ตามสไลด์ 15) ส่ง arXiv ครั้งแรก 23 ตุลาคม 2019
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1910.10683
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 15 และ 16 (ฝังหน้าเปเปอร์และรูปที่ 1)

**ใจความ**

- เสนอกรอบที่แปลงทุกปัญหาข้อความเป็นรูปแบบ **text-to-text** (รูปที่ 1 ในสไลด์: translate, cola, stsb, summarize ใช้โมเดล loss และ hyperparameter ชุดเดียว)
- เปรียบเทียบอย่างเป็นระบบหลายปัจจัยบนงานความเข้าใจภาษาหลายสิบงาน: objective ของ pre-training, สถาปัตยกรรม, ชุดข้อมูลไม่มีป้าย, วิธีถ่ายโอน
- สร้างชุดข้อมูล **C4 (Colossal Clean Crawled Corpus)** เป็นข้อความอังกฤษสะอาดหลายร้อย GB ที่ดึงมาจากเว็บ
- รวมผลการศึกษากับการขยายขนาดถึง 11 พันล้านพารามิเตอร์ ได้ผลระดับ state of the art หลายชุดทดสอบ (สรุป, QA, จัดประเภทข้อความ)
- เปิดเผยชุดข้อมูล โมเดลที่ pre-train แล้ว และโค้ด
- ตามสไลด์ 15: เป้าหมายของเปเปอร์ไม่ใช่เสนอวิธีใหม่ แต่ให้ภาพรวมของสาขา

**เชื่อมกับข้อสอบ:** แหล่งของ prefix, span corruption, sentinel, ไม่มี task head ดูบทเรียน 4.1 ถึง 4.6 และตัวอย่างคำนวณข้อ 1 ถึง 6

### What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?

- **ผู้เขียน:** Thomas Wang, Adam Roberts, Daniel Hesslow, Teven Le Scao, Hyung Won Chung, Iz Beltagy, Julien Launay, Colin Raffel
- **ที่ตีพิมพ์:** ICML 2022 (PMLR) ตามสไลด์ 35 ส่ง arXiv ครั้งแรก 12 เมษายน 2022
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2204.05832
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 35 (อ้างอิงท้ายหน้า) และสไลด์ 39 ("Wang et al. (2022)")

**ใจความ**

- ทดลองขนาดใหญ่เทียบ 3 สถาปัตยกรรม (decoder-only แบบ causal, decoder-only แบบ non-causal, encoder-decoder) กับ 2 objective (autoregressive และ masked LM) ทั้งแบบมีและไม่มี multitask prompted finetuning
- เทรนโมเดลที่มีพารามิเตอร์มากกว่า 5 พันล้านและข้อมูลมากกว่า 170 พันล้าน token
- หลัง pre-train แบบไม่มีป้ายอย่างเดียว **decoder-only แบบ causal ที่เทรนด้วย autoregressive LM ให้ zero-shot ดีที่สุด**
- เมื่อทำ multitask finetuning แล้ว decoder แบบ non-causal ที่เทรนด้วย masked LM ได้ผลรวมดีที่สุดในการทดลองของเปเปอร์
- แปลง decoder แบบ non-causal ให้เป็น causal ได้อย่างมีประสิทธิภาพ และกลับกันก็ได้
- ตัวเลขที่สไลด์ 39 ใช้ (Dec-only 4.8B เทียบกับ Enc-Dec 11B) มาจากเนื้อหาในเปเปอร์ ไม่ได้อยู่ใน abstract จึงไม่ได้ตรวจจากหน้าที่เปิด

**เชื่อมกับข้อสอบ:** แหล่งของ "หลักฐานมีเงื่อนไข ไม่ใช่ของฟรี" ดูบทเรียน 7.3 และตัวอย่างคำนวณข้อ 7 (5.5B เทียบกับ 4.8B)

**กลุ่ม (ข): เปเปอร์ที่สไลด์ยกหน้ามาแสดงหรือเอ่ยชื่อเป็นหัวข้อ**

### Attention Is All You Need

- **ผู้เขียน:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- **ที่ตีพิมพ์:** ปี 2017 ส่ง arXiv ครั้งแรก 12 มิถุนายน 2017 (หน้า arXiv ไม่ระบุการประชุม ที่ว่าลงใน NeurIPS 2017 เป็นความรู้ของผู้เขียน ไม่ได้ตรวจจากหน้าที่เปิด)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1706.03762
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 4 (ฝังรูปที่ 1 และหัวข้อ 3.1, 3.2) และสไลด์ 2 (กล่าวถึงในตาราง)

**ใจความ**

- เสนอ Transformer ที่สร้างจาก attention ล้วน ไม่มี recurrence และไม่มี convolution
- เป็น Encoder-Decoder อยู่แล้วตั้งแต่ต้น: Encoder ชั้นละ self-attention + FFN, Decoder เพิ่ม sub-layer ที่ทำ attention บนเอาต์พุตของ Encoder
- N = 6 ชั้นเหมือนกันต่อฝั่ง d_model = 512 มี residual connection และ layer normalization รอบทุก sub-layer
- ผลลัพธ์ WMT 2014 อังกฤษ-เยอรมัน 28.4 BLEU และอังกฤษ-ฝรั่งเศส 41.8 BLEU (single model) หลังเทรน 3.5 วันบน GPU 8 ตัว
- Decoder มี masking ร่วมกับการเลื่อนเอาต์พุตไป 1 ตำแหน่ง เพื่อให้การทำนายตำแหน่ง i พึ่งตำแหน่งที่น้อยกว่า i เท่านั้น

**เชื่อมกับข้อสอบ:** ลำดับ sublayer ของ decoder (Masked self-attn → Cross-attn → FFN) ดูบทเรียน 1.2 และสูตร `LayerNorm(x + Sublayer(x))` ที่เป็น Post-Norm ดูบทเรียน 1.4

### Multilingual Denoising Pre-training for Neural Machine Translation (mBART)

- **ผู้เขียน:** Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 22 มกราคม 2020 (หน้า arXiv ระบุ "Work in progress" ที่ว่าตีพิมพ์ใน TACL 2020 เป็นความรู้ของผู้เขียน ไม่ได้ตรวจ)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2001.08210
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 29 (ฝังรูปที่ 1 และหัวข้อ 2.2, 2.3 ของเปเปอร์)

**ใจความ**

- mBART เป็น seq2seq denoising autoencoder ที่ pre-train บนข้อความภาษาเดียวหลายภาษาขนาดใหญ่
- fine-tune ตรง ๆ สำหรับแปลแบบมีผู้สอนและไม่มีผู้สอนทั้งระดับประโยคและเอกสาร ไม่ต้องแก้ไขเฉพาะงาน
- รายงานดีขึ้นในเกือบทุกสถานการณ์ ยกเว้นภาษาทรัพยากรสูงสุด: สูงสุด 12 BLEU สำหรับภาษาทรัพยากรน้อย และเกิน 5 BLEU ในหลายโมเดลระดับเอกสารและแบบไม่มีผู้สอน
- ตามหน้าที่ฝังในสไลด์: noise สองแบบ คือ mask span 35% ของคำ (Poisson λ = 3.5) และสลับลำดับประโยค ใส่ token รหัสภาษา (`<LID>`) ทั้งฝั่ง Encoder และ Decoder
- เทรนโมเดล 25 ภาษา (mBART25) บน V100 256 ใบ 500K steps ราว 2.5 สัปดาห์ และมีรุ่นย่อย mBART06, mBART02
- ลำดับความยาวสูงสุดของแต่ละ instance 512 token

**เชื่อมกับข้อสอบ:** แหล่งของ "แปลภาษา (mBART): ใส่ language tag" ดูบทเรียน 3.6

### UL2: Unifying Language Learning Paradigms

- **ผู้เขียน:** Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Denny Zhou, Neil Houlsby, Donald Metzler
- **ที่ตีพิมพ์:** ICLR 2023 (บรรทัดอ้างอิงสไลด์ 42: The Eleventh International Conference on Learning Representations) ส่ง arXiv ครั้งแรก 10 พฤษภาคม 2022
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2205.05131
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 42 (ฝังรูปที่ 2 และ 3 ของเปเปอร์)

**ใจความ**

- เสนอ **Mixture-of-Denoisers (MoD)** เป็น objective ที่รวมหลายแนวทาง pre-training ในที่เดียว
- รูปในสไลด์: R-denoiser (span สั้น corruption ต่ำ), S-denoiser (sequential/prefix LM), X-denoiser (extreme: span ยาวหรือสัดส่วนสูง 3 แบบย่อย)
- ใช้ mode token สลับโหมดตอนเทรนและตอนใช้งาน
- ที่ขนาด 20B พารามิเตอร์ ได้ state of the art บนงาน supervised finetuning 50 งาน และ zero-shot SuperGLUE ดีกว่า GPT-3 175B
- สรุป one-shot ดีกว่า T5-XXL ราวสามเท่า
- Flan-UL2 20B (หลังปรับด้วย FLAN) ได้ MMLU และ Big-Bench แข่งกับ FLAN-PaLM 62B ได้

**เชื่อมกับข้อสอบ:** ข้อสรุปของสไลด์ว่า "objective สำคัญพอ ๆ กับสถาปัตยกรรม" ดูบทเรียน 7.6 (ตาราง [R] [S] [X])

### Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond

- **ผู้เขียน:** Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, Xia Hu
- **ที่ตีพิมพ์:** ACM Transactions on Knowledge Discovery from Data, เล่ม 18 ฉบับ 6 บทความ 160 เมษายน 2024 (ตามเชิงอรรถในสไลด์ 44) ส่ง arXiv ครั้งแรก 26 เมษายน 2023
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2304.13712
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 44 (รูปที่ 1 ต้นไม้วิวัฒนาการของ LLM)

**ใจความ**

- เป็นคู่มือเชิงปฏิบัติสำหรับผู้ใช้ LLM ในงาน NLP ปลายทาง พิจารณาโมเดล ข้อมูล และงาน
- สรุป LLM แบบ GPT-style (decoder-only) และแบบ BERT-style (encoder-only และ encoder-decoder)
- รูปที่ 1 ที่สไลด์ใช้: กิ่ง decoder-only สีน้ำเงิน, encoder-only สีชมพู, encoder-decoder สีเขียว แกนตั้งคือวันที่เปิดตัว สี่เหลี่ยมทึบคือโอเพนซอร์ส สี่เหลี่ยมกลวงคือปิด
- หน้าที่ฝังสรุปว่า encoder-decoder ยังมีความหวัง (ถูกสำรวจต่อเนื่อง ส่วนใหญ่เปิดโอเพนซอร์ส) แต่ความยืดหยุ่นของ decoder-only ทำให้ทางของ Google ที่ยืนกรานฝั่งนี้ดูน่าจะแคบลง
- พิจารณาปัจจัยด้านประสิทธิภาพ ต้นทุน ความหน่วง และ bias ที่เป็นปัญหาลวง (spurious bias) ตอนนำไปใช้

**เชื่อมกับข้อสอบ:** ภาพรวม "Dec-only เป็นแม่น้ำสายหลัก Enc-Dec เป็นสาขาย่อย" ดูบทเรียน 7.7

### Scaling Instruction-Finetuned Language Models (Flan-T5)

- **ผู้เขียน:** Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus และคนอื่น ๆ รวม 35 คน (ไปจนถึง Jason Wei)
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 20 ตุลาคม 2022 (สไลด์ 30 อ้างเป็น Chung et al., 2022 ส่วนชื่อวารสารที่ตีพิมพ์ภายหลังไม่ได้ตรวจ)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/2210.11416
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 30 ("Flan-T5 (Chung et al., 2022)") และสไลด์ 45

**ใจความ**

- ศึกษา instruction finetuning โดยเน้นสามเรื่อง: เพิ่มจำนวนงาน, เพิ่มขนาดโมเดล และ finetune ด้วยข้อมูล chain-of-thought
- ช่วยทุกกลุ่มโมเดล (PaLM, T5, U-PaLM) ทุกรูปแบบ prompt (zero-shot, few-shot, CoT) และหลาย benchmark (MMLU, BBH, TyDiQA, MGSM)
- Flan-PaLM 540B finetune ด้วย 1.8K งาน ชนะ PaLM 540B เฉลี่ย +9.4% และได้ 75.2% บน MMLU แบบ five-shot
- ปล่อยจุดตรวจ Flan-T5 ที่ few-shot ได้ดีเมื่อเทียบกับโมเดลใหญ่กว่ามาก เช่น PaLM 62B
- สไลด์ 30 เขียนจำนวนงานเป็น 1,836 งาน ส่วน abstract เขียน "1.8K" ซึ่งสอดคล้องกัน

**เชื่อมกับข้อสอบ:** ข้อเติมคำ "Flan-T5 fine-tune บน ___ งาน" (คำตอบ 1,836) ดูบทเรียน 4.2 และ 4.7

### Extracting and Composing Robust Features with Denoising Autoencoders

- **ผู้เขียน:** Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol
- **ที่ตีพิมพ์:** Proceedings of the 25th International Conference on Machine Learning (ICML 2008), ACM, หน้า 1096-1103 DOI 10.1145/1390156.1390294
- **ลิงก์ (ตรวจแล้วจากผลค้นหา ไม่ได้เปิดหน้าเปเปอร์ตรง ๆ):** https://mlanthology.org/icml/2008/vincent2008icml-extracting และ https://doi.org/10.1145/1390156.1390294
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 5 ("Vincent (2008) → BERT's MLM (2019) → BART/T5 (2020)")

**ใจความ**

- เสนอหลักการฝึกแบบใหม่สำหรับเรียนรู้ representation แบบไม่มีผู้สอน: ฝึก autoencoder บนอินพุตที่ถูกทำให้เสียโดยตั้งใจ เพื่อให้ representation ทนต่อการเสียบางส่วน
- autoencoder แบบนี้นำมาซ้อนกัน (stack) เพื่อเริ่มต้นสถาปัตยกรรมลึกได้
- ผู้เขียนอธิบายแรงจูงใจจากมุมมองของ manifold learning, ทฤษฎีสารสนเทศ และ generative modeling
- การทดลองจำแนกรูปแบบ (pattern classification) พบประโยชน์จากการทำให้อินพุตเสีย
- ฉบับเต็มตามมาที่ JMLR ปี 2010 ชื่อ Stacked Denoising Autoencoders

**เชื่อมกับข้อสอบ:** ต้นตระกูลของ objective `max log P(x | x̃)` ดูบทเรียน 2.1

### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- **ผู้เขียน:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **ที่ตีพิมพ์:** ปี 2019 (สไลด์ 2 อ้างเป็น Devlin et al., 2019) ส่ง arXiv ครั้งแรก 11 ตุลาคม 2018 (ปรับล่าสุด 24 พฤษภาคม 2019) ที่ว่าลงใน NAACL 2019 เป็นความรู้ของผู้เขียน ไม่ได้ตรวจจากหน้าที่เปิด
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1810.04805
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 2, 6, 9, 14 (อยู่ในรูปที่ 1 ของเปเปอร์ BART ที่ฝังไว้) และสไลด์ 5 ("MLM (2019)")

**ใจความ**

- BERT pre-train representation สองทิศทางจากข้อความไม่มีป้าย โดยให้ทุกชั้นมองทั้งบริบทซ้ายและขวา
- เติมชั้นเอาต์พุตชั้นเดียวก็ fine-tune ได้สำหรับ QA และ NLI ไม่ต้องแก้สถาปัตยกรรมมาก
- ได้ผลดีที่สุดในขณะนั้นบน 11 งาน: GLUE 80.5% (เพิ่มสัมบูรณ์ 7.7 จุด), MultiNLI 86.7% (เพิ่ม 4.6), SQuAD v1.1 F1 93.2 (เพิ่ม 1.5), SQuAD v2.0 F1 83.1 (เพิ่ม 5.1)
- รูปที่ 1 ของ BART ชี้จุดอ่อนที่สไลด์ใช้: BERT ทำนายคำที่ถูก mask แยกกัน จึงสร้างข้อความยาก
- กฎ mask 15% และ 80/10/10 อยู่ในรายละเอียดของเปเปอร์ (ไม่อยู่ใน abstract จึงไม่ได้ตรวจจากหน้าที่เปิด)

**เชื่อมกับข้อสอบ:** กฎ 80/10/10 ที่อาจารย์บอกว่าออกสอบ ดูบทเรียน 2.3 และตัวอย่างคำนวณข้อ 9

**กลุ่ม (ค): เปเปอร์ที่ปรากฏในข้อความของหน้าเปเปอร์ที่ฝังในสไลด์**

### RoBERTa: A Robustly Optimized BERT Pretraining Approach

- **ผู้เขียน:** Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 26 กรกฎาคม 2019
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1907.11692
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 2, 6, 9, 14 (อ้างเป็น "Liu et al., 2019" ในหน้าเปเปอร์ BART) และสไลด์ 13

**ใจความ**

- ผู้เขียนมองว่า BERT เทรนไม่พอ (significantly undertrained) และเมื่อเลือกการตั้งค่าเทรนให้ดีขึ้นก็เทียบเท่าหรือเหนือกว่าโมเดลที่ออกหลังจากนั้นได้
- BART-large เทรนในสเกลเดียวกับ RoBERTa: batch 8,000 และ 500,000 steps (สไลด์ 13) และเปเปอร์ BART รายงานว่าเทียบเท่า RoBERTa บน GLUE และ SQuAD

**เชื่อมกับข้อสอบ:** บริบทของตัวเลข pre-train ใน 3.5 เท่านั้น ไม่น่าเป็นหัวข้อออกสอบโดยตรง

### Improving Language Understanding by Generative Pre-Training (GPT-1)

- **ผู้เขียน:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)
- **ที่ตีพิมพ์:** รายงานทางเทคนิคของ OpenAI ปี 2018 (สไลด์ 2 อ้างเป็น Radford et al., 2018)
- **ลิงก์ (ไม่ได้เปิดหน้านี้โดยตรง ใช้ลิงก์ที่ปรากฏในผลค้นหา):** https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf ยังไม่ยืนยันว่ามีเลข arXiv
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 2, 6, 9, 14 (ในรูปที่ 1 ของเปเปอร์ BART คือกล่อง "GPT")

**ใจความ (จากผลค้นหา ซึ่งเป็นสรุปรอง ไม่ใช่ตัวเปเปอร์)**

- วิธีคือ pre-train โมเดลภาษาแบบ generative บนข้อความไม่มีป้าย แล้ว fine-tune แบบ discriminative ต่อแต่ละงาน
- สถาปัตยกรรมเป็น Transformer decoder 12 ชั้น เทรนบน BooksCorpus (หนังสือที่ยังไม่ตีพิมพ์กว่า 7,000 เล่ม)
- คงงานโมเดลภาษาไว้เป็น objective เสริมตอน fine-tune ซึ่งช่วยให้ generalize และลู่เข้าเร็วขึ้น

**เชื่อมกับข้อสอบ:** ตัวแทนของ Paradigm 2 (CLM, signal 100%) ในตารางสไลด์ 34 ดูบทเรียน 2.3 และ 7.1

### SpanBERT: Improving Pre-training by Representing and Predicting Spans

- **ผู้เขียน:** Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 24 กรกฎาคม 2019
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1907.10529
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 5 และ 7 (ข้อความในหน้าเปเปอร์ BART ที่บอกว่า text infilling ได้แรงบันดาลใจจาก SpanBERT)

**ใจความ**

- SpanBERT mask ช่วงคำที่ติดกัน แล้วทำนายช่วงนั้นจาก representation ที่ขอบของช่วง
- ได้ผลดีกว่า BERT และ baseline ที่แข็งกว่า โดยดีขึ้นมากที่สุดในงานเลือกช่วงข้อความ เช่น QA และ coreference
- ตามหน้าเปเปอร์ BART ที่ฝังในสไลด์: SpanBERT สุ่มความยาวจากการแจกแจงเรขาคณิตแบบถูกตัด และแทนด้วย `[MASK]` จำนวนเท่าความยาวจริง ส่วน BART แทนด้วย `[MASK]` ตัวเดียว

**เชื่อมกับข้อสอบ:** ช่วยแยก "Text Infilling ของ BART" ออกจากการ mask span แบบเดิม ดูบทเรียน 3.4

### Gaussian Error Linear Units (GELUs)

- **ผู้เขียน:** Dan Hendrycks, Kevin Gimpel
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 27 มิถุนายน 2016 (สไลด์อ้างเป็น Hendrycks & Gimpel, 2016)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1606.08415
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 2, 6, 14 (ประโยค "we modify ReLU activation functions to GeLUs" ในหน้าเปเปอร์ BART)

**ใจความ**

- เสนอฟังก์ชัน activation ที่ปรับสเกลอินพุตตามขนาดของมัน แทนที่จะตัดตามเครื่องหมายแบบ ReLU
- รายงานผลดีกว่า ReLU และ ELU ในงานภาพ ภาษา และเสียง

**เชื่อมกับข้อสอบ:** รายละเอียดปลีกย่อยของสถาปัตยกรรม BART (3.1) น่าจะไม่ถูกถามโดยตรง

### Pre-trained Language Model Representations for Language Generation

- **ผู้เขียน:** Sergey Edunov, Alexei Baevski, Michael Auli
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 22 มีนาคม 2019 (สไลด์อ้างเป็น Edunov et al. 2019)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1903.09722
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 5 และ 7 (หัวข้อ 3.4 แปลภาษาของเปเปอร์ BART)

**ใจความ**

- การเติม representation จากโมเดลภาษาที่ pre-train แล้วเข้าไปในฝั่ง Encoder ของโมเดล seq2seq ได้ผลดีกว่าวิธีผสานแบบอื่น โดยช้าลงตอน inference เพียง 14%
- ปรับปรุงการแปลภาษา และได้ state of the art บน CNN/DailyMail
- หน้าเปเปอร์ BART อ้างว่างานก่อนหน้านี้แสดงว่า encoder ที่ pre-train ช่วยได้ แต่ประโยชน์ของการ pre-train ฝั่ง decoder มีจำกัด และ BART แสดงว่าใช้ทั้งโมเดลเป็น decoder ที่ pre-train ได้

**เชื่อมกับข้อสอบ:** บริบทของ BART สำหรับแปลภาษา ดูบทเรียน 3.6

### fairseq: A Fast, Extensible Toolkit for Sequence Modeling

- **ผู้เขียน:** Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 1 เมษายน 2019 (สไลด์อ้างเป็น Ott et al., 2019)
- **ลิงก์ (ตรวจแล้ว):** https://arxiv.org/abs/1904.01038
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 29 (ประโยค "All experiments are done with Fairseq" ในหน้าเปเปอร์ mBART)

**ใจความ**

- เป็นชุดเครื่องมือโอเพนซอร์สบน PyTorch สำหรับสร้างโมเดล sequence แบบกำหนดเอง เช่น แปลภาษาและสรุปความ
- รองรับเทรนแบบกระจายและเทรนแบบ mixed precision ที่เร็วบน GPU รุ่นใหม่

**เชื่อมกับข้อสอบ:** ไม่น่าออกสอบ เป็นเพียงเครื่องมือที่ mBART ใช้

### Phrase-Based & Neural Unsupervised Machine Translation

- **ผู้เขียน:** Guillaume Lample, Myle Ott, Alexis Conneau, Ludovic Denoyer, Marc'Aurelio Ranzato
- **ที่ตีพิมพ์:** ส่ง arXiv ครั้งแรก 20 เมษายน 2018
- **ลิงก์ (ตรวจแล้ว ว่าเป็นเปเปอร์นี้จริง):** https://arxiv.org/abs/1804.07755 ข้อสังเกต: หน้าเปเปอร์ mBART ในสไลด์ 29 อ้างเป็น "Lample et al. (2018c)" ผู้เขียนเดาว่าหมายถึงฉบับนี้ แต่ยังไม่ได้เทียบรายการอ้างอิงของ mBART เพื่อยืนยัน
- **อยู่ในสไลด์:** สัปดาห์ 8 สไลด์ 29 (ประโยค "other noise types, such as those in Lample et al. (2018c)")

**ใจความ**

- โมเดลแปลภาษาแบบไม่ใช้ประโยคคู่ขนานเลย (ทั้งแบบ neural และ phrase-based) ได้ 28.1 BLEU บน WMT'14 อังกฤษ-ฝรั่งเศส และ 25.2 บน WMT'16 เยอรมัน-อังกฤษ
- ดีกว่าวิธีก่อนหน้าเกิน 11 BLEU

**เชื่อมกับข้อสอบ:** บริบทของ noise ทางเลือกใน mBART ไม่น่าออกสอบ

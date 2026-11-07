**Q1** [W8] [BART Noising Strategies] [difficulty: medium]
พิจารณาวิธีการใส่สัญญาณรบกวน (Noising strategies) ของ BART ในขั้นตอน Pre-training ข้อใดเปรียบเทียบความแตกต่างระหว่าง **Token Deletion** และ **Text Infilling** ได้ถูกต้องตามความเป็นจริง?
A. Token Deletion จะแทนที่คำที่ถูกลบด้วยสัญลักษณ์ `[MASK]` ทำให้ความยาวของ source เท่ากับ target ส่วน Text Infilling จะไม่ทิ้ง placeholder ใด ๆ ไว้เลย
B. Token Deletion ลบ token ทิ้งโดยไม่เหลือ placeholder ใด ๆ (ทำให้ src_len < tgt_len) ขณะที่ Text Infilling สุ่มความยาว span จาก Poisson ($\lambda = 3$) แล้วแทนที่ด้วย `[MASK]` เพียงตัวเดียว ซึ่งหากสุ่มได้ความยาว 0 จะเป็นการแทรก `[MASK]` เพื่อบังคับให้โมเดลเรียนรู้ที่จะลบทิ้ง
C. Text Infilling กำหนดให้ 1 token ที่หายไปถูกแทนด้วย 1 `[MASK]` เสมอ ทำให้ความยาว input เท่ากับ target เสมอเช่นเดียวกับ BERT Token Masking
D. ทั้ง Token Deletion และ Text Infilling ต่างใช้ unique sentinel tokens (เช่น `<extra_id_0>`) เพื่อระบุตำแหน่งเริ่มต้นและสิ้นสุดของข้อความที่ขาดหายไป

**Q2** [W8] [T5 Span Corruption and Sentinel Tokens] [difficulty: medium]
ในการ Pre-train โมเดล T5 ด้วยวิธี Span Corruption หากข้อความ input ถูกสุ่มตัดคำออกไป 2 spans ที่ไม่ติดกัน ข้อใดอธิบายโครงสร้างของ Decoder Target และ Sentinel token ได้ถูกต้องที่สุด?
A. Decoder Target จะต้อง reconstruct ข้อความต้นฉบับกลับมาแบบสมบูรณ์ทั้งประโยค (Full reconstruction) โดยขึ้นต้นด้วย `<s>` และลงท้ายด้วย `</s>`
B. Decoder Target จะประกอบด้วยเฉพาะคำที่ถูกตัดออกไปต่อกัน โดยคั่นตำแหน่งด้วย token `[MASK]` เดียวกันทั้งหมด และปิดท้ายด้วย `</s>`
C. Decoder Target จะประกอบด้วย `<extra_id_0>` ตามด้วย span ที่ 1, `<extra_id_1>` ตามด้วย span ที่ 2 และปิดท้ายด้วย token `</s>` ทันทีโดยไม่มี sentinel ปิดท้าย
D. Decoder Target จะประกอบด้วย `<extra_id_0>` ตามด้วย span ที่ 1, `<extra_id_1>` ตามด้วย span ที่ 2 และตามด้วย `<extra_id_2>` เพื่อบ่งบอกว่าไม่มี span ที่หายอีกแล้ว ก่อนจะปิดท้ายด้วย `</s>`

**Q3** [W8] [BART vs T5 Decoder Compute Efficiency] [difficulty: hard]
สมมติเอกสารต้นฉบับมีความยาว $N = 1000$ tokens โดย T5 มีความยาว Decoder Target ประมาณ $0.20N$ (เกิดจาก corrupted tokens 15% รวมกับ sentinels 5%) ในขณะที่ BART กำหนดให้ Decoder ต้องสร้างข้อความเดิมทั้งหมด ($N$ tokens) ข้อใดระบุอัตราส่วนความประหยัดเชิงคำนวณของ **Decoder Self-Attention** ระหว่าง BART ต่อ T5 และความเร็วรวมในการฝึกทั้งโมเดล (Overall training speedup) ได้ถูกต้อง?
A. Decoder Self-Attention ของ T5 ประหยัดกว่า BART 25 เท่า และทำให้ความเร็วการฝึกรวมทั้งโมเดลเร็วขึ้นประมาณ 2-3 เท่า
B. Decoder Self-Attention ของ T5 ประหยัดกว่า BART 5 เท่า และทำให้ความเร็วการฝึกรวมทั้งโมเดลเร็วขึ้นประมาณ 5 เท่า
C. Decoder Self-Attention ของ T5 ประหยัดกว่า BART 25 เท่า และส่งผลให้ความเร็วการฝึกรวมทั้งโมเดลเร็วขึ้น 25 เท่าตามไปด้วย
D. Decoder Self-Attention ของ T5 ประหยัดกว่า BART 2.5 เท่า เนื่องจากคิดจาก $(0.20N)^2 = 0.4N^2$ ทำให้ความเร็วรวมเร็วขึ้นประมาณ 1.5 เท่า

**Q4** [W8] [Architecture Paradigm and Parameter Allocation] [difficulty: medium]
ในการเปรียบเทียบเชิงวิศวกรรมระหว่างสถาปัตยกรรม Encoder-Decoder (เช่น T5) กับ Decoder-only (เช่น GPT) สำหรับงานสร้างภาษา ข้อใดระบุข้อเท็จจริงเกี่ยวกับการจัดสรรพารามิเตอร์และเหตุผลที่ Decoder-only ได้รับความนิยมมากกว่าได้ถูกต้อง?
A. โมเดล Encoder-Decoder ที่มี $P$ พารามิเตอร์ จะใช้พารามิเตอร์ทั้งหมด $P$ ในขั้นตอนการ Generate ข้อความ ทำให้มีศักยภาพในการคำนวณต่อ token สูงกว่า Decoder-only
B. Encoder-Decoder จัดการ KV Cache ได้เรียบง่ายกว่า Decoder-only เนื่องจากไม่ต้องเก็บ cache ของ Decoder ในระหว่างการทำ autoregressive generation
C. โมเดล Encoder-Decoder ที่มี $P$ พารามิเตอร์ จะแบ่งไปอยู่ที่ Encoder ประมาณ $P/2$ และ Decoder $P/2$ ทำให้มีพารามิเตอร์ทำงานตอน generate จริงเพียงครึ่งเดียว ขณะที่ Decoder-only ใช้ทั้ง $P$ พารามิเตอร์ และยังรองรับ multi-turn conversation ในกระบวนการ RLHF ได้อย่างเป็นธรรมชาติกว่า
D. มีผลการทดลองแบบควบคุมตัวแปรที่ยืนยันแล้วว่าที่ขนาดโมเดลมากกว่า 100B พารามิเตอร์ สถาปัตยกรรม Decoder-only มีประสิทธิภาพเหนือกว่า Encoder-Decoder ในทุกด้านอย่างชัดเจน

**Q5** [W8] [T5 Text-to-Text and Task Prefix] [difficulty: hard]
ในการทดลองรันโมเดล T5-small (Pre-trained) เพื่อจำแนกอารมณ์ของประโยคภาษาอังกฤษ (Sentiment Analysis) แต่ผู้พัฒนา **ไม่ได้ใส่ task prefix** (ส่งเฉพาะข้อความรีวิวเข้าไปตรง ๆ โดยไม่มี `sst2 sentence:`) พฤติกรรมใดจะเกิดขึ้นกับผลลัพธ์ของโมเดลตามที่พบใน Demo และเป็นเพราะเหตุใด?
A. โมเดลจะ fallback ไปทำหน้าที่แปลภาษา (เช่น แปลเป็นภาษาเยอรมัน) เนื่องจาก prefix ทำหน้าที่เป็น task switch โดยตรง และ translation เป็นภารกิจที่พบบ่อยใน pre-training mix
B. โมเดลจะไม่สร้างคำตอบใด ๆ ออกมา (Empty string) เนื่องจากไม่พบ task prefix ที่ตรงกับพจนานุกรมคำสั่งของโมเดล
C. โมเดลจะคัดลอก (Copy) ข้อความ input เดิมกลับมาทั้งหมดโดยอัตโนมัติ เนื่องจากถือเป็นคำสั่งเริ่มต้น (Default echo)
D. โมเดลจะยังคงจำแนก sentiment และตอบว่า "positive" หรือ "negative" ได้อย่างถูกต้อง เพราะโมเดลเข้าใจความหมายของประโยคได้โดยไม่ต้องพึ่ง prefix

**Q6** [W8] [Cross-Attention Mechanism and Tensor Shapes] [difficulty: medium]
พิจารณากลไก Cross-Attention ในสถาปัตยกรรม Transformer แบบ Encoder-Decoder ข้อใดอธิบายแหล่งที่มาของเวกเตอร์ (Query, Key, Value), ข้อกำหนดของ Attention Mask และมิติ (Shape) ของ Attention Matrix ได้ถูกต้อง?
A. Query และ Key มาจาก Encoder ส่วน Value มาจาก Decoder โดยต้องใส่ Causal Mask เพื่อป้องกันการมองเห็น token ล่วงหน้า
B. Query และ Key มาจาก Decoder ส่วน Value มาจาก Encoder และมีมิติของ Attention Matrix เป็นรูปสี่เหลี่ยมจัตุรัสขนาด $[src\_len, src\_len]$
C. Query มาจาก Decoder ส่วน Key และ Value มาจาก Encoder โดยไม่ต้องใช้ Causal Mask และมีมิติของ Attention Matrix เป็น $[tgt\_len, src\_len]$
D. Query, Key และ Value ทั้งหมดถูกส่งมาจาก Encoder ผ่านกระบวนการ Bidirectional Masking โดยที่ Decoder ทำหน้าที่เพียงรับผลลัพธ์สุดท้ายไปคำนวณ Softmax

**Q7** [W8] [Extractive vs Generative QA Mechanism] [difficulty: hard]
ในการทดสอบเปรียบเทียบระหว่างโมเดล Extractive QA (BERT) และ Generative QA (T5) หากทำการป้อนคำถามว่า "What is the capital of Germany?" โดย **ไม่กำหนด Context ให้โมเดลเลย (Zero-context)** ข้อใดอธิบายความแตกต่างเชิงกลไก (Mechanism) และผลลัพธ์ที่เกิดขึ้นได้ถูกต้องตามผลการทดลองใน Demo?
A. ทั้งสองโมเดลจะหยุดการทำงานและแจ้งข้อผิดพลาด (Runtime Error) เนื่องจากระบบบังคับให้มี Context เสมอ
B. BERT สามารถใช้ความรู้ภายในโมเดลตอบ "Berlin" ได้ถูกต้อง ขณะที่ T5 ล้มเหลวเนื่องจากไม่มีกลไก Index Pointing
C. BERT จะสุ่มทำนายตำแหน่งคำตอบออกมานอกขอบเขตของประโยคคำถาม ขณะที่ T5 สามารถตอบ "Berlin" ได้อย่างแม่นยำเนื่องจากเป็นสถาปัตยกรรมแบบ Text-to-Text
D. BERT ล้มเหลวเชิงโครงสร้างโดยทำนาย Start/End index บนข้อความคำถามของตัวเอง (ตอบคำถามตัวเอง) ขณะที่ T5 ใช้กลไกการสร้างคำตอบอิสระจึงพยายามตอบได้ แม้ว่า T5-small จะเกิด Hallucination ตอบว่า "Germany" ก็ตาม

**Q8** [W8] [T5 Relative Position Bias] [difficulty: easy]
สถาปัตยกรรม T5 ไม่ได้ใช้ Absolute Positional Encoding แต่เลือกใช้ Relative Position Bias แทน ข้อใดอธิบายลักษณะและการทำงานของ Relative Position Bias ใน T5 ได้ถูกต้อง?
A. นำเวกเตอร์ Positional Embedding แบบ Sinusoidal ไปคูณเข้ากับ Attention Weights ภายหลังการคำนวณ Softmax
B. ใช้ตาราง Lookup Table ขนาด 32 buckets ต่อ head (โดยระยะห่าง 0-7 ใช้ exact bucket และระยะมากกว่านั้นจะแชร์ bucket แบบลอการิทึม) แล้วนำค่า scalar ไปบวกเข้ากับ Attention Logits ก่อนเข้าฟังก์ชัน Softmax
C. มีการกำหนด bucket แยกอิสระสำหรับทุก ๆ ตำแหน่งสัมพัทธ์ตั้งแต่ 0 ถึง 512 โดยไม่มีการแชร์ bucket ร่วมกัน และนำไปบวกเข้ากับ Input Word Embeddings โดยตรง
D. คำนวณระยะห่างสัมพัทธ์ระหว่างตำแหน่ง $i$ และ $j$ แบบต่อเนื่อง แล้วนำค่าระยะทางไปหารสเกลาร์ผลคูณ Dot-Product ของ Query และ Key

**Q9** [W8] [Enc-Dec vs Dec-only FLOPs in Summarization] [difficulty: hard]
กำหนดงานสรุปความ (Summarization) ที่มีขนาดเอกสารต้นฉบับ $L_{doc} = 1000$ tokens และข้อความสรุปผลลัพธ์ $L_{sum} = 100$ tokens โดยสมมติให้โมเดลทั้งสองมีจำนวนพารามิเตอร์รวมเท่ากัน (Encoder-Decoder มีฝั่งละ $N$ เลเยอร์ รวมเป็น $2N$ เลเยอร์ ส่วน Decoder-only มี $2N$ เลเยอร์)
เมื่อคำนวณจำนวน Attention Operations ตามสูตรในสไลด์ ข้อใดเปรียบเทียบต้นทุนการคำนวณ (FLOPs) ระหว่างสองสถาปัตยกรรมนี้ได้ถูกต้อง?
A. Encoder-Decoder มีต้นทุนสูงกว่า Decoder-only ประมาณ 1.50 เท่า เนื่องจากการคำนวณแยกกันสองโมดูล
B. ทั้งสองสถาปัตยกรรมมีต้นทุนการคำนวณเท่ากันพอดีที่ $1,000,000 N$ เนื่องจากมีจำนวนพารามิเตอร์รวมเท่ากัน
C. Decoder-only มีต้นทุนสูงกว่า Encoder-Decoder ประมาณ 1.09 เท่า (คิดเป็น $1,210,000 N$ เทียบกับ $1,110,000 N$)
D. Decoder-only ประหยัดกว่า Encoder-Decoder ประมาณ 2 เท่า เนื่องจากใช้ Causal Masking ทำให้คำนวณเพียงครึ่งหนึ่งของเมทริกซ์ $(1100^2 / 2 = 605,000 N)$

**Q10** [W8] [Modern Application of Encoder-Decoder] [difficulty: easy]
แม้ว่า Decoder-only จะครองตลาด Large Language Models ส่วนใหญ่ในปัจจุบัน แต่สถาปัตยกรรม Encoder-Decoder ยังคงเป็นสถาปัตยกรรมระดับแนวหน้า (State-of-the-art) ในงานประเภทใด และด้วยเหตุผลใด?
A. งานเขียนโค้ดโปรแกรม (Code Generation) เพราะต้องการ Causal Masking ตรวจสอบไวยากรณ์ทีละบรรทัด
B. งานสนทนาโต้ตอบหลายรอบ (Multi-turn Chat) เพราะสามารถบันทึก Static Cache ของผู้ใช้ได้ยาวนานกว่า
C. งาน Retrieval-Augmented Generation (RAG) ขนาดใหญ่ระดับเกิน 100B พารามิเตอร์ เพราะประหยัด KV Cache ที่สุด
D. งาน Machine Translation (เช่น NLLB ของ Meta) และ Speech-to-Text (เช่น Whisper ของ OpenAI) เนื่องจากเป็นงานที่ Input มีขนาดยาวและต้องการ Bidirectional Understanding ที่สมบูรณ์ ขณะที่ Output มีขนาดสั้นกว่า

**F1** [W8] [BART and T5 Architectural Specifications and Denoising]
โมเดล T5 ย่อมาจาก Text-to-Text (1) ___ Transformer โดยในขั้นตอน Pre-training จะใช้กลไก Span Corruption ซึ่งแทนที่ข้อความที่ถูกตัดออกด้วย special token ที่เรียกว่า (2) ___ เช่น `<extra_id_0>` ส่งผลให้ขนาด Decoder Target สั้นลงเหลือประมาณ (3) ___ ของความยาวเอกสาร $N$ ทำให้การคำนวณ Decoder Self-Attention ประหยัดกว่า BART ประมาณ (4) ___ เท่า ในขณะที่ฝั่ง BART มีวิธี noising ที่เป็นเอกลักษณ์คือสุ่มความยาว span ตามการแจกแจงแบบ Poisson ($\lambda = 3$) แล้วแทนที่ด้วย `[MASK]` เพียงตัวเดียว ซึ่งเรียกว่าวิธี (5) ___

### Answer key
- **Q1: B** — Token Deletion ลบ token โดยไม่เหลือ placeholder ($src < tgt$) ขณะที่ Text Infilling แทน span สุ่มจาก Poisson ด้วย `[MASK]` ตัวเดียว และ span ยาว 0 จะเป็นการ insert `[MASK]` บังคับให้โมเดลลบทิ้ง [Notes: 2.5, 4 (Demo 1), 7 (Traps #4, #5)]
- **Q2: D** — Target ของ T5 จะบรรจุเฉพาะ corrupted spans คั่นด้วย sentinel และต้องปิดท้ายด้วย sentinel ถัดไป (`<extra_id_N+1>`) เพื่อระบุว่าหมด span ที่หาย แล้วจึงตามด้วย `</s>` [Notes: 3.1, 4 (Demo 1), 7 (Traps #1, #2)]
- **Q3: A** — Decoder self-attn ratio = $N^2 / (0.2N)^2 = 1 / 0.04 = 25$ เท่า แต่ความเร็วรวมทั้งโมเดลเร็วขึ้นเพียง ~2-3 เท่า เพราะ Encoder ยังประมวลผล input ยาวใกล้เคียงเดิม [Notes: 3.4, 7 (Traps #7, #8)]
- **Q4: C** — Enc-Dec แบ่งพารามิเตอร์ $P$ ไปที่ Encoder $\sim P/2$ และ Decoder $\sim P/2$ ทำให้ตอน generate ทำงานจริงแค่ครึ่งเดียว ขณะที่ Dec-only ใช้เต็ม $P$ และเข้ากับ multi-turn RLHF ได้ดีกว่า [Notes: 2.1, 7 (Traps #22, #23, #24)]
- **Q5: A** — ใน Demo 5 เมื่อไม่ใส่ prefix ให้ T5 โมเดลไม่ได้คัดลอกหรือคืนค่าว่าง แต่ fallback ไปทำ translation ซึ่งเป็น task ที่พบบ่อยสุดใน pre-training แสดงว่า prefix คือ task switch [Notes: 4 (Demo 5), 7 (Trap #14)]
- **Q6: C** — Cross-Attention ใช้ Query จาก Decoder ข้ามไปมอง Key และ Value จาก Encoder โดยไม่ต้องใส่ Mask และ Attention Matrix มีขนาดตามแถวและคอลัมน์คือ $[tgt\_len, src\_len]$ [Notes: 2.1, 2.3, 3.8, 7 (Traps #19, #20)]
- **Q7: D** — เมื่อไม่มี context กลไก Index Pointing ของ BERT ถูกจำกัดโครงสร้างให้ออกนอกข้อความไม่ได้จึงไฮไลต์คำถามตัวเอง ขณะที่ T5 สร้าง token อิสระจึงพยายามตอบแม้จะเกิด hallucination [Notes: 4 (Demo 4), 7 (Traps #17, #18)]
- **Q8: B** — T5 ใช้ Relative Position Bias ขนาด 32 buckets ต่อ head (0-7 exact, 8+ log) โดยนำค่า bias สเกลาร์ไปบวกเข้ากับ logits ก่อนคำนวณ Softmax [Notes: 3.7, 7 (Traps #10, #11, #12)]
- **Q9: C** — Enc-Dec FLOPs $= N \times (1000^2 + 100^2 + 1000 \times 100) = 1,110,000 N$ ขณะที่ Dec-only FLOPs $= 2N \times (1100^2 / 2) = 1,210,000 N$ คิดเป็นอัตราส่วน $1,210,000 / 1,110,000 \approx 1.09$ เท่า [Notes: 3.6, 7 (Trap #27)]
- **Q10: D** — Enc-Dec ยังคงเป็นเลิศในงานที่ Input ขนาดยาวและต้องการ bidirectional encoding ร่วมกับ Output ที่สั้นกว่า เช่น NLLB (Translation) และ Whisper (Speech-to-text) [Notes: 1.4, 6 (#12), 7 (Trap #26)]
- **F1:** (1) Transfer, (2) sentinel token, (3) 0.20 (หรือ 20% / 0.2), (4) 25, (5) Text Infilling [Notes: 2.5, 3.4, 6 (#1, #2, #3, #4)]

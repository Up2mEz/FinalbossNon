**Q61** [W14] [Deduplication] [difficulty: medium]
ในการออกแบบระบบ Deduplication สำหรับชุดข้อมูล Pre-training ระดับ Web-scale ข้อใดเปรียบเทียบหลักการและหน้าที่ของ MinHash, SimHash และ Locality-Sensitive Hashing (LSH) ได้ถูกต้อง?
A. MinHash ทำหน้าที่ลดจำนวนคู่เอกสารทั้งหมดจาก $O(N^2)$ ขณะที่ LSH ทำหน้าที่คำนวณความคล้ายคลึงระหว่างคู่เอกสาร
B. SimHash ใช้ fingerprint ขนาด 64-bit เพื่อประมาณค่า Jaccard similarity ส่วน MinHash ใช้ signature เพื่อรักษาระยะ Cosine similarity
C. LSH banding ช่วยลดจำนวนคู่ที่ต้องนำมาเปรียบเทียบ โดยจัดกลุ่มเอกสารที่มี band ตรงกันเป็น candidate ซึ่งยังจำเป็นต้องนำไปตรวจสอบ (verify) ด้วยค่า Jaccard similarity จริงอีกครั้ง
D. เมื่อเอกสารถูกจับคู่เป็น candidate ใน LSH bucket เดียวกันแล้ว สามารถตัดทิ้งเป็น duplicate ได้ทันทีโดยไม่ต้องเปรียบเทียบ shingle ซ้ำ

**Q62** [W14] [Deduplication & Repetition] [difficulty: medium]
ในขั้นตอนเตรียมข้อมูลสำหรับ Pre-training ทีมงานพบว่าในหน้าเว็บประเภท Cookie banner มักมีข้อความเดิมซ้ำไปซ้ำมาหลายสิบรอบภายในหน้าเดียวกัน (Internal document repetition) แต่เมื่อนำข้อมูลผ่านกระบวนการ MinHash + LSH เอกสารเหล่านี้กลับรอดผ่านไปได้เกือบทั้งหมด เหตุการณ์นี้เกิดขึ้นเพราะเหตุใด และควรแก้ไขด้วยวิธีใดตามแนวคิดของ Gopher pipeline?
A. MinHash และ LSH ออกแบบมาเพื่อตรวจจับ Near-duplicate ข้ามคู่เอกสาร (Pairwise) จึงไม่สามารถตรวจจับการซ้ำภายในเอกสารเดียวกันได้ ต้องใช้ Repetition heuristics ในการตัดทิ้ง
B. เอกสารที่มีข้อความซ้ำกันภายในหน้าจะมีค่า Perplexity สูงมากจนทะลุเกณฑ์ จึงไม่ถูกนับเป็นเอกสารซ้ำซ้อน
C. เกิดจากค่า $r$ (จำนวน rows ต่อ band) ใน LSH สูงเกินไป ทำให้ความน่าจะเป็นในการจับคู่ candidate ลดลงจนเป็นศูนย์
D. MinHash ถูกรบกวนโดย stop words ภาษาอังกฤษ ทำให้ค่า Jaccard similarity ระหว่างเอกสารคำนวณออกมาเป็นลบ

**Q63** [W14] [Quality Filtering] [difficulty: easy]
การคัดกรองคุณภาพข้อมูลเว็บด้วย Perplexity (PPL) โดยใช้โมเดลภาษาอ้างอิง เช่น KenLM 5-gram ที่เทรนบน Wikipedia ใน CCNet ข้อใดอธิบายพฤติกรรมของค่า Perplexity และวิธีการนำไปใช้จริงได้อย่างถูกต้อง?
A. เอกสารที่มีค่า Perplexity ต่ำที่สุดจัดเป็นเอกสารที่มีเนื้อหาเชิงวิชาการและมีคุณภาพสูงที่สุดเสมอ
B. CCNet กำหนดค่า Perplexity threshold คงที่แบบสากล หากเอกสารใดมีค่าเกินเกณฑ์จะถูกตัดทิ้งทันที
C. ข้อความซ้ำซากหรือสแปม (เช่น Cookie banner) มักได้ค่า Perplexity ต่ำมากเพราะทำนายคำถัดไปได้ง่าย ส่วนข้อความวิชาการเฉพาะทางอาจตกไปอยู่กลุ่ม Perplexity สูง (Tail)
D. CCNet จะเลือกเก็บเฉพาะเอกสารที่อยู่ในกลุ่ม Tail เท่านั้น เพราะเป็นกลุ่มที่มีคำศัพท์หลากหลายและมีความรู้แปลกใหม่มากที่สุด

**Q64** [W14] [Filter Pipeline Optimization] [difficulty: hard]
ทีมวิศวกรต้องการออกแบบ Pipeline เพื่อกรองเอกสารจำนวนมหาศาล โดยมีตัวกรอง 2 ชนิดที่ต้องทำงานต่อเนื่องกัน:
- Filter A (Heuristic rule): ต้นทุนประมวลผล $c_A = 0.1\text{ ms/doc}$, อัตราเอกสารที่รอดผ่าน (survival rate) $p_A = 0.3$
- Filter B (Neural classifier): ต้นทุนประมวลผล $c_B = 10\text{ ms/doc}$, อัตราเอกสารที่รอดผ่าน $p_B = 0.5$
หากต้องการให้ระบบมีต้นทุนเวลาเฉลี่ยต่อเอกสาร (ms/doc) ต่ำที่สุด ควรจัดลำดับตัวกรองอย่างไร และมีต้นทุนรวมเฉลี่ยเท่าใด?
A. รัน B ก่อน A เพราะ Neural classifier แม่นยำกว่า โดยมีต้นทุนรวม 10.05 ms/doc
B. รัน A ก่อน B เพราะมีต้นทุนรวมเฉลี่ยเพียง 3.10 ms/doc
C. รัน A ก่อน B เพราะมีต้นทุนรวมเฉลี่ยเพียง 0.15 ms/doc
D. สลับลำดับอย่างไรก็ได้ เนื่องจากตัวกรองทั้งสองเป็น Per-document filter ต้นทุนรวมจะเท่ากันที่ 5.05 ms/doc เสมอ

**Q65** [W14] [Tokenizer Fairness] [difficulty: medium]
จากการทดสอบ Tokenizer สำหรับข้อความภาษาไทยเทียบกับภาษาอังกฤษ พบว่า GPT-2 ให้ค่า TPC (Tokens Per Character) ภาษาไทยเท่ากับ 2.0 (57 ตัวอักษร ได้ 114 tokens) ขณะที่ BERT-base-uncased ให้ค่า TPC ภาษาไทยเพียง 0.0175 (57 ตัวอักษร ได้ 1 token) ข้อสรุปใดถูกต้องตามหลักการทางภาษาและสถาปัตยกรรม Tokenizer?
A. BERT มีสถาปัตยกรรม Vocabulary ที่รองรับและบีบอัดภาษาไทยได้อย่างมีประสิทธิภาพสูงสุดเหนือ GPT-2
B. ค่า TPC ที่ต่ำมากของ BERT เกิดจาก Coverage failure เนื่องจากไม่มีคำศัพท์ภาษาไทยใน Vocabulary ทำให้ข้อความทั้งหมดถูกแปลงเป็น token `[UNK]` เพียงตัวเดียว
C. ค่า TPC คำนวณจากสัดส่วนของจำนวน Token ต่อจำนวนคำ (Words) ดังนั้นค่า 2.0 ของ GPT-2 แสดงว่าโมเดลตัดคำไทยได้ 2 คำต่อ 1 ประโยค
D. อัตราส่วนความเสียเปรียบของภาษาไทย ($\rho \approx 14.2$) เกิดจากการที่ตัวอักษรไทยถูกเข้ารหัสด้วย ASCII ขนาด 1 byte ซึ่งสูญเสียพื้นที่มากกว่า UTF-8

**Q66** [W14] [Evaluation Metrics] [difficulty: medium]
เมื่อนำประโยคสังเคราะห์ Candidate: "mat the on is cat the" ไปเปรียบเทียบกับ Reference: "the cat sat on the mat" พบว่าได้คะแนน BLEU-1 สูงถึง 0.833 ขณะที่ประโยค Paraphrase ที่มีความหมายถูกต้องอย่าง "a feline rested upon the rug" ได้คะแนน BLEU-1 เพียง 0.167 ปรากฏการณ์นี้สะท้อนปัญหาใด และ BERTScore แก้ไขปัญหานี้ได้อย่างไร?
A. สะท้อนปัญหา Length bias โดย BERTScore แก้ไขโดยการใช้ Length Penalty Factor ในการตัดคะแนนคำซ้ำ
B. สะท้อนว่า BLEU ขาดการทำ Positional swap โดย BERTScore แก้ไขด้วยการสลับลำดับคำแบบ Brute-force
C. สะท้อนว่า n-gram overlap ไม่สามารถตรวจจับประโยคที่ถูกต้องได้ โดย BERTScore แก้ไขด้วยการใช้ Rule-based Thesaurus ค้นหาคำไวพจน์
D. สะท้อนข้อจำกัดของ n-gram surface matching ที่ให้คะแนน Word salad สูงเกินจริง โดย BERTScore แก้ไขด้วยการเปรียบเทียบ Contextual Embeddings ผ่าน Greedy Matching บน Cosine Similarity Matrix

**Q67** [W14] [Benchmarks] [difficulty: easy]
ในการประเมินความสามารถด้านการเขียนโปรแกรมด้วย HumanEval โดยสุ่มสร้างโค้ด $n = 200$ ตัวอย่างต่อโจทย์ 1 ข้อ สำหรับโจทย์ระดับยากข้อหนึ่ง พบว่ามีโค้ดที่รันผ่านชุดทดสอบเพียง $c = 1$ ตัวอย่าง หากคำนวณคะแนนด้วยมาตรวัด $pass@100$ สำหรับโจทย์ข้อนี้ จะได้ค่าความน่าจะเป็นเท่ากับเท่าใด?
A. 0.005
B. 0.050
C. 0.500
D. 0.751

**Q68** [W14] [Benchmarks & Evaluation] [difficulty: hard]
การประเมินผล LLM ในปัจจุบันต้องเผชิญกับความท้าทายหลายด้าน ทั้งปัญหา Benchmark Contamination และความซับซ้อนของ Task ข้อใดระบุข้อเท็จจริงเกี่ยวกับชุดทดสอบและระเบียบวิธีประเมินผลได้ถูกต้องตามหลักการ?
A. ใน SWE-bench โมเดลจะได้รับการประเมินว่าแก้ไขปัญหา (Resolved) ได้สำเร็จ ก็ต่อเมื่อการทดสอบส่วนที่เคยล้มเหลวเปลี่ยนเป็นผ่าน (Fail-to-pass) และส่วนที่เคยผ่านอยู่แล้วยังคงผ่านทั้งหมด (Pass-to-pass)
B. หากตรวจพบว่าชุดข้อมูลทดสอบ MMLU มีการรั่วไหล (Contamination) ใน Training Data อยู่ 12% เราสามารถหักลบคะแนนของโมเดลออก 12% ได้โดยตรงเพื่อสะท้อนความสามารถจริง
C. ชุดทดสอบ HarmBench อาศัยการตรวจให้คะแนนโดยผู้เชี่ยวชาญที่เป็นมนุษย์ (Human Evaluation) ในการตัดสิน Attack Success Rate (ASR) เพื่อความเที่ยงตรงสูงสุด
D. คะแนนความสามารถในการแก้ปัญหา Software Engineering บน SWE-bench Lite, SWE-bench Verified และ SWE-bench Original มีมาตรฐานข้อสอบเดียวกันและสามารถนำตัวเลขมาเปรียบเทียบกันได้โดยตรง

**Q69** [W14] [LLM-as-a-Judge] [difficulty: hard]
ในการประเมิน Chatbot โดยใช้ LLM ทำหน้าที่เป็นกรรมการเปรียบเทียบคู่คำตอบ (LLM-as-a-Judge ใน MT-Bench หรือ Chatbot Arena) ข้อใดอธิบายพฤติกรรม Bias ของโมเดลผู้ตัดสินและแนวทางจัดการได้อย่างถูกต้อง?
A. ปัญหา Length Bias สามารถกำจัดได้อย่างสมบูรณ์โดยการนำคะแนนที่ได้มาหารด้วยจำนวนคำของคำตอบโดยตรง
B. หากทำการสลับตำแหน่งคำตอบ (Swap test) ระหว่าง A-B และ B-A แล้วพบว่าคำตัดสินไม่สอดคล้องกัน (Inconsistent) แสดงว่าคำตอบทั้งสองมีคุณภาพเสมอกัน (True Tie) เสมอ
C. โมเดล LLM Judge ระดับแนวหน้าอย่าง GPT-4 และ Claude-v1 มีความต้านทานต่อการใส่ข้อความเติมแต่งไร้สาระ (Padding/Filler) เท่าเทียมกัน โดยทั้งคู่ตัดสินให้คำตอบยาวชนะไม่ถึง 10%
D. LLM Judge มักมีแนวโน้มตัดสินให้คำตอบที่ยาวและมีเนื้อหาเสริมชนะ (Length bias) และชอบคำตอบในตำแหน่งแรก (Positional bias) ซึ่งการทำ Swap test ช่วยลดอคติแต่ไม่สามารถกำจัดอคติได้ทั้งหมด

**Q70** [W14] [RAG Evaluation] [difficulty: medium]
ในการประเมินระบบ Retrieval-Augmented Generation (RAG) ด้วยเฟรมเวิร์ก RAGAS ข้อใดอธิบายหลักการของมาตรวัด Faithfulness และข้อควรระวังในการนำไปใช้ได้อย่างถูกต้อง?
A. หากคำตอบของระบบได้คะแนน Faithfulness = 1.0 จะเป็นการรับประกันว่าคำตอบนั้นถูกต้องตรงตามความจริงของโลก (Factual truth) อย่างแน่นอน
B. Faithfulness วัดว่าประเด็นหรือข้อความ (Claims) ในคำตอบถูกสนับสนุนโดย Context ที่ดึงมาหรือไม่ โดยเป็นมาตรวัดแบบ Reference-free ที่ไม่ต้องพึ่งพาคำตอบเฉลยจากมนุษย์
C. RAGAS ประเมิน Faithfulness ด้วยการวัด Cosine similarity ระหว่าง Question embedding และ Answer embedding เป็นหลัก
D. ในระบบจริง การใช้ Heuristic ตรวจสอบความสอดคล้องของตัวเลขและคำศัพท์สามารถแทนที่ LLM Entailment ได้อย่างสมบูรณ์ เพราะสามารถจับความหมายเชิงปฏิเสธ (Negation) ได้อย่างแม่นยำ

**F9** [W14] [Scaling Law & Data Compute-Optimal]
ตามผลการศึกษา Scaling Law ของ Chinchilla (Hoffmann et al., 2022) สัดส่วน compute-optimal ระบุว่าควรใช้ปริมาณข้อมูลการฝึกฝนประมาณ ___ tokens ต่อ 1 model parameter ดังนั้น สำหรับโมเดลภาษาขนาด 175B parameters จึงควรใช้ปริมาณ pre-training data ประมาณ ___ tokens

**F10** [W14] [Deduplication Algorithms]
ในการลดความซ้ำซ้อนของข้อมูล เทคนิคที่สร้าง fingerprint ขนาด 64-bit และรักษาระยะเชิงมุมหรือค่า Cosine Similarity ผ่านการวัด Hamming Distance คือ ___ ในขณะที่เทคนิคที่ใช้ประมาณค่า Jaccard Similarity ของเซต shingle ผ่านการสุ่ม hash functions คือ ___

### Answer key
- **Q61**: C — LSH banding กรองคู่เพื่อลดการเปรียบเทียบจาก $O(N^2)$ แต่เอกสารที่ผ่านยังเป็นเพียง candidate ที่ต้อง verify ด้วย exact Jaccard อีกครั้ง (Section 3.3, 4, 7 trap 5 & 7).
- **Q62**: A — MinHash/LSH ทำงานเปรียบเทียบระหว่างคู่เอกสาร (pairwise) จึงจับการซ้ำในเอกสารเดียวไม่ได้ ต้องใช้ repetition heuristics ของ Gopher เข้ามาช่วย (Section 4 dedup pipeline, 7 trap 10).
- **Q63**: C — ข้อความซ้ำซาก/cookie spam มี PPL ต่ำมากเพราะทำนายง่าย ขณะที่ศัพท์เฉพาะทางอาจตกไปอยู่ tail และ CCNet แบ่ง 3 bucket ตามขนาดข้อมูลโดยไม่มี fixed cut-off (Section 4 perplexity demo, 7 trap 1 & 2).
- **Q64**: B — ต้นทุนรัน A ก่อน B คือ $c_A + p_A c_B = 0.1 + 0.3(10) = 3.1\text{ ms/doc}$ ซึ่งต่ำกว่ารัน B ก่อน A คือ $10 + 0.5(0.1) = 10.05\text{ ms/doc}$ (Section 3.7, 7 trap 11).
- **Q65**: B — TPC ที่ต่ำมากของ BERT เกิดจาก coverage failure ที่ไม่มีคำศัพท์ภาษาไทยใน vocab ทำให้กลายเป็น `[UNK]` ทั้งหมด ไม่ใช่การบีบอัดที่ดี (Section 3.5, 4, 7 trap 14).
- **Q66**: D — BLEU อิง n-gram surface matching จึงให้คะแนน word salad สูง ส่วน BERTScore แก้ไขโดยเทียบ contextual embeddings ผ่าน greedy matching บน cosine similarity matrix (Section 3.9, 4, 7 trap 18 & 19).
- **Q67**: C — คำนวณจากสูตรกรณีพิเศษ $c = 1$: $pass@k = k/n = 100/200 = 0.500$ (Section 3.1, 7 trap 23).
- **Q68**: A — SWE-bench ตัดสินว่า resolved เมื่อผ่านเทสต์ชุด fail-to-pass และยังคงผ่านชุด pass-to-pass ครบถ้วน (Section 4 pass@k SWE-bench, 6 item 16, 7 trap 24 & 25).
- **Q69**: D — LLM Judge มักมี length bias และ positional bias ซึ่งการทำ swap test ช่วยลดอคติลงได้แต่ไม่สามารถกำจัดให้หมดสิ้น (Section 3.10 mock judge, 7 trap 21 & 22).
- **Q70**: B — RAGAS Faithfulness เป็น reference-free metric ที่วัดการ entailment ระหว่าง claims ของคำตอบกับ context เท่านั้น โดยไม่ได้รับประกัน factual truth ของ context (Section 4 RAGAS proxy, 7 trap 27).
- **F9**: 20, 3.5T (หรือ 3.5 ล้านล้าน) — อัตราส่วน Chinchilla คือ 20 tokens ต่อ parameter โดย $175\text{B} \times 20 = 3,500\text{B} = 3.5\text{T tokens}$ (Section 3.4, 6 item 1).
- **F10**: SimHash, MinHash — SimHash รักษา Cosine similarity ผ่าน Hamming distance ส่วน MinHash ประมาณค่า Jaccard similarity (Section 6 item 10, 7 trap 4).

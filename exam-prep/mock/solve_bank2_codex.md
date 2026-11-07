# Independent solutions — bank 2

n08-X4-01 | B | med | ok | src=51, BART tgt=62, T5 tgt=17; (51²+62²)/(51²+17²)=2.23 ภายใต้สัมประสิทธิ์เท่ากัน
n08-X4-02 | D | high | easier | ใช้ prefix แยกงานและสร้าง label หรือคะแนนเป็น text
n08-X4-03 | A | high | easier | permutation เรียงประโยค, deletion ไม่ทิ้งตำแหน่ง, S เป็น causal continuation
n08-X4-04 | C | high | ok | 11B/2=5.5B มากกว่า 4.8B; ผล zero-shot ไม่อธิบายด้วย decoder size อย่างเดียว
n08-X4-05 | B | high | ok | 12×(4×120+1+2+3+4)=5,880
n08-X4-F1 | 4,222,260; 1,088,000 | high | ok | BART 2×42×50,265; T5 2×17×32,000
n09-X4-01 | B | high | easier | negative ต่อ query เพิ่มจาก 31 เป็น 255; 255/31≈8.23
n09-X4-02 | D | high | ok | cosine ที่เหมาะสุดคือ 1,1,1,0; MSE=(16+9)/4=6.25
n09-X4-03 | A | high | easier | Recall=3/4; MRR=(1/3+1)/2=0.667
n09-X4-04 | C | high | ok | d(a,p)=0.6, d(a,n)≤2; loss ต่ำสุด 0.6−2+1.5=0.1
n09-X4-05 | D | med | easier | ผลต่าง 29.85, 11.67 และ 30.68 จุด; ดูข้อจำกัดของคำว่า ต้อง ใน Issues
n09-X4-F1 | 76.8 วินาที; 0.768 วินาที; 307.2 GB | high | ok | n×d/10⁹, หาร 100 และ n×d×4/10⁹
n10-X4-01 | C | high | easier | 120,000 เกิน FP16 แต่ BF16 ครอบคลุม; ทั้งคู่ 2 bytes/parameter
n10-X4-02 | B | high | easier | model states 128 GB เกิน 80 GB; checkpointing ลด activation และเวลาเพิ่มราว 30%
n10-X4-03 | D | med | ok | TP ภายใน NVLink สองชุดเป็น replicas รองรับคำขอขนาน; ดูข้อจำกัดใน Issues
n10-X4-04 | A | high | easier | exact attention ไม่เก็บ N² ใน HBM; KV=2×32×8×128×2×32,000≈4.19 GB
n10-X4-05 | B | high | easier | TTFT=510 ms, TPS=40; prefill เป็นส่วนใหญ่และ speculative เน้น decode
n10-X4-F1 | 54.5 ล้าน; 0.68%; 6.9 GB | high | ok | 32×64×[2(4096+4096)+2(4096+1024)]=54,525,952; รวม 6.872 GB
n11-X4-01 | C | high | easier | คำสั่งแฝงในอีเมลเป็น indirect injection; จำกัดสิทธิ์และตรวจหลายชั้น
n11-X4-02 | A | med | easier | จับคู่ harmless, honest, helpful; การปฏิเสธมากไปลด helpfulness
n11-X4-03 | D | high | easier | DPO ใช้ preference pairs; GRPO ใช้ reward ที่ตรวจได้และ group baseline
n11-X4-04 | B | med | easier | DPO ใช้ chosen/rejected และ reference เพื่อเพิ่ม preference gap; ดู Issues
n11-X4-05 | C | high | easier | 0.8×300+0.2×5,000=1,240 tokens, 12.4 s; ลดจาก 50 s 75.2%
n11-X4-F5 | 0.6931; 0.3133; 2 | high | ok | −ln σ(0)=ln 2; −ln σ(1)=ln(1+e⁻¹); 0.5×2=1
n12-X4-01 | C | high | ok | starts 0 ถึง 9856 รวม 23; chunk สุดท้าย 64 ตัวอยู่ใน chunk ก่อนทั้งหมด
n12-X4-02 | A | high | ok | Q=.02500, R=.02357, S=.01742, P=.01639
n12-X4-03 | D | high | easier | 10+1+2.5k≤60 จึง k≤19.6 และใช้ได้ 19 pairs
n12-X4-04 | B | high | ok | สี่เทอมของคู่ 1–2 มี loss ln 2; หาร 2N=8 ได้ ln 2/2≈0.347
n12-X4-05 | C | med | easier | alias แยกโหนดทำให้ไม่มี p เดียวเชื่อมสอง edge; ดู Issues
n12-X4-F1 | 21 ประโยค; 20 ค่า; 2 chunks | med | ok | ภายใต้ percentile แบบ linear interpolation ค่าเดียวสูงกว่า p95; ดู Issues
n13-X4-01 | B | high | easier | P(invalid)=(e³+1)/(e³+e²+e+1)=0.676; masked P(id1)=e²/(e²+e)=0.731
n13-X4-02 | D | high | easier | SQL ถูก syntax และรันสำเร็จจึงไม่ trigger retry; ต้องตรวจสิทธิ์และอนุมัติ
n13-X4-03 | A | high | easier | time หมดที่ 60 s ระหว่าง step 9; เสร็จ 8 steps ที่ 56 s
n13-X4-04 | C | high | easier | Routing เป็น Low/Low/Medium ตามตาราง S41; ดู Issues เรื่องตัวเลือก
n13-X4-05 | B | high | easier | 92−15=77 จุด; 92.4−92=0.4 จุด; ต้องพิจารณา scaffold และ contamination
n13-X4-F1 | 81 turn; 740 tokens | high | ok | floor((8000−500−1000)/80)=81; retrieval 500+6×40=740
n14-X4-01 | C | high | easier | 1−C(8,3)/C(10,3)=1−56/120=0.5333
n14-X4-02 | A | high | easier | อังกฤษ 1,000 tokens; ไทย 11,400 tokens สำหรับเนื้อหาเดียวกัน
n14-X4-03 | D | med | ok | ceil(500/0.65)=770 และ ceil(500/0.775)=646; ดู Issues เรื่องการรับประกัน
n14-X4-04 | B | high | ok | จากสูตร X≈1,111 และ 0.36 คู่; Y≈7,846 และ 253 คู่
n14-X4-05 | C | med | ok | ตามเกณฑ์ที่ให้ P1 ติด curly brace, P2 สั้นกว่า 50, P3 ratio=0.15; ดู Issues
n14-X4-F1 | 65 B parameters; 1.7 tokens/parameter | high | easier | 1.3T/20=65B; 300B/175B≈1.7
n08-L4-06 | D | high | easier | ไม่ cache project K,V ซ้ำ 20 ครั้งแทนครั้งเดียว
n09-L3-01 | C | high | easier | 2000×1999/2×0.004/3600≈2.22 ชั่วโมง
n09-L3-03 | D | high | easier | ต้องปรับ encoder เพื่อเปลี่ยน embedding geometry; freeze ไม่รับประกันแก้ anisotropy
n09-L4-01 | D | high | easier | ปิด dropout ได้สอง view เหมือนกัน ขณะที่ negatives ยังอยู่
n09-L4-05 | D | high | easier | ทุกค่าที่ระบุ ≥0.46 สูงกว่า threshold 0.40
n09-L2-F1 | 499,500 ครั้ง; 2,497.5 วินาที | high | ok | 1000×999/2 pairs และคูณ 0.005 วินาที
n10-L4-04 | B | high | easier | Q: 3 TB/s×1.5 FLOP/byte=4.5 TFLOP/s สูงสุด
n10-L4-05 | A | high | easier | tensor scale 12.7/127=0.1; block 3 scale 1.27/127=0.01
n11-L4-04 | B | high | easier | S ห้า paths พร้อมกัน 10 s; R ห้า calls ตามลำดับ 50 s
n11-L4-06 | A | high | easier | อีเมลไม่มี ground truth แบบ math/code; reasoning ใช้ token และ latency เพิ่ม
n12-L3-01 | C | high | harder | TF ratio=(4+1.2×1.75)/(4+1.2×0.625)≈1.28
n12-L3-05 | C | high | easier | graph เชื่อม entity และ relation ข้าม chunks สำหรับ multi-hop
n12-L4-01 | A | high | easier | 4×2.5×45/(45+1.5)=9.677; เพิ่มประมาณ 1.61%
n12-L4-04 | B | high | easier | rank 70 ไม่เข้าชุด rerank 50; เพิ่มคุณภาพ reranker ไม่กู้ candidate นี้
n13-L4-02 | C | high | easier | reasoning error 47% และ search error 23% อธิบาย failure อื่นแม้ hallucination ต่ำ
n13-L4-F1 | 3,000; 500; 250 tokens | high | easier | 60×50; 4×50+300; 5×50
n14-L4-02 | A | high | easier | 175B×20=3.5T; 1.3/3.5≈37%; 15/3.5≈4.3
n14-L4-04 | A | med | easier | ให้ครึ่งคะแนนได้ 69%; ตัดออกได้ 54/70≈77.1%; ดู Issues
n14-L4-06 | C | high | easier | กฎสถิติพื้นฐานและ PPL ต่ำไม่จำเป็นต้องตัดข้อความซ้ำ; ใช้ repetition rules

## Issues

- n08-X4-01: การบอกว่าต้นทุน encoder และ decoder แปรตามกำลังสองแยกกันยังไม่กำหนดค่าคงที่สัดส่วนร่วม คำตอบ B สมมติสัมประสิทธิ์ทั้งสองเท่ากัน หากต่างกันอัตราส่วนรวมเปลี่ยนไป
- n09-X4-05: D เป็นตัวเลือกที่สอดคล้องที่สุดและเลขถูก แต่คำว่า “ต้องใช้” contrastive fine-tuning กว้างเกินหลักฐาน: สไลด์ยังกล่าวถึง post-processing เช่น whitening เป็นแนวทางอื่น ตัวเลขไม่ได้พิสูจน์ว่าจำเป็นหรือผลเพิ่มทั้งหมดเกิดจาก contrastive loss เพียงอย่างเดียว
- n10-X4-03: D เป็นตัวเลือกที่เหมาะภายใต้แนวคิด TP ภายใน node และใช้ replicas แต่โจทย์ไม่กำหนด batch, concurrency, communication cost หรือเป้าหมาย latency/throughput เชิงปริมาณ จึงพิสูจน์ไม่ได้ว่า TP8 คือ layout ที่ดีที่สุดสำหรับ workload จริง
- n11-X4-02: การขอสคริปต์เจาะช่องโหว่ Windows เพียงอย่างเดียวไม่ระบุเจตนาหรือขอบเขตการทดสอบ จึงสรุปว่าเป็นอันตรายจากโจทย์ไม่ได้แน่ชัด แม้ A เป็นการจับคู่ตามเจตนาของข้อสอบ
- n11-X4-04: B เป็นวิธีที่เหมาะที่สุด แต่ DPO loss เพิ่มช่องว่างระหว่าง chosen กับ rejected เทียบ reference ไม่ได้รับประกันว่า probability ของ rejected จะลดลงแยกเดี่ยวทุกครั้ง; สไลด์ W11 แสดงข้อความดังกล่าวเป็น intuition
- n12-X4-05: ชื่อ Sam Altman และ Altman อาจเป็นคนเดียวกัน แต่โจทย์ไม่ได้ยืนยัน entity identity หรือ labels ทั้งหมด โดยเฉพาะโหนดปลายทางต้องมี label COMPANY ตาม Cypher; entity linking จึงเป็นสาเหตุที่เป็นไปได้ ไม่ใช่ข้อสรุปที่พิสูจน์ได้จาก triple ที่ให้
- n12-X4-F1: ไม่ระบุวิธีคำนวณ percentile: linear interpolation ที่ใช้ทั่วไปให้หนึ่งค่าเหนือ p95 และ 2 chunks แต่ nearest-rank สำหรับ 20 ค่าให้ threshold เท่าค่าสูงสุด จึงไม่มีค่าที่สูงกว่าและได้ 1 chunk
- n13-X4-04: ตัด A, B และ D ได้จากระดับ latency หรือ complexity ที่ตัวเลือกเขียนไว้เอง ซึ่งขัดกับเงื่อนไขโดยตรง จึงเลือก C ได้จาก wording โดยไม่ต้องรู้ตาราง S41
- n14-X4-03: 770 และ 646 มาจากการนำอัตรา 65% และ 77.5% มาคิดเป็นค่าเฉลี่ย/สัดส่วนที่คงที่ ไม่รับประกันว่าจะได้ผู้ชนะอย่างน้อย 500 คู่จากผลจริงหากความสอดคล้องเป็นแบบสุ่ม; สัดส่วนที่ให้ยังไม่จำเป็นต้องเป็นจำนวนคู่เต็มพอดีสำหรับขนาดตัวอย่างดังกล่าว
- n14-X4-05: P1 และ P3 ไม่ระบุคุณสมบัติทุกข้อของ Gopher เช่น repetition และ line rules จึงยืนยันด่านแรกที่ตัด P1 ไม่ได้แน่นอน; C ใช้สมมติฐานว่ากฎที่ไม่ได้กล่าวถึงผ่านทั้งหมด
- n14-L4-04: ผลสลับลำดับไม่ตรงกัน 30% บ่งชี้ความไม่สม่ำเสมอ แต่เพียงข้อมูลนี้ยังแยก systematic position bias ออกจากความสุ่มหรือความไม่เสถียรของ judge ไม่ได้ ต้องมีการรันซ้ำหรือวิเคราะห์เพิ่มเติม

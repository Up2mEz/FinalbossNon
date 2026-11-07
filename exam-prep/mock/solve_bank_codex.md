n08-L1-01 | B | high | ok | หมุนให้เริ่มที่ pivot
n08-L1-02 | D | high | ok | masked self-attention ตามด้วย cross-attention และ FFN
n08-L1-03 | A | high | ok | prefix summarize:
n08-L1-04 | C | high | ok | สไลด์ระบุ 12+12, 1024, 16 heads
n08-L1-05 | B | high | ok | learned relative bias บวกใน logits
n08-L1-06 | A | high | ok | reconstruct ต้นฉบับจาก input ที่ corrupt
n08-L2-01 | C | high | easier | masking คงความยาว deletion ลด 2
n08-L2-02 | D | high | ok | ปิดด้วย sentinel ถัดไปและ EOS ตามสไลด์ 17
n08-L2-03 | A | high | ok | 1/(0.25²)=16
n08-L2-04 | B | high | easier | 20/8−1.5=1
n08-L2-05 | C | high | ok | ใช้ final decoder state กับ classifier
n08-L2-06 | D | med | ok | ใช้สมมติฐานแบ่ง parameters ครึ่งต่อครึ่งของสไลด์
n08-L3-01 | A | high | ok | source 30−9+3=24; target 9+4+1=14
n08-L3-02 | C | high | easier | rotation ไม่ได้ลบ token
n08-L3-03 | B | high | ok | ลำดับโมเดลต่างปีไม่ใช่ controlled comparison
n08-L3-04 | D | high | easier | bias บวกก่อน softmax
n08-L3-05 | C | high | ok | 8×12×64×16=98,304
n08-L3-06 | A | high | easier | extractive pointer คืน span จาก context
n08-L4-01 | D | high | ok | 10−1+3=12
n08-L4-02 | B | high | easier | target เต็มทำให้ self-attention ยังคงกำลังสองของ N
n08-L4-03 | A | high | ok | 1.5/(0.5+0.04)=2.78
n08-L4-04 | B | high | ok | 1,110,000/605,000≈1.83
n08-L4-05 | C | high | easier | 400×0.15×0.8=48; 400×0.2=80
n08-L4-06 | D | high | easier | ไม่ cache ต้อง project ซ้ำ 20 ครั้ง
n08-L1-F1 | 1,836 tasks; 101 ภาษา | high | ok | สไลด์ระบุทั้งสองค่า
n08-L2-F2 | Poisson; 3; แทรก | high | ok | zero-length span เป็น insertion
n08-L3-F3 | 277,500; 302,500 | high | ok | แทนค่าตามสูตรที่กำหนด
n08-L4-F4 | 3.47 | high | ok | 52/(10+4+1)=52/15
n09-L1-01 | C | high | ok | 3×768=2,304
n09-L1-02 | A | high | ok | ดึง positive และผลัก negative
n09-L1-03 | D | high | ok | ใช้ weights ร่วมกัน
n09-L1-04 | B | high | ok | mean pooling โดย mask padding
n09-L1-05 | C | high | ok | inner product หลัง normalize เท่ากับ cosine
n09-L1-06 | A | high | easier | 1/4=0.25
n09-L2-01 | B | high | ok | encode n ครั้งและเปรียบเทียบทุกคู่
n09-L2-02 | D | high | easier | max(0,0.4−0.7+0.5)=0.2
n09-L2-03 | A | med | ok | MLM ไม่ได้ปรับ geometry ระดับประโยคโดยตรง
n09-L2-04 | C | high | ok | แทน d=1−sim และตัดค่าคงที่
n09-L2-05 | D | high | ok | 2,000,000×384×4=3.072 GB
n09-L2-06 | A | high | ok | positive อื่นใน batch มี 127 ตัว
n09-L3-01 | C | high | ok | 1,999,000×0.004/3600≈2.22 ชั่วโมง
n09-L3-02 | B | high | ok | 0.3<0.5<0.8 และ loss=0.3
n09-L3-03 | D | med | ok | freeze encoder ไม่ปรับ representation โดยตรง
n09-L3-04 | A | high | easier | ความเร็ว ANN ต้องรายงานร่วมกับ recall
n09-L3-05 | B | high | ok | cosine เลือก d1; dot และ L2 เลือก d2
n09-L3-06 | C | high | ok | u=[3,6], v=[4,4], absolute difference=[1,2]
n09-L4-01 | D | med | ok | ปิด dropout ทำให้ positive views เหมือนกัน
n09-L4-02 | A | high | ok | cosine losses เป็น 0; Euclidean 1+√0.4−√2≈0.218
n09-L4-03 | B | high | ok | 5,000,000/507≈9,862
n09-L4-04 | C | high | ok | 3×3×(1024−768)=2,304 weights
n09-L4-05 | D | high | easier | ทุกค่าที่อยู่ในช่วงที่ระบุสูงกว่า 0.40
n09-L4-06 | B | high | ok | 20M×768×4=61.44 GB ก่อน index overhead
n09-L1-F1 | 2,304 | high | easier | 3×768
n09-L2-F1 | 499,500; 2,497.5 | high | ok | n(n−1)/2 แล้วคูณ 0.005 วินาที
n09-L3-F1 | 0.5; 1.0; 0.3; semi-hard | high | ok | 0.5<1.0<0.5+0.8
n09-L4-F1 | 0.9600; 0.9692; 0.5092 | high | ok | 24/25 และ 63/65
n10-L1-01 | B | high | ok | 13B×2 bytes=26 GB
n10-L1-02 | C | high | ok | 2^30 bytes
n10-L1-03 | A | high | ok | prefill ประมวลผล prompt ก่อน token แรก
n10-L1-04 | D | high | ok | temperature สูงทำให้ softmax แบนลง
n10-L1-05 | B | high | ok | unified memory page ไป CPU RAM
n10-L1-06 | C | high | ok | target model ตรวจ candidates พร้อมกัน
n10-L2-01 | D | high | easier | 3×2^30 bytes เท่ากับ 3 GiB หรือ 3.22 GB
n10-L2-02 | A | high | ok | 8/32=1/4 และ 32/8=4
n10-L2-03 | B | high | ok | IO tiling, paging และ dynamic batching ตามลำดับ
n10-L2-04 | B | high | ok | 16×(1024+4096)=81,920
n10-L2-05 | A | high | ok | 47B×2=94 GB; active ประมาณ 13B
n10-L2-06 | D | high | ok | shared ใช้ทุก token ส่วน routed ถูกเลือก
n10-L3-01 | C | high | ok | 23.5+0.54+1.46=25.5 GB
n10-L3-02 | A | high | ok | KV ต่อ request 0.5369 GB; floor(6.54/0.5369)=12
n10-L3-03 | D | high | ok | optimizer 12N=156 GB; รวม 16N=208 GB; ceil(208/80)=3
n10-L3-04 | B | high | easier | QLoRA freeze base weights
n10-L3-05 | B | high | ok | round(1.5/0.0252)=60; 60×0.0252=1.512
n10-L3-06 | C | high | ok | 16000²×2×32=16.384 GB
n10-L4-01 | D | high | ok | 8.2−8=0.2 GiB≈0.215 GB
n10-L4-02 | A | high | ok | 32×8×(8192+5120)=3,407,872≈0.043%
n10-L4-03 | D | high | easier | base weights เป็นส่วนหลักของ memory
n10-L4-04 | B | high | ok | Q มี 3×1.5=4.5 TFLOP/s สูงสุด
n10-L4-05 | A | low | ok | A ถูกเมื่อ block 3 มี absmax เท่ากับ 1.27 ตามที่ตัวเลือกสมมติ
n10-L4-06 | C | high | ok | floor(12×0.30/0.3)=12; floor(12×0.95/0.3)=38
n10-L1-F1 | 12 | high | easier | 16−2−2=12
n10-L2-F1 | 130.4 | high | ok | 140×10^9/2^30≈130.39
n10-L3-F1 | 0.02; 45; 0.90 | high | ok | 2.54/127=0.02 และ round(45.2)=45
n10-L4-F1 | 131,072; 16.78; 67.11 | high | ok | 2×2×32×8×128 bytes/token; MHA สี่เท่า
n11-L1-01 | C | high | ok | system กำหนดพฤติกรรมระดับสูง
n11-L1-02 | A | high | ok | เพิ่ม assistant header เพื่อเริ่ม generation
n11-L1-03 | D | high | ok | reward head ให้ scalar
n11-L1-04 | B | high | ok | generate แล้ว critique และ refine
n11-L1-05 | A | high | ok | วาง input ระหว่าง header กับ footer
n11-L1-06 | D | high | ok | มีตัวอย่างสองคู่
n11-L2-01 | B | high | ok | logit=0.5×2=1; loss=ln(1+e^-1)
n11-L2-02 | A | high | ok | empathetic คือ tone; formal คือ style ตามกรอบ
n11-L2-03 | D | high | ok | zero-shot ใช้วลี; few-shot ให้ตัวอย่าง reasoning
n11-L2-04 | C | high | ok | ln(1+e^-2)≈0.1269
n11-L2-05 | A | high | ok | base model เรียนการต่อข้อความ
n11-L2-06 | C | high | ok | ใช้ template และ stop token ที่สอดคล้องกับการฝึก
n11-L3-01 | D | high | ok | chosen ratio=3 rejected ratio=−4; loss=ln(1+e^-0.7)
n11-L3-02 | B | high | ok | graph รองรับ merge และวนแก้
n11-L3-03 | D | high | ok | mean=5, sigma=√5, 3/√5≈1.342
n11-L3-04 | A | high | ok | examples, reasoning, executable calculation ตามลำดับ
n11-L3-05 | B | high | ok | KV ×3 และสมมติ compute ×9
n11-L3-06 | C | high | easier | ข้อความสลับ teacher forcing กับ inference
n11-L4-01 | A | high | ok | relative margin=6; ln(1+e^-0.6)=0.4375
n11-L4-02 | C | high | ok | 26 GB ต่อ model; PPO 4, GRPO 3, DPO 2
n11-L4-03 | D | high | ok | plurality คือ 18; sampling เจ็ดครั้งใช้ 2,100 tokens
n11-L4-04 | B | high | ok | 5 parallel calls เทียบกับ 5 sequential calls
n11-L4-05 | C | high | ok | mean=1.5, sigma=1.5√3; advantage=√3
n11-L4-06 | A | med | ok | reward เชิงคุณภาพอีเมลไม่ตรงกับ rule-checkable math/code
n11-L1-F1 | In-context learning; ICL | high | ok | เรียนจาก prompt โดยไม่ปรับ weights
n11-L2-F2 | Critique; วิจารณ์หรือประเมิน | med | ok | สร้างร่างแล้วตรวจและปรับปรุง
n11-L3-F3 | 0.5; 0.4741 | high | ok | 0.1×5 และ ln(1+e^-0.5)
n11-L4-F4 | 0.4; −2; 0.5 | high | ok | mean=0.8 และ population variance=0.16
n12-L1-01 | B | high | ok | b ควบคุม length normalization
n12-L1-02 | C | high | ok | lexical mismatch ทำให้พลาด paraphrase
n12-L1-03 | A | high | ok | linear head ส่ง scalar logit
n12-L1-04 | D | high | ok | graph triple คือ subject relation object
n12-L1-05 | B | high | ok | ตำแหน่งกลางต่ำสุดในตัวอย่าง
n12-L1-06 | C | high | ok | ตัดเมื่อ semantic distance เกิน threshold
n12-L2-01 | D | high | easier | ceiling=k1+1=4
n12-L2-02 | A | high | ok | TF จาก 1 เป็น 10/7 เพิ่ม 42.9%
n12-L2-03 | B | high | ok | (1/61−1/62)/(1/61)=1/62≈1.61%
n12-L2-04 | C | high | ok | MaxSim เป็น late interaction
n12-L2-05 | D | high | easier | สรุป theme ข้าม corpus เหมาะ global
n12-L2-06 | A | high | ok | top 1 ต้น top 2 ท้าย
n12-L3-01 | C | high | ok | TF ratio=6.1/4.75≈1.284
n12-L3-02 | B | high | ok | Z=2/63, Y=1/64+1/66, X=1/61
n12-L3-03 | D | high | ok | max(5,10)+1+200×20/80=61 ms
n12-L3-04 | A | high | easier | เป็น attribute, negation และ relation failures
n12-L3-05 | C | med | ok | graph traversal ช่วยเชื่อมความสัมพันธ์หลาย hop
n12-L3-06 | B | high | easier | เพิ่ม context จำนวนมากไม่รับประกันผลดี
n12-L4-01 | A | med | ok | ตามสูตร 4×2.5×300/301.5≈9.95
n12-L4-02 | D | high | ok | TF ที่ b=1 คือ 2, 1.667, 0.714
n12-L4-03 | C | high | ok | 2/63>1/61 แต่ 2/3<1
n12-L4-04 | B | high | easier | อันดับ 70 ไม่เข้าสู่ candidate pool
n12-L4-05 | D | high | ok | hybrid สำหรับ exact และ semantic; graph สำหรับความสัมพันธ์
n12-L4-06 | A | high | ok | loss=ln(1+3e^(-0.1/τ))
n12-L1-F1 | 0; N/2 | high | ok | เศษและส่วนเท่ากันเมื่อ n_t=N/2
n12-L2-F1 | recall; precision | high | ok | retriever ครอบคลุม reranker เรียงความเกี่ยวข้อง
n12-L3-F1 | 1.571; 2.2 | high | ok | 3×2.2/4.2 และ ceiling=k1+1
n12-L4-F1 | 61; 62 | high | ok | 2/(60+r)>1/61 เมื่อ r<62
n13-L1-01 | C | high | ok | max_steps ป้องกัน loop ไม่สิ้นสุด
n13-L1-02 | A | high | ok | schema เป็นแบบฟอร์มว่างระบุข้อกำหนด
n13-L1-03 | D | high | ok | regex compile เป็น DFA
n13-L1-04 | B | high | ok | ความผิดพลาดต้นทางส่งต่อปลายทาง
n13-L1-05 | A | high | ok | semantic memory เก็บข้อเท็จจริง
n13-L1-06 | D | high | ok | escalate ให้มนุษย์อนุมัติ
n13-L2-01 | B | high | easier | 27.4−29.4=−2; 60.9−56.3=+4.6 จุด
n13-L2-02 | B | high | ok | ไม่มีชื่อใน registry จึง tool not found
n13-L2-03 | C | high | ok | ตารางระบุ routing
n13-L2-04 | C | high | ok | จำนวน valid token เปลี่ยนตาม parser state
n13-L2-05 | A | high | ok | arguments ต้องตรง properties และ required
n13-L2-06 | D | high | ok | similarity retrieval อาจพลาดข้อมูลสำคัญตามเวลา
n13-L3-01 | C | high | ok | search result error อยู่ฝั่ง ReAct เท่านั้นในตาราง
n13-L3-02 | D | high | ok | ต้องดึงสถานะปัจจุบันต่อคำขอ
n13-L3-03 | B | high | ok | (50257−69)/(50257−50146)=50188/111≈452
n13-L3-04 | A | high | ok | debate: medium latency และ complexity, high fault tolerance
n13-L3-05 | D | high | ok | summary เก็บภาพรวมและ retrieval ดึงรายละเอียด
n13-L3-06 | A | high | ok | regex รูปแบบอย่างเดียวไม่ตรวจวันตามปฏิทิน
n13-L4-01 | B | high | ok | floor((124000−15220)/2110)=51; step 61
n13-L4-02 | C | low | ok | reasoning และ search failures อาจหักล้างประโยชน์ด้าน hallucination
n13-L4-03 | A | high | ok | ปิด array ก่อน object ชั้นในและนอก
n13-L4-04 | D | high | ok | bounded depth ทำ finite-state ได้; unbounded nesting ต้องใช้ stack
n13-L4-05 | B | high | ok | summary อาจทิ้งข้อมูลที่จำเป็นต่อ transition
n13-L4-06 | C | high | ok | sequential 26 s; parallel debate 9+4=13 s
n13-L1-F1 | description; parameters | high | ok | description บอกการใช้ parameters ระบุ schema
n13-L2-F1 | HotpotQA; FEVER | high | ok | ตารางเทียบ EM และ accuracy ตามลำดับ
n13-L3-F1 | 5; 2; 50,188 | high | ok | 62+5+2=69; 50,257−69=50,188
n13-L4-F1 | 3,000; 500; 250 | med | ok | 60×50; 4×50+300; 5×50
n14-L1-01 | B | high | ok | สไลด์ระบุ alpha-word ratio อย่างน้อย 0.8
n14-L1-02 | C | high | easier | 100/20=5
n14-L1-03 | A | high | ok | จัดอยู่ใน era semantic matching และ preference
n14-L1-04 | D | high | ok | MMLU ครอบคลุม 57 subjects
n14-L1-05 | C | high | ok | ลำดับมีผลต่อการตัดสิน
n14-L1-06 | B | high | ok | สไลด์ระบุ monthly
n14-L2-01 | D | high | easier | 7/10=0.70
n14-L2-02 | A | high | ok | near-duplicate ใช้ shingles, MinHash, LSH แล้วตรวจจริง
n14-L2-03 | C | high | ok | Dolma รวมหลายแหล่งและมี per-document logs
n14-L2-04 | B | high | ok | max ต่อ reference คือ 0.8, 0.9; เฉลี่ย 0.85
n14-L2-05 | D | high | ok | 1−C(19,5)/C(20,5)=5/20=0.25
n14-L2-06 | A | high | ok | ASR บนชุดโจมตีไม่วัดความสามารถตอบคำขอปกติ
n14-L3-01 | A | high | ok | B ก่อน: 3+0.2×2=3.4; A ก่อน 4.7
n14-L3-02 | D | high | ok | 2²×3²=36; signature เทียบ H ตำแหน่ง
n14-L3-03 | B | high | ok | ไม่ทราบคะแนนแยก subset จากคะแนนรวม
n14-L3-04 | C | high | ok | ควบคุมข้อมูลแล้วเปลี่ยนความยาวเพื่อทดสอบ
n14-L3-05 | B | high | ok | คะแนนคนละ metric เทียบอัตราส่วนตรงไม่ได้
n14-L3-06 | C | high | ok | (1/2+2/3+3/5)/3≈0.589
n14-L4-01 | D | high | ok | 16^(-1/8)≈0.707; P≈0.613
n14-L4-02 | A | high | ok | 175B×20=3.5T; 1.3/3.5≈37%; 15/3.5≈4.29
n14-L4-03 | D | high | ok | (0+1+(1−21/45))/3≈0.511
n14-L4-04 | A | med | ok | tie half-credit 69%; ตัดออก 54/70≈77.1%
n14-L4-05 | B | high | ok | floor((128000−80500)/2110)=22
n14-L4-06 | C | med | ok | repetition filter ตรวจซ้ำภายในเอกสาร
n14-L1-F1 | 176; 0.65 | high | ok | ค่าจากสไลด์ language identification
n14-L2-F1 | 0.72 | high | easier | 2×0.9×0.6/(0.9+0.6)
n14-L3-F1 | 4; 0.42 | high | ok | 128/32=4; (1/32)^(1/4)≈0.4204
n14-L4-F1 | 0.03125; 1,024 | high | ok | 1/(2×16); ลด SE ครึ่งหนึ่งต้องเพิ่ม H สี่เท่า

## Issues

- n08-L2-06: การแบ่ง 400M เป็น encoder และ decoder ฝั่งละประมาณ 200M เป็นแบบจำลองอย่างหยาบตามสไลด์ ไม่ได้ตามจากจำนวนพารามิเตอร์รวมที่เท่ากันเพียงอย่างเดียว; decoder มี cross-attention เพิ่มเติมด้วย
- n08-L3-06: ตัวเลือก B ขัดกันเองว่า decoder ถูกบังคับให้ copy เท่านั้น แต่ paraphrase ได้อิสระ; C อ้างว่าไม่มีทางออกนอก context และ D อ้างว่า extractive pointer hallucinate จึงตัดทิ้งได้จากถ้อยคำและนิยาม ทำให้ A เด่นโดยแทบไม่ต้องวิเคราะห์สถานการณ์
- n09-L3-03: A ใช้คำว่า precompute และใช้ซ้ำได้ “เสมอ” โดยไม่ระบุว่าข้อความและ encoder คงที่; ถ้าเปลี่ยนเอกสารหรือ encoder ต้องคำนวณใหม่ จึงควรจำกัดเงื่อนไข นอกจากนี้ B ใช้คำ cross-attention กับ cross-encoder ซึ่งปกติเป็น joint self-attention ระหว่าง tokens ของทั้งคู่
- n09-L4-01: D ควรระบุว่า positive pair ที่เหมือนกันไม่มีสัญญาณ alignment ที่เป็นประโยชน์ ไม่ใช่ไม่มี gradient หรือการเรียนรู้เลย เพราะ negatives ยังมีส่วนใน loss
- n09-L4-05: ช่วง cosine ที่สไลด์ยกเป็นตัวอย่างไม่ใช่หลักประกันสำหรับคู่ประโยค raw BERT ทุกคู่ทั่วไป; คำตอบ D ใช้ได้ภายใต้ช่วงที่โจทย์สมมติให้เท่านั้น
- n10-L4-05: ไม่มีตัวเลือกที่รับประกันว่าถูก: ค่าใน block 3 มีขนาด “ไม่เกิน” 1.27 ไม่ได้แปลว่า absmax เท่ากับ 1.27 และ q จึงไม่จำเป็นต้องเป็น 4. A เป็นคำตอบที่ตั้งใจหากเพิ่มเงื่อนไขว่า absmax ของ block 3 เท่ากับ 1.27 และ absmax ทั้ง tensor เท่ากับ 12.7
- n11-L4-06: A เป็นแนวทางที่ใกล้ที่สุด แต่กล่าวว่าอีเมล “ไม่มี ground truth ที่ rule ตรวจได้” กว้างเกินไป: บางแง่มุมกำหนดเกณฑ์หรือ ground truth ได้ แม้ความสุภาพและประโยชน์โดยรวมจะตรวจด้วยกฎได้ยาก และสไลด์ยกตัวอย่าง reward ที่ตรวจได้สำหรับ math/code
- n12-L3-05: C เป็นคำตอบที่ใกล้ที่สุด แต่การใช้ Text-to-Cypher ไม่ใช่วิธีเดียวหรือเงื่อนไขจำเป็นในการเชื่อมหลาย hop และ graph ที่ขาดข้อมูลก็ไม่รับประกันคำตอบ
- n12-L4-01: โจทย์กำหนดเอกสารยาว 100 คำแต่มี deep และ learning อย่างละ 300 ครั้ง รวม 600 คำ จึงเป็นเอกสารที่เป็นไปไม่ได้ภายใต้การนับคำเดียวกัน; A เป็นผลแทนตัวเลขในสูตรเชิงสมมติ
- n12-L4-04: A และ C ขัดกับเงื่อนไขว่าตัว reranker เห็นเฉพาะ top 50 ขณะที่เอกสารอยู่อันดับ 70; B จึงเด่นจากถ้อยคำโดยไม่ต้องรู้รายละเอียด cross-encoder
- n13-L4-02: ไม่มีตัวเลือกถูกครบถ้วน: C บอกว่าเป็น failure “ที่ CoT ไม่มี เช่น reasoning error 47%” แต่ในตัวเลือกเองระบุ CoT 16% และสไลด์ 15 ก็ระบุว่า CoT มี reasoning error. C น่าจะเป็นคำตอบที่ตั้งใจหากแยก reasoning error ที่มากกว่าออกจาก search-result error ซึ่งไม่มีใน CoT; สัดส่วนจาก error analysis อธิบายได้เพียงกลไกที่เป็นไปได้ ไม่พิสูจน์สาเหตุของผลคะแนนโดยตรง
- n13-L4-F1: จำนวน turn กับ message ไม่ได้กำหนดความสัมพันธ์กันชัดเจน; 3,000 และ 500 สมมติหนึ่ง message ต่อ turn ถ้า turn หมายถึงคู่ user/assistant ที่แต่ละ message ยาว 50 token จะได้ Buffer 6,000 และ Summary 700 ส่วน Retrieval 250 เพราะระบุ 5 messages
- n14-L3-04: การเพิ่มข้อความซ้ำทดสอบความชอบต่อคำตอบที่ยาวขึ้นในตัวอย่างนั้นได้ แต่ผลเพียงอย่างเดียวไม่พิสูจน์ว่าความยาวเป็นสาเหตุเดียวที่ Alpha ชนะ Beta ทุกครั้ง
- n14-L4-02: ตัวเลือก A คำนวณ 2.7 รอบได้ถูกในฐานะจำนวนรอบสมมติ แต่โจทย์กำหนดว่าไม่วนข้อมูลซ้ำ จึงควรเขียนให้ชัดว่าเป็นจำนวนรอบที่ต้องใช้หากยอมผ่อนเงื่อนไขนั้น
- n14-L4-04: 69% เป็นคะแนนเฉลี่ยเมื่อให้ครึ่งคะแนนแก่คู่ inconsistent ไม่ใช่สัดส่วนคู่ที่ Alpha ชนะจริง และ 30% inconsistent อาจเกิดจากความไม่เสถียรในการสุ่มหรือเหตุอื่นด้วย จึงบ่งชี้ความไวต่อลำดับได้แต่ไม่พิสูจน์ positional bias โดยลำพัง
- n14-L4-06: การซ้ำมากไม่ได้รับประกันว่า perplexity ต่ำโดยไม่ทราบประโยค โมเดลอ้างอิง และ threshold; C สอดคล้องกับบทเรียนเรื่องข้อจำกัดของตัวกรอง แต่ต้องระบุค่า PPL หรือสมมติว่าอยู่ใต้ cutoff หากต้องการยืนยันเหตุผลในตัวเลือก

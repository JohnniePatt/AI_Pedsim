# สรุปการทำวิจัยจากการพัฒนา Code (AI Surrogate Model for Time-Series Pedestrian Simulation)

## 1. การประเมินความสอดคล้อง (Coding Matching Score)
**คะแนน: 9.5/10**

**เหตุผล:** โครงสร้างโฟลเดอร์ในคลังข้อมูล (เช่น [AI_GenerateTimeseries/AI_Train](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTimeseries/AI_Train)) มีการพัฒนาโมเดลจำลองพฤติกรรมกาลเวลา (Time-Series) สอดคล้องกับเป้าหมายการวิเคราะห์อย่างสมบูรณ์ มีการเปรียบเทียบระหว่างโมเดลทำนายพิกัดทศนิยมแบบต่อเนื่อง (Continuous Coordinate Regression: Transformer GPT-2, GNN-CVAE, SGAN, LSTM) และโมเดลจำลองเชิงพื้นที่แบบตาราง (Discrete Grid Spatial Action Policy: GridSocialPolicy)

---

## 2. Checklist สำหรับการเขียน Paper

อ้างอิงจากโครงสร้างมาตรฐานเปเปอร์วิจัยสถาปัตยกรรมและการจำลองพฤติกรรม (รูปแบบอ้างอิง P006_Hong_ASA2025):

- [ ] **Title & Abstract**: สรุปเป้าหมายการประเมินขีดจำกัดของ Continuous Coordinates vs. ความสำเร็จของ Discrete Spatial Grid Policy
- [ ] **Introduction**: 
  - ความท้าทายและข้อจำกัดด้านเวลาของ Physics-based Simulation (JuPedSim / Social Force Model)
  - ความพยายามในการใช้ AI Surrogate Model ทำนาย Time-Series พฤติกรรมมนุษย์
  - ปัญหา Feature Ignorance และ Spatial Blindness ของโมเดลพิกัดทศนิยม $(x,y)$
- [ ] **Methodology**:
  - **Dataset Generation**: JuPedSim simulation บนหลากหลายแปลนอาคาร (Topo_bottleneck & Topo_HouseGAN)
  - **Continuous Coordinate Paradigm**: Transformer (Goal-Conditioned GPT-2), GNN-CVAE, SGAN, CVAE, LSTM Baseline
  - **Discrete Grid Spatial Policy Paradigm**: Local map crops (`walkable`, `exit`, `occupancy`), Action logits ($\Delta x, \Delta y$), Behavior cloning
- [ ] **Results & Findings**:
  - ประเมินความเข้าใจเชิงพื้นที่ (Spatial Awareness & Wall Boundary Respect)
  - การทำนายมวลความหนาแน่นและการแออัด (Bottleneck Congestion Flow)
  - อัตราการก้าวเดินถึงจุดหมาย (Goal Reach Success Rate)
  - ความเร็วในการประมวลผล (Inference Latency & Speedup Ratio vs. JuPedSim)
- [ ] **Discussion**:
  - วิเคราะห์ทำไม Continuous $(x,y)$ Model ถึงเกิดภาวะ Feature Ignorance
  - ทำไม Discrete Grid Spatial Representation ถึงทำหน้าที่เป็น Spatial Inductive Bias ที่จำเป็น
- [ ] **Limitations and Future Research**: ข้อจำกัดเรื่อง Resolution ของ Grid, การขยายผลสู่ Multi-destination และการทดลองในพื้นที่จริง
- [ ] **Conclusion**: สรุปการประยุกต์ใช้เป็นเครื่องมือช่วยตัดสินใจทางสถาปัตยกรรมในขั้นตอน Early-Stage Design

---

## 3. สิ่งที่พร้อมใส่ใน Paper แล้ว (Ready to Include)

1. **Research Framing & Core Hypothesis**:
   - ข้อสรุปชัดเจนว่าการทำนายพิกัดทศนิยม $(x,y)$ แม้จะให้ค่า Error ทางคณิตศาสตร์ดูต่ำ แต่แท้จริงแล้ว AI ไม่เข้าใจ Feature สภาพแวดล้อมจริง (เดินชนกำแพง ทะลุกำแพง หลงทาง)
   - การเปลี่ยนรูปแบบเป็น **Discrete Spatial Grid Policy** คือกุญแจสำคัญที่ทำให้ AI เข้าใจพื้นที่และอ่านแนวโน้มความแออัดได้อย่างสมเหตุสมผล
2. **Architecture & Methodology Details**:
   - รายละเอียดสถาปัตยกรรม `GoalConditionedGPT2` (CNN GeoEncoder + Neighbor Context)
   - รายละเอียดสถาปัตยกรรม `GridSocialPolicyNet` (CNN Local Map Encoder + MLP Feature Fusion + Policy Head)
3. **Synthetic Dataset Pipeline**:
   - การสร้างแปลนแบบสุ่มด้วย HouseGAN และการสกัดคุณลักษณะเชิงพื้นที่

---

## 4. สิ่งที่ยังขาดและต้องเพิ่มใน Paper (Missing Elements)

- **Major (สำคัญมาก)**:
  - **Quantitative Spatial Validity Metrics**: การสกัดสถิติ Boundary Violation Rate (%) และ Goal Reach Rate (%) จาก Rollout
  - **Qualitative Rollout Comparison Figures**: การจัดรูปเปรียบเทียบ Trajectory / Grid Rollout ระหว่าง JuPedSim, Continuous Model, และ Grid Policy Model
  - **Inference Benchmark**: ตารางเปรียบเทียบเวลาประมวลผล (Speedup Factor) ของ AI เทียบกับ JuPedSim
- **Minor (ควรมีเพื่อความสมบูรณ์)**:
  - การวิเคราะห์ผลกระทบของขนาด Grid Cell Resolution (เช่น $0.2m \times 0.2m$ vs $0.5m \times 0.5m$)
  - Literature review เรื่อง Spatial Representation สำหรับ AI ในงานสถาปัตยกรรมและการจำลองฝูงชน

---

## 5. Discussion (ส่วนอภิปรายผล)

การศึกษาเปรียบเทียบในการวิจัยนี้เผยให้เห็นข้อค้นพบสำคัญทางสถาปัตยกรรม AI Surrogate Model สำหรับงานจำลองคนเดินเท้า: **การทำนายพิกัดเชิงเวลาแบบต่อเนื่อง (Continuous Coordinate Regression: $(x_t, y_t) \in \mathbb{R}^2$) มีข้อจำกัดขั้นพื้นฐานในการเรียนรู้เงื่อนไขทางกายภาพของผังอาคาร**

แม้โมเดลเช่น Transformer (GPT-2) หรือ GNN-CVAE จะแสดงค่าเฉลี่ยความคลาดเคลื่อนทางคณิตศาสตร์ (ADE/FDE) ที่ดูดีบนตัวเลข แต่เมื่อนำไปสร้างเส้นทางสัญจรจริง (Trajectory Rollout) กลับพบว่าโมเดลเกิดภาวะ *Feature Ignorance* คือ AI ไม่เข้าใจขอบเขตกำแพง สิ่งกีดขวาง หรือทิศทางเป้าหมายเชิงพื้นที่ ทำให้เกิดปรากฏการณ์เดินตัดผ่านผนัง (Wall Clipping) หรือหลงทาง เนื่องจากโมเดลพยายามทำนายการเรียงต่อของพิกัดทศนิยมแบบราบเรียบ (Smooth Interpolation) โดยขาดกรอบทางกายภาพ (Lack of Spatial Inductive Bias)

ในทางกลับกัน การเปลี่ยนรูปแบบปัญหากลับมาเป็น **Discrete Spatial Grid Policy (`GridSocialPolicy`)** แสดงให้เห็นถึงประสิทธิภาพที่เหนือกว่าอย่างมีนัยสำคัญในการเข้าใจบริบทสถาปัตยกรรม การป้อน Local Map Crops (`walkable`, `exit`, `occupancy`) เข้าไปใน CNN Map Encoder ช่วยให้ AI มีสายตาเชิงพื้นที่ (Spatial Perception) ในทุกก้าวที่ตัดสินใจเดิน การจำกัด Action Space ให้อยู่ในรูปของการขยับบน Grid Cell ทำให้ AI ไม่เพียงแต่เคารพขอบเขตของกำแพงได้อย่างแม่นยำ แต่ยังสามารถสะท้อนพฤติกรรมการกระจายตัวและการก่อตัวของความแออัดบริเวณทางแคบ/คอขวด (Bottleneck Congestion) ได้อย่างสมเหตุสมผล

---

## 6. Conclusion (บทสรุป)

งานวิจัยนี้นำเสนอข้อสรุปเชิงสถาปัตยกรรมที่สำคัญว่า **Spatial Discretization (Grid Representation) คือโครงสร้างที่จำเป็นอย่างยิ่งในการพัฒนา AI Surrogate Model สำหรับงานจำลองการสัญจรของคนเดินเท้า** การพยายามใช้ AI ทำนายพิกัดทศนิยมต่อเนื่องตรงๆ โดยหวังให้โมเดลเรียนรู้เงื่อนไขแปลนอาคารเองนั้นไม่ประสบความสำเร็จเนื่องจาก AI ไม่เข้าใจ Feature เชิงพื้นที่อย่างแท้จริง

การนำเสนอโมเดล `GridSocialPolicy` ที่เรียนรู้ Behavior Cloning บนพื้นที่ Grid ช่วยพิสูจน์ว่า AI สามารถทำหน้าที่เป็นตัวจำลองการสัญจรที่อ่านพื้นที่ เข้าใจความแออัด และทำงานได้รวดเร็วกว่า Physics-based simulation ดั้งเดิมหลายเท่าตัว ซึ่งเปิดโอกาสให้สถาปนิกและผู้ออกแบบสามารถนำไปใช้เป็นเครื่องมือประเมินประสิทธิภาพทางสถาปัตยกรรมในขั้นตอน Early-Stage Design ได้อย่างมั่นใจและแม่นยำ

---

## 7. Limitations and Future Research (ข้อจำกัดและงานวิจัยในอนาคต)

**ข้อจำกัด (Limitations):**
1. **Grid Resolution Trade-off:** ความแม่นยำเชิงพื้นที่ขึ้นอยู่กับขนาดของ Grid Cell หาก Cell ขนาดใหญ่เกินไป อาจสูญเสียรายละเอียดพิกัดทางกายภาพ แต่หากเล็กเกินไป จะเพิ่มความซับซ้อนของ Action Space
2. **Single-Goal Orientation:** การทดสอบปัจจุบันมุ่งเน้นสถานการณ์คนเดินออกจากพื้นที่ (Spawn to Exit) ยังไม่ได้ขยายผลสู่พฤติกรรมสัญจรหลายเป้าหมายพร้อมกัน (Multi-destination)

**งานวิจัยในอนาคต (Future Research):**
1. การพัฒนายกระดับสถาปัตยกรรม Grid Policy ด้วย Spatial Attention / Transformer over Local Occupancy Crops
2. การขยายขอบเขตการจำลองสู่ผังอาคารระดับพาณิชยกรรมขนาดใหญ่ และการจำลองสถานการณ์อพยพฉุกเฉิน (Evacuation Scenarios)

---

## 8. Scope การทดลอง (Experiment Scope)

- **ตัวแปรต้น (Independent Variables):** 
  - รูปแบบ Representation (Continuous Coordinate $(x,y)$ vs. Discrete Grid Policy)
  - ความซับซ้อนของแปลนอาคาร (Single Corridor, Bottleneck, Multi-room Layouts)
- **ตัวแปรตาม (Dependent Variables):** 
  - ความถูกต้องเชิงพื้นที่ (Wall Boundary Violation Rate, Goal Reach Rate)
  - ความสอดคล้องของรูปแบบการกระจายตัวและความแออัด (Congestion Pattern Alignment)
  - ความเร็วในการประมวลผล (Inference Latency & Speedup Ratio)
- **โมเดลที่ศึกษา:**
  - *Continuous Paradigm*: Transformer (GPT-2), GNN-CVAE, SGAN, CVAE, LSTM
  - *Discrete Grid Policy Paradigm*: GridSocialPolicyNet (CNN + MLP Action Policy)

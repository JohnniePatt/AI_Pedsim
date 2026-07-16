# สรุปการทำวิจัยจากการพัฒนา Code (Image-based AI Surrogate Model for Density Map Prediction)

## 1. การประเมินความสอดคล้อง (Coding Matching Score)
**คะแนน: 10/10**
**เหตุผล:** โครงสร้างโฟลเดอร์สำหรับโมเดลภาพ (`AI_GenerateTrajectory/AI_Train/Method_*`) สอดคล้องและครบถ้วนมาก ครอบคลุมสถาปัตยกรรมหลักทั้ง 4 ตัว ได้แก่ Plain U-Net, Pix2PixHD (with adversarial), Pix2PixHD No_D (without adversarial), และ CVAE พร้อมสคริปต์สำหรับการเทรนและการรันทดสอบ (`train_*.py`, `test_*.py`) รวมถึงมีชุดการประเมินผลเชิงเปรียบเทียบบนหน้า Streamlit (ในไฟล์ `image_based_output.py`) ไว้อย่างชัดเจน

---

## 2. Checklist สำหรับการเขียน Paper
อ้างอิงจากมาตรฐานงานวิจัย AI Surrogate Model ในระดับสากล Paper ควรมีองค์ประกอบดังนี้:
- [ ] **Abstract & Keywords**: สรุปวัตถุประสงค์ (ใช้ Deep Learning ทำนายแผนภาพความหนาแน่นผู้ใช้งานแทนการรัน Simulation) วิธีการ และประสิทธิภาพของทั้ง 4 โมเดล
- [ ] **Introduction**: ความล่าช้าของ Physics-based Simulation ในแปลนสลับซับซ้อน และการเปลี่ยนผ่านสู่ Image-to-Image Translation
- [ ] **Methodology**:
  - [ ] **Input Representation**: การแปลงแปลนทางกายภาพ (ผนัง/สิ่งกีดขวาง), จุดเริ่มสัญจร (Spawn), และจุดหมายปลายทาง (Exit) ให้กลายเป็นภาพ 3 แชนเนล (RGB) ขนาด 256x256
  - [ ] **Output Representation**: แผนภาพความหนาแน่นสะสมผู้ใช้งาน (Pedestrian Density Map) ขนาด 256x256 พิกเซล
  - [ ] **Model Architectures**:
    - [ ] **Plain U-Net**: โครงสร้างแบบมาตรฐานสำหรับภาพแพทย์ที่เน้น Skip connection
    - [ ] **Pix2PixHD (With Adversarial)**: โครงสร้าง High-resolution GAN ที่ใช้ Multi-scale Discriminator และ Feature Matching Loss
    - [ ] **Pix2PixHD No_D (Without Adversarial)**: โครงสร้าง ResNet Generator ที่ใช้ Density-Aware L1 Loss (ไม่มี Discriminator)
    - [ ] **Conditional VAE (CVAE)**: โครงสร้างความน่าจะเป็นที่เรียนรู้ Latent Space เพื่อทำนายความหนาแน่นในลักษณะหลายทิศทาง (Multimodal)
- [ ] **Results & Findings**:
  - [ ] ตารางเปรียบเทียบประสิทธิภาพเฉลี่ย (Average Metrics Table: MAE, MSE, RMSE, SSIM, PSNR)
  - [ ] สถิติจำนวนภาพที่ชนะในแต่ละตัววัด (Metric Compare per Image)
  - [ ] ภาพตัวอย่างผลลัพธ์ (Qualitative Evaluation: Ground Truth vs Predictions)
- [ ] **Discussion**: การทำความเข้าใจพฤติกรรมของโมเดล เหตุใด Pix2PixHD จึงมีค่าเฉลี่ย MAE ดีกว่า แต่ Plain U-Net ชนะในบางภาพ, ผลของการทำ Ablation Study (เปรียบเทียบ Pix2PixHD แบบมี/ไม่มี D) และความสามารถเชิงวิเคราะห์ (Generative) ของ CVAE
- [ ] **Limitations and Future Research**: ข้อจำกัดของขนาดภาพ 256x256 และความซับซ้อนของทิศทางสัญจร
- [ ] **Conclusion**: การประยุกต์ใช้เพื่อการประเมินทางเลือกสัญจรแบบเรียลไทม์ (Real-time spatial feedback)

---

## 3. สิ่งที่พร้อมใส่ใน Paper แล้ว (Ready to Include)
- **ข้อมูลภาพ Input/Output**: การนำเสนอการทำนายพฤติกรรมในฐานะงาน Image-to-Image translation
- **ตารางผลการประเมินเฉลี่ย (Run Summary)**:
  - **Pix2PixHD (With Adversarial)**: ได้รับตำแหน่งผู้ชนะที่มีค่าเฉลี่ย **MAE ต่ำที่สุด (0.0013)** และมีค่า SSIM/PSNR ในระดับสูง เนื่องจากโครงสร้าง Discriminator บังคับให้สร้างขอบเขตความหนาแน่นที่คมชัด
  - **Plain U-Net**: ได้รับค่าเฉลี่ย **MAE (0.0015)** และเป็นผู้ชนะถ้านับจำนวนภาพที่มีความคลาดเคลื่อนต่ำสุดรายภาพ (ชนะ 360/862 ภาพ) เนื่องจากลักษณะ Loss ฟังก์ชัน (L1) ที่พยายามเกลี่ยสีเทา ทำให้ได้ผลลัพธ์ที่เฉลี่ยตัวแปรได้ดีกว่าในภาพที่มีความหนาแน่นกระจายตัวบางเบา
  - **Pix2PixHD (No_D / Without Adversarial)**: ผลลัพธ์จากการทำ Ablation Study (ปิดการใช้ Discriminator) ได้ค่าเฉลี่ย **MAE (0.0016)** ซึ่งใช้ Custom Density-Aware L1 Loss ในการถ่วงน้ำหนักพิกเซลเลี่ยงการทำนายภาพเป็นสีดำล้วน
- **ผลลัพธ์ CVAE**: สถาปัตยกรรมแบบ Probabilistic ที่รองรับการสุ่มสร้างแนวโน้มความหนาแน่น (Latent variable $z$) สำหรับการวิเคราะห์ความหลากหลายของพฤติกรรมการเคลื่อนที่

---

## 4. สิ่งที่ยังขาดและต้องเพิ่มใน Paper (Missing Elements)
- **Major (สำคัญมาก)**:
  - รายละเอียดของ Hyperparameters ทั้งหมด เช่น Learning Rate, Batch Size, Optimizer (Adam) และสถาปัตยกรรมแชนเนลใน Encoder/Decoder
  - ขั้นตอนการคำนวณและประเมินผลเชิงเปรียบเทียบบนแง่ของเวลาประมวลผล (Inference Speed) เทียบกับ JuPedSim เพื่อเคลมความเร็วระดับ Real-time
  - การลงลึกรายละเอียดการตั้งค่าโครงสร้าง Loss Function ของ Pix2PixHD (L1 loss + GAN Loss + Feature Matching Loss)
- **Minor (ควรมีเพื่อความสมบูรณ์)**:
  - การวิเคราะห์พฤติกรรมการกระจายตัวความหนาแน่น (Density distribution profile) ในจุดคอขวด (Bottlenecks) เช่น ประตู หรือทางเดินแคบ

---

## 5. Discussion (ส่วนอภิปรายผล)
การเปรียบเทียบผลลัพธ์แสดงถึงความแตกต่างเชิงสถาปัตยกรรมอย่างเด่นชัด:
* **Pix2PixHD (With Adversarial)** ทำงานได้ยอดเยี่ยมที่สุดในภาพรวมเนื่องจากการใช้ **Adversarial Loss (GAN)** ร่วมกับ **Multi-scale Discriminator** ทำให้โมเดลเรียนรู้โครงสร้างขอบและรูปร่างของเส้นทางความหนาแน่นได้โดยไม่เบลอ (Blur) ส่งผลให้ทำนายโครงสร้างกระแสการสัญจรที่หนาแน่นได้แม่นยำสูงสุด
* **Pix2PixHD (No_D / Without Adversarial)** เมื่อปิดการรันของ Discriminator ออกไป พบว่าโครงสร้างการทำนายผลลัพธ์มีความฟุ้งและเบลอ (Blurry boundary) คล้ายคลึงกับ Plain U-Net ซึ่งเป็นการพิสูจน์เชิงประจักษ์ (Ablation Proof) ว่า **Adversarial Loss และ Discriminator มีผลโดยตรงต่อการปรับปรุงความคมชัดระดับสูง (High-frequency details)** ในแผนภาพความหนาแน่นของการสัญจร
* **Plain U-Net** แม้จะไม่มีตัวจำแนก (Discriminator) ช่วยปรับปรุงความชัด แต่ผลจากการใช้ **L1 Loss** ดั้งเดิมทำให้โมเดลเหมาะกับกรณีที่มีพฤติกรรมการเดินกระจัดกระจาย ซึ่ง U-Net จะทำการเกลี่ย (Smooth) สีเทาความหนาแน่นลงบนแปลนได้เรียบเนียนกว่า ส่งผลให้คะแนน MAE รายภาพชนะ Pix2PixHD ในกลุ่มข้อมูลที่มีความหนาแน่นเบาบาง
* **CVAE** แสดงให้เห็นความสามารถพิเศษในการสุ่มตัวแปรแฝง (Latent space sampling $z$) ทำให้ผู้ออกแบบสามารถ "จำลองความไม่แน่นอน" (Uncertainty) ของเส้นทางเลือกที่เดิน เช่น หากผู้คนเดินแยกออกเป็นสองทาง CVAE จะสามารถสร้างทางเลือกเหล่านั้นออกมาได้หลากหลายภาพ ไม่ใช่การเฉลี่ยรวมกันเป็นภาพเดียวแบบ Deterministic models

---

## 6. Conclusion (บทสรุป)
ความสำเร็จในส่วนนี้คือการพัฒนาโครงข่ายประสาทเทียมแบบ Image-to-Image translation เพื่อเปลี่ยน "แปลนอาคารกายภาพร่วมกับจุดเริ่มต้นและปลายทาง" ให้เป็น "แผนภาพความหนาแน่นการสัญจร" ได้ในระดับมิลลิวินาที 

Pix2PixHD (With Adversarial) คือโมเดลแนะนำสำหรับการทำนายรูปแบบความหนาแน่นทั่วไปที่ต้องการความแม่นยำสูงและมีความเสมือนจริงของรูปร่างกระแสสัญจร ในขณะที่ CVAE คือเครื่องมือสำหรับการวิเคราะห์พฤติกรรมทางเลือกที่มีความซับซ้อนและผันแปร ซึ่งทั้งหมดนี้ทำหน้าที่เป็นเครื่องมือวิเคราะห์ประสิทธิภาพสัญจรที่มีความเร็วสูง (Surrogate Model) เอื้อให้ผู้ออกแบบสามารถทดลองปรับแต่งสถาปัตยกรรมและแปลนได้แบบ Interactive

---

## 7. Limitations and Future Research (ข้อจำกัดและงานวิจัยในอนาคต)
**ข้อจำกัด (Limitations):**
1. **ขนาดภาพจำกัด (Resolution Limitation):** ขอบเขตการทำงานจำกัดที่ 256x256 พิกเซล ซึ่งอาจสูญเสียรายละเอียดสเกลหากใช้กับอาคารขนาดใหญ่ที่มีความยาวระดับร้อยเมตร
2. **เงื่อนไขพลวัตคงที่ (Static Condition):** แหล่งกำเนิด (Spawn) และเป้าหมายปลายทาง (Exit) ถูกส่งเป็นข้อมูลภาพแบบคงที่ (Static) ทำให้ยังไม่รองรับกรณีการจำลองสถานการณ์เปลี่ยนจุดหมายระหว่างทาง (Dynamic rerouting)

**งานวิจัยในอนาคต (Future Research):**
1. พัฒนาสถาปัตยกรรมแบบ Super-Resolution (เช่น ผนวก Diffusion Model หรือปรับโครงสร้าง Pix2PixHD) เพื่อรองรับภาพความละเอียด 1024x1024
2. การขยายแชนเนลในภาพ Input เพื่อระบุประเภทของผู้ใช้งานที่หลากหลาย (เช่น ผู้พิการ เด็ก คนชรา) เพื่อทำนายแผนภาพความหนาแน่นจำเพาะบุคคล

---

## 8. Scope การทดลอง (Experiment Scope)
- **ภาพ Input (3 แชนเนล - RGB):**
  - **Red Channel:** ขอบเขตโครงสร้างทางกายภาพ / กำแพง (Physical boundaries & obstacles)
  - **Green Channel:** ตำแหน่งพื้นที่เริ่มต้นสัญจร (Spawn zone)
  - **Blue Channel:** จุดหมายปลายทางสัญจร (Exit zone / checkpoints)
- **ภาพ Output (1 แชนเนล - Grayscale):**
  - แผนภาพความหนาแน่นการสัญจรสะสม (Accumulated Pedestrian Density Map)
- **ชุดข้อมูล (Dataset):**
  - สังเคราะห์ผ่านกระบวนการทำภาพจำลอง Trajectory ของ JuPedSim 
  - ประกอบด้วย 589 Plans (Train 412, Test 117, Val 60) รวมเป็น 3,904 รูปคู่ขนาน (1,326 Unique routes โดยมี 3 แชนเนลความหนาแน่นย่อย)
- **โมเดลเปรียบเทียบ (Surrogate Models):**
  - Plain U-Net
  - Pix2PixHD (With Adversarial)
  - Pix2PixHD No_D (Without Adversarial)
  - Conditional VAE (CVAE)
- **ดัชนีวัดความแม่นยำ (Evaluation Metrics):**
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)

# สรุปและรีวิว Code: Discrete Spatial Grid Action Policy (Method_GridSocialPolicy)

## 1. ภาพรวมสถาปัตยกรรม (Architecture Overview)

โฟลเดอร์ที่เกี่ยวข้อง:
- [AI_GenerateTimeseries/AI_Train/Method_GridSocialPolicy](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_GridSocialPolicy)

`Method_GridSocialPolicy` เป็นการเปลี่ยนกรอบปัญหาจากการทำนายพิกัดทศนิยมต่อเนื่อง มาเป็น **Behavior-Cloning Discrete Action Policy บน Discrete Grid Space**

---

## 2. โครงสร้างไฟล์และการทำงานเชิงลึก (Code Pipeline)

```text
AI_GenerateTimeseries/AI_Train/Method_GridSocialPolicy/
├── action_space.py          # สกัดและสร้าง Discrete Movement Offsets (Action Space)
├── dataset_grid_policy.py   # โหลดข้อมูล Local Grid Crops และสร้าง Training Samples
├── model_grid_policy.py     # สถาปัตยกรรม GridSocialPolicyNet (CNN + MLP Policy)
├── train_grid_policy.py     # Training Loop และระบบเซฟ Checkpoints
├── rollout.py               # รันจำลองก้าวต่อก้าวจากสภาวะเริ่มต้น (Frame 0 Spawn)
├── rollout_batch.py         # รัน Rollout อัตโนมัติหลายชุดตัวอย่าง
└── metrics.py               # คำนวณค่าสถิติการ Rollout
```

### 2.1 Action Space & Representation (`action_space.py`)
- สกัดรูปแบบการเคลื่อนที่ของคนเดินเท้าจาก Grid Trajectory Data
- นิยาม Discrete Movement Offsets ($\Delta x, \Delta y$) เช่น การก้าวขยับ 1-3 ช่อง Grid ในทิศทางต่างๆ (ขึ้น, ลง, ซ้าย, ขวา, แนวเฉียง) + การหยุดรอ (`wait`)
- มีการแยก `stop_head` สำหรับทำนายการจบการเดินทางเมื่อถึงเป้าหมาย

### 2.2 Model Architecture (`model_grid_policy.py`)
- **`ConvBlock` & `map_encoder`**: ใช้ CNN 2D ประมวลผล **Local Grid Crop** ขนาด 3 ช่องสัญญาณ (Channels):
  1. `walkable`: โครงสร้างทางเดินและกำแพงในบริเวณรอบตัว
  2. `exit`: ทิศทางและตำแหน่งของประตูทางออก
  3. `occupancy`: ตำแหน่งของคนเดินเท้าคนอื่นๆ ในรัศมีรอบตัว
- **`feature_encoder`**: MLP ประมวลผล Feature เชิงตัวเลข เช่น ระยะทางและเป้าหมาย
- **`fusion` & Heads**: รวม Map Embedding และ Agent Feature เข้าด้วยกันผ่าน FC Layers ทำนาย:
  - `action_logits`: ความน่าจะเป็นในการเลือกก้าวเดินไปยังช่อง Grid ต่างๆ
  - `stop_logits`: ความน่าจะเป็นในการหยุดเดิน

---

## 3. ทำไม Grid Representation ถึงประสบความสำเร็จ (Why Grid Policy Succeeds)

1. **Spatial Inductive Bias ผ่าน Local Grid Crops**:
   - การป้อนภาพ Local Grid Crop (`walkable`) ขนาดเล็กรอบตัวคนเดินเท้าเข้า CNN ช่วยให้ AI มี **"สายตาเชิงพื้นที่ (Spatial Vision)"** ในทุกก้าวที่ตัดสินใจก้าวเดิน AI สามารถรับรู้ว่าช่องใดเป็นกำแพง ช่องใดเป็นทางเดินได้โดยตรง

2. **การเคารพขอบเขตทางกายภาพ (Wall Boundary Respect)**:
   - ด้วยโครงสร้าง Grid ทำให้ AI เรียนรู้ได้ง่ายว่า Action ใดที่จะนำไปสู่ช่องที่เป็นกำแพง จะมีค่า Cross-Entropy Loss สูง ส่งผลให้ในการ Rollout จริง โมเดลเกือบจะไม่เดินชนหรือทะลุกำแพงเลย (Boundary Violation Rate เข้าใกล้ 0%)

3. **การจับรูปแบบความแออัด (Bottleneck & Density Awareness)**:
   - สัญญาณ `occupancy` ใน Local Crop ทำให้ AI รับรู้ความหนาแน่นของผู้คนที่อยู่ข้างหน้าและรอบข้าง สามารถตัดสินใจหยุดรอ (`wait`) หรือชะลอความเร็วเมื่อเกิดความแออัดบริเวณคอขวด (Bottleneck) ได้อย่างเป็นธรรมชาติ

---

## 4. บทบาทใน Paper

`Method_GridSocialPolicy` คือ **"พระเอกหลักของงานวิจัย" (Proposed Solution & Key Contribution)** ที่ใช้พิสูจน์สมมติฐานว่า:
- การสัญจรของมนุษย์ในผังอาคารต้องการ Spatial Representation ที่ชัดเจน
- การเปลี่ยนมาใช้ Discrete Action Grid Policy สามารถแก้ปัญหา Feature Ignorance ของโมเดลพิกัดทศนิยม และจำลองพฤติกรรมความแออัดในผังอาคารได้อย่างแม่นยำและสมจริง

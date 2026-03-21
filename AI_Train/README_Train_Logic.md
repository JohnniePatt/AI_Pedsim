# คำอธิบายโค้ด Train_Topo2.py (AI LSTM Baseline)

ไฟล์นี้ทำหน้าที่เป็น Pipeline หลักในการเตรียมข้อมูลและเทรนโมเดล LSTM สำหรับทำนายการเคลื่อนที่ของคนเดิน (Pedestrian) โดยแบ่งส่วนการทำงานดังนี้:

---

## 1. การตั้งค่าและจัดการไฟล์ (Setup & Path Management)
- **บรรทัดที่ 41-48:** กำหนด `PROJECT_ROOT` และระบุ Path ไปยังโฟลเดอร์ `geo`, `dataswarm`, และ `spawn_exit_area` อย่างชัดเจนเพื่อให้สคริปต์หาไฟล์เจอไม่ว่าจะรันจากที่ไหน
- **บรรทัดที่ 52-66 (CONFIG):** กำหนดค่า Hyperparameters เช่น `seq_len = 20` (ดูย้อนหลัง 20 เฟรม), `hidden_size = 128`, และชื่อ Feature ที่เราจะใช้เทรนทั้งหมด

## 2. ฟังก์ชันช่วยจัดการข้อมูล (Helper Functions)
- **`load_json_polygons`:** อ่านไฟล์ JSON แผนที่แล้วแปลงเป็นวัตถุ `Polygon` ของ Shapely เพื่อใช้เช็คว่าคนอยู่ที่ไหน
- **`load_exit_polygon`:** อ่านไฟล์ CSV จาก `spawn_exit_area` และใช้ `load_wkt` แปลงข้อความ WKT กลับมาเป็นรูปทรงพื้นที่ทางออก
- **`get_sqlite_metadata`:** ดึงค่าขอบเขตแผนที่ (`xmin`, `xmax` ฯลฯ) จากตาราง metadata ใน SQLite เพื่อใช้ในการทำ Normalization (ปรับสเกลตัวเลขให้อยู่ในช่วง 0-1)

## 3. การสร้างฟีเจอร์ (Feature Engineering - `process_seed_data`)
ส่วนนี้สำคัญที่สุดในการเตรียมข้อมูลให้ AI:
- **Velocity (vx, vy):** ใช้ `groupby('id').shift(1)` เพื่อหาตำแหน่งในเฟรมก่อนหน้ามาลบกับปัจจุบัน ได้มาเป็นความเร็ว/ทิศทางขยับ
- **Goal Vector:** คำนวณระยะห่างระหว่างตัว Agent กับจุดศูนย์กลางของ Exit (`exit_centroid`)
- **Spatial Features:** ใช้ `poly.contains(Point)` เช็คว่าพิกัดนั้นอยู่ในห้อง (`in_room`) หรือทางเดิน (`in_corridor`) หรือไม่
- **Normalization:** นำค่า X, Y และความเร็วทั้งหมดมาหารด้วยขนาดของแผนที่เพื่อให้ค่าอยู่ในช่วงที่โมเดล LSTM เรียนรู้ได้ง่าย

## 4. การสร้างลำดับข้อมูล (Sequence Building - `build_sequences_per_agent`)
- **Sliding Window:** โมเดล LSTM ต้องการข้อมูลเป็นชุด (Sequence) สคริปต์จึงหยิบข้อมูลมาทีละ 20 แถวต่อเนื่องกัน
- **Agent Isolation:** มีการตรวจสอบ `groupby('id')` เพื่อให้แน่ใจว่า **ข้อมูลท้ายของ Agent คนที่ 1 จะไม่ไหลไปปนกับจุดเริ่มต้นของ Agent คนที่ 2** (ป้องกันโมเดลมั่ว)

## 5. โครงสร้างโมเดล (Model Architecture - `LSTM_Baseline`)
- **LSTM Layer:** ใช้ 2 Layer โดยมี Hidden Unit ขนาด 128
- **Many-to-One:** เราใส่ข้อมูลเข้าไป 20 step แต่เอา Output เฉพาะตอนจบของ Sequence (`out[:, -1, :]`) มาผ่าน Linear Layer เพื่อทำนายค่า `dx, dy` (ระยะที่คนจะขยับไปในก้าวถัดไป)

## 6. ลูปรันการเทรน (Training Loop)
- **Data Splitting:** แบ่งข้อมูล 80% ไว้เทรน และ 20% ไว้ตรวจสอบ (Validation)
- **Loss Function:** ใช้ `MSELoss` (Mean Squared Error) เพื่อวัดว่าระยะขยับที่ AI ทาย กับของจริง ต่างกันกี่เมตร
- **Checkpoint:** ทุกครั้งที่ค่า Val Loss ลดลงต่ำที่สุด สคริปต์จะเซฟโมเดลไว้ที่ `best_lstm.pt` ทันที

## 7. ผลลัพธ์ (Outputs)
เมื่อรันเสร็จคุณจะได้ไฟล์ใน `AI_Train/outputs/Topo2/`:
1. `best_lstm.pt`: ตัวโมเดลที่แม่นที่สุด
2. `train_log.csv`: ตารางบอกค่า Error ในแต่ละรอบ (Epoch)
3. `train_config.json`: บันทึกการตั้งค่าทั้งหมดที่ใช้เทรนรอบนั้นๆ

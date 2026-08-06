# AI Technique Notes

วันที่อัปเดต: 2026-08-06

เอกสารนี้สรุปการแก้ไข pipeline ของ `Method_LSTM_SF_01` ที่ทำวันนี้ พร้อมเหตุผลและวิธีรันหลังแก้

## ปัญหาที่เจอ

1. รันคำสั่ง `python3 run_pipeline.py` แล้วจบที่ `stage=plan`
   - สาเหตุคือ `pipeline_common.parser()` ตั้งค่า default ของ `--stage` เป็น `plan`
   - ทำให้คำสั่งที่ไม่ใส่ `--stage` แค่พิมพ์แผน ไม่เริ่ม train จริง

2. หลังเริ่ม train แล้ว validation แตกเพราะหา parquet ไม่เจอ
   - สาเหตุคือ dataset split มีโฟลเดอร์ `case_*` ว่างหรือไม่สมบูรณ์ปะปนอยู่
   - `JointSceneDataset` เดิมเลือกทุก `case_*` directory โดยไม่เช็คว่ามี trajectory parquet และ geometry ครบหรือไม่

3. ตอน train ไม่มี progress ระหว่าง epoch
   - สาเหตุคือ script พิมพ์ผลแค่หลังจบ epoch
   - เมื่อหนึ่ง epoch ใหญ่มาก ผู้ใช้จะเห็นเหมือนโปรแกรมค้าง

4. Default training หนักเกินไป
   - `config_active.json` เดิมมี `train_batches=43504` ต่อ epoch
   - จาก screenshot เห็นว่า 705 batches ใช้เวลาประมาณ 3 นาที และ ETA ต่อ epoch เกือบ 2 ชั่วโมง
   - ยังพบ `loss=nan` ระหว่าง run ทำให้การปล่อยให้ train ต่อไม่มีประโยชน์

## วิธีแก้ที่ทำ

### 1. ให้ `run_pipeline.py` เริ่ม train เป็นค่า default

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/pipeline_common.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/run_pipeline.py`

วิธีแก้:

- เพิ่ม parameter `default_stage` ให้ `pipeline_common.parser()`
- ค่า default กลางยังเป็น `plan` เหมือนเดิม เพื่อไม่กระทบ method อื่น
- ตั้งเฉพาะ `Method_LSTM_SF_01/run_pipeline.py` เป็น `default_stage="train"`
- เพิ่มเมนูเลือก training profile เมื่อรันแบบ interactive โดยไม่ส่ง `--config-train` หรือ `--profile`
- เพิ่ม shortcut `--profile fast` และ `--profile full` สำหรับข้ามเมนู
- เพิ่ม `flush=True` ให้ log pipeline เพื่อให้ข้อความออกตามลำดับก่อนเรียก subprocess

ผลลัพธ์:

```bash
python3 run_pipeline.py
```

จะถามให้เลือก:

```text
Select LSTM-SF training profile:
  1) fast - quick debug/sanity training (recommended)
  2) full - full research-scale training
Choose 1 or 2 [1]:
```

ถ้ากด Enter หรือเลือก `1` จะใช้ fast profile
ถ้าเลือก `2` จะใช้ full profile

ถ้าต้องการดู plan อย่างเดียว ยังใช้ได้:

```bash
python3 run_pipeline.py --stage plan
```

### 2. กรอง dataset case ที่ไม่สมบูรณ์

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/joint_sf.py`

วิธีแก้:

- เพิ่ม helper `_is_complete_case_dir()`
- ให้ `JointSceneDataset` เลือกเฉพาะ case directory ที่มี:
  - trajectory parquet อย่างน้อย 1 ไฟล์
  - `Geo_room.json`
  - `Geo_corridor.json`

ผลลัพธ์:

- ข้ามโฟลเดอร์ `case_*` ที่ว่างหรือไม่มี trajectory
- `max_cases` จะนับจาก case ที่ใช้ train/val/test ได้จริง
- ลดโอกาสแตกกลาง epoch ด้วย `FileNotFoundError: trajectory parquet missing`

### 3. เพิ่ม progress ระหว่าง train และ validation

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/train_joint_sf.py`

วิธีแก้:

- เพิ่ม class `BatchProgress`
- ถ้ามี `tqdm` ใน environment จะใช้ progress bar
- ถ้าไม่มี `tqdm` จะ fallback เป็นข้อความ `batch x/y`
- แสดง progress ทั้งช่วง train และ validation
- แสดงจำนวน case และ batch ก่อนเริ่ม train

ตัวอย่าง output:

```text
[train] train_cases=128 train_batches=64 val_cases=32 val_batches=4
epoch 1/20 train:  50%|...| loss=... teacher=...
epoch 1/20 val: ...
```

### 4. เพิ่ม guard จับ `loss=nan`

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/train_joint_sf.py`

วิธีแก้:

- เพิ่ม `require_finite_losses()`
- เช็คทุก loss ว่า finite หรือไม่
- ถ้าเจอ `nan` หรือ `inf` จะหยุดทันทีด้วย `FloatingPointError`
- error message จะระบุ:
  - epoch
  - batch index
  - case_id
  - loss ที่ผิดปกติ

ผลลัพธ์:

- ไม่เสียเวลาปล่อย training วิ่งต่อเมื่อ loss เสียแล้ว
- ตาม debug case ที่ทำให้เกิด NaN ได้เร็วขึ้น

### 5. ปรับ default config ให้เป็น fast sanity-training profile

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_active.json`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_full.json`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/README.md`

วิธีแก้:

- ลด default run ใน `config_active.json` ให้เหมาะกับการลอง/debug ก่อน
- เพิ่ม `config_full.json` เพื่อเก็บค่า full research-scale เดิมไว้
- อัปเดต README ให้บอกว่า default เป็น fast profile และ full run ใช้ `config_full.json`

ค่า default ใหม่ใน `config_active.json`:

```json
{
  "max_agents": 128,
  "windows_per_case_train": 4,
  "windows_per_case_val": 1,
  "max_train_cases": 128,
  "max_val_cases": 32,
  "batch_size": 8,
  "hidden_dim": 64,
  "epochs": 20,
  "early_stopping_patience": 5,
  "case_cache_size": 64,
  "progress_interval_batches": 5
}
```

ผลลัพธ์:

- จาก `43504` batches/epoch เหลือประมาณ `64` batches/epoch
- เหมาะสำหรับเช็คว่า pipeline, loss, dataset และ checkpoint ทำงานถูกก่อน
- ถ้าต้องการ full training ใช้ config แยก

## วิธีรันหลังแก้

รันแบบ default เร็ว:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01
python3 run_pipeline.py
```

หรือสั่ง fast โดยไม่ต้องตอบเมนู:

```bash
python3 run_pipeline.py --profile fast
```

รัน full research-scale:

```bash
python3 run_pipeline.py --profile full
```

หรือระบุ config ตรง ๆ:

```bash
python3 run_pipeline.py --config-train config_full.json
```

ดู plan อย่างเดียว:

```bash
python3 run_pipeline.py --stage plan
```

ทดสอบ smoke config:

```bash
python3 run_pipeline.py --stage train --config-train config_smoke.json
```

## การตรวจสอบที่ทำแล้ว

1. ตรวจ syntax:

```bash
python -m py_compile pipeline_common.py joint_sf.py train_joint_sf.py Method_LSTM_SF_01/run_pipeline.py
```

2. ตรวจ dry-run:

```bash
python run_pipeline.py --dry-run
```

ยืนยันว่า default stage เป็น `train`

3. ทดสอบ smoke train:

```bash
python run_pipeline.py --stage train --config-train config_smoke.json
```

ผลลัพธ์ผ่าน 1 epoch:

```text
[epoch 1] train=0.13557 val=0.13622 seconds=0.5
```

4. ตรวจจำนวน batch ของ active config ใหม่:

```text
active train_cases 128 train_batches 64 val_cases 32 val_batches 4 epochs 20
```

## หมายเหตุ

- ถ้ามี run เก่ากำลัง train อยู่ ต้องกด `Ctrl+C` แล้วรันใหม่ เพราะ process เดิมจะยังใช้โค้ด/config ที่โหลดไว้ก่อนแก้
- ถ้าเจอ `FloatingPointError: non-finite loss` ให้ดู `case_id` ใน error เพื่อไล่ตรวจข้อมูล case นั้นต่อ
- Default config ใหม่ออกแบบมาเพื่อ debug และ sanity training ก่อน ไม่ใช่ผลวิจัย final
- Full config เดิมยังอยู่ที่ `config_full.json`

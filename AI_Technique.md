# AI Technique Notes

วันที่อัปเดต: 2026-08-06

เอกสารนี้สรุปการแก้ไข pipeline ของ `Method_LSTM_SF_01` ที่ทำวันนี้ พร้อมเหตุผลและวิธีรันหลังแก้

> อัปเดตล่าสุด: `run_pipeline.py` ของ trajectory methods ใช้ interactive operation menu แล้ว
> เมื่อรันโดยไม่ใส่ argument จะไม่เริ่ม train ทันที แต่จะแสดงตัวเลือกพร้อมคำอธิบาย
> ส่วน `--stage` ยังเก็บไว้สำหรับ automation และ backward compatibility

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

### 1. ปรับ `run_pipeline.py` ให้มีเมนูตาม `AGENTS.md`

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/pipeline_common.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/run_pipeline.py`

วิธีแก้:

- เพิ่ม parameter `default_stage` ให้ `pipeline_common.parser()`
- ค่า default กลางยังเป็น `plan` เหมือนเดิม เพื่อไม่กระทบ method อื่น
- เคยทดลองตั้งเฉพาะ `Method_LSTM_SF_01/run_pipeline.py` เป็น `default_stage="train"` เพื่อแก้ปัญหาไม่เริ่ม train
- หลังอ่าน `AGENTS.md` แล้วปรับกลับให้ default เป็น `plan` เพื่อไม่เริ่มใช้ GPU โดยไม่ตั้งใจ
- เพิ่ม interactive operation menu เมื่อรันมือเปล่า โดย default คือ `Check configuration`
- ถ้าเลือก `Train model` หรือ `Train model and evaluate` จะมีเมนูเลือก training profile อีกชั้น
- เพิ่ม shortcut `--stage train --profile fast` และ `--stage train --profile full` สำหรับข้ามเมนู
- ถ้าส่ง `--profile` หรือ `--config-train` โดยไม่ส่ง `--stage` จะถือว่าเป็น `--stage train` เพื่อความสะดวกของ CLI
- เพิ่ม `flush=True` ให้ log pipeline เพื่อให้ข้อความออกตามลำดับก่อนเรียก subprocess

ผลลัพธ์ปัจจุบัน:

```bash
python3 run_pipeline.py
```

จะเปิดเมนู operation:

```text
Social-Force-Informed Joint Multi-Agent LSTM
Select an operation:

  1) Check configuration
  2) Quick smoke test
  3) Train model
  4) Evaluate existing model
  5) Train model and evaluate
  6) View available runs
  0) Exit
Choose [1]:
```

ถ้ากด Enter จะเลือก `Check configuration` และยังไม่เริ่ม train
ถ้าเลือก train จึงจะถามต่อว่าจะใช้ `fast` หรือ `full`

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

### 6. ลดคอขวด data loading ใน full training

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/train_joint_sf.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_active.json`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_full.json`

สาเหตุของคอขวด:

- `JointSceneDataset` โหลด case หนึ่งครั้งด้วยการอ่าน parquet, pivot trajectory, อ่าน geometry JSON, rasterize walkable grid และคำนวณ wall field
- full config เดิมใช้ `shuffle=True` ระดับ sample ทำให้ windows จาก case เดียวกันกระจายทั่ว epoch
- cache เดิมมีแค่ `case_cache_size=2` จึงแทบไม่ hit cache เมื่อ shuffle กระจายแบบนี้
- ผลคือ case เดิมถูกอ่าน parquet และคำนวณ geometry ซ้ำจำนวนมาก

วิธีแก้:

- เพิ่ม `CaseWindowBatchSampler`
- sampler ใหม่ยัง shuffle ลำดับ case ได้ แต่จะจัด windows ของ case เดียวกันให้อยู่ใกล้กัน
- ทำให้ cache ใน `JointSceneDataset` ใช้ซ้ำได้จริง
- เพิ่ม config:
  - `case_grouped_batches`
  - `shuffle_train_cases`
  - `shuffle_train_windows`
  - `persistent_workers`
  - `prefetch_factor`
- ปรับ full config เป็น `num_workers=2`, `case_cache_size=16`, `persistent_workers=true`

ผล benchmark เฉพาะ data loading จาก full config 200 batches:

```text
old_style shuffle=True cache_size=2 workers=0:
  200 batches / 11.03 seconds = 18.13 batch/s

case_grouped_batches=true cache_size=16 workers=2:
  200 batches / 1.09 seconds = 183.33 batch/s
```

แปลว่า data-loading path เร็วขึ้นประมาณ 10 เท่าใน benchmark สั้นนี้ โดยไม่ได้ลดจำนวน full cases/windows

### 7. ลดจำนวน batch ของ full run โดยยังใช้ข้อมูลเท่าเดิม

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/train_joint_sf.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_full.json`

หลักการ:

- จำนวน batch ต่อ epoch คำนวณจาก `num_samples / batch_size`
- `num_samples` ของ full run ยังเป็น `train_cases * windows_per_case_train`
- ถ้าลด `windows_per_case_train`, `max_train_cases` หรือ `epochs` จะลดข้อมูล/ลดรอบฝึก
- ถ้าอยากใช้ข้อมูลเท่าเดิมแต่ batch น้อยลง ให้เพิ่ม `batch_size`

ค่าเดิม:

```text
train_cases ~= 2719
windows_per_case_train = 32
batch_size = 2
train_batches = 2719 * 32 / 2 = 43504
```

ค่าใหม่:

```text
train_cases ~= 2719
windows_per_case_train = 32
batch_size = 8
train_batches = 2719 * 32 / 8 = 10876
```

สิ่งที่เพิ่มเพื่อช่วยให้ batch ใหญ่ขึ้น:

- เปิด `amp=true` ใน config
- เพิ่ม AMP mixed precision ใน train/validation loop ด้วย `torch.amp.autocast` และ `torch.amp.GradScaler`
- benchmark forward/backward จริง 10 batches ด้วย full dimensions:

```text
device=cuda
amp=True
batch_size=8
batches_total=10876
max_memory_gb=3.95
checked_batches=10
batch_per_sec=4.90
```

สรุป:

- batch count ลดจาก `43504` เป็น `10876`
- full run ยังใช้ case/window เท่าเดิม
- ไม่ใช่การลด epoch
- ไม่ใช่การลด dataset
- ความเร็วจริงขึ้นกับ GPU memory และ compute ต่อ batch

### 8. แก้ `stop_loss=nan` จาก all-masked social attention

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/joint_sf.py`

อาการ:

```text
FloatingPointError: non-finite loss at epoch=1 batch=1065
case_id=plan_113_042f_200043_01_single
losses={'loss': nan, 'stop_loss': nan}
```

สาเหตุ:

- ใน full rollout บาง single-agent case โมเดลทำนายว่า agent หยุดหรือถึง exit เร็วกว่าขอบเขต prediction
- timestep ถัดไปจึงมี `current_active=false` ทั้ง scene
- `nn.MultiheadAttention` ได้ `key_padding_mask=true` ทุก key ใน scene นั้น
- PyTorch attention เมื่อ key ถูก mask หมดสามารถคืนค่า NaN ได้
- NaN ไหลต่อไปที่ `stop_logits` และ `stop_loss`

วิธีแก้:

- ใน `JointSocialForcePredictor.predict_next()` ตรวจ scene ที่ไม่มี active agent
- ถ้าไม่มี active key เลย ให้ unmask dummy key หนึ่งตำแหน่งเพื่อไม่ให้ attention เป็น all-masked
- หลัง attention แล้ว zero ค่า social output ของ inactive agents
- ใน `trajectory_losses()` mask `stop_loss` ก่อน reduce เพื่อไม่ให้ค่าที่ไม่ควรถูกนับ contaminate loss

การตรวจสอบ:

```text
targeted case plan_113_042f_200043_01_single:
  amp=False outputs/losses finite
  amp=True  outputs/losses finite

limited full loop:
  completed 1100 batches
  passed the previous failing region
  batch_per_sec ~= 5.93
```

### 9. เพิ่ม profile `quarter` แบบสุ่ม 1/4 ของ train plans ต่อ epoch

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/train_joint_sf.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/run_pipeline.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_quarter_plan.json`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/README.md`

แนวคิด:

- ในแต่ละ epoch เลือกเฉพาะ 25% ของ training plans
- เมื่อขึ้น epoch ใหม่ จะสุ่มชุด plans ใหม่ด้วย `seed + epoch`
- ไม่แตะ validation/test split
- ถ้า plan ถูกเลือก จะใช้ scenarios/windows ของ plan นั้นใน train split
- ใช้ model size เท่า full profile และ `windows_per_case_train=32` เหมือน full

เหตุผลที่เลือกสุ่มระดับ plan ไม่ใช่สุ่ม case:

- ผู้ใช้ต้องการตัวอย่างเช่น “มี 1000 ผัง เลือกมา 250 ผัง”
- การสุ่มระดับ plan ทำให้ concept ตรงกว่า case-level sampling
- ลดความเสี่ยงที่ epoch หนึ่งมี fragment ของ plan กระจายแบบตีความยาก

ผลตรวจจำนวนจาก config ปัจจุบัน:

```text
full:
  train_cases=2719
  train_plans=478
  train_batches=10876
  epochs=100

quarter:
  sampled_train_cases=688/2719
  sampled_train_plans=120/478
  train_batches=2752
  epochs=160
```

หมายเหตุ: จำนวน batches ของ quarter profile อาจแกว่งเล็กน้อยในแต่ละ epoch เพราะแต่ละ plan มีจำนวน scenarios ไม่เท่ากัน ตัวอย่าง epoch 1-3 ได้ประมาณ `2548-2756` batches

ความเห็นแบบไม่อวย:

- ข้อดี: epoch สั้นลงประมาณ 4 เท่า และ coverage จะหมุนเวียนผ่านหลาย epoch
- ข้อดี: ดีกว่า fixed 25% subset เพราะไม่ได้ติดอยู่กับ subset เดิมทั้ง run
- ข้อเสีย: ไม่เท่ากับ full training 100 epoch ถ้าไม่ได้ชดเชย epoch ให้พอ
- ข้อเสีย: validation curve อาจแกว่งกว่า เพราะ train subset เปลี่ยนทุก epoch
- ข้อเสีย: ถ้าใช้เป็นผลวิจัย ต้องระบุ protocol ชัดว่าเป็น stochastic quarter-plan training ไม่ใช่ full-data-per-epoch training

วิธีใช้:

```bash
python3 run_pipeline.py --profile quarter
```

หรือผ่านเมนู:

```text
Train model -> quarter
```

### 10. เพิ่ม profile `half` แบบสุ่ม 1/2 ของ train plans ต่อ epoch

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/run_pipeline.py`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/config_half_plan.json`
- `AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01/README.md`

แนวคิด:

- เหมือน `quarter` แต่เลือก 50% ของ training plans ต่อ epoch
- เมื่อขึ้น epoch ใหม่ จะสุ่ม plans ใหม่
- ใช้ model size, batch size, windows per case และ AMP เหมือน full
- ตั้ง `epochs=120` เพื่อให้ sample exposure รวมประมาณ 60 full-data epochs

ประมาณจำนวนจาก config ปัจจุบัน:

```text
half:
  sampled_train_plans ~= 239/478
  sampled_train_cases ~= 1430/2719
  train_batches ~= 5720
  epochs = 120
```

ความเห็นแบบไม่อวย:

- ดีกว่า `quarter` ถ้ากังวลว่าแต่ละ epoch เห็นข้อมูลน้อยเกินไป
- validation curve มักนิ่งกว่า quarter เพราะ train subset ต่อ epoch ใหญ่กว่า
- ยังเร็วกว่า full ประมาณครึ่งหนึ่งต่อ epoch
- แต่ถ้าตั้ง epochs=120 จะใช้เวลามากกว่า quarter profile
- ไม่เท่ากับ full 100 epochs; ถ้าต้องการ sample exposure เท่า full 100 epochs ต้องใช้ half ประมาณ 200 epochs ซึ่งเวลาอาจใกล้ full มากขึ้น

วิธีใช้:

```bash
python3 run_pipeline.py --profile half
```

หรือผ่านเมนู:

```text
Train model -> half
```

หมายเหตุ research validity:

- benchmark พบ full loader เห็น `train_cases=2719`
- แต่ `AGENTS.md` ระบุ canonical train scenarios เป็น `2603`
- ก่อน final/full research training ต้องตรวจ split membership และ manifest อีกครั้ง หากจำนวนไม่ตรงต้องรายงาน diff และแก้ dataset contract ก่อนอ้างผลเป็น research-valid

### 11. เพิ่ม profile `fast / quarter / full` ให้ `Method_GridSocialPolicy_SF_01`

ไฟล์ที่แก้:

- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/run_pipeline.py`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/train_grid_policy.py`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/config_fast.json`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/config_quarter_plan.json`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/config_full.json`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/config_smoke.json`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/config_train.json`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/README.md`

สิ่งที่ทำ:

- เพิ่มเมนู training profile สำหรับ GridSocialPolicy-SF:

```text
1) fast - quick debug/sanity training (recommended)
2) quarter - rotate 25% of train plans each epoch
3) full - full research-scale training
```

- `fast` ใช้ dataset เล็กและ model เล็กลงเพื่อเช็ก pipeline เร็ว ๆ
- `quarter` ใช้ข้อมูล train ทั้งชุดเป็น pool แต่ต่อ epoch จะสุ่มมา 25% ของ `plan_name` เท่านั้น
- `full` ใช้ train split เต็มตาม canonical Grid manifest
- เพิ่ม `PlanFractionBatchSampler` ใน `train_grid_policy.py` เพื่อสุ่มระดับ plan ไม่ใช่สุ่มระดับ sample
- sampler ใช้ seed เดิมบวกเลข epoch ทำให้แต่ละ epoch สุ่มใหม่ แต่ยัง reproducible
- ใช้ quarter เฉพาะ train split; validation ยังใช้ val split ตาม config เดิม
- ปรับ path ใน config ให้เป็น relative ต่อ method folder เพื่อย้าย workspace ได้ง่ายขึ้น

จำนวนที่ตรวจจาก config ปัจจุบัน:

```text
fast:
  train_plans=109/109
  train_cases=128/128
  train_samples=12288
  train_batches=24
  epochs=20

quarter:
  sampled_train_plans=103/412
  train_cases_pool=2603
  train_samples_pool=663272
  train_batches_per_epoch≈165
  epochs=160

full:
  train_plans=412/412
  train_cases=2603/2603
  train_samples=663272
  train_batches=647
  epochs=100
```

ความเห็นแบบไม่อวย:

- `quarter` ช่วยลดเวลาต่อ epoch ได้มาก เพราะ batch ต่อ epoch เหลือประมาณ 1/4 ของ full
- `quarter` ไม่เท่ากับ full training เพราะแต่ละ epoch เห็นข้อมูลไม่ครบ ต้องรายงาน protocol ว่าเป็น stochastic quarter-plan training
- `quarter` 160 epochs ให้ exposure รวมประมาณ 40 full-equivalent epochs ไม่ใช่ 100 full-equivalent epochs
- ถ้าต้องการ exposure เทียบเท่า full 100 epochs ต้องใช้ quarter ประมาณ 400 epochs ซึ่งเวลาโดยรวมจะเข้าใกล้ full มากขึ้น
- เหมาะใช้เป็นรอบทดลองจริงจังกว่า `fast` และช่วยดูแนวโน้ม loss/metric ก่อนตัดสินใจรัน `full`

วิธีใช้:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py
```

หรือรันตรง:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile fast
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile quarter
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile full
```

### 12. แก้ manifest path เก่าของ `Method_GridSocialPolicy_SF_01`

ไฟล์ที่แก้:

- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/path_utils.py`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/dataset_grid_policy.py`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/action_space.py`
- `AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01/train_grid_policy.py`

อาการ:

```text
FileNotFoundError:
/home/johnnie/programming/AI_Pedsim/AI_Pedsim/Dataset/Data_TrajectoryGrid/Topo_HouseGAN/A/train/.../walkablearea_grid.json
```

สาเหตุ:

- `dataset_root` ที่ส่งจาก config ถูกต้องแล้ว เป็น `/home/johnfaqpc/...`
- แต่ `manifest_trajectory_grid.csv` ยังเก็บ `input_dir` และ `target_dir` เป็น absolute path เก่าจากเครื่อง/โฟลเดอร์ `/home/johnnie/...`
- `dataset_grid_policy.py` อ่าน path จาก manifest ตรง ๆ จึงไปหาไฟล์ที่ตำแหน่งเก่า

วิธีแก้:

- เพิ่ม `resolve_manifest_path()` เพื่อแปลง path เก่ากลับมาใต้ `dataset_root` ปัจจุบัน
- ถ้า absolute path ใน manifest มีไฟล์จริง จะใช้ path นั้นตามเดิม
- ถ้า path ไม่มีจริง แต่เจอ marker `Dataset/Data_TrajectoryGrid/Topo_HouseGAN` จะตัด suffix หลัง marker แล้วประกอบกับ `dataset_root` ใหม่
- ใช้ helper นี้ทั้งใน `dataset_grid_policy.py` ตอนโหลด `walkablearea_grid.json`, `exit_room.json`, `trajectory.parquet`
- ใช้ helper นี้ใน `action_space.py` ตอน scan trajectory เพื่อ build action space
- ปรับ log แรกของ quarter sampler ให้ set epoch 1 ก่อนคำนวณ `train_batches` เพื่อไม่ให้ตัวเลขก่อน epoch กับ epoch 1 ไม่ตรงกัน

การตรวจสอบ:

```text
old manifest path:
  /home/johnnie/.../Topo_HouseGAN/A/train/plan_134_85e1/plan_sim_44_02_full

resolved path:
  /home/johnfaqpc/.../Topo_HouseGAN/A/train/plan_134_85e1/plan_sim_44_02_full

walkablearea_grid.json exists:
  True

first quarter train batch:
  map=torch.Size([1024, 3, 33, 33])
  features=torch.Size([1024, 12])
  action=torch.Size([1024])
  stop=torch.Size([1024])
```

### 13. เพิ่ม profile `fast / quarter / full` ให้ `Method_Transformer_SF_01`

ไฟล์ที่แก้:

- `AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01/run_pipeline.py`
- `AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01/config_fast.json`
- `AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01/config_quarter_plan.json`
- `AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01/config_full.json`
- `AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01/README.md`

สิ่งที่ทำ:

- เพิ่มเมนู training profile สำหรับ Transformer-SF:

```text
1) fast - quick debug/sanity training (recommended)
2) quarter - rotate 25% of train plans each epoch
3) full - full research-scale training
```

- ใช้ trainer กลาง `train_joint_sf.py` เหมือน LSTM-SF โดยส่ง `--architecture transformer`
- `fast` ใช้ dataset/model เล็กลงเพื่อ sanity/debug
- `quarter` ใช้ transformer ขนาด full แต่ตั้ง `case_fraction_per_epoch=0.25` และ `group_fraction_by_plan=true`
- `full` ใช้ train split เต็มทุก epoch
- ตั้ง `case_grouped_batches=true` เพื่อให้ windows ของ case เดียวกันอยู่ใกล้กัน ลดการอ่าน parquet/geometry ซ้ำ
- เปิด `amp=true`, `num_workers=2`, `persistent_workers=true`, `prefetch_factor=2` ใน `quarter/full`
- ใช้ `batch_size=4` สำหรับ transformer เพราะ transformer หนักกว่า LSTM; ลด batch count ลงจาก config เดิมที่ batch size 2 แต่ยัง conservative กว่า LSTM batch size 8

จำนวนที่ตรวจจาก config `quarter`:

```text
sampled_train_cases=688/2719
sampled_train_plans=120/478
train_batches=5504
```

ประมาณ full profile:

```text
train_cases=2719
train_plans=478
train_batches=21752
epochs=100
```

ความเห็นแบบไม่อวย:

- `quarter` ของ Transformer-SF เร็วกว่า full ต่อ epoch ประมาณ 4 เท่า
- แต่เพราะใช้ `epochs=160` exposure รวมจะประมาณ 40 full-equivalent epochs ไม่เท่า full 100 epochs
- ถ้าจะใช้เป็นผลวิจัย ต้องบอก protocol ชัดว่าเป็น stochastic quarter-plan training
- ถ้า GPU ยังไหวและอยากลด batch count อีก อาจค่อยทดลองเพิ่ม `batch_size` จาก 4 เป็น 8 แต่ต้องเช็ก VRAM ก่อน

วิธีใช้:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py
```

หรือรันตรง:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile fast
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile quarter
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile full
```

การตรวจสอบ:

```text
py_compile:
  train_joint_sf.py
  Method_Transformer_SF_01/run_pipeline.py

dry-run:
  --profile fast
  --profile quarter
  --profile full
  default --dry-run -> stage=plan

smoke train:
  train_cases=1
  train_batches=1
  val_cases=1
  val_batches=1
  train=0.16513
  val=0.15400
  epoch_seconds=0.9
```

## วิธีรันปัจจุบัน

รันเมนูหลัก:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py
```

เมนูให้เลือกตรวจ configuration, smoke test, training profile, evaluate, train-and-evaluate
หรือดู run ที่มีอยู่ การกด Enter เลือก `Check configuration` และยังไม่เริ่ม train

คำสั่งด้านล่างยังรองรับสำหรับ automation หรือการระบุ operation โดยตรง

รัน fast โดยตรง:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage train --profile fast
```

รัน full research-scale:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage train --profile full
```

รัน quarter-plan profile:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage train --profile quarter
```

รัน half-plan profile:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage train --profile half
```

หรือระบุ config ตรง ๆ:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage train --config-train config_full.json
```

ดู plan อย่างเดียว:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage plan
```

ทดสอบ smoke config:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --stage train --config-train config_smoke.json
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

ยืนยันว่า default stage เป็น `plan` และไม่เริ่ม train

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

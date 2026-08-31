# JuPedSim computational-time benchmark

โฟลเดอร์นี้ใช้วัด **wall-clock computational time** ของ JuPedSim โดยแยกจากเวลาในโลกจำลอง
(`simulation_duration_s`) และไม่แก้ไขไฟล์ใน `Dataset/` หรือ `Geo_scenario/`

## Files

- `Total_RefFileName.csv` คือ allow-list ที่สร้างจากแถว `status=success` ของ
  `Dataset/Data_Estimate_2/{Train,Val,Test}/data_estimate.csv`
- `src/build_total_ref_filename.py` สร้าง allow-list ใหม่จาก source dataset
- `src/benchmark_jupedsim_runtime.py` รัน JuPedSim จาก allow-list และจับเวลา
- `JuPedSim_Runtime.csv` คือผล benchmark เพียงไฟล์เดียวที่อยู่ใน root ของโฟลเดอร์นี้

ระหว่าง benchmark จะสร้าง SQLite ชั่วคราวใน system temporary directory เพื่อให้ JuPedSim ทำงานด้วย
trajectory writer แบบเดียวกับ pipeline เดิม จากนั้นลบทิ้งหลังจบแต่ละ scenario ไม่มีการสร้าง heatmap,
trajectory plot หรือ metadata ใหม่

## Commands

ใช้ Python environment หลักของโครงการ:

```bash
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  Dataset_TimeCalculate/src/build_total_ref_filename.py

/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  Dataset_TimeCalculate/src/benchmark_jupedsim_runtime.py --dry-run

/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  Dataset_TimeCalculate/src/benchmark_jupedsim_runtime.py \
  --mode missing --split test --limit 1 --warmup 0
```

เมื่อ smoke test ผ่านจึงรัน canonical test ทั้ง 862 scenarios:

```bash
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  Dataset_TimeCalculate/src/benchmark_jupedsim_runtime.py --split test
```

เมื่อเรียกโดยไม่ใส่ option ใน terminal โปรแกรมจะแสดงเมนู:

```text
[0] รันใหม่ทั้งหมดและเขียนทับเฉพาะ JuPedSim_Runtime.csv
[1] รันเฉพาะ reference ที่ยังไม่มีผล success ใน JuPedSim_Runtime.csv
[2] ออกโดยไม่รัน (ค่าเริ่มต้น)
```

สำหรับ automation ให้ใช้ mode ที่ชัดเจน การรันต่อจากผลเดิมใช้:

```bash
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  Dataset_TimeCalculate/src/benchmark_jupedsim_runtime.py \
  --mode missing --split test
```

การรันใหม่ทั้งหมดและแทนที่เฉพาะไฟล์ผล benchmark ใช้ `--mode overwrite --split test`.
คำสั่งนี้ไม่แตะ SQLite, metadata, geometry หรือ dataset ต้นทาง

ผลที่ `timeout`, `deadlock` หรือ `error` จะถูกบันทึกลง CSV แล้ว pipeline จะทำ scenario ถัดไป
โดยค่าเริ่มต้นจำกัดแต่ละ simulation ที่ 5 นาทีตาม config เดิม และถือว่า deadlock เมื่อจำนวน agent
ไม่ลดลงเป็นเวลา 60 วินาที

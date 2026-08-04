# AI_CONFIG_NOR_PREVIEW

บันทึกมาตรฐานการสร้างภาพ Preview สำหรับงานเปรียบเทียบ Time-Series Models
ใน `UI_PerformanceCompare` เพื่อให้ Ground Truth และผลจากทุก Method ใช้รูปแบบ
เดียวกันทั้งสี ขอบเขตภาพ ผนัง ประตู legend และชื่อภาพ

อัปเดตล่าสุด: 2026-08-05

## 1. จุดประสงค์

- ใช้เป็น visual contract กลางของภาพ Ground Truth และ AI rollout ทุก Method
- ทำให้ภาพนำไปวางโครง UI และต้นฉบับวิทยานิพนธ์ได้ในรูปแบบเดียวกัน
- แยกความถูกต้องของการแสดงผลออกจากความถูกต้องของผลวิจัยและ checkpoint
- ป้องกันการ crop ผังตามระยะที่โมเดลเดิน เพราะจะทำให้เห็น floorplan เพียงบางส่วน

`UI_PerformanceCompare/Streamlit` ทำหน้าที่ค้นหาและแสดงไฟล์ PNG เป็นหลัก ไม่ใช่
source renderer ของภาพ มาตรฐานเดิมที่ใช้อ้างอิงอยู่ใน:

- `AI_GenerateTimeseries/generate_gt_previews.py`
- `AI_GenerateTimeseries/AI_Train/Method_GridSocialPolicy/rollout.py`

Transformer renderer ปัจจุบันอยู่ใน:

- `AI_GenerateTimeseries/AI_Train/Method_Transformer/test_transformer.py`

## 2. Standard visual contract

### Canvas และขอบภาพ

- Figure ภายนอกเป็นสีขาว เพื่อให้มีกรอบกระดาษสีขาวและพื้นที่หัวเรื่อง
- พื้นหลังภายในบริเวณ floorplan ใช้ `#101820`
- ใช้ `dpi=150`
- บันทึกด้วย `bbox_inches="tight"`
- ซ่อนแกนด้วย `ax.set_axis_off()` แต่ต้องสร้าง dark-background artist แยกต่างหาก
  เพื่อไม่ให้พื้นภายในหายไปพร้อม axes patch
- Axis limits ต้องคำนวณจาก geometry ทั้งผัง รวม exit area ห้ามคำนวณจาก
  predicted trajectories
- ใช้ `ax.set_aspect("equal", adjustable="box")`

### Dynamic figure size

คำนวณจาก bounds ของ walkable geometry:

```python
aspect = max(floorplan_width / max(floorplan_height, 1e-6), 0.25)
fig_width = min(18.0, max(7.0, 7.0 * aspect))
fig_height = min(12.0, max(4.5, fig_width / aspect))
```

วิธีนี้ทำให้ผังแนวตั้ง แนวนอน และผังจัตุรัสไม่ถูกบีบหรือยืด

### Walkable geometry และผนัง

- Room/corridor fill: `#f3f6f8`
- Wall/edge: `#101820`
- Wall linewidth สำหรับ vector geometry: `2.0`
- Room และ corridor ใช้ fill สีเดียวกัน
- เส้นขอบภายนอกจะกลืนกับ dark background ตามธรรมชาติ
- เส้นขอบภายในทำหน้าที่เป็นผนัง และต้องเว้นช่องเฉพาะตำแหน่งประตู

### ประตูแบบ void

- ห้ามใช้กล่องประตูสีเหลือง
- อ่านตำแหน่งจาก `Geo_door.json`
- ใช้ `door_width` จากข้อมูล; fallback ได้ที่ `1.5 m` เมื่อ field หาย
- ถ้า `horizontal=true` ให้ช่องเปิดยาวตามแกน x
- ถ้า `horizontal=false` ให้ช่องเปิดยาวตามแกน y
- วาด rectangle สีเดียวกับ walkable (`#f3f6f8`) ทับ wall stroke
- ความหนาตั้งฉากกับแนวประตูใช้ประมาณ `0.18 m`
- Door eraser ต้องอยู่เหนือ room/corridor walls แต่ต่ำกว่า trajectory
- ผลที่ต้องการคือผนังขาดเป็นช่องเปิดจริง ไม่ใช่สัญลักษณ์ประตู

### Exit room

- Fill: `#f59e0b`
- Alpha: `0.35`
- Border: `#f97316`
- Border linewidth: `1.3`
- Legend label: `exit room`

### Trajectory และ spawn

- ใช้ Matplotlib default colour cycle เพื่อให้ตรงกับ Ground Truth/Grid rollout
- Trajectory linewidth: `1.2`
- Trajectory alpha: `0.78`
- Spawn fill: `#22c55e`
- Spawn edge: `#052e16`
- Spawn size: `18`
- Spawn edge linewidth: `0.4`
- Legend label: `spawn`
- ภาพของแต่ละ AI Method แสดงเฉพาะ prediction ของ Method นั้น
- ห้ามวาด Ground Truth ซ้อนใน Method preview เว้นแต่กำลังสร้าง diagnostic plot
  ที่แยกชื่อและโฟลเดอร์ออกจาก preview สำหรับ UI

### Title และ legend

- Title ต้องเป็นชื่อ Method โดยตรง เช่น:
  - `Transformer`
  - `GNN-CVAE`
  - `Social GAN`
  - `GridSocialPolicy`
- ห้ามใช้ `DEBUG ONLY`, `Model rollout sample` หรือข้อความสถานะระบบเป็นหัวภาพ
- Warning และ provenance ให้แสดงใน Terminal/metric report แทน
- Legend: `loc="upper right"`, `frameon=True`, `fontsize=8`
- Legend ใช้พื้นสว่างตาม Matplotlib default เพื่อให้เหมือน Ground Truth preview

## 3. Transformer implementation notes

ฟังก์ชันหลัก:

```text
plot_full_case_rollout(...)
```

สิ่งที่ต้องรักษาไว้:

1. โหลด `Geo_room.json`, `Geo_corridor.json`, `Geo_door.json`
2. ใช้ geometry bounds ของทั้ง case กำหนด canvas
3. วาด room/corridor และ wall ก่อน
4. วาด door eraser เพื่อสร้าง void
5. วาด exit overlay
6. วาด prediction trajectories และ spawn points
7. ไม่วาด Ground Truth trajectory ใน Method preview
8. ใช้ title `Transformer`

## 4. สถานะ checkpoint ที่ต้องจำ

### Placeholder ปัจจุบัน

- `Method_Transformer/outputs/run_33` ฝึกบน `Topo_bottleneck`
- `run_33_evaluate/weights/best_model.pth` เป็น weights ชุดเดียวกับ `run_33`
- ปัจจุบันนำมาแสดงบน `Topo_HouseGAN` เพื่อวางโครง UI/เล่มชั่วคราวเท่านั้น
- การรันข้าม topology ต้องใช้ `--allow-dataset-mismatch`
- รูป placeholder ใช้ประกอบการจัด layout ได้
- ADE/FDE, wall violation และ goal success จาก mismatch run ห้ามนำไปสรุปเป็น
  ผลงานวิจัยจริง

คำสั่งสร้าง placeholder ปัจจุบัน:

```bash
cd AI_GenerateTimeseries/AI_Train/Method_Transformer

/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  test_transformer.py \
  --config config_test.json \
  --model_path ../../AI_Result/Method_Transformer/outputs/run_33_evaluate/weights/best_model.pth \
  --run_path ../../AI_Result/Method_Transformer/outputs/run_33_evaluate \
  --allow-dataset-mismatch
```

คำเตือน mismatch ต้องคงอยู่ใน Terminal แม้หัวภาพจะแสดงเพียงชื่อ `Transformer`

## 5. งานที่ต้องทำหลังวางโครงเล่มเสร็จ

1. Train `Method_Transformer` ใหม่บน `Topo_HouseGAN`
2. ใช้ split, seed และ preprocessing contract ที่บันทึกแน่นอน
3. ประเมิน checkpoint ใหม่โดยไม่ใช้ `--allow-dataset-mismatch`
4. สร้าง preview ใหม่ทับ placeholder ในโฟลเดอร์ผลของ run ใหม่
5. สรุปผลหลักเป็นหน่วยเมตรและ metric เชิงพื้นที่:
   - ADE (m)
   - FDE (m)
   - Wall-violation rate
   - Goal-success rate
   - Inference latency
6. แยกผลตาม single/half/full occupancy
7. อัปเดต UI และตารางในเล่มพร้อมกันหลังผลสุดท้ายผ่านการตรวจสอบ

คำสั่ง train รุ่นใหม่:

```bash
cd AI_GenerateTimeseries/AI_Train/Method_Transformer

/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  train_transformer.py --config config_train.json
```

คำสั่ง test รุ่นใหม่:

```bash
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  test_transformer.py \
  --config config_test.json \
  --model_path ../../AI_Result/Method_Transformer/outputs/run_NEW/weights/best_model.pth \
  --run_path ../../AI_Result/Method_Transformer/outputs/run_NEW
```

ห้ามใส่ `--allow-dataset-mismatch` ในการประเมินผลวิจัยสุดท้าย

## 6. Verification checklist

ก่อนนำภาพขึ้น UI หรือใส่ในเล่ม ให้ตรวจทุกข้อ:

- [ ] เห็น floorplan ครบทุกห้องและ exit
- [ ] ไม่มีการ crop ตาม predicted trajectory
- [ ] กรอบนอกและบริเวณ title เป็นสีขาว
- [ ] พื้นใน floorplan เป็น `#101820`
- [ ] ห้องและ corridor เป็น `#f3f6f8`
- [ ] ผนังภายในมองเห็นชัด
- [ ] ประตูเป็น void ไม่มี rectangle สีเหลือง
- [ ] Exit ใช้สีและ alpha ตามมาตรฐาน
- [ ] ไม่มี Ground Truth line ใน AI Method preview
- [ ] Title เป็นชื่อ Method
- [ ] Legend รูปแบบเดียวกับ Ground Truth
- [ ] Aspect ratio ไม่บิดเบี้ยว
- [ ] checkpoint และ dataset topology ตรงกันสำหรับผลวิจัยจริง
- [ ] Metric หลักรายงานเป็นเมตร ไม่เฉลี่ย normalized units ข้ามผังโดยตรง

## 7. UI Performance Gallery case mapping

หน้า `Part D: Per-Model Normalized Rollout Gallery` ใน
`UI_PerformanceCompare/Streamlit/views/time_series_output.py` ใช้ case หลักดังนี้:

- `plan_110_fbd0_42_00_full` -> UI key `plan_110_fbd0`
- `plan_102_8e0f_42_00_full` -> UI key `plan_102_8e0f`
- `plan_110_fbd0_100044_02_half` -> UI key `plan_110_fbd0_half`

สอง plan แรกอยู่ใน split `train` ของ `Data_Traj_Table/Topo_HouseGAN` ไม่ได้อยู่ใน
split `test` ดังนั้นการรัน `test_transformer.py` แบบ default จะไม่มี PNG เหล่านี้
และ UI จะแสดง `Transformer (...) Missing` แม้ใน run จะมีภาพ test อื่นครบแล้ว

สำหรับช่วงวางโครง UI สามารถสร้าง placeholder ด้วย `--split train --case-id ...`
และเก็บใต้:

```text
Method_Transformer/outputs/run_33_evaluate/gallery_placeholders/
```

UI scanner ค้น PNG แบบ recursive และดึง key ด้วย pattern
`plan_<number>_<hex>`; กรณี `plan_110_fbd0_100044_02_half` มี mapping พิเศษเป็น
`plan_110_fbd0_half` อยู่แล้ว

ภาพจาก train split และ mismatched checkpoint เป็น gallery placeholder เท่านั้น
ห้ามใช้ metric เป็นผลวิจัยสุดท้าย หลัง train HouseGAN ใหม่ต้องสร้าง matched evaluation
suite ที่กำหนด floorplan cases ชุดเดียวกันสำหรับทุก Method

## 8. Multi-method framing previews (2026-08-05)

Shared renderer:

```text
AI_GenerateTimeseries/AI_Train/normalized_preview.py
```

Framing generator:

```bash
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  AI_GenerateTimeseries/generate_framing_previews.py \
  --methods gnn sgan lstm gpt
```

Rules:

- The generator executes each method's own checkpoint or retrieval pipeline.
- It does not draw Ground Truth trajectory lines.
- It does not replace a model prediction with A*, interpolation, or seeded curves.
- Output is stored under each evaluate run's `framing_previews/` directory.
- Every method has `framing_previews/preview_manifest.json` with provenance and research-validity status.
- UI gives `framing_previews` priority over legacy A*-generated PNG files.
- All three current gallery cases are from the HouseGAN `train` split. They are for page composition, not held-out metrics.

Current method status:

| Method | Preview source | Provenance status | Required before final research result |
|---|---|---|---|
| Transformer | `run_33` forward inference | trained on `Topo_bottleneck`; HouseGAN mismatch | retrain on frozen HouseGAN train split |
| GNN-CVAE | `run_6` forward inference | trained on `Topo_bottleneck`; HouseGAN mismatch | retrain on frozen HouseGAN train split |
| Social GAN | `run_6/sgan_ep10.pth` forward inference | legacy run has no dataset snapshot; provenance unverified | repair nested-case loader, retrain, save checkpoint metadata |
| LSTM | `run_LSTM_20260327_184506` recursive inference | legacy trainer hard-codes `Topo_bottleneck`; HouseGAN mismatch | retrain on frozen HouseGAN train split |
| GPT+RAG | `topo_bottleneck_v3` retrieval and geometric transfer | index contains both bottleneck and HouseGAN train cases | rebuild/freeze leakage-safe knowledge index and evaluate held-out HouseGAN test |

The old normalized gallery PNGs for LSTM and GPT+RAG were created by the disabled
`run_housegan_evaluations_normalized.py` A* path generator. They remain on disk for
traceability but must not be selected by the UI or cited as model outputs.

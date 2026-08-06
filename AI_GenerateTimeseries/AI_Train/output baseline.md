# Baseline Output Structure Standard

เอกสารนี้กำหนดมาตรฐานชื่อ directory, file และ schema ของผลลัพธ์จากทุกวิธีในงานวิจัย เพื่อให้ UI, Metric Evaluator และการสรุปผลสำหรับเปเปอร์อ่านข้อมูลในรูปแบบเดียวกัน

## 1. หลักการ

- หนึ่ง training run ต้องมี directory ของตัวเองและห้ามเขียนทับ run เดิม
- checkpoint, framing preview และผล final evaluation ต้องแยกออกจากกัน
- ผล `raw` และ `constrained` ต้องแยกไฟล์ ห้ามแทนที่ raw prediction
- Ground Truth ต้องอ้างอิงจาก canonical dataset ห้ามบันทึกปะปนเป็น model prediction
- ทุก evaluation ต้องตรวจ checkpoint/dataset/split compatibility
- ผลที่ไม่ผ่าน compatibility check ต้องมี `research_valid: false`
- ทุกวิธีต้องส่งออก common trajectory schema เดียวกันก่อนคำนวณ Metric

## 2. Method directory names

ชื่อ method ต้องคงที่และตรงกันระหว่าง training code, result directory, UI และตารางผลวิจัย

```text
Method_Transformer_SF_01
Method_SGAN_SF_01
Method_LSTM_SF_01
Method_GNN_CVAE2
Method_KR_GT
Method_GridSocialPolicy
Method_GridSocialPolicy_SF_01
```

ชื่อที่ใช้แสดงในเปเปอร์สามารถกำหนดใน `method_manifest.json` โดยไม่เปลี่ยนชื่อ directory

## 3. Output root

ผลลัพธ์ทั้งหมดเก็บภายใต้:

```text
AI_GenerateTimeseries/
└── AI_Result/
    └── <method_id>/
        └── outputs/
            └── <run_id>/
```

สำหรับ GridPolicy ซึ่งมีโค้ดหลักอยู่ใน `AI_GenerateTrajectoryGrid` ให้ใช้โครงสร้างภายใน run แบบเดียวกัน แม้ output root จะอยู่ต่าง module

## 4. Run ID

รูปแบบที่แนะนำ:

```text
run_<UTC timestamp>_seed<seed>
```

ตัวอย่าง:

```text
run_20260806T091500Z_seed042
run_20260806T091500Z_seed043
run_20260806T091500Z_seed044
```

ห้ามใส่คำว่า `evaluate`, `test`, `debug` หรือชื่อ dataset ต่อท้าย run ID เพราะ evaluation หลายชุดต้องอ้างอิง checkpoint run เดียวกันผ่าน `evaluations/`

## 5. Standard run structure

ทุก method directory ต้องมี `run_pipeline.py` เป็น entry point กลาง โดยใช้
interface ต่อไปนี้:

```text
python run_pipeline.py --stage plan
python run_pipeline.py --stage train
python run_pipeline.py --stage evaluate --run-path <run_dir>
python run_pipeline.py --stage all
```

`plan` ต้องเป็นค่าเริ่มต้นเพื่อป้องกันการเริ่มงานฝึกโดยไม่ตั้งใจ และทุก
pipeline ต้องรองรับ `--dry-run` สำหรับตรวจคำสั่งโดยไม่เปลี่ยนแปลงผลลัพธ์

```text
<run_id>/
├── run_manifest.json
├── method_manifest.json
├── config_train.json
├── dataset_manifest_snapshot.csv
├── environment.json
├── code_provenance.json
├── checkpoints/
│   ├── best_model.pth
│   ├── latest_model.pth
│   └── checkpoint_manifest.json
├── logs/
│   ├── training_history.csv
│   ├── validation_history.csv
│   └── events.log
├── diagnostics/
│   ├── loss_curves.png
│   ├── learning_rate.png
│   └── validation_samples/
├── framing_previews/
│   ├── framing_manifest.json
│   └── <case_id>/
│       ├── raw_preview.png
│       ├── constrained_preview.png
│       └── preview_metadata.json
└── evaluations/
    └── <evaluation_id>/
```

## 6. Evaluation ID

รูปแบบที่แนะนำ:

```text
eval_<dataset_id>_<split>_<protocol_version>
```

ตัวอย่าง:

```text
eval_housegan_canonical_test_v1
eval_housegan_canonical_val_v1
```

หากเปลี่ยน observation length, prediction horizon, sampling interval หรือ Metric definition ต้องเพิ่ม protocol version ใหม่ ห้ามเขียนทับ evaluation เดิม

## 7. Standard evaluation structure

```text
<evaluation_id>/
├── evaluation_manifest.json
├── evaluation_config.json
├── checkpoint_ref.json
├── dataset_ref.json
├── ground_truth_ref.json
├── predictions/
│   └── <case_id>/
│       ├── prediction_raw.parquet
│       ├── prediction_constrained.parquet
│       ├── action_trace.parquet
│       ├── constraint_events.parquet
│       └── prediction_metadata.json
├── metrics/
│   ├── per_agent_metrics.csv
│   ├── per_case_metrics.csv
│   ├── per_floorplan_metrics.csv
│   ├── summary_metrics.csv
│   ├── confidence_intervals.csv
│   └── failure_cases.csv
├── statistics/
│   ├── bootstrap_results.csv
│   ├── paired_test_results.csv
│   └── effect_sizes.csv
├── previews/
│   └── <case_id>/
│       ├── raw_rollout.png
│       ├── constrained_rollout.png
│       └── comparison.png
└── reports/
    ├── evaluation_summary.json
    └── evaluation_summary.md
```

ไฟล์ `prediction_constrained.parquet`, `action_trace.parquet` และ `constraint_events.parquet` ใช้เฉพาะวิธีที่มี safety executor หากไม่มีให้ระบุใน manifest ว่า `constraint_mode: none` โดยไม่สร้างไฟล์เปล่า

## 8. Common prediction schema

ทุกวิธีต้องส่งออกอย่างน้อย:

| Column | Type | Description |
|---|---|---|
| `case_id` | string | Scenario identifier |
| `split` | string | `train`, `val` หรือ `test` |
| `frame` | int64 | Simulation frame |
| `agent_id` | int64 | Agent identifier |
| `pos_x` | float64 | World-coordinate X in metres |
| `pos_y` | float64 | World-coordinate Y in metres |
| `is_active` | bool | Agent ยังอยู่ใน rollout หรือไม่ |

Optional columns:

```text
vel_x
vel_y
acc_x
acc_y
stop_probability
reached_exit
```

ห้ามเก็บ normalized coordinates เป็น `pos_x` และ `pos_y` ผลลัพธ์ต้อง inverse-transform กลับเป็น world coordinates ก่อนบันทึก

## 9. Stochastic prediction schema

SGAN-SF และ GNN-CVAE ต้องเพิ่ม:

| Column | Type | Description |
|---|---|---|
| `sample_id` | int32 | หมายเลข stochastic sample ตั้งแต่ `0` ถึง `K-1` |
| `sample_seed` | int64 | Seed ที่ใช้สร้าง sample |

ค่า `K` และ sampling policy ต้องบันทึกใน `evaluation_config.json` ห้ามเลือก best sample โดยไม่มีการรายงาน mean@K ควบคู่กัน

## 10. Action trace schema

ใช้กับ GridPolicy หรือ continuous model ที่มี safety executor:

```text
case_id
frame
agent_id
proposed_pos_x
proposed_pos_y
executed_pos_x
executed_pos_y
proposed_action
executed_action
intervened
intervention_type
blocked_by_wall
blocked_by_collision
stopped_at_exit
```

`intervention_type` ใช้ค่ามาตรฐาน:

```text
none
wall
out_of_bounds
collision
speed
acceleration
exit
```

## 11. Required manifests

### `run_manifest.json`

```json
{
  "run_id": "run_20260806T091500Z_seed042",
  "method_id": "Method_Transformer_SF_01",
  "seed": 42,
  "status": "completed",
  "dataset_id": "housegan_canonical_imagebase_split_v1",
  "train_split": "train",
  "validation_split": "val",
  "research_valid": true
}
```

### `evaluation_manifest.json`

ต้องมีอย่างน้อย:

```text
evaluation_id
method_id
run_id
checkpoint_path
checkpoint_sha256
dataset_id
dataset_manifest_sha256
split
case_count
floorplan_count
observation_frames
prediction_horizon
frame_stride
coordinate_system
constraint_mode
stochastic_sample_count
research_valid
invalid_reason
created_at_utc
```

Final HouseGAN test evaluation ที่สมบูรณ์ต้องมี:

```text
split = test
case_count = 862
floorplan_count = 117
research_valid = true
```

## 12. Framing preview rules

`framing_previews/` ใช้ตรวจ UI, geometry alignment และรูปแบบ trajectory เท่านั้น ไม่ใช่ final evaluation

`framing_manifest.json` ต้องระบุ:

```json
{
  "purpose": "framing_only",
  "research_valid": false,
  "invalid_reason": "preview subset; not standardized final evaluation"
}
```

Preview ต้อง:

- แสดงชื่อ method
- ใช้พื้นนอกผังสีขาว
- ใช้พื้นที่ non-walkable สีเข้ม
- ใช้พื้นที่ walkable สีอ่อน
- เจาะช่องประตูเป็น void
- แสดง exit และ spawn ตามมาตรฐานกลาง
- ไม่วาด Ground Truth trajectory ทับ model rollout

## 13. Raw and constrained reporting

ชื่อที่ใช้ในตารางและ UI:

```text
Transformer-SF-Raw
Transformer-SF-Constrained
SGAN-SF-Raw
SGAN-SF-Constrained
LSTM-SF-Raw
LSTM-SF-Constrained
GNN-CVAE-Raw
GNN-CVAE-Constrained
GridPolicy-Raw
GridPolicy-Full
KR-GT
```

ผล constrained ห้ามใช้แทน raw result และต้องรายงาน Constraint Intervention Rate เสมอ

## 14. Metric output requirements

`summary_metrics.csv` ต้องมีหนึ่งแถวต่อ method variant และ seed โดยมีอย่างน้อย:

```text
method_id
variant
seed
ADE
FDE
path_length_error
evacuation_time_error
out_of_bounds_rate
wall_crossing_rate
collision_exposure_rate
invalid_step_rate
goal_reach_rate
exit_flow_error
density_map_error
constraint_intervention_rate
latency_ms_per_agent_step
real_time_factor
```

Aggregate ข้าม seed ต้องสร้างเป็นไฟล์ใหม่ ห้ามเขียนทับผลราย seed

## 15. Research-validity gate

ก่อนตั้ง `research_valid: true` ต้องผ่านทุกข้อ:

- checkpoint ฝึกด้วย canonical training split
- evaluation ใช้ canonical test split
- ไม่มี plan overlap ระหว่าง train และ test
- มีครบ 862 test scenarios และ 117 test floor plans
- coordinate inverse transformation ถูกต้อง
- prediction และ Ground Truth ใช้ frame interval เดียวกัน
- stochastic protocol และ seed ถูกบันทึก
- raw output ไม่ผ่าน rule-based correction
- constrained output มี action/constraint trace
- Metric config และ code provenance ถูกบันทึก

หากไม่ผ่านข้อใด ต้องเก็บผลไว้ได้เพื่อ debugging/framing แต่ต้องตั้ง `research_valid: false` พร้อม `invalid_reason`

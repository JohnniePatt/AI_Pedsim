# AI_Pedsim Project Memory (AI_README)

เอกสารนี้คือสรุปภาพรวมโปรเจกต์ + โครงสร้างไฟล์ + workflow ปัจจุบัน  
เป้าหมายคือ: ถ้า chat history หาย สามารถเปิดไฟล์นี้แล้วทำงานต่อได้ทันที

## สถานะล่าสุด: Image-model comparison (ตรวจแล้ว 2026-09-03)

หน้า `UI_PerformanceCompare/Streamlit` ใช้ชุด **corrected representative 2x2**
เป็นค่าเริ่มต้นจากไฟล์:

```text
AI_GenerateImage/model_performance_compare_lock.json
```

ชุดปัจจุบันประกอบด้วย:

1. `Plain U-Net (corrected shared U-Net)` — factorial run, seed 42
2. `Pix2Pix WGAN-GP (corrected shared U-Net)` — factorial run, seed 42
3. `ResNet-9 (original)` — original research run
4. `Pix2PixHD (original full method)` — original research run

คู่ U-Net ใช้ Generator architecture เดียวกันเพื่อแก้ปัญหาการเปรียบเทียบเดิม
ส่วน ResNet-9/Pix2PixHD เก็บ original full-method implementation ตามงานวิจัยเดิม
จึงต้องเรียกชุดนี้ว่า **method-family representative comparison** ไม่ใช่ strict
component-isolated factorial

ไม่ได้เทรนโมเดลเพิ่มตอนสร้าง comparison นี้ ระบบอ่าน prediction ที่มีอยู่แล้วและ
ประเมินใหม่ด้วย protocol `image_density_representative_common_png_256_v1` โดยใช้
saved uint8 PNG, resize แบบ bilinear เป็น 256 x 256 และ canonical HouseGAN test
ครบ 862 scenarios / 117 floor plans ผลหลักคือ:

| Model | MAE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Pix2PixHD (original full method) | **0.001184** | **0.966899** | **0.031826** |
| Pix2Pix WGAN-GP (corrected shared U-Net) | 0.001391 | 0.939124 | 0.040318 |
| Plain U-Net (corrected shared U-Net) | 0.001414 | 0.936195 | 0.041961 |
| ResNet-9 (original) | 0.001508 | 0.941230 | 0.039556 |

ผลและ provenance อยู่ที่:

```text
AI_GenerateImage/AI_Result/RepresentativeComparisons/
  comparison_20260903T143319Z_corrected_representative_2x2_256_v1/
```

### Computational-time source of truth

ตาราง Computational Efficiency ใน UI ห้าม hard-code ตัวเลข แต่ต้องอ่าน:

- AI: `<selected_run>/test_runtime.csv`
- JuPedSim: `Dataset_TimeCalculate/JuPedSim_Runtime.csv`

Image Based Output ใช้ total-runtime comparison: AI ใช้
`test_pipeline_wall_time_s` และ JuPedSim ใช้ `total_wall_time_s` ซึ่งรวม setup,
simulation, SQLite output, trajectory plotting และ density-heatmap generation

| Method | Total Runtime (s) | Average/sample (s) | Speedup |
|---|---:|---:|---:|
| JuPedSim simulation + outputs | 23,649.470879 | 27.435581066 | 1.0x |
| Pix2PixHD | 35.329996 | 0.040986074 | 669.4x |
| ResNet-9 | 25.056682 | 0.029068077 | 943.8x |
| Pix2Pix WGAN-GP | 15.750595 | 0.018272152 | 1,501.5x |
| Plain U-Net | 16.252050 | 0.018853886 | 1,455.2x |

AI runtime ชุดนี้วัดบน `NVIDIA GeForce RTX 5070 Laptop GPU` ตาม artifact จริง
ส่วน JuPedSim runtime เก็บ platform เป็น `x86_64` บน WSL2 แต่ไม่ได้เก็บชื่อ CPU
รุ่นเต็ม จึงห้ามระบุว่าเป็น Intel Core i9 หากไม่มีหลักฐานเพิ่ม

สถานะ comparison โดยรวมเป็น `research_valid: false` เพราะ legacy ResNet-9 และ
Pix2PixHD ไม่มี seed ใน modern provenance manifest แม้ checkpoint hash, canonical
split และ prediction 862 เคสจะตรวจครบ ผลใช้เป็น descriptive comparison ได้ แต่ยัง
ไม่ควรอ้างเป็นผลหลาย seed หรือผล factorial เชิงสถิติ

### Summary Output: MLP/GNN/XGBoost efficiency

ตาราง Computational Efficiency ในหน้า `Summary Output` เป็นคนละตารางกับ image
models และต้องแสดงเฉพาะ MLP, GNN และ XGBoost ที่เลือกอยู่ ฝั่ง JuPedSim ใช้เฉพาะ
`simulation_wall_time_s` ไม่รวม trajectory plotting หรือ density heatmap ส่วน AI
ใช้ `duration_seconds` จาก `test_runtime.json` ตาม artifact ที่มีอยู่:

| Method | Hardware | Samples | Total Runtime (s) | Average/sample (s) | Speedup |
|---|---|---:|---:|---:|---:|
| JuPedSim simulation only | CPU (`x86_64`; model not recorded) | 862 | 6,049.220102 | 7.017656731 | 1.0x |
| MLP | GPU (CUDA; model not recorded) | 862 | 3.687032 | 0.004277299 | 1,640.7x |
| GNN | GPU (CUDA; model not recorded) | 862 | 2.941674 | 0.003412615 | 2,056.4x |
| XGBoost | CPU (model not recorded) | 862 | 1.376539 | 0.001596913 | 4,394.5x |

ค่า AI เหล่านี้เป็น full-test process ไม่ใช่ inference-only โดย MLP/GNN บันทึก
device เป็น CUDA และ XGBoost เป็น CPU ชื่อรุ่น GPU/CPU เต็มไม่ได้ถูกบันทึกใน run
เหล่านี้ จึงห้ามเติมชื่อรุ่นจากการคาดเดา

## 1) ภาพรวมโปรเจกต์

โปรเจกต์นี้มี 2 งานหลักที่ทำคู่กัน:

1. `GeneratePlan_HouseGAN`  
- สร้างผัง (topology/geometry)  
- ทำ simulation ความหนาแน่น (Jupedsim/Social Force workflow)  
- สรุปผลเวลาและ route features

2. `AI_Estimate`  
- เทรนโมเดล MLP เพื่อทำนายเวลาเดินทางจาก A -> B  
- ทำนาย 3 ค่า:
  - `min_agent_time_s`
  - `mean_agent_time_s`
  - `max_agent_time_s`

## 2) สถานะที่ทำเสร็จแล้ว (สำคัญ)

- รีโครงสร้าง `AI_Estimate/AI_Train` เป็น method-based แล้ว
  - `Method_MLP_PyTorch` (ของเดิมที่แยกแล้ว)
  - `Method_MLP_Keras` (ของใหม่ที่เพิ่ม)
  - `Method_GNN` (ของกราฟ - ก้าวข้ามข้อจำกัด MLP)
- รีโครงสร้าง output เป็น `Method_<วิธี>` แล้ว
  - `AI_Estimate/AI_result/Method_MLP_PyTorch/outputs/<run_name>`
  - `AI_Estimate/AI_result/Method_MLP_Keras/outputs/<run_name>`
  - `AI_Estimate/AI_result/Method_GNN/outputs/<run_name>`
- ปรับ Streamlit ของ `AI_Estimate` ให้เลือก method ได้จาก sidebar
- ตัดการใช้ `observed_agents` ออก และใช้ `computed_agents` อย่างเดียวใน pipeline หลัก

## 3) Data/Feature Concept (AI_Estimate)

### Input หลักที่ใช้ตอนนี้ (feature numeric)
- `computed_agents`
- `topology_hop_distance`
- `topology_centerline_distance_m`
- `straight_distance_m`
- `detour_ratio`
- `distance_gap_m`
- `number_of_rooms_between_A_B`
- `door_count_between_A_B`
- `min_door_width_between_A_B`
- `walkable_area_near_path`
- `bottleneck_score`
- `agent_density_near_path`
- `area_per_agent`
- `door_pressure_per_agent`

### Categorical
- `variant_id` -> one-hot (`variant_full`, `variant_half`, `variant_single`)

### Target
- `min_agent_time_s`, `mean_agent_time_s`, `max_agent_time_s`

## 4) โครงสร้างโฟลเดอร์หลัก (ปัจจุบัน)

```text
AI_Pedsim/
├─ AI_README.md
├─ Dataset/
│  ├─ Data_Estimate/
│  │  ├─ Train/data_estimate.csv
│  │  ├─ Val/data_estimate.csv
│  │  ├─ Test/data_estimate.csv
│  │  └─ data_estimate_manifest.json
│  └─ Data_Traj_Table/
├─ Geo_scenario/
│  └─ Topo_HouseGAN/
│     ├─ geo/
│     ├─ dataswarm/
│     ├─ time_summary/
│     └─ route_information/
├─ GeneratePlan_HouseGAN/
│  ├─ Prepare_data/
│  │  ├─ config_housegan.json
│  │  ├─ generate_layout.py
│  │  └─ generate_route_information.py
│  ├─ Simulation/
│  │  ├─ config_density_sim.json
│  │  └─ density_housegan_sim.py
│  └─ Streamlit_ui/
│     └─ app.py
└─ AI_Estimate/
   ├─ AI_Train/
   │  ├─ Method_MLP_PyTorch/
   │  │  ├─ config_train.json
   │  │  ├─ dataset.py
   │  │  ├─ model.py
   │  │  ├─ train_time_estimator.py
   │  │  ├─ test_time_estimator.py
   │  │  └─ visual_time_estimator.py
   │  ├─ Method_MLP_Keras/
   │  │  ├─ config_train.json
   │  │  ├─ dataset_keras.py
   │  │  ├─ train_time_estimator.py
   │  │  ├─ test_time_estimator.py
   │  │  └─ visual_time_estimator.py
   │  └─ Method_GNN/
   │     ├─ config_train.json
   │     ├─ dataset_gnn.py
   │     ├─ model.py
   │     ├─ train_time_estimator.py
   │     ├─ test_time_estimator.py
   │     └─ visual_time_estimator.py
   ├─ Utilities/
   │  └─ format_data_estimate.py
   ├─ Streamlit_ui/
   │  └─ app.py
   └─ AI_result/
      ├─ Method_MLP_PyTorch/outputs/
      ├─ Method_MLP_Keras/outputs/
      └─ Method_GNN/outputs/
```

## 5) Workflow แนะนำ (ทำงานต่อ)

### A) เตรียมข้อมูล estimate
```bash
python AI_Estimate/Utilities/format_data_estimate.py --config AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json
```

### B) เทรน PyTorch
```bash
python AI_Estimate/AI_Train/Method_MLP_PyTorch/train_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json
```

### C) เทรน Keras
```bash
python AI_Estimate/AI_Train/Method_MLP_Keras/train_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_Keras/config_train.json
```

### D) Test
```bash
python AI_Estimate/AI_Train/Method_MLP_PyTorch/test_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json
python AI_Estimate/AI_Train/Method_MLP_Keras/test_time_estimator.py --config AI_Estimate/AI_Train/Method_MLP_Keras/config_train.json
```

### E) Visual report
```bash
python AI_Estimate/AI_Train/Method_MLP_PyTorch/visual_time_estimator.py --run-dir AI_Estimate/AI_result/Method_MLP_PyTorch/outputs/<run_name>
python AI_Estimate/AI_Train/Method_MLP_Keras/visual_time_estimator.py --run-dir AI_Estimate/AI_result/Method_MLP_Keras/outputs/<run_name>
python AI_Estimate/AI_Train/Method_GNN/visual_time_estimator.py --run-dir AI_Estimate/AI_result/Method_GNN/outputs/<run_name>
```

### F) Streamlit
```bash
streamlit run AI_Estimate/Streamlit_ui/app.py
streamlit run GeneratePlan_HouseGAN/Streamlit_ui/app.py
```

## 6) ข้อควรระวัง/แนวทางต่อ

- ต้องรันใน environment เดียวกับโปรเจกต์ (`AI_Pedsim-env`) เสมอ
- ไฟล์ `.pth`/`.keras` เก่าเป็น artifact เก่า อาจมี schema เดิมฝังอยู่ เป็นเรื่องปกติ
- ถ้าปรับ feature ใหม่ ให้ sync 3 จุด:
  1. config
  2. dataset loader
  3. run metadata/manifest

---

ถ้าจะต่อยอดงานทันที แนะนำเริ่มจาก:
1. รัน train ทั้ง PyTorch + Keras อย่างละ 1 run
2. เทียบ `test_metrics.json`
3. เลือก baseline หลัก 1 วิธี แล้วค่อยปรับ feature/optimizer ต่อ

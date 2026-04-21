# AI_Pedsim Project Memory (AI_README)

เอกสารนี้คือสรุปภาพรวมโปรเจกต์ + โครงสร้างไฟล์ + workflow ปัจจุบัน  
เป้าหมายคือ: ถ้า chat history หาย สามารถเปิดไฟล์นี้แล้วทำงานต่อได้ทันที

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


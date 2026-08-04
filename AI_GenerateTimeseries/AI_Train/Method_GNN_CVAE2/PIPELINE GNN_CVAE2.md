# Pipeline: Spatial Graph CVAE สำหรับ Goal-Conditioned Pedestrian Trajectory Generation

---

## ภาพรวม

งานนี้เป็น **feasibility study** ว่า AI สามารถเรียนรู้จาก Social Force Model (SFM) simulation แล้ว generate full pedestrian trajectory ได้ไหม โดยรับแค่ spawn position + exit goal + geometry ของพื้นที่ โดยไม่ต้อง observe การเดินก่อนเลย

---

## ปัญหาที่แก้

Social Force Model (Helbing & Molnar, 1995) ใช้สมการ Newtonian จำลองการเคลื่อนที่ของ pedestrian ซึ่งมีข้อจำกัดคือสมการเป็น hand-crafted และอาจไม่สมจริงในบางสถานการณ์ งานนี้จึงพยายามให้ AI **เรียนรู้ pattern การเดินจาก SFM output** โดยตรง แทนที่จะใช้สมการ

สิ่งที่ต่างจากงานอื่นในวรรณกรรม เช่น CrowdSim (Shi et al., WWW 2023) คือ formulation ของเราเป็น **pure goal-conditioned** — ไม่มี historical observation เลยสักก้าว ซึ่งหายากมากในวรรณกรรมเพราะ benchmark ส่วนใหญ่ออกแบบมาสำหรับ trajectory prediction ที่ต้องมี observation ก่อน

---

## เครื่องมือที่ใช้

| เครื่องมือ | บทบาท |
|-----------|-------|
| **JuPedSim** | Pedestrian simulator ใช้ Social Force Model สร้าง synthetic training data |
| **PyTorch** | Framework หลักสำหรับ train model |
| **Shapely** | จัดการ geometry polygon (room, corridor, door) |
| **Pandas / Parquet** | เก็บและอ่าน trajectory data |
| **NumPy** | คำนวณ array operations |

---

## Data ที่มีต่อ 1 simulation case

```
case_XXXXXX/
├── plan_*_trajectory_data.parquet   ← trajectory จาก JuPedSim
├── Spawn_location_*.csv             ← ตำแหน่งเริ่มต้นของแต่ละ agent
├── Spawn_exit_*.csv                 ← polygon ของ exit area
├── Geo_room.json                    ← polygon ของแต่ละห้อง
├── Geo_corridor.json                ← polygon ของ corridor
└── Geo_door.json                    ← ตำแหน่งและความเชื่อมต่อของประตู
```

### Trajectory Parquet (ต่อ case)
| column | ความหมาย |
|--------|----------|
| `frame` | timestep (integer) |
| `id` | agent ID |
| `pos_x`, `pos_y` | ตำแหน่ง world coordinates (เมตร) |
| `ori_x`, `ori_y` | orientation vector |

---

## Pipeline ทั้งหมด

```
RAW DATA (JuPedSim simulation output)
         │
         ▼
[Step 1] GEOMETRY → SPATIAL GRAPH
         │
         ▼
[Step 2] SPATIAL GRAPH GNN
         │  เรียนรู้ว่าแต่ละ space มีลักษณะยังไง
         ▼
[Step 3] DATASET LOADING + NORMALIZATION
         │
         ▼
[Step 4] AGENT ENCODING → CVAE LATENT
         │  (ใช้เฉพาะตอน train)
         ▼
[Step 5] DECODER LOOP (ทีละ step)
         │  ├── Agent-Room Lookup
         │  ├── Heterogeneous Agent GNN
         │  └── Gaussian Emission Head
         ▼
[Step 6] LOSS COMPUTATION + BACKPROP
         │
         ▼
OUTPUT: Full trajectory (pos_x, pos_y) ต่อ frame ต่อ agent
```

---

## Step 1 — Geometry → Spatial Graph

### ทำอะไร
แปลง JSON polygon ของห้อง, corridor, และประตู ให้เป็น **graph** แทนที่จะ rasterize เป็น pixel grid

### ทำไมถึงเลือกวิธีนี้
วิธีเดิมที่ใช้ CNN กับ occupancy grid 64×64 มีปัญหาหลัก 2 อย่าง:
1. **Global average pooling ทำให้ spatial detail หายหมด** — model ไม่รู้ว่ากำแพงอยู่ตรงไหนจริงๆ
2. **ไม่รู้ structure ของพื้นที่** — ไม่รู้ว่านี่คือ corridor ไม่รู้ว่าประตูเชื่อม room ไหนกับไหน

Graph representation เก็บ semantic information ไว้ได้โดยตรง ว่า "Room-0 เชื่อมกับ Cor-0 ผ่าน Door-0 ที่ตำแหน่ง (3.07, 2.86)"

### Input
- `Geo_room.json` — list ของ polygon coordinates (แต่ละห้อง)
- `Geo_corridor.json` — list ของ polygon coordinates (corridor)
- `Geo_door.json` — list ของ door position + rooms ที่เชื่อมกัน

### Output: Spatial Graph
```
Nodes (N_nodes รวมกัน):
  Room-0  : [cx, cy, area, width, height, 1, 0, 0]  ← type=room
  Room-1  : [cx, cy, area, width, height, 1, 0, 0]
  ...
  Cor-0   : [cx, cy, area, width, height, 0, 1, 0]  ← type=corridor
  Door-0  : [dx, dy, 0,    0.02,  0.02,  0, 0, 1]  ← type=door

Edges (E_edges):
  Room-0 ↔ Door-0  (type 0: space-door)
  Cor-0  ↔ Door-0  (type 0: space-door)
  Room-0 ↔ Cor-0   (type 1: space-space shortcut)
  ...
```

### Agent-Room Assignment
ทุกครั้งที่ agent อยู่ที่ตำแหน่ง (x, y) เราทำ **polygon containment check** ด้วย Shapely:
- ถ้าอยู่ใน polygon ของ Room-2 → agent อยู่ใน node Room-2
- ถ้าไม่อยู่ใน polygon ไหนเลย → ใช้ nearest centroid แทน

---

## Step 2 — Spatial Graph GNN

### ทำอะไร
รัน GNN บน spatial graph เพื่อให้แต่ละ node (ห้อง/corridor/ประตู) ได้ **embedding** ที่สรุปว่าตัวเองเป็นอะไร และเชื่อมกับอะไรบ้าง

### ทำไมถึงใช้ GNN
เพราะ spatial graph เป็น graph โดยธรรมชาติ — ข้อมูลของแต่ละห้องมีความหมายเพิ่มขึ้นเมื่อรู้ว่า neighbor ของมันคืออะไร Room-0 ที่เชื่อมกับ Cor-0 และ Room-2 มีความหมายต่างจาก Room-0 ที่อยู่โดดๆ

### Architecture
```
Input: node_features [N_nodes, 8]
       edge_index    [2, E_edges]
       edge_type     [E_edges]  ← 0=space-door, 1=space-space

2 layers ของ message passing:
  แต่ละ layer:
    1. คำนวณ edge message แยกตาม type:
       - type 0 → edge_mlp_0(h_src, h_dst)
       - type 1 → edge_mlp_1(h_src, h_dst)
    2. Aggregate messages เข้า destination node
    3. Update: h_new = node_update(h_old, agg_type0, agg_type1)
    4. LayerNorm + residual

Output: node_embeds [N_nodes, 64]
```

### สิ่งที่ model เรียนรู้ใน step นี้
- Corridor ที่เชื่อมหลายห้องควรมี embedding ที่แตกต่างจาก corridor ที่เชื่อมแค่ 2 ห้อง
- ห้องที่อยู่ปลายทาง (exit) ควรมี embedding ที่ model จดจำได้
- ประตูแคบกับประตูกว้างมี feature ต่างกัน

---

## Step 3 — Dataset Loading + Normalization

### ทำอะไร
Load trajectory parquet + geometry แล้วแปลง world coordinates → normalized [0, 1] เพื่อให้ model ทำงานใน space เดียวกันทุก case

### Normalization
```python
meta = {min_x, min_y, scale}
gx = (world_x - min_x) / scale   # → [0, 1]
gy = (world_y - min_y) / scale   # → [0, 1]
```

### Trajectory Processing ต่อ case
1. อ่าน parquet → filter เฉพาะ agent ที่ใช้
2. Downsample frames ด้วย `frame_stride=8` (ลด noise จาก SFM)
3. Truncate ที่ `max_seq_len=160`
4. สร้าง agent_mask (True=valid, False=padding)
5. คำนวณ `agent_node_ids` — polygon containment ของ start position ต่อ agent

### Input Tensors ต่อ batch
| tensor | shape | ความหมาย |
|--------|-------|----------|
| `positions` | [B, N, T, 2] | trajectory normalized |
| `agent_mask` | [B, N, T] | valid timestep mask |
| `start_pt` | [B, N, 2] | spawn position normalized |
| `goal_pt` | [B, N, 2] | exit centroid normalized |
| `geo_mask` | [B, 1, 64, 64] | occupancy grid (ยังใช้สำหรับ OOB loss) |
| `node_features` | [N_nodes, 8] | spatial graph node features |
| `edge_index` | [2, E] | spatial graph edges |
| `edge_type` | [E] | edge type (0 หรือ 1) |
| `agent_node_ids` | [B, N] | node idx ของ start position ต่อ agent |

---

## Step 4 — Agent Encoding → CVAE Latent

### ทำอะไร
สรุป trajectory ของแต่ละ agent เป็น **latent vector z** ที่แทน "walking style" ของ agent นั้น

### ทำไมต้องมี CVAE
เพราะ pedestrian แต่ละคนเดินต่างกันแม้จะมี start และ goal เดียวกัน — บางคนเดินตรง บางคนอ้อม บางคนเร็วกว่า CVAE ให้ model สร้าง distribution ของ trajectory ได้แทนที่จะ output เส้นเดียว

### Architecture (Encoder — ใช้เฉพาะ train)
```
Input features ต่อ agent:
  start_pt   [2]  ← spawn position
  goal_pt    [2]  ← exit position
  final_pos  [2]  ← ตำแหน่งสุดท้ายใน GT
  mean_pos   [2]  ← mean position ตลอด trajectory
  mean_vel   [2]  ← mean velocity
  duration   [1]  ← สัดส่วนของ frame ที่ valid

  รวม = 11 features

MLP: 11 → 128 → 128
+ เพิ่ม room_embedding บางส่วน (0.1 weight)

→ mu     [B, N, 32]   ← mean ของ latent distribution
→ logvar [B, N, 32]   ← log variance

Reparameterize: z = mu + eps * exp(0.5 * logvar)
→ latent_z [B, N, 32]
```

### ตอน Inference
ไม่มี GT trajectory → ใช้ **z = 0** (mean of prior) สำหรับ deterministic หรือ sample จาก N(0, I) สำหรับ stochastic

---

## Step 5 — Decoder Loop

### ทำอะไร
Generate trajectory ทีละ step จาก t=0 ถึง T-1 โดยใช้ข้อมูลทั้งหมดที่มี

### ทำไมเป็น Loop ไม่ใช่ one-shot
เพราะต้องการให้ model **react ต่อ agent อื่น** ในแต่ละ timestep — ถ้า generate ทั้งหมดพร้อมกัน model ไม่รู้ว่า agent ข้างๆ อยู่ตรงไหน ณ เวลานั้น

### Decoder Init
```
init_in = [start_pt, goal_pt, latent_z, room_global]
h = tanh(Linear(init_in))   → hidden state [B, N, 128]
current = start_pt
```

### Per-Step Loop (step = 0 → T-2)

#### 5.1 Agent-Room Lookup
ทุก step lookup ว่า agent ปัจจุบันอยู่ใน room ไหน → ดึง node embedding มา:
```
room_local = node_embeds[agent_node_ids]   [B, N, 64]
```

#### 5.2 Heterogeneous Agent GNN (3 layers)
แต่ละ layer ทำ 3 edge types พร้อมกัน:

**Repulsive edges (agent ↔ nearby agents)**
```
rel = pos_j - pos_i                      [B, N, N, 2]
dist = ||rel||                           [B, N, N, 1]

edge_mask = active & ~self & dist ≤ radius

msg = edge_mlp(h_j, rel, dist)
gate = sigmoid(gate_mlp(h_j, rel, dist))
rep_msg = msg * gate                     ← gated (CrowdSim-inspired)

rep_agg = mean(rep_msg over neighbors)  [B, N, 128]
```

เหตุผลที่ใช้ gate: อ้างอิง CrowdSim ที่พิสูจน์ว่า gate mechanism ช่วยให้ model เลือกรับข้อมูลจาก direction ที่สำคัญ (คล้าย visual field ใน SFM)

**Attractive edges (agent ↔ goal)**
```
goal_delta = goal_pt - current           [B, N, 2]
goal_dist  = ||goal_delta||              [B, N, 1]

att_msg = edge_mlp(h, goal_delta, goal_dist)
att_gate = sigmoid(gate_mlp(...))
att_agg = att_msg * att_gate             [B, N, 128]
```

**Room awareness edges (agent ↔ current room)**
```
room_in = [h, room_local]               [B, N, 128+64]

room_msg = edge_mlp(room_in)
room_gate = sigmoid(gate_mlp(room_in))
room_agg = room_msg * room_gate         [B, N, 128]
```

ทั้ง 3 messages รวมกัน:
```
node_in = [h, rep_agg, att_agg, room_agg]  [B, N, 512]
h_new = node_update(node_in)               [B, N, 128]
social = h_new - h                          ← social delta
```

#### 5.3 GRU Cell Update
```
decoder_in = [current, start, goal, latent_z,
              room_global, social, time_t, room_local]
           = [2+2+2+32+64+128+1+64] = 295 dims

h = GRUCell(decoder_in, h)              [B, N, 128]
```

#### 5.4 Gaussian Emission Head (Goal-Anchored)
```
# Linear anchor: เส้นตรงจาก start ไป goal
progress = (step+1) / (T-1)
anchor = start + progress * (goal - start)

# Predict residual จาก anchor
head_in = [h, social, goal_delta]       [B, N, 258]

mu_res     = MLP(head_in)               [B, N, 2]
logvar_res = MLP(head_in)               [B, N, 2]

# Bound residual ด้วย tanh
residual = tanh(mu_res) * max_residual  ← max 0.25 ใน normalized space

# Final position
next_pos = anchor + residual
next_pos = clamp(next_pos, 0, 1)
```

เหตุผลที่ใช้ goal-anchored anchor: ป้องกัน model เดินหลง เพราะตั้งแต่ step แรก anchor ดึงให้ไปทาง goal อยู่แล้ว model เรียนรู้แค่ว่าต้อง bend path ไปทาง corridor ไหน

---

## Step 6 — Loss Computation

### Loss รวม 5 ส่วน

**1. Reconstruction Loss** — path ใกล้ GT
```
L_recon = SmoothL1(pred_positions[:, :, 1:], gt_positions[:, :, 1:])
         เฉพาะ valid timesteps (agent_mask = True)
```

**2. KL Divergence Loss** — regularize latent space
```
L_kl = -0.5 * mean(1 + logvar - mu² - exp(logvar))
ใช้ KL annealing: weight เพิ่มจาก 0 → 0.01 ใน 20 epochs แรก
```

**3. Goal Consistency Loss** — last predicted point ต้องใกล้ goal
```
L_goal = SmoothL1(final_pred, goal_pt)
```

**4. Out-of-Bounds Loss** — ห้ามเดินออกนอก walkable area
```
L_oob = mean(1 - geo_val)
โดย geo_val ≈ 1 ถ้าอยู่ใน walkable, ≈ 0 ถ้าอยู่ใน wall
ใช้ F.grid_sample ทำให้ differentiable ได้
```

**5. Segment OOB Loss** — ห้าม path ตัดผ่านกำแพงระหว่าง 2 frames
```
สุ่ม 5 จุดบน segment ระหว่าง frame t กับ t+1
L_seg_oob = mean(1 - geo_val) บน intermediate points
```

**Total Loss**
```
L = L_recon
  + 0.01 * L_kl
  + 1.0  * L_goal
  + 1.0  * L_oob
  + 0.5  * L_seg_oob
```

---

## สิ่งที่ Model เรียนรู้ทั้งหมด

| Component | เรียนรู้อะไร |
|-----------|------------|
| Spatial Graph GNN | ลักษณะของแต่ละ space และความเชื่อมต่อ |
| Agent Encoder | สรุป walking style เป็น latent vector |
| Repulsive GNN | หลีกเลี่ยง agent อื่นที่อยู่ใกล้ |
| Attractive GNN | เดินไปหา goal |
| Room GNN | navigate ตาม structure ของพื้นที่ |
| GRU Decoder | จำสถานะของการเดินตลอด path |
| Gaussian Head | ความไม่แน่นอนของ trajectory |
| OOB Loss | ไม่เดินทะลุกำแพง |

---

## Output

### ตอน Train
```
pred_positions [B, N, T, 2]  ← trajectory ใน normalized [0, 1]
losses dict                  ← สำหรับ backprop
```

### ตอน Inference (generate)
```
Input:
  start_pt  [B, N, 2]   ← spawn positions (normalized)
  goal_pt   [B, N, 2]   ← exit centroid (normalized)
  geo_mask  [B, 1, H, W] ← occupancy grid
  node_features, edge_index, edge_type  ← spatial graph
  seq_len                ← จำนวน timestep ที่ต้องการ

Output:
  trajectory [B, N, T, 2]  ← normalized
  → แปลงกลับด้วย grid_to_world() → world coordinates (เมตร)
  → เก็บเป็น parquet: frame, id, pos_x, pos_y
```

---

## Curriculum Training (แนะนำ)

เพื่อให้ model เรียนรู้ได้ดี ควร train เป็น stages:

| Step | สิ่งที่เพิ่ม | จุดประสงค์ |
|------|------------|-----------|
| Step 1 | 1 agent, ไม่มี social | เรียนรู้ start→goal เบื้องต้น |
| Step 2 | เพิ่ม geometry loss | เรียนรู้ไม่เดินทะลุกำแพง |
| Step 3 | เพิ่ม multi-agent + social | เรียนรู้หลีกเลี่ยง agent อื่น |
| Step 4 | full dataset | scale ขึ้น |

---

## Metrics ที่ใช้วัด

| Metric | ความหมาย |
|--------|----------|
| **ADE** (Average Displacement Error) | ค่าเฉลี่ยของ error ทุก timestep (เมตร) |
| **FDE** (Final Displacement Error) | error ของ position สุดท้าย vs goal (เมตร) |
| **Collision Rate** | สัดส่วนของ agent คู่ที่ overlap กัน |
| **Out-of-Bounds Rate** | สัดส่วนของ predicted points ที่อยู่นอก walkable area |

# AI_Pedsim Project Instructions

ไฟล์นี้เป็นบริบทถาวรและข้อตกลงการทำงานสำหรับ Codex หรือ AI coding agent ทุกเครื่องที่เปิด repository นี้
ให้ถือว่าเป็นค่าเริ่มต้นของโครงการ แต่คำสั่งล่าสุดที่ผู้ใช้ระบุอย่างชัดเจนใน task ปัจจุบันมีลำดับความสำคัญสูงกว่า

## 1. Project identity and research objective

โครงการนี้ศึกษาการสร้าง full-path pedestrian trajectories บนผังอาคารที่ไม่เคยพบระหว่างการฝึก โดยเปรียบเทียบ
การทำนายพิกัดต่อเนื่องกับ conditional discrete grid policy ภายใต้ข้อมูลผัง เป้าหมาย และสถานการณ์หลาย agent

แกนหลักของ paper คือ:

> การเปลี่ยนปัญหาจาก long-horizon continuous-coordinate prediction ไปเป็นการเลือกการเคลื่อนที่บนกริด
> ซึ่ง conditioned ด้วย floor-plan geometry, destination และ crowd state จะเพิ่ม physical feasibility
> ของ trajectory บน unseen floor plans ได้หรือไม่

อย่าเขียน methodology ให้กลายเป็นเพียงการแข่งขันว่า neural architecture ใดมี ADE/FDE ต่ำที่สุด
ประเด็นหลักคือ representation และ conditioning ส่งผลต่อ physical validity, navigation success และ
long-horizon rollout อย่างไร

### Terminology

- เรียกวิธีหลักว่า **Conditional Discrete Grid Policy** หรือ **GridSocialPolicy**
- อธิบายว่า policy ถูก conditioned ด้วย geometry, goal และ social/crowd state
- อย่าใช้คำว่า hybrid โดยไม่มีคำอธิบาย หากมี rule-based safety executor ให้แยกเป็น post-model constraint layer
- Continuous-coordinate models และ conditional discrete grid policy เป็นคนละ representation family
- ชื่อเต็มของ retrieval baseline คือ **GPT-Assisted Knowledge Retrieval and Geometric Transfer**
- หลีกเลี่ยงตัวย่อ `KR-GT` ในชื่อที่แสดงต่อผู้อ่าน เว้นแต่ได้ให้นิยามไว้ก่อนแล้ว

## 2. Repository map

ตำแหน่งสำคัญ:

```text
AI_GenerateTimeseries/
  AI_Train/                         Continuous-coordinate and retrieval methods
  AI_Result/                        Time-series model runs and evaluations
  AI_CONFIG_NOR_PREVIEW.md          Canonical preview/visual contract

AI_GenerateTrajectoryGrid/
  AI_Train/                         Grid-policy implementations
  AI_Result/                        Grid-policy runs and evaluations

Dataset/
  Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN/
                                    Canonical Image-based split source
  Data_Traj_Table/Topo_HouseGAN/    Continuous-coordinate trajectory dataset
  Data_TrajectoryGrid/Topo_HouseGAN/
                                    Discrete-grid trajectory dataset

UI_PerformanceCompare/Streamlit/    Comparison and research-framing UI
```

ก่อนทำงานให้ยืนยันว่า Git root คือ repository นี้ และอ่านเอกสารต่อไปนี้เมื่อ task เกี่ยวข้อง:

- `AI_GenerateTimeseries/AI_CONFIG_NOR_PREVIEW.md` สำหรับการสร้าง preview
- `AI_GenerateTimeseries/AI_Train/output baseline.md` สำหรับ run/evaluation output contract
- `AI_Technique.md` สำหรับบันทึกเทคนิค ปัญหา และวิธีใช้งาน pipeline ล่าสุด
- README/CONCEPT/DOCUMENT ภายใน directory ของ method ที่กำลังแก้

## 3. Canonical dataset contract

### Source of truth

ให้ใช้รายชื่อไฟล์ใน directory ต่อไปนี้เป็นฐานของ canonical HouseGAN split:

```text
Dataset/Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN/B
```

ข้อห้ามสำคัญ:

- ห้ามแก้ เปลี่ยนชื่อ ย้าย ลบ หรือสร้างไฟล์ทับใน Image-based dataset
- ห้ามจัด split ของ Image-based dataset ใหม่
- เมื่อต้องซ่อมหรือจัด split ของ `Data_Traj_Table` หรือ `Data_TrajectoryGrid` ให้จับคู่ชื่อ plan/scenario
  จาก Image-based `B` เท่านั้น
- หาก trajectory record ขาด ให้ค้นคืนจาก source เช่น `Geo_scenario` หรือขั้นตอนสร้างข้อมูลเดิม
  โดยต้องรักษา canonical split; ห้ามย้าย plan ข้าม split เพื่อทำให้จำนวนครบ
- การเปลี่ยนข้อมูลหรือ manifest ต้องสร้าง backup/audit trail และต้องได้รับคำสั่งที่ชัดเจนจากผู้ใช้ก่อน

### Verified canonical inventory

ตรวจจากไฟล์จริงเมื่อ 2026-08-06:

| Split | Image-based name | Trajectory name | Unique plans | Scenarios |
|---|---|---|---:|---:|
| Train | `train` | `train` | 412 | 2,603 |
| Validation | `validation` | `val` | 60 | 439 |
| Test | `test` | `test` | 117 | 862 |
| Total | — | — | 589 | 3,904 |

ก่อน final training หรือ final evaluation ต้องตรวจจำนวนและ plan membership จาก disk/manifest ใหม่เสมอ
หากไม่ตรงกับตารางนี้ ให้หยุดและรายงาน diff แก่ผู้ใช้ ห้ามแก้ dataset โดยอนุมานเอง

Canonical dataset identifier ที่ใช้อยู่:

```text
housegan_canonical_imagebase_split_v1
```

ห้ามผสม `Topo_bottleneck` กับ `Topo_HouseGAN` ในผลวิจัยเดียวกัน
checkpoint ที่ฝึกจาก topology/dataset อื่นใช้ได้เฉพาะ debugging หรือ framing และต้องระบุ
`research_valid: false` พร้อมเหตุผล

## 4. Method families and protected baselines

### Historical/original implementations

Directory ที่ไม่มี suffix `_SF_01` ให้ถือเป็น baseline/original implementation เช่น:

```text
AI_GenerateTimeseries/AI_Train/Method_Transformer
AI_GenerateTimeseries/AI_Train/Method_SGAN
AI_GenerateTimeseries/AI_Train/Method_GNN_CVAE2
AI_GenerateTimeseries/AI_Train/Method_GPT_Knowledge
AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy
```

กฎเริ่มต้น:

- ห้ามเปลี่ยน architecture, learning objective, model behavior หรือ dataset contract ของ baseline เดิม
  โดยไม่มีคำสั่งชัดเจนจากผู้ใช้
- หากผู้ใช้ขอเพิ่ม Social-Force, joint multi-agent prediction, constraint policy หรือ full-path behavior
  ให้ทำใน implementation ที่แยกชื่อ ไม่แทรกลง baseline เดิม
- Bug fix หรือ output adapter ที่จำเป็นกับ baseline เดิมต้องรายงานไฟล์และผลกระทบก่อนแก้
- อย่าเรียก legacy placeholder, copied checkpoint หรือ dataset-mismatch rollout ว่าผล final

### Active Social-Force-informed implementations

งานใหม่ที่ตั้งใจให้ train และประเมิน full-path แบบหลาย agent อยู่ใน:

```text
AI_GenerateTimeseries/AI_Train/Method_LSTM_SF_01
AI_GenerateTimeseries/AI_Train/Method_SGAN_SF_01
AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01
AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy_SF_01
```

แต่ละ implementation ต้อง:

- ทำนาย active agents พร้อมกันแบบ synchronous joint rollout
- ใช้ geometry, destination/navigation direction และ social context
- รองรับ full-path autoregressive rollout ไม่ใช่แค่ fixed short preview
- แยก direct model output (`Raw`) ออกจาก output หลัง safety executor (`Constrained`)
- เก็บ action/constraint trace เมื่อมีการแทรกแซง
- ห้ามอ้างว่าโมเดล “เรียนรู้ Social Force” หากเป็นเพียง hard-coded post-processing
  ต้องอธิบาย analytic prior, learned residual และ executor แยกกันอย่างตรงไปตรงมา

ชื่อผลลัพธ์มาตรฐาน:

```text
LSTM-SF-Raw
LSTM-SF-Constrained
SGAN-SF-Raw
SGAN-SF-Constrained
Transformer-SF-Raw
Transformer-SF-Constrained
GridPolicy-Raw
GridPolicy-Full
GNN-CVAE-Raw
GNN-CVAE-Constrained
GPT-Assisted Knowledge Retrieval and Geometric Transfer
```

## 5. Raw versus constrained evaluation

ผล Raw และ Constrained ตอบคนละคำถามและห้ามนำมาปนกัน:

- `Raw` คือ output ตรงจากโมเดล ใช้วัดว่าโมเดลเรียนรู้ spatial/social validity ได้เองมากน้อยเพียงใด
- `Constrained` คือ output หลัง walkability/collision/kinematic/exit executor
- ห้ามใช้ Constrained แทน Raw โดยไม่แสดงทั้งสองค่า
- ทุก Constrained result ต้องรายงาน Constraint Intervention Rate
- การ clamp/project trajectory กลับเข้า walkable area ถือเป็น intervention ไม่ใช่ความสามารถดิบของโมเดล

## 6. Training and pipeline rules

Python environment หลัก:

```text
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3
```

ทุก active method ต้องมี `run_pipeline.py` และรองรับอย่างน้อยการตรวจความพร้อม, train และ evaluate ตาม implementation
การใช้งานปกติให้รันเพียง:

```bash
python run_pipeline.py
```

จากนั้นเลือก operation จากเมนูพร้อมคำอธิบาย โดยไม่ต้องจำ `--stage`, config path หรือ checkpoint path
ค่าเริ่มต้นของเมนูต้องเป็นการตรวจ configuration และห้ามเริ่ม train โดยอัตโนมัติ
CLI flags เช่น `--stage` และ `--dry-run` ยังคงใช้สำหรับ automation และ advanced usage

กฎการ execute:

- การตรวจ syntax, unit test, contract test, `--stage plan` และ `--dry-run` ทำได้ตามปกติ
- smoke/fast training ต้องระบุว่าเป็น sanity check และ `research_valid: false`
- อย่าเริ่ม full 100-epoch/all-model training โดยไม่ได้รับคำสั่งชัดเจน เพราะใช้เวลาและ GPU สูง
- ห้ามเลือก checkpoint เพียงเพราะเป็นไฟล์ล่าสุด ต้องตรวจ method, dataset ID, split, seed และ config
- ห้ามใช้ `--allow-dataset-mismatch` ใน final evaluation
- Final training ต้องใช้ canonical train split; model selection ใช้ validation split; test splitใช้ครั้งสุดท้ายสำหรับรายงานผล
- ห้าม tune hyperparameters จาก test results
- Stochastic models ต้องบันทึก `K`, sampling policy และ seed และรายงาน mean@K/วิธี aggregate อย่างชัดเจน
- ใช้ seed และ protocol เดียวกันใน model comparison หรือรายงานความแตกต่างอย่างชัดเจน

หาก config มี absolute path ที่ผูกกับเครื่อง ให้เสนอเปลี่ยนเป็น repo-relative path หรือ environment-based path
เพื่อรองรับคอมเครื่องอื่น แต่ห้าม refactor นอกขอบเขต task โดยเงียบ ๆ

## 7. Output and provenance contract

ให้ยึด `AI_GenerateTimeseries/AI_Train/output baseline.md` เป็น specification หลัก
และสร้าง run ใหม่แทนการเขียนทับ run เดิม

หลักขั้นต่ำ:

- Run directory ใช้ชื่อที่ unique เช่น `run_<UTC timestamp>_seed<seed>`
- เก็บ resolved config, run manifest, dataset reference, source/code provenance, logs และ checkpoints
- Prediction ต้องถูก inverse-transform เป็น world coordinates หน่วยเมตรก่อนบันทึก
- Common prediction columns ขั้นต่ำคือ `case_id`, `split`, `frame`, `agent_id`, `pos_x`, `pos_y`, `is_active`
- Final evaluation ใช้ชื่อแบบ `eval_<dataset_id>_<split>_<protocol_version>`
- เปลี่ยน horizon, stride, observation length หรือ metric definition แล้วต้องสร้าง protocol version ใหม่
- ห้ามเขียนทับ evaluation เดิม

Final HouseGAN test evaluation ที่สมบูรณ์ต้องตรวจได้อย่างน้อยว่า:

```text
split = test
case_count = 862
floorplan_count = 117
dataset_id = housegan_canonical_imagebase_split_v1
research_valid = true
```

## 8. Research-validity gate and anti-fabrication rules

ห้ามแสดงหรือเขียนตัวเลข metric เป็นผลงานวิจัยจริง เว้นแต่มี artifact ที่ตรวจย้อนกลับได้

ตั้ง `research_valid: true` ได้ต่อเมื่อ:

- checkpoint ฝึกด้วย canonical training split
- evaluation ใช้ canonical test splitและไม่มี plan overlap
- มีครบ 862 test scenarios และ 117 test floor plans หรือตัว protocol ระบุ subset อย่างชัดเจนและไม่อ้างว่าเป็น full test
- checkpoint/dataset/protocol hashes หรือ references ตรวจสอบได้
- coordinate system และ frame interval ตรงกัน
- Raw output ไม่ผ่าน rule-based correction
- Constrained output มี trace ของ intervention
- stochastic sampling และ seed ถูกบันทึก
- metric code/config และ provenance ถูกเก็บ

สิ่งต่อไปนี้ต้องเป็น `research_valid: false`:

- smoke test หรือ fast sanity run
- framing preview subset
- placeholder output
- checkpoint/dataset mismatch
- topology mismatch
- incomplete test partition
- missing manifest/provenance ที่ทำให้ย้อนตรวจไม่ได้

ห้ามสร้าง fake metric, hard-code ตัวเลขเพื่อให้ UI มีข้อมูล, คัดลอกผลจาก method อื่น หรือเปลี่ยน label ของผลเก่า
เพื่อให้ดูเหมือนเป็นผลของโมเดลใหม่ หากไม่มีผลจริง UI ต้องแสดง `Missing`, `Not evaluated` หรือสถานะที่ตรงกับความจริง

## 9. Evaluation dimensions

Metric ขั้นต่ำที่ควรพิจารณา โดยนิยามและหน่วยต้องถูก lock ก่อน final run:

- Trajectory accuracy: ADE, FDE, path-length error
- Physical validity: out-of-bounds rate, wall-crossing rate, invalid-step rate
- Navigation: goal/exit reach rate, evacuation-time error
- Social validity: collision-exposure rate และ metric ระยะห่างที่ protocol กำหนด
- Aggregate behavior: exit-flow error, density-map error
- Constraint dependence: constraint-intervention rate
- Efficiency: latency per agent-step และ real-time factor

อย่ารวม metric ที่นิยามต่างกันไว้ในคอลัมน์เดียว และอย่าเฉลี่ย normalized coordinate error ข้ามผัง
แล้วรายงานเป็นเมตรโดยไม่มี inverse transformation

## 10. Preview and UI contract

รายละเอียดภาพทั้งหมดให้ยึด `AI_GenerateTimeseries/AI_CONFIG_NOR_PREVIEW.md`

ข้อกำหนดสำคัญ:

- Canvas รอบนอกและพื้นที่ title เป็นสีขาว
- พื้น non-walkable ภายใน floorplan เป็น `#101820`
- ห้องและ corridor เป็น `#f3f6f8`
- กำหนด bounds จาก geometry ทั้งผังรวม exit ห้าม crop ตาม predicted trajectory
- รักษา aspect ratio เท่ากัน
- ประตูต้องเจาะผนังเป็น void จาก `Geo_door.json` ไม่วาดกล่องประตูสีเหลือง
- Exit room ใช้ overlay และ legend ตาม visual contract
- AI method preview แสดงเฉพาะ trajectory ของ method นั้น ห้ามวาด Ground Truth line ซ้อน
- Title ต้องเป็นชื่อ method เช่น `Transformer`, `Social GAN`, `GNN-CVAE`, `GridSocialPolicy`
- ห้ามใช้ `DEBUG ONLY` หรือ `Model rollout sample` เป็นชื่อภาพสำหรับ gallery
- Warning/provenance แสดงใน manifest, report หรือ terminal ไม่ควรปลอมเป็นชื่อ method
- UI เป็น consumer ของ artifact ไม่ใช่แหล่งสร้าง metric
- UI ต้องไม่ใช้ placeholder เป็น final result และต้องไม่ซ่อน `research_valid: false`

## 11. Safe editing and Git discipline

ก่อนแก้ไฟล์ทุกครั้ง:

1. ตรวจ `git status --short`
2. อ่านไฟล์และ config ที่เกี่ยวข้อง
3. แยก user changes ออกจากงานที่จะทำ
4. ระบุว่าจะกระทบ original baseline หรือ `_SF_01`

กฎ:

- รักษา user changes และหลีกเลี่ยงไฟล์ที่ไม่เกี่ยวข้อง
- ห้ามใช้ `git reset --hard`, destructive checkout หรือ recursive deletion
- ห้าม revert/restore ไฟล์ของผู้ใช้โดยไม่ได้รับอนุญาต
- ห้าม commit, amend, push หรือ force-push เว้นแต่ผู้ใช้สั่ง
- ก่อนลบ ย้าย หรือแทนที่ dataset/checkpoint/run ให้ตรวจ exact resolved path และขออนุญาต
- เพิ่ม implementation ใหม่ด้วย directory ใหม่เมื่อผู้ใช้ต้องการรักษาของเดิม
- หลังแก้ ให้รายงานรายชื่อไฟล์ ผลกระทบ และวิธีทดสอบอย่างตรงไปตรงมา

หากคำสั่งกำกวมระหว่างแก้ baseline เดิมกับ `_SF_01` และการเลือกผิดอาจเปลี่ยนผลวิจัย ให้หยุดถามผู้ใช้ก่อน

## 12. Verification expectations

ตรวจงานตามความเสี่ยง:

- Documentation-only: ตรวจ links/path, spelling ของ method IDs และ `git diff --check`
- Config change: parse JSON และ dry-run pipeline
- Python change: compile/import เฉพาะไฟล์ที่แก้และ run focused tests
- Dataset change: เปรียบเทียบ scenario names, unique plans, split membership และ manifest ก่อน/หลัง
- Renderer/UI change: สร้าง preview ตัวอย่างและตรวจภาพจริง รวมทั้งกรณีผังแนวตั้ง/แนวนอน
- Training/evaluation change: smoke run ก่อน full run และตรวจ manifests/prediction schema

อย่าสรุปว่า “พร้อมทำ paper” เพียงเพราะ code รันได้ ต้องแยกสถานะอย่างน้อยเป็น:

```text
implemented
smoke-tested
trainable
fully trained
evaluated on canonical test
research-valid
```

## 13. Documentation and cross-machine handoff

หลังการเปลี่ยนแปลงที่มีผลต่อ research design, dataset, training protocol, output schema หรือ validity:

- อัปเดตเอกสาร canonical ที่เกี่ยวข้องใน commit เดียวกัน
- บันทึก method ID, dataset ID, split, seed, checkpoint/run ID และสิ่งที่ยังไม่เสร็จ
- อย่าพึ่ง chat memory เป็นแหล่งความจริงเพียงแห่งเดียว
- เมื่อย้ายเครื่อง ให้ clone repository, เปิดที่ Git root และให้ Codex อ่าน `AGENTS.md` ก่อนเริ่มงาน

ข้อความเริ่มต้นที่แนะนำบนเครื่องใหม่:

```text
Read AGENTS.md, AI_GenerateTimeseries/AI_CONFIG_NOR_PREVIEW.md, and
AI_GenerateTimeseries/AI_Train/output baseline.md. Inspect the current Git status,
dataset manifests, and active method configs. Then summarize verified project status,
research-valid results, blockers, and the next safe action before editing anything.
```

## 14. Conflict resolution

หาก code, config, manifest, README และภาพใน UI ให้ข้อมูลไม่ตรงกัน:

1. อย่าเลือกข้อมูลที่ดูดีที่สุด
2. ตรวจ artifact และ dataset บน disk
3. ให้ manifest/provenance ที่ตรวจย้อนกลับได้มีน้ำหนักมากกว่า filename หรือ UI label
4. รายงาน contradiction พร้อม path และหลักฐาน
5. ขอคำยืนยันก่อนแก้สิ่งที่อาจเปลี่ยนผลวิจัยหรือทำลาย baseline

เป้าหมายสูงสุดคือ reproducibility และความซื่อสัตย์ของผลวิจัย ไม่ใช่เพียงทำให้ UI แสดงครบทุกช่อง

# Document_Research Directory Structure

เอกสารนี้นิยามโครงสร้างเชิงลึกของ `Document_Research` ว่าแต่ละ directory แยกตามหน้าที่อะไร และไฟล์ Markdown แต่ละไฟล์ใช้สรุป/อธิบายประเด็นใดในงานวิจัย AI surrogate model สำหรับ pedestrian simulation

Last reviewed: 2026-06-27

## ภาพรวมบทบาทของโฟลเดอร์

`Document_Research` เป็นพื้นที่รวบรวมเอกสารประกอบงานวิจัย ไม่ใช่พื้นที่เก็บ code training โดยตรง โครงสร้างปัจจุบันแบ่งตามชนิดของเอกสารดังนี้:

```text
Document_Research/
├── Output_FrameworkResearch/
│   ├── Imagebase_Output/
│   ├── Summarybase_Output/
│   └── Synthetic_Dataset/
└── Reference_data/
```

แนวคิดการแยก directory คือ:

- `Reference_data/`: เก็บเอกสารต้นทางหรือ reference paper/draft ที่ใช้เป็นฐานอ้างอิงในการเขียนและจัดโครง paper
- `Output_FrameworkResearch/`: เก็บผลผลิตจากการวิเคราะห์ code, dataset, model workflow และ research framework
- `Summarybase_Output/`: สรุปงานหลักฝั่ง scalar prediction หรือการทำนายค่า travel time ด้วย MLP/GNN
- `Synthetic_Dataset/`: สรุปเชิงลึกเกี่ยวกับการสร้าง synthetic layout dataset และเงื่อนไขของ HouseGAN/procedural layout generation
- `Imagebase_Output/`: พื้นที่เตรียมไว้สำหรับผลวิเคราะห์สาย image-based surrogate เช่น density heatmap generation ด้วย Pix2PixHD/CVAE ปัจจุบันยังไม่มีไฟล์ Markdown ภายใน

## 1. Reference_data

Path:

```text
Document_Research/Reference_data/
```

หน้าที่:

- เก็บ paper, draft, หรือ reference documents ที่ใช้เป็นฐานเทียบเคียงในการเขียนงานวิจัย
- เป็น input เชิงเอกสารสำหรับสรุป methodology, abstract, references และ paper structure
- ไม่ใช่ผลวิเคราะห์ขั้นสุดท้าย แต่เป็นแหล่งข้อมูลตั้งต้น

ไฟล์สำคัญ:

```text
P006_Hong_ASA2025.pdf
Time travel prediction_Conference - ASA xjtlu.docx
Density Heatmap Generation with Pix2PixHD and CVAE .docx
```

คำอธิบาย:

- `P006_Hong_ASA2025.pdf`: paper/reference format ที่ใช้เป็นตัวอย่างโครงสร้าง paper เช่น abstract, methodology, results, discussion และ conclusion
- `Time travel prediction_Conference - ASA xjtlu.docx`: draft/reference ของสายงาน travel time prediction ซึ่งสัมพันธ์กับโมเดล MLP/GNN
- `Density Heatmap Generation with Pix2PixHD and CVAE .docx`: draft/reference ของสาย image-based surrogate model ที่เปลี่ยน output จากตัวเลข travel time เป็น density heatmap โดยใช้ Pix2PixHD และ CVAE

หมายเหตุ:

- พบไฟล์ `P006_Hong_ASA2025.pdfZone.Identifier` ซึ่งเป็น metadata จาก Windows/WSL download stream ไม่ใช่เอกสารวิจัยจริง ควร ignore หรือลบออกก่อน commit

## 2. Output_FrameworkResearch

Path:

```text
Document_Research/Output_FrameworkResearch/
```

หน้าที่:

- เก็บเอกสารสังเคราะห์จากการอ่าน code, dataset, output และ reference paper
- เป็นพื้นที่กลางสำหรับแปลง codebase ให้กลายเป็น research narrative
- แยกย่อยตามแกนวิเคราะห์ ได้แก่ summary-based, synthetic dataset, และ image-based output

โครงสร้าง:

```text
Output_FrameworkResearch/
├── Imagebase_Output/
├── Summarybase_Output/
└── Synthetic_Dataset/
```

## 3. Summarybase_Output

Path:

```text
Document_Research/Output_FrameworkResearch/Summarybase_Output/
```

หน้าที่:

- สรุปภาพรวมงานวิจัยฝั่ง travel time prediction
- เชื่อม codebase กับโครงสร้าง paper
- วิเคราะห์โมเดล MLP และ GNN ว่าทำงานอย่างไร จุดแข็ง/จุดอ่อนคืออะไร และควรอธิบายใน paper อย่างไร

ไฟล์ภายใน:

```text
agent_summary.md
agent_summary_code_MLP.md
agent_summary_code_GNN.md
framework_mermaid.md
agent_summary.docx
```

### agent_summary.md

บทบาท:

- เป็นสรุปหลักของงานวิจัย `AI Surrogate Model for Pedestrian Travel Time Prediction`
- ให้คะแนนความสอดคล้องระหว่าง code กับ narrative งานวิจัย (`Coding Matching Score`)
- วาง checklist สำหรับเขียน paper โดยอ้างอิงโครงสร้างมาตรฐานจาก reference paper
- แยกสิ่งที่พร้อมใส่ paper แล้วและสิ่งที่ยังขาด
- มี draft ส่วน Discussion, Conclusion, Limitations/Future Research และ Experiment Scope

สาระสำคัญ:

- งานใช้ synthetic dataset จาก JuPedSim เพื่อ train AI surrogate model
- เปรียบเทียบ MLP กับ GNN สำหรับทำนาย pedestrian travel time
- สรุปว่า MLP ทำผลงานได้ดีกว่า GNN ในผลปัจจุบัน แต่ GNN มีศักยภาพด้าน spatial/generalization หากปรับ graph representation เพิ่ม
- ชี้ประเด็นที่ยังต้องเติม เช่น architecture details, hyperparameters, parity plot, computational efficiency และ graph construction explanation

### agent_summary_code_MLP.md

บทบาท:

- รีวิว code ของ `Method_MLP_PyTorch`
- อธิบาย feature engineering และ pipeline ของ MLP
- ใช้เป็นฐานสำหรับเขียน Methodology ฝั่ง feature-based regression

สาระสำคัญ:

- MLP ใช้ derived features เช่น `detour_ratio`, `agent_density_near_path`, `area_per_agent`, `door_pressure_per_agent`
- มี target transformation ด้วย `log1p` เพื่อลดผลของ right-skewed travel time
- มี feature/target standardization แบบ Z-score
- โครงสร้างโมเดลเป็น feed-forward MLP หลายชั้น เช่น `[128, 64, 32]`
- ใช้ `LayerNorm`, `ReLU`, และ `Dropout`
- จุดแข็งคือ feature engineering ตรงกับโจทย์ regression ทำให้เรียนรู้เร็วและแม่น
- ข้อจำกัดคือ MLP มองข้อมูลเป็น tabular features จึงไม่เข้าใจ topology ของแปลนโดยตรง

### agent_summary_code_GNN.md

บทบาท:

- รีวิว code ของ `Method_GNN`
- อธิบาย graph representation, node features, adjacency normalization และ GNN workflow
- ใช้เป็นฐานสำหรับเขียน Methodology ฝั่ง graph/spatial representation

สาระสำคัญ:

- ใช้ `topological_graph.json` สร้าง adjacency matrix
- ใช้ symmetric normalization แบบ `D^-0.5 A D^-0.5`
- node features รวมทั้ง static features เช่น room area/type และ dynamic route features เช่น start/end, agents, distance, bottleneck factors
- โมเดลเป็น GCN layer หลายชั้นตาม config
- ใช้ global mean pooling รวม node embeddings เป็น graph-level vector
- จุดแข็งคือมีโครงสร้างที่รับรู้ spatial connectivity ได้
- จุดอ่อนสำคัญคือ global mean pooling อาจทำให้ข้อมูล path-specific structure หาย จึงสู้ MLP ไม่ได้ในผลปัจจุบัน
- เสนอแนวทางปรับปรุง เช่น GAT หรือ path-aware pooling

### framework_mermaid.md

บทบาท:

- ให้ Mermaid diagram ของ framework งานวิจัย
- อธิบาย pipeline ตั้งแต่ input configuration, procedural layout generation, graph construction, ไปจนถึง output data structure

สาระสำคัญ:

- เริ่มจาก `config_housegan.json`
- กำหนด complexity, room area mode, number of scenarios และ seed
- generate layout ด้วยการวาง corridor/room และคำนวณ geometry ด้วย Shapely polygons
- detect physical doors และ extract topological graph
- output เป็น directory ใน `Geo_scenario/` พร้อม `topological_graph.json`, `geo_*.json`, `metadata.json` และ preview files

### agent_summary.docx

บทบาท:

- เป็น DOCX version ของ `agent_summary.md`
- ใช้สำหรับนำไปจัดรูปแบบ/ส่งต่อในบริบท paper หรือรายงานที่ต้องการ Word document

## 4. Synthetic_Dataset

Path:

```text
Document_Research/Output_FrameworkResearch/Synthetic_Dataset/
```

หน้าที่:

- สรุปกระบวนการสร้าง synthetic floor plan dataset
- วิเคราะห์ความสัมพันธ์ระหว่าง target complexity ที่สั่ง generate กับ actual complexity ที่เกิดขึ้นจริง
- ใช้เป็นฐานเขียน Methodology/Data Generation ของ paper

ไฟล์ภายใน:

```text
agent_summary.md
Generateplan_conditional.md
Framework_GeneratePlan.md
```

### agent_summary.md

บทบาท:

- สรุปภาพรวม synthetic dataset
- ให้คะแนนความเหมาะสมของ dataset
- อธิบายว่า dataset มีความหลากหลายและเหมาะกับการ train MLP/GNN หรือไม่

สาระสำคัญ:

- วิเคราะห์ code `generate_layout.py`
- ตรวจเงื่อนไข complexity และ room area mode
- สกัด metadata จาก `Geo_scenario`
- สรุปว่ามี 610 layouts ที่นำมาวิเคราะห์
- ให้คะแนนความเหมาะสม `9/10`
- เหตุผลที่ยังไม่เต็มคือ XL และ XXL ยังขาด room area mode แบบ `Big`

### Generateplan_conditional.md

บทบาท:

- วิเคราะห์เชิงลึกว่าการสร้าง layout เกิดจากเงื่อนไขอะไร
- แยก `Target Complexity` ออกจาก `Actual Complexity`
- อธิบายว่าทำไมมี layout ขนาดเล็กเกิดขึ้นจริง ทั้งที่ไม่ได้สั่ง generate เป็น Small

สาระสำคัญ:

- target parameters มี Medium, Large, XL, XXL และ room area mode แบบ Default/Big
- ไม่มีการสั่ง target Small โดยตรง
- actual dataset กลับพบ Small layouts จำนวนมาก เพราะ algorithm วางห้องแบบสุ่มและอาจวางห้องไม่ครบ target
- สาเหตุเชิง code คือการลองวางห้องแบบมี collision detection และจำกัดจำนวน trial หากวางไม่ได้จะ skip
- สรุป cross-tabulation ระหว่าง target complexity กับจำนวนห้องจริง
- เป็นเอกสารสำคัญสำหรับอธิบาย dataset bias และ generation limitation

### Framework_GeneratePlan.md

บทบาท:

- ให้ Mermaid workflow สำหรับ layout generation ด้วย HouseGAN/procedural generator
- ใช้เป็นภาพประกอบ Methodology/Data Generation

สาระสำคัญ:

- input คือ `config_housegan.json`
- pipeline ประกอบด้วย configuration, procedural layout generation, graph construction และ output data structure
- output หลักคือ `Geo_scenario/<Plan Directory>` พร้อมข้อมูล geometry, graph และ metadata

## 5. Imagebase_Output

Path:

```text
Document_Research/Output_FrameworkResearch/Imagebase_Output/
```

หน้าที่ที่ควรใช้:

- เก็บผลวิเคราะห์ของสาย image-based surrogate model
- เหมาะสำหรับสรุปงาน density heatmap generation จาก Pix2PixHD/CVAE
- ควรเป็นพื้นที่คู่ขนานกับ `Summarybase_Output` แต่เปลี่ยน output จาก scalar travel time เป็น image/density map

สถานะปัจจุบัน:

- directory มีอยู่แล้ว แต่ยังไม่มีไฟล์ Markdown ภายใน

ข้อเสนอการจัดไฟล์ในอนาคต:

```text
Imagebase_Output/
├── agent_summary_imagebase.md
├── agent_summary_code_pix2pixhd.md
├── agent_summary_code_cvae.md
└── framework_heatmap_generation.md
```

คำอธิบายไฟล์ที่ควรมี:

- `agent_summary_imagebase.md`: สรุปภาพรวมงาน density heatmap generation และความเชื่อมโยงกับ AI surrogate model
- `agent_summary_code_pix2pixhd.md`: รีวิว code/model Pix2PixHD เช่น input channels, target density map, loss function, resolution preservation
- `agent_summary_code_cvae.md`: รีวิว code/model CVAE เช่น latent representation, reconstruction behavior, limitation ด้าน spatial sharpness
- `framework_heatmap_generation.md`: Mermaid diagram ของ pipeline จาก floor plan/start-exit/agent class ไปสู่ predicted density heatmap

## ความสัมพันธ์ของเอกสารกับ paper narrative

โครงสร้าง `Document_Research` ตอนนี้สามารถตีความเป็น research narrative ได้สองสายหลัก:

### สายที่ 1: Summary-based / Scalar Prediction

โจทย์:

```text
architectural layout + route/agent features -> predicted pedestrian travel time
```

Directory ที่เกี่ยวข้อง:

```text
Output_FrameworkResearch/Summarybase_Output/
```

โมเดล:

- MLP
- GNN

Output:

- predicted travel time
- error metrics เช่น MAE, RMSE, MSE
- parity plots หรือ comparison plots

### สายที่ 2: Image-based / Heatmap Generation

โจทย์:

```text
floor plan + origin/destination + occupant class -> predicted pedestrian density heatmap
```

Directory ที่เกี่ยวข้อง:

```text
Reference_data/Density Heatmap Generation with Pix2PixHD and CVAE .docx
Output_FrameworkResearch/Imagebase_Output/  (ควรเติม Markdown ต่อ)
```

โมเดล:

- Pix2PixHD
- Conditional Variational Autoencoder (CVAE)

Output:

- generated density heatmap
- pixel-level MAE/MSE
- image quality/spatial similarity metrics ที่ควรเพิ่มในอนาคต เช่น SSIM หรือ LPIPS

## ประเด็นที่ควรทำให้สอดคล้องก่อนใช้เขียน paper

1. จำนวน layout ควรใช้ให้ตรงกันทั้งชุดเอกสาร: บางไฟล์เขียน `600 layouts` แต่ analysis ใน `Synthetic_Dataset` พบ `610 layouts`
2. ควรแยก `Target Complexity` และ `Actual Complexity` ให้ชัดใน Methodology เพราะ dataset จริงไม่ได้ตรงกับ parameter ที่สั่ง generate เสมอ
3. ควรย้าย/สรุปงาน density heatmap จาก `Reference_data` ไปเป็น Markdown ใน `Imagebase_Output` เพื่อให้โครงสร้างครบเหมือนสาย travel time
4. ควรเติม metric จริงของแต่ละ model ก่อนเขียน claim เช่น `MLP ดีกว่า GNN` หรือ `Pix2PixHD ดีกว่า CVAE`
5. ควรลบหรือ ignore ไฟล์ `Zone.Identifier` เพราะไม่ใช่ข้อมูลวิจัย

## สรุปการใช้งาน

หากต้องการเขียน paper หรือรายงาน ให้เริ่มอ่านตามลำดับนี้:

1. `Output_FrameworkResearch/Synthetic_Dataset/Generateplan_conditional.md` เพื่อเข้าใจ dataset และ generation bias
2. `Output_FrameworkResearch/Synthetic_Dataset/agent_summary.md` เพื่อเห็นภาพรวม synthetic dataset
3. `Output_FrameworkResearch/Summarybase_Output/agent_summary.md` เพื่อเข้าใจ research narrative หลักของ travel time prediction
4. `Output_FrameworkResearch/Summarybase_Output/agent_summary_code_MLP.md` เพื่อเขียน Methodology ของ MLP
5. `Output_FrameworkResearch/Summarybase_Output/agent_summary_code_GNN.md` เพื่อเขียน Methodology ของ GNN
6. `Output_FrameworkResearch/Summarybase_Output/framework_mermaid.md` และ `Synthetic_Dataset/Framework_GeneratePlan.md` เพื่อใช้สร้างภาพ workflow
7. `Reference_data/Density Heatmap Generation with Pix2PixHD and CVAE .docx` เพื่อขยาย narrative ไปยัง image-based surrogate model


# สรุปและรีวิว Code: Continuous Coordinate Trajectory Prediction Models

## 1. ภาพรวมสถาปัตยกรรม (Architecture Overview)

โฟลเดอร์ที่เกี่ยวข้อง:
- [AI_GenerateTimeseries/AI_Train/Method_Transformer](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_Transformer)
- [AI_GenerateTimeseries/AI_Train/Method_GNN_CVAE](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_GNN_CVAE)
- [AI_GenerateTimeseries/AI_Train/Method_SGAN](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_SGAN)
- [AI_GenerateTimeseries/AI_Train/Method_LSTM_01](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_LSTM_01)

กลุ่มโมเดลเหล่านี้พยายามแก้โจทย์การทำนายเส้นทางสัญจรด้วยการมองพิกัดเป็นตัวเลขทศนิยมแบบต่อเนื่อง:
$$\mathbf{y}_t = (x_t, y_t) \in [0, 1]^2 \quad \text{หรือ} \quad (x_t, y_t) \in \mathbb{R}^2$$

---

## 2. การทำงานเชิงลึกของแต่ละโมเดล (Model Details)

### 2.1 Transformer (Goal-Conditioned GPT-2)
- **ไฟล์หลัก**: `model.py`, `dataset.py`, `train_transformer.py`, `test_transformer.py`
- **โครงสร้างโมเดล**:
  - `GeoEncoder`: ใช้ CNN 2D สกัดภาพผังอาคาร (`geo_mask`) ออกมาเป็น Vector Representation
  - `Neighbor Embedder`: สกัดตำแหน่งและเส้นทางของเพื่อนบ้านรอบตัว
  - `Causal GPT-2 Backbone`: ป้อน Sequence ตำแหน่งในอดีต (`obs_traj`), จุดเริ่มต้น (`start_pt`), จุดหมาย (`end_pt`), และ Geo Feature เพื่อทำนายพิกัดถัดไปแบบ Autoregressive
- **การเทรน**: ใช้ Teacher Forcing และ MSE Loss ระหว่างพิกัดที่ทำนายกับ Ground Truth

### 2.2 GNN + CVAE (Graph Neural Network + Conditional VAE)
- **ไฟล์หลัก**: `model.py`, `dataset.py`, `train_gnn_cvae.py`, `test_gnn_cvae.py`
- **โครงสร้างโมเดล**:
  - สร้าง Graph Node เป็นตัวแทนของคนเดินเท้าแต่ละคน และ Graph Edge เชื่อมต่อคนในรัศมีสังคม
  - ใช้ GCN/GAT Layer ในการสร้าง Social Context Embedding
  - ใช้ CVAE Latent Variable $z \sim \mathcal{N}(\mu, \sigma^2)$ เพื่อสุ่มพฤติกรรมการเดินแบบ Multi-modal

### 2.3 SGAN (Social GAN) & CVAE
- **สถาปัตยกรรม**: ใช้ Generator ทำนายเส้นทางในอนาคตร่วมกับ Discriminator ตรวจสอบความเป็นไปได้ของเส้นทาง หรือใช้ Encoder-Decoder บีบสเปซ Latent

---

## 3. การวิเคราะห์สาเหตุความล้มเหลว (Failure Analysis & Feature Ignorance)

จากการทดลองในโปรเจกต์ พบว่ากลุ่ม Continuous Coordinate Models มีข้อจำกัดร้ายแรง 3 ประการในการนำไปใช้แทน Simulation:

1. **ภาวะ Feature Ignorance (ไม่เข้าใจ Feature ผังอาคาร)**:
   - แม้จะมีการป้อนภาพแปลน (`geo_mask`) ผ่าน CNN GeoEncoder เข้าไปในโมเดล แต่เนื่องจาก Loss Function เป็นเพียง Euclidean Distance (MSE):
     $$\mathcal{L}_{\text{MSE}} = \frac{1}{T}\sum_{t=1}^T ||\hat{\mathbf{y}}_t - \mathbf{y}_t||^2$$
   - ตัวโมเดลเรียนรู้เพียงการ "เฉลี่ยและสไลด์เส้นทางแบบโค้งมน (Smooth Interpolation)" ระหว่างจุด A ไปจุด B แต่ไม่ได้เรียนรู้ขอบเขตทางกายภาพของกำแพง ทำให้เกิดปรากฏการณ์เดินทะลุผนัง (Wall Clipping)

2. **การขาด Spatial Inductive Bias**:
   - พิกัด $(x,y)$ ที่เป็นทศนิยมต่อเนื่องไม่มีโครงสร้างที่บีบให้ AI รับรู้ว่า "บริเวณนี้ก้าวเดินได้" หรือ "บริเวณนี้เป็นสิ่งกีดขวาง" 

3. **ความล้มเหลวในแปลนที่มีความซับซ้อน**:
   - เมื่อเจอแปลนคอขวด (Bottleneck) หรือทางเลี้ยวซิกแซก โมเดลพิกัดทศนิยมมักจะล้มเหลวในการทำนายทิศทาง และเดินพุ่งเข้าหากำแพงตรงๆ เนื่องจาก Feature ที่ป้อนเข้าไปไม่สามารถสื่อสารเงื่อนไขเชิงพื้นที่ให้แก่อัลกอริทึมได้จริง

---

## 4. สรุปบทบาทใน Paper

ในงานวิจัย/เปเปอร์ กลุ่ม Continuous Coordinate Models จะถูกจัดบทบาทเป็น **"กลุ่มทดลองเปรียบเทียบเพื่อชี้ให้เห็นข้อจำกัด" (Comparative Baseline & Failure Case Analysis)** โดยแสดงผลว่าการปรับจูนโมเดลพิกัดทศนิยมให้ซับซ้อนขึ้น (เช่น การใส่ GPT-2 หรือ GNN) ไม่สามารถแก้ปัญหาพื้นฐานเรื่อง Spatial Awareness ได้ หากไม่เปลี่ยนรูปแบบ Representation เชิงพื้นที่

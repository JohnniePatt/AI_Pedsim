# Method_CVAE Develop Log

ไฟล์นี้บันทึกเฉพาะช่วงที่ผู้ใช้เข้ามาปรึกษาเรื่อง **ผลลัพธ์ CVAE ยังไม่ดี** และสิ่งที่เราเลือกวิเคราะห์/ปรับแก้ต่อเนื่องกัน

## develop_cvae_no_0001: ผลลัพธ์ CVAE BW version แรกแย่มาก

### สถานการณ์

หลังจากพัฒนา `Method_CVAE` สำหรับ density-map BW และทดสอบผลลัพธ์ ผู้ใช้แจ้งว่า:

```text
ผลลัพธ์ออกมาแล้ว มันแย่มาก
```

ภาพ prediction ที่ได้ยังไม่สามารถสร้าง density map ที่ดีได้ ลักษณะโดยรวมคือ output จาง เบลอ หรือไม่สามารถจับ spatial density structure ได้ดีเท่าที่ควร

### สิ่งที่วิเคราะห์

จากผลลัพธ์ version แรก คาดว่าปัญหาเกิดได้จากหลายจุด:

```text
1. density map มี background จำนวนมาก ทำให้ loss ถูก background ครอบ
2. pixel-wise L1/MSE อาจทำให้โมเดลเลือกทำนายค่าเฉลี่ย
3. CVAE มี latent variable ทำให้ deterministic prediction อาจเบลอ
4. model capacity เดิมอาจยังไม่พอ
5. BW target ถึงจะดีกว่า ColorJet แต่ยังต้องใช้ loss ที่เหมาะกับ density map
```

### ข้อสรุปของรอบนี้

ยังไม่ควรตัดสินว่า CVAE ใช้ไม่ได้ทันที แต่ควรลองแก้สองเรื่องก่อน:

```text
1. เพิ่ม capacity ของโมเดล
2. ปรับ loss ให้ foreground density สำคัญขึ้น
```

แนวคิดคือ ถ้า output จางเพราะ background เยอะเกินไป เราต้องทำให้บริเวณที่มี density จริงมีน้ำหนักมากขึ้น

## develop_cvae_no_0002: ทดลองแก้ด้วย BW v2 config

### สถานการณ์

หลังจากรอบแรก เราตัดสินใจลองแก้ด้วยการปรับ config ก่อน โดยยังไม่เปลี่ยน architecture หลักของ CVAE

เพิ่ม config:

```text
config_train_densitymap_bw_v2.json
```

### สิ่งที่ปรับใน v2

ค่าหลักของ v2:

```json
"base_filters": 64,
"latent_dim": 64,
"batch_size": 4,
"dropout": 0.0,
"kl_weight": 0.0,
"mse_loss_weight": 1.0,
"edge_loss_weight": 0.25,
"density_foreground_weight": 80.0,
"density_intensity_weight": 40.0
```

### เหตุผลของการปรับ

```text
base_filters = 64
เพิ่มขนาด encoder/decoder เพื่อให้โมเดลมี capacity มากขึ้น

latent_dim = 64
เพิ่ม latent space เผื่อให้รองรับ variation ของ density map ได้ดีขึ้น

dropout = 0.0
ลดการสุ่ม regularization ที่อาจทำให้ output จาง

kl_weight = 0.0
ปิด KL loss ชั่วคราว เพราะ KL อาจกด latent distribution จน prediction เบลอ

density_foreground_weight / density_intensity_weight สูงขึ้น
เพิ่มน้ำหนักบริเวณที่มี density จริง เพื่อแก้ปัญหา background ครอบ loss
```

### ผลหลังลอง v2

ผู้ใช้แจ้งว่าผลลัพธ์ยังไม่ดี:

```text
ผลลัพธ์จากการปรับ config ยังไม่ดีครับเหมือนเดิมเลย
```

### ข้อสรุปของรอบนี้

การเพิ่ม capacity และเพิ่ม foreground weight อย่างเดียวไม่พอ

ดังนั้นปัญหาน่าจะลึกกว่า config tuning ปกติ และควรกลับไปดูเงื่อนไขการ train/test ของ CVAE เอง

## develop_cvae_no_0003: แก้ train/inference mismatch และเพิ่ม BW v3

### สถานการณ์

หลังจาก v2 ยังให้ผลคล้ายเดิม เราวิเคราะห์ต่อว่า CVAE อาจมีปัญหา train/inference mismatch:

```text
ตอน train:
input A + target B -> posterior encoder -> latent z
decoder ได้ z ที่มีข้อมูลจาก target จริง

ตอน test:
ไม่มี target B
decoder ใช้ z = 0
```

ถ้า decoder ระหว่าง train เรียนรู้ที่จะพึ่ง `z` ที่มาจาก target จริงมากเกินไป ตอน test ที่ใช้ `z=0` อาจทำให้ prediction collapse เป็นภาพจางหรือภาพเฉลี่ย

### สมมติฐาน

ปัญหาไม่ได้อยู่ที่ loss weight อย่างเดียว แต่อยู่ที่เงื่อนไขการเรียนรู้:

```text
training condition ไม่เหมือน inference condition
```

ดังนั้น deterministic CVAE output ควรทดลอง train ด้วย condition เดียวกับ inference:

```text
train: z = 0
test:  z = 0
```

เพื่อบังคับให้ decoder เรียนรู้จาก input scenario A เป็นหลัก

### สิ่งที่แก้ใน code

#### cvae_model.py

เพิ่ม argument:

```python
forward_train(image_a, target_b, latent_mode="posterior")
```

รองรับ:

```text
posterior -> ใช้ z จาก posterior encoder ตาม CVAE ปกติ
zero      -> ใช้ z = 0 เหมือน deterministic inference
random    -> sample z จาก prior
```

#### cvae_density_train.py

เพิ่มให้ training loop อ่าน config:

```json
"train_latent_mode": "zero"
```

แล้วส่งเข้า `model.forward_train(...)`

#### cvae_losses.py

เพิ่ม loss ใหม่สำหรับแก้ output จาง:

```text
foreground_l1_loss
mass_loss
gamma_l1_loss
```

ความหมาย:

```text
foreground_l1_loss
วัด error เฉพาะบริเวณที่ target มี density จริง ลดปัญหา background กลบ loss

mass_loss
บังคับค่าเฉลี่ยหรือมวลรวม density ของ prediction ไม่ให้ต่ำกว่า target มากเกินไป

gamma_l1_loss
เปรียบเทียบใน gamma-space เพื่อเน้นค่า density อ่อน ๆ
```

#### cvae_config.py

เพิ่ม default config field:

```json
"foreground_l1_loss_weight": 0.0,
"mass_loss_weight": 0.0,
"gamma_l1_loss_weight": 0.0,
"density_gamma_loss": 1.0,
"train_latent_mode": "posterior"
```

ค่า default ยังทำให้ config เดิมทำงานเหมือนเดิม และเปิดใช้เฉพาะ config ใหม่

### Config ใหม่

เพิ่มไฟล์:

```text
config_train_densitymap_bw_v3.json
```

ค่าหลัก:

```json
"epochs": 80,
"batch_size": 4,
"base_filters": 64,
"latent_dim": 64,
"dropout": 0.0,
"train_latent_mode": "zero",
"kl_weight": 0.0,
"foreground_l1_loss_weight": 8.0,
"mass_loss_weight": 20.0,
"gamma_l1_loss_weight": 1.5,
"density_gamma_loss": 0.5,
"density_foreground_weight": 60.0,
"density_intensity_weight": 30.0
```

### เหตุผลของ v3

v3 เปลี่ยนจากการจูน config ธรรมดา เป็นการแก้เงื่อนไขการ train:

```text
จาก:
train ใช้ posterior z ที่เห็น target
test ใช้ z = 0

เป็น:
train ใช้ z = 0
test ใช้ z = 0
```

และเพิ่ม density-specific losses เพื่อกัน prediction จาง:

```text
foreground_l1_loss -> เน้นบริเวณที่มี density จริง
mass_loss          -> กัน density รวมต่ำเกินไป
gamma_l1_loss      -> เน้นค่า density อ่อน
```

### Command สำหรับทดลอง v3

```bash
cd /home/johnfaqpc/programming/AI_Pedsim

.venv_sim/bin/python \
  AI_GenerateTrajectory/AI_Train/Method_CVAE/train_CVAE_densitymap_bw.py \
  --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_train_densitymap_bw_v3.json
```

### Verification

ตรวจ syntax ด้วย:

```bash
.venv_sim/bin/python -m py_compile \
  AI_GenerateTrajectory/AI_Train/Method_CVAE/cvae_model.py \
  AI_GenerateTrajectory/AI_Train/Method_CVAE/cvae_losses.py \
  AI_GenerateTrajectory/AI_Train/Method_CVAE/cvae_density_train.py \
  AI_GenerateTrajectory/AI_Train/Method_CVAE/cvae_config.py
```

ผล:

```text
compile ผ่าน
```

### สิ่งที่ต้องดูจาก v3

ควรดู sample ระหว่าง epoch โดยเฉพาะช่วง 10-15 epoch แรก:

```text
1. prediction ยังจางเหมือนเดิมหรือไม่
2. density mass ใกล้ target ขึ้นหรือไม่
3. foreground density เริ่มขึ้นตำแหน่งถูกหรือไม่
4. best_mae กับ best_loss ให้ภาพต่างกันหรือไม่
```

### เกณฑ์ตัดสินใจถัดไป

ถ้า v3 ดีขึ้น:

```text
ใช้ v3 เป็น baseline ของ CVAE BW ต่อ
จากนั้นค่อย tune loss weight / epoch / checkpoint selection
```

ถ้า v3 ยังไม่ดี:

```text
หยุดจูน CVAE ชั่วคราว
ทำ deterministic baseline เช่น conditional UNet หรือ Pix2Pix-lite
```

เหตุผล:

```text
ถ้า deterministic baseline ยังทำนาย density ได้ไม่ดี
แปลว่าปัญหาอาจอยู่ที่ dataset pair, target representation หรือ input A ไม่มี signal พอ

ถ้า deterministic baseline ดี แต่ CVAE ยังแย่
แปลว่าปัญหาอยู่ที่การออกแบบ latent/posterior/prior ของ CVAE
```

# Method_pix2pix (Standard Pix2Pix Image-to-Image Translation)

สถาปัตยกรรม **Pix2Pix** มาตรฐาน (Isola et al., CVPR 2017) สำหรับการทำนายภาพความหนาแน่นการสัญจรของผู้คน (Pedestrian Density Map) จากผังอาคาร (Floor Plan)

---

## 1. องค์ประกอบสถาปัตยกรรม (Architecture)

1. **Generator Network (U-Net 256 Architecture):**
   - **Encoder:** 8 ชั้น Downsampling (Conv2d -> BatchNorm -> LeakyReLU 0.2)
   - **Decoder:** 8 ชั้น Upsampling (ConvTranspose2d -> BatchNorm -> ReLU -> Skip Connections)
   - **Skip Connections:** นำ Feature map จากฝั่ง Encoder เชื่อมต่อ (Concatenate) เข้ากับฝั่ง Decoder ในทุกระดับชั้นความละเอียด ป้องกันการสูญหายของขอบเขตผนังและประตู
   - **Output:** ใช้ Sigmoid เพื่อให้ค่าน้ำหนักความหนาแน่นอยู่ในช่วง $[0, 1]$

2. **Discriminator Network (70x70 PatchGAN Discriminator):**
   - **Input:** รับภาพคู่ `[Input A (Floor-Plan 3-ch) + Target B (Density Map 1-ch)]` รวมเป็น 4 แชนเนล
   - **Layers:** 4-layer Convolutional Classifier ตัดสินความสมจริงของภาพเป็นส่วนๆ (Local Patches 70x70)

---

## 2. ฟังก์ชันความสูญเสีย (Loss Functions)

$$\mathcal{L}_{\text{Pix2Pix}} = \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

* **cGAN Loss (Binary Cross Entropy with Logits):**
  กระตุ้นให้ Generator สร้างภาพ Density Map ที่มีความสมจริง คมชัด เพื่อหลอก Discriminator
* **L1 Loss ($\lambda = 100$):**
  ควบคุมไม่ให้ทิศทางสัญจรเบี่ยงเบน ออกห่างจากตำแหน่งพิกัดจริงในภาพ Target

---

## 3. การใช้งาน (Usage)

### ฝึกสอนโมเดล (Train)
```bash
python train_pix2pix_densitymap_bw.py --config config_train.json
```

### ประเมินผล (Test)
```bash
python test_pix2pix_densitymap_bw.py --config config_test.json
```

### รัน Pipeline ทั้งหมด
```bash
python run_pipeline.py
```

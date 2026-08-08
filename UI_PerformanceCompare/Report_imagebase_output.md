# 📊 Comprehensive Benchmark Report: Image-Based Output (UI Performance Compare)

**Project:** AI_Pedsim (Generative AI for Pedestrian Density Map Synthesis)  
**Evaluated Set:** 862 Test Samples (Topo_HouseGAN Dataset)  
**File Generated At:** 2026-08-08  
**Target Path:** `UI_PerformanceCompare/Report_imagebase_output.md`

---

## 1. 🌟 Executive Summary & Overview

รายงานฉบับนี้รวบรวมและสรุปผลการวัดประสิทธิภาพภาพรวมทั้งหมดของ **Image-Based Output Performance Comparison** ที่แสดงผลบนระบบ Streamlit UI (`UI_PerformanceCompare`) โดยเปรียบเทียบโมเดลทั้งหมด **5 สถาปัตยกรรมหลัก + 1 Traditional Physics Simulation Baseline (JuPedSim)** บนชุดข้อมูลทดสอบจำนวน **862 ภาพ (Test Set)**

### 🤖 โมเดลที่เข้าร่วมการทดสอบ:
1. **`JuPedSim`**: Traditional Physics-Based Pedestrian Simulation (Social Force Model)
2. **`pix2pixHD`**: State-of-the-Art High-Resolution GAN (9-Block ResNet Generator + Multi-Scale Discriminators D1, D2, D3 + Feature Matching Loss)
3. **`ResNet-9`**: Deep Residual Generator (9-Block ResNet without Discriminator / Pure L1 Loss)
4. **`Pix2Pix (WGAN-GP)`**: Advanced Pix2Pix Variant (U-Net 256 + PatchGAN Critic with Wasserstein Distance + Gradient Penalty $\lambda_{gp}=10$)
5. **`CVAE`**: Conditional Variational Autoencoder (Generative Probabilistic Baseline)
6. **`Plain U-Net`**: Supervised CNN Baseline (Standard U-Net Architecture)

---

## 2. 📈 Model Benchmark Summary: Full Image (Average 862 Samples)

ตารางสรุปผลการประเมินค่าเฉลี่ยความคลาดเคลื่อนและความคล้ายคลึงของผังความหนาแน่น **คิดคำนวณครอบคลุมทั้งภาพ (Full Image Grid)**:

| Model / Method | MAE ↓ | MSE ↓ | RMSE ↓ | SSIM ↑ | PSNR ↑ (dB) | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pix2pixHD`** 🏆 | **0.001342** | **0.000072** | **0.005723** | **0.963036** | **49.25 dB** | 0.037206 |
| **`ResNet-9`** | 0.001526 | 0.000112 | 0.007079 | 0.942249 | 48.77 dB | 0.037371 |
| **`Pix2Pix (WGAN-GP)`** | 0.001604 | 0.000143 | 0.007598 | 0.921834 | 49.08 dB | **0.035272** 🏆 |
| **`CVAE`** | 0.001878 | 0.000143 | 0.008472 | 0.918709 | 44.25 dB | 0.067590 |
| **`Plain U-Net`** | 0.001988 | 0.000207 | 0.009612 | 0.878082 | 44.98 dB | 0.068607 |

---

## 3. 🚶 Model Benchmark Summary: Only Walkable Area (Average 862 Samples)

ตารางสรุปผลการประเมินค่าเฉลี่ยเฉพาะ **พื้นที่ที่คนสามารถเดินได้จริง (Walkable Space Masked Metrics)** ซึ่งตัดพื้นที่ผนังและสิ่งกีดขวางออก ทำให้เห็นความแม่นยำในการสร้างความหนาแน่นของผู้คนจริง:

| Model / Method | MAE (Walkable) ↓ | MSE (Walkable) ↓ | RMSE (Walkable) ↓ | SSIM (Walkable) ↑ | PSNR (Walkable) ↑ | LPIPS (Walkable) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pix2pixHD`** 🏆 | **0.003608** | **0.000216** | **0.009584** | **0.957365** | **45.33 dB** | **0.026249** 🏆 |
| **`ResNet-9`** | 0.004633 | 0.000365 | 0.012658 | 0.927194 | 43.98 dB | 0.033324 |
| **`Pix2Pix (WGAN-GP)`** | 0.004877 | 0.000454 | 0.013562 | 0.899790 | 43.97 dB | 0.035151 |
| **`Plain U-Net`** | 0.053403 | 0.003154 | 0.055671 | 0.092503 | 25.15 dB | 0.323772 |
| **`CVAE`** | 0.054920 | 0.003313 | 0.057096 | 0.092155 | 24.92 dB | 0.324419 |

---

## ⚡ 4. Computational Efficiency Comparison (Simulation vs. AI)

เปรียบเทียบเวลาในการประมวลผลทำนายผลลัพธ์ (Inference Time) ทั้งหมด 862 ภาพ ระหว่างระบบจำลองฟิสิกส์ดั้งเดิม (JuPedSim บน CPU) กับโมเดล Generative AI (บน NVIDIA GPU):

| Method / Model | Avg Time per Sample (s) | Total Time (s) | Speedup Factor | HW Acceleration |
| :--- | :---: | :---: | :---: | :--- |
| **`JuPedSim (Traditional Sim)`** | 29.57000 | 25,489.32 | **1.0x** | CPU Baseline |
| **`pix2pixHD`** | 0.06100 | 52.58 | **484x** 🚀 | NVIDIA GPU |
| **`ResNet-9`** | 0.06100 | 52.58 | **484x** 🚀 | NVIDIA GPU |
| **`Pix2Pix (WGAN-GP)`** | 0.01486 | 12.81 | **1,990x** 🚀 | NVIDIA GPU |
| **`CVAE`** | 0.00327 | 2.82 | **9,039x** 🚀 | NVIDIA GPU |
| **`Plain U-Net`** | 0.01486 | 12.81 | **1,990x** 🚀 | NVIDIA GPU |

---

## 👥 5. Detailed Breakdown by Occupancy Level

การสกัดผลการประเมินเจาะลึกตามระดับความหนาแน่นของผู้คน 3 ระดับ:
1. **`1-agent (Single)`**: กรณีคนเดิน 1 คน (Single Agent Route)
2. **`N-half (Half)`**: กรณีสัญจรความหนาแน่นปานกลาง (Half Occupancy)
3. **`N (Full)`**: กรณีสัญจรความหนาแน่นสูงหนาแน่นเต็มพื้นที่ (Full Occupancy)

### 5.1 Full Image Metrics by Occupancy Level:

| Occupancy Level | Model | MAE ↓ | MSE ↓ | SSIM ↑ | PSNR ↑ (dB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1-agent (Single)** | **`ResNet-9`** | 0.000183 | 0.000002 | **0.997902** | 60.96 dB |
| | **`Pix2Pix (WGAN-GP)`** | **0.000163** | **0.000002** | 0.997795 | **61.93 dB** |
| | **`pix2pixHD`** | 0.000322 | 0.000002 | 0.996540 | 58.15 dB |
| | **`Plain U-Net`** | 0.000326 | 0.000004 | 0.994728 | 54.57 dB |
| | **`CVAE`** | 0.001181 | 0.000016 | 0.970007 | 48.17 dB |
| **N-half (Half)** | **`pix2pixHD`** 🏆 | **0.001460** | **0.000052** | **0.960157** | **46.62 dB** |
| | **`ResNet-9`** | 0.001739 | 0.000093 | 0.937498 | 44.53 dB |
| | **`Pix2Pix (WGAN-GP)`** | 0.001698 | 0.000105 | 0.921826 | 44.95 dB |
| | **`CVAE`** | 0.002036 | 0.000117 | 0.904024 | 43.54 dB |
| | **`Plain U-Net`** | 0.002069 | 0.000150 | 0.878216 | 42.28 dB |
| **N (Full)** | **`pix2pixHD`** 🏆 | **0.002256** | **0.000161** | **0.931970** | **42.88 dB** |
| | **`ResNet-9`** | 0.002672 | 0.000243 | 0.890615 | 40.71 dB |
| | **`Pix2Pix (WGAN-GP)`** | 0.002969 | 0.000327 | 0.844815 | 40.21 dB |
| | **`CVAE`** | 0.003442 | 0.000429 | 0.782743 | 39.43 dB |
| | **`Plain U-Net`** | 0.003591 | 0.000470 | 0.759663 | 37.99 dB |

---

### 5.2 Walkable Area Metrics by Occupancy Level:

| Occupancy Level | Model | MAE (Walkable) ↓ | MSE (Walkable) ↓ | SSIM (Walkable) ↑ | LPIPS (Walkable) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1-agent (Single)** | **`pix2pixHD`** | 0.000685 | 0.000005 | **0.996913** | **0.001306** |
| | **`ResNet-9`** | 0.000494 | 0.000005 | 0.996788 | 0.001361 |
| | **`Pix2Pix (WGAN-GP)`** | **0.000482** | **0.000005** | 0.996463 | 0.001359 |
| | **`Plain U-Net`** | 0.054801 | 0.003009 | 0.039787 | 0.339387 |
| | **`CVAE`** | 0.056842 | 0.003247 | 0.038433 | 0.341794 |
| **N-half (Half)** | **`pix2pixHD`** 🏆 | **0.003868** | **0.000151** | **0.953851** | **0.030174** |
| | **`ResNet-9`** | 0.005263 | 0.000298 | 0.919220 | 0.039273 |
| | **`Pix2Pix (WGAN-GP)`** | 0.005149 | 0.000330 | 0.894526 | 0.040996 |
| | **`CVAE`** | 0.054029 | 0.003093 | 0.105127 | 0.320135 |
| | **`Plain U-Net`** | 0.052527 | 0.002960 | 0.319969 | 0.319969 |
| **N (Full)** | **`pix2pixHD`** 🏆 | **0.006310** | **0.000495** | **0.920814** | **0.047575** |
| | **`ResNet-9`** | 0.008195 | 0.000798 | 0.864681 | 0.059724 |
| | **`Pix2Pix (WGAN-GP)`** | 0.009060 | 0.001036 | 0.807080 | 0.063510 |
| | **`Plain U-Net`** | 0.052872 | 0.003498 | 0.134461 | 0.311781 |
| | **`CVAE`** | 0.053873 | 0.003601 | 0.133521 | 0.311129 |

---

## 🏆 6. Metric Win-Rate Analysis (% Win across 862 Images)

การนับจำนวนภาพที่แต่ละโมเดลชนะเลิศ (Best Model per Image Sample) คำนวณเป็นเปอร์เซ็นต์อัตราการชนะ (Win Rate %):

### 6.1 Full Image Win Rates (%):
* **SSIM Win Rate (ความคล้ายคลึงรวม):**
  1. **`pix2pixHD`**: **43.50%** 🏆
  2. **`Pix2Pix (WGAN-GP)`**: **21.23%**
  3. **`ResNet-9`**: **11.02%**
  4. **`CVAE`**: **5.10%**
  5. **`Plain U-Net`**: **0.12%**

* **MAE Win Rate (ความคลาดเคลื่อนต่ำสุดรวม):**
  1. **`Pix2Pix (WGAN-GP)`**: **38.17%** 🏆
  2. **`pix2pixHD`**: **31.90%**
  3. **`ResNet-9`**: **8.58%**
  4. **`Plain U-Net`**: **1.97%**
  5. **`CVAE`**: **0.93%**

### 6.2 Walkable Area Win Rates (%):
* **Walkable SSIM Win Rate (ความแม่นยำพื้นที่เดินได้):**
  1. **`pix2pixHD`**: **62.41%** 🏆
  2. **`ResNet-9`**: **12.30%**
  3. **`Pix2Pix (WGAN-GP)`**: **10.56%**

* **Walkable MAE Win Rate (ความคลาดเคลื่อนต่ำสุดพื้นที่เดินได้):**
  1. **`pix2pixHD`**: **48.72%** 🏆
  2. **`Pix2Pix (WGAN-GP)`**: **19.72%**
  3. **`ResNet-9`**: **15.20%**

* **Walkable LPIPS Win Rate (ความคมชัดเชิงการรับรู้สายตา):**
  1. **`pix2pixHD`**: **72.39%** 🏆 (ชนะขาดลอยด้านความสมจริงของภาพ)
  2. **`ResNet-9`**: **8.47%**
  3. **`Pix2Pix (WGAN-GP)`**: **8.12%**

---

## ⚠️ 7. Caveat: Model Architecture Confounding Factors

ในการเปรียบเทียบผลลัพธ์ระหว่างโมเดล มีข้อควรระวัง (Caveat) เรื่องตัวแปรแทรกซ้อน (Confounding factors) ด้านสถาปัตยกรรมของ Discriminator ที่ไม่ได้ถูกควบคุมให้เหมือนกันทั้งหมด:
* **`pix2pixHD`**: ใช้ Discriminator แบบ **Multi-scale (3 scales)** ร่วมกับ Feature Matching Loss
* **`Pix2Pix (WGAN-GP)`**: ใช้ Discriminator (Critic) แบบ **Single-scale** ตามมาตรฐาน PatchGAN

ดังนั้น ประสิทธิภาพของ `pix2pixHD` ที่สูงกว่า `Pix2Pix (WGAN-GP)` อาจไม่ได้มาจากตัว Loss เพียงอย่างเดียว แต่ได้รับอิทธิพลอย่างมากจากสถาปัตยกรรม Multi-scale Discriminator ที่ทรงพลังกว่า หากต้องการขจัด Confounding factor นี้อย่างสมบูรณ์ จะต้องปรับให้ทั้งคู่ใช้ Discriminator สถาปัตยกรรมเดียวกัน (เช่น จับคู่ Multi-scale D + WGAN-GP)

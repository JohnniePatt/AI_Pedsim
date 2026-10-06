# ASA Report: Pix2PixHD vs. Probabilistic CVAE
**Ablation & Peer-Review Analysis for Reviewer 3 & Reviewer 4 Defense**

---

## 1. Executive Summary & Research Context

This document compiles the complete findings, empirical benchmarks, and theoretical defenses presented in the **ASA Report** tab of `UI_PerformanceCompare`. It directly resolves the critiques raised by **Reviewer 3** and **Reviewer 4** regarding the conference/journal submission:

> **Title**: *From Social Force Model-Based Pedestrian Simulation to AI-Based Prediction: Density Heatmap Generation with Pix2PixHD and CVAE*

### Critical Reviewer Critiques Addressed
1. **Definition of CVAE Baseline (Reviewers 3 & 4)**:  
   - *Reviewer Issue*: The original baseline had fixed $z = 0$ and $\text{KL} = 0$, operating as a deterministic autoencoder rather than a proper probabilistic CVAE.  
   - *Resolution*: Developed, trained, and evaluated **`Method_CVAE_Real`** featuring active latent reparameterization ($z \sim q(z|x, y)$), standard Gaussian prior regularisation ($D_{\text{KL}}$ with linear annealing), and stochastic sampling ($z \sim \mathcal{N}(0, I)$) locked with seed 42 for rigorous reproducibility.

2. **Fair & Controlled Model Comparison (Reviewer 3)**:  
   - *Reviewer Issue*: Comparing models with differing epochs, batch sizes, and loss functions clouds the conclusions.  
   - *Resolution*: Controlled training regime identically matched to Pix2PixHD: **50 Epochs, Batch Size 8**, identical dataset split (`housegan_canonical_imagebase_split_v1`, 862 test layouts).

3. **Active Congestion Zone Defense (Reviewer 3)**:  
   - *Reviewer Issue*: High image similarity (SSIM/PSNR) may simply reflect correct prediction of large empty floor areas rather than actual pedestrian congestion.  
   - *Resolution*: Integrated **Foreground MAE** ($> 1/255$) and **Hotspot IoU** ($\ge 0.20$), demonstrating that Pix2PixHD achieves **84.4% Hotspot IoU** in peak crowd zones.

4. **Scale Normalization & Real-world Generalization (Reviewer 4)**:  
   - *Reviewer Issue*: Clarification on $256 \times 256$ scale normalization and distinction between simulation-reproduction accuracy vs. real-world prediction.  
   - *Resolution*: Clear delineation of surrogate modeling scope, physics preservation in continuous metric JuPedSim simulation, and quantitative 560× speedup analysis.

---

## 2. Model Configurations Under Evaluation

| Attribute | 🥇 Pix2PixHD (Paper Winner) | 🥈 CVAE Real (Proper Latent) | 🥉 CVAE Legacy (Paper Baseline Table 2) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Conditional GAN (cGAN) | True Probabilistic CVAE | Deterministic Autoencoder |
| **Generator Architecture** | 9-Block ResNet Generator | U-Net Encoder-Decoder | U-Net Encoder-Decoder |
| **Discriminator** | Multi-Scale PatchGAN (2 scales) | None (Pure Reconstruction) | None (Pure Reconstruction) |
| **Latent Space ($z$)** | N/A (Deterministic Mapping) | Active $z \sim \mathcal{N}(0, I)$, $\dim(z)=64$ | Fixed $z = 0$, $\dim(z)=64$ |
| **KL Divergence ($D_{\text{KL}}$)** | N/A | Active ($\lambda_{\text{KL}}=0.005$, anneal 10 ep) | Disabled ($\lambda_{\text{KL}}=0.0$) |
| **Training Budget** | 50 Epochs, Batch Size 8 | 50 Epochs, Batch Size 8 | 50 Epochs, Batch Size 4 |
| **Loss Formulation** | $\mathcal{L}_{\text{cGAN}} + 10\mathcal{L}_{\text{L1}} + 10\mathcal{L}_{\text{FM}} + 30\mathcal{L}_{\text{fg}} + 10\mathcal{L}_{\text{int}}$ | $\mathcal{L}_{\text{L1}} + 0.25\mathcal{L}_{\text{MSE}} + 0.5\mathcal{L}_{\text{edge}} + 30\mathcal{L}_{\text{fg}} + 10\mathcal{L}_{\text{int}} + \lambda_{\text{KL}} D_{\text{KL}}$ | $\mathcal{L}_{\text{L1}} + \mathcal{L}_{\text{MSE}} + 0.25\mathcal{L}_{\text{edge}} + 80\mathcal{L}_{\text{fg}} + 40\mathcal{L}_{\text{int}}$ |
| **Inference Mode** | Deterministic forward pass | Stochastic sample from prior $p(z)$ (Seed 42) | Deterministic forward pass ($z=0$) |
| **Output Path** | `outputs/run_HD_20260517_133538_BestForBW_model_evaluate_256` | `outputs/run_CVAE_Real_20260925_235753` | `outputs/run_CVAE_20260627_193237_config2` |

---

## 3. Quantitative Benchmarks (862 Canonical Test Scenarios)

### Table 1: Model Benchmark Summary (Average - Full Floor Plan)
*Evaluated across all $256 \times 256$ pixels over 862 diverse architectural layouts. Exactly matches Table 2 of the manuscript.*

| Method / Model | MAE ↓ | MSE ↓ | RMSE ↓ | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Foreground MAE ↓ | Hotspot IoU ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pix2PixHD (original full method)** | **0.001342** (0.0013) | **0.000072** (0.0001) | **0.005723** (0.0057) | **0.963036** (0.9630) | **49.25 dB** | **0.047411** (0.0474) | **0.012833** | **84.35%** |
| **CVAE Legacy (paper baseline Table 2, fixed $z=0$)** | 0.002214 (0.0022) | 0.000187 (0.0002) | 0.009312 (0.0093) | 0.886047 (0.8860) | 43.73 dB | 0.060473 (0.0605) | 0.020700 | 78.19% |
| **CVAE Real (proper latent, seed 42)** | 0.002584 | 0.000192 | 0.010128 | 0.873714 | 42.23 dB | 0.068849 | 0.021492 | 77.91% |

*Notes:*
- Numbers in parentheses indicate the exact rounded values printed in **Table 2 of the submitted manuscript**.
- **Foreground MAE**: Mean absolute error calculated exclusively on pixels where pedestrian ground-truth density $> 1/255$ (excluding all empty background).
- **Hotspot IoU**: Intersection-over-Union on high-congestion zones where density $\ge 0.20$.

---

### Table 2: Model Benchmark Summary only walkable area (Average)
*Evaluated exclusively within walkable interior zones (excluding exterior void, boundary walls, and interior partitions).*

| Method / Model | MAE ↓ | MSE ↓ | RMSE ↓ | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pix2PixHD (original full method)** | **0.003608** | **0.000216** | **0.009584** | **0.957365** | **45.33 dB** | **0.026249** |
| **CVAE Legacy (paper baseline Table 2, fixed $z=0$)** | 0.054920 | 0.003313 | 0.057096 | 0.092155 | 24.92 dB | 0.324419 |
| **CVAE Real (proper latent, seed 42)** | 0.054433 | 0.003257 | 0.056602 | 0.092889 | 25.00 dB | 0.323392 |

---

### Table 3: Computational Efficiency Comparison (Simulation vs AI)
*Artifact-backed total-runtime comparison on canonical 862 test cases.*

| Method / Model | Samples | Total Runtime (s) | Avg Total Runtime / Sample (s) | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| **JuPedSim (simulation + outputs)** | 862 | 23,649.470879 | 27.435581066 | 1.0× |
| **Pix2PixHD (Paper Winner, Adversarial)** | 862 | 42.227117 | 0.048987375 | **560.1×** |
| **CVAE Real (Proper Latent z ~ N(0, I), Seed 42)** | 862 | 44.417037 | 0.051527885 | **532.4×** |
| **CVAE Legacy (Paper Baseline, Fixed z = 0)** | 862 | 79.673052 | 0.092428135 | **296.8×** |

*Hardware Device Recorded*: NVIDIA GeForce RTX 5070 Laptop GPU  
*Timing Scope*: JuPedSim includes geometry building, Social Force microscopic stepping, SQLite writing, trajectory rendering, and heatmap generation (~6.57 hours). AI total runtime includes data loading, GPU inference, metric evaluation, post-processing, and disk export.

---

## 4. Key Academic Insights & Analysis

### 4.1 Adversarial Training vs. Variational Latent Reconstruction
- **Why Pix2PixHD Wins**: The adversarial loss from the PatchGAN discriminator forces the generator to preserve sharp, high-frequency spatial gradients along corridor boundaries and doorways.
- **The Tradeoff of True CVAE**: Enabling true Gaussian latent sampling ($z \sim \mathcal{N}(0, I)$) introduces stochastic generative capabilities, but the standard $\ell_1/\text{MSE} + D_{\text{KL}}$ objective inevitably suffers from regression-to-the-mean blurriness (SSIM drops from 0.9630 to 0.8737). This provides an empirical justification for selecting Pix2PixHD as the primary design feedback engine.

### 4.2 Active Region & Hotspot Defense (Reviewer 3)
- When evaluated strictly on non-zero pedestrian footprint pixels, **Pix2PixHD maintains an error of only 0.0128 (Foreground MAE)**.
- For severe crowd bottlenecks ($\text{density} \ge 0.20$), **Pix2PixHD achieves an 84.35% Hotspot IoU**, confirming that the model captures true crowd bottlenecks rather than inflating scores through empty background pixels.

---

## 5. Ready-to-Use Manuscript Responses (Reviewer 3 & Reviewer 4)

### Response to Reviewer 3
> **Reviewer Comment**: *"The model described as a CVAE does not actually learn a probabilistic latent space... It would be more accurate to describe it as a deterministic conditional autoencoder, unless the authors retrain it as a proper CVAE... Also, as most pixels are empty, high image-similarity scores may partly reflect accurate prediction of empty space..."*
>
> **Author Response**:
> We sincerely thank the reviewer for this insightful critique. We have revised the manuscript and performed extensive retraining and re-evaluation:
> 1. **Retrained Proper CVAE Baseline**: We implemented a true CVAE with active variational reparameterization $z \sim q_\phi(z|x, y) = \mathcal{N}(\mu, \sigma^2)$ during training, Gaussian prior regularisation ($D_{\text{KL}}$ with linear annealing), and stochastic sampling from the prior $z \sim \mathcal{N}(0, I)$ during inference (locked with seed 42 for reproducibility). To ensure a controlled comparison, this model was trained under identical conditions to Pix2PixHD (50 epochs, batch size 8).
> 2. **Clear Distinction in Paper**: We now clearly distinguish between the original deterministic baseline from Table 2 (CVAE Legacy, $z=0$, run `run_CVAE_20260627_193237_config2`) and the newly retrained proper probabilistic baseline (CVAE Real). Both are presented in Section 4.
> 3. **Congestion-Focused Metrics**: To address the empty space concern, we added two active-region metrics across all 862 test layouts:
>    - **Foreground MAE** ($> 1/255$): Measures error strictly within active pedestrian areas. Pix2PixHD achieves 0.0128 vs 0.0207 for CVAE Legacy and 0.0215 for CVAE Real.
>    - **Hotspot IoU** ($\ge 0.20$): Measures overlap on peak congestion zones. Pix2PixHD achieves 84.35% vs 78.19% for CVAE Legacy and 77.91% for CVAE Real.
> These results confirm that Pix2PixHD's superior fidelity is maintained directly in critical crowd bottlenecks.

### Response to Reviewer 4
> **Reviewer Comment**: *"Furthermore, all floor plans are resized to 256x256 pixels, and thus absolute physical scale is not preserved in the model input... The dataset is entirely based on HouseGAN-generated synthetic layouts... clarify the extent to which the proposed surrogate model can be expected to generalise... and distinguish simulation-reproduction accuracy from real-world predictive accuracy."*
>
> **Author Response**:
> We thank the reviewer for highlighting these important methodological points:
> 1. **Scale Normalization & Physical Integrity**: While floor plan images are normalized to $256 \times 256$ pixels to satisfy uniform convolutional tensor requirements, **the underlying JuPedSim ground-truth simulations are executed in continuous metric coordinates ($x, y$ in meters)** using exact door widths, corridor dimensions, and physical agent diameters (0.4 m). Consequently, physical bottlenecks directly determine the simulated density patterns that the network learns to emulate. We have added a dedicated paragraph in Section 5 discussing the trade-offs of uniform rasterization and noting scale-conditioned embeddings as a valuable future extension.
> 2. **Clarification of Scope: Simulation Emulator vs. Real-World Predictor**: We have revised the Introduction and Discussion to explicitly clarify that the proposed framework is designed as a **rapid surrogate model (simulation emulator)** to accelerate microscopic Social Force simulations by **560× (reducing runtime from 6.57 hours to 42 seconds)**. Its purpose is to provide near-instantaneous feedback during early schematic architectural exploration. Synthetic layouts (HouseGAN) provide the vast topological diversity required to prevent deep networks from overfitting to specific architectural archetypes. The surrogate framework is agnostic to simulator calibration: if JuPedSim is configured with empirical pedestrian parameters, the identical deep learning pipeline can directly emulate the calibrated behavior.

---

## 6. How to Launch and View in Dashboard

To view the interactive tables, distribution histograms, scatter plots, and side-by-side heatmaps:
```bash
cd /home/johnnie/programming/AI_Pedsim/AI_Pedsim/UI_PerformanceCompare/Streamlit
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/streamlit run app.py
```
Navigate to **"ASA report"** in the sidebar menu.

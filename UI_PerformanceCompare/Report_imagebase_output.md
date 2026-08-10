# Image-Based Output Benchmark Report

**Project:** AI_Pedsim - pedestrian density heatmap surrogate models

**UI:** `UI_PerformanceCompare/Streamlit`

**Evaluated split:** canonical HouseGAN test split, 862 scenarios from 117 unseen floor plans

**Dataset ID:** `housegan_canonical_imagebase_split_v1`

**Updated:** 2026-08-09
**Paper source:** `ICCEA_FP_Image 4.pdf`

---

## 1. Scope and comparison decision

The main Image-Based Output comparison now contains four deterministic surrogate configurations plus JuPedSim as the physics-based timing reference:

1. **pix2pixHD** - nine-block ResNet-style generator with adversarial training and a multi-scale discriminator.
2. **ResNet-9** - non-adversarial nine-block ResNet-style generator.
3. **Pix2Pix (WGAN-GP)** - U-Net generator with a single-scale PatchGAN critic and WGAN-GP training. This is named `pix2pix` in the paper.
4. **Plain U-Net** - the matching U-Net generator trained non-adversarially.
5. **JuPedSim** - Social Force Model simulation used only as the computational-time baseline, not as an image-prediction competitor.

CVAE is excluded from all comparison tables, charts, winner calculations, failure-case ranking, and image grids. The evaluated CVAE v3 run fixed `z=0` during training and inference and used `kl_weight=0`; it therefore did not provide a valid probabilistic CVAE comparison. Its artifacts remain on disk for provenance and have not been deleted.

---

## 2. Research design represented by the UI

After removing CVAE, the comparison is the paper's core 2 x 2 factorial-style ablation:

| Backbone family | Non-adversarial | Adversarial |
| :--- | :--- | :--- |
| U-Net | Plain U-Net | Pix2Pix (WGAN-GP) |
| ResNet-style | ResNet-9 | pix2pixHD |

The U-Net pair shares the same generator, so it isolates the discriminator contribution precisely. The ResNet pair consists of independent implementations that share only the backbone family; it also differs in discriminator scale. Its adversarial comparison is therefore approximate rather than fully controlled.

This distinction is essential: the results support a backbone-dependent adversarial effect, but only the U-Net-side effect is based on an exactly matched generator.

---

## 3. Dataset and simulation protocol

The paper reports 610 synthetically generated HouseGAN layouts across six size categories. It then reports 1,400 candidate routes, 1,326 valid routes after route filtering, and 3,904 completed occupancy scenarios after JuPedSim processing.

The canonical image-based dataset currently verified on disk contains:

| Split | Scenarios | Unique floor plans |
| :--- | ---: | ---: |
| Train | 2,603 | 412 |
| Validation | 439 | 60 |
| Test | 862 | 117 |
| **Total** | **3,904** | **589** |

The paper's `610 floor-plan layouts` describes the generated source population, whereas the canonical evaluated split contains 589 unique plan identifiers. The difference must remain explicit; the report does not silently relabel 610 generated layouts as 610 evaluated layouts.

Reference density fields were produced with JuPedSim v1.3.2 using the Social Force Model and default parameters without case-specific calibration. Maximum occupancy was derived from 2 persons/m2 of origin-room area. Every route was represented at three occupancy levels: `N`, `N/2`, and `1 agent`.

Each input is an RGB raster encoding walkable area, walls, destination, and origin agents. Targets are time-averaged Voronoi local-density fields computed with PedPy on a 0.5 x 0.5 m grid, rendered in grayscale with `v_max = 5.0 persons/m2` mapped to `[0,255]`. Splitting is performed at floor-plan level to prevent geometric leakage.

---

## 4. Evaluation metrics

All image metrics are computed on the 862-scenario held-out test set:

- **MAE, MSE, RMSE:** pixel-level error on grayscale density normalized to `[0,1]`; lower is better.
- **SSIM:** structural similarity; higher is better.
- **PSNR:** signal fidelity in decibels; higher is better.
- **LPIPS:** perceptual distance using a pretrained AlexNet backbone; lower is better.
- **Walkable metrics:** the same measures restricted to the walkable-area mask.

Normalized density can be converted back to physical density using `v_max = 5.0 persons/m2`.

---

## 5. Aggregate test results

### 5.1 Full-image metrics (mean over 862 scenarios)

| Model | MAE down | MSE down | RMSE down | SSIM up | PSNR up (dB) | LPIPS down |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **pix2pixHD** | **0.001342** | **0.000072** | **0.005723** | **0.963036** | **49.25** | 0.037206 |
| **ResNet-9** | 0.001526 | 0.000112 | 0.007079 | 0.942249 | 48.77 | 0.037371 |
| **Pix2Pix (WGAN-GP)** | 0.001604 | 0.000143 | 0.007598 | 0.921834 | 49.08 | **0.035272** |
| **Plain U-Net** | 0.001988 | 0.000207 | 0.009612 | 0.878082 | 44.98 | 0.068607 |

pix2pixHD leads five of the six aggregate metrics. Pix2Pix (WGAN-GP) has the lowest LPIPS.

### 5.2 Walkable-area metrics (mean over 862 scenarios)

| Model | MAE down | MSE down | RMSE down | SSIM up | PSNR up (dB) | LPIPS down |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **pix2pixHD** | **0.003608** | **0.000216** | **0.009584** | **0.957365** | **45.33** | **0.026249** |
| **ResNet-9** | 0.004633 | 0.000365 | 0.012658 | 0.927194 | 43.98 | 0.033324 |
| **Pix2Pix (WGAN-GP)** | 0.004877 | 0.000454 | 0.013562 | 0.899790 | 43.97 | 0.035151 |
| **Plain U-Net** | 0.053403 | 0.003154 | 0.055671 | 0.092503 | 25.15 | 0.323772 |

Walkable-area evaluation strengthens the same ordering and exposes the large foreground-density error of Plain U-Net that is diluted by full-image background pixels.

---

## 6. Occupancy-level analysis

### 6.1 Full-image results by occupancy

| Occupancy | Model | MAE down | MSE down | SSIM up | PSNR up (dB) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1-agent | ResNet-9 | 0.000183 | 0.000002 | **0.997902** | 60.96 |
| | Pix2Pix (WGAN-GP) | **0.000163** | **0.000002** | 0.997795 | **61.93** |
| | pix2pixHD | 0.000322 | 0.000002 | 0.996540 | 58.15 |
| | Plain U-Net | 0.000326 | 0.000004 | 0.994728 | 54.57 |
| N/2 | pix2pixHD | **0.001460** | **0.000052** | **0.960157** | **46.62** |
| | ResNet-9 | 0.001739 | 0.000093 | 0.937498 | 44.53 |
| | Pix2Pix (WGAN-GP) | 0.001698 | 0.000105 | 0.921826 | 44.95 |
| | Plain U-Net | 0.002069 | 0.000150 | 0.878216 | 42.28 |
| N (full) | pix2pixHD | **0.002256** | **0.000161** | **0.931970** | **42.88** |
| | ResNet-9 | 0.002672 | 0.000243 | 0.890615 | 40.71 |
| | Pix2Pix (WGAN-GP) | 0.002969 | 0.000327 | 0.844815 | 40.21 |
| | Plain U-Net | 0.003591 | 0.000470 | 0.759663 | 37.99 |

All four models degrade as occupancy rises. At full occupancy, the ResNet-style configurations retain substantially higher SSIM than the U-Net configurations. The discriminator gain is also larger for the U-Net pair (`0.8448 - 0.7597 = 0.0851`) than for the approximate ResNet pair (`0.9320 - 0.8906 = 0.0414`).

### 6.2 Walkable-area results by occupancy

| Occupancy | Model | MAE down | MSE down | SSIM up | LPIPS down |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1-agent | pix2pixHD | 0.000685 | 0.000005 | **0.996913** | **0.001306** |
| | ResNet-9 | 0.000494 | 0.000005 | 0.996788 | 0.001361 |
| | Pix2Pix (WGAN-GP) | **0.000482** | **0.000005** | 0.996463 | 0.001359 |
| | Plain U-Net | 0.054801 | 0.003009 | 0.039787 | 0.339387 |
| N/2 | pix2pixHD | **0.003868** | **0.000151** | **0.953851** | **0.030174** |
| | ResNet-9 | 0.005263 | 0.000298 | 0.919220 | 0.039273 |
| | Pix2Pix (WGAN-GP) | 0.005149 | 0.000330 | 0.894526 | 0.040996 |
| | Plain U-Net | 0.052527 | 0.002960 | 0.319969 | 0.319969 |
| N (full) | pix2pixHD | **0.006310** | **0.000495** | **0.920814** | **0.047575** |
| | ResNet-9 | 0.008195 | 0.000798 | 0.864681 | 0.059724 |
| | Pix2Pix (WGAN-GP) | 0.009060 | 0.001036 | 0.807080 | 0.063510 |
| | Plain U-Net | 0.052872 | 0.003498 | 0.134461 | 0.311781 |

---

## 7. Per-image win rates after CVAE exclusion

The following rates were recomputed from the four selected runs. They must not reuse the former five-model percentages.

### 7.1 Full image

| Metric | pix2pixHD | ResNet-9 | Pix2Pix (WGAN-GP) | Plain U-Net | Tie |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MAE | 36.89% | 11.95% | **48.26%** | 2.90% | 0.00% |
| SSIM | **49.42%** | 17.75% | 32.37% | 0.46% | 0.00% |
| LPIPS | **41.42%** | 18.56% | 39.10% | 0.00% | 0.93% |

### 7.2 Walkable area

| Metric | pix2pixHD | ResNet-9 | Pix2Pix (WGAN-GP) | Plain U-Net | Tie |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MAE | **54.41%** | 19.84% | 24.25% | 0.00% | 1.51% |
| SSIM | **66.36%** | 19.84% | 12.53% | 0.00% | 1.28% |
| LPIPS | **77.84%** | 10.67% | 10.56% | 0.00% | 0.93% |

---

## 8. Route-length relationship

The paper reports Pearson correlation between route length and each metric. Removing CVAE does not alter the correlations of the remaining models:

| Model | MAE r | MSE r | RMSE r | SSIM r | PSNR r | LPIPS r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| pix2pixHD | 0.0931 | 0.1314 | 0.1952 | -0.2360 | -0.0744 | 0.0678 |
| ResNet-9 | 0.0775 | 0.1106 | 0.1645 | -0.1810 | 0.0052 | 0.1288 |
| Pix2Pix (WGAN-GP) | 0.0642 | 0.0879 | 0.1435 | -0.1410 | 0.0209 | 0.1658 |
| Plain U-Net | 0.0090 | 0.0791 | 0.1262 | -0.1270 | -0.0518 | 0.1807 |

All absolute correlations remain below 0.24 for the four-model comparison. Occupancy level is therefore more strongly associated with degradation than route length within this dataset.

---

## 9. Failure-case analysis after CVAE exclusion

The UI now ranks cases dynamically across the four included models. Recalculation gives an overall pooled SSIM mean of `0.9263`, standard deviation `0.1237`, and a `mean +/- 0.5 SD` normal-case range of `[0.8645, 0.9881]`. A total of 176 test scenarios satisfy the normal-case criterion simultaneously across all four models.

The three lowest mean-SSIM cases from unique layouts are:

1. `plan_141_4cca__45_03_full.png` - mean SSIM `0.4075`
2. `plan_191_3f10__45_03_full.png` - mean SSIM `0.4232`
3. `plan_104_c330__42_00_full.png` - mean SSIM `0.4707`

The paper's qualitative diagnosis remains applicable to the four included models: the hardest cases are full-occupancy scenarios with concentrated doorway or corridor bottlenecks. pix2pixHD, ResNet-9, and Pix2Pix generally localize these hotspots but often overestimate their spatial extent; Plain U-Net may produce a small misplaced blob or near-uniform low density. Small single-agent cases can generate near-zero target fields and should not be treated as strong evidence of model capability merely because their metrics are high.

---

## 10. Computational efficiency

Timing is reported for 862 test scenarios. AI models ran on an NVIDIA RTX 3080; JuPedSim ran on an Intel Core i9 12th-generation CPU.

| Method | Average time/sample (s) | Total time (s) | Speedup vs. JuPedSim |
| :--- | ---: | ---: | ---: |
| JuPedSim | 29.57000 | 25,489.32 | 1.0x |
| Plain U-Net | 0.01486 | 12.81 | 1,990x |
| Pix2Pix (WGAN-GP) | 0.01486 | 12.81 | 1,990x |
| ResNet-9 | 0.06100 | 52.58 | 484x |
| pix2pixHD | 0.06100 | 52.58 | 484x |

The paired methods have identical inference times because discriminators are used only during training. The timing comparison demonstrates workflow acceleration but confounds algorithm and hardware; it is not a controlled CPU-versus-CPU or GPU-versus-GPU benchmark.

---

## 11. Main findings

1. **Backbone choice is the strongest observed factor.** ResNet-style configurations outperform U-Net configurations under both adversarial and non-adversarial objectives.
2. **The adversarial contribution depends on backbone.** Aggregate SSIM improves by `+0.0437` for the exactly matched U-Net pair and `+0.0208` for the approximately matched ResNet pair.
3. **Congestion is the primary unresolved condition.** Full occupancy produces the largest performance gaps and the most severe localized-hotspot failures.
4. **Route length has only a weak relationship with error.** The four retained models have `|r| < 0.24` across the reported route-length correlations.
5. **All retained AI surrogates are substantially faster than JuPedSim**, although the hardware differs.

---

## 12. Limitations and next experiments

- Each configuration was trained once. Differences are descriptive, not statistically validated; repeated seeds and significance testing are required.
- Only the U-Net pair is generator-matched. A matched ResNet generator with and without the same discriminator is needed to isolate the adversarial contribution cleanly.
- Density is clipped at `v_max = 5.0 persons/m2`, so more extreme congestion is not represented.
- Layouts are rescaled to a fixed image frame, meaning absolute physical scale varies across samples.
- The study uses synthetic HouseGAN layouts only; generalization to professionally designed buildings remains untested.
- Suitable deterministic future models include Attention U-Net, UNet++, and SegFormer. Probabilistic models should be revisited only with a protocol containing repeated outcomes per identical condition and explicit stochastic evaluation.

---

## 13. Reproducibility note

The UI reads metrics from evaluation artifacts; it does not manufacture or relabel results. Removing CVAE changes only the comparison consumer and this report. CVAE code, checkpoints, predictions, and historical metrics remain untouched for auditability.

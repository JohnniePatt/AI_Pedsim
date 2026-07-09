# Pix2PixHD No_D Concept

This method uses the Pix2PixHD ResNet Generator architecture but trains **without a Discriminator**.

- Grayscale density-map predictions are saved as `predictions/{filename}.png`.
- Grayscale density-map target images are saved under `targets/{filename}.png`.

Entry points:
- `train_pix2pixhd_NoD_densitymap_bw.py`: Model training pipeline.
- `test_pix2pixhd_NoD_densitymap_bw.py`: Model evaluation pipeline.

---

## Core Idea

Pix2PixHD typically relies on adversarial training (GAN) with multi-scale Discriminators and Feature Matching loss. In this variant, the **Discriminator is completely disabled**. The generator (based on a high-capacity ResNet backbone) is trained in a purely supervised manner.

This serves as an ablation study to isolate the impact of:
1. High-resolution multi-scale ResNet generator architecture.
2. Loss weights designed for sparse density maps.
3. Lack of GAN-induced high-frequency hallucinations.

```text
Input A (3ch RGB layout) 
  |
  v
Front-End Downsampling 
  |
  v
ResNet Residual Blocks (Feature Mapping)
  |
  v
Back-End Upsampling -> Output B (3ch RGB representation of BW density)
```

---

## Target Representation

In this method, the target grayscale density maps are loaded as **3-channel RGB representation of BW density**. During inference, output pixel channels are normalized and mapped back to ColorJet for presentation in the UI, and Grayscale maps are backed up as `MASK_<filename>`.

---

## Loss Design

Since there is no adversarial discriminator, the network is optimized using a custom **Density-Aware L1 Reconstruction Loss**:

$$\text{Loss} = \text{L1\_Loss}(\text{Fake\_B}, \text{Real\_B}) \times \text{Weights}$$

Where the pixel-wise weights are calculated on the target mask to focus learning on pedestrian locations:

```python
# Compute loss weights dynamically based on target density
weights = 1.0
if target_foreground:
    weights += density_foreground_weight (default: 30.0)
if target_intensity:
    weights += target_gray * density_intensity_weight (default: 10.0)
```

This prevents the generator from taking the easy shortcut of predicting all-black (empty) images due to the sparse nature of pedestrian density fields.

---

## Expected Training Details

- **Dataset size**:
  - Train: ~2603 images
  - Validation: ~439 images
  - Test: ~862 images
- **Default Resolution**: 256x256 (internally padded/resized to multiples of 32)
- **Batch Size**: 8
- **Learning Rate**: 0.0002
- **L1 Loss Weight**: 10.0

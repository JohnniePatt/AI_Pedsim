# PlainUNet Concept

This method supports direct layout-to-density mapping using a standard vanilla UNet architecture.

- Grayscale density-map predictions are saved as `predictions/{filename}.png`.
- Grayscale density-map target images are saved under `targets/{filename}.png`.

Entry points:
- `train_PlainUnet_densitymap.py`: Model training pipeline.
- `test_PlainUnet_densitymap.py`: Model evaluation pipeline.

---

## Core Idea

Vanilla UNet is a fully convolutional network that passes features from layout maps directly to predicted grayscale density fields. The architecture uses symmetrical downsampling (encoder) and upsampling (decoder) paths, bridged by skip connections. This ensures spatial context (e.g. wall layouts) is directly accessible during the upsampling recovery of pedestrian heatmaps.

```text
Input A (3ch RGB layout) 
  |
  v
ConvBlock (Encoder) ------ [Skip Conn] ------ ConvBlock (Decoder) -> Output B (1ch Grayscale Density)
  |                                            ^
  v                                            |
  +------> Pool ----> Bottleneck ----> Up -----+
```

---

## Target Representation

Unlike Pix2PixHD, which processes Grayscale images using 3 channels (RGB), PlainUNet targets a strict **1-channel grayscale representation** (`L` mode) during the core network operations. This provides a direct linear mapping of scalar density values.

During testing/evaluation:
- The network outputs a 1-channel grayscale tensor in range `[0, 1]` via sigmoid activation.
- The 1-channel outputs are converted to visual 3-channel COLORJET images (saved as main prediction/target files) using a fallback Jet color ramp mapping.
- The pure scalar 1-channel grayscale inputs are backed up as `MASK_<filename>` for exact density mapping tasks.

---

## Loss Design

PlainUNet relies on a straightforward **L1 Reconstruction Loss**:

$$\text{Loss} = \frac{1}{N} \sum | \hat{B}_i - B_i |$$

This encourages pixel-wise similarity without the structural complexities or unstable training curves of adversarial networks (GANs) or conditional probabilities (CVAEs). It serves as a strong baseline model for deterministic mapping of pedestrian trajectory density.

---

## Expected Training Details

- **Dataset size**:
  - Train: ~2603 images
  - Validation: ~439 images
  - Test: ~862 images
- **Default Resolution**: 256x256
- **Batch Size**: 8
- **Learning Rate**: 0.0002

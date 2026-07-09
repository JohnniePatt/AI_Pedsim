# Pix2PixHD Concept

This method uses the standard high-resolution Pix2PixHD generative adversarial network (GAN) model.

- Grayscale density-map predictions are saved as `predictions/{filename}.png`.
- Grayscale density-map target images are saved under `targets/{filename}.png`.

Entry points:
- `train_pix2pixHD_densitymap_bw.py`: Model training pipeline.
- `test_pix2pixHD_densitymap_bw.py`: Model evaluation pipeline.

---

## Core Idea

Pix2PixHD is a state-of-the-art conditional GAN (cGAN) framework designed for high-resolution image-to-image translation. In our project, it maps layout images (A) directly to grayscale pedestrian density maps (B). 

It features:
1. **Coarse-to-fine Generator**: A global generator network ($G_1$) combined with a local enhancer network ($G_2$) to output clean, crisp, and high-resolution layout details.
2. **Multi-scale Discriminators**: Uses two or three patch-based discriminators ($D_1$, $D_2$, etc.) operating at different spatial resolutions to capture both global structure and local details.

```text
Input A (3ch RGB layout) 
  |
  v
Generator G (G1 + G2 Enhancer) -> Predicted Output B (Fake B)
  |
  +----------------------+
  |                      |
  v                      v
Discriminator D1       Discriminator D2 (Downscaled)
  |                      |
  +----------> Loss <----+
```

---

## Target Representation

Grayscale density maps are learned as **3-channel RGB representations of BW density**. Grayscale inputs are converted to ColorJet in the UI and backed up as `MASK_<filename>`.

---

## Loss Design

The model is optimized using a combination of multiple loss functions:

1. **Adversarial Loss**: Encourages realistic and sharp density distribution.
2. **Feature Matching Loss**: Minimizes the difference between discriminator activations of real and fake images across multiple scales.
3. **Density-Aware L1 Reconstruction Loss**: A weighted L1 loss focusing on foreground (pedestrian density locations) and intensity profiles.

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{GAN}}(G, D) + \lambda_{\text{FM}} \mathcal{L}_{\text{FM}}(G, D) + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}}(G)$$

---

## Expected Training Details

- **Dataset size**:
  - Train: ~2603 images
  - Validation: ~439 images
  - Test: ~862 images
- **Default Resolution**: 256x256 (pad/resize to multiple of 32)
- **Batch Size**: 8
- **Learning Rate**: 0.0002

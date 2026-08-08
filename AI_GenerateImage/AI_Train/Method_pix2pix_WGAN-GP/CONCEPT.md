# Method_pix2pix_WGAN-GP Architecture & Concept

## Overview
`Method_pix2pix_WGAN-GP` trains a **Pix2Pix Conditional GAN** where the generator uses a 256-scale U-Net architecture with skip connections, and the discriminator uses a 70x70 PatchGAN Critic trained with **Wasserstein GAN Loss with Gradient Penalty (WGAN-GP)** instead of standard Binary Cross-Entropy (BCE).

## Objective Function
1. **Critic Loss (Wasserstein Distance + Gradient Penalty):**
   $$\mathcal{L}_D = \mathbb{E}[\mathcal{D}(A, G(A))] - \mathbb{E}[\mathcal{D}(A, B)] + \lambda_{gp} \mathbb{E}[(\|\nabla_{\hat{B}} \mathcal{D}(A, \hat{B})\|_2 - 1)^2]$$
   where $\lambda_{gp} = 10.0$.

2. **Generator Loss:**
   $$\mathcal{L}_G = -\mathbb{E}[\mathcal{D}(A, G(A))] + \lambda_{L1} \| B - G(A) \|_1$$
   where $\lambda_{L1} = 100.0$.

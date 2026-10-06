# Method_CVAE_Real: True Probabilistic Latent CVAE

This directory implements a true probabilistic Conditional Variational Autoencoder (CVAE) for pedestrian density heatmap generation. It directly addresses the critical feedback from **Reviewer 3** and **Reviewer 4**:
- **Learned Latent Space with KL Regularization:** Unlike the legacy `Method_CVAE` (which fixed $z=0$ and set `kl_weight=0`), `Method_CVAE_Real` samples $z \sim q(z|x, y) = \mathcal{N}(\mu, \sigma^2)$ during training and regularizes the latent space with Gaussian KL divergence ($D_{\mathrm{KL}}(q(z|x, y) \parallel p(z))$) with KL annealing.
- **Reproducible Stochastic Inference:** Supports random prior sampling ($z \sim \mathcal{N}(0, I)$) with a fixed random seed (`--seed 42`), ensuring that all experimental evaluations produce identical, reproducible results across runs.
- **Computational Runtime Tracking:** Implements comprehensive timing tracking matching `Method_pix2pix_WGAN-GP`:
  - `logs/training_history.csv`: Records per-epoch loss and metrics.
  - `logs/training_runtime.csv`: Records training duration and samples.
  - `test_runtime.csv`: Records inference `Time Generate`, `Average Time Generate Per Image`, `runtime_excluding_metrics_s`, and device metadata.
- **Congestion & Active Region Metrics:** Calculates `Foreground_MAE` (error only where pedestrian density is present) and `Hotspot_IoU` (IoU on high-congestion zones $\ge 0.20$) to prevent evaluation bias caused by empty floor plan spaces.

## Usage

### 1. Training
Run training using the preconfigured JSON config:
```bash
python3 train_CVAE_real_densitymap_bw.py --config config_train_real_cvae.json
```

Outputs will be saved under:
`AI_GenerateImage/AI_Result/Method_CVAE_Real/outputs/run_CVAE_Real_<timestamp>/`

### 2. Testing / Evaluation
Run evaluation on a trained checkpoint:
```bash
python3 test_CVAE_real_densitymap_bw.py \
    --run_path <path_to_run_dir> \
    --checkpoint_mode best_mae \
    --latent_mode random \
    --seed 42 \
    --num_samples 1
```

To run Monte Carlo stochastic averaging (e.g., 5 samples per layout):
```bash
python3 test_CVAE_real_densitymap_bw.py \
    --run_path <path_to_run_dir> \
    --checkpoint_mode best_mae \
    --latent_mode random \
    --seed 42 \
    --num_samples 5
```

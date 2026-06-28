# Method_CVAE

PyTorch Conditional Variational Autoencoder (CVAE) for density-map image output.

This method supports two density-map target representations:

- ColorJet direct prediction
- BW/grayscale scalar density prediction

The older trajectory-line CVAE scripts were removed from this folder. `Method_CVAE` is now scoped to density-map experiments only.

## Files

```text
train_CVAE_densitymap_color.py
test_CVAE_densitymap_color.py
config_train_densitymap_color.json
config_test_densitymap_color.json

train_CVAE_densitymap_bw.py
test_CVAE_densitymap_bw.py
config_train_densitymap_bw.json
config_train_densitymap_bw_v2.json
config_train_densitymap_bw_v3.json
config_test_densitymap_bw.json
```

Shared implementation files:

```text
cvae_density_train.py
cvae_density_test.py
cvae_model.py
cvae_data.py
cvae_losses.py
cvae_io.py
cvae_config.py
```

## Dataset

Expected structure:

```text
A/train, A/validation, A/test
B/train, B/validation, B/test
```

ColorJet dataset:

```text
../Dataset/Data_ImageUNet/DensityMap_COLORJET_dataset/Topo_HouseGAN
```

BW dataset:

```text
../Dataset/Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN
```

Current split size for both datasets:

```text
train      = 2603
validation = 439
test       = 862
```

## Channel Design

ColorJet CVAE:

```text
input A  = RGB, 3 channels
target B = ColorJet RGB, 3 channels
output   = 3 channels
```

BW CVAE:

```text
input A  = RGB, 3 channels
target B = grayscale density, 1 channel
output   = 1 channel
```

This makes BW the canonical scalar density target. ColorJet direct prediction is kept as an ablation experiment.

## Train

ColorJet:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/train_CVAE_densitymap_color.py --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_train_densitymap_color.json
```

BW:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/train_CVAE_densitymap_bw.py --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_train_densitymap_bw.json
```

BW v3 recovery experiment:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/train_CVAE_densitymap_bw.py --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_train_densitymap_bw_v3.json
```

The v3 config is intended for weak or blurry CVAE outputs. It trains with `train_latent_mode=zero` so training matches deterministic inference, then adds foreground-normalized L1, density-mass, and gamma-space L1 losses to reduce under-predicted density maps.

## Test

ColorJet:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/test_CVAE_densitymap_color.py --run_path AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name> --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_test_densitymap_color.json
```

BW:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/test_CVAE_densitymap_bw.py --run_path AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name> --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_test_densitymap_bw.json
```

Checkpoint modes:

- `best_mae`: selected by lowest validation MAE.
- `best_loss`: selected by lowest validation total loss.
- `final`: final epoch checkpoint.

Stochastic inference:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/test_CVAE_densitymap_bw.py --run_path AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name> --num_samples 4
```

## Output Structure

```text
AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name>/
checkpoints/best.pt
checkpoints/best_mae.pt
checkpoints/best_loss.pt
checkpoints/final.pt
logs/training_history.csv
samples/*.png
test_results/best_mae/
test_results/best_loss/
```

Each test result contains:

```text
predictions/
inputs/
targets/
error_maps/
test_evaluation_summary.csv
test_evaluation_summary.json
test_per_image_metrics.csv
test_scalar_density_summary.csv
```

For BW runs, the test script also publishes a Pix2PixHD-style final evaluation view at the run root:

```text
test_evaluation_summary.csv
test_evaluation_per_image.csv
test_results/inputs/
test_results/predictions/
test_results/targets/
```

The BW prediction and target images are rendered as ColorJet images in `test_results/predictions` and `test_results/targets`. The original grayscale density images are preserved with a `MASK_` prefix:

```text
test_results/predictions/plan_xxx.png       # ColorJet display image
test_results/predictions/MASK_plan_xxx.png  # original grayscale density
```

This mirrors the Pix2PixHD BW workflow. Pix2PixHD saves BW output as grayscale RGB first, then the UI/utility flow converts it to ColorJet while keeping `MASK_` backups. CVAE now does that conversion automatically when testing `best_mae`.

## Metrics

The density-map test reports:

- MAE
- MSE
- RMSE
- PSNR
- SSIM

For ColorJet, the test also writes `test_scalar_density_summary.csv`, which averages RGB channels into a scalar field. This is not a perfect inverse of ColorJet, but it helps compare the rough density structure.

## Estimated Training Time

With `image_size=256`, `batch_size=8`, and `epochs=50`:

| Device class | Estimated time per model |
|---|---:|
| RTX 4060 / 3060 class laptop GPU | 2-5 hours |
| RTX 4070 / 3080 / 4080 class GPU | 1-3 hours |
| CPU only | 20+ hours and not recommended |

Training both BW and ColorJet should take about twice the single-model time.

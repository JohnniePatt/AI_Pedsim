# CVAE Concept

This method now supports density-map image generation with two target representations:

1. `colorjet`: direct ColorJet density-map prediction.
2. `bw`: grayscale density-map prediction.

The legacy trajectory-line mask CVAE scripts were removed from this method folder so the method is now focused on density-map output only. Use these entrypoints:

- `train_CVAE_densitymap_color.py`
- `test_CVAE_densitymap_color.py`
- `train_CVAE_densitymap_bw.py`
- `test_CVAE_densitymap_bw.py`

## Core Idea

CVAE is used because density-map prediction can be one-to-many: the same environment layout can plausibly produce multiple pedestrian-density patterns. The latent variable `z` allows the model to represent that uncertainty.

```text
condition image A -> condition encoder
target image B    -> posterior encoder -> mu, logvar -> z
condition features + z -> decoder -> predicted density map
```

During deterministic inference the model uses `z=0`. During stochastic inference, `--num_samples N` samples multiple latent vectors and averages the predicted maps.

## Target Representation

The important experimental condition is the target representation, not only the architecture.

| Experiment | Target representation | Target channels | Meaning |
|---|---:|---:|---|
| CVAE ColorJet | `colorjet` | 3 | The model directly learns RGB ColorJet heatmap colors. |
| CVAE BW | `bw` | 1 | The model learns the scalar grayscale density field. |

This is intentionally different from the old Pix2PixHD observation where BW was still trained as 3-channel RGB. For the CVAE density-map version, BW is treated as the canonical scalar density output.

## Why BW Is the Main Target

ColorJet is visually useful, but its RGB values are not linearly related to the underlying density. Pixel-wise RGB losses may penalize color transitions rather than density error.

BW/grayscale is closer to the physical density field. Therefore:

```text
main learning target = BW density map
visualization target = ColorJet rendering
```

The direct ColorJet CVAE remains useful as an ablation experiment.

## Loss Design

The density-map CVAE uses:

- weighted L1 reconstruction
- optional weighted MSE reconstruction
- Sobel edge loss
- KL regularization with annealing
- optional foreground-normalized L1 loss
- optional density-mass loss
- optional gamma-space L1 loss

For BW, foreground and intensity weights are enabled by default:

```text
density_foreground_weight = 30.0
density_intensity_weight  = 10.0
```

## BW v3 Recovery Experiment

The first BW CVAE runs produced weak or blurry predictions. The likely cause is not only loss weight, but train/inference mismatch:

```text
training:  posterior encoder sees A + target B -> latent z
testing:   target B is unavailable -> latent z = 0
```

If the decoder learns to depend on target-informed `z`, test-time prediction can collapse toward an average density map. `config_train_densitymap_bw_v3.json` therefore trains with:

```text
train_latent_mode = zero
kl_weight         = 0.0
```

This keeps the CVAE-shaped architecture but makes the decoder learn the same condition it will receive during deterministic inference. The v3 config also adds foreground, mass, and gamma-space losses to reduce under-predicted density intensity.

For ColorJet, these weights default to zero because RGB color is not a direct scalar density field.

## Expected Training Time

Current dataset size:

```text
train      = 2603 images
validation = 439 images
test       = 862 images
image_size = 256
batch_size = 8
epochs     = 50
```

Estimated training time depends strongly on the GPU:

| Device class | Estimated time per model |
|---|---:|
| RTX 4060 / 3060 class laptop GPU | 2-5 hours |
| RTX 4070 / 3080 / 4080 class GPU | 1-3 hours |
| CPU only | not recommended, likely 20+ hours |

Training both BW and ColorJet should roughly double the time.

## Final Evaluation Display

BW is trained and evaluated as a 1-channel scalar density field. For visual comparison, the final evaluation view renders BW predictions and targets as ColorJet images while preserving the original grayscale density maps as `MASK_<filename>`.

This follows the Pix2PixHD BW convention:

```text
normal filename -> ColorJet display
MASK_filename   -> original grayscale density
```

The metric values are still computed on the scalar density field, not on the rendered ColorJet image.

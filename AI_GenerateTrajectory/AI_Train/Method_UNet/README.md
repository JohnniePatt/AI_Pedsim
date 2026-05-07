# Method_UNet

Segmentation-style U-Net for trajectory-line mask prediction.

## Install

WSL2 / Linux with NVIDIA GPU:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

The root `requirements.txt` is the main WSL project environment. It keeps the
PyTorch CUDA stack pinned for pix2pixHD and Method_UNet.

Apple Silicon / M-series Mac, including M5:

```bash
python3 -m venv AI_Pedsim-mac-env
source AI_Pedsim-mac-env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements_MAC.txt
```

Verify PyTorch and MPS detection:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS:", torch.backends.mps.is_available())
PY
```

If `MPS` is false, the script still runs on CPU. On Mac, do not install CUDA/cuDNN packages.

Verify PyTorch CUDA detection:

```bash
nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

If CUDA is false, the script still runs on CPU.

## Train

```bash
python AI_GenerateTrajectory/AI_Train/Method_UNet/train_unet_trajectory_mask.py --config AI_GenerateTrajectory/AI_Train/Method_UNet/config_train.json
```

Outputs are written to:

```text
AI_GenerateTrajectory/AI_Result/Method_UNet/outputs/run_UNet_<timestamp>/
```

Run contents:

```text
checkpoints/best.pt
checkpoints/best_dice.pt
checkpoints/best_loss.pt
checkpoints/final.pt
checkpoints/model_architecture.txt
logs/training_history.csv
progress.json
run_config_snapshot.json
test_results/best_dice/
test_results/best_loss/
```

## Test Existing Run

```bash
python AI_GenerateTrajectory/AI_Train/Method_UNet/test_unet_trajectory_mask.py --run_path AI_GenerateTrajectory/AI_Result/Method_UNet/outputs/<run_name>
```

The test script writes:

```text
test_results/<checkpoint_mode>/predictions/
test_results/<checkpoint_mode>/probability_maps/
test_results/<checkpoint_mode>/inputs/
test_results/<checkpoint_mode>/targets/
test_results/<checkpoint_mode>/test_evaluation_summary.csv
test_results/<checkpoint_mode>/test_per_image_metrics.csv
test_results/<checkpoint_mode>/test_threshold_metrics.csv
```

Checkpoint modes:

- `best_dice` is selected by highest validation Dice score.
- `best_loss` is selected by lowest validation loss.
- `final` is the last epoch checkpoint.

## Mask Notes

- `A` is loaded as RGB and resized with bilinear interpolation.
- `B` is loaded as 1-channel binary mask and resized with nearest-neighbor interpolation.
- Model output is 1-channel sigmoid probability.
- Predictions are saved as binary masks using `mask_threshold`.

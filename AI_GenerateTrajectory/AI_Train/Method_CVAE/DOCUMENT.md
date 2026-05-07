# Method_CVAE

PyTorch Conditional Variational Autoencoder (CVAE) for trajectory-line mask generation.

This method now uses the same WSL/NVIDIA PyTorch environment as pix2pixHD and Method_UNet.

## Dataset

Default dataset:

```text
Dataset/Data_ImageUNet/Trajectory_line_mask_dataset/Topo_HouseGAN
```

Expected structure:

```text
A/train, A/validation, A/test
B/train, B/validation, B/test
```

`A` is the scenario image. `B` is a binary trajectory-line mask.

## Train

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/train_cvae_trajectoryLine.py --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_train.json
```

## Test

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/test_cvae_trajectoryLine.py --run_path AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name> --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_test.json
```

Checkpoint modes:

- `best_dice` is selected by highest validation Dice score.
- `best_loss` is selected by lowest validation total loss.
- `final` is the last epoch checkpoint.

You can evaluate a specific mode:

```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/test_cvae_trajectoryLine.py --run_path AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name> --checkpoint_mode best_loss --output_name best_loss
```

## Output Structure

```text
AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name>/
checkpoints/best.pt
checkpoints/best_dice.pt
checkpoints/best_loss.pt
checkpoints/final.pt
logs/training_history.csv
samples/*.png
test_results/best_dice/
test_results/best_loss/
```

Each test result contains:

```text
predictions/
probability_maps/
inputs/
targets/
test_evaluation_summary.csv
test_per_image_metrics.csv
test_threshold_metrics.csv
```

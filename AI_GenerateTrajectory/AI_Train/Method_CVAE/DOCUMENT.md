# Method_CVAE

## Purpose
Conditional Variational Autoencoder (CVAE) for trajectory-line image generation from map-condition image pairs.

## Train
```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/train_cvae_trajectoryLine.py --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_train.json
```

## Test
```bash
python AI_GenerateTrajectory/AI_Train/Method_CVAE/test_cvae_trajectoryLine.py --run_path AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name> --config AI_GenerateTrajectory/AI_Train/Method_CVAE/config_test.json
```

## Output Structure
- `AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/<run_name>/checkpoints`
- `.../logs/training_history.csv`
- `.../samples/*.png`
- `.../test_results/{inputs,targets,predictions}`
- `.../test_evaluation_summary.csv`

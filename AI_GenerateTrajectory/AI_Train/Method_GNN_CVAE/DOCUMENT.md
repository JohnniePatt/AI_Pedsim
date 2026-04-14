# Method_GNN_CVAE

This method is a lighter multi-agent baseline than the transformer pipeline.

## What it does
- loads one simulation case as one training sample
- keeps all agents together
- encodes walkable geometry from room + corridor union
- builds per-frame social interaction with a custom distance-based GNN
- uses a CVAE latent per agent to represent walking style
- decodes the whole trajectory autoregressively across time

## Saved Outputs
Each training run stores:
- `logs/training_history.csv`
- `weights/latest_model.pth`
- `weights/best_model.pth`
- `weights/epoch_XXX.pth`
- `samples/epoch_XXX/AI_pred_*.parquet`
- `samples/epoch_XXX/GT_real_*.parquet`

## Metrics
The baseline logger writes:
- train loss
- validation loss
- ADE (metres)
- FDE (metres)
- collision rate
- out-of-bounds rate

## Notes
The first version intentionally avoids heavy external graph dependencies such as `torch_geometric`.
This keeps the environment simpler and makes iteration easier before we scale the model up.

## Step Layout
Scoped training stages now live in subdirectories:
- `Step_01_GoalOnly`
- `Step_01_5_GoalGeometry`
- `Step_02_GoalSocial`
- `Step_03_FullDataset`

Each step contains its own:
- `config_train.json`
- `config_test.json`
- `train_gnn_cvae.py`
- `test_gnn_cvae.py`

Those step scripts are lightweight wrappers around the shared core files in the method root.

Recommended order:
1. `Step_01_GoalOnly`
2. `Step_01_5_GoalGeometry`
3. `Step_02_GoalSocial`
4. `Step_03_FullDataset`

# AI README

This file is a compact handoff note for future sessions working on `AI_GenerateImage/AI_Train/Method_GNN_CVAE`.

## Current Training Story

The method is being treated as a curriculum, not a single one-shot training run.

Current intended order:
1. `Step_01_GoalOnly`
2. `Step_01_5_GoalGeometry`
3. `Step_02_GoalSocial`
4. `Step_03_FullDataset`

The idea is to teach the model the basics in layers:
- first learn how to move from start to goal
- then learn how to stay inside the walkable area
- then add social interaction
- finally scale up to the full dataset

## Why Step_01_5 Was Added

There was a gap between `GoalOnly` and `GoalSocial`.

The missing skill was geometry understanding:
- which space is walkable
- which space is wall / hole / outside map
- how to avoid drifting through walls or leaving the map

Without this intermediate step, the model can start using social context before it has learned the simpler rule that valid trajectories should stay inside the walkable area.

## Important Discovery About GT

The ground-truth parquet exported for visuals is not the full raw trajectory.

In the shared dataset loader:
- frames are downsampled by `frame_stride`
- then truncated by `max_seq_len`

So `GT_real_*.parquet` inside run samples is the truncated training tensor, not the original full raw trajectory from the dataset folder.

Example found during inspection:
- raw case `100007` had 3859 frames
- exported GT used only 160 frames
- with `frame_stride=8`, the exported GT only covered frames `0..1272`

This explains why GT plots may stop well before the goal star.

## Important Discovery About The Goal Star

The goal star shown in visuals is the centroid of `exit_area`.

Raw trajectories often disappear once agents reach the exit boundary or enter the exit zone. That means even the raw GT may stop near the exit, not exactly at the centroid star.

So there are two different effects:
- GT is truncated by dataset settings
- even full raw GT may end at the exit boundary, not at the centroid

## How The Shared Core Works

Shared files:
- `dataset.py`
- `model.py`
- `train_gnn_cvae.py`
- `test_gnn_cvae.py`
- `visual_gnn_cvae.py`

Step folders are wrappers plus step-specific configs.

Each step changes mostly:
- number of agents
- amount of data
- social on/off
- sequence length
- loss emphasis
- whether to resume from a previous checkpoint

## Step Intent

### Step_01_GoalOnly
- `max_agents = 1`
- `use_social = false`
- purpose: learn basic start-to-goal trajectory generation

### Step_01_5_GoalGeometry
- `max_agents = 4`
- `use_social = false`
- lower LR than step 1
- much stronger `oob_weight`
- extra `segment_oob_weight` to punish wall-cutting between frames
- smaller `max_residual` to reduce free-form detours through obstacles
- purpose: learn to stay in walkable corridors before adding social complexity

### Step_02_GoalSocial
- multi-agent
- `use_social = true`
- purpose: preserve goal-following while reacting to nearby agents

### Step_03_FullDataset
- full-scale version of step 2
- more agents
- more data
- longer sequences

## Resume Strategy

The intended training chain is:
- best checkpoint from step 1 -> step 1.5
- best checkpoint from step 1.5 -> step 2
- best checkpoint from step 2 -> step 3

This is curriculum fine-tuning, not isolated training.

The `resume_checkpoint` value in `Step_01_5_GoalGeometry/config_train.json` is a placeholder and should be replaced with the actual `best_model.pth` path from the finished step 1 run before training.

Recommended pattern for later steps is the same.

## What To Watch During Training

For geometry-focused training, the most important signs are:
- `out_of_bounds_rate` should drop clearly
- prediction should stay inside walkable corridors
- trajectories should not cut through walls
- FDE should remain reasonable while geometry improves

If `oob_weight` is too low:
- the model may still drift through walls

If `oob_weight` is too high:
- the model may become too conservative and hurt trajectory quality

## Known Conceptual Caveat

This model behaves more like full-trajectory conditional generation than classic forecasting from a short observed prefix.

Why:
- one sample contains a whole scene
- the encoder uses trajectory statistics from the whole stored sequence
- the decoder generates a full sequence of the same stored length

So `obs_len` is much less central than in standard trajectory-prediction setups.

## Files Added In This Session

- `Step_01_5_GoalGeometry/train_gnn_cvae.py`
- `Step_01_5_GoalGeometry/test_gnn_cvae.py`
- `Step_01_5_GoalGeometry/config_train.json`
- `Step_01_5_GoalGeometry/config_test.json`
- `AI_README.md`

## If You Continue Later

Best next actions:
1. replace the placeholder resume path in step 1.5 with the real best checkpoint from step 1
2. train step 1.5 and inspect `out_of_bounds_rate`
3. if geometry is still weak, raise `oob_weight` or reduce trajectory freedom further
4. only move to step 2 once walkable-area behavior is stable

# Method_GridSocialPolicy README_v01

## Purpose

`Method_GridSocialPolicy` is the first grid-based behavior-cloning baseline for `AI_GenerateTrajectoryGrid`.

Goal: generate pedestrian trajectories from only:

- walkable grid map
- frame-0 spawn positions of all agents
- exit room trigger polygon

The model predicts agent actions on a grid rollout, while rule logic enforces walkable cells, collision rejection, and exit-room stopping.

## AI Architecture

Model class: `GridSocialPolicyNet`

Source:

```text
AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy/model_grid_policy.py
```

Input has 2 branches.

### Map Encoder

Input tensor shape:

```text
[batch, 3, crop_size, crop_size]
```

Channels:

```text
0 = local walkable grid crop
1 = local exit-room / goal crop
2 = local occupancy crop from other active agents
```

Network:

```text
ConvBlock(3 -> base_channels)
MaxPool2d
ConvBlock(base_channels -> base_channels * 2)
MaxPool2d
ConvBlock(base_channels * 2 -> base_channels * 4)
AdaptiveAvgPool2d(1)
Flatten
```

Default:

```text
base_channels = 32
crop_size = 33
```

### Agent Feature Encoder

Scalar feature vector size:

```text
feature_dim = 8
```

Features:

```text
0 = normalized grid_x
1 = normalized grid_y
2 = normalized dx to exit centroid
3 = normalized dy to exit centroid
4 = normalized distance to exit centroid
5 = nearest-agent distance in local crop
6 = active-agent count normalized by 250
7 = is_walkable_cell
```

Network:

```text
Linear(8 -> hidden_dim)
ReLU
Dropout
Linear(hidden_dim -> hidden_dim)
ReLU
```

Default:

```text
hidden_dim = 128
dropout = 0.1
```

### Fusion And Heads

Fusion:

```text
concat(map_feature, agent_feature)
Linear(map_dim + hidden_dim -> hidden_dim)
ReLU
Dropout
Linear(hidden_dim -> hidden_dim)
ReLU
```

Outputs:

```text
policy_head -> action_logits over movement actions + wait
stop_head   -> stop logit
```

Important: `stop` is not part of the action class. It is a separate binary head.

## Action Space

Source:

```text
AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy/action_space.py
```

Current action space:

```text
20 movement actions
1 wait action
1 separate stop head
```

Action space is built from training data by scanning trajectory grid deltas.

Current config:

```text
movement_action_count = 20
action_frame_stride = 5
```

`action_frame_stride=5` means the label action is computed from the grid delta between frame `t` and frame `t+5`, not `t+1`.

Reason: source data is 25 FPS. Per-frame grid movement creates too many `wait` labels. Stride 5 makes action labels represent a real movement step better.

## Learning Principle

Training style:

```text
Behavior Cloning / Supervised Learning
```

For each sampled trajectory row:

```text
Input:
  local map crop
  local goal crop
  local occupancy crop
  scalar agent/goal/social features

Target:
  action_id from future grid delta
  stop_target from end-of-agent trajectory
```

The model learns to imitate Social Force simulation trajectories after they are normalized into grid actions.

Current rollout is autoregressive:

```text
frame 0 spawn -> predict action -> update grid position -> repeat
```

At rollout time:

```text
1. model predicts action logits and stop logit
2. if stop probability >= stop_threshold, agent stops
3. if agent center is inside exit room polygon, agent stops
4. proposed move is rejected if outside walkable grid
5. proposed move is rejected if multiple agents choose same cell
```

## Loss Function

Source:

```text
AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy/train_grid_policy.py
```

Total loss:

```text
total_loss = action_loss + stop_loss_weight * stop_loss
```

Action loss:

```text
CrossEntropyLoss(action_logits, action_target)
```

Stop loss:

```text
BCEWithLogitsLoss(stop_logit, stop_target)
```

Current class weighting:

```text
move_loss_weight = 1.0
wait_loss_weight = 0.2
```

Reason: `wait` labels are still frequent even with stride. Lowering wait weight reduces the model's tendency to learn "always wait".

Current optimizer:

```text
optimizer = adamw
learning_rate = 0.0003
weight_decay = 0.0001
lr scheduler = ReduceLROnPlateau
gradient clipping = 1.0
```

## Desired Outcome

The wanted rollout behavior:

```text
1. all agents start from frame-0 spawn positions
2. agents move only on walkable grid cells
3. agents avoid occupying the same cell
4. agents follow learned Social Force-like motion patterns
5. agents navigate from start room to exit room
6. agents stop when they enter the exit room trigger polygon
```

Primary evaluation should not rely only on train/val loss.

Important rollout metrics:

```text
moving_agents
movement_steps
wait_steps
move_decisions
stopped_agents
collision_count
blocked_by_wall_steps
blocked_by_collision_steps
walkable_ratio
mean_path_cells
max_path_cells
```

Recommended future metrics:

```text
reached_exit_ratio
mean_final_distance_to_exit
escaped_spawn_room_ratio
path_length_error_vs_ground_truth
```

## Current Known Limitation

This v0.1 model is still a local policy.

It sees:

```text
local crop around agent
exit centroid direction
local occupancy
nearest-agent distance
```

It does not yet have:

```text
global shortest path field
distance-to-exit map
next-step direction-to-exit map
graph/path planning guidance
agent attention/GNN
rollout-level loss
```

Observed failure:

```text
In large rooms with many agents, the model can move locally but fail to exit the spawn room.
It may wait too much, loop inside the start room, or cluster near obstacles/doors.
```

Likely next improvement:

```text
Add BFS/A* distance-to-exit and direction-to-exit channels to the map input.
```

## Result Folder Structure

Root:

```text
AI_GenerateTrajectoryGrid/AI_Result/Method_GridSocialPolicy/
```

Contains one folder per training run:

```text
run_YYYYMMDD_HHMMSS/
```

### Run Folder

Example:

```text
run_20260507_004920/
  action_space.json
  config_train.json
  metrics.csv
  model_architecture.txt
  checkpoints/
  rollouts/
```

### `config_train.json`

Stores the exact config used for the run.

Important fields:

```text
dataset_root
epochs
batch_size
optimizer
learning_rate
weight_decay
crop_size
movement_action_count
action_frame_stride
move_loss_weight
wait_loss_weight
base_channels
hidden_dim
dropout
```

Use this to reproduce or compare a run.

### `action_space.json`

Stores action ids used by the checkpoint.

Important because model outputs action ids, and ids must map back to `(dx, dy)`.

Never mix a checkpoint with another run's action space.

### `metrics.csv`

Training/validation metrics per epoch.

Columns:

```text
epoch
train_loss
train_action_loss
train_stop_loss
train_action_acc
train_stop_acc
val_loss
val_action_loss
val_stop_loss
val_action_acc
val_stop_acc
lr
```

Note: lower validation loss does not guarantee good rollout. Autoregressive rollout must be checked separately.

### `model_architecture.txt`

Text dump of the PyTorch model structure.

Useful for quick architecture comparison between runs.

### `checkpoints/`

```text
best.pt
last.pt
```

`best.pt`:

```text
checkpoint with lowest validation loss
```

`last.pt`:

```text
latest epoch checkpoint
```

Checkpoint includes:

```text
model_state_dict
optimizer_state_dict
epoch
config
action_space
best_val_loss
```

### `rollouts/`

Contains rollout outputs for one or more test cases.

Single rollout folder:

```text
rollouts/<split>_<plan_name>_<sqlite_stem>/
  rollout.parquet
  action_trace.parquet
  summary.json
  samples/
    rollout_preview.png
```

Batch summary:

```text
rollouts/batch_rollout_<split>.csv
rollouts/batch_rollout_<split>.json
```

### `rollout.parquet`

AI-generated trajectory table.

Typical columns:

```text
frame
agent_id
pos_x
pos_y
grid_x
grid_y
grid_row
stopped
```

This is the AI output path in grid/world coordinates.

### `action_trace.parquet`

Per-agent decision log.

Typical columns:

```text
frame
agent_id
action_id
action_name
action_kind
dx
dy
stop_prob
proposed_grid_x
proposed_grid_y
proposed_grid_row
blocked_by_wall
blocked_by_collision
accepted
```

Use this to debug why agents wait, collide, get blocked, or fail to reach exit.

### `summary.json`

Rollout-level metrics.

Important fields:

```text
frames
agents
rows
walkable_ratio
collision_count
stopped_agents
moving_agents
movement_steps
mean_path_cells
max_path_cells
wait_steps
move_decisions
blocked_by_wall_steps
blocked_by_collision_steps
action_counts
sample_preview
```

Use this for fast case review before opening plots.

### `samples/rollout_preview.png`

Saved visualization of AI rollout.

Shows:

```text
walkable grid
exit room
spawn points
AI predicted paths
end points
```

## Dataset Dependency

This method expects prepared dataset:

```text
Dataset/Data_TrajectoryGrid/Topo_HouseGAN/
```

Expected structure:

```text
A/<split>/<plan>/<sqlite_stem>/
  walkablearea_grid.json
  spawn_agent.parquet
  exit_room.json

B/<split>/<plan>/<sqlite_stem>/
  trajectory.parquet

metadata/<split>/<plan>/<sqlite_stem>.json
manifest_trajectory_grid.csv
```

`A` is input.

`B` is training target.

Raw comparison images:

```text
Geo_scenario/Topo_HouseGAN/trajectory_line/<plan>/trajectory_<case>.png
```

Streamlit result comparison uses:

```text
Ground truth raw       -> trajectory_line image
Ground truth on grid   -> B trajectory plotted on A grid
AI rollout output      -> rollout preview or rollout.parquet
```

## Next Recommended Direction

Next model/data upgrade should prioritize navigation guidance:

```text
1. Add distance_to_exit_grid channel.
2. Add next_step_direction_to_exit channels from BFS/A*.
3. Train policy with these channels.
4. Evaluate reached_exit_ratio and final_distance_to_exit.
5. After navigation works, improve social collision behavior with attention/GNN.
```


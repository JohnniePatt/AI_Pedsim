# Method_GridSocialPolicy_SF_01

> Social-Force-conditioned discrete policy ที่แยกจาก
> `Method_GridSocialPolicy` เดิมอย่างสมบูรณ์ ข้อมูล policy เพิ่ม explicit
> agent-repulsion และ wall-repulsion vectors ร่วมกับ goal, occupancy และ
> walkability map; training และ synchronous full rollout ใช้ feature contract
> 12 มิติเดียวกัน และเรียกผ่าน `run_pipeline.py`

`GridSocialPolicyNet` is the first trainable baseline for `AI_GenerateTrajectoryGrid`.

It is a behavior-cloning policy model:

```text
input  = local walkable grid crop + exit-room crop + local agent occupancy crop + scalar agent/goal features
output = movement action logits + separate stop logit
```

The action space is learned from the prepared dataset. The utility scans train trajectories,
counts grid deltas, keeps the most common movement offsets, then adds one `wait` action.
`stop` is not an action class; it is trained as a separate head.

## Files

```text
action_space.py            Build/load movement action space.
dataset_grid_policy.py     Load A/B prepared cases and create training samples.
model_grid_policy.py       CNN + MLP policy model.
train_grid_policy.py       Training loop and checkpoint writer.
rollout.py                 Run a trained checkpoint from frame 0 spawn.
rollout_batch.py           Run and save 1-20 rollout preview samples from manifest cases.
metrics.py                 Basic rollout summary.
config_train.json          Default training config.
```

## Train

Quick smoke test:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 train_grid_policy.py --config config_smoke.json
```

Full baseline training:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 train_grid_policy.py --config config_train.json
```

Outputs go to:

```text
AI_GenerateTrajectoryGrid/AI_Result/Method_GridSocialPolicy/run_YYYYMMDD_HHMMSS/
  action_space.json
  config_train.json
  metrics.csv
  model_architecture.txt
  checkpoints/
    best.pt
    last.pt
```

## Rollout

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 rollout.py \
  --checkpoint /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Result/Method_GridSocialPolicy/<run>/checkpoints/best.pt \
  --input-dir /home/johnfaqpc/programming/AI_Pedsim/Dataset/Data_TrajectoryGrid/Topo_HouseGAN/A/train/<plan>/<sqlite_stem> \
  --output-dir /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Result/Method_GridSocialPolicy/<run>/rollout_sample
```

Each rollout writes:

```text
rollout.parquet
action_trace.parquet
summary.json
samples/rollout_preview.png
```

For multiple preview samples:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 rollout_batch.py \
  --checkpoint /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Result/Method_GridSocialPolicy/<run>/checkpoints/best.pt \
  --dataset-root /home/johnfaqpc/programming/AI_Pedsim/Dataset/Data_TrajectoryGrid/Topo_HouseGAN \
  --output-root /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Result/Method_GridSocialPolicy/<run>/rollouts \
  --split val \
  --sample-count 10
```

## Current Scope

This is v0. It includes local social context through an occupancy crop and nearest-agent
feature. A later v1 can replace or augment that with GAT/Transformer attention over agents.

## Data Loading Notes

Training is intentionally case-blocked for speed. `dataloader_shuffle` defaults to `false`;
the dataset shuffles case order and samples inside each case instead. This keeps parquet
reads/cache hot and avoids starving the GPU with random cross-case file loading.

The default full config uses:

```text
action_frame_stride = 5
wait_loss_weight = 0.2
batch_size = 1024
optimizer = adamw
num_workers = 4
pin_memory = true
```

`action_frame_stride=5` trains one action from a 5-frame delta instead of a 1-frame
delta. This reduces the 25 FPS wait-label problem and gives the 20 movement actions
real speed/direction meaning.

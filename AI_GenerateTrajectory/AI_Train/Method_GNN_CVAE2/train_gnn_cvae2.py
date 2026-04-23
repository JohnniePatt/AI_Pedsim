"""
train_gnn_cvae2.py
------------------
Train GNNCVAE2 on HouseGAN multi-agent simulation trajectories.

Spatial graphs vary per case, so each batch is processed sample-by-sample
with gradient accumulation over the whole batch before each optimiser step.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimulationTrajectoryDataset, collate_fn
from model import GNNCVAE2
from prepare_geometry_gnn_cvae2 import grid_to_world, point_is_inside_walkable


# ── utilities ────────────────────────────────────────────────────────────────

def build_optimizer(model, cfg):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr   = cfg.get("lr", 3e-4)
    wd   = cfg.get("weight_decay", 1e-4)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def get_next_run_dir(base: pathlib.Path) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    idx  = max(nums) + 1 if nums else 1
    return base / f"run_{idx}"


def get_kl_weight(epoch: int, cfg: dict) -> float:
    target = cfg.get("kl_weight", 0.01)
    anneal = cfg.get("kl_anneal_epochs", 20)
    return min(target, target * epoch / max(anneal, 1))


def graph_to_device(nf, ei, et, device):
    return nf.to(device), ei.to(device), et.to(device)


# ── per-sample forward (handles variable-size graphs) ───────────────────────

def forward_sample(model, sample_idx: int, batch: dict, device, cfg, epoch: int, train: bool):
    """Run model forward on one sample from the batch. Returns loss scalar."""
    B_agents = batch["positions"].shape[1]

    positions  = batch["positions"][sample_idx:sample_idx+1].to(device)   # [1, N, T, 2]
    agent_mask = batch["agent_mask"][sample_idx:sample_idx+1].to(device)
    start_pt   = batch["start_pt"][sample_idx:sample_idx+1].to(device)
    goal_pt    = batch["goal_pt"][sample_idx:sample_idx+1].to(device)
    geo_mask   = batch["geo_mask"][sample_idx:sample_idx+1].to(device)

    nf = batch["node_features_list"][sample_idx].to(device)
    ei = batch["edge_index_list"][sample_idx].to(device)
    et = batch["edge_type_list"][sample_idx].to(device)
    ni = batch["agent_node_ids_list"][sample_idx].unsqueeze(0).to(device)  # [1, N_agents]

    # Pad agent_node_ids to match padded batch dimension
    na = ni.shape[1]
    if na < B_agents:
        ni_padded = torch.zeros(1, B_agents, dtype=torch.long, device=device)
        ni_padded[0, :na] = ni[0]
        ni = ni_padded

    kl_w      = get_kl_weight(epoch, cfg)
    goal_w    = cfg.get("goal_weight", 1.0)
    oob_warmup = cfg.get("oob_warmup_epochs", 0)
    oob_w     = 0.0 if epoch <= oob_warmup else cfg.get("oob_weight", 1.0)
    seg_oob_w = 0.0 if epoch <= oob_warmup else cfg.get("segment_oob_weight", 0.5)

    out = model(
        positions, agent_mask, start_pt, goal_pt, geo_mask,
        nf, ei, et, ni,
        sample_latent=train,
        goal_weight=goal_w,
        kl_weight=kl_w,
        oob_weight=oob_w,
        segment_oob_weight=seg_oob_w,
        teacher_forcing=train,
    )
    return out


# ── train / validate ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None

    for batch in tqdm(loader, desc="  Train", leave=False):
        B = batch["positions"].shape[0]
        optimizer.zero_grad()
        batch_loss = 0.0

        for i in range(B):
            with autocast("cuda", enabled=use_amp):
                out = forward_sample(model, i, batch, device, cfg, epoch, train=True)
            loss_i = out["loss"] / B
            if use_amp:
                scaler.scale(loss_i).backward()
            else:
                loss_i.backward()
            batch_loss += out["loss"].item()

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += batch_loss / B
        n     += 1

    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, cfg, epoch):
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  Val  ", leave=False):
        B = batch["positions"].shape[0]
        for i in range(B):
            out = forward_sample(model, i, batch, device, cfg, epoch, train=False)
            total += out["loss"].item()
        n += B
    return total / max(n, 1)


# ── metrics / export ─────────────────────────────────────────────────────────

def denorm_positions(norm: np.ndarray, meta: dict) -> np.ndarray:
    world = np.zeros_like(norm)
    for a in range(norm.shape[0]):
        for t in range(norm.shape[1]):
            world[a, t] = grid_to_world(norm[a, t, 0], norm[a, t, 1], meta)
    return world


def compute_scene_metrics(pred_world, gt_world, mask, walkable, collision_threshold_m, exit_centroid_world=None):
    valid_steps = mask[:, 1:]
    pred_future = pred_world[:, 1:]
    gt_future   = gt_world[:, 1:]

    dists, final_dists = [], []
    outside_count = point_count = collision_pairs = total_pairs = 0

    for a in range(mask.shape[0]):
        vi = np.where(valid_steps[a])[0]
        if len(vi) == 0:
            continue
        d = np.linalg.norm(pred_future[a, vi] - gt_future[a, vi], axis=1)
        dists.extend(d.tolist())
        last_pred = pred_future[a, vi[-1]]
        final_dists.append(
            float(np.linalg.norm(last_pred - exit_centroid_world)) if exit_centroid_world is not None
            else float(d[-1])
        )
        for t in vi.tolist():
            x, y = pred_future[a, t]
            outside_count += 0 if point_is_inside_walkable(x, y, walkable) else 1
            point_count   += 1

    for t in range(1, mask.shape[1]):
        act = np.where(mask[:, t])[0]
        if len(act) < 2:
            continue
        pts = pred_world[act, t]
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                total_pairs += 1
                if np.linalg.norm(pts[i] - pts[j]) < collision_threshold_m:
                    collision_pairs += 1

    ade   = float(np.mean(dists))       if dists       else 0.0
    fde   = float(np.mean(final_dists)) if final_dists else 0.0
    col_r = float(collision_pairs / max(total_pairs, 1))
    oob_r = float(outside_count  / max(point_count,  1))
    return ade, fde, col_r, oob_r


def export_scene_parquet(path, positions_world, mask, agent_ids, frames):
    rows = []
    for a, aid in enumerate(agent_ids.tolist()):
        if aid < 0:
            continue
        for t, frame in enumerate(frames.tolist()):
            if t >= mask.shape[1] or not mask[a, t]:
                continue
            rows.append({"frame": int(frame), "id": int(aid),
                         "pos_x": float(positions_world[a, t, 0]),
                         "pos_y": float(positions_world[a, t, 1])})
    pd.DataFrame(rows).to_parquet(path)


@torch.no_grad()
def run_epoch_report(model, val_dataset, report_dir, epoch, device, cfg):
    model.eval()
    s       = val_dataset.samples[0]
    meta    = s["meta"]
    walkable = s["walkable"]
    case_id = s["case_id"]
    case_dir = pathlib.Path(s["case_dir"])

    positions  = s["positions"].unsqueeze(0).to(device)
    agent_mask = s["agent_mask"].unsqueeze(0).to(device)
    start_pt   = s["start_pt"].unsqueeze(0).to(device)
    goal_pt    = s["goal_pt"].unsqueeze(0).to(device)
    geo_mask   = s["geo_mask"].unsqueeze(0).to(device)
    nf = s["node_features"].to(device)
    ei = s["edge_index"].to(device)
    et = s["edge_type"].to(device)
    ni = s["agent_node_ids"].unsqueeze(0).to(device)

    out = model(
        positions, agent_mask, start_pt, goal_pt, geo_mask,
        nf, ei, et, ni,
        sample_latent=False, teacher_forcing=False,
        goal_weight=cfg.get("goal_weight", 1.0),
        kl_weight=get_kl_weight(epoch, cfg),
        oob_weight=cfg.get("oob_weight", 1.0),
        segment_oob_weight=cfg.get("segment_oob_weight", 0.5),
    )

    pred_norm  = out["positions"][0].cpu().numpy()
    gt_norm    = s["positions"].numpy()
    mask_np    = s["agent_mask"].numpy()
    agent_ids  = s["agent_ids"].numpy()
    frames     = s["frames"].numpy()

    pred_world = denorm_positions(pred_norm, meta)
    gt_world   = denorm_positions(gt_norm,   meta)

    goal_n = s["goal_pt"][0].numpy()
    exit_centroid_world = np.array(grid_to_world(float(goal_n[0]), float(goal_n[1]), meta), dtype=np.float32)

    ade, fde, col_r, oob_r = compute_scene_metrics(
        pred_world, gt_world, mask_np, walkable,
        cfg.get("collision_threshold_m", 0.4), exit_centroid_world,
    )

    epoch_dir = report_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    export_scene_parquet(epoch_dir / f"AI_pred_{case_id}.parquet", pred_world, mask_np, agent_ids, frames)
    export_scene_parquet(epoch_dir / f"GT_real_{case_id}.parquet", gt_world,   mask_np, agent_ids, frames)

    for fname in ["Geo_room.json", "Geo_corridor.json",
                  f"Spawn_location_{case_id}.csv", f"Spawn_exit_{case_id}.csv"]:
        src = case_dir / fname
        if src.exists():
            shutil.copy(src, epoch_dir / fname)

    return ade, fde, col_r, oob_r


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str((cfg_file.parent / dataset_path).resolve())

    project_root = pathlib.Path(__file__).resolve().parents[3]
    run_dir      = get_next_run_dir(project_root / "AI_GenerateTrajectory" / "AI_Result" / "Method_GNN_CVAE2" / "outputs")
    logs_dir     = run_dir / "logs"
    weights_dir  = run_dir / "weights"
    samples_dir  = run_dir / "samples"
    for p in [logs_dir, weights_dir, samples_dir]:
        p.mkdir(parents=True, exist_ok=True)

    shutil.copy(cfg_file, run_dir / "config_train.json")
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump({"config": str(cfg_file), "scope_name": cfg.get("scope_name", "")}, f, indent=2)

    print(f"[Train] Run:    {run_dir}")
    print(f"[Train] Config: {cfg_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    ds_kw = dict(
        obs_len=cfg.get("obs_len", 1),
        frame_stride=cfg.get("frame_stride", 8),
        max_seq_len=cfg.get("max_seq_len", 160),
        grid_size=cfg.get("grid_size", 64),
        geo_padding=cfg.get("geo_padding", 1.0),
        max_agents=cfg.get("max_agents", 64),
        subset_percent=cfg.get("subset_percent", 100.0),
        data_percent=cfg.get("data_percent"),
    )
    train_ds = SimulationTrajectoryDataset(dataset_path, split="train", **ds_kw)
    val_ds   = SimulationTrajectoryDataset(dataset_path, split="val",   **ds_kw)
    if not train_ds or not val_ds:
        raise RuntimeError("Dataset is empty — check dataset_path.")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 1), shuffle=True,
                              num_workers=cfg.get("num_workers", 0), collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.get("batch_size", 1), shuffle=False,
                              num_workers=cfg.get("num_workers", 0), collate_fn=collate_fn)

    model = GNNCVAE2(
        hidden_dim=cfg.get("hidden_dim", 128),
        latent_dim=cfg.get("latent_dim", 32),
        node_feat_dim=cfg.get("node_feat_dim", 8),
        node_embed_dim=cfg.get("node_embed_dim", 64),
        spatial_gnn_layers=cfg.get("spatial_gnn_layers", 2),
        agent_gnn_layers=cfg.get("agent_gnn_layers", 3),
        neighbor_radius=cfg.get("neighbor_radius", 0.08),
        dropout=cfg.get("dropout", 0.1),
        max_residual=cfg.get("max_residual", 0.25),
        segment_samples=cfg.get("segment_samples", 5),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {total_params:,}")

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    resume = cfg.get("resume_checkpoint")
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"[Train] Resumed from {resume}")

    scaler = None  # Disable AMP for now (dtype issues with spatial GNN in autocast)
    epochs     = cfg.get("epochs", 60)
    save_every = cfg.get("save_every", 10)
    best_val   = float("inf")

    history_file = logs_dir / "training_history.csv"
    progress_file = run_dir / "progress.json"
    with open(history_file, "w") as f:
        f.write("epoch,train_loss,val_loss,ade_m,fde_m,collision_rate,out_of_bounds_rate\n")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch:3d}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val_loss   = validate(model, val_loader, device, cfg, epoch)
        scheduler.step(val_loss)
        ade, fde, col_r, oob_r = run_epoch_report(model, val_ds, samples_dir, epoch, device, cfg)

        print(f"  train={train_loss:.5f}  val={val_loss:.5f}  "
              f"ADE={ade:.3f}m  FDE={fde:.3f}m  coll={col_r:.4f}  oob={oob_r:.4f}")

        with open(history_file, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{ade:.4f},{fde:.4f},{col_r:.6f},{oob_r:.6f}\n")

        with open(progress_file, "w") as f:
            json.dump({"epoch": epoch, "total_epochs": epochs,
                       "percentage": int(epoch / epochs * 100), "loss": val_loss}, f)

        torch.save(model.state_dict(), weights_dir / "latest_model.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), weights_dir / "best_model.pth")
            print(f"  [*] Best saved (val_loss={best_val:.5f})")
        if epoch % save_every == 0:
            torch.save(model.state_dict(), weights_dir / f"epoch_{epoch:03d}.pth")

    print(f"\n[Train] Done — {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_train.json")
    args = parser.parse_args()
    main(args.config)

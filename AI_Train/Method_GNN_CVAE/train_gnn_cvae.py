"""
train_gnn_cvae.py
-----------------
Train GNNCVAE on full multi-agent simulation trajectories.
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
from model import GNNCVAE
from prepare_geometry_gnn_cvae import grid_to_world, point_is_inside_walkable


def build_optimizer(model, cfg):
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adam' or 'adamw'.")


def get_next_run_dir(base: pathlib.Path) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    idx = max(nums) + 1 if nums else 1
    return base / f"run_{idx}"


def batch_to_device(batch: dict, device: torch.device) -> dict:
    keys = ["positions", "agent_mask", "start_pt", "goal_pt", "geo_mask", "agent_ids"]
    result = {k: batch[k].to(device) for k in keys if k in batch}
    for key in ["case_ids", "case_dirs", "metas", "walkables", "frames"]:
        result[key] = batch[key]
    return result


def get_kl_weight(epoch: int, cfg: dict) -> float:
    """Linear KL annealing: ramp from 0 to kl_weight over kl_anneal_epochs."""
    target       = cfg.get("kl_weight", 0.01)
    anneal_epochs = cfg.get("kl_anneal_epochs", 20)
    return min(target, target * epoch / max(anneal_epochs, 1))


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch: int):
    model.train()
    total_loss, n = 0.0, 0
    use_amp  = scaler is not None
    kl_w     = get_kl_weight(epoch, cfg)
    goal_w   = cfg.get("goal_weight", 1.0)
    oob_w    = cfg.get("oob_weight", 0.5)

    for batch in tqdm(loader, desc="  Train", leave=False):
        b = batch_to_device(batch, device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            out = model(
                positions       = b["positions"],
                agent_mask      = b["agent_mask"],
                start_pt        = b["start_pt"],
                goal_pt         = b["goal_pt"],
                geo_mask        = b["geo_mask"],
                goal_weight     = goal_w,
                kl_weight       = kl_w,
                oob_weight      = oob_w,
                teacher_forcing = True,
            )
        loss = out["loss"]
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, cfg, epoch: int):
    model.eval()
    total_loss, n = 0.0, 0
    use_amp = device.type == "cuda"
    kl_w    = get_kl_weight(epoch, cfg)
    goal_w  = cfg.get("goal_weight", 1.0)
    oob_w   = cfg.get("oob_weight", 0.5)

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        b = batch_to_device(batch, device)
        with autocast("cuda", enabled=use_amp):
            out = model(
                positions       = b["positions"],
                agent_mask      = b["agent_mask"],
                start_pt        = b["start_pt"],
                goal_pt         = b["goal_pt"],
                geo_mask        = b["geo_mask"],
                goal_weight     = goal_w,
                kl_weight       = kl_w,
                oob_weight      = oob_w,
                teacher_forcing = True,
                sample_latent   = False,
            )
        total_loss += out["loss"].item()
        n += 1
    return total_loss / max(n, 1)


def denorm_positions(norm_positions: np.ndarray, meta: dict) -> np.ndarray:
    world = np.zeros_like(norm_positions)
    for aidx in range(norm_positions.shape[0]):
        for tidx in range(norm_positions.shape[1]):
            world[aidx, tidx] = grid_to_world(norm_positions[aidx, tidx, 0], norm_positions[aidx, tidx, 1], meta)
    return world


def compute_scene_metrics(pred_world: np.ndarray, gt_world: np.ndarray, mask: np.ndarray, walkable, collision_threshold_m: float):
    valid_steps = mask[:, 1:]
    pred_future = pred_world[:, 1:]
    gt_future = gt_world[:, 1:]

    dists = []
    final_dists = []
    outside_count = 0
    point_count = 0
    collision_pairs = 0
    total_pairs = 0

    n_agents, t_future = valid_steps.shape
    for aidx in range(n_agents):
        valid_idx = np.where(valid_steps[aidx])[0]
        if len(valid_idx) == 0:
            continue
        agent_dists = np.linalg.norm(pred_future[aidx, valid_idx] - gt_future[aidx, valid_idx], axis=1)
        dists.extend(agent_dists.tolist())
        final_dists.append(float(agent_dists[-1]))
        for tidx in valid_idx.tolist():
            x, y = pred_future[aidx, tidx]
            outside_count += 0 if point_is_inside_walkable(x, y, walkable) else 1
            point_count += 1

    for tidx in range(1, mask.shape[1]):
        active_idx = np.where(mask[:, tidx])[0]
        if len(active_idx) < 2:
            continue
        pts = pred_world[active_idx, tidx]
        for i in range(len(active_idx)):
            for j in range(i + 1, len(active_idx)):
                dist = float(np.linalg.norm(pts[i] - pts[j]))
                total_pairs += 1
                if dist < collision_threshold_m:
                    collision_pairs += 1

    ade = float(np.mean(dists)) if dists else 0.0
    fde = float(np.mean(final_dists)) if final_dists else 0.0
    collision_rate = float(collision_pairs / max(total_pairs, 1))
    out_of_bounds_rate = float(outside_count / max(point_count, 1))
    return ade, fde, collision_rate, out_of_bounds_rate


def export_scene_parquet(path: pathlib.Path, positions_world: np.ndarray, mask: np.ndarray, agent_ids: np.ndarray, frames: np.ndarray):
    rows = []
    for aidx, agent_id in enumerate(agent_ids.tolist()):
        if agent_id < 0:
            continue
        for tidx, frame in enumerate(frames.tolist()):
            if tidx >= mask.shape[1] or not mask[aidx, tidx]:
                continue
            rows.append({
                "frame": int(frame),
                "id": int(agent_id),
                "pos_x": float(positions_world[aidx, tidx, 0]),
                "pos_y": float(positions_world[aidx, tidx, 1]),
            })
    pd.DataFrame(rows).to_parquet(path)


@torch.no_grad()
def run_epoch_report(model, val_dataset, report_dir: pathlib.Path, epoch: int, device, cfg):
    model.eval()
    sample = val_dataset.samples[0]
    meta = sample["meta"]
    walkable = sample["walkable"]
    case_id = sample["case_id"]
    case_dir = pathlib.Path(sample["case_dir"])

    positions = sample["positions"].unsqueeze(0).to(device)
    agent_mask = sample["agent_mask"].unsqueeze(0).to(device)
    start_pt = sample["start_pt"].unsqueeze(0).to(device)
    goal_pt = sample["goal_pt"].unsqueeze(0).to(device)
    geo_mask = sample["geo_mask"].unsqueeze(0).to(device)

    out = model(
        positions       = positions,
        agent_mask      = agent_mask,
        start_pt        = start_pt,
        goal_pt         = goal_pt,
        geo_mask        = geo_mask,
        teacher_forcing = False,
        sample_latent   = False,
        goal_weight     = cfg.get("goal_weight", 1.0),
        kl_weight       = cfg.get("kl_weight", 0.01),
        oob_weight      = cfg.get("oob_weight", 0.5),
    )

    pred_norm = out["positions"][0].cpu().numpy()
    gt_norm = sample["positions"].numpy()
    mask = sample["agent_mask"].numpy()
    agent_ids = sample["agent_ids"].numpy()
    frames = sample["frames"].numpy()

    pred_world = denorm_positions(pred_norm, meta)
    gt_world = denorm_positions(gt_norm, meta)
    ade, fde, collision_rate, out_of_bounds_rate = compute_scene_metrics(
        pred_world,
        gt_world,
        mask,
        walkable,
        collision_threshold_m=cfg.get("collision_threshold_m", 0.4),
    )

    epoch_dir = report_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    export_scene_parquet(epoch_dir / f"AI_pred_{case_id}.parquet", pred_world, mask, agent_ids, frames)
    export_scene_parquet(epoch_dir / f"GT_real_{case_id}.parquet", gt_world, mask, agent_ids, frames)

    for fname in [
        "Geo_room.json",
        "Geo_corridor.json",
        f"Spawn_location_{case_id}.csv",
        f"Spawn_exit_{case_id}.csv",
    ]:
        src = case_dir / fname
        if src.exists():
            shutil.copy(src, epoch_dir / fname)

    return ade, fde, collision_rate, out_of_bounds_rate


def main(config_path: str):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str((cfg_file.parent / dataset_path).resolve())

    project_root = pathlib.Path(__file__).resolve().parents[2]
    run_dir = get_next_run_dir(project_root / "AI_Result" / "Method_GNN_CVAE" / "outputs")
    logs_dir = run_dir / "logs"
    weights_dir = run_dir / "weights"
    samples_dir = run_dir / "samples"
    for path in [logs_dir, weights_dir, samples_dir]:
        path.mkdir(parents=True, exist_ok=True)

    shutil.copy(cfg_file, run_dir / "config_train.json")
    print(f"[Train] Run: {run_dir}")
    print(f"[Train] Scope: {cfg.get('scope_name', 'unnamed_scope')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    ds_kwargs = dict(
        obs_len=cfg.get("obs_len", 1),
        frame_stride=cfg.get("frame_stride", 8),
        max_seq_len=cfg.get("max_seq_len", 160),
        grid_size=cfg.get("grid_size", 64),
        geo_padding=cfg.get("geo_padding", 1.0),
        max_agents=cfg.get("max_agents", 64),
        subset_percent=cfg.get("subset_percent", 100.0),
        data_percent=cfg.get("data_percent"),
    )
    train_ds = SimulationTrajectoryDataset(dataset_path, split="train", **ds_kwargs)
    val_ds = SimulationTrajectoryDataset(dataset_path, split="val", **ds_kwargs)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset is empty - check dataset_path or split structure.")

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 2), shuffle=True, num_workers=cfg.get("num_workers", 0), collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 2), shuffle=False, num_workers=cfg.get("num_workers", 0), collate_fn=collate_fn)

    model = GNNCVAE(
        hidden_dim=cfg.get("hidden_dim", 128),
        latent_dim=cfg.get("latent_dim", 32),
        geo_dim=cfg.get("geo_dim", 64),
        gnn_layers=cfg.get("gnn_layers", 2),
        neighbor_radius=cfg.get("neighbor_radius", 0.08),
        dropout=cfg.get("dropout", 0.1),
        use_social=cfg.get("use_social", True),
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    resume = cfg.get("resume_checkpoint")
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"[Train] Resumed from {resume}")

    scaler = GradScaler("cuda") if device.type == "cuda" else None
    epochs = cfg.get("epochs", 60)
    save_every = cfg.get("save_every", 10)
    best_val = float("inf")

    history_file = logs_dir / "training_history.csv"
    progress_file = run_dir / "progress.json"
    with open(history_file, "w") as f:
        f.write("epoch,train_loss,val_loss,ade_m,fde_m,collision_rate,out_of_bounds_rate\n")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch:3d}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)
        val_loss   = validate(model, val_loader, device, cfg, epoch)
        scheduler.step(val_loss)
        ade, fde, collision_rate, out_of_bounds_rate = run_epoch_report(model, val_ds, samples_dir, epoch, device, cfg)

        print(
            f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"ADE={ade:.3f} m  FDE={fde:.3f} m  coll={collision_rate:.4f}  oob={out_of_bounds_rate:.4f}"
        )

        with open(history_file, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{ade:.4f},{fde:.4f},{collision_rate:.6f},{out_of_bounds_rate:.6f}\n")

        with open(progress_file, "w") as f:
            json.dump({"epoch": epoch, "total_epochs": epochs, "percentage": int((epoch / epochs) * 100), "loss": val_loss}, f)

        torch.save(model.state_dict(), weights_dir / "latest_model.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), weights_dir / "best_model.pth")
            print(f"  [*] Best model saved (val_loss={best_val:.5f})")
        if epoch % save_every == 0:
            torch.save(model.state_dict(), weights_dir / f"epoch_{epoch:03d}.pth")

    print(f"\n[Train] Done - {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_train.json")
    args = parser.parse_args()
    main(args.config)



"""
train_transformer.py
──────────────────────
Train GoalConditionedGPT2 (with social context) on full pedestrian trajectories.

Usage:
    python train_transformer.py --config config_train.json

Each epoch:
  1. Teacher-forcing pass on full training set  → train_loss
  2. Validation loss
  3. Inference on first val sample              → FDE / ADE  +  report files
  4. Logs: logs/training_history.csv
  5. Checkpoints: weights/

Output auto-created at:
    AI_Result/Method_Transformer/outputs/run_{N}/
"""

import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import torch
from shapely import wkt as shapely_wkt
from shapely.geometry import Point
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PedestrianDataset, collate_fn
from model import GoalConditionedGPT2
from prepare_geometry_transformer import grid_to_world


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_next_run_dir(base: pathlib.Path) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    nums     = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    idx      = max(nums) + 1 if nums else 1
    return base / f"run_{idx}"


def batch_to_device(batch: dict, device: torch.device) -> dict:
    keys = ["obs_traj", "pred_traj", "start_pt", "end_pt",
            "geo_mask", "neighbor_trajs", "neighbor_mask"]
    return {k: batch[k].to(device) for k in keys if k in batch}


# ─── Training / validation loops ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    use_amp = scaler is not None

    for batch in tqdm(loader, desc="  Train", leave=False):
        b = batch_to_device(batch, device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            out = model(
                b["obs_traj"], b["start_pt"], b["end_pt"], b["geo_mask"],
                neighbor_trajs = b.get("neighbor_trajs"),
                neighbor_mask  = b.get("neighbor_mask"),
                labels         = b["pred_traj"],
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
        n          += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    use_amp = device.type == "cuda"

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        b   = batch_to_device(batch, device)
        with autocast("cuda", enabled=use_amp):
            out = model(
                b["obs_traj"], b["start_pt"], b["end_pt"], b["geo_mask"],
                neighbor_trajs = b.get("neighbor_trajs"),
                neighbor_mask  = b.get("neighbor_mask"),
                labels         = b["pred_traj"],
            )
        total_loss += out["loss"].item()
        n          += 1

    return total_loss / max(n, 1)


# ─── Per-epoch validation report ──────────────────────────────────────────────

@torch.no_grad()
def run_epoch_report(model, val_dataset, report_dir: pathlib.Path, epoch: int, device):
    """
    Inference on the FIRST val sample.
    Saves: AI_pred parquet, GT parquet, Geo JSONs, Spawn CSVs.
    Returns (FDE_metres, ADE_metres).
    """
    model.eval()

    sample   = val_dataset.samples[0]
    case_id  = sample["case_id"]
    meta     = sample["meta"]
    case_dir = pathlib.Path(sample["case_dir"])

    def _to_device(t):
        return t.unsqueeze(0).to(device)

    obs      = _to_device(sample["obs_traj"])
    start_pt = _to_device(sample["start_pt"])
    end_pt   = _to_device(sample["end_pt"])
    geo_mask = _to_device(sample["geo_mask"])
    neigh_t  = _to_device(sample["neighbor_trajs"])
    neigh_m  = _to_device(sample["neighbor_mask"])

    gt_norm  = sample["pred_traj"].numpy()       # [T, 2]
    pred_len = len(gt_norm)

    out       = model(
        obs, start_pt, end_pt, geo_mask,
        neighbor_trajs=neigh_t, neighbor_mask=neigh_m,
        pred_len=pred_len,
    )
    pred_norm = out["logits"][0].cpu().numpy()   # [T, 2]

    def denorm(pts):
        return np.array([grid_to_world(gx, gy, meta) for gx, gy in pts])

    obs_world  = denorm(sample["obs_traj"].numpy())
    pred_world = denorm(pred_norm)
    gt_world   = denorm(gt_norm)

    # ── Load exit polygon ────────────────────────────────────────────────────────
    # GT ends when the real pedestrian first steps into the exit polygon (bottom
    # edge), NOT at the exit centroid (the displayed star).  The model has no
    # stopping mechanism and overshoots past the exit.
    # Fix: trim AI at the first predicted step inside the exit polygon, then
    # measure FDE as distance to the exit CENTROID (the star) – the true goal.
    exit_csv = case_dir / f"Spawn_exit_{case_id}.csv"
    exit_df  = pd.read_csv(exit_csv)
    exit_poly = shapely_wkt.loads(
        exit_df[exit_df["type"] == "exit_area"].iloc[0]["area"]
    )
    exit_centroid = np.array(
        [exit_poly.centroid.x, exit_poly.centroid.y], dtype=np.float32
    )
    exit_buffered = exit_poly.buffer(2.0)   # 2 m tolerance

    stop_idx = len(pred_world)              # default: keep all steps
    for i, (px, py) in enumerate(pred_world):
        if exit_buffered.contains(Point(float(px), float(py))):
            stop_idx = i + 1
            break
    pred_world = pred_world[:stop_idx]

    full_pred = np.concatenate([obs_world, pred_world], axis=0)
    full_gt   = np.concatenate([obs_world, gt_world],   axis=0)

    # FDE = distance from AI's final predicted position to the exit centroid (⭐).
    #   • If AI reached the exit polygon: trimmed final ≈ polygon entry point → small FDE
    #   • If AI never reached exit: full final position → large FDE (correctly penalised)
    # ADE = mean step-by-step error over the shared trajectory length.
    fde     = float(np.linalg.norm(pred_world[-1] - exit_centroid))
    min_len = min(len(pred_world), len(gt_world))
    ade     = float(np.linalg.norm(pred_world[:min_len] - gt_world[:min_len], axis=1).mean())

    # Save report files
    epoch_dir = report_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(full_pred, columns=["pos_x", "pos_y"]).to_parquet(
        epoch_dir / f"AI_pred_{case_id}.parquet"
    )
    pd.DataFrame(full_gt, columns=["pos_x", "pos_y"]).to_parquet(
        epoch_dir / f"GT_real_{case_id}.parquet"
    )
    for fname in [
        "Geo_room.json", "Geo_corridor.json",
        f"Spawn_location_{case_id}.csv",
        f"Spawn_exit_{case_id}.csv",
    ]:
        src = case_dir / fname
        if src.exists():
            shutil.copy(src, epoch_dir / fname)

    return fde, ade


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(config_path: str):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str((cfg_file.parent / dataset_path).resolve())

    # Output dirs
    project_root = pathlib.Path(__file__).resolve().parents[2]
    run_dir      = get_next_run_dir(
        project_root / "AI_Result" / "Method_Transformer" / "outputs"
    )
    logs_dir    = run_dir / "logs"
    weights_dir = run_dir / "weights"
    samples_dir = run_dir / "samples"
    for d in [logs_dir, weights_dir, samples_dir]:
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(cfg_file, run_dir / "config_train.json")
    print(f"\n[Train] Run: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Datasets
    ds_kwargs = dict(
        obs_len        = cfg.get("obs_len", 5),
        frame_stride   = cfg.get("frame_stride", 8),
        max_seq_len    = cfg.get("max_seq_len", 512),
        grid_size      = cfg.get("grid_size", 64),
        geo_padding    = cfg.get("geo_padding", 1.0),
        max_neighbors  = cfg.get("max_neighbors", 10),
        subset_percent = cfg.get("subset_percent", 100.0),
    )

    train_ds = PedestrianDataset(dataset_path, split="train", **ds_kwargs)
    val_ds   = PedestrianDataset(dataset_path, split="val",   **ds_kwargs)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset is empty – check dataset_path in config.")

    n_workers = cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 16),
        shuffle=True,  num_workers=n_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=cfg.get("batch_size", 16),
        shuffle=False, num_workers=n_workers, collate_fn=collate_fn,
    )

    # Model
    model = GoalConditionedGPT2(
        d_model       = cfg.get("d_model", 128),
        nhead         = cfg.get("nhead", 4),
        num_layers    = cfg.get("num_layers", 4),
        max_seq_len   = cfg.get("max_seq_len", 512),
        dropout       = cfg.get("dropout", 0.1),
        max_neighbors = cfg.get("max_neighbors", 10),
        obs_len       = cfg.get("obs_len", 5),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {n_params:,}")

    # torch.compile gives ~20% free speedup (requires PyTorch 2.0+; skipped on CPU)
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[Train] torch.compile: enabled")
        except Exception as e:
            print(f"[Train] torch.compile: skipped ({e})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.get("lr", 3e-4),
        weight_decay = cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    resume = cfg.get("resume_checkpoint", None)
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"[Train] Resumed from {resume}")

    # AMP scaler (None on CPU → falls back to fp32)
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    epochs     = cfg.get("epochs", 100)
    save_every = cfg.get("save_every", 10)
    best_val   = float("inf")

    history_file = logs_dir / "training_history.csv"
    with open(history_file, "w") as f:
        f.write("epoch,train_loss,val_loss,fde_m,ade_m\n")

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch:3d}/{epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss   = validate(model, val_loader, device)
        scheduler.step(val_loss)

        fde, ade = run_epoch_report(model, val_ds, samples_dir, epoch, device)

        print(
            f"  train_loss={train_loss:.5f}  "
            f"val_loss={val_loss:.5f}  "
            f"FDE={fde:.3f} m  ADE={ade:.3f} m"
        )

        with open(history_file, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{fde:.4f},{ade:.4f}\n")

        torch.save(model.state_dict(), weights_dir / "latest_model.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), weights_dir / "best_model.pth")
            print(f"  [*] Best model saved (val_loss={best_val:.5f})")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), weights_dir / f"epoch_{epoch:03d}.pth")

    print(f"\n[Train] Done — {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_train.json")
    args = parser.parse_args()
    main(args.config)

"""
test_gnn_cvae.py
----------------
Evaluate GNNCVAE on the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimulationTrajectoryDataset, collate_fn
from model import GNNCVAE
from prepare_geometry_gnn_cvae import grid_to_world, point_is_inside_walkable


def denorm_positions(norm_positions, meta):
    world = norm_positions.copy()
    for aidx in range(norm_positions.shape[0]):
        for tidx in range(norm_positions.shape[1]):
            world[aidx, tidx] = grid_to_world(norm_positions[aidx, tidx, 0], norm_positions[aidx, tidx, 1], meta)
    return world


def compute_metrics(pred_world, gt_world, mask, walkable, collision_threshold_m):
    import numpy as np

    dists = []
    finals = []
    outside = 0
    points = 0
    collisions = 0
    pairs = 0

    for aidx in range(mask.shape[0]):
        valid_idx = np.where(mask[aidx, 1:])[0]
        if len(valid_idx) == 0:
            continue
        agent_dists = np.linalg.norm(pred_world[aidx, 1:][valid_idx] - gt_world[aidx, 1:][valid_idx], axis=1)
        dists.extend(agent_dists.tolist())
        finals.append(float(agent_dists[-1]))
        for tidx in valid_idx.tolist():
            x, y = pred_world[aidx, 1:][tidx]
            outside += 0 if point_is_inside_walkable(x, y, walkable) else 1
            points += 1

    for tidx in range(1, mask.shape[1]):
        active = np.where(mask[:, tidx])[0]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                pairs += 1
                if float(np.linalg.norm(pred_world[active[i], tidx] - pred_world[active[j], tidx])) < collision_threshold_m:
                    collisions += 1

    return {
        "ADE_m": float(sum(dists) / max(len(dists), 1)),
        "FDE_m": float(sum(finals) / max(len(finals), 1)),
        "collision_rate": float(collisions / max(pairs, 1)),
        "out_of_bounds_rate": float(outside / max(points, 1)),
    }


def main(config_path: str, model_path: str | None = None, run_path: str | None = None):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str((cfg_file.parent / dataset_path).resolve())

    if run_path:
        run_dir_from_arg = pathlib.Path(run_path)
        candidate = run_dir_from_arg / "weights" / "best_model.pth"
        if candidate.exists():
            model_path = str(candidate)

    if model_path is None:
        model_path = cfg.get("model_checkpoint", "")
    model_path = pathlib.Path(model_path) if model_path else None

    if model_path and model_path.exists():
        run_dir = model_path.parents[1]
    else:
        run_dir = cfg_file.parent / "test_results_manual"

    out_dir = run_dir / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")

    test_ds = SimulationTrajectoryDataset(
        dataset_path,
        split="test",
        obs_len=cfg.get("obs_len", 1),
        frame_stride=cfg.get("frame_stride", 8),
        max_seq_len=cfg.get("max_seq_len", 160),
        grid_size=cfg.get("grid_size", 64),
        geo_padding=cfg.get("geo_padding", 1.0),
        max_agents=cfg.get("max_agents", 64),
        subset_percent=cfg.get("subset_percent", 100.0),
        data_percent=cfg.get("data_percent"),
    )
    if len(test_ds) == 0:
        print("[Test] Empty test set. Aborting.")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 0), collate_fn=collate_fn)
    model = GNNCVAE(
        hidden_dim=cfg.get("hidden_dim", 128),
        latent_dim=cfg.get("latent_dim", 32),
        geo_dim=cfg.get("geo_dim", 64),
        gnn_layers=cfg.get("gnn_layers", 2),
        neighbor_radius=cfg.get("neighbor_radius", 0.08),
        dropout=cfg.get("dropout", 0.0),
        use_social=cfg.get("use_social", True),
        max_residual=cfg.get("max_residual", 0.15),
        segment_samples=cfg.get("segment_samples", 5),
    ).to(device)

    if model_path and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Test] Loaded: {model_path}")
    else:
        print("[Test] WARNING: no weights loaded - random init.")

    model.eval()
    rows = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            positions = batch["positions"].to(device)
            agent_mask = batch["agent_mask"].to(device)
            start_pt = batch["start_pt"].to(device)
            goal_pt = batch["goal_pt"].to(device)
            geo_mask = batch["geo_mask"].to(device)

            out = model(
                positions=positions,
                agent_mask=agent_mask,
                start_pt=start_pt,
                goal_pt=goal_pt,
                geo_mask=geo_mask,
                teacher_forcing=False,
                sample_latent=False,
                goal_weight=0.0,
                kl_weight=0.0,
            )

            pred_norm = out["positions"][0].cpu().numpy()
            gt_norm = batch["positions"][0].cpu().numpy()
            mask = batch["agent_mask"][0].cpu().numpy()
            meta = batch["metas"][0]
            walkable = batch["walkables"][0]
            metrics = compute_metrics(
                denorm_positions(pred_norm, meta),
                denorm_positions(gt_norm, meta),
                mask,
                walkable,
                cfg.get("collision_threshold_m", 0.4),
            )
            metrics["case_id"] = batch["case_ids"][0]
            rows.append(metrics)

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"metric": "ADE_m", "value": float(df["ADE_m"].mean())},
        {"metric": "FDE_m", "value": float(df["FDE_m"].mean())},
        {"metric": "collision_rate", "value": float(df["collision_rate"].mean())},
        {"metric": "out_of_bounds_rate", "value": float(df["out_of_bounds_rate"].mean())},
        {"metric": "n_cases", "value": float(len(df))},
    ])
    df.to_csv(out_dir / "test_case_metrics.csv", index=False)
    summary.to_csv(out_dir / "test_evaluation_summary.csv", index=False)
    print(f"[Test] Saved -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_test.json")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--run_path", default=None)
    args = parser.parse_args()
    main(args.config, args.model_path, args.run_path)

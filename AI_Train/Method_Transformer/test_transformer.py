"""
test_transformer.py
────────────────────
Evaluate GoalConditionedGPT2 (with social context) on the test split.

Usage:
    python test_transformer.py --config config_test.json
    python test_transformer.py --config config_test.json \\
        --model_path ../../AI_Result/Method_Transformer/outputs/run_5/weights/best_model.pth

Output in {run_dir}/test_results/:
  test_evaluation_summary.csv
  test_evaluation.txt
  trajectories/eval_scene_NNNN.png   (first 20 samples)
"""

import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PedestrianDataset, collate_fn
from model import GoalConditionedGPT2
from prepare_geometry_transformer import grid_to_world


def batch_to_device(batch: dict, device: torch.device) -> dict:
    keys = ["obs_traj", "pred_traj", "start_pt", "end_pt",
            "geo_mask", "neighbor_trajs", "neighbor_mask"]
    return {k: batch[k].to(device) for k in keys if k in batch}


def main(config_path: str, model_path: str | None = None):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str((cfg_file.parent / dataset_path).resolve())

    if model_path is None:
        model_path = cfg.get("model_checkpoint", "")
    model_path = pathlib.Path(model_path) if model_path else None

    # Output directory (inside same run folder as model weights)
    if model_path and model_path.exists():
        run_dir = model_path.parents[1]
    else:
        run_dir = cfg_file.parent / "test_results_manual"

    out_dir  = run_dir / "test_results"
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")

    # Dataset
    test_ds = PedestrianDataset(
        dataset_path,
        split          = "test",
        obs_len        = cfg.get("obs_len", 5),
        frame_stride   = cfg.get("frame_stride", 8),
        max_seq_len    = cfg.get("max_seq_len", 512),
        grid_size      = cfg.get("grid_size", 64),
        geo_padding    = cfg.get("geo_padding", 1.0),
        max_neighbors  = cfg.get("max_neighbors", 10),
        subset_percent = cfg.get("subset_percent", 100.0),
    )
    if len(test_ds) == 0:
        print("[Test] Empty test set. Aborting.")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.get("batch_size", 8),
        shuffle     = False,
        num_workers = cfg.get("num_workers", 0),
        collate_fn  = collate_fn,
    )

    # Model
    model = GoalConditionedGPT2(
        d_model       = cfg.get("d_model", 128),
        nhead         = cfg.get("nhead", 4),
        num_layers    = cfg.get("num_layers", 4),
        max_seq_len   = cfg.get("max_seq_len", 512),
        dropout       = 0.0,
        max_neighbors = cfg.get("max_neighbors", 10),
        obs_len       = cfg.get("obs_len", 5),
    ).to(device)

    if model_path and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Test] Loaded: {model_path}")
    else:
        print("[Test] WARNING: no weights loaded – random init.")

    model.eval()

    total_fde, total_ade, total_n = 0.0, 0.0, 0
    vis_count = 0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            b       = batch_to_device(batch, device)
            lengths = batch["lengths"]

            batch_max = int(lengths.max().item())
            with autocast(enabled=use_amp):
                out = model(
                    b["obs_traj"], b["start_pt"], b["end_pt"], b["geo_mask"],
                    neighbor_trajs = b.get("neighbor_trajs"),
                    neighbor_mask  = b.get("neighbor_mask"),
                    pred_len       = batch_max,
                )
            preds = out["logits"]   # [B, batch_max, 2]

            for i in range(len(lengths)):
                T         = int(lengths[i].item())
                pred_norm = preds[i, :T].cpu().numpy()
                gt_norm   = batch["pred_traj"][i, :T].cpu().numpy()

                dist = np.linalg.norm(pred_norm - gt_norm, axis=1)   # [T]
                total_fde += float(dist[-1])
                total_ade += float(dist.mean())
                total_n   += 1

                # Visualisation (first 20)
                if vis_count < 20:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    gm = batch["geo_mask"][i].squeeze().cpu().numpy()
                    ax.imshow(gm, cmap="gray", origin="lower", alpha=0.35,
                              extent=[0, 1, 0, 1])

                    ox = b["obs_traj"][i].cpu().numpy()
                    ax.plot(ox[:, 0], ox[:, 1], "b-o", ms=3, label="Observed")
                    ax.plot(gt_norm[:, 0], gt_norm[:, 1], "g--", lw=1.5, label="Ground Truth")
                    ax.plot(pred_norm[:, 0], pred_norm[:, 1], "r-", lw=1.5, label="AI Prediction")
                    ax.scatter(*b["end_pt"][i].cpu().numpy(), c="yellow",
                               marker="*", s=150, zorder=5, label="Goal")

                    # Draw neighbour obs trajectories (faint)
                    nm = batch["neighbor_mask"][i].cpu().numpy()
                    nt = batch["neighbor_trajs"][i].cpu().numpy()   # [K, obs_len, 2]
                    ego_pos = b["obs_traj"][i, 0].cpu().numpy()
                    for k in range(len(nm)):
                        if nm[k]:
                            abs_nt = nt[k] + ego_pos               # back to absolute
                            ax.plot(abs_nt[:, 0], abs_nt[:, 1],
                                    "c-", lw=0.6, alpha=0.5)

                    ax.set_title(f"FDE={dist[-1]:.3f}  ADE={dist.mean():.3f}")
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.legend(fontsize=7)
                    ax.set_aspect("equal")
                    ax.grid(True, alpha=0.3)

                    fig.savefig(traj_dir / f"eval_scene_{vis_count:04d}.png", dpi=120)
                    plt.close(fig)
                    vis_count += 1

    avg_fde = total_fde / max(total_n, 1)
    avg_ade = total_ade / max(total_n, 1)

    print(f"\n[Test] Samples : {total_n}")
    print(f"[Test] ADE      : {avg_ade:.4f}  (normalised [0,1])")
    print(f"[Test] FDE      : {avg_fde:.4f}  (normalised [0,1])")

    pd.DataFrame([
        {"metric": "ADE_norm",  "value": avg_ade},
        {"metric": "FDE_norm",  "value": avg_fde},
        {"metric": "n_samples", "value": total_n},
    ]).to_csv(out_dir / "test_evaluation_summary.csv", index=False)

    with open(out_dir / "test_evaluation.txt", "w") as f:
        f.write(
            f"Test Evaluation\n"
            f"Samples : {total_n}\n"
            f"ADE     : {avg_ade:.4f}  (normalised)\n"
            f"FDE     : {avg_fde:.4f}  (normalised)\n"
        )

    print(f"[Test] Saved → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config_test.json")
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()
    main(args.config, args.model_path)

"""
test_gnn_cvae2.py
-----------------
Evaluate GNNCVAE2 on the test split.

Usage:
  # Auto-find best_model.pth from a run directory
  python3 test_gnn_cvae2.py --config config_test.json --run_path ../../AI_Result/Method_GNN_CVAE2/outputs/run_2

  # Or specify checkpoint directly
  python3 test_gnn_cvae2.py --config config_test.json --model_path ../../AI_Result/Method_GNN_CVAE2/outputs/run_2/weights/best_model.pth

Results are saved to <run_dir>/test_eval/
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimulationTrajectoryDataset, collate_fn
from model import GNNCVAE2
from prepare_geometry_gnn_cvae2 import grid_to_world, point_is_inside_walkable

AI_TRAIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(AI_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(AI_TRAIN_DIR))
from baseline_output import (  # noqa: E402
    create_evaluation_layout,
    finalize_evaluation,
    resolve_checkpoint,
    write_case_prediction,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def denorm_positions(norm: np.ndarray, meta: dict) -> np.ndarray:
    world = np.zeros_like(norm)
    for a in range(norm.shape[0]):
        for t in range(norm.shape[1]):
            world[a, t] = grid_to_world(norm[a, t, 0], norm[a, t, 1], meta)
    return world


def compute_metrics(pred_world, gt_world, mask, walkable, collision_threshold_m, exit_centroid_world=None):
    dists, finals = [], []
    outside = points = collisions = pairs = 0

    for a in range(mask.shape[0]):
        vi = np.where(mask[a, 1:])[0]
        if len(vi) == 0:
            continue
        d = np.linalg.norm(pred_world[a, 1:][vi] - gt_world[a, 1:][vi], axis=1)
        dists.extend(d.tolist())
        last_pred = pred_world[a, 1:][vi[-1]]
        finals.append(
            float(np.linalg.norm(last_pred - exit_centroid_world)) if exit_centroid_world is not None
            else float(d[-1])
        )
        for t in vi.tolist():
            x, y = pred_world[a, 1:][t]
            outside += 0 if point_is_inside_walkable(x, y, walkable) else 1
            points  += 1

    for t in range(1, mask.shape[1]):
        act = np.where(mask[:, t])[0]
        for i in range(len(act)):
            for j in range(i + 1, len(act)):
                pairs += 1
                if np.linalg.norm(pred_world[act[i], t] - pred_world[act[j], t]) < collision_threshold_m:
                    collisions += 1

    return {
        "ADE_m":             float(np.mean(dists))   if dists   else 0.0,
        "FDE_m":             float(np.mean(finals))  if finals  else 0.0,
        "collision_rate":    float(collisions / max(pairs,  1)),
        "out_of_bounds_rate":float(outside   / max(points, 1)),
    }


def prediction_frame(case_id, split, positions_world, mask, agent_ids, frames):
    rows = []
    for a, aid in enumerate(agent_ids.tolist()):
        if aid < 0:
            continue
        for t, frame in enumerate(frames.tolist()):
            if t >= mask.shape[1] or not mask[a, t]:
                continue
            rows.append({
                "case_id": str(case_id), "split": split, "frame": int(frame),
                "agent_id": int(aid), "pos_x": float(positions_world[a, t, 0]),
                "pos_y": float(positions_world[a, t, 1]), "is_active": True,
            })
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str, model_path: str | None = None, run_path: str | None = None):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    # Dataset path
    dp = cfg["dataset_path"]
    dataset_path = dp if os.path.isabs(dp) else str((cfg_file.parent / dp).resolve())

    # Resolve model checkpoint
    if run_path:
        candidate = resolve_checkpoint(run_path)
        if candidate:
            model_path = str(candidate)

    if model_path is None:
        model_path = cfg.get("model_checkpoint", "")
    model_ckpt = pathlib.Path(model_path) if model_path else None

    # Output directory lives beside the run that owns the checkpoint
    if model_ckpt and model_ckpt.exists():
        run_dir = model_ckpt.parents[1]
    elif run_path:
        run_dir = pathlib.Path(run_path)
    else:
        run_dir = cfg_file.parent / "test_results_manual"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")

    # Dataset
    test_ds = SimulationTrajectoryDataset(
        dataset_path, split="test",
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
        print("[Test] Empty test set — check dataset_path.")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg.get("num_workers", 0), collate_fn=collate_fn)

    # Model
    model = GNNCVAE2(
        hidden_dim=cfg.get("hidden_dim", 128),
        latent_dim=cfg.get("latent_dim", 32),
        node_feat_dim=cfg.get("node_feat_dim", 8),
        node_embed_dim=cfg.get("node_embed_dim", 64),
        spatial_gnn_layers=cfg.get("spatial_gnn_layers", 2),
        agent_gnn_layers=cfg.get("agent_gnn_layers", 3),
        neighbor_radius=cfg.get("neighbor_radius", 0.08),
        dropout=0.0,
        max_residual=cfg.get("max_residual", 0.25),
        segment_samples=cfg.get("segment_samples", 5),
    ).to(device)

    compatibility_ok = False
    if model_ckpt and model_ckpt.exists():
        saved = torch.load(model_ckpt, map_location=device)
        if isinstance(saved, dict) and "model_state_dict" in saved:
            state_dict = saved["model_state_dict"]
            trained_name = str(saved.get("data_config", {}).get("dataset_name", ""))
            compatibility_ok = bool(trained_name) and trained_name.casefold() == pathlib.Path(dataset_path).name.casefold()
        else:
            state_dict = saved
        model.load_state_dict(state_dict)
        print(f"[Test] Loaded: {model_ckpt}")
    else:
        raise FileNotFoundError("a valid --model_path or --run_path checkpoint is required")

    dataset_id = cfg.get("dataset_id", "housegan_canonical_imagebase_split_v1")
    dataset_manifest = pathlib.Path(dataset_path) / "manifest_housegan_cases.csv"
    eval_layout = create_evaluation_layout(
        run_dir,
        method_id="Method_GNN_CVAE2",
        dataset_id=dataset_id,
        split="test",
        protocol_version=cfg.get("protocol_version", "v1"),
        checkpoint_path=model_ckpt,
        evaluation_config=cfg,
        dataset_manifest=dataset_manifest if dataset_manifest.exists() else None,
        stochastic_sample_count=1,
        compatibility_ok=compatibility_ok,
        invalid_reason=None if compatibility_ok else "legacy checkpoint lacks verifiable dataset provenance",
    )
    out_dir = eval_layout.root
    print(f"[Test] Output: {out_dir}")

    model.eval()
    rows = []
    col_thresh = cfg.get("collision_threshold_m", 0.4)
    export_pq = cfg.get("export_parquet", True)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test] Evaluating"):
            i = 0
            positions  = batch["positions"][i:i+1].to(device)
            agent_mask = batch["agent_mask"][i:i+1].to(device)
            start_pt   = batch["start_pt"][i:i+1].to(device)
            goal_pt    = batch["goal_pt"][i:i+1].to(device)
            geo_mask   = batch["geo_mask"][i:i+1].to(device)
            nf = batch["node_features_list"][i].to(device)
            ei = batch["edge_index_list"][i].to(device)
            et = batch["edge_type_list"][i].to(device)
            ni = batch["agent_node_ids_list"][i].unsqueeze(0).to(device)

            na = ni.shape[1]
            max_a = positions.shape[1]
            if na < max_a:
                ni_pad = torch.zeros(1, max_a, dtype=torch.long, device=device)
                ni_pad[0, :na] = ni[0]
                ni = ni_pad

            out = model(
                positions, agent_mask, start_pt, goal_pt, geo_mask,
                nf, ei, et, ni,
                teacher_forcing=False, sample_latent=False,
                goal_weight=0.0, kl_weight=0.0, oob_weight=0.0, segment_oob_weight=0.0,
            )

            pred_norm = out["positions"][0].cpu().numpy()
            gt_norm   = batch["positions"][0].cpu().numpy()
            mask_np   = batch["agent_mask"][0].cpu().numpy()
            meta      = batch["metas"][0]
            walkable  = batch["walkables"][0]
            case_id   = batch["case_ids"][0]
            agent_ids = batch["agent_ids"][0].numpy()
            frames    = batch["frames"][0].numpy()

            pred_world = denorm_positions(pred_norm, meta)
            gt_world   = denorm_positions(gt_norm,   meta)

            # FDE vs exit centroid
            goal_n = batch["goal_pt"][0, 0].numpy()
            exit_c = np.array(grid_to_world(float(goal_n[0]), float(goal_n[1]), meta), dtype=np.float32)

            metrics = compute_metrics(pred_world, gt_world, mask_np, walkable, col_thresh, exit_c)
            metrics["case_id"] = case_id
            rows.append(metrics)

            if export_pq:
                pred_df = prediction_frame(
                    case_id, "test", pred_world, mask_np, agent_ids, frames
                )
                pred_df["sample_id"] = 0
                pred_df["sample_seed"] = int(cfg.get("seed", 42))
                write_case_prediction(
                    eval_layout, case_id, pred_df, variant="raw", stochastic=True
                )

    # Summary
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"metric": "ADE_m",             "mean": float(df["ADE_m"].mean()),             "std": float(df["ADE_m"].std())},
        {"metric": "FDE_m",             "mean": float(df["FDE_m"].mean()),             "std": float(df["FDE_m"].std())},
        {"metric": "collision_rate",    "mean": float(df["collision_rate"].mean()),    "std": float(df["collision_rate"].std())},
        {"metric": "out_of_bounds_rate","mean": float(df["out_of_bounds_rate"].mean()),"std": float(df["out_of_bounds_rate"].std())},
        {"metric": "n_cases",           "mean": float(len(df)),                        "std": 0.0},
    ])

    df.to_csv(eval_layout.metrics / "per_case_metrics.csv", index=False)
    summary.to_csv(eval_layout.metrics / "summary_metrics_long.csv", index=False)
    pd.DataFrame([{
        "method_id": "Method_GNN_CVAE2", "variant": "raw", "seed": cfg.get("seed", 42),
        "ADE": float(df["ADE_m"].mean()), "FDE": float(df["FDE_m"].mean()),
        "out_of_bounds_rate": float(df["out_of_bounds_rate"].mean()),
        "collision_exposure_rate": float(df["collision_rate"].mean()),
        "constraint_intervention_rate": 0.0,
    }]).to_csv(eval_layout.metrics / "summary_metrics.csv", index=False)

    plan_by_case = {}
    if dataset_manifest.exists():
        manifest_df = pd.read_csv(dataset_manifest, usecols=["case_id", "plan_name"])
        plan_by_case = dict(zip(manifest_df["case_id"].astype(str), manifest_df["plan_name"].astype(str)))
    research_valid = finalize_evaluation(
        eval_layout,
        case_ids=df["case_id"].astype(str),
        floorplan_ids={plan_by_case[c] for c in df["case_id"].astype(str) if c in plan_by_case},
        compatibility_ok=compatibility_ok,
        canonical_test_required=(dataset_id == "housegan_canonical_imagebase_split_v1"),
        additional_failures=(
            ["dataset subset is not a final evaluation"]
            if float(cfg.get("subset_percent", 100.0)) != 100.0 else []
        ),
    )

    print(f"\n[Test] Results over {len(df)} cases:")
    print(f"  ADE  = {float(df['ADE_m'].mean()):.4f} m  ± {float(df['ADE_m'].std()):.4f}")
    print(f"  FDE  = {float(df['FDE_m'].mean()):.4f} m  ± {float(df['FDE_m'].std()):.4f}")
    print(f"  Coll = {float(df['collision_rate'].mean()):.4f}")
    print(f"  OOB  = {float(df['out_of_bounds_rate'].mean()):.4f}")
    print(f"  Research valid = {research_valid}")
    print(f"\n[Test] Saved → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config_test.json")
    parser.add_argument("--model_path", default=None, help="Path to .pth checkpoint")
    parser.add_argument("--run_path",   default=None, help="Path to run_N dir (auto-finds best_model.pth)")
    args = parser.parse_args()
    main(args.config, args.model_path, args.run_path)

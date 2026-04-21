import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Runtime CUDA libs bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_cuda_ld_library_path():
    venv_env = os.environ.get("VIRTUAL_ENV")
    candidate_venvs = []
    if venv_env:
        candidate_venvs.append(Path(venv_env))
    candidate_venvs.append(Path(sys.executable).resolve().parents[1])
    candidate_venvs.append(Path(__file__).resolve().parents[4] / "AI_Pedsim-env")

    nvidia_root = None
    for venv_root in candidate_venvs:
        if not venv_root: continue
        lib_root = venv_root / "lib"
        if not lib_root.exists(): continue
        for py_dir in lib_root.glob("python*"):
            candidate = py_dir / "site-packages" / "nvidia"
            if candidate.exists():
                nvidia_root = candidate
                break
        if nvidia_root: break

    if not nvidia_root: return
    lib_dirs = []
    for name in ["cuda_runtime", "cudnn", "cublas", "cufft", "curand", "cusolver", "cusparse", "nccl", "nvjitlink"]:
        lib_dir = nvidia_root / name / "lib"
        if lib_dir.exists(): lib_dirs.append(str(lib_dir))
    
    patterns = ["libcudart.so*", "libcublas.so*", "libcudnn.so*", "libnccl.so*"]
    for lib_dir in lib_dirs:
        for pattern in patterns:
            for so_path in sorted(Path(lib_dir).glob(pattern)):
                try: ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                except OSError: pass

_bootstrap_cuda_ld_library_path()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dataset_gnn import build_gnn_data_bundle
from model import build_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inverse_target_transform(values, scaler):
    mean = np.asarray(scaler["target_mean"], dtype=np.float32)
    std = np.asarray(scaler["target_std"], dtype=np.float32)
    return np.expm1(values * std + mean).clip(min=0)

def collate_graphs(batch):
    max_nodes = max(item['x'].shape[0] for item in batch)
    num_features = batch[0]['x'].shape[1]
    batch_x, batch_adj, batch_y = [], [], []
    for item in batch:
        num_nodes = item['x'].shape[0]
        x_padded = torch.zeros((max_nodes, num_features))
        x_padded[:num_nodes, :] = item['x']
        adj_padded = torch.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = item['adj']
        batch_x.append(x_padded)
        batch_adj.append(adj_padded)
        batch_y.append(item['y'])
    return {"x": torch.stack(batch_x), "adj": torch.stack(batch_adj), "y": torch.stack(batch_y)}

# ---------------------------------------------------------------------------
# Testing Logic
# ---------------------------------------------------------------------------

def test(config_path, checkpoint_path=None, output_dir=None):
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).resolve()
        run_dir = checkpoint_path.parent
    else:
        run_root = config_path.parent / config["output"]["root"]
        runs = sorted([d for d in run_root.iterdir() if d.is_dir()])
        if not runs: return
        run_dir = runs[-1]
        checkpoint_path = run_dir / "best_result.pth"

    print(f"[AI_Estimate][GNN][Test] Run: {run_dir.name}")
    print(f"[AI_Estimate][GNN][Test] Checkpoint: {checkpoint_path}")

    metadata = torch.load(run_dir / "metadata.pth", weights_only=False)
    scaler = metadata["scaler"]
    input_dim = metadata["input_dim"]

    _, _, test_ds, _ = build_gnn_data_bundle(config, str(config_path))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_graphs)

    model = build_model(input_dim, config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    all_rows = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, adj, y = batch["x"].float().to(device), batch["adj"].float().to(device), batch["y"].float().to(device)
            output = model(x, adj)
            
            p_s = inverse_target_transform(output.cpu().numpy(), scaler)
            t_s = inverse_target_transform(y.cpu().numpy(), scaler)
            
            all_preds.append(p_s)
            all_targets.append(t_s)
            
            # Metadata for predictions.csv
            row_meta = test_ds.df.iloc[i].copy()
            # Add predictions
            for j, target in enumerate(config["features"]["target"]):
                row_meta[f"true_{target}"] = t_s[0, j]
                row_meta[f"pred_{target}"] = p_s[0, j]
                row_meta[f"abs_error_{target}"] = abs(p_s[0, j] - t_s[0, j])
            all_rows.append(row_meta)

    all_preds, all_targets = np.vstack(all_preds), np.vstack(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets), axis=0)
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2, axis=0))
    
    metrics = {
        "final_test": {
            "mae_overall_s": float(np.mean(mae)),
            "rmse_overall_s": float(np.mean(rmse)),
        },
        "all_metrics": {
            "mae": mae.tolist(),
            "rmse": rmse.tolist()
        },
        "targets": config["features"]["target"]
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed predictions
    eval_dir = Path(output_dir).resolve() if output_dir else run_dir / "test_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(eval_dir / "predictions.csv", index=False)

    print(f"[AI_Estimate][GNN][Test] MAE: {mae}")
    print(f"[AI_Estimate][GNN][Test] Results saved to: {eval_dir}")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_train.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    test(args.config, args.checkpoint, args.output_dir)

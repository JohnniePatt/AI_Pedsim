import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Runtime CUDA libs bootstrap (for TensorFlow/PyTorch GPU in venv)
# ---------------------------------------------------------------------------

def _append_unique_path(existing, new_paths):
    parts = [p for p in (existing or "").split(":") if p]
    seen = set(parts)
    for path in new_paths:
        if path and path not in seen:
            parts.append(path)
            seen.add(path)
    return ":".join(parts)

def _bootstrap_cuda_ld_library_path():
    candidate_venvs = []
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        candidate_venvs.append(Path(venv_env))
    candidate_venvs.append(Path(sys.executable).resolve().parents[1])
    candidate_venvs.append(Path(__file__).resolve().parents[4] / "AI_Pedsim-env")

    nvidia_root = None
    for venv_root in candidate_venvs:
        if not venv_root:
            continue
        lib_root = venv_root / "lib"
        if not lib_root.exists():
            continue
        for py_dir in lib_root.glob("python*"):
            candidate = py_dir / "site-packages" / "nvidia"
            if candidate.exists():
                nvidia_root = candidate
                break
        if nvidia_root:
            break

    if not nvidia_root:
        return
    lib_dirs = []
    for name in [
        "cuda_runtime",
        "cudnn",
        "cublas",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nccl",
        "nvjitlink",
    ]:
        lib_dir = nvidia_root / name / "lib"
        if lib_dir.exists():
            lib_dirs.append(str(lib_dir))
    os.environ["LD_LIBRARY_PATH"] = _append_unique_path(os.environ.get("LD_LIBRARY_PATH", ""), lib_dirs)
    _preload_cuda_libs(lib_dirs)

def _preload_cuda_libs(lib_dirs):
    patterns = [
        "libcudart.so*",
        "libcublas.so*",
        "libcublasLt.so*",
        "libcudnn.so*",
        "libcudnn_*.so*",
        "libcusolver.so*",
        "libcusparse.so*",
        "libcurand.so*",
        "libnccl.so*",
        "libnvJitLink.so*",
    ]
    loaded = set()
    for lib_dir in lib_dirs:
        path_obj = Path(lib_dir)
        if not path_obj.exists():
            continue
        for pattern in patterns:
            for so_path in sorted(path_obj.glob(pattern)):
                key = so_path.name
                if key in loaded:
                    continue
                try:
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                    loaded.add(key)
                except OSError:
                    pass

_bootstrap_cuda_ld_library_path()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dataset_gnn import build_gnn_data_bundle
from model import build_model, choose_device


def print_system_status(device_type, model_type="Graph Neural Network (GNN)"):
    import platform
    import psutil
    print("-" * 60)
    print(f"🚀 [AI_Estimate] Hardware & Model Status")
    print(f"   • Model Type: {model_type}")
    print(f"   • Processor : {platform.processor()}")
    print(f"   • CPUs      : {psutil.cpu_count(logical=True)} logical cores")
    print(f"   • RAM       : {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
    print(f"   • Device    : {device_type}")
    print("-" * 60)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collate_graphs(batch):
    """
    Custom collate to handle graphs with different number of nodes by padding.
    """
    max_nodes = max(item['x'].shape[0] for item in batch)
    num_features = batch[0]['x'].shape[1]
    num_targets = batch[0]['y'].shape[0]
    
    batch_x = []
    batch_adj = []
    batch_y = []
    
    for item in batch:
        num_nodes = item['x'].shape[0]
        # Pad x: [Nodes, Features] -> [max_nodes, Features]
        x_padded = torch.zeros((max_nodes, num_features))
        x_padded[:num_nodes, :] = item['x']
        
        # Pad adj: [Nodes, Nodes] -> [max_nodes, max_nodes]
        adj_padded = torch.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = item['adj']
        
        batch_x.append(x_padded)
        batch_adj.append(adj_padded)
        batch_y.append(item['y'])
        
    return {
        "x": torch.stack(batch_x),
        "adj": torch.stack(batch_adj),
        "y": torch.stack(batch_y)
    }

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def train(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        config = json.load(f)

    # Setup Device
    device = choose_device(config)
    print_system_status(device)

    # Load Data
    train_ds, val_ds, _, scaler = build_gnn_data_bundle(config, str(config_path))
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["train"]["batch_size"], 
        shuffle=True, 
        collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["train"]["batch_size"], 
        shuffle=False, 
        collate_fn=collate_graphs
    )

    # Build Model
    input_dim = 10 # [Area, IsCorridor, IsStart, IsEnd, Agents, DistStraight, DistTopo, DoorWidth, DoorCount, Bottleneck]
    model = build_model(input_dim, config).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["train"]["learning_rate"], 
        weight_decay=config["train"]["weight_decay"]
    )

    # Output Dir
    run_name = config["output"].get("run_name", "auto")
    if run_name == "auto":
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    
    output_root = (config_path.parent / config["output"]["root"]).resolve()
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training Loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"[AI_Estimate][GNN][Train] Starting: {run_name}")
    print(f" - Train batches: {len(train_loader)}")
    print(f" - Val batches: {len(val_loader)}")
    print("-" * 30, flush=True)

    def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{total_epochs} [Train]", leave=True)
        for batch in pbar:
            x, adj, y = batch["x"].float().to(device), batch["adj"].float().to(device), batch["y"].float().to(device)
            optimizer.zero_grad()
            output = model(x, adj)
            
            # Weighted MSE
            weights = torch.ones_like(y)
            weights = weights + torch.relu(y) * 0.5
            weights[:, 2] *= 1.2 
            
            loss = (weights * (output - y)**2).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})
        return total_loss / len(loader)


    def validate_epoch(model, loader, device, epoch, total_epochs):
        model.eval()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d}/{total_epochs} [Val]", leave=True)
        with torch.no_grad():
            for batch in pbar:
                x, adj, y = batch["x"].float().to(device), batch["adj"].float().to(device), batch["y"].float().to(device)
                output = model(x, adj)
                
                weights = torch.ones_like(y)
                weights = weights + torch.relu(y) * 0.5
                weights[:, 2] *= 1.2
                
                loss = (weights * (output - y)**2).mean()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.5f}"})
        return total_loss / len(loader)

    total_epochs = config["train"]["epochs"]
    for epoch in range(total_epochs):
        avg_train = train_epoch(model, train_loader, optimizer, device, epoch, total_epochs)
        avg_val = validate_epoch(model, val_loader, device, epoch, total_epochs)
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_dir / "best_result.pth")
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

    # Save results
    history_df = pd.DataFrame({
        "epoch": np.arange(1, len(history["train_loss"]) + 1),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"]
    })
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    
    torch.save({
        "scaler": scaler,
        "config": config,
        "input_dim": input_dim
    }, output_dir / "metadata.pth")

    print(f"[AI_Estimate][GNN][Train] Best Val Loss: {best_val_loss:.6f}")
    print(f"[AI_Estimate][GNN][Train] Run: {output_dir}")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_train.json")
    args = parser.parse_args()
    train(args.config)

import os
import csv
import json
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Set path roots
wsl_root = "/home/johnnie/programming/AI_Pedsim/AI_Pedsim"

mlp_config_path = os.path.join(wsl_root, "AI_Estimate", "AI_Train", "Method_MLP_PyTorch", "config_train.json")
mlp_checkpoint_path = os.path.join(wsl_root, "AI_Estimate", "AI_result", "Method_MLP_PyTorch", "outputs", "run_20260413_130929", "best_result.pth")

gnn_config_path = os.path.join(wsl_root, "AI_Estimate", "AI_Train", "Method_GNN", "config_train.json")
gnn_checkpoint_path = os.path.join(wsl_root, "AI_Estimate", "AI_result", "Method_GNN", "outputs", "run_20260421_161900", "best_result.pth")
gnn_metadata_path = os.path.join(wsl_root, "AI_Estimate", "AI_result", "Method_GNN", "outputs", "run_20260421_161900", "metadata.pth")

output_csv_path = os.path.join(wsl_root, "Document_Research", "Output_FrameworkResearch", "Summarybase_Output", "comparative_time_method.csv")

# 1. Sum of traditional simulation time on test set
# Load from MLP predictions.csv (which contains all test set rows)
mlp_predictions_csv = os.path.join(wsl_root, "AI_Estimate", "AI_result", "Method_MLP_PyTorch", "outputs", "run_20260413_130929", "test_eval", "predictions.csv")

traditional_time = 0.0
with open(mlp_predictions_csv, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    dur_idx = header.index("simulation_duration_s")
    for row in reader:
        if row:
            traditional_time += float(row[dur_idx])

print(f"Traditional simulation time (test set sum): {traditional_time:.6f} seconds")

# 2. Measure MLP PyTorch inference time
import sys
sys.path.append(os.path.join(wsl_root, "AI_Estimate", "AI_Train", "Method_MLP_PyTorch"))
from model import build_model as build_mlp_model

checkpoint_mlp = torch.load(mlp_checkpoint_path, map_location="cpu", weights_only=False)

# Load raw test CSV
test_csv_path = os.path.join(wsl_root, "Dataset", "Data_Estimate", "Test", "data_estimate.csv")
test_df = pd.read_csv(test_csv_path)

# Prepare features manually based on checkpoint expectations
if "observed_agents" not in test_df.columns:
    test_df["observed_agents"] = test_df["computed_agents"]

# Ensure derived features are present
test_df["computed_agents"] = pd.to_numeric(test_df.get("computed_agents", 0), errors="coerce").fillna(0)
test_df["observed_agents"] = pd.to_numeric(test_df.get("observed_agents", 0), errors="coerce").fillna(0)
test_df["topology_centerline_distance_m"] = pd.to_numeric(test_df["topology_centerline_distance_m"], errors="coerce").fillna(0)
test_df["straight_distance_m"] = pd.to_numeric(test_df["straight_distance_m"], errors="coerce").fillna(0)
test_df["walkable_area_near_path"] = pd.to_numeric(test_df["walkable_area_near_path"], errors="coerce").fillna(0)
test_df["door_count_between_A_B"] = pd.to_numeric(test_df["door_count_between_A_B"], errors="coerce").fillna(0)
test_df["min_door_width_between_A_B"] = pd.to_numeric(test_df["min_door_width_between_A_B"], errors="coerce").fillna(1.5)

test_df["detour_ratio"] = test_df.apply(
    lambda r: r["topology_centerline_distance_m"] / r["straight_distance_m"] if r["straight_distance_m"] > 1e-9 else 1.0, axis=1
)
test_df["distance_gap_m"] = (test_df["topology_centerline_distance_m"] - test_df["straight_distance_m"]).clip(lower=0)
test_df["agent_density_near_path"] = test_df.apply(
    lambda r: r["computed_agents"] / r["walkable_area_near_path"] if r["walkable_area_near_path"] > 1e-9 else 0.0, axis=1
)
test_df["area_per_agent"] = test_df.apply(
    lambda r: r["walkable_area_near_path"] / max(r["computed_agents"], 1) if r["walkable_area_near_path"] > 1e-9 else 0.0, axis=1
)
test_df["door_pressure_per_agent"] = test_df.apply(
    lambda r: (r["computed_agents"] * r["door_count_between_A_B"]) / max(r["min_door_width_between_A_B"], 0.1), axis=1
)

# Variants
for variant in ["full", "half", "single"]:
    test_df[f"variant_{variant}"] = (test_df["variant_id"].astype(str) == variant).astype(float)

# Extract features
mlp_features = test_df[checkpoint_mlp["feature_columns"]].astype(float).to_numpy(dtype=np.float32)

# Standardize
mean = np.array(checkpoint_mlp["scaler"]["feature_mean"], dtype=np.float32)
std = np.array(checkpoint_mlp["scaler"]["feature_std"], dtype=np.float32)
x_test_mlp = torch.from_numpy((mlp_features - mean) / std)

model_mlp = build_mlp_model(len(checkpoint_mlp["feature_columns"]), checkpoint_mlp["config"])
model_mlp.load_state_dict(checkpoint_mlp["model_state_dict"])
model_mlp.eval()

# Warm up
with torch.no_grad():
    for _ in range(10):
        _ = model_mlp(x_test_mlp)

# Measure
t0 = time.perf_counter()
with torch.no_grad():
    _ = model_mlp(x_test_mlp)
t1 = time.perf_counter()
mlp_inference_time = t1 - t0
print(f"MLP inference time: {mlp_inference_time:.6f} seconds")

# Clean up sys.path
sys.path.remove(os.path.join(wsl_root, "AI_Estimate", "AI_Train", "Method_MLP_PyTorch"))
if "model" in sys.modules: del sys.modules["model"]
if "dataset" in sys.modules: del sys.modules["dataset"]

# 3. Measure GNN PyTorch inference time
sys.path.append(os.path.join(wsl_root, "AI_Estimate", "AI_Train", "Method_GNN"))
from dataset_gnn import build_gnn_data_bundle
from model import build_model as build_gnn_model
from test_time_estimator import collate_graphs

with open(gnn_config_path, "r") as f:
    config_gnn = json.load(f)

checkpoint_gnn = torch.load(gnn_checkpoint_path, map_location="cpu", weights_only=False)
metadata_gnn = torch.load(gnn_metadata_path, map_location="cpu", weights_only=False)

# Build GNN dataset bundle
_, _, test_ds_gnn, _ = build_gnn_data_bundle(config_gnn, gnn_config_path)
test_loader_gnn = DataLoader(test_ds_gnn, batch_size=len(test_ds_gnn), shuffle=False, collate_fn=collate_graphs)
batch_gnn = next(iter(test_loader_gnn))
x_gnn = batch_gnn["x"].float()
adj_gnn = batch_gnn["adj"].float()

model_gnn = build_gnn_model(metadata_gnn["input_dim"], config_gnn)
model_gnn.load_state_dict(checkpoint_gnn)
model_gnn.eval()

# Warm up
with torch.no_grad():
    for _ in range(10):
        _ = model_gnn(x_gnn, adj_gnn)

# Measure
t0 = time.perf_counter()
with torch.no_grad():
    _ = model_gnn(x_gnn, adj_gnn)
t1 = time.perf_counter()
gnn_inference_time = t1 - t0
print(f"GNN inference time: {gnn_inference_time:.6f} seconds")

# 4. Generate comparative_time_method.csv
with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Method", "Times"])
    writer.writerow(["Traditional simulation", f"{traditional_time:.2f} s"])
    writer.writerow(["MLP", f"{mlp_inference_time:.5f} s"])
    writer.writerow(["GNN", f"{gnn_inference_time:.5f} s"])

print(f"Successfully generated comparative_time_method.csv at {output_csv_path}")

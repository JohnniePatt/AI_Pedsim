import os
import argparse
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pathlib

from dataset import TrajectorySlidingWindowDataset
from model import GoalConditionedGPT2

def compute_auc_mae(error_array):
    """
    error_array: 1D array of errors across prediction frames (length pred_len)
    Computes the Area Under the Curve (AUC) using trapezoidal rule.
    """
    # Assuming frame steps of 1 unit
    return np.trapz(error_array, dx=1)

def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print("--- Loading Test Set ---")
    test_dataset = TrajectorySlidingWindowDataset(config["dataset_path"], config, split="test", shuffle=True)
    
    if len(test_dataset) == 0:
        print("Error: Test dataset is empty.")
        return
        
    # MODEL Setup
    model = GoalConditionedGPT2(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"]
    ).to(device)
    
    model_ckpt = pathlib.Path(config.get("model_checkpoint", ""))
    if model_ckpt.exists():
        model.load_state_dict(torch.load(model_ckpt, map_location=device))
        print(f"Loaded trained weights from: {model_ckpt}")
    else:
        print(f"⚠️ Warning: No trained model found at {model_ckpt}. Running with untrained weights.")
        
    model.eval()
    pred_len = config["pred_len"]
    
    # Sample 5 Sequences
    num_samples = min(5, len(test_dataset))
    sampled_indices = random.sample(range(len(test_dataset)), num_samples)
    
    total_auc_x = 0.0
    total_auc_y = 0.0
    
    output_dir = pathlib.Path("../../AI_Result/Method_Transformer/test_latest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Testing {num_samples} Samples ---")
    
    for i, idx in enumerate(sampled_indices):
        data = test_dataset[idx]
        
        obs = data["obs_traj"].unsqueeze(0).to(device) # [1, 20, 2]
        true_pred = data["pred_traj"].numpy() # [10, 2]
        
        start_pt = data["start_pt"].unsqueeze(0).to(device)
        end_pt = data["end_pt"].unsqueeze(0).to(device)
        geo_mask = data["geo_mask"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds_tensor = model(obs, start_pt, end_pt, geo_mask, pred_len=pred_len)
            preds = preds_tensor[0].cpu().numpy() # [10, 2]
            
        # Error calculation
        abs_err_x = np.abs(preds[:, 0] - true_pred[:, 0])
        abs_err_y = np.abs(preds[:, 1] - true_pred[:, 1])
        
        # Calculate AUC for MAE
        auc_x = compute_auc_mae(abs_err_x)
        auc_y = compute_auc_mae(abs_err_y)
        
        total_auc_x += auc_x
        total_auc_y += auc_y
        
        print(f"Sample {i+1} - AUC Error X: {auc_x:.4f}, AUC Error Y: {auc_y:.4f}")
        
        # Plot Trajectory (Path)
        fig_traj, ax_traj = plt.subplots(figsize=(6, 6))
        
        # Simplified geo visualization using the mask
        mask_np = data["geo_mask"][0].numpy()
        ax_traj.imshow(mask_np, cmap="gray", origin='lower', extent=[0, mask_np.shape[1], 0, mask_np.shape[0]], alpha=0.5)
        
        obs_np = data["obs_traj"].numpy()
        
        # We roughly map the coords to arbitrary pixel space for visual purposes if not denormalized
        # In a real scenario, you'd denormalize the coordinates
        ax_traj.plot(obs_np[:, 0], obs_np[:, 1], 'co-', label="Observation (Past)")
        ax_traj.plot(true_pred[:, 0], true_pred[:, 1], 'go-', label="True Future")
        ax_traj.plot(preds[:, 0], preds[:, 1], 'rx-', label="Predicted Future")
        ax_traj.scatter(data["end_pt"][0].item(), data["end_pt"][1].item(), c='yellow', marker='*', s=200, label="Destination")
        
        ax_traj.set_title(f"Sample {i+1}: Trajectory Prediction")
        ax_traj.legend()
        fig_traj.savefig(output_dir / f"sample_{i+1}_trajectory.png")
        plt.close(fig_traj)
        
        # Plot Errors over time
        fig_err, ax_err = plt.subplots(figsize=(6, 4))
        frames = np.arange(1, pred_len + 1)
        ax_err.plot(frames, abs_err_x, 'b.-', label=f"Abs Error X (AUC={auc_x:.2f})")
        ax_err.plot(frames, abs_err_y, 'r.-', label=f"Abs Error Y (AUC={auc_y:.2f})")
        
        # Fill area under curve
        ax_err.fill_between(frames, 0, abs_err_x, alpha=0.2, color='blue')
        ax_err.fill_between(frames, 0, abs_err_y, alpha=0.2, color='red')
        
        ax_err.set_title(f"Sample {i+1}: Absolute Error Curve per Frame")
        ax_err.set_xlabel("Prediction Frame")
        ax_err.set_ylabel("Absolute Error (m)")
        ax_err.legend()
        fig_err.savefig(output_dir / f"sample_{i+1}_error_curve.png")
        plt.close(fig_err)
        
    # Final Averages
    avg_auc_x = total_auc_x / num_samples
    avg_auc_y = total_auc_y / num_samples
    
    print("\n" + "="*40)
    print("📋 OVERALL EVALUATION RESULTS (5 Samples)")
    print("="*40)
    print(f"Average MAE (Area Under Curve X): {avg_auc_x:.4f}")
    print(f"Average MAE (Area Under Curve Y): {avg_auc_y:.4f}")
    print("="*40)
    
    # Save test results
    results = {
        "avg_auc_x": avg_auc_x,
        "avg_auc_y": avg_auc_y,
        "num_samples": num_samples,
        "obs_len": config["obs_len"],
        "pred_len": pred_len
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Graphs and results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_test.json")
    args = parser.parse_args()
    main(args.config)

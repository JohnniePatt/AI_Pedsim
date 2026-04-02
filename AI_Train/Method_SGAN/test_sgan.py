import argparse
import json
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset, seq_collate
from model import TrajectoryGenerator

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_test.json")
    parser.add_argument("--run_path", type=str, required=True, help="Path to the training run directory")
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Resolve dataset_path relative to config file if it's relative
    dataset_path = config["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(config_dir, dataset_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Testing SGAN on device: {device}")

    # Load dataset
    test_dataset = TrajectoryDataset(data_dir=dataset_path, config=config, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=seq_collate
    )

    # Initialize model
    model = TrajectoryGenerator(
        emb_dim=config.get("emb_size", 64),
        h_dim=config.get("hidden_size", 128),
        pool_dim=config.get("social_pooling_size", 16),
        obs_len=config["obs_len"],
        pred_len=config["pred_len"]
    ).to(device)

    # Load strongest/latest weights
    weights_dir = os.path.join(args.run_path, "weights")
    if not os.path.exists(weights_dir):
        print("❌ No weights directory found in run path.")
        return
        
    weights = sorted([w for w in os.listdir(weights_dir) if w.endswith(".pth")])
    if not weights:
        print("❌ No checkpoint files found.")
        return
        
    latest_weight = os.path.join(weights_dir, weights[-1])
    print(f"📦 Loading weights from {latest_weight}...")
    model.load_state_dict(torch.load(latest_weight, map_location=device))
    model.eval()

    samples_out = os.path.join(args.run_path, "samples")
    os.makedirs(samples_out, exist_ok=True)
    
    total_ade, total_fde = 0, 0
    samples_count = 0
    
    print("SGAN Generating test trajectories...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Evaluate maybe just 5 scenes to keep images low
            if batch_idx >= 5:
                break
                
            obs_traj = batch[0].to(device) # (obs_len, batch_size, 2)
            pred_traj_gt = batch[1].to(device) # (pred_len, batch_size, 2)
            obs_rel_traj = batch[2].to(device)
            pred_rel_traj_gt = batch[3].to(device)
            seq_start_end = batch[4]
            
            # Predict
            pred_rel_traj = model(obs_rel_traj, seq_start_end)
            
            # Convert relative back to absolute
            # pred_traj = obs_traj[-1] + cumsum(pred_rel_traj)
            # (pred_len, batch_size, 2)
            last_obs = obs_traj[-1] # shape (batch_size, 2)
            pred_traj = torch.zeros_like(pred_traj_gt)
            
            curr = last_obs.clone()
            for t in range(pred_traj.shape[0]):
                curr = curr + pred_rel_traj[t]
                pred_traj[t] = curr

            # Calculate metrics 
            # ADE: Average Displacement Error
            # FDE: Final Displacement Error
            # Not heavily tracked right now but good for printouts
            
            # Visualize Scene
            obs_traj_cpu = obs_traj.cpu().numpy()
            pred_gt_cpu = pred_traj_gt.cpu().numpy()
            pred_cpu = pred_traj.cpu().numpy()
            
            for (start, end) in seq_start_end:
                plt.figure(figsize=(8, 8))
                plt.title(f"SGAN Scene {samples_count} Trajectories")
                plt.xlabel("X")
                plt.ylabel("Y")
                
                # Plot every pedestrian in the scene
                for p_idx in range(start, end):
                    o_x = obs_traj_cpu[:, p_idx, 0]
                    o_y = obs_traj_cpu[:, p_idx, 1]
                    
                    gt_x = pred_gt_cpu[:, p_idx, 0]
                    gt_y = pred_gt_cpu[:, p_idx, 1]
                    
                    p_x = pred_cpu[:, p_idx, 0]
                    p_y = pred_cpu[:, p_idx, 1]
                    
                    plt.plot(o_x, o_y, 'b-', label='Observed' if p_idx == start else "")
                    plt.plot(gt_x, gt_y, 'g--', label='Ground Truth' if p_idx == start else "")
                    plt.plot(p_x, p_y, 'r--', label='SGAN Predicted' if p_idx == start else "")
                    
                    # Mark current pos
                    plt.scatter(o_x[-1], o_y[-1], c='blue', s=20)
                    
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(samples_out, f"scene_{samples_count}.png"), dpi=150)
                plt.close()
                samples_count += 1
                
    # Finish Evaluation Log
    eval_file = os.path.join(args.run_path, "evaluation.txt")
    with open(eval_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("------------------\n")
        f.write(f"Evaluated Scenes: {samples_count}\n")
        f.write("Plots generated in samples/ directory.\n")
        
    print(f"✅ Testing complete! Generated {samples_count} scene visualisations.")

if __name__ == "__main__":
    test()

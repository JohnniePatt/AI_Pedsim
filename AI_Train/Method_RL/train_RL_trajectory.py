"""
train_RL_trajectory.py
Modularized training script for RL trajectory prediction.
Standardized for AI Training Dashboard.
"""
import os
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import argparse
import sys
from shapely.geometry import Polygon
from shapely.wkt import loads as load_wkt

# Local imports
from vir_pedsim import PedsimRL_Env
from model_rl import ActorCritic

# --- Unified Configuration ---
class TrainingConfiguration:
    # 1. Hyperparameters
    seq_len = 20
    input_size = 14
    num_neighbors = 5
    hidden_size = 128
    num_layers = 2
    lr = 3e-4
    max_episodes = 1000
    max_gen_frames = 1000
    gamma = 0.99
    eps_clip = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Paths
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent
    TOPO_DIR = PROJECT_ROOT / "Topo_bottleneck"
    
    # Defaults for data (Can be overridden via JSON)
    train_file = str(TOPO_DIR / "dataswarm" / "test" / "double-botteleneck_100801.sqlite")
    spawn_exit_csv = str(TOPO_DIR / "spawn_exit_area" / "test" / "spawn_exit_100801.csv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_RL_{timestamp}"
    METHOD_NAME = BASE_DIR.name
    RUNS_ROOT = PROJECT_ROOT / "AI_Result" / METHOD_NAME / "outputs"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name
    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"
    LOG_DIR = CURRENT_RUN_DIR / "logs"

    def __init__(self):
        self.RUNS_ROOT.mkdir(exist_ok=True)
        self.CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

config = TrainingConfiguration()

def load_config_from_json(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            if hasattr(config, k): setattr(config, k, v)
    print(f"✅ Loaded config from {json_path}")

def write_progress(epoch, total_epochs, score, avg_val=0.0):
    progress_file = config.BASE_DIR / "progress.json"
    data = {
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "progress_percent": round((epoch + 1) / total_epochs * 100, 2),
        "loss": round(float(score), 2), # In RL, we use Score as 'loss' for display
        "val_loss": round(float(avg_val), 2),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(progress_file, "w") as f: json.dump(data, f, indent=4)

def load_exit_polygon(csv_path: Path):
    df = pd.read_csv(csv_path)
    return load_wkt(df[df['type'] == 'exit_area'].iloc[0]['area'])

def execute_training():
    write_progress(-1, config.max_episodes, 0)
    
    # Save Config Snapshot
    snap_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    with open(snap_path, "w") as f:
        json.dump({k: v for k, v in config.__class__.__dict__.items() if not k.startswith("__") and not callable(v)}, f, indent=4, default=str)

    device = torch.device(config.device)
    
    # 1. Geometry
    GEO_DIR = config.TOPO_DIR / "geo"
    with open(GEO_DIR / "geo_room.json", 'r') as f: room_data = json.load(f)
    with open(GEO_DIR / "geo_corridor.json", 'r') as f: corridor_data = json.load(f)
    room_polys = [Polygon(p) for p in room_data]
    corridor_polys = [Polygon(p) for p in corridor_data]

    # 2. Env & Model
    # env expects a dict for config
    env_cfg = {
        "seq_len": config.seq_len,
        "num_neighbors": config.num_neighbors,
        "max_gen_frames": config.max_gen_frames
    }
    env = PedsimRL_Env(env_cfg, room_polys, corridor_polys)
    exit_centroid = load_exit_polygon(Path(config.spawn_exit_csv)).centroid
    
    model = ActorCritic(int(config.input_size), int(config.hidden_size), int(config.num_layers), 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    log_path = config.LOG_DIR / "training_history.csv"
    with open(log_path, "w") as f: f.write("episode,score,lr\n")

    print(f"🚀 Starting RL Training: {config.run_name}")
    best_score = -float('inf')

    for episode in range(config.max_episodes):
        state = env.reset(Path(config.train_file), exit_centroid).to(device)
        
        # Rollout
        log_probs, rewards = [], []
        total_reward = 0
        
        model.train()
        for f in range(int(config.max_gen_frames)):
            action, log_prob, _ = model.act(state.unsqueeze(0))
            next_state_np, reward, done = env.step(action[0].cpu().numpy())
            state = next_state_np.to(device)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            if done: break
            
        # Optimization (Simple REINFORCE for stability)
        optimizer.zero_grad()
        R = 0; returns = []
        for r in reversed(rewards):
            R = r + config.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        policy_loss = []
        for lp, R_norm in zip(log_probs, returns):
            policy_loss.append(-lp * R_norm)
            
        final_loss = torch.stack(policy_loss).sum()
        final_loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        lr = optimizer.param_groups[0]['lr']
        with open(log_path, "a") as f: f.write(f"{episode},{total_reward:.2f},{lr:.6f}\n")
        
        if (episode + 1) % 10 == 0:
            print(f"🌟 Ep {episode+1:04d} | Score: {total_reward:.1f} | LR: {lr:.6f}")
            write_progress(episode, config.max_episodes, total_reward)

        if total_reward > best_score:
            best_score = total_reward
            torch.save(model.state_dict(), config.CURRENT_RUN_DIR / "best_rl_brain.pt")
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")

        if (episode + 1) % 100 == 0:
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{episode+1}.pth")

    # --- Trigger Standalone Test ---
    print("\n--- Triggering Standalone RL Test ---")
    import subprocess
    test_script = config.BASE_DIR / "test_RL_trajectory.py"
    subprocess.run([sys.executable, str(test_script), "--run_path", str(config.CURRENT_RUN_DIR)])
    
    print(f"🏁 RL Training Finished! Results in {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_active.json")
    args = parser.parse_args()
    
    load_config_from_json(config.BASE_DIR / args.config)
    execute_training()

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from shapely.geometry import Polygon
from shapely.wkt import loads as load_wkt

# Import local dependencies
from model_rl import ActorCritic
from vir_pedsim import PedsimRL_Env

class TestConfig:
    def __init__(self, run_path=None):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self.TOPO_DIR = self.PROJECT_ROOT / "Topo_2"
        
        # Defaults
        self.input_size = 14
        self.hidden_size = 128
        self.num_layers = 2
        self.max_gen_frames = 1000
        
        if run_path:
            self.CURRENT_RUN_DIR = Path(run_path).resolve()
            snap = self.CURRENT_RUN_DIR / "run_config_snapshot.json"
            if snap.exists():
                with open(snap, "r") as f:
                    data = json.load(f)
                    for k, v in data.items(): setattr(self, k, v)
                self.CURRENT_RUN_DIR = Path(run_path).resolve()
            
            self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
            self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)

def load_exit_polygon(csv_path: Path):
    df = pd.read_csv(csv_path)
    return load_wkt(df[df['type'] == 'exit_area'].iloc[0]['area'])

def run_evaluation(run_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] RL Evaluation on: {device_status}\n{'='*50}\n")
    config = TestConfig(run_path)
    
    print(f"🔍 [TEST RL] Evaluating Run: {config.CURRENT_RUN_DIR.name}")
    
    # 1. Geometry
    GEO_DIR = config.TOPO_DIR / "geo"
    with open(GEO_DIR / "geo_room.json", 'r') as f: room_data = json.load(f)
    with open(GEO_DIR / "geo_corridor.json", 'r') as f: corridor_data = json.load(f)
    room_polys = [Polygon(p) for p in room_data]
    corridor_polys = [Polygon(p) for p in corridor_data]

    # 2. Env & Model
    # We use a static test file for evaluation
    test_file = config.TOPO_DIR / "dataswarm" / "test" / "double-botteleneck_100801.sqlite"
    spawn_csv = config.TOPO_DIR / "spawn_exit_area" / "test" / "spawn_exit_100801.csv"
    
    env = PedsimRL_Env({"seq_len": config.seq_len, "num_neighbors": int(config.num_neighbors)}, room_polys, corridor_polys)
    exit_centroid = load_exit_polygon(spawn_csv).centroid

    model = ActorCritic(int(config.input_size), int(config.hidden_size), int(config.num_layers), 2).to(device)
    
    best_ckpt = config.CURRENT_RUN_DIR / "best_rl_brain.pt"
    if not best_ckpt.exists():
        best_ckpt = config.CURRENT_RUN_DIR / "checkpoints" / "generator_best.pth"
        
    if not best_ckpt.exists():
        print(f"❌ [ERROR] No brain found at {best_ckpt}"); return
    
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()

    # 3. Inference
    print("🧪 Running RL Inference (10 episodes)...")
    scores = []
    with torch.no_grad():
        for i in range(10):
            state = env.reset(test_file, exit_centroid).to(device)
            total_reward = 0
            for f in range(int(config.max_gen_frames)):
                action, _, _ = model.act(state.unsqueeze(0))
                next_state_np, reward, done = env.step(action[0].cpu().numpy())
                total_reward += reward
                state = next_state_np.to(device)
                if done: break
            scores.append(total_reward)
            print(f"  Ep {i+1}: Score={total_reward:.2f}")

    avg_score = np.mean(scores)
    score_path = config.CURRENT_RUN_DIR / "test_evaluation_summary.csv"
    with open(score_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"Average Reward,{avg_score:.4f}\n")
        f.write(f"Min Reward,{np.min(scores):.4f}\n")
        f.write(f"Max Reward,{np.max(scores):.4f}\n")

    print(f"📊 [EVAL RL] Avg Score: {avg_score:.2f}")
    print(f"✅ [DONE] Evaluation results saved to {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True, help="Path to run folder")
    args = parser.parse_args()
    run_evaluation(args.run_path)

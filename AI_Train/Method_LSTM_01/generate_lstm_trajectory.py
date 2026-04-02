"""
generate_lstm_trajectory.py
The "Product Show" script for LSTM - Generates full trajectory predictions and visualizations.
Saves results into standardized outputs/gen_xxx folders.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import sys

# --- Model Arch ---
class LSTM_Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__(); self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True); self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])

# --- Helper Functions ---
def load_json_polygons(filepath: Path) -> list:
    with open(filepath, 'r') as f: data = json.load(f)
    return [Polygon(coords) for coords in data]

def load_exit_polygon(csv_path: Path) -> Polygon:
    df = pd.read_csv(csv_path); return load_wkt(df[df['type'] == 'exit_area'].iloc[0]['area'])

def get_sqlite_metadata(sqlite_path: Path):
    conn = sqlite3.connect(sqlite_path)
    meta = {k: float(conn.execute("SELECT value FROM metadata WHERE key = ?", (k,)).fetchone()[0]) for k in ['xmin', 'xmax', 'ymin', 'ymax']}
    conn.close(); return meta

def run_product_show(run_path):
    run_dir = Path(run_path).resolve()
    
    # 📄 Load Configuration Snapshot for Consistency
    params = {"hidden_size": 256, "num_layers": 3, "seq_len": 20, "predict_len": 1}
    config_p = run_dir / "config_active.json"
    if not config_p.exists(): config_p = run_dir / "run_config_snapshot.json"
    if config_p.exists():
        with open(config_p, "r") as f:
            snap = json.load(f)
            for k in ["hidden_size", "num_layers", "seq_len", "predict_len"]:
                if k in snap: params[k] = int(snap[k])
        print(f"📖 [CONFIG] Loaded params from snapshot: {params}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # New Standardized Output Path
    gen_dir = run_dir / "test_results" / "trajectories"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Generation on: {device_status}\n{'='*50}\n")
    
    # 📁 Static Paths relative to This Script's Location
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent # AI_Pedsim
    TOPO_DIR = PROJECT_ROOT / "Topo_2"
    DATASWARM_TEST_DIR = TOPO_DIR / "dataswarm" / "test"
    SPAWN_EXIT_TEST_DIR = TOPO_DIR / "spawn_exit_area" / "test"
    GEO_DIR = TOPO_DIR / "geo"
    
    print(f"🌟 [PRODUCT SHOW] Starting generation in {gen_dir.name}")
    
    # 1. Load Model
    model_path = run_dir / "checkpoints" / "generator_best.pth"
    if not model_path.exists(): model_path = run_dir / "best_lstm.pt"
    
    model = LSTM_Baseline(9, params["hidden_size"], params["num_layers"], 2 * params["predict_len"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Geometry for Background
    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")
    all_walkable = unary_union(room_polys + corridor_polys)
    
    # 3. Pick Sample Seeds (Product Show usually shows a few good examples)
    sql_files = sorted(list(DATASWARM_TEST_DIR.glob("*.sqlite")))[:5]
    
    for sql_f in tqdm(sql_files, desc="🎬 Generating Demos"):
        seed_id = sql_f.stem.split('_')[1]
        meta = get_sqlite_metadata(sql_f)
        exit_poly = load_exit_polygon(SPAWN_EXIT_TEST_DIR / f"spawn_exit_{seed_id}.csv")
        
        # Load Ground Truth trajectories
        conn = sqlite3.connect(sql_f)
        gt_df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data", conn)
        conn.close()
        
        # 🚀 Step-by-Step AI Prediction (Simplified Vectorized)
        all_seeds = [agt.iloc[:params["seq_len"]] for _, agt in gt_df.groupby('id') if len(agt) >= params["seq_len"]]
        if not all_seeds: continue
        
        num_agents = len(all_seeds)
        w, h = meta['xmax'] - meta['xmin'], meta['ymax'] - meta['ymin']
        diag = np.sqrt(w**2 + h**2)
        
        # Current status
        pos_history = np.array([s[['pos_x', 'pos_y']].values for s in all_seeds], dtype=np.float32) # (N, seq_len, 2)
        agent_ids = [s['id'].iloc[0] for s in all_seeds]
        active_mask = np.ones(num_agents, dtype=bool)
        
        results = []
        
        for f in range(500): # Max 500 prediction frames for show
            if not active_mask.any(): break
            
            # Feature extraction
            hist_n = pos_history[:, -params["seq_len"]:, :]
            cx, cy = hist_n[:, :, 0], hist_n[:, :, 1]
            last_x, last_y = cx[:, -1], cy[:, -1]
            
            # Predict
            vx = np.diff(cx, axis=1, prepend=cx[:, :1])
            vy = np.diff(cy, axis=1, prepend=cy[:, :1])
            gdx = (exit_poly.centroid.x - cx) / w
            gdy = (exit_poly.centroid.y - cy) / h
            dist_norm = np.sqrt((exit_poly.centroid.x - cx)**2 + (exit_poly.centroid.y - cy)**2) / diag
            
            # Area checks (Simplified constant for show if moving fast)
            in_r, in_c = np.ones((num_agents, params["seq_len"])), np.zeros((num_agents, params["seq_len"])) # Placeholder for speed
            
            feats = np.stack([ (cx-meta['xmin'])/w, (cy-meta['ymin'])/h, vx, vy, gdx, gdy, dist_norm, in_r, in_c ], axis=-1)
            feat_tensor = torch.tensor(feats, dtype=torch.float32).to(device)
            
            # Predict - Output indices [0, 1] represent dx, dy of first step in predict window
            with torch.no_grad():
                preds = model(feat_tensor).cpu().numpy()
            
            dx, dy = preds[:, 0], preds[:, 1]
            next_x = last_x + dx
            next_y = last_y + dy
            
            new_pos = np.stack([next_x, next_y], axis=-1)[:, np.newaxis, :]
            pos_history = np.concatenate([pos_history, new_pos], axis=1)
            
            # Check exit
            d_exit = np.sqrt((exit_poly.centroid.x - next_x)**2 + (exit_poly.centroid.y - next_y)**2)
            active_mask = active_mask & (d_exit > 0.5)
            
        # 🎨 Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        # Background map
        import matplotlib.patches as patches
        for p in room_polys: 
            coords = np.array(p.exterior.coords)
            ax.add_patch(patches.Polygon(coords, facecolor='#f0f0f0', edgecolor='#cccccc', alpha=0.5))
        for p in corridor_polys: 
            coords = np.array(p.exterior.coords)
            ax.add_patch(patches.Polygon(coords, facecolor='#e0e0e0', edgecolor='#cccccc', alpha=0.5))
            
        # Ground Truth (Thin blue)
        for _, agt in gt_df.groupby('id'):
            ax.plot(agt['pos_x'], agt['pos_y'], color='blue', alpha=0.1, linewidth=0.5)
        
        # AI Predicted (Solid Orange)
        for i in range(num_agents):
            ax.plot(pos_history[i, :, 0], pos_history[i, :, 1], color='#ff7f0e', alpha=0.8, linewidth=1.5)
            # Add a small dot to mark start point
            ax.scatter(pos_history[i, 0, 0], pos_history[i, 0, 1], color='green', s=10, zorder=5)
            
        ax.set_aspect('equal'); ax.set_title(f"Trajectory Product Show: Seed {seed_id}\n(Blue=Reality, Orange=AI Recursive)"); ax.axis('off')
        plt.tight_layout()
        plt.savefig(gen_dir / f"product_show_{seed_id}.png", dpi=200)
        plt.close(fig)
        
    print(f"✅ Product Show generated in: {gen_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    args = parser.parse_args()
    run_product_show(args.run_path)

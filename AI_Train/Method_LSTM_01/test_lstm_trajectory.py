"""
test_lstm_trajectory.py
Standardized evaluation script for LSTM - Directly from SQLite.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import json
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, Polygon

# --- Model ---
class LSTM_Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__(); self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True); self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])

def load_json_polygons(filepath: Path) -> list:
    with open(filepath, 'r') as f: data = json.load(f)
    return [Polygon(coords) for coords in data]

def load_exit_polygon(csv_path: Path) -> Polygon:
    df = pd.read_csv(csv_path); exit_row = df[df['type'] == 'exit_area']
    return load_wkt(exit_row.iloc[0]['area'])

def get_sqlite_data(sqlite_path: Path):
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY id, frame", conn)
    meta = {k: float(conn.execute("SELECT value FROM metadata WHERE key = ?", (k,)).fetchone()[0]) for k in ['xmin', 'xmax', 'ymin', 'ymax']}
    conn.close(); return df, meta

def process_sqlite_to_sequences(sqlite_path: Path, csv_path: Path, room_polys, corridor_polys, seq_len, feat_cols, target_cols):
    try:
        df, meta = get_sqlite_data(sqlite_path); exit_poly = load_exit_polygon(csv_path)
        exit_centroid = exit_poly.centroid
        dw, dh = meta['xmax'] - meta['xmin'], meta['ymax'] - meta['ymin']
        diag = np.sqrt(dw**2 + dh**2)
        
        df['vx'] = df.groupby('id')['pos_x'].diff().fillna(0)
        df['vy'] = df.groupby('id')['pos_y'].diff().fillna(0)
        df['target_dx'] = df.groupby('id')['pos_x'].shift(-1) - df['pos_x']
        df['target_dy'] = df.groupby('id')['pos_y'].shift(-1) - df['pos_y']
        df = df.dropna(subset=['target_dx', 'target_dy']).copy()
        
        df['goal_dx'] = exit_centroid.x - df['pos_x']
        df['goal_dy'] = exit_centroid.y - df['pos_y']
        df['dist_to_exit'] = np.sqrt(df['goal_dx']**2 + df['goal_dy']**2)
        
        import matplotlib.path as mpltPath
        pts = df[['pos_x', 'pos_y']].values
        df['in_room'] = 0.0; df['in_corridor'] = 0.0
        for p in room_polys: df['in_room'] += mpltPath.Path(np.array(p.exterior.coords)).contains_points(pts).astype(float)
        for p in corridor_polys: df['in_corridor'] += mpltPath.Path(np.array(p.exterior.coords)).contains_points(pts).astype(float)
        
        df['x_norm'] = (df['pos_x'] - meta['xmin']) / dw
        df['y_norm'] = (df['pos_y'] - meta['ymin']) / dh
        df['vx_norm'] = df['vx']; df['vy_norm'] = df['vy']
        df['goal_dx_norm'] = df['goal_dx'] / dw; df['goal_dy_norm'] = df['goal_dy'] / dh
        df['dist_to_exit_norm'] = df['dist_to_exit'] / diag
        df['target_dx_norm'] = df['target_dx']; df['target_dy_norm'] = df['target_dy']
        
        X_all, Y_all = [], []
        for agent_id, agent_data in df.groupby('id'):
            if len(agent_data) < seq_len: continue
            feat_np = agent_data[feat_cols].values
            target_np = agent_data[target_cols].values
            for i in range(len(agent_data) - seq_len + 1):
                X_all.append(feat_np[i : i + seq_len])
                Y_all.append(target_np[i + seq_len - 1])
        return X_all, Y_all
    except Exception as e:
        print(f"⚠️ Skip {sqlite_path.name}: {e}"); return [], []

def run_evaluation(run_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Evaluation on: {device_status}\n{'='*50}\n")
    run_dir = Path(run_path).resolve()
    
    # 📁 Static Paths relative to scripts
    PROJECT_ROOT = run_dir.parent.parent.parent
    TOPO_DIR = PROJECT_ROOT / "Topo_2"
    DATASWARM_TEST_DIR = TOPO_DIR / "dataswarm" / "test"
    SPAWN_EXIT_TEST_DIR = TOPO_DIR / "spawn_exit_area" / "test"
    GEO_DIR = TOPO_DIR / "geo"

    # Load Model
    params = {"hidden_size": 256, "num_layers": 3, "feat_n": 9, "target_n": 2, "seq_len": 20}
    feat_cols = ["x_norm", "y_norm", "vx_norm", "vy_norm", "goal_dx_norm", "goal_dy_norm", "dist_to_exit_norm", "in_room", "in_corridor"]
    target_cols = ["target_dx_norm", "target_dy_norm"]

    model = LSTM_Baseline(params["feat_n"], params["hidden_size"], params["num_layers"], params["target_n"]).to(device)
    ckpt = run_dir / "checkpoints" / "generator_best.pth"
    if not ckpt.exists(): ckpt = run_dir / "best_lstm.pt"
    if not ckpt.exists(): print(f"❌ No checkpoint at {ckpt}"); return
    
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Process SQLite files from TEST set
    files = sorted(list(DATASWARM_TEST_DIR.glob("*.sqlite")))[:20] # Limit for speed
    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")
    
    X_test, Y_test = [], []
    print(f"📂 [SQLITE] Loading TEST data from {len(files)} files...")
    for f in tqdm(files):
        seed = f.stem.split('_')[1]
        csv_f = SPAWN_EXIT_TEST_DIR / f"spawn_exit_{seed}.csv"
        x_s, y_s = process_sqlite_to_sequences(f, csv_f, room_polys, corridor_polys, params["seq_len"], feat_cols, target_cols)
        X_test.extend(x_s); Y_test.extend(y_s)

    if not X_test: print("❌ No test data found."); return
    
    X_t, Y_t = torch.tensor(np.array(X_test), dtype=torch.float32).to(device), torch.tensor(np.array(Y_test), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(X_t)
        mae = nn.L1Loss()(out, Y_t).item()
        mse = nn.MSELoss()(out, Y_t).item()
        rmse = np.sqrt(mse)

    # Save Summary
    score_path = run_dir / "test_evaluation_summary.csv"
    with open(score_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"MAE (L1),{mae:.6f}\n")
        f.write(f"MSE,{mse:.6f}\n")
        f.write(f"RMSE,{rmse:.6f}\n")
        f.write(f"Total Test Samples,{len(X_t)}\n")

    print(f"\n🏆 Results: MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    print(f"✅ Saved to {score_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    args = parser.parse_args()
    run_evaluation(args.run_path)

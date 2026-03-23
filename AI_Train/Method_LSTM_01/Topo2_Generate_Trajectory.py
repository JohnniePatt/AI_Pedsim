"""
Topo2_Generate_Trajectory.py (VERSION: MEMORY OPTIMIZED & VECTORIZED)

# เป้าหมาย
สคริปต์รัน AI เดินยาวที่ประหยัดแรมสูงสุด 
- ใช้ NumPy Array แทน List of Dictionaries
- เขียนลงไฟล์ทีละก้อน ไม่เก็บค้างในแรม
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
import json
from tqdm import tqdm
import gc # Garbage Collector

# ===================================================================== #
# CONFIGURATION
# ===================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_2"
GEO_DIR = TOPO_DIR / "geo"
DATASWARM_DIR = TOPO_DIR / "dataswarm" / "test" 
SPAWN_EXIT_DIR = TOPO_DIR / "spawn_exit_area" / "test"
MODEL_PATH = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2" / "best_lstm.pt"
OUTPUT_DIR = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2" / "generated_paths"

CONFIG = {
    "seq_len": 20,
    "max_gen_frames": 2000, 
    "hidden_size": 256,
    "num_layers": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "feature_cols": [
        "x_norm", "y_norm", "vx_norm", "vy_norm", 
        "goal_dx_norm", "goal_dy_norm", "dist_to_exit_norm", 
        "in_room", "in_corridor"
    ],
    "target_cols": ["target_dx_norm", "target_dy_norm"]
}

# ===================================================================== #
# 1. CORE ARCHITECTURE
# ===================================================================== #

class LSTM_Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Baseline, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        return self.fc(out)

def load_json_polygons(filepath: Path) -> list:
    with open(filepath, 'r') as f: data = json.load(f)
    return [Polygon(coords) for coords in data]

def load_exit_polygon(csv_path: Path) -> Polygon:
    df = pd.read_csv(csv_path)
    return load_wkt(df[df['type'] == 'exit_area'].iloc[0]['area'])

# ===================================================================== #
# 2. GENERATION (MEMORY CONSERVATIVE)
# ===================================================================== #

def batch_generate(all_agents_seed, model, device, meta, exit_centroid, room_polys, corridor_polys):
    num_agents = len(all_agents_seed)
    w = meta['xmax'] - meta['xmin']
    h = meta['ymax'] - meta['ymin']
    diag = np.sqrt(w**2 + h**2)

    # 🧊 1. ใช้ Pre-allocated NumPy (ประหยัดแรมกว่า List ของ Dicts มหาศาล)
    # เราคาดการณ์ว่าพื้นที่เก็บข้อมูลสูงสุดคือ (N agents * max_frames)
    max_storage = num_agents * CONFIG["max_gen_frames"]
    results_arr = np.zeros((max_storage, 6), dtype=np.float32) # [frame, id, x, y, in_r, in_c]
    ptr = 0 # ตัวชี้ตำแหน่งในอาเรย์
    
    # History Buffer (N, 20, 2)
    pos_history = np.array([seed[['pos_x', 'pos_y']].values.tolist() for seed in all_agents_seed], dtype=np.float32)
    
    agent_ids = np.array([seed['id'].iloc[0] for seed in all_agents_seed], dtype=np.int32)
    start_frames = np.array([seed['frame'].iloc[-1] for seed in all_agents_seed], dtype=np.int32)
    
    active_mask = np.ones(num_agents, dtype=bool)

    for f in tqdm(range(CONFIG["max_gen_frames"]), desc="   🚶 Walking Agents", leave=False):
        if not active_mask.any(): break
        
        # 📐 Vectorized Features
        history_20 = pos_history[:, -CONFIG["seq_len"]:, :]
        x, y = history_20[:, :, 0], history_20[:, :, 1]
        
        x_norm = (x - meta['xmin']) / w
        y_norm = (y - meta['ymin']) / h
        vx = np.diff(x, axis=1, prepend=x[:, :1])
        vy = np.diff(y, axis=1, prepend=y[:, :1])
        gdx = (exit_centroid.x - x) / w
        gdy = (exit_centroid.y - y) / h
        dist_norm = np.sqrt((exit_centroid.x - x)**2 + (exit_centroid.y - y)**2) / diag
        
        # Area Check (เฉพาะจุดล่าสุด)
        last_x, last_y = x[:, -1], y[:, -1]
        in_r, in_c = np.zeros(num_agents, dtype=np.float32), np.zeros(num_agents, dtype=np.float32)
        for idx in np.where(active_mask)[0]:
            p = Point(last_x[idx], last_y[idx])
            if any(poly.contains(p) for poly in room_polys): in_r[idx] = 1.0
            if any(poly.contains(p) for poly in corridor_polys): in_c[idx] = 1.0
        
        # Stack Features
        features = np.stack([
            x_norm, y_norm, vx, vy, gdx, gdy, dist_norm,
            np.tile(in_r[:, None], (1, CONFIG["seq_len"])),
            np.tile(in_c[:, None], (1, CONFIG["seq_len"]))
        ], axis=-1)
        
        # AI Predict
        feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(feat_tensor).cpu().numpy()
            
        # 🟢 FIX: คูณกลับด้วย w, h เพื่อให้ก้าวเท้ายาวเท่าคนจริง
        next_x = last_x + preds[:, 0] * w
        next_y = last_y + preds[:, 1] * h
        
        # Update History
        new_pos = np.stack([next_x, next_y], axis=-1)[:, np.newaxis, :]
        pos_history = np.concatenate([pos_history, new_pos], axis=1)

        # Store Results (Vectorized)
        batch_active_idx = np.where(active_mask)[0]
        n_active = len(batch_active_idx)
        
        results_arr[ptr : ptr + n_active, 0] = start_frames[batch_active_idx] + f + 1
        results_arr[ptr : ptr + n_active, 1] = agent_ids[batch_active_idx]
        results_arr[ptr : ptr + n_active, 2] = next_x[batch_active_idx]
        results_arr[ptr : ptr + n_active, 3] = next_y[batch_active_idx]
        results_arr[ptr : ptr + n_active, 4] = in_r[batch_active_idx]
        results_arr[ptr : ptr + n_active, 5] = in_c[batch_active_idx]
        ptr += n_active
        
        # Check Stop Condition
        d_exit = np.sqrt((exit_centroid.x - next_x)**2 + (exit_centroid.y - next_y)**2)
        active_mask = active_mask & (d_exit > 0.6)
                
    return pd.DataFrame(results_arr[:ptr, :], columns=['frame', 'id', 'pos_x', 'pos_y', 'in_room', 'in_corridor'])

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG["device"])
    
    print("🛠️ Loading Maps and Model...")
    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")
    
    model = LSTM_Baseline(len(CONFIG["feature_cols"]), CONFIG["hidden_size"], CONFIG["num_layers"], len(CONFIG["target_cols"])).to(device)
    
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if any(k.startswith('module.') for k in ckpt.keys()):
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in ckpt.items(): new_sd[k[7:]] = v
        ckpt = new_sd
    model.load_state_dict(ckpt)
    model.eval()

    test_sql_files = sorted(list(DATASWARM_DIR.glob("*.sqlite")))[:1]
    
    final_gen_paths = [] # เราจะไม่เก็บ data มหาศาลในนี้แล้ว

    for sql_path in tqdm(test_sql_files, desc="🎬 Processing Files"):
        seed_id = sql_path.stem.split('_')[1]
        
        # ดึง Metadata & โหลดคนจริง
        conn = sqlite3.connect(sql_path)
        meta = { k: float(conn.execute("SELECT value FROM metadata WHERE key = ?", (k,)).fetchone()[0]) for k in ['xmin', 'xmax', 'ymin', 'ymax']}
        gt_df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data", conn)
        conn.close()
        
        exit_poly = load_exit_polygon(SPAWN_EXIT_DIR / f"spawn_exit_{seed_id}.csv")
        
        # จัดชุด Seed
        all_seeds = [agt.iloc[:CONFIG["seq_len"]] for _, agt in gt_df.groupby('id') if len(agt) >= CONFIG["seq_len"]]
        if not all_seeds: continue
        
        # 🚀 Generation
        gen_df = batch_generate(all_seeds, model, device, meta, exit_poly.centroid, room_polys, corridor_polys)
        gen_df['seed'] = seed_id
        
        # 📐 วัดสถิติ Error แล้วเคลียร์แรม
        merged = pd.merge(gen_df, gt_df, on=['frame', 'id'], suffixes=('_pred', '_true'))
        if not merged.empty:
            mae_x = np.abs(merged['pos_x_pred'] - merged['pos_x_true']).mean()
            mae_y = np.abs(merged['pos_y_pred'] - merged['pos_y_true']).mean()
            print(f"   📊 [Seed {seed_id}] MAE X: {mae_x:.4f}m, Y: {mae_y:.4f}m")
        
        # บันทึกเป็น Parquet แยกไฟล์ทันที (ไม่อมไว้ในแรม)
        gen_df.to_parquet(OUTPUT_DIR / f"paths_seed_{seed_id}.parquet")
        
        # เคลียร์ตัวแปรหนักๆ ทิ้งทุกรอบ
        del gen_df, gt_df, merged, all_seeds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✅ All results saved individually in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

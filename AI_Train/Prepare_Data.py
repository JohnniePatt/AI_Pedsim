import os
import sqlite3
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import json

# ===================================================================== #
# CONFIGURATION
# ===================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_2"
GEO_DIR = TOPO_DIR / "geo"
DATASWARM_DIR = TOPO_DIR / "dataswarm"
SPAWN_EXIT_DIR = TOPO_DIR / "spawn_exit_area"
PROCESSED_DIR = PROJECT_ROOT / "AI_Train" / "dataswarm_processed_topo2"

CHUNK_SIZE = 5 # ลดจาก 15 เหลือ 5 เพื่อเลี่ยง Error ไฟล์ใหญ่เกินไปบนระบบ Lustre (HPC)
SEQ_LEN = 20
FEATURE_COLS = [
    "x_norm", "y_norm", "vx_norm", "vy_norm", 
    "goal_dx_norm", "goal_dy_norm", "dist_to_exit_norm", 
    "in_room", "in_corridor"
]
TARGET_COLS = ["target_dx_norm", "target_dy_norm"]

# ===================================================================== #
# 1. HELPER FUNCTIONS
# ===================================================================== #

def load_json_polygons(filepath: Path) -> list:
    if not filepath.exists():
        raise FileNotFoundError(f"Missing {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]

def check_point_in_polygons(point: Point, polygons: list) -> float:
    for poly in polygons:
        if poly.contains(point):
            return 1.0
    return 0.0

def load_exit_polygon(csv_path: Path) -> Polygon:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    exit_row = df[df['type'] == 'exit_area']
    if exit_row.empty:
        raise ValueError(f"No 'exit_area' found in {csv_path}")
    return load_wkt(exit_row.iloc[0]['area'])

def get_sqlite_metadata(sqlite_path: Path) -> dict:
    conn = sqlite3.connect(sqlite_path)
    meta = {}
    for key_str in ['xmin', 'xmax', 'ymin', 'ymax', 'fps']:
        res = conn.execute("SELECT value FROM metadata WHERE key = ?", (key_str,)).fetchone()
        meta[key_str] = float(res[0]) if res else 0.0
    conn.close()
    return meta

def load_trajectory_data(sqlite_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY id, frame", conn)
    conn.close()
    return df

# ===================================================================== #
# 2. FEATURE ENGINEERING
# ===================================================================== #

def process_seed_data(sqlite_path: Path, csv_path: Path, room_polys: list, corridor_polys: list) -> pd.DataFrame:
    try:
        seed = int(sqlite_path.stem.split('_')[1])
        df = load_trajectory_data(sqlite_path)
        meta = get_sqlite_metadata(sqlite_path)
        exit_poly = load_exit_polygon(csv_path)
    except Exception as e:
        print(f"Error processing {sqlite_path.name}: {e}")
        return None
        
    if df.empty:
        return None

    exit_centroid = exit_poly.centroid
    domain_width = meta['xmax'] - meta['xmin']
    domain_height = meta['ymax'] - meta['ymin']
    domain_diag = np.sqrt(domain_width**2 + domain_height**2)
    
    if domain_diag == 0: domain_width = domain_height = domain_diag = 100.0 

    # Calculate velocities
    df['prev_x'] = df.groupby('id')['pos_x'].shift(1).fillna(df['pos_x'])
    df['prev_y'] = df.groupby('id')['pos_y'].shift(1).fillna(df['pos_y'])
    df['vx'] = df['pos_x'] - df['prev_x']
    df['vy'] = df['pos_y'] - df['prev_y']

    # Sub-step Target
    df['next_x'] = df.groupby('id')['pos_x'].shift(-1)
    df['next_y'] = df.groupby('id')['pos_y'].shift(-1)
    df = df.dropna(subset=['next_x', 'next_y']).copy()

    df['target_dx'] = df['next_x'] - df['pos_x']
    df['target_dy'] = df['next_y'] - df['pos_y']

    # Goal & Distance Vectors
    df['goal_dx'] = exit_centroid.x - df['pos_x']
    df['goal_dy'] = exit_centroid.y - df['pos_y']
    df['dist_to_exit'] = np.sqrt(df['goal_dx']**2 + df['goal_dy']**2)

    # ⚡ Vectorized Matplotlib Array-based Point-in-Polygon 
    try:
        from matplotlib.path import Path as mplPath
        pts = df[['pos_x', 'pos_y']].values
        
        room_mask = np.zeros(len(df), dtype=bool)
        for p in room_polys:
            room_mask |= mplPath(np.array(p.exterior.coords)).contains_points(pts)
        df['in_room'] = room_mask.astype(float)
        
        corridor_mask = np.zeros(len(df), dtype=bool)
        for p in corridor_polys:
            corridor_mask |= mplPath(np.array(p.exterior.coords)).contains_points(pts)
        df['in_corridor'] = corridor_mask.astype(float)
    except ImportError:
        df['in_room'] = df.apply(lambda r: check_point_in_polygons(Point(r['pos_x'], r['pos_y']), room_polys), axis=1)
        df['in_corridor'] = df.apply(lambda r: check_point_in_polygons(Point(r['pos_x'], r['pos_y']), corridor_polys), axis=1)

    # Final Normalization
    df['x_norm'] = (df['pos_x'] - meta['xmin']) / domain_width
    df['y_norm'] = (df['pos_y'] - meta['ymin']) / domain_height
    df['goal_dx_norm'] = df['goal_dx'] / domain_width
    df['goal_dy_norm'] = df['goal_dy'] / domain_height
    df['dist_to_exit_norm'] = df['dist_to_exit'] / domain_diag
    
    # 🐛 FIX BUG: ทิ้งความเร็วเป็นเมตร/เฟรม เพื่อไม่ให้ Loss มัน underflow
    df['vx_norm'] = df['vx']
    df['vy_norm'] = df['vy']
    df['target_dx_norm'] = df['target_dx']
    df['target_dy_norm'] = df['target_dy']
    
    df['seed'] = seed
    
    return df

# ===================================================================== #
# 3. SEQUENCE GENERATION
# ===================================================================== #

def build_sequences_per_agent(df: pd.DataFrame, seq_len: int, feat_cols: list, target_cols: list):
    X_list, y_list = [], []
    for agent_id, agent_data in df.groupby('id'):
        agent_data = agent_data.sort_values(by='frame').reset_index(drop=True)
        if len(agent_data) < seq_len:
            continue
            
        features_np = agent_data[feat_cols].values
        targets_np = agent_data[target_cols].values
        
        for i in range(len(agent_data) - seq_len + 1):
            X_list.append(features_np[i : i + seq_len])
            y_list.append(targets_np[i + seq_len - 1])
    return X_list, y_list

def process_and_sequence_seed(sqlite_path: Path, csv_path: Path, room_polys: list, corridor_polys: list, seq_len: int, feature_cols: list, target_cols: list):
    df = process_seed_data(sqlite_path, csv_path, room_polys, corridor_polys)
    if df is None:
        return [], []
    return build_sequences_per_agent(df, seq_len, feature_cols, target_cols)

# ===================================================================== #
# 4. CHUNKING & SAVING PIPELINE
# ===================================================================== #

def save_chunks(split_name: str, room_polys: list, corridor_polys: list):
    split_dir = DATASWARM_DIR / split_name
    csv_dir = SPAWN_EXIT_DIR / split_name
    out_dir = PROCESSED_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(split_dir.glob("*.sqlite"))
    if not files:
        print(f"  [?] No SQLite files found in {split_dir}")
        return

    print(f"\n🚀 Processing {split_name.upper()} Data ({len(files)} files into chunks of {CHUNK_SIZE})...")
    
    num_chunks = int(np.ceil(len(files) / CHUNK_SIZE))
    
    for i in range(0, len(files), CHUNK_SIZE):
        chunk_idx = i // CHUNK_SIZE + 1
        chunk_files = files[i : i + CHUNK_SIZE]
        
        X_all, y_all = [], []
        for sqlite_file in tqdm(chunk_files, desc=f"  -> Chunk {chunk_idx:03d}/{num_chunks:03d}", leave=False):
            try:
                seed = sqlite_file.stem.split('_')[1]
                csv_file = csv_dir / f"spawn_exit_{seed}.csv"
                x_s, y_s = process_and_sequence_seed(
                    sqlite_file, csv_file, room_polys, corridor_polys, 
                    SEQ_LEN, FEATURE_COLS, TARGET_COLS
                )
                if x_s and y_s:
                    X_all.extend(x_s)
                    y_all.extend(y_s)
            except Exception as e:
                print(f"Error on {sqlite_file.name}: {e}")
                continue
                
        if X_all:
            # แปลงเป็น PyTorch Tensor แล้วเซฟลงดิสก์!
            X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32)
            y_tensor = torch.tensor(np.array(y_all), dtype=torch.float32)
            
            x_out_path = out_dir / f"X_{split_name}_{chunk_idx:03d}.pt"
            y_out_path = out_dir / f"y_{split_name}_{chunk_idx:03d}.pt"
            
            torch.save(X_tensor, x_out_path)
            torch.save(y_tensor, y_out_path)
            print(f"  ✅ Saved Chunk {chunk_idx:03d}: Generated {len(X_all)} sequences -> {x_out_path.name}")
        else:
            print(f"  ❌ Skipped Chunk {chunk_idx:03d} (No valid sequences)")

def main():
    print("==================================================")
    print("🛠️  DATA PREPARATION SCRIPT (Separating X, y) 🛠️")
    print("==================================================")
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading Topology Geometry...")
    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")
    print("   -> Topology Loaded!")

    print("2. Processing Datasets & Saving Tensors...")
    save_chunks("train", room_polys, corridor_polys)
    save_chunks("validation", room_polys, corridor_polys)
    save_chunks("test", room_polys, corridor_polys)
    
    print("\n🎉 All data pre-processing complete! Tensors are saved in:")
    print(f"   {PROCESSED_DIR}")
    print("\nNext Step: Run Train_Topo2.py to use these pre-made X_train, y_train files.")

if __name__ == "__main__":
    main()

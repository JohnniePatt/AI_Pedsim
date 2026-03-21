"""
train_lstm.py

# เป้าหมาย
สคริปต์สำหรับฝึกโมเดล LSTM พยากรณ์เส้นทางเดินคน (Next-step Prediction)
อัปเดตใหม่: รองรับโครงสร้างแยก Train / Validation / Test อย่างเด็ดขาดในระดับแบบโฟลเดอร์ 
เพื่อป้องกัน Data Leakage ตามมาตรฐานงานวิจัย

# โครงสร้างโฟลเดอร์ที่คาดหวัง:
- Topo_2/dataswarm/train/
- Topo_2/dataswarm/validation/
- Topo_2/dataswarm/test/
และใน spawn_exit_area ต้องมีโฟลเดอร์ในระนาบเดียวกันให้ครบถ้วน
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, Polygon
from tqdm import tqdm

print("Initializing Training Script...")

# ===================================================================== #
# CONFIGURATION
# ===================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_2"
GEO_DIR = TOPO_DIR / "geo"
DATASWARM_DIR = TOPO_DIR / "dataswarm"
SPAWN_EXIT_DIR = TOPO_DIR / "spawn_exit_area"
OUTPUT_DIR = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2"

CONFIG = {
    "seq_len": 20,
    "batch_size": 128,
    "hidden_size": 128,
    "num_layers": 2,
    "lr": 1e-3,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "feature_cols": [
        "x_norm", "y_norm", "vx_norm", "vy_norm", 
        "goal_dx_norm", "goal_dy_norm", "dist_to_exit_norm", 
        "in_room", "in_corridor"
    ],
    "target_cols": ["target_dx_norm", "target_dy_norm"]
}

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
    df['dist_to_exit'] = df.apply(lambda row: Point(row['pos_x'], row['pos_y']).distance(exit_poly), axis=1)

    # Contextual containment
    df['in_room'] = df.apply(lambda r: check_point_in_polygons(Point(r['pos_x'], r['pos_y']), room_polys), axis=1)
    df['in_corridor'] = df.apply(lambda r: check_point_in_polygons(Point(r['pos_x'], r['pos_y']), corridor_polys), axis=1)

    # Final Normalization
    df['x_norm'] = (df['pos_x'] - meta['xmin']) / domain_width
    df['y_norm'] = (df['pos_y'] - meta['ymin']) / domain_height
    df['vx_norm'] = df['vx'] / domain_width
    df['vy_norm'] = df['vy'] / domain_height
    df['goal_dx_norm'] = df['goal_dx'] / domain_width
    df['goal_dy_norm'] = df['goal_dy'] / domain_height
    df['dist_to_exit_norm'] = df['dist_to_exit'] / domain_diag
    df['target_dx_norm'] = df['target_dx'] / domain_width
    df['target_dy_norm'] = df['target_dy'] / domain_height
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

# ===================================================================== #
# 4. NEURAL NETWORK ARCHITECTURE
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

# ===================================================================== #
# 5. DATASET PIPELINE (FOLDER-AWARE SPLIT)
# ===================================================================== #

def load_split_dataset(split_name: str, room_polys, corridor_polys):
    """กระบวนการโหลดและแยกข้อมูลตามชื่อ Subdirectory แบบตายตัว (Train, Validation, Test)"""
    print(f"\n--- Loading '{split_name.upper()}' Split ---")
    data_dir = DATASWARM_DIR / split_name
    csv_dir = SPAWN_EXIT_DIR / split_name
    
    if not data_dir.exists():
        print(f"Skipping {split_name}: Folder {data_dir} does not exist.")
        return None, None, 0
        
    sqlite_files = list(data_dir.glob("double-botteleneck_*.sqlite"))
    if not sqlite_files:
        print(f"Skipping {split_name}: No sqlite files found.")
        return None, None, 0

    df_list = []
    for sqlite_file in tqdm(sqlite_files, desc=f"Parsing {split_name} features"):
        seed = sqlite_file.stem.split('_')[1]
        csv_file = csv_dir / f"spawn_exit_{seed}.csv"
        
        df_seed = process_seed_data(sqlite_file, csv_file, room_polys, corridor_polys)
        if df_seed is not None:
            df_list.append(df_seed)

    if not df_list:
        return None, None, 0

    master_df = pd.concat(df_list, ignore_index=True)
    X_all, y_all = [], []
    
    for seed, seed_df in master_df.groupby('seed'):
        x_s, y_s = build_sequences_per_agent(seed_df, CONFIG["seq_len"], CONFIG["feature_cols"], CONFIG["target_cols"])
        X_all.extend(x_s)
        y_all.extend(y_s)

    if not X_all:
        return None, None, 0

    X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_all), dtype=torch.float32)
    
    print(f"[{split_name.upper()}] Sequences generated: {len(X_tensor)} from {len(sqlite_files)} seeds.")
    return X_tensor, y_tensor, len(sqlite_files)

# ===================================================================== #
# 6. TRAINING ROUTINE
# ===================================================================== #

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")

    # Load splits independently corresponding exactly to folder structures!
    # เลิกใช้ train_test_split() แล้ว เปลี่ยนมาโหลดจากโฟลเดอร์ตายตัว
    X_train, y_train, train_count = load_split_dataset("train", room_polys, corridor_polys)
    X_val, y_val, val_count = load_split_dataset("validation", room_polys, corridor_polys)
    X_test, y_test, test_count = load_split_dataset("test", room_polys, corridor_polys) # เก็บตุนไว้สำหรับ Evaluate ท้ายสุด

    if X_train is None or X_val is None:
        print("ERROR: Missing Train or Validation data. Cannot start training.")
        return

    CONFIG["seeds_used"] = {"train": train_count, "validation": val_count, "test": test_count}
    CONFIG["sequences"] = {"train": len(X_train), "validation": len(X_val), "test": len(X_test) if X_test is not None else 0}

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    device = torch.device(CONFIG["device"])
    print(f"\nModel Initialization (Device: {device})")
    
    model = LSTM_Baseline(
        input_size=len(CONFIG["feature_cols"]),
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=len(CONFIG["target_cols"])
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    epoch_logs = []
    best_val_loss = float('inf')

    print(f"\n--- Starting Training ({CONFIG['epochs']} Epochs) ---")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        train_loss_accum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * batch_x.size(0)
            
        avg_train_loss = train_loss_accum / len(train_loader.dataset)

        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss_accum += loss.item() * batch_x.size(0)
                
        avg_val_loss = val_loss_accum / len(val_loader.dataset)
        
        epoch_logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            CONFIG["best_val_loss"] = best_val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_lstm.pt")

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.6f}")
    
    # 7. ถ่ายทอดผลการ Evaluate บนกลุ่ม TEST แยกต่างหากให้ดูหลังสุด (ถ้ามี)
    if X_test is not None:
        print("\n--- Evaluating on Unseen TEST Set ---")
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_lstm.pt")) # เลือกโมเดลที่ท็อปฟอร์มสุดมาทดสอบ
        model.eval()
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        test_loss_accum = 0.0
        with torch.no_grad():
             for batch_x, batch_y in test_loader:
                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                 loss = criterion(model(batch_x), batch_y)
                 test_loss_accum += loss.item() * batch_x.size(0)
        final_test_mse = test_loss_accum / len(test_loader.dataset)
        print(f"Final Test MSE Loss: {final_test_mse:.6f}")
        CONFIG["final_test_loss"] = final_test_mse
    
    torch.save(model.state_dict(), OUTPUT_DIR / "last_lstm.pt")
    pd.DataFrame(epoch_logs).to_csv(OUTPUT_DIR / "train_log.csv", index=False)
    with open(OUTPUT_DIR / "train_config.json", 'w') as f:
        json.dump(CONFIG, f, indent=4)
        
    print(f"\nAll weights and configs successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    "seq_len": 20,        # ดูย้อนหลัง 20 เฟรม (กำลังดี ไม่สั้นไม่ยาวไป)
    "batch_size": 1024,   # เพิ่มจาก 128 -> ป้อนทีละ 1024 ให้คุ้มพลังและ VRAM 10GB ของ RTX 3080 (ทำให้เทรนไวขึ้นมาก)
    "hidden_size": 256,   # เพิ่มจาก 128 -> เพิ่มขนาดสมองเซลล์ LSTM ให้จำแพทเทิร์นได้เยอะและลึกขึ้น
    "num_layers": 3,      # เพิ่มจาก 2 -> ซ้อนเลเยอร์ 3 ชั้น (Deep) เพื่อช่วยสกัดฟีเจอร์ที่ซับซ้อนของการชะลอตัวเวลาคนหนาแน่น
    "lr": 1e-3,           # ความเร็วการเรียนรู้มาตรฐานของ Adam
    "epochs": 100,        # รัน 100 รอบ เพื่อให้โมเดลมีเวลาเรียนรู้จนกว่าจะลู่เข้าหา Loss ที่ต่ำที่สุด (ปกติงานวิจัยรันข้ามคืน 100-300 รอบ)
    "train_files_per_epoch": 60, # 🌟 จำนวนไฟล์ต่อ 1 รอบ (ชิฟต์สลับไปเรื่อยๆ) เพื่อให้ไม่ต้องรอนานกว่าจะขึ้น Validation
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
    # ⚡ Vectorized Euclidean Distance (ไวขึ้น 1000x ไม่กิน RAM เลตติ้งขยะ Point)
    df['dist_to_exit'] = np.sqrt(df['goal_dx']**2 + df['goal_dy']**2)

    # ⚡ Vectorized Matplotlib Array-based Point-in-Polygon (ไม่ใช้ CPU Apply ที่อืดและหนักเครื่อง)
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
        # หากไม่มีกราฟิก รันแบบเก่า
        df['in_room'] = df.apply(lambda r: check_point_in_polygons(Point(r['pos_x'], r['pos_y']), room_polys), axis=1)
        df['in_corridor'] = df.apply(lambda r: check_point_in_polygons(Point(r['pos_x'], r['pos_y']), corridor_polys), axis=1)

    # Final Normalization
    df['x_norm'] = (df['pos_x'] - meta['xmin']) / domain_width
    df['y_norm'] = (df['pos_y'] - meta['ymin']) / domain_height
    df['goal_dx_norm'] = df['goal_dx'] / domain_width
    df['goal_dy_norm'] = df['goal_dy'] / domain_height
    df['dist_to_exit_norm'] = df['dist_to_exit'] / domain_diag
    
    # 🐛 FIX BUG: ความเร็ว และ ระยะเดินก้าวถัดไป ไม่ควรเอาไปหารกระดาน 110 เมตร
    # ไม่งั้นความเร็วคนเดินแค่ 0.05 เมตร พอโดนหาร 110 มันจะเล็กเป็น 0.0004 จน Loss เป็น 0 หมด!
    # เราทิ้งไว้ให้เป็นตัวเลข raw "เมตร/เฟรม" ตรงๆ เลย (AI เทรนง่ายกว่ามาก)
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
    """Wrapper function to process dataframe and generate sequences concurrently per seed."""
    df = process_seed_data(sqlite_path, csv_path, room_polys, corridor_polys)
    if df is None:
        return [], []
    return build_sequences_per_agent(df, seq_len, feature_cols, target_cols)

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

import random

# ===================================================================== #
# 5. CHUNK-BASED PIPELINE (แก้ปัญหา OOM / นำเข้า GPU ทันที)
# ===================================================================== #

def load_chunk_dataset(sqlite_files_chunk, split_name: str, room_polys, corridor_polys):
    """โหลดข้อมูลแค่ 10-20 ไฟล์พอ ไม่โหลดหมด! เพื่อประหยัด RAM"""
    csv_dir = SPAWN_EXIT_DIR / split_name
    X_all, y_all = [], []
    
    for sqlite_file in sqlite_files_chunk:
        try:
            seed = sqlite_file.stem.split('_')[1]
            csv_file = csv_dir / f"spawn_exit_{seed}.csv"
            x_s, y_s = process_and_sequence_seed(sqlite_file, csv_file, room_polys, corridor_polys, CONFIG["seq_len"], CONFIG["feature_cols"], CONFIG["target_cols"])
            if x_s and y_s:
                X_all.extend(x_s)
                y_all.extend(y_s)
        except Exception:
            continue

    if not X_all:
        return None, None

    return torch.tensor(np.array(X_all), dtype=torch.float32), torch.tensor(np.array(y_all), dtype=torch.float32)

# ===================================================================== #
# 6. TRAINING ROUTINE
# ===================================================================== #

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG["device"])
    
    print("\n" + "="*50)
    print("🖥️  HARDWARE SYSTEM CHECK")
    print("="*50)
    print(f"Device Assigned: {device.type.upper()}")
    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        print(f"GPUs Detected:   {gpu_count} unit(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  [{i}] GPU Model:  {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("WARNING:         🚨 No GPU detected! Running on CPU.")
    print("="*50 + "\n")

    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")

    # กวาดหาไฟล์ทั้งหมด (แต่ยังไม่โหลดลง RAM)
    train_files = list((DATASWARM_DIR / "train").glob("double-botteleneck_*.sqlite"))
    val_files   = list((DATASWARM_DIR / "validation").glob("double-botteleneck_*.sqlite"))
    test_files  = list((DATASWARM_DIR / "test").glob("double-botteleneck_*.sqlite"))

    if not train_files or not val_files:
        print("ERROR: Missing sqlite files in 'train' or 'validation' folders.")
        return

    # ล้างไฟล์ขยะ Config เก่าๆ
    CONFIG["seeds_used"] = {"train": len(train_files), "validation": len(val_files), "test": len(test_files)}
    
    # 📏 ค่าคงที่สำหรับแปลงสเกลกลับเป็น 'เมตร' 
    # (ใช้ประมาณพื้นที่ Topo_2 ที่กว้าง 110m ยาว 130m)
    SCALE_X_M = 110.0  
    SCALE_Y_M = 130.0

    model = LSTM_Baseline(
        input_size=len(CONFIG["feature_cols"]),
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=len(CONFIG["target_cols"])
    )
    
    # 🔥 ฟีเจอร์ลับสำหรับเซิร์ฟเวอร์รวยๆ: ตรวจจับหลายการ์ดจอ และกระจายงานแบ่งกันคำนวณ!
    if torch.cuda.device_count() > 1:
        print(f"🔥 Multi-GPU DETECTED: Distributing batch across {torch.cuda.device_count()} GPUs! 🔥")
        model = nn.DataParallel(model)
        
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    epoch_logs = []
    best_val_loss = float('inf')
    chunk_size = 15 # ❤️ โหลดทีละ 15 ซีดแล้วป้อน GPU เลย แรมจะได้ไม่เต็ม!

    print(f"\n--- 🚀 Starting STREAMING Training ({CONFIG['epochs']} Epochs) ---")
    
    # 🌟 ตัวแปรพิเศษสำหรับไถ (Shift) ข้อมูลทีละส่วน
    train_file_idx = 0
    random.shuffle(train_files)
    files_per_epoch = min(CONFIG.get("train_files_per_epoch", len(train_files)), len(train_files))
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n[ Epoch {epoch:03d}/{CONFIG['epochs']} ]")
        
        # 🚗 ทยอยหยิบไฟล์ย่อยสำหรับเทรนรอบนี้ (Shift ข้อมูลหน้ากระดาน)
        epoch_train_files = []
        for _ in range(files_per_epoch):
            if train_file_idx >= len(train_files):
                random.shuffle(train_files) # พอไถจนหมดลิสต์ ก็สับไพ่และไถกลับไปหน้าสุดใหม่
                train_file_idx = 0
            epoch_train_files.append(train_files[train_file_idx])
            train_file_idx += 1
            
        # -----------------------------
        # TRAIN LOOP (Chunked Stream)
        # -----------------------------
        model.train()
        train_loss_accum = 0.0
        train_sequences_seen = 0
        num_train_chunks = int(np.ceil(len(epoch_train_files) / chunk_size))
        
        for i in range(0, len(epoch_train_files), chunk_size):
            chunk_idx = i // chunk_size + 1
            chunk = epoch_train_files[i : i + chunk_size]
            print(f"  > [Train Chunk {chunk_idx:02d}/{num_train_chunks:02d}] ⏳ Parsing {len(chunk)} files... ", end="", flush=True)
            X_chunk, y_chunk = load_chunk_dataset(chunk, "train", room_polys, corridor_polys)
            
            if X_chunk is None: 
                print("Skipped.")
                continue
            
            print(f"✅ Found {len(X_chunk)} seqs.")
            dataset = torch.utils.data.TensorDataset(X_chunk, y_chunk)
            loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
            
            # 🚀 แถบหลอดแบบใหม่: โชว์เปอร์เซ็นต์การเขมือบข้อมูลของ GPU "เฉพาะ ภายในชุด 15 ไฟล์นี้"
            batch_pbar = tqdm(loader, desc=f"    ⚡ GPU Training  ", leave=False, colour="green")
            for batch_x, batch_y in batch_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item() * batch_x.size(0)
                train_sequences_seen += batch_x.size(0)
            
            batch_pbar.close()
            del X_chunk, y_chunk, dataset, loader # ล้างขยะออกจาก RAM ทันที!

        avg_train_loss = train_loss_accum / max(1, train_sequences_seen)

        # -----------------------------
        # VALIDATION LOOP (Chunked Stream)
        # -----------------------------
        model.eval()
        val_loss_accum = 0.0
        val_ade_accum = 0.0 # Average Displacement Error (m)
        val_sequences_seen = 0
        num_val_chunks = int(np.ceil(len(val_files) / chunk_size))
        
        with torch.no_grad():
            for i in range(0, len(val_files), chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk = val_files[i : i + chunk_size]
                print(f"  > [Val Chunk   {chunk_idx:02d}/{num_val_chunks:02d}] ⏳ Parsing... ", end="", flush=True)
                X_chunk, y_chunk = load_chunk_dataset(chunk, "validation", room_polys, corridor_polys)
                
                if X_chunk is None:
                    print("Skipped.")
                    continue
                
                print(f"✅")
                dataset = torch.utils.data.TensorDataset(X_chunk, y_chunk)
                loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
                
                val_pbar = tqdm(loader, desc=f"    🔍 GPU Validating", leave=False, colour="blue")
                for batch_x, batch_y in val_pbar:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss_accum += loss.item() * batch_x.size(0)
                    
                    err_dx_m = outputs[:, 0] - batch_y[:, 0]
                    err_dy_m = outputs[:, 1] - batch_y[:, 1]
                    dist_err_m = torch.sqrt(err_dx_m**2 + err_dy_m**2)
                    val_ade_accum += dist_err_m.sum().item()
                    
                    val_sequences_seen += batch_x.size(0)
                
                val_pbar.close()
                del X_chunk, y_chunk, dataset, loader 

        avg_val_loss = val_loss_accum / max(1, val_sequences_seen)
        avg_val_ade = val_ade_accum / max(1, val_sequences_seen)
        
        epoch_logs.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_ade": avg_val_ade})
        
        print(f"🎯 Epoch {epoch:03d} Result | Train Loss (MSE): {avg_train_loss:.6f} | Val Loss (MSE): {avg_val_loss:.6f} | 📏 Val Error (ADE): {avg_val_ade:.3f} meters")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            CONFIG["best_val_loss"] = best_val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_lstm.pt")
            print("   -> 🌟 New Best Model Saved!")

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.6f}")
    
    # Eval Test Set ที่กั๊กไว้
    if test_files:
        print("\n--- Evaluating on Unseen TEST Set ---")
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_lstm.pt"))
        model.eval()
        test_loss_accum = 0.0
        test_seq_seen = 0
        num_test_chunks = int(np.ceil(len(test_files) / chunk_size))
        
        with torch.no_grad():
            for i in range(0, len(test_files), chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk = test_files[i : i + chunk_size]
                print(f"  > [Test Chunk  {chunk_idx:02d}/{num_test_chunks:02d}] ⏳ Parsing... ", end="", flush=True)
                X_chunk, y_chunk = load_chunk_dataset(chunk, "test", room_polys, corridor_polys)
                
                if X_chunk is None:
                    print("Skipped.")
                    continue
                    
                print("✅")
                dataset = torch.utils.data.TensorDataset(X_chunk, y_chunk)
                loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
                
                test_pbar = tqdm(loader, desc=f"    🏆 GPU Testing   ", leave=False, colour="magenta")
                for batch_x, batch_y in test_pbar:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    loss = criterion(model(batch_x), batch_y)
                    test_loss_accum += loss.item() * batch_x.size(0)
                    test_seq_seen += batch_x.size(0)
                
                test_pbar.close()
                del X_chunk, y_chunk, dataset, loader
        
        final_test_mse = test_loss_accum / max(1, test_seq_seen)
        print(f"Final Test MSE Loss: {final_test_mse:.6f}")
        CONFIG["final_test_loss"] = final_test_mse

    torch.save(model.state_dict(), OUTPUT_DIR / "last_lstm.pt")
    pd.DataFrame(epoch_logs).to_csv(OUTPUT_DIR / "train_log.csv", index=False)
    with open(OUTPUT_DIR / "train_config.json", 'w') as f:
        json.dump(CONFIG, f, indent=4)
        
    print(f"\nAll weights successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

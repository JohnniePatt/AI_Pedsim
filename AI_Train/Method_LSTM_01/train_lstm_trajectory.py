import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random
import argparse
from datetime import datetime
import shutil
import sys
import json
from shapely.wkt import loads as load_wkt
from shapely.geometry import Point, Polygon
import gc

class TrainingConfiguration:
    seq_len = 20
    batch_size = 1024
    hidden_size = 256
    num_layers = 3
    learning_rate = 1e-3
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🌟 Streaming Settings
    files_per_chunk = 20      # Number of SQLite files to load into memory at once
    train_chunks_per_epoch = 10 # How many chunks to train on per epoch (keeps it fast)
    frame_step = 1            # Skip frames to downsample (1 = all frames, 5 = 5fps)
    predict_len = 1           # Number of frames to predict ahead (Multi-step)
    
    # Feature columns (calculated on-the-fly from SQLite)
    feature_cols = ["x_norm", "y_norm", "vx_norm", "vy_norm", "goal_dx_norm", "goal_dy_norm", "dist_to_exit_norm", "in_room", "in_corridor"]
    target_cols = ["target_dx_norm", "target_dy_norm"]

    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent.parent
    TOPO_DIR = PROJECT_ROOT / "Topo_2"
    DATASWARM_DIR = TOPO_DIR / "dataswarm"
    GEO_DIR = TOPO_DIR / "geo"
    SPAWN_EXIT_DIR = TOPO_DIR / "spawn_exit_area"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_LSTM_{timestamp}"
    RUNS_ROOT = BASE_DIR / "runs"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name
    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"

    def setup_directories(self):
        self.RUNS_ROOT.mkdir(exist_ok=True)
        self.CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(exist_ok=True)

def load_config_from_json(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f: data = json.load(f)
    for key, value in data.items():
        if hasattr(config, key): setattr(config, key, value)
    print(f"📂 [CONFIG] Loaded parameters from {json_path}")

# Initialize Config
config = TrainingConfiguration()
config.setup_directories()
load_config_from_json(config.BASE_DIR / "config_active.json")

# --- Data Processing Helpers ---
def load_json_polygons(filepath: Path) -> list:
    if not filepath.exists(): return []
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]

def load_exit_polygon(csv_path: Path) -> Polygon:
    df = pd.read_csv(csv_path); exit_row = df[df['type'] == 'exit_area']
    return load_wkt(exit_row.iloc[0]['area'])

def get_sqlite_data(sqlite_path: Path):
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data ORDER BY id, frame", conn)
    meta = {k: float(conn.execute("SELECT value FROM metadata WHERE key = ?", (k,)).fetchone()[0]) for k in ['xmin', 'xmax', 'ymin', 'ymax']}
    conn.close(); return df, meta

# --- Optimized Data Structures ---
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, agent_data_list, seq_len, predict_len):
        """
        agent_data_list: List of dicts, each containing 'features' (Nx9) and 'targets' (Nx2)
        """
        self.agent_data = agent_data_list
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.indices = []
        
        for a_idx, data in enumerate(self.agent_data):
            n_frames = len(data['features'])
            # We need seq_len frames for input and predict_len more for targets
            # Since target[i] depends on pos[i+1], we need total seq_len + predict_len frames.
            if n_frames >= self.seq_len + self.predict_len:
                for start_f in range(n_frames - self.seq_len - self.predict_len + 1):
                    self.indices.append((a_idx, start_f))
                    
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        a_idx, start_f = self.indices[idx]
        data = self.agent_data[a_idx]
        
        # Slice on the fly
        x = data['features'][start_f : start_f + self.seq_len]
        
        # Targets: Slice predict_len steps starting from the end of the input sequence
        # Target of first step is at start_f + seq_len - 1
        y = data['targets'][start_f + self.seq_len - 1 : start_f + self.seq_len - 1 + self.predict_len]
        y_flat = y.flatten() # Flatten list of [dx, dy] into [dx0, dy0, dx1, dy1, ...]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_flat, dtype=torch.float32)

def process_sqlite_to_raw_agents(sqlite_path: Path, csv_path: Path, room_polys, corridor_polys):
    """Processes SQLite into a list of dictionaries (one per agent) with full feature arrays."""
    try:
        df, meta = get_sqlite_data(sqlite_path); exit_poly = load_exit_polygon(csv_path)
        exit_centroid = exit_poly.centroid
        dw, dh = meta['xmax'] - meta['xmin'], meta['ymax'] - meta['ymin']
        diag = np.sqrt(dw**2 + dh**2)
        
        # 🔻 Downsampling Per Agent (Efficient Vectorized Way)
        if config.frame_step > 1:
            # Filters every Nth frame within each agent group without dropping columns
            mask = df.groupby('id').cumcount() % config.frame_step == 0
            df = df[mask].copy()

        # Velocity and Targets (Calculated on the downsampled timeline)
        df['vx'] = df.groupby('id')['pos_x'].diff().fillna(0)
        df['vy'] = df.groupby('id')['pos_y'].diff().fillna(0)
        df['target_dx'] = df.groupby('id')['pos_x'].shift(-1) - df['pos_x']
        df['target_dy'] = df.groupby('id')['pos_y'].shift(-1) - df['pos_y']
        df = df.dropna(subset=['target_dx', 'target_dy']).copy()
        
        # Goal and Distance
        df['goal_dx'] = exit_centroid.x - df['pos_x']
        df['goal_dy'] = exit_centroid.y - df['pos_y']
        df['dist_to_exit'] = np.sqrt(df['goal_dx']**2 + df['goal_dy']**2)
        
        # Room/Corridor check
        import matplotlib.path as mpltPath
        pts = df[['pos_x', 'pos_y']].values
        df['in_room'] = 0.0; df['in_corridor'] = 0.0
        for p in room_polys: df['in_room'] += mpltPath.Path(np.array(p.exterior.coords)).contains_points(pts).astype(float)
        for p in corridor_polys: df['in_corridor'] += mpltPath.Path(np.array(p.exterior.coords)).contains_points(pts).astype(float)
        
        # Normalization
        df['x_norm'] = (df['pos_x'] - meta['xmin']) / dw
        df['y_norm'] = (df['pos_y'] - meta['ymin']) / dh
        df['vx_norm'] = df['vx']; df['vy_norm'] = df['vy']
        df['goal_dx_norm'] = df['goal_dx'] / dw; df['goal_dy_norm'] = df['goal_dy'] / dh
        df['dist_to_exit_norm'] = df['dist_to_exit'] / diag
        df['target_dx_norm'] = df['target_dx']; df['target_dy_norm'] = df['target_dy']
        
        # Result Construction
        agent_list = []
        for agent_id, agent_data in df.groupby('id'):
            agent_list.append({
                'features': agent_data[config.feature_cols].values.astype(np.float32),
                'targets': agent_data[config.target_cols].values.astype(np.float32)
            })
        return agent_list
    except Exception as e:
        print(f"⚠️ Skip {sqlite_path.name}: {e}"); return []

def load_sqlite_batch(file_list, room_polys, corridor_polys):
    all_agents = []
    for f in file_list:
        try: seed = f.stem.split('_')[1]
        except: seed = "0"
        
        subset = f.parent.name
        c_dir = config.SPAWN_EXIT_DIR / subset
        
        agents = process_sqlite_to_raw_agents(f, c_dir / f"spawn_exit_{seed}.csv", room_polys, corridor_polys)
        all_agents.extend(agents)
    
    if not all_agents: return None
    return SequenceDataset(all_agents, config.seq_len, config.predict_len)

# --- Model ---
class LSTM_Baseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__(); self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True); self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x): out, _ = self.lstm(x); return self.fc(out[:, -1, :])

def execute_training():
    device = torch.device(config.device)
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Training on: {device_status}\n{'='*50}\n")

    # 💾 Archive Config
    orig_config = config.BASE_DIR / "config_active.json"
    if orig_config.exists(): shutil.copy(orig_config, config.CURRENT_RUN_DIR / "config_active.json")
    
    # 📂 Preparation
    room_polys = load_json_polygons(config.GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(config.GEO_DIR / "geo_corridor.json")
    
    train_files = sorted(list((config.DATASWARM_DIR / "train").glob("*.sqlite")))
    val_files = sorted(list((config.DATASWARM_DIR / "validation").glob("*.sqlite")))
    
    # 🏗️ Model Setup
    model = LSTM_Baseline(len(config.feature_cols), int(config.hidden_size), int(config.num_layers), len(config.target_cols) * int(config.predict_len))
    
    # 🔌 Loading Pretrained Checkpoint
    if hasattr(config, "pretrained_checkpoint") and config.pretrained_checkpoint != "-" and os.path.exists(config.pretrained_checkpoint):
        print(f"🔄 [RESUME] Loading weights from: {config.pretrained_checkpoint}")
        model.load_state_dict(torch.load(config.pretrained_checkpoint, map_location="cpu"))

    model = model.to(device)
    
    # 🔥 Multi-GPU Support
    if torch.cuda.device_count() > 1:
        print(f"🔥 Multi-GPU DETECTED: Distributing batch across {torch.cuda.device_count()} GPUs! 🔥")
        model = nn.DataParallel(model)

    criterion = nn.MSELoss(); optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    history_file = config.CURRENT_RUN_DIR / "training_history.csv"
    best_loss = float('inf')

    print(f"\n--- 🚀 Starting ON-THE-FLY SQLite Streaming ({config.epochs} Epochs) ---")
    
    file_idx = 0
    random.shuffle(train_files)

    for epoch in range(int(config.epochs)):
        model.train(); epoch_t_loss = 0.0; chunks_processed = 0
        
        # 🚗 Batch Loop (Streaming from SQLite)
        for chunk_i in range(config.train_chunks_per_epoch):
            batch_files = []
            for _ in range(config.files_per_chunk):
                if file_idx >= len(train_files):
                    random.shuffle(train_files); file_idx = 0
                batch_files.append(train_files[file_idx]); file_idx += 1
            
            print(f"  > [Epoch {epoch+1:02d} | Chunk {chunk_i+1:02d}] ⏳ Processing {len(batch_files)} SQLite files... ", end="", flush=True)
            train_dataset = load_sqlite_batch(batch_files, room_polys, corridor_polys)
            
            if train_dataset is None: print("⚠️ Empty chunk, skipping."); continue
            print(f"✅ {len(train_dataset)} sequences.")
            
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True)
            
            chunk_loss = 0.0
            pbar = tqdm(loader, desc=f"    ⚡ GPU Training  ", leave=False, colour="green")
            for bx, by in pbar:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(); out = model(bx); loss = criterion(out, by); loss.backward(); optimizer.step()
                chunk_loss += loss.item()
            
            epoch_t_loss += (chunk_loss / len(loader))
            chunks_processed += 1
            
            # 🧹 Explicitly Free Memory
            del train_dataset, loader; gc.collect(); torch.cuda.empty_cache()

        avg_t_loss = epoch_t_loss / max(1, chunks_processed)
        
        # 🔍 Validation
        model.eval(); v_loss = 0.0
        val_subset = val_files[:config.files_per_chunk]
        val_dataset = load_sqlite_batch(val_subset, room_polys, corridor_polys)
        
        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(config.batch_size), shuffle=False)
            with torch.no_grad():
                for bx, by in val_loader: v_loss += criterion(model(bx.to(device)), by.to(device)).item()
            v_loss /= len(val_loader)
            del val_dataset, val_loader; gc.collect()
        
        # 💾 Immediate History Saving (Appending to CSV)
        epoch_stats = {"epoch": epoch+1, "train_loss": avg_t_loss, "val_loss": v_loss}
        print(f"✨ Epoch {epoch+1:03d}: Train Loss {avg_t_loss:.6f} | Val Loss {v_loss:.6f}")
        
        pd.DataFrame([epoch_stats]).to_csv(history_file, mode='a', header=not os.path.exists(history_file), index=False)

        if v_loss < best_loss:
            best_loss = v_loss
            # Save actual model (for DataParallel, we must save model.module or use standard saving)
            save_obj = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(save_obj, config.CHECKPOINT_DIR / "generator_best.pth")
            print("   -> 🏆 New Best Model Saved!")

    print(f"✅ Training Complete. Best Val Loss: {best_loss:.6f}")
    
    # Final Test Trigger
    test_script = config.BASE_DIR / 'test_lstm_trajectory.py'
    if test_script.exists():
        print(f"🚀 Launching Final Test Evaluation...")
        os.system(f"{sys.executable} {test_script} --run_path {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    execute_training()

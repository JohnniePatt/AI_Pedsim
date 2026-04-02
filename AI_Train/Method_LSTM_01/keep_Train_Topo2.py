"""
train_lstm.py

# เป้าหมาย
สคริปต์สำหรับฝึกโมเดล LSTM พยากรณ์เส้นทางเดินคน (Next-step Prediction)
อัปเดตใหม่: โค้ดสะอาดตาสไตล์ Normal ML Pipeline (โหลด X_train, y_train ที่หั่นไว้แล้วจากฮาร์ดดิสก์)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import random

print("Initializing Training Script (Clean Pipeline)...")

# ===================================================================== #
# CONFIGURATION
# ===================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_2"
PROCESSED_DIR = PROJECT_ROOT / "AI_Train" / "dataswarm_processed_topo2"
OUTPUT_DIR = PROJECT_ROOT / "AI_Result" / "Method_LSTM_01" / "outputs" / "Topo2"

CONFIG = {
    "seq_len": 20,        
    "batch_size": 1024,   
    "hidden_size": 256,   
    "num_layers": 3,      
    "lr": 1e-3,           
    "epochs": 100,        
    "train_chunks_per_epoch": 4, # 🌟 จำนวนโฟลเดอร์ Chunk (ปกติ 1 chunk = 15 ไฟล์) ที่จะหยิบมาต่อบอก
    "pretrained_checkpoint": "-", 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "feature_cols": [
        "x_norm", "y_norm", "vx_norm", "vy_norm", 
        "goal_dx_norm", "goal_dy_norm", "dist_to_exit_norm", 
        "in_room", "in_corridor"
    ],
    "target_cols": ["target_dx_norm", "target_dy_norm"]
}

# ===================================================================== #
# 1. NEURAL NETWORK ARCHITECTURE
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
# 2. TRAINING ROUTINE
# ===================================================================== #

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
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

    # กวาดหาไฟล์ .pt (ก้อน X_train, y_train ที่ถูกตระเตรียมไว้ล่วงหน้าจาก Prepare_Data.py)
    x_train_files = sorted(list((PROCESSED_DIR / "train").glob("X_train_*.pt")))
    x_val_files   = sorted(list((PROCESSED_DIR / "validation").glob("X_validation_*.pt")))
    x_test_files  = sorted(list((PROCESSED_DIR / "test").glob("X_test_*.pt")))

    if not x_train_files or not x_val_files:
        print("❌ ERROR: Missing pre-processed .pt files! Please run Prepare_Data.py first.")
        return

    CONFIG["chunks_available"] = {"train": len(x_train_files), "validation": len(x_val_files), "test": len(x_test_files)}

    model = LSTM_Baseline(
        input_size=len(CONFIG["feature_cols"]),
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=len(CONFIG["target_cols"])
    )
    
    # 🌟 Transfer Learning Loading
    if CONFIG.get("pretrained_checkpoint", "-") != "-":
        ckpt_path = Path(CONFIG["pretrained_checkpoint"])
        if ckpt_path.exists():
            print(f"\n🧠 [PRE-TRAINED] Loading Master Weights from: {ckpt_path.name}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"      -> 📜 Resumed from History: Epoch {checkpoint.get('epoch', '?')} (Val Loss: {checkpoint.get('val_loss', 'N/A')})")
            else:
                model.load_state_dict(checkpoint) 
                print("      -> ✨ Successfully Injected Weights (best_lstm style).")
        else:
            print(f"\n❌ [ERROR] Pre-trained file missing at: {ckpt_path}\n      -> ⚠️ Fallback: Training completely from SCRATCH (Blank Brain)!")

    # Multi-GPU Logic
    if torch.cuda.device_count() > 1:
        print(f"🔥 Multi-GPU DETECTED: Distributing batch across {torch.cuda.device_count()} GPUs! 🔥")
        model = nn.DataParallel(model)
        
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])

    epoch_logs = []
    best_val_loss = float('inf')

    print(f"\n--- 🚀 Starting STREAMING Training ({CONFIG['epochs']} Epochs) ---")
    
    # ตัวแปรเลื่อนไหล (Shift Chunk)
    train_chunk_idx = 0
    random.shuffle(x_train_files)
    chunks_per_epoch = min(CONFIG.get("train_chunks_per_epoch", len(x_train_files)), len(x_train_files))
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n[ Epoch {epoch:03d}/{CONFIG['epochs']} ]")
        
        # 🚗 จัดทำลิสต์ก้อน Chunk สำหรับ Epoch นี้
        epoch_x_train = []
        for _ in range(chunks_per_epoch):
            if train_chunk_idx >= len(x_train_files):
                random.shuffle(x_train_files) 
                train_chunk_idx = 0
            epoch_x_train.append(x_train_files[train_chunk_idx])
            train_chunk_idx += 1
            
        # -----------------------------
        # TRAIN LOOP (Loading pre-made X_train, y_train)
        # -----------------------------
        model.train()
        train_loss_accum = 0.0
        train_sequences_seen = 0
        
        for i, x_file in enumerate(epoch_x_train):
            y_file = x_file.parent / x_file.name.replace("X_", "y_")
            print(f"  > [Train Chunk {i+1:02d}/{chunks_per_epoch:02d}] ⏳ Loading {x_file.name}... ", end="", flush=True)
            
            X_train = torch.load(x_file, weights_only=True)
            y_train = torch.load(y_file, weights_only=True)
            print(f"✅ Found {len(X_train)} seqs.")
            
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
            
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
            del X_train, y_train, dataset, loader 

        avg_train_loss = train_loss_accum / max(1, train_sequences_seen)

        # -----------------------------
        # VALIDATION LOOP 
        # -----------------------------
        model.eval()
        val_loss_accum = 0.0
        val_mae_x_acc = 0.0
        val_mae_y_acc = 0.0
        val_mre_x_acc = 0.0
        val_mre_y_acc = 0.0
        val_mse_x_acc = 0.0
        val_mse_y_acc = 0.0
        val_ade_accum = 0.0 
        val_sequences_seen = 0
        
        with torch.no_grad():
            for i, x_file in enumerate(x_val_files):
                y_file = x_file.parent / x_file.name.replace("X_", "y_")
                print(f"  > [Val Chunk   {i+1:02d}/{len(x_val_files):02d}] ⏳ Loading {x_file.name}... ", end="", flush=True)
                
                X_val = torch.load(x_file, weights_only=True)
                y_val = torch.load(y_file, weights_only=True)
                print(f"✅")
                
                dataset = torch.utils.data.TensorDataset(X_val, y_val)
                loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
                
                val_pbar = tqdm(loader, desc=f"    🔍 GPU Validating", leave=False, colour="blue")
                for batch_x, batch_y in val_pbar:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss_accum += loss.item() * batch_x.size(0)
                    
                    err_dx_m = outputs[:, 0] - batch_y[:, 0]
                    err_dy_m = outputs[:, 1] - batch_y[:, 1]
                    
                    val_mae_x_acc += torch.abs(err_dx_m).sum().item()
                    val_mae_y_acc += torch.abs(err_dy_m).sum().item()
                    
                    val_mse_x_acc += (err_dx_m**2).sum().item()
                    val_mse_y_acc += (err_dy_m**2).sum().item()
                    
                    mre_x = torch.abs(err_dx_m) / (torch.abs(batch_y[:, 0]) + 1e-4)
                    mre_y = torch.abs(err_dy_m) / (torch.abs(batch_y[:, 1]) + 1e-4)
                    val_mre_x_acc += mre_x.sum().item()
                    val_mre_y_acc += mre_y.sum().item()
                    
                    dist_err_m = torch.sqrt(err_dx_m**2 + err_dy_m**2)
                    val_ade_accum += dist_err_m.sum().item()
                    
                    val_sequences_seen += batch_x.size(0)
                
                val_pbar.close()
                del X_val, y_val, dataset, loader 

        n_seq = max(1, val_sequences_seen)
        avg_val_loss = val_loss_accum / n_seq
        avg_mae_x = val_mae_x_acc / n_seq
        avg_mae_y = val_mae_y_acc / n_seq
        avg_rmse_x = float(np.sqrt(val_mse_x_acc / n_seq))
        avg_rmse_y = float(np.sqrt(val_mse_y_acc / n_seq))
        avg_mre_x = val_mre_x_acc / n_seq
        avg_mre_y = val_mre_y_acc / n_seq
        avg_val_ade = val_ade_accum / n_seq 
        
        print(f"🎯 Epoch {epoch:03d} Result | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | 📏 ADE(MDE): {avg_val_ade:.3f} m")

        epoch_logs.append({
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss, 
            "val_ade_mde": avg_val_ade,
            "val_mae_x": avg_mae_x,
            "val_mae_y": avg_mae_y,
            "val_rmse_x": avg_rmse_x,
            "val_rmse_y": avg_rmse_y,
            "val_mre_x": avg_mre_x,
            "val_mre_y": avg_mre_y
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            CONFIG["best_val_loss"] = best_val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_lstm.pt")
            print("   -> 🌟 New Best Model Saved!")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pt")
        
        pd.DataFrame(epoch_logs).to_csv(OUTPUT_DIR / "train_log.csv", index=False)

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.6f}")
    
    # ===================================================================== #
    # 3. TEST LOOP
    # ===================================================================== #
    if x_test_files:
        print("\n--- Evaluating on Unseen TEST Set ---")
        model.load_state_dict(torch.load(OUTPUT_DIR / "best_lstm.pt", weights_only=True))
        model.eval()
        test_loss_accum = 0.0
        test_seq_seen = 0
        
        with torch.no_grad():
            for i, x_file in enumerate(x_test_files):
                y_file = x_file.parent / x_file.name.replace("X_", "y_")
                print(f"  > [Test Chunk  {i+1:02d}/{len(x_test_files):02d}] ⏳ Loading... ", end="", flush=True)
                
                X_test = torch.load(x_file, weights_only=True)
                y_test = torch.load(y_file, weights_only=True)
                print("✅")
                
                dataset = torch.utils.data.TensorDataset(X_test, y_test)
                loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
                
                test_pbar = tqdm(loader, desc=f"    🏆 GPU Testing   ", leave=False, colour="magenta")
                for batch_x, batch_y in test_pbar:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    loss = criterion(model(batch_x), batch_y)
                    test_loss_accum += loss.item() * batch_x.size(0)
                    test_seq_seen += batch_x.size(0)
                
                test_pbar.close()
                del X_test, y_test, dataset, loader
        
        final_test_mse = test_loss_accum / max(1, test_seq_seen)
        print(f"Final Test MSE Loss: {final_test_mse:.6f}")
        CONFIG["final_test_loss"] = final_test_mse

    torch.save(model.state_dict(), OUTPUT_DIR / "last_lstm.pt")
    with open(OUTPUT_DIR / "train_config.json", 'w') as f:
        json.dump(CONFIG, f, indent=4)
        
    print(f"\nAll weights and configs successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

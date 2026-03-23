"""
Topo2_Test.py

# เป้าหมาย
สคริปต์สำหรับทดสอบประสิทธิภาพโมเดลหลังจากเทรนเสร็จแล้ว
โดยใช้ข้อมูลชุด TEST (ที่ AI ไม่เคยเห็นมาก่อน) 
เพื่อวัดค่าความแม่นยำ (ADE/MAE/RMSE) ออกมาเป็นตัวเลขสถิติ
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import gc

# ===================================================================== #
# CONFIGURATION
# ===================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "AI_Train" / "dataswarm_processed" / "topo2" /"test"
MODEL_PATH = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2" / "best_lstm.pt"
OUTPUT_DIR = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2" / "test_results"

CONFIG = {
    "batch_size": 2048,
    "hidden_size": 256,
    "num_layers": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "feature_cols_count": 9,
    "target_cols_count": 2
}

# ===================================================================== #
# 1. MODEL ARCHITECTURE (ต้องตรงกับไฟล์เทรน)
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
# 2. EVALUATION ROUTINE
# ===================================================================== #

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG["device"])
    
    print(f"\n--- 🧪 Starting AI Evaluation on TEST SET ({device}) ---")

    # 1. โหลดโมเดล
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return

    model = LSTM_Baseline(
        input_size=CONFIG["feature_cols_count"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=CONFIG["target_cols_count"]
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    
    # 🩹 FIX: แก้ปัญหา DataParallel (module. prefix)
    if any(k.startswith('module.') for k in checkpoint.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # ตัด 'module.' ออก
            new_state_dict[name] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Successfully Loaded [best_lstm.pt]")

    # 2. ค้นหาไฟล์ข้อมูลชุด Test
    test_files = sorted(list(PROCESSED_DIR.glob("X_test_*.pt")))
    if not test_files:
        print(f"❌ ERROR: No test tensors found in {PROCESSED_DIR}")
        return
    print(f"Found {len(test_files)} test chunks to evaluate.")

    # 3. เตรียมตัวแปรเก็บผลลัพธ์
    all_preds = []
    all_trues = []
    
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for i, x_file in enumerate(test_files):
            y_file = x_file.parent / x_file.name.replace("X_", "y_")
            print(f"  > Processing {x_file.name}... ", end="", flush=True)
            
            X_test_chunk = torch.load(x_file, weights_only=True).to(device)
            y_test_chunk = torch.load(y_file, weights_only=True).to(device)
            
            outputs = model(X_test_chunk)
            loss = criterion(outputs, y_test_chunk)
            total_loss += loss.item() * X_test_chunk.size(0)
            
            # เก็บผลไว้คำนวณสถิติละเอียด
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(y_test_chunk.cpu().numpy())
            
            # 🧹 เคลียร์แรมก้อนนี้ทิ้งทันที
            del X_test_chunk, y_test_chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅")

    # 4. คำนวณ Error Metrics
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    n_samples = len(trues)
    
    # คำนวณความต่างแต่ละแกน (เมตร)
    err_dx = preds[:, 0] - trues[:, 0]
    err_dy = preds[:, 1] - trues[:, 1]
    
    # ADE (Average Displacement Error) หรือ Mean Distance Error (MDE)
    dist_err = np.sqrt(err_dx**2 + err_dy**2)
    ade = np.mean(dist_err)
    
    # MAE (Mean Absolute Error)
    mae_x = np.mean(np.abs(err_dx))
    mae_y = np.mean(np.abs(err_dy))
    
    # RMSE (Root Mean Square Error)
    rmse_x = np.sqrt(np.mean(err_dx**2))
    rmse_y = np.sqrt(np.mean(err_dy**2))
    
    # 5. สรุปผลลัพธ์
    results_summary = {
        "Total_Samples": int(n_samples),
        "Final_Test_MSE": float(total_loss / n_samples),
        "ADE_MDE_meters": float(ade),
        "MAE_X_meters": float(mae_x),
        "MAE_Y_meters": float(mae_y),
        "RMSE_X_meters": float(rmse_x),
        "RMSE_Y_meters": float(rmse_y)
    }

    print("\n" + "="*50)
    print("🏆 FINAL TEST RESULTS (UNSEEN DATA)")
    print("="*50)
    print(f"📊 Total Sequences Tested: {n_samples:,}")
    print(f"📏 ADE (Mean Distance Error): {ade:.4f} meters")
    print(f"🟦 MAE (X Axis):            {mae_x:.4f} meters")
    print(f"🟪 MAE (Y Axis):            {mae_y:.4f} meters")
    print(f"📉 RMSE (X Axis):           {rmse_x:.4f} meters")
    print(f"📉 RMSE (Y Axis):           {rmse_y:.4f} meters")
    print("="*50)

    # 6. บันทึกลงไฟล์
    with open(OUTPUT_DIR / "final_test_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    # 7. วาดกราฟเปรียบเทียบ (Scatter Plot)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(trues[:1000, 0], preds[:1000, 0], alpha=0.3, color='blue')
    plt.plot([-0.2, 0.2], [-0.2, 0.2], '--k')
    plt.title("Target DX: True vs Pred")
    plt.xlabel("Actual (m)")
    plt.ylabel("Predicted (m)")

    plt.subplot(1, 2, 2)
    plt.scatter(trues[:1000, 1], preds[:1000, 1], alpha=0.3, color='purple')
    plt.plot([-0.2, 0.2], [-0.2, 0.2], '--k')
    plt.title("Target DY: True vs Pred")
    plt.xlabel("Actual (m)")
    plt.ylabel("Predicted (m)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_scatter_plot.png")
    print(f"\n🎨 Results and Visualization saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

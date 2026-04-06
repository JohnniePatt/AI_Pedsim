"""
AI_Train/Method_RL/config_rl.py

# เป้าหมาย
เก็บค่า Hyperparameters ทั้งหมดของ Reinforcement Learning
"""

from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_bottleneck"

CONFIG = {
    # 🎯 ตัวที่ใช้เทรน (ระบุไฟล์เดียวตามต้องการ)
    "train_file": TOPO_DIR / "dataswarm" / "test" / "double-botteleneck_100801.sqlite",
    "spawn_exit_csv": TOPO_DIR / "spawn_exit_area" / "test" / "spawn_exit_100801.csv",
    
    # 🧠 โมเดล
    "seq_len": 20,
    "input_size": 14,         # (x, y) + (goal_dx, goal_dy) + 5 * (neighbor_dx, neighbor_dy)
    "num_neighbors": 5,       # จำนวนคนที่เรดาร์มองเห็น
    "hidden_size": 128,
    "num_layers": 2,
    "lr": 3e-4,
    
    # 🏃 การเรียนรู้ (RL)
    "max_episodes": 1000,     # ซ้อมกี่ครั้ง (รอบ)
    "max_gen_frames": 1000,   # เดินได้นานสุดกี่เฟรม
    "gamma": 0.99,            # มองการณ์ไกล (ยิ่งเยอะ ยิ่งวางแผนยาว)
    "eps_clip": 0.2,          # ขอบเขตการปรับสมองไม่ให้เปลี่ยนเร็วเกินไป
    
    # 💻 ฮาร์ดแวร์
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

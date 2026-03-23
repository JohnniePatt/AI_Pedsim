"""
AI_Train/Method_RL/train_rl.py (THE ENTRY POINT)

# เป้าหมาย
โค้ดสำหรับสั่งรุกคืบ (Train) AI ด้วยเทคนิค PPO (RL)
โฟกัสไฟล์เดียวเพื่อรีด Accuracy ให้สูงสุด!
"""

import os
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# โหลดพวกเราเอง
from config_rl import CONFIG
from vir_pedsim import PedsimRL_Env
from model_rl import ActorCritic

# ฟังก์ชันเสริมสำหรับโหลด Exit
from shapely.wkt import loads as load_wkt
def load_exit_polygon(csv_path: Path):
    df = pd.read_csv(csv_path)
    return load_wkt(df[df['type'] == 'exit_area'].iloc[0]['area'])

def main():
    device = torch.device(CONFIG["device"])
    print(f"🛠️  Using Device: {device}")

    # 1. เตรียมแผนที่ (Geometry)
    # เราขอเอา Geometry จากด่านหลัก (Topo_2) มาใช้
    BASE_DIR = Path(__file__).resolve().parent.parent.parent / "Topo_2" / "geo"
    with open(BASE_DIR / "geo_room.json", 'r') as f: room_data = json.load(f)
    with open(BASE_DIR / "geo_corridor.json", 'r') as f: corridor_data = json.load(f)
    from shapely.geometry import Polygon
    room_polys = [Polygon(p) for p in room_data]
    corridor_polys = [Polygon(p) for p in corridor_data]

    # 2. เตรียม Env และ Model
    env = PedsimRL_Env(CONFIG, room_polys, corridor_polys)
    exit_centroid = load_exit_polygon(CONFIG["spawn_exit_csv"]).centroid
    
    # สมอง AI (14 input Features จากเรดาร์ค้นหา 5 คนแรก)
    model = ActorCritic(input_size=CONFIG["input_size"], hidden_size=CONFIG["hidden_size"], 
                        num_layers=CONFIG["num_layers"], output_size=2).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    # 🟢 เพิ่ม Scheduler เพื่อให้สมอง "นิ่งขึ้น" เมื่อเวลาผ่านไป (ลดการลองเสี่ยงดวงมั่วๆ)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    
    # 3. เริ่มซ้อม (The Training Loop)
    print(f"🎬 Starting Training for {CONFIG['max_episodes']} episodes...")
    for episode in range(1, CONFIG["max_episodes"] + 1):
        state = env.reset(CONFIG["train_file"], exit_centroid).to(device)
        
        # Buffer สำหรับเก็บข้อมูลใน 1 รอบ (Rollout)
        states, actions, log_probs, rewards, values, masks = [], [], [], [], [], []
        total_reward = 0
        
        for f in range(CONFIG["max_gen_frames"]):
            # 🚶 AI ตัดสินใจ (Actor-Critic)
            # x: (1, 20, 2)
            action, log_prob, value = model.act(state.unsqueeze(0))
            
            # 🦶 ก้าวจิงๆ ใน Env
            next_state_np, reward, done = env.step(action[0].cpu().numpy())
            next_state = next_state_np.to(device)
            
            # เก็บข้อมูล
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            masks.append(1 - done)
            
            total_reward += reward
            state = next_state
            
            if done: break
            
        # 🟢 เมื่อจบ 1 รอบ (Episode) -> ปรับสมอง (Gradient Descent)
        # (ใน RL เราจะคำนวณ Error แบบ PPO หรือ Advantage)
        # เพื่อความง่าย ผมเขียนแบบ REINFORCE (Policy Gradient) ให้ก่อนเพื่อให้เห็นภาพครับ
        # -------------------------------------------------------------
        optimizer.zero_grad()
        # คำนวณ Discounted Rewards
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(device)
        # 🟢 ตัวนี้คือหัวใจสำคัญของ RL! ถ้าไม่ Normalize ค่า R ที่ติดลบมากๆ สมองจะรวนทันที
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        policy_loss = []
        for lp, R_norm in zip(log_probs, returns):
            policy_loss.append(-lp * R_norm) # ลบเพราะเราต้องการ maximize
            
        final_loss = torch.stack(policy_loss).sum()
        final_loss.backward()
        optimizer.step()
        scheduler.step() # 🟢 อัปเดตความนิ่งของสมอง (ลดการสุ่มมั่ว)
        # -------------------------------------------------------------
        
        if episode % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   🌟 Ep {episode:04d} | Frames: {len(rewards):04d} | Score: {total_reward:.1f} | LR: {lr:.6f}")

    # 4. เซฟสมองที่เก่งขึ้นแล้วทิ้งไว้
    save_path = Path(__file__).resolve().parent / "best_rl_brain.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 ซ้อมเสร็จแล้ว! สมองถูกเซฟไว้ที่: {save_path.name}")

if __name__ == "__main__":
    main()

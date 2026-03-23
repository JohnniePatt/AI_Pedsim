"""
AI_Train/Method_RL/env_pedsim.py

# เป้าหมาย
คลาสจำลองสถานการณ์ (Environment) สำหรับการเดินของคนเดินเท้า
รองศรับ Reinforcement Learning โดยมีระบบวัดผล (Reward) และสถานะ (State)
"""

import numpy as np
import sqlite3
import pandas as pd
import torch
from pathlib import Path
from shapely.geometry import Point, Polygon
import json

class PedsimRL_Env:
    def __init__(self, config, room_polys, corridor_polys):
        self.config = config
        self.room_polys = room_polys
        self.corridor_polys = corridor_polys
        
        # สถานะปัจจุบัน
        self.current_pos = None
        self.current_frame = 0
        self.history = None # สำหรับเก็บ 20 เฟรมย้อนหลัง (State)
        self.exit_centroid = None
        self.meta = None
        
        # ข้อมูลเฉลย (Ground Truth) สำหรับเทียบ
        self.all_agents_df = None
        self.gt_df = None
        
    def reset(self, sqlite_path, exit_poly_centroid):
        """ล้างหน้ากระดาน เริ่มเดินใหม่จากจุดเริ่มต้นของไฟล์ SQLite"""
        self.exit_centroid = exit_poly_centroid
        
        # 🟢 โหลดข้อมูลเข้า RAM แค่ "ครั้งแรกครั้งเดียว" (Memory Cache)
        # ป้องกันอาการ RAM บวมและ CPU ทำงานหนักจากการโหลดไฟล์ซ้ำๆ ทุก Episode
        if self.all_agents_df is None:
            conn = sqlite3.connect(sqlite_path)
            self.meta = { k: float(conn.execute("SELECT value FROM metadata WHERE key = ?", (k,)).fetchone()[0]) for k in ['xmin', 'xmax', 'ymin', 'ymax']}
            self.all_agents_df = pd.read_sql_query("SELECT frame, id, pos_x, pos_y FROM trajectory_data", conn)
            conn.close()
            
            # สร้าง Dictionary สำหรับทำ Radar ล่วงหน้าแค่ครั้งเดียว
            self.frame_to_agents = {}
            for row in self.all_agents_df.itertuples():
                f = row.frame
                if f not in self.frame_to_agents:
                    self.frame_to_agents[f] = []
                self.frame_to_agents[f].append((row.id, row.pos_x, row.pos_y))
        
        # โฟกัสสอน ID แรกสุด (MIN) คนเดียวให้เก่งก่อน ตามแผน Overfitting
        self.tracked_id = self.all_agents_df['id'].min()
        self.gt_df = self.all_agents_df[self.all_agents_df['id'] == self.tracked_id].sort_values('frame')
        
        # เริ่มต้นก้าวแรกหลังจากมีประวัติ 20 อันแรก
        seed_data = self.gt_df.iloc[:self.config['seq_len']]
        self.current_frame = seed_data['frame'].iloc[-1] + 1
        
        self.current_pos = np.array([seed_data['pos_x'].iloc[-1], seed_data['pos_y'].iloc[-1]])
        
        # เตรียมประวัติ (Buffer สำหรับ LSTM)
        self.history = seed_data[['pos_x', 'pos_y']].values.tolist()
        
        return self._get_observation()

    def step(self, action):
        """ก้าวเดิน 1 ครั้ง (Action คือการเลื่อนที่ Dx, Dy)"""
        # 1. 🚶 AI ก้าวขา พร้อมป้องกันไม่ให้มันบ้าพลังก้าวกระโดดเกิน 1 เมตรต่อเฟรม
        dx, dy = np.clip(action[0], -1.0, 1.0), np.clip(action[1], -1.0, 1.0)
        self.current_pos[0] += dx
        self.current_pos[1] += dy
        self.current_frame += 1
        
        # 2. 🗺️ เช็คสถานะแผนที่
        p = Point(self.current_pos[0], self.current_pos[1])
        in_area = any(poly.contains(p) for poly in self.room_polys + self.corridor_polys)
        
        # 3. 🎯 คำนวณรางวัล (REWARD LOGIC - โฟกัสการเดินตามเฉลยเป๊ะๆ)
        reward = 0.5 # 🟢 โบนัสการอยู่รอด (Survival Bonus) พื้นฐานทุกก้าว
        done = False
        
        # - เช็คว่าเฉลยในเฟรมนี้อยู่ตรงไหน
        gt_row = self.gt_df[self.gt_df['frame'] == self.current_frame]
        
        if not gt_row.empty:
            gt_pos = np.array([gt_row['pos_x'].iloc[0], gt_row['pos_y'].iloc[0]])
            dist_to_gt = np.linalg.norm(self.current_pos - gt_pos)
            
            # --- 🟢 ปรับกติกายางยืด (Elastic Reward) ---
            if dist_to_gt < 0.5:
                # ยิ่งใกล้เฉลย ยิ่งได้แต้มบวก (สูงสุด +5.0 ถ้าทับรอยเป๊ะ)
                reward += (0.5 - dist_to_gt) * 10.0
            else:
                # ถ้าห่างเกิน 0.5 เมตร ถึงจะเริ่มหักแต้มจุกจิก
                reward -= dist_to_gt 
                
            # ถ้าเดินหลงทิศ ห่างจากเฉลยฉีกออกไปเกิน 2 เมตร -> สั่งจบเกม (เริ่มใหม่) ทันที!
            if dist_to_gt > 2.0:
                reward -= 50.0
                done = True
        else:
            # แปลว่า AI เดินแกะรอยตามจนจบไฟล์เฉลยแล้ว! (ทำภารกิจสำเร็จ 100%)
            reward += 200.0
            done = True
            
        # - บทลงโทษถ้าเดินตกขอบแผนที่ / เดินชนกำแพง
        if not in_area:
            reward -= 10.0 # โดนหักแต้มเจ็บระบม แต่ยอมให้เดินทะลุกำแพงได้ (ไม่สั่งจบเกม) 

        # 4. อัปเดตประวัติ
        self.history.append(self.current_pos.tolist())
        self.history.pop(0)

        if self.current_frame >= self.config['max_gen_frames']:
            done = True

        return self._get_observation(), reward, done

    def _get_observation(self):
        """แปลงประวัติการเดินให้เป็นรูปแบบที่สมอง (LSTM) เข้าใจ พร้อมระบบเรดาร์ (Neighbors)"""
        w = self.meta['xmax'] - self.meta['xmin']
        h = self.meta['ymax'] - self.meta['ymin']
        domain_diag = np.sqrt(w**2 + h**2)
        if domain_diag == 0: domain_diag = 100.0

        obs_list = []
        # ต้องประมวลผลย้อนหลัง 20 เฟรมตามประวัติ (history)
        start_frame = self.current_frame - len(self.history)
        
        for i, (hx, hy) in enumerate(self.history):
            frame_idx = start_frame + i
            
            # 1. พิกัดตัวเอง (Normalized)
            x_norm = (hx - self.meta['xmin']) / w
            y_norm = (hy - self.meta['ymin']) / h
            
            # 2. ระยะกระจัดไปหาทางออก (Goal Vector)
            goal_dx_norm = (self.exit_centroid.x - hx) / w
            goal_dy_norm = (self.exit_centroid.y - hy) / h
            
            # 3. ระบบเรดาร์ค้นหาพิกัดเพื่อนบ้านที่ใกล้ที่สุด (Nearest Neighbors)
            neighbors_feat = []
            
            # ดึงข้อมูลทุกคนในเฟรมนั้นๆ ด้วย Dictionary (เร็วปรู๊ดปร๊าด)
            agents_in_frame = self.frame_to_agents.get(frame_idx, [])
            other_agents = [a for a in agents_in_frame if a[0] != self.tracked_id]
            
            if other_agents:
                other_pos = np.array([[a[1], a[2]] for a in other_agents])
                dists = np.linalg.norm(other_pos - [hx, hy], axis=1)
                
                # เรียงลำดับจากคนใกล้ที่สุดไปไกลที่สุด
                idx = np.argsort(dists)[:self.config.get('num_neighbors', 5)]
                
                for k in idx:
                    ndx = (other_pos[k][0] - hx) / w
                    ndy = (other_pos[k][1] - hy) / h
                    neighbors_feat.extend([ndx, ndy])
                    
            # ถ้าอยู่ในห้องโล่งๆ แล้วเจอคนไม่ถึง 5 คน ให้เอาศูนย์ (0.0) มาเติมให้ครบ (Padding)
            expected_n_len = self.config.get('num_neighbors', 5) * 2
            if len(neighbors_feat) < expected_n_len:
                neighbors_feat.extend([0.0] * (expected_n_len - len(neighbors_feat)))
                
            # รวมสายตาทั้งหมดของเฟรมนี้: [self_x, self_y, goal_dx, goal_dy, n1x, n1y, n2x, n2y...] เป็น 14 มิติ
            frame_obs = [x_norm, y_norm, goal_dx_norm, goal_dy_norm] + neighbors_feat
            obs_list.append(frame_obs)

        # รวมประวัติความจำทั้งหมดเข้าด้วยกันเป็น Tensor (20 ก้าว, 14 มิติ)
        return torch.tensor(obs_list, dtype=torch.float32)

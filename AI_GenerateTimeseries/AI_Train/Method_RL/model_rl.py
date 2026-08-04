"""
AI_Train/Method_RL/model_rl.py

# เป้าหมาย
สมอง AI แบบ Actor-Critic (PPO Compatible)
- Actor: ตัดสินใจทิศทางการเดิน (Mean/Sigma)
- Critic: ประเมินมูลค่าของตำแหน่งที่ยืนอยู่ (Value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ActorCritic, self).__init__()
        
        # 1. Shared Brain (LSTM): จำจังหวะก้าวก่อนหน้า
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 2. Actor Head: ส่วนตัดสินใจ (จะเดินไปไหน Dx, Dy)
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size * 2) # *2 สำหรับ Mean และ Sigma
        )
        
        # 3. Critic Head: ส่วนวิจารณ์ (ดีไม่ดียังไง)
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1) # คืนค่าเป็นคะแนนความพอใจ (Scalar)
        )
        
    def forward(self, x):
        # x: (Batch, Seql_len, Features)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] # เอาผลลัพธ์จากเฟรมสุดท้าย
        
        # Actor: ทายพิกัดก้าว (Mean และ Log-Sigma)
        mu_sigma = self.actor_fc(last_out)
        
        # 🟢 บีบให้ก้าวเดินต่อ 1 เฟรม ไม่เกิน 0.2 เมตร (แทนที่จะเหาะไปเป็นเมตร)
        mu = torch.tanh(mu_sigma[:, :2]) * 0.2 
        
        # 🟢 ป้องกัน Sigma สุ่มวงกว้างเกินไปตอนที่โมเดลยังโง่อยู่
        log_sigma = torch.clamp(mu_sigma[:, 2:], -20, -1)
        sigma = torch.exp(log_sigma)
        
        # Critic: ประเมินค่า
        value = self.critic_fc(last_out)
        
        return mu, sigma, value

    def act(self, x):
        """สำหรับสุ่มเลือกก้าว (Exploration) ระหว่างเทรน"""
        mu, sigma, value = self.forward(x)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_log_prob, value

    def evaluate(self, x, action):
        """สำหรับคำนวณ Error ระหว่างอัปเดตสมอง"""
        mu, sigma, value = self.forward(x)
        dist = Normal(mu, sigma)
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_log_prob, value, dist_entropy

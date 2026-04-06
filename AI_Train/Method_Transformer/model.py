import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class GeoEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Input: [B, 1, 128, 128]
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # [B, 128, 1, 1]
            nn.Flatten(),
            nn.Linear(128, d_model)
        )
    def forward(self, x):
        return self.net(x)

class GoalConditionedGPT2(nn.Module):
    """
    GPT-2 Architecture with Geo Mask and Goal Context for continuous trajectory generation.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, max_seq_len=60):
        super(GoalConditionedGPT2, self).__init__()
        
        config = GPT2Config(
            n_embd=d_model,
            n_layer=num_layers,
            n_head=nhead,
            n_positions=max_seq_len + 10,
            n_ctx=max_seq_len + 10,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            use_cache=False 
        )
        self.gpt2 = GPT2Model(config)
        
        self.geo_enc = GeoEncoder(d_model)
        self.cond_proj = nn.Linear(4, d_model)
        self.context_fuse = nn.Linear(d_model * 2, d_model) # fuse geo + goal
        
        self.input_proj = nn.Linear(2, d_model)
        self.output_head = nn.Linear(d_model, 2)

    def forward(self, obs_traj, start_pt, end_pt, geo_mask, pred_len=10):
        """
        obs_traj: [Batch, obs_len, 2]
        start_pt: [Batch, 2], end_pt: [Batch, 2]
        geo_mask: [Batch, 1, 128, 128]
        pred_len: number of frames to predict autoregressively.
        """
        bs = obs_traj.shape[0]
        
        # 1. Encode Context
        geo_feat = self.geo_enc(geo_mask) # [B, d_model]
        goal_feat = self.cond_proj(torch.cat([start_pt, end_pt], dim=-1)) # [B, d_model]
        
        context_embeds = self.context_fuse(torch.cat([geo_feat, goal_feat], dim=-1)).unsqueeze(1) # [B, 1, d_model]
        
        # 2. Autoregressive loop
        current_seq = obs_traj
        predictions = []
        
        for step in range(pred_len):
            # Project current sequence
            inputs_embeds = self.input_proj(current_seq) # [B, SeqLength, d_model]
            
            # Combine Context + Sequence
            # Sequence: [Context, p1, p2, ..., p_current]
            full_embeds = torch.cat([context_embeds, inputs_embeds], dim=1)
            
            # GPT2 Forward
            outputs = self.gpt2(inputs_embeds=full_embeds)
            last_hidden_state = outputs.last_hidden_state # [B, 1 + SeqLength, d_model]
            
            # Predict next point from the last token's embeddings
            next_pt = self.output_head(last_hidden_state[:, -1, :]) # [B, 2]
            
            # Save prediction
            predictions.append(next_pt)
            
            # Append prediction to current sequence for next step
            # next_pt has shape [B, 2]. Add a dimension to make it [B, 1, 2]
            current_seq = torch.cat([current_seq, next_pt.unsqueeze(1)], dim=1)
            
        # Return all predictions as [B, pred_len, 2]
        return torch.stack(predictions, dim=1)

if __name__ == "__main__":
    # Test model forward pass
    model = GoalConditionedGPT2()
    dummy_traj = torch.randn(2, 20, 2) # obs_len = 20
    dummy_start = torch.randn(2, 2)
    dummy_end = torch.randn(2, 2)
    dummy_mask = torch.ones(2, 1, 128, 128)
    
    out = model(dummy_traj, dummy_start, dummy_end, dummy_mask, pred_len=10)
    print(f"Output shape: {out.shape}") # Expected [2, 10, 2]

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=128):
        super(Encoder, self).__init__()
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, 1, batch_first=False)

    def forward(self, obs_rel_traj):
        # obs_rel_traj shape: (obs_len, batch, 2)
        emb = nn.functional.relu(self.spatial_embedding(obs_rel_traj))
        # emb shape: (obs_len, batch, emb_dim)
        output, (hn, cn) = self.encoder(emb)
        return hn.squeeze(0), cn.squeeze(0)  # shape: (batch, h_dim)

class SocialPooling(nn.Module):
    def __init__(self, h_dim=128, pool_dim=16):
        super(SocialPooling, self).__init__()
        # Simplified Social Pooling: Max pooling over encoded hidden states
        self.pool_embedding = nn.Linear(h_dim, pool_dim)
        
    def forward(self, seq_start_end, h_states):
        # h_states shape: (total_batch_peds, h_dim)
        pool_h = torch.zeros((h_states.shape[0], self.pool_embedding.out_features), device=h_states.device)
        for (start, end) in seq_start_end:
            scene_h = h_states[start:end] # (peds_in_scene, h_dim)
            if scene_h.shape[0] > 1:
                # max pooling over pedestrians in scene
                pooled = torch.max(scene_h, dim=0, keepdim=True)[0] 
                # assign to all
                pool_h[start:end] = self.pool_embedding(pooled.expand_as(scene_h))
        return pool_h

class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=128, pool_dim=16, seq_len=12):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim
        
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, h_dim, 1, batch_first=False)
        self.social_pooling = SocialPooling(h_dim, pool_dim)
        
        # Predict relative coordinate (dx, dy)
        self.hidden2pos = nn.Linear(h_dim + pool_dim, 2)

    def forward(self, last_obs, last_h, last_c, seq_start_end):
        # last_obs shape: (batch, 2) - usually the last relative step, or zero vector to kickstart
        preds = []
        curr_obs = last_obs
        h, c = last_h.unsqueeze(0), last_c.unsqueeze(0)
        
        for t in range(self.seq_len):
            emb = nn.functional.relu(self.spatial_embedding(curr_obs)).unsqueeze(0) # (1, batch, emb_dim)
            output, (h, c) = self.decoder(emb, (h, c))
            
            # extract hidden state (batch, h_dim)
            h_curr = h.squeeze(0)
            
            # apply social pooling
            pool_h = self.social_pooling(seq_start_end, h_curr)
            
            # concatenate and predict
            mlp_in = torch.cat([h_curr, pool_h], dim=1)
            pred_rel_pos = self.hidden2pos(mlp_in) # (batch, 2)
            preds.append(pred_rel_pos)
            
            curr_obs = pred_rel_pos
            
        # preds shape: (seq_len, batch, 2)
        return torch.stack(preds, dim=0)

class TrajectoryGenerator(nn.Module):
    def __init__(self, emb_dim=64, h_dim=128, pool_dim=16, obs_len=8, pred_len=12):
        super(TrajectoryGenerator, self).__init__()
        self.encoder = Encoder(emb_dim, h_dim)
        self.decoder = Decoder(emb_dim, h_dim, pool_dim, pred_len)

    def forward(self, obs_rel_traj, seq_start_end):
        h, c = self.encoder(obs_rel_traj)
        last_obs = obs_rel_traj[-1] # last step
        pred_rel_traj = self.decoder(last_obs, h, c, seq_start_end)
        return pred_rel_traj

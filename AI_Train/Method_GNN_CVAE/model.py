"""
model.py
--------
Custom GNN-CVAE baseline for multi-agent trajectory generation.

Key design choices
------------------
- Goal-anchored decoding: each predicted position is
    pos_t = linear_interp(start, goal, t) + learned_residual
  This eliminates delta-accumulation error (the main cause of large FDE).
- OOB soft loss: differentiable penalty via F.grid_sample on the geo_mask,
  so the model learns to stay inside walkable area.
- Proper hidden-state initialisation via init_mlp (no hack padding).
- KL annealing is handled in the training script.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, geo_mask: torch.Tensor) -> torch.Tensor:
        return self.net(geo_mask)


class SocialGNN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        hidden: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        bsz, n_agents, _ = positions.shape
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel  = pos_j - pos_i                                  # [B, N, N, 2]
        dist = torch.norm(rel, dim=-1, keepdim=True)          # [B, N, N, 1]

        h_j     = hidden.unsqueeze(1).expand(-1, n_agents, -1, -1)
        msg_in  = torch.cat([h_j, rel, dist], dim=-1)
        messages = self.msg_mlp(msg_in)

        active_pair = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
        eye = torch.eye(n_agents, device=positions.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = active_pair & (~eye) & (dist.squeeze(-1) <= radius)

        masked_messages = messages * edge_mask.unsqueeze(-1).float()
        denom  = edge_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        pooled = masked_messages.sum(dim=2) / denom
        pooled = pooled * active_mask.unsqueeze(-1).float()
        return self.out_norm(pooled)


class GNNCVAE(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        geo_dim: int = 64,
        gnn_layers: int = 2,
        neighbor_radius: float = 0.08,
        dropout: float = 0.1,
        use_social: bool = True,
        oob_weight: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.latent_dim      = latent_dim
        self.geo_dim         = geo_dim
        self.gnn_layers      = gnn_layers
        self.neighbor_radius = neighbor_radius
        self.use_social      = use_social
        self.oob_weight      = oob_weight

        self.geo_encoder  = GeoEncoder(geo_dim)
        self.agent_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head  = nn.Linear(hidden_dim, latent_dim)
        self.geo_to_hidden = nn.Linear(geo_dim, hidden_dim)   # encoder: small geo bias

        # ── Proper decoder initialisation ──────────────────────────────────
        # Input: start(2) + goal(2) + latent(latent_dim) + geo(geo_dim)
        self.init_mlp = nn.Sequential(
            nn.Linear(2 + 2 + latent_dim + geo_dim, hidden_dim),
            nn.Tanh(),
        )

        self.social_layers = nn.ModuleList(
            [SocialGNN(hidden_dim) for _ in range(gnn_layers)]
        )

        # GRU input: current(2)+start(2)+goal(2)+latent(ld)+geo(gd)+social(hd)+time(1)
        gru_in = 2 + 2 + 2 + latent_dim + geo_dim + hidden_dim + 1
        self.decoder_cell = nn.GRUCell(gru_in, hidden_dim)

        # ── Position head: predicts RESIDUAL from linear anchor ─────────────
        # Replaces the old delta_head.  Output is NOT a delta from current
        # position; it is a deviation from the straight-line interpolation
        # between start and goal.  This eliminates accumulation drift.
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        return (x * mask.unsqueeze(-1).float()).sum(dim=-2) / denom

    def _last_valid(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=-1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.shape[-1])
        return torch.gather(x, dim=2, index=idx).squeeze(2)

    # ── Encoder ──────────────────────────────────────────────────────────────

    def encode_latent(
        self,
        positions:  torch.Tensor,
        agent_mask: torch.Tensor,
        start_pt:   torch.Tensor,
        goal_pt:    torch.Tensor,
        geo_feat:   torch.Tensor,
    ):
        deltas     = positions[:, :, 1:] - positions[:, :, :-1]
        delta_mask = agent_mask[:, :, 1:] & agent_mask[:, :, :-1]
        mean_pos   = self._masked_mean(positions, agent_mask)
        mean_vel   = self._masked_mean(deltas, delta_mask)
        final_pos  = self._last_valid(positions, agent_mask)
        duration   = (agent_mask.sum(dim=-1, keepdim=True).float()
                      / agent_mask.shape[-1]).clamp(max=1.0)

        features   = torch.cat([start_pt, goal_pt, final_pos, mean_pos, mean_vel, duration], dim=-1)
        hidden     = self.agent_encoder(features)
        geo_expand = geo_feat.unsqueeze(1).expand(-1, positions.shape[1], -1)
        hidden     = hidden + 0.1 * self.geo_to_hidden(geo_expand)
        mu     = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Decoder ──────────────────────────────────────────────────────────────

    def decode(
        self,
        start_pt:          torch.Tensor,
        goal_pt:           torch.Tensor,
        geo_feat:          torch.Tensor,
        latent_z:          torch.Tensor,
        seq_len:           int,
        teacher_positions: torch.Tensor | None = None,
        agent_mask:        torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_agents, _ = start_pt.shape
        device    = start_pt.device
        geo_expand = geo_feat.unsqueeze(1).expand(-1, n_agents, -1)   # [B,N,geo_dim]

        # Proper initialisation
        init_in = torch.cat([start_pt, goal_pt, latent_z, geo_expand], dim=-1)
        h = self.init_mlp(init_in)                                     # [B, N, hidden_dim]

        preds   = [start_pt]
        current = start_pt

        for step in range(seq_len - 1):
            if teacher_positions is not None:
                current = teacher_positions[:, :, step]

            active_mask = (
                agent_mask[:, :, step]
                if agent_mask is not None
                else torch.ones(bsz, n_agents, dtype=torch.bool, device=device)
            )

            # Social context
            social = torch.zeros_like(h)
            if self.use_social:
                h_s = h
                for layer in self.social_layers:
                    s = layer(current, active_mask, h_s, self.neighbor_radius)
                    h_s  = h_s + s
                    social = social + s

            time_scalar = torch.full(
                (bsz, n_agents, 1),
                float(step) / max(seq_len - 1, 1),
                device=device,
            )
            decoder_in = torch.cat(
                [current, start_pt, goal_pt, latent_z, geo_expand, social, time_scalar],
                dim=-1,
            )

            flat_in  = decoder_in.reshape(bsz * n_agents, -1)
            flat_h   = h.reshape(bsz * n_agents, -1)
            updated_h = self.decoder_cell(flat_in, flat_h).view(bsz, n_agents, self.hidden_dim)
            active_f  = active_mask.unsqueeze(-1).float()
            h = updated_h * active_f + h * (1.0 - active_f)

            # ── Goal-anchored absolute position ──────────────────────────────
            # progress goes 1/(T-1) … 1.0 so the anchor ends exactly at goal
            progress      = (step + 1) / max(seq_len - 1, 1)
            linear_anchor = start_pt + progress * (goal_pt - start_pt)

            goal_delta = goal_pt - current
            residual   = self.pos_head(torch.cat([h, social, goal_delta], dim=-1))
            next_pos   = linear_anchor + residual

            if agent_mask is not None:
                next_mask = agent_mask[:, :, step + 1].unsqueeze(-1).float()
                next_pos  = next_pos * next_mask

            preds.append(next_pos)
            if teacher_positions is None:
                current = next_pos

        return torch.stack(preds, dim=2)   # [B, N, T, 2]

    # ── Loss ─────────────────────────────────────────────────────────────────

    def _oob_loss(
        self,
        pred_positions: torch.Tensor,   # [B, N, T, 2]  normalised [0,1]
        agent_mask:     torch.Tensor,   # [B, N, T]  bool
        geo_mask:       torch.Tensor,   # [B, 1, H, W]  float {0,1}
    ) -> torch.Tensor:
        """
        Differentiable out-of-bounds penalty via bilinear sampling of geo_mask.
        geo_val ≈ 1 → walkable, 0 → wall.  Loss = mean(1 - geo_val).
        padding_mode='zeros' treats positions outside [0,1] as walls.
        """
        B, N, T, _ = pred_positions.shape

        # grid_sample expects coordinates in [-1, 1]
        # grid shape must be [B, H_out, W_out, 2]
        grid = pred_positions.view(B, N * T, 1, 2) * 2.0 - 1.0   # [B, N*T, 1, 2]

        geo_vals = F.grid_sample(
            geo_mask.float(),
            grid,
            mode          = "bilinear",
            align_corners = True,
            padding_mode  = "zeros",   # out-of-bounds → 0 (wall)
        )                                                          # [B, 1, N*T, 1]
        geo_vals = geo_vals.squeeze(1).squeeze(-1).view(B, N, T)  # [B, N, T]

        penalty = (1.0 - geo_vals) * agent_mask.float()
        return penalty.sum() / (agent_mask.float().sum() + 1e-8)

    def compute_losses(
        self,
        pred_positions: torch.Tensor,
        gt_positions:   torch.Tensor,
        agent_mask:     torch.Tensor,
        mu:             torch.Tensor,
        logvar:         torch.Tensor,
        goal_pt:        torch.Tensor,
        geo_mask:       torch.Tensor,
        goal_weight:    float = 1.0,
        kl_weight:      float = 0.01,
        oob_weight:     float = 0.5,
    ) -> dict:
        point_mask = agent_mask.unsqueeze(-1).float()

        # Reconstruction (skip t=0 which is always start_pt)
        recon      = self.loss_fn(pred_positions[:, :, 1:], gt_positions[:, :, 1:])
        recon_mask = point_mask[:, :, 1:]
        recon_loss = (recon * recon_mask).sum() / (recon_mask.sum() * 2.0 + 1e-8)

        # KL divergence
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        agent_valid = (agent_mask.sum(dim=-1) > 0).float().unsqueeze(-1)
        kl_loss = (kl * agent_valid).sum() / (agent_valid.sum() * self.latent_dim + 1e-8)

        # Goal-consistency: last valid predicted position vs goal
        final_pred = self._last_valid(pred_positions, agent_mask)
        goal_loss  = F.smooth_l1_loss(final_pred, goal_pt, reduction="none")
        goal_loss  = (goal_loss * agent_valid).sum() / (agent_valid.sum() * 2.0 + 1e-8)

        # OOB soft penalty (differentiable via geo_mask bilinear sampling)
        oob_loss = self._oob_loss(pred_positions, agent_mask, geo_mask)

        total = (
            recon_loss
            + kl_weight  * kl_loss
            + goal_weight * goal_loss
            + oob_weight  * oob_loss
        )
        return {
            "loss":       total,
            "recon_loss": recon_loss,
            "kl_loss":    kl_loss,
            "goal_loss":  goal_loss,
            "oob_loss":   oob_loss,
        }

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        positions:       torch.Tensor,
        agent_mask:      torch.Tensor,
        start_pt:        torch.Tensor,
        goal_pt:         torch.Tensor,
        geo_mask:        torch.Tensor,
        sample_latent:   bool  = True,
        goal_weight:     float = 1.0,
        kl_weight:       float = 0.01,
        oob_weight:      float = 0.5,
        teacher_forcing: bool  = True,
        return_dict:     bool  = True,
    ) -> dict:
        geo_feat  = self.geo_encoder(geo_mask)
        mu, logvar = self.encode_latent(positions, agent_mask, start_pt, goal_pt, geo_feat)
        latent_z   = self.reparameterize(mu, logvar) if sample_latent else mu

        pred_positions = self.decode(
            start_pt          = start_pt,
            goal_pt           = goal_pt,
            geo_feat          = geo_feat,
            latent_z          = latent_z,
            seq_len           = positions.shape[2],
            teacher_positions = positions if teacher_forcing else None,
            agent_mask        = agent_mask,
        )

        losses = self.compute_losses(
            pred_positions, positions, agent_mask,
            mu, logvar, goal_pt, geo_mask,
            goal_weight = goal_weight,
            kl_weight   = kl_weight,
            oob_weight  = oob_weight,
        )
        result = {**losses, "positions": pred_positions, "mu": mu, "logvar": logvar}
        if return_dict:
            return result
        return result["loss"], result["positions"]

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        start_pt:  torch.Tensor,
        goal_pt:   torch.Tensor,
        geo_mask:  torch.Tensor,
        seq_len:   int,
        latent_z:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        geo_feat = self.geo_encoder(geo_mask)
        if latent_z is None:
            latent_z = torch.zeros(
                start_pt.shape[0], start_pt.shape[1], self.latent_dim,
                device=start_pt.device,
            )   # use mean of prior (z=0) for deterministic generation
        return self.decode(
            start_pt          = start_pt,
            goal_pt           = goal_pt,
            geo_feat          = geo_feat,
            latent_z          = latent_z,
            seq_len           = seq_len,
            teacher_positions = None,
            agent_mask        = torch.ones(
                start_pt.shape[0], start_pt.shape[1], seq_len,
                dtype=torch.bool, device=start_pt.device,
            ),
        )

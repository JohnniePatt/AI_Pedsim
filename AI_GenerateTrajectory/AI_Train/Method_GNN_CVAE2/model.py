"""
model.py
--------
GNN-CVAE2: Spatial Graph CVAE for goal-conditioned pedestrian trajectory generation.

Architecture:
  - SpatialGraphGNN: edge-type-aware 2-layer GNN on room/corridor/door graph
  - CVAE encoder: summarises trajectory into latent z
  - Decoder loop: heterogeneous agent GNN (repulsive/attractive/room) + GRU + goal-anchored emission
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Spatial Graph GNN ───────────────────────────────────────────────────────

class SpatialGraphGNN(nn.Module):
    """2-layer edge-type-aware message passing on the floor-plan graph."""

    def __init__(self, node_feat_dim: int = 8, embed_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.input_proj = nn.Linear(node_feat_dim, embed_dim)

        self.edge_mlps = nn.ModuleList()
        self.node_updates = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(n_layers):
            self.edge_mlps.append(nn.ModuleList([
                nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)),
                nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)),
            ]))
            self.node_updates.append(
                nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
            )
            self.norms.append(nn.LayerNorm(embed_dim))

    def forward(
        self,
        node_features: torch.Tensor,  # [N, 8]
        edge_index: torch.Tensor,      # [2, E]
        edge_type: torch.Tensor,       # [E]
    ) -> torch.Tensor:                 # [N, embed_dim]
        h = self.input_proj(node_features)
        N = h.shape[0]

        for i in range(self.n_layers):
            if edge_index.shape[1] == 0:
                h = self.norms[i](h)
                continue

            src, dst = edge_index[0], edge_index[1]
            edge_in = torch.cat([h[src], h[dst]], dim=-1)  # [E, 2D]

            msgs = torch.zeros(edge_index.shape[1], self.embed_dim, device=h.device, dtype=h.dtype)
            for t in range(2):
                mask = (edge_type == t)
                if mask.any():
                    msgs[mask] = self.edge_mlps[i][t](edge_in[mask])

            agg = [torch.zeros(N, self.embed_dim, device=h.device, dtype=h.dtype) for _ in range(2)]
            for t in range(2):
                mask = (edge_type == t)
                if mask.any():
                    agg[t].scatter_add_(0, dst[mask].unsqueeze(-1).expand(-1, self.embed_dim), msgs[mask])

            h_new = self.node_updates[i](torch.cat([h, agg[0], agg[1]], dim=-1))
            h = self.norms[i](h + h_new)

        return h


# ── Heterogeneous Agent GNN layer ───────────────────────────────────────────

class AgentGNNLayer(nn.Module):
    """One pass of the 3-edge-type gated agent GNN."""

    def __init__(self, hidden_dim: int = 128, node_embed_dim: int = 64):
        super().__init__()
        # Repulsive: agent ↔ neighbour agents
        self.rep_msg  = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.rep_gate = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # Attractive: agent → goal
        self.att_msg  = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.att_gate = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # Room awareness: agent ↔ current room node
        self.room_msg  = nn.Sequential(nn.Linear(hidden_dim + node_embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.room_gate = nn.Sequential(nn.Linear(hidden_dim + node_embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        h: torch.Tensor,            # [B, N, H]
        current: torch.Tensor,      # [B, N, 2]
        goal_pt: torch.Tensor,      # [B, N, 2]
        room_local: torch.Tensor,   # [B, N, D]
        active_mask: torch.Tensor,  # [B, N] bool
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = h.shape

        # Repulsive
        rel  = current.unsqueeze(1) - current.unsqueeze(2)          # [B, N, N, 2]  pos_j - pos_i
        dist = rel.norm(dim=-1, keepdim=True)                        # [B, N, N, 1]
        h_j  = h.unsqueeze(1).expand(-1, N, -1, -1)
        rep_in   = torch.cat([h_j, rel, dist], dim=-1)
        gated_rep = self.rep_msg(rep_in) * self.rep_gate(rep_in)

        active_pair = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
        eye = torch.eye(N, device=h.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = active_pair & ~eye & (dist.squeeze(-1) <= radius)
        gated_rep = gated_rep * edge_mask.unsqueeze(-1).float().to(h.dtype)
        rep_agg = gated_rep.sum(dim=2) / edge_mask.sum(dim=2, keepdim=True).clamp(min=1).float().to(h.dtype)

        # Attractive
        goal_delta = goal_pt - current
        goal_dist  = goal_delta.norm(dim=-1, keepdim=True)
        att_in  = torch.cat([h, goal_delta, goal_dist], dim=-1)
        att_agg = self.att_msg(att_in) * self.att_gate(att_in)

        # Room
        room_in  = torch.cat([h, room_local], dim=-1)
        room_agg = self.room_msg(room_in) * self.room_gate(room_in)

        h_new  = self.node_update(torch.cat([h, rep_agg, att_agg, room_agg], dim=-1))
        social = h_new - h
        return h_new, social


# ── Main model ───────────────────────────────────────────────────────────────

class GNNCVAE2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        node_feat_dim: int = 8,
        node_embed_dim: int = 64,
        spatial_gnn_layers: int = 2,
        agent_gnn_layers: int = 3,
        neighbor_radius: float = 0.08,
        dropout: float = 0.1,
        max_residual: float = 0.25,
        segment_samples: int = 5,
    ):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.latent_dim      = latent_dim
        self.node_embed_dim  = node_embed_dim
        self.neighbor_radius = neighbor_radius
        self.max_residual    = max_residual
        self.segment_samples = max(int(segment_samples), 2)

        self.spatial_gnn = SpatialGraphGNN(node_feat_dim, node_embed_dim, spatial_gnn_layers)
        self.global_proj = nn.Linear(node_embed_dim, node_embed_dim)

        # Encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head  = nn.Linear(hidden_dim, latent_dim)
        self.room_to_enc  = nn.Linear(node_embed_dim, hidden_dim)

        # Decoder init: start(2)+goal(2)+latent(ld)+room_global(ned)
        self.init_mlp = nn.Sequential(
            nn.Linear(2 + 2 + latent_dim + node_embed_dim, hidden_dim),
            nn.Tanh(),
        )

        self.agent_gnn_layers_list = nn.ModuleList(
            [AgentGNNLayer(hidden_dim, node_embed_dim) for _ in range(agent_gnn_layers)]
        )

        # GRU input: current(2)+start(2)+goal(2)+latent(ld)+room_global(ned)+social(hd)+time(1)+room_local(ned)
        gru_in = 2 + 2 + 2 + latent_dim + node_embed_dim + hidden_dim + 1 + node_embed_dim
        self.decoder_cell = nn.GRUCell(gru_in, hidden_dim)

        # Emission head: h(hd) + social(hd) + goal_delta(2)
        head_in = hidden_dim + hidden_dim + 2
        self.pos_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )

        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _masked_mean(self, x, mask):
        denom = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        return (x * mask.unsqueeze(-1).float()).sum(dim=-2) / denom

    def _last_valid(self, x, mask):
        lengths = mask.sum(dim=-1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.shape[-1])
        return torch.gather(x, dim=2, index=idx).squeeze(2)

    # ── spatial GNN ──────────────────────────────────────────────────────────

    def encode_spatial(
        self,
        node_features: torch.Tensor,  # [N, 8]
        edge_index: torch.Tensor,      # [2, E]
        edge_type: torch.Tensor,       # [E]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_embeds    = self.spatial_gnn(node_features, edge_index, edge_type)  # [N, D]
        room_global    = self.global_proj(node_embeds.mean(dim=0))               # [D]
        node_positions = node_features[:, :2]                                    # [N, 2] normalised centroids
        return node_embeds, room_global, node_positions

    # ── OOB snapping ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _snap_to_walkable(
        self,
        positions: torch.Tensor,             # [B, N, 2]
        geo_mask: torch.Tensor,              # [B, 1, H, W]
        walkable_pts_cache: list,            # precomputed per-batch walkable cell centres
        active_mask: torch.Tensor | None,    # [B, N] bool
    ) -> torch.Tensor:
        """Move OOB active agents to the nearest walkable cell (no gradient)."""
        B, N, _ = positions.shape
        grid_coords = positions.view(B, N, 1, 2) * 2.0 - 1.0
        geo_vals = F.grid_sample(
            geo_mask.float(), grid_coords, mode="nearest",
            align_corners=True, padding_mode="zeros",
        ).view(B, N)

        oob = geo_vals < 0.5
        if active_mask is not None:
            oob = oob & active_mask
        if not oob.any():
            return positions

        result = positions.clone()
        for b in range(B):
            wpts = walkable_pts_cache[b]
            if wpts is None:
                continue
            oob_agents = oob[b].nonzero(as_tuple=True)[0]
            if len(oob_agents) == 0:
                continue
            oob_pos = positions[b, oob_agents].float()
            dists   = torch.cdist(oob_pos, wpts.float())      # [M, K]
            nearest = wpts[dists.argmin(dim=-1)]               # [M, 2]
            result[b, oob_agents] = nearest.to(positions.dtype)
        return result

    # ── CVAE encoder ─────────────────────────────────────────────────────────

    def encode_latent(self, positions, agent_mask, start_pt, goal_pt, node_embeds, agent_node_ids):
        deltas     = positions[:, :, 1:] - positions[:, :, :-1]
        delta_mask = agent_mask[:, :, 1:] & agent_mask[:, :, :-1]
        mean_pos   = self._masked_mean(positions, agent_mask)
        mean_vel   = self._masked_mean(deltas, delta_mask)
        final_pos  = self._last_valid(positions, agent_mask)
        duration   = (agent_mask.sum(dim=-1, keepdim=True).float() / agent_mask.shape[-1]).clamp(max=1.0)

        features = torch.cat([start_pt, goal_pt, final_pos, mean_pos, mean_vel, duration], dim=-1)
        hidden   = self.agent_encoder(features)

        # Add start-room bias
        start_room = node_embeds[agent_node_ids]  # [B, N, D]
        hidden = hidden + 0.1 * self.room_to_enc(start_room)

        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # ── decoder ──────────────────────────────────────────────────────────────

    def decode(
        self,
        start_pt, goal_pt, latent_z, room_global, node_embeds, agent_node_ids,
        seq_len, teacher_positions=None, agent_mask=None,
        node_positions=None, geo_mask=None,
    ):
        B, N, _ = start_pt.shape
        device   = start_pt.device
        autoregressive = teacher_positions is None

        rg      = room_global.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # [B, N, D]
        init_in = torch.cat([start_pt, goal_pt, latent_z, rg], dim=-1)
        h       = self.init_mlp(init_in)

        preds            = [start_pt]
        current          = start_pt
        current_node_ids = agent_node_ids  # [B, N] — updated each step

        # Precompute walkable cell centres once for snapping
        walkable_cache = None
        if autoregressive and geo_mask is not None:
            H, W = geo_mask.shape[-2:]
            walkable_cache = []
            for b in range(B):
                wm = geo_mask[b, 0]
                ys, xs = (wm > 0.5).nonzero(as_tuple=True)
                if len(xs) > 0:
                    cx = (xs.float() + 0.5) / W
                    cy = (ys.float() + 0.5) / H
                    walkable_cache.append(torch.stack([cx, cy], dim=-1).to(device))
                else:
                    walkable_cache.append(None)

        for step in range(seq_len - 1):
            if not autoregressive:
                current = teacher_positions[:, :, step]

            active_mask = (
                agent_mask[:, :, step] if agent_mask is not None
                else torch.ones(B, N, dtype=torch.bool, device=device)
            )

            # Dynamic room lookup: find nearest graph node for each agent's current position
            if node_positions is not None:
                node_pos  = node_positions.to(device)
                cur_flat  = current.reshape(B * N, 2).float()
                dists_rn  = torch.cdist(cur_flat, node_pos.float())   # [B*N, num_nodes]
                current_node_ids = dists_rn.argmin(dim=-1).reshape(B, N)

            room_local = node_embeds[current_node_ids]  # [B, N, D]

            # Heterogeneous agent GNN
            social = torch.zeros_like(h)
            h_s    = h
            for layer in self.agent_gnn_layers_list:
                h_s, s = layer(h_s, current, goal_pt, room_local, active_mask, self.neighbor_radius)
                social = social + s

            time_scalar = torch.full((B, N, 1), float(step) / max(seq_len - 1, 1), device=device)
            dec_in = torch.cat([current, start_pt, goal_pt, latent_z, rg, social, time_scalar, room_local], dim=-1)

            flat_h  = h.reshape(B * N, -1)
            flat_in = dec_in.reshape(B * N, -1)
            h_new   = self.decoder_cell(flat_in, flat_h).view(B, N, self.hidden_dim)
            af      = active_mask.unsqueeze(-1).float()
            h       = h_new * af + h * (1.0 - af)

            # Velocity-based emission: predict step delta from current position
            goal_delta = goal_pt - current
            step_vec   = torch.tanh(self.pos_head(torch.cat([h, social, goal_delta], dim=-1))) * self.max_residual
            next_pos   = (current + step_vec).clamp(0.0, 1.0)

            if agent_mask is not None:
                next_pos = next_pos * agent_mask[:, :, step + 1].unsqueeze(-1).float()

            # Record unsnapped prediction (honest OOB loss signal)
            preds.append(next_pos)

            if autoregressive:
                # Use snapped position as next input to prevent error compounding
                current = next_pos
                if walkable_cache is not None:
                    active_next = (
                        agent_mask[:, :, step + 1] if agent_mask is not None
                        else torch.ones(B, N, dtype=torch.bool, device=device)
                    )
                    current = self._snap_to_walkable(next_pos, geo_mask, walkable_cache, active_next)

        return torch.stack(preds, dim=2)  # [B, N, T, 2]

    # ── losses ───────────────────────────────────────────────────────────────

    def _oob_loss(self, pred_positions, agent_mask, geo_mask):
        B, N, T, _ = pred_positions.shape
        grid = pred_positions.view(B, N * T, 1, 2) * 2.0 - 1.0
        geo_vals = F.grid_sample(geo_mask.float(), grid, mode="bilinear",
                                 align_corners=True, padding_mode="zeros")
        geo_vals = geo_vals.squeeze(1).squeeze(-1).view(B, N, T)
        penalty  = (1.0 - geo_vals) * agent_mask.float()
        return penalty.sum() / (agent_mask.float().sum() + 1e-8)

    def _segment_oob_loss(self, pred_positions, agent_mask, geo_mask):
        seg_mask = agent_mask[:, :, :-1] & agent_mask[:, :, 1:]
        if not torch.any(seg_mask):
            return pred_positions.new_tensor(0.0)
        alphas = torch.linspace(0.0, 1.0, self.segment_samples, device=pred_positions.device)[1:-1]
        if alphas.numel() == 0:
            return pred_positions.new_tensor(0.0)
        start = pred_positions[:, :, :-1]
        end   = pred_positions[:, :, 1:]
        pts   = start.unsqueeze(3) + (end - start).unsqueeze(3) * alphas.view(1, 1, 1, -1, 1)
        B, N, Tm1, ns, _ = pts.shape
        grid = pts.reshape(B, N * Tm1 * ns, 1, 2) * 2.0 - 1.0
        geo_vals = F.grid_sample(geo_mask.float(), grid, mode="bilinear",
                                 align_corners=True, padding_mode="zeros")
        geo_vals = geo_vals.squeeze(1).squeeze(-1).view(B, N, Tm1, ns)
        penalty  = (1.0 - geo_vals) * seg_mask.unsqueeze(-1).float()
        return penalty.sum() / (seg_mask.float().sum() * ns + 1e-8)

    def compute_losses(
        self, pred_positions, gt_positions, agent_mask, mu, logvar, goal_pt, geo_mask,
        goal_weight=1.0, kl_weight=0.01, oob_weight=1.0, segment_oob_weight=0.5,
    ):
        point_mask = agent_mask.unsqueeze(-1).float()
        recon      = self.loss_fn(pred_positions[:, :, 1:], gt_positions[:, :, 1:])
        recon_loss = (recon * point_mask[:, :, 1:]).sum() / (point_mask[:, :, 1:].sum() * 2.0 + 1e-8)

        kl         = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        agent_valid = (agent_mask.sum(dim=-1) > 0).float().unsqueeze(-1)
        kl_loss    = (kl * agent_valid).sum() / (agent_valid.sum() * self.latent_dim + 1e-8)

        final_pred = self._last_valid(pred_positions, agent_mask)
        goal_loss  = F.smooth_l1_loss(final_pred, goal_pt, reduction="none")
        goal_loss  = (goal_loss * agent_valid).sum() / (agent_valid.sum() * 2.0 + 1e-8)

        oob_loss     = self._oob_loss(pred_positions, agent_mask, geo_mask)
        seg_oob_loss = self._segment_oob_loss(pred_positions, agent_mask, geo_mask)

        total = (recon_loss + kl_weight * kl_loss + goal_weight * goal_loss
                 + oob_weight * oob_loss + segment_oob_weight * seg_oob_loss)
        return {
            "loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss,
            "goal_loss": goal_loss, "oob_loss": oob_loss, "segment_oob_loss": seg_oob_loss,
        }

    # ── forward / generate ───────────────────────────────────────────────────

    def forward(
        self,
        positions, agent_mask, start_pt, goal_pt, geo_mask,
        node_features, edge_index, edge_type, agent_node_ids,
        sample_latent=True, goal_weight=1.0, kl_weight=0.01,
        oob_weight=1.0, segment_oob_weight=0.5, teacher_forcing=True,
    ):
        node_embeds, room_global, node_positions = self.encode_spatial(node_features, edge_index, edge_type)
        mu, logvar = self.encode_latent(positions, agent_mask, start_pt, goal_pt, node_embeds, agent_node_ids)
        latent_z   = self.reparameterize(mu, logvar) if sample_latent else mu

        pred_positions = self.decode(
            start_pt, goal_pt, latent_z, room_global, node_embeds, agent_node_ids,
            positions.shape[2],
            teacher_positions=positions if teacher_forcing else None,
            agent_mask=agent_mask,
            node_positions=node_positions,
            geo_mask=geo_mask,
        )

        losses = self.compute_losses(
            pred_positions, positions, agent_mask, mu, logvar, goal_pt, geo_mask,
            goal_weight, kl_weight, oob_weight, segment_oob_weight,
        )
        return {**losses, "positions": pred_positions, "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def generate(
        self,
        start_pt, goal_pt, geo_mask,
        node_features, edge_index, edge_type, agent_node_ids,
        seq_len, latent_z=None,
    ):
        node_embeds, room_global, node_positions = self.encode_spatial(node_features, edge_index, edge_type)
        if latent_z is None:
            latent_z = torch.zeros(
                start_pt.shape[0], start_pt.shape[1], self.latent_dim, device=start_pt.device
            )
        return self.decode(
            start_pt, goal_pt, latent_z, room_global, node_embeds, agent_node_ids,
            seq_len, teacher_positions=None,
            agent_mask=torch.ones(
                start_pt.shape[0], start_pt.shape[1], seq_len, dtype=torch.bool, device=start_pt.device
            ),
            node_positions=node_positions,
            geo_mask=geo_mask,
        )

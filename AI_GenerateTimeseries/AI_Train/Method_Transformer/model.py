"""
model.py
─────────
Goal-conditioned GPT-2 with social context (neighbouring agents).

Architecture
────────────

  ┌─────────────────────────────────────────────────────────────┐
  │  Context tokens (prepended before the ego trajectory)       │
  │  ┌──────────────┐  ┌──────────────┐     ┌──────────────┐   │
  │  │ geo+goal     │  │ neighbour 1  │ … │ │ neighbour K  │   │
  │  │ token  [d]   │  │ token  [d]   │     │ token  [d]   │   │
  │  └──────────────┘  └──────────────┘     └──────────────┘   │
  └─────────────────────────────────────────────────────────────┘
                  ↓  fed into GPT-2  ↓
  ┌──────────────────────────────────────────────────────┐
  │ ego obs step 1 │ … │ ego obs step obs_len │          │
  │          (teacher-forced target during training)      │
  └──────────────────────────────────────────────────────┘
                  ↓
              OutputHead → (x, y)

Neighbour tokens:
  Each neighbour's obs_traj  [obs_len, 2]  is encoded by NeighborEncoder
  into a single d-dimensional token.  Real vs padded slots are separated
  by a boolean mask.  The model learns to attend to nearby agents and
  ignore padded (zero) slots.

Forward modes
─────────────
  Training  : pass labels=[B,T,2]     → teacher forcing  →  loss + logits
  Inference : pass pred_len=N          → autoregressive   →  logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


# ─── Geometry encoder ─────────────────────────────────────────────────────────

class GeoEncoder(nn.Module):
    """CNN: [B, 1, 64, 64] → [B, d_model]."""

    def __init__(self, d_model: int, mode: str = "spatial"):
        super().__init__()
        if mode not in {"spatial", "pooled"}:
            raise ValueError(f"unsupported geo encoder mode: {mode}")
        self.mode = mode
        self.net = nn.Sequential(
            nn.Conv2d(1,   16, 4, stride=2, padding=1),   # 32
            nn.ReLU(),
            nn.Conv2d(16,  32, 4, stride=2, padding=1),   # 16
            nn.ReLU(),
            nn.Conv2d(32,  64, 4, stride=2, padding=1),   # 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),   # 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) if mode == "spatial" else nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 if mode == "spatial" else 128, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Neighbour encoder ────────────────────────────────────────────────────────

class NeighborEncoder(nn.Module):
    """
    Encode K neighbours' observed trajectories into K context tokens.

    Input  : [B, K, obs_len, 2]  – relative positions (neighbour - ego at t=0)
    Mask   : [B, K]               – True for real neighbours, False for padding
    Output : [B, K, d_model]
    """

    def __init__(self, d_model: int, obs_len: int):
        super().__init__()
        self.step_proj = nn.Linear(2, d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(
        self,
        neighbor_trajs: torch.Tensor,   # [B, K, obs_len, 2]
        neighbor_mask:  torch.Tensor,   # [B, K]  bool
    ) -> torch.Tensor:
        B, K, T, _ = neighbor_trajs.shape

        # Per-step embedding  →  mean over time
        step_emb = self.step_proj(neighbor_trajs.view(B * K, T, 2))   # [B*K, T, d]
        pooled   = step_emb.mean(dim=1)                                # [B*K, d]
        out      = self.norm(self.out_proj(pooled)).view(B, K, -1)    # [B, K, d]

        # Zero-out padded neighbour slots (mask=False)
        out = out * neighbor_mask.float().unsqueeze(-1)                # [B, K, d]
        return out


# ─── Main model ───────────────────────────────────────────────────────────────

class GoalConditionedGPT2(nn.Module):
    """
    GPT-2 trajectory model with social context.

    Parameters
    ----------
    d_model      : embedding dimension
    nhead        : attention heads
    num_layers   : transformer layers
    max_seq_len  : max total sequence length
    dropout      : dropout on attention / residuals
    max_neighbors: K neighbours included as context tokens
    obs_len      : number of observed (seed) frames

    Inputs
    ------
    obs_traj       [B, obs_len, 2]     – ego's observed frames (normalised)
    start_pt       [B, 2]              – ego spawn (normalised)
    end_pt         [B, 2]              – exit centroid (normalised)
    geo_mask       [B, 1, H, W]        – occupancy grid
    neighbor_trajs [B, K, obs_len, 2]  – neighbours' relative obs frames
    neighbor_mask  [B, K]              – True = real neighbour

    Training  (pass labels)    → {"loss": scalar, "logits": [B,T,2]}
    Inference (pass pred_len)  → {"logits": [B, pred_len, 2]}
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        max_neighbors: int = 10,
        obs_len: int = 5,
        geo_encoder_type: str = "spatial",
        prediction_mode: str = "delta",
        max_step_size: float = 0.1,
        walkability_loss_weight: float = 0.05,
    ):
        super().__init__()

        self.max_neighbors = max_neighbors
        if prediction_mode not in {"delta", "absolute"}:
            raise ValueError(f"unsupported prediction mode: {prediction_mode}")
        self.prediction_mode = prediction_mode
        self.max_step_size = float(max_step_size)
        self.walkability_loss_weight = float(walkability_loss_weight)
        # Total context prefix length: 1 (geo+goal) + K (neighbours)
        n_ctx_tokens = 1 + max_neighbors

        gpt_cfg = GPT2Config(
            n_embd      = d_model,
            n_layer     = num_layers,
            n_head      = nhead,
            n_positions = max_seq_len + n_ctx_tokens + 2,
            n_ctx       = max_seq_len + n_ctx_tokens + 2,
            resid_pdrop = dropout,
            embd_pdrop  = dropout,
            attn_pdrop  = dropout,
            use_cache   = True,
        )
        self.gpt2 = GPT2Model(gpt_cfg)

        # Encoders
        self.geo_enc       = GeoEncoder(d_model, mode=geo_encoder_type)
        self.cond_proj     = nn.Linear(4, d_model)          # start(2)+end(2)
        self.context_fuse  = nn.Linear(d_model * 2, d_model)
        self.neighbor_enc  = NeighborEncoder(d_model, obs_len)

        # Trajectory embedding / head
        self.input_proj    = nn.Linear(2, d_model)
        self.output_head   = nn.Linear(d_model, 2)

        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        obs_traj:       torch.Tensor,
        start_pt:       torch.Tensor,
        end_pt:         torch.Tensor,
        geo_mask:       torch.Tensor,
        neighbor_trajs: torch.Tensor | None = None,
        neighbor_mask:  torch.Tensor | None = None,
        pred_len:       int = 10,
        labels:         torch.Tensor | None = None,
        lengths:        torch.Tensor | None = None,
        return_dict:    bool = True,
    ) -> dict:

        B      = obs_traj.shape[0]
        device = obs_traj.device

        # ── 1. Build context tokens ──────────────────────────────────────────
        # 1a. geo + goal  →  single token [B, 1, d]
        geo_feat  = self.geo_enc(geo_mask)
        goal_feat = self.cond_proj(torch.cat([start_pt, end_pt], dim=-1))
        geo_goal  = self.context_fuse(
            torch.cat([geo_feat, goal_feat], dim=-1)
        ).unsqueeze(1)                                           # [B, 1, d]

        # 1b. Neighbours  →  K tokens [B, K, d]
        if neighbor_trajs is not None:
            if neighbor_mask is None:
                # Infer mask: real if any value is non-zero
                neighbor_mask = (
                    neighbor_trajs.abs().sum(dim=(-1, -2)) > 0
                )
            neigh_tokens = self.neighbor_enc(neighbor_trajs, neighbor_mask)  # [B,K,d]
        else:
            # No social context (backward compat / ablation)
            K = self.max_neighbors
            neigh_tokens = torch.zeros(B, K, geo_goal.shape[-1], device=device)

        # Concatenate: [B, 1+K, d]
        context = torch.cat([geo_goal, neigh_tokens], dim=1)

        # ── 2a. Training – teacher forcing ────────────────────────────────────
        if labels is not None:
            T = labels.shape[1]

            # Input sequence: obs + labels[:-1]
            seq_pts = torch.cat([obs_traj, labels[:, :-1, :]], dim=1)  # [B, obs+T-1, 2]
            seq_emb = self.input_proj(seq_pts)                          # [B, obs+T-1, d]
            full    = torch.cat([context, seq_emb], dim=1)             # [B, C+obs+T-1, d]

            S       = full.shape[1]
            pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

            hidden  = self.gpt2(
                inputs_embeds=full, position_ids=pos_ids, use_cache=False
            ).last_hidden_state                                         # [B, S, d]

            # Last T hidden states predict labels[0..T-1]
            pred_hidden = hidden[:, -T:, :]
            raw_preds   = self.output_head(pred_hidden)                # [B, T, 2]
            if self.prediction_mode == "delta":
                # Bounded displacement prediction makes teacher forcing and
                # autoregressive rollout obey the same transition contract.
                base_pts = torch.cat([obs_traj[:, -1:, :], labels[:, :-1, :]], dim=1)
                preds = torch.clamp(
                    base_pts + self.max_step_size * torch.tanh(raw_preds),
                    min=0.0,
                    max=1.0,
                )
            else:
                preds = raw_preds

            # Use explicit lengths: (0, 0) can be a real normalised coordinate
            # and therefore must not double as a padding sentinel.
            if lengths is not None:
                step_ids = torch.arange(T, device=device).unsqueeze(0)
                mask = (step_ids < lengths.unsqueeze(1)).unsqueeze(-1).float()
            else:
                mask = (labels.abs().sum(dim=-1, keepdim=True) > 0).float()
            elem_loss = self.loss_fn(preds, labels)
            trajectory_loss = (elem_loss * mask).sum() / (mask.sum() * 2 + 1e-8)

            # Penalise predictions in walls/outside the map.  Coordinates and
            # the occupancy grid share the same [0,1] frame.
            sample_grid = (preds * 2.0 - 1.0).unsqueeze(2)              # [B,T,1,2]
            walkability = F.grid_sample(
                geo_mask,
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(1).squeeze(-1).unsqueeze(-1)                     # [B,T,1]
            collision_loss = ((1.0 - walkability) * mask).sum() / (mask.sum() + 1e-8)
            loss = trajectory_loss + self.walkability_loss_weight * collision_loss

            result = {
                "loss": loss,
                "trajectory_loss": trajectory_loss.detach(),
                "collision_loss": collision_loss.detach(),
                "logits": preds,
            }

        # ── 2b. Inference – autoregressive generation with KV-cache ─────────────
        else:
            preds = []

            # ── Prefill: feed context + full obs in one shot ──────────────────
            obs_emb = self.input_proj(obs_traj)                        # [B, obs_len, d]
            prefill  = torch.cat([context, obs_emb], dim=1)           # [B, C+obs_len, d]
            S_pre    = prefill.shape[1]
            pos_ids  = torch.arange(S_pre, device=device).unsqueeze(0).expand(B, -1)

            out_pre  = self.gpt2(inputs_embeds=prefill, position_ids=pos_ids)
            past_kv  = out_pre.past_key_values
            step_pos = S_pre   # position index of the next token

            # Last hidden state from prefill → first predicted point
            raw_next = self.output_head(out_pre.last_hidden_state[:, -1, :])
            if self.prediction_mode == "delta":
                next_pt = torch.clamp(
                    obs_traj[:, -1, :] + self.max_step_size * torch.tanh(raw_next),
                    min=0.0,
                    max=1.0,
                )
            else:
                next_pt = raw_next
            preds.append(next_pt)

            # ── Decode: one token per step, cache growing ─────────────────────
            for _ in range(pred_len - 1):
                tok_emb = self.input_proj(next_pt.unsqueeze(1))        # [B, 1, d]
                pos_id  = torch.full((B, 1), step_pos, dtype=torch.long, device=device)

                out_step = self.gpt2(
                    inputs_embeds  = tok_emb,
                    position_ids   = pos_id,
                    past_key_values= past_kv,
                )
                past_kv  = out_step.past_key_values
                step_pos += 1

                raw_next = self.output_head(out_step.last_hidden_state[:, -1, :])
                if self.prediction_mode == "delta":
                    next_pt = torch.clamp(
                        next_pt + self.max_step_size * torch.tanh(raw_next),
                        min=0.0,
                        max=1.0,
                    )
                else:
                    next_pt = raw_next
                preds.append(next_pt)

            result = {"logits": torch.stack(preds, dim=1)}            # [B, pred_len, 2]

        if return_dict:
            return result
        if "loss" in result:
            return result["loss"], result["logits"]
        return (result["logits"],)

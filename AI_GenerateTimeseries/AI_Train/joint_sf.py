"""Joint multi-agent data and Social-Force-informed neural building blocks.

The contract is deliberately architecture-neutral: every scene is represented
as synchronized agent trajectories, an active mask, a goal, and rasterized
walkability/wall-normal fields. Neural models learn only a bounded residual on
top of the same desired-motion, agent-repulsion, and wall-repulsion prior.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon
from torch.utils.data import Dataset


def _load_polygons(path: pathlib.Path) -> list[Polygon]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return [Polygon(coords).buffer(0) for coords in payload if len(coords) >= 3]


def _rasterize_polygon(poly: Polygon, points: np.ndarray) -> np.ndarray:
    inside = MplPath(np.asarray(poly.exterior.coords)).contains_points(points)
    for interior in poly.interiors:
        inside &= ~MplPath(np.asarray(interior.coords)).contains_points(points)
    return inside


def geometry_fields(case_dir: pathlib.Path, grid_size: int, padding: float) -> tuple[np.ndarray, np.ndarray, dict]:
    polygons = _load_polygons(case_dir / "Geo_room.json") + _load_polygons(case_dir / "Geo_corridor.json")
    if not polygons:
        raise ValueError(f"no room/corridor polygons in {case_dir}")
    bounds = np.asarray([poly.bounds for poly in polygons], dtype=np.float64)
    min_x = float(bounds[:, 0].min() - padding)
    min_y = float(bounds[:, 1].min() - padding)
    max_x = float(bounds[:, 2].max() + padding)
    max_y = float(bounds[:, 3].max() + padding)
    scale = max(max_x - min_x, max_y - min_y)
    xs = np.linspace(min_x, min_x + scale, grid_size)
    ys = np.linspace(min_y, min_y + scale, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    walkable = np.zeros(len(points), dtype=bool)
    for poly in polygons:
        if poly.geom_type == "Polygon":
            walkable |= _rasterize_polygon(poly, points)
        else:
            for component in poly.geoms:
                walkable |= _rasterize_polygon(component, points)
    walkable = walkable.reshape(grid_size, grid_size)
    distance = distance_transform_edt(walkable).astype(np.float32)
    grad_y, grad_x = np.gradient(distance)
    norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    wall_field = np.stack([grad_x / norm, grad_y / norm], axis=0).astype(np.float32)
    wall_field *= np.exp(-distance / 3.0)[None, ...]
    meta = {"min_x": min_x, "min_y": min_y, "scale": scale, "grid_size": grid_size}
    return walkable.astype(np.float32), wall_field, meta


def exit_centroid(case_dir: pathlib.Path, case_id: str) -> np.ndarray:
    candidates = [case_dir / f"Spawn_exit_{case_id}.csv", *sorted(case_dir.glob("Spawn_exit_*.csv"))]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        rows = frame[frame["type"] == "exit_area"]
        if not rows.empty:
            poly = shapely_wkt.loads(rows.iloc[0]["area"])
            return np.asarray([poly.centroid.x, poly.centroid.y], dtype=np.float32)
    raise FileNotFoundError(f"exit_area not found in {case_dir}")


def _trajectory_columns(frame: pd.DataFrame) -> tuple[str, str, str, str]:
    def choose(candidates: tuple[str, ...]) -> str:
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        raise KeyError(f"missing trajectory column; expected one of {candidates}")
    return (
        choose(("frame", "frame_id", "FR")),
        choose(("agent_id", "id", "ID")),
        choose(("pos_x", "x", "X")),
        choose(("pos_y", "y", "Y")),
    )


def _is_complete_case_dir(case_dir: pathlib.Path) -> bool:
    return (
        any(case_dir.glob("*.parquet"))
        and (case_dir / "Geo_room.json").is_file()
        and (case_dir / "Geo_corridor.json").is_file()
    )


class JointSceneDataset(Dataset):
    """Sample synchronized windows without materializing every overlapping window."""

    def __init__(
        self,
        dataset_root: str | pathlib.Path,
        split: str,
        *,
        obs_len: int = 8,
        pred_len: int = 24,
        frame_stride: int = 5,
        max_agents: int = 64,
        windows_per_case: int = 32,
        max_cases: int | None = None,
        grid_size: int = 64,
        geo_padding: float = 1.0,
        seed: int = 42,
        cache_size: int = 2,
    ):
        self.dataset_root = pathlib.Path(dataset_root).resolve()
        self.split = split
        self.obs_len = int(obs_len)
        self.pred_len = int(pred_len)
        self.frame_stride = max(int(frame_stride), 1)
        self.max_agents = int(max_agents)
        self.windows_per_case = int(windows_per_case)
        self.grid_size = int(grid_size)
        self.geo_padding = float(geo_padding)
        self.seed = int(seed)
        self.epoch = 0
        self.cache_size = int(cache_size)
        split_dir = self.dataset_root / split
        self.case_dirs = sorted(
            path for path in split_dir.glob("case_*")
            if path.is_dir() and _is_complete_case_dir(path)
        )
        if max_cases is not None:
            self.case_dirs = self.case_dirs[: int(max_cases)]
        if not self.case_dirs:
            raise RuntimeError(f"no case_* directories found in {split_dir}")
        self.samples = [
            (case_index, window_index)
            for case_index in range(len(self.case_dirs))
            for window_index in range(self.windows_per_case)
        ]
        self.cache: OrderedDict[int, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_case(self, case_index: int) -> dict:
        if case_index in self.cache:
            item = self.cache.pop(case_index)
            self.cache[case_index] = item
            return item
        case_dir = self.case_dirs[case_index]
        case_id = case_dir.name.removeprefix("case_")
        parquet_files = sorted(case_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"trajectory parquet missing in {case_dir}")
        raw = pd.read_parquet(parquet_files[0])
        frame_col, agent_col, x_col, y_col = _trajectory_columns(raw)
        raw = raw[[frame_col, agent_col, x_col, y_col]].drop_duplicates([frame_col, agent_col])
        pivot_x = raw.pivot(index=frame_col, columns=agent_col, values=x_col).sort_index()
        pivot_y = raw.pivot(index=frame_col, columns=agent_col, values=y_col).reindex(pivot_x.index)
        walkable, wall_field, meta = geometry_fields(case_dir, self.grid_size, self.geo_padding)
        goal_world = exit_centroid(case_dir, case_id)
        goal = (goal_world - np.asarray([meta["min_x"], meta["min_y"]], dtype=np.float32)) / meta["scale"]
        item = {
            "case_id": case_id,
            "case_dir": case_dir,
            "frames": pivot_x.index.to_numpy(dtype=np.int64),
            "agent_ids": pivot_x.columns.to_numpy(dtype=np.int64),
            "x": pivot_x.to_numpy(dtype=np.float32),
            "y": pivot_y.to_numpy(dtype=np.float32),
            "walkable": walkable,
            "wall_field": wall_field,
            "meta": meta,
            "goal": goal.astype(np.float32),
        }
        self.cache[case_index] = item
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return item

    def __getitem__(self, sample_index: int) -> dict:
        case_index, window_index = self.samples[sample_index]
        case = self._load_case(case_index)
        total_steps = self.obs_len + self.pred_len
        required_raw = (total_steps - 1) * self.frame_stride + 1
        frame_count = len(case["frames"])
        max_start = max(frame_count - required_raw, 0)
        if self.split == "train":
            rng = np.random.default_rng(
                self.seed + self.epoch * 10000019 + case_index * 100003 + window_index
            )
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
        else:
            start = int(round(max_start * window_index / max(self.windows_per_case - 1, 1)))
        raw_indices = start + np.arange(total_steps) * self.frame_stride
        in_range = raw_indices < frame_count
        clipped = np.minimum(raw_indices, max(frame_count - 1, 0))
        x = case["x"][clipped]
        y = case["y"][clipped]
        active = np.isfinite(x) & np.isfinite(y) & in_range[:, None]
        valid_agents = np.flatnonzero(active[: self.obs_len].all(axis=0))
        if not len(valid_agents):
            valid_agents = np.flatnonzero(active[: self.obs_len].any(axis=0))
        original_agent_count = len(valid_agents)
        if len(valid_agents) > self.max_agents:
            valid_agents = valid_agents[: self.max_agents]
        n_agents = len(valid_agents)
        positions = np.zeros((self.max_agents, total_steps, 2), dtype=np.float32)
        mask = np.zeros((self.max_agents, total_steps), dtype=bool)
        agent_ids = np.full(self.max_agents, -1, dtype=np.int64)
        if n_agents:
            selected_x = pd.DataFrame(x[:, valid_agents]).ffill().bfill().fillna(0.0).to_numpy(np.float32)
            selected_y = pd.DataFrame(y[:, valid_agents]).ffill().bfill().fillna(0.0).to_numpy(np.float32)
            origin = np.asarray([case["meta"]["min_x"], case["meta"]["min_y"]], dtype=np.float32)
            world = np.stack([selected_x, selected_y], axis=-1)
            norm = np.clip((world - origin) / case["meta"]["scale"], 0.0, 1.0)
            positions[:n_agents] = np.transpose(norm, (1, 0, 2))
            mask[:n_agents] = active[:, valid_agents].T
            agent_ids[:n_agents] = case["agent_ids"][valid_agents]
        return {
            "positions": torch.from_numpy(positions),
            "active": torch.from_numpy(mask),
            "agent_ids": torch.from_numpy(agent_ids),
            "goal": torch.from_numpy(case["goal"]),
            "walkable": torch.from_numpy(case["walkable"][None, ...]),
            "wall_field": torch.from_numpy(case["wall_field"]),
            "frames": torch.from_numpy(case["frames"][clipped].astype(np.int64)),
            "min_xy": torch.tensor([case["meta"]["min_x"], case["meta"]["min_y"]], dtype=torch.float32),
            "scale": torch.tensor(case["meta"]["scale"], dtype=torch.float32),
            "case_id": case["case_id"],
            "case_dir": str(case["case_dir"]),
            "original_agent_count": torch.tensor(original_agent_count, dtype=torch.int64),
            "agents_truncated": torch.tensor(original_agent_count > self.max_agents, dtype=torch.bool),
        }


def social_force_prior(
    position: torch.Tensor,
    velocity: torch.Tensor,
    goal: torch.Tensor,
    active: torch.Tensor,
    wall_field: torch.Tensor,
    *,
    desired_step: float,
    agent_strength: float,
    agent_sigma: float,
    wall_strength: float,
) -> torch.Tensor:
    goal_delta = goal[:, None, :] - position
    desired = F.normalize(goal_delta, dim=-1, eps=1e-6) * desired_step
    diff = position[:, :, None, :] - position[:, None, :, :]
    distance = torch.linalg.vector_norm(diff, dim=-1).clamp_min(1e-4)
    pair_mask = active[:, :, None] & active[:, None, :]
    eye = torch.eye(position.shape[1], device=position.device, dtype=torch.bool)[None, ...]
    pair_mask &= ~eye
    repulsion = (
        F.normalize(diff, dim=-1, eps=1e-6)
        * torch.exp(-distance / max(agent_sigma, 1e-4))[..., None]
        * pair_mask[..., None]
    ).sum(dim=2) * agent_strength
    grid = (position * 2.0 - 1.0).unsqueeze(2)
    wall = F.grid_sample(wall_field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    wall = wall.squeeze(-1).transpose(1, 2) * wall_strength
    damping = -0.05 * velocity
    prior = desired + repulsion + wall + damping
    max_norm = desired_step * 2.5
    norm = torch.linalg.vector_norm(prior, dim=-1, keepdim=True).clamp_min(1e-6)
    prior = prior * torch.clamp(max_norm / norm, max=1.0)
    return prior * active[..., None]


class JointSocialForcePredictor(nn.Module):
    """LSTM or Transformer residual predictor with synchronous social attention."""

    def __init__(
        self,
        architecture: str,
        *,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        max_residual: float = 0.03,
        noise_dim: int = 0,
        desired_step: float = 0.012,
        agent_strength: float = 0.004,
        agent_sigma: float = 0.04,
        wall_strength: float = 0.006,
    ):
        super().__init__()
        if architecture not in {"lstm", "transformer", "sgan"}:
            raise ValueError(architecture)
        self.architecture = architecture
        self.hidden_dim = int(hidden_dim)
        self.max_residual = float(max_residual)
        self.noise_dim = int(noise_dim if architecture == "sgan" else 0)
        self.force_kwargs = {
            "desired_step": float(desired_step),
            "agent_strength": float(agent_strength),
            "agent_sigma": float(agent_sigma),
            "wall_strength": float(wall_strength),
        }
        self.input_projection = nn.Linear(6, hidden_dim)
        if architecture == "transformer":
            layer = nn.TransformerEncoderLayer(hidden_dim, nhead, hidden_dim * 4, dropout, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.temporal = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.social_attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        decoder_input = hidden_dim * 2 + 2 + self.noise_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def predict_next(
        self,
        history: torch.Tensor,
        active_history: torch.Tensor,
        goal: torch.Tensor,
        wall_field: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, agents, steps, _ = history.shape
        velocity = torch.zeros_like(history)
        velocity[:, :, 1:] = history[:, :, 1:] - history[:, :, :-1]
        goal_delta = goal[:, None, None, :] - history
        temporal_input = torch.cat([history, velocity, goal_delta], dim=-1)
        tokens = self.input_projection(temporal_input).reshape(batch * agents, steps, self.hidden_dim)
        key_padding = ~active_history.reshape(batch * agents, steps)
        all_padding = key_padding.all(dim=1)
        if all_padding.any():
            key_padding = key_padding.clone()
            key_padding[all_padding, -1] = False
        if self.architecture == "transformer":
            encoded = self.temporal(tokens, src_key_padding_mask=key_padding)
            temporal_state = encoded[:, -1]
        else:
            _, (hidden, _) = self.temporal(tokens)
            temporal_state = hidden[-1]
        temporal_state = temporal_state.reshape(batch, agents, self.hidden_dim)
        current_active = active_history[:, :, -1]
        social_key_padding = ~current_active
        no_active_scene = social_key_padding.all(dim=1)
        if no_active_scene.any():
            social_key_padding = social_key_padding.clone()
            social_key_padding[no_active_scene, 0] = False
        social, _ = self.social_attention(
            temporal_state, temporal_state, temporal_state,
            key_padding_mask=social_key_padding,
        )
        social = torch.where(current_active[..., None], social, torch.zeros_like(social))
        current = history[:, :, -1]
        current_velocity = velocity[:, :, -1]
        prior = social_force_prior(current, current_velocity, goal, current_active, wall_field, **self.force_kwargs)
        pieces = [temporal_state, social, prior]
        if self.noise_dim:
            if noise is None:
                noise = torch.randn(batch, agents, self.noise_dim, device=history.device)
            pieces.append(noise)
        decoded = self.decoder(torch.cat(pieces, dim=-1))
        residual = self.max_residual * torch.tanh(decoded[..., :2])
        next_position = torch.clamp(current + prior + residual, 0.0, 1.0)
        next_position = torch.where(current_active[..., None], next_position, current)
        return next_position, decoded[..., 2], prior

    def rollout(
        self,
        observed: torch.Tensor,
        observed_active: torch.Tensor,
        goal: torch.Tensor,
        wall_field: torch.Tensor,
        pred_len: int,
        *,
        teacher_positions: torch.Tensor | None = None,
        teacher_active: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        stop_threshold: float = 0.5,
        exit_radius: float = 0.025,
    ) -> dict[str, torch.Tensor]:
        history = observed
        active_history = observed_active
        predictions, stop_logits, priors, active_predictions = [], [], [], []
        for step in range(int(pred_len)):
            next_position, stop_logit, prior = self.predict_next(history, active_history, goal, wall_field)
            current_active = active_history[:, :, -1]
            reached = torch.linalg.vector_norm(next_position - goal[:, None, :], dim=-1) <= exit_radius
            predicted_stop = torch.sigmoid(stop_logit) >= stop_threshold
            next_active = current_active & ~(reached & predicted_stop)
            predictions.append(next_position)
            stop_logits.append(stop_logit)
            priors.append(prior)
            active_predictions.append(next_active)
            use_teacher = (
                self.training and teacher_positions is not None and
                torch.rand((), device=history.device) < teacher_forcing_ratio
            )
            history_next = teacher_positions[:, :, step] if use_teacher else next_position
            active_next = teacher_active[:, :, step] if use_teacher and teacher_active is not None else next_active
            history = torch.cat([history, history_next[:, :, None, :]], dim=2)
            active_history = torch.cat([active_history, active_next[:, :, None]], dim=2)
        return {
            "positions": torch.stack(predictions, dim=2),
            "stop_logits": torch.stack(stop_logits, dim=2),
            "active": torch.stack(active_predictions, dim=2),
            "force_prior": torch.stack(priors, dim=2),
        }


class SceneDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.LSTM(4, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, positions: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(positions)
        velocity[:, :, 1:] = positions[:, :, 1:] - positions[:, :, :-1]
        features = torch.cat([positions, velocity], dim=-1)
        batch, agents, steps, dims = features.shape
        encoded, _ = self.encoder(features.reshape(batch * agents, steps, dims))
        state = encoded[:, -1].reshape(batch, agents, -1)
        weights = active[:, :, -1].float()
        pooled = (state * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def trajectory_losses(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_active: torch.Tensor,
    walkable: torch.Tensor,
    *,
    walkability_weight: float,
    stop_weight: float,
) -> dict[str, torch.Tensor]:
    mask = target_active[..., None].float()
    position = F.smooth_l1_loss(outputs["positions"], target, reduction="none")
    position_loss = (position * mask).sum() / (mask.sum() * 2.0 + 1e-6)
    grid = (outputs["positions"] * 2.0 - 1.0).reshape(target.shape[0], -1, 1, 2)
    sampled = F.grid_sample(walkable, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(target.shape[0], target.shape[1], target.shape[2])
    walkability_loss = ((1.0 - sampled) * target_active.float()).sum() / target_active.float().sum().clamp_min(1.0)
    stop_target = torch.zeros_like(target_active, dtype=torch.float32)
    stop_target[:, :, :-1] = target_active[:, :, :-1].float() * (~target_active[:, :, 1:]).float()
    stop_loss = F.binary_cross_entropy_with_logits(outputs["stop_logits"], stop_target, reduction="none")
    valid_stop = target_active.float()
    # The window boundary is not an evacuation event; its following state is
    # unknown, so it must not teach the model to stop at every training horizon.
    valid_stop[:, :, -1] = 0.0
    stop_loss = torch.where(valid_stop.bool(), stop_loss, torch.zeros_like(stop_loss))
    stop_loss = (stop_loss * valid_stop).sum() / valid_stop.sum().clamp_min(1.0)
    total = position_loss + walkability_weight * walkability_loss + stop_weight * stop_loss
    return {"loss": total, "position_loss": position_loss, "walkability_loss": walkability_loss, "stop_loss": stop_loss}

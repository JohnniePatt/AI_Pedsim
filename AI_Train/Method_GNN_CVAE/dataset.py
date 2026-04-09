"""
dataset.py
----------
Dataset for Method_GNN_CVAE.
Each sample corresponds to one simulation case and contains all agents together.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from shapely import wkt as shapely_wkt
from torch.utils.data import Dataset
from tqdm import tqdm

from prepare_geometry_gnn_cvae import create_occupancy_grid, world_to_grid


class SimulationTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        obs_len: int = 1,
        frame_stride: int = 8,
        max_seq_len: int = 160,
        grid_size: int = 64,
        geo_padding: float = 1.0,
        max_agents: int = 64,
        subset_percent: float = 100.0,
        data_percent: float | None = None,
    ):
        self.obs_len = obs_len
        self.frame_stride = frame_stride
        self.max_seq_len = max_seq_len
        self.grid_size = grid_size
        self.geo_padding = geo_padding
        self.max_agents = max_agents

        if data_percent is not None:
            subset_percent = data_percent

        split_dir = os.path.join(data_dir, split)
        case_dirs = sorted(glob.glob(os.path.join(split_dir, "case_*")))
        n = max(1, int(len(case_dirs) * subset_percent / 100.0)) if case_dirs else 0
        case_dirs = case_dirs[:n]

        self.samples: list[dict] = []
        self._geo_cache: dict[str, tuple[np.ndarray, dict, object]] = {}

        if not case_dirs:
            print(f"[Dataset] WARNING: no case_* folders in {split_dir}")
            return

        print(f"[Dataset] Loading [{split}] - {len(case_dirs)} simulations ...")
        for case_dir in tqdm(case_dirs, desc=f"Building [{split}]"):
            try:
                sample = self._load_case(case_dir)
                if sample is not None:
                    self.samples.append(sample)
            except Exception as exc:
                print(f"  [skip] {os.path.basename(case_dir)}: {exc}")

        print(f"[Dataset] [{split}] ready - {len(self.samples)} simulation samples.")

    def _resolve_columns(self, df: pd.DataFrame) -> tuple[str, str, str, str]:
        id_col = "id" if "id" in df.columns else "AgentID"
        fr_col = "frame" if "frame" in df.columns else "Frame"
        x_col = "pos_x" if "pos_x" in df.columns else "x"
        y_col = "pos_y" if "pos_y" in df.columns else "y"
        return id_col, fr_col, x_col, y_col

    def _load_case(self, case_dir: str) -> dict | None:
        case_id = os.path.basename(case_dir).replace("case_", "")
        room_json = os.path.join(case_dir, "Geo_room.json")
        corridor_json = os.path.join(case_dir, "Geo_corridor.json")
        cache_key = f"{room_json}|{corridor_json}|{self.grid_size}|{self.geo_padding}"

        if cache_key not in self._geo_cache:
            grid, meta, walkable = create_occupancy_grid(
                room_json,
                corridor_json,
                grid_size=self.grid_size,
                padding=self.geo_padding,
            )
            self._geo_cache[cache_key] = (grid, meta, walkable)

        grid, meta, walkable = self._geo_cache[cache_key]
        geo_mask = torch.from_numpy(grid).unsqueeze(0)

        exit_csv = os.path.join(case_dir, f"Spawn_exit_{case_id}.csv")
        exit_df = pd.read_csv(exit_csv)
        exit_row = exit_df[exit_df["type"] == "exit_area"].iloc[0]
        exit_poly = shapely_wkt.loads(exit_row["area"])
        goal_x, goal_y = world_to_grid(exit_poly.centroid.x, exit_poly.centroid.y, meta)

        spawn_csv = os.path.join(case_dir, f"Spawn_location_{case_id}.csv")
        spawn_df = pd.read_csv(spawn_csv)

        parquet_files = glob.glob(os.path.join(case_dir, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("no .parquet found in case folder")
        traj_df = pd.read_parquet(parquet_files[0])
        id_col, fr_col, x_col, y_col = self._resolve_columns(traj_df)

        all_ids = sorted([int(x) for x in traj_df[id_col].dropna().unique().tolist()])
        if not all_ids:
            return None

        kept_ids = all_ids[: self.max_agents]
        frames = np.array(sorted(traj_df[fr_col].dropna().unique().tolist()))[:: self.frame_stride]
        if len(frames) > self.max_seq_len:
            frames = frames[: self.max_seq_len]
        if len(frames) <= self.obs_len:
            return None

        frame_to_idx = {frame: idx for idx, frame in enumerate(frames.tolist())}
        n_agents = len(kept_ids)
        t_len = len(frames)

        positions = np.zeros((n_agents, t_len, 2), dtype=np.float32)
        mask = np.zeros((n_agents, t_len), dtype=bool)
        start_pts = np.zeros((n_agents, 2), dtype=np.float32)
        goal_pts = np.zeros((n_agents, 2), dtype=np.float32)

        spawn_lookup = {int(row["id"]): row for _, row in spawn_df.iterrows()}
        id_to_row = {agent_id: idx for idx, agent_id in enumerate(kept_ids)}

        traj_df = traj_df[traj_df[id_col].isin(kept_ids)].copy()
        for row in traj_df.itertuples(index=False):
            agent_id = int(getattr(row, id_col))
            frame = getattr(row, fr_col)
            if frame not in frame_to_idx:
                continue
            ridx = id_to_row[agent_id]
            tidx = frame_to_idx[frame]
            gx, gy = world_to_grid(float(getattr(row, x_col)), float(getattr(row, y_col)), meta)
            positions[ridx, tidx] = (gx, gy)
            mask[ridx, tidx] = True

        valid_agents = mask.sum(axis=1) > self.obs_len
        if not np.any(valid_agents):
            return None

        positions = positions[valid_agents]
        mask = mask[valid_agents]
        kept_ids = [aid for aid, keep in zip(kept_ids, valid_agents.tolist()) if keep]
        n_agents = len(kept_ids)

        for idx, agent_id in enumerate(kept_ids):
            spawn_row = spawn_lookup.get(agent_id)
            if spawn_row is not None:
                sx, sy = world_to_grid(float(spawn_row["pos_x"]), float(spawn_row["pos_y"]), meta)
            else:
                first_idx = int(np.argmax(mask[idx]))
                sx, sy = positions[idx, first_idx].tolist()
            start_pts[idx] = (sx, sy)
            goal_pts[idx] = (goal_x, goal_y)

        return {
            "positions": torch.from_numpy(positions),
            "agent_mask": torch.from_numpy(mask),
            "start_pt": torch.from_numpy(start_pts),
            "goal_pt": torch.from_numpy(goal_pts),
            "geo_mask": geo_mask,
            "case_id": case_id,
            "case_dir": case_dir,
            "meta": meta,
            "walkable": walkable,
            "frames": torch.tensor(frames, dtype=torch.long),
            "agent_ids": torch.tensor(kept_ids, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_agents = max(sample["positions"].shape[0] for sample in batch)
    max_t = max(sample["positions"].shape[1] for sample in batch)
    grid_h = batch[0]["geo_mask"].shape[-2]
    grid_w = batch[0]["geo_mask"].shape[-1]

    positions = torch.zeros(batch_size, max_agents, max_t, 2, dtype=torch.float32)
    agent_mask = torch.zeros(batch_size, max_agents, max_t, dtype=torch.bool)
    start_pt = torch.zeros(batch_size, max_agents, 2, dtype=torch.float32)
    goal_pt = torch.zeros(batch_size, max_agents, 2, dtype=torch.float32)
    geo_mask = torch.zeros(batch_size, 1, grid_h, grid_w, dtype=torch.float32)
    agent_ids = torch.full((batch_size, max_agents), -1, dtype=torch.long)
    case_ids = []
    case_dirs = []
    metas = []
    walkables = []
    frames_list = []

    for bidx, sample in enumerate(batch):
        n_agents, t_len, _ = sample["positions"].shape
        positions[bidx, :n_agents, :t_len] = sample["positions"]
        agent_mask[bidx, :n_agents, :t_len] = sample["agent_mask"]
        start_pt[bidx, :n_agents] = sample["start_pt"]
        goal_pt[bidx, :n_agents] = sample["goal_pt"]
        geo_mask[bidx] = sample["geo_mask"]
        agent_ids[bidx, :n_agents] = sample["agent_ids"]
        case_ids.append(sample["case_id"])
        case_dirs.append(sample["case_dir"])
        metas.append(sample["meta"])
        walkables.append(sample["walkable"])
        frames_list.append(sample["frames"])

    return {
        "positions": positions,
        "agent_mask": agent_mask,
        "start_pt": start_pt,
        "goal_pt": goal_pt,
        "geo_mask": geo_mask,
        "agent_ids": agent_ids,
        "case_ids": case_ids,
        "case_dirs": case_dirs,
        "metas": metas,
        "walkables": walkables,
        "frames": frames_list,
    }

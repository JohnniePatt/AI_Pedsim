"""
dataset.py
----------
Dataset for Method_GNN_CVAE2.
Extends GNN_CVAE dataset by adding spatial graph data per case.
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

from prepare_geometry_gnn_cvae2 import (
    agent_to_node,
    build_spatial_graph,
    create_occupancy_grid,
    world_to_grid,
)


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
        self.obs_len      = obs_len
        self.frame_stride = frame_stride
        self.max_seq_len  = max_seq_len
        self.grid_size    = grid_size
        self.geo_padding  = geo_padding
        self.max_agents   = max_agents

        if data_percent is not None:
            subset_percent = data_percent

        split_dir = os.path.join(data_dir, split)
        case_dirs = sorted(glob.glob(os.path.join(split_dir, "case_*")))
        n = max(1, int(len(case_dirs) * subset_percent / 100.0)) if case_dirs else 0
        case_dirs = case_dirs[:n]

        self.samples: list[dict] = []
        self._geo_cache: dict[str, tuple] = {}

        if not case_dirs:
            print(f"[Dataset] WARNING: no case_* folders in {split_dir}")
            return

        print(f"[Dataset] Loading [{split}] — {len(case_dirs)} cases …")
        for case_dir in tqdm(case_dirs, desc=f"[{split}]"):
            try:
                sample = self._load_case(case_dir)
                if sample is not None:
                    self.samples.append(sample)
            except Exception as exc:
                print(f"  [skip] {os.path.basename(case_dir)}: {exc}")

        print(f"[Dataset] [{split}] ready — {len(self.samples)} samples.")

    def _resolve_columns(self, df: pd.DataFrame):
        id_col = "id" if "id" in df.columns else "AgentID"
        fr_col = "frame" if "frame" in df.columns else "Frame"
        x_col  = "pos_x" if "pos_x" in df.columns else "x"
        y_col  = "pos_y" if "pos_y" in df.columns else "y"
        return id_col, fr_col, x_col, y_col

    def _load_case(self, case_dir: str) -> dict | None:
        case_id   = os.path.basename(case_dir).replace("case_", "")
        room_json = os.path.join(case_dir, "Geo_room.json")
        cor_json  = os.path.join(case_dir, "Geo_corridor.json")
        door_json = os.path.join(case_dir, "Geo_door.json")

        cache_key = f"{room_json}|{cor_json}|{self.grid_size}|{self.geo_padding}"
        if cache_key not in self._geo_cache:
            grid, meta, walkable = create_occupancy_grid(
                room_json, cor_json, grid_size=self.grid_size, padding=self.geo_padding
            )
            graph = build_spatial_graph(room_json, cor_json, door_json, meta)
            self._geo_cache[cache_key] = (grid, meta, walkable, graph)

        grid, meta, walkable, graph = self._geo_cache[cache_key]
        geo_mask = torch.from_numpy(grid).unsqueeze(0)  # [1, H, W]

        # Exit goal
        exit_csv = os.path.join(case_dir, f"Spawn_exit_{case_id}.csv")
        exit_df  = pd.read_csv(exit_csv)
        exit_row = exit_df[exit_df["type"] == "exit_area"].iloc[0]
        exit_poly = shapely_wkt.loads(exit_row["area"])
        goal_x, goal_y = world_to_grid(exit_poly.centroid.x, exit_poly.centroid.y, meta)

        # Spawn positions
        spawn_csv = os.path.join(case_dir, f"Spawn_location_{case_id}.csv")
        spawn_df  = pd.read_csv(spawn_csv)

        # Trajectory
        parquet_files = glob.glob(os.path.join(case_dir, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("no .parquet in case folder")
        traj_df = pd.read_parquet(parquet_files[0])
        id_col, fr_col, x_col, y_col = self._resolve_columns(traj_df)

        all_ids = sorted([int(x) for x in traj_df[id_col].dropna().unique().tolist()])
        if not all_ids:
            return None

        kept_ids = all_ids[: self.max_agents]
        frames   = np.array(sorted(traj_df[fr_col].dropna().unique().tolist()))[:: self.frame_stride]
        if len(frames) > self.max_seq_len:
            frames = frames[: self.max_seq_len]
        if len(frames) <= self.obs_len:
            return None

        frame_to_idx = {f: i for i, f in enumerate(frames.tolist())}
        n_agents, t_len = len(kept_ids), len(frames)

        positions  = np.zeros((n_agents, t_len, 2), dtype=np.float32)
        mask       = np.zeros((n_agents, t_len), dtype=bool)
        start_pts  = np.zeros((n_agents, 2), dtype=np.float32)
        goal_pts   = np.zeros((n_agents, 2), dtype=np.float32)

        spawn_lookup = {int(r["id"]): r for _, r in spawn_df.iterrows()}
        id_to_row    = {aid: idx for idx, aid in enumerate(kept_ids)}

        for row in traj_df[traj_df[id_col].isin(kept_ids)].itertuples(index=False):
            aid   = int(getattr(row, id_col))
            frame = getattr(row, fr_col)
            if frame not in frame_to_idx:
                continue
            ridx, tidx = id_to_row[aid], frame_to_idx[frame]
            gx, gy = world_to_grid(float(getattr(row, x_col)), float(getattr(row, y_col)), meta)
            positions[ridx, tidx] = (gx, gy)
            mask[ridx, tidx] = True

        valid_agents = mask.sum(axis=1) > self.obs_len
        if not np.any(valid_agents):
            return None

        positions = positions[valid_agents]
        mask      = mask[valid_agents]
        kept_ids  = [aid for aid, keep in zip(kept_ids, valid_agents.tolist()) if keep]
        n_agents  = len(kept_ids)

        agent_node_ids = np.zeros(n_agents, dtype=np.int64)

        for idx, agent_id in enumerate(kept_ids):
            spawn_row = spawn_lookup.get(agent_id)
            if spawn_row is not None:
                sx, sy = world_to_grid(float(spawn_row["pos_x"]), float(spawn_row["pos_y"]), meta)
            else:
                first = int(np.argmax(mask[idx]))
                sx, sy = positions[idx, first].tolist()
            start_pts[idx]      = (sx, sy)
            goal_pts[idx]       = (goal_x, goal_y)
            agent_node_ids[idx] = agent_to_node(float(sx), float(sy), graph, meta)

        return {
            "positions":      torch.from_numpy(positions),
            "agent_mask":     torch.from_numpy(mask),
            "start_pt":       torch.from_numpy(start_pts),
            "goal_pt":        torch.from_numpy(goal_pts),
            "geo_mask":       geo_mask,
            # spatial graph (per-sample, variable size)
            "node_features":  torch.from_numpy(graph["node_features"]),
            "edge_index":     torch.from_numpy(graph["edge_index"]),
            "edge_type":      torch.from_numpy(graph["edge_type"]),
            "agent_node_ids": torch.from_numpy(agent_node_ids),
            # metadata
            "case_id":    case_id,
            "case_dir":   case_dir,
            "meta":       meta,
            "walkable":   walkable,
            "frames":     torch.tensor(frames, dtype=torch.long),
            "agent_ids":  torch.tensor(kept_ids, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    B        = len(batch)
    max_a    = max(s["positions"].shape[0] for s in batch)
    max_t    = max(s["positions"].shape[1] for s in batch)
    grid_h   = batch[0]["geo_mask"].shape[-2]
    grid_w   = batch[0]["geo_mask"].shape[-1]

    positions  = torch.zeros(B, max_a, max_t, 2)
    agent_mask = torch.zeros(B, max_a, max_t, dtype=torch.bool)
    start_pt   = torch.zeros(B, max_a, 2)
    goal_pt    = torch.zeros(B, max_a, 2)
    geo_mask   = torch.zeros(B, 1, grid_h, grid_w)
    agent_ids  = torch.full((B, max_a), -1, dtype=torch.long)
    # agent_node_ids: per-sample list (graphs differ in size)
    agent_node_ids_list = []

    case_ids, case_dirs, metas, walkables, frames_list = [], [], [], [], []
    node_features_list, edge_index_list, edge_type_list = [], [], []

    for i, s in enumerate(batch):
        na, tl, _ = s["positions"].shape
        positions[i, :na, :tl]  = s["positions"]
        agent_mask[i, :na, :tl] = s["agent_mask"]
        start_pt[i, :na]        = s["start_pt"]
        goal_pt[i, :na]         = s["goal_pt"]
        geo_mask[i]              = s["geo_mask"]
        agent_ids[i, :na]        = s["agent_ids"]
        agent_node_ids_list.append(s["agent_node_ids"])  # [na]

        node_features_list.append(s["node_features"])
        edge_index_list.append(s["edge_index"])
        edge_type_list.append(s["edge_type"])

        case_ids.append(s["case_id"])
        case_dirs.append(s["case_dir"])
        metas.append(s["meta"])
        walkables.append(s["walkable"])
        frames_list.append(s["frames"])

    return {
        "positions":          positions,
        "agent_mask":         agent_mask,
        "start_pt":           start_pt,
        "goal_pt":            goal_pt,
        "geo_mask":           geo_mask,
        "agent_ids":          agent_ids,
        # graph data as lists (variable size per sample)
        "node_features_list": node_features_list,
        "edge_index_list":    edge_index_list,
        "edge_type_list":     edge_type_list,
        "agent_node_ids_list":agent_node_ids_list,
        # meta
        "case_ids":   case_ids,
        "case_dirs":  case_dirs,
        "metas":      metas,
        "walkables":  walkables,
        "frames":     frames_list,
    }

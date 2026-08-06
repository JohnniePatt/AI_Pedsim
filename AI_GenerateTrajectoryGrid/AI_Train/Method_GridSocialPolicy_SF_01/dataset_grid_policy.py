from __future__ import annotations

import json
import pathlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point, Polygon
from torch.utils.data import Dataset

from action_space import ActionSpace


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def grid_to_array(grid_rows: list[str]) -> np.ndarray:
    height = len(grid_rows)
    width = len(grid_rows[0])
    data = np.frombuffer("".join(grid_rows).encode("ascii"), dtype=np.uint8).reshape(height, width)
    return (data == ord("1")).astype(np.float32)


def make_goal_grid(exit_payload: dict, grid_meta: dict) -> np.ndarray:
    width = int(grid_meta["width"])
    height = int(grid_meta["height"])
    cell = float(grid_meta["cell_size_m"])
    origin_x = float(grid_meta["origin_x"])
    origin_y = float(grid_meta["origin_y"])
    poly = Polygon(exit_payload["exit_node"]["polygon"])
    goal = np.zeros((height, width), dtype=np.float32)

    min_x, min_y, max_x, max_y = poly.bounds
    gx0 = max(0, int(np.floor((min_x - origin_x) / cell)) - 1)
    gx1 = min(width - 1, int(np.ceil((max_x - origin_x) / cell)) + 1)
    gy0 = max(0, int(np.floor((min_y - origin_y) / cell)) - 1)
    gy1 = min(height - 1, int(np.ceil((max_y - origin_y) / cell)) + 1)
    for gy in range(gy0, gy1 + 1):
        y = origin_y + (gy + 0.5) * cell
        row = height - 1 - gy
        for gx in range(gx0, gx1 + 1):
            x = origin_x + (gx + 0.5) * cell
            if poly.covers(Point(float(x), float(y))):
                goal[row, gx] = 1.0
    return goal


def crop_centered(array: np.ndarray, center_row: int, center_col: int, crop_size: int, pad_value: float = 0.0) -> np.ndarray:
    half = crop_size // 2
    padded = np.pad(array, ((half, half), (half, half)), mode="constant", constant_values=pad_value)
    row = int(center_row) + half
    col = int(center_col) + half
    return padded[row - half : row + half + 1, col - half : col + half + 1]


def build_next_index(traj: pd.DataFrame, frame_stride: int = 1) -> np.ndarray:
    next_idx = np.full(len(traj), -1, dtype=np.int64)
    stride = max(int(frame_stride), 1)
    for _, agent_df in traj.groupby("agent_id", sort=False):
        ordered = agent_df.sort_values("frame")
        idx = ordered.index.to_numpy(dtype=np.int64)
        if len(idx) > stride:
            next_idx[idx[:-stride]] = idx[stride:]
    return next_idx


def dataframe_arrays(traj: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {}
    for col in ["frame", "agent_id", "pos_x", "pos_y", "grid_x", "grid_y", "grid_row", "is_walkable_cell"]:
        arrays[col] = traj[col].to_numpy()
    return arrays


def build_frame_arrays(traj: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    frames: dict[int, dict[str, np.ndarray]] = {}
    for frame, indices in traj.groupby("frame", sort=False).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        frames[int(frame)] = {
            "grid_x": traj["grid_x"].to_numpy()[idx].astype(np.int64, copy=False),
            "grid_row": traj["grid_row"].to_numpy()[idx].astype(np.int64, copy=False),
            "agent_id": traj["agent_id"].to_numpy()[idx].astype(np.int64, copy=False),
        }
    return frames


def make_wall_force_field(walkable: np.ndarray) -> np.ndarray:
    distance = distance_transform_edt(walkable > 0.5).astype(np.float32)
    grad_row, grad_col = np.gradient(distance)
    norm = np.sqrt(grad_col**2 + grad_row**2) + 1e-6
    field = np.stack([grad_col / norm, -grad_row / norm], axis=0)
    return (field * np.exp(-distance / 3.0)[None, ...]).astype(np.float32)


class GridPolicyDataset(Dataset):
    def __init__(
        self,
        dataset_root: pathlib.Path,
        split: str,
        action_space: ActionSpace,
        crop_size: int = 33,
        max_samples_per_case: int = 256,
        max_cases: int | None = None,
        seed: int = 42,
        cache_size: int = 4,
        action_frame_stride: int = 1,
    ):
        self.dataset_root = pathlib.Path(dataset_root)
        self.split = split
        self.action_space = action_space
        self.crop_size = int(crop_size)
        self.cache_size = int(cache_size)
        self.action_frame_stride = max(int(action_frame_stride), 1)
        self.cache: OrderedDict[int, dict] = OrderedDict()

        manifest_path = self.dataset_root / "manifest_trajectory_grid.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        rng = np.random.default_rng(seed + {"train": 0, "val": 1000, "test": 2000}.get(split, 3000))
        if split == "train":
            manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if max_cases is not None:
            manifest = manifest.head(int(max_cases)).reset_index(drop=True)
        if manifest.empty:
            raise RuntimeError(f"No cases for split={split}")
        self.cases = manifest.to_dict("records")

        self.samples: list[tuple[int, int]] = []
        for case_idx, row in enumerate(self.cases):
            total = int(row["trajectory_rows"])
            count = min(max_samples_per_case, total)
            if split == "train":
                row_indices = rng.choice(total, size=count, replace=False)
            else:
                row_indices = np.linspace(0, total - 1, count, dtype=np.int64)
            self.samples.extend((case_idx, int(i)) for i in row_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_case(self, case_idx: int) -> dict:
        if case_idx in self.cache:
            item = self.cache.pop(case_idx)
            self.cache[case_idx] = item
            return item

        row = self.cases[case_idx]
        input_dir = pathlib.Path(row["input_dir"])
        target_dir = pathlib.Path(row["target_dir"])
        grid_payload = load_json(input_dir / "walkablearea_grid.json")
        exit_payload = load_json(input_dir / "exit_room.json")
        traj = pd.read_parquet(
            target_dir / "trajectory.parquet",
            columns=["frame", "agent_id", "pos_x", "pos_y", "grid_x", "grid_y", "grid_row", "is_walkable_cell"],
        )
        traj = traj.reset_index(drop=True)
        next_idx = build_next_index(traj, frame_stride=self.action_frame_stride)

        walkable = grid_to_array(grid_payload["grid"])
        goal = make_goal_grid(exit_payload, grid_payload["meta"])

        item = {
            "row": row,
            "arrays": dataframe_arrays(traj),
            "next_idx": next_idx,
            "walkable": walkable,
            "goal": goal,
            "wall_force": make_wall_force_field(walkable),
            "grid_meta": grid_payload["meta"],
            "exit_payload": exit_payload,
            "frames": build_frame_arrays(traj),
        }
        self.cache[case_idx] = item
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        case_idx, row_idx = self.samples[idx]
        case = self._load_case(case_idx)
        arrays = case["arrays"]

        grid_h, grid_w = case["walkable"].shape
        frame = int(arrays["frame"][row_idx])
        agent_id = int(arrays["agent_id"][row_idx])
        grid_x = int(arrays["grid_x"][row_idx])
        grid_y = int(arrays["grid_y"][row_idx])
        center_row = int(arrays["grid_row"][row_idx])
        center_col = grid_x
        occupancy = np.zeros_like(case["walkable"], dtype=np.float32)
        frame_agents = case["frames"].get(frame)
        nearest_dist = float(self.crop_size)
        if frame_agents is not None:
            gx = frame_agents["grid_x"]
            grow = frame_agents["grid_row"]
            valid = (gx >= 0) & (gx < grid_w) & (grow >= 0) & (grow < grid_h)
            occupancy[grow[valid], gx[valid]] = 1.0
            other = valid & (frame_agents["agent_id"] != agent_id)
            if np.any(other):
                dist = np.sqrt((gx[other] - center_col) ** 2 + (grow[other] - center_row) ** 2)
                nearest_dist = float(np.min(dist))
        if 0 <= center_row < grid_h and 0 <= center_col < grid_w:
            occupancy[center_row, center_col] = 0.0

        map_crop = crop_centered(case["walkable"], center_row, center_col, self.crop_size, pad_value=0.0)
        goal_crop = crop_centered(case["goal"], center_row, center_col, self.crop_size, pad_value=0.0)
        occ_crop = crop_centered(occupancy, center_row, center_col, self.crop_size, pad_value=0.0)
        x_map = np.stack([map_crop, goal_crop, occ_crop], axis=0).astype(np.float32)

        next_row_idx = int(case["next_idx"][row_idx])
        stop_target = 1.0 if next_row_idx < 0 else 0.0
        if next_row_idx < 0:
            action_target = self.action_space.wait_action_id
        else:
            dx = int(arrays["grid_x"][next_row_idx]) - grid_x
            dy = int(arrays["grid_y"][next_row_idx]) - grid_y
            action_target = self.action_space.action_id_for_delta(dx, dy)

        exit_node = case["exit_payload"]["exit_node"]
        goal_cx, goal_cy = exit_node["centroid"]
        cell = float(case["grid_meta"]["cell_size_m"])
        dx_goal_m = float(goal_cx) - float(arrays["pos_x"][row_idx])
        dy_goal_m = float(goal_cy) - float(arrays["pos_y"][row_idx])
        dist_goal_m = (dx_goal_m**2 + dy_goal_m**2) ** 0.5
        agent_count = len(frame_agents["agent_id"]) if frame_agents is not None else 1
        repulsion = np.zeros(2, dtype=np.float32)
        if frame_agents is not None:
            other = frame_agents["agent_id"] != agent_id
            dx = grid_x - frame_agents["grid_x"][other]
            other_grid_y = grid_h - 1 - frame_agents["grid_row"][other]
            dy = grid_y - other_grid_y
            distance = np.sqrt(dx**2 + dy**2).clip(min=1e-4)
            if len(distance):
                weights = np.exp(-distance / 4.0)
                repulsion = np.tanh(
                    np.array([(dx / distance * weights).sum(), (dy / distance * weights).sum()], dtype=np.float32)
                )
        wall_force = np.zeros(2, dtype=np.float32)
        if 0 <= center_row < grid_h and 0 <= center_col < grid_w:
            wall_force = case["wall_force"][:, center_row, center_col]

        features = np.array(
            [
                float(grid_x) / max(grid_w - 1, 1),
                float(grid_y) / max(grid_h - 1, 1),
                dx_goal_m / max(grid_w * cell, 1e-6),
                dy_goal_m / max(grid_h * cell, 1e-6),
                dist_goal_m / max(((grid_w * cell) ** 2 + (grid_h * cell) ** 2) ** 0.5, 1e-6),
                min(nearest_dist / max(self.crop_size, 1), 1.0),
                min(agent_count / 250.0, 1.0),
                float(arrays["is_walkable_cell"][row_idx]),
                float(repulsion[0]),
                float(repulsion[1]),
                float(wall_force[0]),
                float(wall_force[1]),
            ],
            dtype=np.float32,
        )

        return {
            "map": torch.from_numpy(x_map),
            "features": torch.from_numpy(features),
            "action": torch.tensor(action_target, dtype=torch.long),
            "stop": torch.tensor(stop_target, dtype=torch.float32),
        }

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_rollout(trajectory: pd.DataFrame, walkable_grid: np.ndarray, action_trace: pd.DataFrame | None = None) -> dict:
    if trajectory.empty:
        return {"frames": 0, "agents": 0, "rows": 0, "walkable_ratio": 0.0, "collision_count": 0}

    grid_h, grid_w = walkable_grid.shape
    inside = (
        (trajectory["grid_x"] >= 0)
        & (trajectory["grid_x"] < grid_w)
        & (trajectory["grid_row"] >= 0)
        & (trajectory["grid_row"] < grid_h)
    )
    walkable = []
    for _, row in trajectory.iterrows():
        if not bool(inside.loc[row.name]):
            walkable.append(False)
        else:
            walkable.append(bool(walkable_grid[int(row["grid_row"]), int(row["grid_x"])] > 0.5))

    collision_count = 0
    for _, frame_df in trajectory.groupby("frame"):
        occupancy = frame_df.groupby(["grid_x", "grid_y"]).size()
        collision_count += int((occupancy > 1).sum())

    path_lengths = []
    moving_agents = 0
    for _, agent_df in trajectory.groupby("agent_id", sort=False):
        ordered = agent_df.sort_values("frame")
        dx = ordered["grid_x"].diff().fillna(0).abs().to_numpy()
        dy = ordered["grid_y"].diff().fillna(0).abs().to_numpy()
        path_len = float(np.sum(dx + dy))
        path_lengths.append(path_len)
        if path_len > 0:
            moving_agents += 1

    summary = {
        "frames": int(trajectory["frame"].nunique()),
        "agents": int(trajectory["agent_id"].nunique()),
        "rows": int(len(trajectory)),
        "walkable_ratio": float(np.mean(walkable)),
        "collision_count": collision_count,
        "stopped_agents": int(trajectory.groupby("agent_id")["stopped"].max().sum()) if "stopped" in trajectory.columns else 0,
        "moving_agents": int(moving_agents),
        "movement_steps": int(sum(path_lengths)),
        "mean_path_cells": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "max_path_cells": float(np.max(path_lengths)) if path_lengths else 0.0,
    }

    if action_trace is not None and not action_trace.empty:
        action_counts = action_trace["action_name"].value_counts().to_dict()
        summary["wait_steps"] = int(action_counts.get("wait", 0))
        summary["move_decisions"] = int(action_trace["action_kind"].eq("move").sum()) if "action_kind" in action_trace.columns else 0
        summary["blocked_by_wall_steps"] = int(action_trace.get("blocked_by_wall", pd.Series(dtype=bool)).sum())
        summary["blocked_by_collision_steps"] = int(action_trace.get("blocked_by_collision", pd.Series(dtype=bool)).sum())
        summary["action_counts"] = {str(k): int(v) for k, v in action_counts.items()}

    return summary

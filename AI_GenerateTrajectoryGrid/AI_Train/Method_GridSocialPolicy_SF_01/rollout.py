from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap
from shapely.geometry import Point, Polygon

from action_space import ActionSpace
from dataset_grid_policy import crop_centered, grid_to_array, make_goal_grid
from metrics import summarize_rollout
from model_grid_policy import GridSocialPolicyNet


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def grid_center(gx: int, gy: int, meta: dict) -> tuple[float, float]:
    cell = float(meta["cell_size_m"])
    return float(meta["origin_x"]) + (gx + 0.5) * cell, float(meta["origin_y"]) + (gy + 0.5) * cell


def world_to_grid_plot(x: float, y: float, meta: dict) -> tuple[float, float]:
    cell = float(meta["cell_size_m"])
    gx = (float(x) - float(meta["origin_x"])) / cell
    gy = (float(y) - float(meta["origin_y"])) / cell
    row = int(meta["height"]) - gy
    return gx, row


def save_rollout_preview(
    output_path: pathlib.Path,
    walkable: np.ndarray,
    rollout_df: pd.DataFrame,
    spawn_df: pd.DataFrame,
    exit_payload: dict,
    grid_meta: dict,
    summary: dict | None = None,
) -> None:
    height, width = walkable.shape
    aspect = max(width / max(height, 1), 0.25)
    fig_width = min(18.0, max(7.0, 7.0 * aspect))
    fig_height = min(12.0, max(4.5, fig_width / aspect))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    cmap = ListedColormap(["#101820", "#f3f6f8"])
    ax.imshow(walkable, cmap=cmap, origin="upper", interpolation="nearest")

    if exit_payload.get("exit_node", {}).get("polygon"):
        poly_points = [world_to_grid_plot(x, y, grid_meta) for x, y in exit_payload["exit_node"]["polygon"]]
        xs = [p[0] for p in poly_points]
        ys = [p[1] for p in poly_points]
        ax.fill(xs, ys, color="#f59e0b", alpha=0.35, label="exit room")
        ax.plot(xs, ys, color="#f97316", linewidth=1.3)

    if not rollout_df.empty:
        for _, agent_df in rollout_df.groupby("agent_id"):
            ordered = agent_df.sort_values("frame")
            moved = bool(((ordered["grid_x"].diff().fillna(0) != 0) | (ordered["grid_row"].diff().fillna(0) != 0)).any())
            if moved:
                ax.plot(ordered["grid_x"], ordered["grid_row"], linewidth=1.2, alpha=0.78)
                ax.scatter(ordered["grid_x"].iloc[-1], ordered["grid_row"].iloc[-1], s=13, c="#ef4444", zorder=4)
            else:
                ax.scatter(
                    ordered["grid_x"].iloc[0],
                    ordered["grid_row"].iloc[0],
                    s=60,
                    facecolors="none",
                    edgecolors="#ef4444",
                    linewidths=1.0,
                    zorder=4,
                )

    ax.scatter(spawn_df["grid_x"], spawn_df["grid_row"], s=18, c="#22c55e", edgecolors="#052e16", linewidths=0.4, label="spawn")
    if summary and int(summary.get("movement_steps", 0)) == 0:
        ax.text(
            0.015,
            0.985,
            "model predicted no movement (all paths length = 0 cells)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#ef4444",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#ef4444", "alpha": 0.92},
        )
    ax.set_title("Model rollout sample")
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_agent_features(
    agent: dict,
    frame_agents: list[dict],
    exit_payload: dict,
    grid_meta: dict,
    crop_size: int,
    wall_force: np.ndarray,
) -> np.ndarray:
    width = int(grid_meta["width"])
    height = int(grid_meta["height"])
    cell = float(grid_meta["cell_size_m"])
    goal_cx, goal_cy = exit_payload["exit_node"]["centroid"]
    x, y = grid_center(agent["grid_x"], agent["grid_y"], grid_meta)
    dx_goal_m = float(goal_cx) - x
    dy_goal_m = float(goal_cy) - y
    dist_goal_m = (dx_goal_m**2 + dy_goal_m**2) ** 0.5
    nearest = float(crop_size)
    repulsion_terms = []
    for other in frame_agents:
        if other["agent_id"] == agent["agent_id"] or other["stopped"]:
            continue
        dist = ((other["grid_x"] - agent["grid_x"]) ** 2 + (other["grid_row"] - agent["grid_row"]) ** 2) ** 0.5
        nearest = min(nearest, dist)
        dx = agent["grid_x"] - other["grid_x"]
        dy = agent["grid_y"] - other["grid_y"]
        metric = max((dx**2 + dy**2) ** 0.5, 1e-4)
        weight = np.exp(-metric / 4.0)
        repulsion_terms.append((dx / metric * weight, dy / metric * weight))
    repulsion = (
        np.tanh(np.asarray(repulsion_terms, dtype=np.float32).sum(axis=0))
        if repulsion_terms else np.zeros(2, dtype=np.float32)
    )
    row = int(agent["grid_row"])
    col = int(agent["grid_x"])
    wall = wall_force[:, row, col] if 0 <= row < wall_force.shape[1] and 0 <= col < wall_force.shape[2] else np.zeros(2)
    active_count = sum(1 for item in frame_agents if not item["stopped"])
    return np.array(
        [
            agent["grid_x"] / max(width - 1, 1),
            agent["grid_y"] / max(height - 1, 1),
            dx_goal_m / max(width * cell, 1e-6),
            dy_goal_m / max(height * cell, 1e-6),
            dist_goal_m / max(((width * cell) ** 2 + (height * cell) ** 2) ** 0.5, 1e-6),
            min(nearest / max(crop_size, 1), 1.0),
            min(active_count / 250.0, 1.0),
            1.0,
            float(repulsion[0]),
            float(repulsion[1]),
            float(wall[0]),
            float(wall[1]),
        ],
        dtype=np.float32,
    )


def rollout_case(
    checkpoint_path: pathlib.Path,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    max_steps: int,
    stop_threshold: float,
    crop_size: int,
    wait_logit_bias: float = 0.0,
    disable_wait: bool = False,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    action_space = ActionSpace(checkpoint["action_space"])
    cfg = checkpoint.get("config", {})

    grid_payload = load_json(input_dir / "walkablearea_grid.json")
    exit_payload = load_json(input_dir / "exit_room.json")
    spawn = pd.read_parquet(input_dir / "spawn_agent.parquet")
    walkable = grid_to_array(grid_payload["grid"])
    wall_distance = distance_transform_edt(walkable > 0.5).astype(np.float32)
    wall_grad_row, wall_grad_col = np.gradient(wall_distance)
    wall_norm = np.sqrt(wall_grad_col**2 + wall_grad_row**2) + 1e-6
    wall_force = np.stack([wall_grad_col / wall_norm, -wall_grad_row / wall_norm], axis=0)
    wall_force *= np.exp(-wall_distance / 3.0)[None, ...]
    goal = make_goal_grid(exit_payload, grid_payload["meta"])
    exit_poly = Polygon(exit_payload["exit_node"]["polygon"])

    device = get_device()
    model = GridSocialPolicyNet(
        num_actions=action_space.num_actions,
        feature_dim=int(cfg.get("feature_dim", 12)),
        base_channels=int(cfg.get("base_channels", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    grid_h, grid_w = walkable.shape
    agents = []
    for _, row in spawn.iterrows():
        agents.append(
            {
                "agent_id": int(row["agent_id"]),
                "grid_x": int(row["grid_x"]),
                "grid_y": int(row["grid_y"]),
                "grid_row": int(row["grid_row"]),
                "stopped": False,
            }
        )

    rows = []
    action_trace_rows = []
    with torch.no_grad():
        for frame in range(max_steps + 1):
            for agent in agents:
                x, y = grid_center(agent["grid_x"], agent["grid_y"], grid_payload["meta"])
                rows.append(
                    {
                        "frame": frame,
                        "agent_id": agent["agent_id"],
                        "pos_x": x,
                        "pos_y": y,
                        "grid_x": agent["grid_x"],
                        "grid_y": agent["grid_y"],
                        "grid_row": agent["grid_row"],
                        "stopped": agent["stopped"],
                    }
                )
            if all(agent["stopped"] for agent in agents):
                break
            if frame >= max_steps:
                break

            occupancy = np.zeros_like(walkable, dtype=np.float32)
            for agent in agents:
                if not agent["stopped"] and 0 <= agent["grid_row"] < grid_h and 0 <= agent["grid_x"] < grid_w:
                    occupancy[agent["grid_row"], agent["grid_x"]] = 1.0

            proposals = {}
            trace_index_by_agent_id = {}
            active_agents = []
            map_crops = []
            feature_rows = []
            for agent in agents:
                if agent["stopped"]:
                    continue
                x, y = grid_center(agent["grid_x"], agent["grid_y"], grid_payload["meta"])
                if exit_poly.covers(Point(x, y)):
                    agent["stopped"] = True
                    action_trace_rows.append(
                        {
                            "frame": frame,
                            "agent_id": agent["agent_id"],
                            "action_id": -1,
                            "action_name": "stop_exit_trigger",
                            "action_kind": "stop",
                            "dx": 0,
                            "dy": 0,
                            "stop_prob": 1.0,
                            "proposed_grid_x": agent["grid_x"],
                            "proposed_grid_y": agent["grid_y"],
                            "proposed_grid_row": agent["grid_row"],
                            "blocked_by_wall": False,
                            "blocked_by_collision": False,
                            "accepted": True,
                        }
                    )
                    continue

                map_crop = crop_centered(walkable, agent["grid_row"], agent["grid_x"], crop_size, pad_value=0.0)
                goal_crop = crop_centered(goal, agent["grid_row"], agent["grid_x"], crop_size, pad_value=0.0)
                occ_crop = crop_centered(occupancy, agent["grid_row"], agent["grid_x"], crop_size, pad_value=0.0)
                occ_crop[crop_size // 2, crop_size // 2] = 0.0
                map_crops.append(np.stack([map_crop, goal_crop, occ_crop], axis=0).astype(np.float32))
                feature_rows.append(
                    make_agent_features(agent, agents, exit_payload, grid_payload["meta"], crop_size, wall_force)
                )
                active_agents.append(agent)

            if active_agents:
                grid_tensor = torch.from_numpy(np.stack(map_crops, axis=0)).to(device)
                feature_tensor = torch.from_numpy(np.stack(feature_rows, axis=0)).to(device)
                action_logits, stop_logits = model(grid_tensor, feature_tensor)
                stop_probs = torch.sigmoid(stop_logits)
                if wait_logit_bias:
                    action_logits[:, action_space.wait_action_id] += float(wait_logit_bias)
                if disable_wait:
                    action_logits[:, action_space.wait_action_id] = -1.0e9
                action_ids = torch.argmax(action_logits, dim=1).detach().cpu().numpy()
                stop_probs_np = stop_probs.detach().cpu().numpy()

            for agent, action_id_value, stop_prob_value in zip(active_agents, action_ids, stop_probs_np):
                stop_prob = float(stop_prob_value)
                if stop_prob >= stop_threshold:
                    agent["stopped"] = True
                    action_trace_rows.append(
                        {
                            "frame": frame,
                            "agent_id": agent["agent_id"],
                            "action_id": -1,
                            "action_name": "stop_head",
                            "action_kind": "stop",
                            "dx": 0,
                            "dy": 0,
                            "stop_prob": stop_prob,
                            "proposed_grid_x": agent["grid_x"],
                            "proposed_grid_y": agent["grid_y"],
                            "proposed_grid_row": agent["grid_row"],
                            "blocked_by_wall": False,
                            "blocked_by_collision": False,
                            "accepted": True,
                        }
                    )
                    continue
                action_id = int(action_id_value)
                action_payload = action_space.actions[action_id]
                dx, dy = action_space.delta_for_action_id(action_id)
                new_gx = agent["grid_x"] + dx
                new_gy = agent["grid_y"] + dy
                new_row = grid_h - 1 - new_gy
                blocked_by_wall = False
                if not (0 <= new_gx < grid_w and 0 <= new_row < grid_h and walkable[new_row, new_gx] > 0.5):
                    blocked_by_wall = True
                    new_gx, new_gy, new_row = agent["grid_x"], agent["grid_y"], agent["grid_row"]
                proposals[agent["agent_id"]] = (new_gx, new_gy, new_row)
                trace_index_by_agent_id[agent["agent_id"]] = len(action_trace_rows)
                action_trace_rows.append(
                    {
                        "frame": frame,
                        "agent_id": agent["agent_id"],
                        "action_id": action_id,
                        "action_name": str(action_payload["name"]),
                        "action_kind": str(action_payload["kind"]),
                        "dx": int(dx),
                        "dy": int(dy),
                        "stop_prob": stop_prob,
                        "proposed_grid_x": int(new_gx),
                        "proposed_grid_y": int(new_gy),
                        "proposed_grid_row": int(new_row),
                        "blocked_by_wall": bool(blocked_by_wall),
                        "blocked_by_collision": False,
                        "accepted": False,
                    }
                )

            counts = {}
            for cell in proposals.values():
                counts[(cell[0], cell[1])] = counts.get((cell[0], cell[1]), 0) + 1
            for agent in agents:
                if agent["agent_id"] not in proposals:
                    continue
                new_gx, new_gy, new_row = proposals[agent["agent_id"]]
                if counts[(new_gx, new_gy)] > 1:
                    trace_idx = trace_index_by_agent_id.get(agent["agent_id"])
                    if trace_idx is not None:
                        action_trace_rows[trace_idx]["blocked_by_collision"] = True
                    continue
                trace_idx = trace_index_by_agent_id.get(agent["agent_id"])
                if trace_idx is not None:
                    action_trace_rows[trace_idx]["accepted"] = True
                agent["grid_x"], agent["grid_y"], agent["grid_row"] = new_gx, new_gy, new_row

    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_df = pd.DataFrame(rows)
    action_trace_df = pd.DataFrame(action_trace_rows)
    rollout_df.to_parquet(output_dir / "rollout.parquet", index=False)
    action_trace_df.to_parquet(output_dir / "action_trace.parquet", index=False)
    summary = summarize_rollout(rollout_df, walkable, action_trace=action_trace_df)
    summary["wait_logit_bias"] = float(wait_logit_bias)
    summary["disable_wait"] = bool(disable_wait)
    preview_path = output_dir / "samples" / "rollout_preview.png"
    save_rollout_preview(preview_path, walkable, rollout_df, spawn, exit_payload, grid_payload["meta"], summary=summary)
    summary["sample_preview"] = str(preview_path)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out GridSocialPolicyNet from one prepared A case.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stop-threshold", type=float, default=0.8)
    parser.add_argument("--crop-size", type=int, default=33)
    parser.add_argument("--wait-logit-bias", type=float, default=0.0, help="Add this value to the wait action logit during rollout. Negative values reduce wait.")
    parser.add_argument("--disable-wait", action="store_true", help="Mask the wait action during rollout diagnostics.")
    args = parser.parse_args()

    summary = rollout_case(
        args.checkpoint.resolve(),
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        args.max_steps,
        args.stop_threshold,
        args.crop_size,
        wait_logit_bias=args.wait_logit_bias,
        disable_wait=args.disable_wait,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

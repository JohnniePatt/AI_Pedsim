from __future__ import annotations

import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

base = pathlib.Path(__file__).resolve().parent.parent
dataset_root = base / "Dataset/Data_TrajectoryGrid/Topo_HouseGAN"
manifest_p = dataset_root / "manifest_trajectory_grid.csv"

out_gt_dir = base / "AI_GenerateTimeseries/AI_Result/GroundTruth_Previews"
out_gt_dir.mkdir(parents=True, exist_ok=True)

def grid_to_array(grid_rows: list[str]) -> np.ndarray:
    height = len(grid_rows)
    width = len(grid_rows[0])
    data = np.frombuffer("".join(grid_rows).encode("ascii"), dtype=np.uint8).reshape(height, width)
    return (data == ord("1")).astype(np.float32)

def world_to_grid_plot(x: float, y: float, meta: dict) -> tuple[float, float]:
    cell = float(meta["cell_size_m"])
    gx = (float(x) - float(meta["origin_x"])) / cell
    gy = (float(y) - float(meta["origin_y"])) / cell
    row = int(meta["height"]) - gy
    return gx, row

if manifest_p.exists():
    manifest = pd.read_csv(manifest_p)
    target_plans = ["plan_110_fbd0", "plan_102_8e0f"]

    for target in target_plans:
        sub = manifest[manifest["plan_name"] == target]
        for _, row in sub.iterrows():
            a_dir = dataset_root / str(row["input_dir"])
            b_dir = dataset_root / str(row["target_dir"])
            gt_parquet = b_dir / "trajectory.parquet"
            grid_json = a_dir / "walkablearea_grid.json"
            exit_json = a_dir / "exit_room.json"
            spawn_p = a_dir / "spawn_agent.parquet"
            
            if gt_parquet.exists() and grid_json.exists():
                payload = json.loads(grid_json.read_text())
                walkable = grid_to_array(payload["grid"])
                grid_meta = payload.get("meta", payload.get("metadata", {"cell_size_m": 0.1, "origin_x": 0, "origin_y": 0, "height": walkable.shape[0], "width": walkable.shape[1]}))
                
                exit_payload = json.loads(exit_json.read_text())
                spawn_df = pd.read_parquet(spawn_p)
                gt_df = pd.read_parquet(gt_parquet)
                
                case_name = f"{row['split']}_{row['plan_name']}_{row['sqlite_stem']}"
                
                height, width = walkable.shape
                aspect = max(width / max(height, 1), 0.25)
                fig_w = min(18.0, max(7.0, 7.0 * aspect))
                fig_h = min(12.0, max(4.5, fig_w / aspect))

                # ── 1. Ground Truth (grid) Preview ──
                out_grid_img = out_gt_dir / f"{case_name}_gt_grid_preview.png"
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
                cmap = ListedColormap(["#101820", "#f3f6f8"])
                ax.imshow(walkable, cmap=cmap, origin="upper", interpolation="nearest")

                if exit_payload.get("exit_node", {}).get("polygon"):
                    poly = exit_payload["exit_node"]["polygon"]
                    poly_pts = [world_to_grid_plot(x, y, grid_meta) for x, y in poly]
                    xs = [p[0] for p in poly_pts]
                    ys = [p[1] for p in poly_pts]
                    ax.fill(xs, ys, color="#f59e0b", alpha=0.35, label="exit room")
                    ax.plot(xs, ys, color="#f97316", linewidth=1.3)

                for _, agent_df in gt_df.groupby("agent_id"):
                    ordered = agent_df.sort_values("frame")
                    ax.plot(ordered["grid_x"], ordered["grid_row"], linewidth=1.2, alpha=0.78)

                ax.scatter(spawn_df["grid_x"], spawn_df["grid_row"], s=18, c="#22c55e", edgecolors="#052e16", linewidths=0.4, label="spawn")
                ax.set_title("Ground truth (grid)")
                ax.set_axis_off()
                ax.legend(loc="upper right", frameon=True, fontsize=8)
                fig.tight_layout()
                fig.savefig(out_grid_img, bbox_inches="tight")
                plt.close(fig)

                # ── 2. Ground Truth (raw) Preview (Exact Canvas & Frame Match) ──
                out_raw_img = out_gt_dir / f"{case_name}_gt_raw_preview.png"
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
                ax.imshow(walkable, cmap=cmap, origin="upper", interpolation="nearest")

                if exit_payload.get("exit_node", {}).get("polygon"):
                    poly = exit_payload["exit_node"]["polygon"]
                    poly_pts = [world_to_grid_plot(x, y, grid_meta) for x, y in poly]
                    xs = [p[0] for p in poly_pts]
                    ys = [p[1] for p in poly_pts]
                    ax.fill(xs, ys, color="#f59e0b", alpha=0.35, label="exit room")
                    ax.plot(xs, ys, color="#f97316", linewidth=1.3)

                for _, agent_df in gt_df.groupby("agent_id"):
                    ordered = agent_df.sort_values("frame")
                    gx_pts = []
                    gy_pts = []
                    for px, py in zip(ordered["pos_x"], ordered["pos_y"]):
                        gx, gy = world_to_grid_plot(px, py, grid_meta)
                        gx_pts.append(gx)
                        gy_pts.append(gy)
                    ax.plot(gx_pts, gy_pts, linewidth=1.2, alpha=0.78)

                ax.scatter(spawn_df["grid_x"], spawn_df["grid_row"], s=18, c="#22c55e", edgecolors="#052e16", linewidths=0.4, label="spawn")
                ax.set_title("Ground truth (raw)")
                ax.set_axis_off()
                ax.legend(loc="upper right", frameon=True, fontsize=8)
                fig.tight_layout()
                fig.savefig(out_raw_img, bbox_inches="tight")
                plt.close(fig)
                print(f"Generated 100% Identical Canvas GT Raw & Grid previews for: {case_name}")

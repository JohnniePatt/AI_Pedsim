from __future__ import annotations

import json
import os
import pathlib
import sys
import numpy as np
import pandas as pd
import torch

# Base project paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
GRID_POLICY_DIR = PROJECT_ROOT / "AI_GenerateTimeseries/AI_Train/Method_GridSocialPolicy"
sys.path.insert(0, str(GRID_POLICY_DIR))

from rollout import grid_to_array, load_json, save_rollout_preview
from dataset_grid_policy import grid_to_array

TS_RESULT_DIR = PROJECT_ROOT / "AI_GenerateTimeseries/AI_Result"
HOUSEGAN_ROOT = PROJECT_ROOT / "Dataset/Data_TrajectoryGrid/Topo_HouseGAN"


def get_housegan_10_full_cases() -> list[dict]:
    """Gets HouseGAN N-agent cases prioritizing user requested target plans."""
    manifest_p = HOUSEGAN_ROOT / "manifest_trajectory_grid.csv"
    if not manifest_p.exists():
        return []

    manifest = pd.read_csv(manifest_p)
    full_df = manifest[manifest["sqlite_stem"].str.contains("_full") | manifest["sqlite_stem"].str.contains("100044_02")].reset_index(drop=True)
    
    target_names = ["plan_110_fbd0", "plan_102_8e0f", "plan_100_d769"]
    cases = []
    
    # Priority targets
    for target in target_names:
        sub = full_df[full_df["plan_name"] == target]
        if not sub.empty:
            full_sub = sub[sub["sqlite_stem"].str.contains("_full")]
            row = full_sub.iloc[0] if not full_sub.empty else sub.iloc[0]
            input_dir = HOUSEGAN_ROOT / str(row["input_dir"])
            cases.append(
                {
                    "case_name": f"{row['split']}_{row['plan_name']}_{row['sqlite_stem']}",
                    "plan_name": str(row["plan_name"]),
                    "split": str(row["split"]),
                    "input_dir": input_dir,
                }
            )
            
    # Explicit half case for plan_110_fbd0
    half_sub = full_df[(full_df["plan_name"] == "plan_110_fbd0") & (full_df["sqlite_stem"].str.contains("100044_02"))]
    if not half_sub.empty:
        row = half_sub.iloc[0]
        input_dir = HOUSEGAN_ROOT / str(row["input_dir"])
        cases.append(
            {
                "case_name": f"{row['split']}_{row['plan_name']}_{row['sqlite_stem']}",
                "plan_name": str(row["plan_name"]),
                "split": str(row["split"]),
                "input_dir": input_dir,
            }
        )
            
    # Additional unique cases
    unique_df = full_df.drop_duplicates(subset=["plan_name"]).reset_index(drop=True)
    for idx, row in unique_df.head(10).iterrows():
        pname = str(row["plan_name"])
        if any(c["plan_name"] == pname for c in cases):
            continue
        input_dir = HOUSEGAN_ROOT / str(row["input_dir"])
        cases.append(
            {
                "case_name": f"{row['split']}_{row['plan_name']}_{row['sqlite_stem']}",
                "plan_name": pname,
                "split": str(row["split"]),
                "input_dir": input_dir,
            }
        )
    return cases


def run_model_inference_and_plot(model_name: str, weights_path: pathlib.Path, out_evaluate_dir: pathlib.Path):
    """Runs evaluation inference for a specific model and generates standardized rollout preview plots."""
    raise RuntimeError(
        "Disabled invalid evaluator: this function generated A* paths for every "
        "model without executing the supplied checkpoint. Use each method's real "
        "test script; Transformer uses AI_Train/Method_Transformer/test_transformer.py."
    )
    out_evaluate_dir.mkdir(parents=True, exist_ok=True)
    cases = get_housegan_10_full_cases()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating {model_name} on {len(cases)} HouseGAN N-Agent Full Cases ===")
    
    # Check if checkpoint exists
    has_weights = weights_path.exists()
    if has_weights:
        print(f"  [Loaded Weights] {weights_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [Warning] Checkpoint missing at {weights_path}, generating forward prediction outputs.")

    for idx, c_info in enumerate(cases):
        case_name = c_info["case_name"]
        input_dir = c_info["input_dir"]
        
        if not (input_dir / "walkablearea_grid.json").exists():
            continue

        grid_payload = load_json(input_dir / "walkablearea_grid.json")
        exit_payload = load_json(input_dir / "exit_room.json")
        spawn_df = pd.read_parquet(input_dir / "spawn_agent.parquet")
        walkable = grid_to_array(grid_payload["grid"])
        grid_meta = grid_payload["meta"]
        
        height, width = walkable.shape
        cell_size = float(grid_meta["cell_size_m"])

        # Generate predicted agent trajectories for N agents following walkable grid paths
        rollout_rows = []
        
        goal_cx, goal_cy = exit_payload["exit_node"]["centroid"]
        goal_gx = int((goal_cx - grid_meta["origin_x"]) / cell_size)
        goal_gy = int((goal_cy - grid_meta["origin_y"]) / cell_size)
        goal_row = height - goal_gy

        # Ensure goal cell is within bounds and walkable if possible
        goal_gx = max(0, min(width - 1, goal_gx))
        goal_row = max(0, min(height - 1, goal_row))

        import heapq

        def find_grid_path(walkable_grid: np.ndarray, start_x: int, start_r: int, end_x: int, end_r: int) -> list[tuple[int, int]]:
            h, w = walkable_grid.shape
            start = (start_x, start_r)
            goal = (end_x, end_r)
            if start == goal:
                return [start]
            
            # If goal cell is not walkable, find nearest walkable cell near goal
            if walkable_grid[goal[1], goal[0]] < 0.5:
                found = False
                for rad in range(1, 10):
                    for dx in range(-rad, rad + 1):
                        for dr in range(-rad, rad + 1):
                            nx, nr = goal[0] + dx, goal[1] + dr
                            if 0 <= nx < w and 0 <= nr < h and walkable_grid[nr, nx] > 0.5:
                                goal = (nx, nr)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

            open_set = []
            heapq.heappush(open_set, (0, start))
            came_from = {}
            g_score = {start: 0.0}
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

            while open_set:
                _, current = heapq.heappop(open_set)
                if current == goal:
                    path = [goal]
                    while path[-1] in came_from:
                        path.append(came_from[path[-1]])
                    path.reverse()
                    return path

                cx, cr = current
                for dx, dr in dirs:
                    nx, nr = cx + dx, cr + dr
                    if 0 <= nx < w and 0 <= nr < h:
                        if walkable_grid[nr, nx] > 0.5:
                            cost = 1.414 if (dx != 0 and dr != 0) else 1.0
                            tentative = g_score[current] + cost
                            nxt = (nx, nr)
                            if tentative < g_score.get(nxt, float("inf")):
                                came_from[nxt] = current
                                g_score[nxt] = tentative
                                f_score = tentative + float(np.hypot(nx - goal[0], nr - goal[1]))
                                heapq.heappush(open_set, (f_score, nxt))

            # Fallback path if disconnected
            return [start, goal]

        for _, sp in spawn_df.iterrows():
            aid = int(sp["agent_id"])
            sx = max(0, min(width - 1, int(sp["grid_x"])))
            srow = max(0, min(height - 1, int(sp["grid_row"])))

            grid_path = find_grid_path(walkable, sx, srow, goal_gx, goal_row)

            for step_i, (px, pr) in enumerate(grid_path):
                rollout_rows.append(
                    {
                        "agent_id": aid,
                        "frame": step_i,
                        "grid_x": int(px),
                        "grid_row": int(pr),
                    }
                )

        rollout_df = pd.DataFrame(rollout_rows)
        save_img_path = out_evaluate_dir / f"{case_name}_rollout_preview.png"
        
        save_rollout_preview(
            output_path=save_img_path,
            walkable=walkable,
            rollout_df=rollout_df,
            spawn_df=spawn_df,
            exit_payload=exit_payload,
            grid_meta=grid_meta,
            summary={"movement_steps": len(rollout_df)},
        )
        print(f"  [{idx + 1}/10] Saved {model_name} -> {save_img_path.relative_to(PROJECT_ROOT)}")


def main():
    print("=== Running Normalized HouseGAN Standardized Plot Evaluations ===")
    
    models = [
        (
            "Transformer",
            TS_RESULT_DIR / "Method_Transformer/outputs/run_33/weights/best_model.pth",
            TS_RESULT_DIR / "Method_Transformer/outputs/run_33_evaluate",
        ),
        (
            "GNN-CVAE",
            TS_RESULT_DIR / "Method_GNN_CVAE/outputs/run_6/weights/best_model.pth",
            TS_RESULT_DIR / "Method_GNN_CVAE/outputs/run_6_evaluate",
        ),
        (
            "Social GAN",
            TS_RESULT_DIR / "Method_SGAN/outputs/run_6/weights/sgan_ep10.pth",
            TS_RESULT_DIR / "Method_SGAN/outputs/run_6_evaluate",
        ),
        (
            "LSTM Baseline",
            TS_RESULT_DIR / "Method_LSTM_01/run_LSTM_20260327_184506/checkpoints/generator_best.pth",
            TS_RESULT_DIR / "Method_LSTM_01/outputs/run_LSTM_20260327_184506_evaluate",
        ),
        (
            "GPT_Knowledge",
            TS_RESULT_DIR / "Method_GPT_Knowledge/knowledge_manifest.json",
            TS_RESULT_DIR / "Method_GPT_Knowledge/outputs/run_gpt_knowledge_evaluate",
        ),
    ]
    
    for m_name, w_p, out_p in models:
        run_model_inference_and_plot(m_name, w_p, out_p)

    print("=== All Normalized HouseGAN Evaluations Complete! ===")


if __name__ == "__main__":
    main()

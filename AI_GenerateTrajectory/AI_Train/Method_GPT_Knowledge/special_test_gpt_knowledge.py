"""
special_test_gpt_knowledge.py
-----------------------------
Run a blind/special test on a manually provided scene and optionally compare
the generated result against an uploaded ground-truth trajectory parquet.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import pandas as pd

from generate_gpt_knowledge import generate_case_prediction, load_config as load_generate_config, resolve_from_config, save_generation
from prepare_geometry_gpt_knowledge import extract_agent_paths, infer_case_id, load_scene, point_is_inside_walkable, resample_path
from validate_gpt_knowledge import compute_collision_rate
from visual_gpt_knowledge import build_agent_colors, compute_metrics, plot_agent_trajectories, plot_polygon

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from shapely import wkt as shapely_wkt


def load_goal_centroid(case_dir: pathlib.Path):
    case_id = infer_case_id(case_dir)
    spawn_exit_path = case_dir / f"Spawn_exit_{case_id}.csv"
    if not spawn_exit_path.exists():
        return None
    df = pd.read_csv(spawn_exit_path)
    exit_rows = df[df["type"] == "exit_area"]
    if exit_rows.empty:
        return None
    poly = shapely_wkt.loads(exit_rows.iloc[0]["area"])
    return poly.centroid


def plot_special_visual(case_dir: pathlib.Path, pred_df: pd.DataFrame, out_path: pathlib.Path, gt_df: pd.DataFrame | None = None, title_prefix: str = ""):
    walkable = load_scene(case_dir)["walkable"]
    goal_centroid = load_goal_centroid(case_dir)
    titles = ["Walkable + GT vs AI", "Ground Truth", "AI Prediction"] if gt_df is not None else ["Walkable + AI", "AI Prediction"]
    ncols = 3 if gt_df is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(18 if ncols == 3 else 12, 6), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, title in zip(axes, titles):
        plot_polygon(ax, walkable, facecolor="#dfeedd", edgecolor="#4c6a55", alpha=0.85, linewidth=1.1)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if goal_centroid is not None:
            ax.scatter(goal_centroid.x, goal_centroid.y, c="#f2c94c", s=180, marker="*", edgecolors="black", linewidths=0.6, zorder=6)

    all_agent_ids = sorted(set(pred_df["id"].tolist()) | (set(gt_df["id"].tolist()) if gt_df is not None else set()))
    colors = build_agent_colors([int(x) for x in all_agent_ids])

    if gt_df is not None:
        gt_handles = plot_agent_trajectories(axes[0], gt_df, colors, linestyle="--", linewidth=1.8, alpha=0.8, label_prefix="GT")
        plot_agent_trajectories(axes[0], pred_df, colors, linestyle="-", linewidth=1.4, alpha=0.9, label_prefix=None)
        plot_agent_trajectories(axes[1], gt_df, colors, linestyle="--", linewidth=2.0, alpha=0.9, label_prefix=None)
        plot_agent_trajectories(axes[2], pred_df, colors, linestyle="-", linewidth=2.0, alpha=0.95, label_prefix=None)
    else:
        gt_handles = []
        plot_agent_trajectories(axes[0], pred_df, colors, linestyle="-", linewidth=1.8, alpha=0.95, label_prefix="AI")
        plot_agent_trajectories(axes[1], pred_df, colors, linestyle="-", linewidth=2.0, alpha=0.95, label_prefix=None)

    bounds = walkable.bounds
    pad_x = max((bounds[2] - bounds[0]) * 0.05, 1.0)
    pad_y = max((bounds[3] - bounds[1]) * 0.05, 1.0)
    for ax in axes:
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    legend_handles = [
        Patch(facecolor="#dfeedd", edgecolor="#4c6a55", label="Walkable Area"),
        plt.Line2D([0], [0], color="#d62728", linestyle="-", linewidth=2.0, label="AI Trajectory"),
        plt.Line2D([0], [0], color="#f2c94c", marker="*", linestyle="None", markersize=12, markeredgecolor="black", label="Goal"),
    ]
    if gt_df is not None:
        legend_handles.insert(1, plt.Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=2.0, label="GT Trajectory"))
    axes[0].legend(handles=legend_handles + gt_handles[:8], loc="best", fontsize=8)

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_with_gt(case_dir: pathlib.Path, pred_df: pd.DataFrame, gt_parquet: pathlib.Path, output_dir: pathlib.Path, cfg: dict):
    gt_df = pd.read_parquet(gt_parquet).sort_values(["id", "frame"]).reset_index(drop=True)
    shutil.copy2(gt_parquet, output_dir / "GT_uploaded.parquet")

    gt_paths = extract_agent_paths(gt_df)
    pred_paths = {int(k): {"points": g[["pos_x", "pos_y"]].to_numpy(dtype=np.float64)} for k, g in pred_df.groupby("id")}
    eval_steps = int(cfg.get("eval_steps", 100))
    rows = []
    for agent_id in sorted(gt_paths):
        if agent_id not in pred_paths:
            continue
        gt = gt_paths[agent_id]["points"]
        pred = pred_paths[agent_id]["points"]
        gt_eval = resample_path(gt, eval_steps)
        pred_eval = resample_path(pred, eval_steps)
        dists = np.linalg.norm(pred_eval - gt_eval, axis=1)
        rows.append(
            {
                "case_id": str(infer_case_id(case_dir)),
                "agent_id": int(agent_id),
                "ade_m": float(np.mean(dists)),
                "fde_m": float(dists[-1]),
                "gt_duration_frames": int(len(gt)),
                "pred_duration_frames": int(len(pred)),
                "duration_abs_error_frames": int(abs(len(pred) - len(gt))),
            }
        )

    walkable = load_scene(case_dir)["walkable"]
    outside = 0
    for row in pred_df.itertuples(index=False):
        outside += 0 if point_is_inside_walkable(row.pos_x, row.pos_y, walkable) else 1
    oob_rate = outside / max(len(pred_df), 1)
    collision_rate = compute_collision_rate(pred_paths, float(cfg.get("collision_threshold_m", 0.3)))
    ade_m, fde_m = compute_metrics(gt_df, pred_df)

    per_agent = pd.DataFrame(rows)
    summary = {
        "case_id": str(infer_case_id(case_dir)),
        "n_agents_eval": int(len(per_agent)),
        "path_ade_m": float(per_agent["ade_m"].mean()) if len(per_agent) else float(ade_m),
        "path_fde_m": float(per_agent["fde_m"].mean()) if len(per_agent) else float(fde_m),
        "duration_abs_error_frames": float(per_agent["duration_abs_error_frames"].mean()) if len(per_agent) else 0.0,
        "collision_rate": float(collision_rate),
        "out_of_bounds_rate": float(oob_rate),
    }

    per_agent.to_csv(output_dir / "compare_agent_metrics.csv", index=False)
    with open(output_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_special_visual(
        case_dir,
        pred_df,
        output_dir / "compare_ai_vs_gt.png",
        gt_df=gt_df,
        title_prefix=f"special_test | case {summary['case_id']} | ADE={summary['path_ade_m']:.3f} m | FDE={summary['path_fde_m']:.3f} m",
    )


def main(config_path: str, input_case_dir: str, output_dir: str, gt_parquet: str | None = None):
    cfg = load_generate_config(config_path)
    input_case_dir = pathlib.Path(input_case_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    knowledge_dir = resolve_from_config(config_path, cfg["knowledge_dir"])
    generation_dir = output_dir / "generation"
    compare_dir = output_dir / "compare"

    result = generate_case_prediction(input_case_dir, knowledge_dir, cfg)
    save_generation(result, generation_dir)

    case_id = result["case_id"]
    manifest = {
        "case_id": case_id,
        "input_case_dir": str(input_case_dir),
        "knowledge_dir": str(knowledge_dir),
        "output_dir": str(output_dir),
        "retrieved_cases": result["retrieved_cases"],
    }
    with open(output_dir / "special_test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    plot_special_visual(
        input_case_dir,
        result["prediction_df"],
        output_dir / "ai_prediction.png",
        gt_df=None,
        title_prefix=f"special_test | case {case_id} | blend_top={len(result.get('blend_cases', []))} | lead={result['retrieved_cases'][0]['case_id']} | AI prediction only",
    )

    print(f"[SpecialTest] Generated AI prediction for case_{case_id}")
    print(f"[SpecialTest] Output dir: {output_dir}")
    print(f"[SpecialTest] Retrieved top case: {result['retrieved_cases'][0]['case_id']} score={result['retrieved_cases'][0]['score']:.4f}")
    print(f"[SpecialTest] Blend cases used: {len(result.get('blend_cases', []))}")

    if gt_parquet:
        gt_parquet_path = pathlib.Path(gt_parquet).resolve()
        compare_dir.mkdir(parents=True, exist_ok=True)
        compare_with_gt(input_case_dir, result["prediction_df"], gt_parquet_path, compare_dir, cfg)
        print(f"[SpecialTest] Compared AI vs GT using {gt_parquet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(pathlib.Path(__file__).with_name("config_test.json")))
    parser.add_argument("--input_case_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gt_parquet", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.input_case_dir, args.output_dir, args.gt_parquet)

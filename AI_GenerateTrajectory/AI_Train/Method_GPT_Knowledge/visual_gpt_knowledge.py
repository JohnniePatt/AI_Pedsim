"""
visual_gpt_knowledge.py
-----------------------
Create visual comparisons for Method_GPT_Knowledge generated samples.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon

from prepare_geometry_gpt_knowledge import build_walkable_area, load_trajectory, resample_path


MAX_LEGEND_AGENTS = 8


def iter_sample_dirs(outputs_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted([p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("sample_case_")], key=lambda p: p.name)


def load_traj_table(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"frame", "id", "pos_x", "pos_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def plot_polygon(ax, geom, facecolor: str, edgecolor: str, alpha: float, linewidth: float = 1.0):
    if geom.is_empty:
        return
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            plot_polygon(ax, part, facecolor, edgecolor, alpha, linewidth)
        return
    if not isinstance(geom, Polygon):
        return

    ext = np.asarray(geom.exterior.coords)
    ax.fill(ext[:, 0], ext[:, 1], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    for hole in geom.interiors:
        coords = np.asarray(hole.coords)
        ax.fill(coords[:, 0], coords[:, 1], facecolor="white", edgecolor=edgecolor, alpha=1.0, linewidth=linewidth * 0.75)


def build_agent_colors(agent_ids: list[int]):
    cmap = plt.get_cmap("tab20")
    return {agent_id: cmap(idx % 20) for idx, agent_id in enumerate(agent_ids)}


def plot_agent_trajectories(ax, df: pd.DataFrame, colors: dict[int, tuple], linestyle: str, linewidth: float, alpha: float, label_prefix: str | None = None):
    handles = []
    shown = 0
    for agent_id, grp in df.groupby("id"):
        grp = grp.sort_values("frame")
        color = colors[int(agent_id)]
        line, = ax.plot(grp["pos_x"], grp["pos_y"], linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color)
        ax.scatter(grp.iloc[0]["pos_x"], grp.iloc[0]["pos_y"], color=color, s=14, alpha=min(alpha + 0.1, 1.0))
        ax.scatter(grp.iloc[-1]["pos_x"], grp.iloc[-1]["pos_y"], color=color, marker="x", s=28, alpha=min(alpha + 0.1, 1.0))
        if label_prefix is not None and shown < MAX_LEGEND_AGENTS:
            line.set_label(f"{label_prefix} agent {int(agent_id)}")
            handles.append(line)
            shown += 1
    return handles


def compute_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[float, float]:
    merged = gt_df.merge(pred_df, on=["frame", "id"], suffixes=("_gt", "_pred"))
    if not merged.empty:
        merged["dist"] = np.sqrt((merged["pos_x_pred"] - merged["pos_x_gt"]) ** 2 + (merged["pos_y_pred"] - merged["pos_y_gt"]) ** 2)
        ade = float(merged["dist"].mean())
        final_rows = merged.sort_values("frame").groupby("id").tail(1)
        fde = float(final_rows["dist"].mean()) if not final_rows.empty else 0.0
        return ade, fde

    gt_paths = {int(k): g[["pos_x", "pos_y"]].to_numpy(dtype=np.float64) for k, g in gt_df.groupby("id")}
    pred_paths = {int(k): g[["pos_x", "pos_y"]].to_numpy(dtype=np.float64) for k, g in pred_df.groupby("id")}
    dists = []
    final_dists = []
    for agent_id in sorted(set(gt_paths) & set(pred_paths)):
        gt_eval = resample_path(gt_paths[agent_id], 100)
        pred_eval = resample_path(pred_paths[agent_id], 100)
        path_dist = np.linalg.norm(pred_eval - gt_eval, axis=1)
        dists.extend(path_dist.tolist())
        final_dists.append(float(path_dist[-1]))
    ade = float(np.mean(dists)) if dists else 0.0
    fde = float(np.mean(final_dists)) if final_dists else 0.0
    return ade, fde


def load_goal_centroid(case_dir: pathlib.Path):
    case_id = case_dir.name.replace("case_", "")
    spawn_exit_path = case_dir / f"Spawn_exit_{case_id}.csv"
    if not spawn_exit_path.exists():
        return None
    df = pd.read_csv(spawn_exit_path)
    exit_rows = df[df["type"] == "exit_area"]
    if exit_rows.empty:
        return None
    poly = shapely_wkt.loads(exit_rows.iloc[0]["area"])
    return poly.centroid


def make_sample_plot(sample_dir: pathlib.Path, out_path: pathlib.Path) -> dict:
    pred_path = next(sample_dir.glob("AI_pred_*.parquet"), None)
    summary_path = sample_dir / "generation_summary.json"
    if pred_path is None or not summary_path.exists():
        raise FileNotFoundError(f"Missing AI_pred parquet or generation_summary.json in {sample_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case_dir = pathlib.Path(summary["case_dir"])
    gt_df = load_trajectory(case_dir)
    pred_df = load_traj_table(pred_path)
    walkable = build_walkable_area(case_dir / "Geo_room.json", case_dir / "Geo_corridor.json")
    goal_centroid = load_goal_centroid(case_dir)
    ade_m, fde_m = compute_metrics(gt_df, pred_df)

    all_agent_ids = sorted(set(gt_df["id"].tolist()) | set(pred_df["id"].tolist()))
    colors = build_agent_colors([int(x) for x in all_agent_ids])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = ["Walkable + GT vs AI", "Ground Truth", "AI Prediction"]
    for ax, title in zip(axes, titles):
        plot_polygon(ax, walkable, facecolor="#dfeedd", edgecolor="#4c6a55", alpha=0.85, linewidth=1.1)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        if goal_centroid is not None:
            ax.scatter(goal_centroid.x, goal_centroid.y, c="#f2c94c", s=180, marker="*", edgecolors="black", linewidths=0.6, zorder=6)

    gt_handles = plot_agent_trajectories(axes[0], gt_df, colors, linestyle="--", linewidth=1.8, alpha=0.8, label_prefix="GT")
    plot_agent_trajectories(axes[0], pred_df, colors, linestyle="-", linewidth=1.4, alpha=0.9, label_prefix=None)
    plot_agent_trajectories(axes[1], gt_df, colors, linestyle="--", linewidth=2.0, alpha=0.9, label_prefix=None)
    plot_agent_trajectories(axes[2], pred_df, colors, linestyle="-", linewidth=2.0, alpha=0.95, label_prefix=None)

    bounds = walkable.bounds
    pad_x = max((bounds[2] - bounds[0]) * 0.05, 1.0)
    pad_y = max((bounds[3] - bounds[1]) * 0.05, 1.0)
    for ax in axes:
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    legend_handles = [
        Patch(facecolor="#dfeedd", edgecolor="#4c6a55", label="Walkable Area"),
        plt.Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=2.0, label="GT Trajectory"),
        plt.Line2D([0], [0], color="#d62728", linestyle="-", linewidth=2.0, label="AI Trajectory"),
        plt.Line2D([0], [0], color="#f2c94c", marker="*", linestyle="None", markersize=12, markeredgecolor="black", label="Goal"),
    ]
    axes[0].legend(handles=legend_handles + gt_handles[:MAX_LEGEND_AGENTS], loc="best", fontsize=8)

    case_id = str(summary["case_id"])
    retrieved_case_id = summary.get("retrieved_cases", [{}])[0].get("case_id", "-")
    fig.suptitle(
        f"{sample_dir.name} | case {case_id} | retrieved={retrieved_case_id} | agents={len(all_agent_ids)} | ADE={ade_m:.3f} m | FDE={fde_m:.3f} m",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample": sample_dir.name,
        "case_id": case_id,
        "retrieved_case_id": retrieved_case_id,
        "num_agents": len(all_agent_ids),
        "ade_m": ade_m,
        "fde_m": fde_m,
        "out_of_bounds_rate": float(summary.get("out_of_bounds_rate", 0.0)),
        "plot_path": str(out_path),
    }


def main(result_dir: str | None = None):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    resolved_result_dir = pathlib.Path(result_dir).resolve() if result_dir else project_root / "AI_Result" / "Method_GPT_Knowledge"
    outputs_dir = resolved_result_dir / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"No outputs directory found in {resolved_result_dir}")

    sample_dirs = iter_sample_dirs(outputs_dir)
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_case_* directories found in {outputs_dir}")

    out_dir = resolved_result_dir / "visuals" / "generated_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[Visual] Result dir : {resolved_result_dir}")
    print(f"[Visual] Samples    : {len(sample_dirs)}")

    for sample_dir in sample_dirs:
        out_path = out_dir / f"{sample_dir.name}.png"
        row = make_sample_plot(sample_dir, out_path)
        rows.append(row)
        print(f"[Visual] {sample_dir.name} -> ADE={row['ade_m']:.3f} m  FDE={row['fde_m']:.3f} m  saved={out_path.name}")

    metrics_csv = out_dir / "sample_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"[Visual] Metrics saved -> {metrics_csv}")
    print(f"[Visual] Plots saved   -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default=None, help="Path to AI_Result/Method_GPT_Knowledge. Defaults to the project result directory.")
    args = parser.parse_args()
    main(args.result_dir)

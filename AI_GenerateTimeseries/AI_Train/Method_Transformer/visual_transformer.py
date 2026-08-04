"""
visual_transformer.py
---------------------
Create per-epoch validation visualisations for Method_Transformer runs.

Expected input structure:
    AI_Result/Method_Transformer/outputs/run_N/samples/epoch_XXX/
        AI_pred_{case_id}.parquet
        GT_real_{case_id}.parquet
        Geo_room.json
        Geo_corridor.json
        Spawn_location_{case_id}.csv
        Spawn_exit_{case_id}.csv

Usage:
    python visual_transformer.py --run_dir ../../AI_Result/Method_Transformer/outputs/run_9

Outputs:
    {run_dir}/visuals/val_epochs/epoch_XXX.png
    {run_dir}/visuals/val_epochs/epoch_metrics.csv
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Polygon

from prepare_geometry_transformer import build_walkable_area


def find_latest_run(base_dir: pathlib.Path) -> pathlib.Path:
    runs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")
    return runs[-1]


def load_xy_parquet(path: pathlib.Path) -> np.ndarray:
    df = pd.read_parquet(path)
    required = {"pos_x", "pos_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return df[["pos_x", "pos_y"]].to_numpy(dtype=np.float32)


def infer_obs_prefix_len(gt_world: np.ndarray, pred_world: np.ndarray, atol: float = 1e-6) -> int:
    max_common = min(len(gt_world), len(pred_world))
    prefix_len = 0
    for idx in range(max_common):
        if np.allclose(gt_world[idx], pred_world[idx], atol=atol, rtol=0.0):
            prefix_len += 1
        else:
            break
    return prefix_len


def compute_metrics(gt_world: np.ndarray, pred_world: np.ndarray, obs_prefix_len: int) -> tuple[float, float]:
    gt_future = gt_world[obs_prefix_len:]
    pred_future = pred_world[obs_prefix_len:]
    if len(gt_future) == 0 or len(pred_future) == 0:
        return 0.0, 0.0

    min_len = min(len(gt_future), len(pred_future))
    dists = np.linalg.norm(pred_future[:min_len] - gt_future[:min_len], axis=1)
    return float(dists[-1]), float(dists.mean())


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


def read_spawn_points(epoch_dir: pathlib.Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    spawn_location = next(epoch_dir.glob("Spawn_location_*.csv"), None)
    spawn_exit = next(epoch_dir.glob("Spawn_exit_*.csv"), None)

    start_pt = None
    goal_pt = None

    if spawn_location is not None:
        df = pd.read_csv(spawn_location)
        if {"pos_x", "pos_y"}.issubset(df.columns) and not df.empty:
            start_pt = df[["pos_x", "pos_y"]].iloc[0].to_numpy(dtype=np.float32)

    if spawn_exit is not None:
        df = pd.read_csv(spawn_exit)
        if {"type", "area"}.issubset(df.columns):
            exit_rows = df[df["type"] == "exit_area"]
            if not exit_rows.empty:
                from shapely import wkt as shapely_wkt

                poly = shapely_wkt.loads(exit_rows.iloc[0]["area"])
                goal_pt = np.array([poly.centroid.x, poly.centroid.y], dtype=np.float32)

    return start_pt, goal_pt


def iter_epoch_dirs(samples_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    return sorted([p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")], key=lambda p: p.name)


def make_epoch_plot(epoch_dir: pathlib.Path, out_path: pathlib.Path) -> dict:
    ai_path = next(epoch_dir.glob("AI_pred_*.parquet"), None)
    gt_path = next(epoch_dir.glob("GT_real_*.parquet"), None)
    room_json = epoch_dir / "Geo_room.json"
    corridor_json = epoch_dir / "Geo_corridor.json"

    if ai_path is None or gt_path is None:
        raise FileNotFoundError(f"Missing AI_pred_*.parquet or GT_real_*.parquet in {epoch_dir}")
    if not room_json.exists() or not corridor_json.exists():
        raise FileNotFoundError(f"Missing Geo_room.json or Geo_corridor.json in {epoch_dir}")

    pred_world = load_xy_parquet(ai_path)
    gt_world = load_xy_parquet(gt_path)
    walkable = build_walkable_area(str(room_json), str(corridor_json))
    obs_prefix_len = infer_obs_prefix_len(gt_world, pred_world)
    fde_m, ade_m = compute_metrics(gt_world, pred_world, obs_prefix_len)
    start_pt, goal_pt = read_spawn_points(epoch_dir)

    gt_obs = gt_world[:obs_prefix_len]
    gt_future = gt_world[obs_prefix_len:]
    pred_future = pred_world[obs_prefix_len:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = [
        "Walkable + GT vs AI",
        "Ground Truth",
        "AI Prediction",
    ]
    for ax, title in zip(axes, titles):
        plot_polygon(ax, walkable, facecolor="#dfeedd", edgecolor="#4c6a55", alpha=0.85, linewidth=1.1)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    if len(gt_obs) > 0:
        for ax in axes:
            ax.plot(gt_obs[:, 0], gt_obs[:, 1], color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label="Observed")

    if len(gt_future) > 0:
        axes[0].plot(gt_future[:, 0], gt_future[:, 1], color="#2ca02c", linewidth=2.2, linestyle="--", label="GT Future")
        axes[1].plot(gt_future[:, 0], gt_future[:, 1], color="#2ca02c", linewidth=2.4, linestyle="--", label="GT Future")

    if len(pred_future) > 0:
        axes[0].plot(pred_future[:, 0], pred_future[:, 1], color="#d62728", linewidth=2.2, label="AI Future")
        axes[2].plot(pred_future[:, 0], pred_future[:, 1], color="#d62728", linewidth=2.4, label="AI Future")

    if start_pt is None and len(gt_obs) > 0:
        start_pt = gt_obs[0]
    if goal_pt is None and len(gt_future) > 0:
        goal_pt = gt_future[-1]

    if start_pt is not None:
        for ax in axes:
            ax.scatter(start_pt[0], start_pt[1], c="#1f77b4", s=70, marker="o", edgecolors="black", linewidths=0.5, zorder=5)

    if goal_pt is not None:
        for ax in axes:
            ax.scatter(goal_pt[0], goal_pt[1], c="#f2c94c", s=180, marker="*", edgecolors="black", linewidths=0.6, zorder=6)

    bounds = walkable.bounds
    pad_x = max((bounds[2] - bounds[0]) * 0.05, 1.0)
    pad_y = max((bounds[3] - bounds[1]) * 0.05, 1.0)
    for ax in axes:
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    legend_handles = [
        Patch(facecolor="#dfeedd", edgecolor="#4c6a55", label="Walkable Area"),
        plt.Line2D([0], [0], color="#1f77b4", marker="o", linewidth=2, markersize=4, label="Observed"),
        plt.Line2D([0], [0], color="#2ca02c", linestyle="--", linewidth=2.2, label="GT Future"),
        plt.Line2D([0], [0], color="#d62728", linewidth=2.2, label="AI Future"),
        plt.Line2D([0], [0], color="#f2c94c", marker="*", linestyle="None", markersize=12, markeredgecolor="black", label="Goal"),
    ]
    axes[0].legend(handles=legend_handles, loc="best", fontsize=9)

    case_id = ai_path.stem.replace("AI_pred_", "")
    fig.suptitle(
        f"{epoch_dir.name} | case {case_id} | observed={obs_prefix_len} | future={len(gt_future)} | "
        f"FDE={fde_m:.3f} m | ADE={ade_m:.3f} m",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "epoch": epoch_dir.name,
        "case_id": case_id,
        "obs_prefix_len": obs_prefix_len,
        "future_len": len(gt_future),
        "fde_m": fde_m,
        "ade_m": ade_m,
        "plot_path": str(out_path),
    }


def main(run_dir: str | None = None):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    base_outputs = project_root / "AI_Result" / "Method_Transformer" / "outputs"
    resolved_run_dir = pathlib.Path(run_dir).resolve() if run_dir else find_latest_run(base_outputs)

    samples_dir = resolved_run_dir / "samples"
    if not samples_dir.exists():
        raise FileNotFoundError(f"No samples directory found in {resolved_run_dir}")

    out_dir = resolved_run_dir / "visuals" / "val_epochs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    epoch_dirs = list(iter_epoch_dirs(samples_dir))
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* directories found in {samples_dir}")

    print(f"[Visual] Run dir   : {resolved_run_dir}")
    print(f"[Visual] Epoch dirs: {len(epoch_dirs)}")

    for epoch_dir in epoch_dirs:
        out_path = out_dir / f"{epoch_dir.name}.png"
        row = make_epoch_plot(epoch_dir, out_path)
        rows.append(row)
        print(
            f"[Visual] {epoch_dir.name} -> "
            f"FDE={row['fde_m']:.3f} m  ADE={row['ade_m']:.3f} m  saved={out_path.name}"
        )

    metrics_csv = out_dir / "epoch_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"[Visual] Metrics saved -> {metrics_csv}")
    print(f"[Visual] Plots saved   -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Path to a specific run directory. Defaults to the latest run_* under AI_Result/Method_Transformer/outputs.",
    )
    args = parser.parse_args()
    main(args.run_dir)

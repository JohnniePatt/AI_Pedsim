"""
visual_gnn_cvae.py
------------------
Create per-epoch validation visualisations for Method_GNN_CVAE runs.
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

from prepare_geometry_gnn_cvae import build_walkable_area


MAX_LEGEND_AGENTS = 8


def find_latest_run(base_dir: pathlib.Path) -> pathlib.Path:
    runs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")], key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {base_dir}")
    return runs[-1]


def iter_epoch_dirs(samples_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    return sorted([p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")], key=lambda p: p.name)


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


def compute_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[float, float]:
    merged = gt_df.merge(pred_df, on=["frame", "id"], suffixes=("_gt", "_pred"))
    if merged.empty:
        return 0.0, 0.0

    merged["dist"] = np.sqrt((merged["pos_x_pred"] - merged["pos_x_gt"]) ** 2 + (merged["pos_y_pred"] - merged["pos_y_gt"]) ** 2)
    ade = float(merged["dist"].mean())
    final_rows = merged.sort_values("frame").groupby("id").tail(1)
    fde = float(final_rows["dist"].mean()) if not final_rows.empty else 0.0
    return ade, fde


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


def plot_goals(ax, epoch_dir: pathlib.Path):
    spawn_exit = next(epoch_dir.glob("Spawn_exit_*.csv"), None)
    if spawn_exit is None:
        return None
    df = pd.read_csv(spawn_exit)
    if not {"type", "area"}.issubset(df.columns):
        return None
    exit_rows = df[df["type"] == "exit_area"]
    if exit_rows.empty:
        return None

    from shapely import wkt as shapely_wkt

    poly = shapely_wkt.loads(exit_rows.iloc[0]["area"])
    ax.scatter(poly.centroid.x, poly.centroid.y, c="#f2c94c", s=180, marker="*", edgecolors="black", linewidths=0.6, zorder=6)
    return poly.centroid


def make_epoch_plot(epoch_dir: pathlib.Path, out_path: pathlib.Path) -> dict:
    ai_path = next(epoch_dir.glob("AI_pred_*.parquet"), None)
    gt_path = next(epoch_dir.glob("GT_real_*.parquet"), None)
    room_json = epoch_dir / "Geo_room.json"
    corridor_json = epoch_dir / "Geo_corridor.json"

    if ai_path is None or gt_path is None:
        raise FileNotFoundError(f"Missing AI_pred_*.parquet or GT_real_*.parquet in {epoch_dir}")
    if not room_json.exists() or not corridor_json.exists():
        raise FileNotFoundError(f"Missing Geo_room.json or Geo_corridor.json in {epoch_dir}")

    pred_df = load_traj_table(ai_path)
    gt_df = load_traj_table(gt_path)
    walkable = build_walkable_area(str(room_json), str(corridor_json))
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
        plot_goals(ax, epoch_dir)

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

    case_id = ai_path.stem.replace("AI_pred_", "")
    fig.suptitle(
        f"{epoch_dir.name} | case {case_id} | agents={len(all_agent_ids)} | ADE={ade_m:.3f} m | FDE={fde_m:.3f} m",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "epoch": epoch_dir.name,
        "case_id": case_id,
        "num_agents": len(all_agent_ids),
        "ade_m": ade_m,
        "fde_m": fde_m,
        "plot_path": str(out_path),
    }


def main(run_dir: str | None = None):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    base_outputs = project_root / "AI_Result" / "Method_GNN_CVAE" / "outputs"
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
        print(f"[Visual] {epoch_dir.name} -> ADE={row['ade_m']:.3f} m  FDE={row['fde_m']:.3f} m  saved={out_path.name}")

    metrics_csv = out_dir / "epoch_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)
    print(f"[Visual] Metrics saved -> {metrics_csv}")
    print(f"[Visual] Plots saved   -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None, help="Path to a specific run directory. Defaults to the latest run_* under AI_Result/Method_GNN_CVAE/outputs.")
    args = parser.parse_args()
    main(args.run_dir)

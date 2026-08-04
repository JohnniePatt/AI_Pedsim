"""
visual_gnn_cvae2.py
-------------------
Visualise GNN-CVAE2 run results.  Three modes:

  --mode val_epochs   per-epoch validation trajectory plots (default)
  --mode test         test_eval trajectory plots (per case)
  --mode curve        training-history loss/metric curves
  --mode all          all three

Usage examples:
  python3 visual_gnn_cvae2.py --run_path ../../AI_Result/Method_GNN_CVAE2/outputs/run_2
  python3 visual_gnn_cvae2.py --run_path ../../AI_Result/Method_GNN_CVAE2/outputs/run_2 --mode curve
  python3 visual_gnn_cvae2.py --run_path ../../AI_Result/Method_GNN_CVAE2/outputs/run_2 --mode test --max_cases 20
  python3 visual_gnn_cvae2.py --run_path ../../AI_Result/Method_GNN_CVAE2/outputs/run_2 --mode all
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Polygon

from prepare_geometry_gnn_cvae2 import build_walkable_area

MAX_LEGEND_AGENTS = 8


# ── geometry helpers ─────────────────────────────────────────────────────────

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
    ax.fill(ext[:, 0], ext[:, 1], facecolor=facecolor, edgecolor=edgecolor,
            alpha=alpha, linewidth=linewidth)
    for hole in geom.interiors:
        coords = np.asarray(hole.coords)
        ax.fill(coords[:, 0], coords[:, 1], facecolor="white", edgecolor=edgecolor,
                alpha=1.0, linewidth=linewidth * 0.75)


def plot_goals(ax, case_dir: pathlib.Path):
    spawn_exit = next(case_dir.glob("Spawn_exit_*.csv"), None)
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
    ax.scatter(poly.centroid.x, poly.centroid.y, c="#f2c94c", s=180, marker="*",
               edgecolors="black", linewidths=0.6, zorder=6)
    return poly.centroid


def build_agent_colors(agent_ids: list):
    cmap = plt.get_cmap("tab20")
    return {int(aid): cmap(idx % 20) for idx, aid in enumerate(agent_ids)}


def plot_trajectories(ax, df: pd.DataFrame, colors: dict, linestyle: str,
                      linewidth: float, alpha: float, label_prefix: str | None = None):
    handles, shown = [], 0
    for agent_id, grp in df.groupby("id"):
        grp = grp.sort_values("frame")
        color = colors[int(agent_id)]
        line, = ax.plot(grp["pos_x"], grp["pos_y"], linestyle=linestyle,
                        linewidth=linewidth, alpha=alpha, color=color)
        ax.scatter(grp.iloc[0]["pos_x"],  grp.iloc[0]["pos_y"],  color=color, s=14,
                   alpha=min(alpha + 0.1, 1.0))
        ax.scatter(grp.iloc[-1]["pos_x"], grp.iloc[-1]["pos_y"], color=color, s=28,
                   marker="x", alpha=min(alpha + 0.1, 1.0))
        if label_prefix and shown < MAX_LEGEND_AGENTS:
            line.set_label(f"{label_prefix} {int(agent_id)}")
            handles.append(line)
            shown += 1
    return handles


def compute_ade_fde(gt_df, pred_df, exit_centroid=None):
    merged = gt_df.merge(pred_df, on=["frame", "id"], suffixes=("_gt", "_pred"))
    if merged.empty:
        return 0.0, 0.0
    merged["dist"] = np.sqrt(
        (merged["pos_x_pred"] - merged["pos_x_gt"]) ** 2 +
        (merged["pos_y_pred"] - merged["pos_y_gt"]) ** 2
    )
    ade = float(merged["dist"].mean())
    if exit_centroid is not None:
        ai_final = pred_df.sort_values("frame").groupby("id").tail(1)
        fde_vals = np.sqrt(
            (ai_final["pos_x"].values - exit_centroid[0]) ** 2 +
            (ai_final["pos_y"].values - exit_centroid[1]) ** 2
        )
        fde = float(fde_vals.mean()) if len(fde_vals) > 0 else 0.0
    else:
        fde = float(merged.sort_values("frame").groupby("id").tail(1)["dist"].mean())
    return ade, fde


# ── shared plot builder ───────────────────────────────────────────────────────

def make_traj_plot(case_dir: pathlib.Path, out_path: pathlib.Path, title_prefix: str = "") -> dict:
    ai_path  = next(case_dir.glob("AI_pred_*.parquet"), None)
    gt_path  = next(case_dir.glob("GT_real_*.parquet"),  None)
    room_json = case_dir / "Geo_room.json"
    cor_json  = case_dir / "Geo_corridor.json"

    if not (ai_path and gt_path and room_json.exists() and cor_json.exists()):
        raise FileNotFoundError(f"Missing files in {case_dir}")

    pred_df = pd.read_parquet(ai_path).sort_values(["id", "frame"]).reset_index(drop=True)
    gt_df   = pd.read_parquet(gt_path).sort_values(["id", "frame"]).reset_index(drop=True)
    walkable = build_walkable_area(str(room_json), str(cor_json))

    exit_centroid = None
    spawn_exit = next(case_dir.glob("Spawn_exit_*.csv"), None)
    if spawn_exit is not None:
        edf = pd.read_csv(spawn_exit)
        if {"type", "area"}.issubset(edf.columns):
            erows = edf[edf["type"] == "exit_area"]
            if not erows.empty:
                from shapely import wkt as shapely_wkt
                poly = shapely_wkt.loads(erows.iloc[0]["area"])
                exit_centroid = np.array([poly.centroid.x, poly.centroid.y])

    ade, fde = compute_ade_fde(gt_df, pred_df, exit_centroid)
    all_ids  = sorted(set(gt_df["id"].tolist()) | set(pred_df["id"].tolist()))
    colors   = build_agent_colors([int(x) for x in all_ids])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    bounds = walkable.bounds
    pad_x  = max((bounds[2] - bounds[0]) * 0.05, 1.0)
    pad_y  = max((bounds[3] - bounds[1]) * 0.05, 1.0)

    for ax, title in zip(axes, ["GT vs AI (overlay)", "Ground Truth", "AI Prediction"]):
        plot_polygon(ax, walkable, "#dfeedd", "#4c6a55", 0.85, 1.1)
        plot_goals(ax, case_dir)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    plot_trajectories(axes[0], gt_df,   colors, "--", 1.8, 0.8)
    plot_trajectories(axes[0], pred_df, colors, "-",  1.4, 0.9)
    plot_trajectories(axes[1], gt_df,   colors, "--", 2.0, 0.9)
    plot_trajectories(axes[2], pred_df, colors, "-",  2.0, 0.95)

    legend_handles = [
        Patch(facecolor="#dfeedd", edgecolor="#4c6a55", label="Walkable Area"),
        plt.Line2D([0], [0], color="steelblue", linestyle="--", lw=2, label="GT"),
        plt.Line2D([0], [0], color="tomato",    linestyle="-",  lw=2, label="AI"),
        plt.Line2D([0], [0], color="#f2c94c", marker="*", linestyle="None",
                   markersize=12, markeredgecolor="black", label="Goal"),
    ]
    axes[0].legend(handles=legend_handles, loc="best", fontsize=8)

    case_id = ai_path.stem.replace("AI_pred_", "")
    fig.suptitle(
        f"{title_prefix}{case_dir.name}  |  case {case_id}"
        f"  |  agents={len(all_ids)}  |  ADE={ade:.3f} m  |  FDE={fde:.3f} m",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"case_id": case_id, "num_agents": len(all_ids), "ade_m": ade, "fde_m": fde}


# ── mode: val_epochs ─────────────────────────────────────────────────────────

def run_val_epochs(run_dir: pathlib.Path, step: int = 1):
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        print(f"[Visual] No samples/ in {run_dir}")
        return

    epoch_dirs = sorted(
        [p for p in samples_dir.iterdir() if p.is_dir() and p.name.startswith("epoch_")],
        key=lambda p: p.name,
    )[::step]

    out_dir = run_dir / "visuals" / "val_epochs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[Visual] val_epochs: {len(epoch_dirs)} epochs → {out_dir}")
    for ep_dir in epoch_dirs:
        out_png = out_dir / f"{ep_dir.name}.png"
        try:
            row = make_traj_plot(ep_dir, out_png, title_prefix="Validation | ")
            row["epoch"] = ep_dir.name
            rows.append(row)
            print(f"  {ep_dir.name}  ADE={row['ade_m']:.3f} m  FDE={row['fde_m']:.3f} m")
        except Exception as e:
            print(f"  [skip] {ep_dir.name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "epoch_metrics.csv", index=False)
    print(f"[Visual] val_epochs done → {out_dir}")


# ── mode: test ────────────────────────────────────────────────────────────────

def run_test(run_dir: pathlib.Path, max_cases: int = 50):
    test_dir = run_dir / "test_eval" / "samples"
    if not test_dir.exists():
        print(f"[Visual] No test_eval/samples/ in {run_dir}  (run test_gnn_cvae2.py first)")
        return

    case_dirs = sorted([p for p in test_dir.iterdir() if p.is_dir()])[:max_cases]
    out_dir   = run_dir / "visuals" / "test_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"[Visual] test: {len(case_dirs)} cases → {out_dir}")
    for cd in case_dirs:
        out_png = out_dir / f"{cd.name}.png"
        try:
            row = make_traj_plot(cd, out_png, title_prefix="Test | ")
            row["case_dir"] = cd.name
            rows.append(row)
            print(f"  {cd.name}  ADE={row['ade_m']:.3f} m  FDE={row['fde_m']:.3f} m")
        except Exception as e:
            print(f"  [skip] {cd.name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "test_case_metrics.csv", index=False)
    print(f"[Visual] test done → {out_dir}")


# ── mode: curve ───────────────────────────────────────────────────────────────

def run_curve(run_dir: pathlib.Path):
    csv_path = run_dir / "logs" / "training_history.csv"
    if not csv_path.exists():
        print(f"[Visual] No training_history.csv in {run_dir / 'logs'}")
        return

    df = pd.read_csv(csv_path)
    out_dir = run_dir / "visuals" / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("GNN-CVAE2 Training Progress", fontsize=14)

    # Loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tomato")
    best_epoch = int(df.loc[df["val_loss"].idxmin(), "epoch"])
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    # ADE
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["ade_m"], color="mediumseagreen")
    ax.set_title("ADE (m)"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    # FDE
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["fde_m"], color="darkorange")
    ax.set_title("FDE (m)"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    # Collision + OOB
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["collision_rate"],    label="Collision Rate", color="crimson")
    ax.plot(df["epoch"], df["out_of_bounds_rate"],label="OOB Rate",       color="mediumpurple")
    ax.set_title("Collision & OOB Rate"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(True, alpha=0.3)

    out_png = out_dir / "training_curves.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_row = df.loc[df["val_loss"].idxmin()]
    print(f"[Visual] curve done → {out_png}")
    print(f"  Best epoch : {int(best_row['epoch'])}")
    print(f"  Val loss   : {best_row['val_loss']:.5f}")
    print(f"  ADE        : {best_row['ade_m']:.3f} m")
    print(f"  FDE        : {best_row['fde_m']:.3f} m")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path",  default=None,
                        help="Path to run_N directory (defaults to latest run)")
    parser.add_argument("--mode",      default="val_epochs",
                        choices=["val_epochs", "test", "curve", "all"],
                        help="What to visualise")
    parser.add_argument("--epoch_step",default=1, type=int,
                        help="Plot every N-th epoch (val_epochs mode)")
    parser.add_argument("--max_cases", default=50, type=int,
                        help="Max test cases to plot (test mode)")
    args = parser.parse_args()

    # Resolve run dir
    if args.run_path:
        run_dir = pathlib.Path(args.run_path).resolve()
    else:
        base = pathlib.Path(__file__).resolve().parents[3] \
               / "AI_GenerateImage" / "AI_Result" / "Method_GNN_CVAE2" / "outputs"
        runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")])
        if not runs:
            raise FileNotFoundError(f"No run_* directories in {base}")
        run_dir = runs[-1]

    print(f"[Visual] Run: {run_dir}")

    if args.mode in ("val_epochs", "all"):
        run_val_epochs(run_dir, step=args.epoch_step)
    if args.mode in ("curve", "all"):
        run_curve(run_dir)
    if args.mode in ("test", "all"):
        run_test(run_dir, max_cases=args.max_cases)


if __name__ == "__main__":
    main()

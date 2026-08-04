"""
test_transformer.py
────────────────────
Evaluate GoalConditionedGPT2 (with social context) on the test split.

Usage:
    python test_transformer.py --config config_test.json
    python test_transformer.py --config config_test.json \\
        --model_path ../../AI_Result/Method_Transformer/outputs/run_5/weights/best_model.pth

Output in {run_dir}/test_results/:
  test_evaluation_summary.csv
  test_evaluation.txt
  trajectories/eval_scene_NNNN.png   (first 20 samples)
"""

import argparse
import json
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from shapely import wkt as shapely_wkt
from shapely.geometry import Point

from dataset import PedestrianDataset, collate_fn
from model import GoalConditionedGPT2
from prepare_geometry_transformer import build_walkable_area


def batch_to_device(batch: dict, device: torch.device) -> dict:
    keys = ["obs_traj", "pred_traj", "start_pt", "end_pt",
            "geo_mask", "neighbor_trajs", "neighbor_mask"]
    return {k: batch[k].to(device) for k in keys if k in batch}


def plot_full_case_rollout(case_dir: pathlib.Path, agent_trajectories: list, output_path: pathlib.Path, title: str = "Model rollout sample"):
    """
    Plots ALL N agents on the floorplan in high-quality vector style:
    - Dark background (#0B0E14)
    - Off-white room/corridor polygons (#F8FAFC) with dark walls (#101820)
    - Door openings cut through wall strokes (no coloured door patches)
    - Light orange exit room polygon (#FED7AA, alpha=0.7) with orange border (#EA580C)
    - Green spawn dots (#10B981)
    - Multi-colored trajectory lines for every agent
    """
    import json
    from matplotlib.patches import Polygon as MplPolygon, Rectangle
    from shapely import wkt as shapely_wkt

    room_json = case_dir / "Geo_room.json"
    corridor_json = case_dir / "Geo_corridor.json"
    door_json = case_dir / "Geo_door.json"

    # Match the canvas sizing contract used by generate_gt_previews.py and
    # GridSocialPolicy/rollout.py: adapt the figure to the floorplan aspect.
    walkable_area = build_walkable_area(str(room_json), str(corridor_json))
    bound_min_x, bound_min_y, bound_max_x, bound_max_y = walkable_area.bounds
    bound_width = max(bound_max_x - bound_min_x, 1e-6)
    bound_height = max(bound_max_y - bound_min_y, 1e-6)
    aspect = max(bound_width / bound_height, 0.25)
    fig_width = min(18.0, max(7.0, 7.0 * aspect))
    fig_height = min(12.0, max(4.5, fig_width / aspect))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    layout_x, layout_y = [], []

    def draw_polys(json_path, facecolor, edgecolor, linewidth, alpha=1.0, zorder=1):
        if not json_path.exists(): return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            for poly_coords in data:
                if isinstance(poly_coords, list) and len(poly_coords) > 2:
                    coords = np.asarray(poly_coords, dtype=np.float64)
                    layout_x.extend(coords[:, 0].tolist())
                    layout_y.extend(coords[:, 1].tolist())
                    patch = MplPolygon(poly_coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
                    ax.add_patch(patch)
        except Exception as e:
            print(f"[Plot Warning] Failed to draw polys from {json_path.name}: {e}")

    # 1. Render Walkable Floorplan Layout (Rooms, Corridors) using Vector Polygons
    draw_polys(room_json, facecolor="#f3f6f8", edgecolor="#101820", linewidth=2.0, zorder=1)
    # A single walkable fill colour makes door openings read as continuous voids.
    draw_polys(corridor_json, facecolor="#f3f6f8", edgecolor="#101820", linewidth=2.0, zorder=1)

    # 2. Render Doors
    if door_json.exists():
        try:
            with open(door_json, 'r') as f:
                doors = json.load(f)
            for d in doors:
                pos = d["pos"]
                width = d.get("door_width", 1.5)  # Fallback if door_width is missing
                is_horiz = d.get("horizontal", False)
                # Cover the shared wall stroke with walkable fill.  The extra
                # thickness fully erases anti-aliased wall pixels and produces
                # an actual opening rather than a symbolic yellow door.
                dw, dh = (width, 0.18) if is_horiz else (0.18, width)
                rect = Rectangle(
                    (pos[0] - dw / 2, pos[1] - dh / 2),
                    dw,
                    dh,
                    facecolor="#f3f6f8",
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=2,
                )
                ax.add_patch(rect)
                layout_x.extend([pos[0] - dw / 2, pos[0] + dw / 2])
                layout_y.extend([pos[1] - dh / 2, pos[1] + dh / 2])
        except Exception as e:
            print(f"[Plot Warning] Failed to draw doors: {e}")

    # 3. Load Exit Area Polygon
    case_id = case_dir.name.replace("case_", "")
    exit_csv = case_dir / f"Spawn_exit_{case_id}.csv"
    if not exit_csv.exists():
        exit_files = list(case_dir.glob("Spawn_exit_*.csv"))
        if exit_files:
            exit_csv = exit_files[0]

    if exit_csv.exists():
        try:
            exit_df = pd.read_csv(exit_csv)
            exit_rows = exit_df[exit_df["type"] == "exit_area"]
            if not exit_rows.empty:
                raw_exit = exit_rows.iloc[0]["area"]
                if isinstance(raw_exit, str):
                    exit_poly = shapely_wkt.loads(raw_exit)
                    if exit_poly.geom_type == "Polygon":
                        coords = list(exit_poly.exterior.coords)
                        layout_x.extend([point[0] for point in coords])
                        layout_y.extend([point[1] for point in coords])
                        exit_patch = MplPolygon(
                            coords,
                            closed=True,
                            facecolor="#f59e0b",
                            edgecolor="#f97316",
                            linewidth=1.3,
                            alpha=0.35,
                            zorder=3,
                            label="exit room",
                        )
                        ax.add_patch(exit_patch)
        except Exception as e:
            pass

    # 4. Plot Spawn Points & Trajectories for ALL agents
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    spawn_handles_added = False

    for idx, trajectory_parts in enumerate(agent_trajectories):
        obs_pts, pred_pts = trajectory_parts[:2]
        color = colors[idx % len(colors)]
        full_traj = np.concatenate([obs_pts, pred_pts], axis=0) if len(obs_pts) > 0 else pred_pts
        
        if len(full_traj) > 0:
            sp_label = "spawn" if not spawn_handles_added else None
            ax.scatter(
                full_traj[0, 0],
                full_traj[0, 1],
                s=18,
                c="#22c55e",
                edgecolors="#052e16",
                linewidths=0.4,
                zorder=5,
                label=sp_label,
            )
            spawn_handles_added = True

        # Draw trajectory path
        ax.plot(full_traj[:, 0], full_traj[:, 1], color=color, linewidth=1.2, alpha=0.78, zorder=4)

    # 5. Fix aspect ratio and scale
    ax.set_aspect("equal", adjustable="box")

    # The canvas is defined by the complete floorplan, never by how far a
    # prediction happened to travel.  Trajectory-derived limits caused the
    # reported "half floorplan" images whenever a rollout stopped early.
    if layout_x and layout_y:
        x_min, x_max = min(layout_x), max(layout_x)
        y_min, y_max = min(layout_y), max(layout_y)
        dx = max((x_max - x_min) * 0.04, 0.5)
        dy = max((y_max - y_min) * 0.04, 0.5)
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)

    # Use an explicit dark artist so set_axis_off() behaves exactly like the
    # raster previews while preserving a white outer figure frame.
    bg_x0, bg_x1 = ax.get_xlim()
    bg_y0, bg_y1 = ax.get_ylim()
    background = Rectangle(
        (bg_x0, bg_y0),
        bg_x1 - bg_x0,
        bg_y1 - bg_y0,
        facecolor="#101820",
        edgecolor="none",
        zorder=-10,
    )
    ax.add_patch(background)
    ax.set_xlim(bg_x0, bg_x1)
    ax.set_ylim(bg_y0, bg_y1)
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    rendered_limits = (ax.get_xlim(), ax.get_ylim())
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rendered_limits


def load_checkpoint_bundle(model_path: pathlib.Path, device: torch.device) -> tuple[dict, dict, dict]:
    """Return (state_dict, model_config, data_config), including legacy runs."""
    saved = torch.load(model_path, map_location=device)
    if isinstance(saved, dict) and "model_state_dict" in saved:
        state_dict = saved["model_state_dict"]
        model_cfg = {k: v for k, v in saved.get("model_config", {}).items() if v is not None}
        data_cfg = dict(saved.get("data_config", {}))
    else:
        state_dict = saved
        model_cfg = {"geo_encoder_type": "pooled", "prediction_mode": "absolute"}
        data_cfg = {}
        legacy_cfg_path = model_path.parents[1] / "config_train.json"
        if not legacy_cfg_path.exists():
            legacy_cfg_path = model_path.parents[1] / "config_test.json"
        if legacy_cfg_path.exists():
            with open(legacy_cfg_path) as file:
                legacy_cfg = json.load(file)
            trained_path = legacy_cfg.get("dataset_path", "")
            data_cfg["dataset_name"] = pathlib.Path(trained_path).name
            data_cfg["dataset_path"] = trained_path
            for key in ["d_model", "nhead", "num_layers", "max_seq_len", "dropout", "max_neighbors", "obs_len"]:
                if key in legacy_cfg:
                    model_cfg[key] = legacy_cfg[key]

    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    return state_dict, model_cfg, data_cfg


def assert_checkpoint_compatible(checkpoint_data: dict, dataset_path: str, allow_dataset_mismatch: bool) -> None:
    trained_name = str(checkpoint_data.get("dataset_name", "")).strip()
    requested_name = pathlib.Path(dataset_path).name
    if trained_name and trained_name.casefold() != requested_name.casefold():
        message = (
            "checkpoint/dataset mismatch: checkpoint was trained on "
            f"'{trained_name}', but evaluation requested '{requested_name}'. "
            "Metrics and rollouts from this combination are not valid research results."
        )
        if not allow_dataset_mismatch:
            raise ValueError(message + " Pass --allow-dataset-mismatch only for debugging.")
        print(f"[Test] WARNING: {message}")


def load_case_metric_geometry(case_dir: pathlib.Path, case_id: str) -> tuple[object, object]:
    walkable = build_walkable_area(
        str(case_dir / "Geo_room.json"),
        str(case_dir / "Geo_corridor.json"),
    ).buffer(0.05)
    exit_csv = case_dir / f"Spawn_exit_{case_id}.csv"
    if not exit_csv.exists():
        exit_csv = next(case_dir.glob("Spawn_exit_*.csv"))
    exit_df = pd.read_csv(exit_csv)
    exit_poly = shapely_wkt.loads(exit_df[exit_df["type"] == "exit_area"].iloc[0]["area"])
    return walkable, exit_poly


def main(
    config_path: str,
    model_path: str | None = None,
    run_path: str | None = None,
    case_id: str | None = None,
    allow_dataset_mismatch: bool = False,
    split: str = "test",
):
    cfg_file = pathlib.Path(config_path).resolve()
    with open(cfg_file) as f:
        cfg = json.load(f)

    dataset_path = cfg["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = str((cfg_file.parent / dataset_path).resolve())

    if model_path is None:
        configured_path = cfg.get("model_checkpoint", "")
        model_path = str((cfg_file.parent / configured_path).resolve()) if configured_path else None
    model_path = pathlib.Path(model_path).resolve() if model_path else None

    # Output directory
    if run_path:
        run_dir = pathlib.Path(run_path).resolve()
    elif model_path and model_path.exists():
        run_dir = model_path.parents[1]
    else:
        run_dir = cfg_file.parent / "test_results_manual"

    out_dir  = run_dir / "test_results"
    traj_dir = out_dir / "predictions"
    traj_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device: {device}")

    if not model_path or not model_path.exists():
        raise FileNotFoundError("a valid --model_path/model_checkpoint is required for evaluation")
    state_dict, checkpoint_model_cfg, checkpoint_data_cfg = load_checkpoint_bundle(model_path, device)
    assert_checkpoint_compatible(checkpoint_data_cfg, dataset_path, allow_dataset_mismatch)
    # Dataset
    test_ds = PedestrianDataset(
        data_dir       = dataset_path,
        split          = split,
        obs_len        = cfg.get("obs_len", 5),
        frame_stride   = cfg.get("frame_stride", 8),
        max_seq_len    = cfg.get("max_seq_len", 512),
        grid_size      = cfg.get("grid_size", 64),
        geo_padding    = cfg.get("geo_padding", 1.0),
        max_neighbors  = cfg.get("max_neighbors", 10),
        subset_percent = cfg.get("subset_percent", 100.0),
        case_id         = case_id,
    )
    if len(test_ds) == 0:
        raise RuntimeError(
            f"no {split} samples found{f' for case {case_id}' if case_id else ''}"
        )

    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.get("batch_size", 8),
        shuffle     = False,
        num_workers = cfg.get("num_workers", 0),
        collate_fn  = collate_fn,
    )

    # The checkpoint is authoritative for architecture fields.  This preserves
    # legacy pooled/absolute runs while new runs use spatial/delta prediction.
    effective_model_cfg = {
        "d_model": cfg.get("d_model", 128),
        "nhead": cfg.get("nhead", 4),
        "num_layers": cfg.get("num_layers", 4),
        "max_seq_len": cfg.get("max_seq_len", 512),
        "dropout": 0.0,
        "max_neighbors": cfg.get("max_neighbors", 10),
        "obs_len": cfg.get("obs_len", 5),
        "geo_encoder_type": cfg.get("geo_encoder_type", "spatial"),
        "prediction_mode": cfg.get("prediction_mode", "delta"),
        "max_step_size": cfg.get("max_step_size", 0.1),
        "walkability_loss_weight": cfg.get("walkability_loss_weight", 0.05),
    }
    effective_model_cfg.update(checkpoint_model_cfg)
    effective_model_cfg["dropout"] = 0.0
    model = GoalConditionedGPT2(**effective_model_cfg).to(device)

    if model_path and model_path.exists():
        model.load_state_dict(state_dict)
        print(f"[Test] Loaded: {model_path}")
        print(
            "[Test] Contract: "
            f"geo={effective_model_cfg['geo_encoder_type']}, "
            f"prediction={effective_model_cfg['prediction_mode']}"
        )
    else:
        print("[Test] WARNING: no weights loaded – random init.")

    model.eval()

    total_fde_m, total_ade_m = 0.0, 0.0
    total_fde_norm, total_ade_norm, total_n = 0.0, 0.0, 0
    wall_violation_steps, total_pred_steps, goal_successes = 0, 0, 0
    geometry_cache = {}
    use_amp = device.type == "cuda"
    case_predictions = {}  # case_id -> {"case_dir": Path, "agents": list}

    start_test_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            b       = batch_to_device(batch, device)
            lengths = batch["lengths"]

            batch_max = int(lengths.max().item())
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(
                    b["obs_traj"], b["start_pt"], b["end_pt"], b["geo_mask"],
                    neighbor_trajs = b.get("neighbor_trajs"),
                    neighbor_mask  = b.get("neighbor_mask"),
                    pred_len       = batch_max,
                )
            preds = out["logits"]   # [B, batch_max, 2]

            for i in range(len(lengths)):
                T         = int(lengths[i].item())
                pred_norm = preds[i, :T].cpu().numpy()
                gt_norm   = batch["pred_traj"][i, :T].cpu().numpy()

                c_id = batch["case_ids"][i] if "case_ids" in batch else "default"
                c_dir = pathlib.Path(batch["case_dirs"][i]) if "case_dirs" in batch else pathlib.Path(".")
                meta = batch["metas"][i] if "metas" in batch else None

                dist_norm = np.linalg.norm(pred_norm - gt_norm, axis=1)
                scale_m = float(meta["scale"]) if meta else 1.0
                dist_m = dist_norm * scale_m
                total_fde_norm += float(dist_norm[-1])
                total_ade_norm += float(dist_norm.mean())
                total_fde_m += float(dist_m[-1])
                total_ade_m += float(dist_m.mean())
                total_n += 1

                # Denormalize coordinates to world coordinates for display
                ox = b["obs_traj"][i].cpu().numpy()
                if meta:
                    ox_w = ox * meta["scale"] + np.array([meta["min_x"], meta["min_y"]])
                    pred_w = pred_norm * meta["scale"] + np.array([meta["min_x"], meta["min_y"]])
                    gt_w = gt_norm * meta["scale"] + np.array([meta["min_x"], meta["min_y"]])
                else:
                    ox_w, pred_w, gt_w = ox, pred_norm, gt_norm

                cache_key = str(c_dir.resolve())
                if cache_key not in geometry_cache:
                    geometry_cache[cache_key] = load_case_metric_geometry(c_dir, c_id)
                walkable, exit_poly = geometry_cache[cache_key]
                pred_points = [Point(float(x), float(y)) for x, y in pred_w]
                wall_violation_steps += sum(not walkable.covers(point) for point in pred_points)
                total_pred_steps += len(pred_points)
                goal_successes += int(any(exit_poly.covers(point) for point in pred_points))

                if c_id not in case_predictions:
                    case_predictions[c_id] = {"case_dir": c_dir, "agents": []}
                case_predictions[c_id]["agents"].append((ox_w, pred_w, gt_w))

    print(f"\n[Test] Generating multi-agent floorplan previews for {len(case_predictions)} test cases …")
    for c_id, c_data in tqdm(case_predictions.items(), desc="Saving Plots"):
        out_img = traj_dir / f"{c_id}_rollout_preview.png"
        plot_full_case_rollout(
            c_data["case_dir"],
            c_data["agents"],
            out_img,
            title="Transformer",
        )

    total_test_time_sec = time.time() - start_test_time
    avg_latency_ms = (total_test_time_sec / max(total_n, 1)) * 1000.0
    avg_fde_m = total_fde_m / max(total_n, 1)
    avg_ade_m = total_ade_m / max(total_n, 1)
    avg_fde_norm = total_fde_norm / max(total_n, 1)
    avg_ade_norm = total_ade_norm / max(total_n, 1)
    wall_violation_rate = wall_violation_steps / max(total_pred_steps, 1)
    goal_success_rate = goal_successes / max(total_n, 1)

    print(f"\n[Test] Samples         : {total_n}")
    print(f"[Test] ADE             : {avg_ade_m:.4f} m")
    print(f"[Test] FDE             : {avg_fde_m:.4f} m")
    print(f"[Test] Wall violations : {wall_violation_rate:.2%} of predicted steps")
    print(f"[Test] Goal success    : {goal_success_rate:.2%} of agents")
    print(f"[Test] Total Time (s)  : {total_test_time_sec:.2f} s ({total_test_time_sec/60.0:.2f} min)")
    print(f"[Test] Avg Latency (ms): {avg_latency_ms:.2f} ms/sample")

    pd.DataFrame([
        {"metric": "ADE_m",               "value": avg_ade_m},
        {"metric": "FDE_m",               "value": avg_fde_m},
        {"metric": "ADE_norm",            "value": avg_ade_norm},
        {"metric": "FDE_norm",            "value": avg_fde_norm},
        {"metric": "wall_violation_rate", "value": wall_violation_rate},
        {"metric": "goal_success_rate",   "value": goal_success_rate},
        {"metric": "n_samples",           "value": total_n},
        {"metric": "total_time_sec",      "value": round(total_test_time_sec, 2)},
        {"metric": "avg_latency_ms",      "value": round(avg_latency_ms, 2)},
    ]).to_csv(out_dir / "test_evaluation_summary.csv", index=False)

    with open(out_dir / "test_evaluation.txt", "w") as f:
        f.write(
            f"Test Evaluation Report\n"
            f"======================\n"
            f"Samples           : {total_n}\n"
            f"ADE (metres)      : {avg_ade_m:.4f}\n"
            f"FDE (metres)      : {avg_fde_m:.4f}\n"
            f"ADE (normalised)  : {avg_ade_norm:.4f}\n"
            f"FDE (normalised)  : {avg_fde_norm:.4f}\n"
            f"Wall violations   : {wall_violation_rate:.4%}\n"
            f"Goal success      : {goal_success_rate:.4%}\n"
            f"Total Test Time   : {total_test_time_sec:.2f} sec ({total_test_time_sec/60.0:.2f} min)\n"
            f"Average Latency   : {avg_latency_ms:.2f} ms/sample\n"
        )

    print(f"[Test] Saved → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config_test.json")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--run_path",   default=None)
    parser.add_argument("--case-id", default=None, help="Evaluate one exact case id (without case_ prefix).")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to render. Keep 'test' for research metrics; train/val are gallery placeholders only.",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Debug only: allow a checkpoint trained on a different topology.",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.model_path,
        args.run_path,
        case_id=args.case_id,
        allow_dataset_mismatch=args.allow_dataset_mismatch,
        split=args.split,
    )

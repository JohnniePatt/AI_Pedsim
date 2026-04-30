"""prepare_densitymap_dataset.py
Prepare A/B image datasets for density-map prediction (pix2pixHD / UNet).

A image (input – spawn/exit map):   IDENTICAL to prepare_unet_image_dataset.py
B image (target – density heatmap): Voronoi-based density (same formula as
    generate_heatmaps / bottleneck_archhouseplan.py generate_heatmaps)

A image colour scheme:
    black  (0,0,0)     = walls / non-walkable
    red    (255,0,0)   = walkable floor
    green  (0,255,0)   = spawn position (small dot at spawn points / room centroid)
    blue   (0,0,255)   = exit room (filled area)

B image colour scheme (grayscale, R=G=B):
    black  (0)         = non-walkable OR zero density
    gray   (1-255)     = density in persons/m²  mapped [0, vmax=5] → [0, 255]
    NO wall outlines   = pure density only; floor plan structure is in A image
"""

import argparse
import concurrent.futures
import csv
import json
import multiprocessing as mp
import os
import pathlib
import random
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby

import numpy as np
import shapely.wkt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
from tqdm.auto import tqdm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# ──────────────────────────────────────────────────────────────────────────────
# Generic dataset utilities
# ──────────────────────────────────────────────────────────────────────────────

def count_split_images(dataset_root: pathlib.Path) -> dict:
    summary = {
        "A_total": 0,
        "B_total": 0,
        "total_png": 0,
        "dataset_layout": "ab_flat",
    }
    for side in ("A", "B"):
        n = len(list((dataset_root / side).glob("*.png")))
        summary[f"{side}_total"] = n
        summary["total_png"] += n
    return summary


def ensure_clean_output(output_dir: pathlib.Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite to replace): {output_dir}")
        print(f"[CLEAN] Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_rgb(arr: np.ndarray, path: pathlib.Path) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def _resolve_torch_device(device_preference: str, force_gpu: bool = False):
    pref = str(device_preference).strip().lower()
    if pref not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device_preference}'. Use auto|cpu|cuda.")

    if not HAS_TORCH:
        if pref == "cuda" or force_gpu:
            raise RuntimeError("PyTorch is not installed, cannot run CUDA mode.")
        return None, "cpu (torch-unavailable)"

    if pref == "cpu":
        return torch.device("cpu"), "cpu"

    cuda_available = torch.cuda.is_available()
    if pref == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda"), "cuda"

    # auto
    if cuda_available:
        return torch.device("cuda"), "cuda"
    if force_gpu:
        raise RuntimeError("GPU was forced but CUDA is not available.")
    return torch.device("cpu"), "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CanvasTransform:
    min_x: float
    max_y: float
    span_x: float
    span_y: float
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    width: int
    height: int


def _build_canvas_transform(
    bounds: tuple[float, float, float, float],
    size_wh: tuple[int, int],
    preserve_aspect: bool = True,
) -> CanvasTransform:
    min_x, min_y, max_x, max_y = bounds
    w, h = size_wh
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    sx = (w - 1) / span_x
    sy = (h - 1) / span_y

    if preserve_aspect:
        scale = min(sx, sy)
        sx = sy = scale
        offset_x = 0.5 * ((w - 1) - span_x * sx)
        offset_y = 0.5 * ((h - 1) - span_y * sy)
    else:
        offset_x = 0.0
        offset_y = 0.0

    return CanvasTransform(
        min_x=min_x,
        max_y=max_y,
        span_x=span_x,
        span_y=span_y,
        scale_x=sx,
        scale_y=sy,
        offset_x=offset_x,
        offset_y=offset_y,
        width=w,
        height=h,
    )


def _world_to_px(
    x: float,
    y: float,
    transform: CanvasTransform,
) -> tuple[int, int]:
    px = int(round(transform.offset_x + (x - transform.min_x) * transform.scale_x))
    py = int(round(transform.offset_y + (transform.max_y - y) * transform.scale_y))
    return (
        max(0, min(transform.width - 1, px)),
        max(0, min(transform.height - 1, py)),
    )


def _canvas_bounds(walkable: Polygon, padding: float = 1.0) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = walkable.bounds
    return min_x - padding, min_y - padding, max_x + padding, max_y + padding


# ──────────────────────────────────────────────────────────────────────────────
# Shapely → PIL rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_shapely(
    draw: ImageDraw.ImageDraw,
    geom,
    transform: CanvasTransform,
    fill,
    hole_fill=(0, 0, 0),
) -> None:
    if geom is None or geom.is_empty:
        return
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        ext_px = [_world_to_px(x, y, transform) for x, y in poly.exterior.coords]
        if len(ext_px) >= 3:
            draw.polygon(ext_px, fill=fill)
        for interior in poly.interiors:
            int_px = [_world_to_px(x, y, transform) for x, y in interior.coords]
            if len(int_px) >= 3:
                draw.polygon(int_px, fill=hole_fill)


# ──────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_walkable_from_sqlite(sqlite_path: pathlib.Path) -> Polygon | None:
    if not sqlite_path.exists():
        return None
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT wkt FROM geometry LIMIT 1")
        row = cur.fetchone()
        if row:
            return shapely.wkt.loads(row[0])
    except Exception as exc:
        print(f"  [WARN] geometry load failed ({sqlite_path.name}): {exc}")
    finally:
        conn.close()
    return None


def _load_all_positions_from_sqlite(
    sqlite_path: pathlib.Path,
) -> tuple[list[tuple[float, float]], int]:
    """Return (positions, n_frames) from trajectory_data.

    positions : list of (pos_x, pos_y) — all agents, all frames
    n_frames  : number of unique frame indices in the simulation
    """
    if not sqlite_path.exists():
        return [], 0
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        traj_tbl = next((t for t in tables if "trajectory" in t.lower()), None)
        if traj_tbl is None:
            return [], 0
        cur.execute(f"SELECT pos_x, pos_y, frame FROM {traj_tbl}")
        rows = cur.fetchall()
        positions = [(float(r[0]), float(r[1])) for r in rows]
        n_frames = len(set(r[2] for r in rows))
        return positions, max(n_frames, 1)
    finally:
        conn.close()


def _load_sampled_positions_from_sqlite(
    sqlite_path: pathlib.Path,
    frame_step: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return sampled (x,y) positions and sampled frame count from trajectory table."""
    if not sqlite_path.exists():
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), 0

    step = max(1, int(frame_step))
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        traj_tbl = next((t for t in tables if "trajectory" in t.lower()), None)
        if traj_tbl is None:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), 0

        if step == 1:
            cur.execute(f"SELECT pos_x, pos_y, frame FROM {traj_tbl}")
        else:
            cur.execute(f"SELECT pos_x, pos_y, frame FROM {traj_tbl} WHERE (frame % ?) = 0", (step,))
        rows = cur.fetchall()
        if not rows:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), 0

        arr = np.asarray(rows, dtype=np.float64)
        x = arr[:, 0].astype(np.float32, copy=False)
        y = arr[:, 1].astype(np.float32, copy=False)
        frames = arr[:, 2].astype(np.int64, copy=False)
        n_frames = int(np.unique(frames).size)
        return x, y, max(n_frames, 1)
    finally:
        conn.close()


def _fast_blur2d(arr: np.ndarray, iters: int = 1) -> np.ndarray:
    """Very fast 3x3 Gaussian-like blur using NumPy only."""
    if iters <= 0:
        return arr
    out = arr.astype(np.float32, copy=False)
    for _ in range(int(iters)):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] +
            2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:] +
            p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
        ) / 16.0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Geo JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_room_polygons(topo_root: pathlib.Path, plan_dir: str) -> list[Polygon]:
    p = topo_root / "geo" / plan_dir / "geo_room.json"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]


def _load_route_meta(topo_root: pathlib.Path, plan_dir: str, route_idx: int) -> dict:
    path = topo_root / "metadata" / plan_dir / f"route_{route_idx:02d}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _extract_spawn_points(meta: dict, variant_id: str | None, seed_str: str) -> list[tuple[float, float]]:
    if not meta:
        return []
    target_seed = None
    try:
        target_seed = int(seed_str)
    except ValueError:
        pass
    variants = meta.get("variants", [])
    for v in variants:
        vid = str(v.get("variant_id", "")).lower()
        vseed = v.get("route_seed")
        if variant_id is not None and vid != str(variant_id).lower():
            continue
        if target_seed is not None and vseed is not None and int(vseed) != target_seed:
            continue
        pts = v.get("spawn_positions_preview", [])
        if pts:
            return [(float(x), float(y)) for x, y in pts]
    pts = meta.get("spawn_positions_preview", [])
    return [(float(x), float(y)) for x, y in pts] if pts else []


def _node_index(node_id: str | None) -> int | None:
    if node_id and "-" in node_id:
        try:
            return int(node_id.split("-")[1])
        except ValueError:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Image renderers
# ──────────────────────────────────────────────────────────────────────────────

def _dot_radius(size_wh: tuple[int, int]) -> int:
    return max(4, int(min(size_wh) * 0.02))


def _draw_dot(draw, cx, cy, transform, fill, radius):
    px, py = _world_to_px(cx, cy, transform)
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill)


def _render_a_image(
    walkable: Polygon,
    bounds: tuple,
    spawn_zone: Polygon | None,
    exit_zone: Polygon | None,
    spawn_points: list[tuple[float, float]] | None,
    size_wh: tuple[int, int],
    preserve_aspect: bool = True,
) -> np.ndarray:
    """A (input) image — identical to prepare_unet_image_dataset.py."""
    canvas = Image.new("RGB", size_wh, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    transform = _build_canvas_transform(bounds, size_wh, preserve_aspect=preserve_aspect)

    _draw_shapely(draw, walkable, transform, fill=(255, 0, 0))

    if exit_zone and not exit_zone.is_empty:
        clipped = exit_zone.intersection(walkable)
        _draw_shapely(draw, clipped, transform, fill=(0, 0, 255))

    if spawn_points:
        rad = max(1, int(_dot_radius(size_wh) * 0.30))
        for sx, sy in spawn_points:
            _draw_dot(draw, sx, sy, transform, fill=(0, 255, 0), radius=rad)
    elif spawn_zone and not spawn_zone.is_empty:
        clipped = spawn_zone.intersection(walkable)
        if not clipped.is_empty:
            cx, cy = clipped.centroid.x, clipped.centroid.y
            _draw_dot(draw, cx, cy, transform, fill=(0, 255, 0), radius=_dot_radius(size_wh))

    return np.array(canvas)


def _render_b_densitymap(
    walkable: Polygon,
    bounds: tuple,
    sqlite_path: pathlib.Path,
    size_wh: tuple[int, int],
    grid_size_m: float = 0.5,
    vmax_density: float = 5.0,
    preserve_aspect: bool = True,
    density_scale_mode: str = "fixed",
    density_percentile: float = 99.0,
    density_gamma: float = 0.35,
    frame_step: int = 60,
    walkable_zero_gray: int = 8,
    torch_device=None,
    speed_mode: str = "accurate",
    turbo_blur_iters: int = 2,
) -> np.ndarray:
    """Render B density map as grayscale from the exact same density pipeline.

    Density computation itself stays identical to the baseline (pedpy CPU pipeline).
    GPU is used for grid mapping and tone-mapping when `torch_device` is CUDA.
    """
    speed_mode = str(speed_mode).strip().lower()
    if speed_mode not in {"accurate", "turbo"}:
        raise ValueError("speed_mode must be one of: accurate, turbo")

    W, H = size_wh
    transform = _build_canvas_transform(bounds, size_wh, preserve_aspect=preserve_aspect)

    if speed_mode == "accurate":
        import pedpy
        import pandas as pd

        # Same as reference heatmap generation.
        trajectory_data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=sqlite_path)
        loaded_walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=sqlite_path)

        individual_speed = pedpy.compute_individual_speed(
            traj_data=trajectory_data,
            frame_step=5,
            speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
        )
        individual_voronoi_cells = pedpy.compute_individual_voronoi_polygons(
            traj_data=trajectory_data,
            walkable_area=loaded_walkable_area,
            cut_off=pedpy.Cutoff(radius=0.8, quad_segments=3),
        )

        sum_density = None
        count = 0
        frames_to_process = individual_speed["frame"].unique()[::frame_step]
        for f in frames_to_process:
            speed_f = individual_speed[individual_speed.frame == f]
            cells_f = individual_voronoi_cells[individual_voronoi_cells.frame == f]
            frame_data = pd.merge(cells_f, speed_f, on=["id", "frame"])
            if frame_data.empty:
                continue
            d_profile, _ = pedpy.compute_profiles(
                individual_voronoi_speed_data=frame_data,
                walkable_area=loaded_walkable_area.polygon,
                grid_size=grid_size_m,
                speed_method=pedpy.SpeedMethod.ARITHMETIC,
            )
            arr = d_profile[0] if hasattr(d_profile, "__getitem__") else np.array(d_profile)
            if sum_density is None:
                sum_density = arr.copy().astype(np.float32)
            else:
                sum_density += arr.astype(np.float32)
            count += 1

        if count == 0 or sum_density is None:
            return np.zeros((H, W, 3), dtype=np.uint8)

        mean_density = sum_density / count
        min_x_w, _, _, max_y_w = loaded_walkable_area.polygon.bounds
        ny_ped, nx_ped = mean_density.shape
    else:
        # Turbo mode: occupancy-based density (much faster, approximate).
        min_x_w, min_y_w, max_x_w, max_y_w = walkable.bounds
        nx_ped = max(1, int(np.ceil((max_x_w - min_x_w) / grid_size_m)))
        ny_ped = max(1, int(np.ceil((max_y_w - min_y_w) / grid_size_m)))
        x, y, n_frames = _load_sampled_positions_from_sqlite(sqlite_path, frame_step=frame_step)
        if x.size == 0 or n_frames <= 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        if HAS_TORCH and torch_device is not None and getattr(torch_device, "type", "cpu") == "cuda":
            with torch.no_grad():
                x_t = torch.as_tensor(x, dtype=torch.float32, device=torch_device)
                y_t = torch.as_tensor(y, dtype=torch.float32, device=torch_device)
                ci_t = torch.floor((x_t - min_x_w) / grid_size_m).to(torch.int64)
                ri_t = torch.floor((max_y_w - y_t) / grid_size_m).to(torch.int64)
                valid_t = (ci_t >= 0) & (ci_t < nx_ped) & (ri_t >= 0) & (ri_t < ny_ped)
                if not bool(valid_t.any()):
                    return np.zeros((H, W, 3), dtype=np.uint8)
                flat_idx_t = (ri_t[valid_t] * nx_ped + ci_t[valid_t]).to(torch.int64)
                counts_t = torch.bincount(flat_idx_t, minlength=ny_ped * nx_ped).to(torch.float32)
                mean_density = counts_t.reshape(ny_ped, nx_ped) / (float(n_frames) * (grid_size_m ** 2))
                mean_density = mean_density.cpu().numpy()
        else:
            ci = np.floor((x - min_x_w) / grid_size_m).astype(np.int64)
            ri = np.floor((max_y_w - y) / grid_size_m).astype(np.int64)
            valid = (ci >= 0) & (ci < nx_ped) & (ri >= 0) & (ri < ny_ped)
            if not np.any(valid):
                return np.zeros((H, W, 3), dtype=np.uint8)
            flat_idx = ri[valid] * nx_ped + ci[valid]
            counts = np.bincount(flat_idx, minlength=ny_ped * nx_ped).astype(np.float32)
            mean_density = counts.reshape(ny_ped, nx_ped) / (float(n_frames) * (grid_size_m ** 2))

        mean_density = _fast_blur2d(mean_density, iters=max(0, int(turbo_blur_iters)))

    mask_img = Image.new("L", size_wh, 0)
    _draw_shapely(ImageDraw.Draw(mask_img), walkable, transform, fill=255, hole_fill=0)
    mask = np.array(mask_img, dtype=bool)

    use_gpu = bool(HAS_TORCH and torch_device is not None and getattr(torch_device, "type", "cpu") == "cuda")
    mode = density_scale_mode.lower().strip()
    if mode not in {"fixed", "image_max", "percentile"}:
        raise ValueError("density_scale_mode must be one of: fixed, image_max, percentile")

    if use_gpu:
        with torch.no_grad():
            device = torch_device
            mean_t = torch.as_tensor(mean_density, dtype=torch.float32, device=device)
            px_t = torch.arange(W, dtype=torch.float32, device=device)
            py_t = torch.arange(H, dtype=torch.float32, device=device)
            py_grid_t, px_grid_t = torch.meshgrid(py_t, px_t, indexing="ij")

            x_world_t = transform.min_x + (px_grid_t - transform.offset_x) / transform.scale_x
            y_world_t = transform.max_y - (py_grid_t - transform.offset_y) / transform.scale_y

            ci_t = torch.floor((x_world_t - min_x_w) / grid_size_m).to(torch.int64)
            # pedpy profile row indexing starts at top (max y), not bottom (min y).
            ri_t = torch.floor((max_y_w - y_world_t) / grid_size_m).to(torch.int64)

            render_mask_t = (
                (px_grid_t >= transform.offset_x)
                & (px_grid_t <= transform.offset_x + transform.span_x * transform.scale_x)
                & (py_grid_t >= transform.offset_y)
                & (py_grid_t <= transform.offset_y + transform.span_y * transform.scale_y)
            )
            valid_t = render_mask_t & (ci_t >= 0) & (ci_t < nx_ped) & (ri_t >= 0) & (ri_t < ny_ped)

            ci_safe_t = ci_t.clamp(0, nx_ped - 1)
            ri_safe_t = ri_t.clamp(0, ny_ped - 1)
            gather_idx_t = (ri_safe_t * nx_ped + ci_safe_t).reshape(-1)
            flat_mean_t = mean_t.reshape(-1)

            density_t = torch.zeros((H, W), dtype=torch.float32, device=device)
            density_t.view(-1)[valid_t.view(-1)] = flat_mean_t[gather_idx_t[valid_t.view(-1)]]

            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
            density_t[~mask_t] = 0.0

            valid_density_t = density_t[mask_t & (density_t > 0.0)]
            if valid_density_t.numel() == 0:
                return np.zeros((H, W, 3), dtype=np.uint8)

            if mode == "fixed":
                scale_ref = torch.tensor(float(vmax_density), dtype=torch.float32, device=device)
            elif mode == "image_max":
                scale_ref = valid_density_t.max()
            else:
                q = float(np.clip(density_percentile, 50.0, 100.0)) / 100.0
                try:
                    scale_ref = torch.quantile(valid_density_t, q)
                except Exception:
                    scale_ref = torch.tensor(
                        float(np.percentile(valid_density_t.detach().cpu().numpy(), q * 100.0)),
                        dtype=torch.float32,
                        device=device,
                    )

            scale_ref = torch.clamp(scale_ref, min=1e-9)
            gamma = max(float(density_gamma), 1e-6)
            base_gray = float(np.clip(walkable_zero_gray, 0, 254))
            norm_t = torch.clamp(density_t / scale_ref, 0.0, 1.0).pow(gamma)
            gray_t = torch.round(base_gray + norm_t * (255.0 - base_gray)).to(torch.uint8)
            gray_t[~mask_t] = 0
            gray = gray_t.cpu().numpy()
    else:
        px_arr = np.arange(W, dtype=np.float32)
        py_arr = np.arange(H, dtype=np.float32)
        px_grid, py_grid = np.meshgrid(px_arr, py_arr)
        x_world = transform.min_x + (px_grid - transform.offset_x) / transform.scale_x
        y_world = transform.max_y - (py_grid - transform.offset_y) / transform.scale_y

        ci = np.floor((x_world - min_x_w) / grid_size_m).astype(int)
        # pedpy profile row indexing starts at top (max y), not bottom (min y).
        ri = np.floor((max_y_w - y_world) / grid_size_m).astype(int)

        render_mask = (
            (px_grid >= transform.offset_x)
            & (px_grid <= transform.offset_x + transform.span_x * transform.scale_x)
            & (py_grid >= transform.offset_y)
            & (py_grid <= transform.offset_y + transform.span_y * transform.scale_y)
        )
        valid = render_mask & (ci >= 0) & (ci < nx_ped) & (ri >= 0) & (ri < ny_ped)
        ci_safe = np.clip(ci, 0, nx_ped - 1)
        ri_safe = np.clip(ri, 0, ny_ped - 1)
        density = np.where(valid, mean_density[ri_safe, ci_safe], 0.0)
        density[~mask] = 0.0

        valid_density = density[mask & (density > 0.0)]
        if valid_density.size == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        if mode == "fixed":
            scale_ref = float(vmax_density)
        elif mode == "image_max":
            scale_ref = float(np.max(valid_density))
        else:
            p = float(np.clip(density_percentile, 50.0, 100.0))
            scale_ref = float(np.percentile(valid_density, p))

        scale_ref = max(scale_ref, 1e-9)
        norm = np.power(np.clip(density / scale_ref, 0.0, 1.0), max(density_gamma, 1e-6))
        base_gray = int(np.clip(walkable_zero_gray, 0, 254))
        gray = np.rint(base_gray + norm * (255.0 - base_gray)).astype(np.uint8)
        gray[~mask] = 0

    return np.stack([gray, gray, gray], axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Pair collection (same logic as prepare_unet_image_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_traj_suffix(suffix: str) -> tuple[str, int, str | None]:
    s = suffix.replace(".png", "")
    parts = s.split("_")
    variants = {"full", "half", "single"}
    if parts and parts[-1].lower() in variants:
        return parts[-3], int(parts[-2]), parts[-1]
    return parts[-2], int(parts[-1]), None


def _collect_housegan_pairs(
    topo_root: pathlib.Path,
    trajectory_dir: pathlib.Path,
) -> list[dict]:
    pairs: list[dict] = []
    for traj_png in sorted(trajectory_dir.rglob("trajectory_*.png")):
        plan_dir = traj_png.parent.name
        suffix   = traj_png.name.replace("trajectory_", "", 1)
        pair_name = f"{plan_dir}__{suffix}"
        try:
            seed_str, route_idx, variant_id = _parse_traj_suffix(suffix)
        except (IndexError, ValueError) as exc:
            print(f"  [SKIP] Cannot parse '{suffix}': {exc}")
            continue

        sqlite_name = f"plan_sim_{suffix.replace('.png', '')}.sqlite"
        sqlite_path = topo_root / "dataswarm" / plan_dir / sqlite_name

        meta         = _load_route_meta(topo_root, plan_dir, route_idx)
        start_idx    = _node_index(meta.get("start_node"))
        end_idx      = _node_index(meta.get("end_node"))
        spawn_points = _extract_spawn_points(meta, variant_id, seed_str)

        pairs.append({
            "plan_dir":    plan_dir,
            "pair_name":   pair_name,
            "sqlite_path": sqlite_path,
            "start_idx":   start_idx,
            "end_idx":     end_idx,
            "spawn_points": spawn_points,
        })
    return pairs


def _render_and_save_pair_task(task: dict) -> dict:
    """Worker task: render one A/B pair and save directly to disk."""
    topo_root = pathlib.Path(task["topo_root"])
    output_dir = pathlib.Path(task["output_dir"])
    target_size_wh = tuple(task["target_size_wh"])

    plan_dir = task["plan_dir"]
    pair_name = task["pair_name"]
    sqlite_path = pathlib.Path(task["sqlite_path"])
    start_idx = task["start_idx"]
    end_idx = task["end_idx"]
    spawn_points = task.get("spawn_points", [])
    try:
        walkable = _load_walkable_from_sqlite(sqlite_path)
        if walkable is None or walkable.is_empty:
            return {"status": "skip", "pair_name": pair_name, "reason": f"No walkable geometry in {sqlite_path.name}"}

        if not sqlite_path.exists():
            return {"status": "skip", "pair_name": pair_name, "reason": f"sqlite not found: {sqlite_path.name}"}

        bounds = _canvas_bounds(walkable, padding=float(task["canvas_padding_m"]))
        rooms = _load_room_polygons(topo_root, plan_dir)
        spawn_zone = rooms[start_idx] if (start_idx is not None and start_idx < len(rooms)) else None
        exit_zone = rooms[end_idx] if (end_idx is not None and end_idx < len(rooms)) else None

        a_img = _render_a_image(
            walkable,
            bounds,
            spawn_zone,
            exit_zone,
            spawn_points,
            target_size_wh,
            preserve_aspect=bool(task["preserve_aspect"]),
        )

        torch_device, _ = _resolve_torch_device(task["device_preference"], force_gpu=bool(task["force_gpu"]))
        b_img = _render_b_densitymap(
            walkable,
            bounds,
            sqlite_path,
            target_size_wh,
            grid_size_m=float(task["grid_size_m"]),
            frame_step=int(task["frame_step"]),
            vmax_density=float(task["vmax_density"]),
            preserve_aspect=bool(task["preserve_aspect"]),
            density_scale_mode=task["density_scale_mode"],
            density_percentile=float(task["density_percentile"]),
            density_gamma=float(task["density_gamma"]),
            walkable_zero_gray=int(task["walkable_zero_gray"]),
            torch_device=torch_device,
            speed_mode=task["speed_mode"],
            turbo_blur_iters=int(task["turbo_blur_iters"]),
        )

        save_rgb(a_img, output_dir / "A" / pair_name)
        save_rgb(b_img, output_dir / "B" / pair_name)
        return {"status": "ok", "pair_name": pair_name}
    except Exception as exc:
        return {"status": "skip", "pair_name": pair_name, "reason": f"Render failed for {sqlite_path.name}: {exc}"}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_densitymap_dataset(
    topo_root: pathlib.Path,
    trajectory_dir: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite: bool,
    seed: int,
    max_pairs: int = 0,
    target_size_wh: tuple[int, int] = (512, 512),
    canvas_padding_m: float = 0.0,
    grid_size_m: float = 0.5,
    frame_step: int = 60,
    vmax_density: float = 5.0,
    preserve_aspect: bool = True,
    density_scale_mode: str = "fixed",
    density_percentile: float = 99.0,
    density_gamma: float = 0.35,
    walkable_zero_gray: int = 8,
    device_preference: str = "auto",
    force_gpu: bool = False,
    workers: int = 1,
    speed_mode: str = "accurate",
    turbo_blur_iters: int = 2,
) -> None:
    """Build A/B pairs where:
       A = floor-plan with spawn dot + exit area   (same as trajectory dataset)
       B = density heatmap in persons/m², grayscale, fixed scale [0, vmax_density]

    Both images share the same canonical coordinate system derived from the
    SQLite geometry table, ensuring pixel-perfect alignment.
    """
    ensure_clean_output(output_dir, overwrite=overwrite)
    torch_device, device_label = _resolve_torch_device(device_preference, force_gpu=force_gpu)
    print(f"[DEVICE] density map device: {device_label}")
    workers = max(1, int(workers))
    print(f"[WORKERS] parallel workers: {workers}")
    print(f"[SPEED] mode: {speed_mode}")

    pairs = _collect_housegan_pairs(topo_root, trajectory_dir)
    if not pairs:
        raise RuntimeError(f"No renderable pairs found under {trajectory_dir}.")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]

    for side in ("A", "B"):
        (output_dir / side).mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    tasks = []
    for pair in pairs:
        tasks.append(
            {
                "topo_root": str(topo_root),
                "output_dir": str(output_dir),
                "target_size_wh": list(target_size_wh),
                "canvas_padding_m": canvas_padding_m,
                "grid_size_m": grid_size_m,
                "frame_step": frame_step,
                "vmax_density": vmax_density,
                "preserve_aspect": preserve_aspect,
                "density_scale_mode": density_scale_mode,
                "density_percentile": density_percentile,
                "density_gamma": density_gamma,
                "walkable_zero_gray": walkable_zero_gray,
                "device_preference": device_preference,
                "force_gpu": force_gpu,
                "speed_mode": speed_mode,
                "turbo_blur_iters": turbo_blur_iters,
                **pair,
            }
        )

    progress = tqdm(
        total=len(tasks),
        desc=f"[{topo_root.name}] Render A/B",
        unit="pair",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    if workers == 1:
        for task in tasks:
            result = _render_and_save_pair_task(task)
            if result["status"] == "ok":
                ok += 1
            else:
                skip += 1
                print(f"  [SKIP] {result.get('reason', result.get('pair_name', 'unknown'))}")
            progress.update(1)
            progress.set_postfix(rendered=ok, skipped=skip, refresh=False)
    else:
        # CUDA cannot be safely re-initialized in forked subprocesses.
        # Always use "spawn" context for multiprocessing workers.
        spawn_ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=spawn_ctx) as ex:
            futures = [ex.submit(_render_and_save_pair_task, task) for task in tasks]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"status": "skip", "reason": f"Worker crashed: {exc}"}

                if result.get("status") == "ok":
                    ok += 1
                else:
                    skip += 1
                    print(f"  [SKIP] {result.get('reason', result.get('pair_name', 'unknown'))}")

                progress.update(1)
                progress.set_postfix(rendered=ok, skipped=skip, refresh=False)

    progress.close()

    print(f"[BUILD] {ok} pairs rendered, {skip} skipped  →  {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare density-map A/B datasets from Geo_scenario source data."
    )
    parser.add_argument("--source_root",   type=pathlib.Path, required=True,
                        help="Root of Geo_scenario (e.g. .../Geo_scenario)")
    parser.add_argument("--output_root",   type=pathlib.Path, required=True,
                        help="Root of Dataset/Data_ImageUNet (e.g. .../Dataset/Data_ImageUNet)")
    parser.add_argument("--topologies",    nargs="+", required=True,
                        help="Topology folder names, e.g. Topo_HouseGAN")
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--max_pairs",     type=int, default=0,
                        help="Debug: limit pairs (0 = all)")
    parser.add_argument("--target_size",   type=int, default=512,
                        help="Square canvas size in pixels (default 512)")
    parser.add_argument("--canvas_padding", type=float, default=0.0,
                        help="Padding around walkable area in metres (default 0.0)")
    parser.add_argument("--stretch_to_canvas", action="store_true",
                        help="Legacy behavior: stretch X/Y independently to fill the square canvas.")
    parser.add_argument("--grid_size",     type=float, default=0.5,
                        help="Coarse grid cell size in metres (default 0.5, same as heatmap script)")
    parser.add_argument("--frame_step",    type=int, default=60,
                        help="Sample every N frames when averaging density (default 60). Larger = faster.")
    parser.add_argument("--vmax_density",  type=float, default=5.0,
                        help="Density scale ceiling in persons/m² (default 5.0, same as heatmap script)")
    parser.add_argument("--density_scale_mode", type=str, default="fixed",
                        choices=["fixed", "image_max", "percentile"],
                        help="How to scale density to grayscale (default: fixed).")
    parser.add_argument("--density_percentile", type=float, default=99.0,
                        help="Percentile used when --density_scale_mode=percentile (default: 99.0).")
    parser.add_argument("--density_gamma", type=float, default=0.35,
                        help="Gamma for grayscale tone mapping, <1 brightens low density (default: 0.35).")
    parser.add_argument("--walkable_zero_gray", type=int, default=8,
                        help="Gray value for walkable pixels where density is zero (default: 8).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for density map raster/tone operations (default: auto).")
    parser.add_argument("--force_gpu", action="store_true",
                        help="Fail fast if CUDA is unavailable instead of falling back.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for pair rendering (default: 1).")
    parser.add_argument("--speed_mode", type=str, default="accurate", choices=["accurate", "turbo"],
                        help="accurate: pedpy Voronoi baseline; turbo: much faster occupancy approximation.")
    parser.add_argument("--turbo_blur_iters", type=int, default=2,
                        help="Smoothing iterations for turbo mode (default: 2).")
    args = parser.parse_args()

    source_root    = args.source_root.resolve()
    output_root    = args.output_root.resolve()
    target_size_wh = (args.target_size, args.target_size)

    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"source_root  : {source_root}")
    print(f"output_root  : {output_root}")
    print(f"topologies   : {', '.join(args.topologies)}")
    print(f"target_size  : {args.target_size}×{args.target_size} px")
    print(f"grid_size    : {args.grid_size} m")
    print(f"frame_step   : {args.frame_step}")
    print(f"aspect_mode  : {'stretch' if args.stretch_to_canvas else 'preserve'}")
    print(
        "density_map  : "
        f"mode={args.density_scale_mode}, p={args.density_percentile}, "
        f"gamma={args.density_gamma}, zero_gray={args.walkable_zero_gray}"
    )
    print(f"device       : {args.device} (force_gpu={args.force_gpu})")
    print(f"workers      : {args.workers}")
    print(f"speed_mode   : {args.speed_mode}")
    print(f"turbo_blur   : {args.turbo_blur_iters}")
    print("=" * 70)

    rows = []
    for topo in args.topologies:
        # Source trajectory_line PNGs (used only for filename enumeration)
        topo_root      = source_root / topo
        trajectory_dir = topo_root / "trajectory_line_dataset" / "Cleandata_1"
        # For HouseGAN the PNG files sit under plan_XXXX sub-dirs
        if not trajectory_dir.exists():
            trajectory_dir = topo_root / "trajectory_line"
        if not trajectory_dir.exists():
            print(f"[MISS] No trajectory source for '{topo}': {trajectory_dir}")
            rows.append({"topology": topo, "status": "missing_source"})
            continue

        dst_dataset = output_root / "DensityMap_dataset" / topo

        try:
            build_densitymap_dataset(
                topo_root        = topo_root,
                trajectory_dir   = trajectory_dir,
                output_dir       = dst_dataset,
                overwrite        = args.overwrite,
                seed             = args.seed,
                max_pairs        = args.max_pairs,
                target_size_wh   = target_size_wh,
                canvas_padding_m = args.canvas_padding,
                grid_size_m      = args.grid_size,
                frame_step       = args.frame_step,
                vmax_density     = args.vmax_density,
                preserve_aspect  = not args.stretch_to_canvas,
                density_scale_mode = args.density_scale_mode,
                density_percentile = args.density_percentile,
                density_gamma      = args.density_gamma,
                walkable_zero_gray = args.walkable_zero_gray,
                device_preference  = args.device,
                force_gpu          = args.force_gpu,
                workers            = args.workers,
                speed_mode         = args.speed_mode,
                turbo_blur_iters   = args.turbo_blur_iters,
            )
        except Exception as exc:
            print(f"[ERROR] {topo}: {exc}")
            rows.append({"topology": topo, "status": f"error: {exc}"})
            continue

        counts = count_split_images(dst_dataset)
        rows.append({
            "topology":       topo,
            "status":         "prepared",
            "output_dataset": str(dst_dataset),
            **counts,
            "updated_at":     datetime.now().isoformat(timespec="seconds"),
        })
        print(f"[STAT] {topo} | total_png={counts['total_png']}")

    manifest_path = output_root / "manifest_densitymap_dataset.csv"
    fieldnames = ["topology", "output_dataset", "status",
                  "A_total", "B_total",
                  "total_png", "dataset_layout", "updated_at"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    prepared = sum(1 for r in rows if r.get("status") == "prepared")
    print("-" * 70)
    print(f"[SUMMARY] prepared={prepared}  total={len(rows)}")
    print(f"[SUMMARY] manifest={manifest_path}")


if __name__ == "__main__":
    main()

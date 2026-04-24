"""prepare_unet_image_dataset.py
Prepare A/B image datasets for UNet/pix2pix training.

For Topo_HouseGAN the images are re-rendered from source data
(SQLite geometry + trajectory, geo JSON room polygons, route metadata)
so that A and B always share the same exact coordinate system.

A image colour scheme  (input – spawn/exit map):
    black  (0,0,0)     = walls / non-walkable
    red    (255,0,0)   = walkable floor
    green  (0,255,0)   = spawn room (clipped to walkable area)
    blue   (0,0,255)   = exit  room (clipped to walkable area)

B image colour scheme  (ground truth – trajectory map):
    black      (0,0,0)       = walls / non-walkable
    gray       (220,220,220) = walkable floor
    pink       (255,182,193) = pedestrian trajectory lines
"""

import argparse
import csv
import json
import pathlib
import random
import shutil
import sqlite3
from datetime import datetime
from itertools import groupby

import numpy as np
import shapely.wkt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, shape as shapely_shape


# ──────────────────────────────────────────────────────────────────────────────
# Generic dataset utilities (unchanged between topologies)
# ──────────────────────────────────────────────────────────────────────────────

def count_split_images(dataset_root: pathlib.Path) -> dict:
    summary = {
        "A_train": 0, "A_test": 0, "A_validation": 0,
        "B_train": 0, "B_test": 0, "B_validation": 0,
        "total_png": 0, "dataset_layout": "ab_split",
    }
    for side in ("A", "B"):
        for split in ("train", "test", "validation"):
            n = len(list((dataset_root / side / split).glob("*.png")))
            summary[f"{side}_{split}"] = n
            summary["total_png"] += n
    return summary


def copy_dataset(source_dir: pathlib.Path, output_dir: pathlib.Path, overwrite: bool) -> int:
    if output_dir.exists():
        if not overwrite:
            print(f"[SKIP] Output exists (use --overwrite to replace): {output_dir}")
            return 0
        print(f"[CLEAN] Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COPY] {source_dir} -> {output_dir}")
    shutil.copytree(source_dir, output_dir)
    n = len(list(output_dir.rglob("*")))
    print(f"[DONE] Copied {n} entries.")
    return n


def ensure_clean_output(output_dir: pathlib.Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite to replace): {output_dir}")
        print(f"[CLEAN] Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def detect_layout(source_dir: pathlib.Path, topo_root: pathlib.Path) -> str:
    has_ab = all((source_dir / s).exists() for s in ("A", "B"))
    has_split = all((source_dir / "A" / s).exists() for s in ("train", "test", "validation"))
    if has_ab and has_split:
        return "ab_ready"
    if source_dir.name == "trajectory_line" and (topo_root / "spawn_exit").exists():
        return "trajectory_plus_spawn_exit"
    return "unsupported"


def split_bucket(index: int, n: int) -> str:
    if n == 0:
        return "train"
    if index < int(0.8 * n):
        return "train"
    if index < int(0.9 * n):
        return "validation"
    return "test"


def save_rgb(arr: np.ndarray, path: pathlib.Path) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ──────────────────────────────────────────────────────────────────────────────

def _world_to_px(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    size_wh: tuple[int, int],
) -> tuple[int, int]:
    """World (x, y) → image pixel (col, row).
    Y-axis is flipped: world max_y maps to image row 0 (top).
    """
    min_x, min_y, max_x, max_y = bounds
    w, h = size_wh
    px = int(round(((x - min_x) / max(max_x - min_x, 1e-9)) * (w - 1)))
    py = int(round(((max_y - y) / max(max_y - min_y, 1e-9)) * (h - 1)))
    return max(0, min(w - 1, px)), max(0, min(h - 1, py))


def _canvas_bounds(walkable: Polygon, padding: float = 1.0) -> tuple[float, float, float, float]:
    """Walkable polygon bounds expanded by `padding` metres on all sides."""
    min_x, min_y, max_x, max_y = walkable.bounds
    return min_x - padding, min_y - padding, max_x + padding, max_y + padding


# ──────────────────────────────────────────────────────────────────────────────
# Shapely → PIL rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_shapely(
    draw: ImageDraw.ImageDraw,
    geom,
    bounds: tuple,
    size_wh: tuple[int, int],
    fill: tuple[int, int, int],
    hole_fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Rasterise a shapely Polygon or MultiPolygon onto *draw*.

    Exterior ring is filled with *fill*.
    Interior rings (holes, e.g. wall islands) are filled with *hole_fill*.
    """
    if geom is None or geom.is_empty:
        return
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        if poly.is_empty:
            continue
        # Exterior
        ext_px = [_world_to_px(x, y, bounds, size_wh) for x, y in poly.exterior.coords]
        if len(ext_px) >= 3:
            draw.polygon(ext_px, fill=fill)
        # Interior rings (holes → back to background / obstacle colour)
        for interior in poly.interiors:
            int_px = [_world_to_px(x, y, bounds, size_wh) for x, y in interior.coords]
            if len(int_px) >= 3:
                draw.polygon(int_px, fill=hole_fill)


# ──────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_walkable_from_sqlite(sqlite_path: pathlib.Path) -> Polygon | None:
    """Read the walkable-area WKT from a jupedsim SQLite file.

    The `geometry` table stores the exact polygon that the simulation used,
    including correct wall cutouts and door openings.
    """
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


def _load_trajectories_from_sqlite(
    sqlite_path: pathlib.Path,
) -> list[list[tuple[float, float]]]:
    """Return a list of agent paths.  Each path is a list of (x, y) world coords
    ordered by frame number.
    """
    if not sqlite_path.exists():
        return []
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        traj_tbl = next((t for t in tables if "trajectory" in t.lower()), None)
        if traj_tbl is None:
            return []
        cur.execute(f"SELECT id, pos_x, pos_y FROM {traj_tbl} ORDER BY id, frame")
        rows = cur.fetchall()
        paths = []
        for _agent_id, pts_iter in groupby(rows, key=lambda r: r[0]):
            coords = [(r[1], r[2]) for r in pts_iter]
            if len(coords) >= 2:
                paths.append(coords)
        return paths
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Geo JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_room_polygons(topo_root: pathlib.Path, plan_dir: str) -> list[Polygon]:
    """Load room polygons from geo_room.json as shapely Polygons."""
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
    """Extract spawn_positions_preview for the exact variant/seed in route metadata."""
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
    # fallback: route-level preview
    pts = meta.get("spawn_positions_preview", [])
    return [(float(x), float(y)) for x, y in pts] if pts else []


def _node_index(node_id: str | None) -> int | None:
    """'Room-6' → 6,  None / bad format → None."""
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
    """Dot radius that scales with canvas size (≈ 2% of shorter side, min 4px)."""
    return max(4, int(min(size_wh) * 0.02))


def _draw_dot(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    bounds: tuple,
    size_wh: tuple[int, int],
    fill: tuple[int, int, int],
    radius: int,
) -> None:
    """Draw a filled circle at world position (cx, cy)."""
    px, py = _world_to_px(cx, cy, bounds, size_wh)
    draw.ellipse(
        [px - radius, py - radius, px + radius, py + radius],
        fill=fill,
    )


def _render_a_image(
    walkable: Polygon,
    bounds: tuple,
    spawn_zone: Polygon | None,
    exit_zone: Polygon | None,
    spawn_points: list[tuple[float, float]] | None,
    size_wh: tuple[int, int],
) -> np.ndarray:
    """Render the A (input) image.

    Layer order (bottom → top):
      1. Black canvas              → wall / obstacle
      2. Walkable area (red)       → floor
      3. Exit room (blue, filled)  → destination area
      4. Spawn dot (green circle)  → spawn position (centroid of spawn room)
    """
    canvas = Image.new("RGB", size_wh, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    _draw_shapely(draw, walkable, bounds, size_wh, fill=(255, 0, 0))

    # Exit zone: fill the whole room area (destination)
    if exit_zone and not exit_zone.is_empty:
        clipped = exit_zone.intersection(walkable)
        _draw_shapely(draw, clipped, bounds, size_wh, fill=(0, 0, 255))

    # Spawn: draw all spawn points (preferred), fallback to centroid
    if spawn_points:
        rad = max(1, int(_dot_radius(size_wh) * 0.30))
        for sx, sy in spawn_points:
            _draw_dot(draw, sx, sy, bounds, size_wh, fill=(0, 255, 0), radius=rad)
    elif spawn_zone and not spawn_zone.is_empty:
        clipped = spawn_zone.intersection(walkable)
        if not clipped.is_empty:
            cx, cy = clipped.centroid.x, clipped.centroid.y
            _draw_dot(draw, cx, cy, bounds, size_wh, fill=(0, 255, 0), radius=_dot_radius(size_wh))

    return np.array(canvas)


def _render_b_image(
    walkable: Polygon,
    bounds: tuple,
    paths: list[list[tuple[float, float]]],
    size_wh: tuple[int, int],
    line_width: int = 2,
) -> np.ndarray:
    """Render the B (ground-truth) image.

    Layer order (bottom → top):
      1. Black canvas  → wall / obstacle
      2. Walkable area (gray)
      3. Trajectory lines (pink), one polyline per agent
    """
    canvas = Image.new("RGB", size_wh, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    _draw_shapely(draw, walkable, bounds, size_wh, fill=(220, 220, 220))

    for path in paths:
        px_path = [_world_to_px(x, y, bounds, size_wh) for x, y in path]
        if len(px_path) >= 2:
            draw.line(px_path, fill=(255, 182, 193), width=line_width)

    return np.array(canvas)


# ──────────────────────────────────────────────────────────────────────────────
# Pair collection
# ──────────────────────────────────────────────────────────────────────────────

def _parse_traj_suffix(suffix: str) -> tuple[str, int, str | None]:
    """'42_00_full.png' → ('42', 0, 'full');  '42_00.png' → ('42', 0, None)."""
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
    """Walk trajectory_line/ and build a record for every renderable A/B pair."""
    pairs: list[dict] = []

    for traj_png in sorted(trajectory_dir.rglob("trajectory_*.png")):
        plan_dir = traj_png.parent.name          # e.g. "plan_100_8ec0"
        suffix   = traj_png.name.replace("trajectory_", "", 1)   # "42_00_full.png"
        pair_name = f"{plan_dir}__{suffix}"

        try:
            seed_str, route_idx, variant_id = _parse_traj_suffix(suffix)
        except (IndexError, ValueError) as exc:
            print(f"  [SKIP] Cannot parse '{suffix}': {exc}")
            continue

        # SQLite: dataswarm/plan_dir/plan_sim_{suffix_no_ext}.sqlite
        sqlite_name = f"plan_sim_{suffix.replace('.png', '')}.sqlite"
        sqlite_path = topo_root / "dataswarm" / plan_dir / sqlite_name

        # Route metadata → which rooms are spawn / exit
        meta       = _load_route_meta(topo_root, plan_dir, route_idx)
        start_idx  = _node_index(meta.get("start_node"))
        end_idx    = _node_index(meta.get("end_node"))
        spawn_points = _extract_spawn_points(meta, variant_id, seed_str)

        pairs.append({
            "plan_dir":   plan_dir,
            "pair_name":  pair_name,
            "sqlite_path": sqlite_path,
            "start_idx":  start_idx,   # room index for spawn
            "end_idx":    end_idx,     # room index for exit
            "spawn_points": spawn_points,
        })

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_ab_dataset_from_housegan(
    topo_root: pathlib.Path,
    trajectory_dir: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite: bool,
    seed: int,
    max_pairs: int = 0,
    target_size_wh: tuple[int, int] = (512, 512),
    canvas_padding_m: float = 1.0,
    line_width: int = 2,
) -> None:
    """Build perfectly-aligned A/B pairs from geo + SQLite source data.

    Both A and B are rendered on the SAME canonical coordinate system:
      - Canvas bounds  = walkable_polygon.bounds + canvas_padding_m
      - Walkable area  = exact polygon from SQLite geometry table
                         (correct walls, doors, corridors – not approximated)
      - Spawn / Exit   = geo_room.json polygon clipped to walkable area
      - Trajectories   = world (x,y) from SQLite trajectory_data table

    The matplotlib-generated PNG images are NOT used at all.
    """
    ensure_clean_output(output_dir, overwrite=overwrite)

    pairs = _collect_housegan_pairs(topo_root, trajectory_dir)
    if not pairs:
        raise RuntimeError(f"No renderable pairs found under {trajectory_dir}.")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]

    for side in ("A", "B"):
        for split in ("train", "test", "validation"):
            (output_dir / side / split).mkdir(parents=True, exist_ok=True)

    ok = skip = 0

    for i, pair in enumerate(pairs):
        plan_dir    = pair["plan_dir"]
        pair_name   = pair["pair_name"]
        sqlite_path: pathlib.Path = pair["sqlite_path"]
        start_idx   = pair["start_idx"]
        end_idx     = pair["end_idx"]
        spawn_points = pair.get("spawn_points", [])

        # ── Walkable area (from SQLite – exact simulation geometry) ──────────
        walkable = _load_walkable_from_sqlite(sqlite_path)
        if walkable is None or walkable.is_empty:
            print(f"  [SKIP] No walkable geometry in {sqlite_path.name}")
            skip += 1
            continue

        bounds = _canvas_bounds(walkable, padding=canvas_padding_m)

        # ── Room polygons for spawn / exit ────────────────────────────────────
        rooms = _load_room_polygons(topo_root, plan_dir)
        spawn_zone = rooms[start_idx] if (start_idx is not None and start_idx < len(rooms)) else None
        exit_zone  = rooms[end_idx]   if (end_idx   is not None and end_idx   < len(rooms)) else None

        # ── Trajectory paths (from SQLite) ─────────────────────────────────
        paths = _load_trajectories_from_sqlite(sqlite_path)

        # ── Render ─────────────────────────────────────────────────────────
        split_name = split_bucket(i, len(pairs))

        a_img = _render_a_image(walkable, bounds, spawn_zone, exit_zone, spawn_points, target_size_wh)
        b_img = _render_b_image(walkable, bounds, paths, target_size_wh, line_width=line_width)

        save_rgb(a_img, output_dir / "A" / split_name / pair_name)
        save_rgb(b_img, output_dir / "B" / split_name / pair_name)
        ok += 1

    print(f"[BUILD] {ok} pairs rendered, {skip} skipped  →  {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare UNet image datasets from Geo_scenario to Dataset/Data_ImageUNet."
    )
    parser.add_argument("--source_root",     type=pathlib.Path, required=True)
    parser.add_argument("--output_root",     type=pathlib.Path, required=True)
    parser.add_argument("--dataset_subpath", type=str, default="trajectory_line_dataset/Cleandata_1")
    parser.add_argument("--output_group",    type=str, default="Trajectory_line_dataset")
    parser.add_argument("--topologies",      nargs="+", required=True)
    parser.add_argument("--overwrite",       action="store_true")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--max_pairs",       type=int, default=0,
                        help="Debug: limit number of pairs (0 = all).")
    parser.add_argument("--target_size",     type=int, default=512,
                        help="Square canvas size in pixels (default 512).")
    parser.add_argument("--canvas_padding",  type=float, default=1.0,
                        help="Padding around walkable area in metres (default 1.0).")
    parser.add_argument("--line_width",      type=int, default=2,
                        help="Trajectory line width in pixels (default 2).")
    args = parser.parse_args()

    source_root    = args.source_root.resolve()
    output_root    = args.output_root.resolve()
    dataset_parts  = pathlib.Path(args.dataset_subpath)
    target_size_wh = (args.target_size, args.target_size)

    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"source_root  : {source_root}")
    print(f"output_root  : {output_root}")
    print(f"dataset_path : {dataset_parts}")
    print(f"topologies   : {', '.join(args.topologies)}")
    print(f"target_size  : {args.target_size}×{args.target_size} px")
    print(f"padding      : {args.canvas_padding} m")
    print("=" * 70)

    rows = []
    for topo in args.topologies:
        src_dataset = source_root / topo / dataset_parts
        dst_dataset = output_root / args.output_group / topo

        if not src_dataset.exists():
            print(f"[MISS] Source not found for '{topo}': {src_dataset}")
            rows.append({"topology": topo, "status": "missing_source",
                         "source_dataset": str(src_dataset), "output_dataset": str(dst_dataset),
                         "A_train": 0, "A_test": 0, "A_validation": 0,
                         "B_train": 0, "B_test": 0, "B_validation": 0,
                         "total_png": 0, "dataset_layout": "missing_source",
                         "updated_at": datetime.now().isoformat(timespec="seconds")})
            continue

        layout = detect_layout(src_dataset, source_root / topo)

        if layout == "ab_ready":
            copy_dataset(src_dataset, dst_dataset, overwrite=args.overwrite)

        elif layout == "trajectory_plus_spawn_exit":
            build_ab_dataset_from_housegan(
                topo_root       = source_root / topo,
                trajectory_dir  = src_dataset,
                output_dir      = dst_dataset,
                overwrite       = args.overwrite,
                seed            = args.seed,
                max_pairs       = args.max_pairs,
                target_size_wh  = target_size_wh,
                canvas_padding_m = args.canvas_padding,
                line_width      = args.line_width,
            )

        else:
            print(f"[MISS] Unsupported layout for '{topo}': {src_dataset}")
            rows.append({"topology": topo, "status": "unsupported_layout",
                         "source_dataset": str(src_dataset), "output_dataset": str(dst_dataset),
                         "A_train": 0, "A_test": 0, "A_validation": 0,
                         "B_train": 0, "B_test": 0, "B_validation": 0,
                         "total_png": 0, "dataset_layout": "unsupported_layout",
                         "updated_at": datetime.now().isoformat(timespec="seconds")})
            continue

        counts = count_split_images(dst_dataset)
        rows.append({"topology": topo, "status": "prepared",
                     "source_dataset": str(src_dataset), "output_dataset": str(dst_dataset),
                     **counts,
                     "updated_at": datetime.now().isoformat(timespec="seconds")})
        print(f"[STAT] {topo} | total_png={counts['total_png']}")

    manifest_path = output_root / "manifest_unet_image_dataset.csv"
    fieldnames = ["topology", "source_dataset", "output_dataset", "status",
                  "A_train", "A_test", "A_validation",
                  "B_train", "B_test", "B_validation",
                  "total_png", "dataset_layout", "updated_at"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    prepared = sum(1 for r in rows if r["status"] == "prepared")
    missing  = len(rows) - prepared
    print("-" * 70)
    print(f"[SUMMARY] prepared={prepared}  missing={missing}")
    print(f"[SUMMARY] manifest={manifest_path}")


if __name__ == "__main__":
    main()

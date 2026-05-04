"""Prepare A/B image datasets for trajectory-line mask prediction.

This is the mask version of the trajectory-line dataset builder:

A image (input):
    black (0,0,0)   = walls / non-walkable
    red   (255,0,0) = walkable floor
    green (0,255,0) = spawn position
    blue  (0,0,255) = exit room

B image (target mask):
    black (0,0,0)       = background
    white (255,255,255) = pedestrian trajectory line only

For Topo_HouseGAN, B is rendered from SQLite trajectory coordinates so A and B
share the exact same coordinate transform. The old matplotlib PNGs are used
only to enumerate which simulations should be rendered.
"""

import argparse
import csv
import pathlib
import random
import sqlite3
import sys
from datetime import datetime
from itertools import groupby

import numpy as np
import shapely.wkt
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from prepare_densitymap_dataset import (
    _build_canvas_transform,
    _canvas_bounds,
    _collect_housegan_pairs,
    _load_room_polygons,
    _render_a_image,
    _world_to_px,
    count_split_images,
    ensure_clean_output,
    save_rgb,
)
from prepare_split_dataset import extract_group_key, normalize_ratios, split_group_keys


def _readonly_sqlite_uri(sqlite_path: pathlib.Path) -> str:
    """Open simulation SQLite files without taking locks; we never write them."""
    return f"file:{sqlite_path}?mode=ro&immutable=1"


def _load_walkable_from_sqlite(sqlite_path: pathlib.Path):
    """Read the walkable-area WKT from a jupedsim SQLite file."""
    if not sqlite_path.exists():
        return None

    conn = sqlite3.connect(_readonly_sqlite_uri(sqlite_path), uri=True)
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
    """Return one ordered (x, y) path per pedestrian agent."""
    if not sqlite_path.exists():
        return []

    conn = sqlite3.connect(_readonly_sqlite_uri(sqlite_path), uri=True)
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
            coords = [(float(r[1]), float(r[2])) for r in pts_iter]
            if len(coords) >= 2:
                paths.append(coords)
        return paths
    finally:
        conn.close()


def _render_b_trajectory_mask(
    bounds: tuple[float, float, float, float],
    paths: list[list[tuple[float, float]]],
    size_wh: tuple[int, int],
    line_width: int = 2,
    preserve_aspect: bool = True,
) -> np.ndarray:
    """Render B as a binary-like RGB mask: black background, white lines."""
    canvas = Image.new("RGB", size_wh, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    transform = _build_canvas_transform(bounds, size_wh, preserve_aspect=preserve_aspect)

    for path in paths:
        px_path = [_world_to_px(x, y, transform) for x, y in path]
        if len(px_path) >= 2:
            draw.line(px_path, fill=(255, 255, 255), width=line_width)

    return np.array(canvas)


def _split_for_pairs(
    pairs: list[dict],
    seed: int,
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    """Assign split by plan/group key to avoid leaking the same layout."""
    train_r, test_r, val_r = normalize_ratios(train_ratio, test_ratio, val_ratio)
    del val_r  # split_group_keys derives validation from the remainder.

    group_keys = sorted({extract_group_key(str(pair["pair_name"])) for pair in pairs})
    split_keys = split_group_keys(group_keys, train_r, test_r, seed)

    split_by_group = {}
    for split_name, keys in split_keys.items():
        for key in keys:
            split_by_group[key] = split_name
    return split_by_group


def build_trajectory_mask_dataset(
    topo_root: pathlib.Path,
    trajectory_dir: pathlib.Path,
    output_dir: pathlib.Path,
    overwrite: bool,
    seed: int,
    max_pairs: int = 0,
    target_size_wh: tuple[int, int] = (512, 512),
    canvas_padding_m: float = 0.0,
    line_width: int = 2,
    preserve_aspect: bool = True,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> None:
    """Build split A/B pairs where B is a white trajectory-line mask."""
    ensure_clean_output(output_dir, overwrite=overwrite)

    pairs = _collect_housegan_pairs(topo_root, trajectory_dir)
    if not pairs:
        raise RuntimeError(f"No renderable pairs found under {trajectory_dir}.")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_pairs and max_pairs > 0:
        pairs = pairs[:max_pairs]

    split_by_group = _split_for_pairs(pairs, seed, train_ratio, test_ratio, val_ratio)

    for side in ("A", "B"):
        for split_name in ("train", "test", "validation"):
            (output_dir / side / split_name).mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    manifest_rows: list[dict[str, str]] = []

    progress = tqdm(
        pairs,
        total=len(pairs),
        desc=f"[{topo_root.name}] Render A/mask-B",
        unit="pair",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for pair in progress:
        plan_dir = pair["plan_dir"]
        pair_name = pair["pair_name"]
        sqlite_path = pair["sqlite_path"]
        start_idx = pair["start_idx"]
        end_idx = pair["end_idx"]
        spawn_points = pair.get("spawn_points", [])

        walkable = _load_walkable_from_sqlite(sqlite_path)
        if walkable is None or walkable.is_empty:
            print(f"  [SKIP] No walkable geometry in {sqlite_path.name}")
            skip += 1
            progress.set_postfix(rendered=ok, skipped=skip, refresh=False)
            continue

        paths = _load_trajectories_from_sqlite(sqlite_path)
        if not paths:
            print(f"  [SKIP] No trajectory paths in {sqlite_path.name}")
            skip += 1
            progress.set_postfix(rendered=ok, skipped=skip, refresh=False)
            continue

        bounds = _canvas_bounds(walkable, padding=canvas_padding_m)
        rooms = _load_room_polygons(topo_root, plan_dir)
        spawn_zone = rooms[start_idx] if (start_idx is not None and start_idx < len(rooms)) else None
        exit_zone = rooms[end_idx] if (end_idx is not None and end_idx < len(rooms)) else None

        group_key = extract_group_key(str(pair_name))
        split_name = split_by_group.get(group_key, "train")

        a_img = _render_a_image(
            walkable,
            bounds,
            spawn_zone,
            exit_zone,
            spawn_points,
            target_size_wh,
            preserve_aspect=preserve_aspect,
        )
        b_img = _render_b_trajectory_mask(
            bounds,
            paths,
            target_size_wh,
            line_width=line_width,
            preserve_aspect=preserve_aspect,
        )

        save_rgb(a_img, output_dir / "A" / split_name / pair_name)
        save_rgb(b_img, output_dir / "B" / split_name / pair_name)
        manifest_rows.append(
            {
                "filename": pair_name,
                "group_key": group_key,
                "split": split_name,
            }
        )
        ok += 1
        progress.set_postfix(rendered=ok, skipped=skip, refresh=False)

    progress.close()

    manifest_path = output_dir / "split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "group_key", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[BUILD] {ok} pairs rendered, {skip} skipped -> {output_dir}")
    print(f"[SPLIT] manifest={manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare trajectory-line mask A/B datasets from Geo_scenario source data."
    )
    parser.add_argument("--source_root", type=pathlib.Path, required=True)
    parser.add_argument("--output_root", type=pathlib.Path, required=True)
    parser.add_argument("--topologies", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs", type=int, default=0, help="Debug: limit pairs (0 = all)")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--canvas_padding", type=float, default=0.0)
    parser.add_argument("--line_width", type=int, default=2)
    parser.add_argument(
        "--stretch_to_canvas",
        action="store_true",
        help="Legacy behavior: stretch X/Y independently to fill the square canvas.",
    )
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--output_group", type=str, default="Trajectory_line_mask_dataset")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    target_size_wh = (args.target_size, args.target_size)

    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"source_root  : {source_root}")
    print(f"output_root  : {output_root}")
    print(f"output_group : {args.output_group}")
    print(f"topologies   : {', '.join(args.topologies)}")
    print(f"target_size  : {args.target_size}x{args.target_size} px")
    print(f"line_width   : {args.line_width}")
    print(f"aspect_mode  : {'stretch' if args.stretch_to_canvas else 'preserve'}")
    print(f"split_ratio  : train={args.train}, test={args.test}, val={args.val}")
    print("B mask       : black background, white trajectory line only")
    print("=" * 70)

    rows = []
    for topo in args.topologies:
        topo_root = source_root / topo
        trajectory_dir = topo_root / "trajectory_line_dataset" / "Cleandata_1"
        if not trajectory_dir.exists():
            trajectory_dir = topo_root / "trajectory_line"
        if not trajectory_dir.exists():
            print(f"[MISS] No trajectory source for '{topo}': {trajectory_dir}")
            rows.append({"topology": topo, "status": "missing_source"})
            continue

        dst_dataset = output_root / args.output_group / topo

        try:
            build_trajectory_mask_dataset(
                topo_root=topo_root,
                trajectory_dir=trajectory_dir,
                output_dir=dst_dataset,
                overwrite=args.overwrite,
                seed=args.seed,
                max_pairs=args.max_pairs,
                target_size_wh=target_size_wh,
                canvas_padding_m=args.canvas_padding,
                line_width=args.line_width,
                preserve_aspect=not args.stretch_to_canvas,
                train_ratio=args.train,
                test_ratio=args.test,
                val_ratio=args.val,
            )
        except Exception as exc:
            print(f"[ERROR] {topo}: {exc}")
            rows.append({"topology": topo, "status": f"error: {exc}"})
            continue

        counts = count_split_images(dst_dataset)
        rows.append(
            {
                "topology": topo,
                "status": "prepared",
                "output_dataset": str(dst_dataset),
                **counts,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        print(f"[STAT] {topo} | total_png={counts['total_png']}")

    manifest_path = output_root / "manifest_trajectory_line_mask_dataset.csv"
    fieldnames = [
        "topology",
        "output_dataset",
        "status",
        "A_train",
        "A_test",
        "A_validation",
        "B_train",
        "B_test",
        "B_validation",
        "total_png",
        "dataset_layout",
        "updated_at",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    prepared = sum(1 for r in rows if r.get("status") == "prepared")
    print("-" * 70)
    print(f"[SUMMARY] prepared={prepared} total={len(rows)}")
    print(f"[SUMMARY] manifest={manifest_path}")


if __name__ == "__main__":
    main()

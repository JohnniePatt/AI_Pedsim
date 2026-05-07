"""Rasterize Topo_HouseGAN layouts into walkable-area grids.

The script reads each plan folder under Geo_scenario/Topo_HouseGAN/geo,
unions geo_room.json and geo_corridor.json, and writes:

    <plan_dir>/walkablearea_grid.json

Grid convention:
    - grid rows are stored top-to-bottom for image-like inspection.
    - "1" means the cell center is inside the walkable area.
    - "0" means wall / outside / non-walkable.
    - world_to_grid uses bottom-left origin in metric coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Iterable

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep


DEFAULT_CELL_SIZE_M = 0.125
DEFAULT_PADDING_M = 0.0
DEFAULT_WALL_THICKNESS_SCALE = 1.0


@dataclass(frozen=True)
class GridMeta:
    plan_name: str
    cell_size_m: float
    padding_m: float
    origin_x: float
    origin_y: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width: int
    height: int
    row_order: str
    value_0: str
    value_1: str


def load_json(file_path: pathlib.Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_polygons(json_path: pathlib.Path) -> list[Polygon]:
    if not json_path.exists():
        return []

    raw_items = load_json(json_path)

    polygons: list[Polygon] = []
    for coords in raw_items:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            polygons.append(poly)
    return polygons


def first_existing(parent: pathlib.Path, *names: str) -> pathlib.Path:
    for name in names:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return parent / names[0]


def load_doors(json_path: pathlib.Path) -> list[dict]:
    if not json_path.exists():
        return []
    raw_items = load_json(json_path)
    return raw_items if isinstance(raw_items, list) else []


def load_default_door_width(plan_dir: pathlib.Path) -> float:
    meta_path = plan_dir / "metadata.json"
    if not meta_path.exists():
        return 1.5
    try:
        meta = load_json(meta_path)
        params = meta.get("params", {})
        return float(params.get("door_width", 1.5))
    except Exception:
        return 1.5


def build_door_openings(doors: list[dict], default_door_width: float, wall_thickness_m: float):
    door_boxes = []
    depth = wall_thickness_m * 2.0
    for door in doors:
        pos = door.get("pos")
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        cx, cy = float(pos[0]), float(pos[1])
        door_width = float(door.get("door_width", default_door_width))
        horizontal = bool(door.get("horizontal", False))

        if horizontal:
            half_wx = door_width * 0.5
            half_wy = depth * 0.5
        else:
            half_wx = depth * 0.5
            half_wy = door_width * 0.5
        door_boxes.append(box(cx - half_wx, cy - half_wy, cx + half_wx, cy + half_wy))

    if not door_boxes:
        return None
    return unary_union(door_boxes)


def iter_plan_dirs(geo_root: pathlib.Path, plan_name: str | None) -> Iterable[pathlib.Path]:
    if plan_name:
        plan_dir = geo_root / plan_name
        if not plan_dir.exists():
            raise FileNotFoundError(f"Plan not found: {plan_dir}")
        yield plan_dir
        return

    yield from sorted(
        (p for p in geo_root.iterdir() if p.is_dir() and p.name.startswith("plan_")),
        key=lambda p: p.name,
    )


def build_walkable_grid(
    plan_dir: pathlib.Path,
    cell_size_m: float,
    padding_m: float,
    wall_thickness_m: float,
) -> dict:
    cell_size_m = round(float(cell_size_m), 3)
    padding_m = round(float(padding_m), 3)
    wall_thickness_m = round(float(wall_thickness_m), 3)

    room_polys = load_polygons(first_existing(plan_dir, "geo_room.json", "Geo_room.json"))
    corridor_polys = load_polygons(first_existing(plan_dir, "geo_corridor.json", "Geo_corridor.json"))
    doors = load_doors(first_existing(plan_dir, "geo_door.json", "Geo_door.json"))
    default_door_width = load_default_door_width(plan_dir)
    walkable_parts = room_polys + corridor_polys
    if not walkable_parts:
        raise RuntimeError(f"No room/corridor polygons found in {plan_dir}")

    walkable_union = unary_union(walkable_parts)
    if not walkable_union.is_valid:
        walkable_union = walkable_union.buffer(0)
    if walkable_union.is_empty:
        raise RuntimeError(f"Walkable union is empty in {plan_dir}")

    # Build non-walkable walls from polygon boundaries and reopen door cuts.
    boundary_geoms = [poly.boundary.buffer(wall_thickness_m * 0.5, cap_style=2, join_style=2) for poly in walkable_parts]
    wall_band = unary_union(boundary_geoms)
    door_openings = build_door_openings(doors, default_door_width, wall_thickness_m)
    if door_openings is not None:
        wall_band = wall_band.difference(door_openings)

    walkable = walkable_union.difference(wall_band)
    if not walkable.is_valid:
        walkable = walkable.buffer(0)
    if walkable.is_empty:
        raise RuntimeError(f"Walkable result became empty after wall/door carving in {plan_dir}")

    min_x, min_y, max_x, max_y = walkable_union.bounds
    origin_x = min_x - padding_m
    origin_y = min_y - padding_m
    padded_max_x = max_x + padding_m
    padded_max_y = max_y + padding_m

    width = max(1, int(math.ceil((padded_max_x - origin_x) / cell_size_m)))
    height = max(1, int(math.ceil((padded_max_y - origin_y) / cell_size_m)))

    prepared_walkable = prep(walkable)
    rows_top_to_bottom: list[str] = []
    walkable_count = 0

    for row_top in range(height):
        gy = height - 1 - row_top
        y = origin_y + (gy + 0.5) * cell_size_m
        row_chars: list[str] = []
        for gx in range(width):
            x = origin_x + (gx + 0.5) * cell_size_m
            is_walkable = prepared_walkable.covers(Point(x, y))
            if is_walkable:
                walkable_count += 1
                row_chars.append("1")
            else:
                row_chars.append("0")
        rows_top_to_bottom.append("".join(row_chars))

    meta = GridMeta(
        plan_name=plan_dir.name,
        cell_size_m=cell_size_m,
        padding_m=padding_m,
        origin_x=origin_x,
        origin_y=origin_y,
        min_x=origin_x,
        min_y=origin_y,
        max_x=origin_x + width * cell_size_m,
        max_y=origin_y + height * cell_size_m,
        width=width,
        height=height,
        row_order="top_to_bottom",
        value_0="non_walkable",
        value_1="walkable",
    )

    return {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "meta": asdict(meta),
        "source_files": {
            "rooms": "geo_room.json",
            "corridors": "geo_corridor.json",
        },
        "geometry_summary": {
            "room_count": len(room_polys),
            "corridor_count": len(corridor_polys),
            "door_count": len(doors),
            "wall_thickness_m": wall_thickness_m,
            "walkable_area_m2": walkable.area,
        },
        "grid_summary": {
            "total_cells": width * height,
            "walkable_cells": walkable_count,
            "non_walkable_cells": width * height - walkable_count,
            "walkable_ratio": walkable_count / float(width * height),
        },
        "grid": rows_top_to_bottom,
    }


def write_grid(plan_dir: pathlib.Path, payload: dict, overwrite: bool) -> pathlib.Path:
    out_path = plan_dir / "walkablearea_grid.json"
    if out_path.exists() and not overwrite:
        return out_path

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
    return out_path


def prepare_layouts(
    geo_root: pathlib.Path,
    plan_name: str | None,
    cell_size_m: float,
    padding_m: float,
    wall_thickness_m: float,
    overwrite: bool,
) -> list[dict]:
    cell_size_m = round(float(cell_size_m), 3)
    padding_m = round(float(padding_m), 3)

    if cell_size_m <= 0:
        raise ValueError("--cell-size-m must be greater than 0")
    if padding_m < 0:
        raise ValueError("--padding-m must be >= 0")
    if wall_thickness_m <= 0:
        raise ValueError("--wall-thickness-m must be > 0")
    if not geo_root.exists():
        raise FileNotFoundError(f"Geo root not found: {geo_root}")

    manifest_rows: list[dict] = []
    plan_dirs = list(iter_plan_dirs(geo_root, plan_name))
    print(f"[Grid] Plans found: {len(plan_dirs)}")

    for i, plan_dir in enumerate(plan_dirs, start=1):
        out_path = plan_dir / "walkablearea_grid.json"
        if out_path.exists() and not overwrite:
            print(f"[{i}/{len(plan_dirs)}] skip {plan_dir.name}: output exists")
            manifest_rows.append({"plan_name": plan_dir.name, "status": "skipped", "output": str(out_path)})
            continue

        try:
            payload = build_walkable_grid(plan_dir, cell_size_m, padding_m, wall_thickness_m)
            out_path = write_grid(plan_dir, payload, overwrite=True)
            summary = payload["grid_summary"]
            meta = payload["meta"]
            print(
                f"[{i}/{len(plan_dirs)}] wrote {plan_dir.name}: "
                f"{meta['width']}x{meta['height']} cells, "
                f"walkable={summary['walkable_cells']}"
            )
            manifest_rows.append(
                {
                    "plan_name": plan_dir.name,
                    "status": "written",
                    "output": str(out_path),
                    "width": meta["width"],
                    "height": meta["height"],
                    "walkable_cells": summary["walkable_cells"],
                    "walkable_ratio": summary["walkable_ratio"],
                }
            )
        except Exception as exc:
            print(f"[{i}/{len(plan_dirs)}] failed {plan_dir.name}: {exc}")
            manifest_rows.append({"plan_name": plan_dir.name, "status": "failed", "error": str(exc)})

    return manifest_rows


def default_geo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "Geo_scenario" / "Topo_HouseGAN" / "geo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create walkable-area grid JSON files for Topo_HouseGAN layouts.")
    parser.add_argument("--geo-root", type=pathlib.Path, default=default_geo_root())
    parser.add_argument("--plan-name", type=str, default=None, help="Optional single plan folder name, e.g. plan_137_05aa")
    parser.add_argument("--cell-size-m", type=float, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--padding-m", type=float, default=DEFAULT_PADDING_M)
    parser.add_argument(
        "--wall-thickness-m",
        type=float,
        default=None,
        help="Non-walkable wall thickness. Default is cell_size_m * wall_thickness_scale.",
    )
    parser.add_argument(
        "--wall-thickness-scale",
        type=float,
        default=DEFAULT_WALL_THICKNESS_SCALE,
        help="Multiplier for cell_size_m when wall-thickness-m is not provided.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    wall_thickness_m = args.wall_thickness_m
    if wall_thickness_m is None:
        wall_thickness_m = float(args.cell_size_m) * float(args.wall_thickness_scale)

    rows = prepare_layouts(
        geo_root=args.geo_root.resolve(),
        plan_name=args.plan_name,
        cell_size_m=args.cell_size_m,
        padding_m=args.padding_m,
        wall_thickness_m=wall_thickness_m,
        overwrite=args.overwrite,
    )
    written = sum(1 for row in rows if row["status"] == "written")
    skipped = sum(1 for row in rows if row["status"] == "skipped")
    failed = sum(1 for row in rows if row["status"] == "failed")
    print(f"[Grid] Done. written={written}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()

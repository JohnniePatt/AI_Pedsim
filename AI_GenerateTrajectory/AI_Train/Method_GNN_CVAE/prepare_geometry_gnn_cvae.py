"""
prepare_geometry_gnn_cvae.py
---------------------------
Geometry helpers for Method_GNN_CVAE.
"""

import json

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def load_polygons_from_json(file_path: str) -> list[Polygon]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]


def build_walkable_area(room_json: str, corridor_json: str):
    rooms = load_polygons_from_json(room_json)
    corridors = load_polygons_from_json(corridor_json)
    return unary_union(rooms + corridors)


def compute_meta(walkable_poly, padding: float = 1.0, grid_size: int = 64) -> dict:
    minx, miny, maxx, maxy = walkable_poly.bounds
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    scale = max(maxx - minx, maxy - miny)
    return {
        "min_x": minx,
        "min_y": miny,
        "scale": scale,
        "grid_size": grid_size,
    }


def _rasterize(poly, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path

    h, w = len(ys), len(xs)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    inside = Path(np.asarray(poly.exterior.coords)).contains_points(pts)
    for interior in poly.interiors:
        inside &= ~Path(np.asarray(interior.coords)).contains_points(pts)
    return inside.reshape(h, w)


def create_occupancy_grid(room_json: str, corridor_json: str, grid_size: int = 64, padding: float = 1.0):
    walkable = build_walkable_area(room_json, corridor_json)
    meta = compute_meta(walkable, padding=padding, grid_size=grid_size)

    xs = np.linspace(meta["min_x"], meta["min_x"] + meta["scale"], grid_size)
    ys = np.linspace(meta["min_y"], meta["min_y"] + meta["scale"], grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=bool)

    polys = list(walkable.geoms) if hasattr(walkable, "geoms") else [walkable]
    for poly in polys:
        grid |= _rasterize(poly, xs, ys)

    return grid.astype(np.float32), meta, walkable


def world_to_grid(x: float, y: float, meta: dict) -> tuple[float, float]:
    gx = (x - meta["min_x"]) / meta["scale"]
    gy = (y - meta["min_y"]) / meta["scale"]
    return gx, gy


def grid_to_world(gx: float, gy: float, meta: dict) -> tuple[float, float]:
    x = gx * meta["scale"] + meta["min_x"]
    y = gy * meta["scale"] + meta["min_y"]
    return x, y


def point_is_inside_walkable(x: float, y: float, walkable) -> bool:
    pt = Point(float(x), float(y))
    return bool(walkable.covers(pt))

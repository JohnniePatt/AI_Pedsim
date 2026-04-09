"""
prepare_geometry_transformer.py
────────────────────────────────
Geometry utilities for the GPT-2 trajectory model.

Walkable-area logic matches the JuPedSim simulation setup:
  walkable_area = unary_union(room_polygons + corridor_polygons)

All coordinate helpers use a square bounding box so the
occupancy grid and the trajectory normalisation share the
same coordinate frame.
"""

import json
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_polygons_from_json(file_path: str) -> list[Polygon]:
    """Load a JSON file that contains a list of polygon coordinate lists."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]


def build_walkable_area(room_json: str, corridor_json: str):
    """
    Merge room and corridor polygons into a single walkable-area polygon.
    Identical to the approach used in main_script_sim.py.
    """
    rooms     = load_polygons_from_json(room_json)
    corridors = load_polygons_from_json(corridor_json)
    return unary_union(rooms + corridors)


# ─── Bounding-box meta ─────────────────────────────────────────────────────────

def compute_meta(walkable_poly, padding: float = 1.0, grid_size: int = 64) -> dict:
    """
    Compute a square bounding box around the walkable area with padding.
    The same square is used for the occupancy grid AND for trajectory
    normalisation so that grid coordinates == normalised trajectory coords.

    Returns a dict:
        min_x, min_y  – lower-left corner of the square
        scale         – side length of the square  (max(width, height) + 2*padding)
        grid_size     – grid resolution
    """
    minx, miny, maxx, maxy = walkable_poly.bounds
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding
    scale = max(maxx - minx, maxy - miny)
    return {
        "min_x":     minx,
        "min_y":     miny,
        "scale":     scale,
        "grid_size": grid_size,
    }


# ─── Occupancy grid ────────────────────────────────────────────────────────────

def _rasterize(poly, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Fill *poly* (a single Shapely Polygon) into a boolean grid.
    Uses matplotlib.path for vectorised point-in-polygon (fast).
    """
    from matplotlib.path import Path

    H, W = len(ys), len(xs)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    ext_coords = np.array(poly.exterior.coords)
    inside = Path(ext_coords).contains_points(pts)

    # Subtract holes
    for interior in poly.interiors:
        hole_coords = np.array(interior.coords)
        inside &= ~Path(hole_coords).contains_points(pts)

    return inside.reshape(H, W)


def create_occupancy_grid(
    room_json: str,
    corridor_json: str,
    grid_size: int = 64,
    padding: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Build a [grid_size × grid_size] float32 occupancy grid.
      1.0 = walkable
      0.0 = obstacle / wall

    Returns (grid, meta) where meta is the coordinate-frame descriptor
    produced by compute_meta().
    """
    walkable = build_walkable_area(room_json, corridor_json)
    meta     = compute_meta(walkable, padding=padding, grid_size=grid_size)

    xs = np.linspace(meta["min_x"], meta["min_x"] + meta["scale"], grid_size)
    ys = np.linspace(meta["min_y"], meta["min_y"] + meta["scale"], grid_size)

    # Use bool array for accumulation, convert to float32 at the end
    grid = np.zeros((grid_size, grid_size), dtype=bool)

    # walkable may be a Polygon or MultiPolygon
    if hasattr(walkable, "geoms"):
        polys = list(walkable.geoms)
    else:
        polys = [walkable]

    for p in polys:
        grid |= _rasterize(p, xs, ys)

    return grid.astype(np.float32), meta


# ─── Coordinate helpers ────────────────────────────────────────────────────────

def world_to_grid(x: float, y: float, meta: dict) -> tuple[float, float]:
    """
    World (x, y)  →  normalised [0, 1] grid coordinates.
    This is the normalisation applied to every trajectory point.
    """
    gx = (x - meta["min_x"]) / meta["scale"]
    gy = (y - meta["min_y"]) / meta["scale"]
    return gx, gy


def grid_to_world(gx: float, gy: float, meta: dict) -> tuple[float, float]:
    """Inverse of world_to_grid — recover real-world metres."""
    x = gx * meta["scale"] + meta["min_x"]
    y = gy * meta["scale"] + meta["min_y"]
    return x, y

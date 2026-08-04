"""
prepare_geometry_gnn_cvae2.py
-----------------------------
Geometry helpers for Method_GNN_CVAE2.
Adds spatial graph construction on top of the original occupancy-grid helpers.
"""

import json
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


# ── polygon / walkable helpers (same as GNN_CVAE) ──────────────────────────

def load_polygons_from_json(file_path: str) -> list:
    with open(file_path, "r") as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]


def load_doors_from_json(file_path: str) -> list:
    with open(file_path, "r") as f:
        return json.load(f)


def build_walkable_area(room_json: str, corridor_json: str):
    rooms = load_polygons_from_json(room_json)
    corridors = load_polygons_from_json(corridor_json)
    return unary_union(rooms + corridors)


def compute_meta(walkable_poly, padding: float = 1.0, grid_size: int = 64) -> dict:
    minx, miny, maxx, maxy = walkable_poly.bounds
    minx -= padding; miny -= padding; maxx += padding; maxy += padding
    scale = max(maxx - minx, maxy - miny)
    return {"min_x": minx, "min_y": miny, "scale": scale, "grid_size": grid_size}


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


def world_to_grid(x: float, y: float, meta: dict) -> tuple:
    return (x - meta["min_x"]) / meta["scale"], (y - meta["min_y"]) / meta["scale"]


def grid_to_world(gx: float, gy: float, meta: dict) -> tuple:
    return gx * meta["scale"] + meta["min_x"], gy * meta["scale"] + meta["min_y"]


def point_is_inside_walkable(x: float, y: float, walkable) -> bool:
    return bool(walkable.covers(Point(float(x), float(y))))


# ── spatial graph construction ──────────────────────────────────────────────

def build_spatial_graph(room_json: str, corridor_json: str, door_json: str, meta: dict) -> dict:
    """Build heterogeneous spatial graph from geometry files.

    Node features [N, 8]: [cx_norm, cy_norm, area_norm, width_norm, height_norm,
                            is_room, is_corridor, is_door]
    Edge types: 0 = space↔door, 1 = space↔space (via shared door)
    """
    rooms = load_polygons_from_json(room_json)
    corridors = load_polygons_from_json(corridor_json)
    doors = load_doors_from_json(door_json)

    # Build node list: rooms first, then corridors, then doors
    nodes = []  # (name, geometry, is_room, is_cor, is_door)
    for i, poly in enumerate(rooms):
        nodes.append((f"Room-{i}", poly, 1, 0, 0))
    for i, poly in enumerate(corridors):
        nodes.append((f"Cor-{i}", poly, 0, 1, 0))
    door_start = len(nodes)
    for i, d in enumerate(doors):
        nodes.append((f"Door-{i}", Point(d["pos"][0], d["pos"][1]), 0, 0, 1))

    name_to_idx = {n[0]: idx for idx, n in enumerate(nodes)}
    s = meta["scale"]

    def node_feat(geom, is_room, is_cor, is_door):
        if isinstance(geom, Point):
            cx_w, cy_w = geom.x, geom.y
            area_n, w_n, h_n = 0.0, 0.02 / max(s, 1e-6), 0.02 / max(s, 1e-6)
        else:
            cx_w, cy_w = geom.centroid.x, geom.centroid.y
            b = geom.bounds
            area_n = geom.area / max(s * s, 1e-12)
            w_n = (b[2] - b[0]) / max(s, 1e-6)
            h_n = (b[3] - b[1]) / max(s, 1e-6)
        cx_n, cy_n = world_to_grid(cx_w, cy_w, meta)
        return [cx_n, cy_n, area_n, w_n, h_n, float(is_room), float(is_cor), float(is_door)]

    node_features = np.array([node_feat(n[1], n[2], n[3], n[4]) for n in nodes], dtype=np.float32)
    node_centroids = node_features[:, :2].copy()

    # Edges
    src, dst, etype = [], [], []

    def add_undirected(a, b, t):
        src.extend([a, b]); dst.extend([b, a]); etype.extend([t, t])

    # Type-0: space ↔ door
    for didx, d in enumerate(doors):
        di = door_start + didx
        for rname in d.get("rooms", []):
            if rname in name_to_idx:
                add_undirected(name_to_idx[rname], di, 0)

    # Type-1: space ↔ space (rooms sharing a door)
    connected = set()
    for d in doors:
        valid = [name_to_idx[r] for r in d.get("rooms", []) if r in name_to_idx]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pair = (min(valid[i], valid[j]), max(valid[i], valid[j]))
                if pair not in connected:
                    connected.add(pair)
                    add_undirected(pair[0], pair[1], 1)

    if src:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_type_arr = np.array(etype, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type_arr = np.zeros(0, dtype=np.int64)

    return {
        "node_features": node_features,       # [N, 8]
        "edge_index": edge_index,              # [2, E]
        "edge_type": edge_type_arr,            # [E]
        "node_centroids": node_centroids,      # [N, 2] normalised
        "node_polys": [n[1] for n in nodes],   # for containment check
        "node_names": [n[0] for n in nodes],
    }


def agent_to_node(x_norm: float, y_norm: float, graph: dict, meta: dict) -> int:
    """Map a normalised position to a graph node via containment, then nearest centroid."""
    x_w = x_norm * meta["scale"] + meta["min_x"]
    y_w = y_norm * meta["scale"] + meta["min_y"]
    pt = Point(x_w, y_w)
    for idx, geom in enumerate(graph["node_polys"]):
        if isinstance(geom, Polygon) and geom.contains(pt):
            return idx
    # Fallback: nearest centroid
    dists = np.linalg.norm(graph["node_centroids"] - np.array([x_norm, y_norm], dtype=np.float32), axis=1)
    return int(np.argmin(dists))

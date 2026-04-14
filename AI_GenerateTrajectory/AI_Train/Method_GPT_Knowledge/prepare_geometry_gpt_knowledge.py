"""
prepare_geometry_gpt_knowledge.py
---------------------------------
Geometry and scene loading helpers for Method_GPT_Knowledge.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def infer_case_id(case_dir: str | pathlib.Path) -> str:
    match = re.search(r"case_(.+)$", str(case_dir))
    if not match:
        raise ValueError(f"Unable to infer case id from '{case_dir}'.")
    return match.group(1)


def load_polygons_from_json(file_path: str | pathlib.Path) -> list[Polygon]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Polygon(coords) for coords in data]


def _build_housegan_walkable_area(room_json: str | pathlib.Path, corridor_json: str | pathlib.Path, door_json: str | pathlib.Path):
    rooms = load_polygons_from_json(room_json)
    corridors = load_polygons_from_json(corridor_json)
    with open(door_json, "r", encoding="utf-8") as f:
        doors_data = json.load(f)

    footprint = unary_union(rooms + corridors)
    wall_thickness = 0.2
    door_width = 1.5

    raw_walls = unary_union([poly.exterior.buffer(wall_thickness / 2.0, join_style=2) for poly in rooms + corridors])
    door_cutouts = []
    for door in doors_data:
        px, py = door["pos"]
        cut_depth = wall_thickness * 4.0
        if bool(door.get("horizontal", False)):
            cutout = box(px - door_width / 2.0, py - cut_depth / 2.0, px + door_width / 2.0, py + cut_depth / 2.0)
        else:
            cutout = box(px - cut_depth / 2.0, py - door_width / 2.0, px + cut_depth / 2.0, py + door_width / 2.0)
        door_cutouts.append(cutout)

    if door_cutouts:
        raw_walls = raw_walls.difference(unary_union(door_cutouts))

    return footprint.difference(raw_walls)


def build_walkable_area(room_json: str | pathlib.Path, corridor_json: str | pathlib.Path, door_json: str | pathlib.Path | None = None):
    if door_json is not None and pathlib.Path(door_json).exists():
        return _build_housegan_walkable_area(room_json, corridor_json, door_json)
    rooms = load_polygons_from_json(room_json)
    corridors = load_polygons_from_json(corridor_json)
    return unary_union(rooms + corridors)


def load_spawn_exit(case_dir: str | pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_dir = pathlib.Path(case_dir)
    case_id = infer_case_id(case_dir)
    spawn_path = case_dir / f"Spawn_location_{case_id}.csv"
    exit_path = case_dir / f"Spawn_exit_{case_id}.csv"

    if not spawn_path.exists():
        matches = sorted(case_dir.glob("Spawn_location_*.csv"))
        if len(matches) == 1:
            spawn_path = matches[0]
        else:
            raise FileNotFoundError(f"Missing Spawn_location file in {case_dir}")

    if not exit_path.exists():
        matches = sorted(case_dir.glob("Spawn_exit_*.csv"))
        if len(matches) == 1:
            exit_path = matches[0]
        else:
            raise FileNotFoundError(f"Missing Spawn_exit file in {case_dir}")

    spawn_df = pd.read_csv(spawn_path)
    exit_df = pd.read_csv(exit_path)
    return spawn_df, exit_df


def load_trajectory(case_dir: str | pathlib.Path) -> pd.DataFrame:
    case_dir = pathlib.Path(case_dir)
    files = sorted(case_dir.glob("*_trajectory_data.parquet"))
    if not files:
        raise FileNotFoundError(f"No trajectory parquet found in {case_dir}.")
    df = pd.read_parquet(files[0])
    return df.sort_values(["id", "frame"]).reset_index(drop=True)


def load_scene(case_dir: str | pathlib.Path) -> dict:
    case_dir = pathlib.Path(case_dir)
    spawn_df, exit_df = load_spawn_exit(case_dir)
    door_path = case_dir / "Geo_door.json"
    walkable = build_walkable_area(case_dir / "Geo_room.json", case_dir / "Geo_corridor.json", door_path if door_path.exists() else None)

    spawn_poly = None
    exit_poly = None
    for _, row in exit_df.iterrows():
        geom = wkt.loads(row["area"])
        if row["type"] == "spawning_area":
            spawn_poly = geom
        elif row["type"] == "exit_area":
            exit_poly = geom

    if spawn_poly is None or exit_poly is None:
        raise ValueError(f"Missing spawning_area or exit_area in {case_dir}.")

    return {
        "case_id": infer_case_id(case_dir),
        "case_dir": str(case_dir),
        "spawn_df": spawn_df,
        "exit_df": exit_df,
        "spawn_polygon": spawn_poly,
        "exit_polygon": exit_poly,
        "walkable": walkable,
    }


def point_is_inside_walkable(x: float, y: float, walkable) -> bool:
    return bool(walkable.covers(Point(float(x), float(y))))


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / norm


def scene_centers(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    spawn_center = np.array(scene["spawn_polygon"].centroid.coords[0], dtype=np.float64)
    exit_center = np.array(scene["exit_polygon"].centroid.coords[0], dtype=np.float64)
    return spawn_center, exit_center


def local_frame(spawn_center: np.ndarray, exit_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = _safe_unit(exit_center - spawn_center)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
    return forward, lateral


def to_local(points_xy: np.ndarray, origin: np.ndarray, forward: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    shifted = points_xy - origin[None, :]
    return np.column_stack([shifted @ forward, shifted @ lateral])


def from_local(points_uv: np.ndarray, origin: np.ndarray, forward: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    return origin[None, :] + points_uv[:, [0]] * forward[None, :] + points_uv[:, [1]] * lateral[None, :]


def build_similarity_transform(src_spawn: np.ndarray, src_exit: np.ndarray, dst_spawn: np.ndarray, dst_exit: np.ndarray):
    src_forward, src_lateral = local_frame(src_spawn, src_exit)
    dst_forward, dst_lateral = local_frame(dst_spawn, dst_exit)
    src_dist = max(float(np.linalg.norm(src_exit - src_spawn)), 1e-6)
    dst_dist = max(float(np.linalg.norm(dst_exit - dst_spawn)), 1e-6)
    scale = dst_dist / src_dist

    def transform(points_xy: np.ndarray) -> np.ndarray:
        local = to_local(points_xy, src_spawn, src_forward, src_lateral)
        local[:, 0] *= scale
        local[:, 1] *= scale
        return from_local(local, dst_spawn, dst_forward, dst_lateral)

    return transform


def extract_agent_paths(df: pd.DataFrame) -> dict[int, dict]:
    paths: dict[int, dict] = {}
    for agent_id, group in df.groupby("id"):
        arr = group[["pos_x", "pos_y"]].to_numpy(dtype=np.float64)
        frames = group["frame"].to_numpy(dtype=np.int64)
        if len(arr) == 0:
            continue
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        path_length = float(deltas.sum()) if len(deltas) else 0.0
        paths[int(agent_id)] = {
            "points": arr,
            "frames": frames,
            "duration_frames": int(len(arr)),
            "start": arr[0],
            "end": arr[-1],
            "path_length": path_length,
        }
    return paths


def resample_path(points_xy: np.ndarray, n_steps: int) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((n_steps, 2), dtype=np.float64)
    if len(points_xy) == 1:
        return np.repeat(points_xy, n_steps, axis=0)
    if n_steps <= 1:
        return points_xy[[0]].copy()

    src = np.linspace(0.0, 1.0, len(points_xy))
    dst = np.linspace(0.0, 1.0, n_steps)
    x = np.interp(dst, src, points_xy[:, 0])
    y = np.interp(dst, src, points_xy[:, 1])
    return np.column_stack([x, y])


def scene_feature_row(case_dir: str | pathlib.Path, require_trajectory: bool = True) -> dict:
    scene = load_scene(case_dir)
    spawn_center, exit_center = scene_centers(scene)
    has_trajectory = True
    durations = []
    path_lengths = []
    try:
        traj = load_trajectory(case_dir)
        agent_paths = extract_agent_paths(traj)
        durations = [v["duration_frames"] for v in agent_paths.values()]
        path_lengths = [v["path_length"] for v in agent_paths.values()]
    except FileNotFoundError:
        has_trajectory = False
        if require_trajectory:
            raise

    minx, miny, maxx, maxy = scene["walkable"].bounds
    start_goal = exit_center - spawn_center

    return {
        "case_id": str(scene["case_id"]),
        "case_dir": str(case_dir),
        "n_agents": int(len(scene["spawn_df"])),
        "spawn_center_x": float(spawn_center[0]),
        "spawn_center_y": float(spawn_center[1]),
        "exit_center_x": float(exit_center[0]),
        "exit_center_y": float(exit_center[1]),
        "goal_dx": float(start_goal[0]),
        "goal_dy": float(start_goal[1]),
        "goal_distance": float(np.linalg.norm(start_goal)),
        "walkable_area": float(scene["walkable"].area),
        "walkable_min_x": float(minx),
        "walkable_min_y": float(miny),
        "walkable_max_x": float(maxx),
        "walkable_max_y": float(maxy),
        "has_trajectory": has_trajectory,
        "mean_duration_frames": float(np.mean(durations)) if durations else 0.0,
        "mean_path_length": float(np.mean(path_lengths)) if path_lengths else 0.0,
    }

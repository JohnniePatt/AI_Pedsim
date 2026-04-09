"""
generate_gpt_knowledge.py
-------------------------
Retrieve similar train cases and transfer their trajectories into a target scene.
"""

from __future__ import annotations

import argparse
import heapq
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from prepare_geometry_gpt_knowledge import (
    build_similarity_transform,
    extract_agent_paths,
    load_scene,
    load_trajectory,
    local_frame,
    point_is_inside_walkable,
    resample_path,
    scene_centers,
    scene_feature_row,
    to_local,
)


@dataclass
class RetrievedCase:
    case_id: str
    case_dir: str
    score: float


def load_config(config_path: str | pathlib.Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_config(config_path: str | pathlib.Path, raw_path: str) -> pathlib.Path:
    config_path = pathlib.Path(config_path).resolve()
    path = pathlib.Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_scene_index(knowledge_dir: pathlib.Path) -> pd.DataFrame:
    scene_index_path = knowledge_dir / "scene_index.parquet"
    if not scene_index_path.exists():
        raise FileNotFoundError(f"Missing scene_index.parquet in {knowledge_dir}. Run build_knowledge.py first.")
    return pd.read_parquet(scene_index_path)


def scene_distance(target: dict, candidate_row: pd.Series, weights: dict) -> float:
    goal_dist_term = abs(target["goal_distance"] - float(candidate_row["goal_distance"]))
    goal_vec_term = np.hypot(
        target["goal_dx"] - float(candidate_row["goal_dx"]),
        target["goal_dy"] - float(candidate_row["goal_dy"]),
    )
    walkable_term = abs(target["walkable_area"] - float(candidate_row["walkable_area"]))
    agents_term = abs(target["n_agents"] - int(candidate_row["n_agents"]))
    duration_term = abs(target["mean_duration_frames"] - float(candidate_row["mean_duration_frames"]))
    duration_weight = weights.get("duration", 0.01)
    if not bool(target.get("has_trajectory", True)):
        duration_weight = 0.0
    return (
        weights.get("goal_distance", 1.0) * goal_dist_term
        + weights.get("goal_vector", 1.0) * goal_vec_term
        + weights.get("walkable_area", 0.05) * walkable_term
        + weights.get("n_agents", 1.0) * agents_term
        + duration_weight * duration_term
    )


def retrieve_cases(target_case_dir: pathlib.Path, scene_index: pd.DataFrame, cfg: dict) -> list[RetrievedCase]:
    target = scene_feature_row(target_case_dir, require_trajectory=False)
    weights = cfg.get("retrieval_weights", {})
    candidates = []
    target_id = str(target["case_id"])

    for _, row in scene_index.iterrows():
        if str(row["case_id"]) == target_id and pathlib.Path(row["case_dir"]).resolve() == target_case_dir.resolve():
            continue
        score = scene_distance(target, row, weights)
        candidates.append(RetrievedCase(case_id=str(row["case_id"]), case_dir=str(row["case_dir"]), score=float(score)))

    candidates.sort(key=lambda x: x.score)
    return candidates[: int(cfg.get("top_k_cases", 5))]


def _retrieval_blend_weights(retrieved_cases: list[RetrievedCase], cfg: dict) -> np.ndarray:
    if not retrieved_cases:
        return np.zeros((0,), dtype=np.float64)
    temperatures = float(cfg.get("blend_temperature", 1.0))
    temperatures = max(temperatures, 1e-6)
    scores = np.array([item.score for item in retrieved_cases], dtype=np.float64)
    shifted = scores - scores.min()
    raw = np.exp(-shifted / temperatures)
    raw_sum = raw.sum()
    if raw_sum <= 0:
        return np.full_like(raw, 1.0 / len(raw))
    return raw / raw_sum


def _greedy_match(target_local_starts: np.ndarray, source_local_starts: np.ndarray) -> list[int]:
    remaining = set(range(len(source_local_starts)))
    assignment = []
    for tpt in target_local_starts:
        best_idx = None
        best_dist = None
        for sidx in remaining:
            dist = float(np.linalg.norm(tpt - source_local_starts[sidx]))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = sidx
        if best_idx is None:
            best_idx = 0
        assignment.append(best_idx)
        if best_idx in remaining:
            remaining.remove(best_idx)
    return assignment


def _grid_index(point_xy: np.ndarray, minx: float, miny: float, step: float) -> tuple[int, int]:
    return (int(round((float(point_xy[0]) - minx) / step)), int(round((float(point_xy[1]) - miny) / step)))


def _grid_point(ix: int, iy: int, minx: float, miny: float, step: float) -> np.ndarray:
    return np.array([minx + ix * step, miny + iy * step], dtype=np.float64)


def _nearest_walkable_grid(point_xy: np.ndarray, walkable, minx: float, miny: float, step: float, max_radius: int = 12) -> tuple[int, int]:
    base_ix, base_iy = _grid_index(point_xy, minx, miny, step)
    best = None
    best_dist = None
    for radius in range(max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                cand = _grid_point(base_ix + dx, base_iy + dy, minx, miny, step)
                if not point_is_inside_walkable(cand[0], cand[1], walkable):
                    continue
                dist = float(np.linalg.norm(cand - point_xy))
                if best_dist is None or dist < best_dist:
                    best = (base_ix + dx, base_iy + dy)
                    best_dist = dist
        if best is not None:
            return best
    return (base_ix, base_iy)


def _build_route_skeleton(walkable, start_xy: np.ndarray, end_xy: np.ndarray, step: float) -> np.ndarray:
    minx, miny, maxx, maxy = walkable.bounds
    start_idx = _nearest_walkable_grid(start_xy, walkable, minx, miny, step)
    end_idx = _nearest_walkable_grid(end_xy, walkable, minx, miny, step)

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start_idx))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start_idx: 0.0}

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    def heuristic(node: tuple[int, int]) -> float:
        dx = node[0] - end_idx[0]
        dy = node[1] - end_idx[1]
        return float(np.hypot(dx, dy))

    visited = set()
    max_nodes = int(((maxx - minx) / step + 3) * ((maxy - miny) / step + 3))

    while open_heap and len(visited) < max_nodes:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        if current == end_idx:
            break

        current_pt = _grid_point(current[0], current[1], minx, miny, step)
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            nxt_pt = _grid_point(nxt[0], nxt[1], minx, miny, step)
            if not point_is_inside_walkable(nxt_pt[0], nxt_pt[1], walkable):
                continue
            segment = LineString([tuple(current_pt.tolist()), tuple(nxt_pt.tolist())])
            if not walkable.covers(segment):
                continue
            move_cost = float(np.hypot(dx, dy))
            tentative = g_score[current] + move_cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))

    if end_idx not in came_from and end_idx != start_idx:
        return np.vstack([start_xy, end_xy])

    path_nodes = [end_idx]
    while path_nodes[-1] != start_idx:
        path_nodes.append(came_from[path_nodes[-1]])
    path_nodes.reverse()
    path = np.vstack([_grid_point(ix, iy, minx, miny, step) for ix, iy in path_nodes])
    path[0] = np.asarray(start_xy, dtype=np.float64)
    path[-1] = np.asarray(end_xy, dtype=np.float64)
    simplified = np.asarray(LineString(path).simplify(step * 0.5, preserve_topology=False).coords, dtype=np.float64)
    if len(simplified) < 2:
        simplified = path
    simplified[0] = np.asarray(start_xy, dtype=np.float64)
    simplified[-1] = np.asarray(end_xy, dtype=np.float64)
    return simplified


def _route_normals(route_points: np.ndarray) -> np.ndarray:
    if len(route_points) < 2:
        return np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (len(route_points), 1))
    tangents = np.zeros_like(route_points)
    tangents[0] = route_points[1] - route_points[0]
    tangents[-1] = route_points[-1] - route_points[-2]
    if len(route_points) > 2:
        tangents[1:-1] = route_points[2:] - route_points[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    tangents = tangents / norms
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def _candidate_lateral_offsets(candidate_path: np.ndarray, route_resampled: np.ndarray, route_normals: np.ndarray) -> np.ndarray:
    if len(candidate_path) == 0:
        return np.zeros((len(route_resampled),), dtype=np.float64)
    candidate_resampled = resample_path(np.asarray(candidate_path, dtype=np.float64), len(route_resampled))
    delta = candidate_resampled - route_resampled
    return np.sum(delta * route_normals, axis=1)


def _blend_lateral_offsets(offset_series: list[np.ndarray], weights: np.ndarray, blend_strength: float) -> np.ndarray:
    if not offset_series:
        return np.zeros((0,), dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.full((len(offset_series),), 1.0 / len(offset_series), dtype=np.float64)
    else:
        weights = weights / total
    base = offset_series[0]
    avg = np.zeros_like(base)
    for offsets, weight in zip(offset_series, weights.tolist()):
        avg += float(weight) * offsets
    blend_strength = float(np.clip(blend_strength, 0.0, 1.0))
    return base + blend_strength * (avg - base)


def _smooth_offset_series(offsets: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or len(offsets) < 3:
        return np.asarray(offsets, dtype=np.float64)
    smoothed = np.asarray(offsets, dtype=np.float64).copy()
    for _ in range(passes):
        nxt = smoothed.copy()
        nxt[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = nxt
    return smoothed


def _apply_offset_taper(offsets: np.ndarray, end_zero_ratio: float) -> np.ndarray:
    tapered = np.asarray(offsets, dtype=np.float64).copy()
    if len(tapered) == 0:
        return tapered
    progress = np.linspace(0.0, 1.0, len(tapered), dtype=np.float64)
    end_zero_ratio = float(np.clip(end_zero_ratio, 0.05, 0.95))
    taper = np.ones_like(progress)
    mask = progress >= end_zero_ratio
    if np.any(mask):
        taper[mask] = np.clip((1.0 - progress[mask]) / max(1.0 - end_zero_ratio, 1e-6), 0.0, 1.0)
    return tapered * taper


def _limit_offset_delta(offsets: np.ndarray, max_delta: float) -> np.ndarray:
    limited = np.asarray(offsets, dtype=np.float64).copy()
    if len(limited) < 2 or max_delta <= 0:
        return limited
    for idx in range(1, len(limited)):
        delta = limited[idx] - limited[idx - 1]
        if delta > max_delta:
            limited[idx] = limited[idx - 1] + max_delta
        elif delta < -max_delta:
            limited[idx] = limited[idx - 1] - max_delta
    return limited


def _project_outside_points(blended_points: np.ndarray, walkable) -> np.ndarray:
    fixed = np.asarray(blended_points, dtype=np.float64).copy()
    boundary = walkable.boundary
    for idx, (x, y) in enumerate(fixed.tolist()):
        if point_is_inside_walkable(x, y, walkable):
            continue
        nearest = boundary.interpolate(boundary.project(Point(float(x), float(y))))
        fixed[idx] = np.array(nearest.coords[0], dtype=np.float64)
    return fixed


def _smooth_path(points_xy: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or len(points_xy) < 3:
        return points_xy
    smoothed = np.asarray(points_xy, dtype=np.float64).copy()
    for _ in range(passes):
        nxt = smoothed.copy()
        nxt[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed = nxt
    return smoothed


def _align_path_start(points_xy: np.ndarray, target_start: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy
    aligned = np.asarray(points_xy, dtype=np.float64).copy()
    delta = np.asarray(target_start, dtype=np.float64) - aligned[0]
    fade = np.linspace(1.0, 0.0, len(aligned), dtype=np.float64)[:, None]
    aligned += fade * delta[None, :]
    aligned[0] = np.asarray(target_start, dtype=np.float64)
    return aligned


def _enforce_agent_separation(predictions: dict[int, dict], walkable, cfg: dict) -> dict[int, dict]:
    min_sep = float(cfg.get("min_agent_separation_m", 0.45))
    iterations = int(cfg.get("separation_iterations", 2))
    if min_sep <= 0 or iterations <= 0 or len(predictions) < 2:
        return predictions

    agent_ids = sorted(predictions)
    max_steps = max(len(predictions[agent_id]["points"]) for agent_id in agent_ids)
    for _ in range(iterations):
        for t in range(max_steps):
            points_at_t = []
            for agent_id in agent_ids:
                pts = predictions[agent_id]["points"]
                if t < len(pts):
                    points_at_t.append((agent_id, pts[t].copy()))
            for idx in range(len(points_at_t)):
                aid_i, p_i = points_at_t[idx]
                for jdx in range(idx + 1, len(points_at_t)):
                    aid_j, p_j = points_at_t[jdx]
                    vec = p_j - p_i
                    dist = float(np.linalg.norm(vec))
                    if dist >= min_sep:
                        continue
                    direction = vec / max(dist, 1e-6) if dist > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
                    push = 0.5 * (min_sep - dist) * direction
                    predictions[aid_i]["points"][t] = predictions[aid_i]["points"][t] - push
                    predictions[aid_j]["points"][t] = predictions[aid_j]["points"][t] + push
        for agent_id in agent_ids:
            pts = predictions[agent_id]["points"]
            pts = _project_outside_points(pts, walkable)
            if len(pts):
                pts[0] = predictions[agent_id]["start_point"]
            predictions[agent_id]["points"] = pts
    return predictions


def _plot_route_skeleton(scene: dict, route_skeleton: np.ndarray, out_path: pathlib.Path):
    import matplotlib.pyplot as plt
    from visual_gpt_knowledge import plot_polygon

    spawn_center, exit_center = scene_centers(scene)
    walkable = scene["walkable"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    plot_polygon(ax, walkable, facecolor="#dfeedd", edgecolor="#4c6a55", alpha=0.85, linewidth=1.1)
    ax.plot(route_skeleton[:, 0], route_skeleton[:, 1], color="#d62728", linewidth=3.0, label="Route Skeleton")
    ax.scatter(spawn_center[0], spawn_center[1], c="#1f77b4", s=90, marker="o", label="Spawn Center", zorder=5)
    ax.scatter(exit_center[0], exit_center[1], c="#f2c94c", s=180, marker="*", edgecolors="black", linewidths=0.6, label="Exit Center", zorder=6)
    ax.set_title(f"Route Skeleton | case {scene['case_id']}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    bounds = walkable.bounds
    pad_x = max((bounds[2] - bounds[0]) * 0.05, 1.0)
    pad_y = max((bounds[3] - bounds[1]) * 0.05, 1.0)
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _prediction_rows_from_predictions(predictions: dict[int, dict]) -> list[dict]:
    rows = []
    for agent_id in sorted(predictions):
        pred_points = predictions[agent_id]["points"]
        pred_frames = predictions[agent_id]["frames"]
        primary_source_case = predictions[agent_id]["source_case_id"]
        primary_source_agent = predictions[agent_id]["source_agent_id"]
        for frame, (x, y) in zip(pred_frames.tolist(), pred_points.tolist()):
            rows.append(
                {
                    "frame": int(frame),
                    "id": int(agent_id),
                    "pos_x": float(x),
                    "pos_y": float(y),
                    "source_case_id": str(primary_source_case),
                    "source_agent_id": int(primary_source_agent),
                }
            )
    return rows


def _compute_oob_rate_from_rows(rows: list[dict], walkable) -> float:
    if not rows:
        return 0.0
    outside = 0
    for item in rows:
        outside += 0 if point_is_inside_walkable(item["pos_x"], item["pos_y"], walkable) else 1
    return outside / max(len(rows), 1)


def _compute_path_quality(predictions: dict[int, dict], route_resampled: np.ndarray, cfg: dict) -> dict:
    if not predictions:
        return {"max_segment_length": 0.0, "max_route_deviation": 0.0, "quality_fail": False}

    route_step = np.linalg.norm(np.diff(route_resampled, axis=0), axis=1)
    typical_step = float(np.median(route_step)) if len(route_step) else 0.0
    max_seg = 0.0
    max_dev = 0.0
    for agent_id in predictions:
        pts = np.asarray(predictions[agent_id]["points"], dtype=np.float64)
        if len(pts) >= 2:
            max_seg = max(max_seg, float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        if len(pts) == len(route_resampled):
            max_dev = max(max_dev, float(np.max(np.linalg.norm(pts - route_resampled, axis=1))))

    seg_factor = float(cfg.get("quality_max_segment_factor", 2.5))
    dev_limit = float(cfg.get("quality_max_route_deviation_m", 2.0))
    seg_limit = seg_factor * max(typical_step, 1e-6)
    quality_fail = (max_seg > seg_limit) or (max_dev > dev_limit)
    return {
        "typical_route_step": typical_step,
        "max_segment_length": max_seg,
        "max_route_deviation": max_dev,
        "segment_limit": seg_limit,
        "deviation_limit": dev_limit,
        "quality_fail": bool(quality_fail),
    }


def build_planner_prompt(target_scene: dict, retrieved_cases: list[RetrievedCase]) -> str:
    spawn_center, exit_center = scene_centers(target_scene)
    lines = [
        "You are a trajectory planning assistant for pedestrian simulations.",
        "Target scene summary:",
        f"- case_id: {target_scene['case_id']}",
        f"- n_agents: {len(target_scene['spawn_df'])}",
        f"- spawn_center: ({spawn_center[0]:.3f}, {spawn_center[1]:.3f})",
        f"- exit_center: ({exit_center[0]:.3f}, {exit_center[1]:.3f})",
        f"- walkable_area: {target_scene['walkable'].area:.3f}",
        "",
        "Retrieved reference cases:",
    ]
    for idx, item in enumerate(retrieved_cases, start=1):
        lines.append(f"{idx}. case_{item.case_id} | score={item.score:.4f} | dir={item.case_dir}")
    lines.extend(
        [
            "",
            "Task:",
            "- infer a plausible multi-agent movement policy from the retrieved cases",
            "- keep agents inside walkable space",
            "- move from spawn area toward exit area",
            "- preserve the general crowd flow shape seen in the references",
        ]
    )
    return "\n".join(lines)


def _load_candidate_case_bundle(case_dir: str | pathlib.Path) -> dict[str, Any]:
    scene = load_scene(case_dir)
    source_df = load_trajectory(case_dir)
    source_paths = extract_agent_paths(source_df)
    source_ids = list(sorted(source_paths))
    src_spawn_center, src_exit_center = scene_centers(scene)
    src_forward, src_lateral = local_frame(src_spawn_center, src_exit_center)
    source_start_points = np.stack([source_paths[aid]["start"] for aid in source_ids], axis=0)
    source_local = to_local(source_start_points, src_spawn_center, src_forward, src_lateral)
    return {
        "scene": scene,
        "paths": source_paths,
        "ids": source_ids,
        "spawn_center": src_spawn_center,
        "exit_center": src_exit_center,
        "source_local": source_local,
    }


def _generate_predictions_once(
    target_scene: dict,
    target_spawn: pd.DataFrame,
    target_local: np.ndarray,
    candidate_bundles: list[dict[str, Any]],
    route_resampled: np.ndarray,
    route_normals: np.ndarray,
    cfg: dict,
    blend_strength: float,
    max_lateral_offset: float,
    use_blend: bool,
) -> tuple[dict[int, dict], list[dict], float, dict]:
    common_steps = len(route_resampled)
    smooth_passes = int(cfg.get("smoothing_passes", 1))
    offset_smooth_passes = int(cfg.get("offset_smoothing_passes", 2))
    offset_end_zero_ratio = float(cfg.get("offset_end_zero_ratio", 0.75))
    max_offset_delta = float(cfg.get("max_offset_delta_m", 0.25))
    predictions = {}

    for bundle in candidate_bundles:
        bundle["assignments"] = _greedy_match(target_local, bundle["source_local"])

    for target_idx, (target_row, target_local_start) in enumerate(zip(target_spawn.itertuples(index=False), target_local)):
        candidate_offsets = []
        candidate_sources = []
        for bundle in candidate_bundles:
            assign_idx = bundle["assignments"][target_idx]
            source_agent_id = bundle["ids"][assign_idx]
            transform = build_similarity_transform(bundle["spawn_center"], bundle["exit_center"], route_resampled[0], route_resampled[-1])
            pred_points = transform(bundle["paths"][source_agent_id]["points"])
            candidate_offsets.append(_candidate_lateral_offsets(pred_points, route_resampled, route_normals))
            candidate_sources.append(
                {
                    "case_id": str(bundle["meta"].case_id),
                    "agent_id": int(source_agent_id),
                    "weight": float(bundle["weight"]),
                    "score": float(bundle["meta"].score),
                }
            )

        target_start = np.array([float(target_row.pos_x), float(target_row.pos_y)], dtype=np.float64)
        weights = np.array([item["weight"] for item in candidate_sources], dtype=np.float64)
        if use_blend:
            blended_offsets = _blend_lateral_offsets(candidate_offsets, weights, blend_strength)
        else:
            blended_offsets = candidate_offsets[0] if candidate_offsets else np.zeros((common_steps,), dtype=np.float64)

        spawn_lateral_bias = float(target_local_start[1]) if use_blend else 0.0
        progress = np.linspace(0.0, 1.0, common_steps, dtype=np.float64)
        spawn_bias_decay = (1.0 - progress) ** 1.5
        blended_offsets = blended_offsets + spawn_bias_decay * spawn_lateral_bias
        blended_offsets = _smooth_offset_series(blended_offsets, offset_smooth_passes)
        blended_offsets = _limit_offset_delta(blended_offsets, max_offset_delta)
        blended_offsets = _apply_offset_taper(blended_offsets, offset_end_zero_ratio)
        blended_offsets = np.clip(blended_offsets, -max_lateral_offset, max_lateral_offset)

        blended_points = route_resampled + blended_offsets[:, None] * route_normals
        blended_points = _align_path_start(blended_points, target_start)
        blended_points = _smooth_path(blended_points, smooth_passes)
        blended_points = _align_path_start(blended_points, target_start)
        blended_points[-1] = route_resampled[-1]
        blended_points = _project_outside_points(blended_points, target_scene["walkable"])
        pred_frames = np.arange(len(blended_points), dtype=np.int64)
        primary_source = candidate_sources[0] if candidate_sources else {"case_id": "", "agent_id": -1}
        predictions[int(target_row.id)] = {
            "points": blended_points,
            "frames": pred_frames,
            "start_point": target_start,
            "source_case_id": str(primary_source["case_id"]),
            "source_agent_id": int(primary_source["agent_id"]),
            "blended_sources": candidate_sources,
        }

    predictions = _enforce_agent_separation(predictions, target_scene["walkable"], cfg)
    rows = _prediction_rows_from_predictions(predictions)
    oob_rate = _compute_oob_rate_from_rows(rows, target_scene["walkable"])
    quality = _compute_path_quality(predictions, route_resampled, cfg)
    return predictions, rows, oob_rate, quality


def generate_case_prediction(target_case_dir: str | pathlib.Path, knowledge_dir: str | pathlib.Path, cfg: dict) -> dict:
    target_case_dir = pathlib.Path(target_case_dir).resolve()
    knowledge_dir = pathlib.Path(knowledge_dir).resolve()
    scene_index = load_scene_index(knowledge_dir)
    target_scene = load_scene(target_case_dir)
    retrieved_cases = retrieve_cases(target_case_dir, scene_index, cfg)
    if not retrieved_cases:
        raise RuntimeError("No retrieved reference cases were found.")

    target_spawn = target_scene["spawn_df"].sort_values("id").reset_index(drop=True)
    dst_spawn_center, dst_exit_center = scene_centers(target_scene)
    dst_forward, dst_lateral = local_frame(dst_spawn_center, dst_exit_center)
    target_start_points = target_spawn[["pos_x", "pos_y"]].to_numpy(dtype=np.float64)
    target_local = to_local(target_start_points, dst_spawn_center, dst_forward, dst_lateral)
    route_grid_step = float(cfg.get("route_grid_step_m", 0.4))
    route_skeleton = _build_route_skeleton(target_scene["walkable"], dst_spawn_center, dst_exit_center, route_grid_step)

    candidate_limit = int(cfg.get("blend_case_count", min(len(retrieved_cases), 3)))
    candidate_limit = max(1, min(candidate_limit, len(retrieved_cases)))
    blend_cases = retrieved_cases[:candidate_limit]
    blend_weights = _retrieval_blend_weights(blend_cases, cfg)
    candidate_bundles = []
    for idx, item in enumerate(blend_cases):
        try:
            candidate_bundles.append(
                {
                    "meta": item,
                    "weight": float(blend_weights[idx]),
                    **_load_candidate_case_bundle(item.case_dir),
                }
            )
        except FileNotFoundError:
            continue
    if not candidate_bundles:
        raise RuntimeError("All retrieved reference cases were missing trajectory data.")

    common_steps = int(cfg.get("blend_steps", 80))
    common_steps = max(common_steps, 8)
    route_resampled = resample_path(route_skeleton, common_steps)
    route_normals = _route_normals(route_resampled)
    max_attempts = int(cfg.get("max_generation_attempts", 4))
    blend_strength_base = float(cfg.get("blend_strength", 0.35))
    max_lateral_offset_base = float(cfg.get("max_lateral_offset_m", 1.5))
    retry_decay = float(cfg.get("retry_offset_decay", 0.5))
    fallback_to_route_only = bool(cfg.get("fallback_to_route_only", True))
    predictions = {}
    rows = []
    oob_rate = 1.0
    quality = {"quality_fail": True}
    attempt_history = []

    for attempt_idx in range(max_attempts):
        decay = retry_decay ** attempt_idx
        use_blend = True
        if fallback_to_route_only and attempt_idx == max_attempts - 1:
            use_blend = False
            decay = 0.0

        predictions, rows, oob_rate, quality = _generate_predictions_once(
            target_scene=target_scene,
            target_spawn=target_spawn,
            target_local=target_local,
            candidate_bundles=candidate_bundles,
            route_resampled=route_resampled,
            route_normals=route_normals,
            cfg=cfg,
            blend_strength=blend_strength_base * decay,
            max_lateral_offset=max_lateral_offset_base * max(decay, 0.15),
            use_blend=use_blend,
        )
        attempt_history.append(
            {
                "attempt": attempt_idx + 1,
                "used_blend": use_blend,
                "blend_strength": float(blend_strength_base * decay),
                "max_lateral_offset_m": float(max_lateral_offset_base * max(decay, 0.15)),
                "out_of_bounds_rate": float(oob_rate),
                "quality_fail": bool(quality.get("quality_fail", False)),
                "max_segment_length": float(quality.get("max_segment_length", 0.0)),
                "max_route_deviation": float(quality.get("max_route_deviation", 0.0)),
            }
        )
        if oob_rate <= 0.0 and not bool(quality.get("quality_fail", False)):
            break

    return {
        "case_id": str(target_scene["case_id"]),
        "case_dir": str(target_case_dir),
        "retrieved_cases": [item.__dict__ for item in retrieved_cases],
        "blend_cases": [item["meta"].__dict__ | {"weight": item["weight"]} for item in candidate_bundles],
        "route_skeleton": route_skeleton.tolist(),
        "generation_attempts": attempt_history,
        "quality_summary": quality,
        "planner_prompt": build_planner_prompt(target_scene, retrieved_cases),
        "predictions": predictions,
        "prediction_df": pd.DataFrame(rows),
        "out_of_bounds_rate": float(oob_rate),
    }


def save_generation(result: dict, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    result["prediction_df"].to_parquet(output_dir / f"AI_pred_{result['case_id']}.parquet", index=False)
    route_skeleton = np.asarray(result.get("route_skeleton", []), dtype=np.float64)
    if len(route_skeleton):
        with open(output_dir / "route_skeleton.json", "w", encoding="utf-8") as f:
            json.dump({"case_id": result["case_id"], "route_skeleton": result["route_skeleton"]}, f, indent=2)
        scene = load_scene(result["case_dir"])
        _plot_route_skeleton(scene, route_skeleton, output_dir / "route_skeleton.png")
    with open(output_dir / "retrieved_cases.json", "w", encoding="utf-8") as f:
        json.dump(result["retrieved_cases"], f, indent=2)
    with open(output_dir / "planner_prompt.txt", "w", encoding="utf-8") as f:
        f.write(result["planner_prompt"])
    with open(output_dir / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": result["case_id"],
                "case_dir": result["case_dir"],
                "out_of_bounds_rate": result["out_of_bounds_rate"],
                "retrieved_cases": result["retrieved_cases"],
                "blend_cases": result.get("blend_cases", []),
                "route_skeleton_points": len(result.get("route_skeleton", [])),
                "generation_attempts": result.get("generation_attempts", []),
                "quality_summary": result.get("quality_summary", {}),
            },
            f,
            indent=2,
        )


def main(config_path: str):
    cfg = load_config(config_path)
    case_dir = resolve_from_config(config_path, cfg["input_case_dir"])
    knowledge_dir = resolve_from_config(config_path, cfg["knowledge_dir"])
    output_dir = resolve_from_config(config_path, cfg["output_dir"])
    result = generate_case_prediction(case_dir, knowledge_dir, cfg)
    save_generation(result, output_dir)
    print(f"[Generate] Saved prediction for case_{result['case_id']} to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(pathlib.Path(__file__).with_name("config_generate.json")))
    args = parser.parse_args()
    main(args.config)

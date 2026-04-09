"""
generate_gpt_knowledge.py
-------------------------
Retrieve similar train cases and transfer their trajectories into a target scene.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    case_id: int
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


def generate_case_prediction(target_case_dir: str | pathlib.Path, knowledge_dir: str | pathlib.Path, cfg: dict) -> dict:
    target_case_dir = pathlib.Path(target_case_dir).resolve()
    knowledge_dir = pathlib.Path(knowledge_dir).resolve()
    scene_index = load_scene_index(knowledge_dir)
    target_scene = load_scene(target_case_dir)
    retrieved_cases = retrieve_cases(target_case_dir, scene_index, cfg)
    if not retrieved_cases:
        raise RuntimeError("No retrieved reference cases were found.")

    source_scene = load_scene(retrieved_cases[0].case_dir)
    source_df = load_trajectory(retrieved_cases[0].case_dir)
    source_paths = extract_agent_paths(source_df)
    source_ids = list(sorted(source_paths))

    target_spawn = target_scene["spawn_df"].sort_values("id").reset_index(drop=True)
    src_spawn_center, src_exit_center = scene_centers(source_scene)
    dst_spawn_center, dst_exit_center = scene_centers(target_scene)
    transform = build_similarity_transform(src_spawn_center, src_exit_center, dst_spawn_center, dst_exit_center)

    src_forward, src_lateral = local_frame(src_spawn_center, src_exit_center)
    dst_forward, dst_lateral = local_frame(dst_spawn_center, dst_exit_center)

    source_start_points = np.stack([source_paths[aid]["start"] for aid in source_ids], axis=0)
    source_local = to_local(source_start_points, src_spawn_center, src_forward, src_lateral)
    target_start_points = target_spawn[["pos_x", "pos_y"]].to_numpy(dtype=np.float64)
    target_local = to_local(target_start_points, dst_spawn_center, dst_forward, dst_lateral)

    assign_idx = _greedy_match(target_local, source_local)
    predictions = {}
    rows = []

    for target_row, sidx in zip(target_spawn.itertuples(index=False), assign_idx):
        source_agent_id = source_ids[sidx]
        source_path = source_paths[source_agent_id]
        pred_points = transform(source_path["points"])
        pred_frames = np.arange(len(pred_points), dtype=np.int64)
        predictions[int(target_row.id)] = {
            "points": pred_points,
            "frames": pred_frames,
            "source_case_id": str(retrieved_cases[0].case_id),
            "source_agent_id": int(source_agent_id),
        }
        for frame, (x, y) in zip(pred_frames.tolist(), pred_points.tolist()):
            rows.append(
                {
                    "frame": int(frame),
                    "id": int(target_row.id),
                    "pos_x": float(x),
                    "pos_y": float(y),
                    "source_case_id": str(retrieved_cases[0].case_id),
                    "source_agent_id": int(source_agent_id),
                }
            )

    oob_rate = 0.0
    if rows:
        outside = 0
        for item in rows:
            outside += 0 if point_is_inside_walkable(item["pos_x"], item["pos_y"], target_scene["walkable"]) else 1
        oob_rate = outside / max(len(rows), 1)

    return {
        "case_id": str(target_scene["case_id"]),
        "case_dir": str(target_case_dir),
        "retrieved_cases": [item.__dict__ for item in retrieved_cases],
        "planner_prompt": build_planner_prompt(target_scene, retrieved_cases),
        "predictions": predictions,
        "prediction_df": pd.DataFrame(rows),
        "out_of_bounds_rate": float(oob_rate),
    }


def save_generation(result: dict, output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    result["prediction_df"].to_parquet(output_dir / f"AI_pred_{result['case_id']}.parquet", index=False)
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

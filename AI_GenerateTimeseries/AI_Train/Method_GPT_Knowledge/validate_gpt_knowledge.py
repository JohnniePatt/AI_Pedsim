"""
validate_gpt_knowledge.py
-------------------------
Evaluate retrieval-based scene generation against a validation or test split.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from generate_gpt_knowledge import generate_case_prediction
from prepare_geometry_gpt_knowledge import (
    extract_agent_paths,
    infer_case_id,
    load_scene,
    load_trajectory,
    point_is_inside_walkable,
    resample_path,
)


def load_config(config_path: str | pathlib.Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_config(config_path: str | pathlib.Path, raw_path: str) -> pathlib.Path:
    config_path = pathlib.Path(config_path).resolve()
    path = pathlib.Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def iter_case_dirs(dataset_root: pathlib.Path, split: str) -> list[pathlib.Path]:
    split_dir = dataset_root / split
    return sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("case_")])


def apply_case_sampling(case_dirs: list[pathlib.Path], cfg: dict) -> list[pathlib.Path]:
    percent = float(cfg.get("validation_percent", 100.0))
    percent = max(0.0, min(100.0, percent))
    if percent >= 100.0 or len(case_dirs) <= 1:
        return case_dirs
    if percent <= 0.0:
        return []

    keep = max(1, int(round(len(case_dirs) * percent / 100.0)))
    seed = int(cfg.get("validation_seed", 42))
    rng = random.Random(seed)
    sampled = list(case_dirs)
    rng.shuffle(sampled)
    sampled = sampled[:keep]
    sampled.sort()
    return sampled


def compute_collision_rate(paths: dict[int, dict], threshold_m: float) -> float:
    max_len = max((len(item["points"]) for item in paths.values()), default=0)
    collisions = 0
    pair_count = 0
    ids = sorted(paths)
    for tidx in range(max_len):
        active = []
        for agent_id in ids:
            pts = paths[agent_id]["points"]
            if tidx < len(pts):
                active.append((agent_id, pts[tidx]))
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                pair_count += 1
                if np.linalg.norm(active[i][1] - active[j][1]) < threshold_m:
                    collisions += 1
    return float(collisions / max(pair_count, 1))


def evaluate_case(case_dir: pathlib.Path, knowledge_dir: pathlib.Path, cfg: dict) -> tuple[dict, pd.DataFrame]:
    prediction = generate_case_prediction(case_dir, knowledge_dir, cfg)
    gt_df = load_trajectory(case_dir)
    gt_paths = extract_agent_paths(gt_df)
    pred_paths = prediction["predictions"]
    walkable = load_scene(case_dir)["walkable"]
    eval_steps = int(cfg.get("eval_steps", 100))
    rows = []

    for agent_id in sorted(gt_paths):
        if agent_id not in pred_paths:
            continue
        gt = gt_paths[agent_id]["points"]
        pred = pred_paths[agent_id]["points"]
        gt_eval = resample_path(gt, eval_steps)
        pred_eval = resample_path(pred, eval_steps)
        dists = np.linalg.norm(pred_eval - gt_eval, axis=1)
        rows.append(
            {
                "case_id": str(infer_case_id(case_dir)),
                "agent_id": int(agent_id),
                "ade_m": float(np.mean(dists)),
                "fde_m": float(dists[-1]),
                "gt_duration_frames": int(len(gt)),
                "pred_duration_frames": int(len(pred)),
                "duration_abs_error_frames": int(abs(len(pred) - len(gt))),
            }
        )

    pred_oob_rows = prediction["prediction_df"]
    outside = 0
    for row in pred_oob_rows.itertuples(index=False):
        outside += 0 if point_is_inside_walkable(row.pos_x, row.pos_y, walkable) else 1
    oob_rate = outside / max(len(pred_oob_rows), 1)
    collision_rate = compute_collision_rate(pred_paths, float(cfg.get("collision_threshold_m", 0.3)))

    per_agent = pd.DataFrame(rows)
    summary = {
        "case_id": str(infer_case_id(case_dir)),
        "case_dir": str(case_dir),
        "n_agents_eval": int(len(per_agent)),
        "path_ade_m": float(per_agent["ade_m"].mean()) if len(per_agent) else 0.0,
        "path_fde_m": float(per_agent["fde_m"].mean()) if len(per_agent) else 0.0,
        "duration_abs_error_frames": float(per_agent["duration_abs_error_frames"].mean()) if len(per_agent) else 0.0,
        "collision_rate": float(collision_rate),
        "out_of_bounds_rate": float(oob_rate),
        "retrieved_case_id": str(prediction["retrieved_cases"][0]["case_id"]),
        "retrieval_score": float(prediction["retrieved_cases"][0]["score"]),
    }
    return summary, per_agent


def main(config_path: str):
    cfg = load_config(config_path)
    dataset_root = resolve_from_config(config_path, cfg["dataset_root"])
    split = cfg.get("split", "test")
    knowledge_dir = resolve_from_config(config_path, cfg["knowledge_dir"])
    output_dir = resolve_from_config(config_path, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = iter_case_dirs(dataset_root, split)
    total_cases = len(case_dirs)
    case_dirs = apply_case_sampling(case_dirs, cfg)
    max_cases = int(cfg.get("max_cases", 0))
    if max_cases > 0:
        case_dirs = case_dirs[:max_cases]

    print(
        f"[Validate] Split={split} | selected_cases={len(case_dirs)}/{total_cases} "
        f"(validation_percent={float(cfg.get('validation_percent', 100.0)):.1f}%)"
    )

    summaries = []
    agent_tables = []
    for case_dir in tqdm(case_dirs, desc=f"Validate [{split}]"):
        summary, per_agent = evaluate_case(case_dir, knowledge_dir, cfg)
        summaries.append(summary)
        agent_tables.append(per_agent)

    summary_df = pd.DataFrame(summaries)
    agent_df = pd.concat(agent_tables, ignore_index=True) if agent_tables else pd.DataFrame()
    summary_df.to_csv(output_dir / f"{split}_scene_metrics.csv", index=False)
    agent_df.to_csv(output_dir / f"{split}_agent_metrics.csv", index=False)

    overview = {
        "split": split,
        "num_cases": int(len(summary_df)),
        "mean_path_ade_m": float(summary_df["path_ade_m"].mean()) if len(summary_df) else 0.0,
        "mean_path_fde_m": float(summary_df["path_fde_m"].mean()) if len(summary_df) else 0.0,
        "mean_duration_abs_error_frames": float(summary_df["duration_abs_error_frames"].mean()) if len(summary_df) else 0.0,
        "mean_collision_rate": float(summary_df["collision_rate"].mean()) if len(summary_df) else 0.0,
        "mean_out_of_bounds_rate": float(summary_df["out_of_bounds_rate"].mean()) if len(summary_df) else 0.0,
    }
    with open(output_dir / f"{split}_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"[Validate] Saved metrics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(pathlib.Path(__file__).with_name("config_validate.json")))
    args = parser.parse_args()
    main(args.config)

"""Full-path raw evaluation for joint Social-Force-informed continuous models."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baseline_output import create_evaluation_layout, finalize_evaluation, write_case_prediction
from joint_sf import JointSceneDataset
from train_joint_sf import make_model, move_batch


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", type=pathlib.Path, required=True)
    cli.add_argument("--method-id", required=True)
    cli.add_argument("--architecture", choices=("lstm", "transformer", "sgan"), required=True)
    cli.add_argument("--run-path", type=pathlib.Path, required=True)
    cli.add_argument("--checkpoint", type=pathlib.Path, required=True)
    cli.add_argument("--max-cases", type=int, default=None)
    args = cli.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as stream:
        eval_cfg = json.load(stream)
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location="cpu")
    cfg = dict(saved.get("model_config", {}))
    cfg.update({key: value for key, value in eval_cfg.items() if value is not None})
    dataset_path = pathlib.Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    cfg["dataset_path"] = str(dataset_path)
    max_steps = int(cfg.get("max_rollout_steps", 512))
    dataset = JointSceneDataset(
        dataset_path, "test", obs_len=cfg.get("obs_len", 8), pred_len=max_steps,
        frame_stride=cfg.get("frame_stride", 5), max_agents=cfg.get("max_agents", 64),
        windows_per_case=1, max_cases=args.max_cases or cfg.get("max_test_cases"),
        grid_size=cfg.get("grid_size", 64), geo_padding=cfg.get("geo_padding", 1.0),
        seed=cfg.get("seed", 42), cache_size=cfg.get("case_cache_size", 2),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(cfg, args.architecture).to(device)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    trained_name = str(saved.get("data_config", {}).get("dataset_name", ""))
    compatibility_ok = trained_name.casefold() == dataset_path.name.casefold()
    manifest = dataset_path / "manifest_housegan_cases.csv"
    sample_count = int(cfg.get("stochastic_sample_count", 20)) if args.architecture == "sgan" else 1
    layout = create_evaluation_layout(
        args.run_path.resolve(), method_id=args.method_id,
        dataset_id=cfg.get("dataset_id", "housegan_canonical_imagebase_split_v1"),
        split="test", protocol_version=cfg.get("protocol_version", "joint_sf_full_path_v1"),
        checkpoint_path=checkpoint, evaluation_config=cfg,
        dataset_manifest=manifest if manifest.exists() else None,
        stochastic_sample_count=sample_count, compatibility_ok=compatibility_ok,
        invalid_reason=None if compatibility_ok else "checkpoint/dataset mismatch",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(cfg.get("num_workers", 0)))
    case_rows, case_ids, truncated_cases = [], [], []
    total_agent_steps = 0
    started = time.time()
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            obs_len = int(cfg.get("obs_len", 8))
            case_id = str(raw["case_id"][0]); case_ids.append(case_id)
            if bool(raw["agents_truncated"][0]): truncated_cases.append(case_id)
            agent_ids = raw["agent_ids"][0].numpy()
            min_xy = raw["min_xy"][0].numpy(); scale = float(raw["scale"][0])
            start_frame = int(raw["frames"][0, obs_len - 1])
            rows = []
            sample_metrics = []
            target = batch["positions"][:, :, obs_len:].cpu().numpy()[0]
            target_active = batch["active"][:, :, obs_len:].cpu().numpy()[0]
            walkable_grid = raw["walkable"][0, 0].numpy()
            for sample_id in range(sample_count):
                sample_seed = int(cfg.get("seed", 42)) + sample_id
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(sample_seed)
                output = model.rollout(
                    batch["positions"][:, :, :obs_len], batch["active"][:, :, :obs_len],
                    batch["goal"], batch["wall_field"], max_steps,
                    stop_threshold=cfg.get("stop_threshold", 0.5),
                    exit_radius=cfg.get("exit_radius_norm", 0.025),
                )
                predictions = output["positions"][0].cpu().numpy()
                predicted_active = output["active"][0].cpu().numpy()
                for agent_index, agent_id in enumerate(agent_ids):
                    if agent_id < 0:
                        continue
                    for step in range(max_steps):
                        world = predictions[agent_index, step] * scale + min_xy
                        row = {
                            "case_id": case_id, "split": "test",
                            "frame": start_frame + (step + 1) * int(cfg.get("frame_stride", 5)),
                            "agent_id": int(agent_id), "pos_x": float(world[0]), "pos_y": float(world[1]),
                            "is_active": bool(predicted_active[agent_index, step]),
                        }
                        if args.architecture == "sgan":
                            row.update({"sample_id": sample_id, "sample_seed": sample_seed})
                        rows.append(row)
                valid = target_active & predicted_active
                distances = np.linalg.norm(predictions - target, axis=-1)
                ade = float(distances[valid].mean()) * scale if valid.any() else np.nan
                finals = []
                for agent_index in range(len(agent_ids)):
                    indices = np.flatnonzero(valid[agent_index])
                    if len(indices): finals.append(distances[agent_index, indices[-1]] * scale)
                goal = batch["goal"][0].cpu().numpy()
                reached = np.linalg.norm(predictions - goal[None, None, :], axis=-1) <= float(cfg.get("exit_radius_norm", 0.025))
                grid_x = np.clip(np.rint(predictions[..., 0] * (walkable_grid.shape[1] - 1)).astype(int), 0, walkable_grid.shape[1] - 1)
                grid_y = np.clip(np.rint(predictions[..., 1] * (walkable_grid.shape[0] - 1)).astype(int), 0, walkable_grid.shape[0] - 1)
                walkable_steps = walkable_grid[grid_y, grid_x] > 0.5
                invalid_rate = float((~walkable_steps & predicted_active).sum() / max(predicted_active.sum(), 1))
                collision_count = 0
                pair_count = 0
                collision_threshold = float(cfg.get("collision_threshold_m", 0.4)) / max(scale, 1e-6)
                for step in range(max_steps):
                    active_indices = np.flatnonzero(predicted_active[:, step] & (agent_ids >= 0))
                    if len(active_indices) < 2:
                        continue
                    points = predictions[active_indices, step]
                    pair_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
                    upper = np.triu_indices(len(points), 1)
                    collision_count += int((pair_distances[upper] < collision_threshold).sum())
                    pair_count += len(upper[0])
                sample_metrics.append((
                    ade,
                    float(np.mean(finals)) if finals else np.nan,
                    float(reached.any(axis=1)[agent_ids >= 0].mean()),
                    invalid_rate,
                    float(collision_count / max(pair_count, 1)),
                ))
            prediction_df = pd.DataFrame(rows)
            total_agent_steps += len(prediction_df)
            write_case_prediction(layout, case_id, prediction_df, variant="raw", stochastic=args.architecture == "sgan")
            metric_array = np.asarray(sample_metrics, dtype=np.float64)
            case_rows.append({
                "case_id": case_id,
                "ADE": float(np.nanmean(metric_array[:, 0])),
                "FDE": float(np.nanmean(metric_array[:, 1])),
                "goal_reach_rate": float(np.nanmean(metric_array[:, 2])),
                "non_walkable_rate": float(np.nanmean(metric_array[:, 3])),
                "collision_exposure_rate": float(np.nanmean(metric_array[:, 4])),
                "agents_truncated": bool(raw["agents_truncated"][0]),
            })

    elapsed = time.time() - started
    per_case = pd.DataFrame(case_rows)
    per_case.to_csv(layout.metrics / "per_case_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "method_id": args.method_id, "variant": "raw", "seed": cfg.get("seed", 42),
        "ADE": per_case["ADE"].mean(), "FDE": per_case["FDE"].mean(),
        "path_length_error": np.nan,
        "evacuation_time_error": np.nan,
        "out_of_bounds_rate": 0.0,
        "wall_crossing_rate": np.nan,
        "collision_exposure_rate": per_case["collision_exposure_rate"].mean(),
        "invalid_step_rate": per_case["non_walkable_rate"].mean(),
        "goal_reach_rate": per_case["goal_reach_rate"].mean(),
        "exit_flow_error": np.nan,
        "density_map_error": np.nan,
        "constraint_intervention_rate": 0.0,
        "latency_ms_per_agent_step": elapsed * 1000.0 / max(total_agent_steps, 1),
        "real_time_factor": np.nan,
    }])
    summary.to_csv(layout.metrics / "summary_metrics.csv", index=False)
    plan_by_case = {}
    if manifest.exists():
        manifest_df = pd.read_csv(manifest, usecols=["case_id", "plan_name"])
        plan_by_case = dict(zip(manifest_df["case_id"].astype(str), manifest_df["plan_name"].astype(str)))
    failures = []
    if args.max_cases or cfg.get("max_test_cases"): failures.append("partial test selection")
    if truncated_cases: failures.append(f"agent truncation occurred in {len(truncated_cases)} cases")
    valid_research = finalize_evaluation(
        layout, case_ids=case_ids,
        floorplan_ids={plan_by_case[item] for item in case_ids if item in plan_by_case},
        compatibility_ok=compatibility_ok, canonical_test_required=True,
        additional_failures=failures,
    )
    print(f"[evaluate] cases={len(case_ids)} research_valid={valid_research} output={layout.root}")


if __name__ == "__main__":
    main()

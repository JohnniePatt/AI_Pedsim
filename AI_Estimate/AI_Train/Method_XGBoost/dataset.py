import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_VARIANTS = ["full", "half", "single"]
SPLIT_DIRS = {"train": "Train", "val": "Val", "test": "Test"}


@dataclass
class DataBundle:
    frames: dict
    x: dict
    y_seconds: dict
    y_model: dict
    feature_columns: list
    target_columns: list
    dataset_root: Path
    source_manifest: dict
    split_files: dict


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_path(raw_path, config_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_divide(a, b, default=0.0):
    a = float(a) if pd.notna(a) else 0.0
    b = float(b) if pd.notna(b) else 0.0
    return a / b if abs(b) >= 1e-9 else default


def add_derived_features(frame):
    for column in [
        "computed_agents",
        "topology_centerline_distance_m",
        "straight_distance_m",
        "walkable_area_near_path",
        "door_count_between_A_B",
    ]:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0)
    frame["min_door_width_between_A_B"] = pd.to_numeric(
        frame.get("min_door_width_between_A_B", 1.5), errors="coerce"
    ).fillna(1.5)
    frame["detour_ratio"] = frame.apply(
        lambda row: _safe_divide(
            row["topology_centerline_distance_m"], row["straight_distance_m"], 1.0
        ),
        axis=1,
    )
    frame["distance_gap_m"] = (
        frame["topology_centerline_distance_m"] - frame["straight_distance_m"]
    ).clip(lower=0)
    frame["agent_density_near_path"] = frame.apply(
        lambda row: _safe_divide(row["computed_agents"], row["walkable_area_near_path"]),
        axis=1,
    )
    frame["area_per_agent"] = frame.apply(
        lambda row: _safe_divide(row["walkable_area_near_path"], max(row["computed_agents"], 1)),
        axis=1,
    )
    frame["door_pressure_per_agent"] = frame.apply(
        lambda row: _safe_divide(
            row["computed_agents"] * row["door_count_between_A_B"],
            max(row["min_door_width_between_A_B"], 0.1),
        ),
        axis=1,
    )


def add_variant_columns(frame):
    variants = sorted(set(DEFAULT_VARIANTS) | set(frame["variant_id"].dropna().astype(str)))
    for variant in variants:
        frame[f"variant_{variant}"] = (frame["variant_id"].astype(str) == variant).astype(float)


def feature_columns_from_config(frame, config):
    columns = list(config["features"].get("numeric", []))
    if "variant_id" in config["features"].get("categorical", []):
        columns.extend(
            sorted(
                column
                for column in frame
                if column.startswith("variant_")
                and pd.api.types.is_numeric_dtype(frame[column])
            )
        )
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if len(columns) != 17:
        raise ValueError(f"Expected the MLP-compatible 17 features, found {len(columns)}: {columns}")
    return columns


def _validate_split(frame, split, config):
    expected_rows = config["data"].get("expected_rows", {}).get(split)
    expected_plans = config["data"].get("expected_plans", {}).get(split)
    actual_plans = int(frame["plan"].nunique())
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(f"{split} row count mismatch: expected {expected_rows}, found {len(frame)}")
    if expected_plans is not None and actual_plans != int(expected_plans):
        raise ValueError(f"{split} plan count mismatch: expected {expected_plans}, found {actual_plans}")


def build_data_bundle(config, config_path):
    dataset_root = resolve_path(config["data"]["data_estimate_root"], config_path)
    manifest_path = dataset_root / "data_estimate_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    source_manifest = read_json(manifest_path)
    expected_id = config["data"].get("dataset_id")
    if source_manifest.get("dataset_id") != expected_id:
        raise ValueError(
            f"Dataset ID mismatch: expected {expected_id}, found {source_manifest.get('dataset_id')}"
        )

    frames = {}
    split_files = {}
    for split, directory in SPLIT_DIRS.items():
        path = dataset_root / directory / "data_estimate.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
        if "split" not in frame:
            frame["split"] = split
        frame = frame[frame["split"].astype(str).str.lower() == split].copy()
        status = config["data"].get("filter_status")
        if status and "status" in frame:
            frame = frame[frame["status"].astype(str).str.lower() == str(status).lower()].copy()
        frame.reset_index(drop=True, inplace=True)
        _validate_split(frame, split, config)
        frames[split] = frame
        split_files[split] = {"path": str(path), "sha256": sha256_file(path)}

    plan_sets = {split: set(frame["plan"].astype(str)) for split, frame in frames.items()}
    overlaps = {
        "train_val": sorted(plan_sets["train"] & plan_sets["val"]),
        "train_test": sorted(plan_sets["train"] & plan_sets["test"]),
        "val_test": sorted(plan_sets["val"] & plan_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Plan split overlap detected: {overlaps}")

    combined = pd.concat(frames.values(), ignore_index=True)
    add_derived_features(combined)
    add_variant_columns(combined)
    feature_columns = feature_columns_from_config(combined, config)
    target_columns = list(config["features"]["target"])
    required = feature_columns + target_columns
    if combined[required].isna().any().any():
        bad = combined[required].columns[combined[required].isna().any()].tolist()
        raise ValueError(f"Missing values in model columns: {bad}")

    processed_frames = {}
    x = {}
    y_seconds = {}
    y_model = {}
    for split in SPLIT_DIRS:
        processed = combined[combined["split"].astype(str).str.lower() == split].copy().reset_index(drop=True)
        processed_frames[split] = processed
        x[split] = processed[feature_columns].astype(float).to_numpy(dtype=np.float32)
        y_seconds[split] = processed[target_columns].astype(float).to_numpy(dtype=np.float32)
        y_model[split] = np.log1p(y_seconds[split]).astype(np.float32)

    return DataBundle(
        frames=processed_frames,
        x=x,
        y_seconds=y_seconds,
        y_model=y_model,
        feature_columns=feature_columns,
        target_columns=target_columns,
        dataset_root=dataset_root,
        source_manifest=source_manifest,
        split_files=split_files,
    )


def inverse_target_transform(values):
    return np.expm1(np.asarray(values, dtype=np.float64)).clip(min=0)


def ordered_time_predictions(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sort(values, axis=1)


def compute_metrics(pred_seconds, true_seconds, target_columns):
    error = np.asarray(pred_seconds) - np.asarray(true_seconds)
    metrics = {
        "rows": int(len(error)),
        "values": int(error.size),
        "mae_overall_s": float(np.mean(np.abs(error))),
        "mse_overall_s2": float(np.mean(error ** 2)),
        "rmse_overall_s": float(np.sqrt(np.mean(error ** 2))),
    }
    for index, target in enumerate(target_columns):
        target_error = error[:, index]
        metrics[f"mae_{target}"] = float(np.mean(np.abs(target_error)))
        metrics[f"mse_{target}"] = float(np.mean(target_error ** 2))
        metrics[f"rmse_{target}"] = float(np.sqrt(np.mean(target_error ** 2)))
    consistency = (
        (pred_seconds[:, 0] <= pred_seconds[:, 1])
        & (pred_seconds[:, 1] <= pred_seconds[:, 2])
    )
    metrics["target_order_valid_rows"] = int(consistency.sum())
    metrics["target_order_violation_rows"] = int((~consistency).sum())
    metrics["target_order_violation_rate"] = float((~consistency).mean())
    return metrics

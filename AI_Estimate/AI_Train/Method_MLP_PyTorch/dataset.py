import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_VARIANTS = ["full", "half", "single"]


@dataclass
class EstimateDataBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    dataframe: pd.DataFrame
    splits: dict
    feature_columns: list
    target_columns: list
    scaler: dict


class TimeEstimateDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_columns, scaler):
        self.meta = dataframe.reset_index(drop=True)
        features = self.meta[feature_columns].astype(float).to_numpy(dtype=np.float32)
        targets = self.meta[target_columns].astype(float).to_numpy(dtype=np.float32)
        self.x = standardize(features, scaler["feature_mean"], scaler["feature_std"])
        self.y_seconds = targets
        self.y = standardize(np.log1p(targets), scaler["target_mean"], scaler["target_std"])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.x[index]),
            torch.from_numpy(self.y[index]),
        )


def split_dir_name(split_name):
    return {"train": "Train", "val": "Val", "test": "Test"}[split_name]


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def resolve_path(raw_path, config_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def resolve_data_estimate_root(config, config_path):
    raw_root = config.get("data", {}).get("data_estimate_root", "../../../Dataset/Data_Estimate")
    return resolve_path(raw_root, config_path)


def formatted_split_path(config, config_path, split_name):
    return resolve_data_estimate_root(config, config_path) / split_dir_name(split_name) / "data_estimate.csv"


def formatted_manifest_path(config, config_path):
    return resolve_data_estimate_root(config, config_path) / "data_estimate_manifest.json"


def formatted_splits_exist(config, config_path):
    return all(formatted_split_path(config, config_path, split).exists() for split in ["train", "val", "test"])


def safe_divide(a, b, default=0.0):
    a = float(a) if pd.notna(a) else 0.0
    b = float(b) if pd.notna(b) else 0.0
    if abs(b) < 1e-9:
        return default
    return a / b


def load_joined_dataframe(config, config_path):
    data_cfg = config.get("data", {})
    time_csv = resolve_path(data_cfg["time_summary_csv"], config_path)
    route_csv = resolve_path(data_cfg["route_information_csv"], config_path)

    time_df = pd.read_csv(time_csv)
    route_df = pd.read_csv(route_csv)
    required_key = ["plan", "start_node", "end_node"]

    time_df = time_df.dropna(subset=required_key + ["variant_id"])
    status = data_cfg.get("filter_status", "success")
    if status:
        time_df = time_df[time_df["status"].astype(str).str.lower() == str(status).lower()]

    route_df = route_df.dropna(subset=required_key)
    merged = time_df.merge(route_df, on=required_key, how="inner", suffixes=("", "_route"))
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=config["features"]["target"])
    add_derived_features(merged)
    add_variant_columns(merged)
    return merged.reset_index(drop=True)


def add_derived_features(df):
    df["computed_agents"] = pd.to_numeric(df.get("computed_agents", 0), errors="coerce").fillna(0)
    df["topology_centerline_distance_m"] = pd.to_numeric(df["topology_centerline_distance_m"], errors="coerce").fillna(0)
    df["straight_distance_m"] = pd.to_numeric(df["straight_distance_m"], errors="coerce").fillna(0)
    df["walkable_area_near_path"] = pd.to_numeric(df["walkable_area_near_path"], errors="coerce").fillna(0)
    df["door_count_between_A_B"] = pd.to_numeric(df["door_count_between_A_B"], errors="coerce").fillna(0)
    df["min_door_width_between_A_B"] = pd.to_numeric(df["min_door_width_between_A_B"], errors="coerce").fillna(1.5)

    df["detour_ratio"] = df.apply(
        lambda row: safe_divide(row["topology_centerline_distance_m"], row["straight_distance_m"], default=1.0),
        axis=1,
    )
    df["distance_gap_m"] = (df["topology_centerline_distance_m"] - df["straight_distance_m"]).clip(lower=0)
    df["agent_density_near_path"] = df.apply(
        lambda row: safe_divide(row["computed_agents"], row["walkable_area_near_path"], default=0.0),
        axis=1,
    )
    df["area_per_agent"] = df.apply(
        lambda row: safe_divide(row["walkable_area_near_path"], max(row["computed_agents"], 1), default=0.0),
        axis=1,
    )
    df["door_pressure_per_agent"] = df.apply(
        lambda row: safe_divide(
            row["computed_agents"] * row["door_count_between_A_B"],
            max(row["min_door_width_between_A_B"], 0.1),
            default=0.0,
        ),
        axis=1,
    )


def add_variant_columns(df):
    variants = sorted(set(DEFAULT_VARIANTS).union(set(df["variant_id"].dropna().astype(str).unique())))
    for variant in variants:
        df[f"variant_{variant}"] = (df["variant_id"].astype(str) == variant).astype(float)


def feature_columns_from_config(df, config):
    numeric_columns = list(config["features"].get("numeric", []))
    categorical_columns = []
    for name in config["features"].get("categorical", []):
        if name == "variant_id":
            categorical_columns.extend(
                sorted(
                    [
                        col
                        for col in df.columns
                        if col.startswith("variant_") and pd.api.types.is_numeric_dtype(df[col])
                    ]
                )
            )
    columns = numeric_columns + categorical_columns
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return columns


def split_plans(df, config):
    data_cfg = config.get("data", {})
    train_percent = float(data_cfg.get("train_percent", 70))
    val_percent = float(data_cfg.get("val_percent", 15))
    test_percent = float(data_cfg.get("test_percent", 15))
    total = train_percent + val_percent + test_percent
    if total <= 0:
        raise ValueError("train_percent + val_percent + test_percent must be > 0")

    plans = sorted(df["plan"].dropna().astype(str).unique().tolist())
    rng = random.Random(int(data_cfg.get("random_seed", 42)))
    rng.shuffle(plans)

    n = len(plans)
    n_train = max(1, int(math.floor(n * train_percent / total)))
    n_val = max(1, int(math.floor(n * val_percent / total))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1 if n >= 3 else 0

    train_plans = plans[:n_train]
    val_plans = plans[n_train : n_train + n_val]
    test_plans = plans[n_train + n_val :]
    if not test_plans and val_plans:
        test_plans = [val_plans.pop()]

    return {"train": train_plans, "val": val_plans, "test": test_plans}


def load_formatted_split_dataframe(config, config_path, split_name):
    path = formatted_split_path(config, config_path, split_name)
    if not path.exists():
        root = resolve_data_estimate_root(config, config_path)
        raise FileNotFoundError(
            f"Missing formatted {split_dir_name(split_name)} data: {path}. "
            f"Please place your prepared split file in {root}."
        )
    df = pd.read_csv(path)
    if "split" not in df.columns:
        df["split"] = split_name
    return df


def load_formatted_dataframes(config, config_path):
    frames = {
        split: load_formatted_split_dataframe(config, config_path, split)
        for split in ["train", "val", "test"]
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    add_derived_features(combined)
    add_variant_columns(combined)
    combined = combined.reset_index(drop=True)
    frames = {
        split: combined[combined["split"].astype(str).str.lower() == split].copy()
        for split in ["train", "val", "test"]
    }
    return frames, combined


def scaler_from_train(df, feature_columns, target_columns):
    features = df[feature_columns].astype(float).to_numpy(dtype=np.float32)
    targets = np.log1p(df[target_columns].astype(float).to_numpy(dtype=np.float32))
    return {
        "feature_mean": features.mean(axis=0).tolist(),
        "feature_std": np.maximum(features.std(axis=0), 1e-6).tolist(),
        "target_mean": targets.mean(axis=0).tolist(),
        "target_std": np.maximum(targets.std(axis=0), 1e-6).tolist(),
    }


def standardize(values, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def inverse_target_transform(values, scaler):
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler["target_mean"], dtype=np.float32)
    std = np.asarray(scaler["target_std"], dtype=np.float32)
    return np.expm1(values * std + mean).clip(min=0)


def build_data_bundle(config, config_path):
    use_formatted = bool(config.get("data", {}).get("use_formatted_data", True))
    if use_formatted:
        split_frames, df = load_formatted_dataframes(config, config_path)
    else:
        raise ValueError(
            "use_formatted_data=false is disabled in this workflow. "
            "Use pre-split files under Dataset/Data_Estimate/{Train,Val,Test}."
        )

    feature_columns = feature_columns_from_config(df, config)
    target_columns = list(config["features"]["target"])

    train_df = split_frames["train"].copy()
    val_df = split_frames["val"].copy()
    test_df = split_frames["test"].copy()
    splits = {
        "train": sorted(train_df["plan"].dropna().astype(str).unique().tolist()),
        "val": sorted(val_df["plan"].dropna().astype(str).unique().tolist()),
        "test": sorted(test_df["plan"].dropna().astype(str).unique().tolist()),
    }
    if train_df.empty:
        raise ValueError("Training dataframe is empty after split.")

    scaler = scaler_from_train(train_df, feature_columns, target_columns)
    return EstimateDataBundle(
        train=TimeEstimateDataset(train_df, feature_columns, target_columns, scaler),
        val=TimeEstimateDataset(val_df, feature_columns, target_columns, scaler),
        test=TimeEstimateDataset(test_df, feature_columns, target_columns, scaler),
        dataframe=df,
        splits=splits,
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler=scaler,
    )

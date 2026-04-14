import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


DEFAULT_VARIANTS = ["full", "half", "single"]


@dataclass
class KerasDataBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    dataframe: pd.DataFrame
    splits: dict
    feature_columns: list
    target_columns: list
    scaler: dict
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(raw_path, config_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def resolve_data_estimate_root(config, config_path):
    raw_root = config.get("data", {}).get("data_estimate_root", "../../../Dataset/Data_Estimate")
    return resolve_path(raw_root, config_path)


def _split_dir(split_name):
    return {"train": "Train", "val": "Val", "test": "Test"}[split_name]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split_csv(config, config_path, split_name):
    path = resolve_data_estimate_root(config, config_path) / _split_dir(split_name) / "data_estimate.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    df = pd.read_csv(path)
    if "split" not in df.columns:
        df["split"] = split_name
    return df


def _safe_divide(a, b, default=0.0):
    a = float(a) if pd.notna(a) else 0.0
    b = float(b) if pd.notna(b) else 0.0
    return a / b if abs(b) >= 1e-9 else default


def _add_derived_features(df):
    for col in ["computed_agents", "topology_centerline_distance_m", "straight_distance_m",
                "walkable_area_near_path", "door_count_between_A_B"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["min_door_width_between_A_B"] = pd.to_numeric(
        df.get("min_door_width_between_A_B", 1.5), errors="coerce"
    ).fillna(1.5)

    df["detour_ratio"] = df.apply(
        lambda r: _safe_divide(r["topology_centerline_distance_m"], r["straight_distance_m"], default=1.0),
        axis=1,
    )
    df["distance_gap_m"] = (df["topology_centerline_distance_m"] - df["straight_distance_m"]).clip(lower=0)
    df["agent_density_near_path"] = df.apply(
        lambda r: _safe_divide(r["computed_agents"], r["walkable_area_near_path"]), axis=1
    )
    df["area_per_agent"] = df.apply(
        lambda r: _safe_divide(r["walkable_area_near_path"], max(r["computed_agents"], 1)), axis=1
    )
    df["door_pressure_per_agent"] = df.apply(
        lambda r: _safe_divide(
            r["computed_agents"] * r["door_count_between_A_B"],
            max(r["min_door_width_between_A_B"], 0.1),
        ),
        axis=1,
    )


def _add_variant_columns(df):
    variants = sorted(set(DEFAULT_VARIANTS) | set(df["variant_id"].dropna().astype(str).unique()))
    for v in variants:
        df[f"variant_{v}"] = (df["variant_id"].astype(str) == v).astype(float)


def _feature_columns(df, config):
    numeric = list(config["features"].get("numeric", []))
    categorical = []
    for name in config["features"].get("categorical", []):
        if name == "variant_id":
            categorical += sorted(
                c for c in df.columns
                if c.startswith("variant_") and pd.api.types.is_numeric_dtype(df[c])
            )
    columns = numeric + categorical
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return columns


# ---------------------------------------------------------------------------
# Scaler (standardize on log1p targets)
# ---------------------------------------------------------------------------

def _fit_scaler(df, feature_columns, target_columns):
    x = df[feature_columns].astype(float).to_numpy(dtype=np.float32)
    y = np.log1p(df[target_columns].astype(float).to_numpy(dtype=np.float32))
    return {
        "feature_mean": x.mean(axis=0).tolist(),
        "feature_std": np.maximum(x.std(axis=0), 1e-6).tolist(),
        "target_mean": y.mean(axis=0).tolist(),
        "target_std": np.maximum(y.std(axis=0), 1e-6).tolist(),
    }


def _standardize(values, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _to_xy(df, feature_columns, target_columns, scaler):
    x = _standardize(
        df[feature_columns].astype(float).to_numpy(dtype=np.float32),
        scaler["feature_mean"], scaler["feature_std"],
    )
    y = _standardize(
        np.log1p(df[target_columns].astype(float).to_numpy(dtype=np.float32)),
        scaler["target_mean"], scaler["target_std"],
    )
    return x, y


# ---------------------------------------------------------------------------
# tf.data.Dataset factory
# ---------------------------------------------------------------------------

def _make_tf_dataset(x, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Inverse transform helpers (used by train / test)
# ---------------------------------------------------------------------------

def inverse_target_transform(values, scaler):
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler["target_mean"], dtype=np.float32)
    std = np.asarray(scaler["target_std"], dtype=np.float32)
    return np.expm1(values * std + mean).clip(min=0)


def ordered_time_predictions(values):
    """Enforce min <= mean <= max ordering across the three output targets."""
    values = np.asarray(values, dtype=np.float32)
    ordered = values.copy()
    low = np.minimum.reduce(values[:, [0, 1, 2]].T)
    high = np.maximum.reduce(values[:, [0, 1, 2]].T)
    ordered[:, 0] = low
    ordered[:, 1] = np.clip(values[:, 1], low, high)
    ordered[:, 2] = high
    return ordered


# ---------------------------------------------------------------------------
# Main bundle builder
# ---------------------------------------------------------------------------

def build_data_bundle(config, config_path, batch_size=64):
    split_frames = {s: _load_split_csv(config, config_path, s) for s in ["train", "val", "test"]}
    df = pd.concat(split_frames.values(), ignore_index=True).replace([np.inf, -np.inf], np.nan)
    _add_derived_features(df)
    _add_variant_columns(df)

    feature_columns = _feature_columns(df, config)
    target_columns = list(config["features"]["target"])

    train_df = df[df["split"].astype(str).str.lower() == "train"].copy().reset_index(drop=True)
    val_df   = df[df["split"].astype(str).str.lower() == "val"].copy().reset_index(drop=True)
    test_df  = df[df["split"].astype(str).str.lower() == "test"].copy().reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training dataframe is empty.")

    scaler = _fit_scaler(train_df, feature_columns, target_columns)

    x_train, y_train = _to_xy(train_df, feature_columns, target_columns, scaler)
    x_val,   y_val   = _to_xy(val_df,   feature_columns, target_columns, scaler)
    x_test,  y_test  = _to_xy(test_df,  feature_columns, target_columns, scaler)

    splits = {
        "train": sorted(train_df["plan"].dropna().astype(str).unique().tolist()),
        "val":   sorted(val_df["plan"].dropna().astype(str).unique().tolist()),
        "test":  sorted(test_df["plan"].dropna().astype(str).unique().tolist()),
    }

    return KerasDataBundle(
        train_ds=_make_tf_dataset(x_train, y_train, batch_size, shuffle=True),
        val_ds=_make_tf_dataset(x_val,   y_val,   batch_size, shuffle=False),
        test_ds=_make_tf_dataset(x_test,  y_test,  batch_size, shuffle=False),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        dataframe=df.reset_index(drop=True),
        splits=splits,
        feature_columns=feature_columns,
        target_columns=target_columns,
        scaler=scaler,
        x_train=x_train, y_train=y_train,
        x_val=x_val,     y_val=y_val,
        x_test=x_test,   y_test=y_test,
    )

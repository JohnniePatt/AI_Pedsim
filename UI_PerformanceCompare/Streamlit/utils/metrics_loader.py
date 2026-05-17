from __future__ import annotations

from pathlib import Path

import pandas as pd


METRIC_ORDER = ["MAE", "MSE", "RMSE", "SSIM", "PSNR", "LPIPS"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_summary(run_path: Path) -> pd.DataFrame:
    df = _read_csv(run_path / "test_evaluation_summary.csv")
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["metric", "value"])
    df = df[["metric", "value"]].copy()
    df["metric"] = df["metric"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_per_image(run_path: Path) -> pd.DataFrame:
    df = _read_csv(run_path / "test_evaluation_per_image.csv")
    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "mae": "MAE",
        "mse": "MSE",
        "rmse": "RMSE",
        "ssim": "SSIM",
        "psnr": "PSNR",
        "lpips": "LPIPS",
    }
    df = df.rename(columns=rename_map)
    for metric in METRIC_ORDER:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def metric_summary_from_per_image(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_ORDER:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "worst": values.min() if metric in {"SSIM", "PSNR"} else values.max(),
            }
        )
    return pd.DataFrame(rows)


def attach_run_label(df: pd.DataFrame, run_label: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "run", run_label)
    return out

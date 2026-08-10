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


import numpy as np


def load_summary(run_path: Path) -> pd.DataFrame:
    candidates = [
        run_path / "test_evaluation_summary.csv",
        run_path / "test_results" / "test_evaluation_summary.csv",
        run_path / "logs" / "test_evaluation_summary.csv",
    ]
    df = pd.DataFrame()
    for c in candidates:
        df = _read_csv(c)
        if not df.empty and "metric" in df.columns and "value" in df.columns:
            break

    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["metric", "value"])
    df = df[["metric", "value"]].copy()
    df["metric"] = df["metric"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_per_image(run_path: Path) -> pd.DataFrame:
    candidates = [
        run_path / "test_evaluation_per_image.csv",
        run_path / "test_results" / "test_evaluation_per_image.csv",
        run_path / "logs" / "test_evaluation.csv",
        run_path / "logs" / "test_evaluation_per_image.csv",
    ]
    df = pd.DataFrame()
    for c in candidates:
        df = _read_csv(c)
        if not df.empty:
            break

    if df.empty:
        return pd.DataFrame()

    # Normalize column names to uppercase except file_name
    new_cols = []
    for col in df.columns:
        c_clean = str(col).strip()
        if c_clean.lower() in ("filename", "file_name"):
            new_cols.append("file_name")
        else:
            new_cols.append(c_clean.upper())
    df.columns = new_cols

    if "MAE" in df.columns and "MSE" in df.columns:
        if "RMSE" not in df.columns:
            df["RMSE"] = np.sqrt(pd.to_numeric(df["MSE"], errors="coerce"))
        if "PSNR" not in df.columns:
            df["PSNR"] = 20.0 * np.log10(1.0 / np.maximum(df["RMSE"], 1e-12))

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


def load_walkable_per_image(run_path: Path) -> pd.DataFrame:
    candidates = [
        run_path / "test_evaluation_walkable_per_image.csv",
        run_path / "test_results" / "test_evaluation_walkable_per_image.csv",
        run_path / "logs" / "test_evaluation_walkable_per_image.csv",
    ]
    df = pd.DataFrame()
    for c in candidates:
        df = _read_csv(c)
        if not df.empty:
            break
    if df.empty:
        return pd.DataFrame()

    new_cols = []
    for col in df.columns:
        c_clean = str(col).strip()
        if c_clean.lower() in ("filename", "file_name"):
            new_cols.append("file_name")
        else:
            new_cols.append(c_clean.upper())
    df.columns = new_cols

    for col in METRIC_ORDER:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derivations if RMSE or PSNR is missing
    if "MSE" in df.columns and "RMSE" not in df.columns:
        df["RMSE"] = np.sqrt(df["MSE"].clip(lower=0))

    if "RMSE" in df.columns and "PSNR" not in df.columns:
        def _calc_psnr(val):
            if pd.isna(val) or val <= 1e-12:
                return np.nan
            return 20.0 * np.log10(1.0 / val)
        df["PSNR"] = df["RMSE"].apply(_calc_psnr)

    cols = ["file_name"] + [m for m in METRIC_ORDER if m in df.columns]
    return df[cols].copy()


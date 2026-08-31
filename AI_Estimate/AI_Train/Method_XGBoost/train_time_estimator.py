import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import (
    build_data_bundle,
    compute_metrics,
    inverse_target_transform,
    read_json,
    sha256_file,
    write_json,
)


def require_xgboost():
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is not installed. Install Method_XGBoost/requirements.txt in the active environment."
        ) from exc
    return xgboost


def resolve_output_root(config, config_path):
    raw = Path(config["output"]["root"])
    return raw if raw.is_absolute() else (Path(config_path).resolve().parent / raw).resolve()


def make_run_dir(config, config_path):
    seed = int(config["train"]["random_seed"])
    run_name = config["output"].get("run_name", "auto")
    if run_name == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"run_{timestamp}_seed{seed:03d}"
    run_dir = resolve_output_root(config, config_path) / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing to overwrite: {run_dir}")
    for relative in ["checkpoints", "logs", "diagnostics", "evaluations"]:
        (run_dir / relative).mkdir(parents=True, exist_ok=False)
    return run_dir


def git_value(args, cwd):
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_parameters(config, target_index):
    model = config["model"]
    return {
        "booster": model.get("booster", "gbtree"),
        "objective": model.get("objective", "reg:squarederror"),
        "n_estimators": int(model.get("n_estimators", 500)),
        "learning_rate": float(model.get("learning_rate", 0.05)),
        "max_depth": int(model.get("max_depth", 6)),
        "min_child_weight": float(model.get("min_child_weight", 1.0)),
        "subsample": float(model.get("subsample", 1.0)),
        "colsample_bytree": float(model.get("colsample_bytree", 1.0)),
        "reg_alpha": float(model.get("reg_alpha", 0.0)),
        "reg_lambda": float(model.get("reg_lambda", 1.0)),
        "tree_method": model.get("tree_method", "hist"),
        "n_jobs": int(model.get("n_jobs", -1)),
        "random_state": int(config["train"]["random_seed"]) + target_index,
        "eval_metric": config["train"].get("eval_metric", "rmse"),
        "early_stopping_rounds": int(config["train"].get("early_stopping_rounds", 30)),
    }


def predict_matrix(models, x):
    return np.column_stack([model.predict(x) for model in models])


def train(config_path):
    xgboost = require_xgboost()
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    bundle = build_data_bundle(config, config_path)
    run_dir = make_run_dir(config, config_path)
    started_at = datetime.now(timezone.utc)

    models = []
    histories = []
    checkpoint_entries = []
    for index, target in enumerate(bundle.target_columns):
        model = xgboost.XGBRegressor(**model_parameters(config, index))
        model.fit(
            bundle.x["train"],
            bundle.y_model["train"][:, index],
            eval_set=[
                (bundle.x["train"], bundle.y_model["train"][:, index]),
                (bundle.x["val"], bundle.y_model["val"][:, index]),
            ],
            verbose=bool(config["train"].get("verbose", False)),
        )
        checkpoint = run_dir / "checkpoints" / f"best_model_{target}.json"
        model.save_model(checkpoint)
        checkpoint_entries.append(
            {
                "target": target,
                "path": str(checkpoint.relative_to(run_dir)),
                "sha256": sha256_file(checkpoint),
                "best_iteration": int(getattr(model, "best_iteration", -1)),
                "best_score": float(getattr(model, "best_score", np.nan)),
            }
        )
        result = model.evals_result()
        train_values = result.get("validation_0", {}).get(config["train"]["eval_metric"], [])
        val_values = result.get("validation_1", {}).get(config["train"]["eval_metric"], [])
        histories.append(
            pd.DataFrame(
                {
                    "round": np.arange(1, len(train_values) + 1),
                    "target": target,
                    "train_rmse_log": train_values,
                    "val_rmse_log": val_values,
                }
            )
        )
        models.append(model)

    train_pred = inverse_target_transform(predict_matrix(models, bundle.x["train"]))
    val_pred = inverse_target_transform(predict_matrix(models, bundle.x["val"]))
    train_metrics = compute_metrics(train_pred, bundle.y_seconds["train"], bundle.target_columns)
    val_metrics = compute_metrics(val_pred, bundle.y_seconds["val"], bundle.target_columns)
    ended_at = datetime.now(timezone.utc)

    pd.concat(histories, ignore_index=True).to_csv(run_dir / "logs" / "training_history.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(run_dir / "training_history.csv", index=False)
    write_json(run_dir / "config_used.json", config)
    write_json(run_dir / "config_resolved.json", config)
    write_json(
        run_dir / "method_manifest.json",
        {
            "method_id": "Method_XGBoost",
            "display_name": "XGBoost",
            "model_family": "gradient_boosted_decision_trees",
            "implementation": "vanilla gbtree; one independent regressor per target",
            "input_feature_count": len(bundle.feature_columns),
            "output_target_count": len(bundle.target_columns),
            "constraint_mode": "none",
            "target_order_postprocessing": bool(config["features"].get("enforce_target_order", False)),
        },
    )
    dataset_manifest = {
        "dataset_id": bundle.source_manifest["dataset_id"],
        "canonical_dataset_id": bundle.source_manifest.get("canonical_dataset_id"),
        "source_manifest_path": str(bundle.dataset_root / "data_estimate_manifest.json"),
        "source_manifest_sha256": sha256_file(bundle.dataset_root / "data_estimate_manifest.json"),
        "rows": sum(len(frame) for frame in bundle.frames.values()),
        "train_rows": len(bundle.frames["train"]),
        "val_rows": len(bundle.frames["val"]),
        "test_rows": len(bundle.frames["test"]),
        "train_plans": int(bundle.frames["train"]["plan"].nunique()),
        "val_plans": int(bundle.frames["val"]["plan"].nunique()),
        "test_plans": int(bundle.frames["test"]["plan"].nunique()),
        "feature_columns": bundle.feature_columns,
        "target_columns": bundle.target_columns,
        "split_files": bundle.split_files,
    }
    write_json(run_dir / "dataset_manifest.json", dataset_manifest)
    write_json(run_dir / "dataset_manifest_snapshot.json", bundle.source_manifest)
    write_json(
        run_dir / "model_bundle.json",
        {
            "feature_columns": bundle.feature_columns,
            "target_columns": bundle.target_columns,
            "target_transform": "log1p",
            "input_scaling": "none",
            "checkpoints": checkpoint_entries,
        },
    )
    write_json(run_dir / "checkpoints" / "checkpoint_manifest.json", {"models": checkpoint_entries})
    write_json(run_dir / "best_result.json", {"models": checkpoint_entries})
    write_json(run_dir / "metrics.json", {"best": {"train": train_metrics, "val": val_metrics}})
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cpu_count": os.cpu_count(),
        },
    )
    repo_root = Path(__file__).resolve().parents[3]
    write_json(
        run_dir / "code_provenance.json",
        {
            "git_commit": git_value(["rev-parse", "HEAD"], repo_root),
            "git_branch": git_value(["branch", "--show-current"], repo_root),
            "git_status_short": git_value(["status", "--short"], repo_root),
            "source_files": {
                name: sha256_file(Path(__file__).resolve().parent / name)
                for name in ["train_time_estimator.py", "dataset.py", "config_train.json"]
            },
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_dir.name,
            "method_id": "Method_XGBoost",
            "dataset_id": bundle.source_manifest["dataset_id"],
            "seed": int(config["train"]["random_seed"]),
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": ended_at.isoformat(),
            "duration_seconds": (ended_at - started_at).total_seconds(),
            "status": "trained",
            "research_valid": False,
            "research_valid_reason": "Training complete; canonical test evaluation is a separate stage.",
        },
    )
    bundle.frames["train"].head(500).to_csv(run_dir / "dataset_preview.csv", index=False)
    print(f"[AI_Estimate][XGBoost][Train] run={run_dir}")
    print(f"[AI_Estimate][XGBoost][Train] val_mae={val_metrics['mae_overall_s']:.3f}s")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train vanilla XGBoost travel-time estimators.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_train.json")))
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

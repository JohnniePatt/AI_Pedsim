import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dataset import (
    build_data_bundle,
    compute_metrics,
    inverse_target_transform,
    ordered_time_predictions,
    read_json,
    sha256_file,
    write_json,
)
from train_time_estimator import require_xgboost, resolve_output_root


def latest_run(output_root):
    candidates = sorted(
        (path for path in Path(output_root).glob("run_*") if (path / "model_bundle.json").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No trained XGBoost runs under {output_root}")
    return candidates[0]


def resolve_run(config, config_path, checkpoint):
    if checkpoint:
        path = Path(checkpoint).resolve()
        if path.is_dir():
            return path
        if path.name == "best_result.json":
            return path.parent
        if path.parent.name == "checkpoints":
            return path.parent.parent
        return path.parent
    return latest_run(resolve_output_root(config, config_path))


def load_models(xgboost, run_dir, model_bundle):
    models = []
    for index, entry in enumerate(model_bundle["checkpoints"]):
        checkpoint = run_dir / entry["path"]
        if sha256_file(checkpoint) != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        model = xgboost.XGBRegressor()
        model.load_model(checkpoint)
        models.append(model)
    return models


def evaluate(config_path, checkpoint=None, output_dir=None):
    xgboost = require_xgboost()
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    run_dir = resolve_run(config, config_path, checkpoint)
    model_bundle = read_json(run_dir / "model_bundle.json")
    dataset_manifest = read_json(run_dir / "dataset_manifest.json")
    bundle = build_data_bundle(config, config_path)

    if model_bundle["feature_columns"] != bundle.feature_columns:
        raise ValueError("Feature schema differs from the training run")
    if model_bundle["target_columns"] != bundle.target_columns:
        raise ValueError("Target schema differs from the training run")
    if dataset_manifest["dataset_id"] != bundle.source_manifest["dataset_id"]:
        raise ValueError("Dataset ID differs from the training run")
    models = load_models(xgboost, run_dir, model_bundle)
    pred_log = np.column_stack([model.predict(bundle.x["test"]) for model in models])
    pred_raw_seconds = inverse_target_transform(pred_log)
    enforce_order = bool(config["features"].get("enforce_target_order", False))
    pred_seconds = ordered_time_predictions(pred_raw_seconds) if enforce_order else pred_raw_seconds
    true_seconds = bundle.y_seconds["test"].astype(np.float64)
    metrics = compute_metrics(pred_seconds, true_seconds, bundle.target_columns)
    raw_metrics = compute_metrics(pred_raw_seconds, true_seconds, bundle.target_columns)

    output_dir = Path(output_dir).resolve() if output_dir else run_dir / "test_eval"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = bundle.frames["test"].copy().reset_index(drop=True)
    for index, target in enumerate(bundle.target_columns):
        result[f"true_{target}"] = true_seconds[:, index]
        result[f"pred_raw_{target}"] = pred_raw_seconds[:, index]
        result[f"pred_{target}"] = pred_seconds[:, index]
        result[f"abs_error_{target}"] = np.abs(pred_seconds[:, index] - true_seconds[:, index])
    result["target_order_intervened"] = np.any(pred_seconds != pred_raw_seconds, axis=1)
    result.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "test_metrics.json", metrics)
    write_json(output_dir / "raw_test_metrics.json", raw_metrics)
    write_json(
        output_dir / "evaluation_manifest.json",
        {
            "evaluation_id": (
                f"eval_{bundle.source_manifest['dataset_id']}_test_"
                f"{config['evaluation']['protocol_version']}"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method_id": "Method_XGBoost",
            "run_id": run_dir.name,
            "dataset_id": bundle.source_manifest["dataset_id"],
            "split": "test",
            "case_count": len(result),
            "floorplan_count": int(result["plan"].nunique()),
            "target_transform": "log1p",
            "input_scaling": "none",
            "target_order_postprocessing": enforce_order,
            "research_valid": (
                not bool(config["train"].get("smoke_test", False))
                and len(result) == 862
                and result["plan"].nunique() == 117
            ),
        },
    )
    write_json(
        output_dir / "checkpoint_ref.json",
        {"run_id": run_dir.name, "models": model_bundle["checkpoints"]},
    )
    write_json(
        output_dir / "dataset_ref.json",
        {
            "dataset_id": bundle.source_manifest["dataset_id"],
            "test_csv": bundle.split_files["test"],
        },
    )
    run_metrics_path = run_dir / "metrics.json"
    run_metrics = read_json(run_metrics_path)
    run_metrics["final_test"] = metrics
    write_json(run_metrics_path, run_metrics)
    run_manifest = read_json(run_dir / "run_manifest.json")
    run_manifest["status"] = "evaluated"
    run_manifest["research_valid"] = bool(
        not config["train"].get("smoke_test", False)
        and len(result) == 862
        and result["plan"].nunique() == 117
    )
    run_manifest["research_valid_reason"] = (
        "Canonical Data_Estimate_2 test split evaluated with full provenance."
        if run_manifest["research_valid"]
        else "Smoke test or canonical test inventory mismatch."
    )
    write_json(run_dir / "run_manifest.json", run_manifest)
    print(f"[AI_Estimate][XGBoost][Test] run={run_dir}")
    print(json.dumps(metrics, indent=2))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost travel-time run.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_train.json")))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.run_path or args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()

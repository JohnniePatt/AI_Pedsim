import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_data_bundle, inverse_target_transform, read_json, write_json
from model import build_model, choose_device
from train_time_estimator import ordered_time_predictions


def latest_checkpoint(output_root):
    output_root = Path(output_root)
    candidates = sorted(output_root.glob("run_*/best_result.pth"))
    if not candidates:
        candidates = sorted(output_root.glob("*/best_result.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_result.pth found under {output_root}")
    return candidates[-1]


def resolve_output_root(config, config_path):
    raw_root = config.get("output", {}).get("root", "../../AI_result/Method_MLP_PyTorch/outputs")
    path = Path(raw_root)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


@torch.no_grad()
def predict_dataset(model, dataset, checkpoint, device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    preds = []
    truths = []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        truths.append(y.numpy())
    pred_seconds = ordered_time_predictions(inverse_target_transform(np.vstack(preds), checkpoint["scaler"]))
    true_seconds = inverse_target_transform(np.vstack(truths), checkpoint["scaler"])
    return pred_seconds, true_seconds


def compute_metrics(pred, true, target_columns):
    error = pred - true
    mse = float(np.mean(error**2))
    metrics = {
        "rows": int(len(pred)),
        "mae_overall_s": float(np.mean(np.abs(error))),
        "mse_overall_s": mse,
        "rmse_overall_s": float(np.sqrt(mse)),
    }
    for idx, name in enumerate(target_columns):
        target_mse = float(np.mean(error[:, idx] ** 2))
        metrics[f"mae_{name}"] = float(np.mean(np.abs(error[:, idx])))
        metrics[f"mse_{name}"] = target_mse
        metrics[f"rmse_{name}"] = float(np.sqrt(target_mse))
    return metrics


def test(config_path, checkpoint_path=None, output_dir=None):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    output_root = resolve_output_root(config, config_path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else latest_checkpoint(output_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    bundle = build_data_bundle(config, config_path)
    if list(bundle.feature_columns) != list(checkpoint["feature_columns"]):
        meta = bundle.test.meta.copy()
        if "observed_agents" in checkpoint["feature_columns"] and "observed_agents" not in meta.columns:
            meta["observed_agents"] = meta["computed_agents"]
        from dataset import TimeEstimateDataset
        bundle.test = TimeEstimateDataset(meta, checkpoint["feature_columns"], checkpoint["target_columns"], checkpoint["scaler"])

    device = choose_device(config)
    model = build_model(len(checkpoint["feature_columns"]), checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred, true = predict_dataset(model, bundle.test, checkpoint, device)
    metrics = compute_metrics(pred, true, checkpoint["target_columns"])
    output_dir = Path(output_dir) if output_dir else checkpoint_path.parent / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = bundle.test.meta.reset_index(drop=True).copy()
    for idx, target in enumerate(checkpoint["target_columns"]):
        meta[f"true_{target}"] = true[:, idx]
        meta[f"pred_{target}"] = pred[:, idx]
        meta[f"abs_error_{target}"] = np.abs(pred[:, idx] - true[:, idx])
    meta.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "test_metrics.json", metrics)
    print(f"[AI_Estimate][Test] checkpoint={checkpoint_path}")
    print(json.dumps(metrics, indent=2))
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AI_Estimate time estimator.")
    parser.add_argument("--config", default="AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    test(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()

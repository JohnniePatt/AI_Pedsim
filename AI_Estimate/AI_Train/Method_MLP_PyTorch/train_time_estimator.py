import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import build_data_bundle, inverse_target_transform, read_json, write_json
from model import build_model, choose_device


def resolve_output_root(config, config_path):
    raw_root = config.get("output", {}).get("root", "../../AI_result/Method_MLP_PyTorch/outputs")
    path = Path(raw_root)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def make_run_dir(config, config_path):
    output_root = resolve_output_root(config, config_path)
    run_name = config.get("output", {}).get("run_name", "auto")
    if run_name == "auto":
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ordered_time_predictions(values):
    values = np.asarray(values, dtype=np.float32)
    ordered = values.copy()
    low = np.minimum.reduce(values[:, [0, 1, 2]].T)
    high = np.maximum.reduce(values[:, [0, 1, 2]].T)
    middle = np.clip(values[:, 1], low, high)
    ordered[:, 0] = low
    ordered[:, 1] = middle
    ordered[:, 2] = high
    return ordered


def batch_to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        x, y = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(x)
        total_rows += len(x)
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, scaler, target_columns):
    model.eval()
    losses = []
    preds = []
    truths = []
    for batch in loader:
        x, y = batch_to_device(batch, device)
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(float(loss.item()) * len(x))
        preds.append(pred.cpu().numpy())
        truths.append(y.cpu().numpy())

    if not preds:
        return {"loss": 0.0, "rows": 0}

    pred_seconds = ordered_time_predictions(inverse_target_transform(np.vstack(preds), scaler))
    true_seconds = inverse_target_transform(np.vstack(truths), scaler)
    error = pred_seconds - true_seconds
    metrics = {
        "loss": sum(losses) / len(pred_seconds),
        "rows": int(len(pred_seconds)),
        "mae_overall_s": float(np.mean(np.abs(error))),
        "rmse_overall_s": float(np.sqrt(np.mean(error**2))),
    }
    for idx, name in enumerate(target_columns):
        metrics[f"mae_{name}"] = float(np.mean(np.abs(error[:, idx])))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(error[:, idx] ** 2)))
    return metrics


def save_checkpoint(path, model, optimizer, epoch, config, bundle, metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "feature_columns": bundle.feature_columns,
            "target_columns": bundle.target_columns,
            "scaler": bundle.scaler,
            "metrics": metrics,
        },
        path,
    )


def train(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    run_dir = make_run_dir(config, config_path)
    bundle = build_data_bundle(config, config_path)
    device = choose_device(config)

    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(bundle.test, batch_size=batch_size, shuffle=False)

    model = build_model(len(bundle.feature_columns), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )
    criterion = nn.MSELoss()
    epochs = int(train_cfg.get("epochs", 120))
    save_every_epoch = bool(train_cfg.get("save_every_epoch", True))

    write_json(run_dir / "config_used.json", config)
    write_json(
        run_dir / "dataset_manifest.json",
        {
            "rows": int(len(bundle.dataframe)),
            "train_rows": int(len(bundle.train)),
            "val_rows": int(len(bundle.val)),
            "test_rows": int(len(bundle.test)),
            "train_plans": len(bundle.splits["train"]),
            "val_plans": len(bundle.splits["val"]),
            "test_plans": len(bundle.splits["test"]),
            "feature_columns": bundle.feature_columns,
            "target_columns": bundle.target_columns,
            "splits": bundle.splits,
        },
    )
    bundle.dataframe.head(500).to_csv(run_dir / "dataset_preview.csv", index=False)

    history = []
    best_val = float("inf")
    best_metrics = {}

    print(f"[AI_Estimate][Train] run={run_dir}")
    print(f"[AI_Estimate][Train] device={device} rows={len(bundle.dataframe)} train={len(bundle.train)} val={len(bundle.val)} test={len(bundle.test)}")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate(model, train_loader, criterion, device, bundle.scaler, bundle.target_columns)
        val_metrics = evaluate(model, val_loader, criterion, device, bundle.scaler, bundle.target_columns)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss", 0.0),
            "train_mae_overall_s": train_metrics.get("mae_overall_s", 0.0),
            "val_mae_overall_s": val_metrics.get("mae_overall_s", 0.0),
            "val_rmse_overall_s": val_metrics.get("rmse_overall_s", 0.0),
        }
        for key, value in val_metrics.items():
            if key.startswith("mae_") and key != "mae_overall_s":
                row[f"val_{key}"] = value
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

        val_score = val_metrics.get("mae_overall_s", float("inf"))
        checkpoint_metrics = {"train": train_metrics, "val": val_metrics}
        if save_every_epoch:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, config, bundle, checkpoint_metrics)
        if val_score < best_val:
            best_val = val_score
            best_metrics = checkpoint_metrics
            save_checkpoint(run_dir / "best_result.pth", model, optimizer, epoch, config, bundle, checkpoint_metrics)

        print(
            f"[Epoch {epoch:03d}/{epochs}] train_loss={train_loss:.5f} "
            f"val_mae={val_metrics.get('mae_overall_s', 0.0):.3f}s "
            f"val_rmse={val_metrics.get('rmse_overall_s', 0.0):.3f}s"
        )

    final_test_metrics = evaluate(model, test_loader, criterion, device, bundle.scaler, bundle.target_columns)
    write_json(run_dir / "metrics.json", {"best": best_metrics, "final_test": final_test_metrics})
    print(f"[AI_Estimate][Train] best_val_mae={best_val:.3f}s")
    print(f"[AI_Estimate][Train] final_test_mae={final_test_metrics.get('mae_overall_s', 0.0):.3f}s")
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI_Estimate time estimator.")
    parser.add_argument("--config", default="AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

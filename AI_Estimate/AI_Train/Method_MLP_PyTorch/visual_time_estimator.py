import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def latest_run(output_root):
    output_root = Path(output_root)
    runs = [path for path in output_root.iterdir() if path.is_dir()] if output_root.exists() else []
    runs = [path for path in runs if (path / "training_history.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"No training runs found under {output_root}")
    return sorted(runs, key=lambda path: path.stat().st_mtime)[-1]


def plot_training_curves(run_dir):
    history_path = run_dir / "training_history.csv"
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    if "train_mae_overall_s" in history:
        ax.plot(history["epoch"], history["train_mae_overall_s"], label="Train MAE", linewidth=2)
    if "val_mae_overall_s" in history:
        ax.plot(history["epoch"], history["val_mae_overall_s"], label="Val MAE", linewidth=2)
    if "val_rmse_overall_s" in history:
        ax.plot(history["epoch"], history["val_rmse_overall_s"], label="Val RMSE", linewidth=2, alpha=0.75)
    ax.set_title("AI_Estimate training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = run_dir / "training_curves.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_prediction_scatter(run_dir):
    predictions_path = run_dir / "test_eval" / "predictions.csv"
    if not predictions_path.exists():
        return None

    df = pd.read_csv(predictions_path)
    pairs = [
        ("min_agent_time_s", "Fastest agent time"),
        ("mean_agent_time_s", "Average agent time"),
        ("max_agent_time_s", "Slowest agent time"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (target, title) in zip(axes, pairs):
        true_col = f"true_{target}"
        pred_col = f"pred_{target}"
        if true_col not in df or pred_col not in df:
            ax.axis("off")
            continue
        ax.scatter(df[true_col], df[pred_col], s=18, alpha=0.7)
        low = min(df[true_col].min(), df[pred_col].min())
        high = max(df[true_col].max(), df[pred_col].max())
        ax.plot([low, high], [low, high], color="#ef4444", linewidth=2, label="Perfect")
        ax.set_title(title)
        ax.set_xlabel("True (s)")
        ax.set_ylabel("Predicted (s)")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = run_dir / "prediction_scatter.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_error_histogram(run_dir):
    predictions_path = run_dir / "test_eval" / "predictions.csv"
    if not predictions_path.exists():
        return None

    df = pd.read_csv(predictions_path)
    columns = [col for col in df.columns if col.startswith("abs_error_")]
    if not columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in columns:
        label = col.replace("abs_error_", "").replace("_agent_time_s", "")
        ax.hist(df[col], bins=24, alpha=0.45, label=label)
    ax.set_title("Absolute error distribution")
    ax.set_xlabel("Absolute error (s)")
    ax.set_ylabel("Rows")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = run_dir / "error_histogram.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def visualise(run_dir=None, output_root=None):
    if run_dir:
        run_dir = Path(run_dir).resolve()
    else:
        run_dir = latest_run(Path(output_root).resolve())
    outputs = [plot_training_curves(run_dir)]
    for maybe_path in [plot_prediction_scatter(run_dir), plot_error_histogram(run_dir)]:
        if maybe_path:
            outputs.append(maybe_path)
    print(f"[AI_Estimate][Visual] run={run_dir}")
    for path in outputs:
        print(f"[AI_Estimate][Visual] saved={path}")
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Create visual report for AI_Estimate runs.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-root", default="AI_Estimate/AI_result/Method_MLP_PyTorch/outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    visualise(args.run_dir, args.output_root)


if __name__ == "__main__":
    main()

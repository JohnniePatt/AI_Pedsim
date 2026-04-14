import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_run(output_root):
    output_root = Path(output_root)
    runs = [p for p in output_root.iterdir() if p.is_dir()] if output_root.exists() else []
    runs = [p for p in runs if (p / "training_history.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"No training runs found under {output_root}")
    return sorted(runs, key=lambda p: p.stat().st_mtime)[-1]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_training_curves(run_dir):
    history = pd.read_csv(run_dir / "training_history.csv")
    has_mae_s = "val_mae_overall_s" in history.columns

    n_panels = 2 if has_mae_s else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(10 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: MSE loss
    ax = axes[0]
    if "train_loss" in history.columns:
        ax.plot(history["epoch"], history["train_loss"], label="Train loss (MSE)", linewidth=2)
    if "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"],   label="Val loss (MSE)",   linewidth=2)
    ax.set_title("AI_Estimate Keras — Loss curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Panel 2: MAE in real seconds (from RealSecondsMAECallback)
    if has_mae_s:
        ax2 = axes[1]
        if "train_mae_overall_s" in history.columns:
            ax2.plot(history["epoch"], history["train_mae_overall_s"], label="Train MAE (s)", linewidth=2)
        ax2.plot(history["epoch"], history["val_mae_overall_s"], label="Val MAE (s)", linewidth=2)
        ax2.set_title("AI_Estimate Keras — MAE in seconds")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE (seconds)")
        ax2.grid(True, alpha=0.25)
        ax2.legend()

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
        ("min_agent_time_s",  "Fastest agent time"),
        ("mean_agent_time_s", "Average agent time"),
        ("max_agent_time_s",  "Slowest agent time"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (target, title) in zip(axes, pairs):
        true_col = f"true_{target}"
        pred_col = f"pred_{target}"
        if true_col not in df or pred_col not in df:
            ax.axis("off")
            continue
        ax.scatter(df[true_col], df[pred_col], s=18, alpha=0.7)
        low  = min(df[true_col].min(), df[pred_col].min())
        high = max(df[true_col].max(), df[pred_col].max())
        ax.plot([low, high], [low, high], color="#ef4444", linewidth=2)
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
    columns = [c for c in df.columns if c.startswith("abs_error_")]
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


# ---------------------------------------------------------------------------
# Main visualise function
# ---------------------------------------------------------------------------

def visualise(run_dir=None, output_root=None):
    run_dir = Path(run_dir).resolve() if run_dir else _latest_run(Path(output_root).resolve())
    outputs = [plot_training_curves(run_dir)]
    for maybe_path in [plot_prediction_scatter(run_dir), plot_error_histogram(run_dir)]:
        if maybe_path:
            outputs.append(maybe_path)
    print(f"[AI_Estimate][Keras][Visual] run={run_dir}")
    for path in outputs:
        print(f"[AI_Estimate][Keras][Visual] saved={path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Create visual report for AI_Estimate Keras runs.")
    parser.add_argument("--run-dir",     default=None)
    parser.add_argument("--output-root", default="AI_Estimate/AI_result/Method_MLP_Keras/outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    visualise(args.run_dir, args.output_root)


if __name__ == "__main__":
    main()

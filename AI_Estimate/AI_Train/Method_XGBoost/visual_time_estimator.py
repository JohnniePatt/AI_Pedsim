import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGETS = [
    ("min_agent_time_s", "Fastest"),
    ("mean_agent_time_s", "Average"),
    ("max_agent_time_s", "Slowest"),
]


def visualise(run_dir):
    run_dir = Path(run_dir).resolve()
    predictions = pd.read_csv(run_dir / "test_eval" / "predictions.csv")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, (target, title) in zip(axes, TARGETS):
        true = predictions[f"true_{target}"]
        pred = predictions[f"pred_{target}"]
        low = min(true.min(), pred.min())
        high = max(true.max(), pred.max())
        axis.scatter(true, pred, s=16, alpha=0.6)
        axis.plot([low, high], [low, high], color="#ef4444", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Ground truth (s)")
        axis.set_ylabel("Predicted time (s)")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    output = run_dir / "prediction_scatter.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[AI_Estimate][XGBoost][Visual] saved={output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    visualise(args.run_dir)


if __name__ == "__main__":
    main()

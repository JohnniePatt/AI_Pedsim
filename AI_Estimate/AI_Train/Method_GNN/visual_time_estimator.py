import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Visualization Logic
# ---------------------------------------------------------------------------

def visualize(run_dir):
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists(): return

    # 1. Plot Training History
    history_path = run_dir / "history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            history = json.load(f)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("GNN Training History")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(run_dir / "training_history.png")
        plt.close()

    # 2. Show Metrics
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print("\n[AI_Estimate][GNN][Visual] Metrics:")
        for i, target in enumerate(metrics["targets"]):
            print(f"  {target}: MAE={metrics['mae'][i]:.4f}")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    visualize(args.run_dir)

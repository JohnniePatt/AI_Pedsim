import argparse
import json
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGETS = [
    ("min_agent_time_s", "min"),
    ("mean_agent_time_s", "mean"),
    ("max_agent_time_s", "max"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_run(output_root):
    output_root = Path(output_root)
    runs = [path for path in output_root.iterdir() if path.is_dir()] if output_root.exists() else []
    runs = [path for path in runs if (path / "test_eval" / "predictions.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"No run with predictions.csv found under {output_root}")
    return sorted(runs, key=lambda path: path.stat().st_mtime)[-1]

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def summarize_predictions(predictions_path):
    df = pd.read_csv(predictions_path)
    if "trajectory_file" not in df.columns:
        # Fallback to group by plan/route if trajectory_file is missing
        group_col = "plan"
    else:
        group_col = "trajectory_file"

    rows = []
    for group_val, group in df.groupby(group_col, dropna=False):
        file_name = Path(str(group_val)).name if pd.notna(group_val) else "unknown_file"
        plan_name = str(group["plan"].iloc[0]) if "plan" in group.columns and len(group) else "unknown_plan"
        record = {
            "file_name": file_name,
            "plan": plan_name,
            "display_name": f"{plan_name} | {file_name}",
            "rows": int(len(group)),
        }
        for target, key in TARGETS:
            true_col = f"true_{target}"
            pred_col = f"pred_{target}"
            err_col = f"abs_error_{target}"
            record[f"{key}_real_s"] = float(group[true_col].mean()) if true_col in group else 0.0
            record[f"{key}_ai_s"] = float(group[pred_col].mean()) if pred_col in group else 0.0
            record[f"{key}_error_s"] = float(group[err_col].mean()) if err_col in group else 0.0
        rows.append(record)
    rows = sorted(rows, key=lambda item: item["file_name"])
    return rows

def build_report(run_dir):
    run_dir = Path(run_dir).resolve()
    predictions_path = run_dir / "test_eval" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv in {predictions_path.parent}")
    
    rows = summarize_predictions(predictions_path)
    report = {
        "run_dir": str(run_dir),
        "predictions_csv": str(predictions_path),
        "files": len(rows),
        "rows": rows,
    }
    
    out_path = run_dir / "test_eval" / "sample_time_report.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    
    print(f"[AI_Estimate][GNN][Sample] report saved: {out_path}")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-root", default="../../../AI_result/Method_GNN/outputs")
    args = parser.parse_args()
    
    try:
        run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run(Path(args.output_root).resolve())
        build_report(run_dir)
    except Exception as e:
        print(f"[AI_Estimate][GNN][Sample] Error: {e}")

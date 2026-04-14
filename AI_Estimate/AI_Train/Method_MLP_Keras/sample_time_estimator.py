import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = [
    ("min_agent_time_s", "min"),
    ("mean_agent_time_s", "mean"),
    ("max_agent_time_s", "max"),
]


def latest_run(output_root):
    output_root = Path(output_root)
    runs = [path for path in output_root.iterdir() if path.is_dir()] if output_root.exists() else []
    runs = [path for path in runs if (path / "test_eval" / "predictions.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"No run with predictions.csv found under {output_root}")
    return sorted(runs, key=lambda path: path.stat().st_mtime)[-1]


def summarize_predictions(predictions_path):
    df = pd.read_csv(predictions_path)
    if "trajectory_file" not in df.columns:
        raise ValueError("predictions.csv must contain 'trajectory_file' column")

    rows = []
    for traj_file, group in df.groupby("trajectory_file", dropna=False):
        file_name = Path(str(traj_file)).name if pd.notna(traj_file) else "unknown_file"
        plan_name = str(group["plan"].iloc[0]) if "plan" in group.columns and len(group) else "unknown_plan"
        record = {
            "trajectory_file": str(traj_file),
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
    out_table_path = run_dir / "test_eval" / "sample_time_report.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    pd.DataFrame(rows).to_csv(out_table_path, index=False)
    print(f"[AI_Estimate][Sample] report={out_path}")
    print(f"[AI_Estimate][Sample] table={out_table_path}")
    print(f"[AI_Estimate][Sample] files={len(rows)}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-file sample time summary from test predictions.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-root", default="AI_Estimate/AI_result/Method_MLP_Keras/outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run(Path(args.output_root).resolve())
    build_report(run_dir)


if __name__ == "__main__":
    main()

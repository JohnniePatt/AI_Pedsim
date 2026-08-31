import argparse
import json
from pathlib import Path

import pandas as pd


def create_report(run_dir):
    run_dir = Path(run_dir).resolve()
    frame = pd.read_csv(run_dir / "test_eval" / "predictions.csv")
    rows = []
    group_column = "trajectory_file" if "trajectory_file" in frame else "plan"
    for name, group in frame.groupby(group_column, dropna=False):
        rows.append(
            {
                "file_name": str(name),
                "display_name": str(name),
                "rows": len(group),
                "min_real_s": float(group["true_min_agent_time_s"].mean()),
                "min_ai_s": float(group["pred_min_agent_time_s"].mean()),
                "min_error_s": float(group["abs_error_min_agent_time_s"].mean()),
                "mean_real_s": float(group["true_mean_agent_time_s"].mean()),
                "mean_ai_s": float(group["pred_mean_agent_time_s"].mean()),
                "mean_error_s": float(group["abs_error_mean_agent_time_s"].mean()),
                "max_real_s": float(group["true_max_agent_time_s"].mean()),
                "max_ai_s": float(group["pred_max_agent_time_s"].mean()),
                "max_error_s": float(group["abs_error_max_agent_time_s"].mean()),
            }
        )
    report = {"files": len(rows), "rows": rows}
    output = run_dir / "test_eval" / "sample_time_report.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"[AI_Estimate][XGBoost][Sample] report={output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    create_report(args.run_dir)


if __name__ == "__main__":
    main()

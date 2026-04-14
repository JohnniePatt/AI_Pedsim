import argparse
import sys
from pathlib import Path

import pandas as pd

TRAIN_ROOT = Path(__file__).resolve().parents[1] / "AI_Train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from dataset import (
    feature_columns_from_config,
    formatted_manifest_path,
    formatted_split_path,
    load_joined_dataframe,
    read_json,
    resolve_data_estimate_root,
    split_dir_name,
    split_plans,
    write_json,
)


def split_dataframe(df, splits):
    frames = {}
    for split_name, plan_names in splits.items():
        frame = df[df["plan"].isin(plan_names)].copy()
        frame.insert(0, "split", split_name)
        frames[split_name] = frame.reset_index(drop=True)
    return frames


def write_split_files(frames, config, config_path):
    output_paths = {}
    for split_name, frame in frames.items():
        path = formatted_split_path(config, config_path, split_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        output_paths[split_name] = str(path)
    return output_paths


def format_data_estimate(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    output_root = resolve_data_estimate_root(config, config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_joined_dataframe(config, config_path)
    feature_columns = feature_columns_from_config(df, config)
    target_columns = list(config["features"]["target"])
    splits = split_plans(df, config)
    frames = split_dataframe(df, splits)
    output_paths = write_split_files(frames, config, config_path)

    combined_path = output_root / "all_data_estimate.csv"
    pd.concat(frames.values(), ignore_index=True).to_csv(combined_path, index=False)

    manifest = {
        "output_root": str(output_root),
        "all_data_estimate_csv": str(combined_path),
        "source_rows": int(len(df)),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "splits": {
            split_name: {
                "directory": split_dir_name(split_name),
                "csv": output_paths[split_name],
                "rows": int(len(frame)),
                "plans": int(frame["plan"].nunique()),
            }
            for split_name, frame in frames.items()
        },
        "split_plan_names": splits,
    }
    write_json(formatted_manifest_path(config, config_path), manifest)

    print(f"[AI_Estimate][Format] output={output_root}")
    print(
        "[AI_Estimate][Format] rows "
        f"train={len(frames['train'])} val={len(frames['val'])} test={len(frames['test'])}"
    )
    print(
        "[AI_Estimate][Format] plans "
        f"train={frames['train']['plan'].nunique()} "
        f"val={frames['val']['plan'].nunique()} "
        f"test={frames['test']['plan'].nunique()}"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Format AI_Estimate data into separate Train/Val/Test folders.")
    parser.add_argument("--config", default="AI_Estimate/AI_Train/Method_MLP_PyTorch/config_train.json")
    return parser.parse_args()


def main():
    args = parse_args()
    format_data_estimate(args.config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from rollout import rollout_case


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def make_rollout_name(split: str, plan_name: str, sqlite_stem: str, suffix: str = "") -> str:
    base = f"{split}_{plan_name}_{sqlite_stem}"
    return f"{base}_{suffix}" if suffix else base


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GridSocialPolicyNet rollout on multiple manifest cases.")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stop-threshold", type=float, default=0.8)
    parser.add_argument("--crop-size", type=int, default=33)
    parser.add_argument("--wait-logit-bias", type=float, default=0.0)
    parser.add_argument("--disable-wait", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    manifest_path = args.dataset_root / "manifest_trajectory_grid.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    split_df = manifest[manifest["split"] == args.split].reset_index(drop=True)
    if split_df.empty:
        raise RuntimeError(f"No cases for split={args.split}")

    start = max(int(args.start_index), 0)
    end = min(start + max(int(args.sample_count), 1), len(split_df))
    selected = split_df.iloc[start:end].reset_index(drop=True)

    batch_rows = []
    for idx, row in selected.iterrows():
        rollout_name = make_rollout_name(args.split, str(row["plan_name"]), str(row["sqlite_stem"]), args.suffix)
        output_dir = args.output_root / rollout_name
        print(f"[{idx + 1}/{len(selected)}] rollout {row['plan_name']}/{row['sqlite_stem']} -> {output_dir}")
        summary = rollout_case(
            checkpoint_path=args.checkpoint.resolve(),
            input_dir=pathlib.Path(row["input_dir"]).resolve(),
            output_dir=output_dir.resolve(),
            max_steps=int(args.max_steps),
            stop_threshold=float(args.stop_threshold),
            crop_size=int(args.crop_size),
            wait_logit_bias=float(args.wait_logit_bias),
            disable_wait=bool(args.disable_wait),
        )
        batch_rows.append(
            {
                "split": args.split,
                "plan_name": row["plan_name"],
                "sqlite_stem": row["sqlite_stem"],
                "output_dir": str(output_dir.resolve()),
                **summary,
            }
        )

    batch_df = pd.DataFrame(batch_rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(args.output_root / f"batch_rollout_{args.split}.csv", index=False)
    write_json(
        args.output_root / f"batch_rollout_{args.split}.json",
        {
            "split": args.split,
            "sample_count": int(len(selected)),
            "start_index": start,
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "disable_wait": bool(args.disable_wait),
            "wait_logit_bias": float(args.wait_logit_bias),
        },
    )
    print(f"[DONE] wrote {len(selected)} rollout samples")
    print(f"[DONE] summary_csv={args.output_root / f'batch_rollout_{args.split}.csv'}")


if __name__ == "__main__":
    main()

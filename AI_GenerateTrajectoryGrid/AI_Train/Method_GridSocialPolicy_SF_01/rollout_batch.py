from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import pandas as pd
import torch

from rollout import rollout_case

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
BASELINE_TRAIN_DIR = PROJECT_ROOT / "AI_GenerateTimeseries" / "AI_Train"
if str(BASELINE_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_TRAIN_DIR))
from baseline_output import (  # noqa: E402
    create_evaluation_layout,
    finalize_evaluation,
    write_case_prediction,
)


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
    parser.add_argument("--output-root", type=pathlib.Path, default=None,
                        help="Legacy output directory. Omit to write the standard evaluations/ layout beside the checkpoint.")
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

    checkpoint = args.checkpoint.resolve()
    run_dir = checkpoint.parents[1]
    saved = torch.load(checkpoint, map_location="cpu")
    checkpoint_cfg = saved.get("config", {}) if isinstance(saved, dict) else {}
    trained_dataset = pathlib.Path(str(checkpoint_cfg.get("dataset_root", ""))).name
    compatibility_ok = bool(trained_dataset) and trained_dataset.casefold() == args.dataset_root.resolve().name.casefold()
    eval_layout = create_evaluation_layout(
        run_dir,
        method_id="Method_GridSocialPolicy_SF_01",
        dataset_id="housegan_canonical_imagebase_split_v1",
        split=args.split,
        protocol_version="v1",
        checkpoint_path=checkpoint,
        evaluation_config={
            "max_steps": args.max_steps,
            "stop_threshold": args.stop_threshold,
            "crop_size": args.crop_size,
            "wait_logit_bias": args.wait_logit_bias,
            "disable_wait": args.disable_wait,
            "sample_count": args.sample_count,
            "start_index": args.start_index,
        },
        dataset_manifest=manifest_path,
        constraint_mode="grid_walkability_and_collision_executor",
        compatibility_ok=compatibility_ok,
        invalid_reason=None if compatibility_ok else "checkpoint/dataset provenance mismatch or unavailable",
    )
    standard_output = args.output_root is None
    output_root = eval_layout.predictions if standard_output else args.output_root.resolve()

    batch_rows = []
    for idx, row in selected.iterrows():
        rollout_name = make_rollout_name(args.split, str(row["plan_name"]), str(row["sqlite_stem"]), args.suffix)
        case_id = f"{row['plan_name']}_{str(row['sqlite_stem']).removeprefix('plan_sim_')}"
        output_dir = output_root / case_id
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
        if standard_output:
            rollout_path = output_dir / "rollout.parquet"
            rollout_df = pd.read_parquet(rollout_path)
            rollout_df.insert(0, "split", args.split)
            rollout_df.insert(0, "case_id", case_id)
            rollout_df["is_active"] = ~rollout_df["stopped"].astype(bool)
            write_case_prediction(
                eval_layout,
                case_id,
                rollout_df,
                variant="constrained",
            )
            rollout_path.unlink()
            preview_src = output_dir / "samples" / "rollout_preview.png"
            if preview_src.exists():
                preview_dir = eval_layout.previews / case_id
                preview_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(preview_src, preview_dir / "constrained_rollout.png")
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
    summary_root = eval_layout.metrics if standard_output else output_root
    summary_root.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(summary_root / "per_case_metrics.csv", index=False)
    if standard_output:
        total_rows = max(float(batch_df.get("rows", pd.Series(dtype=float)).sum()), 1.0)
        blocked = float(batch_df.get("blocked_by_wall_steps", pd.Series(dtype=float)).sum()) + float(
            batch_df.get("blocked_by_collision_steps", pd.Series(dtype=float)).sum()
        )
        decisions = max(float(batch_df.get("move_decisions", pd.Series(dtype=float)).sum()), 1.0)
        pd.DataFrame([{
            "method_id": "Method_GridSocialPolicy_SF_01",
            "variant": "constrained",
            "seed": int(saved.get("config", {}).get("seed", 42)) if isinstance(saved, dict) else 42,
            "out_of_bounds_rate": 1.0 - float(batch_df["walkable_ratio"].mean()),
            "collision_exposure_rate": float(batch_df["collision_count"].sum()) / total_rows,
            "invalid_step_rate": 1.0 - float(batch_df["walkable_ratio"].mean()),
            "constraint_intervention_rate": blocked / decisions,
        }]).to_csv(eval_layout.metrics / "summary_metrics.csv", index=False)
    write_json(
        eval_layout.reports / "evaluation_summary.json" if standard_output else output_root / f"batch_rollout_{args.split}.json",
        {
            "split": args.split,
            "sample_count": int(len(selected)),
            "start_index": start,
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "output_root": str(output_root),
            "disable_wait": bool(args.disable_wait),
            "wait_logit_bias": float(args.wait_logit_bias),
        },
    )
    research_valid = finalize_evaluation(
        eval_layout,
        case_ids=(
            [f"{row['plan_name']}_{str(row['sqlite_stem']).removeprefix('plan_sim_')}" for _, row in selected.iterrows()]
            if standard_output else []
        ),
        floorplan_ids=selected["plan_name"].astype(str) if standard_output else [],
        compatibility_ok=compatibility_ok,
        canonical_test_required=True,
        additional_failures=(
            (["legacy --output-root mode is not a standard evaluation"] if not standard_output else [])
            + (["partial rollout selection is framing only"] if len(selected) != len(split_df) else [])
        ),
    )
    print(f"[DONE] wrote {len(selected)} rollout samples")
    print(f"[DONE] research_valid={research_valid}")
    print(f"[DONE] summary_csv={summary_root / 'per_case_metrics.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
COMMON_DIR = PROJECT_ROOT / "AI_GenerateTimeseries" / "AI_Train"
sys.path.insert(0, str(COMMON_DIR))

from pipeline_common import (  # noqa: E402
    load_json, parser, require_sf_ready, resolve_checkpoint, resolve_run, run,
    wants_evaluate, wants_train,
)


def main() -> None:
    cli = parser("Plan, train, or evaluate Method_GridSocialPolicy_SF_01.")
    cli.add_argument("--config-train", type=pathlib.Path, default=HERE / "config_train.json")
    cli.add_argument("--sample-count", type=int, default=0,
                     help="0 evaluates every case in the selected split.")
    cli.add_argument("--split", choices=("train", "val", "test"), default="test")
    cli.add_argument("--max-steps", type=int, default=1000)
    args = cli.parse_args()
    config = load_json(args.config_train.resolve())
    require_sf_ready(config, "Method_GridSocialPolicy_SF_01", dry_run=args.dry_run or args.stage == "plan")
    outputs = PROJECT_ROOT / "AI_GenerateTrajectoryGrid" / "AI_Result" / "Method_GridSocialPolicy_SF_01" / "outputs"
    dataset_root = pathlib.Path(config["dataset_root"]).resolve()

    print(f"[pipeline] method=Method_GridSocialPolicy_SF_01 stage={args.stage} outputs={outputs}")
    if wants_train(args.stage):
        run([args.python, HERE / "train_grid_policy.py", "--config", args.config_train.resolve()], cwd=HERE, dry_run=args.dry_run)
    if wants_evaluate(args.stage):
        run_dir = resolve_run(args.run_path, outputs)
        checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
        sample_count = args.sample_count
        if sample_count <= 0:
            import pandas as pd
            manifest = pd.read_csv(dataset_root / "manifest_trajectory_grid.csv", usecols=["split"])
            sample_count = int((manifest["split"] == args.split).sum())
        run([args.python, HERE / "rollout_batch.py", "--checkpoint", checkpoint,
             "--dataset-root", dataset_root, "--split", args.split,
             "--sample-count", sample_count, "--max-steps", args.max_steps], cwd=HERE, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
COMMON_DIR = PROJECT_ROOT / "AI_GenerateTimeseries" / "AI_Train"
sys.path.insert(0, str(COMMON_DIR))

from pipeline_common import (  # noqa: E402
    choose_operation, choose_run, confirm_full_training, load_json, parser,
    print_available_runs, require_sf_ready, resolve_checkpoint, resolve_run,
    run, wants_evaluate, wants_train,
)


TRAIN_PROFILES = {
    "fast": HERE / "config_fast.json",
    "quarter": HERE / "config_quarter_plan.json",
    "full": HERE / "config_full.json",
}


def choose_train_profile(input_func=input) -> str:
    print("\nSelect GridSocialPolicy-SF training profile:")
    print("  1) fast - quick debug/sanity training (recommended)")
    print("  2) quarter - rotate 25% of train plans each epoch")
    print("  3) full - full research-scale training")
    while True:
        try:
            choice = input_func("Choose 1, 2, or 3 [1]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[pipeline] Cancelled; training was not started.")
            return "cancel"
        if not choice or choice in {"1", "fast"}:
            return "fast"
        if choice in {"2", "quarter", "quarter-plan", "quarter_plan"}:
            return "quarter"
        if choice in {"3", "full"}:
            return "full"
        print("Please enter 1 for fast, 2 for quarter, or 3 for full.")


def choose_train_config(args) -> pathlib.Path:
    if args.config_train is not None:
        return args.config_train
    if args.profile is not None:
        return TRAIN_PROFILES[args.profile]
    return TRAIN_PROFILES["fast"]


def main() -> None:
    cli = parser("Plan, train, or evaluate Method_GridSocialPolicy_SF_01.")
    cli.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default=None,
                     help="Training profile shortcut. If omitted in an interactive train run, a menu is shown.")
    cli.add_argument("--config-train", type=pathlib.Path, default=None,
                     help="Explicit training config path. Overrides --profile and the interactive menu.")
    cli.add_argument("--sample-count", type=int, default=0,
                     help="0 evaluates every case in the selected split.")
    cli.add_argument("--split", choices=("train", "val", "test"), default="test")
    cli.add_argument("--max-steps", type=int, default=1000)
    args = cli.parse_args()
    outputs = PROJECT_ROOT / "AI_GenerateTrajectoryGrid" / "AI_Result" / "Method_GridSocialPolicy_SF_01" / "outputs"
    stage_was_provided = any(item == "--stage" or item.startswith("--stage=") for item in sys.argv[1:])
    interactive = len(sys.argv) == 1 and sys.stdin.isatty()
    if not stage_was_provided and (args.profile is not None or args.config_train is not None):
        args.stage = "train"
    action = None
    if interactive:
        action = choose_operation("Social-Force-Conditioned Discrete Grid Policy")
        if action == "exit":
            return
        if action == "runs":
            print_available_runs(outputs)
            return
        args.stage = {"check": "plan", "smoke": "all", "train": "train",
                      "evaluate": "evaluate", "all": "all"}[action]
        if action == "smoke":
            args.config_train = HERE / "config_smoke.json"
            args.sample_count = 1
            args.max_steps = 20
        elif action in {"train", "all"}:
            args.profile = choose_train_profile()
            if args.profile == "cancel":
                return
        if action == "evaluate":
            args.run_path = choose_run(outputs)
            if args.run_path is None:
                return
    train_config = choose_train_config(args)
    if interactive and action in {"train", "all"} and args.profile in {"quarter", "full"}:
        if not confirm_full_training("Method_GridSocialPolicy_SF_01", train_config):
            return
    config = load_json(train_config.resolve())
    require_sf_ready(config, "Method_GridSocialPolicy_SF_01", dry_run=args.dry_run or args.stage == "plan")
    dataset_root = pathlib.Path(config["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (train_config.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    print(
        f"[pipeline] method=Method_GridSocialPolicy_SF_01 stage={args.stage} "
        f"train_config={train_config.resolve()} outputs={outputs}",
        flush=True,
    )
    if wants_train(args.stage):
        run([args.python, HERE / "train_grid_policy.py", "--config", train_config.resolve()], cwd=HERE, dry_run=args.dry_run)
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

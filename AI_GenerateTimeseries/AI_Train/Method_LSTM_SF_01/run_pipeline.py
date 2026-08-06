from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

from pipeline_common import (  # noqa: E402
    load_json, parser, require_sf_ready, resolve_run, run, wants_evaluate, wants_train,
)


TRAIN_PROFILES = {
    "fast": HERE / "config_active.json",
    "full": HERE / "config_full.json",
}


def choose_train_config(args) -> pathlib.Path:
    if args.config_train is not None:
        return args.config_train
    if args.profile is not None:
        return TRAIN_PROFILES[args.profile]
    if not wants_train(args.stage) or not sys.stdin.isatty():
        return TRAIN_PROFILES["fast"]

    print("\nSelect LSTM-SF training profile:")
    print("  1) fast - quick debug/sanity training (recommended)")
    print("  2) full - full research-scale training")
    choice = input("Choose 1 or 2 [1]: ").strip()
    if choice in {"2", "full", "Full", "FULL"}:
        return TRAIN_PROFILES["full"]
    return TRAIN_PROFILES["fast"]


def main() -> None:
    cli = parser("Train, plan, or evaluate Method_LSTM_SF_01.", default_stage="train")
    cli.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default=None,
                     help="Training profile shortcut. If omitted in an interactive train run, a menu is shown.")
    cli.add_argument("--config-train", type=pathlib.Path, default=None,
                     help="Explicit training config path. Overrides --profile and the interactive menu.")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_test.json")
    args = cli.parse_args()
    train_config = choose_train_config(args)
    outputs = HERE.parents[1] / "AI_Result" / "Method_LSTM_SF_01" / "outputs"
    train_cfg = load_json(train_config.resolve())
    require_sf_ready(train_cfg, "Method_LSTM_SF_01", dry_run=args.dry_run or args.stage == "plan")

    print(
        f"[pipeline] method=Method_LSTM_SF_01 stage={args.stage} "
        f"train_config={train_config.resolve()} outputs={outputs}",
        flush=True,
    )
    if wants_train(args.stage):
        run([args.python, AI_TRAIN / "train_joint_sf.py", "--config", train_config.resolve(),
             "--method-id", "Method_LSTM_SF_01", "--architecture", "lstm"], cwd=AI_TRAIN, dry_run=args.dry_run)
    if wants_evaluate(args.stage):
        run_dir = resolve_run(args.run_path, outputs)
        checkpoint = args.checkpoint or (run_dir / "checkpoints" / "best_model.pth")
        run([args.python, AI_TRAIN / "evaluate_joint_sf.py", "--config", args.config_test.resolve(),
             "--method-id", "Method_LSTM_SF_01", "--architecture", "lstm",
             "--run-path", run_dir, "--checkpoint", checkpoint], cwd=AI_TRAIN, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

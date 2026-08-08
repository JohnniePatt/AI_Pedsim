from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

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
    print("\nSelect Transformer-SF training profile:")
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
    cli = parser("Check, train, or evaluate Method_Transformer_SF_01.", default_stage="plan")
    cli.add_argument("--profile", choices=tuple(TRAIN_PROFILES), default=None,
                     help="Training profile shortcut. If omitted in an interactive train run, a menu is shown.")
    cli.add_argument("--config-train", type=pathlib.Path, default=None,
                     help="Explicit training config path. Overrides --profile and the interactive menu.")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_test.json")
    args = cli.parse_args()
    outputs = HERE.parents[1] / "AI_Result" / "Method_Transformer_SF_01" / "outputs"
    stage_was_provided = any(item == "--stage" or item.startswith("--stage=") for item in sys.argv[1:])
    interactive = len(sys.argv) == 1 and sys.stdin.isatty()
    if not stage_was_provided and (args.profile is not None or args.config_train is not None):
        args.stage = "train"
    action = None
    if interactive:
        action = choose_operation("Social-Force-Informed Joint Multi-Agent Transformer")
        if action == "exit":
            return
        if action == "runs":
            print_available_runs(outputs)
            return
        args.stage = {"check": "plan", "smoke": "all", "train": "train",
                      "evaluate": "evaluate", "all": "all"}[action]
        if action == "smoke":
            args.config_train = HERE / "config_smoke.json"
            args.config_test = HERE / "config_test_smoke.json"
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
        if not confirm_full_training("Method_Transformer_SF_01", train_config):
            return
    train_cfg = load_json(train_config.resolve())
    require_sf_ready(train_cfg, "Method_Transformer_SF_01", dry_run=args.dry_run or args.stage == "plan")

    print(
        f"[pipeline] method=Method_Transformer_SF_01 stage={args.stage} "
        f"train_config={train_config.resolve()} outputs={outputs}",
        flush=True,
    )
    if wants_train(args.stage):
        run([args.python, AI_TRAIN / "train_joint_sf.py", "--config", train_config.resolve(),
             "--method-id", "Method_Transformer_SF_01", "--architecture", "transformer"], cwd=AI_TRAIN, dry_run=args.dry_run)
    if wants_evaluate(args.stage):
        run_dir = resolve_run(args.run_path, outputs)
        checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
        command = [args.python, AI_TRAIN / "evaluate_joint_sf.py", "--config", args.config_test.resolve(),
                   "--method-id", "Method_Transformer_SF_01", "--architecture", "transformer",
                   "--run-path", run_dir, "--checkpoint", checkpoint]
        run(command, cwd=AI_TRAIN, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

from pipeline_common import (  # noqa: E402
    choose_operation, choose_run, confirm_full_training, load_json, parser,
    print_available_runs, require_sf_ready, resolve_run, run, wants_evaluate,
    wants_train,
)


def main() -> None:
    cli = parser("Plan, train, or evaluate Method_SGAN_SF_01.")
    cli.add_argument("--config-train", type=pathlib.Path, default=HERE / "config_train.json")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_test.json")
    args = cli.parse_args()
    outputs = HERE.parents[1] / "AI_Result" / "Method_SGAN_SF_01" / "outputs"
    interactive = len(sys.argv) == 1
    action = None
    if interactive:
        action = choose_operation("Social-Force-Informed Joint Multi-Agent Social GAN")
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
            if not confirm_full_training("Method_SGAN_SF_01", args.config_train):
                return
        if action == "evaluate":
            args.run_path = choose_run(outputs)
            if args.run_path is None:
                return
    train_cfg = load_json(args.config_train.resolve())
    require_sf_ready(train_cfg, "Method_SGAN_SF_01", dry_run=args.dry_run or args.stage == "plan")

    print(f"[pipeline] method=Method_SGAN_SF_01 stage={args.stage} outputs={outputs}")
    if wants_train(args.stage):
        run([args.python, AI_TRAIN / "train_joint_sf.py", "--config", args.config_train.resolve(),
             "--method-id", "Method_SGAN_SF_01", "--architecture", "sgan"], cwd=AI_TRAIN, dry_run=args.dry_run)
    if wants_evaluate(args.stage):
        run_dir = resolve_run(args.run_path, outputs)
        checkpoint = args.checkpoint or (run_dir / "checkpoints" / "best_model.pth")
        run([args.python, AI_TRAIN / "evaluate_joint_sf.py", "--config", args.config_test.resolve(),
             "--method-id", "Method_SGAN_SF_01", "--architecture", "sgan",
             "--run-path", run_dir, "--checkpoint", checkpoint], cwd=AI_TRAIN, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

from pipeline_common import (  # noqa: E402
    load_json, parser, require_sf_ready, resolve_checkpoint, resolve_run, run,
    wants_evaluate, wants_train,
)


def main() -> None:
    cli = parser("Plan, train, or evaluate Method_Transformer_SF_01.")
    cli.add_argument("--config-train", type=pathlib.Path, default=HERE / "config_train.json")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_test.json")
    args = cli.parse_args()
    outputs = HERE.parents[1] / "AI_Result" / "Method_Transformer_SF_01" / "outputs"
    train_cfg = load_json(args.config_train.resolve())
    require_sf_ready(train_cfg, "Method_Transformer_SF_01", dry_run=args.dry_run or args.stage == "plan")

    print(f"[pipeline] method=Method_Transformer_SF_01 stage={args.stage} outputs={outputs}")
    if wants_train(args.stage):
        run([args.python, AI_TRAIN / "train_joint_sf.py", "--config", args.config_train.resolve(),
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

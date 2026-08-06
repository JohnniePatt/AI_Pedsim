from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

from pipeline_common import parser, resolve_checkpoint, resolve_run, run, wants_evaluate, wants_train  # noqa: E402


def main() -> None:
    cli = parser("Plan, train, or evaluate Method_GNN_CVAE2.")
    cli.add_argument("--config-train", type=pathlib.Path, default=HERE / "config_train.json")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_test.json")
    args = cli.parse_args()
    outputs = HERE.parents[1] / "AI_Result" / "Method_GNN_CVAE2" / "outputs"
    print(f"[pipeline] method=Method_GNN_CVAE2 stage={args.stage} outputs={outputs}")
    if wants_train(args.stage):
        run([args.python, HERE / "train_gnn_cvae2.py", "--config", args.config_train.resolve()], cwd=HERE, dry_run=args.dry_run)
    if wants_evaluate(args.stage):
        run_dir = resolve_run(args.run_path, outputs)
        checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
        run([args.python, HERE / "test_gnn_cvae2.py", "--config", args.config_test.resolve(),
             "--run_path", run_dir, "--model_path", checkpoint], cwd=HERE, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

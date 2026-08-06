from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
AI_TRAIN = HERE.parent
sys.path.insert(0, str(AI_TRAIN))

from pipeline_common import (  # noqa: E402
    choose_operation, choose_run, confirm_full_training, parser,
    print_available_runs, resolve_checkpoint, resolve_run, run, wants_evaluate,
    wants_train,
)


def main() -> None:
    cli = parser("Plan, train, or evaluate Method_GNN_CVAE2.")
    cli.add_argument("--config-train", type=pathlib.Path, default=HERE / "config_train.json")
    cli.add_argument("--config-test", type=pathlib.Path, default=HERE / "config_test.json")
    args = cli.parse_args()
    outputs = HERE.parents[1] / "AI_Result" / "Method_GNN_CVAE2" / "outputs"
    interactive = len(sys.argv) == 1
    action = None
    if interactive:
        action = choose_operation("Graph Neural Network with Conditional Variational Autoencoder",
                                  supports_smoke=False)
        if action == "exit":
            return
        if action == "runs":
            print_available_runs(outputs)
            return
        args.stage = {"check": "plan", "train": "train", "evaluate": "evaluate", "all": "all"}[action]
        if action in {"train", "all"}:
            if not confirm_full_training("Method_GNN_CVAE2", args.config_train):
                return
        if action == "evaluate":
            args.run_path = choose_run(outputs)
            if args.run_path is None:
                return
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

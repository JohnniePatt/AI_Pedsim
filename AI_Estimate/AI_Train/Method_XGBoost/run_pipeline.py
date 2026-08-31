import argparse
import subprocess
import sys
from pathlib import Path

from dataset import build_data_bundle, read_json


METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = METHOD_DIR / "config_train.json"


def plan(config_path):
    config = read_json(config_path)
    bundle = build_data_bundle(config, config_path)
    try:
        import xgboost
        dependency = f"available ({xgboost.__version__})"
    except ImportError:
        dependency = "MISSING (install requirements.txt)"
    print("XGBoost travel-time estimator plan")
    print(f"  config: {config_path}")
    print(f"  dataset: {bundle.source_manifest['dataset_id']}")
    print(f"  rows: train={len(bundle.frames['train'])}, val={len(bundle.frames['val'])}, test={len(bundle.frames['test'])}")
    print(f"  plans: train={bundle.frames['train']['plan'].nunique()}, val={bundle.frames['val']['plan'].nunique()}, test={bundle.frames['test']['plan'].nunique()}")
    print(f"  input features: {len(bundle.feature_columns)}")
    print(f"  targets: {', '.join(bundle.target_columns)}")
    print(f"  xgboost: {dependency}")


def command_for(stage, config_path, run_path=None):
    if stage == "train":
        return [sys.executable, str(METHOD_DIR / "train_time_estimator.py"), "--config", str(config_path)]
    if stage == "evaluate":
        command = [sys.executable, str(METHOD_DIR / "test_time_estimator.py"), "--config", str(config_path)]
        if run_path:
            command.extend(["--run-path", str(run_path)])
        return command
    raise ValueError(stage)


def menu():
    print("XGBoost Travel-Time Estimator")
    print("  1) Check configuration")
    print("  2) Train model")
    print("  3) Evaluate existing model")
    print("  4) Train model and evaluate")
    print("  0) Exit")
    choice = input("Choose [1]: ").strip() or "1"
    return {"1": "plan", "2": "train", "3": "evaluate", "4": "all", "0": "exit"}.get(choice, "plan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["plan", "train", "evaluate", "all"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    stage = args.stage or menu()
    if stage == "exit":
        return
    plan(config_path)
    if stage == "plan":
        return
    if stage in {"train", "evaluate"}:
        command = command_for(stage, config_path, args.run_path)
        print("command:", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
        return
    train_command = command_for("train", config_path)
    print("command:", " ".join(train_command))
    if args.dry_run:
        return
    subprocess.run(train_command, check=True)
    output_root = (METHOD_DIR / read_json(config_path)["output"]["root"]).resolve()
    latest = max(output_root.glob("run_*"), key=lambda path: path.stat().st_mtime)
    subprocess.run(command_for("evaluate", config_path, latest), check=True)


if __name__ == "__main__":
    main()

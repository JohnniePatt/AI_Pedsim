from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


def load_parent_main():
    method_dir = pathlib.Path(__file__).resolve().parents[1]
    if str(method_dir) not in sys.path:
        sys.path.insert(0, str(method_dir))
    module_path = method_dir / "train_gnn_cvae.py"
    spec = importlib.util.spec_from_file_location("gnn_cvae_train_core", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(pathlib.Path(__file__).with_name("config_train.json")))
    args = parser.parse_args()
    expected = pathlib.Path(__file__).with_name("config_train.json").resolve()
    requested = pathlib.Path(args.config).resolve()
    if requested != expected:
        raise ValueError(
            f"Step_01_5_GoalGeometry must use its own config.\n"
            f"Expected: {expected}\n"
            f"Got: {requested}"
        )
    load_parent_main()(str(expected))

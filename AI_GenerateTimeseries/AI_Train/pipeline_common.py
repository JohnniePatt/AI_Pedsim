"""Small orchestration helpers shared by per-method ``run_pipeline.py`` files."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Iterable


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--stage",
        choices=("plan", "train", "evaluate", "all"),
        default="plan",
        help="Default is plan so a long training job cannot start accidentally.",
    )
    result.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    result.add_argument("--run-path", type=pathlib.Path, default=None)
    result.add_argument("--checkpoint", type=pathlib.Path, default=None)
    result.add_argument("--python", default=sys.executable)
    return result


def load_json(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def command_text(command: Iterable[object]) -> str:
    return shlex.join(str(item) for item in command)


def run(command: list[object], *, cwd: pathlib.Path, dry_run: bool) -> None:
    print(f"[pipeline] cwd={cwd}")
    print(f"[pipeline] {command_text(command)}")
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=cwd, check=True)


def latest_run(outputs_root: pathlib.Path) -> pathlib.Path:
    candidates = [path for path in outputs_root.glob("run_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run_* directory found under {outputs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def resolve_run(run_path: pathlib.Path | None, outputs_root: pathlib.Path) -> pathlib.Path:
    return run_path.resolve() if run_path else latest_run(outputs_root.resolve())


def resolve_checkpoint(
    run_dir: pathlib.Path,
    explicit: pathlib.Path | None,
    names: tuple[str, ...] = ("best_model.pth", "latest_model.pth"),
) -> pathlib.Path:
    if explicit:
        checkpoint = explicit.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    for directory in (run_dir / "checkpoints", run_dir / "weights"):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no compatible checkpoint found under {run_dir}")


def require_sf_ready(config: dict, method_id: str, *, dry_run: bool) -> None:
    if config.get("sf_implementation_ready") is True:
        return
    message = (
        f"{method_id} is still a protected baseline copy; implement and validate its SF contract, "
        "then set sf_implementation_ready=true."
    )
    if dry_run:
        print(f"[pipeline] BLOCKED FOR EXECUTION: {message}")
        return
    raise RuntimeError(message)


def wants_train(stage: str) -> bool:
    return stage in {"train", "all"}


def wants_evaluate(stage: str) -> bool:
    return stage in {"evaluate", "all"}


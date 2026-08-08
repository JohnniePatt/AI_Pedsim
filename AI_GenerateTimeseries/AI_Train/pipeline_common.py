"""Small orchestration helpers shared by per-method ``run_pipeline.py`` files."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
from typing import Callable, Iterable


STANDARD_MENU = (
    (
        "check",
        "Check configuration",
        "ตรวจ method, config, output path และ contract ที่รองรับ โดยยังไม่เริ่ม train",
    ),
    (
        "smoke",
        "Quick smoke test",
        "ทดลอง train/evaluate ชุดเล็กเพื่อตรวจ pipeline; ผลนี้ไม่ใช่ผลวิจัย",
    ),
    (
        "train",
        "Train model",
        "ฝึกด้วย training config หลัก; ตรวจ dataset, subset และจำนวน epoch ก่อนเริ่ม",
    ),
    (
        "evaluate",
        "Evaluate existing model",
        "เลือก run/checkpoint ที่มีอยู่เพื่อสร้าง trajectory และคำนวณ metrics",
    ),
    (
        "all",
        "Train model and evaluate",
        "ฝึกด้วย config ที่เลือก แล้วประเมิน checkpoint ที่ดีที่สุดต่อทันที",
    ),
    (
        "runs",
        "View available runs",
        "แสดง run, checkpoint, dataset และสถานะ research-validity ที่ตรวจพบ",
    ),
    (
        "exit",
        "Exit",
        "ออกจากโปรแกรมโดยไม่ดำเนินการ",
    ),
)


RETRIEVAL_MENU = (
    (
        "check",
        "Check configuration",
        "ตรวจ training-only knowledge source, query split และความเสี่ยง data leakage",
    ),
    (
        "train",
        "Build knowledge index",
        "สร้างฐานความรู้จาก canonical training split ตาม config ที่กำหนด",
    ),
    (
        "evaluate",
        "Validate existing knowledge index",
        "ตรวจ index, provenance และการแยก training/test dataset",
    ),
    (
        "all",
        "Build and validate",
        "สร้าง knowledge index แล้วตรวจสอบ contract ต่อเนื่องในคำสั่งเดียว",
    ),
    (
        "runs",
        "View configured artifacts",
        "แสดงตำแหน่ง knowledge index และ evaluation artifacts ที่ config อ้างถึง",
    ),
    (
        "exit",
        "Exit",
        "ออกจากโปรแกรมโดยไม่ดำเนินการ",
    ),
)


def parser(description: str, *, default_stage: str = "plan") -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument(
        "--stage",
        choices=("plan", "train", "evaluate", "all"),
        default=default_stage,
        help=f"Pipeline stage to run. Default is {default_stage}.",
    )
    result.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    result.add_argument("--run-path", type=pathlib.Path, default=None)
    result.add_argument("--checkpoint", type=pathlib.Path, default=None)
    result.add_argument("--python", default=sys.executable)
    return result


def load_json(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def choose_operation(
    method_name: str,
    *,
    supports_smoke: bool = True,
    retrieval: bool = False,
    input_func: Callable[[str], str] = input,
) -> str:
    """Show the user-facing pipeline menu and return a stable action name."""
    menu = RETRIEVAL_MENU if retrieval else STANDARD_MENU
    if not supports_smoke:
        menu = tuple(item for item in menu if item[0] != "smoke")
    operations = tuple(item for item in menu if item[0] != "exit")
    exit_item = next(item for item in menu if item[0] == "exit")

    print(f"\n{method_name}")
    print("Select an operation:\n")
    for number, (_, title, description) in enumerate(operations, start=1):
        print(f"  {number}) {title}")
        print(f"     {description}\n")
    print(f"  0) {exit_item[1]}")
    print(f"     {exit_item[2]}\n")

    while True:
        try:
            raw = input_func("Choose [1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[pipeline] Cancelled; no operation was started.")
            return "exit"
        if not raw:
            return operations[0][0]
        if raw == "0":
            return "exit"
        if raw.isdigit() and 1 <= int(raw) <= len(operations):
            return operations[int(raw) - 1][0]
        print(f"Please enter a number from 0 to {len(operations)}.")


def confirm_full_training(
    method_name: str,
    config_path: pathlib.Path,
    *,
    input_func: Callable[[str], str] = input,
) -> bool:
    """Require an explicit confirmation before an interactive full run."""
    print("\nTraining selected")
    print(f"  method : {method_name}")
    print(f"  config : {config_path.resolve()}")
    print("  note   : inspect the config above; this may use the GPU for several hours")
    try:
        answer = input_func("Start training? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[pipeline] Cancelled; training was not started.")
        return False
    confirmed = answer in {"y", "yes"}
    if not confirmed:
        print("[pipeline] Cancelled; training was not started.")
    return confirmed


def available_runs(outputs_root: pathlib.Path) -> list[pathlib.Path]:
    if not outputs_root.is_dir():
        return []
    return sorted(
        (path for path in outputs_root.glob("run_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def _run_manifest_summary(run_dir: pathlib.Path) -> str:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return "manifest=missing"
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError):
        return "manifest=unreadable"
    fields = []
    for key in ("status", "dataset_id", "seed", "research_valid"):
        if key in manifest:
            fields.append(f"{key}={manifest[key]}")
    return " ".join(fields) if fields else "manifest=present"


def print_available_runs(outputs_root: pathlib.Path, *, limit: int = 20) -> list[pathlib.Path]:
    runs = available_runs(outputs_root)
    print(f"\nAvailable runs: {outputs_root.resolve()}")
    if not runs:
        print("  No run directories were found.")
        return []
    for number, run_dir in enumerate(runs[:limit], start=1):
        checkpoint = any(
            (run_dir / directory / name).is_file()
            for directory in ("checkpoints", "weights")
            for name in ("best_model.pth", "latest_model.pth", "best.pt", "last.pt")
        )
        print(
            f"  {number}) {run_dir.name} checkpoint={'yes' if checkpoint else 'no'} "
            f"{_run_manifest_summary(run_dir)}"
        )
    if len(runs) > limit:
        print(f"  ... {len(runs) - limit} older run(s) not shown")
    return runs


def choose_run(
    outputs_root: pathlib.Path,
    *,
    input_func: Callable[[str], str] = input,
) -> pathlib.Path | None:
    """Let an interactive user choose a run without typing a filesystem path."""
    runs = print_available_runs(outputs_root)
    if not runs:
        return None
    visible = runs[:20]
    while True:
        try:
            raw = input_func("Select run [1 = newest, 0 = cancel]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[pipeline] Cancelled; evaluation was not started.")
            return None
        if not raw:
            return visible[0]
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(visible):
            return visible[int(raw) - 1]
        print(f"Please enter a number from 0 to {len(visible)}.")


def command_text(command: Iterable[object]) -> str:
    return shlex.join(str(item) for item in command)


def run(command: list[object], *, cwd: pathlib.Path, dry_run: bool) -> None:
    print(f"[pipeline] cwd={cwd}", flush=True)
    print(f"[pipeline] {command_text(command)}", flush=True)
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

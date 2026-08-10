from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_RESULT_ROOT = PROJECT_ROOT / "AI_GenerateImage" / "AI_Result"


@dataclass(frozen=True)
class RunInfo:
    method: str
    run_name: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.method} / {self.run_name}"


def _candidate_run_dirs(method_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    outputs_dir = method_dir / "outputs"
    if outputs_dir.exists():
        candidates.extend([p for p in outputs_dir.iterdir() if p.is_dir()])
    candidates.extend([p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("run")])
    return sorted(set(candidates), key=lambda p: p.name, reverse=True)


def discover_runs() -> list[RunInfo]:
    if not AI_RESULT_ROOT.exists():
        return []

    runs: list[RunInfo] = []
    for method_dir in sorted([p for p in AI_RESULT_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name):
        for run_dir in _candidate_run_dirs(method_dir):
            has_summary = (
                (run_dir / "test_evaluation_summary.csv").exists()
                or (run_dir / "logs" / "test_evaluation.csv").exists()
                or (run_dir / "logs" / "training_history.csv").exists()
            )
            has_per_image = (run_dir / "test_evaluation_per_image.csv").exists()
            test_res = run_dir / "test_results"
            has_images = test_res.exists() and (
                (test_res / "predictions").exists()
                or (test_res / "colorjet").exists()
                or (test_res / "bw").exists()
                or any(p.suffix == ".png" for p in test_res.iterdir() if p.is_file())
                or any(
                    (sub / "predictions").exists() or (sub / "colorjet").exists() or (sub / "bw").exists()
                    for sub in test_res.iterdir() if sub.is_dir()
                )
            )
            if has_summary or has_per_image or has_images:
                runs.append(RunInfo(method=method_dir.name, run_name=run_dir.name, path=run_dir))
    return sorted(runs, key=lambda run: run.path.stat().st_mtime, reverse=True)


def get_run_by_label(runs: list[RunInfo], label: str) -> RunInfo | None:
    for run in runs:
        if run.label == label:
            return run
    return None

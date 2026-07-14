from __future__ import annotations

from pathlib import Path


def image_triplet(run_path: Path, file_name: str) -> tuple[Path | None, Path | None, Path | None]:
    result_dir = run_path / "test_results"
    
    def find_path(sub_name: str) -> Path | None:
        # Try direct: e.g. run_path / "test_results" / "predictions" / file_name
        p = result_dir / sub_name / file_name
        if p.exists():
            return p
        # Try subdirectories inside test_results: e.g. run_path / "test_results" / "best_loss" / "predictions" / file_name
        if result_dir.exists():
            for sub in result_dir.iterdir():
                if sub.is_dir():
                    p_sub = sub / sub_name / file_name
                    if p_sub.exists():
                        return p_sub
        return None

    return (
        find_path("inputs"),
        find_path("predictions"),
        find_path("targets"),
    )

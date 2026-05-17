from __future__ import annotations

from pathlib import Path


def image_triplet(run_path: Path, file_name: str) -> tuple[Path | None, Path | None, Path | None]:
    result_dir = run_path / "test_results"
    input_path = result_dir / "inputs" / file_name
    pred_path = result_dir / "predictions" / file_name
    target_path = result_dir / "targets" / file_name

    return (
        input_path if input_path.exists() else None,
        pred_path if pred_path.exists() else None,
        target_path if target_path.exists() else None,
    )

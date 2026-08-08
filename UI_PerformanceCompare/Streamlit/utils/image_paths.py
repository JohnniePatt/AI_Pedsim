from __future__ import annotations

from pathlib import Path


def image_triplet(run_path: Path, file_name: str) -> tuple[Path | None, Path | None, Path | None]:
    result_dir = run_path / "test_results"
    
    def find_path(sub_name: str) -> Path | None:
        if sub_name == "predictions":
            sub_candidates = ["predictions", "colorjet", "bw", ""]
        else:
            sub_candidates = [sub_name, ""]

        clean_name = file_name[5:] if file_name.startswith("MASK_") else file_name
        mask_name = f"MASK_{clean_name}"
        name_candidates = [clean_name, mask_name, file_name]

        for name in name_candidates:
            for sc in sub_candidates:
                p = (result_dir / sc / name) if sc else (result_dir / name)
                if p.exists() and p.is_file():
                    return p

        if result_dir.exists():
            for sub in result_dir.iterdir():
                if sub.is_dir():
                    for name in name_candidates:
                        for sc in sub_candidates:
                            p_sub = (sub / sc / name) if sc else (sub / name)
                            if p_sub.exists() and p_sub.is_file():
                                return p_sub
        return None

    return (
        find_path("inputs"),
        find_path("predictions"),
        find_path("targets"),
    )

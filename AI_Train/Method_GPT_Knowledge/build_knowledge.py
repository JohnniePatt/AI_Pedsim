"""
build_knowledge.py
------------------
Build a retrieval-friendly knowledge index from the train split.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd
from tqdm import tqdm

from prepare_geometry_gpt_knowledge import scene_feature_row


def load_config(config_path: str | pathlib.Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_config(config_path: str | pathlib.Path, raw_path: str) -> pathlib.Path:
    config_path = pathlib.Path(config_path).resolve()
    path = pathlib.Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def resolve_dataset_roots(config_path: str | pathlib.Path, cfg: dict) -> list[pathlib.Path]:
    raw = cfg.get("dataset_roots", cfg.get("dataset_root"))
    if raw is None:
        raise KeyError("Config must contain 'dataset_roots' or 'dataset_root'.")
    if isinstance(raw, (str, pathlib.Path)):
        return [resolve_from_config(config_path, str(raw))]
    if isinstance(raw, list):
        return [resolve_from_config(config_path, str(item)) for item in raw]
    raise TypeError(f"dataset_roots/dataset_root must be a string or list, got {type(raw).__name__}")


def iter_case_dirs(dataset_root: pathlib.Path, split: str) -> list[pathlib.Path]:
    split_dir = dataset_root / split
    return sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("case_")])


def main(config_path: str):
    cfg = load_config(config_path)
    dataset_roots = resolve_dataset_roots(config_path, cfg)
    split = cfg.get("split", "train")
    max_cases = int(cfg.get("max_cases", 0))
    output_dir = resolve_from_config(config_path, cfg["knowledge_output_dir"])

    case_dirs = []
    for dataset_root in dataset_roots:
        dataset_name = dataset_root.name
        for case_dir in iter_case_dirs(dataset_root, split):
            case_dirs.append((dataset_name, dataset_root, case_dir))
    if max_cases > 0:
        case_dirs = case_dirs[:max_cases]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for dataset_name, dataset_root, case_dir in tqdm(case_dirs, desc=f"Build knowledge [{split}]"):
        try:
            row = scene_feature_row(case_dir)
            row["dataset_name"] = dataset_name
            row["dataset_root"] = str(dataset_root)
            rows.append(row)
        except Exception as e:
            skipped.append({
                "dataset_name": dataset_name,
                "case_dir": str(case_dir),
                "error_type": type(e).__name__,
                "error": str(e),
            })
            print(f"[Knowledge][Skip] {case_dir} -> {type(e).__name__}: {e}")

    scene_index = pd.DataFrame(rows)
    if not scene_index.empty:
        scene_index = scene_index.sort_values(["dataset_name", "case_id"]).reset_index(drop=True)
    scene_index.to_parquet(output_dir / "scene_index.parquet", index=False)
    scene_index.to_csv(output_dir / "scene_index.csv", index=False)
    pd.DataFrame(skipped).to_csv(output_dir / "skipped_cases.csv", index=False)

    manifest = {
        "dataset_roots": [str(p) for p in dataset_roots],
        "split": split,
        "num_cases": int(len(scene_index)),
        "num_skipped": int(len(skipped)),
        "datasets": scene_index["dataset_name"].value_counts().to_dict() if not scene_index.empty and "dataset_name" in scene_index.columns else {},
        "index_files": {
            "scene_index_parquet": str(output_dir / "scene_index.parquet"),
            "scene_index_csv": str(output_dir / "scene_index.csv"),
            "skipped_cases_csv": str(output_dir / "skipped_cases.csv"),
        },
        "notes": [
            "This knowledge index stores scene-level features and pointers back to original case directories.",
            "Full trajectories stay in the dataset and are loaded lazily during generation.",
            "The first baseline uses retrieval + geometric transfer, with an optional GPT planner slot later.",
        ],
    }
    with open(output_dir / "knowledge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Knowledge] Built {len(scene_index)} cases into {output_dir}")
    print(f"[Knowledge] Skipped {len(skipped)} cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(pathlib.Path(__file__).with_name("config_build.json")))
    args = parser.parse_args()
    main(args.config)

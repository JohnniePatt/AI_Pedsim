#!/usr/bin/env python3
"""Build the immutable JuPedSim benchmark allow-list from Data_Estimate_2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset" / "Data_Estimate_2"
DEFAULT_OUTPUT = PROJECT_ROOT / "Dataset_TimeCalculate" / "Total_RefFileName.csv"
SPLITS = (("Train", "train"), ("Val", "val"), ("Test", "test"))

OUTPUT_COLUMNS = [
    "reference_id",
    "split",
    "plan",
    "route_index",
    "variant_id",
    "variant_label",
    "status",
    "computed_agents",
    "fps",
    "max_frame",
    "simulated_duration_s",
    "source_trajectory_filename",
    "source_trajectory_file",
    "route_metadata_file",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def source_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_rows(dataset_root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, object]]]:
    rows: list[dict[str, str]] = []
    inventory: dict[str, dict[str, object]] = {}
    seen: set[str] = set()

    for directory_name, expected_split in SPLITS:
        csv_path = dataset_root / directory_name / "data_estimate.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")

        split_rows = 0
        plans: set[str] = set()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle):
                split = (source.get("split") or "").strip().lower()
                if split != expected_split:
                    raise ValueError(
                        f"Unexpected split {split!r} in {csv_path}; expected {expected_split!r}"
                    )
                if (source.get("status") or "").strip().lower() != "success":
                    continue

                trajectory_raw = (source.get("trajectory_file") or "").strip()
                if not trajectory_raw:
                    raise ValueError(f"Missing trajectory_file in {csv_path}")
                trajectory = source_path(trajectory_raw)
                if not trajectory.is_file():
                    raise FileNotFoundError(f"Referenced trajectory does not exist: {trajectory}")

                plan = (source.get("plan") or "").strip()
                route_index = int(source["route_index"])
                variant_id = (source.get("variant_id") or "").strip()
                reference_id = f"{plan}__{trajectory.stem}"
                if reference_id in seen:
                    raise ValueError(f"Duplicate reference_id: {reference_id}")
                seen.add(reference_id)

                route_meta = (
                    PROJECT_ROOT
                    / "Geo_scenario"
                    / "Topo_HouseGAN"
                    / "metadata"
                    / plan
                    / f"route_{route_index:02d}.json"
                )
                if not route_meta.is_file():
                    raise FileNotFoundError(f"Missing route metadata: {route_meta}")

                rows.append(
                    {
                        "reference_id": reference_id,
                        "split": split,
                        "plan": plan,
                        "route_index": str(route_index),
                        "variant_id": variant_id,
                        "variant_label": (source.get("variant_label") or "").strip(),
                        "status": "success",
                        "computed_agents": str(int(float(source["computed_agents"]))),
                        "fps": source["fps"],
                        "max_frame": str(int(float(source["max_frame"]))),
                        "simulated_duration_s": source["simulation_duration_s"],
                        "source_trajectory_filename": trajectory.name,
                        "source_trajectory_file": portable_path(trajectory),
                        "route_metadata_file": portable_path(route_meta),
                    }
                )
                split_rows += 1
                plans.add(plan)

        inventory[expected_split] = {
            "rows": split_rows,
            "plans": len(plans),
            "source_csv": portable_path(csv_path),
            "source_sha256": sha256_file(csv_path),
        }

    rows.sort(
        key=lambda row: (
            {"train": 0, "val": 1, "test": 2}[row["split"]],
            row["plan"],
            int(row["route_index"]),
            {"full": 0, "half": 1, "single": 2}.get(row["variant_id"], 99),
        )
    )
    return rows, inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Total_RefFileName.csv from successful Data_Estimate_2 rows."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    rows, inventory = build_rows(dataset_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"output": str(output), "rows": len(rows), "splits": inventory}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

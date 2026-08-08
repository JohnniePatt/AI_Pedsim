from __future__ import annotations

import pathlib


DATASET_MARKER = ("Dataset", "Data_TrajectoryGrid", "Topo_HouseGAN")


def resolve_manifest_path(dataset_root: pathlib.Path, manifest_value: str | pathlib.Path) -> pathlib.Path:
    """Resolve stale absolute paths from a manifest against the active dataset root."""

    dataset_root = pathlib.Path(dataset_root).resolve()
    path = pathlib.Path(manifest_value)
    if path.is_absolute() and path.exists():
        return path

    parts = path.parts
    for index in range(0, max(len(parts) - len(DATASET_MARKER) + 1, 0)):
        if tuple(parts[index : index + len(DATASET_MARKER)]) == DATASET_MARKER:
            suffix = parts[index + len(DATASET_MARKER) :]
            return dataset_root.joinpath(*suffix)

    if path.is_absolute():
        return path
    return dataset_root / path

"""
prepare_dataset_trajectory_grid.py
----------------------------------
Build an A/B trajectory-grid dataset directly from Topo_HouseGAN raw outputs.

Input source:
    Geo_scenario/Topo_HouseGAN/
        dataswarm/<plan_name>/<sqlite_name>.sqlite
        geo/<plan_name>/walkablearea_grid.json
        geo/<plan_name>/geo_room.json
        geo/<plan_name>/geo_corridor.json
        metadata/<plan_name>/route_<route_index>.json

Output:
    Dataset/Data_TrajectoryGrid/Topo_HouseGAN/
        A/<split>/<plan_name>/<sqlite_stem>/
            walkablearea_grid.json
            spawn_agent.parquet
            exit_room.json
        B/<split>/<plan_name>/<sqlite_stem>/
            trajectory.parquet
        metadata/<split>/<plan_name>/<sqlite_stem>.json
        manifest_trajectory_grid.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from shapely.geometry import Polygon

from normalize_housegan_dataset import load_trajectory_table, parse_sqlite_name
from prepare_layout_to_grid_walkablearea import build_walkable_grid


DEFAULT_CELL_SIZE_M = 0.125
DEFAULT_PADDING_M = 0.0
DEFAULT_WALL_THICKNESS_M = 0.125


@dataclass(frozen=True)
class DatasetPaths:
    source_root: pathlib.Path
    dataswarm_root: pathlib.Path
    geo_root: pathlib.Path
    metadata_root: pathlib.Path
    output_root: pathlib.Path


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def default_source_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "Geo_scenario" / "Topo_HouseGAN"


def default_output_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "Dataset" / "Data_TrajectoryGrid" / "Topo_HouseGAN"


def make_paths(source_root: pathlib.Path, output_root: pathlib.Path) -> DatasetPaths:
    paths = DatasetPaths(
        source_root=source_root,
        dataswarm_root=source_root / "dataswarm",
        geo_root=source_root / "geo",
        metadata_root=source_root / "metadata",
        output_root=output_root,
    )
    for required in [paths.dataswarm_root, paths.geo_root, paths.metadata_root]:
        if not required.exists():
            raise FileNotFoundError(f"Required source directory not found: {required}")
    return paths


def stable_plan_split(plan_name: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    key = f"{seed}:{plan_name}".encode("utf-8")
    bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def iter_sqlite_files(dataswarm_root: pathlib.Path, plan_name: str | None):
    plan_dirs = [dataswarm_root / plan_name] if plan_name else sorted(dataswarm_root.glob("plan_*"), key=lambda p: p.name)
    for plan_dir in plan_dirs:
        if not plan_dir.exists() or not plan_dir.is_dir():
            continue
        for sqlite_path in sorted(plan_dir.glob("*.sqlite"), key=lambda p: p.name):
            yield plan_dir.name, sqlite_path


def ensure_grid_json(
    geo_dir: pathlib.Path,
    cell_size_m: float,
    padding_m: float,
    wall_thickness_m: float,
    regenerate_grid: bool,
) -> pathlib.Path:
    grid_path = geo_dir / "walkablearea_grid.json"
    if grid_path.exists() and not regenerate_grid:
        return grid_path

    payload = build_walkable_grid(
        plan_dir=geo_dir,
        cell_size_m=cell_size_m,
        padding_m=padding_m,
        wall_thickness_m=wall_thickness_m,
    )
    write_json(grid_path, payload)
    return grid_path


def grid_rows_to_array(grid_rows: list[str]) -> list[str]:
    if not grid_rows:
        raise ValueError("walkablearea_grid.json has empty grid rows")
    width = len(grid_rows[0])
    if any(len(row) != width for row in grid_rows):
        raise ValueError("walkablearea_grid.json grid rows have inconsistent widths")
    return grid_rows


def add_grid_columns(df: pd.DataFrame, grid_payload: dict) -> pd.DataFrame:
    meta = grid_payload["meta"]
    grid_rows = grid_rows_to_array(grid_payload["grid"])
    cell = float(meta["cell_size_m"])
    origin_x = float(meta["origin_x"])
    origin_y = float(meta["origin_y"])
    width = int(meta["width"])
    height = int(meta["height"])

    out = df.copy()
    grid_x = ((out["pos_x"].to_numpy(dtype="float64") - origin_x) / cell)
    grid_y = ((out["pos_y"].to_numpy(dtype="float64") - origin_y) / cell)
    out["grid_x"] = [int(math.floor(v)) for v in grid_x]
    out["grid_y"] = [int(math.floor(v)) for v in grid_y]
    out["grid_row"] = height - 1 - out["grid_y"]
    out["is_inside_grid"] = (
        (out["grid_x"] >= 0)
        & (out["grid_x"] < width)
        & (out["grid_y"] >= 0)
        & (out["grid_y"] < height)
    )

    walkable_flags: list[bool] = []
    for gx, row, inside in zip(out["grid_x"], out["grid_row"], out["is_inside_grid"]):
        if not bool(inside):
            walkable_flags.append(False)
        else:
            walkable_flags.append(grid_rows[int(row)][int(gx)] == "1")
    out["is_walkable_cell"] = walkable_flags
    return out


def normalize_trajectory(sqlite_path: pathlib.Path, grid_payload: dict, table_filter: str) -> pd.DataFrame:
    df = load_trajectory_table(sqlite_path, table_filter=table_filter)
    rename_map = {"id": "agent_id"}
    df = df.rename(columns=rename_map)
    df["frame"] = df["frame"].astype("int64")
    df["agent_id"] = df["agent_id"].astype("int64")
    df = add_grid_columns(df, grid_payload)

    ordered = ["frame", "agent_id", "pos_x", "pos_y", "grid_x", "grid_y", "grid_row", "is_inside_grid", "is_walkable_cell"]
    for optional in ["ori_x", "ori_y"]:
        if optional in df.columns:
            ordered.append(optional)
    return df[ordered].sort_values(["frame", "agent_id"]).reset_index(drop=True)


def make_spawn_agent(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    first_frame = int(trajectory_df["frame"].min())
    cols = ["frame", "agent_id", "pos_x", "pos_y", "grid_x", "grid_y", "grid_row", "is_inside_grid", "is_walkable_cell"]
    return (
        trajectory_df.loc[trajectory_df["frame"] == first_frame, cols]
        .sort_values("agent_id")
        .reset_index(drop=True)
    )


def load_node_polygons(geo_dir: pathlib.Path) -> dict[str, Polygon]:
    node_polys: dict[str, Polygon] = {}
    for prefix, filename in [("Room", "geo_room.json"), ("Cor", "geo_corridor.json")]:
        path = geo_dir / filename
        if not path.exists():
            continue
        for i, coords in enumerate(load_json(path)):
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            node_polys[f"{prefix}-{i}"] = poly
    return node_polys


def polygon_payload(poly: Polygon) -> dict:
    min_x, min_y, max_x, max_y = poly.bounds
    centroid = poly.centroid
    exterior = [[float(x), float(y)] for x, y in poly.exterior.coords]
    return {
        "polygon": exterior,
        "bbox": [float(min_x), float(min_y), float(max_x), float(max_y)],
        "centroid": [float(centroid.x), float(centroid.y)],
        "area_m2": float(poly.area),
    }


def route_json_path(metadata_dir: pathlib.Path, route_idx_str: str) -> pathlib.Path:
    return metadata_dir / f"route_{int(route_idx_str):02d}.json"


def find_variant_metadata(route_meta: dict, sqlite_name: str, variant: str | None) -> dict | None:
    for item in route_meta.get("variants", []):
        trajectory_file = str(item.get("trajectory_file", ""))
        variant_id = item.get("variant_id")
        if pathlib.Path(trajectory_file).name == sqlite_name:
            return item
        if variant and variant_id == variant:
            return item
    return None


def build_exit_room_payload(
    plan_name: str,
    sqlite_path: pathlib.Path,
    seed_str: str,
    route_idx_str: str,
    variant: str | None,
    geo_dir: pathlib.Path,
    metadata_dir: pathlib.Path,
) -> dict:
    route_path = route_json_path(metadata_dir, route_idx_str)
    if not route_path.exists():
        raise FileNotFoundError(f"Missing route metadata: {route_path}")

    route_meta = load_json(route_path)
    start_node = route_meta.get("start_node")
    end_node = route_meta.get("end_node")
    if not start_node or not end_node:
        raise ValueError(f"Route metadata missing start_node/end_node: {route_path}")

    node_polys = load_node_polygons(geo_dir)
    if end_node not in node_polys:
        raise ValueError(f"End node {end_node} polygon not found in {geo_dir}")
    if start_node not in node_polys:
        raise ValueError(f"Start node {start_node} polygon not found in {geo_dir}")

    variant_meta = find_variant_metadata(route_meta, sqlite_path.name, variant)
    end_kind = "room" if str(end_node).startswith("Room-") else "corridor"
    start_kind = "room" if str(start_node).startswith("Room-") else "corridor"

    return {
        "schema_version": 1,
        "plan_name": plan_name,
        "sqlite_name": sqlite_path.name,
        "sqlite_stem": sqlite_path.stem,
        "seed": int(seed_str),
        "route_index": int(route_idx_str),
        "variant": variant or "none",
        "source_metadata": str(route_path),
        "start_node": {
            "name": start_node,
            "kind": start_kind,
            **polygon_payload(node_polys[start_node]),
        },
        "exit_node": {
            "name": end_node,
            "kind": end_kind,
            **polygon_payload(node_polys[end_node]),
        },
        "exit_room": end_node if end_kind == "room" else None,
        "topological_path": route_meta.get("topological_path", []),
        "trigger": {
            "type": "room_polygon" if end_kind == "room" else "node_polygon",
            "node_name": end_node,
            "remove_agent_on_enter": True,
            "condition": "agent_position_inside_polygon",
        },
        "simulation_context": {
            "computed_agents": route_meta.get("computed_agents"),
            "variant_agents": None if not variant_meta else variant_meta.get("computed_agents"),
            "status": route_meta.get("status"),
            "variant_status": None if not variant_meta else variant_meta.get("status"),
        },
    }


def case_is_complete(input_dir: pathlib.Path, target_dir: pathlib.Path, metadata_path: pathlib.Path) -> bool:
    return (
        (input_dir / "walkablearea_grid.json").exists()
        and (input_dir / "spawn_agent.parquet").exists()
        and (input_dir / "exit_room.json").exists()
        and (target_dir / "trajectory.parquet").exists()
        and metadata_path.exists()
    )


def prepare_dataset(
    paths: DatasetPaths,
    plan_name: str | None,
    cell_size_m: float,
    padding_m: float,
    wall_thickness_m: float,
    table_filter: str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    max_cases: int | None,
    overwrite: bool,
    regenerate_grid: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths.output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    error_rows: list[dict] = []
    processed = 0

    for plan, sqlite_path in iter_sqlite_files(paths.dataswarm_root, plan_name):
        if max_cases is not None and processed >= max_cases:
            break

        geo_dir = paths.geo_root / plan
        metadata_dir = paths.metadata_root / plan
        try:
            seed_str, route_idx_str, variant = parse_sqlite_name(sqlite_path)
            split = stable_plan_split(plan, split_seed, train_ratio, val_ratio)
            case_key = sqlite_path.stem

            input_dir = paths.output_root / "A" / split / plan / case_key
            target_dir = paths.output_root / "B" / split / plan / case_key
            metadata_path = paths.output_root / "metadata" / split / plan / f"{case_key}.json"

            if case_is_complete(input_dir, target_dir, metadata_path) and not overwrite:
                manifest_rows.append(
                    {
                        "status": "skipped",
                        "split": split,
                        "plan_name": plan,
                        "sqlite_name": sqlite_path.name,
                        "sqlite_stem": sqlite_path.stem,
                        "seed": int(seed_str),
                        "route_index": int(route_idx_str),
                        "variant": variant or "none",
                        "input_dir": str(input_dir),
                        "target_dir": str(target_dir),
                        "metadata_path": str(metadata_path),
                    }
                )
                processed += 1
                continue

            if not geo_dir.exists():
                raise FileNotFoundError(f"Missing geo plan directory: {geo_dir}")
            if not metadata_dir.exists():
                raise FileNotFoundError(f"Missing metadata plan directory: {metadata_dir}")

            grid_path = ensure_grid_json(geo_dir, cell_size_m, padding_m, wall_thickness_m, regenerate_grid)
            grid_payload = load_json(grid_path)
            trajectory_df = normalize_trajectory(sqlite_path, grid_payload, table_filter)
            spawn_df = make_spawn_agent(trajectory_df)
            exit_payload = build_exit_room_payload(
                plan_name=plan,
                sqlite_path=sqlite_path,
                seed_str=seed_str,
                route_idx_str=route_idx_str,
                variant=variant,
                geo_dir=geo_dir,
                metadata_dir=metadata_dir,
            )

            input_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(grid_path, input_dir / "walkablearea_grid.json")
            spawn_df.to_parquet(input_dir / "spawn_agent.parquet", index=False)
            write_json(input_dir / "exit_room.json", exit_payload)
            trajectory_df.to_parquet(target_dir / "trajectory.parquet", index=False)

            metadata_payload = {
                "schema_version": 1,
                "created_at_unix": time.time(),
                "source": {
                    "sqlite": str(sqlite_path),
                    "geo_dir": str(geo_dir),
                    "walkablearea_grid": str(grid_path),
                    "route_metadata": exit_payload["source_metadata"],
                },
                "output": {
                    "input_dir": str(input_dir),
                    "target_dir": str(target_dir),
                    "metadata_path": str(metadata_path),
                },
                "case": {
                    "plan_name": plan,
                    "sqlite_name": sqlite_path.name,
                    "sqlite_stem": sqlite_path.stem,
                    "split": split,
                    "seed": int(seed_str),
                    "route_index": int(route_idx_str),
                    "variant": variant or "none",
                    "start_node": exit_payload["start_node"]["name"],
                    "exit_node": exit_payload["exit_node"]["name"],
                    "topological_path": exit_payload["topological_path"],
                },
                "data_summary": {
                    "agent_count": int(trajectory_df["agent_id"].nunique()),
                    "spawn_agent_count": int(len(spawn_df)),
                    "frame_count": int(trajectory_df["frame"].nunique()),
                    "min_frame": int(trajectory_df["frame"].min()),
                    "max_frame": int(trajectory_df["frame"].max()),
                    "trajectory_rows": int(len(trajectory_df)),
                    "positions_outside_grid": int((~trajectory_df["is_inside_grid"]).sum()),
                    "positions_on_non_walkable_cell": int((~trajectory_df["is_walkable_cell"]).sum()),
                },
                "grid_meta": grid_payload.get("meta", {}),
            }
            write_json(metadata_path, metadata_payload)

            manifest_rows.append(
                {
                    "status": "written",
                    "split": split,
                    "plan_name": plan,
                    "sqlite_name": sqlite_path.name,
                    "sqlite_stem": sqlite_path.stem,
                    "seed": int(seed_str),
                    "route_index": int(route_idx_str),
                    "variant": variant or "none",
                    "start_node": exit_payload["start_node"]["name"],
                    "exit_node": exit_payload["exit_node"]["name"],
                    "agent_count": int(trajectory_df["agent_id"].nunique()),
                    "frame_count": int(trajectory_df["frame"].nunique()),
                    "trajectory_rows": int(len(trajectory_df)),
                    "positions_outside_grid": int((~trajectory_df["is_inside_grid"]).sum()),
                    "positions_on_non_walkable_cell": int((~trajectory_df["is_walkable_cell"]).sum()),
                    "input_dir": str(input_dir),
                    "target_dir": str(target_dir),
                    "metadata_path": str(metadata_path),
                }
            )
            print(f"[TrajectoryGrid] wrote {split}/{plan}/{case_key}")
            processed += 1
        except Exception as exc:
            error_rows.append({"plan_name": plan, "sqlite_path": str(sqlite_path), "error": str(exc)})
            print(f"[TrajectoryGrid] failed {plan}/{sqlite_path.name}: {exc}")

    manifest_df = pd.DataFrame(manifest_rows)
    error_df = pd.DataFrame(error_rows)
    manifest_df.to_csv(paths.output_root / "manifest_trajectory_grid.csv", index=False)
    if not error_df.empty:
        error_df.to_csv(paths.output_root / "manifest_trajectory_grid_errors.csv", index=False)
    return manifest_df, error_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare A/B trajectory-grid dataset from Topo_HouseGAN raw outputs.")
    parser.add_argument("--source-root", type=pathlib.Path, default=default_source_root())
    parser.add_argument("--output-root", type=pathlib.Path, default=default_output_root())
    parser.add_argument("--plan-name", type=str, default=None, help="Optional single plan folder name.")
    parser.add_argument("--cell-size-m", type=float, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--padding-m", type=float, default=DEFAULT_PADDING_M)
    parser.add_argument("--wall-thickness-m", type=float, default=DEFAULT_WALL_THICKNESS_M)
    parser.add_argument("--filter", type=str, default="trajectory_data", help="SQLite table name filter.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit for testing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-grid", action="store_true", help="Regenerate geo/<plan>/walkablearea_grid.json before copying.")
    args = parser.parse_args()

    if args.cell_size_m <= 0:
        raise ValueError("--cell-size-m must be > 0")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be less than 1")

    paths = make_paths(args.source_root.resolve(), args.output_root.resolve())
    manifest_df, error_df = prepare_dataset(
        paths=paths,
        plan_name=args.plan_name,
        cell_size_m=round(float(args.cell_size_m), 3),
        padding_m=round(float(args.padding_m), 3),
        wall_thickness_m=round(float(args.wall_thickness_m), 3),
        table_filter=args.filter,
        split_seed=args.split_seed,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        max_cases=args.max_cases,
        overwrite=args.overwrite,
        regenerate_grid=args.regenerate_grid,
    )

    written = int((manifest_df.get("status", pd.Series(dtype=str)) == "written").sum()) if not manifest_df.empty else 0
    skipped = int((manifest_df.get("status", pd.Series(dtype=str)) == "skipped").sum()) if not manifest_df.empty else 0
    failed = len(error_df)
    print(f"[TrajectoryGrid] Done. written={written}, skipped={skipped}, failed={failed}")
    print(f"[TrajectoryGrid] Output: {paths.output_root}")


if __name__ == "__main__":
    main()

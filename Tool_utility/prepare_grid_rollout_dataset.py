"""Build grid rollout cases from existing Topo_HouseGAN trajectory cases.

Source case folders are expected to contain:
    Geo_room.json
    Geo_corridor.json
    Geo_door.json
    Spawn_location_*.csv
    Spawn_exit_*.csv
    *_trajectory_data.parquet

Output case folders contain only the agreed rollout dataset files:
    input.json
    agents_initial.parquet
    map/walkablearea_grid.json
    map/room_grid.json
    target/trajectory.parquet
    metadata.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time
from dataclasses import dataclass

import pandas as pd
import shapely.wkt
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from prepare_layout_to_grid_walkablearea import build_walkable_grid, first_existing, load_polygons


DEFAULT_CELL_SIZE_M = 0.125
DEFAULT_PADDING_M = 0.0
DEFAULT_WALL_THICKNESS_M = 0.125
FPS = 25


@dataclass(frozen=True)
class CaseSources:
    case_dir: pathlib.Path
    case_id: str
    split: str
    plan_name: str
    trajectory_path: pathlib.Path
    spawn_location_path: pathlib.Path
    spawn_exit_path: pathlib.Path | None


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def parse_plan_name(case_id: str) -> str:
    marker = "_42_"
    if marker in case_id:
        return case_id.split(marker, 1)[0]
    parts = case_id.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:-2])
    return case_id


def load_manifest(source_root: pathlib.Path) -> dict[str, dict]:
    manifest_path = source_root / "manifest_housegan_cases.csv"
    if not manifest_path.exists():
        return {}
    df = pd.read_csv(manifest_path)
    return {str(row["case_id"]): row.to_dict() for _, row in df.iterrows()}


def find_one(case_dir: pathlib.Path, pattern: str) -> pathlib.Path | None:
    matches = sorted(case_dir.glob(pattern), key=lambda p: p.name)
    return matches[0] if matches else None


def iter_source_cases(source_root: pathlib.Path) -> list[CaseSources]:
    cases: list[CaseSources] = []
    for split in ("train", "val", "test"):
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        for case_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            case_id = case_dir.name.removeprefix("case_")
            trajectory_path = find_one(case_dir, "*_trajectory_data.parquet")
            spawn_location_path = find_one(case_dir, "Spawn_location_*.csv")
            if trajectory_path is None or spawn_location_path is None:
                continue
            cases.append(
                CaseSources(
                    case_dir=case_dir,
                    case_id=case_id,
                    split=split,
                    plan_name=parse_plan_name(case_id),
                    trajectory_path=trajectory_path,
                    spawn_location_path=spawn_location_path,
                    spawn_exit_path=find_one(case_dir, "Spawn_exit_*.csv"),
                )
            )
    return cases


def world_to_grid(pos_x, pos_y, meta: dict) -> tuple[pd.Series, pd.Series]:
    cell = float(meta["cell_size_m"])
    origin_x = float(meta["origin_x"])
    origin_y = float(meta["origin_y"])
    grid_x = ((pos_x - origin_x) / cell).astype("int64")
    grid_y = ((pos_y - origin_y) / cell).astype("int64")
    return grid_x, grid_y


def build_room_grid(case_dir: pathlib.Path, walkable_payload: dict) -> dict:
    meta = walkable_payload["meta"]
    width = int(meta["width"])
    height = int(meta["height"])
    cell = float(meta["cell_size_m"])
    origin_x = float(meta["origin_x"])
    origin_y = float(meta["origin_y"])

    room_polys = load_polygons(first_existing(case_dir, "Geo_room.json", "geo_room.json"))
    corridor_polys = load_polygons(first_existing(case_dir, "Geo_corridor.json", "geo_corridor.json"))
    prepared_rooms = [(f"Room-{i}", prep(poly)) for i, poly in enumerate(room_polys)]
    prepared_corridors = [(f"Cor-{i}", prep(poly)) for i, poly in enumerate(corridor_polys)]

    label_codes: dict[str, int] = {"non_walkable": 0, "open_walkable": 1}
    for i, _ in enumerate(corridor_polys):
        label_codes[f"Cor-{i}"] = 1000 + i
    for i, _ in enumerate(room_polys):
        label_codes[f"Room-{i}"] = 2000 + i

    code_labels = {str(code): label for label, code in label_codes.items()}
    rows: list[list[int]] = []
    walkable_rows = walkable_payload["grid"]

    for row_top in range(height):
        gy = height - 1 - row_top
        y = origin_y + (gy + 0.5) * cell
        row_codes: list[int] = []
        for gx in range(width):
            if walkable_rows[row_top][gx] != "1":
                row_codes.append(label_codes["non_walkable"])
                continue

            x = origin_x + (gx + 0.5) * cell
            pt = Point(x, y)
            label = None
            for room_label, prepared in prepared_rooms:
                if prepared.covers(pt):
                    label = room_label
                    break
            if label is None:
                for corridor_label, prepared in prepared_corridors:
                    if prepared.covers(pt):
                        label = corridor_label
                        break
            if label is None:
                label = "open_walkable"
            row_codes.append(label_codes[label])
        rows.append(row_codes)

    return {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "meta": meta,
        "labels": code_labels,
        "label_codes": label_codes,
        "grid": rows,
    }


def derive_goal_room(case: CaseSources, manifest_row: dict | None) -> tuple[str | None, str]:
    if manifest_row and isinstance(manifest_row.get("end_node"), str):
        end_node = manifest_row["end_node"]
        if end_node.startswith("Room-"):
            return end_node, "manifest"

    if case.spawn_exit_path is None:
        return None, "missing_spawn_exit"

    try:
        spawn_exit = pd.read_csv(case.spawn_exit_path)
        exit_rows = spawn_exit[spawn_exit["type"].astype(str).str.lower() == "exit_area"]
        if exit_rows.empty:
            return None, "missing_exit_area"
        exit_geom = shapely.wkt.loads(str(exit_rows.iloc[0]["area"]))
        rooms = load_polygons(first_existing(case.case_dir, "Geo_room.json", "geo_room.json"))
        best_room = None
        best_area = 0.0
        for i, room in enumerate(rooms):
            area = exit_geom.intersection(room).area
            if area > best_area:
                best_area = area
                best_room = f"Room-{i}"
        if best_room is None:
            return None, "exit_area_no_room_match"
        return best_room, "spawn_exit_overlap"
    except Exception as exc:
        return None, f"goal_room_failed:{exc}"


def build_agents_initial(spawn_path: pathlib.Path, meta: dict) -> pd.DataFrame:
    df = pd.read_csv(spawn_path)
    if "id" in df.columns:
        df = df.rename(columns={"id": "agent_id"})
    keep = ["agent_id", "pos_x", "pos_y"]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"Spawn file missing columns: {missing}")
    out = df[keep].copy()
    out["grid_x"], out["grid_y"] = world_to_grid(out["pos_x"], out["pos_y"], meta)
    return out.sort_values("agent_id").reset_index(drop=True)


def build_target_trajectory(trajectory_path: pathlib.Path, meta: dict) -> pd.DataFrame:
    df = pd.read_parquet(trajectory_path)
    if "id" in df.columns:
        df = df.rename(columns={"id": "agent_id"})
    required = ["frame", "agent_id", "pos_x", "pos_y"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Trajectory missing columns: {missing}")
    df = df.sort_values(["frame", "agent_id"]).reset_index(drop=True)
    df["grid_x"], df["grid_y"] = world_to_grid(df["pos_x"], df["pos_y"], meta)
    return df


def prepare_case(
    case: CaseSources,
    output_root: pathlib.Path,
    manifest_row: dict | None,
    cell_size_m: float,
    padding_m: float,
    wall_thickness_m: float,
    overwrite: bool,
) -> dict:
    output_case_dir = output_root / case.split / case.case_dir.name
    if output_case_dir.exists():
        if not overwrite:
            return {"case_id": case.case_id, "split": case.split, "status": "skipped_exists"}
        shutil.rmtree(output_case_dir)

    for primary, fallback in (("Geo_room.json", "geo_room.json"), ("Geo_corridor.json", "geo_corridor.json"), ("Geo_door.json", "geo_door.json")):
        if not first_existing(case.case_dir, primary, fallback).exists():
            return {"case_id": case.case_id, "split": case.split, "status": "failed", "error": f"missing {primary}"}

    try:
        goal_room, goal_source = derive_goal_room(case, manifest_row)
        if goal_room is None:
            return {"case_id": case.case_id, "split": case.split, "status": "failed", "error": goal_source}

        walkable_payload = build_walkable_grid(case.case_dir, cell_size_m, padding_m, wall_thickness_m)
        room_grid_payload = build_room_grid(case.case_dir, walkable_payload)
        agents_initial = build_agents_initial(case.spawn_location_path, walkable_payload["meta"])
        trajectory = build_target_trajectory(case.trajectory_path, walkable_payload["meta"])

        map_dir = output_case_dir / "map"
        target_dir = output_case_dir / "target"
        map_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        write_json(map_dir / "walkablearea_grid.json", walkable_payload)
        write_json(map_dir / "room_grid.json", room_grid_payload)
        agents_initial.to_parquet(output_case_dir / "agents_initial.parquet", index=False)
        trajectory.to_parquet(target_dir / "trajectory.parquet", index=False)

        input_payload = {
            "schema_version": 1,
            "case_id": case.case_id,
            "plan_name": case.plan_name,
            "fps": FPS,
            "goal_room": goal_room,
            "num_agents": int(agents_initial["agent_id"].nunique()),
            "agents_initial": "agents_initial.parquet",
            "map": {
                "walkable_grid": "map/walkablearea_grid.json",
                "room_grid": "map/room_grid.json",
            },
        }
        metadata = {
            "schema_version": 1,
            "case_id": case.case_id,
            "plan_name": case.plan_name,
            "split": case.split,
            "goal_room": goal_room,
            "goal_room_source": goal_source,
            "fps": FPS,
            "num_agents": int(agents_initial["agent_id"].nunique()),
            "num_frames": int(trajectory["frame"].nunique()),
            "cell_size_m": float(walkable_payload["meta"]["cell_size_m"]),
            "padding_m": float(walkable_payload["meta"]["padding_m"]),
            "wall_thickness_m": wall_thickness_m,
            "source_case_dir": str(case.case_dir),
            "source_parquet": str(case.trajectory_path),
            "source_spawn_location": str(case.spawn_location_path),
            "source_spawn_exit": str(case.spawn_exit_path) if case.spawn_exit_path else None,
        }
        write_json(output_case_dir / "input.json", input_payload)
        write_json(output_case_dir / "metadata.json", metadata)

        return {
            "case_id": case.case_id,
            "split": case.split,
            "status": "written",
            "plan_name": case.plan_name,
            "goal_room": goal_room,
            "goal_room_source": goal_source,
            "num_agents": metadata["num_agents"],
            "num_frames": metadata["num_frames"],
            "output_case_dir": str(output_case_dir),
        }
    except Exception as exc:
        return {"case_id": case.case_id, "split": case.split, "status": "failed", "error": str(exc)}


def prepare_dataset(
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    cell_size_m: float,
    padding_m: float,
    wall_thickness_m: float,
    overwrite: bool,
    max_cases: int,
) -> pd.DataFrame:
    manifest = load_manifest(source_root)
    cases = iter_source_cases(source_root)
    if max_cases > 0:
        cases = cases[:max_cases]
    if not cases:
        raise RuntimeError(f"No source cases found under {source_root}")

    rows = []
    print(f"[GridRollout] Cases found: {len(cases)}")
    for i, case in enumerate(cases, start=1):
        row = prepare_case(
            case=case,
            output_root=output_root,
            manifest_row=manifest.get(case.case_id),
            cell_size_m=cell_size_m,
            padding_m=padding_m,
            wall_thickness_m=wall_thickness_m,
            overwrite=overwrite,
        )
        rows.append(row)
        status = row.get("status")
        detail = row.get("error") or row.get("goal_room") or ""
        print(f"[{i}/{len(cases)}] {status} {case.split}/{case.case_dir.name} {detail}")

    output_root.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output_root / "manifest_grid_rollout_cases.csv", index=False)
    return report


def default_source_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "Dataset" / "Data_Traj_Table" / "Topo_HouseGAN"


def default_output_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "Dataset" / "Data_GridRollout" / "Topo_HouseGAN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare grid rollout dataset from existing Topo_HouseGAN cases.")
    parser.add_argument("--source-root", type=pathlib.Path, default=default_source_root())
    parser.add_argument("--output-root", type=pathlib.Path, default=default_output_root())
    parser.add_argument("--cell-size-m", type=float, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--padding-m", type=float, default=DEFAULT_PADDING_M)
    parser.add_argument("--wall-thickness-m", type=float, default=DEFAULT_WALL_THICKNESS_M)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="Debug limit. 0 means all cases.")
    args = parser.parse_args()

    report = prepare_dataset(
        source_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        cell_size_m=args.cell_size_m,
        padding_m=args.padding_m,
        wall_thickness_m=args.wall_thickness_m,
        overwrite=args.overwrite,
        max_cases=args.max_cases,
    )
    counts = report["status"].value_counts().to_dict()
    print(f"[GridRollout] Done. {counts}")
    print(f"[GridRollout] Report: {args.output_root / 'manifest_grid_rollout_cases.csv'}")


if __name__ == "__main__":
    main()

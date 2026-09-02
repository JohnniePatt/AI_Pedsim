#!/usr/bin/env python3
"""Measure JuPedSim wall-clock runtime without changing source datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from shapely.geometry import shape


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = PROJECT_ROOT / "Dataset_TimeCalculate" / "Total_RefFileName.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "Dataset_TimeCalculate" / "JuPedSim_Runtime.csv"
GENERATOR_PATH = PROJECT_ROOT / "GeneratePlan_HouseGAN" / "Simulation" / "density_housegan_sim.py"
PROTECTED_ROOTS = (PROJECT_ROOT / "Dataset", PROJECT_ROOT / "Geo_scenario")
REQUIRED_JUPEDSIM_VERSION = "1.3.2"

RESULT_COLUMNS = [
    "senario_name",
    "recorded_at_utc",
    "benchmark_iteration",
    "split",
    "reference_id",
    "source_trajectory_filename",
    "source_trajectory_file",
    "plan",
    "route_index",
    "variant_id",
    "seed",
    "agent_count",
    "status",
    "plan_setup_wall_time_s",
    "setup_wall_time_s",
    "simulation_wall_time_s",
    "sqlite_save_wall_time_s",
    "trajectory_plot_wall_time_s",
    "density_heatmap_wall_time_s",
    "total_wall_time_s",
    "simulated_duration_s",
    "real_time_factor",
    "iterations",
    "temp_output_deleted",
    "error",
    "hostname",
    "os",
    "cpu",
    "python_version",
    "jupedsim_version",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text() -> str:
    return utc_now().isoformat()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_safe_output(path: Path) -> None:
    for root in PROTECTED_ROOTS:
        if is_relative_to(path, root):
            raise ValueError(f"Refusing to write to protected dataset path: {path}")


def load_generator_module():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(f"Missing simulation generator: {GENERATOR_PATH}")
    spec = importlib.util.spec_from_file_location("density_housegan_sim", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_references(
    path: Path, split: str = "all", limit: int | None = None
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "success"]
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No matching successful references found in {path}")
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def memory_percent() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return 0.0
    values: dict[str, float] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        token = rest.strip().split(" ")[0]
        try:
            values[key] = float(token)
        except ValueError:
            continue
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, ((total - available) / total) * 100.0))


def cleanup_runtime_memory() -> None:
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.close("all")
    gc.collect()


def stage_prefix(progress_label: str, reference_id: str, stage: str) -> str:
    prefix = f"{progress_label} " if progress_label else ""
    return f"{prefix}[ram={memory_percent():.1f}%] [stage={stage}] {reference_id}"


def run_with_stage_heartbeat(
    stage: str,
    reference_id: str,
    progress_label: str,
    progress_interval_s: float,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    started_ns = time.perf_counter_ns()
    print(f"{stage_prefix(progress_label, reference_id, stage)} start", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        while True:
            try:
                result = future.result(timeout=progress_interval_s)
                elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000
                print(
                    f"{stage_prefix(progress_label, reference_id, stage)} done elapsed={elapsed_s:.3f}s",
                    flush=True,
                )
                return result, elapsed_s
            except concurrent.futures.TimeoutError:
                elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000
                print(
                    f"{stage_prefix(progress_label, reference_id, stage)} running elapsed={elapsed_s:.1f}s",
                    flush=True,
                )


def require_jupedsim_version() -> str:
    version = package_version("jupedsim")
    if version != REQUIRED_JUPEDSIM_VERSION:
        raise RuntimeError(
            f"JuPedSim version mismatch: this benchmark requires exact version "
            f"{REQUIRED_JUPEDSIM_VERSION}, but found {version} in the current Python environment. "
            f"Please install or activate an environment with: python3 -m pip install jupedsim=={REQUIRED_JUPEDSIM_VERSION}"
        )
    return version


def machine_info() -> dict[str, str]:
    cpu = platform.processor() or platform.machine() or "unknown"
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu,
        "python_version": platform.python_version(),
        "jupedsim_version": require_jupedsim_version(),
    }


def close_writer(simulation: Any) -> None:
    writer = getattr(simulation, "_writer", None)
    if writer is not None and hasattr(writer, "close"):
        writer.close()


def load_plan_context(generator: Any, plan: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter_ns()
    topo_root = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
    plan_dir = topo_root / "geo" / plan
    corridor_file = plan_dir / "geo_corridor.json"
    room_file = plan_dir / "geo_room.json"
    door_file = plan_dir / "geo_door.json"
    for required in (corridor_file, room_file, door_file):
        if not required.is_file():
            raise FileNotFoundError(f"Missing plan geometry: {required}")

    corridor_polys = generator.load_polygons_from_json(corridor_file)
    room_polys = generator.load_polygons_from_json(room_file)
    doors_data = read_json(door_file)

    summary_path = topo_root / "metadata" / plan / "simulation_summary.json"
    summary = read_json(summary_path)
    config = summary.get("config", {})
    walkable_area = generator.build_walkable_area(
        corridor_polys,
        room_polys,
        doors_data,
        config.get("geometry_policy", {}),
    )
    _, node_polys = generator.build_route_graph(corridor_polys, room_polys, doors_data)
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return {
        "config": config,
        "walkable_area": walkable_area,
        "node_polys": node_polys,
    }, elapsed


def simulate_reference(
    generator: Any,
    reference: dict[str, str],
    plan_context: dict[str, Any],
    temp_sqlite: Path,
    timeout_minutes_override: float | None,
    no_progress_timeout_s: float,
    progress_label: str,
    progress_interval_s: float,
) -> dict[str, Any]:
    import jupedsim as jps

    setup_started = time.perf_counter_ns()
    route_meta = read_json(PROJECT_ROOT / reference["route_metadata_file"])
    variant = next(
        (item for item in route_meta.get("variants", []) if item.get("variant_id") == reference["variant_id"]),
        None,
    )
    if variant is None:
        raise ValueError(
            f"Missing variant={reference['variant_id']} in {reference['route_metadata_file']}"
        )

    config = plan_context["config"]
    spawn_policy = config.get("spawn_policy", {})
    agent_policy = config.get("agent_policy", {})
    walkable_area = plan_context["walkable_area"]
    safe_spawn_geojson = route_meta.get("safe_spawn_geojson")
    if not safe_spawn_geojson:
        raise ValueError(f"Missing safe_spawn_geojson in {reference['route_metadata_file']}")
    safe_spawn = shape(safe_spawn_geojson).buffer(0)
    end_node = route_meta["end_node"]
    exit_area = plan_context["node_polys"][end_node]
    clip_exit = exit_area.intersection(walkable_area).buffer(0)
    if hasattr(clip_exit, "geoms") and len(clip_exit.geoms) > 0:
        clip_exit = max(clip_exit.geoms, key=lambda polygon: polygon.area)
    if safe_spawn.is_empty or safe_spawn.area < 0.1:
        raise ValueError("safe spawn area is empty")
    if clip_exit.is_empty or clip_exit.area < 0.1:
        raise ValueError("exit area is empty")

    seed = int(variant["route_seed"])
    agent_count = int(reference["computed_agents"])
    positions = jps.distributions.distribute_by_number(
        polygon=safe_spawn,
        number_of_agents=agent_count,
        distance_to_agents=float(spawn_policy.get("distance_to_agents_m", 0.3)),
        distance_to_polygon=float(spawn_policy.get("distance_to_polygon_m", 0.15)),
        seed=seed,
    )
    if len(positions) != agent_count:
        raise ValueError(f"Distributed {len(positions)} agents; expected {agent_count}")

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=walkable_area,
        trajectory_writer=jps.SqliteTrajectoryWriter(output_file=temp_sqlite),
    )
    exit_poly = clip_exit.buffer(-0.1).convex_hull
    if exit_poly.is_empty:
        exit_poly = clip_exit.convex_hull
    exit_id = simulation.add_exit_stage(exit_poly.exterior.coords[:-1])
    journey_id = simulation.add_journey(jps.JourneyDescription([exit_id]))

    rng = np.random.default_rng(seed)
    speeds = rng.normal(
        float(agent_policy.get("desired_speed_mean", 1.34)),
        float(agent_policy.get("desired_speed_std", 0.05)),
        agent_count,
    )
    radius = float(agent_policy.get("radius_m", 0.15))
    for position, speed in zip(positions, speeds):
        simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                journey_id=journey_id,
                stage_id=exit_id,
                position=position,
                desired_speed=float(speed),
                radius=radius,
            )
        )

    setup_s = (time.perf_counter_ns() - setup_started) / 1_000_000_000
    timeout_minutes = (
        float(timeout_minutes_override)
        if timeout_minutes_override is not None
        else float(config.get("timeout_minutes", 5))
    )
    simulation_started = time.perf_counter_ns()
    last_progress_ns = simulation_started
    last_count = simulation.agent_count()
    iterations = 0
    status = "success"
    error = ""
    loop_finished_ns = simulation_started
    sqlite_save_s = 0.0
    last_heartbeat_ns = simulation_started
    print(
        f"{stage_prefix(progress_label, reference['reference_id'], 'simulation')} "
        f"start agents={agent_count}",
        flush=True,
    )

    try:
        while simulation.agent_count() > 0:
            simulation.iterate()
            iterations += 1
            now_ns = time.perf_counter_ns()
            current_count = simulation.agent_count()
            if current_count < last_count:
                last_count = current_count
                last_progress_ns = now_ns
            if (now_ns - last_heartbeat_ns) / 1_000_000_000 >= progress_interval_s:
                elapsed_s = (now_ns - simulation_started) / 1_000_000_000
                print(
                    f"{stage_prefix(progress_label, reference['reference_id'], 'simulation')} "
                    f"running elapsed={elapsed_s:.1f}s agents_left={current_count} iterations={iterations}",
                    flush=True,
                )
                last_heartbeat_ns = now_ns
            if (now_ns - simulation_started) / 1_000_000_000 > timeout_minutes * 60:
                status = "timeout"
                error = f"wall-clock timeout after {timeout_minutes:g} minutes"
                break
            if (now_ns - last_progress_ns) / 1_000_000_000 > no_progress_timeout_s:
                status = "deadlock"
                error = f"no agent progress for {no_progress_timeout_s:g} seconds"
                break
    finally:
        loop_finished_ns = time.perf_counter_ns()
        sqlite_save_started = time.perf_counter_ns()
        print(
            f"{stage_prefix(progress_label, reference['reference_id'], 'sqlite_save')} start",
            flush=True,
        )
        close_writer(simulation)
        sqlite_save_s = (time.perf_counter_ns() - sqlite_save_started) / 1_000_000_000
        print(
            f"{stage_prefix(progress_label, reference['reference_id'], 'sqlite_save')} "
            f"done elapsed={sqlite_save_s:.3f}s",
            flush=True,
        )

    simulation_s = (loop_finished_ns - simulation_started) / 1_000_000_000
    print(
        f"{stage_prefix(progress_label, reference['reference_id'], 'simulation')} "
        f"done status={status} elapsed={simulation_s:.3f}s iterations={iterations}",
        flush=True,
    )
    return {
        "seed": seed,
        "agent_count": agent_count,
        "status": status,
        "setup_wall_time_s": setup_s,
        "simulation_wall_time_s": simulation_s,
        "sqlite_save_wall_time_s": sqlite_save_s,
        "iterations": iterations,
        "error": error,
    }


def generate_density_heatmap(
    trajectory_data: Any,
    loaded_walkable_area: Any,
    output_path: Path,
    dpi: int,
    grid_size: float,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import pedpy

    individual_speed = pedpy.compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=5,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )
    individual_voronoi_cells = pedpy.compute_individual_voronoi_polygons(
        traj_data=trajectory_data,
        walkable_area=loaded_walkable_area,
        cut_off=pedpy.Cutoff(radius=0.8, quad_segments=3),
    )

    sum_density = None
    count = 0
    for frame in individual_speed["frame"].unique()[::60]:
        speed_f = individual_speed[individual_speed.frame == frame]
        cells_f = individual_voronoi_cells[individual_voronoi_cells.frame == frame]
        frame_data = pd.merge(cells_f, speed_f, on=["id", "frame"])
        density_profile, _ = pedpy.compute_profiles(
            individual_voronoi_speed_data=frame_data,
            walkable_area=loaded_walkable_area.polygon,
            grid_size=grid_size,
            speed_method=pedpy.SpeedMethod.ARITHMETIC,
        )
        if sum_density is None:
            sum_density = np.copy(density_profile[0])
        else:
            sum_density += density_profile[0]
        count += 1

    if count <= 0 or sum_density is None:
        return

    mean_density_map = sum_density / count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    pedpy.plot_profiles(
        walkable_area=loaded_walkable_area,
        profiles=[mean_density_map],
        axes=ax,
        label=r"$\rho$ / 1/$m^2$",
        vmin=0,
        vmax=5,
        title="Average Density",
    )
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def empty_result_row(senario_name: str) -> dict[str, Any]:
    row = {column: "" for column in RESULT_COLUMNS}
    row["senario_name"] = senario_name
    return row


def read_result_rows(path: Path, references: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if not path.is_file():
        return [empty_result_row(reference_id) for reference_id in references]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "senario_name" not in fieldnames and "reference_id" not in fieldnames:
            raise ValueError(f"Existing result CSV must contain senario_name or reference_id: {path}")
        rows: list[dict[str, Any]] = []
        for raw_row in reader:
            senario_name = raw_row.get("senario_name") or raw_row.get("reference_id") or ""
            if not senario_name:
                continue
            row = empty_result_row(senario_name)
            for column in RESULT_COLUMNS:
                if column in raw_row:
                    row[column] = raw_row.get(column, "")
            row["senario_name"] = senario_name
            rows.append(row)
    if not rows:
        raise ValueError(f"Existing result CSV has no scenario rows: {path}")
    missing = [row["senario_name"] for row in rows if row["senario_name"] not in references]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} senario_name value(s) are missing from the reference CSV, "
            f"for example: {preview}"
        )
    return rows


def reset_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [empty_result_row(str(row["senario_name"])) for row in rows]


def write_result_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def choose_interactive_mode() -> str:
    print("\nJuPedSim computational-time benchmark")
    print("  [0] รันใหม่ทั้งหมดและเขียนทับเฉพาะ JuPedSim_Runtime.csv")
    print("  [1] รันเฉพาะ reference ที่ยังไม่มีผล success")
    print("  [2] ออกโดยไม่รัน (ค่าเริ่มต้น)")
    choice = input("เลือก 0, 1 หรือ 2 [2]: ").strip() or "2"
    return {"0": "overwrite", "1": "missing", "2": "exit"}.get(choice, "invalid")


def confirm_interactive_run(
    mode: str,
    selected: int,
    pending: int,
    result_csv: Path,
    max_attempts: int,
    ram_cache_clear_threshold_percent: float,
) -> bool:
    print("\nสรุปก่อนรัน")
    print(f"  mode: {mode}")
    print(f"  reference ที่เลือก: {selected}")
    print(f"  simulation ที่ต้องรัน: {pending}")
    print(f"  retry ต่อ simulation: สูงสุด {max_attempts} attempt")
    print(f"  ล้าง plan cache เมื่อ RAM >= {ram_cache_clear_threshold_percent:.1f}%")
    print(f"  ไฟล์ผลลัพธ์ถาวร: {result_csv}")
    print("  ห้ามเขียนข้อมูลลง Dataset/ และ Geo_scenario/")
    return input("พิมพ์ RUN เพื่อเริ่ม: ").strip() == "RUN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark JuPedSim wall-clock time using the Data_Estimate_2 allow-list."
    )
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--mode",
        choices=("overwrite", "missing"),
        default=None,
        help="Explicit non-interactive mode. Without this flag, an interactive menu is shown on a TTY.",
    )
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--postprocess-dpi", type=int, default=220)
    parser.add_argument(
        "--stage-progress-seconds",
        type=float,
        default=10.0,
        help="Print a heartbeat for long-running simulation/postprocess stages at this interval.",
    )
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument("--no-progress-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--ram-cache-clear-threshold-percent",
        type=float,
        default=85.0,
        help="Clear the in-process plan cache before a reference when system RAM usage is at least this percent.",
    )
    parser.add_argument(
        "--max-attempts-per-reference",
        type=int,
        default=2,
        help="Retry the same reference up to this many attempts before moving to the next one.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.postprocess_dpi < 1:
        raise ValueError("--postprocess-dpi must be at least 1")
    if args.stage_progress_seconds <= 0:
        raise ValueError("--stage-progress-seconds must be positive")
    if args.max_attempts_per_reference < 1:
        raise ValueError("--max-attempts-per-reference must be at least 1")
    if args.ram_cache_clear_threshold_percent <= 0 or args.ram_cache_clear_threshold_percent > 100:
        raise ValueError("--ram-cache-clear-threshold-percent must be in (0, 100]")
    if args.no_progress_timeout_seconds <= 0:
        raise ValueError("--no-progress-timeout-seconds must be positive")
    if args.timeout_minutes is not None and args.timeout_minutes <= 0:
        raise ValueError("--timeout-minutes must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    require_jupedsim_version()
    reference_csv = args.reference_csv.resolve()
    result_csv = args.output_csv.resolve()
    assert_safe_output(result_csv)
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Missing reference CSV: {reference_csv}")

    all_references_list = read_references(reference_csv, "all")
    references_map = {row["reference_id"]: row for row in all_references_list}

    result_rows = read_result_rows(result_csv, references_map)
    row_index_by_senario = {row["senario_name"]: i for i, row in enumerate(result_rows)}
    selected_references: list[dict[str, str]] = []
    for row in result_rows:
        reference = references_map[row["senario_name"]]
        if args.split == "all" or reference.get("split") == args.split:
            selected_references.append(reference)
    if args.limit is not None:
        selected_references = selected_references[: args.limit]
    if not selected_references:
        raise ValueError(f"No selected references match split={args.split!r}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "reference_csv": str(reference_csv),
                    "split": args.split,
                    "selected_references": len(selected_references),
                    "warmup": args.warmup,
                    "postprocess_dpi": args.postprocess_dpi,
                    "stage_progress_seconds": args.stage_progress_seconds,
                    "max_attempts_per_reference": args.max_attempts_per_reference,
                    "ram_cache_clear_threshold_percent": args.ram_cache_clear_threshold_percent,
                    "planned_rows": len(selected_references),
                    "output_csv": str(result_csv),
                    "source_writes": False,
                },
                indent=2,
            )
        )
        return 0

    interactive = args.mode is None and sys.stdin.isatty()
    if interactive:
        mode = choose_interactive_mode()
        if mode == "invalid":
            print("Invalid selection. Nothing was run.")
            return 2
        if mode == "exit":
            print("Exit: nothing was run.")
            return 0
    elif args.mode is None:
        print("No --mode supplied in a non-interactive session; nothing was run.")
        print("Use --dry-run, --mode missing, or --mode overwrite.")
        return 0
    else:
        mode = args.mode

    if mode == "overwrite":
        result_rows = reset_result_rows(result_rows)
        write_result_rows(result_csv, result_rows)

    # Determine pending scenarios
    pending_references = []
    for ref in selected_references:
        senario_name = ref["reference_id"]
        idx = row_index_by_senario.get(senario_name)
        if idx is None:
            continue
        existing_status = result_rows[idx].get("status")
        if mode == "overwrite" or existing_status != "success":
            pending_references.append(ref)

    pending = len(pending_references)
    if interactive and not confirm_interactive_run(
        mode,
        len(selected_references),
        pending,
        result_csv,
        args.max_attempts_per_reference,
        args.ram_cache_clear_threshold_percent,
    ):
        print("Confirmation not received. Nothing was run.")
        return 0
    if pending == 0:
        print(f"All selected references already have successful results in {result_csv}")
        return 0

    generator = load_generator_module()
    environment = machine_info()
    plan_cache: dict[str, dict[str, Any]] = {}

    def execute(
        reference: dict[str, str],
        iteration: int,
        progress_label: str = "",
    ) -> dict[str, Any]:
        nonlocal plan_cache
        cleanup_runtime_memory()
        ram_before_s = memory_percent()
        if ram_before_s >= args.ram_cache_clear_threshold_percent and plan_cache:
            plan_cache.clear()
            cleanup_runtime_memory()
            ram_before_s = memory_percent()
        plan = reference["plan"]
        plan_setup_s = 0.0
        if plan not in plan_cache:
            print(
                f"{stage_prefix(progress_label, reference['reference_id'], 'plan_setup')} start plan={plan}",
                flush=True,
            )
            plan_cache[plan], plan_setup_s = load_plan_context(generator, plan)
            print(
                f"{stage_prefix(progress_label, reference['reference_id'], 'plan_setup')} "
                f"done elapsed={plan_setup_s:.3f}s plan={plan}",
                flush=True,
            )

        temp_deleted = False
        benchmark: dict[str, Any] = {
            "seed": "",
            "agent_count": reference["computed_agents"],
            "status": "error",
            "setup_wall_time_s": 0.0,
            "simulation_wall_time_s": 0.0,
            "sqlite_save_wall_time_s": 0.0,
            "iterations": 0,
            "error": "",
        }
        trajectory_plot_s = 0.0
        density_heatmap_s = 0.0

        with tempfile.TemporaryDirectory(prefix="jupedsim_runtime_") as temp_directory:
            temp_sqlite = Path(temp_directory) / "trajectory.sqlite"
            try:
                benchmark = simulate_reference(
                    generator,
                    reference,
                    plan_cache[plan],
                    temp_sqlite,
                    args.timeout_minutes,
                    args.no_progress_timeout_seconds,
                    progress_label,
                    args.stage_progress_seconds,
                )
                if benchmark["status"] == "success":
                    temp_traj_dir = Path(temp_directory) / "traj_out"
                    temp_traj_dir.mkdir(parents=True, exist_ok=True)
                    (trajectory_data, loaded_walkable_area), trajectory_plot_s = run_with_stage_heartbeat(
                        "trajectory_plot",
                        reference["reference_id"],
                        progress_label,
                        args.stage_progress_seconds,
                        generator.generate_trajectory_plot,
                        temp_sqlite,
                        reference["reference_id"],
                        temp_traj_dir,
                        args.postprocess_dpi,
                    )

                    temp_density_png = Path(temp_directory) / "density.png"
                    _, density_heatmap_s = run_with_stage_heartbeat(
                        "density_heatmap",
                        reference["reference_id"],
                        progress_label,
                        args.stage_progress_seconds,
                        generate_density_heatmap,
                        trajectory_data,
                        loaded_walkable_area,
                        temp_density_png,
                        dpi=args.postprocess_dpi,
                        grid_size=float(plan_cache[plan]["config"].get("grid_size", 0.5)),
                    )
            except Exception as exc:
                benchmark["status"] = "error"
                benchmark["error"] = f"{type(exc).__name__}: {exc}"
        temp_deleted = not Path(temp_directory).exists()

        total_s = (
            plan_setup_s
            + float(benchmark.get("setup_wall_time_s", 0.0))
            + float(benchmark.get("simulation_wall_time_s", 0.0))
            + float(benchmark.get("sqlite_save_wall_time_s", 0.0))
            + trajectory_plot_s
            + density_heatmap_s
        )
        simulated_duration_s = float(reference["simulated_duration_s"])
        row = {
            "senario_name": reference["reference_id"],
            "recorded_at_utc": utc_text(),
            "benchmark_iteration": iteration,
            "split": reference["split"],
            "reference_id": reference["reference_id"],
            "source_trajectory_filename": reference["source_trajectory_filename"],
            "source_trajectory_file": reference["source_trajectory_file"],
            "plan": plan,
            "route_index": reference["route_index"],
            "variant_id": reference["variant_id"],
            "seed": benchmark.get("seed", ""),
            "agent_count": benchmark.get("agent_count", reference["computed_agents"]),
            "status": benchmark["status"],
            "plan_setup_wall_time_s": f"{plan_setup_s:.9f}",
            "setup_wall_time_s": f"{float(benchmark.get('setup_wall_time_s', 0.0)):.9f}",
            "simulation_wall_time_s": f"{float(benchmark.get('simulation_wall_time_s', 0.0)):.9f}",
            "sqlite_save_wall_time_s": f"{float(benchmark.get('sqlite_save_wall_time_s', 0.0)):.9f}",
            "trajectory_plot_wall_time_s": f"{trajectory_plot_s:.9f}",
            "density_heatmap_wall_time_s": f"{density_heatmap_s:.9f}",
            "total_wall_time_s": f"{total_s:.9f}",
            "simulated_duration_s": f"{simulated_duration_s:.9f}",
            "real_time_factor": f"{simulated_duration_s / total_s:.9f}" if total_s > 0 else "",
            "iterations": benchmark.get("iterations", 0),
            "temp_output_deleted": str(temp_deleted).lower(),
            "error": benchmark.get("error", ""),
            **environment,
        }
        cleanup_runtime_memory()
        ram_after_s = memory_percent()
        prefix = f"{progress_label} " if progress_label else ""
        print(
            f"{prefix}[ram={ram_after_s:.1f}%] [{row['status']}] {reference['reference_id']} "
            f"total={row['total_wall_time_s']}s temp_deleted={row['temp_output_deleted']}"
        )
        return row

    if args.warmup > 0 and selected_references:
        for warmup_index in range(args.warmup):
            print(f"[warmup {warmup_index + 1}/{args.warmup}] {selected_references[0]['reference_id']}")
            execute(selected_references[0], 0)
        plan_cache.clear()

    completed_attempts = 0
    for reference in pending_references:
        completed_attempts += 1
        final_row: dict[str, Any] | None = None
        for attempt in range(1, args.max_attempts_per_reference + 1):
            attempt_label = (
                f" retry={attempt}/{args.max_attempts_per_reference}"
                if attempt > 1
                else ""
            )
            final_row = execute(
                reference,
                1,
                progress_label=f"[{completed_attempts}/{pending}]{attempt_label}",
            )
            if final_row["status"] == "success":
                break
            if attempt < args.max_attempts_per_reference:
                print(
                    f"[{completed_attempts}/{pending}] retrying "
                    f"{reference['reference_id']} after status={final_row['status']}"
                )
        if final_row is None:
            raise RuntimeError("Internal error: no benchmark attempt was executed")

        # Update result_rows in-place and write to disk
        senario_name = reference["reference_id"]
        idx = row_index_by_senario.get(senario_name)
        if idx is None:
            raise RuntimeError(f"Missing senario_name in result template: {senario_name}")
        result_rows[idx] = final_row
        write_result_rows(result_csv, result_rows)

    print(f"Results: {result_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure JuPedSim wall-clock runtime without changing source datasets."""

from __future__ import annotations

import argparse
import csv
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

import numpy as np
from shapely.geometry import shape


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = PROJECT_ROOT / "Dataset_TimeCalculate" / "Total_RefFileName.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "Dataset_TimeCalculate" / "JuPedSim_Runtime.csv"
GENERATOR_PATH = PROJECT_ROOT / "GeneratePlan_HouseGAN" / "Simulation" / "density_housegan_sim.py"
PROTECTED_ROOTS = (PROJECT_ROOT / "Dataset", PROJECT_ROOT / "Geo_scenario")

RESULT_COLUMNS = [
    "run_id",
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
    return utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_run_id() -> str:
    return utc_now().strftime("run_%Y%m%dT%H%M%SZ")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_safe_output(path: Path) -> None:
    resolved = path.resolve()
    for protected in PROTECTED_ROOTS:
        if is_relative_to(resolved, protected.resolve()):
            raise ValueError(f"Output is inside protected source tree: {resolved}")


def load_generator_module():
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(f"Missing JuPedSim generator: {GENERATOR_PATH}")
    spec = importlib.util.spec_from_file_location("housegan_density_sim_benchmark_source", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load generator module: {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_references(path: Path, split: str, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    rows = [row for row in rows if row.get("status") == "success"]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No successful references selected for split={split!r}")
    return rows


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def machine_info() -> dict[str, str]:
    cpu = platform.processor().strip()
    if not cpu and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu": cpu or "unknown",
        "python_version": platform.python_version(),
        "jupedsim_version": package_version("jupedsim"),
    }


def close_writer(simulation: Any) -> None:
    writer = getattr(simulation, "_writer", None)
    if writer is not None:
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
    return {"walkable_area": walkable_area, "node_polys": node_polys, "config": config}, elapsed


def simulate_reference(
    generator: Any,
    reference: dict[str, str],
    plan_context: dict[str, Any],
    temp_sqlite: Path,
    timeout_minutes_override: float | None,
    no_progress_timeout_s: float,
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

    try:
        while simulation.agent_count() > 0:
            simulation.iterate()
            iterations += 1
            now_ns = time.perf_counter_ns()
            current_count = simulation.agent_count()
            if current_count < last_count:
                last_count = current_count
                last_progress_ns = now_ns
            if (now_ns - simulation_started) / 1_000_000_000 > timeout_minutes * 60:
                status = "timeout"
                error = f"wall-clock timeout after {timeout_minutes:g} minutes"
                break
            if (now_ns - last_progress_ns) / 1_000_000_000 > no_progress_timeout_s:
                status = "deadlock"
                error = f"no agent progress for {no_progress_timeout_s:g} seconds"
                break
    finally:
        close_writer(simulation)

    simulation_s = (time.perf_counter_ns() - simulation_started) / 1_000_000_000
    return {
        "seed": seed,
        "agent_count": agent_count,
        "status": status,
        "setup_wall_time_s": setup_s,
        "simulation_wall_time_s": simulation_s,
        "iterations": iterations,
        "error": error,
    }


def completed_keys(result_csv: Path) -> set[tuple[str, str]]:
    if not result_csv.is_file():
        return set()
    with result_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(
                f"Existing result CSV has an incompatible header: {result_csv}"
            )
        return {
            (row["reference_id"], row["benchmark_iteration"])
            for row in reader
            if row.get("status") == "success"
        }


def append_result(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())


def initialize_result(path: Path) -> None:
    """Replace only the benchmark CSV after the user explicitly selects mode 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n").writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def choose_interactive_mode() -> str:
    print("\nJuPedSim computational-time benchmark")
    print("  [0] รันใหม่ทั้งหมดและเขียนทับเฉพาะ JuPedSim_Runtime.csv")
    print("  [1] รันเฉพาะ reference ที่ยังไม่มีผล success")
    print("  [2] ออกโดยไม่รัน (ค่าเริ่มต้น)")
    choice = input("เลือก 0, 1 หรือ 2 [2]: ").strip() or "2"
    return {"0": "overwrite", "1": "missing", "2": "exit"}.get(choice, "invalid")


def confirm_interactive_run(mode: str, selected: int, pending: int, result_csv: Path) -> bool:
    print("\nสรุปก่อนรัน")
    print(f"  mode: {mode}")
    print(f"  reference ที่เลือก: {selected}")
    print(f"  simulation ที่ต้องรัน: {pending}")
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
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-minutes", type=float, default=None)
    parser.add_argument("--no-progress-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.no_progress_timeout_seconds <= 0:
        raise ValueError("--no-progress-timeout-seconds must be positive")
    if args.timeout_minutes is not None and args.timeout_minutes <= 0:
        raise ValueError("--timeout-minutes must be positive")


def main() -> int:
    args = parse_args()
    validate_args(args)
    reference_csv = args.reference_csv.resolve()
    result_csv = args.output_csv.resolve()
    assert_safe_output(result_csv)
    if not reference_csv.is_file():
        raise FileNotFoundError(f"Missing reference CSV: {reference_csv}")
    references = read_references(reference_csv, args.split, args.limit)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "reference_csv": str(reference_csv),
                    "split": args.split,
                    "selected_references": len(references),
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "planned_rows": len(references) * args.repeats,
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

    existing_done = completed_keys(result_csv)
    all_keys = {
        (reference["reference_id"], str(iteration))
        for iteration in range(1, args.repeats + 1)
        for reference in references
    }
    done = set() if mode == "overwrite" else existing_done
    pending = len(all_keys - done)
    if interactive and not confirm_interactive_run(mode, len(all_keys), pending, result_csv):
        print("Confirmation not received. Nothing was run.")
        return 0
    if pending == 0:
        print(f"All selected references already have successful results in {result_csv}")
        return 0

    if mode == "overwrite":
        initialize_result(result_csv)
    else:
        result_csv.parent.mkdir(parents=True, exist_ok=True)

    run_id = new_run_id()
    generator = load_generator_module()
    environment = machine_info()
    plan_cache: dict[str, dict[str, Any]] = {}

    def execute(reference: dict[str, str], iteration: int, record: bool) -> None:
        nonlocal plan_cache
        plan = reference["plan"]
        plan_setup_s = 0.0
        if plan not in plan_cache:
            plan_cache[plan], plan_setup_s = load_plan_context(generator, plan)

        temp_deleted = False
        benchmark: dict[str, Any] = {
            "seed": "",
            "agent_count": reference["computed_agents"],
            "status": "error",
            "setup_wall_time_s": 0.0,
            "simulation_wall_time_s": 0.0,
            "iterations": 0,
            "error": "",
        }
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
                )
            except Exception as exc:
                benchmark["status"] = "error"
                benchmark["error"] = f"{type(exc).__name__}: {exc}"
        temp_deleted = not Path(temp_directory).exists()

        total_s = (
            plan_setup_s
            + float(benchmark.get("setup_wall_time_s", 0.0))
            + float(benchmark.get("simulation_wall_time_s", 0.0))
        )
        simulated_duration_s = float(reference["simulated_duration_s"])
        row = {
            "run_id": run_id,
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
            "total_wall_time_s": f"{total_s:.9f}",
            "simulated_duration_s": f"{simulated_duration_s:.9f}",
            "real_time_factor": f"{simulated_duration_s / total_s:.9f}" if total_s > 0 else "",
            "iterations": benchmark.get("iterations", 0),
            "temp_output_deleted": str(temp_deleted).lower(),
            "error": benchmark.get("error", ""),
            **environment,
        }
        if record:
            append_result(result_csv, row)
            print(
                f"[{row['status']}] {reference['reference_id']} "
                f"total={row['total_wall_time_s']}s temp_deleted={row['temp_output_deleted']}"
            )

    for warmup_index in range(args.warmup):
        print(f"[warmup {warmup_index + 1}/{args.warmup}] {references[0]['reference_id']}")
        execute(references[0], 0, record=False)
    plan_cache.clear()

    for iteration in range(1, args.repeats + 1):
        for reference in references:
            key = (reference["reference_id"], str(iteration))
            if key in done:
                print(f"[skip] {reference['reference_id']} iteration={iteration}")
                continue
            execute(reference, iteration, record=True)

    print(f"Results: {result_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

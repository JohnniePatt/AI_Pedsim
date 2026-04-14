import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_ARCH_PATH = PROJECT_ROOT / "Prepare_data" / "Architecture_housePlan"
if str(OLD_ARCH_PATH) not in sys.path:
    sys.path.append(str(OLD_ARCH_PATH))

import jupedsim as jps
import pedpy
from bottleneck_archhouseplan import generate_heatmaps, generate_trajectory_plot, load_polygons_from_json


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_config(config_path):
    cfg = read_json(config_path)
    cfg.setdefault("scenario_name", "Topo_HouseGAN")
    cfg.setdefault("spawn_policy", {})
    cfg.setdefault("agent_policy", {})
    cfg.setdefault("geometry_policy", {})
    return cfg


def build_walkable_area(corridor_polys, room_polys, doors_data, geometry_policy):
    union_buffer = float(geometry_policy.get("union_buffer_m", 0.02))
    wall_thickness = float(geometry_policy.get("wall_thickness_m", 0.2))
    default_door_width = float(geometry_policy.get("door_width_m", 1.5))
    door_depth_multiplier = float(geometry_policy.get("door_depth_multiplier", 2.5))

    all_geoms = corridor_polys + room_polys
    footprint = unary_union([p.buffer(union_buffer) for p in all_geoms]).buffer(-union_buffer)
    walls = unary_union([poly.exterior.buffer(wall_thickness / 2, join_style=2) for poly in all_geoms])

    door_cutouts = []
    for door in doors_data:
        px, py = door["pos"]
        door_width = float(door.get("door_width", default_door_width))
        door_depth = wall_thickness * door_depth_multiplier
        if door.get("horizontal", False):
            cutout = box(px - door_width / 2, py - door_depth / 2, px + door_width / 2, py + door_depth / 2)
        else:
            cutout = box(px - door_depth / 2, py - door_width / 2, px + door_depth / 2, py + door_width / 2)
        door_cutouts.append(cutout)

    if door_cutouts:
        walls = walls.difference(unary_union(door_cutouts))

    walkable = footprint.difference(walls)
    if hasattr(walkable, "geoms"):
        walkable = max(walkable.geoms, key=lambda p: p.area)

    return walkable.buffer(0)


def build_route_graph(corridor_polys, room_polys, doors_data):
    graph = nx.Graph()
    node_polys = {}

    for i, poly in enumerate(corridor_polys):
        node_id = f"Cor-{i}"
        graph.add_node(node_id, type="Corridor")
        node_polys[node_id] = poly
    for i, poly in enumerate(room_polys):
        node_id = f"Room-{i}"
        graph.add_node(node_id, type="Room")
        node_polys[node_id] = poly

    for door in doors_data:
        rooms = door.get("rooms", [])
        if len(rooms) == 2 and rooms[0] in graph.nodes and rooms[1] in graph.nodes:
            graph.add_edge(rooms[0], rooms[1])

    return graph, node_polys


def find_longest_routes(graph):
    if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
        return []

    diameter = nx.diameter(graph)
    periphery = nx.periphery(graph)
    routes = []
    seen = set()

    for i, start in enumerate(periphery):
        for end in periphery[i + 1 :]:
            path = nx.shortest_path(graph, start, end)
            if len(path) - 1 != diameter:
                continue
            pair = tuple(sorted((start, end)))
            if pair in seen:
                continue
            seen.add(pair)
            routes.append((start, end, path))

    return routes


def choose_safe_spawn_area(raw_spawn_area, walkable_area, requested_offset):
    clipped = raw_spawn_area.intersection(walkable_area).buffer(0)
    if hasattr(clipped, "geoms"):
        clipped = max(clipped.geoms, key=lambda p: p.area)

    offsets = [requested_offset, 0.75, 0.5, 0.25, 0.1, 0.0]
    offsets = sorted({max(0.0, float(v)) for v in offsets}, reverse=True)
    for offset in offsets:
        safe = clipped.buffer(-offset).buffer(0) if offset > 0 else clipped
        if hasattr(safe, "geoms") and len(safe.geoms) > 0:
            safe = max(safe.geoms, key=lambda p: p.area)
        if not safe.is_empty and safe.area > 0.1:
            return clipped, safe, offset

    return clipped, clipped, 0.0


def compute_agent_count(safe_area, spawn_policy):
    sqm_per_person = float(spawn_policy.get("sqm_per_person", 2.0))
    min_agents = int(spawn_policy.get("min_agents", 1))
    max_agents = int(spawn_policy.get("max_agents", 250))
    raw_count = math.floor(safe_area / max(0.01, sqm_per_person))
    return int(max(min_agents, min(max_agents, raw_count)))


def simulation_variants(base_agent_count):
    base_agent_count = max(1, int(base_agent_count))
    return [
        {
            "id": "full",
            "label": "N Agent",
            "agent_count": base_agent_count,
            "description": "Agent count from configured density",
        },
        {
            "id": "half",
            "label": "N/2 Agent",
            "agent_count": max(1, base_agent_count // 2),
            "description": "Half of configured density result",
        },
        {
            "id": "single",
            "label": "1 Agent",
            "agent_count": 1,
            "description": "Single-agent baseline",
        },
    ]


def run_density_simulation(seed, num_agents, walkable_area, trajectory_file, safe_spawn_area, exit_area, spawn_policy, agent_policy, timeout_minutes):
    clip_exit = exit_area.intersection(walkable_area).buffer(0)
    if hasattr(clip_exit, "geoms") and len(clip_exit.geoms) > 0:
        clip_exit = max(clip_exit.geoms, key=lambda p: p.area)

    if safe_spawn_area.is_empty or safe_spawn_area.area < 0.1:
        return False, [], "safe spawn area is empty"
    if clip_exit.is_empty or clip_exit.area < 0.1:
        return False, [], "exit area is empty"

    distance_to_agents = float(spawn_policy.get("distance_to_agents_m", 0.3))
    distance_to_polygon = float(spawn_policy.get("distance_to_polygon_m", 0.15))

    try:
        positions = jps.distributions.distribute_by_number(
            polygon=safe_spawn_area,
            number_of_agents=num_agents,
            distance_to_agents=distance_to_agents,
            distance_to_polygon=distance_to_polygon,
            seed=seed,
        )
    except Exception as exc:
        return False, [], f"failed to distribute agents: {exc}"

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=walkable_area,
        trajectory_writer=jps.SqliteTrajectoryWriter(output_file=trajectory_file),
    )

    exit_poly = clip_exit.buffer(-0.1).convex_hull
    if exit_poly.is_empty:
        exit_poly = clip_exit.convex_hull
    exit_id = simulation.add_exit_stage(exit_poly.exterior.coords[:-1])
    journey = jps.JourneyDescription([exit_id])
    journey_id = simulation.add_journey(journey)

    rng = np.random.default_rng(seed)
    speeds = rng.normal(float(agent_policy.get("desired_speed_mean", 1.34)), float(agent_policy.get("desired_speed_std", 0.05)), num_agents)
    radius = float(agent_policy.get("radius_m", 0.15))

    for pos, speed in zip(positions, speeds):
        simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                journey_id=journey_id,
                stage_id=exit_id,
                position=pos,
                desired_speed=float(speed),
                radius=radius,
            )
        )

    start_time = time.time()
    last_change_time = time.time()
    max_duration_seconds = timeout_minutes * 60
    no_progress_timeout = 60

    with tqdm(total=num_agents, desc=f"Simulating {seed}") as pbar:
        last_count = num_agents
        while simulation.agent_count() > 0:
            simulation.iterate()
            now = time.time()
            if now - start_time > max_duration_seconds:
                simulation._writer.close()
                if Path(trajectory_file).exists():
                    os.remove(trajectory_file)
                return False, positions, f"timeout after {timeout_minutes} minutes"

            current_count = simulation.agent_count()
            if current_count < last_count:
                pbar.update(last_count - current_count)
                last_count = current_count
                last_change_time = now
            elif now - last_change_time > no_progress_timeout:
                simulation._writer.close()
                if Path(trajectory_file).exists():
                    os.remove(trajectory_file)
                return False, positions, f"deadlock: no progress for {no_progress_timeout}s"

    simulation._writer.close()
    return True, positions, ""


def plot_polygon(ax, geom, **kwargs):
    if geom.is_empty:
        return
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            plot_polygon(ax, part, **kwargs)
        return
    ax.fill(*geom.exterior.xy, **kwargs)


def set_focus_bounds(ax, geom, padding=1.0):
    if geom.is_empty:
        return
    minx, miny, maxx, maxy = geom.bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad = max(float(padding), 0.15 * max(width, height))
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def set_full_plan_bounds(ax, walkable_area, padding=1.0):
    if walkable_area.is_empty:
        return
    minx, miny, maxx, maxy = walkable_area.bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    pad = max(float(padding), 0.05 * max(width, height))
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def plot_offset_area(walkable_area, raw_spawn, clipped_spawn, safe_spawn, used_offset, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    pedpy.plot_walkable_area(walkable_area=pedpy.WalkableArea(walkable_area), axes=ax)
    if not clipped_spawn.is_empty:
        plot_polygon(ax, clipped_spawn, color="#d1d5db", alpha=0.55, label=f"Walkable spawn clip: {clipped_spawn.area:.2f} m2")
    if not raw_spawn.is_empty:
        ax.plot(*raw_spawn.exterior.xy, color="#6b7280", linewidth=1.8, label=f"Raw spawn room: {raw_spawn.area:.2f} m2")
    if not safe_spawn.is_empty:
        plot_polygon(ax, safe_spawn, color="#2563eb", alpha=0.5, label=f"After offset {used_offset:.2f} m: {safe_spawn.area:.2f} m2")
    set_full_plan_bounds(ax, walkable_area, padding=1.0)
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spawn_policy(walkable_area, raw_spawn, safe_spawn, positions, exit_area, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    pedpy.plot_walkable_area(walkable_area=pedpy.WalkableArea(walkable_area), axes=ax)
    if not raw_spawn.is_empty:
        plot_polygon(ax, raw_spawn, color="#bdbdbd", alpha=0.35, label="Raw spawn room")
    if not safe_spawn.is_empty:
        plot_polygon(ax, safe_spawn, color="#64b5f6", alpha=0.45, label="Offset safe spawn")
    if not exit_area.is_empty:
        plot_polygon(ax, exit_area, color="#ef5350", alpha=0.35, label="Exit room")
    if positions:
        ax.scatter(*zip(*positions), s=8, color="#1e40af", label="Agents")
    set_full_plan_bounds(ax, walkable_area, padding=1.0)
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_plan(plan_name, cfg):
    scenario_name = cfg.get("scenario_name", "Topo_HouseGAN")
    topo_root = PROJECT_ROOT / "Geo_scenario" / scenario_name
    plan_dir = topo_root / "geo" / plan_name
    if not plan_dir.exists():
        print(f"[Sim] Missing plan directory: {plan_dir}")
        return False

    corridor_file = plan_dir / "geo_corridor.json"
    room_file = plan_dir / "geo_room.json"
    door_file = plan_dir / "geo_door.json"
    if not corridor_file.exists() or not room_file.exists() or not door_file.exists():
        print(f"[Sim] Missing geometry files in {plan_dir}")
        return False

    corridor_polys = load_polygons_from_json(corridor_file)
    room_polys = load_polygons_from_json(room_file)
    doors_data = read_json(door_file)

    walkable_area = build_walkable_area(corridor_polys, room_polys, doors_data, cfg.get("geometry_policy", {}))
    graph, node_polys = build_route_graph(corridor_polys, room_polys, doors_data)
    routes = find_longest_routes(graph)
    if not routes:
        print(f"[Sim] No valid connected longest route for {plan_name}")
        return False

    output_dirs = {
        "dataswarm": topo_root / "dataswarm" / plan_name,
        "density": topo_root / "heatmap_density" / plan_name,
        "speed": topo_root / "heatmap_speed" / plan_name,
        "trajectory": topo_root / "trajectory_line" / plan_name,
        "spawn_exit": topo_root / "spawn_exit" / plan_name,
        "offset_area": topo_root / "offset_area" / plan_name,
        "metadata": topo_root / "metadata" / plan_name,
        "previews": topo_root / "previews" / plan_name,
    }
    if not cfg.get("preview_only", False):
        for key, output_dir in output_dirs.items():
            if output_dir.exists() and key in {"dataswarm", "density", "speed", "trajectory", "spawn_exit", "offset_area", "metadata", "previews"}:
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    spawn_policy = cfg.get("spawn_policy", {})
    agent_policy = cfg.get("agent_policy", {})
    requested_offset = float(spawn_policy.get("wall_offset_m", 1.0))
    grid_size = float(cfg.get("grid_size", 0.5))
    timeout_minutes = int(cfg.get("timeout_minutes", 5))

    summary = {
        "plan_name": plan_name,
        "scenario_name": scenario_name,
        "route_count": len(routes),
        "walkable_area_m2": float(walkable_area.area),
        "config": cfg,
        "routes": [],
    }
    write_json(output_dirs["metadata"] / "simulation_config.json", cfg)

    print(f"[Sim] Plan={plan_name} routes={len(routes)} walkable={walkable_area.area:.2f} m2")
    for route_idx, (start_nid, end_nid, path) in enumerate(routes):
        route_seed = seed + route_idx
        suffix = f"{route_seed}_{route_idx:02d}"
        trajectory_file = output_dirs["dataswarm"] / f"plan_sim_{suffix}.sqlite"
        raw_spawn = node_polys[start_nid]
        exit_area = node_polys[end_nid]
        clipped_spawn, safe_spawn, used_offset = choose_safe_spawn_area(raw_spawn, walkable_area, requested_offset)
        agent_count = compute_agent_count(float(safe_spawn.area), spawn_policy)

        route_meta = {
            "route_index": route_idx,
            "route_seed": route_seed,
            "start_node": start_nid,
            "end_node": end_nid,
            "topological_path": path,
            "raw_spawn_area_m2": float(raw_spawn.area),
            "clipped_spawn_area_m2": float(clipped_spawn.area) if not clipped_spawn.is_empty else 0.0,
            "safe_spawn_area_m2": float(safe_spawn.area) if not safe_spawn.is_empty else 0.0,
            "wall_offset_requested_m": requested_offset,
            "wall_offset_used_m": used_offset,
            "sqm_per_person": float(spawn_policy.get("sqm_per_person", 2.0)),
            "computed_agents": agent_count,
            "status": "pending",
        }

        if cfg.get("preview_only", False):
            if trajectory_file.exists():
                trajectory_data, loaded_walkable_area = generate_trajectory_plot(trajectory_file, suffix, output_dirs["trajectory"], 220)
                generate_heatmaps(trajectory_data, loaded_walkable_area, suffix, output_dirs["density"], output_dirs["speed"], 220, grid_size=grid_size)
            route_meta["status"] = "preview_only"
            summary["routes"].append(route_meta)
            continue

        route_meta["variants"] = []
        for variant_idx, variant in enumerate(simulation_variants(agent_count)):
            variant_seed = route_seed + (variant_idx * 100000)
            variant_suffix = f"{variant_seed}_{route_idx:02d}_{variant['id']}"
            trajectory_file = output_dirs["dataswarm"] / f"plan_sim_{variant_suffix}.sqlite"

            print(
                f"[Sim] route={route_idx:02d} {start_nid}->{end_nid} variant={variant['id']} "
                f"safe_area={safe_spawn.area:.2f}m2 offset={used_offset:.2f}m agents={variant['agent_count']}"
            )
            ok, positions, error = run_density_simulation(
                variant_seed,
                variant["agent_count"],
                walkable_area,
                trajectory_file,
                safe_spawn,
                exit_area,
                spawn_policy,
                agent_policy,
                timeout_minutes,
            )

            variant_meta = {
                "variant_id": variant["id"],
                "variant_label": variant["label"],
                "variant_description": variant["description"],
                "route_seed": variant_seed,
                "computed_agents": variant["agent_count"],
                "status": "success" if ok else "failed",
                "error": error,
                "trajectory_file": str(trajectory_file),
                "spawn_positions_preview": [[float(p[0]), float(p[1])] for p in positions[:20]],
                "agent_count_distributed": len(positions),
                "raw_spawn_geojson": mapping(raw_spawn) if not raw_spawn.is_empty else None,
                "clipped_spawn_geojson": mapping(clipped_spawn) if not clipped_spawn.is_empty else None,
                "safe_spawn_geojson": mapping(safe_spawn) if not safe_spawn.is_empty else None,
            }
            route_meta["variants"].append(variant_meta)

            plot_offset_area(
                walkable_area,
                raw_spawn,
                clipped_spawn,
                safe_spawn,
                used_offset,
                f"Offset Area | {plan_name} | {start_nid}->{end_nid} | {variant['label']}",
                output_dirs["offset_area"] / f"offset_area_{variant_suffix}.png",
            )

            plot_spawn_policy(
                walkable_area,
                raw_spawn,
                safe_spawn,
                positions,
                exit_area,
                f"Density Spawn | {plan_name} | {start_nid}->{end_nid} | {variant['label']}",
                output_dirs["spawn_exit"] / f"spawn_exit_{variant_suffix}.png",
            )

            if ok:
                trajectory_data, loaded_walkable_area = generate_trajectory_plot(trajectory_file, variant_suffix, output_dirs["trajectory"], 220)
                generate_heatmaps(trajectory_data, loaded_walkable_area, variant_suffix, output_dirs["density"], output_dirs["speed"], 220, grid_size=grid_size)

            plt.close("all")
            gc.collect()

        full_variant = next((v for v in route_meta["variants"] if v["variant_id"] == "full"), route_meta["variants"][0])
        route_meta.update(
            {
                "status": full_variant["status"],
                "error": full_variant.get("error", ""),
                "trajectory_file": full_variant.get("trajectory_file", ""),
                "spawn_positions_preview": full_variant.get("spawn_positions_preview", []),
                "agent_count_distributed": full_variant.get("agent_count_distributed", 0),
                "raw_spawn_geojson": mapping(raw_spawn) if not raw_spawn.is_empty else None,
                "clipped_spawn_geojson": mapping(clipped_spawn) if not clipped_spawn.is_empty else None,
                "safe_spawn_geojson": mapping(safe_spawn) if not safe_spawn.is_empty else None,
            }
        )
        write_json(output_dirs["metadata"] / f"route_{route_idx:02d}.json", route_meta)
        summary["routes"].append(route_meta)

    write_json(output_dirs["metadata"] / "simulation_summary.json", summary)
    return any(
        variant.get("status") in {"success", "preview_only"}
        for route in summary["routes"]
        for variant in route.get("variants", [{"status": route.get("status")}])
    )


def plans_to_run(cfg, cli_plan, batch):
    scenario_name = cfg.get("scenario_name", "Topo_HouseGAN")
    geo_root = PROJECT_ROOT / "Geo_scenario" / scenario_name / "geo"
    if batch:
        if not geo_root.exists():
            return []
        topo_root = PROJECT_ROOT / "Geo_scenario" / scenario_name
        swarm_root = topo_root / "dataswarm"
        plans = []
        for plan_dir in sorted(geo_root.iterdir()):
            if not plan_dir.is_dir():
                continue
            plan_swarm = swarm_root / plan_dir.name
            if not plan_swarm.exists() or not any(plan_swarm.glob("*.sqlite")):
                plans.append(plan_dir.name)
        return plans
    plan = cli_plan or cfg.get("plan", "")
    return [plan] if plan else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config_density_sim.json")))
    parser.add_argument("--plan", type=str, default="")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sqm-per-person", type=float, default=None)
    parser.add_argument("--wall-offset", type=float, default=None)
    parser.add_argument("--max-agents", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.sqm_per_person is not None:
        cfg["spawn_policy"]["sqm_per_person"] = args.sqm_per_person
    if args.wall_offset is not None:
        cfg["spawn_policy"]["wall_offset_m"] = args.wall_offset
    if args.max_agents is not None:
        cfg["spawn_policy"]["max_agents"] = args.max_agents
    if args.timeout is not None:
        cfg["timeout_minutes"] = args.timeout
    if args.preview:
        cfg["preview_only"] = True

    selected_plans = plans_to_run(cfg, args.plan, args.batch or bool(cfg.get("batch", False)))
    if not selected_plans:
        print("[Sim] No plan selected. Use --plan <plan_name> or --batch.")
        return

    ok_count = 0
    for plan_name in selected_plans:
        ok_count += int(run_plan(plan_name, cfg))

    print(f"[Sim] Completed {ok_count}/{len(selected_plans)} plan(s)")


if __name__ == "__main__":
    main()

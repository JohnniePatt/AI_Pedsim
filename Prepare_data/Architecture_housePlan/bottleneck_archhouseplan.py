import os
import time
import shutil
import pathlib
import json
import sqlite3
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from numpy.random import normal
from shapely import GeometryCollection, Polygon
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
import psutil

def print_memory_usage(step_name, seed):
    """ฟังก์ชันเสริมสำหรับพิมพ์สถานะ RAM ปัจจุบันของระบบและของ Process นี้"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    
    process_mb = mem_info.rss / 1024 ** 2
    total_mb = sys_mem.total / 1024 ** 2
    used_mb = sys_mem.used / 1024 ** 2
    percent = sys_mem.percent
    
    print(f"[{seed}] [RAM '{step_name}'] Process: {process_mb:.1f} MB | System Used: {used_mb:.1f}/{total_mb:.1f} MB ({percent}%)")

import jupedsim as jps
import pedpy

try:
    from jupedsim.internal.notebook_utils import read_sqlite_file
except ImportError:
    read_sqlite_file = None

def load_polygons_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [Polygon(poly_coords) for poly_coords in data]

def run_pedsim_simulation(current_seed, num_agents, area, trajectory_file, spawning_area, exit_area, timeout_minutes=3):
    print_memory_usage("Start Simulation Setup", current_seed)
    print(f"[{current_seed}] Starting Setup...")
    
    # 1. Clipping areas
    print(f"[{current_seed}] Clipping Geometries...")
    clip_spawning = spawning_area.intersection(area)
    if hasattr(clip_spawning, "geoms"): clip_spawning = max(clip_spawning.geoms, key=lambda p: p.area)
    
    clip_exit = exit_area.intersection(area)
    if hasattr(clip_exit, "geoms"): clip_exit = max(clip_exit.geoms, key=lambda p: p.area)

    # เช็คว่ามีพื้นที่เหลือให้วางคนหรือไม่
    if clip_spawning.is_empty or clip_spawning.area < 0.1:
        print(f"[{current_seed}] Error: Spawning area too small or empty after clipping.")
        return False, clip_spawning, clip_exit, None

    try:
        # 2. Distributing agents
        print(f"[{current_seed}] Distributing {num_agents} agents in area {clip_spawning.area:.2f} m2...")
        pos_in_spawning_area = jps.distributions.distribute_by_number(
            polygon=clip_spawning,
            number_of_agents=num_agents,
            distance_to_agents=0.3, # ลดจาก 0.4
            distance_to_polygon=0.15, # ลดจาก 0.5 เพื่อให้วางในห้องเล็กได้
            seed=current_seed,
        )
    except Exception as e:
        print(f"Warning: Failed to distribute {num_agents} agents inside spawning area for seed {current_seed}. Error: {e}")
        return False, clip_spawning, clip_exit, None

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=area,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=trajectory_file
        ),
    )
    
    # เพิ่ม Exit โดยใช้ Convex Hull เพื่อให้ JuPedSim ยอมรับ (ต้องเป็นพื้นที่นูน)
    # และหดระยะเข้ามานิดหน่อยเพื่อไม่ให้ทับเส้นขอบ geometry (Buffer)
    exit_poly = clip_exit.buffer(-0.1).convex_hull
    exit_id = simulation.add_exit_stage(exit_poly.exterior.coords[:-1])
    journey = jps.JourneyDescription([exit_id])
    journey_id = simulation.add_journey(journey)
    
    rng = np.random.default_rng(current_seed)
    v_distribution = rng.normal(1.34, 0.05, num_agents)
    
    for pos, v0 in zip(pos_in_spawning_area, v_distribution):
        simulation.add_agent(
            jps.CollisionFreeSpeedModelAgentParameters(
                journey_id=journey_id,
                stage_id=exit_id,
                position=pos,
                desired_speed=v0,
                radius=0.15,
            )
        )
    
    print_memory_usage("Before Running Iterate", current_seed)
    print(f"[{current_seed}] Running Simulation (Timeout: {timeout_minutes} min)...", flush=True)
    
    start_time = time.time()
    last_change_time = time.time() # ติดตามเวลาที่มีคนออกล่าสุด
    max_duration_seconds = timeout_minutes * 60
    no_progress_timeout = 60 # ถ้าผ่านไป 1 นาทีไม่มีคนออกเพิ่มเลย ถือว่าค้าง
    
    with tqdm(total=num_agents, desc=f"Simulating (Seed {current_seed})") as pbar:
        last_count = num_agents
        while simulation.agent_count() > 0:
            simulation.iterate()
            
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 1. Total Timeout Check
            if elapsed_time > max_duration_seconds:
                print(f"\n[!] Total Timeout: Simulation taking too long (> {timeout_minutes} min). Aborting seed {current_seed}.", flush=True)
                simulation._writer.close()
                if pathlib.Path(trajectory_file).exists():
                    os.remove(trajectory_file) 
                return False, clip_spawning, clip_exit, None
            
            current_count = simulation.agent_count()
            
            # 2. Progress Check (Deadlock detection)
            if current_count < last_count:
                pbar.update(last_count - current_count)
                last_count = current_count
                last_change_time = current_time # อัปเดตเวลาล่าสุดที่คนออกได้จริง
            else:
                # ถ้าจำนวนคนเท่าเดิม เช็คว่าค้างนานเกินไปหรือยัง
                if current_time - last_change_time > no_progress_timeout:
                    print(f"\n[!] Deadlock Detected: No progress for {no_progress_timeout}s. Aborting seed {current_seed}.", flush=True)
                    simulation._writer.close()
                    if pathlib.Path(trajectory_file).exists():
                        os.remove(trajectory_file)
                    return False, clip_spawning, clip_exit, None

    simulation._writer.close()
    
    print_memory_usage("After Iterate Finish", current_seed)
    return True, clip_spawning, clip_exit, pos_in_spawning_area

def plot_simulation_configuration(walkable_area, spawning_area, pos_in_spawning_area, exit_area, current_seed, spawn_exit_dir, dpi):
    print(f"[{current_seed}] Generating Spawn/Exit Plot...")
    fig, axes = plt.subplots(figsize=(8, 8))
    pedpy.plot_walkable_area(walkable_area=walkable_area, axes=axes)
    axes.fill(*spawning_area.exterior.xy, color="lightgrey", alpha=0.5, label="Spawning Area")
    axes.fill(*exit_area.exterior.xy, color="indianred", alpha=0.5, label="Exit Area")
    
    if pos_in_spawning_area:
        axes.scatter(*zip(*pos_in_spawning_area), s=5, color="blue", label="Agents")
        
    axes.set_xlabel("x/m")
    axes.set_ylabel("y/m")
    axes.set_aspect("equal")
    axes.set_title(f"Simulation Configuration (Seed: {current_seed})")
    
    img_path = spawn_exit_dir / f"spawn_exit{current_seed}.png"
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {img_path}")

def generate_trajectory_plot(trajectory_file, current_seed, trajectory_line_dir, dpi):
    print_memory_usage("Start Trajectory Plot", current_seed)
    print(f"[{current_seed}] Generating Trajectory Plot...")
    
    if read_sqlite_file is not None:
        try:
            trajectory_data, loaded_walkable_area = read_sqlite_file(str(trajectory_file))
        except Exception as e:
            trajectory_data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=trajectory_file)
            loaded_walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_file)
    else:
        trajectory_data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=trajectory_file)
        loaded_walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_file)
        
    fig, ax = plt.subplots(figsize=(8, 8))
    pedpy.plot_measurement_setup(
        walkable_area=loaded_walkable_area,
        traj=trajectory_data,
        traj_alpha=0.5,
        traj_width=1,
        ml_color="b",
        ma_line_width=1,
        ma_alpha=0.2,
        axes=ax,
    ).set_aspect("equal")
    plt.title(f"Trajectory Line (Seed: {current_seed})")
    
    traj_img_path = trajectory_line_dir / f"trajectory_{current_seed}.png"
    plt.savefig(traj_img_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {traj_img_path}")
    
    return trajectory_data, loaded_walkable_area

def generate_heatmaps(trajectory_data, loaded_walkable_area, current_seed, density_dir, speed_dir, dpi, grid_size=0.5):
    print_memory_usage("Start Heatmaps calc", current_seed)
    print(f"[{current_seed}] Calculating Heatmaps (Grid: {grid_size}m)...")
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
    sum_speed = None
    count = 0
    frame_n = 60
    # grid_size is now a parameter
    
    frames_to_process = individual_speed['frame'].unique()[::frame_n]

    for f in tqdm(frames_to_process, desc=f"Heatmap (Seed {current_seed})"):
        speed_f = individual_speed[individual_speed.frame == f]
        cells_f = individual_voronoi_cells[individual_voronoi_cells.frame == f]
        frame_data = pd.merge(cells_f, speed_f, on=["id", "frame"])

        d_profile, s_profile = pedpy.compute_profiles(
            individual_voronoi_speed_data=frame_data,
            walkable_area=loaded_walkable_area.polygon,
            grid_size=grid_size,
            speed_method=pedpy.SpeedMethod.ARITHMETIC,
        )
        if sum_density is None:
            sum_density = np.copy(d_profile[0])
            sum_speed = np.copy(s_profile[0])
        else:
            sum_density += d_profile[0]
            sum_speed += s_profile[0]

        count += 1
        
        del frame_data
        del speed_f
        del cells_f

    del individual_voronoi_cells
    del individual_speed
    gc.collect()

    print_memory_usage("After Heatmap loop", current_seed)

    if count > 0:
        mean_density_map = sum_density / count
        mean_speed_map = sum_speed / count

        fig_den, ax_den = plt.subplots(figsize=(8, 8))
        pedpy.plot_profiles(
            walkable_area=loaded_walkable_area,
            profiles=[mean_density_map],
            axes=ax_den,
            label="$\\rho$ / 1/$m^2$",
            vmin=0,
            vmax=5,
            title=f"Average Density (Seed: {current_seed})",
        )
        den_img_path = density_dir / f"heatmap_density{current_seed}.png"
        plt.savefig(den_img_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_den)
        print(f"Saved {den_img_path}")

        fig_spd, ax_spd = plt.subplots(figsize=(8, 8))
        pedpy.plot_profiles(
            walkable_area=loaded_walkable_area,
            profiles=[mean_speed_map],
            axes=ax_spd,
            label="v / m/s",
            vmin=0,
            vmax=1.5,
            title=f"Average Speed (Seed: {current_seed})",
        )
        spd_img_path = speed_dir / f"heatmap_speed{current_seed}.png"
        plt.savefig(spd_img_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_spd)
        print(f"Saved {spd_img_path}")

# ==========================================
# setup
# ==========================================
START_SEED = 42 
END_SEED = 42 
NUM_AGENTS = 50 
DPI = 300  

BASE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent.parent

def run_full_simulation_flow(plan_name, agents, seed, grid_size, timeout_minutes=5, preview_only=False):
    global PLAN_FOLDER_NAME, PLAN_DIR, START_SEED, END_SEED, NUM_AGENTS
    
    PLAN_FOLDER_NAME = plan_name
    PLAN_DIR = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo" / PLAN_FOLDER_NAME
    START_SEED = seed
    END_SEED = seed
    NUM_AGENTS = agents
    
    GEO_DIR = PLAN_DIR 
    GEO_CORRIDOR_FILE = GEO_DIR / "geo_corridor.json"
    GEO_ROOM_FILE = GEO_DIR / "geo_room.json"
    GEO_DOOR_FILE = GEO_DIR / "geo_door.json"
    
    # Define output folders
    OUTPUT_BASE = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
    DATASWARM_DIR = OUTPUT_BASE / "dataswarm" / PLAN_FOLDER_NAME
    HEATMAP_DENSITY_DIR = OUTPUT_BASE / "heatmap_density" / PLAN_FOLDER_NAME
    HEATMAP_SPEED_DIR = OUTPUT_BASE / "heatmap_speed" / PLAN_FOLDER_NAME
    TRAJECTORY_LINE_DIR = OUTPUT_BASE / "trajectory_line" / PLAN_FOLDER_NAME
    SPAWN_EXIT_DIR = OUTPUT_BASE / "spawn_exit" / PLAN_FOLDER_NAME

    # Clean output folders for this specific plan (only if not preview_only)
    if not preview_only:
        for d in [DATASWARM_DIR, HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, TRAJECTORY_LINE_DIR, SPAWN_EXIT_DIR]:
            if d.exists():
                print(f"Cleaning existing output sub-directory: {d}")
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    if not GEO_CORRIDOR_FILE.exists() or not GEO_ROOM_FILE.exists() or not GEO_DOOR_FILE.exists():
        print(f"ERROR: Geometry files not found in {PLAN_DIR}")
        return False

    print(f"\n>>> Processing Plan: {PLAN_FOLDER_NAME}")
    print(f"Loading Global Geometry...")
    corridor_polys = load_polygons_from_json(GEO_CORRIDOR_FILE)
    room_polys = load_polygons_from_json(GEO_ROOM_FILE)
    
    with open(GEO_DOOR_FILE, 'r') as f:
        doors_data = json.load(f)

    all_geoms = corridor_polys + room_polys
    footprint = unary_union([p.buffer(0.02) for p in all_geoms]).buffer(-0.02)
    
    WALL_THICKNESS = 0.2
    DOOR_WIDTH = 1.5 
    walls = unary_union([poly.exterior.buffer(WALL_THICKNESS/2, join_style=2) for poly in all_geoms])
    
    door_cutouts = []
    for d in doors_data:
        px, py = d["pos"]
        dw = DOOR_WIDTH
        dt = WALL_THICKNESS * 2.5 
        if d["horizontal"]: cutout = box(px - dw/2, py - dt/2, px + dw/2, py + dt/2)
        else: cutout = box(px - dt/2, py - dw/2, px + dt/2, py + dw/2)
        door_cutouts.append(cutout)

    if door_cutouts:
        doors_poly = unary_union(door_cutouts)
        walls = walls.difference(doors_poly)
        
    area = footprint.difference(walls)
    
    if hasattr(area, "geoms"): # MultiPolygon
        area = max(area.geoms, key=lambda p: p.area)
    
    # Identify longest routes and run sim
    import networkx as nx
    G = nx.Graph()
    node_polys = {} 
    
    for i, p in enumerate(corridor_polys):
        nid = f"Cor-{i}"; G.add_node(nid, type='Corridor'); node_polys[nid] = p
    for i, p in enumerate(room_polys):
        nid = f"Room-{i}"; G.add_node(nid, type='Room'); node_polys[nid] = p
        
    for d in doors_data:
        if 'rooms' in d and len(d['rooms']) == 2:
            u, v = d['rooms']
            if u in G.nodes and v in G.nodes: G.add_edge(u, v)

    if not nx.is_connected(G):
        print(f"ERROR: Layout {PLAN_FOLDER_NAME} is not connected. Skipping.")
        return False

    diameter = nx.diameter(G)
    periphery = nx.periphery(G)
    routes_to_sim = []
    processed_pairs = set()
    for i, start_nid in enumerate(periphery):
        for end_nid in periphery[i+1:]:
            path = nx.shortest_path(G, start_nid, end_nid)
            if len(path) - 1 == diameter:
                pair = tuple(sorted((start_nid, end_nid)))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    routes_to_sim.append((start_nid, end_nid))
    
    print(f"Found {len(routes_to_sim)} longest routes")

    for r_idx, (start_nid, end_nid) in enumerate(routes_to_sim):
        suffix = f"{r_idx:02d}"
        sim_name = f"plan_sim_{seed}_{suffix}"
        trajectory_file = DATASWARM_DIR / f"{sim_name}.sqlite"
        
        if preview_only:
            if not trajectory_file.exists(): continue
            trajectory_data, loaded_walkable_area = generate_trajectory_plot(trajectory_file, f"{seed}_{suffix}", TRAJECTORY_LINE_DIR, DPI)
            generate_heatmaps(trajectory_data, loaded_walkable_area, f"{seed}_{suffix}", HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, DPI, grid_size=grid_size)
        else:
            success, _, _, pos_in_spawning_area = run_pedsim_simulation(seed, agents, area, trajectory_file, node_polys[start_nid], node_polys[end_nid], timeout_minutes=timeout_minutes)
            if success:
                plot_simulation_configuration(pedpy.WalkableArea(area), node_polys[start_nid], pos_in_spawning_area, node_polys[end_nid], f"{seed}_{suffix}", SPAWN_EXIT_DIR, DPI)
                trajectory_data, loaded_walkable_area = generate_trajectory_plot(trajectory_file, f"{seed}_{suffix}", TRAJECTORY_LINE_DIR, DPI)
                generate_heatmaps(trajectory_data, loaded_walkable_area, f"{seed}_{suffix}", HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, DPI, grid_size=grid_size)
        
        plt.close('all'); gc.collect() 
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, default=None, help="Name of the plan folder")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--grid", type=float, default=0.5, help="Heatmap grid size")
    parser.add_argument("--preview", action="store_true", help="Only regenerate previews")
    parser.add_argument("--batch", action="store_true", help="Run for all unsimulated HouseGAN plans")
    parser.add_argument("--timeout", type=int, default=5, help="Simulation timeout in minutes")
    args = parser.parse_args()

    if args.batch:
        topo_root = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN"
        geo_root = topo_root / "geo"
        swarm_root = topo_root / "dataswarm"
        
        all_plans = sorted([d.name for d in geo_root.iterdir() if d.is_dir() and d.name.startswith("plan_")])
        unsimulated = []
        for p in all_plans:
            p_swarm = swarm_root / p
            if not p_swarm.exists() or not any(p_swarm.glob("*.sqlite")):
                unsimulated.append(p)
        
        print(f"Found {len(unsimulated)} unsimulated HouseGAN plans out of {len(all_plans)} total.")
        for i, p_name in enumerate(unsimulated):
            print(f"\n[Batch {i+1}/{len(unsimulated)}] Starting Planet: {p_name}")
            run_full_simulation_flow(p_name, args.agents, args.seed, args.grid, timeout_minutes=args.timeout)
    else:
        p_name = args.plan if args.plan else PLAN_FOLDER_NAME
        run_full_simulation_flow(p_name, args.agents, args.seed, args.grid, timeout_minutes=args.timeout, preview_only=args.preview)

    print("\n=== All Tasks Completed ===")

import os
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

def run_pedsim_simulation(current_seed, num_agents, area, trajectory_file, spawning_area, exit_area):
    print_memory_usage("Start Simulation Setup", current_seed)
    print(f"[{current_seed}] Starting Setup...")
    
    print(f"[{current_seed}] Spawning Area Centroid: {spawning_area.centroid}")
    print(f"[{current_seed}] Exit Area Centroid: {exit_area.centroid}")
    
    try:
        # กระจายคนในพื้นที่ Spawn โดยเว้นระยะห่างจากขอบพอสมควรเพื่อไม่ให้ติด Constraint
        pos_in_spawning_area = jps.distributions.distribute_by_number(
            polygon=spawning_area,
            number_of_agents=num_agents,
            distance_to_agents=0.4,
            distance_to_polygon=0.4, 
            seed=current_seed,
        )
    except Exception as e:
        print(f"Warning: Failed to distribute {num_agents} agents inside spawning area for seed {current_seed}. Error: {e}")
        return False, spawning_area, exit_area, None

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=area,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=trajectory_file
        ),
    )
    
    # เพิ่ม Exit โดยหดระยะเข้ามานิดหน่อย (Buffer) เพื่อไม่ให้ทับเส้นขอบ geometry
    exit_id = simulation.add_exit_stage(exit_area.buffer(-0.1).exterior.coords[:-1])
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
    print(f"[{current_seed}] Running Simulation...")
    while simulation.agent_count() > 0:
        simulation.iterate()
    simulation._writer.close()
    
    print_memory_usage("After Iterate Finish", current_seed)
    return True, spawning_area, exit_area, pos_in_spawning_area

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

# กำหนดโฟลเดอร์ของ Plan
PLAN_FOLDER_NAME = f"plan_42_default"
PLAN_DIR = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo" / PLAN_FOLDER_NAME

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, default=None, help="Name of the plan folder")
    parser.add_argument("--agents", type=int, default=50, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--grid", type=float, default=0.5, help="Heatmap grid size")
    parser.add_argument("--preview", action="store_true", help="Only regenerate previews from existing sqlite files")
    args = parser.parse_args()

    if args.plan:
        PLAN_FOLDER_NAME = args.plan
        START_SEED = args.seed
        END_SEED = args.seed
        NUM_AGENTS = args.agents
        PLAN_DIR = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo" / PLAN_FOLDER_NAME
    else:
        PLAN_DIR = PROJECT_ROOT / "Geo_scenario" / "Topo_HouseGAN" / "geo" / PLAN_FOLDER_NAME

    GEO_CORRIDOR_FILE = PLAN_DIR / "geo_corridor.json"
    GEO_ROOM_FILE = PLAN_DIR / "geo_room.json"
    GEO_DOOR_FILE = PLAN_DIR / "geo_door.json"
    
    # Define output folders
    OUTPUT_BASE = BASE_DIR / "outputs" / PLAN_FOLDER_NAME
    DATASWARM_DIR = OUTPUT_BASE / "dataswarm"
    HEATMAP_DENSITY_DIR = OUTPUT_BASE / "heatmap_density"
    HEATMAP_SPEED_DIR = OUTPUT_BASE / "heatmap_speed"
    TRAJECTORY_LINE_DIR = OUTPUT_BASE / "trajectory_line"
    SPAWN_EXIT_DIR = OUTPUT_BASE / "spawn_exit"

    for d in [OUTPUT_BASE, DATASWARM_DIR, HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, TRAJECTORY_LINE_DIR, SPAWN_EXIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not GEO_CORRIDOR_FILE.exists() or not GEO_ROOM_FILE.exists() or not GEO_DOOR_FILE.exists():
        print(f"ERROR: Geometry files not found in {PLAN_DIR}")
        exit(1)

    print(f"Loading Global Geometry for {PLAN_FOLDER_NAME}...")
    corridor_polys = load_polygons_from_json(GEO_CORRIDOR_FILE)
    room_polys = load_polygons_from_json(GEO_ROOM_FILE)
    
    with open(GEO_DOOR_FILE, 'r') as f:
        doors_data = json.load(f)

    all_geoms = corridor_polys + room_polys
    footprint = unary_union(all_geoms)
    
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

    # ==========================================
    # loop and process
    # ==========================================
    for current_seed in range(START_SEED, END_SEED + 1):
        print(f"\n{'='*40}")
        print(f"Processing Seed: {current_seed}")
        print(f"{'='*40}")

        # --- Identify Longest Paths for Simulation ---
        import networkx as nx
        G = nx.Graph()
        node_polys = {} 
        
        for i, p in enumerate(corridor_polys):
            nid = f"Cor-{i}"
            G.add_node(nid, type='Corridor')
            node_polys[nid] = p
        for i, p in enumerate(room_polys):
            nid = f"Room-{i}"
            G.add_node(nid, type='Room')
            node_polys[nid] = p
            
        for d in doors_data:
            if 'rooms' in d and len(d['rooms']) == 2:
                u, v = d['rooms']
                if u in G.nodes and v in G.nodes:
                    G.add_edge(u, v)

        if not nx.is_connected(G):
            print(f"[{current_seed}] ERROR: Layout is not connected. Skipping.")
            continue

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
        
        print(f"[{current_seed}] Found {len(routes_to_sim)} longest routes (Diameter: {diameter})")

        for r_idx, (start_nid, end_nid) in enumerate(routes_to_sim):
            suffix = f"{r_idx:02d}"
            sim_name = f"plan_sim_{current_seed}_{suffix}"
            trajectory_file = DATASWARM_DIR / f"{sim_name}.sqlite"
            
            print(f"\n--- Running Route {suffix}: {start_nid} -> {end_nid} ---")
            
            if args.preview:
                if not trajectory_file.exists():
                    print(f"[{current_seed}] Skip Preview: {trajectory_file} not found.")
                    continue
                print(f"[{current_seed}] Regenerating Previews for {trajectory_file}...")
                trajectory_data, loaded_walkable_area = generate_trajectory_plot(trajectory_file, f"{current_seed}_{suffix}", TRAJECTORY_LINE_DIR, DPI)
                generate_heatmaps(trajectory_data, loaded_walkable_area, f"{current_seed}_{suffix}", HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, DPI, grid_size=args.grid)
            else:
                # 1. Run Simulation
                success, _, _, pos_in_spawning_area = run_pedsim_simulation(
                    current_seed, NUM_AGENTS, area, trajectory_file, node_polys[start_nid], node_polys[end_nid]
                )
                
                if not success: continue
                    
                plot_simulation_configuration(pedpy.WalkableArea(area), node_polys[start_nid], pos_in_spawning_area, node_polys[end_nid], f"{current_seed}_{suffix}", SPAWN_EXIT_DIR, DPI)
                trajectory_data, loaded_walkable_area = generate_trajectory_plot(trajectory_file, f"{current_seed}_{suffix}", TRAJECTORY_LINE_DIR, DPI)
                generate_heatmaps(trajectory_data, loaded_walkable_area, f"{current_seed}_{suffix}", HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, DPI, grid_size=args.grid)
            
            plt.close('all'); gc.collect() 
            
        print_memory_usage("End of Seed Cleanup", current_seed)

    print("\n=== All Tasks Completed ===")

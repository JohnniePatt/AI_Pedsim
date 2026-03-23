import os
import time
import pathlib
import json
import sqlite3
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely import GeometryCollection, Polygon
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
from tqdm.auto import tqdm
import psutil

def print_memory_usage(step_name, seed):
    """ฟังก์ชันเสริมสำหรับพิมพ์สถานะ RAM ปัจจุบันของระบบและของ Process นี้"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    
    # แปลง Byte เป็น Megabyte (MB)
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

# ==========================================
# function
# ==========================================
def load_polygons_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [Polygon(poly_coords) for poly_coords in data]

def run_pedsim_simulation(current_seed, num_agents, all_rooms, area, trajectory_file, timeout_minutes):
    """รัน simulation ของ Jupedsim แบบมีจำกัดเวลาทำงานจริง"""
    print_memory_usage("Start Simulation Setup", current_seed)
    print(f"[{current_seed}] Starting Setup...")
    rand_gen = random.Random(current_seed)
    
    if len(all_rooms) < 2:
        raise ValueError("Not enough rooms generated for spawn and exit. Need at least 2.")

    selected_rooms = rand_gen.sample(all_rooms, k=2)
    spawning_area = selected_rooms[0]
    exit_area = selected_rooms[1]
    
    print(f"[{current_seed}] Spawning Area: {spawning_area.centroid}")
    print(f"[{current_seed}] Exit Area: {exit_area.centroid}")
    
    try:
        pos_in_spawning_area = jps.distributions.distribute_by_number(
            polygon=spawning_area,
            number_of_agents=num_agents,
            distance_to_agents=0.3,
            distance_to_polygon=0.15,
            seed=current_seed,
        )
    except Exception as e:
        print(f"Warning: Failed to distribute {num_agents} agents inside spawning area for seed {current_seed}. Error: {e}")
        return False, None, None, None

    simulation = jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=area,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=trajectory_file
        ),
    )
    
    exit_id = simulation.add_exit_stage(exit_area.exterior.coords[:-1])
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
    
    start_time = time.time()
    max_duration_seconds = timeout_minutes * 60
    
    # ใช้หลอด tqdm ติดตามการอพยพของคน (จำนวนคนที่ออกจากพื้นที่แล้ว)
    with tqdm(total=num_agents, desc=f"Simulating (Seed {current_seed})") as pbar:
        last_count = num_agents
        while simulation.agent_count() > 0:
            simulation.iterate()
            
            # เช็คเวลาทำงานทุกรอบ
            elapsed_time = time.time() - start_time
            if elapsed_time > max_duration_seconds:
                print(f"\n[!] ว้าวุ่นแล้ว: ทะเยอทะยานเกินไป หรือ Deadlock ทำยอดไม่สำเร็จ (ใช้เวลาเกิน {timeout_minutes} นาที)")
                print(f"[!] ขอยกเลิกและลบข้อมูล Seed {current_seed} ทิ้งทันที!")
                simulation._writer.close()
                if pathlib.Path(trajectory_file).exists():
                    os.remove(trajectory_file) # ลบไฟล์ฐานข้อมูลที่พังทิ้ง
                return False, None, None, None
            
            current_count = simulation.agent_count()
            # อัปเดตหลอดตามจำนวนคนที่หายไป (อพยพออกแล้ว)
            if current_count < last_count:
                pbar.update(last_count - current_count)
                last_count = current_count
                
    simulation._writer.close()
    
    print_memory_usage("After Iterate Finish", current_seed)
    return True, spawning_area, exit_area, pos_in_spawning_area

def plot_simulation_configuration(
    walkable_area, spawning_area, pos_in_spawning_area, exit_area, 
    current_seed, spawn_exit_dir, dpi
):
    """วาดพื้นที่เกิด พื้นที่ออก และตำแหน่งเริ่มต้นของคน"""
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
    """อ่านข้อมูล trajectory และสร้างรูป Trajectory Line"""
    print_memory_usage("Start Trajectory Plot", current_seed)
    print(f"[{current_seed}] Generating Trajectory Plot...")
    
    if read_sqlite_file is not None:
        try:
            trajectory_data, loaded_walkable_area = read_sqlite_file(str(trajectory_file))
        except Exception as e:
            trajectory_data = pedpy.load_trajectory_from_jupedsim_sqlite(
                trajectory_file=trajectory_file
            )
            loaded_walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(
                trajectory_file=trajectory_file
            )
    else:
        trajectory_data = pedpy.load_trajectory_from_jupedsim_sqlite(
            trajectory_file=trajectory_file
        )
        loaded_walkable_area = pedpy.load_walkable_area_from_jupedsim_sqlite(
            trajectory_file=trajectory_file
        )
        
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

# ==========================================
# setup
# ==========================================
START_SEED = 100508
END_SEED = 100600 # Adjust for multiple seeds (e.g., 5 or 100)
NUM_AGENTS = 100
DPI = 300  # Default DPI for all saved figures
TIMEOUT_MINUTES = 5 # ให้เวลายอมแพ้ 2 นาที (ปรับได้ครับ)

# Base directory setup (relative to this script's location)
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Define input/output folders
GEO_DIR = BASE_DIR / "geo"
DATASWARM_DIR = BASE_DIR / "dataswarm"
TRAJECTORY_LINE_DIR = BASE_DIR / "trajectory_line"
SPAWN_EXIT_DIR = BASE_DIR / "spawn_exit"

# Create directories if they do not exist
for directory in [GEO_DIR, DATASWARM_DIR, TRAJECTORY_LINE_DIR, SPAWN_EXIT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Define geometry paths
GEO_CORRIDOR_FILE = GEO_DIR / "geo_corridor.json"
GEO_ROOM_FILE = GEO_DIR / "geo_room.json"

if __name__ == "__main__":
    if not GEO_CORRIDOR_FILE.exists() or not GEO_ROOM_FILE.exists():
        print(f"ERROR: Geometry files not found in {GEO_DIR}")
        print("Please make sure 'geo_corridor.json' and 'geo_room.json' exist in the 'geo' folder.")
        exit(1)

    print("Loading Global Geometry...")
    corridor_polys = load_polygons_from_json(GEO_CORRIDOR_FILE)
    room_polys = load_polygons_from_json(GEO_ROOM_FILE)

    all_geoms = corridor_polys + room_polys
    all_areas = unary_union(all_geoms)
    area = all_areas
    all_rooms = room_polys

    # ==========================================
    # loop and process (Simulation Only)
    # ==========================================
    for current_seed in range(START_SEED, END_SEED + 1):
        print(f"\n{'='*40}")
        print(f"Processing Seed (Sim): {current_seed}")
        print(f"{'='*40}")
        
        trajectory_file = DATASWARM_DIR / f"double-botteleneck_{current_seed}.sqlite"
        
        # 1. Run Simulation
        success, spawning_area, exit_area, pos_in_spawning_area = run_pedsim_simulation(
            current_seed, NUM_AGENTS, all_rooms, area, trajectory_file, TIMEOUT_MINUTES
        )
        if not success:
            print(f"[{current_seed}] Skipped or Aborted. (No valid trajectory saved)")
            continue
            
        # 1.1 Plot Simulation Configuration
        plot_simulation_configuration(
            walkable_area=pedpy.WalkableArea(area), 
            spawning_area=spawning_area, 
            pos_in_spawning_area=pos_in_spawning_area, 
            exit_area=exit_area, 
            current_seed=current_seed, 
            spawn_exit_dir=SPAWN_EXIT_DIR, 
            dpi=DPI
        )

        # 2. Plot Trajectory Line
        trajectory_data, loaded_walkable_area = generate_trajectory_plot(
            trajectory_file, current_seed, TRAJECTORY_LINE_DIR, DPI
        )
        
        # --- Aggressive Memory Cleanup ---
        del trajectory_data
        del loaded_walkable_area
        del spawning_area
        del exit_area
        del pos_in_spawning_area
        
        plt.close('all') 
        gc.collect() 
        print_memory_usage("End of Loop Cleanup (Sim)", current_seed)
        # --------------------------------

    print("\n=== Simulation Script Completed ===")

import os
import pathlib
import json
import sqlite3
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def generate_heatmaps(trajectory_file, current_seed, density_dir, speed_dir, dpi):
    """อ่านข้อมูล trajectory จาก SQLite, คำนวณและสร้างรูป Heatmap ของ Density และ Speed"""
    
    print_memory_usage("Loading Data from SQLite", current_seed)
    
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

    print_memory_usage("Start Heatmaps calc", current_seed)
    print(f"[{current_seed}] Calculating Heatmaps...")
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
    grid_size = 0.5
    
    # หาเฟรมทั้งหมดที่จะทำ (เพื่อทำ tqdm และ progress)
    frames_to_process = individual_speed['frame'].unique()[::frame_n]

    for f in tqdm(frames_to_process, desc=f"Heatmap (Seed {current_seed})"):
        # กรองข้อมูลเฉพาะเฟรม f จากตารางดิบ
        speed_f = individual_speed[individual_speed.frame == f]
        cells_f = individual_voronoi_cells[individual_voronoi_cells.frame == f]
        
        # Merge เฉพาะของทีละเฟรม ช่วยลดขนาดตารางที่ต้องอยู่ใน RAM มหาศาล
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
        
        # เคลียร์ตัวแปรของเฟรมทิ้งให้ RAM กลับคืนมา
        del frame_data
        del speed_f
        del cells_f

    # ล้างตารางหลัก
    del individual_voronoi_cells
    del individual_speed
    gc.collect()

    print_memory_usage("After Heatmap loop", current_seed)

    if count > 0:
        mean_density_map = sum_density / count
        mean_speed_map = sum_speed / count

        # Plot Density
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

        # Plot Speed
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
        
    return trajectory_data, loaded_walkable_area

# ==========================================
# setup
# ==========================================
START_SEED = 100095
END_SEED = 100100  # Adjust for multiple seeds (e.g., 5 or 100)
DPI = 300  # Default DPI for all saved figures

# Base directory setup (relative to this script's location)
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Define input/output folders
DATASWARM_DIR = BASE_DIR / "dataswarm"
HEATMAP_DENSITY_DIR = BASE_DIR / "heatmap_density"
HEATMAP_SPEED_DIR = BASE_DIR / "heatmap_speed"

# Create directories if they do not exist
for directory in [HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # ==========================================
    # loop and process (Heatmap Only)
    # ==========================================
    for current_seed in range(START_SEED, END_SEED + 1):
        print(f"\n{'='*40}")
        print(f"Processing Seed (Heatmap): {current_seed}")
        print(f"{'='*40}")
        
        trajectory_file = DATASWARM_DIR / f"double-botteleneck_{current_seed}.sqlite"
        
        if not trajectory_file.exists():
            print(f"Warning: File {trajectory_file} not found. Did you run the Simulation script first?")
            continue
            
        # 1. Run Heatmap Analysis
        trajectory_data, loaded_walkable_area = generate_heatmaps(
            trajectory_file, current_seed, 
            HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR, DPI
        )
        
        # --- Aggressive Memory Cleanup ---
        # คืนค่า RAM แบบบังคับลบตัวแปรใหญ่ๆ หลังจากจบรอบ
        del trajectory_data
        del loaded_walkable_area
        
        plt.close('all') # ปิดกราฟทั้งหมดของ matplotlib ที่อาจค้างในหน่วยความจำ
        gc.collect() # ขอให้ Python คืน Memory ให้ OS
        print_memory_usage("End of Loop Cleanup (Heatmap)", current_seed)
        # --------------------------------

    print("\n=== Heatmap Script Completed ===")

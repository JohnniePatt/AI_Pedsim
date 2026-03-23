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
import time

# --- Pytorch for GPU ---
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def print_system_resources():
    print("=" * 60)
    print("System Resources & Configuration")
    print("=" * 60)
    sys_mem = psutil.virtual_memory()
    total_mb = sys_mem.total / 1024 ** 2
    print(f"Total System RAM: {total_mb:.1f} MB")
    
    if HAS_TORCH and torch.cuda.is_available():
        print(f"GPU Available: YES")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"GPU Available: NO (Will fallback to CPU or fail if forced to use GPU)")
    print("=" * 60 + "\n")

def print_memory_usage(step_name, seed):
    """ฟังก์ชันเสริมสำหรับพิมพ์สถานะ RAM ปัจจุบันของระบบและของ Process นี้"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    
    # แปลง Byte เป็น Megabyte (MB)
    process_mb = mem_info.rss / 1024 ** 2
    percent = sys_mem.percent
    
    print(f"[{seed}] [{step_name}] Process RAM: {process_mb:.1f} MB | Sys RAM: {percent}%")

import jupedsim as jps
import pedpy
import shapely

try:
    from jupedsim.internal.notebook_utils import read_sqlite_file
except ImportError:
    read_sqlite_file = None

def compute_gpu_profiles(ped_x, ped_y, ped_speeds, grid_coords_gpu, mask_gpu, grid_size, ny, nx, device):
    """
    Simulates pedpy.compute_profiles using Pytorch on GPU.
    """
    if len(ped_x) == 0:
        return np.zeros((ny, nx)), np.zeros((ny, nx))
        
    # [N, 2]
    ped_coords = torch.tensor(np.column_stack((ped_x, ped_y)), dtype=torch.float32, device=device)
    ped_speeds_t = torch.tensor(ped_speeds, dtype=torch.float32, device=device)
    
    # Compute distances from all grid points to all pedestrians
    # dist shape: [num_grid_points, N]
    dist = torch.cdist(grid_coords_gpu, ped_coords)
    
    # closest pedestrian index for each grid point
    # nearest_ped_idx shape: [num_grid_points]
    nearest_ped_idx = dist.argmin(dim=1)
    
    # Compute Voronoi Area for each pedestrian
    # Count how many grid cells belong to each pedestrian, **ONLY INSIDE WALKABLE AREA**
    valid_nearest_ped_idx = nearest_ped_idx[mask_gpu]
    bincounts = torch.bincount(valid_nearest_ped_idx, minlength=len(ped_coords))
    ped_areas = bincounts * (grid_size ** 2)
    
    # Density = 1 / Area
    # Add a small epsilon to prevent division by zero for pedestrians with 0 area
    ped_densities = 1.0 / torch.clamp(ped_areas, min=1e-5)
    
    # Map back to grid
    grid_density = ped_densities[nearest_ped_idx]
    grid_speed = ped_speeds_t[nearest_ped_idx]
    
    # Apply walkable area mask (set outside to NaN or 0)
    grid_density[~mask_gpu] = float('nan')
    grid_speed[~mask_gpu] = float('nan')
    
    # Reshape back to 2D image (ny, nx)
    grid_density_2d = grid_density.reshape(ny, nx).cpu().numpy()
    grid_speed_2d = grid_speed.reshape(ny, nx).cpu().numpy()
    
    return grid_density_2d, grid_speed_2d

def setup_gpu_grid(walkable_area_polygon, grid_size, device):
    """
    Precomputes the grid coordinates and the boolean mask of the walkable area on the GPU.
    Because pedpy relies on a specific grid alignment, we mimic its bounds.
    """
    min_x, min_y, max_x, max_y = walkable_area_polygon.bounds
    nx = int(np.ceil((max_x - min_x) / grid_size))
    ny = int(np.ceil((max_y - min_y) / grid_size))
    
    # Generate grid cell centers
    x_coords = np.arange(nx) * grid_size + min_x + (grid_size / 2.0)
    y_coords = np.arange(ny) * grid_size + min_y + (grid_size / 2.0)
    
    xx, yy = np.meshgrid(x_coords, y_coords) # yy shape: (ny, nx)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Check which grid points are inside the polygon using shapely (only done ONCE)
    try:
        mask = shapely.contains(walkable_area_polygon, shapely.points(grid_points))
    except AttributeError:
        # Shapely 1.x fallback
        mask = np.array([walkable_area_polygon.contains(shapely.geometry.Point(p)) for p in grid_points])
        
    # Send to GPU
    grid_coords_gpu = torch.tensor(grid_points, dtype=torch.float32, device=device)
    mask_gpu = torch.tensor(mask, dtype=torch.bool, device=device)
    return grid_coords_gpu, mask_gpu, nx, ny


# ==========================================
# function
# ==========================================
def generate_heatmaps(trajectory_file, current_seed, density_dir, speed_dir, dpi):
    """อ่านข้อมูล trajectory จาก SQLite, คำนวณและสร้างรูป Heatmap ของ Density และ Speed ด้วย GPU"""
    
    file_id = current_seed # used to hold the file stem
    t0 = time.time()
    
    # เลือก Device
    device = torch.device('cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("[WARNING] Pytorch is using CPU! This will not be faster than original.")

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

    individual_speed = pedpy.compute_individual_speed(
        traj_data=trajectory_data,
        frame_step=5,
        speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
    )

    # ---------------------------------------------
    # GPU Setup Section
    # ---------------------------------------------
    grid_size = 0.5
    grid_coords_gpu, mask_gpu, nx, ny = setup_gpu_grid(loaded_walkable_area.polygon, grid_size, device)

    sum_density = np.zeros((ny, nx))
    sum_speed = np.zeros((ny, nx))
    count = 0
    frame_n = 60
    
    frames_to_process = individual_speed['frame'].unique()[::frame_n]
    
    # ---------------------------------------------
    # Main GPU Processing Loop
    # ---------------------------------------------
    
    # Progress bar แบบมี Memory Update ให้ดูด้วย
    pbar = tqdm(frames_to_process, desc=f"GPU Heatmap ({file_id})")
    
    for f in pbar:
        # อัปเดต % RAM ในหลอด tqdm เพื่อการ Debug
        mem_percent = psutil.virtual_memory().percent
        pbar.set_postfix({'RAM%': mem_percent})
        
        speed_f = individual_speed[individual_speed.frame == f]
        
        # ป้องกันกรณี index ไม่ตรง ให้ merge เพื่อให้มั่นใจ
        # แต่เพื่อความเป๊ะ pedpy trajectory_data เก็บพิกัด x, y ใน data
        # ลอง merge ข้อมูลก่อน เพื่อดึงพิกัดจากตารางดิบ
        raw_f = trajectory_data.data[trajectory_data.data.frame == f]
        frame_data = pd.merge(raw_f, speed_f, on=["id", "frame"], how="inner")
        
        ped_x_cleaned = frame_data['x'].values
        ped_y_cleaned = frame_data['y'].values
        ped_speeds_cleaned = frame_data['speed'].values
        
        d_prof, s_prof = compute_gpu_profiles(
            ped_x_cleaned, ped_y_cleaned, ped_speeds_cleaned, 
            grid_coords_gpu, mask_gpu, grid_size, ny, nx, device
        )
        
        sum_density += d_prof
        sum_speed += s_prof
        count += 1

    # Cleanup
    del individual_speed
    del grid_coords_gpu
    del mask_gpu
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    if count > 0:
        mean_density_map = sum_density / count
        mean_speed_map = sum_speed / count

        # ---------------------------------------------
        # Plotting
        # ---------------------------------------------
        min_x, min_y, max_x, max_y = loaded_walkable_area.polygon.bounds
        extent = [min_x, max_x, min_y, max_y]
        
        # Plot Density
        fig_den, ax_den = plt.subplots(figsize=(8, 8))
        im_den = ax_den.imshow(mean_density_map, origin='lower', extent=extent, vmin=0, vmax=5, cmap='jet', interpolation='nearest')
        # วาดเส้นกรอบ walkable area ทับเพื่อให้เห็นว่าส่วนไหนคือฉาก
        if hasattr(loaded_walkable_area.polygon, 'exterior'):
            x_ext, y_ext = loaded_walkable_area.polygon.exterior.xy
            ax_den.plot(x_ext, y_ext, color='white', linewidth=2)
        plt.colorbar(im_den, ax=ax_den, label="$\\rho$ / 1/$m^2$")
        ax_den.set_title(f"Average Density ({file_id}) - GPU")
        
        den_img_path = density_dir / f"heatmap_density_{file_id}.png"
        plt.savefig(den_img_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_den)
        print(f"Saved {den_img_path}")

        # Plot Speed
        fig_spd, ax_spd = plt.subplots(figsize=(8, 8))
        im_spd = ax_spd.imshow(mean_speed_map, origin='lower', extent=extent, vmin=0, vmax=1.5, cmap='jet', interpolation='nearest')
        if hasattr(loaded_walkable_area.polygon, 'exterior'):
            ax_spd.plot(x_ext, y_ext, color='white', linewidth=2)
        plt.colorbar(im_spd, ax=ax_spd, label="v / m/s")
        ax_spd.set_title(f"Average Speed ({file_id}) - GPU")
        
        spd_img_path = speed_dir / f"heatmap_speed_{file_id}.png"
        plt.savefig(spd_img_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_spd)
        print(f"Saved {spd_img_path}")
        
    t1 = time.time()
    print(f"[{file_id}] Finished in {t1-t0:.2f} seconds.")
    return trajectory_data, loaded_walkable_area

# ==========================================
# setup
# ==========================================
START_SEED = 100095
END_SEED = 100100  # Adjust for multiple seeds (e.g., 5 or 100)
DPI = 300  # Default DPI for all saved figures

# Base directory setup (relative to this script's location)
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Define input/output folders based on new structure
TOPO_DIR = BASE_DIR.parent.parent / "Topo_2"
DATASWARM_DIR = TOPO_DIR / "dataswarm"
HEATMAP_DENSITY_DIR = TOPO_DIR / "heatmap_density"
HEATMAP_SPEED_DIR = TOPO_DIR / "heatmap_speed"

# Create directories if they do not exist
for directory in [HEATMAP_DENSITY_DIR, HEATMAP_SPEED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print_system_resources()
    
    # ==========================================
    # loop and process (Heatmap Only)
    # ==========================================
    sqlite_files = list(DATASWARM_DIR.rglob("*.sqlite"))
    
    if not sqlite_files:
        print(f"Warning: No .sqlite files found in {DATASWARM_DIR}")
        
    for trajectory_file in sqlite_files:
        rel_path = trajectory_file.relative_to(DATASWARM_DIR)
        parent_subfolder = rel_path.parent
        file_stem = trajectory_file.stem
        
        # Create corresponding subdirectories in output folders
        out_density_dir = HEATMAP_DENSITY_DIR / parent_subfolder
        out_speed_dir = HEATMAP_SPEED_DIR / parent_subfolder
        
        out_density_dir.mkdir(parents=True, exist_ok=True)
        out_speed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*40}")
        print(f"Processing File (Heatmap): {rel_path}")
        print(f"{'='*40}")
            
        # 1. Run Heatmap Analysis
        trajectory_data, loaded_walkable_area = generate_heatmaps(
            trajectory_file, file_stem, 
            out_density_dir, out_speed_dir, DPI
        )
        
        # --- Aggressive Memory Cleanup ---
        del trajectory_data
        del loaded_walkable_area
        
        plt.close('all') 
        gc.collect() 
        print_memory_usage("End of Loop", file_stem)
        # --------------------------------

    print("\n=== Heatmap Script Completed ===")

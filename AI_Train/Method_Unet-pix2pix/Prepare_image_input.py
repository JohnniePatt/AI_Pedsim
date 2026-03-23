import os
import pathlib
import sqlite3
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time
import shapely
import cv2
import json
import random
from shapely import wkt
from shapely.geometry import Polygon, Point

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found. Fast processing requires PyTorch.")

import pedpy
try:
    from jupedsim.internal.notebook_utils import read_sqlite_file
except ImportError:
    read_sqlite_file = None

# --- Helpers ---
def setup_gpu_grid(walkable_area_polygon, grid_size, device):
    """ใช้ 0,0 เป็นจุดเริ่มคงที่ (Global Origin)"""
    min_x, min_y = 0.0, 0.0
    max_x, max_y = walkable_area_polygon.bounds[2], walkable_area_polygon.bounds[3]
    nx = int(np.ceil(max_x / grid_size))
    ny = int(np.ceil(max_y / grid_size))
    
    x_coords = np.arange(nx) * grid_size + (grid_size / 2.0)
    y_coords = np.arange(ny) * grid_size + (grid_size / 2.0)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    mask = np.array([walkable_area_polygon.contains(Point(p)) for p in grid_points])
        
    grid_coords_gpu = torch.tensor(grid_points, dtype=torch.float32, device=device)
    mask_gpu = torch.tensor(mask, dtype=torch.bool, device=device)
    return grid_coords_gpu, mask_gpu, nx, ny

def compute_gpu_density(ped_x, ped_y, grid_coords_gpu, mask_gpu, nx, ny, grid_size, device):
    if len(ped_x) == 0: return np.zeros((ny, nx))
    ped_coords = torch.tensor(np.column_stack((ped_x, ped_y)), dtype=torch.float32, device=device)
    dist = torch.cdist(grid_coords_gpu, ped_coords)
    nearest_ped_idx = dist.argmin(dim=1)
    ped_areas = torch.bincount(nearest_ped_idx[mask_gpu], minlength=len(ped_coords)) * (grid_size**2)
    grid_density = (1.0 / torch.clamp(ped_areas, min=1e-5))[nearest_ped_idx]
    grid_density[~mask_gpu] = 0.0
    return grid_density.reshape(ny, nx).cpu().numpy()

def draw_wkt_pure(img, wkt_str_or_poly, grid_size, color_rgb):
    """วาด Polygon แบบ 'Pure Color' (ล้างสีอื่นทิ้ง ณ จุดนั้น)"""
    try:
        if isinstance(wkt_str_or_poly, str):
            poly = wkt.loads(wkt_str_or_poly)
        else:
            poly = wkt_str_or_poly
            
        if hasattr(poly, 'exterior') and poly.exterior:
            ext_pts = (np.array(poly.exterior.coords) / grid_size).astype(np.int32)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [ext_pts], 255)
            img[mask > 0] = 0
            cv2.fillPoly(img, [ext_pts], color_rgb)
            if hasattr(poly, 'interiors'):
                for interior in poly.interiors:
                    int_pts = (np.array(interior.coords) / grid_size).astype(np.int32)
                    cv2.fillPoly(img, [int_pts], (0, 0, 0))
    except Exception: pass

# --- Setup Paths ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
TOPO_DIR = BASE_DIR.parent.parent / "Topo_2"
DATASWARM_DIR = TOPO_DIR / "dataswarm"
AREA_DIR = TOPO_DIR / "spawn_exit_area"
DATASET_DIR = TOPO_DIR / "heatmap_density" / "Cleandata_1"

def process_file_final(trajectory_file, out_A_dir, out_B_dir, mode, spawn_mode):
    device = torch.device('cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu')
    file_stem = trajectory_file.stem
    seed = file_stem.split('_')[-1]
    
    if read_sqlite_file:
        try: traj_data, walkable = read_sqlite_file(str(trajectory_file))
        except: 
            traj_data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=trajectory_file)
            walkable = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_file)
    else:
        traj_data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=trajectory_file)
        walkable = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_file)
    
    agent_count = traj_data.data['id'].nunique()
    grid_size = 0.5
    bounds = walkable.polygon.bounds
    nx = int(np.ceil(bounds[2] / grid_size))
    ny = int(np.ceil(bounds[3] / grid_size))

    if nx < ny:
        new_w, new_h = 512, int(ny * (512 / nx))
    else:
        new_h, new_w = 512, int(nx * (512 / ny))

    # --- TASK B: Density Map ---
    if mode in [2, 3]:
        ind_speed = pedpy.compute_individual_speed(traj_data=traj_data, frame_step=5, speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED)
        grid_coords, mask_gpu, _, _ = setup_gpu_grid(walkable.polygon, grid_size, device)
        frames = ind_speed['frame'].unique()[::60]
        sum_d = np.zeros((ny, nx))
        for f in frames:
            pts = traj_data.data[traj_data.data.frame == f][['x', 'y']].values
            sum_d += compute_gpu_density(pts[:, 0], pts[:, 1], grid_coords, mask_gpu, nx, ny, grid_size, device)
        density_img = cv2.resize(np.log1p(sum_d/len(frames)), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        density_img = np.flipud(density_img)
        density_img_u8 = np.clip((density_img / np.log1p(64.0)) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_B_dir / f"{file_stem}.png"), density_img_u8)

    # --- TASK A: Mask Image ---
    if mode in [1, 3]:
        subfolder = trajectory_file.relative_to(DATASWARM_DIR).parent
        area_csv = AREA_DIR / subfolder / f"spawn_exit_{seed}.csv"
        area_df = pd.read_csv(area_csv)
        spawn_wkt = area_df[area_df['type'] == 'spawning_area']['area'].values[0]
        exit_wkt = area_df[area_df['type'] == 'exit_area']['area'].values[0]

        img_a = np.zeros((ny, nx, 3), dtype=np.uint8)
        # 1. Floor (Red)
        draw_wkt_pure(img_a, walkable.polygon, grid_size, (255, 0, 0))
        
        # 2. Spawn (Green)
        if spawn_mode == 2: # Dots Mode
            # ดึงพิกัดแรกสุดของแต่ละ ID
            starts = traj_data.data.groupby('id')[['x', 'y']].first().values
            spawn_val = int(np.clip((agent_count / 2000.0) * 255.0, 150, 255))
            for x, y in starts:
                px, py = int(x / grid_size), int(y / grid_size)
                if 0 <= px < nx and 0 <= py < ny:
                    img_a[py, px] = [0, spawn_val, 0] # Pure Green dot
        else: # Rectangle Mode (1)
            spawn_val = int(np.clip((agent_count / 2000.0) * 255.0, 150, 255))
            draw_wkt_pure(img_a, spawn_wkt, grid_size, (0, spawn_val, 0))
        
        # 3. Exit (Blue)
        draw_wkt_pure(img_a, exit_wkt, grid_size, (0, 0, 255))
        
        input_img_res = np.flipud(cv2.resize(img_a, (new_w, new_h), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(out_A_dir / f"{file_stem}.png"), cv2.cvtColor(input_img_res, cv2.COLOR_RGB2BGR))

    return agent_count

if __name__ == "__main__":
    print("\n--- Pix2Pix Preparation Tool ---")
    print("1: Generate A (Input), 2: Generate B (Target), 3: BOTH")
    try: choice = int(input("Select Mode: ").strip())
    except: choice = 3
    
    spawn_mode = 1
    if choice in [1, 3]:
        print("\nSelect Spawn Visualization:")
        print("1: Rectangle (Solid), 2: Dots (Agent Positions)")
        try: spawn_mode = int(input("Select Style: ").strip())
        except: spawn_mode = 1

    sqlite_files = list(DATASWARM_DIR.rglob("*.sqlite"))
    total_files = len(sqlite_files)
    pbar = tqdm(total=total_files, desc="Progress")
    for i, traj_file in enumerate(sqlite_files):
        out_A = DATASET_DIR / "A" / traj_file.relative_to(DATASWARM_DIR).parent
        out_B = DATASET_DIR / "B" / traj_file.relative_to(DATASWARM_DIR).parent
        out_A.mkdir(parents=True, exist_ok=True); out_B.mkdir(parents=True, exist_ok=True)
        pbar.set_description(f"File: {traj_file.name}")
        try:
            agents = process_file_final(traj_file, out_A, out_B, choice, spawn_mode)
            tqdm.write(f"[{i+1}/{total_files}] DONE: {traj_file.name} | Agents: {agents}")
        except Exception as e: tqdm.write(f"[ERROR] {traj_file.name}: {e}")
        pbar.update(1)
    pbar.close()

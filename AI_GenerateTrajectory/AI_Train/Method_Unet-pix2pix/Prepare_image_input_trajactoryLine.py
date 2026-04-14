import os
import pathlib
import sqlite3
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2
import json
import random
from shapely import wkt
from shapely.geometry import Polygon, Point, LineString

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import pedpy

# --- Helpers ---
def draw_wkt_pure(img, wkt_str_or_poly, grid_size, color_bgr):
    """Draw a polygon with a pure color."""
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
            cv2.fillPoly(img, [ext_pts], color_bgr)
            if hasattr(poly, 'interiors'):
                for interior in poly.interiors:
                    int_pts = (np.array(interior.coords) / grid_size).astype(np.int32)
                    cv2.fillPoly(img, [int_pts], (0, 0, 0))
    except Exception: pass

def get_bgr_color(rgb_tuple):
    """Convert RGB to BGR for OpenCV."""
    return (rgb_tuple[2], rgb_tuple[1], rgb_tuple[0])

# --- Setup Paths ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_bottleneck"
DATASWARM_DIR = TOPO_DIR / "dataswarm"
AREA_DIR = TOPO_DIR / "spawn_exit_area"
# New Dataset Directory for Trajectory Lines
DATASET_DIR = TOPO_DIR / "trajectory_line_dataset" / "Cleandata_1"

def process_file_trajectory(trajectory_file, out_A_dir, out_B_dir, mode, spawn_mode):
    file_stem = trajectory_file.stem
    seed = file_stem.split('_')[-1]
    
    # Load trajectory and walkable area
    traj_data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file=trajectory_file)
    walkable = pedpy.load_walkable_area_from_jupedsim_sqlite(trajectory_file=trajectory_file)
    
    agent_count = traj_data.data['id'].nunique()
    grid_size = 0.5
    bounds = walkable.polygon.bounds
    nx = int(np.ceil(bounds[2] / grid_size))
    ny = int(np.ceil(bounds[3] / grid_size))

    # Calculate target dimensions (standardizing on 512 for one side)
    if nx < ny:
        new_w, new_h = 512, int(ny * (512 / nx))
    else:
        new_h, new_w = 512, int(nx * (512 / ny))

    # --- TASK A: Mask Image (Input) ---
    # Design: Red Floor, Green Spawn, Blue Exit
    if mode in [1, 3]:
        subfolder = trajectory_file.relative_to(DATASWARM_DIR).parent
        area_csv = AREA_DIR / subfolder / f"spawn_exit_{seed}.csv"
        
        spawn_wkt = None
        exit_wkt = None
        if area_csv.exists():
            area_df = pd.read_csv(area_csv)
            spawn_wkt = area_df[area_df['type'] == 'spawning_area']['area'].values[0]
            exit_wkt = area_df[area_df['type'] == 'exit_area']['area'].values[0]

        img_a = np.zeros((ny, nx, 3), dtype=np.uint8)
        # 1. Floor (Red in RGB -> (0, 0, 255) in BGR)
        draw_wkt_pure(img_a, walkable.polygon, grid_size, (0, 0, 255))
        
        # 2. Spawn (Green in RGB -> (0, 255, 0) in BGR)
        spawn_val = int(np.clip((agent_count / 2000.0) * 255.0, 150, 255))
        if spawn_mode == 2: # Dots Mode
            starts = traj_data.data.groupby('id')[['x', 'y']].first().values
            for x, y in starts:
                px, py = int(x / grid_size), int(y / grid_size)
                if 0 <= px < nx and 0 <= py < ny:
                    img_a[py, px] = [0, spawn_val, 0]
        elif spawn_wkt: # Rectangle Mode
            draw_wkt_pure(img_a, spawn_wkt, grid_size, (0, spawn_val, 0))
        
        # 3. Exit (Blue in RGB -> (255, 0, 0) in BGR)
        if exit_wkt:
            draw_wkt_pure(img_a, exit_wkt, grid_size, (255, 0, 0))
        
        input_img_res = np.flipud(cv2.resize(img_a, (new_w, new_h), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(out_A_dir / f"{file_stem}.png"), input_img_res)

    # --- TASK B: Trajectory Line Plot (Target) ---
    # Design: Gray environment with Pink lines
    if mode in [2, 3]:
        img_b = np.zeros((ny, nx, 3), dtype=np.uint8)
        
        # 1. Draw Gray Floor (BGR: 220, 220, 220)
        draw_wkt_pure(img_b, walkable.polygon, grid_size, (220, 220, 220))
        
        # 2. Draw Trajectories (Pink in RGB -> BGR)
        # Pink color: BGR (203, 192, 255) or similar
        line_color = (180, 150, 255) # Light Pinkish
        
        for aid, agent_data in traj_data.data.groupby('id'):
            # Points list in (pix_x, pix_y)
            pts = agent_data[['x', 'y']].values
            pixel_pts = (pts / grid_size).astype(np.int32)
            
            # Draw line segments
            if len(pixel_pts) > 1:
                cv2.polylines(img_b, [pixel_pts], False, line_color, thickness=1, lineType=cv2.LINE_AA)

        target_img_res = np.flipud(cv2.resize(img_b, (new_w, new_h), interpolation=cv2.INTER_LINEAR))
        cv2.imwrite(str(out_B_dir / f"{file_stem}.png"), target_img_res)

    return agent_count

if __name__ == "__main__":
    print("\n--- Pix2Pix Trajectory Line Preparation Tool ---")
    print("1: Generate A (Input), 2: Generate B (Target), 3: BOTH")
    try: choice = int(input("Select Mode [Default: 3]: ").strip() or "3")
    except: choice = 3
    
    spawn_mode = 1
    if choice in [1, 3]:
        print("\nSelect Spawn Visualization:")
        print("1: Rectangle (Solid), 2: Dots (Agent Positions)")
        try: spawn_mode = int(input("Select Style [Default: 1]: ").strip() or "1")
        except: spawn_mode = 1

    sqlite_files = list(DATASWARM_DIR.rglob("*.sqlite"))
    total_files = len(sqlite_files)
    
    if total_files == 0:
        print(f"ERROR: No sqlite files found in {DATASWARM_DIR}")
        exit(1)

    pbar = tqdm(total=total_files, desc="Progress")
    for i, traj_file in enumerate(sqlite_files):
        out_A = DATASET_DIR / "A" / traj_file.relative_to(DATASWARM_DIR).parent
        out_B = DATASET_DIR / "B" / traj_file.relative_to(DATASWARM_DIR).parent
        out_A.mkdir(parents=True, exist_ok=True); out_B.mkdir(parents=True, exist_ok=True)
        pbar.set_description(f"File: {traj_file.name}")
        try:
            agents = process_file_trajectory(traj_file, out_A, out_B, choice, spawn_mode)
            # tqdm.write(f"[{i+1}/{total_files}] DONE: {traj_file.name} | Agents: {agents}")
        except Exception as e: tqdm.write(f"[ERROR] {traj_file.name}: {e}")
        pbar.update(1)
    pbar.close()
    print(f"\n✅ Dataset prepared at: {DATASET_DIR}")

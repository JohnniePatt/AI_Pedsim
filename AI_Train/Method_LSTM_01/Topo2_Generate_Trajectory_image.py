"""
Topo2_Generate_Trajectory_image.py

# เป้าหมาย
สร้างรูปวาดเส้นทาง (Trajectory Line Plots) เพื่อเปรียบเทียบ
ระหว่างข้อมูลจริง (Ground Truth - Blue) และ AI ทาย (AI Generated - Orange)
อ้างอิงสไตล์การวาดจาก main_script_sim.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
import json
from tqdm import tqdm
import gc

# สำหรับวาดแผนที่แบบเดียวกับ main_script_sim.py
import pedpy

# ===================================================================== #
# CONFIGURATION
# ===================================================================== #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOPO_DIR = PROJECT_ROOT / "Topo_2"
GEO_DIR = TOPO_DIR / "geo"
DATASWARM_DIR = TOPO_DIR / "dataswarm" / "test"
GEN_PATH_DIR = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2" / "generated_paths"
IMG_OUTPUT_DIR = PROJECT_ROOT / "AI_Train" / "outputs" / "Topo2" / "generated_images"

# ===================================================================== #
# HELPERS
# ===================================================================== #

def load_json_polygons(filepath: Path) -> list:
    with open(filepath, 'r') as f: data = json.load(f)
    return [Polygon(coords) for coords in data]

def get_walkable_area(room_polys, corridor_polys):
    all_geoms = room_polys + corridor_polys
    return unary_union(all_geoms)

# ===================================================================== #
# CORE PLOTTING ENGINE
# ===================================================================== #

def plot_comparison(seed_id, gen_df, gt_df, area_union, room_polys, corridor_polys):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. วาดฉากหลัง (Walkable Area)
    # ใช้ pedpy ช่วยวาดขอบเขตสีเทา/ดำ
    walk_area = pedpy.WalkableArea(area_union)
    pedpy.plot_walkable_area(walkable_area=walk_area, axes=ax)

    # 2. วาดเส้นทาง Ground Truth (สีน้ำเงิน - โปร่งแสงนิดๆ)
    for aid, agent_data in gt_df.groupby('id'):
        ax.plot(agent_data['pos_x'], agent_data['pos_y'], color='blue', alpha=0.3, linewidth=0.5, label='Actual' if aid == gt_df['id'].iloc[0] else "")

    # 3. วาดเส้นทาง AI Generated (สีส้ม - เด่นชัดขึ้นมาหน่อย)
    for aid, agent_data in gen_df.groupby('id'):
        ax.plot(agent_data['pos_x'], agent_data['pos_y'], color='#ff7f0e', alpha=0.8, linewidth=0.8, label='AI Predicted' if aid == gen_df['id'].iloc[0] else "")

    # 4. ตกแต่งรูป
    ax.set_aspect('equal')
    ax.set_title(f"Trajectory Prediction Comparison (Seed: {seed_id})")
    ax.legend(loc='upper right')
    
    img_path = IMG_OUTPUT_DIR / f"comparison_{seed_id}.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return img_path

def main():
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Loading Geometry...")
    room_polys = load_json_polygons(GEO_DIR / "geo_room.json")
    corridor_polys = load_json_polygons(GEO_DIR / "geo_corridor.json")
    area_union = get_walkable_area(room_polys, corridor_polys)
    
    # ค้นหาไฟล์ .parquet ที่ AI สร้างไว้
    parquet_files = sorted(list(GEN_PATH_DIR.glob("paths_seed_*.parquet")))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {GEN_PATH_DIR}")
        print("Please run Topo2_Generate_Trajectory.py first!")
        return

    print(f"🚀 Found {len(parquet_files)} seeds. Starting plots...")

    for p_file in tqdm(parquet_files, desc="🖼️ Drawing Images"):
        seed_id = p_file.stem.split('_')[2]
        
        # 1. โหลดข้อมูล AI
        gen_df = pd.read_parquet(p_file)
        
        # 2. โหลดข้อมูลจริง (SQLite) เพื่อมาเทียบ
        sqlite_path = DATASWARM_DIR / f"double-botteleneck_{seed_id}.sqlite"
        if not sqlite_path.exists():
            print(f"⚠️ Missing SQLite for seed {seed_id}, skipping ground truth.")
            gt_df = pd.DataFrame(columns=['id', 'pos_x', 'pos_y'])
        else:
            conn = sqlite3.connect(sqlite_path)
            gt_df = pd.read_sql_query("SELECT id, pos_x, pos_y FROM trajectory_data", conn)
            conn.close()

        # 3. วาดรูป
        img_p = plot_comparison(seed_id, gen_df, gt_df, area_union, room_polys, corridor_polys)
        
        # ประหยัดแรม
        del gen_df, gt_df
        gc.collect()

    print(f"\n✅ All images saved in {IMG_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

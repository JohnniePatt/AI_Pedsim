import os
import shutil
import sqlite3
import pandas as pd
import glob
import json

def generate_spawn_location(parquet_path, output_path):
    """Extracts Frame 0 from parquet and saves as Spawn_location_xxx.csv"""
    try:
        df = pd.read_parquet(parquet_path)
        # Handle flexible column names
        cols = {c.lower().replace(' ', '').replace('_', ''): c for c in df.columns}
        
        f_col = cols.get('frame')
        id_col = cols.get('id') or cols.get('agentid')
        x_col = cols.get('posx') or cols.get('positionx') or cols.get('x')
        y_col = cols.get('posy') or cols.get('positiony') or cols.get('y')

        if all([f_col, id_col, x_col, y_col]):
            frame_0 = df[df[f_col] == 0][[id_col, x_col, y_col]].copy()
            frame_0.columns = ['id', 'pos_x', 'pos_y']
            frame_0.to_csv(output_path, index=False)
            return True
    except Exception as e:
        print(f"      ⚠️ Failed to generate spawn location: {e}")
    return False

def reorganize_bottleneck():
    source_root = "Geo_scenario/Topo_bottleneck"
    source_parquet_root = os.path.join(source_root, "dataswarm_parquet")
    source_geo_dir = os.path.join(source_root, "geo")
    source_exit_root = os.path.join(source_root, "spawn_exit_area")
    dest_root = "Dataset_Traj_Table/Topo_bottleneck"
    
    os.makedirs(dest_root, exist_ok=True)
    
    # Define global geo files for bottleneck
    geo_corridor_src = os.path.join(source_geo_dir, "geo_corridor.json")
    geo_room_src = os.path.join(source_geo_dir, "geo_room.json")
    
    for split in ["train", "test", "validation"]:
        split_dir = os.path.join(source_parquet_root, split)
        exit_split_dir = os.path.join(source_exit_root, split)
        if not os.path.exists(split_dir): continue
            
        print(f"📦 Processing Topo_bottleneck [{split}]...")
        for parquet_file in glob.glob(os.path.join(split_dir, "*.parquet")):
            filename = os.path.basename(parquet_file)
            # Find Case ID (e.g. 100102)
            case_id = None
            for segment in filename.split('_'):
                if segment.isdigit() and len(segment) >= 6:
                    case_id = segment
                    break
            
            if not case_id: continue
            
            case_dir = os.path.join(dest_root, f"case_{case_id}")
            os.makedirs(case_dir, exist_ok=True)
            
            # 1. Copy Parquet
            target_parquet = os.path.join(case_dir, filename)
            shutil.copy2(parquet_file, target_parquet)
            
            # 2. Copy Geo files (rename to exact names requested)
            if os.path.exists(geo_corridor_src):
                shutil.copy2(geo_corridor_src, os.path.join(case_dir, "Geo_corridor.json"))
            if os.path.exists(geo_room_src):
                shutil.copy2(geo_room_src, os.path.join(case_dir, "Geo_room.json"))
            
            # 3. Copy Spawn_exit_xxx.csv
            exit_src = os.path.join(exit_split_dir, f"spawn_exit_{case_id}.csv")
            if os.path.exists(exit_src):
                shutil.copy2(exit_src, os.path.join(case_dir, f"Spawn_exit_{case_id}.csv"))
            
            # 4. Generate Spawn_location_xxx.csv (Frame 0)
            generate_spawn_location(target_parquet, os.path.join(case_dir, f"Spawn_location_{case_id}.csv"))
            
            print(f"   ✅ Case {case_id} organized.")

def reorganize_housegan():
    source_root = "Geo_scenario/Topo_HouseGAN"
    source_geo_root = os.path.join(source_root, "geo")
    source_exit_root = os.path.join(source_root, "spawn_exit_area")
    # For HouseGAN, parquet might be in outputs or converted already
    source_parquet_root = os.path.join(source_root, "dataswarm_parquet") 
    dest_root = "Dataset_Traj_Table/Topo_HouseGAN"
    
    os.makedirs(dest_root, exist_ok=True)
    if not os.path.exists(source_parquet_root):
        print(f"⚠️ Topo_HouseGAN: dataswarm_parquet not found. Skipping trajectory organization.")
        return

    for split in ["train", "test", "validation"]:
        split_dir = os.path.join(source_parquet_root, split)
        exit_split_dir = os.path.join(source_exit_root, split)
        if not os.path.exists(split_dir): continue

        print(f"🏠 Processing Topo_HouseGAN [{split}]...")
        for parquet_file in glob.glob(os.path.join(split_dir, "*.parquet")):
            filename = os.path.basename(parquet_file)
            case_id = filename.split('_')[1] if '_' in filename else filename.split('.')[0]
            
            case_dir = os.path.join(dest_root, f"case_{case_id}")
            os.makedirs(case_dir, exist_ok=True)
            
            # 1. Copy Parquet
            target_parquet = os.path.join(case_dir, filename)
            shutil.copy2(parquet_file, target_parquet)
            
            # 2. Copy Geo files (find matching plan folder)
            plan_folder_prefix = f"plan_{case_id.split('_')[0]}" # Heuristic for HouseGAN plans
            plan_dirs = glob.glob(os.path.join(source_geo_root, f"{plan_folder_prefix}*"))
            if plan_dirs:
                src_geo_dir = plan_dirs[0]
                shutil.copy2(os.path.join(src_geo_dir, "geo_corridor.json"), os.path.join(case_dir, "Geo_corridor.json"))
                shutil.copy2(os.path.join(src_geo_dir, "geo_room.json"), os.path.join(case_dir, "Geo_room.json"))
            
            # 3. Copy Spawn exit
            exit_src = os.path.join(exit_split_dir, f"spawn_exit_{case_id}.csv")
            if os.path.exists(exit_src):
                shutil.copy2(exit_src, os.path.join(case_dir, f"Spawn_exit_{case_id}.csv"))

            # 4. Generate Spawn location
            generate_spawn_location(target_parquet, os.path.join(case_dir, f"Spawn_location_{case_id}.csv"))
            
            print(f"   ✅ Case {case_id} organized.")

if __name__ == "__main__":
    import sys
    print("🚀 Starting Professional Dataset Reorganization...", flush=True)
    try:
        reorganize_bottleneck()
        reorganize_housegan()
        print("✨ Reorganization complete. Your Dataset_Traj_Table is now AI-Ready!", flush=True)
    except Exception as e:
        print(f"❌ Critical Error: {e}", file=sys.stderr, flush=True)

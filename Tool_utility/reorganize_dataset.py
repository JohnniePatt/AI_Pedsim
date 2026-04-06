import os
import shutil
import sqlite3
import pandas as pd
import glob
import json

def reorganize_bottleneck():
    source_parquet_root = "Geo_scenario/Topo_bottleneck/dataswarm_parquet"
    source_geo_dir = "Geo_scenario/Topo_bottleneck/geo"
    dest_root = "Dataset_Traj_Table/Topo_bottleneck"
    
    os.makedirs(dest_root, exist_ok=True)
    
    # Define source geo files
    geo_corridor_src = os.path.join(source_geo_dir, "geo_corridor.json")
    geo_room_src = os.path.join(source_geo_dir, "geo_room.json")
    
    # Process train, test, validation folders
    for split in ["train", "test", "validation"]:
        split_dir = os.path.join(source_parquet_root, split)
        if not os.path.exists(split_dir):
            continue
            
        for parquet_file in glob.glob(os.path.join(split_dir, "*.parquet")):
            filename = os.path.basename(parquet_file)
            # Example: double-botteleneck_100102_trajectory_data.parquet
            # Extract case ID
            parts = filename.split('_')
            if len(parts) >= 2:
                case_id = parts[1]
                case_dir = os.path.join(dest_root, f"case_{case_id}")
                os.makedirs(case_dir, exist_ok=True)
                
                # Copy Parquet
                shutil.copy2(parquet_file, os.path.join(case_dir, filename))
                
                # Copy Geo files (rename to exact names)
                shutil.copy2(geo_corridor_src, os.path.join(case_dir, "Geo_corridor.json"))
                shutil.copy2(geo_room_src, os.path.join(case_dir, "Geo_room.json"))
                
                print(f"Processed Topo_bottleneck case: {case_id}")

def reorganize_housegan():
    source_geo_root = "Geo_scenario/Topo_HouseGAN/geo"
    source_output_root = "Prepare_data/Architecture_housePlan/outputs"
    dest_root = "Dataset_Traj_Table/Topo_HouseGAN"
    
    os.makedirs(dest_root, exist_ok=True)
    
    # Find all plan directories
    plan_dirs = [d for d in os.listdir(source_geo_root) if os.path.isdir(os.path.join(source_geo_root, d)) and d.startswith("plan_")]
    
    for plan_id in plan_dirs:
        # Extract case ID from plan name (e.g. plan_46_481d -> case_46_481d)
        case_id = plan_id.replace("plan_", "")
        case_dir = os.path.join(dest_root, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)
        
        # 1. Copy Geo files
        plan_src_dir = os.path.join(source_geo_root, plan_id)
        shutil.copy2(os.path.join(plan_src_dir, "geo_corridor.json"), os.path.join(case_dir, "Geo_corridor.json"))
        shutil.copy2(os.path.join(plan_src_dir, "geo_room.json"), os.path.join(case_dir, "Geo_room.json"))
        
        # 2. Find and convert Trajectory SQLite to Parquet
        # Check source_output_root/plan_id/dataswarm/
        sqlite_dir = os.path.join(source_output_root, plan_id, "dataswarm")
        if os.path.exists(sqlite_dir):
            sqlite_files = glob.glob(os.path.join(sqlite_dir, "*.sqlite"))
            for sqlite_file in sqlite_files:
                try:
                    # Convert SQLite to Parquet using pandas
                    conn = sqlite3.connect(sqlite_file)
                    # Find table name
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    for table in tables:
                        table_name = table[0]
                        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        
                        # Ensure standard column names
                        # Mapping might be needed depending on SQLite schema
                        # The user wants: Frame, Agent id, position X, Y
                        # Let's check common names: frame, agent_id, pos_x, pos_y
                        rename_map = {
                            'frame': 'Frame',
                            'agent_id': 'Agent id',
                            'pos_x': 'position X',
                            'pos_y': 'Y'
                        }
                        # Find existing columns and rename them
                        existing_cols = {c.lower().replace('_', '').replace(' ', ''): c for c in df.columns}
                        final_rename = {}
                        if 'frame' in existing_cols: final_rename[existing_cols['frame']] = 'Frame'
                        if 'agentid' in existing_cols: final_rename[existing_cols['agentid']] = 'Agent id'
                        if 'posx' in existing_cols: final_rename[existing_cols['posx']] = 'position X'
                        if 'posy' in existing_cols: final_rename[existing_cols['posy']] = 'Y'
                        
                        df.rename(columns=final_rename, inplace=True)
                        
                        # Save as Parquet
                        parquet_name = f"housegan_{case_id}_{table_name}.parquet"
                        df.to_parquet(os.path.join(case_dir, parquet_name))
                        print(f"Converted {sqlite_file} table {table_name} to Parquet for case {case_id}")
                    
                    conn.close()
                except Exception as e:
                    print(f"Error converting SQLite {sqlite_file}: {e}")
        else:
            print(f"No trajectory data found for HouseGAN case {case_id} in {sqlite_dir}")

if __name__ == "__main__":
    print("Starting dataset reorganization...")
    reorganize_bottleneck()
    reorganize_housegan()
    print("Reorganization complete.")

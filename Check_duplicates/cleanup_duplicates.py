import os
import json
import re
from pathlib import Path

def extract_id(filename):
    """Extracts the ID from filenames like spawn_exit_100171.csv -> 100171"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def get_all_linked_files(base_repo, target_id):
    """Finds all files in Topo_2 that contain the target_id."""
    # List of common patterns:
    # Topo_2/spawn_exit_area/split/spawn_exit_{id}.csv
    # Topo_2/spawn_exit/spawn_exit{id}.png (No underscore)
    # Topo_2/trajectory_line/trajectory_{id}.png
    # Topo_2/dataswarm/split/spawn_exit_{id}.sqlite
    # Topo_2/heatmap_density/heatmap_density{id}.png
    # Topo_2/heatmap_speed/heatmap_speed{id}.png
    # Topo_2/geo/spawn_exit_{id}.geo.json (maybe?)
    
    files_to_remove = []
    topo_path = Path(base_repo) / "Topo_2"
    
    # We'll search for files that have the target_id in their name
    # Be careful not to match substrings of longer IDs if possible, 
    # but here IDs seem to be unique enough.
    
    # Common subfolders to check
    subfolders = [
        "spawn_exit", "trajectory_line", "spawn_exit_area", 
        "dataswarm", "heatmap_density", "heatmap_speed", "geo", "trajectory_line_dataset"
    ]
    
    for sub in subfolders:
        sub_path = topo_path / sub
        if not sub_path.exists(): continue
        
        # Search recursively for files containing the ID
        for file_path in sub_path.rglob(f"*{target_id}*"):
            # Ensure it's not a directory and it's a direct match to the ID context
            if file_path.is_file():
                # Check for common naming patterns to avoid accidents
                name = file_path.name
                if f"_{target_id}." in name or f"exit{target_id}." in name or f"density{target_id}." in name or f"speed{target_id}." in name or f"trajectory_{target_id}." in name:
                    files_to_remove.append(str(file_path))
                    
    return files_to_remove

def cleanup():
    duplicate_json = "csv_duplicates.json"
    if not os.path.exists(duplicate_json):
        print("Error: csv_duplicates.json not found.")
        return
    
    with open(duplicate_json, "r") as f:
        groups = json.load(f)
        
    all_files_removed = []
    ids_to_remove = []
    
    for group in groups:
        if not group: continue
        # Keep the first ID, remove others
        keep_id = extract_id(group[0])
        for other_filename in group[1:]:
            remove_id = extract_id(other_filename)
            if remove_id and remove_id != keep_id:
                ids_to_remove.append(remove_id)
    
    ids_to_remove = sorted(list(set(ids_to_remove)))
    print(f"Found {len(ids_to_remove)} IDs to remove.")
    
    base_dir = "." # Assumes we run in AI_Pedsim root
    
    for rid in ids_to_remove:
        files = get_all_linked_files(base_dir, rid)
        for f in files:
            try:
                os.remove(f)
                all_files_removed.append(f)
            except Exception as e:
                print(f"Error removing {f}: {e}")
                
    # Final count
    print("\n--- Summary ---")
    print(f"IDs with duplicate scenarios cleaned: {len(ids_to_remove)}")
    print(f"Total files deleted: {len(all_files_removed)}")
    
    # Also save the list of removed files
    with open("deleted_files_log.txt", "w") as f:
        for line in all_files_removed:
            f.write(line + "\n")

if __name__ == "__main__":
    cleanup()

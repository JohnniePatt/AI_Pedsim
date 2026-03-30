import os
import glob
from collections import defaultdict
import json

def find_duplicates_in_csvs(base_dir):
    area_map = defaultdict(list)
    
    # Search in train, test, validation
    search_pattern = os.path.join(base_dir, "**", "*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Checking {len(files)} CSV files...")
    
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) < 3: continue
                # Lines are Expected: Header, Spawn, Exit
                # Use simplified content (just the polygons)
                spawn = lines[1].strip()
                exit_pt = lines[2].strip()
                key = (spawn, exit_pt)
                area_map[key].append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Process results
    duplicates = [names for names in area_map.values() if len(names) > 1]
    return duplicates

if __name__ == "__main__":
    base_path = "Topo_2/spawn_exit_area"
    results = find_duplicates_in_csvs(base_path)
    
    if not results:
        print("✅ No duplicate spawn/exit areas found across the dataset.")
    else:
        print(f"⚠️ Found {len(results)} groups of duplicate scenarios!")
        for i, group in enumerate(results[:10]): # Show first 10
            print(f" Group {i+1}: {', '.join(group)}")
        
        # Save to file
        with open("csv_duplicates.json", "w") as f:
            json.dump(results, f, indent=4)

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np

def get_image_hash_ignoring_margins(filepath):
    """Calculates hash of the center part of the image using matplotlib/numpy."""
    try:
        # read image as numpy array
        img = plt.imread(str(filepath))
        height, width, _ = img.shape
        # Crop out margins (top title, bottom/left axes)
        left = int(width * 0.1)
        right = int(width * 0.9)
        top = int(height * 0.1)
        bottom = int(height * 0.9)
        
        cropped = img[top:bottom, left:right, :]
        
        # Convert to bytes for hashing
        # float32 values might have tiny variations, so we'll round and scale
        bytes_data = (cropped * 255).astype(np.uint8).tobytes()
        
        hasher = hashlib.md5()
        hasher.update(bytes_data)
        return hasher.hexdigest()
    except Exception as e:
        return f"ERROR_{e}"

def find_duplicates(directory):
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory {directory} not found."}

    hash_map = defaultdict(list)
    files = sorted(list(path.glob("*.png")))
    total = len(files)
    print(f"🔍 Analyzing {total} images in {directory}...")
    
    for i, file_path in enumerate(files):
        h = get_image_hash_ignoring_margins(file_path)
        hash_map[h].append(file_path.name)
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{total}...")

    duplicates = [names for h, names in hash_map.items() if len(names) > 1 and not h.startswith("ERROR")]
    return {
        "count": total,
        "duplicates": duplicates
    }

if __name__ == "__main__":
    results = {
        "spawn_exit_scenarios": find_duplicates("Topo_2/spawn_exit"),
        "trajectory_scenarios": find_duplicates("Topo_2/trajectory_line")
    }
    
    with open("scenario_duplicates.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Also print any groups
    if results['spawn_exit_scenarios']['duplicates']:
        print("Duplicate spawn_exit paires found!")
        # Print first 5 groups
        for g in results['spawn_exit_scenarios']['duplicates'][:5]:
            print(f" - {g}")
    else:
        print("No duplicate spawn_exit scenarios found.")

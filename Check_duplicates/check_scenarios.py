import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json
from PIL import Image

def get_image_hash_ignoring_margins(filepath):
    """Calculates hash of the center part of the image to ignore seed labels."""
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        width, height = img.size
        # Crop out top title (contains seed) and bottom/left axes
        # Typical margins: 10% title, 10% axis
        left = int(width * 0.1)
        right = int(width * 0.9)
        top = int(height * 0.1)
        bottom = int(height * 0.9)
        
        cropped = img.crop((left, top, right, bottom))
        
        # We can also resize to make it faster/more robust to small shifts if needed
        # but let's keep exact pixels for now.
        
        hasher = hashlib.md5()
        hasher.update(cropped.tobytes())
        return hasher.hexdigest()

def find_duplicates(directory):
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory {directory} not found."}

    hash_map = defaultdict(list)
    files = list(path.glob("*.png"))
    print(f"🔍 Analyzing {len(files)} scenarios in {directory}...")
    
    for i, file_path in enumerate(files):
        try:
            h = get_image_hash_ignoring_margins(file_path)
            hash_map[h].append(file_path.name)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}...")

    duplicates = [names for names in hash_map.values() if len(names) > 1]
    return {
        "count": len(files),
        "duplicates": duplicates
    }

if __name__ == "__main__":
    results = {
        "spawn_exit_scenarios": find_duplicates("Topo_2/spawn_exit")
    }
    
    with open("scenario_duplicates.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Done! Found {len(results['spawn_exit_scenarios']['duplicates'])} sets of duplicate scenarios.")

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json

def get_image_hash(filepath):
    """Calculates MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def find_duplicates(directory):
    path = Path(directory)
    if not path.exists():
        return {"error": f"Directory {directory} not found."}

    hash_map = defaultdict(list)
    files = list(path.glob("*.png"))
    for file_path in files:
        h = get_image_hash(file_path)
        hash_map[h].append(file_path.name)

    duplicates = [names for names in hash_map.values() if len(names) > 1]
    return {
        "count": len(files),
        "duplicates": duplicates
    }

if __name__ == "__main__":
    results = {
        "spawn_exit": find_duplicates("Topo_bottleneck/spawn_exit"),
        "trajectory_line": find_duplicates("Topo_bottleneck/trajectory_line")
    }
    
    with open("duplicate_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Done! Results written to duplicate_results.json")

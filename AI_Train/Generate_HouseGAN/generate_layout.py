import argparse
import json
import time
import uuid
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union

class Room:
    def __init__(self, x, y, w, h, name="Room", type="room"):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.name = name
        self.type = type
        self.poly = box(x, y, x + w, y + h)
    
    def get_coords(self):
        c = list(self.poly.exterior.coords)
        return [[float(p[0]), float(p[1])] for p in c]

def generate_procedural_layout(total_rooms_range, num_corridors, door_width, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    total_nodes = random.randint(*total_rooms_range)
    rooms = []
    edges = [] # Graph edges
    
    # --- Phase 1: Skeleton (Corridors) ---
    for i in range(num_corridors):
        placed = False
        for _ in range(100):
            if not rooms:
                w, h = random.uniform(6, 10), random.uniform(2.5, 3.5)
                if random.random() > 0.5: w, h = h, w
                rooms.append(Room(0, 0, w, h, name=f"Cor-0", type="corridor"))
                placed = True; break
            
            target_idx = random.randrange(len(rooms))
            corridors_only = [idx for idx, r in enumerate(rooms) if r.type == 'corridor']
            if corridors_only: target_idx = random.choice(corridors_only)
            
            target_room = rooms[target_idx]
            side = random.choice(['N', 'S', 'E', 'W'])
            w, h = random.uniform(5, 8), random.uniform(2.5, 3.5)
            if side in ['E', 'W']: w, h = h, w
            
            if side == 'N': nx, ny = round(target_room.x + random.uniform(0, target_room.w - w), 2), target_room.y + target_room.h
            elif side == 'S': nx, ny = round(target_room.x + random.uniform(0, target_room.w - w), 2), target_room.y - h
            elif side == 'E': nx, ny = target_room.x + target_room.w, round(target_room.y + random.uniform(0, target_room.h - h), 2)
            else: nx, ny = target_room.x - w, round(target_room.y + random.uniform(0, target_room.h - h), 2)
            
            new_room = Room(nx, ny, w, h, name=f"Cor-{i}", type="corridor")
            if not any(new_room.poly.intersects(r.poly) and new_room.poly.intersection(r.poly).area > 0.05 for r in rooms):
                rooms.append(new_room)
                edges.append((target_idx, len(rooms) - 1)) # Connect to parent
                placed = True; break
                
    # --- Phase 2: Rooms ---
    num_rooms_to_add = max(0, total_nodes - len(rooms))
    for i in range(num_rooms_to_add):
        placed = False
        for _ in range(100):
            target_idx = random.randrange(len(rooms))
            target_room = rooms[target_idx]
            side = random.choice(['N', 'S', 'E', 'W'])
            w, h = random.uniform(3, 5), random.uniform(3, 5)
            
            if side == 'N': nx, ny = round(target_room.x + random.uniform(0, target_room.w - w), 2), target_room.y + target_room.h
            elif side == 'S': nx, ny = round(target_room.x + random.uniform(0, target_room.w - w), 2), target_room.y - h
            elif side == 'E': nx, ny = target_room.x + target_room.w, round(target_room.y + random.uniform(0, target_room.h - h), 2)
            else: nx, ny = target_room.x - w, round(target_room.y + random.uniform(0, target_room.h - h), 2)
            
            new_room = Room(nx, ny, w, h, name=f"Room-{i}", type="room")
            if not any(new_room.poly.intersects(r.poly) and new_room.poly.intersection(r.poly).area > 0.05 for r in rooms):
                rooms.append(new_room)
                edges.append((target_idx, len(rooms) - 1)) # Connect to parent
                placed = True; break

    # --- Phase 3: Robust Doors and Connectivity ---
    physical_doors = []
    TOL = 0.05 # 5cm tolerance for detection
    
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1, r2 = rooms[i], rooms[j]
            
            # Find actual overlap area with a small detection buffer
            # This is MUCH more robust than line-to-line intersection
            inter = r1.poly.buffer(TOL).intersection(r2.poly.buffer(TOL))
            
            if inter.is_empty or inter.area < 0.001:
                continue
            
            # Get the bounding box of the intersection area
            minx, miny, maxx, maxy = inter.bounds
            inter_w = maxx - minx
            inter_h = maxy - miny
            
            # Decide orientation based on which dimension is larger
            # (If it's a vertical shared wall, the intersection 'box' will be tall and thin)
            is_v = inter_h > inter_w
            is_h = inter_w > inter_h
            
            # Only carve door if the shared boundary is long enough
            shared_len = max(inter_w, inter_h)
            if shared_len > 0.5:
                mid_x = (minx + maxx) / 2
                mid_y = (miny + maxy) / 2
                
                physical_doors.append({
                    "pos": (float(mid_x), float(mid_y)),
                    "rooms": (r1.name, r2.name),
                    "horizontal": bool(is_h)
                })
                # Ensure connectivity in graph
                if (i, j) not in edges and (j, i) not in edges:
                    edges.append((i, j))

    return rooms, edges, physical_doors

def save_graph_preview(rooms, edges, img_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = {}
    n = len(rooms)
    for i in range(n):
        angle = 2 * np.pi * i / n
        dist = 0.8 if rooms[i].type == 'room' else 0.4
        pos[i] = (dist * np.cos(angle), dist * np.sin(angle))
    
    # Draw Edges
    for start, end in edges:
        p1, p2 = pos[start], pos[end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#95a5a6', lw=1.2, zorder=1)
    
    # Draw Nodes
    for i, r in enumerate(rooms):
        px, py = pos[i]
        color = '#27ae60' if r.type == 'corridor' else '#ecf0f1'
        ax.scatter(px, py, s=700 if r.type == 'corridor' else 400, color=color, edgecolors='#2c3e50', lw=2, zorder=2)
        ax.text(px, py, str(i), ha='center', va='center', fontsize=7, fontweight='bold')

    # Add margin to prevent cropping
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title("Topological Structure (Bubble Diagram)", fontsize=10)
    plt.savefig(img_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def save_visual_preview(rooms, doors, img_path, seed):
    fig, ax = plt.subplots(figsize=(8, 8))
    union = unary_union([r.poly for r in rooms])
    minx, miny, maxx, maxy = union.bounds
    
    for i, r in enumerate(rooms):
        coords = np.array(r.poly.exterior.coords)
        color = '#c8e6c9' if r.type == 'corridor' else '#f5f5f5'
        ax.add_patch(patches.Polygon(coords, fill=True, color=color, ec='black', lw=1.5))
        ax.text(r.x + r.w/2, r.y + r.h/2, f"{i}", ha='center', va='center', fontsize=8, fontweight='bold')
    
    for d in doors:
        px, py = d["pos"]
        col = '#f1c40f'
        if d["horizontal"]: ax.plot([px-0.6, px+0.6], [py, py], color=col, lw=5, ls='--', zorder=5)
        else: ax.plot([px, px], [py-0.6, py+0.7], color=col, lw=5, ls='--', zorder=5)
    
    # Increase padding for layout preview too
    ax.set_xlim(minx - 2.5, maxx + 2.5)
    ax.set_ylim(miny - 2.5, maxy + 2.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"Procedural Plan (Seed: {seed})", fontsize=10)
    plt.savefig(img_path, dpi=150, bbox_inches='tight'); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_housegan.json")
    args = parser.parse_args()
    
    config = {"num_scenarios": 5, "num_corridors": 1, "random_seed": 42, "door_width": 1.5, "complexity": "Medium (5-8 Rooms)"}
    if Path(args.config).exists():
        with open(args.config, "r") as f: config.update(json.load(f))

    comp = config["complexity"]
    r_range = (3, 5) if "Low" in comp else (8, 15) if "High" in comp else (5, 8)
    
    for i in range(config["num_scenarios"]):
        current_seed = config["random_seed"] + i
        rooms, edges, doors = generate_procedural_layout(r_range, config["num_corridors"], config["door_width"], current_seed)
        
        run_name = f"plan_{current_seed}_{uuid.uuid4().hex[:4]}"
        project_root = Path(__file__).resolve().parent.parent.parent
        run_dir = project_root / "Geo_scenario" / "Topo_HouseGAN" / "geo" /run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        graph_data = {"nodes": [{"id": i, "name": r.name, "type": r.type} for i, r in enumerate(rooms)], "edges": [{"from": e[0], "to": e[1]} for e in edges]}
        with open(run_dir / "topological_graph.json", "w") as f: json.dump(graph_data, f, indent=4)
        with open(run_dir / "metadata.json", "w") as f: json.dump({"seed": current_seed, "params": config, "rooms": len(rooms), "corridors": config["num_corridors"]}, f, indent=4)
        
        with open(run_dir / "geo_room.json", "w") as f: json.dump([r.get_coords() for r in rooms if r.type=='room'], f, indent=4)
        with open(run_dir / "geo_corridor.json", "w") as f: json.dump([r.get_coords() for r in rooms if r.type=='corridor'], f, indent=4)
        with open(run_dir / "geo_door.json", "w") as f: json.dump(doors, f, indent=4)
        
        save_visual_preview(rooms, doors, run_dir / "preview.png", current_seed)
        save_graph_preview(rooms, edges, run_dir / "preview_graph.png")
        print(f"✅ Generated Topology: {run_dir.name} in {run_dir}")

if __name__ == "__main__": main()

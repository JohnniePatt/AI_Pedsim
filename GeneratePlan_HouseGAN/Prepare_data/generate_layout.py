import argparse
import json
import random
import time
import uuid
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

AREA_PROFILES = {
    "default": {
        "room_size_range": (2.5, 7.0),
        "corridor_first_length_range": (6.0, 12.0),
        "corridor_attached_length_range": (5.0, 10.0),
        "corridor_width_range": (2.5, 4.0),
    },
    "big": {
        "room_size_range": (10.0, 25.0),
        "corridor_first_length_range": (12.0, 25.0),
        "corridor_attached_length_range": (10.0, 22.0),
        "corridor_width_range": (3.0, 6.0),
    },
}


class Room:
    def __init__(self, x, y, w, h, name="Room", type="room"):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.name = name
        self.type = type
        self.poly = box(x, y, x + w, y + h)

    def get_coords(self):
        coords = list(self.poly.exterior.coords)
        return [[float(p[0]), float(p[1])] for p in coords]


def can_place_attached_room(new_room, existing_rooms, target_idx, max_target_overlap=0.7):
    for idx, room in enumerate(existing_rooms):
        if not new_room.poly.intersects(room.poly):
            continue
        overlap_area = new_room.poly.intersection(room.poly).area
        if idx == target_idx:
            if overlap_area > max_target_overlap:
                return False
            continue
        if overlap_area > 0.051:
            return False
    return True


def generate_procedural_layout(total_rooms_range, max_corridors, door_width, seed, size_profile):
    random.seed(seed)
    np.random.seed(seed)

    total_nodes = random.randint(*total_rooms_range)
    num_corridors = random.randint(1, max_corridors)
    rooms = []
    edges = []

    for i in range(num_corridors):
        placed = False
        for _ in range(100):
            if not rooms:
                w = random.uniform(*size_profile["corridor_first_length_range"])
                h = random.uniform(*size_profile["corridor_width_range"])
                if random.random() > 0.5:
                    w, h = h, w
                rooms.append(Room(0, 0, w, h, name="Cor-0", type="corridor"))
                placed = True
                break

            corridors_only = [idx for idx, r in enumerate(rooms) if r.type == "corridor"]
            target_idx = random.choice(corridors_only) if corridors_only else random.randrange(len(rooms))
            target_room = rooms[target_idx]
            side = random.choice(["N", "S", "E", "W"])
            w = random.uniform(*size_profile["corridor_attached_length_range"])
            h = random.uniform(*size_profile["corridor_width_range"])
            if side in ["E", "W"]:
                w, h = h, w

            overlap = 0.05
            if side == "N":
                nx = round(target_room.x + random.uniform(0, max(0.01, target_room.w - w)), 2)
                ny = target_room.y + target_room.h - overlap
            elif side == "S":
                nx = round(target_room.x + random.uniform(0, max(0.01, target_room.w - w)), 2)
                ny = target_room.y - h + overlap
            elif side == "E":
                nx = target_room.x + target_room.w - overlap
                ny = round(target_room.y + random.uniform(0, max(0.01, target_room.h - h)), 2)
            else:
                nx = target_room.x - w + overlap
                ny = round(target_room.y + random.uniform(0, max(0.01, target_room.h - h)), 2)

            new_room = Room(nx, ny, w, h, name=f"Cor-{i}", type="corridor")
            if can_place_attached_room(new_room, rooms, target_idx):
                rooms.append(new_room)
                edges.append((target_idx, len(rooms) - 1))
                placed = True
                break
        if not placed:
            print(f"[Generate] Warning: corridor {i} could not be placed for seed {seed}")

    num_rooms_to_add = max(0, total_nodes - len(rooms))
    for i in range(num_rooms_to_add):
        placed = False
        for _ in range(100):
            target_idx = random.randrange(len(rooms))
            target_room = rooms[target_idx]
            side = random.choice(["N", "S", "E", "W"])
            w = random.uniform(*size_profile["room_size_range"])
            h = random.uniform(*size_profile["room_size_range"])

            overlap = 0.05
            if side == "N":
                nx = round(target_room.x + random.uniform(0, max(0.01, target_room.w - w)), 2)
                ny = target_room.y + target_room.h - overlap
            elif side == "S":
                nx = round(target_room.x + random.uniform(0, max(0.01, target_room.w - w)), 2)
                ny = target_room.y - h + overlap
            elif side == "E":
                nx = target_room.x + target_room.w - overlap
                ny = round(target_room.y + random.uniform(0, max(0.01, target_room.h - h)), 2)
            else:
                nx = target_room.x - w + overlap
                ny = round(target_room.y + random.uniform(0, max(0.01, target_room.h - h)), 2)

            new_room = Room(nx, ny, w, h, name=f"Room-{i}", type="room")
            if can_place_attached_room(new_room, rooms, target_idx):
                rooms.append(new_room)
                edges.append((target_idx, len(rooms) - 1))
                placed = True
                break
        if not placed:
            print(f"[Generate] Warning: room {i} could not be placed for seed {seed}")

    physical_doors = []
    tol = 0.05
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1, r2 = rooms[i], rooms[j]
            inter = r1.poly.buffer(tol).intersection(r2.poly.buffer(tol))
            if inter.is_empty or inter.area < 0.001:
                continue

            minx, miny, maxx, maxy = inter.bounds
            inter_w = maxx - minx
            inter_h = maxy - miny
            shared_len = max(inter_w, inter_h)
            if shared_len <= 0.5:
                continue

            mid_x = (minx + maxx) / 2
            mid_y = (miny + maxy) / 2
            is_horizontal = inter_w > inter_h
            physical_doors.append(
                {
                    "pos": [float(mid_x), float(mid_y)],
                    "rooms": [r1.name, r2.name],
                    "horizontal": bool(is_horizontal),
                    "door_width": float(door_width),
                }
            )
            if (i, j) not in edges and (j, i) not in edges:
                edges.append((i, j))

    return rooms, edges, physical_doors


def save_graph_preview(rooms, edges, img_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = {}
    n = len(rooms)
    for i in range(n):
        angle = 2 * np.pi * i / max(1, n)
        dist = 0.8 if rooms[i].type == "room" else 0.4
        pos[i] = (dist * np.cos(angle), dist * np.sin(angle))

    for start, end in edges:
        p1, p2 = pos[start], pos[end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#95a5a6", lw=1.2, zorder=1)

    for i, room in enumerate(rooms):
        px, py = pos[i]
        color = "#27ae60" if room.type == "corridor" else "#ecf0f1"
        ax.scatter(px, py, s=700 if room.type == "corridor" else 400, color=color, edgecolors="#2c3e50", lw=2, zorder=2)
        ax.text(px, py, str(i), ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Topological Structure", fontsize=10)
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_visual_preview(rooms, doors, img_path, seed):
    fig, ax = plt.subplots(figsize=(8, 8))
    union = unary_union([r.poly for r in rooms])
    minx, miny, maxx, maxy = union.bounds

    for i, room in enumerate(rooms):
        coords = np.array(room.poly.exterior.coords)
        color = "#c8e6c9" if room.type == "corridor" else "#f5f5f5"
        ax.add_patch(patches.Polygon(coords, fill=True, color=color, ec="black", lw=1.5))
        ax.text(room.x + room.w / 2, room.y + room.h / 2, f"{i}", ha="center", va="center", fontsize=8, fontweight="bold")

    for door in doors:
        px, py = door["pos"]
        door_width = float(door.get("door_width", 1.5))
        if door["horizontal"]:
            ax.plot([px - door_width / 2, px + door_width / 2], [py, py], color="#f1c40f", lw=5, ls="--", zorder=5)
        else:
            ax.plot([px, px], [py - door_width / 2, py + door_width / 2], color="#f1c40f", lw=5, ls="--", zorder=5)

    ax.set_xlim(minx - 2.5, maxx + 2.5)
    ax.set_ylim(miny - 2.5, maxy + 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Procedural HouseGAN Plan (Seed: {seed})", fontsize=10)
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def complexity_to_range(complexity):
    if "3-5" in complexity:
        return (3, 5)
    if "5-8" in complexity:
        return (5, 8)
    if "8-15" in complexity:
        return (8, 15)
    if "15-20" in complexity:
        return (15, 20)
    if "20-30" in complexity:
        return (20, 30)
    return (5, 8)


def resolve_size_profile(config):
    requested_mode = str(config.get("room_area_mode", "default")).strip().lower()
    mode = requested_mode if requested_mode in AREA_PROFILES else "default"
    profile = AREA_PROFILES[mode].copy()
    return mode, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config_housegan.json")))
    args = parser.parse_args()

    config = {
        "num_scenarios": 5,
        "num_corridors": 1,
        "random_seed": 42,
        "door_width": 1.5,
        "complexity": "Medium (5-8 Rooms)",
        "room_area_mode": "default",
        "output_scenario": "Topo_HouseGAN",
    }
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    project_root = Path(__file__).resolve().parents[2]
    output_root = project_root / "Geo_scenario" / config.get("output_scenario", "Topo_HouseGAN") / "geo"
    output_root.mkdir(parents=True, exist_ok=True)

    room_range = complexity_to_range(config["complexity"])
    room_area_mode, size_profile = resolve_size_profile(config)
    config["room_area_mode"] = room_area_mode
    for i in range(int(config["num_scenarios"])):
        current_seed = int(config["random_seed"]) + i
        rooms, edges, doors = generate_procedural_layout(
            room_range,
            int(config["num_corridors"]),
            float(config["door_width"]),
            current_seed,
            size_profile,
        )

        run_name = f"plan_{current_seed}_{uuid.uuid4().hex[:4]}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        graph_data = {
            "nodes": [{"id": i, "name": r.name, "type": r.type, "area_m2": float(r.poly.area)} for i, r in enumerate(rooms)],
            "edges": [{"from": e[0], "to": e[1]} for e in edges],
        }
        metadata = {
            "plan_name": run_name,
            "created_at_unix": time.time(),
            "seed": current_seed,
            "params": config,
            "room_count": sum(1 for r in rooms if r.type == "room"),
            "corridor_count": sum(1 for r in rooms if r.type == "corridor"),
            "door_count": len(doors),
            "output_scenario": config.get("output_scenario", "Topo_HouseGAN"),
        }

        with open(run_dir / "topological_graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        with open(run_dir / "geo_room.json", "w", encoding="utf-8") as f:
            json.dump([r.get_coords() for r in rooms if r.type == "room"], f, indent=2)
        with open(run_dir / "geo_corridor.json", "w", encoding="utf-8") as f:
            json.dump([r.get_coords() for r in rooms if r.type == "corridor"], f, indent=2)
        with open(run_dir / "geo_door.json", "w", encoding="utf-8") as f:
            json.dump(doors, f, indent=2)

        save_visual_preview(rooms, doors, run_dir / "preview.png", current_seed)
        save_graph_preview(rooms, edges, run_dir / "preview_graph.png")
        print(f"[Generate] Created {run_dir}")


if __name__ == "__main__":
    main()

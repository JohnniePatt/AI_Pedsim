import os
import pathlib
import json
import random
import csv
from shapely.geometry import Polygon

# ==========================================
# setup
# ==========================================
BASE_DIR = pathlib.Path(__file__).parent.resolve()

GEO_DIR = BASE_DIR / "geo"
DATASWARM_DIR = BASE_DIR / "dataswarm"
SPAWN_EXIT_ROOT = BASE_DIR / "spawn_exit_area" # เปลี่ยนเป็น Root

# Define geometry paths
GEO_ROOM_FILE = GEO_DIR / "geo_room.json"

def load_polygons_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [Polygon(poly_coords) for poly_coords in data]

if __name__ == "__main__":
    if not GEO_ROOM_FILE.exists():
        print(f"ERROR: Geometry files not found in {GEO_DIR}")
        exit(1)

    print("Loading Global Geometry for Rooms...")
    all_rooms = load_polygons_from_json(GEO_ROOM_FILE)
    
    # 1. ใช้วิธี rglob เพื่อค้นหาไฟล์ .sqlite ในทุกโฟลเดอร์ย่อย (train/test/validation)
    sqlite_files = list(DATASWARM_DIR.rglob("double-botteleneck_*.sqlite"))
    
    if not sqlite_files:
        print(f"No sqlite files found in {DATASWARM_DIR}")
        exit(1)
        
    print(f"Found {len(sqlite_files)} sqlite files. Processing...")

    for sqlite_file in sqlite_files:
        # 2. คำนวณ Relative Path เพื่อหาว่าไฟล์นี้อยู่ในโฟลเดอร์ย่อยไหน (เช่น 'train' หรือ 'test')
        relative_subdir = sqlite_file.parent.relative_to(DATASWARM_DIR)
        
        # 3. สร้าง Target Folder ให้ตรงกันที่ฝั่ง spawn_exit_area
        target_dir = SPAWN_EXIT_ROOT / relative_subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        # ดึงเลข Seed จากชื่อไฟล์
        filename = sqlite_file.stem
        try:
            current_seed = int(filename.split('_')[1])
        except (IndexError, ValueError):
            continue

        # 4. ใช้ Seed เดิมสุ่มแบบเดิม (Reconstruct)
        rand_gen = random.Random(current_seed)
        selected_rooms = rand_gen.sample(all_rooms, k=2)
        spawning_area = selected_rooms[0]
        exit_area = selected_rooms[1]
        
        # 5. บันทึกข้อมูลลง CSV ในโฟลเดอร์ย่อยที่ถูกต้อง
        csv_file = target_dir / f"spawn_exit_{current_seed}.csv"
        
        data = [
            {"type": "spawning_area", "area": spawning_area.wkt},
            {"type": "exit_area", "area": exit_area.wkt}
        ]
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["type", "area"])
            writer.writeheader()
            writer.writerows(data)
            
        print(f"[{relative_subdir}/{current_seed}] Saved {csv_file.name}")

    print("\n=== Spawn/Exit Extraction Completed with Subdirectory support ===")

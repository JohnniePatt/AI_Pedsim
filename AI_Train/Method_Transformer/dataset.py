import os
import json
import random
import pathlib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from tqdm import tqdm

def render_geo_mask(room_json, corridor_json, resolution=0.1, max_size=128):
    """
    Reads Geo_room.json and Geo_corridor.json.
    Computes a bounding box encompassing all points, renders a binary occupancy map
    scaled to the specified max_size for CNN feeding.
    Returns: torch tensor [1, max_size, max_size] (0 = obstacle, 1 = walkable area).
    """
    polygons = []
    
    def load_polys(json_path):
        if not json_path.exists(): return []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return [Polygon(coords) for coords in data if len(coords) >= 3]
        except:
            return []

    rooms = load_polys(room_json)
    corr = load_polys(corridor_json)
    polygons.extend(rooms)
    polygons.extend(corr)

    if not polygons:
        return torch.zeros((1, max_size, max_size))

    # Find bounds
    try:
        min_x = min([p.bounds[0] for p in polygons])
        min_y = min([p.bounds[1] for p in polygons])
        max_x = max([p.bounds[2] for p in polygons])
        max_y = max([p.bounds[3] for p in polygons])
    except:
        return torch.zeros((1, max_size, max_size))
    
    width_m = max(1.0, max_x - min_x)
    height_m = max(1.0, max_y - min_y)
    
    scale_x = max_size / width_m
    scale_y = max_size / height_m
    scale = min(scale_x, scale_y) # preserve aspect ratio
    
    img = Image.new('L', (max_size, max_size), color=0)
    draw = ImageDraw.Draw(img)
    
    tx = (max_size - width_m * scale) / 2
    ty = (max_size - height_m * scale) / 2

    for poly in polygons:
        img_coords = []
        try:
            for x, y in poly.exterior.coords:
                ix = int((x - min_x) * scale + tx)
                iy = int((y - min_y) * scale + ty)
                img_coords.append((ix, iy))
            draw.polygon(img_coords, fill=255)
        except:
            continue
        
    mask = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(mask).unsqueeze(0) # [1, H, W]


class TrajectorySlidingWindowDataset(Dataset):
    """
    Rock-Solid Memory-Efficient Standard Dataset.
    Stores lightweight pointers to numpy arrays instead of thousands of Duplicate Tensors.
    Never hangs, allows full CPU multiprocessing, and uses extremely little RAM.
    """
    def __init__(self, data_dir, config, split="train", shuffle=True):
        super(TrajectorySlidingWindowDataset, self).__init__()
        
        self.split = split
        self.data_dir = pathlib.Path(data_dir) / split
        
        self.obs_len = config.get("obs_len", 8)
        self.pred_len = config.get("pred_len", 12)
        self.seq_len = self.obs_len + self.pred_len
        
        self.col_frame = config.get("col_frame", "Frame")
        self.col_agent = config.get("col_agent", "Agent id")
        self.col_x = config.get("col_x", "position X")
        self.col_y = config.get("col_y", "Y")

        self.samples = []
        self.shared_trajs = [] # Stores numeric matrices
        self._geo_cache = {}
        
        subset_percent = config.get("data_subset_percent", 100)
        
        if not self.data_dir.exists():
            print(f"⚠️ Directory {self.data_dir} does not exist.")
            return

        all_cases = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
        if subset_percent < 100 and len(all_cases) > 0:
            num_keep = max(1, int((subset_percent / 100.0) * len(all_cases)))
            all_cases = random.sample(all_cases, num_keep)
            print(f"📉 Using {subset_percent}% of data: {num_keep} out of {len(all_cases)} cases.")
            
        print(f"📂 Pre-scanning {self.split} cases (Lighting fast RAM-efficient mode)...")
        
        # We process files right away and store lightweight indices!
        for case_dir in tqdm(all_cases):
            self._process_case(case_dir)
            
        if shuffle:
            random.shuffle(self.samples)
            
        print(f"✅ Ready! Found {len(self.samples)} training sequences.")

    def _get_geo_mask(self, case_dir):
        # Most of the time map is identical, use global cache key
        cache_key = "global_scenario_mask"
        if cache_key not in self._geo_cache:
            room_json = case_dir / "Geo_room.json"
            corr_json = case_dir / "Geo_corridor.json"
            self._geo_cache[cache_key] = render_geo_mask(room_json, corr_json)
        return self._geo_cache[cache_key]

    def _process_case(self, case_dir):
        try:
            geo_mask = self._get_geo_mask(case_dir)
            
            parquet_files = list(case_dir.glob("*.parquet"))
            if not parquet_files: return
            
            df = pd.read_parquet(parquet_files[0])
            
            required_cols = {
                'agent': [self.col_agent, 'Agent id', 'id', 'agent_id'],
                'frame': [self.col_frame, 'Frame', 'frame', 'frame_id'],
                'x': [self.col_x, 'position X', 'pos_x', 'X', 'x'],
                'y': [self.col_y, 'Y', 'pos_y', 'Y', 'y']
            }
            
            actual = {}
            for key, candidates in required_cols.items():
                for c in candidates:
                    if c in df.columns:
                        actual[key] = c
                        break
                if key not in actual: return

            for agent_id, group in df.groupby(actual['agent']):
                group = group.sort_values(actual['frame'])
                traj_np = group[[actual['x'], actual['y']]].values.astype(np.float32)
                
                if len(traj_np) < self.seq_len: continue
                
                # Store the single master array in memory
                traj_idx = len(self.shared_trajs)
                self.shared_trajs.append((traj_np, geo_mask))
                
                # Create lightweight windows pointing to start index
                # Stride 2 to reduce redundancy
                for i in range(0, len(traj_np) - self.seq_len + 1, 2):
                    self.samples.append((traj_idx, i))
                    
        except Exception as e:
            # Uncomment if needed, but keeping console clean
            pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Construct Tensor on-the-fly! This is why RAM stays low.
        traj_idx, start_frame = self.samples[idx]
        traj_np, geo_mask = self.shared_trajs[traj_idx]
        
        window = traj_np[start_frame : start_frame + self.seq_len]
        obs = window[:self.obs_len]
        pred = window[self.obs_len:]
        
        return {
            "obs_traj": torch.from_numpy(obs).float(),
            "pred_traj": torch.from_numpy(pred).float(),
            "start_pt": torch.from_numpy(obs[0]).float(),
            "end_pt": torch.from_numpy(traj_np[-1]).float(),
            "geo_mask": geo_mask
        }

if __name__ == "__main__":
    test_config = {"obs_len": 20, "pred_len": 10, "data_subset_percent": 10, "col_frame": "frame", "col_agent": "id", "col_x": "pos_x", "col_y": "pos_y"}
    path = pathlib.Path("../../Dataset_Traj_Table/Topo_bottleneck")
    if path.exists():
        ds = TrajectorySlidingWindowDataset(str(path), test_config, split="train")
        print(f"Total sequences loaded: {len(ds)}")

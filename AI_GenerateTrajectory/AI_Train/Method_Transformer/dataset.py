"""
dataset.py
──────────
PedestrianDataset — loads ALL agents from every simulation.

Social context: for each ego-agent, the K nearest neighbours'
observed trajectories (first obs_len steps) are stored alongside
the ego's own data.  During training the model sees:

  [geo+goal token]  [neighbour_1 token] … [neighbour_K token]
  [ego obs frames]  →  predict [ego future frames]

At inference time the neighbour observations are known (we observed
them), so there is no leakage.

Expected folder layout
──────────────────────
Dataset/Data_Traj_Table/Topo_bottleneck/{train|test|val}/case_{id}/
  Geo_room.json
  Geo_corridor.json
  Spawn_location_{id}.csv   – per-agent spawn positions (100 agents)
  Spawn_exit_{id}.csv       – spawning_area + exit_area WKT polygons
  *_trajectory_data.parquet – ALL agents' trajectories for this sim
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from shapely import wkt as shapely_wkt
from torch.utils.data import Dataset
from tqdm import tqdm

from prepare_geometry_transformer import create_occupancy_grid, world_to_grid


class PedestrianDataset(Dataset):
    """
    Full-path pedestrian trajectory prediction dataset with social context.

    Parameters
    ----------
    data_dir       : root containing train / test / val sub-folders
    split          : "train" | "test" | "val"
    obs_len        : seed frames given to the model (observed steps)
    frame_stride   : keep every N-th frame (reduces 25 fps → ~3 fps for stride=8)
    max_seq_len    : maximum total frames after striding (obs + pred)
    grid_size      : occupancy-grid resolution
    geo_padding    : padding (m) added around the bounding box
    max_neighbors  : K nearest neighbours to include as social context
    subset_percent : use only this % of cases (useful for quick tests)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        obs_len: int = 5,
        frame_stride: int = 8,
        max_seq_len: int = 512,
        grid_size: int = 64,
        geo_padding: float = 1.0,
        max_neighbors: int = 10,
        subset_percent: float = 100.0,
    ):
        self.obs_len       = obs_len
        self.frame_stride  = frame_stride
        self.max_seq_len   = max_seq_len
        self.grid_size     = grid_size
        self.geo_padding   = geo_padding
        self.max_neighbors = max_neighbors

        split_dir = os.path.join(data_dir, split)
        case_dirs = sorted(glob.glob(os.path.join(split_dir, "case_*")))

        n = max(1, int(len(case_dirs) * subset_percent / 100.0))
        case_dirs = case_dirs[:n]

        if not case_dirs:
            print(f"[Dataset] WARNING: no case_* folders in {split_dir}")
            self.samples = []
            return

        print(f"[Dataset] Loading [{split}] – {len(case_dirs)} cases …")

        self._geo_cache: dict = {}   # cache occupancy grids per geometry
        self.samples: list   = []

        for case_dir in tqdm(case_dirs, desc=f"Building [{split}]"):
            try:
                self._load_case(case_dir)
            except Exception as exc:
                print(f"  [skip] {os.path.basename(case_dir)}: {exc}")

        print(
            f"[Dataset] [{split}] ready — "
            f"{len(self.samples)} agent samples "
            f"from {len(case_dirs)} simulations."
        )

    # ── internal loader ───────────────────────────────────────────────────────

    def _load_case(self, case_dir: str):
        case_id = os.path.basename(case_dir).replace("case_", "")

        # 1. Occupancy grid (cached per unique JSON content)
        room_json     = os.path.join(case_dir, "Geo_room.json")
        corridor_json = os.path.join(case_dir, "Geo_corridor.json")
        cache_key     = f"{room_json}|{corridor_json}"

        if cache_key not in self._geo_cache:
            grid, meta = create_occupancy_grid(
                room_json, corridor_json,
                grid_size=self.grid_size,
                padding=self.geo_padding,
            )
            self._geo_cache[cache_key] = (grid, meta)

        grid, meta = self._geo_cache[cache_key]
        geo_mask = torch.from_numpy(grid).unsqueeze(0)          # [1, H, W]

        # 2. Exit centroid → goal point (normalised)
        exit_csv  = os.path.join(case_dir, f"Spawn_exit_{case_id}.csv")
        exit_df   = pd.read_csv(exit_csv)
        exit_row  = exit_df[exit_df["type"] == "exit_area"].iloc[0]
        exit_poly = shapely_wkt.loads(exit_row["area"])
        ne_x, ne_y = world_to_grid(exit_poly.centroid.x, exit_poly.centroid.y, meta)
        end_pt = torch.tensor([ne_x, ne_y], dtype=torch.float32)

        # 3. Spawn positions (one row per agent)
        spawn_csv = os.path.join(case_dir, f"Spawn_location_{case_id}.csv")
        spawn_df  = pd.read_csv(spawn_csv)

        # 4. Trajectory parquet (ALL agents in this simulation)
        parquet_files = glob.glob(os.path.join(case_dir, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError("no .parquet found in case folder")

        traj_df = pd.read_parquet(parquet_files[0])

        id_col = "id"    if "id"    in traj_df.columns else "AgentID"
        fr_col = "frame" if "frame" in traj_df.columns else "Frame"
        x_col  = "pos_x" if "pos_x" in traj_df.columns else "x"
        y_col  = "pos_y" if "pos_y" in traj_df.columns else "y"

        # 5. Pre-process every agent's trajectory (strided + normalised)
        #    Stored as a lookup dict for fast neighbour retrieval.
        all_norm: dict[int, np.ndarray] = {}   # agent_id → [T, 2] norm coords

        for agent_id, grp in traj_df.groupby(id_col):
            raw = grp.sort_values(fr_col)[[x_col, y_col]].values
            raw = raw[:: self.frame_stride]
            if len(raw) > self.max_seq_len:
                raw = raw[: self.max_seq_len]
            norm = np.stack(
                [
                    (raw[:, 0] - meta["min_x"]) / meta["scale"],
                    (raw[:, 1] - meta["min_y"]) / meta["scale"],
                ],
                axis=1,
            ).astype(np.float32)
            all_norm[int(agent_id)] = norm

        # 6. Build one sample per agent
        for _, spawn_row in spawn_df.iterrows():
            agent_id = int(spawn_row["id"])
            if agent_id not in all_norm:
                continue

            norm = all_norm[agent_id]
            if len(norm) <= self.obs_len:
                continue

            ns_x, ns_y = world_to_grid(spawn_row["pos_x"], spawn_row["pos_y"], meta)
            start_pt = torch.tensor([ns_x, ns_y], dtype=torch.float32)

            obs_traj  = torch.from_numpy(norm[: self.obs_len])      # [obs_len, 2]
            pred_traj = torch.from_numpy(norm[self.obs_len :])       # [variable, 2]

            # 7. Social context: K nearest neighbours (by distance at t=0)
            neigh_trajs, neigh_mask = self._get_neighbors(
                agent_id, norm[0], all_norm
            )

            self.samples.append(
                {
                    "obs_traj":      obs_traj,          # [obs_len, 2]
                    "pred_traj":     pred_traj,          # [variable, 2]
                    "start_pt":      start_pt,           # [2]
                    "end_pt":        end_pt,             # [2]
                    "geo_mask":      geo_mask,           # [1, H, W]
                    "neighbor_trajs": neigh_trajs,       # [K, obs_len, 2]
                    "neighbor_mask":  neigh_mask,        # [K]  bool
                    "case_id":        case_id,
                    "agent_id":       agent_id,
                    "meta":           meta,
                    "case_dir":       case_dir,
                }
            )

    def _get_neighbors(
        self,
        ego_id: int,
        ego_pos_norm: np.ndarray,       # [2]  ego position at t=0 (normalised)
        all_norm: dict[int, np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (neighbor_trajs [K, obs_len, 2], neighbor_mask [K] bool).
        Neighbours are the K agents closest to the ego at t=0.
        Relative positions: neighbour pos - ego pos (translation invariant).
        Empty slots are zero-padded, mask=False.
        """
        K        = self.max_neighbors
        obs_len  = self.obs_len

        dists = []
        for aid, traj in all_norm.items():
            if aid == ego_id:
                continue
            if len(traj) < obs_len:
                continue
            d = float(np.linalg.norm(traj[0] - ego_pos_norm))
            dists.append((d, aid))

        dists.sort(key=lambda x: x[0])
        top_k = dists[:K]

        neigh_np   = np.zeros((K, obs_len, 2), dtype=np.float32)
        mask_np    = np.zeros(K, dtype=bool)

        for i, (_, aid) in enumerate(top_k):
            traj = all_norm[aid][:obs_len]          # [obs_len, 2]
            # Use relative position so model is translation-invariant
            neigh_np[i] = traj - ego_pos_norm       # relative to ego's t=0
            mask_np[i]  = True

        return (
            torch.from_numpy(neigh_np),
            torch.from_numpy(mask_np),
        )

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ─── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    Stack fixed-size tensors; pad variable-length pred_traj.
    Includes social context tensors (neighbor_trajs, neighbor_mask).
    """
    obs    = torch.stack([b["obs_traj"]      for b in batch])   # [B, obs_len, 2]
    starts = torch.stack([b["start_pt"]      for b in batch])   # [B, 2]
    ends   = torch.stack([b["end_pt"]        for b in batch])   # [B, 2]
    geos   = torch.stack([b["geo_mask"]      for b in batch])   # [B, 1, H, W]
    neighs = torch.stack([b["neighbor_trajs"] for b in batch])  # [B, K, obs_len, 2]
    nmask  = torch.stack([b["neighbor_mask"]  for b in batch])  # [B, K]

    preds   = [b["pred_traj"] for b in batch]
    lengths = torch.tensor([len(p) for p in preds], dtype=torch.long)
    max_len = int(lengths.max().item())
    B       = len(batch)
    padded  = torch.zeros(B, max_len, 2, dtype=torch.float32)
    for i, p in enumerate(preds):
        padded[i, : len(p)] = p

    return {
        "obs_traj":       obs,      # [B, obs_len, 2]
        "pred_traj":      padded,   # [B, max_pred, 2]
        "lengths":        lengths,  # [B]
        "start_pt":       starts,   # [B, 2]
        "end_pt":         ends,     # [B, 2]
        "geo_mask":       geos,     # [B, 1, H, W]
        "neighbor_trajs": neighs,   # [B, K, obs_len, 2]
        "neighbor_mask":  nmask,    # [B, K]  True=real, False=padded
    }

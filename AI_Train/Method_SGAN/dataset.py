import os
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
import pathlib
import random

class TrajectoryDataset(IterableDataset):
    """
    Dataloader for the Trajectory datasets saved in Parquet format.
    """
    def __init__(self, data_dir, config, split="train", shuffle=True):
        super(TrajectoryDataset, self).__init__()
        
        self.split = split
        self.shuffle = shuffle
        
        self.data_dir = pathlib.Path(data_dir) / split
        self.obs_len = config.get("obs_len", 8)
        self.pred_len = config.get("pred_len", 12)
        self.skip = config.get("skip", 1)
        self.seq_len = self.obs_len + self.pred_len
        
        self.col_frame = config.get("col_frame", "frame_id")
        self.col_agent = config.get("col_agent", "id")
        self.col_x = config.get("col_x", "pos_x")
        self.col_y = config.get("col_y", "pos_y")

        self.all_files = list(self.data_dir.glob("*.parquet"))
        if len(self.all_files) == 0:
            print(f"No parquet files found in {self.data_dir}")
            
        print(f"Found {len(self.all_files)} files from {split}. Will stream them on-the-fly.")
        
        # สร้างโฟลเดอร์ Cache เพื่อเก็บข้อมูลที่ประมวลผลแล้ว
        self.cache_dir = self.data_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _extract_from_file(self, df):
        """Extract valid scenes with pre-calculated relative trajectories. Returns a list of (obs, pred, obs_rel, pred_rel)."""
        samples = []
        if self.col_frame not in df.columns or self.col_agent not in df.columns:
            return samples
            
        df = df.drop_duplicates(subset=[self.col_frame, self.col_agent])
        pivot_x = df.pivot(index=self.col_frame, columns=self.col_agent, values=self.col_x)
        pivot_y = df.pivot(index=self.col_frame, columns=self.col_agent, values=self.col_y)
        
        frames = pivot_x.index.values
        x_mat = pivot_x.values
        y_mat = pivot_y.values
        
        num_frames = len(frames)
        window_size = self.seq_len * self.skip
        
        for i in range(0, num_frames - window_size + 1, self.skip):
            x_window = x_mat[i : i + window_size : self.skip, :] 
            y_window = y_mat[i : i + window_size : self.skip, :] 
            
            valid_mask = ~np.isnan(x_window).any(axis=0)
            if not np.any(valid_mask):
                continue
                
            # Slices: (seq_len, num_valid) -> Transpose to (num_valid, seq_len)
            vx = x_window[:, valid_mask].T
            vy = y_window[:, valid_mask].T
            
            # Combine into (num_valid, seq_len, 2)
            traj = np.stack((vx, vy), axis=-1)
            traj_torch = torch.from_numpy(traj).float()
            
            # Vectorized relative trajectory calculation
            rel_traj = torch.zeros_like(traj_torch)
            rel_traj[:, 1:, :] = traj_torch[:, 1:, :] - traj_torch[:, :-1, :]
            
            obs_traj = traj_torch[:, :self.obs_len, :]
            pred_traj = traj_torch[:, self.obs_len:, :]
            obs_rel_traj = rel_traj[:, :self.obs_len, :]
            pred_rel_traj = rel_traj[:, self.obs_len:, :]
            
            samples.append((obs_traj, pred_traj, obs_rel_traj, pred_rel_traj))
            
        return samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            files_to_process = self.all_files
        else:
            files_to_process = [
                f for i, f in enumerate(self.all_files) 
                if i % worker_info.num_workers == worker_info.id
            ]
            
        if self.shuffle:
            random.shuffle(files_to_process)
            
        buffer = []
        buffer_size = 100  # เพิ่มขนาด buffer สำหรับช่วยการ shuffle ระหว่างหลายไฟล์
        
        for path in files_to_process:
            cache_file = self.cache_dir / (path.stem + ".pt")
            
            try:
                # 🛠️ ตรวจสอบว่ามี Cache หรือไม่
                if cache_file.exists():
                    file_samples = torch.load(cache_file)
                else:
                    # ถ้าไม่มีค่อยอ่าน Parquet และเซฟเกึบไว้
                    df = pd.read_parquet(path)
                    file_samples = self._extract_from_file(df)
                    torch.save(file_samples, cache_file)
                
                for sample in file_samples:
                    # sample = (obs, pred, obs_rel, pred_rel)
                    # We add a '1' at the end to signify one chunk of data from a file is being sent
                    # This is a bit hacky but allows tracking file progress without complicated IPC
                    if self.shuffle:
                        buffer.append(sample)
                        if len(buffer) >= buffer_size:
                            idx = random.randint(0, len(buffer) - 1)
                            yield buffer.pop(idx)
                    else:
                        yield sample
            except Exception:
                continue
                
        if self.shuffle:
            random.shuffle(buffer)
            for sample in buffer:
                yield sample

def seq_collate(data):
    """
    Custom collate function for SGAN to handle variable number of pedestrians per scene.
    Packs data from multiple scenes into a single batch dimension.
    """
    obs_traj_list, pred_traj_list, obs_rel_traj_list, pred_rel_traj_list = zip(*data)
    
    # Concatenate along the 'num_peds' dimension (dim=0 here) 
    # to form a big batch of shape (total_peds_in_batch, seq_len, 2)
    obs_traj = torch.cat(obs_traj_list, dim=0)
    pred_traj = torch.cat(pred_traj_list, dim=0)
    obs_rel_traj = torch.cat(obs_rel_traj_list, dim=0)
    pred_rel_traj = torch.cat(pred_rel_traj_list, dim=0)
    
    # seq_start_end tells us which pedestrians belong to which scene
    seq_start_end = []
    start = 0
    for obs in obs_traj_list:
        end = start + obs.shape[0]
        seq_start_end.append((start, end))
        start = end
        
    # SGAN expects input shape (seq_len, batch_size, 2)
    obs_traj = obs_traj.permute(1, 0, 2)
    pred_traj = pred_traj.permute(1, 0, 2)
    obs_rel_traj = obs_rel_traj.permute(1, 0, 2)
    pred_rel_traj = pred_rel_traj.permute(1, 0, 2)
    
    return obs_traj, pred_traj, obs_rel_traj, pred_rel_traj, seq_start_end

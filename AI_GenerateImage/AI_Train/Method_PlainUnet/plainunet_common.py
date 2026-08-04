import json
import math
import pathlib
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def resolve_project_root(script_dir: pathlib.Path) -> pathlib.Path:
    return script_dir.parent.parent

def resolve_path(path_value, project_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute(): return path
    return (project_root / path).resolve()

def make_run_dirs(base_dir: pathlib.Path) -> dict:
    project_root = resolve_project_root(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_PlainUNet_{timestamp}"
    current_run_dir = project_root / "AI_Result" / "Method_PlainUnet" / "outputs" / run_name
    paths = {
        "PROJECT_ROOT": project_root,
        "CURRENT_RUN_DIR": current_run_dir,
        "CHECKPOINT_DIR": current_run_dir / "checkpoints",
        "TEST_RESULT_DIR": current_run_dir / "test_results",
        "LOG_DIR": current_run_dir / "logs",
        "SAMPLE_DIR": current_run_dir / "samples",
    }
    for key in ("CHECKPOINT_DIR", "TEST_RESULT_DIR", "LOG_DIR", "SAMPLE_DIR"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths

class PlainUNetDataset(Dataset):
    def __init__(self, dataset_root, split, image_size):
        self.dataset_root = pathlib.Path(dataset_root)
        self.split = split
        self.image_size = int(image_size)
        a_dir, b_dir = self.dataset_root / "A" / split, self.dataset_root / "B" / split
        if not a_dir.exists() or not b_dir.exists():
            raise FileNotFoundError(f"Missing A or B in {dataset_root}/{split}")
        a_names = {p.name for p in a_dir.glob("*.png")}
        b_names = {p.name for p in b_dir.glob("*.png")}
        self.names = sorted(a_names & b_names)

    def __len__(self): return len(self.names)
    def __getitem__(self, index):
        name = self.names[index]
        a = Image.open(self.dataset_root / "A" / self.split / name).convert("RGB").resize((self.image_size, self.image_size), BILINEAR)
        b = Image.open(self.dataset_root / "B" / self.split / name).convert("L").resize((self.image_size, self.image_size), BILINEAR)
        a_arr = np.asarray(a, dtype=np.float32) / 255.0
        b_arr = np.asarray(b, dtype=np.float32) / 255.0
        return torch.from_numpy(a_arr).permute(2, 0, 1).contiguous(), torch.from_numpy(b_arr[None, ...]).contiguous(), name

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, drop=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        ]
        if drop > 0: layers.append(nn.Dropout2d(drop))
        self.b = nn.Sequential(*layers)
    def forward(self, x): return self.b(x)

class PlainUNet(nn.Module):
    def __init__(self, base=32, drop=0.1):
        super().__init__()
        self.c1 = ConvBlock(3, base, 0.0)
        self.c2 = ConvBlock(base, base * 2, drop)
        self.c3 = ConvBlock(base * 2, base * 4, drop)
        self.c4 = ConvBlock(base * 4, base * 8, drop)
        self.bn = ConvBlock(base * 8, base * 16, drop)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.u4 = ConvBlock(base * 16 + base * 8, base * 8, drop)
        self.u3 = ConvBlock(base * 8 + base * 4, base * 4, drop)
        self.u2 = ConvBlock(base * 4 + base * 2, base * 2, drop)
        self.u1 = ConvBlock(base * 2 + base, base, 0.0)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        bn = self.bn(self.pool(c4))
        x = self.u4(torch.cat([self.up(bn), c4], dim=1))
        x = self.u3(torch.cat([self.up(x), c3], dim=1))
        x = self.u2(torch.cat([self.up(x), c2], dim=1))
        x = self.u1(torch.cat([self.up(x), c1], dim=1))
        return self.out(x) # Output is raw logits, use sigmoid for density

def tensor_density_metrics(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, dtype=np.float32), 0.0, 1.0)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))
    mu_x, mu_y = float(np.mean(y_true)), float(np.mean(y_pred))
    var_x, var_y = float(np.var(y_true)), float(np.var(y_pred))
    cov_xy = float(np.mean((y_true - mu_x) * (y_pred - mu_y)))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return {"mae": mae, "mse": mse, "rmse": rmse, "psnr": psnr, "ssim": float(ssim)}

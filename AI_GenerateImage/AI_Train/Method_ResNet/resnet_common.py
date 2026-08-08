import os
import json
import math
import pathlib
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def resolve_project_root(script_dir: pathlib.Path) -> pathlib.Path:
    return script_dir.parents[2]

def resolve_path(path_value: str | pathlib.Path, project_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

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

def convert_bw_to_colorjet(bw_tensor_or_array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(bw_tensor_or_array, torch.Tensor):
        bw_data = bw_tensor_or_array.detach().cpu().numpy()
    else:
        bw_data = np.array(bw_tensor_or_array)

    if bw_data.ndim == 3 and bw_data.shape[0] in (1, 3):
        bw_data = bw_data[0]

    gray_norm = np.clip(bw_data, 0.0, 1.0).astype(np.float32)
    x = gray_norm
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).clip(0, 255).astype(np.uint8)
    return rgb

def make_run_dirs(script_dir: pathlib.Path, method_name: str = "Method_ResNet") -> dict[str, pathlib.Path]:
    project_root = resolve_project_root(script_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_ResNet_{timestamp}"
    runs_root = project_root / "AI_GenerateImage" / "AI_Result" / method_name / "outputs"
    current_run_dir = runs_root / run_name
    paths = {
        "PROJECT_ROOT": project_root,
        "RUNS_ROOT": runs_root,
        "CURRENT_RUN_DIR": current_run_dir,
        "CHECKPOINT_DIR": current_run_dir / "checkpoints",
        "LOG_DIR": current_run_dir / "logs",
        "SAMPLE_DIR": current_run_dir / "samples",
        "TEST_RESULT_DIR": current_run_dir / "test_results",
        "FINAL_EVALUATION_DIR": current_run_dir / "final_evaluation",
    }
    for key in ("CHECKPOINT_DIR", "LOG_DIR", "SAMPLE_DIR", "TEST_RESULT_DIR", "FINAL_EVALUATION_DIR"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths

# --- ResNet Block & ResNet Generator Architecture (9 Residual Blocks) ---

class ResNetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1, num_resnet_blocks: int = 9, ngf: int = 64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # 3 Downsampling Layers
        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # 9 ResNet Residual Blocks Bottleneck
        mult = 2 ** n_downsampling
        for i in range(num_resnet_blocks):
            model += [ResNetBlock(ngf * mult)]

        # 3 Upsampling Layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, kernel_size=7, padding=0),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- Dataset Loader ---

class ResNetDataset(Dataset):
    def __init__(self, dataset_root: pathlib.Path, split: str = "train", image_size: int = 256, target_channels: int = 1):
        self.dataset_root = pathlib.Path(dataset_root)
        if split == "val" and not (self.dataset_root / "A" / "val").exists() and (self.dataset_root / "A" / "validation").exists():
            split = "validation"
        self.split = split
        self.image_size = image_size
        self.target_channels = target_channels

        self.dir_a = self.dataset_root / "A" / split
        self.dir_b = self.dataset_root / "B" / split

        if not self.dir_a.exists() or not self.dir_b.exists():
            raise FileNotFoundError(f"Dataset directory missing: {self.dir_a} or {self.dir_b}")

        self.filenames = sorted([
            f for f in os.listdir(self.dir_a)
            if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.filenames[idx]
        path_a = self.dir_a / fname
        path_b = self.dir_b / fname

        img_a = Image.open(path_a).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        img_b = Image.open(path_b).convert("L" if self.target_channels == 1 else "RGB").resize((self.image_size, self.image_size), Image.BILINEAR)

        arr_a = np.array(img_a, dtype=np.float32) / 255.0
        arr_b = np.array(img_b, dtype=np.float32) / 255.0

        tensor_a = torch.from_numpy(arr_a).permute(2, 0, 1)
        if self.target_channels == 1:
            tensor_b = torch.from_numpy(arr_b).unsqueeze(0)
        else:
            tensor_b = torch.from_numpy(arr_b).permute(2, 0, 1)

        return tensor_a, tensor_b, fname

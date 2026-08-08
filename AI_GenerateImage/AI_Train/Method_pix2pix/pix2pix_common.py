import os
import json
import pathlib
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def resolve_project_root(script_dir: pathlib.Path) -> pathlib.Path:
    return script_dir.parent.parent

def resolve_path(path_value: str | pathlib.Path, project_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

def tensor_density_metrics(y_true, y_pred):
    import math
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

def make_run_dirs(script_dir: pathlib.Path, method_name: str = "Method_pix2pix") -> dict[str, pathlib.Path]:
    project_root = resolve_project_root(script_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_pix2pix_{timestamp}"
    runs_root = project_root / "AI_Result" / method_name / "outputs"
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

# --- U-Net Generator for Pix2Pix ---
class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_dropout: bool = False):
        super().__init__()
        if down:
            self.model = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        # Encoder (Downsampling)
        self.d1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.d2 = UNetBlock(64, 128, down=True)
        self.d3 = UNetBlock(128, 256, down=True)
        self.d4 = UNetBlock(256, 512, down=True)
        self.d5 = UNetBlock(512, 512, down=True)
        self.d6 = UNetBlock(512, 512, down=True)
        self.d7 = UNetBlock(512, 512, down=True)
        self.d8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU(inplace=True))

        # Decoder (Upsampling with skip connections)
        self.u1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.u2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.u3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.u4 = UNetBlock(1024, 512, down=False)
        self.u5 = UNetBlock(1024, 256, down=False)
        self.u6 = UNetBlock(512, 128, down=False)
        self.u7 = UNetBlock(256, 64, down=False)
        self.u8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))
        return self.u8(torch.cat([u7, d1], dim=1))

# --- PatchGAN Discriminator (70x70) ---
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 4): # Input A (3) + Target B (1) = 4
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, input_a: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_a, target_b], dim=1)
        return self.model(x)

# --- Dataset ---
class Pix2PixDataset(Dataset):
    def __init__(self, dataset_root: str | pathlib.Path, subset: str = "train", image_size: int = 256, target_channels: int = 1):
        self.dir_A = pathlib.Path(dataset_root) / "A" / subset
        self.dir_B = pathlib.Path(dataset_root) / "B" / subset
        self.image_size = image_size
        self.target_channels = target_channels
        if not self.dir_A.exists() or not self.dir_B.exists():
            raise FileNotFoundError(f"Dataset subset directories missing: {self.dir_A} or {self.dir_B}")
        a_files = {f.name for f in self.dir_A.glob("*.png")}
        b_files = {f.name for f in self.dir_B.glob("*.png")}
        self.files = sorted(list(a_files & b_files))

        self.transform_a = transforms.Compose([
            transforms.Resize((image_size, image_size), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        
        self.transform_b = transforms.Compose([
            transforms.Resize((image_size, image_size), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_name = self.files[idx]
        img_a = Image.open(self.dir_A / file_name).convert("RGB")
        if self.target_channels == 1:
            img_b = Image.open(self.dir_B / file_name).convert("L")
        else:
            img_b = Image.open(self.dir_B / file_name).convert("RGB")
            
        tensor_a = self.transform_a(img_a)
        tensor_b = self.transform_b(img_b)
        return tensor_a, tensor_b, file_name

def convert_bw_to_colorjet(bw_data: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Converts a 1-channel Grayscale density map (in range 0.0 to 1.0 or 0 to 255)
    into a 3-channel RGB COLORJET image (numpy array uint8, shape [H, W, 3]).
    """
    import cv2
    if isinstance(bw_data, torch.Tensor):
        bw_arr = bw_data.detach().cpu().numpy()
    else:
        bw_arr = np.array(bw_data)

    if bw_arr.ndim == 3 and bw_arr.shape[0] in (1, 3):
        bw_arr = bw_arr[0]
    elif bw_arr.ndim == 3 and bw_arr.shape[2] in (1, 3):
        bw_arr = bw_arr[..., 0]

    if bw_arr.dtype in (np.float32, np.float64):
        bw_uint8 = (np.clip(bw_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        bw_uint8 = np.clip(bw_arr, 0, 255).astype(np.uint8)

    color_bgr = cv2.applyColorMap(bw_uint8, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return color_rgb


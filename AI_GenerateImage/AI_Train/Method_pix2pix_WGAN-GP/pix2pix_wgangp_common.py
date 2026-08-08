import os
import math
import json
import pathlib
import datetime
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def resolve_project_root(script_path: pathlib.Path) -> pathlib.Path:
    current = script_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "Dataset").exists() and (parent / "AI_GenerateImage").exists():
            return parent
    return current.parents[2]

def resolve_path(path_str: str, project_root: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def make_run_dirs(script_dir: pathlib.Path, method_name: str = "Method_pix2pix_WGAN-GP") -> dict[str, pathlib.Path]:
    project_root = resolve_project_root(script_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_pix2pix_wgangp_{timestamp}"
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

class Pix2PixDataset(Dataset):
    def __init__(self, dataset_root: pathlib.Path, split: str = "train", image_size: int = 512):
        self.dataset_root = dataset_root
        self.split = "validation" if split in ("val", "validation") else split
        self.image_size = image_size
        
        self.dir_a = self.dataset_root / "A" / self.split
        self.dir_b = self.dataset_root / "B" / self.split
        
        if not self.dir_a.exists():
            raise FileNotFoundError(f"Input dir A not found: {self.dir_a}")
        if not self.dir_b.exists():
            raise FileNotFoundError(f"Target dir B not found: {self.dir_b}")
            
        self.filenames = sorted([f.name for f in self.dir_a.glob("*.png") if (self.dir_b / f.name).exists()])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        path_a = self.dir_a / fname
        path_b = self.dir_b / fname
        
        img_a = Image.open(path_a).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        img_b = Image.open(path_b).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        
        arr_a = np.array(img_a, dtype=np.float32) / 255.0
        arr_b = np.array(img_b, dtype=np.float32) / 255.0
        
        # Pix2Pix Generator normalization [-1, 1] for A
        tensor_a = torch.from_numpy(arr_a).permute(2, 0, 1) * 2.0 - 1.0
        # Target B in [0, 1]
        tensor_b = torch.from_numpy(arr_b).unsqueeze(0)
        
        return tensor_a, tensor_b, fname

# UNet 256 Skip Connection Generator
class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.InstanceNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.InstanceNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, num_downs=8, ngf=64, use_dropout=False):
        super().__init__()
        # Construct UNet structure
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, use_dropout=use_dropout)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block)
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

# PatchGAN Discriminator with InstanceNorm (WGAN-GP compliant)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=4, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(netD, real_samples, fake_samples, input_a, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_inputs = torch.cat([input_a, interpolates], dim=1)
    d_interpolates = netD(d_inputs)
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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

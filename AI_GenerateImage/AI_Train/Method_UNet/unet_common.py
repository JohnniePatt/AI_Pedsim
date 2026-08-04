import csv
import json
import pathlib
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


def resolve_project_root(script_dir: pathlib.Path) -> pathlib.Path:
    return script_dir.parent.parent


def resolve_path(path_value: str | pathlib.Path, project_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_json_config(config_path: pathlib.Path) -> dict:
    if not config_path or not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_run_dirs(base_dir: pathlib.Path, method_name: str = "Method_UNet") -> dict[str, pathlib.Path]:
    project_root = resolve_project_root(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_UNet_{timestamp}"
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
    }
    for key in ("CHECKPOINT_DIR", "LOG_DIR", "SAMPLE_DIR", "TEST_RESULT_DIR"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def list_paired_files(dataset_root: pathlib.Path, split: str) -> list[str]:
    a_dir = dataset_root / "A" / split
    b_dir = dataset_root / "B" / split
    if not a_dir.exists() or not b_dir.exists():
        raise FileNotFoundError(f"Expected A/{split} and B/{split} under {dataset_root}")
    a_names = {p.name for p in a_dir.glob("*.png")}
    b_names = {p.name for p in b_dir.glob("*.png")}
    names = sorted(a_names & b_names)
    if not names:
        raise RuntimeError(f"No paired PNG files found for split='{split}' under {dataset_root}")
    return names


class TrajectoryMaskDataset(Dataset):
    def __init__(self, dataset_root: pathlib.Path, split: str, image_size: int):
        self.dataset_root = pathlib.Path(dataset_root)
        self.split = split
        self.image_size = int(image_size)
        self.names = list_paired_files(self.dataset_root, split)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        a_path = self.dataset_root / "A" / self.split / name
        b_path = self.dataset_root / "B" / self.split / name
        a = Image.open(a_path).convert("RGB").resize((self.image_size, self.image_size), BILINEAR)
        b = Image.open(b_path).convert("L").resize((self.image_size, self.image_size), NEAREST)

        a_arr = np.asarray(a, dtype=np.float32) / 255.0
        b_arr = (np.asarray(b, dtype=np.float32) >= 128).astype(np.float32)
        a_tensor = torch.from_numpy(a_arr).permute(2, 0, 1).contiguous()
        b_tensor = torch.from_numpy(b_arr[None, ...]).contiguous()
        return a_tensor, b_tensor, name


def make_loader(dataset_root: pathlib.Path, split: str, image_size: int, batch_size: int, shuffle: bool, num_workers: int = 0):
    dataset = TrajectoryMaskDataset(dataset_root, split, image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(int(num_workers), 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, dataset.names


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TrajectoryMaskUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_filters: int = 32, dropout: float = 0.1):
        super().__init__()
        f = int(base_filters)
        self.c1 = ConvBlock(in_channels, f, 0.0)
        self.c2 = ConvBlock(f, f * 2, dropout)
        self.c3 = ConvBlock(f * 2, f * 4, dropout)
        self.c4 = ConvBlock(f * 4, f * 8, dropout)
        self.bn = ConvBlock(f * 8, f * 16, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.u4 = ConvBlock(f * 16 + f * 8, f * 8, dropout)
        self.u3 = ConvBlock(f * 8 + f * 4, f * 4, dropout)
        self.u2 = ConvBlock(f * 4 + f * 2, f * 2, dropout)
        self.u1 = ConvBlock(f * 2 + f, f, 0.0)
        self.out = nn.Conv2d(f, 1, 1)

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
        return self.out(x)


def build_unet(base_filters: int = 32, dropout: float = 0.1) -> nn.Module:
    return TrajectoryMaskUNet(base_filters=base_filters, dropout=dropout)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_from_probs(y_true: torch.Tensor, y_prob: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    y_true = y_true.float().reshape(y_true.shape[0], -1)
    y_prob = y_prob.float().reshape(y_prob.shape[0], -1)
    intersection = (y_true * y_prob).sum(dim=1)
    denom = y_true.sum(dim=1) + y_prob.sum(dim=1)
    return ((2.0 * intersection + smooth) / (denom + smooth)).mean()


def iou_from_probs(y_true: torch.Tensor, y_prob: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    y_true = y_true.float().reshape(y_true.shape[0], -1)
    y_prob = y_prob.float().reshape(y_prob.shape[0], -1)
    intersection = (y_true * y_prob).sum(dim=1)
    union = y_true.sum(dim=1) + y_prob.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


def make_loss_fn(
    bce_weight: float,
    dice_weight: float,
    foreground_weight: float,
    device: torch.device,
    focal_gamma: float = 0.0,
):
    pos_weight = torch.tensor([float(foreground_weight)], dtype=torch.float32, device=device)

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        if float(focal_gamma) > 0:
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            bce_map = ((1.0 - p_t).clamp(min=1e-6) ** float(focal_gamma)) * bce_map
        dice_loss = 1.0 - dice_from_probs(targets, probs)
        return float(bce_weight) * bce_map.mean() + float(dice_weight) * dice_loss

    return loss_fn


def hard_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = (y_true >= 0.5).astype(np.uint8)
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    eps = 1e-9
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    dice = (2 * tp) / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, eps)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def write_progress(path: pathlib.Path, epoch: int, total_epochs: int, loss: float, val_dice: float):
    data = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "percentage": round((epoch / max(total_epochs, 1)) * 100, 2),
        "loss": round(float(loss), 6),
        "val_dice": round(float(val_dice), 6),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_summary_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

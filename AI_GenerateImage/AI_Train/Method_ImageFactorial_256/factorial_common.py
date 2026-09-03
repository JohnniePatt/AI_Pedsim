from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pathlib
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
EXPECTED_CASES = {"train": 2603, "validation": 439, "test": 862}
EXPECTED_PLANS = {"train": 412, "validation": 60, "test": 117}


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def resolve_path(value: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def verify_canonical_dataset(dataset_root: pathlib.Path, snapshot_path: pathlib.Path | None = None):
    inventory = {}
    plan_sets = {}
    rows = []
    for split in ("train", "validation", "test"):
        a_names = {p.name for p in (dataset_root / "A" / split).glob("*.png")}
        b_names = {p.name for p in (dataset_root / "B" / split).glob("*.png")}
        paired = sorted(a_names & b_names)
        plans = {name.split("__", 1)[0] for name in paired}
        inventory[split] = {
            "input_count": len(a_names),
            "target_count": len(b_names),
            "paired_count": len(paired),
            "plan_count": len(plans),
            "input_only_count": len(a_names - b_names),
            "target_only_count": len(b_names - a_names),
        }
        plan_sets[split] = plans
        rows.extend((split, name, name.split("__", 1)[0]) for name in paired)

    overlap = {
        "train_validation": len(plan_sets["train"] & plan_sets["validation"]),
        "train_test": len(plan_sets["train"] & plan_sets["test"]),
        "validation_test": len(plan_sets["validation"] & plan_sets["test"]),
    }
    problems = []
    for split in EXPECTED_CASES:
        current = inventory[split]
        if current["paired_count"] != EXPECTED_CASES[split]:
            problems.append(f"{split} cases={current['paired_count']}")
        if current["plan_count"] != EXPECTED_PLANS[split]:
            problems.append(f"{split} plans={current['plan_count']}")
        if current["input_only_count"] or current["target_only_count"]:
            problems.append(f"{split} A/B mismatch")
    if any(overlap.values()):
        problems.append(f"plan overlap={overlap}")
    if problems:
        raise RuntimeError("Canonical dataset verification failed: " + "; ".join(problems))

    snapshot_hash = None
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "file_name", "plan_id"])
            writer.writerows(rows)
        snapshot_hash = sha256_file(snapshot_path)
    return inventory, overlap, snapshot_hash


class FactorialDensityDataset(Dataset):
    """One preprocessing contract shared by all four factorial cells."""

    def __init__(self, dataset_root: pathlib.Path, split: str, image_size: int = 256):
        self.dataset_root = pathlib.Path(dataset_root)
        self.split = "validation" if split in {"val", "validation"} else split
        self.image_size = int(image_size)
        self.dir_a = self.dataset_root / "A" / self.split
        self.dir_b = self.dataset_root / "B" / self.split
        a_names = {p.name for p in self.dir_a.glob("*.png")}
        b_names = {p.name for p in self.dir_b.glob("*.png")}
        self.filenames = sorted(a_names & b_names)
        if not self.filenames:
            raise RuntimeError(f"No paired images in {self.dir_a} and {self.dir_b}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        name = self.filenames[index]
        image_a = Image.open(self.dir_a / name).convert("RGB").resize(
            (self.image_size, self.image_size), BILINEAR
        )
        image_b = Image.open(self.dir_b / name).convert("L").resize(
            (self.image_size, self.image_size), BILINEAR
        )
        input_01 = np.asarray(image_a, dtype=np.float32) / 255.0
        target_01 = np.asarray(image_b, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(input_01).permute(2, 0, 1).contiguous()
        input_tensor = input_tensor * 2.0 - 1.0
        target_tensor = torch.from_numpy(target_01[None, ...]).contiguous()
        return input_tensor, target_tensor, name


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class SharedUNetGenerator(nn.Module):
    """Exact generator used by both unet_l1 and unet_wgangp."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.c1 = ConvBlock(3, base)
        self.c2 = ConvBlock(base, base * 2)
        self.c3 = ConvBlock(base * 2, base * 4)
        self.c4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.u4 = ConvBlock(base * 24, base * 8)
        self.u3 = ConvBlock(base * 12, base * 4)
        self.u2 = ConvBlock(base * 6, base * 2)
        self.u1 = ConvBlock(base * 3, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        c1 = self.c1(value)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        value = self.bottleneck(self.pool(c4))
        value = self.u4(torch.cat([self.up(value), c4], dim=1))
        value = self.u3(torch.cat([self.up(value), c3], dim=1))
        value = self.u2(torch.cat([self.up(value), c2], dim=1))
        value = self.u1(torch.cat([self.up(value), c1], dim=1))
        return torch.sigmoid(self.output(value))


class ResNetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class SharedResNet9Generator(nn.Module):
    """Exact generator used by both resnet_l1 and resnet_wgangp."""

    def __init__(self, ngf: int = 64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        for index in range(3):
            multiplier = 2**index
            layers.extend([
                nn.Conv2d(ngf * multiplier, ngf * multiplier * 2, 3, 2, 1),
                nn.InstanceNorm2d(ngf * multiplier * 2),
                nn.ReLU(True),
            ])
        for _ in range(9):
            layers.append(ResNetBlock(ngf * 8))
        for index in range(3):
            multiplier = 2 ** (3 - index)
            layers.extend([
                nn.ConvTranspose2d(
                    ngf * multiplier,
                    ngf * multiplier // 2,
                    3,
                    2,
                    1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(ngf * multiplier // 2),
                nn.ReLU(True),
            ])
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, 7), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class SharedPatchCritic(nn.Module):
    """One conditional PatchGAN critic shared by both adversarial cells."""

    def __init__(self, input_channels: int = 4, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


def build_generator(architecture: str) -> nn.Module:
    if architecture == "unet":
        return SharedUNetGenerator()
    if architecture == "resnet9":
        return SharedResNet9Generator()
    raise ValueError(f"Unknown architecture: {architecture}")


def gradient_penalty(
    critic: nn.Module,
    input_tensor: torch.Tensor,
    real_target: torch.Tensor,
    fake_target: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real_target.size(0), 1, 1, 1, device=real_target.device)
    interpolated = (alpha * real_target + (1.0 - alpha) * fake_target).requires_grad_(True)
    score = critic(torch.cat([input_tensor, interpolated], dim=1))
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def density_metrics(target, prediction):
    target = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
    difference = prediction - target
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference**2))
    rmse = float(math.sqrt(mse))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))
    mean_t, mean_p = float(target.mean()), float(prediction.mean())
    var_t, var_p = float(target.var()), float(prediction.var())
    covariance = float(np.mean((target - mean_t) * (prediction - mean_p)))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_t * mean_p + c1) * (2 * covariance + c2)) / (
        (mean_t**2 + mean_p**2 + c1) * (var_t + var_p + c2)
    )
    return {"mae": mae, "mse": mse, "rmse": rmse, "psnr": psnr, "ssim": float(ssim)}


def jet_image(value) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


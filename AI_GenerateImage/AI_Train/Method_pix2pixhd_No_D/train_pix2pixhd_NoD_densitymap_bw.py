import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pathlib
import sys
from tqdm import tqdm
from datetime import datetime
import argparse
import json

# ==========================================
# TRAINING CONFIGURATION (Pix2PixHD No_D DensityMap BW)
# ==========================================
class TrainingConfiguration:
    # 1. Hyperparameters
    generator_learning_rate = 0.0002
    beta1 = 0.0
    beta2 = 0.9
    batch_size = 8
    epochs = 50

    # 2. Loss Weights
    l1_loss_weight = 10.0
    # Density BW targets are sparse. These weights prevent the generator from
    # taking the easy shortcut of predicting an all-black image.
    density_foreground_weight = 30.0
    density_intensity_weight = 10.0
    density_foreground_threshold = 1.0 / 255.0

    # 3. Resume
    resume_checkpoint_path = "-"

    # 4. Image Settings
    image_size = 256         # resize to this; must be multiple of 32
    input_channels = 3
    output_channels = 3

    # 5. Dataset (fixed default for DensityMap BW; override via config if needed)
    dataset_root = "../Dataset/Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN"
    loaded_config_path = ""

    # 6. Run Organization
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = BASE_DIR.parent.parent
    DATASET_ROOT = (PROJECT_ROOT / dataset_root).resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_HD_NoD_{timestamp}"
    METHOD_NAME = BASE_DIR.name
    RUNS_ROOT = PROJECT_ROOT / "AI_Result" / METHOD_NAME / "outputs"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name

    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"
    LOG_DIR = CURRENT_RUN_DIR / "logs"
    SAMPLE_DIR = CURRENT_RUN_DIR / "samples"
    TEST_RESULT_DIR = CURRENT_RUN_DIR / "test_results"

# Initialize Config
config = TrainingConfiguration()

def load_config_from_json(json_path):
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Resolve dataset_root string → DATASET_ROOT Path
    if config.dataset_root:
        p = pathlib.Path(config.dataset_root)
        if not p.is_absolute():
            p = config.PROJECT_ROOT / p
        config.DATASET_ROOT = p
    config.loaded_config_path = str(pathlib.Path(json_path).resolve())

    # Ensure image_size is a valid multiple of 32
    config.image_size = ((config.image_size + 31) // 32) * 32

    print(f"📂 [CONFIG] Loaded parameters from {config.loaded_config_path}")
    print(f"📂 [CONFIG] DATASET_ROOT = {config.DATASET_ROOT}")
    print(f"🖼️  [CONFIG] image_size = {config.image_size} | batch = {config.batch_size}")
    print(
        f"🎯 [DENSITY BW] fg_weight={config.density_foreground_weight} | "
        f"intensity_weight={config.density_intensity_weight} | "
        f"threshold={config.density_foreground_threshold}"
    )

def write_progress(epoch, total_epochs, loss, val_l1):
    progress_file = config.CURRENT_RUN_DIR / "progress.json"
    data = {
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "percentage": round(((epoch + 1) / total_epochs) * 100, 2),
        "loss": round(loss, 6),
        "val_l1": round(val_l1, 6),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(progress_file, "w") as f:
        json.dump(data, f, indent=4)


def density_aware_l1_loss(fake_b, real_b, base_criterion):
    if config.density_foreground_weight <= 0 and config.density_intensity_weight <= 0:
        return base_criterion(fake_b, real_b)

    target_01 = ((real_b + 1.0) * 0.5).clamp(0.0, 1.0)
    target_gray = target_01.mean(dim=1, keepdim=True)
    weights = torch.ones_like(target_gray)

    if config.density_foreground_weight > 0:
        foreground = (target_gray > float(config.density_foreground_threshold)).float()
        weights = weights + foreground * float(config.density_foreground_weight)

    if config.density_intensity_weight > 0:
        weights = weights + target_gray * float(config.density_intensity_weight)

    return (torch.abs(fake_b - real_b) * weights).mean()

# --- Generator Components (ResNet-based) ---
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True)
        )
    def forward(self, x):
        return x + self.block(x)

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        ]
        model += [
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.InstanceNorm2d(512, affine=True), nn.ReLU(True)
        ]
        for _ in range(n_blocks):
            model += [ResNetBlock(512)]
        model += [
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(64, affine=True), nn.ReLU(True)
        ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)

# --- Dataset ---
class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="train", image_size=256):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        self.file_list = sorted([f.name for f in self.directory_A.glob("*.png")])

        # Use config image_size (already snapped to multiple of 32)
        target = ((image_size + 31) // 32) * 32
        self.target_w = target
        self.target_h = target
        print(f"📏 [DATASET-{subset}] {len(self.file_list)} images | resize → {self.target_w}x{self.target_h}")

        self.transforms = transforms.Compose([
            transforms.Resize((self.target_h, self.target_w), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_a = Image.open(self.directory_A / name).convert("RGB")
        img_b = Image.open(self.directory_B / name).convert("RGB")
        return self.transforms(img_a), self.transforms(img_b)

def execute_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Training on: {device_status}\n{'='*50}\n")
    print(f"🚀 [INIT NoD] Device: {device} | Run: {config.run_name}")

    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR, config.TEST_RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    run_config_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    config_dict = {
        k: str(v) if isinstance(v, pathlib.Path) else v
        for k, v in config.__class__.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }
    for k, v in config.__dict__.items():
        config_dict[k] = str(v) if isinstance(v, pathlib.Path) else v
    with open(run_config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"📄 [CONFIG] Run snapshot saved to {run_config_path}")

    raw_config_path = pathlib.Path(config.loaded_config_path) if config.loaded_config_path else (config.BASE_DIR / "config_train.json")
    if not raw_config_path.exists():
        raw_config_path = config.BASE_DIR / "config_active.json"
    if raw_config_path.exists():
        import shutil
        shutil.copy(raw_config_path, config.CURRENT_RUN_DIR / raw_config_path.name)
        print(f"💾 [ARCHIVE] {raw_config_path.name} copied to {config.CURRENT_RUN_DIR}")

    write_progress(-1, config.epochs, 0.0, 0.0)

    generator = GeneratorNetwork(config.input_channels, config.output_channels).to(device)

    if config.resume_checkpoint_path not in ["-", "", None]:
        print(f"🔄 [RESUME] Loading {config.resume_checkpoint_path}")
        generator.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))

    generator_optimizer = optim.Adam(generator.parameters(), lr=config.generator_learning_rate, betas=(config.beta1, config.beta2))

    log_hist_path = config.LOG_DIR / "training_history.csv"
    with open(log_hist_path, "w") as f:
        f.write("epoch,l1,val_l1\n")

    pixel_loss_criterion = nn.L1Loss()

    train_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "train", config.image_size)
    val_ds   = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "validation", config.image_size)
    test_ds  = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", config.image_size)
    if len(train_ds) == 0:
        raise RuntimeError(
            f"Dataset is empty at '{config.DATASET_ROOT}'. "
            f"Expected PNG files under A/train and B/train."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"Validation dataset is empty at '{config.DATASET_ROOT}'. "
            f"Expected PNG files under A/validation and B/validation."
        )
    if len(test_ds) == 0:
        raise RuntimeError(
            f"Test dataset is empty at '{config.DATASET_ROOT}'. "
            f"Expected PNG files under A/test and B/test."
        )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val = float('inf')
    global_step = 0
    for epoch in range(config.epochs):
        generator.train()
        epoch_metrics = {"l1": 0.0}

        pbar = tqdm(train_loader, leave=False, disable=not sys.stdout.isatty())
        for real_a, real_b in pbar:
            global_step += 1
            real_a, real_b = real_a.to(device), real_b.to(device)

            # --- Train Generator ---
            generator_optimizer.zero_grad()
            fake_b = generator(real_a)

            loss_g_l1 = density_aware_l1_loss(fake_b, real_b, pixel_loss_criterion)
            loss_total = loss_g_l1 * config.l1_loss_weight
            loss_total.backward()
            generator_optimizer.step()

            epoch_metrics["l1"] += loss_g_l1.item()
            pbar.set_description(f"E{epoch}")
            pbar.set_postfix(G_L1=f"{loss_total.item():.3f}")

        # Validation
        generator.eval(); val_l1 = 0.0
        with torch.no_grad():
            for va, vb in val_loader:
                val_l1 += density_aware_l1_loss(generator(va.to(device)), vb.to(device), pixel_loss_criterion).item()

        avg_val = (val_l1 / len(val_loader)) * config.l1_loss_weight
        metrics = [
            epoch,
            epoch_metrics['l1'] / len(train_loader),
            avg_val
        ]
        with open(log_hist_path, "a") as f:
            f.write(",".join([f"{m:.6f}" if isinstance(m, float) else str(m) for m in metrics]) + "\n")

        print(f"✨ [EPOCH {epoch}] L1_Loss: {metrics[1]:.4f} | Val L1: {avg_val:.4f}")
        write_progress(epoch, config.epochs, metrics[1], avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")
            print(f"  🏆 New Best NoD Model (Val L1: {best_val:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")

    print("\n--- Triggering Standalone Test Evaluation ---")
    import subprocess
    test_script = pathlib.Path(__file__).parent / "test_pix2pixhd_NoD_densitymap_bw.py"
    test_config = pathlib.Path(__file__).parent / "config_test.json"
    subprocess.run([
        sys.executable,
        str(test_script),
        "--run_path",
        str(config.CURRENT_RUN_DIR),
        "--config",
        str(test_config),
    ])

    print(f"🏁 Training Finished! Results in {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    config_candidates = []
    if os.path.isabs(args.config):
        config_candidates.append(args.config)
    else:
        # Prefer path as provided from current working directory, then script-relative.
        config_candidates.append(args.config)
        config_candidates.append(os.path.join(script_dir, args.config))

    selected_config = next((p for p in config_candidates if os.path.exists(p)), None)
    if selected_config:
        load_config_from_json(selected_config)
    else:
        fallback_train = os.path.join(script_dir, "config_train.json")
        if os.path.exists(fallback_train):
            print(f"[WARN] Config '{args.config}' not found. Fallback to '{fallback_train}'.")
            load_config_from_json(fallback_train)
        else:
            fallback = os.path.join(script_dir, "config_active.json")
            if os.path.exists(fallback):
                print(f"[WARN] Config '{args.config}' not found. Fallback to '{fallback}'.")
                load_config_from_json(fallback)
            else:
                raise FileNotFoundError(
                    f"Config not found: {args.config} and fallback config_train.json is missing."
                )

    execute_training()

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
# TRAINING CONFIGURATION (Pix2PixHD v04 — Fast Mode)
# ==========================================
class TrainingConfiguration:
    # 1. Hyperparameters
    generator_learning_rate = 0.0002
    discriminator_learning_rate = 0.0001
    beta1 = 0.0
    beta2 = 0.9
    batch_size = 8
    epochs = 50

    # 2. Loss Weights (HD specific)
    l1_loss_weight = 10.0
    feature_matching_weight = 2.0
    lambda_gp = 10.0  # Gradient Penalty weight
    mask_bce_loss_weight = 6.0
    mask_dice_loss_weight = 6.0
    mask_threshold = 0.5
    mask_foreground_weight = 20.0
    dice_smooth = 1e-6

    # 3. GAN Options
    num_discriminators = 1   # single scale — faster, sufficient for this task
    use_label_smoothing = False
    n_critic = 1             # 1× D per G step — L1+FM already guides G

    # 4. Resume
    resume_checkpoint_path = "-"

    # 5. Image Settings
    image_size = 256         # resize to this; must be multiple of 32
    input_channels = 3
    output_channels = 3

    # 6. Dataset (override via config_train.json "dataset_root")
    dataset_root = ""  # set in config_train.json; empty = use default below

    # 7. Run Organization
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = BASE_DIR.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "Topo_bottleneck" / "trajectory_line_dataset" / "Cleandata_1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_HD_{timestamp}"
    METHOD_NAME = BASE_DIR.name
    RUNS_ROOT = PROJECT_ROOT / "AI_Result" / METHOD_NAME / "outputs"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name

    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"
    LOG_DIR = CURRENT_RUN_DIR / "logs"
    SAMPLE_DIR = CURRENT_RUN_DIR / "samples"
    TEST_RESULT_DIR = CURRENT_RUN_DIR / "test_results"

# Initialize Config
config = TrainingConfiguration()


def resolve_input_path(path_str, script_dir):
    """Resolve CLI path from absolute, CWD-relative, then script-relative."""
    if not path_str:
        return None

    p = pathlib.Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)

    cwd_candidate = pathlib.Path.cwd() / p
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    script_candidate = pathlib.Path(script_dir) / p
    if script_candidate.exists():
        return str(script_candidate.resolve())

    return None

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

    # Ensure image_size is a valid multiple of 32
    config.image_size = ((config.image_size + 31) // 32) * 32

    print(f"📂 [CONFIG] Loaded parameters from {json_path}")
    print(f"📂 [CONFIG] DATASET_ROOT = {config.DATASET_ROOT}")
    print(f"🖼️  [CONFIG] image_size = {config.image_size} | batch = {config.batch_size} | n_critic = {config.n_critic} | D scales = {config.num_discriminators}")

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


def tensor_to_gray01(image_tensor):
    """
    Convert normalized RGB tensor in [-1, 1] to grayscale [0, 1].
    Input shape: (N, 3, H, W)
    Output shape: (N, 1, H, W)
    """
    image_01 = (image_tensor + 1.0) * 0.5
    r = image_01[:, 0:1, :, :]
    g = image_01[:, 1:2, :, :]
    b = image_01[:, 2:3, :, :]
    return 0.299 * r + 0.587 * g + 0.114 * b


def soft_mask_from_gray(gray_tensor, threshold):
    """
    Build a smooth mask probability from grayscale intensity.
    """
    denom = max(1.0 - float(threshold), 1e-6)
    return torch.clamp((gray_tensor - threshold) / denom, 0.0, 1.0)


def hard_mask_from_gray(gray_tensor, threshold):
    return (gray_tensor >= threshold).float()


def dice_loss(pred_prob, target_mask, smooth=1e-6):
    pred = pred_prob.contiguous().view(pred_prob.size(0), -1)
    target = target_mask.contiguous().view(target_mask.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice_score = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice_score.mean()


def weighted_bce_loss(pred_prob, target_mask, foreground_weight=1.0, eps=1e-6):
    pred_prob = torch.clamp(pred_prob, eps, 1.0 - eps)
    weights = 1.0 + (float(foreground_weight) - 1.0) * target_mask
    bce = -(target_mask * torch.log(pred_prob) + (1.0 - target_mask) * torch.log(1.0 - pred_prob))
    return (weights * bce).mean()

# --- Pix2PixHD Components: ResNet-based Generator ---
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
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
        ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)

# --- Multi-scale Discriminator ---
class SingleDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        # WGAN-GP: InstanceNorm (not BatchNorm) to avoid gradient penalty issues
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128, affine=True), nn.LeakyReLU(0.2, True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, True))
        self.layer5 = nn.Conv2d(512, 1, 4, 1, 1)

    def forward(self, x):
        f1 = self.layer1(x); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3)
        return [f1, f2, f3, f4, self.layer5(f4)]

class DiscriminatorNetwork(nn.Module):
    def __init__(self, in_channels=6, num_scales=1):
        super().__init__()
        self.scales = num_scales
        self.discriminators = nn.ModuleList([SingleDiscriminator(in_channels) for _ in range(num_scales)])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for i in range(self.scales):
            outputs.append(self.discriminators[i](x))
            if i < self.scales - 1: x = self.downsample(x)
        return outputs

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    gp = 0
    for i in range(len(d_interpolates)):
        gradients = torch.autograd.grad(
            outputs=d_interpolates[i][-1],
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates[i][-1]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gp += ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp / len(d_interpolates)

# --- Dataset ---
class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="train", image_size=256):
        base_root = pathlib.Path(root_directory)
        subset_aliases = {
            "train": ["train", "training"],
            "validation": ["validation", "val", "valid"],
            "test": ["test", "testing"],
        }
        candidate_splits = subset_aliases.get(subset, [subset])

        self.directory_A = None
        self.directory_B = None
        for split_name in candidate_splits:
            cand_a = base_root / "A" / split_name
            cand_b = base_root / "B" / split_name
            if cand_a.exists() and cand_b.exists():
                self.directory_A = cand_a
                self.directory_B = cand_b
                break

        if self.directory_A is None or self.directory_B is None:
            a_root = base_root / "A"
            b_root = base_root / "B"
            a_splits = sorted([p.name for p in a_root.iterdir()]) if a_root.exists() else []
            b_splits = sorted([p.name for p in b_root.iterdir()]) if b_root.exists() else []
            raise FileNotFoundError(
                f"[DATASET-{subset}] split not found in {base_root}. "
                f"Tried splits: {candidate_splits}. "
                f"Available A splits: {a_splits if a_splits else 'A missing/empty'}, "
                f"B splits: {b_splits if b_splits else 'B missing/empty'}"
            )

        a_names = {f.name for f in self.directory_A.glob("*.png")}
        b_names = {f.name for f in self.directory_B.glob("*.png")}
        missing_b = sorted(a_names - b_names)
        missing_a = sorted(b_names - a_names)
        if missing_a or missing_b:
            raise RuntimeError(
                f"[DATASET-{subset}] A/B filename mismatch: "
                f"missing_in_B={len(missing_b)}, missing_in_A={len(missing_a)}"
            )
        self.file_list = sorted(a_names & b_names)

        target = ((image_size + 31) // 32) * 32
        self.target_w = target
        self.target_h = target
        print(f"[DATASET-{subset}] {len(self.file_list)} images | resize -> {self.target_w}x{self.target_h}")

        self.transform_a = transforms.Compose([
            transforms.Resize((self.target_h, self.target_w), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform_b = transforms.Compose([
            # B is a binary mask; interpolating it creates gray halos and thick blobs.
            transforms.Resize((self.target_h, self.target_w), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_a = Image.open(self.directory_A / name).convert("RGB")
        img_b = Image.open(self.directory_B / name).convert("RGB")
        return self.transform_a(img_a), self.transform_b(img_b)

def execute_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Training on: {device_status}\n{'='*50}\n")
    print(f"🚀 [INIT HD] Device: {device} | Run: {config.run_name}")

    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR, config.TEST_RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    run_config_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    config_dict = {k: str(v) if isinstance(v, pathlib.Path) else v
                   for k, v in config.__class__.__dict__.items() if not k.startswith("__")}
    with open(run_config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"📄 [CONFIG] Run snapshot saved to {run_config_path}")

    raw_config_path = config.BASE_DIR / "config_train.json"
    if not raw_config_path.exists():
        raw_config_path = config.BASE_DIR / "config_active.json"
    if raw_config_path.exists():
        import shutil
        shutil.copy(raw_config_path, config.CURRENT_RUN_DIR / raw_config_path.name)
        print(f"💾 [ARCHIVE] {raw_config_path.name} copied to {config.CURRENT_RUN_DIR}")

    write_progress(-1, config.epochs, 0.0, 0.0)

    generator = GeneratorNetwork(config.input_channels, config.output_channels).to(device)
    discriminator = DiscriminatorNetwork(config.input_channels + config.output_channels, config.num_discriminators).to(device)

    if config.resume_checkpoint_path not in ["-", "", None]:
        print(f"🔄 [RESUME] Loading {config.resume_checkpoint_path}")
        generator.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))

    generator_optimizer = optim.Adam(generator.parameters(), lr=config.generator_learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(config.beta1, config.beta2))

    log_hist_path = config.LOG_DIR / "training_history.csv"
    with open(log_hist_path, "w") as f:
        f.write("epoch,d_loss,g_adv,fm,l1,bce,dice,val_l1_raw,val_l1,val_bce,val_dice,val_total\n")

    pixel_loss_criterion = nn.L1Loss()
    fm_loss_criterion = nn.L1Loss()
    mask_bce_criterion = nn.BCELoss()

    train_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "train", config.image_size)
    val_ds   = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "validation", config.image_size)
    test_ds  = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", config.image_size)
    if len(train_ds) == 0:
        raise RuntimeError(
            f"[DATASET] train split is empty at {train_ds.directory_A}. "
            "Check --config path and dataset_root in config."
        )
    if len(val_ds) == 0:
        raise RuntimeError(
            f"[DATASET] validation split is empty at {val_ds.directory_A}. "
            "Create validation data or point dataset_root to the correct dataset."
        )
    if len(test_ds) == 0:
        raise RuntimeError(
            f"[DATASET] test split is empty at {test_ds.directory_A}. "
            "Create test data or point dataset_root to the correct dataset."
        )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_val = float('inf')
    global_step = 0
    for epoch in range(config.epochs):
        generator.train(); discriminator.train()
        epoch_metrics = {"d": 0.0, "g_adv": 0.0, "fm": 0.0, "l1": 0.0, "bce": 0.0, "dice": 0.0}

        pbar = tqdm(train_loader, leave=False, disable=not sys.stdout.isatty())
        for real_a, real_b in pbar:
            global_step += 1
            real_a, real_b = real_a.to(device), real_b.to(device)

            # --- Train Critic (n_critic times) ---
            loss_d_val = 0
            for _ in range(config.n_critic):
                fake_b_d = generator(real_a).detach()
                joint_real = torch.cat([real_a, real_b], 1)
                joint_fake = torch.cat([real_a, fake_b_d], 1)

                discriminator_optimizer.zero_grad()
                real_outputs = discriminator(joint_real)
                fake_outputs = discriminator(joint_fake)

                loss_d = 0
                for i in range(config.num_discriminators):
                    loss_d += (fake_outputs[i][-1].mean() - real_outputs[i][-1].mean())

                gp = compute_gradient_penalty(discriminator, joint_real, joint_fake, device)
                loss_total_d = loss_d + config.lambda_gp * gp
                loss_total_d.backward()
                discriminator_optimizer.step()
                loss_d_val = loss_total_d.item()

            # --- Train Generator (1 step) ---
            fake_b = generator(real_a)
            joint_real = torch.cat([real_a, real_b], 1)
            joint_fake_g = torch.cat([real_a, fake_b], 1)

            generator_optimizer.zero_grad()
            fake_outputs_g = discriminator(joint_fake_g)
            real_outputs_g = discriminator(joint_real)

            loss_g_adv = 0; loss_fm = 0
            for i in range(config.num_discriminators):
                loss_g_adv += -fake_outputs_g[i][-1].mean()
                for j in range(len(fake_outputs_g[i]) - 1):
                    loss_fm += fm_loss_criterion(fake_outputs_g[i][j], real_outputs_g[i][j].detach())

            loss_g_l1 = pixel_loss_criterion(fake_b, real_b)

            real_gray = tensor_to_gray01(real_b)
            fake_gray = tensor_to_gray01(fake_b)
            gt_mask = hard_mask_from_gray(real_gray, config.mask_threshold)
            pred_mask_prob = torch.clamp(fake_gray, 1e-6, 1.0 - 1e-6)

            loss_g_bce = weighted_bce_loss(
                pred_mask_prob,
                gt_mask,
                foreground_weight=config.mask_foreground_weight,
            )
            loss_g_dice = dice_loss(pred_mask_prob, gt_mask, smooth=config.dice_smooth)
            loss_total = (
                loss_g_adv
                + (loss_fm * config.feature_matching_weight)
                + (loss_g_l1 * config.l1_loss_weight)
                + (loss_g_bce * config.mask_bce_loss_weight)
                + (loss_g_dice * config.mask_dice_loss_weight)
            )
            loss_total.backward()
            generator_optimizer.step()

            epoch_metrics["d"] += loss_d_val
            epoch_metrics["g_adv"] += loss_g_adv.item()
            epoch_metrics["fm"] += loss_fm.item()
            epoch_metrics["l1"] += loss_g_l1.item()
            epoch_metrics["bce"] += loss_g_bce.item()
            epoch_metrics["dice"] += loss_g_dice.item()
            pbar.set_description(f"E{epoch}")
            pbar.set_postfix(
                D=f"{loss_d_val:.3f}",
                G=f"{loss_total.item():.3f}",
                L1=f"{loss_g_l1.item():.3f}",
                BCE=f"{loss_g_bce.item():.3f}",
                Dice=f"{loss_g_dice.item():.3f}",
            )

        # Validation
        generator.eval(); val_l1 = 0.0; val_bce = 0.0; val_dice = 0.0
        with torch.no_grad():
            for va, vb in val_loader:
                va = va.to(device)
                vb = vb.to(device)
                vpred = generator(va)
                val_l1 += pixel_loss_criterion(vpred, vb).item()

                vb_gray = tensor_to_gray01(vb)
                vp_gray = tensor_to_gray01(vpred)
                v_gt_mask = hard_mask_from_gray(vb_gray, config.mask_threshold)
                v_pred_mask_prob = torch.clamp(vp_gray, 1e-6, 1.0 - 1e-6)
                val_bce += weighted_bce_loss(
                    v_pred_mask_prob,
                    v_gt_mask,
                    foreground_weight=config.mask_foreground_weight,
                ).item()
                val_dice += dice_loss(v_pred_mask_prob, v_gt_mask, smooth=config.dice_smooth).item()

        avg_val_l1_raw = (val_l1 / len(val_loader))
        avg_val_l1 = avg_val_l1_raw * config.l1_loss_weight
        avg_val_bce = (val_bce / len(val_loader))
        avg_val_dice = (val_dice / len(val_loader))
        avg_val = (
            avg_val_l1
            + avg_val_bce * config.mask_bce_loss_weight
            + avg_val_dice * config.mask_dice_loss_weight
        )
        metrics = [
            epoch,
            epoch_metrics['d'] / len(train_loader),
            epoch_metrics['g_adv'] / len(train_loader),
            epoch_metrics['fm'] / len(train_loader),
            epoch_metrics['l1'] / len(train_loader),
            epoch_metrics['bce'] / len(train_loader),
            epoch_metrics['dice'] / len(train_loader),
            avg_val_l1_raw,
            avg_val_l1,
            avg_val_bce,
            avg_val_dice,
            avg_val
        ]
        with open(log_hist_path, "a") as f:
            f.write(",".join([f"{m:.6f}" if isinstance(m, float) else str(m) for m in metrics]) + "\n")

        print(
            f"[EPOCH {epoch}] "
            f"D: {metrics[1]:.4f} | FM: {metrics[3]:.4f} | "
            f"Train L1/BCE/Dice: {metrics[4]:.4f}/{metrics[5]:.4f}/{metrics[6]:.4f} | "
            f"Val L1raw/L1w/BCE/Dice: {metrics[7]:.4f}/{metrics[8]:.4f}/{metrics[9]:.4f}/{metrics[10]:.4f} | "
            f"Val Total: {avg_val:.4f}"
        )
        write_progress(epoch, config.epochs, metrics[1], avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")
            print(f"  New Best HD Model (Val Total: {best_val:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")

    print("\n--- Triggering Standalone HD Test Evaluation ---")
    import subprocess
    test_script = pathlib.Path(__file__).parent / "test_pix2pixHD_trajectoryLine.py"
    subprocess.run([sys.executable, str(test_script), "--run_path", str(config.CURRENT_RUN_DIR)])

    print(f"🏁 HD Training Finished! Results in {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train_04.json")
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    cpath = resolve_input_path(args.config, script_dir)
    if cpath:
        load_config_from_json(cpath)
    else:
        fallback_train = resolve_input_path("config_train.json", script_dir)
        if fallback_train:
            print(f"[WARN] Config not found: {args.config}. Falling back to {fallback_train}")
            load_config_from_json(fallback_train)
        else:
            fallback = resolve_input_path("config_active.json", script_dir)
            if fallback:
                print(f"[WARN] Config not found: {args.config}. Falling back to {fallback}")
                load_config_from_json(fallback)
            else:
                print(f"[WARN] No config file found. Using in-script defaults.")

    execute_training()



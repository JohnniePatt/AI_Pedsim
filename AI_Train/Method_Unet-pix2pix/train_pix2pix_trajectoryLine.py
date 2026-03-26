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
import torch.nn.functional as F
from datetime import datetime
import argparse
import json

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
class TrainingConfiguration:
    # 1. Hyperparameters
    generator_learning_rate = 0.0002
    discriminator_learning_rate = 0.0001 # TTUR: Train D slower if it's too strong
    beta1 = 0.5
    beta2 = 0.999
    batch_size = 4
    epochs = 100
    l1_loss_weight = 100.0
    
    # 2. GAN Balancing
    use_label_smoothing = True 
    d_train_freq = 1 # Train D every N batches (default 1)
    
    # 3. Resume Training
    # Set to a path (e.g. "runs_trajectory/run_xxx/checkpoints/generator_best.pth") or "-" to start new
    resume_checkpoint_path = "-" 
    
    # 4. Image Settings
    image_size = 512
    input_channels = 3
    output_channels = 3
    
    # 5. Run Organization
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = BASE_DIR.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "Topo_2" / "trajectory_line_dataset" / "Cleandata_1"
    
    # Create timestamped run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    RUNS_ROOT = BASE_DIR / "runs_trajectory"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name
    
    # Derived Paths
    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"
    LOG_DIR = CURRENT_RUN_DIR / "logs"
    SAMPLE_DIR = CURRENT_RUN_DIR / "samples"

# Initialize Config
config = TrainingConfiguration()

def load_config_from_json(json_path):
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    print(f"📂 [CONFIG] Loaded parameters from {json_path}")

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

# --- GAN Architecture: U-Net Generator ---
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=True, use_dropout=False):
        super().__init__()
        if downsampling:
            self.operation = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.operation = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.operation(x)
        if self.use_dropout: x = self.dropout_layer(x)
        return x

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.encoder2 = UNetBlock(64, 128, downsampling=True)
        self.encoder3 = UNetBlock(128, 256, downsampling=True)
        self.encoder4 = UNetBlock(256, 512, downsampling=True)
        self.encoder5 = UNetBlock(512, 512, downsampling=True)
        self.encoder6 = UNetBlock(512, 512, downsampling=True)
        self.encoder7 = UNetBlock(512, 512, downsampling=True)
        self.encoder8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        
        # Decoder
        self.decoder1 = UNetBlock(512, 512, downsampling=False, use_dropout=True)
        self.decoder2 = UNetBlock(1024, 512, downsampling=False, use_dropout=True)
        self.decoder3 = UNetBlock(1024, 512, downsampling=False, use_dropout=True)
        self.decoder4 = UNetBlock(1024, 512, downsampling=False)
        self.decoder5 = UNetBlock(1024, 256, downsampling=False)
        self.decoder6 = UNetBlock(512, 128, downsampling=False)
        self.decoder7 = UNetBlock(256, 64, downsampling=False)
        self.decoder8 = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.encoder1(x); d2 = self.encoder2(d1); d3 = self.encoder3(d2); d4 = self.encoder4(d3)
        d5 = self.encoder5(d4); d6 = self.encoder6(d5); d7 = self.encoder7(d6); d8 = self.encoder8(d7)
        u1 = self.decoder1(d8)
        u2 = self.decoder2(torch.cat([u1, d7], 1)); u3 = self.decoder3(torch.cat([u2, d6], 1))
        u4 = self.decoder4(torch.cat([u3, d5], 1)); u5 = self.decoder5(torch.cat([u4, d4], 1))
        u6 = self.decoder6(torch.cat([u5, d3], 1)); u7 = self.decoder7(torch.cat([u6, d2], 1))
        return self.decoder8(torch.cat([u7, d1], 1))

class DiscriminatorNetwork(nn.Module):
    def __init__(self, joint_channels=6):
        super().__init__()
        self.structure = nn.Sequential(
            nn.Conv2d(joint_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, input_image, target_image):
        return self.structure(torch.cat([input_image, target_image], 1))

# --- Dataset ---
class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="train", image_size=512):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        self.file_list = sorted([f.name for f in self.directory_A.glob("*.png")])
        
        # 🔍 Auto-Detect Dimension
        if self.file_list:
            with Image.open(self.directory_A / self.file_list[0]) as img:
                self.orig_w, self.orig_h = img.size
            self.target_w = ((self.orig_w + 31) // 32) * 32
            self.target_h = ((self.orig_h + 31) // 32) * 32
            print(f"📏 [DATASET] Auto-Detected: {self.orig_w}x{self.orig_h} | Scaling to: {self.target_w}x{self.target_h}")
        else:
            self.target_w, self.target_h = 512, 512

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

# --- Main Training Loop ---
def execute_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Training on: {device_status}\n{'='*50}\n")
    print(f"🚀 [INIT] Device: {device} | Run: {config.run_name}")

    # Prepare Directories
    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR]: d.mkdir(parents=True, exist_ok=True)

    # 📁 Save a copy of the finalized configuration for this specific run
    run_config_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    config_dict = {k: str(v) if isinstance(v, pathlib.Path) else v 
                   for k, v in config.__class__.__dict__.items() if not k.startswith("__")}
    with open(run_config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"📄 [CONFIG] Run snapshot saved to {run_config_path}")

    # 💾 Also ARCHIVE the raw config_active.json in the run directory
    raw_config_path = config.BASE_DIR / "config_active.json"
    if raw_config_path.exists():
        import shutil
        shutil.copy(raw_config_path, config.CURRENT_RUN_DIR / "config_active.json")
        print(f"💾 [ARCHIVE] config_active.json copied to {config.CURRENT_RUN_DIR}")
    
    # 🚀 Initial Progress (0%)
    write_progress(-1, config.epochs, 0.0, 0.0)

    generator = GeneratorNetwork(config.input_channels, config.output_channels).to(device)
    discriminator = DiscriminatorNetwork(config.input_channels + config.output_channels).to(device)

    # Resume Training if requested
    if config.resume_checkpoint_path not in ["-", "", None]:
        print(f"🔄 [RESUME] Loading weights from {config.resume_checkpoint_path}")
        try:
            generator.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))
        except Exception as e: print(f"⚠️ [WARN] Could not load checkpoint: {e}")

    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=config.generator_learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(config.beta1, config.beta2))

    # Loss Settings
    gan_criterion = nn.BCEWithLogitsLoss()
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    
    # Logs
    log_path = config.LOG_DIR / "training_history.csv"
    with open(log_path, "w") as f:
        f.write("epoch,d_loss,g_adv,l1,val_l1\n")

    # Data
    train_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "train", config.image_size)
    val_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "validation", config.image_size)
    test_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", config.image_size)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    test_result_dir = config.CURRENT_RUN_DIR / "test_results"
    test_result_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    global_step = 0
    for epoch in range(config.epochs):
        generator.train(); discriminator.train()
        epoch_metrics = {"d": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        
        pbar = tqdm(train_loader, leave=False, disable=not sys.stdout.isatty())
        for real_a, real_b in pbar:
            global_step += 1
            real_a, real_b = real_a.to(device), real_b.to(device)
            fake_b = generator(real_a)

            # --- Train Discriminator (Conditional) ---
            loss_d_val = 0
            if global_step % config.d_train_freq == 0:
                pred_real = discriminator(real_a, real_b)
                label_real = torch.full_like(pred_real, 0.9) if config.use_label_smoothing else torch.ones_like(pred_real)
                loss_d_real = gan_criterion(pred_real, label_real)
                
                pred_fake = discriminator(real_a, fake_b.detach())
                loss_d_fake = gan_criterion(pred_fake, torch.zeros_like(pred_fake))
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                discriminator_optimizer.zero_grad(); loss_d.backward(); discriminator_optimizer.step()
                loss_d_val = loss_d.item()

            # --- Train Generator ---
            pred_fake_g = discriminator(real_a, fake_b)
            loss_g_adv = gan_criterion(pred_fake_g, torch.ones_like(pred_fake_g))
            loss_g_l1 = mae_criterion(fake_b, real_b)
            
            loss_g = loss_g_adv + (loss_g_l1 * config.l1_loss_weight)
            generator_optimizer.zero_grad(); loss_g.backward(); generator_optimizer.step()

            epoch_metrics["d"] += loss_d_val; epoch_metrics["g_adv"] += loss_g_adv.item(); epoch_metrics["g_l1"] += loss_g_l1.item()
            pbar.set_description(f"E{epoch}"); pbar.set_postfix(D=loss_d_val, G=loss_g.item())

        # Validation
        generator.eval(); val_l1 = 0.0
        with torch.no_grad():
            for va, vb in val_loader:
                va, vb = va.to(device), vb.to(device)
                val_l1 += mae_criterion(generator(va), vb).item()
        
        avg_val = (val_l1 / len(val_loader)) * config.l1_loss_weight
        metrics = [epoch, epoch_metrics['d']/len(train_loader), epoch_metrics['g_adv']/len(train_loader), epoch_metrics['g_l1']/len(train_loader), avg_val]
        with open(log_path, "a") as f: f.write(",".join([f"{m:.6f}" if isinstance(m, float) else str(m) for m in metrics]) + "\n")
        
        print(f"✨ [EPOCH {epoch}] D_Loss: {metrics[1]:.4f} | Val L1: {avg_val:.4f}")
        write_progress(epoch, config.epochs, metrics[1], avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")
            print(f"  🏆 New Best (Val L1: {best_val:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")

    # --- Final Test Evaluation (Standalone Trigger) ---
    print("\n--- Triggering Standalone Test Evaluation ---")
    import subprocess
    test_script = pathlib.Path(__file__).parent / "test_pix2pix_trajectoryLine.py"
    subprocess.run([sys.executable, str(test_script), "--run_path", str(config.CURRENT_RUN_DIR)])
    
    print(f"🏁 Training Finished! Results in {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config_active.json")
    args = parser.parse_args()
    
    if args.config:
        # If relative path, join with script directory
        cpath = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
        load_config_from_json(cpath)
        
    execute_training()

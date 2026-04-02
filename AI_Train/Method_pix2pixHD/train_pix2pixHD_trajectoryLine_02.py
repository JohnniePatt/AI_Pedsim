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
# TRAINING CONFIGURATION (Pix2PixHD Version)
# ==========================================
class TrainingConfiguration:
    # 1. Hyperparameters
    generator_learning_rate = 0.0001
    discriminator_learning_rate = 0.0001
    beta1 = 0.0
    beta2 = 0.9
    batch_size = 2 # Pix2PixHD is memory heavy, start small
    epochs = 100
    
    # 2. Loss Weights (HD specific)
    l1_loss_weight = 10.0 # Standard L1
    feature_matching_weight = 10.0 # Help stabilize HD details
    lambda_gp = 10.0 # Gradient Penalty weight
    
    # 3. GAN Options
    num_discriminators = 2 
    use_label_smoothing = False # Not needed for WGAN
    d_train_freq = 1 # Train D every N steps (default 1)
    
    # 4. Resume
    resume_checkpoint_path = "-"
    
    # 5. Image Settings
    image_size = 512
    input_channels = 3
    output_channels = 3
    
    # 6. Run Organization
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = BASE_DIR.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "Topo_2" / "trajectory_line_dataset" / "Cleandata_1"
    
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

# --- Pix2PixHD Components: ResNet-based Generator ---
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super().__init__()
        # Initial Extraction (Down 0)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ]
        # Downsampling (Down 1 & 2)
        model += [
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True)
        ]
        # ResNet Blocks (The logic part)
        for _ in range(n_blocks):
            model += [ResNetBlock(512)]
        # Upsampling
        model += [
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
        ]
        # Output
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
        # Return list of feature maps for Feature Matching
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True))
        self.layer5 = nn.Conv2d(512, 1, 4, 1, 1)

    def forward(self, x):
        f1 = self.layer1(x); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3)
        return [f1, f2, f3, f4, self.layer5(f4)]

class DiscriminatorNetwork(nn.Module):
    def __init__(self, in_channels=6, num_scales=2):
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
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    
    fake = torch.ones(d_interpolates[0][-1].shape, device=device, requires_grad=False)
    
    # Get gradient w.r.t. interpolates (for the first scale of multi-scale discriminator)
    # WGAN-GP usually applies to one scale or sum of scales? 
    # Usually applied to the main scale or all. Applying to all for multi-scale.
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

# --- Dataset and Training ---
class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="train", image_size=512):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        self.file_list = sorted([f.name for f in self.directory_A.glob("*.png")])
        
        # 🔍 Auto-Detect Dataset Dimensions and Calculate Nearest 32 Multiple
        if self.file_list:
            with Image.open(self.directory_A / self.file_list[0]) as img:
                self.orig_w, self.orig_h = img.size
            # Calculate multiples of 32 to keep GPU happy while preserving aspect ratio
            self.target_w = ((self.orig_w + 31) // 32) * 32
            self.target_h = ((self.orig_h + 31) // 32) * 32
            print(f"📏 [DATASET] Auto-Detected Size: {self.orig_w}x{self.orig_h} | Training at: {self.target_w}x{self.target_h}")
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

def execute_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Training on: {device_status}\n{'='*50}\n")
    print(f"🚀 [INIT HD] Device: {device} | Run: {config.run_name}")

    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR, config.TEST_RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 📁 Save a copy of the finalized configuration for this specific run
    run_config_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    config_dict = {k: str(v) if isinstance(v, pathlib.Path) else v 
                   for k, v in config.__class__.__dict__.items() if not k.startswith("__")}
    with open(run_config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"📄 [CONFIG] Run snapshot saved to {run_config_path}")

    # 💾 Also ARCHIVE the raw config_train.json (or config_active) in the run directory
    raw_config_path = config.BASE_DIR / "config_train.json"
    if not raw_config_path.exists():
        raw_config_path = config.BASE_DIR / "config_active.json" # Fallback
        
    if raw_config_path.exists():
        import shutil
        shutil.copy(raw_config_path, config.CURRENT_RUN_DIR / raw_config_path.name)
        print(f"💾 [ARCHIVE] {raw_config_path.name} copied to {config.CURRENT_RUN_DIR}")

    # 🚀 Initial Progress (0%)
    write_progress(-1, config.epochs, 0.0, 0.0)

    generator = GeneratorNetwork(config.input_channels, config.output_channels).to(device)
    discriminator = DiscriminatorNetwork(config.input_channels + config.output_channels, config.num_discriminators).to(device)

    # Resume
    if config.resume_checkpoint_path not in ["-", "", None]:
        print(f"🔄 [RESUME] Loading {config.resume_checkpoint_path}")
        generator.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))

    generator_optimizer = optim.Adam(generator.parameters(), lr=config.generator_learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(config.beta1, config.beta2))

    log_hist_path = config.LOG_DIR / "training_history.csv"
    with open(log_hist_path, "w") as f:
        f.write("epoch,d_loss,g_adv,fm,l1,val_l1\n")

    # WGAN-GP replaces BCE with mean-loss
    # gan_loss_criterion = nn.BCEWithLogitsLoss() 
    pixel_loss_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    fm_loss_criterion = nn.L1Loss() # Feature Matching uses L1

    train_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "train", config.image_size)
    val_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "validation", config.image_size)
    test_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", config.image_size)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    best_val = float('inf')
    global_step = 0
    for epoch in range(config.epochs):
        generator.train(); discriminator.train()
        epoch_metrics = {"d": 0.0, "g_adv": 0.0, "fm": 0.0, "l1": 0.0}
        
        pbar = tqdm(train_loader, leave=False, disable=not sys.stdout.isatty())
        for real_a, real_b in pbar:
            global_step += 1
            real_a, real_b = real_a.to(device), real_b.to(device)
            fake_b = generator(real_a)

            joint_real = torch.cat([real_a, real_b], 1)
            joint_fake = torch.cat([real_a, fake_b.detach()], 1)
            
            # --- Train Critic (Discriminator) ---
            loss_d_val = 0
            if global_step % config.d_train_freq == 0:
                discriminator_optimizer.zero_grad()
                
                real_outputs = discriminator(joint_real)
                fake_outputs = discriminator(joint_fake)
                
                loss_d = 0
                for i in range(config.num_discriminators):
                    # WGAN Loss: mean(Critic(Fake)) - mean(Critic(Real))
                    loss_d += (fake_outputs[i][-1].mean() - real_outputs[i][-1].mean())
                
                # Add Gradient Penalty
                gp = compute_gradient_penalty(discriminator, joint_real, joint_fake, device)
                loss_total_d = loss_d + config.lambda_gp * gp
                
                loss_total_d.backward(); discriminator_optimizer.step()
                loss_d_val = loss_total_d.item()

            # --- Train Generator ---
            generator_optimizer.zero_grad()
            joint_fake_g = torch.cat([real_a, fake_b], 1)
            fake_outputs_g = discriminator(joint_fake_g); real_outputs_g = discriminator(joint_real)
            
            loss_g_adv = 0; loss_fm = 0
            for i in range(config.num_discriminators):
                # WGAN Adversarial Loss for Generator: -mean(Critic(Fake))
                loss_g_adv += -fake_outputs_g[i][-1].mean()
                # Feature Matching Loss (compare layers 0 to N-1)
                for j in range(len(fake_outputs_g[i]) - 1):
                    loss_fm += fm_loss_criterion(fake_outputs_g[i][j], real_outputs_g[i][j].detach())
            
            loss_g_l1 = pixel_loss_criterion(fake_b, real_b)
            loss_total = loss_g_adv + (loss_fm * config.feature_matching_weight) + (loss_g_l1 * config.l1_loss_weight)
            
            loss_total.backward(); generator_optimizer.step()

            epoch_metrics["d"] += loss_d_val; epoch_metrics["g_adv"] += loss_g_adv.item()
            epoch_metrics["fm"] += loss_fm.item(); epoch_metrics["l1"] += loss_g_l1.item()
            pbar.set_description(f"E{epoch}"); pbar.set_postfix(D=loss_d_val, G=loss_total.item())

        # Validation
        generator.eval(); val_l1 = 0.0
        with torch.no_grad():
            for va, vb in val_loader:
                val_l1 += pixel_loss_criterion(generator(va.to(device)), vb.to(device)).item()
        
        avg_val = (val_l1 / len(val_loader)) * config.l1_loss_weight
        metrics = [epoch, epoch_metrics['d']/len(train_loader), epoch_metrics['g_adv']/len(train_loader), 
                   epoch_metrics['fm']/len(train_loader), epoch_metrics['l1']/len(train_loader), avg_val]
        with open(log_hist_path, "a") as f: f.write(",".join([f"{m:.6f}" if isinstance(m, float) else str(m) for m in metrics]) + "\n")
        
        print(f"✨ [EPOCH {epoch}] D_Loss: {metrics[1]:.4f} | FM_Loss: {metrics[3]:.4f} | Val L1: {avg_val:.4f}")
        write_progress(epoch, config.epochs, metrics[1], avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")
            print(f"  🏆 New Best HD Model (Val L1: {best_val:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")

    # Final Test (Standalone Trigger)
    print("\n--- Triggering Standalone HD Test Evaluation ---")
    import subprocess
    test_script = pathlib.Path(__file__).parent / "test_pix2pixHD_trajectoryLine.py"
    subprocess.run([sys.executable, str(test_script), "--run_path", str(config.CURRENT_RUN_DIR)])
    
    print(f"🏁 HD Training Finished! Results in {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json", help="Path to training config (e.g. config_train.json)")
    args = parser.parse_args()
    
    # If explicitly provided or default exists, load it
    script_dir = os.path.dirname(__file__)
    # Support both command line path and local script folder path
    cpath = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    
    if os.path.exists(cpath):
        load_config_from_json(cpath)
    else:
        # Fallback for older naming convention
        fallback = os.path.join(script_dir, "config_active.json")
        if os.path.exists(fallback):
            load_config_from_json(fallback)
        
    execute_training()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pathlib
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime

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
    use_label_smoothing = True # Use 0.9 instead of 1.0 for real labels
    
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
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
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
    print(f"🚀 [INIT] Device: {device} | Run: {config.run_name}")

    # Prepare Directories
    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR]: d.mkdir(parents=True, exist_ok=True)

    # Initialize Networks
    generator = GeneratorNetwork(config.input_channels, config.output_channels).to(device)
    discriminator = DiscriminatorNetwork(config.input_channels + config.output_channels).to(device)

    # Resume Training if requested
    if config.resume_checkpoint_path not in ["-", "", None]:
        print(f"🔄 [RESUME] Loading weights from {config.resume_checkpoint_path}")
        try:
            generator.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))
            # Note: Optional to load discriminator too if available
            # discriminator.load_state_dict(...)
        except Exception as e: print(f"⚠️ [WARN] Could not load checkpoint: {e}")

    # Optimizers (Separate Learning Rates)
    generator_optimizer = optim.Adam(generator.parameters(), lr=config.generator_learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(config.beta1, config.beta2))

    # Loss Settings
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()
    pixel_loss_criterion = nn.L1Loss()
    
    # Logs
    log_path = config.LOG_DIR / "training_history.csv"
    with open(log_path, "w") as f:
        f.write("epoch,discriminator_loss,generator_adversarial_loss,generator_l1_loss,validation_l1_loss\n")

    # Data
    train_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "train", config.image_size)
    val_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "validation", config.image_size)
    test_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", config.image_size) # Add Test Dataset

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4) # Add Test Loader

    print(f"📊 [DATA] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Prepare Test Result Dir
    test_result_dir = config.CURRENT_RUN_DIR / "test_results"
    test_result_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    for epoch in range(config.epochs):
        generator.train(); discriminator.train()
        epoch_metrics = {"d": 0.0, "g_adv": 0.0, "g_l1": 0.0}
        
        pbar = tqdm(train_loader, leave=False)
        for real_a, real_b in pbar:
            real_a, real_b = real_a.to(device), real_b.to(device)

            # --- Train Discriminator ---
            fake_b = generator(real_a)
            pred_real = discriminator(real_a, real_b)
            # Label Smoothing: use 0.9 for real instead of 1.0
            real_target = torch.full_like(pred_real, 0.9) if config.use_label_smoothing else torch.ones_like(pred_real)
            loss_d_real = adversarial_loss_criterion(pred_real, real_target)
            
            pred_fake = discriminator(real_a, fake_b.detach())
            loss_d_fake = adversarial_loss_criterion(pred_fake, torch.zeros_like(pred_fake))
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            discriminator_optimizer.zero_grad(); loss_d.backward(); discriminator_optimizer.step()

            # --- Train Generator ---
            pred_fake_g = discriminator(real_a, fake_b)
            loss_g_adv = adversarial_loss_criterion(pred_fake_g, torch.ones_like(pred_fake_g))
            loss_g_l1 = pixel_loss_criterion(fake_b, real_b)
            
            loss_g = loss_g_adv + (loss_g_l1 * config.l1_loss_weight)
            generator_optimizer.zero_grad(); loss_g.backward(); generator_optimizer.step()

            epoch_metrics["d"] += loss_d.item(); epoch_metrics["g_adv"] += loss_g_adv.item(); epoch_metrics["g_l1"] += loss_g_l1.item()
            pbar.set_description(f"Epoch {epoch}"); pbar.set_postfix(D=loss_d.item(), G=loss_g.item())

        # Validation
        generator.eval(); val_l1 = 0.0
        with torch.no_grad():
            for va, vb in val_loader:
                va, vb = va.to(device), vb.to(device)
                val_l1 += pixel_loss_criterion(generator(va), vb).item()
        
        avg_val = (val_l1 / len(val_loader)) * config.l1_loss_weight
        # Log results with full names
        metrics = [epoch, epoch_metrics['d']/len(train_loader), epoch_metrics['g_adv']/len(train_loader), epoch_metrics['g_l1']/len(train_loader), avg_val]
        with open(log_path, "a") as f:
            f.write(",".join([f"{m:.6f}" if isinstance(m, float) else str(m) for m in metrics]) + "\n")
        
        print(f"✨ [EPOCH {epoch}] D_Loss: {metrics[1]:.4f} | Val L1: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")
            print(f"  🏆 New Best (Val L1: {best_val:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")
            with torch.no_grad():
                sample_a, sample_b = next(iter(val_loader))
                res = generator(sample_a.to(device))[0].cpu().numpy().transpose(1, 2, 0)
                img = ((res * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(config.SAMPLE_DIR / f"sample_epoch_{epoch+1}.png")

    # --- Final Test Evaluation ---
    print("\n--- Running Final Test Evaluation ---")
    if (config.CHECKPOINT_DIR / "generator_best.pth").exists():
        generator.load_state_dict(torch.load(config.CHECKPOINT_DIR / "generator_best.pth", map_location=device))
    
    generator.eval()
    test_l1_total = 0
    with torch.no_grad():
        for i, (test_a, test_b) in enumerate(test_loader):
            test_a, test_b = test_a.to(device), test_b.to(device)
            t_fake_b = generator(test_a)
            test_l1_total += pixel_loss_criterion(t_fake_b, test_b).item()
            
            # Save a few test results
            if i < 10:
                res = (t_fake_b[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                img = res.clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(test_result_dir / f"test_result_{i}.png")
                
    avg_test_l1 = (test_l1_total / len(test_loader)) * config.l1_loss_weight
    print(f"✅ Final Test L1 Loss: {avg_test_l1:.4f}")
    print(f"🏁 Trajectory Line Training Finished! Best Val L1: {best_val:.4f}")

if __name__ == "__main__":
    execute_training()

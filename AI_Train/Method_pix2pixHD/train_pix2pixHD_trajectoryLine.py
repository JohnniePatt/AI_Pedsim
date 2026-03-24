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
from datetime import datetime

# ==========================================
# TRAINING CONFIGURATION (Pix2PixHD Version)
# ==========================================
class TrainingConfiguration:
    # 1. Hyperparameters
    generator_learning_rate = 0.0002
    discriminator_learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    batch_size = 2 # Pix2PixHD is memory heavy, start small
    epochs = 100
    
    # 2. Loss Weights (HD specific)
    l1_loss_weight = 10.0 # Standard L1
    feature_matching_weight = 10.0 # Help stabilize HD details
    
    # 3. GAN Options
    num_discriminators = 2 # Multi-scale: 512px and 256px
    use_label_smoothing = True
    
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
    RUNS_ROOT = BASE_DIR / "runs_trajectory"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name
    
    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"
    LOG_DIR = CURRENT_RUN_DIR / "logs"
    SAMPLE_DIR = CURRENT_RUN_DIR / "samples"
    TEST_RESULT_DIR = CURRENT_RUN_DIR / "test_results"

config = TrainingConfiguration()

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

# --- Dataset and Training ---
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

def execute_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 [INIT HD] Device: {device} | Run: {config.run_name}")

    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR, config.TEST_RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    generator = GeneratorNetwork(config.input_channels, config.output_channels).to(device)
    discriminator = DiscriminatorNetwork(config.input_channels + config.output_channels, config.num_discriminators).to(device)

    # Resume
    if config.resume_checkpoint_path not in ["-", "", None]:
        print(f"🔄 [RESUME] Loading {config.resume_checkpoint_path}")
        generator.load_state_dict(torch.load(config.resume_checkpoint_path, map_location=device))

    generator_optimizer = optim.Adam(generator.parameters(), lr=config.generator_learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(config.beta1, config.beta2))

    gan_loss_criterion = nn.BCEWithLogitsLoss()
    pixel_loss_criterion = nn.L1Loss()
    fm_loss_criterion = nn.L1Loss() # Feature Matching uses L1

    train_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "train", config.image_size)
    val_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "validation", config.image_size)
    test_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", config.image_size)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    log_path = config.LOG_DIR / "training_history.csv"
    with open(log_path, "w") as f:
        f.write("epoch,discriminator_loss,generator_adversarial_loss,feature_matching_loss,generator_l1_loss,validation_l1_loss\n")

    best_val = float('inf')
    for epoch in range(config.epochs):
        generator.train(); discriminator.train()
        epoch_metrics = {"d": 0.0, "g_adv": 0.0, "fm": 0.0, "l1": 0.0}
        
        pbar = tqdm(train_loader, leave=False)
        for real_a, real_b in pbar:
            real_a, real_b = real_a.to(device), real_b.to(device)
            fake_b = generator(real_a)

            # --- Train Discriminator ---
            discriminator_optimizer.zero_grad()
            loss_d = 0
            joint_real = torch.cat([real_a, real_b], 1); joint_fake = torch.cat([real_a, fake_b.detach()], 1)
            
            real_outputs = discriminator(joint_real); fake_outputs = discriminator(joint_fake)
            for i in range(config.num_discriminators):
                # Only use final layer output [ -1] for GAN loss
                r_pred = real_outputs[i][-1]; f_pred = fake_outputs[i][-1]
                r_target = torch.full_like(r_pred, 0.9) if config.use_label_smoothing else torch.ones_like(r_pred)
                loss_d += (gan_loss_criterion(r_pred, r_target) + gan_loss_criterion(f_pred, torch.zeros_like(f_pred))) * 0.5
            
            loss_d.backward(); discriminator_optimizer.step()

            # --- Train Generator ---
            generator_optimizer.zero_grad()
            joint_fake_g = torch.cat([real_a, fake_b], 1)
            fake_outputs_g = discriminator(joint_fake_g); real_outputs_g = discriminator(joint_real)
            
            loss_g_adv = 0; loss_fm = 0
            for i in range(config.num_discriminators):
                # GAN Loss
                loss_g_adv += gan_loss_criterion(fake_outputs_g[i][-1], torch.ones_like(fake_outputs_g[i][-1]))
                # Feature Matching Loss (compare layers 0 to N-1)
                for j in range(len(fake_outputs_g[i]) - 1):
                    loss_fm += fm_loss_criterion(fake_outputs_g[i][j], real_outputs_g[i][j].detach())
            
            loss_g_l1 = pixel_loss_criterion(fake_b, real_b)
            loss_total = loss_g_adv + (loss_fm * config.feature_matching_weight) + (loss_g_l1 * config.l1_loss_weight)
            
            loss_total.backward(); generator_optimizer.step()

            epoch_metrics["d"] += loss_d.item(); epoch_metrics["g_adv"] += loss_g_adv.item()
            epoch_metrics["fm"] += loss_fm.item(); epoch_metrics["l1"] += loss_g_l1.item()
            pbar.set_description(f"Epoch {epoch}"); pbar.set_postfix(D=loss_d.item(), G=loss_total.item())

        # Validation
        generator.eval(); val_l1 = 0.0
        with torch.no_grad():
            for va, vb in val_loader:
                val_l1 += pixel_loss_criterion(generator(va.to(device)), vb.to(device)).item()
        
        avg_val = (val_l1 / len(val_loader)) * config.l1_loss_weight
        metrics = [epoch, epoch_metrics['d']/len(train_loader), epoch_metrics['g_adv']/len(train_loader), 
                   epoch_metrics['fm']/len(train_loader), epoch_metrics['l1']/len(train_loader), avg_val]
        with open(log_path, "a") as f: f.write(",".join([f"{m:.6f}" if isinstance(m, float) else str(m) for m in metrics]) + "\n")
        
        print(f"✨ [EPOCH {epoch}] D_Loss: {metrics[1]:.4f} | FM_Loss: {metrics[3]:.4f} | Val L1: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / "generator_best.pth")
            print(f"  🏆 New Best HD Model (Val L1: {best_val:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIR / f"generator_epoch_{epoch+1}.pth")
            with torch.no_grad():
                sample_a, _ = next(iter(val_loader))
                res = generator(sample_a.to(device))[0].cpu().numpy().transpose(1, 2, 0)
                img = ((res * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(config.SAMPLE_DIR / f"sample_epoch_{epoch+1}.png")

    # Final Test
    print("\n--- Running Final HD Test Evaluation ---")
    if (config.CHECKPOINT_DIR / "generator_best.pth").exists():
        generator.load_state_dict(torch.load(config.CHECKPOINT_DIR / "generator_best.pth", map_location=device))
    generator.eval(); test_l1 = 0
    with torch.no_grad():
        for i, (ta, tb) in enumerate(test_loader):
            tfb = generator(ta.to(device))
            test_l1 += pixel_loss_criterion(tfb, tb.to(device)).item()
            if i < 10:
                img = ((tfb[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(config.TEST_RESULT_DIR / f"test_result_{i}.png")
    print(f"🏁 HD Training Finished! Best Val L1: {best_val:.4f}")

if __name__ == "__main__":
    execute_training()

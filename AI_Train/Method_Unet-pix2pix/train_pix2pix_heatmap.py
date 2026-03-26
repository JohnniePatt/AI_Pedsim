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

# --- GAN Architecture: U-Net Generator ---
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
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

    def forward(self, x):
        x = self.model(x)
        if self.use_dropout: x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # Encoder
        self.d1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.d2 = UNetBlock(64, 128, down=True)
        self.d3 = UNetBlock(128, 256, down=True)
        self.d4 = UNetBlock(256, 512, down=True)
        self.d5 = UNetBlock(512, 512, down=True)
        self.d6 = UNetBlock(512, 512, down=True)
        self.d7 = UNetBlock(512, 512, down=True)
        self.d8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        
        # Decoder
        self.u1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.u2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.u3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.u4 = UNetBlock(1024, 512, down=False)
        self.u5 = UNetBlock(1024, 256, down=False)
        self.u6 = UNetBlock(512, 128, down=False)
        self.u7 = UNetBlock(256, 64, down=False)
        self.u8 = nn.Sequential(nn.ConvTranspose2d(128, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        return self.u8(torch.cat([u7, d1], 1))

# --- PatchGAN Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, in_ch=4): # A (3) + B (1) = 4
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

# --- Dataset Handler ---
class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, subset="train"):
        self.dir_A = pathlib.Path(root_dir) / "A" / subset
        self.dir_B = pathlib.Path(root_dir) / "B" / subset
        
        # Get all png files in the subset directory
        self.files = sorted([f.name for f in self.dir_A.glob("*.png")])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img_a = Image.open(self.dir_A / file_name).convert("RGB")
        img_b = Image.open(self.dir_B / file_name).convert("L") # Heatmap is Gray
        return self.transform(img_a), self.transform(img_b)

# --- Training Script ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Training on: {device_status}\n{'='*50}\n")

    # Paths
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    DATA_DIR = BASE_DIR.parent.parent / "Topo_2" / "heatmap_density" / "Cleandata_1"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    OUTPUT_DIR = BASE_DIR / "outputs"
    LOG_DIR = BASE_DIR / "log_loss"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Log file init
    log_file = LOG_DIR / "training_history.csv"
    with open(log_file, "w") as f:
        f.write("epoch,d_loss,g_gan_loss,l1_mae_loss,mse_loss,val_l1_loss\n")

    # Models
    net_g = Generator(in_ch=3, out_ch=1).to(device)
    net_d = Discriminator(in_ch=4).to(device)

    # Optimizers
    opt_g = optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss Functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()  # Also known as MAE
    criterion_mse = nn.MSELoss()

    # Data
    if not (DATA_DIR / "A" / "train").exists():
        print(f"ERROR: Data subdirectories not found at {DATA_DIR}")
        print("Please check your folder structure (A/train, B/train, etc.)")
        return

    train_dataset = Pix2PixDataset(DATA_DIR, subset="train")
    val_dataset = Pix2PixDataset(DATA_DIR, subset="validation")
    test_dataset = Pix2PixDataset(DATA_DIR, subset="test")

    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    epochs = 100
    L1_LAMBDA = 100
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- Training Loop ---
        net_g.train()
        net_d.train()
        
        train_metrics = {"d": 0, "g_gan": 0, "g_l1": 0, "g_mse": 0}
        
        train_loop = tqdm(train_loader, leave=False)
        for i, (real_a, real_b) in enumerate(train_loop):
            real_a, real_b = real_a.to(device), real_b.to(device)

            # --- Train Discriminator ---
            fake_b = net_g(real_a)
            
            # Real
            pred_real = net_d(real_a, real_b)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Fake
            pred_fake = net_d(real_a, fake_b.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # --- Train Generator ---
            pred_fake = net_d(real_a, fake_b)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake_b, real_b)
            loss_g_mse = criterion_mse(fake_b, real_b)
            
            total_loss_g = loss_g_gan + (loss_g_l1 * L1_LAMBDA)
            
            opt_g.zero_grad()
            total_loss_g.backward()
            opt_g.step()

            # Accumulate metrics
            train_metrics["d"] += loss_d.item()
            train_metrics["g_gan"] += loss_g_gan.item()
            train_metrics["g_l1"] += loss_g_l1.item()
            train_metrics["g_mse"] += loss_g_mse.item()

            train_loop.set_description(f"Epoch [{epoch}/{epochs}] Train")
            train_loop.set_postfix(D_loss=loss_d.item(), G_loss=total_loss_g.item())

        # Average training metrics for the epoch
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        # --- Validation Loop ---
        net_g.eval()
        val_l1_total = 0
        with torch.no_grad():
            for val_a, val_b in val_loader:
                val_a, val_b = val_a.to(device), val_b.to(device)
                v_fake_b = net_g(val_a)
                val_l1_total += criterion_l1(v_fake_b, val_b).item()
        
        avg_val_l1 = (val_l1_total / len(val_loader)) * L1_LAMBDA
        print(f"Epoch [{epoch}/{epochs}] - Train G_GAN: {train_metrics['g_gan']:.4f} | Val L1 Loss: {avg_val_l1:.4f}")

        # Write Log to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch},{train_metrics['d']:.6f},{train_metrics['g_gan']:.6f},{train_metrics['g_l1']:.6f},{train_metrics['g_mse']:.6f},{avg_val_l1:.6f}\n")

        # Save Best Model
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            torch.save(net_g.state_dict(), CHECKPOINT_DIR / "gen_best.pth")
            print(f"  --> Saved Best Model (L1: {best_val_loss:.4f})")

        # Save Regular Checkpoint & Samples
        if (epoch + 1) % 10 == 0:
            torch.save(net_g.state_dict(), CHECKPOINT_DIR / f"gen_epoch_{epoch+1}.pth")
            
            # Save visual result from Validation Set for monitoring
            with torch.no_grad():
                val_a_sample, val_b_sample = next(iter(val_loader))
                val_a_sample = val_a_sample.to(device)
                sample_out = net_g(val_a_sample)
                # Denormalize
                res = (sample_out[0].cpu().numpy() * 0.5 + 0.5) * 255
                Image.fromarray(res[0].astype(np.uint8)).save(OUTPUT_DIR / f"sample_epoch_{epoch+1}.png")

    # --- Final Test Evaluation ---
    print("\n--- Running Final Test Evaluation ---")
    if (CHECKPOINT_DIR / "gen_best.pth").exists():
        net_g.load_state_dict(torch.load(CHECKPOINT_DIR / "gen_best.pth"))
    
    net_g.eval()
    test_l1_total = 0
    with torch.no_grad():
        for i, (test_a, test_b) in enumerate(test_loader):
            test_a, test_b = test_a.to(device), test_b.to(device)
            t_fake_b = net_g(test_a)
            test_l1_total += criterion_l1(t_fake_b, test_b).item()
            
            # Save a few test results
            if i < 5:
                res = (t_fake_b[0].cpu().numpy() * 0.5 + 0.5) * 255
                Image.fromarray(res[0].astype(np.uint8)).save(OUTPUT_DIR / f"test_result_{i}.png")
                
    avg_test_l1 = (test_l1_total / len(test_loader)) * L1_LAMBDA
    print(f"Final Test L1 Loss: {avg_test_l1:.4f}")
    print(f"Training and Testing finished! Best Val L1: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()

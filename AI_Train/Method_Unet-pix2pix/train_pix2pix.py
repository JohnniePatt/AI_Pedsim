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
    def __init__(self, root_dir):
        self.dir_A = pathlib.Path(root_dir) / "A"
        self.dir_B = pathlib.Path(root_dir) / "B"
        self.files = sorted([f.relative_to(self.dir_A) for f in self.dir_A.rglob("*.png")])
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path_rel = self.files[idx]
        img_a = Image.open(self.dir_A / path_rel).convert("RGB")
        img_b = Image.open(self.dir_B / path_rel).convert("L") # Heatmap is Gray
        return self.transform(img_a), self.transform(img_b)

# --- Training Script ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Paths
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    DATA_DIR = BASE_DIR.parent.parent / "Topo_2" / "heatmap_density" / "Cleandata_1"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    OUTPUT_DIR = BASE_DIR / "outputs"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Models
    net_g = Generator(in_ch=3, out_ch=1).to(device)
    net_d = Discriminator(in_ch=4).to(device)

    # Optimizers
    opt_g = optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss Functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    # Data
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found at {DATA_DIR}")
        print("Please run Prepare_image_input.py first.")
        return

    dataset = Pix2PixDataset(DATA_DIR)
    if len(dataset) == 0:
        print(f"ERROR: No images found in {DATA_DIR}")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    epochs = 100
    L1_LAMBDA = 100

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for i, (real_a, real_b) in enumerate(loop):
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
            loss_g_l1 = criterion_l1(fake_b, real_b) * L1_LAMBDA
            
            loss_g = loss_g_gan + loss_g_l1
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(D_loss=loss_d.item(), G_loss=loss_g.item())

        # Save Checkpoint & Samples
        if (epoch + 1) % 10 == 0:
            torch.save(net_g.state_dict(), CHECKPOINT_DIR / f"gen_epoch_{epoch+1}.pth")
            
            # Save visual result
            with torch.no_grad():
                # Denormalize
                res = (fake_b[0].cpu().numpy() * 0.5 + 0.5) * 255
                Image.fromarray(res[0].astype(np.uint8)).save(OUTPUT_DIR / f"sample_epoch_{epoch+1}.png")

    print("Training finished!")

if __name__ == "__main__":
    train()

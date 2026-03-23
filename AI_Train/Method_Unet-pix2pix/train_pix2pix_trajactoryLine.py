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

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
class TrainingConfiguration:
    # Hyperparameters
    learning_rate = 0.002
    beta1 = 0.5
    beta2 = 0.999
    batch_size = 4
    epochs = 100
    l1_loss_weight = 100.0  # Lambda for L1 Loss
    
    # Image Settings
    image_size = 512
    input_channels = 3  # RGB environment mask
    output_channels = 3  # RGB trajectory line plot
    
    # Paths
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = BASE_DIR.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "Topo_2" / "trajectory_line_dataset" / "Cleandata_1"
    
    CHECKPOINT_DIRECTORY = BASE_DIR / "checkpoints_traj"
    OUTPUT_IMAGE_DIRECTORY = BASE_DIR / "outputs_traj"
    LOG_DIRECTORY = BASE_DIR / "log_loss_traj"

# Initialize Config
config = TrainingConfiguration()

# --- GAN Architecture: U-Net Generator ---
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=True, use_dropout=False):
        super().__init__()
        if downsampling:
            self.operation = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.operation = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.operation(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        return x

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder (Downsampling)
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.encoder2 = UNetBlock(64, 128, downsampling=True)
        self.encoder3 = UNetBlock(128, 256, downsampling=True)
        self.encoder4 = UNetBlock(256, 512, downsampling=True)
        self.encoder5 = UNetBlock(512, 512, downsampling=True)
        self.encoder6 = UNetBlock(512, 512, downsampling=True)
        self.encoder7 = UNetBlock(512, 512, downsampling=True)
        self.encoder8 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.ReLU())
        
        # Decoder (Upsampling with skip connections)
        self.decoder1 = UNetBlock(512, 512, downsampling=False, use_dropout=True)
        self.decoder2 = UNetBlock(1024, 512, downsampling=False, use_dropout=True)
        self.decoder3 = UNetBlock(1024, 512, downsampling=False, use_dropout=True)
        self.decoder4 = UNetBlock(1024, 512, downsampling=False)
        self.decoder5 = UNetBlock(1024, 256, downsampling=False)
        self.decoder6 = UNetBlock(512, 128, downsampling=False)
        self.decoder7 = UNetBlock(256, 64, downsampling=False)
        self.decoder8 = nn.Sequential(nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        # Encoder forward pass
        d1 = self.encoder1(x)
        d2 = self.encoder2(d1)
        d3 = self.encoder3(d2)
        d4 = self.encoder4(d3)
        d5 = self.encoder5(d4)
        d6 = self.encoder6(d5)
        d7 = self.encoder7(d6)
        d8 = self.encoder8(d7)
        
        # Decoder forward pass with skip connections
        u1 = self.decoder1(d8)
        u2 = self.decoder2(torch.cat([u1, d7], dim=1))
        u3 = self.decoder3(torch.cat([u2, d6], dim=1))
        u4 = self.decoder4(torch.cat([u3, d5], dim=1))
        u5 = self.decoder5(torch.cat([u4, d4], dim=1))
        u6 = self.decoder6(torch.cat([u5, d3], dim=1))
        u7 = self.decoder7(torch.cat([u6, d2], dim=1))
        return self.decoder8(torch.cat([u7, d1], dim=1))

# --- PatchGAN Discriminator ---
class DiscriminatorNetwork(nn.Module):
    def __init__(self, joint_channels=6): # Combined Input A (3) and Target B (3)
        super().__init__()
        self.structure = nn.Sequential(
            nn.Conv2d(joint_channels, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input_image, target_image):
        return self.structure(torch.cat([input_image, target_image], dim=1))

# --- Dataset Handler ---
class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="train", image_size=512):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        
        self.file_list = sorted([f.name for f in self.directory_A.glob("*.png")])
        
        self.augmentation_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        current_file = self.file_list[index]
        input_image = Image.open(self.directory_A / current_file).convert("RGB")
        target_image = Image.open(self.directory_B / current_file).convert("RGB") # Trajectory Line is now RGB
        return self.augmentation_transforms(input_image), self.augmentation_transforms(target_image)

# --- Training Execution ---
def execute_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Starting Training on: {device}")

    # Create distinct directories for trajectory lines
    os.makedirs(config.CHECKPOINT_DIRECTORY, exist_ok=True)
    os.makedirs(config.OUTPUT_IMAGE_DIRECTORY, exist_ok=True)
    os.makedirs(config.LOG_DIRECTORY, exist_ok=True)

    # Initialize log files
    training_log_path = config.LOG_DIRECTORY / "training_history.csv"
    if not training_log_path.exists():
        with open(training_log_path, "w") as f:
            f.write("epoch,discriminator_loss,generator_adversarial_loss,l1_mae_loss,val_l1_loss\n")

    # Initialize Networks
    generator = GeneratorNetwork(in_channels=config.input_channels, out_channels=config.output_channels).to(device)
    discriminator = DiscriminatorNetwork(joint_channels=(config.input_channels + config.output_channels)).to(device)

    # Initialize Optimizers with explicit learning rates
    generator_optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

    # Loss Functions
    binary_cross_entropy_loss = nn.BCEWithLogitsLoss()
    l1_loss_criterion = nn.L1Loss()

    # Dataset Verification
    if not (config.DATASET_ROOT / "A" / "train").exists():
        print(f"❌ ERROR: Dataset not found at {config.DATASET_ROOT}")
        return

    # Data Loaders
    training_dataset = Pix2PixTrajectoryDataset(config.DATASET_ROOT, subset="train", image_size=config.image_size)
    validation_dataset = Pix2PixTrajectoryDataset(config.DATASET_ROOT, subset="validation", image_size=config.image_size)
    
    print(f"📊 Dataset: {len(training_dataset)} train, {len(validation_dataset)} validation")

    training_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    best_validation_loss = float('inf')

    # Training Loop
    for epoch in range(config.epochs):
        generator.train()
        discriminator.train()
        
        epoch_metrics = {"discriminator": 0.0, "gen_adv": 0.0, "gen_l1": 0.0}
        
        progress_bar = tqdm(training_loader, leave=False)
        for i, (real_inputs, real_targets) in enumerate(progress_bar):
            real_inputs, real_targets = real_inputs.to(device), real_targets.to(device)

            # --- Update Discriminator ---
            # Produce fake images
            fake_targets = generator(real_inputs)
            
            # Discriminator loss for Real images
            real_predictions = discriminator(real_inputs, real_targets)
            real_loss = binary_cross_entropy_loss(real_predictions, torch.ones_like(real_predictions))
            
            # Discriminator loss for Fake images
            fake_predictions = discriminator(real_inputs, fake_targets.detach())
            fake_loss = binary_cross_entropy_loss(fake_predictions, torch.zeros_like(fake_predictions))
            
            discriminator_total_loss = (real_loss + fake_loss) * 0.5
            
            discriminator_optimizer.zero_grad()
            discriminator_total_loss.backward()
            discriminator_optimizer.step()

            # --- Update Generator ---
            # Generator wants to fool the discriminator
            fake_predictions_adv = discriminator(real_inputs, fake_targets)
            generator_adversarial_loss = binary_cross_entropy_loss(fake_predictions_adv, torch.ones_like(fake_predictions_adv))
            generator_l1_loss = l1_loss_criterion(fake_targets, real_targets)
            
            generator_total_loss = generator_adversarial_loss + (generator_l1_loss * config.l1_loss_weight)
            
            generator_optimizer.zero_grad()
            generator_total_loss.backward()
            generator_optimizer.step()

            # Track Metrics
            epoch_metrics["discriminator"] += discriminator_total_loss.item()
            epoch_metrics["gen_adv"] += generator_adversarial_loss.item()
            epoch_metrics["gen_l1"] += generator_l1_loss.item()

            progress_bar.set_description(f"Epoch [{epoch}/{config.epochs}]")
            progress_bar.set_postfix(D_loss=discriminator_total_loss.item(), G_loss=generator_total_loss.item())

        # Validation Step
        generator.eval()
        val_l1_accumulator = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in validation_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                predicted_targets = generator(val_inputs)
                val_l1_accumulator += l1_loss_criterion(predicted_targets, val_targets).item()
        
        average_val_l1 = (val_l1_accumulator / len(validation_loader)) * config.l1_loss_weight
        
        # Calculate Averages for Logging
        avg_d_loss = epoch_metrics["discriminator"] / len(training_loader)
        avg_g_adv = epoch_metrics["gen_adv"] / len(training_loader)
        avg_g_l1 = epoch_metrics["gen_l1"] / len(training_loader)

        print(f"✨ Epoch {epoch} Results | D_Loss: {avg_d_loss:.4f} | Val L1: {average_val_l1:.4f}")

        # Update CSV Log
        with open(training_log_path, "a") as f:
            f.write(f"{epoch},{avg_d_loss:.6f},{avg_g_adv:.6f},{avg_g_l1:.6f},{average_val_l1:.6f}\n")

        # Save Best Model
        if average_val_l1 < best_validation_loss:
            best_validation_loss = average_val_l1
            torch.save(generator.state_dict(), config.CHECKPOINT_DIRECTORY / "generator_best.pth")
            print(f"  🏆 New Best Model Saved (L1: {best_validation_loss:.4f})")

        # Save Periodical Samples
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), config.CHECKPOINT_DIRECTORY / f"generator_epoch_{epoch+1}.pth")
            
            with torch.no_grad():
                val_sample_batch = next(iter(validation_loader))
                sample_inputs = val_sample_batch[0].to(device)
                sample_results = generator(sample_inputs)
                
                # Convert back to uint8 image (denormalize)
                # Take the first image in the batch
                output_image_tensor = (sample_results[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                output_image_tensor = output_image_tensor.clip(0, 255).astype(np.uint8)
                Image.fromarray(output_image_tensor).save(config.OUTPUT_IMAGE_DIRECTORY / f"sample_epoch_{epoch+1}.png")

    print("\n🏁 Trajectory Line Training Finished!")

if __name__ == "__main__":
    execute_training()

import os
import json
import pathlib
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from pix2pix_wgangp_common import (
    Pix2PixDataset,
    UNetGenerator,
    PatchGANDiscriminator,
    compute_gradient_penalty,
    convert_bw_to_colorjet,
    get_device,
    make_run_dirs,
    resolve_path,
    resolve_project_root
)

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = script_dir / "config_train.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    project_root = resolve_project_root(script_dir)
    dataset_root = resolve_path(cfg["dataset_root"], project_root)
    run_paths = make_run_dirs(script_dir, method_name="Method_pix2pix_WGAN-GP")

    device = get_device()
    print(f"🚀 Training Method_pix2pix_WGAN-GP on device: {device}")
    print(f"📁 Output Run Directory: {run_paths['CURRENT_RUN_DIR']}")

    # Save run config snapshot
    with open(run_paths["CURRENT_RUN_DIR"] / "run_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Datasets and Loaders
    image_size = cfg.get("image_size", 512)
    train_dataset = Pix2PixDataset(dataset_root, split="train", image_size=image_size)
    val_dataset = Pix2PixDataset(dataset_root, split="val", image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # Initialize Networks
    netG = UNetGenerator(input_nc=3, output_nc=1, num_downs=8, ngf=64).to(device)
    netD = PatchGANDiscriminator(input_nc=4, ndf=64, n_layers=3).to(device)

    # Optimizers
    lr = cfg.get("lr", 0.0002)
    b1 = cfg.get("b1", 0.5)
    b2 = cfg.get("b2", 0.999)
    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(b1, b2))

    criterion_l1 = nn.L1Loss()
    lambda_l1 = cfg.get("lambda_l1", 100.0)
    lambda_gp = cfg.get("lambda_gp", 10.0)

    epochs = cfg.get("epochs", 50)
    best_val_loss = float("inf")

    history_csv = run_paths["LOG_DIR"] / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_g_loss", "train_d_loss", "val_l1_loss"])

    for epoch in range(1, epochs + 1):
        netG.train()
        netD.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{epochs:03d}]")
        for real_a, real_b, _ in pbar:
            real_a, real_b = real_a.to(device), real_b.to(device)

            # ---------------------
            #  Train Discriminator (WGAN-GP Critic)
            # ---------------------
            optimizer_D.zero_grad()

            fake_b = netG(real_a)
            real_pair = torch.cat([real_a, real_b], dim=1)
            fake_pair = torch.cat([real_a, fake_b.detach()], dim=1)

            pred_real = netD(real_pair)
            pred_fake = netD(fake_pair)

            # WGAN Critic Loss: E[D(fake)] - E[D(real)]
            d_loss_wasserstein = torch.mean(pred_fake) - torch.mean(pred_real)
            
            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(netD, real_b, fake_b.detach(), real_a, device)
            
            d_loss = d_loss_wasserstein + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()

            fake_pair_g = torch.cat([real_a, fake_b], dim=1)
            pred_fake_g = netD(fake_pair_g)

            # Generator Loss: -E[D(fake)] + lambda_l1 * L1(fake, real)
            g_loss_adv = -torch.mean(pred_fake_g)
            g_loss_l1 = criterion_l1(fake_b, real_b) * lambda_l1
            g_loss = g_loss_adv + g_loss_l1

            g_loss.backward()
            optimizer_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            pbar.set_postfix({"D": f"{d_loss.item():.4f}", "G": f"{g_loss.item():.4f}"})

        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)

        # Validation
        netG.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for real_a, real_b, _ in val_loader:
                real_a, real_b = real_a.to(device), real_b.to(device)
                fake_b = netG(real_a)
                val_l1 += criterion_l1(fake_b, real_b).item()

        avg_val_l1 = val_l1 / len(val_loader)
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | Val L1: {avg_val_l1:.4f}")

        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_g_loss, avg_d_loss, avg_val_l1])

        # Checkpoint Saving
        checkpoint_data = {
            "epoch": epoch,
            "netG_state_dict": netG.state_dict(),
            "netD_state_dict": netD.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
            "val_loss": avg_val_l1,
            "config": cfg,
        }

        torch.save(checkpoint_data, run_paths["CHECKPOINT_DIR"] / "final.pt")
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            torch.save(checkpoint_data, run_paths["CHECKPOINT_DIR"] / "best_loss.pt")

        # Save Sample Grids every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                sample_a, sample_b, _ = next(iter(val_loader))
                sample_a, sample_b = sample_a.to(device), sample_b.to(device)
                pred_b = netG(sample_a)

                colorjet_np = convert_bw_to_colorjet(pred_b[0, 0])
                colorjet_tensor = torch.from_numpy(colorjet_np).permute(2, 0, 1).float() / 255.0

                grid = torch.cat([sample_a[0].cpu(), sample_b[0].repeat(3, 1, 1).cpu(), pred_b[0].repeat(3, 1, 1).cpu(), colorjet_tensor], dim=2)
                save_image(grid, run_paths["SAMPLE_DIR"] / f"sample_epoch_{epoch:03d}.png")

    print(f"✅ Training Method_pix2pix_WGAN-GP completed successfully!")

if __name__ == "__main__":
    main()

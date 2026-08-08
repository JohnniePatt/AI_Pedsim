import argparse
import json
import pathlib
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from pix2pix_common import (
    Pix2PixDataset,
    UNetGenerator,
    PatchGANDiscriminator,
    convert_bw_to_colorjet,
    get_device,
    resolve_path,
    resolve_project_root,
    make_run_dirs
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = script_dir / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    project_root = resolve_project_root(script_dir)
    dataset_root = resolve_path(cfg["dataset_root"], project_root)
    
    device = get_device()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Pix2Pix Training on: {device_name}\n{'='*50}\n")

    image_size = cfg.get("image_size", 256)
    target_channels = cfg.get("target_channels", 1)
    l1_lambda = cfg.get("l1_lambda", 100.0)
    lr = cfg.get("learning_rate", 2e-4)
    batch_size = cfg.get("batch_size", 8)
    epochs = cfg.get("epochs", 50)
    patience = cfg.get("early_stopping_patience", 12)

    # Initialize Networks
    net_g = UNetGenerator(in_ch=3, out_ch=target_channels).to(device)
    net_d = PatchGANDiscriminator(in_ch=3 + target_channels).to(device)

    # Optimizers
    opt_g = optim.Adam(net_g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(net_d.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    train_dataset = Pix2PixDataset(dataset_root, "train", image_size, target_channels)
    val_dataset = Pix2PixDataset(dataset_root, "validation", image_size, target_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    run_dirs = make_run_dirs(script_dir)
    
    # Save config snapshot
    with open(run_dirs["CURRENT_RUN_DIR"] / "run_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    log_csv = run_dirs["LOG_DIR"] / "training_history.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "d_loss", "g_gan_loss", "g_l1_loss", "val_l1_loss"])

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        net_g.train()
        net_d.train()

        running_d_loss = 0.0
        running_g_gan_loss = 0.0
        running_g_l1_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{epochs:03d}] Train", leave=False)
        for real_a, real_b, _ in train_loop:
            real_a, real_b = real_a.to(device), real_b.to(device)

            # -------------------------------------------------------------
            # 1. Train Discriminator
            # -------------------------------------------------------------
            opt_d.zero_grad()

            fake_b = net_g(real_a)

            # Real Pair Loss
            pred_real = net_d(real_a, real_b)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            # Fake Pair Loss
            pred_fake = net_d(real_a, fake_b.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # -------------------------------------------------------------
            # 2. Train Generator
            # -------------------------------------------------------------
            opt_g.zero_grad()

            pred_fake = net_d(real_a, fake_b)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake_b, real_b)

            total_loss_g = loss_g_gan + (loss_g_l1 * l1_lambda)
            total_loss_g.backward()
            opt_g.step()

            running_d_loss += loss_d.item()
            running_g_gan_loss += loss_g_gan.item()
            running_g_l1_loss += loss_g_l1.item()

            train_loop.set_postfix(D_loss=loss_d.item(), G_loss=total_loss_g.item(), L1=loss_g_l1.item())

        epoch_d_loss = running_d_loss / len(train_loader)
        epoch_g_gan_loss = running_g_gan_loss / len(train_loader)
        epoch_g_l1_loss = running_g_l1_loss / len(train_loader)

        # -------------------------------------------------------------
        # Validation Loop
        # -------------------------------------------------------------
        net_g.eval()
        val_l1_loss = 0.0
        with torch.no_grad():
            for val_a, val_b, _ in val_loader:
                val_a, val_b = val_a.to(device), val_b.to(device)
                val_fake_b = net_g(val_a)
                val_l1_loss += criterion_l1(val_fake_b, val_b).item()

        val_l1_loss /= len(val_loader)

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | D Loss: {epoch_d_loss:.4f} | G GAN: {epoch_g_gan_loss:.4f} | G L1: {epoch_g_l1_loss:.4f} | Val L1: {val_l1_loss:.4f}")

        # Log CSV
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_d_loss, epoch_g_gan_loss, epoch_g_l1_loss, val_l1_loss])

        # Progress JSON
        progress_data = {
            "epoch": epoch,
            "total_epochs": epochs,
            "percentage": round((epoch / epochs) * 100.0, 2),
            "loss": epoch_g_l1_loss,
            "val_l1_loss": val_l1_loss
        }
        with open(run_dirs["CURRENT_RUN_DIR"] / "progress.json", "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=4)

        # Save Sample Visual Grid
        with torch.no_grad():
            sample_a, sample_b, _ = next(iter(val_loader))
            sample_a, sample_b = sample_a.to(device), sample_b.to(device)
            sample_pred = net_g(sample_a)

            a_arr = (sample_a.cpu().numpy()[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            b_arr = (sample_b.cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.uint8)
            p_arr = (sample_pred.cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.uint8)
            colorjet_arr = convert_bw_to_colorjet(sample_pred[0, 0])

            a_img = Image.fromarray(a_arr, mode="RGB")
            b_img = Image.fromarray(b_arr, mode="L").convert("RGB")
            p_img = Image.fromarray(p_arr, mode="L").convert("RGB")
            c_img = Image.fromarray(colorjet_arr, mode="RGB")

            w, h = a_img.size
            grid = Image.new("RGB", (w * 4, h))
            grid.paste(a_img, (0, 0))
            grid.paste(b_img, (w, 0))
            grid.paste(p_img, (w * 2, 0))
            grid.paste(c_img, (w * 3, 0))
            grid.save(run_dirs["SAMPLE_DIR"] / f"epoch_{epoch:03d}.png")

        # Save Checkpoints
        state = {
            "epoch": epoch,
            "model_state_dict": net_g.state_dict(),
            "discriminator_state_dict": net_d.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            "config": cfg
        }

        if val_l1_loss < best_val_loss:
            best_val_loss = val_l1_loss
            no_improve = 0
            torch.save(state, run_dirs["CHECKPOINT_DIR"] / "best_loss.pt")
            print(f"  -> Saved new best_loss.pt (Val L1: {best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"\nTraining Complete! Best Val L1 Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

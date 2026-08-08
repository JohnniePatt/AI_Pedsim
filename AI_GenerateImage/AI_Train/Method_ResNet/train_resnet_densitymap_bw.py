import os
import json
import csv
import pathlib
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import numpy as np

from resnet_common import (
    ResNetDataset,
    ResNetGenerator,
    convert_bw_to_colorjet,
    get_device,
    make_run_dirs,
    resolve_path,
    resolve_project_root
)

def main():
    parser = argparse.ArgumentParser(description="Train Method_ResNet (9-Block ResNet DensityMap BW)")
    parser.add_argument("--config", type=str, default="config_train.json", help="Path to config file")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = script_dir / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    project_root = resolve_project_root(script_dir)
    dataset_root = resolve_path(cfg["dataset_root"], project_root)
    run_paths = make_run_dirs(script_dir, "Method_ResNet")
    run_dir = run_paths["CURRENT_RUN_DIR"]

    print("=" * 50)
    print(f"🛰️ [SYSTEM] ResNet Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"📁 Run Output Dir: {run_dir}")
    print("=" * 50)

    # Save run config snapshot
    snapshot_config = dict(cfg)
    snapshot_config["dataset_root"] = str(dataset_root)
    with open(run_dir / "run_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot_config, f, indent=4)

    device = get_device()
    epochs = cfg.get("epochs", 50)
    batch_size = cfg.get("batch_size", 4)
    lr = cfg.get("learning_rate", 0.0002)
    l1_weight = cfg.get("l1_weight", 100.0)
    num_resnet_blocks = cfg.get("num_resnet_blocks", 9)
    image_size = cfg.get("image_size", 256)
    target_channels = cfg.get("target_channels", 1)

    train_dataset = ResNetDataset(dataset_root, "train", image_size, target_channels)
    val_dataset = ResNetDataset(dataset_root, "val", image_size, target_channels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    generator = ResNetGenerator(in_ch=3, out_ch=target_channels, num_resnet_blocks=num_resnet_blocks).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_l1 = nn.L1Loss()

    history_csv = run_paths["LOG_DIR"] / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_l1", "val_l1"])

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        generator.train()
        epoch_l1 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{epochs:03d}]")
        for real_a, real_b, _ in pbar:
            real_a, real_b = real_a.to(device), real_b.to(device)

            optimizer.zero_grad()
            fake_b = generator(real_a)
            loss_l1 = criterion_l1(fake_b, real_b) * l1_weight
            loss_l1.backward()
            optimizer.step()

            epoch_l1 += loss_l1.item()
            pbar.set_postfix({"L1": f"{loss_l1.item():.4f}"})

        avg_l1 = epoch_l1 / len(train_loader)

        # Validation
        generator.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for real_a, real_b, _ in val_loader:
                real_a, real_b = real_a.to(device), real_b.to(device)
                fake_b = generator(real_a)
                val_l1 += criterion_l1(fake_b, real_b).item()

        avg_val_l1 = val_l1 / len(val_loader)

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | L1 Loss: {avg_l1:.4f} | Val L1: {avg_val_l1:.4f}")

        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_l1, avg_val_l1])

        # Checkpoint Saving
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
                pred_b = generator(sample_a)

                colorjet_np = convert_bw_to_colorjet(pred_b[0, 0])
                colorjet_tensor = torch.from_numpy(colorjet_np).permute(2, 0, 1).float() / 255.0

                grid = torch.cat([sample_a[0].cpu(), sample_b[0].repeat(3, 1, 1).cpu(), pred_b[0].repeat(3, 1, 1).cpu(), colorjet_tensor], dim=2)
                save_image(grid, run_paths["SAMPLE_DIR"] / f"sample_epoch_{epoch:03d}.png")

    print(f"\nTraining Complete! Best Val L1 Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

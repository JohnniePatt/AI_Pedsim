import argparse
import json
import pathlib
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from plainunet_common import PlainUNetDataset, PlainUNet, get_device, resolve_path, resolve_project_root, make_run_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = script_dir / args.config
    with open(config_path) as f: cfg = json.load(f)

    project_root = resolve_project_root(script_dir)
    dataset_root = resolve_path(cfg["dataset_root"], project_root)
    
    device = get_device()
    model = PlainUNet(base=cfg.get("base_filters", 32), drop=cfg.get("dropout", 0.1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 2e-4))
    criterion = torch.nn.L1Loss()

    train_loader = DataLoader(PlainUNetDataset(dataset_root, "train", cfg["image_size"]), batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PlainUNetDataset(dataset_root, "validation", cfg["image_size"]), batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    run_dirs = make_run_dirs(script_dir)
    best_loss = float("inf")
    patience = cfg.get("early_stopping_patience", 12)
    no_improve = 0

    # Save config snapshot
    with open(run_dirs["CURRENT_RUN_DIR"] / "run_config_snapshot.json", "w") as f: json.dump(cfg, f, indent=4)

    with open(run_dirs["LOG_DIR"] / "training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "l1", "val_l1"])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0
        for a, b, _ in tqdm(train_loader, desc=f"Train E{epoch}", leave=False):
            a, b = a.to(device), b.to(device)
            optimizer.zero_grad()
            pred = torch.sigmoid(model(a))
            loss = criterion(pred, b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for a, b, _ in tqdm(val_loader, desc=f"Val E{epoch}", leave=False):
                a, b = a.to(device), b.to(device)
                pred = torch.sigmoid(model(a))
                val_loss += criterion(pred, b).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch:03d}/{cfg['epochs']:03d} | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f}")

        with open(run_dirs["LOG_DIR"] / "training_history.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        with torch.no_grad():
            sample_a, sample_b, _ = next(iter(val_loader))
            sample_a, sample_b = sample_a.to(device), sample_b.to(device)
            sample_pred = torch.sigmoid(model(sample_a))
            a_arr = (sample_a.cpu().numpy()[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            b_arr = (sample_b.cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.uint8)
            p_arr = (sample_pred.cpu().numpy()[0, 0] * 255).clip(0, 255).astype(np.uint8)
            a_img = Image.fromarray(a_arr, mode="RGB")
            b_img = Image.fromarray(b_arr, mode="L").convert("RGB")
            p_img = Image.fromarray(p_arr, mode="L").convert("RGB")
            w, h = a_img.size
            grid = Image.new("RGB", (w * 3, h))
            grid.paste(a_img, (0, 0))
            grid.paste(p_img, (w, 0))
            grid.paste(b_img, (w * 2, 0))
            grid.save(run_dirs["SAMPLE_DIR"] / f"epoch_{epoch:03d}.png")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg}
        
        progress_data = {
            "epoch": epoch,
            "total_epochs": cfg["epochs"],
            "percentage": (epoch / cfg["epochs"]) * 100.0,
            "loss": train_loss
        }
        with open(run_dirs["CURRENT_RUN_DIR"] / "progress.json", "w") as f:
            json.dump(progress_data, f, indent=4)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(state, run_dirs["CHECKPOINT_DIR"] / "best_loss.pt")
            print("  -> Saved new best_loss.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    main()

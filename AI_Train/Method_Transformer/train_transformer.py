import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib

from dataset import TrajectorySlidingWindowDataset
from model import GoalConditionedGPT2

def train(model, dataloader, optimizer, device, pred_len):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        obs_traj = batch["obs_traj"].to(device)
        pred_traj = batch["pred_traj"].to(device)
        start_pt = batch["start_pt"].to(device)
        end_pt = batch["end_pt"].to(device)
        geo_mask = batch["geo_mask"].to(device)
        
        optimizer.zero_grad()
        preds = model(obs_traj, start_pt, end_pt, geo_mask, pred_len=pred_len)
        loss = torch.abs(preds - pred_traj).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device, pred_len):
    model.eval()
    total_loss = 0
    err_x_total = 0
    err_y_total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            obs_traj = batch["obs_traj"].to(device)
            pred_traj = batch["pred_traj"].to(device)
            start_pt = batch["start_pt"].to(device)
            end_pt = batch["end_pt"].to(device)
            geo_mask = batch["geo_mask"].to(device)
            
            preds = model(obs_traj, start_pt, end_pt, geo_mask, pred_len=pred_len)
            abs_diff = torch.abs(preds - pred_traj)
            
            loss = abs_diff.mean()
            total_loss += loss.item()
            err_x_total += abs_diff[:, :, 0].mean().item()
            err_y_total += abs_diff[:, :, 1].mean().item()
            
    num_batches = len(dataloader)
    return total_loss / num_batches, err_x_total / num_batches, err_y_total / num_batches

def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # OUTPUT DIR Setup
    # Go up from Method_Transformer to AI_Train, then up to Root
    project_root = pathlib.Path(__file__).resolve().parents[2]
    method_name = "Method_Transformer"
    output_base_dir = project_root / "AI_Result" / method_name / "outputs"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple run grouping (like SGAN)
    existing_runs = [d for d in output_base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    run_idx = len(existing_runs) + 1
    output_dir = output_base_dir / f"run_{run_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve dataset path relative to config if needed
    dataset_path = config["dataset_path"]
    if not os.path.isabs(dataset_path):
        # dataset_path is relative to config_train.json location
        config_dir = pathlib.Path(config_path).resolve().parent
        dataset_path = str((config_dir / dataset_path).resolve())

    # DATASET
    print(f"--- Loading Training Set from {dataset_path} ---")
    train_dataset = TrajectorySlidingWindowDataset(dataset_path, config, split="train")
    print("--- Loading Validation Set ---")
    val_dataset = TrajectorySlidingWindowDataset(dataset_path, config, split="val")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ Error: Dataset is empty. Please check path or run Data Formatter.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    # MODEL
    model = GoalConditionedGPT2(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    
    # TRAINING LOOP
    best_val_loss = float('inf')
    epochs = config["epochs"]
    pred_len = config["pred_len"]
    save_every = config.get("save_every", 10)
    
    history_logs = []
    progress_file = output_dir / "progress.json"
    
    for epoch in range(epochs):
        epoch_num = epoch + 1
        print(f"\nEpoch {epoch_num}/{epochs}")
        
        train_loss = train(model, train_loader, optimizer, device, pred_len)
        val_loss, err_x, err_y = validate(model, val_loader, device, pred_len)
        
        print(f"Train L1 Loss: {train_loss:.4f}")
        print(f"Val L1 Loss: {val_loss:.4f} | Abs Err X: {err_x:.4f} | Abs Err Y: {err_y:.4f}")
        
        # Save history
        history_logs.append({
            "epoch": epoch_num,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_err_x": err_x,
            "val_err_y": err_y
        })
        
        # Format as progress.json for UI (Manual write to avoid ModuleNotFoundError)
        with open(progress_file, "w") as f:
            json.dump({
                "epoch": epoch_num,
                "total_epochs": epochs,
                "percentage": int((epoch_num / epochs) * 100),
                "loss": train_loss,
                "val_loss": val_loss,
                "val_mae": (err_x + err_y) / 2
            }, f)
        
        # 1. Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"🌟 New Best Model! (Saved to {output_dir}/best_model.pth)")

        # 2. Save Every X Epochs
        if epoch_num % save_every == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch_num}.pth")
            print(f"💾 Periodic Save at epoch {epoch_num}")

        # 3. Always Save Latest
        torch.save(model.state_dict(), output_dir / "latest_model.pth")
            
    print(f"✅ Training Complete. Results saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()
    main(args.config)

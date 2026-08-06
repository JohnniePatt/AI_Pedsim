import argparse
import json
import os
import pathlib
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import TrajectoryDataset, seq_collate
from model import TrajectoryGenerator

AI_TRAIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(AI_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(AI_TRAIN_DIR))
from baseline_output import (  # noqa: E402
    create_run_layout,
    mark_run_completed,
    update_checkpoint_manifest,
)

def get_run_dir(base_dir):
    # base_dir is e.g. .../AI_Pedsim/AI_Train/Method_SGAN
    method_name = os.path.basename(base_dir)
    # Go up twice from Method_SGAN: .../AI_Pedsim
    project_root = os.path.dirname(os.path.dirname(base_dir))
    runs_dir = os.path.join(project_root, "AI_Result", method_name, "outputs")
    os.makedirs(runs_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(runs_dir) if d.startswith("run_")]
    run_idx = len(existing_runs) + 1
    run_dir = os.path.join(runs_dir, f"run_{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    return run_dir

def l2_loss(pred_traj, pred_traj_gt, mode='average'):
    loss = (pred_traj_gt - pred_traj)**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / pred_traj.shape[0]
    return torch.sum(loss)

def train():
    raise RuntimeError(
        "Legacy non-adversarial trainer is disabled in Method_SGAN_SF_01; use run_pipeline.py, "
        "which invokes the genuine joint SGAN-SF trainer."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    if not config.get("sf_implementation_ready", False):
        raise RuntimeError(
            "Method_SGAN_SF_01 is a protected baseline copy: implement the genuine joint SGAN/Social-Force "
            "contract, then set sf_implementation_ready=true before training."
        )

    # Resolve dataset_path relative to config file if it's relative
    dataset_path = config["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(config_dir, dataset_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting SGAN Training on device: {device}")

    # Initialize Dataset
    print(f"Loading dataset from {dataset_path}...")
    train_dataset = TrajectoryDataset(data_dir=dataset_path, config=config, split="train", shuffle=True)
    if len(train_dataset.all_files) == 0:
        print("❌ Dataset is empty. Please check the dataset path or format Parquet first.")
        return
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # shuffle handled internally by IterableDataset
        collate_fn=seq_collate,
        pin_memory=True,
        num_workers=12,            # เพิ่มเป็น 12 หัวช่วยแกะไฟล์ Parquet พร้อมกัน
        persistent_workers=True    # เก็บ worker ไว้ไม่ต้องโหลดใหม่ทุก epoch
    )
    print(f"Dataset streaming initialized. Files to process: {len(train_dataset.all_files)}")

    # Model
    model = TrajectoryGenerator(
        emb_dim=config.get("emb_size", 64),
        h_dim=config.get("hidden_size", 128),
        pool_dim=config.get("social_pooling_size", 16),
        obs_len=config["obs_len"],
        pred_len=config["pred_len"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    project_root = pathlib.Path(__file__).resolve().parents[2]
    dataset_manifest = pathlib.Path(dataset_path) / "manifest_housegan_cases.csv"
    run_layout = create_run_layout(
        project_root / "AI_Result" / "Method_SGAN_SF_01" / "outputs",
        method_id="Method_SGAN_SF_01",
        method_display_name="Social-Force-Informed Joint Multi-Agent Social GAN",
        method_family="continuous_coordinate_social",
        seed=config.get("seed", 42),
        dataset_id=config.get("dataset_id", "housegan_canonical_imagebase_split_v1"),
        config=config,
        dataset_manifest=dataset_manifest if dataset_manifest.exists() else None,
        project_root=project_root.parent,
    )
    run_dir = str(run_layout.root)
    progress_file = os.path.join(run_dir, "progress.json")
    history_file = os.path.join(run_dir, "logs", "training_history.csv")
    
    with open(history_file, "w") as f:
        f.write("epoch,loss\n")

    epochs = config["epochs"]
    total_files = len(train_dataset.all_files)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # หลอด 1: แสดงความคืบหน้าการโหลดไฟล์ (Streaming Data)
        file_pbar = tqdm(total=total_files, desc=f"📂 Loading Data [Ep {epoch}]", position=0, leave=False)
        # หลอด 2: แสดงความคืบหน้าการเทรน (Training Batches)
        train_pbar = tqdm(desc=f"🚀 Training [Ep {epoch}]", unit="batch", position=1, leave=True)
        
        processed_files = set()
        
        for batch_idx, batch in enumerate(train_loader):
            # ใน batch นี้อาจรวบรวมคนเดินมาจากหลายไฟล์ เราจะนับไฟล์ที่เจอผ่าน IDs ถ้าทำได้
            # แต่เพื่อความง่าย เราจะประมาณการความคืบหน้าของไฟล์ตามสัดส่วน batch
            # หรือถ้าอยากแม่นยำ เราส่งค่ากลับมาจาก Dataset (แต่เพื่อความเร็วจะใช้ postfix ช่วย)
            
            obs_traj, pred_traj_gt = batch[0].to(device), batch[1].to(device)
            obs_rel_traj, pred_rel_traj_gt = batch[2].to(device), batch[3].to(device)
            seq_start_end = batch[4]
            
            optimizer.zero_grad()
            pred_rel_traj = model(obs_rel_traj, seq_start_end)
            loss = l2_loss(pred_rel_traj, pred_rel_traj_gt)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            
            train_pbar.update(1)
            train_pbar.set_postfix({"loss": f"{current_loss:.4f}", "avg": f"{total_loss/(batch_idx+1):.4f}"})
            
            # อัปเดตหลอดไฟล์ (เนื่องจาก 4 workers โหลดพร้อมกัน เราจะประมาณค่าจาก batch)
            # ถ้า 603 ไฟล์ แบตช์ละ 64... เราจะอัปเดตไฟล์แบบคร่าวๆ
            if batch_idx % 2 == 0: 
                file_pbar.update(min(1, total_files - file_pbar.n))

        file_pbar.close()
        train_pbar.close()
        
        avg_loss = total_loss / max(1, batch_idx + 1)
        tqdm.write(f"✅ Epoch [{epoch}/{epochs}] Finished. Average Loss: {avg_loss:.4f}")

        # Update Progress for Streamlit UI
        with open(progress_file, "w") as f:
            json.dump({
                "epoch": epoch,
                "total_epochs": epochs,
                "percentage": int((epoch / epochs) * 100),
                "loss": avg_loss
            }, f)
            
        # Update History CSV
        with open(history_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f}\n")

        # Save weights occasionally
        if epoch % 10 == 0 or epoch == epochs:
            payload = {
                "format_version": 2,
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "emb_dim": config.get("emb_size", 64),
                    "h_dim": config.get("hidden_size", 128),
                    "pool_dim": config.get("social_pooling_size", 16),
                    "obs_len": config["obs_len"],
                    "pred_len": config["pred_len"],
                },
                "data_config": {
                    "dataset_path": str(pathlib.Path(dataset_path).resolve()),
                    "dataset_name": pathlib.Path(dataset_path).name,
                    "dataset_id": config.get("dataset_id", "housegan_canonical_imagebase_split_v1"),
                },
                "epoch": epoch,
            }
            checkpoint_path = run_layout.checkpoints / "latest_model.pth"
            torch.save(payload, checkpoint_path)
            update_checkpoint_manifest(run_layout.root, checkpoint_path, "latest")

    mark_run_completed(run_layout.root)
    print("✅ Training Completed Successfully!")

if __name__ == "__main__":
    train()

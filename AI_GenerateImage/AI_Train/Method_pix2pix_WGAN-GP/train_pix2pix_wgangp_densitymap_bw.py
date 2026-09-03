import os
import json
import pathlib
import csv
import argparse
import datetime
import hashlib
import platform
import random
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

EXPECTED_SPLIT_COUNTS = {"train": 2603, "validation": 439, "test": 862}
EXPECTED_PLAN_COUNTS = {"train": 412, "validation": 60, "test": 117}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_inventory(dataset_root, snapshot_path):
    inventories = {}
    plan_sets = {}
    rows = []
    for split in ("train", "validation", "test"):
        a_names = {p.name for p in (dataset_root / "A" / split).glob("*.png")}
        b_names = {p.name for p in (dataset_root / "B" / split).glob("*.png")}
        paired = sorted(a_names & b_names)
        plans = {name.split("__", 1)[0] for name in paired}
        inventories[split] = {
            "input_count": len(a_names),
            "target_count": len(b_names),
            "paired_count": len(paired),
            "plan_count": len(plans),
            "input_only_count": len(a_names - b_names),
            "target_only_count": len(b_names - a_names),
        }
        plan_sets[split] = plans
        rows.extend((split, name, name.split("__", 1)[0]) for name in paired)

    overlap = {
        "train_validation": len(plan_sets["train"] & plan_sets["validation"]),
        "train_test": len(plan_sets["train"] & plan_sets["test"]),
        "validation_test": len(plan_sets["validation"] & plan_sets["test"]),
    }
    problems = []
    for split in EXPECTED_SPLIT_COUNTS:
        inv = inventories[split]
        if inv["paired_count"] != EXPECTED_SPLIT_COUNTS[split]:
            problems.append(f"{split}: paired_count={inv['paired_count']}")
        if inv["plan_count"] != EXPECTED_PLAN_COUNTS[split]:
            problems.append(f"{split}: plan_count={inv['plan_count']}")
        if inv["input_only_count"] or inv["target_only_count"]:
            problems.append(f"{split}: A/B membership differs")
    if any(overlap.values()):
        problems.append(f"plan overlap detected: {overlap}")
    if problems:
        raise RuntimeError("Canonical dataset verification failed: " + "; ".join(problems))

    with open(snapshot_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "file_name", "plan_id"])
        writer.writerows(rows)
    return inventories, overlap, sha256_file(snapshot_path)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_value(project_root, args, fallback="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def main():
    parser = argparse.ArgumentParser(description="Train Method_pix2pix_WGAN-GP")
    parser.add_argument("--config", default="config_train.json", help="Config path, relative to this method directory")
    parser.add_argument("--dry-run", action="store_true", help="Verify data and one forward pass without creating a run")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    project_root = resolve_project_root(script_dir)
    dataset_root = resolve_path(cfg["dataset_root"], project_root)
    seed = int(cfg.get("seed", 42))
    configure_seed(seed)

    device = get_device()
    image_size = int(cfg.get("image_size", 512))
    if image_size != 256 and config_path.name == "config_train_256.json":
        raise ValueError(f"256 protocol requires image_size=256, got {image_size}")

    if args.dry_run:
        temp_snapshot = script_dir / ".dataset_manifest_snapshot.dry_run.csv"
        try:
            inventory, overlap, snapshot_hash = canonical_inventory(dataset_root, temp_snapshot)
        finally:
            temp_snapshot.unlink(missing_ok=True)
        dataset = Pix2PixDataset(dataset_root, split="train", image_size=image_size)
        input_a, target_b, filename = dataset[0]
        model = UNetGenerator(input_nc=3, output_nc=1, num_downs=8, ngf=64).to(device).eval()
        with torch.no_grad():
            output = model(input_a.unsqueeze(0).to(device))
        print(json.dumps({
            "status": "dry_run_passed",
            "config": str(config_path),
            "dataset_id": cfg.get("dataset_id"),
            "inventory": inventory,
            "plan_overlap": overlap,
            "dataset_snapshot_sha256": snapshot_hash,
            "image_size": image_size,
            "sample": filename,
            "input_shape": list(input_a.shape),
            "target_shape": list(target_b.shape),
            "output_shape": list(output.shape),
            "device": str(device),
        }, indent=2))
        return

    utc_now = datetime.datetime.now(datetime.timezone.utc)
    run_name = f"run_{utc_now.strftime('%Y%m%dT%H%M%SZ')}_seed{seed:03d}"
    run_paths = make_run_dirs(
        script_dir, method_name="Method_pix2pix_WGAN-GP", run_name=run_name
    )
    print(f"🚀 Training Method_pix2pix_WGAN-GP on device: {device}")
    print(f"📁 Output Run Directory: {run_paths['CURRENT_RUN_DIR']}")

    resolved_cfg = dict(cfg)
    resolved_cfg["dataset_root"] = str(dataset_root)
    resolved_cfg["seed"] = seed
    write_json(run_paths["CURRENT_RUN_DIR"] / "config_train.json", resolved_cfg)
    write_json(run_paths["CURRENT_RUN_DIR"] / "run_config_snapshot.json", resolved_cfg)

    inventory, overlap, snapshot_hash = canonical_inventory(
        dataset_root, run_paths["CURRENT_RUN_DIR"] / "dataset_manifest_snapshot.csv"
    )
    commit = git_value(project_root, ["rev-parse", "HEAD"])
    git_dirty = bool(git_value(project_root, ["status", "--porcelain"], fallback=""))
    code_files = [
        script_dir / "train_pix2pix_wgangp_densitymap_bw.py",
        script_dir / "pix2pix_wgangp_common.py",
        config_path,
    ]
    write_json(run_paths["CURRENT_RUN_DIR"] / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    })
    write_json(run_paths["CURRENT_RUN_DIR"] / "code_provenance.json", {
        "git_commit": commit,
        "git_dirty": git_dirty,
        "files": {str(p.relative_to(project_root)): sha256_file(p) for p in code_files},
    })
    manifest_path = run_paths["CURRENT_RUN_DIR"] / "run_manifest.json"
    manifest = {
        "run_id": run_name,
        "method_id": "Method_pix2pix_WGAN-GP",
        "status": "training",
        "started_at_utc": utc_now.isoformat(),
        "dataset_id": cfg.get("dataset_id", "unknown"),
        "dataset_inventory": inventory,
        "plan_overlap": overlap,
        "dataset_manifest_sha256": snapshot_hash,
        "train_split": "train",
        "model_selection_split": "validation",
        "image_size": image_size,
        "seed": seed,
        "epochs": int(cfg.get("epochs", 50)),
        "research_valid": False,
        "research_valid_reason": "Training/evaluation not complete",
    }
    write_json(manifest_path, manifest)

    # Datasets and Loaders
    train_dataset = Pix2PixDataset(dataset_root, split="train", image_size=image_size)
    val_dataset = Pix2PixDataset(dataset_root, split="val", image_size=image_size)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True, generator=loader_generator)
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
    best_epoch = None
    training_started = time.perf_counter()

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
            "dataset_id": cfg.get("dataset_id"),
            "seed": seed,
            "run_id": run_name,
        }

        torch.save(checkpoint_data, run_paths["CHECKPOINT_DIR"] / "final.pt")
        if avg_val_l1 < best_val_loss:
            best_val_loss = avg_val_l1
            best_epoch = epoch
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

    training_wall_time_s = time.perf_counter() - training_started
    write_json(run_paths["CHECKPOINT_DIR"] / "checkpoint_manifest.json", {
        "selection_metric": "validation_l1_loss",
        "best_checkpoint": "best_loss.pt",
        "latest_checkpoint": "final.pt",
        "best_validation_l1": best_val_loss,
        "best_epoch": best_epoch,
        "completed_epochs": epochs,
    })
    with open(run_paths["LOG_DIR"] / "training_runtime.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epochs", "train_samples", "validation_samples", "wall_time_s", "wall_time_hours"])
        writer.writerow([epochs, len(train_dataset), len(val_dataset), training_wall_time_s, training_wall_time_s / 3600.0])
    manifest.update({
        "status": "trained",
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_wall_time_s": training_wall_time_s,
        "best_validation_l1": best_val_loss,
        "best_epoch": best_epoch,
        "research_valid_reason": "Canonical training completed; canonical test evaluation pending",
    })
    write_json(manifest_path, manifest)
    print(f"✅ Training Method_pix2pix_WGAN-GP completed successfully!")

if __name__ == "__main__":
    main()

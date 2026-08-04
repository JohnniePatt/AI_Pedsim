import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import torch
from tqdm import tqdm

from unet_common import (
    build_unet,
    dice_from_probs,
    get_device,
    iou_from_probs,
    load_json_config,
    make_loader,
    make_loss_fn,
    make_run_dirs,
    resolve_path,
    resolve_project_root,
    write_progress,
    write_summary_csv,
)


def build_config(config_path: pathlib.Path) -> dict:
    script_dir = pathlib.Path(__file__).parent.resolve()
    project_root = resolve_project_root(script_dir)
    cfg = {
        "epochs": 50,
        "batch_size": 8,
        "learning_rate": 2e-4,
        "image_size": 256,
        "base_filters": 32,
        "dropout": 0.1,
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "foreground_weight": 25.0,
        "focal_gamma": 0.0,
        "mask_threshold": 0.5,
        "early_stopping_patience": 12,
        "reduce_lr_patience": 5,
        "num_workers": 4,
        "dataset_root": "../Dataset/Data_ImageUNet/Trajectory_line_mask_dataset/Topo_HouseGAN",
        "resume_checkpoint_path": "-",
        "run_test_after_train": True,
    }
    cfg.update(load_json_config(config_path))
    cfg["BASE_DIR"] = str(script_dir)
    cfg["PROJECT_ROOT"] = str(project_root)
    cfg["DATASET_ROOT"] = str(resolve_path(cfg["dataset_root"], project_root))
    return cfg


def print_system(device: torch.device):
    print("=" * 70)
    print(f"[SYSTEM] PyTorch: {torch.__version__}")
    print(f"[SYSTEM] Device: {device}")
    if device.type == "cuda":
        print(f"[SYSTEM] CUDA: {torch.version.cuda}")
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)


def run_epoch(model, loader, optimizer, loss_fn, device: torch.device, train: bool, epoch: int, total_epochs: int):
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0
    phase = "Train" if train else "Val"
    progress = tqdm(
        loader,
        desc=f"{phase} {epoch:03d}/{total_epochs:03d}",
        leave=False,
        dynamic_ncols=True,
    )

    for a, b, _ in progress:
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(a)
            loss = loss_fn(logits, b)
            probs = torch.sigmoid(logits)
            dice = dice_from_probs(b, probs)
            iou = iou_from_probs(b, probs)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_dice += float(dice.detach().cpu())
        total_iou += float(iou.detach().cpu())
        total_batches += 1
        progress.set_postfix(
            loss=f"{total_loss / total_batches:.4f}",
            dice=f"{total_dice / total_batches:.4f}",
            iou=f"{total_iou / total_batches:.4f}",
        )

    denom = max(total_batches, 1)
    return {
        "loss": total_loss / denom,
        "dice": total_dice / denom,
        "iou": total_iou / denom,
    }


def save_checkpoint(
    path: pathlib.Path,
    model,
    optimizer,
    epoch: int,
    cfg: dict,
    best_val_dice: float,
    best_val_loss: float | None = None,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "best_val_dice": best_val_dice,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch U-Net for trajectory-line masks.")
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = pathlib.Path.cwd() / config_path
        if not config_path.exists():
            config_path = script_dir / args.config

    cfg = build_config(config_path)
    dataset_root = pathlib.Path(cfg["DATASET_ROOT"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    device = get_device()
    print_system(device)

    run_dirs = make_run_dirs(script_dir, method_name="Method_UNet")
    current_run_dir = run_dirs["CURRENT_RUN_DIR"]
    cfg.update({k: str(v) for k, v in run_dirs.items()})
    cfg["run_name"] = current_run_dir.name
    cfg["framework"] = "pytorch"

    snapshot_path = current_run_dir / "run_config_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
    if config_path.exists():
        shutil.copy2(config_path, current_run_dir / "config_train.json")

    print(f"[CONFIG] Loaded: {config_path}")
    print(f"[CONFIG] Dataset: {dataset_root}")
    print(f"[RUN] {current_run_dir}")

    train_loader, train_names = make_loader(
        dataset_root,
        "train",
        int(cfg["image_size"]),
        int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    val_loader, val_names = make_loader(
        dataset_root,
        "validation",
        int(cfg["image_size"]),
        int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    print(f"[DATA] train={len(train_names)} validation={len(val_names)}")

    model = build_unet(base_filters=int(cfg["base_filters"]), dropout=float(cfg["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(cfg["reduce_lr_patience"]),
        min_lr=1e-7,
    )
    loss_fn = make_loss_fn(
        bce_weight=float(cfg["bce_weight"]),
        dice_weight=float(cfg["dice_weight"]),
        foreground_weight=float(cfg["foreground_weight"]),
        device=device,
        focal_gamma=float(cfg.get("focal_gamma", 0.0)),
    )

    with open(run_dirs["CHECKPOINT_DIR"] / "model_architecture.txt", "w", encoding="utf-8") as f:
        f.write(str(model))

    resume_path = str(cfg.get("resume_checkpoint_path", "-"))
    best_val_dice = -1.0
    best_val_loss = float("inf")
    start_epoch = 1
    if resume_path not in ("-", "", "None", "none", None):
        resume = pathlib.Path(resume_path)
        if not resume.is_absolute():
            resume = resolve_path(resume, pathlib.Path(cfg["PROJECT_ROOT"]))
        print(f"[RESUME] Loading checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_dice = float(checkpoint.get("best_val_dice", best_val_dice))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    progress_path = current_run_dir / "progress.json"
    write_progress(progress_path, 0, int(cfg["epochs"]), 0.0, 0.0)

    history_rows = []
    patience_counter = 0
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            train=True,
            epoch=epoch,
            total_epochs=int(cfg["epochs"]),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer,
                loss_fn,
                device,
                train=False,
                epoch=epoch,
                total_epochs=int(cfg["epochs"]),
            )
        scheduler.step(val_metrics["dice"])

        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "dice_coef": train_metrics["dice"],
            "iou_coef": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice_coef": val_metrics["dice"],
            "val_iou_coef": val_metrics["iou"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        write_summary_csv(run_dirs["LOG_DIR"] / "training_history.csv", history_rows)
        write_progress(progress_path, epoch, int(cfg["epochs"]), train_metrics["loss"], val_metrics["dice"])

        print(
            f"[EPOCH {epoch:03d}/{int(cfg['epochs']):03d}] "
            f"loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            patience_counter = 0
            save_checkpoint(
                run_dirs["CHECKPOINT_DIR"] / "best_dice.pt",
                model,
                optimizer,
                epoch,
                cfg,
                best_val_dice,
                best_val_loss,
            )
            save_checkpoint(
                run_dirs["CHECKPOINT_DIR"] / "best.pt",
                model,
                optimizer,
                epoch,
                cfg,
                best_val_dice,
                best_val_loss,
            )
            print(f"[CKPT] Saved best_dice.pt val_dice={best_val_dice:.4f}")
        else:
            patience_counter += 1

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                run_dirs["CHECKPOINT_DIR"] / "best_loss.pt",
                model,
                optimizer,
                epoch,
                cfg,
                best_val_dice,
                best_val_loss,
            )
            print(f"[CKPT] Saved best_loss.pt val_loss={best_val_loss:.4f}")

        if patience_counter >= int(cfg["early_stopping_patience"]):
            print(f"[EARLY STOP] No val_dice improvement for {patience_counter} epochs.")
            break

    save_checkpoint(run_dirs["CHECKPOINT_DIR"] / "final.pt", model, optimizer, epoch, cfg, best_val_dice, best_val_loss)
    print(f"[DONE] Training finished: {current_run_dir}")

    if bool(cfg.get("run_test_after_train", True)):
        test_script = script_dir / "test_unet_trajectory_mask.py"
        for checkpoint_mode in ("best_dice", "best_loss"):
            checkpoint_path = run_dirs["CHECKPOINT_DIR"] / f"{checkpoint_mode}.pt"
            if not checkpoint_path.exists():
                continue
            cmd = [
                sys.executable,
                str(test_script),
                "--run_path",
                str(current_run_dir),
                "--checkpoint_mode",
                checkpoint_mode,
                "--output_name",
                checkpoint_mode,
            ]
            print(f"[TEST] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()

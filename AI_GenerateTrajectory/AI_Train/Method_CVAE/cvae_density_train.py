import argparse
import pathlib
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm

from cvae_config import build_train_config, make_run_dirs, save_run_snapshot, write_progress, write_summary_csv
from cvae_data import make_density_dataset
from cvae_io import save_triptych_sample
from cvae_losses import DensityLossComputer, tensor_density_metrics
from cvae_model import CVAE


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_system(device):
    print("=" * 70)
    print(f"[SYSTEM] PyTorch: {torch.__version__}")
    print(f"[SYSTEM] Device: {device}")
    if device.type == "cuda":
        print(f"[SYSTEM] CUDA: {torch.version.cuda}")
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)


def configure_torch_backend(cfg):
    use_cudnn = bool(cfg.get("use_cudnn", True))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.enabled = use_cudnn
        print(f"[SYSTEM] cuDNN enabled: {torch.backends.cudnn.enabled}")


def metric_arrays(batch_b, pred):
    true_np = batch_b.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    rows = []
    for i in range(true_np.shape[0]):
        rows.append(tensor_density_metrics(true_np[i], pred_np[i]))
    return rows


def average_metric_rows(rows):
    if not rows:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 0.0, "ssim": 0.0}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def run_epoch(model, loader, optimizer, loss_comp, device, train, epoch, total_epochs, kl_weight):
    model.train(train)
    phase = "Train" if train else "Val"
    latent_mode = str(loss_comp.cfg.get("train_latent_mode", "posterior" if train else "zero"))
    totals = {"total": 0.0, "l1": 0.0, "mse_loss": 0.0, "edge": 0.0, "kl": 0.0}
    metric_rows = []
    batches = 0
    progress = tqdm(loader, desc=f"{phase} {epoch:03d}/{total_epochs:03d}", leave=False, dynamic_ncols=True)
    for batch_a, batch_b, _ in progress:
        batch_a = batch_a.to(device, non_blocking=True)
        batch_b = batch_b.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits, mu, logvar = model.forward_train(batch_a, batch_b, latent_mode=latent_mode)
            loss_total, loss_l1, loss_mse, loss_edge, loss_kl = loss_comp.compute(
                logits, batch_b, mu, logvar, kl_weight
            )
            pred = torch.sigmoid(logits)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                optimizer.step()
        totals["total"] += float(loss_total.detach().cpu())
        totals["l1"] += float(loss_l1.detach().cpu())
        totals["mse_loss"] += float(loss_mse.detach().cpu())
        totals["edge"] += float(loss_edge.detach().cpu())
        totals["kl"] += float(loss_kl.detach().cpu())
        metric_rows.extend(metric_arrays(batch_b, pred))
        batches += 1
        avg_metrics = average_metric_rows(metric_rows)
        progress.set_postfix(
            total=f"{totals['total'] / batches:.4f}",
            mae=f"{avg_metrics['mae']:.4f}",
            ssim=f"{avg_metrics['ssim']:.4f}",
            kl=f"{totals['kl'] / batches:.4f}",
        )
    out = {key: value / max(batches, 1) for key, value in totals.items()}
    out.update({f"metric_{key}": value for key, value in average_metric_rows(metric_rows).items()})
    return out


def save_checkpoint(path, model, optimizer, epoch, cfg, best_val_mae, best_val_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "best_val_mae": best_val_mae,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def save_epoch_samples(model, samples, sample_dir, epoch, device):
    model.eval()
    with torch.no_grad():
        for idx, (a, b) in enumerate(samples):
            a = a.to(device).unsqueeze(0)
            pred = torch.sigmoid(model.forward_infer(a))[0]
            save_triptych_sample(
                a[0].detach().cpu(),
                pred.detach().cpu(),
                b.detach().cpu(),
                sample_dir / f"epoch_{epoch:04d}_sample_{idx:02d}.png",
            )


def main(default_config, target_representation, target_channels, test_script_name):
    parser = argparse.ArgumentParser(description=f"Train CVAE density map ({target_representation}).")
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = pathlib.Path.cwd() / config_path
        if not config_path.exists():
            config_path = script_dir / args.config

    cfg = build_train_config(config_path)
    cfg["target_representation"] = str(cfg.get("target_representation", target_representation))
    cfg["target_channels"] = int(cfg.get("target_channels", target_channels))
    cfg["metric_mode"] = str(cfg.get("metric_mode", "rgb" if cfg["target_channels"] == 3 else "density_scalar"))
    dataset_root = pathlib.Path(cfg["DATASET_ROOT"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    device = get_device()
    configure_torch_backend(cfg)
    print_system(device)

    run_dirs = make_run_dirs(script_dir, method_name="Method_CVAE")
    cfg = save_run_snapshot(cfg, config_path, run_dirs)
    print(f"[CONFIG] Loaded: {config_path}")
    print(f"[CONFIG] Dataset: {dataset_root}")
    print(f"[CONFIG] Target: {cfg['target_representation']} channels={cfg['target_channels']}")
    print(f"[RUN] {run_dirs['CURRENT_RUN_DIR']}")

    train_loader, train_pairs = make_density_dataset(
        dataset_root,
        "train",
        cfg["batch_size"],
        cfg["image_size"],
        True,
        target_representation=cfg["target_representation"],
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 0),
    )
    val_loader, val_pairs = make_density_dataset(
        dataset_root,
        "validation",
        cfg["batch_size"],
        cfg["image_size"],
        False,
        target_representation=cfg["target_representation"],
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 0),
    )
    test_loader, test_pairs = make_density_dataset(
        dataset_root,
        "test",
        cfg["batch_size"],
        cfg["image_size"],
        False,
        target_representation=cfg["target_representation"],
        seed=cfg.get("seed", 42),
        num_workers=0,
    )
    print(f"[DATA] train={len(train_pairs)} validation={len(val_pairs)} test={len(test_pairs)}")

    model = CVAE(
        cfg["image_size"],
        cfg["base_filters"],
        cfg["latent_dim"],
        dropout=cfg.get("dropout", 0.1),
        target_channels=cfg["target_channels"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(cfg.get("reduce_lr_patience", 5)),
        min_lr=1e-7,
    )
    loss_comp = DensityLossComputer(cfg, device)

    with open(run_dirs["CHECKPOINT_DIR"] / "model_architecture.txt", "w", encoding="utf-8") as f:
        f.write(str(model))

    resume_path = str(cfg.get("resume_checkpoint_path", "-"))
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    start_epoch = 1
    if resume_path not in ("-", "", "None", "none", None):
        resume = pathlib.Path(resume_path)
        if not resume.is_absolute():
            resume = pathlib.Path(cfg["PROJECT_ROOT"]) / resume
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_mae = float(checkpoint.get("best_val_mae", best_val_mae))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[RESUME] Loaded {resume}")

    fixed_samples = []
    for a, b, _ in test_loader:
        for i in range(a.shape[0]):
            fixed_samples.append((a[i], b[i]))
            if len(fixed_samples) >= int(cfg.get("sample_count", 4)):
                break
        if len(fixed_samples) >= int(cfg.get("sample_count", 4)):
            break

    progress_path = run_dirs["CURRENT_RUN_DIR"] / "progress.json"
    write_progress(progress_path, 0, int(cfg["epochs"]), 0.0, 0.0)
    history_rows = []
    patience_counter = 0

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        kl_weight = float(cfg.get("kl_weight", 0.01))
        anneal = int(cfg.get("kl_anneal_epochs", 0))
        if anneal > 0:
            kl_weight *= min(1.0, epoch / float(anneal))

        train_metrics = run_epoch(model, train_loader, optimizer, loss_comp, device, True, epoch, int(cfg["epochs"]), kl_weight)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, loss_comp, device, False, epoch, int(cfg["epochs"]), kl_weight)
        scheduler.step(val_metrics["total"])

        row = {"epoch": epoch, "kl_weight": kl_weight, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history_rows.append(row)
        write_summary_csv(run_dirs["LOG_DIR"] / "training_history.csv", history_rows)
        write_progress(progress_path, epoch, int(cfg["epochs"]), train_metrics["total"], val_metrics["total"])

        print(
            f"[EPOCH {epoch:03d}/{int(cfg['epochs']):03d}] "
            f"train_total={train_metrics['total']:.4f} train_mae={train_metrics['metric_mae']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_mae={val_metrics['metric_mae']:.4f} "
            f"val_ssim={val_metrics['metric_ssim']:.4f} kl_w={kl_weight:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if epoch % int(cfg.get("sample_every_epochs", 1)) == 0 and fixed_samples:
            save_epoch_samples(model, fixed_samples, run_dirs["SAMPLE_DIR"], epoch, device)

        improved_mae = val_metrics["metric_mae"] < best_val_mae
        improved_loss = val_metrics["total"] < best_val_loss
        if improved_mae:
            best_val_mae = val_metrics["metric_mae"]
            patience_counter = 0
            save_checkpoint(run_dirs["CHECKPOINT_DIR"] / "best_mae.pt", model, optimizer, epoch, cfg, best_val_mae, best_val_loss)
            save_checkpoint(run_dirs["CHECKPOINT_DIR"] / "best.pt", model, optimizer, epoch, cfg, best_val_mae, best_val_loss)
            print(f"[CKPT] Saved best_mae.pt val_mae={best_val_mae:.6f}")
        else:
            patience_counter += 1
        if improved_loss:
            best_val_loss = val_metrics["total"]
            save_checkpoint(run_dirs["CHECKPOINT_DIR"] / "best_loss.pt", model, optimizer, epoch, cfg, best_val_mae, best_val_loss)
            print(f"[CKPT] Saved best_loss.pt val_total={best_val_loss:.6f}")

        if epoch % int(cfg.get("checkpoint_every_epochs", 10)) == 0:
            save_checkpoint(run_dirs["CHECKPOINT_DIR"] / f"epoch_{epoch:04d}.pt", model, optimizer, epoch, cfg, best_val_mae, best_val_loss)
        if patience_counter >= int(cfg.get("early_stopping_patience", 10)):
            print(f"[EARLY STOP] No val_mae improvement for {patience_counter} epochs.")
            break

    save_checkpoint(run_dirs["CHECKPOINT_DIR"] / "final.pt", model, optimizer, epoch, cfg, best_val_mae, best_val_loss)
    print(f"[DONE] Training finished: {run_dirs['CURRENT_RUN_DIR']}")

    if bool(cfg.get("run_test_after_train", True)):
        test_script = script_dir / test_script_name
        for checkpoint_mode in cfg.get("test_checkpoint_modes", ["best_mae", "best_loss"]):
            checkpoint_path = run_dirs["CHECKPOINT_DIR"] / f"{checkpoint_mode}.pt"
            if not checkpoint_path.exists():
                continue
            cmd = [
                sys.executable,
                str(test_script),
                "--run_path",
                str(run_dirs["CURRENT_RUN_DIR"]),
                "--checkpoint_mode",
                checkpoint_mode,
                "--output_name",
                checkpoint_mode,
            ]
            print(f"[TEST] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)

"""Train LSTM-SF, Transformer-SF, or SGAN-SF under one fair data contract."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline_output import create_run_layout, mark_run_completed, update_checkpoint_manifest
from joint_sf import JointSceneDataset, JointSocialForcePredictor, SceneDiscriminator, trajectory_losses


DISPLAY_NAMES = {
    "Method_Transformer_SF_01": "Social-Force-Informed Joint Multi-Agent Transformer",
    "Method_LSTM_SF_01": "Social-Force-Informed Joint Multi-Agent LSTM",
    "Method_SGAN_SF_01": "Social-Force-Informed Joint Multi-Agent Social GAN",
}


def make_dataset(cfg: dict, split: str) -> JointSceneDataset:
    return JointSceneDataset(
        cfg["dataset_path"], split,
        obs_len=cfg.get("obs_len", 8), pred_len=cfg.get("pred_len", 24),
        frame_stride=cfg.get("frame_stride", 5), max_agents=cfg.get("max_agents", 64),
        windows_per_case=cfg.get("windows_per_case_train" if split == "train" else "windows_per_case_val", 32 if split == "train" else 4),
        max_cases=cfg.get("max_train_cases" if split == "train" else "max_val_cases"),
        grid_size=cfg.get("grid_size", 64), geo_padding=cfg.get("geo_padding", 1.0),
        seed=cfg.get("seed", 42), cache_size=cfg.get("case_cache_size", 2),
    )


def make_model(cfg: dict, architecture: str) -> JointSocialForcePredictor:
    return JointSocialForcePredictor(
        architecture,
        hidden_dim=cfg.get("hidden_dim", cfg.get("d_model", 128)),
        num_layers=cfg.get("num_layers", 2), nhead=cfg.get("nhead", 4),
        dropout=cfg.get("dropout", 0.1), max_residual=cfg.get("max_residual", 0.03),
        noise_dim=cfg.get("noise_dim", 16), desired_step=cfg.get("desired_step", 0.012),
        agent_strength=cfg.get("agent_repulsion_strength", 0.004),
        agent_sigma=cfg.get("agent_repulsion_sigma", 0.04),
        wall_strength=cfg.get("wall_repulsion_strength", 0.006),
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def model_rollout(model, batch, cfg, teacher_forcing_ratio: float):
    obs_len = int(cfg.get("obs_len", 8))
    positions = batch["positions"]
    active = batch["active"]
    return model.rollout(
        positions[:, :, :obs_len], active[:, :, :obs_len], batch["goal"], batch["wall_field"],
        positions.shape[2] - obs_len,
        teacher_positions=positions[:, :, obs_len:], teacher_active=active[:, :, obs_len:],
        teacher_forcing_ratio=teacher_forcing_ratio,
        stop_threshold=cfg.get("stop_threshold", 0.5), exit_radius=cfg.get("exit_radius_norm", 0.025),
    )


def validation(model, loader, device, cfg) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "position_loss": 0.0, "walkability_loss": 0.0, "stop_loss": 0.0}
    batches = 0
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            outputs = model_rollout(model, batch, cfg, 0.0)
            obs_len = int(cfg.get("obs_len", 8))
            losses = trajectory_losses(
                outputs, batch["positions"][:, :, obs_len:], batch["active"][:, :, obs_len:], batch["walkable"],
                walkability_weight=cfg.get("walkability_loss_weight", 0.1), stop_weight=cfg.get("stop_loss_weight", 0.2),
            )
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--config", type=pathlib.Path, required=True)
    cli.add_argument("--method-id", required=True, choices=tuple(DISPLAY_NAMES))
    cli.add_argument("--architecture", required=True, choices=("lstm", "transformer", "sgan"))
    args = cli.parse_args()
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    if cfg.get("sf_implementation_ready") is not True:
        raise RuntimeError("sf_implementation_ready must be true after the joint SF contract is configured")
    dataset_path = pathlib.Path(cfg["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    cfg["dataset_path"] = str(dataset_path)

    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_module = pathlib.Path(__file__).resolve().parents[1]
    manifest = dataset_path / "manifest_housegan_cases.csv"
    layout = create_run_layout(
        project_module / "AI_Result" / args.method_id / "outputs",
        method_id=args.method_id, method_display_name=DISPLAY_NAMES[args.method_id],
        method_family="joint_social_force_continuous_coordinate",
        seed=seed, dataset_id=cfg.get("dataset_id", "housegan_canonical_imagebase_split_v1"),
        config=cfg, dataset_manifest=manifest if manifest.exists() else None,
        project_root=project_module.parent,
    )
    print(f"[train] method={args.method_id} architecture={args.architecture} device={device} run={layout.root}")
    train_ds, val_ds = make_dataset(cfg, "train"), make_dataset(cfg, "val")
    loader_kwargs = dict(
        batch_size=int(cfg.get("batch_size", 4)), num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", device.type == "cuda")),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    model = make_model(cfg, args.architecture).to(device)
    generator_optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("learning_rate", cfg.get("lr", 3e-4))),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    discriminator = SceneDiscriminator(cfg.get("hidden_dim", 128)).to(device) if args.architecture == "sgan" else None
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(cfg.get("discriminator_lr", 2e-4))) if discriminator else None
    history_path = layout.logs / "training_history.csv"
    fields = ["epoch", "train_loss", "position_loss", "walkability_loss", "stop_loss", "adversarial_loss", "discriminator_loss", "val_loss", "epoch_seconds"]
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    best_val = float("inf")
    patience = int(cfg.get("early_stopping_patience", 10)); bad_epochs = 0

    for epoch in range(1, int(cfg.get("epochs", 100)) + 1):
        train_ds.set_epoch(epoch)
        started = time.time(); model.train()
        if discriminator: discriminator.train()
        sums = {
            "loss": 0.0,
            "position_loss": 0.0,
            "walkability_loss": 0.0,
            "stop_loss": 0.0,
            "adversarial_loss": 0.0,
            "discriminator_loss": 0.0,
        }
        batches = 0
        ratio_start = float(cfg.get("teacher_forcing_start", 1.0)); ratio_end = float(cfg.get("teacher_forcing_end", 0.1))
        progress = (epoch - 1) / max(int(cfg.get("epochs", 100)) - 1, 1)
        teacher_ratio = ratio_start + (ratio_end - ratio_start) * progress
        for raw in train_loader:
            batch = move_batch(raw, device)
            obs_len = int(cfg.get("obs_len", 8))
            outputs = model_rollout(model, batch, cfg, teacher_ratio)
            losses = trajectory_losses(
                outputs, batch["positions"][:, :, obs_len:], batch["active"][:, :, obs_len:], batch["walkable"],
                walkability_weight=cfg.get("walkability_loss_weight", 0.1), stop_weight=cfg.get("stop_loss_weight", 0.2),
            )
            adversarial_loss = torch.zeros((), device=device); discriminator_loss = torch.zeros((), device=device)
            if discriminator:
                full_real = batch["positions"]
                full_fake = torch.cat([batch["positions"][:, :, :obs_len], outputs["positions"]], dim=2)
                full_active = batch["active"]
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_score = discriminator(full_real, full_active)
                fake_score = discriminator(full_fake.detach(), full_active)
                discriminator_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score)) +
                    F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
                )
                discriminator_loss.backward(); discriminator_optimizer.step()
                adversarial_loss = F.binary_cross_entropy_with_logits(
                    discriminator(full_fake, full_active), torch.ones_like(real_score)
                )
                losses["loss"] = losses["loss"] + float(cfg.get("adversarial_loss_weight", 0.05)) * adversarial_loss
            generator_optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 1.0)))
            generator_optimizer.step()
            for key in ("loss", "position_loss", "walkability_loss", "stop_loss"):
                sums[key] += float(losses[key].detach().cpu())
            sums["adversarial_loss"] += float(adversarial_loss.detach().cpu())
            sums["discriminator_loss"] += float(discriminator_loss.detach().cpu()); batches += 1

        val = validation(model, val_loader, device, cfg)
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / max(batches, 1),
            "position_loss": sums["position_loss"] / max(batches, 1),
            "walkability_loss": sums["walkability_loss"] / max(batches, 1),
            "stop_loss": sums["stop_loss"] / max(batches, 1),
            "adversarial_loss": sums["adversarial_loss"] / max(batches, 1),
            "discriminator_loss": sums["discriminator_loss"] / max(batches, 1),
            "val_loss": val["loss"],
            "epoch_seconds": time.time() - started,
        }
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)
        payload = {
            "format_version": 1, "method_id": args.method_id, "architecture": args.architecture,
            "model_state_dict": model.state_dict(), "model_config": cfg,
            "data_config": {"dataset_path": str(dataset_path), "dataset_name": dataset_path.name,
                            "dataset_id": cfg.get("dataset_id"), "frame_stride": cfg.get("frame_stride")},
            "epoch": epoch,
        }
        if discriminator:
            payload["discriminator_state_dict"] = discriminator.state_dict()
        latest = layout.checkpoints / "latest_model.pth"; torch.save(payload, latest)
        update_checkpoint_manifest(layout.root, latest, "latest")
        if val["loss"] < best_val:
            best_val = val["loss"]; bad_epochs = 0
            best = layout.checkpoints / "best_model.pth"; torch.save(payload, best)
            update_checkpoint_manifest(layout.root, best, "best")
        else:
            bad_epochs += 1
        print(f"[epoch {epoch}] train={row['train_loss']:.5f} val={val['loss']:.5f} seconds={row['epoch_seconds']:.1f}")
        if bad_epochs >= patience:
            print(f"[train] early stopping after {bad_epochs} non-improving epochs")
            break
    mark_run_completed(layout.root)
    print(f"[train] completed: {layout.root}")


if __name__ == "__main__":
    main()

"""Train LSTM-SF, Transformer-SF, or SGAN-SF under one fair data contract."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from baseline_output import create_run_layout, mark_run_completed, update_checkpoint_manifest
from joint_sf import JointSceneDataset, JointSocialForcePredictor, SceneDiscriminator, trajectory_losses

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on the local training env
    tqdm = None


DISPLAY_NAMES = {
    "Method_Transformer_SF_01": "Social-Force-Informed Joint Multi-Agent Transformer",
    "Method_LSTM_SF_01": "Social-Force-Informed Joint Multi-Agent LSTM",
    "Method_SGAN_SF_01": "Social-Force-Informed Joint Multi-Agent Social GAN",
}


class CaseWindowBatchSampler(Sampler[list[int]]):
    """Shuffle cases while keeping each case's windows close for dataset cache hits."""

    def __init__(
        self,
        dataset: JointSceneDataset,
        *,
        batch_size: int,
        seed: int,
        shuffle_cases: bool = True,
        shuffle_windows: bool = True,
        case_fraction_per_epoch: float = 1.0,
        group_fraction_by_plan: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.seed = int(seed)
        self.shuffle_cases = bool(shuffle_cases)
        self.shuffle_windows = bool(shuffle_windows)
        self.case_fraction_per_epoch = min(max(float(case_fraction_per_epoch), 0.0), 1.0)
        self.group_fraction_by_plan = bool(group_fraction_by_plan)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.plan_groups = self._make_plan_groups()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _plan_key(case_dir: pathlib.Path) -> str:
        case_id = case_dir.name.removeprefix("case_")
        pieces = case_id.split("_")
        return "_".join(pieces[:3]) if len(pieces) >= 3 else case_id

    def _make_plan_groups(self) -> list[list[int]]:
        groups: dict[str, list[int]] = {}
        for case_index, case_dir in enumerate(self.dataset.case_dirs):
            groups.setdefault(self._plan_key(case_dir), []).append(case_index)
        return [groups[key] for key in sorted(groups)]

    def selected_case_indices(self) -> list[int]:
        rng = np.random.default_rng(self.seed + self.epoch * 1000003)
        if self.case_fraction_per_epoch >= 1.0:
            case_indices = np.arange(len(self.dataset.case_dirs), dtype=np.int64)
            if self.shuffle_cases:
                rng.shuffle(case_indices)
            return [int(item) for item in case_indices]
        if self.group_fraction_by_plan:
            group_indices = np.arange(len(self.plan_groups), dtype=np.int64)
            if self.shuffle_cases:
                rng.shuffle(group_indices)
            group_count = max(1, int(np.ceil(len(group_indices) * self.case_fraction_per_epoch)))
            selected = []
            for group_index in group_indices[:group_count]:
                selected.extend(self.plan_groups[int(group_index)])
            if self.shuffle_cases:
                rng.shuffle(selected)
            return [int(item) for item in selected]
        case_indices = np.arange(len(self.dataset.case_dirs), dtype=np.int64)
        if self.shuffle_cases:
            rng.shuffle(case_indices)
        case_count = max(1, int(np.ceil(len(case_indices) * self.case_fraction_per_epoch)))
        return [int(item) for item in case_indices[:case_count]]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + 17)
        batch = []
        for case_index in self.selected_case_indices():
            window_indices = np.arange(self.dataset.windows_per_case, dtype=np.int64)
            if self.shuffle_windows:
                rng.shuffle(window_indices)
            base = int(case_index) * self.dataset.windows_per_case
            for window_index in window_indices:
                batch.append(base + int(window_index))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        sample_count = len(self.selected_case_indices()) * self.dataset.windows_per_case
        if self.drop_last:
            return sample_count // self.batch_size
        return (sample_count + self.batch_size - 1) // self.batch_size


class BatchProgress:
    def __init__(self, label: str, total: int, *, fallback_interval: int):
        self.label = label
        self.total = max(int(total), 1)
        self.fallback_interval = max(int(fallback_interval), 1)
        self.last_print = 0.0
        self.bar = tqdm(total=self.total, desc=label, unit="batch", dynamic_ncols=True, file=sys.stdout) if tqdm else None

    def update(self, count: int, **metrics: float) -> None:
        if self.bar:
            self.bar.update(1)
            if metrics:
                self.bar.set_postfix(
                    {key: f"{value:.5f}" for key, value in metrics.items()},
                    refresh=False,
                )
            return
        now = time.time()
        if count == 1 or count == self.total or count % self.fallback_interval == 0 or now - self.last_print >= 10.0:
            metric_text = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            suffix = f" {metric_text}" if metric_text else ""
            print(f"{self.label} batch {count}/{self.total}{suffix}", flush=True)
            self.last_print = now

    def close(self) -> None:
        if self.bar:
            self.bar.close()


def case_id_text(raw: dict) -> str:
    case_ids = raw.get("case_id", [])
    if isinstance(case_ids, str):
        return case_ids
    if isinstance(case_ids, (list, tuple)):
        return ", ".join(str(item) for item in case_ids)
    return str(case_ids)


def require_finite_losses(losses: dict[str, torch.Tensor], raw: dict, *, epoch: int, batch_index: int) -> None:
    bad = {key: float(value.detach().cpu()) for key, value in losses.items() if not torch.isfinite(value).all()}
    if not bad:
        return
    raise FloatingPointError(
        f"non-finite loss at epoch={epoch} batch={batch_index} "
        f"case_id={case_id_text(raw)} losses={bad}"
    )


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


def cuda_autocast(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)


def validation(model, loader, device, cfg, *, epoch: int, label: str, fallback_interval: int, use_amp: bool) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "position_loss": 0.0, "walkability_loss": 0.0, "stop_loss": 0.0}
    batches = 0
    progress = BatchProgress(label, len(loader), fallback_interval=fallback_interval)
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            with cuda_autocast(use_amp):
                outputs = model_rollout(model, batch, cfg, 0.0)
                obs_len = int(cfg.get("obs_len", 8))
                losses = trajectory_losses(
                    outputs, batch["positions"][:, :, obs_len:], batch["active"][:, :, obs_len:], batch["walkable"],
                    walkability_weight=cfg.get("walkability_loss_weight", 0.1), stop_weight=cfg.get("stop_loss_weight", 0.2),
                )
            require_finite_losses(losses, raw, epoch=epoch, batch_index=batches + 1)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            batches += 1
            progress.update(batches, loss=totals["loss"] / batches)
    progress.close()
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
    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 0))
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", device.type == "cuda")),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    grouped_batches = bool(cfg.get("case_grouped_batches", True))
    train_sampler = None
    if grouped_batches:
        train_sampler = CaseWindowBatchSampler(
            train_ds,
            batch_size=batch_size,
            seed=seed,
            shuffle_cases=bool(cfg.get("shuffle_train_cases", True)),
            shuffle_windows=bool(cfg.get("shuffle_train_windows", True)),
            case_fraction_per_epoch=float(cfg.get("case_fraction_per_epoch", 1.0)),
            group_fraction_by_plan=bool(cfg.get("group_fraction_by_plan", True)),
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    use_amp = bool(cfg.get("amp", device.type == "cuda")) and device.type == "cuda"
    print(
        f"[train] train_cases={len(train_ds.case_dirs)} train_batches={len(train_loader)} "
        f"val_cases={len(val_ds.case_dirs)} val_batches={len(val_loader)} "
        f"grouped_batches={grouped_batches} workers={num_workers} amp={use_amp} "
        f"case_fraction_per_epoch={cfg.get('case_fraction_per_epoch', 1.0)}",
        flush=True,
    )
    model = make_model(cfg, args.architecture).to(device)
    generator_optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("learning_rate", cfg.get("lr", 3e-4))),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    discriminator = SceneDiscriminator(cfg.get("hidden_dim", 128)).to(device) if args.architecture == "sgan" else None
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(cfg.get("discriminator_lr", 2e-4))) if discriminator else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history_path = layout.logs / "training_history.csv"
    fields = ["epoch", "train_loss", "position_loss", "walkability_loss", "stop_loss", "adversarial_loss", "discriminator_loss", "val_loss", "epoch_seconds"]
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    best_val = float("inf")
    patience = int(cfg.get("early_stopping_patience", 10)); bad_epochs = 0
    epochs = int(cfg.get("epochs", 100))
    progress_interval = int(cfg.get("progress_interval_batches", max(1, len(train_loader) // 20)))

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            selected_cases = len(train_sampler.selected_case_indices())
            print(
                f"[epoch {epoch}] sampled_train_cases={selected_cases}/{len(train_ds.case_dirs)} "
                f"train_batches={len(train_loader)}",
                flush=True,
            )
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
        epoch_progress = (epoch - 1) / max(epochs - 1, 1)
        teacher_ratio = ratio_start + (ratio_end - ratio_start) * epoch_progress
        batch_progress = BatchProgress(
            f"epoch {epoch}/{epochs} train",
            len(train_loader),
            fallback_interval=progress_interval,
        )
        for raw in train_loader:
            batch = move_batch(raw, device)
            with cuda_autocast(use_amp):
                obs_len = int(cfg.get("obs_len", 8))
                outputs = model_rollout(model, batch, cfg, teacher_ratio)
                losses = trajectory_losses(
                    outputs, batch["positions"][:, :, obs_len:], batch["active"][:, :, obs_len:], batch["walkable"],
                    walkability_weight=cfg.get("walkability_loss_weight", 0.1), stop_weight=cfg.get("stop_loss_weight", 0.2),
                )
            require_finite_losses(losses, raw, epoch=epoch, batch_index=batches + 1)
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
                scaler.scale(discriminator_loss).backward()
                scaler.step(discriminator_optimizer)
                with cuda_autocast(use_amp):
                    adversarial_loss = F.binary_cross_entropy_with_logits(
                        discriminator(full_fake, full_active), torch.ones_like(real_score)
                    )
                losses["loss"] = losses["loss"] + float(cfg.get("adversarial_loss_weight", 0.05)) * adversarial_loss
            generator_optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(generator_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("gradient_clip", 1.0)))
            scaler.step(generator_optimizer)
            scaler.update()
            for key in ("loss", "position_loss", "walkability_loss", "stop_loss"):
                sums[key] += float(losses[key].detach().cpu())
            sums["adversarial_loss"] += float(adversarial_loss.detach().cpu())
            sums["discriminator_loss"] += float(discriminator_loss.detach().cpu()); batches += 1
            batch_progress.update(batches, loss=sums["loss"] / batches, teacher=teacher_ratio)
        batch_progress.close()

        val = validation(
            model,
            val_loader,
            device,
            cfg,
            epoch=epoch,
            label=f"epoch {epoch}/{epochs} val",
            fallback_interval=max(1, len(val_loader) // 10),
            use_amp=use_amp,
        )
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

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from action_space import ActionSpace, build_action_space
from dataset_grid_policy import GridPolicyDataset
from model_grid_policy import GridSocialPolicyNet

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
BASELINE_TRAIN_DIR = PROJECT_ROOT / "AI_GenerateTimeseries" / "AI_Train"
if str(BASELINE_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_TRAIN_DIR))
from baseline_output import (  # noqa: E402
    create_run_layout,
    mark_run_completed,
    update_checkpoint_manifest,
)


def load_json(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PlanFractionBatchSampler(Sampler[list[int]]):
    """Sample a rotating fraction of training plans each epoch."""

    def __init__(
        self,
        dataset: GridPolicyDataset,
        *,
        batch_size: int,
        seed: int,
        plan_fraction_per_epoch: float,
        shuffle_plans: bool = True,
        shuffle_samples: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.seed = int(seed)
        self.plan_fraction_per_epoch = min(max(float(plan_fraction_per_epoch), 0.0), 1.0)
        self.shuffle_plans = bool(shuffle_plans)
        self.shuffle_samples = bool(shuffle_samples)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.plan_to_cases = self._make_plan_groups()
        self.case_to_sample_indices = self._make_case_sample_indices()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_plan_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for case_index, row in enumerate(self.dataset.cases):
            groups.setdefault(str(row["plan_name"]), []).append(case_index)
        return groups

    def _make_case_sample_indices(self) -> dict[int, list[int]]:
        indices: dict[int, list[int]] = {}
        for sample_index, (case_index, _) in enumerate(self.dataset.samples):
            indices.setdefault(case_index, []).append(sample_index)
        return indices

    def selected_plan_names(self) -> list[str]:
        names = sorted(self.plan_to_cases)
        if self.plan_fraction_per_epoch >= 1.0:
            return names
        rng = np.random.default_rng(self.seed + self.epoch * 1000003)
        indices = np.arange(len(names), dtype=np.int64)
        if self.shuffle_plans:
            rng.shuffle(indices)
        count = max(1, int(np.ceil(len(indices) * self.plan_fraction_per_epoch)))
        return [names[int(index)] for index in indices[:count]]

    def selected_case_indices(self) -> list[int]:
        cases: list[int] = []
        for plan_name in self.selected_plan_names():
            cases.extend(self.plan_to_cases[plan_name])
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + 17)
        if self.shuffle_plans:
            rng.shuffle(cases)
        return [int(item) for item in cases]

    def selected_sample_indices(self) -> list[int]:
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + 31)
        samples: list[int] = []
        for case_index in self.selected_case_indices():
            case_samples = list(self.case_to_sample_indices.get(case_index, ()))
            if self.shuffle_samples:
                rng.shuffle(case_samples)
            samples.extend(case_samples)
        return samples

    def __iter__(self):
        batch: list[int] = []
        for sample_index in self.selected_sample_indices():
            batch.append(sample_index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        sample_count = len(self.selected_sample_indices())
        if self.drop_last:
            return sample_count // self.batch_size
        return (sample_count + self.batch_size - 1) // self.batch_size


def make_loader(cfg: dict, split: str, action_space: ActionSpace, shuffle: bool) -> tuple[DataLoader, PlanFractionBatchSampler | None]:
    per_case_key = "max_samples_per_case_train" if split == "train" else "max_samples_per_case_eval"
    max_cases_key = "max_train_cases" if split == "train" else "max_eval_cases"
    dataset = GridPolicyDataset(
        dataset_root=pathlib.Path(cfg["dataset_root"]),
        split=split,
        action_space=action_space,
        crop_size=int(cfg["crop_size"]),
        max_samples_per_case=int(cfg[per_case_key]),
        max_cases=cfg.get(max_cases_key),
        seed=int(cfg.get("seed", 42)),
        cache_size=int(cfg.get("case_cache_size", 4)),
        action_frame_stride=int(cfg.get("action_frame_stride", 1)),
    )
    num_workers = int(cfg.get("num_workers", 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", False)),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    plan_fraction = float(cfg.get("plan_fraction_per_epoch", 1.0)) if split == "train" else 1.0
    if split == "train" and bool(cfg.get("plan_fraction_batches", plan_fraction < 1.0)):
        sampler = PlanFractionBatchSampler(
            dataset,
            batch_size=int(cfg["batch_size"]),
            seed=int(cfg.get("seed", 42)),
            plan_fraction_per_epoch=plan_fraction,
            shuffle_plans=bool(cfg.get("shuffle_train_plans", True)),
            shuffle_samples=bool(cfg.get("shuffle_train_samples", True)),
            drop_last=True,
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs), sampler
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        drop_last=split == "train",
        **kwargs,
    ), None


def make_action_loss_weights(action_space: ActionSpace, cfg: dict, device: torch.device) -> torch.Tensor | None:
    wait_weight = float(cfg.get("wait_loss_weight", 1.0))
    move_weight = float(cfg.get("move_loss_weight", 1.0))
    if wait_weight == 1.0 and move_weight == 1.0:
        return None
    weights = torch.full((action_space.num_actions,), move_weight, dtype=torch.float32, device=device)
    weights[action_space.wait_action_id] = wait_weight
    return weights


def run_epoch(model, loader, optimizer, device, train: bool, stop_loss_weight: float, action_loss_weights: torch.Tensor | None = None):
    model.train(train)
    ce_loss = nn.CrossEntropyLoss(weight=action_loss_weights)
    bce_loss = nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "action_loss": 0.0, "stop_loss": 0.0, "action_acc": 0.0, "stop_acc": 0.0}
    batches = 0
    label = "Train" if train else "Val"
    progress = tqdm(loader, desc=label, leave=False, dynamic_ncols=True)

    for batch in progress:
        grid_map = batch["map"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        action_target = batch["action"].to(device, non_blocking=True)
        stop_target = batch["stop"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            action_logits, stop_logits = model(grid_map, features)
            action_loss = ce_loss(action_logits, action_target)
            stop_loss = bce_loss(stop_logits, stop_target)
            loss = action_loss + stop_loss_weight * stop_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        action_pred = torch.argmax(action_logits, dim=1)
        stop_pred = (torch.sigmoid(stop_logits) >= 0.5).float()
        totals["loss"] += float(loss.detach().cpu())
        totals["action_loss"] += float(action_loss.detach().cpu())
        totals["stop_loss"] += float(stop_loss.detach().cpu())
        totals["action_acc"] += float((action_pred == action_target).float().mean().detach().cpu())
        totals["stop_acc"] += float((stop_pred == stop_target).float().mean().detach().cpu())
        batches += 1
        progress.set_postfix(loss=f"{totals['loss'] / batches:.4f}", action_acc=f"{totals['action_acc'] / batches:.3f}", stop_acc=f"{totals['stop_acc'] / batches:.3f}")

    return {k: v / max(batches, 1) for k, v in totals.items()}


def save_checkpoint(path: pathlib.Path, model, optimizer, epoch: int, cfg: dict, action_space_payload: dict, best_val_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "action_space": action_space_payload,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def make_optimizer(model, cfg: dict):
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg["learning_rate"])
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer={optimizer_name}. Use adamw or adam.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GridSocialPolicyNet v0.")
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config_train.json"))
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = args.config if args.config.is_absolute() else script_dir / args.config
    cfg = load_json(config_path)
    if not cfg.get("sf_implementation_ready", False):
        raise RuntimeError(
            "Method_GridSocialPolicy_SF_01 is a protected baseline copy: implement and validate the "
            "Social-Force additions, then set sf_implementation_ready=true before training."
        )

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_root = pathlib.Path(cfg["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (config_path.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    cfg["dataset_root"] = str(dataset_root)

    if cfg.get("output_root"):
        outputs_root = pathlib.Path(cfg["output_root"])
        if not outputs_root.is_absolute():
            outputs_root = (config_path.parent / outputs_root).resolve()
        else:
            outputs_root = outputs_root.resolve()
        if outputs_root.name == "Method_GridSocialPolicy_SF_01":
            outputs_root = outputs_root / "outputs"
    else:
        outputs_root = script_dir.parents[1] / "AI_Result" / "Method_GridSocialPolicy_SF_01" / "outputs"
    dataset_manifest = dataset_root / "manifest_trajectory_grid.csv"
    run_layout = create_run_layout(
        outputs_root,
        method_id="Method_GridSocialPolicy_SF_01",
        method_display_name="Social-Force-Conditioned Discrete Grid Policy",
        method_family="conditional_discrete_grid_policy",
        seed=seed,
        dataset_id=cfg.get("dataset_id", "housegan_canonical_imagebase_split_v1"),
        config=cfg,
        dataset_manifest=dataset_manifest if dataset_manifest.exists() else None,
        project_root=PROJECT_ROOT,
    )
    run_dir = run_layout.root

    action_space_path = pathlib.Path(cfg.get("action_space_path", run_dir / "action_space.json"))
    if not action_space_path.is_absolute():
        action_space_path = run_dir / action_space_path
    if bool(cfg.get("rebuild_action_space", True)) or not action_space_path.exists():
        action_space_payload = build_action_space(
            dataset_root=dataset_root,
            output_path=action_space_path,
            movement_count=int(cfg.get("movement_action_count", 20)),
            split="train",
            max_cases=cfg.get("action_space_max_cases"),
            frame_stride=int(cfg.get("action_frame_stride", 1)),
        )
    else:
        action_space_payload = load_json(action_space_path)
    action_space = ActionSpace(action_space_payload)

    device = get_device()
    print(f"[SYSTEM] torch={torch.__version__} device={device}")
    print(f"[DATA] dataset_root={dataset_root}")
    print(f"[RUN] {run_dir}")
    print(f"[ACTION] actions={action_space.num_actions} wait_id={action_space.wait_action_id}")

    train_loader, train_sampler = make_loader(cfg, "train", action_space, shuffle=bool(cfg.get("dataloader_shuffle", False)))
    val_loader, _ = make_loader(cfg, "val", action_space, shuffle=False)
    if train_sampler is not None:
        train_sampler.set_epoch(1)
    print(f"[DATA] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)}")
    print(
        f"[DATA] train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"plan_fraction_per_epoch={cfg.get('plan_fraction_per_epoch', 1.0)}"
    )

    model = GridSocialPolicyNet(
        num_actions=action_space.num_actions,
        feature_dim=int(cfg.get("feature_dim", 12)),
        base_channels=int(cfg.get("base_channels", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=int(cfg.get("lr_patience", 3)))
    action_loss_weights = make_action_loss_weights(action_space, cfg, device)

    with (run_dir / "model_architecture.txt").open("w", encoding="utf-8") as f:
        f.write(str(model))

    metrics_path = run_layout.logs / "training_history.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_action_loss",
                "train_stop_loss",
                "train_action_acc",
                "train_stop_acc",
                "val_loss",
                "val_action_loss",
                "val_stop_loss",
                "val_action_acc",
                "val_stop_acc",
                "lr",
            ],
        )
        writer.writeheader()

    best_val_loss = float("inf")
    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        print(f"[EPOCH] {epoch}/{epochs}")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            print(
                f"[EPOCH] sampled_train_plans={len(train_sampler.selected_plan_names())}/{len(train_sampler.plan_to_cases)} "
                f"sampled_train_cases={len(train_sampler.selected_case_indices())}/{len(train_loader.dataset.cases)} "
                f"train_batches={len(train_loader)}"
            )
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            stop_loss_weight=float(cfg.get("stop_loss_weight", 1.0)),
            action_loss_weights=action_loss_weights,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            stop_loss_weight=float(cfg.get("stop_loss_weight", 1.0)),
            action_loss_weights=action_loss_weights,
        )
        scheduler.step(val_metrics["loss"])
        lr = optimizer.param_groups[0]["lr"]

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_action_loss",
                    "train_stop_loss",
                    "train_action_acc",
                    "train_stop_acc",
                    "val_loss",
                    "val_action_loss",
                    "val_stop_loss",
                    "val_action_acc",
                    "val_stop_acc",
                    "lr",
                ],
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_action_loss": train_metrics["action_loss"],
                    "train_stop_loss": train_metrics["stop_loss"],
                    "train_action_acc": train_metrics["action_acc"],
                    "train_stop_acc": train_metrics["stop_acc"],
                    "val_loss": val_metrics["loss"],
                    "val_action_loss": val_metrics["action_loss"],
                    "val_stop_loss": val_metrics["stop_loss"],
                    "val_action_acc": val_metrics["action_acc"],
                    "val_stop_acc": val_metrics["stop_acc"],
                    "lr": lr,
                }
            )

        latest_path = run_layout.checkpoints / "latest_model.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, cfg, action_space_payload, best_val_loss)
        update_checkpoint_manifest(run_dir, latest_path, "latest")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = run_layout.checkpoints / "best_model.pth"
            save_checkpoint(best_path, model, optimizer, epoch, cfg, action_space_payload, best_val_loss)
            update_checkpoint_manifest(run_dir, best_path, "best")

        print(
            f"[METRIC] train_loss={train_metrics['loss']:.4f} train_action={train_metrics['action_acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_action={val_metrics['action_acc']:.3f} val_stop={val_metrics['stop_acc']:.3f}"
        )

    print(f"[DONE] best_val_loss={best_val_loss:.4f}")
    mark_run_completed(run_dir)
    print(f"[DONE] checkpoint={run_layout.checkpoints / 'best_model.pth'}")


if __name__ == "__main__":
    main()

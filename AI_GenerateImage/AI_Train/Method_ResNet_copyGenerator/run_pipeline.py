from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import GeneratorNetwork


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
EXPECTED = {
    "train": (2603, 412),
    "validation": (439, 60),
    "test": (862, 117),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: pathlib.Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def load_config() -> dict:
    config = json.loads((SCRIPT_DIR / "config.json").read_text(encoding="utf-8"))
    if config["image_size"] != 256 or config["seed"] != 42:
        raise ValueError("This experiment is locked to image_size=256 and seed=42")
    return config


def dataset_root(config: dict) -> pathlib.Path:
    path = pathlib.Path(config["dataset_root"])
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def verify_dataset(root: pathlib.Path, snapshot: pathlib.Path | None = None):
    inventory = {}
    plan_sets = {}
    rows = []
    for split, (expected_cases, expected_plans) in EXPECTED.items():
        a_names = {p.name for p in (root / "A" / split).glob("*.png")}
        b_names = {p.name for p in (root / "B" / split).glob("*.png")}
        paired = sorted(a_names & b_names)
        plans = {name.split("__", 1)[0] for name in paired}
        if len(paired) != expected_cases or len(plans) != expected_plans:
            raise RuntimeError(
                f"Canonical {split} mismatch: cases={len(paired)}, plans={len(plans)}"
            )
        if a_names != b_names:
            raise RuntimeError(f"Canonical {split} A/B filenames do not match")
        inventory[split] = {"cases": len(paired), "plans": len(plans)}
        plan_sets[split] = plans
        rows.extend((split, name, name.split("__", 1)[0]) for name in paired)
    overlap = {
        "train_validation": len(plan_sets["train"] & plan_sets["validation"]),
        "train_test": len(plan_sets["train"] & plan_sets["test"]),
        "validation_test": len(plan_sets["validation"] & plan_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Canonical split overlap detected: {overlap}")
    if snapshot is not None:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["split", "file_name", "plan_id"])
            writer.writerows(rows)
    return inventory, overlap


class DensityDataset(Dataset):
    """Exact Pix2PixHD RGB/Tanh data representation."""

    def __init__(self, root: pathlib.Path, split: str, size: int):
        self.dir_a = root / "A" / split
        self.dir_b = root / "B" / split
        self.names = sorted(p.name for p in self.dir_a.glob("*.png"))
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]
        image_a = Image.open(self.dir_a / name).convert("RGB").resize(
            (self.size, self.size), Image.Resampling.BICUBIC
        )
        image_b = Image.open(self.dir_b / name).convert("RGB").resize(
            (self.size, self.size), Image.Resampling.BICUBIC
        )
        input_tensor = torch.from_numpy(np.asarray(image_a, dtype=np.float32) / 255.0)
        target_tensor = torch.from_numpy(np.asarray(image_b, dtype=np.float32) / 255.0)
        input_tensor = input_tensor.permute(2, 0, 1).contiguous() * 2.0 - 1.0
        target_tensor = target_tensor.permute(2, 0, 1).contiguous() * 2.0 - 1.0
        return input_tensor, target_tensor, name


def density_aware_l1(prediction, target, config):
    target_01 = ((target + 1.0) * 0.5).clamp(0.0, 1.0)
    target_gray = target_01.mean(dim=1, keepdim=True)
    weights = torch.ones_like(target_gray)
    weights += (target_gray > config["density_foreground_threshold"]).float() * config["density_foreground_weight"]
    weights += target_gray * config["density_intensity_weight"]
    return (torch.abs(prediction - target) * weights).mean()


def make_loader(dataset, config, shuffle: bool, epoch: int = 0):
    generator = torch.Generator().manual_seed(config["seed"] * 100_000 + epoch)
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers_train"] if shuffle else config["num_workers_eval"],
        pin_memory=True,
        worker_init_fn=worker_seed,
        generator=generator,
    )


def atomic_save(payload, path: pathlib.Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def create_run(config: dict) -> pathlib.Path:
    run_id = f"run_{utc_stamp()}_seed{config['seed']:03d}"
    run_dir = REPO_ROOT / "AI_GenerateImage" / "AI_Result" / config["method_id"] / "outputs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "evaluations"):
        (run_dir / name).mkdir()
    shutil.copy2(SCRIPT_DIR / "config.json", run_dir / "config_train.json")
    shutil.copy2(SCRIPT_DIR / "config_test.json", run_dir / "config_test.json")
    inventory, overlap = verify_dataset(dataset_root(config), run_dir / "dataset_manifest_snapshot.csv")
    write_json(run_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    })
    write_json(run_dir / "code_provenance.json", {
        "created_at_utc": utc_now(),
        "source_generator": "AI_GenerateImage/AI_Train/Method_pix2pixHD/train_pix2pixHD_densitymap_bw.py::GeneratorNetwork",
        "files": {
            name: sha256_file(SCRIPT_DIR / name)
            for name in ("run_pipeline.py", "model.py", "config.json", "config_test.json")
        },
    })
    write_json(run_dir / "run_manifest.json", {
        "run_id": run_id,
        "method_id": config["method_id"],
        "seed": config["seed"],
        "dataset_id": config["dataset_id"],
        "dataset_inventory": inventory,
        "plan_overlap": overlap,
        "train_split": "train",
        "model_selection_split": "validation",
        "architecture": "exact_pix2pixhd_generator_rgb_tanh_affine_instance_norm",
        "objective": "density_aware_l1_only",
        "discriminator": False,
        "status": "created",
        "research_valid": False,
        "research_valid_reason": "Training and canonical test evaluation not complete",
    })
    return run_dir


def train(config: dict, run_dir: pathlib.Path) -> None:
    root = dataset_root(config)
    configure_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneratorNetwork(3, 3, 9).to(device)
    initial_hash = state_dict_sha256(model)
    optimizer = optim.Adam(
        model.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"])
    )
    train_ds = DensityDataset(root, "train", 256)
    val_ds = DensityDataset(root, "validation", 256)
    val_loader = make_loader(val_ds, config, False)
    best_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    elapsed_before = 0.0
    latest = run_dir / "checkpoints" / "latest_resume.pt"
    if latest.exists():
        payload = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(payload["generator"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = payload["epoch"] + 1
        best_loss = payload["best_validation_loss"]
        best_epoch = payload["best_epoch"]
        elapsed_before = payload.get("training_wall_time_s", 0.0)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "training", "initial_generator_sha256": initial_hash})
    write_json(manifest_path, manifest)
    history = run_dir / "logs" / "training_history.csv"
    if not history.exists():
        history.write_text("epoch,train_density_l1,validation_density_l1,epoch_time_s,elapsed_time_s\n", encoding="utf-8")
    started = time.perf_counter()
    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_sum = 0.0
        loader = make_loader(train_ds, config, True, epoch)
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = density_aware_l1(model(inputs), targets, config)
            loss = reconstruction * config["l1_loss_weight"]
            loss.backward()
            optimizer.step()
            train_sum += float(reconstruction.detach())
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                prediction = model(inputs.to(device, non_blocking=True))
                validation_sum += float(density_aware_l1(prediction, targets.to(device, non_blocking=True), config))
        validation_loss = validation_sum / len(val_loader)
        epoch_time = time.perf_counter() - epoch_started
        elapsed = elapsed_before + time.perf_counter() - started
        with history.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([
                epoch, train_sum / len(loader), validation_loss, epoch_time, elapsed
            ])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            atomic_save({
                "generator": model.state_dict(),
                "epoch": epoch,
                "validation_density_l1": validation_loss,
                "method_id": config["method_id"],
                "dataset_id": config["dataset_id"],
                "seed": config["seed"],
                "architecture": "exact_pix2pixhd_generator_rgb_tanh_affine_instance_norm",
                "objective": "density_aware_l1_only",
                "initial_generator_sha256": initial_hash,
            }, run_dir / "checkpoints" / "best_model.pt")
        atomic_save({
            "generator": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_wall_time_s": elapsed,
        }, latest)
        eta = (elapsed / epoch) * (config["epochs"] - epoch)
        print(
            f"[epoch {epoch:02d}/{config['epochs']}] train={train_sum / len(loader):.8f} "
            f"val={validation_loss:.8f} best={best_loss:.8f}@{best_epoch} "
            f"time={epoch_time:.1f}s eta={eta / 60:.1f}min",
            flush=True,
        )
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    write_json(run_dir / "checkpoints" / "checkpoint_manifest.json", {
        "checkpoint": "best_model.pt",
        "sha256": sha256_file(checkpoint),
        "best_epoch": best_epoch,
        "best_validation_density_l1": best_loss,
        "initial_generator_sha256": initial_hash,
    })
    manifest.update({
        "status": "trained",
        "trained_at_utc": utc_now(),
        "best_epoch": best_epoch,
        "best_validation_density_l1": best_loss,
        "checkpoint_sha256": sha256_file(checkpoint),
    })
    write_json(manifest_path, manifest)


def evaluate(config: dict, run_dir: pathlib.Path) -> None:
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_checkpoint = run_dir / "checkpoints" / "generator_best.pth"
    atomic_save(payload["generator"], raw_checkpoint)
    evaluator = SCRIPT_DIR.parent / "Method_pix2pixHD" / "test_pix2pixHD_densitymap_bw.py"
    subprocess.run([
        sys.executable,
        str(evaluator),
        "--run_path", str(run_dir),
        "--config", str(SCRIPT_DIR / "config_test.json"),
    ], check=True)
    runtime_path = run_dir / "test_runtime.csv"
    rows = list(csv.DictReader(runtime_path.open(encoding="utf-8")))
    if len(rows) != 1 or int(rows[0]["sample_count"]) != EXPECTED["test"][0]:
        raise RuntimeError("Evaluator did not produce the complete canonical test artifact")
    rows[0]["method_id"] = config["method_id"]
    with runtime_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    eval_dir = run_dir / "evaluations" / "eval_test_runtime_v2"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in ("test_runtime.csv", "test_evaluation_summary.csv", "test_evaluation_per_image.csv"):
        shutil.copy2(run_dir / name, eval_dir / name)
    checkpoint_sha = sha256_file(checkpoint)
    write_json(eval_dir / "evaluation_manifest.json", {
        "evaluation_id": f"eval_{config['dataset_id']}_test_{config['protocol_id']}",
        "method_id": config["method_id"],
        "run_id": run_dir.name,
        "checkpoint_path": "checkpoints/best_model.pt",
        "checkpoint_sha256": checkpoint_sha,
        "dataset_id": config["dataset_id"],
        "split": "test",
        "case_count": EXPECTED["test"][0],
        "floorplan_count": EXPECTED["test"][1],
        "image_size": 256,
        "constraint_mode": "none",
        "seed": config["seed"],
        "objective": "density_aware_l1_only",
        "discriminator": False,
        "research_valid": True,
        "created_at_utc": utc_now(),
    })
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "evaluated",
        "evaluated_at_utc": utc_now(),
        "checkpoint_sha256": checkpoint_sha,
        "test_case_count": EXPECTED["test"][0],
        "test_floorplan_count": EXPECTED["test"][1],
        "research_valid": True,
        "research_valid_reason": "Canonical train/validation/test protocol complete for this run",
        "comparison_limitation": "The retained original Pix2PixHD run has no recorded seed; cross-run comparison is descriptive, not a paired-seed factorial effect.",
    })
    write_json(manifest_path, manifest)


def resolve_run(value: str | None, config: dict) -> pathlib.Path:
    if value:
        return pathlib.Path(value).resolve()
    root = REPO_ROOT / "AI_GenerateImage" / "AI_Result" / config["method_id"] / "outputs"
    runs = sorted(path for path in root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError("No run exists; use --stage train or --stage all")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("plan", "train", "evaluate", "all"), default="plan")
    parser.add_argument("--run-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config()
    inventory, overlap = verify_dataset(dataset_root(config))
    print(json.dumps({
        "method": config["method_id"],
        "architecture": "exact Pix2PixHD Generator only",
        "discriminator": False,
        "seed": config["seed"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "image_size": config["image_size"],
        "inventory": inventory,
        "plan_overlap": overlap,
    }, indent=2))
    if args.stage == "plan" or args.dry_run:
        return
    run_dir = resolve_run(args.run_path, config) if args.stage == "evaluate" else create_run(config)
    print(f"RUN_DIR={run_dir}", flush=True)
    if args.stage in {"train", "all"}:
        train(config, run_dir)
    if args.stage in {"evaluate", "all"}:
        evaluate(config, run_dir)


if __name__ == "__main__":
    main()

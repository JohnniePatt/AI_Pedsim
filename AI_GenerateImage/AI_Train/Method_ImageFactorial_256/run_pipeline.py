from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from factorial_common import (
    EXPECTED_CASES,
    EXPECTED_PLANS,
    FactorialDensityDataset,
    SharedPatchCritic,
    build_generator,
    configure_seed,
    density_metrics,
    gradient_penalty,
    jet_image,
    project_root,
    read_json,
    resolve_path,
    sha256_file,
    state_dict_sha256,
    verify_canonical_dataset,
    worker_seed,
    write_json,
)


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FINAL_SUFFIX = "__model_evaluate_256_factorial"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_value(arguments, fallback="unknown"):
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=project_root(), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def load_config(path_value: str):
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    cfg = read_json(path.resolve())
    if cfg.get("image_size") != 256:
        raise ValueError("Factorial protocol is locked to image_size=256")
    seeds = cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Factorial config must contain a non-empty list of unique seeds")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Every factorial seed must be a non-negative integer")
    expected_cells = {"unet_l1", "unet_wgangp", "resnet_l1", "resnet_wgangp"}
    if {cell["cell_id"] for cell in cfg["cells"]} != expected_cells:
        raise ValueError("Factorial config must contain exactly the four required cells")
    return cfg, path.resolve()


def print_plan(cfg):
    dataset_root = resolve_path(cfg["dataset_root"])
    inventory, overlap, _ = verify_canonical_dataset(dataset_root)
    print("2x2 factorial matrix")
    print("architecture | L1-only | WGAN-GP + L1")
    print("U-Net       | Plain U-Net | Pix2Pix WGAN-GP")
    print("ResNet-9    | ResNet-9 | Pix2PixHD factorial variant")
    print(f"seeds={cfg['seeds']} epochs={cfg['epochs']} batch={cfg['batch_size']} size=256")
    print(json.dumps({"inventory": inventory, "plan_overlap": overlap}, indent=2))


def create_experiment(cfg, config_path: pathlib.Path):
    stamp = utc_stamp()
    experiment_dir = (
        project_root()
        / "AI_GenerateImage"
        / "AI_Result"
        / "FactorialExperiments"
        / f"experiment_{stamp}_image_2x2_factorial_256_v1"
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, experiment_dir / "config_factorial.json")
    inventory, overlap, dataset_hash = verify_canonical_dataset(
        resolve_path(cfg["dataset_root"]), experiment_dir / "dataset_manifest_snapshot.csv"
    )
    code_files = [SCRIPT_DIR / "run_pipeline.py", SCRIPT_DIR / "factorial_common.py", config_path]
    write_json(experiment_dir / "code_provenance.json", {
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--porcelain"], fallback="")),
        "files": {
            str(path.relative_to(project_root())): sha256_file(path) for path in code_files
        },
    })
    write_json(experiment_dir / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    })
    runs = []
    for seed in cfg["seeds"]:
        for cell in cfg["cells"]:
            run_name = f"run_{cell['run_stem']}_{stamp}_seed{seed:03d}"
            run_dir = (
                project_root()
                / "AI_GenerateImage"
                / "AI_Result"
                / cell["method_dir"]
                / "outputs"
                / run_name
            )
            runs.append({
                **cell,
                "seed": seed,
                "run_id": run_name,
                "run_dir": str(run_dir),
                "status": "pending",
            })
    state = {
        "experiment_id": cfg["experiment_id"],
        "experiment_dir": str(experiment_dir),
        "created_at_utc": utc_now(),
        "status": "created",
        "dataset_id": cfg["dataset_id"],
        "dataset_inventory": inventory,
        "plan_overlap": overlap,
        "dataset_manifest_sha256": dataset_hash,
        "runs": runs,
    }
    write_json(experiment_dir / "experiment_manifest.json", state)
    return experiment_dir, state


def load_experiment(experiment_dir_value: str):
    experiment_dir = pathlib.Path(experiment_dir_value).resolve()
    return experiment_dir, read_json(experiment_dir / "experiment_manifest.json")


def save_state(experiment_dir: pathlib.Path, state) -> None:
    write_json(experiment_dir / "experiment_manifest.json", state)


def run_dirs(run_dir: pathlib.Path):
    paths = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "samples": run_dir / "samples",
        "test": run_dir / "test_results",
        "evaluations": run_dir / "evaluations",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def atomic_torch_save(payload, path: pathlib.Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def training_loader(dataset, cfg, seed: int, epoch: int):
    generator = torch.Generator().manual_seed(seed * 100_000 + epoch)
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers_train"],
        pin_memory=True,
        worker_init_fn=worker_seed,
        generator=generator,
    )


def train_one(run, cfg, experiment_dir: pathlib.Path):
    run_dir = pathlib.Path(run["run_dir"])
    paths = run_dirs(run_dir)
    dataset_root = resolve_path(cfg["dataset_root"])
    train_dataset = FactorialDensityDataset(dataset_root, "train", 256)
    validation_dataset = FactorialDensityDataset(dataset_root, "validation", 256)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers_eval"],
        pin_memory=True,
    )
    seed = int(run["seed"])
    configure_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = build_generator(run["architecture"]).to(device)
    initial_hash = state_dict_sha256(generator)
    critic = SharedPatchCritic().to(device) if run["objective"] == "wgangp_l1" else None
    generator_optimizer = optim.Adam(
        generator.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )
    critic_optimizer = None
    if critic is not None:
        critic_optimizer = optim.Adam(
            critic.parameters(),
            lr=cfg["learning_rate"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
    criterion = nn.L1Loss()
    latest_path = paths["checkpoints"] / "latest_resume.pt"
    best_path = paths["checkpoints"] / "best_model.pt"
    history_path = paths["logs"] / "training_history.csv"
    start_epoch = 1
    best_epoch = None
    best_validation_l1 = float("inf")
    elapsed_before = 0.0
    if latest_path.exists():
        resume = torch.load(latest_path, map_location=device, weights_only=False)
        generator.load_state_dict(resume["generator"])
        generator_optimizer.load_state_dict(resume["generator_optimizer"])
        if critic is not None:
            critic.load_state_dict(resume["critic"])
            critic_optimizer.load_state_dict(resume["critic_optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = resume["best_epoch"]
        best_validation_l1 = float(resume["best_validation_l1"])
        elapsed_before = float(resume.get("training_wall_time_s", 0.0))
        torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng_state_all"]])

    resolved_config = {
        **cfg,
        "dataset_root": str(dataset_root),
        "cell": {key: run[key] for key in ("cell_id", "method", "architecture", "objective")},
        "seed": seed,
    }
    write_json(run_dir / "config_train.json", resolved_config)
    run_manifest = {
        "run_id": run["run_id"],
        "experiment_id": cfg["experiment_id"],
        "cell_id": run["cell_id"],
        "method": run["method"],
        "architecture": run["architecture"],
        "objective": run["objective"],
        "seed": seed,
        "image_size": 256,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "dataset_id": cfg["dataset_id"],
        "dataset_manifest_sha256": sha256_file(experiment_dir / "dataset_manifest_snapshot.csv"),
        "train_split": "train",
        "model_selection_split": "validation",
        "initial_generator_sha256": initial_hash,
        "status": "training",
        "research_valid": False,
        "research_valid_reason": "Training/evaluation not complete",
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "train_generator_loss", "train_reconstruction_l1", "train_adversarial_loss", "train_critic_loss", "validation_l1", "epoch_time_s"]
            )

    started = time.perf_counter()
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        epoch_started = time.perf_counter()
        generator.train()
        if critic is not None:
            critic.train()
        sums = {"generator": 0.0, "l1": 0.0, "adv": 0.0, "critic": 0.0}
        loader = training_loader(train_dataset, cfg, seed, epoch)
        progress = tqdm(
            loader,
            desc=f"{run['cell_id']} seed{seed} [{epoch:03d}/{cfg['epochs']:03d}]",
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for input_tensor, target_tensor, _ in progress:
            input_tensor = input_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)
            generator_optimizer.zero_grad(set_to_none=True)
            prediction = generator(input_tensor)
            critic_loss_value = 0.0
            adversarial_loss = torch.zeros((), device=device)
            if critic is not None:
                critic_optimizer.zero_grad(set_to_none=True)
                real_score = critic(torch.cat([input_tensor, target_tensor], dim=1))
                fake_score = critic(torch.cat([input_tensor, prediction.detach()], dim=1))
                gp = gradient_penalty(critic, input_tensor, target_tensor, prediction.detach())
                critic_loss = fake_score.mean() - real_score.mean() + cfg["lambda_gp"] * gp
                critic_loss.backward()
                critic_optimizer.step()
                critic_loss_value = float(critic_loss.detach())
                adversarial_loss = -critic(torch.cat([input_tensor, prediction], dim=1)).mean()
            reconstruction_l1 = criterion(prediction, target_tensor)
            generator_loss = adversarial_loss + cfg["lambda_l1"] * reconstruction_l1
            generator_loss.backward()
            generator_optimizer.step()
            sums["generator"] += float(generator_loss.detach())
            sums["l1"] += float(reconstruction_l1.detach())
            sums["adv"] += float(adversarial_loss.detach())
            sums["critic"] += critic_loss_value

        generator.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for input_tensor, target_tensor, _ in validation_loader:
                prediction = generator(input_tensor.to(device, non_blocking=True))
                validation_sum += float(criterion(prediction, target_tensor.to(device, non_blocking=True)))
        validation_l1 = validation_sum / len(validation_loader)
        epoch_time = time.perf_counter() - epoch_started
        batch_count = len(loader)
        row = [
            epoch,
            sums["generator"] / batch_count,
            sums["l1"] / batch_count,
            sums["adv"] / batch_count,
            sums["critic"] / batch_count,
            validation_l1,
            epoch_time,
        ]
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        if validation_l1 < best_validation_l1:
            best_validation_l1 = validation_l1
            best_epoch = epoch
            atomic_torch_save({
                "generator": generator.state_dict(),
                "epoch": epoch,
                "validation_l1": validation_l1,
                "run_id": run["run_id"],
                "cell_id": run["cell_id"],
                "architecture": run["architecture"],
                "objective": run["objective"],
                "seed": seed,
                "image_size": 256,
                "dataset_id": cfg["dataset_id"],
                "protocol_id": cfg["protocol_id"],
                "initial_generator_sha256": initial_hash,
            }, best_path)
        elapsed = elapsed_before + time.perf_counter() - started
        atomic_torch_save({
            "epoch": epoch,
            "generator": generator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "critic": critic.state_dict() if critic is not None else None,
            "critic_optimizer": critic_optimizer.state_dict() if critic_optimizer is not None else None,
            "best_epoch": best_epoch,
            "best_validation_l1": best_validation_l1,
            "training_wall_time_s": elapsed,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, latest_path)
        print(
            f"[{run['cell_id']} seed={seed} epoch={epoch:02d}] "
            f"val_l1={validation_l1:.8f} best={best_validation_l1:.8f}@{best_epoch} time={epoch_time:.1f}s"
        )
        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            sample_input, sample_target, _ = next(iter(validation_loader))
            with torch.no_grad():
                sample_prediction = generator(sample_input[:1].to(device)).cpu()
            sample_input = ((sample_input[:1] + 1.0) * 0.5).clamp(0.0, 1.0)
            sample_target = sample_target[:1].repeat(1, 3, 1, 1)
            sample_prediction = sample_prediction.repeat(1, 3, 1, 1)
            save_image(
                torch.cat([sample_input, sample_target, sample_prediction], dim=3),
                paths["samples"] / f"sample_epoch_{epoch:03d}.png",
            )

    training_wall_time = elapsed_before + time.perf_counter() - started
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    write_json(paths["checkpoints"] / "checkpoint_manifest.json", {
        "best_checkpoint": "best_model.pt",
        "latest_resume_checkpoint": "latest_resume.pt",
        "selection_metric": "validation_l1",
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_l1": best_checkpoint["validation_l1"],
        "best_checkpoint_sha256": sha256_file(best_path),
    })
    with (paths["logs"] / "training_runtime.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epochs", "train_samples", "validation_samples", "wall_time_s", "wall_time_hours"])
        writer.writerow([cfg["epochs"], len(train_dataset), len(validation_dataset), training_wall_time, training_wall_time / 3600.0])
    run_manifest.update({
        "status": "trained",
        "trained_at_utc": utc_now(),
        "training_wall_time_s": training_wall_time,
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_l1": best_checkpoint["validation_l1"],
        "research_valid_reason": "Canonical training complete; test evaluation pending",
    })
    write_json(run_dir / "run_manifest.json", run_manifest)
    run.update({
        "status": "trained",
        "initial_generator_sha256": initial_hash,
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_l1": best_checkpoint["validation_l1"],
        "training_wall_time_s": training_wall_time,
    })
    return run


def summarize(rows, keys):
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def evaluate_one(run, cfg):
    full_pipeline_started = time.perf_counter()
    run_dir = pathlib.Path(run["run_dir"])
    paths = run_dirs(run_dir)
    checkpoint_path = paths["checkpoints"] / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["cell_id"] != run["cell_id"] or checkpoint["seed"] != run["seed"]:
        raise RuntimeError(f"Checkpoint mismatch for {run['run_id']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = build_generator(run["architecture"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    dataset = FactorialDensityDataset(resolve_path(cfg["dataset_root"]), "test", 256)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers_eval"],
        pin_memory=True,
    )
    result_dirs = {
        "predictions": paths["test"] / "predictions",
        "predictions_float": paths["test"] / "predictions_float",
        "bw": paths["test"] / "bw",
        "colorjet": paths["test"] / "colorjet",
        "inputs": paths["test"] / "inputs",
        "targets": paths["test"] / "targets",
    }
    for path in result_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    rows = []
    walkable_rows = []
    time_generate = 0.0
    metrics_time = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pipeline_started = time.perf_counter()
    with torch.no_grad():
        for input_tensor, target_tensor, names in tqdm(
            loader,
            desc=f"test {run['cell_id']} seed{run['seed']}",
            disable=not sys.stderr.isatty(),
        ):
            name = names[0]
            input_device = input_tensor.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generate_started = time.perf_counter()
            prediction = generator(input_device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            time_generate += time.perf_counter() - generate_started
            prediction_np = prediction[0, 0].cpu().numpy().astype(np.float32)
            target_np = target_tensor[0, 0].numpy().astype(np.float32)
            input_01 = ((input_tensor[0] + 1.0) * 0.5).clamp(0.0, 1.0).numpy()
            metrics_started = time.perf_counter()
            metrics = density_metrics(target_np, prediction_np)
            mask = input_01.sum(axis=0) > 0
            walkable = density_metrics(target_np[mask], prediction_np[mask])
            metrics_time += time.perf_counter() - metrics_started
            rows.append({"file_name": name, **metrics})
            walkable_rows.append({
                "file_name": name,
                "MAE": walkable["mae"],
                "MSE": walkable["mse"],
                "RMSE": walkable["rmse"],
                "PSNR": walkable["psnr"],
                "SSIM": walkable["ssim"],
                "LPIPS": float("nan"),
            })
            prediction_uint8 = (np.clip(prediction_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            target_uint8 = (np.clip(target_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            input_uint8 = (np.clip(input_01.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(prediction_uint8).save(result_dirs["predictions"] / name)
            Image.fromarray(prediction_uint8).save(result_dirs["bw"] / name)
            Image.fromarray(jet_image(prediction_np)).save(result_dirs["colorjet"] / name)
            Image.fromarray(target_uint8).save(result_dirs["targets"] / name)
            Image.fromarray(input_uint8).save(result_dirs["inputs"] / name)
            np.save(result_dirs["predictions_float"] / f"{pathlib.Path(name).stem}.npy", prediction_np)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pipeline_time = time.perf_counter() - pipeline_started

    lpips_metrics_started = time.perf_counter()
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        lpips_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=cfg["num_workers_eval"])
        row_index = {row["file_name"]: row for row in walkable_rows}
        with torch.no_grad():
            for input_tensor, target_tensor, names in lpips_loader:
                predictions = torch.stack([
                    torch.from_numpy(np.load(result_dirs["predictions_float"] / f"{pathlib.Path(name).stem}.npy"))
                    for name in names
                ]).unsqueeze(1)
                masks = (((input_tensor + 1.0) * 0.5).sum(dim=1, keepdim=True) > 0).float()
                pred_rgb = (predictions * masks).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
                target_rgb = (target_tensor * masks).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
                values = lpips_model(pred_rgb, target_rgb).reshape(-1).cpu().numpy()
                for name, value in zip(names, values):
                    row_index[name]["LPIPS"] = float(value)
    except Exception as error:
        print(f"Warning: LPIPS unavailable for {run['run_id']}: {error}")
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics_time += time.perf_counter() - lpips_metrics_started

    summary_metrics_started = time.perf_counter()
    main_keys = ["mae", "mse", "rmse", "psnr", "ssim"]
    walk_keys = ["MAE", "MSE", "RMSE", "PSNR", "SSIM", "LPIPS"]
    main_summary = summarize(rows, main_keys)
    walkable_summary = summarize(walkable_rows, walk_keys)
    with (run_dir / "test_evaluation_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", *main_keys])
        writer.writeheader(); writer.writerows(rows)
    with (run_dir / "test_evaluation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["metric", "value"])
        for key in main_keys:
            writer.writerow([key.upper(), main_summary[key]])
    with (run_dir / "test_evaluation_walkable_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", *walk_keys])
        writer.writeheader(); writer.writerows(walkable_rows)
    with (run_dir / "test_evaluation_walkable_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["metric", "value"])
        for key in walk_keys:
            writer.writerow([key, walkable_summary[key]])
    metrics_time += time.perf_counter() - summary_metrics_started
    full_pipeline_time = time.perf_counter() - full_pipeline_started
    timing_protocol_id = cfg.get("timing_protocol_id", "image_test_runtime_v1")
    runtime = {
        "method_id": run["method_dir"],
        "cell_id": run["cell_id"],
        "seed": run["seed"],
        "split": "test",
        "sample_count": len(rows),
        "Time Generate": f"{time_generate:.6f}",
        "Average Time Generate Per Image": f"{time_generate / len(rows):.9f}",
        "timing_protocol_id": timing_protocol_id,
        "test_pipeline_scope": "checkpoint_load_through_metric_summary_write",
        "test_pipeline_wall_time_s": f"{full_pipeline_time:.6f}",
        "prediction_output_loop_wall_time_s": f"{pipeline_time:.6f}",
        "metrics_wall_time_s": f"{metrics_time:.6f}",
        "runtime_excluding_metrics_s": f"{full_pipeline_time - metrics_time:.6f}",
        "mean_runtime_excluding_metrics_per_image_s": f"{(full_pipeline_time - metrics_time) / len(rows):.9f}",
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint_path": "checkpoints/best_model.pt",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "measured_at_utc": utc_now(),
    }
    floorplans = {name.split("__", 1)[0] for name in dataset.filenames}
    valid = (
        len(rows) == EXPECTED_CASES["test"]
        and len(floorplans) == EXPECTED_PLANS["test"]
        and checkpoint["dataset_id"] == cfg["dataset_id"]
        and checkpoint["image_size"] == 256
    )
    evaluation_id = (
        f"eval_{cfg['dataset_id']}_test_{cfg['protocol_id']}_"
        f"seed{run['seed']:03d}_{timing_protocol_id}"
    )
    evaluation_dir = paths["evaluations"] / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "test_runtime.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runtime.keys()); writer.writeheader(); writer.writerow(runtime)
    with (evaluation_dir / "test_runtime.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runtime.keys()); writer.writeheader(); writer.writerow(runtime)
    write_json(evaluation_dir / "evaluation_manifest.json", {
        "evaluation_id": evaluation_id,
        "cell_id": run["cell_id"],
        "seed": run["seed"],
        "dataset_id": cfg["dataset_id"],
        "split": "test",
        "protocol_id": cfg["protocol_id"],
        "timing_protocol_id": timing_protocol_id,
        "case_count": len(rows),
        "floorplan_count": len(floorplans),
        "image_size": 256,
        "checkpoint_sha256": runtime["checkpoint_sha256"],
        "research_valid": valid,
        "metrics_from_float_predictions": True,
        "png_outputs_are_visualizations": True,
    })
    manifest = read_json(run_dir / "run_manifest.json")
    manifest.update({
        "status": "evaluated",
        "evaluated_at_utc": utc_now(),
        "evaluation_id": evaluation_id,
        "research_valid": valid,
        "research_valid_reason": "Canonical factorial evaluation complete" if valid else "Evaluation gate failed",
    })
    write_json(run_dir / "run_manifest.json", manifest)
    if not valid:
        raise RuntimeError(f"Research-validity gate failed for {run['run_id']}")

    already_locked = run_dir.name.endswith(FINAL_SUFFIX)
    final_dir = run_dir if already_locked else run_dir.with_name(run_dir.name + FINAL_SUFFIX)
    if not already_locked:
        if final_dir.exists():
            raise FileExistsError(f"Final factorial run already exists: {final_dir}")
        run_dir.rename(final_dir)
    manifest["locked_directory_name"] = final_dir.name
    manifest["status"] = "locked_evaluated"
    write_json(final_dir / "run_manifest.json", manifest)
    run.update({
        "run_dir": str(final_dir),
        "status": "locked_evaluated",
        "evaluation_id": evaluation_id,
        "checkpoint_sha256": runtime["checkpoint_sha256"],
        "test_summary": main_summary,
        "walkable_summary": walkable_summary,
        "time_generate_s": time_generate,
        "time_generate_per_image_s": time_generate / len(rows),
    })
    return run


def verify_initial_pairs(state):
    problems = []
    for seed in sorted({run["seed"] for run in state["runs"]}):
        for architecture in ("unet", "resnet9"):
            hashes = {
                run.get("initial_generator_sha256")
                for run in state["runs"]
                if run["seed"] == seed and run["architecture"] == architecture
            }
            if len(hashes) != 1 or None in hashes:
                problems.append({"seed": seed, "architecture": architecture, "hashes": list(hashes)})
    if problems:
        raise RuntimeError(f"Initial generator equality check failed: {problems}")


def analyze_factorial(experiment_dir: pathlib.Path, state):
    result_rows = []
    for run in state["runs"]:
        if run["status"] != "locked_evaluated":
            raise RuntimeError(f"Run not evaluated: {run['run_id']}")
        row = {
            "cell_id": run["cell_id"],
            "method": run["method"],
            "architecture": run["architecture"],
            "objective": run["objective"],
            "seed": run["seed"],
            "run_dir": run["run_dir"],
            "MAE": run["test_summary"]["mae"],
            "MSE": run["test_summary"]["mse"],
            "RMSE": run["test_summary"]["rmse"],
            "PSNR": run["test_summary"]["psnr"],
            "SSIM": run["test_summary"]["ssim"],
            "walkable_MAE": run["walkable_summary"]["MAE"],
            "walkable_SSIM": run["walkable_summary"]["SSIM"],
            "LPIPS": run["walkable_summary"]["LPIPS"],
            "Time_Generate_s": run["time_generate_s"],
            "Time_Generate_per_image_s": run["time_generate_per_image_s"],
        }
        result_rows.append(row)
    results_path = experiment_dir / "factorial_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys()); writer.writeheader(); writer.writerows(result_rows)

    response_names = ["MAE", "MSE", "RMSE", "PSNR", "SSIM", "walkable_MAE", "walkable_SSIM", "LPIPS", "Time_Generate_per_image_s"]
    per_seed_effects = []
    for seed in sorted({row["seed"] for row in result_rows}):
        seed_rows = [row for row in result_rows if row["seed"] == seed]
        for response in response_names:
            values = {}
            for row in seed_rows:
                a = 1 if row["architecture"] == "resnet9" else -1
                b = 1 if row["objective"] == "wgangp_l1" else -1
                values[(a, b)] = float(row[response])
            effect_a = np.mean([values[(1, -1)], values[(1, 1)]]) - np.mean([values[(-1, -1)], values[(-1, 1)]])
            effect_b = np.mean([values[(-1, 1)], values[(1, 1)]]) - np.mean([values[(-1, -1)], values[(1, -1)]])
            effect_ab = np.mean([values[(-1, -1)], values[(1, 1)]]) - np.mean([values[(-1, 1)], values[(1, -1)]])
            for effect, value in (("architecture", effect_a), ("objective", effect_b), ("interaction", effect_ab)):
                per_seed_effects.append({"seed": seed, "response": response, "effect": effect, "value": float(value)})
    with (experiment_dir / "factorial_effects_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_seed_effects[0].keys()); writer.writeheader(); writer.writerows(per_seed_effects)
    aggregate = []
    for response in response_names:
        for effect in ("architecture", "objective", "interaction"):
            values = np.array([row["value"] for row in per_seed_effects if row["response"] == response and row["effect"] == effect], dtype=float)
            mean = float(values.mean())
            if len(values) >= 2:
                sd = float(values.std(ddof=1))
                se = sd / math.sqrt(len(values))
                # 95% two-sided Student-t critical values for the supported
                # pilot/final seed counts (df=1 and df=2 respectively).
                t_critical = {2: 12.7062047364, 3: 4.3026527299}.get(len(values))
                half = t_critical * se if t_critical is not None else float("nan")
                ci95_low = mean - half
                ci95_high = mean + half
            else:
                # A single seed yields a valid factorial contrast, but it cannot
                # estimate between-seed variance or a confidence interval.
                sd = se = ci95_low = ci95_high = float("nan")
            aggregate.append({
                "response": response,
                "effect": effect,
                "n_seeds": len(values),
                "mean_effect": mean,
                "sd": sd,
                "se": se,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
            })
    with (experiment_dir / "factorial_effects_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate[0].keys()); writer.writeheader(); writer.writerows(aggregate)
    lock = {
        "lock_id": f"{state['experiment_id']}_{len(set(run['seed'] for run in state['runs']))}_seed_lock",
        "status": "locked",
        "locked_at_utc": utc_now(),
        "dataset_id": state["dataset_id"],
        "seeds": sorted({run["seed"] for run in state["runs"]}),
        "run_count": len(state["runs"]),
        "runs": [
            {
                "cell_id": run["cell_id"],
                "seed": run["seed"],
                "run_dir": run["run_dir"],
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
            for run in state["runs"]
        ],
    }
    write_json(experiment_dir / "factorial_lock.json", lock)
    state.update({"status": "complete", "completed_at_utc": utc_now(), "factorial_lock": str(experiment_dir / "factorial_lock.json")})
    save_state(experiment_dir, state)


def smoke_test(cfg):
    dataset = FactorialDensityDataset(resolve_path(cfg["dataset_root"]), "train", 256)
    input_tensor, target_tensor, _ = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
    hashes = {}
    for architecture in ("unet", "resnet9"):
        for objective in ("l1", "wgangp_l1"):
            configure_seed(42)
            generator = build_generator(architecture).to(device)
            initial_hash = state_dict_sha256(generator)
            hashes.setdefault(architecture, set()).add(initial_hash)
            prediction = generator(input_tensor)
            loss = cfg["lambda_l1"] * nn.functional.l1_loss(prediction, target_tensor)
            if objective == "wgangp_l1":
                critic = SharedPatchCritic().to(device)
                real = critic(torch.cat([input_tensor, target_tensor], 1))
                fake = critic(torch.cat([input_tensor, prediction.detach()], 1))
                d_loss = fake.mean() - real.mean() + cfg["lambda_gp"] * gradient_penalty(critic, input_tensor, target_tensor, prediction.detach())
                d_loss.backward()
                critic.zero_grad(set_to_none=True)
                loss = loss - critic(torch.cat([input_tensor, prediction], 1)).mean()
            loss.backward()
            print(f"smoke {architecture}/{objective}: output={tuple(prediction.shape)} loss={float(loss.detach()):.6f} initial={initial_hash[:12]}")
            del generator
            if objective == "wgangp_l1":
                del critic
    if any(len(values) != 1 for values in hashes.values()):
        raise RuntimeError(f"Generator initialization mismatch: {hashes}")
    print("Smoke checks passed; paired generator initial hashes are identical.")


def execute(args):
    cfg, config_path = load_config(args.config)
    if args.stage == "plan" or args.dry_run:
        print_plan(cfg)
        return
    if args.stage == "smoke":
        print_plan(cfg)
        smoke_test(cfg)
        return
    if args.experiment_dir:
        experiment_dir, state = load_experiment(args.experiment_dir)
    else:
        experiment_dir, state = create_experiment(cfg, config_path)
        print(f"Created experiment: {experiment_dir}")
    if args.stage == "retest":
        verify_initial_pairs(state)
        state["status"] = "retesting"
        save_state(experiment_dir, state)
        for run in state["runs"]:
            if run["status"] != "locked_evaluated":
                raise RuntimeError(f"Retest requires a locked evaluated run: {run['run_id']}")
            print(f"RETEST {run['cell_id']} seed={run['seed']}")
            evaluate_one(run, cfg)
            save_state(experiment_dir, state)
        analyze_factorial(experiment_dir, state)
        print(f"Factorial retest complete: {experiment_dir}")
        return
    if args.stage in {"train", "all"}:
        state["status"] = "training"
        save_state(experiment_dir, state)
        for run in state["runs"]:
            if run["status"] in {"trained", "locked_evaluated"}:
                continue
            print(f"TRAIN {run['cell_id']} seed={run['seed']} -> {run['run_dir']}")
            try:
                train_one(run, cfg, experiment_dir)
                save_state(experiment_dir, state)
            except Exception as error:
                run["status"] = "failed_training"
                run["error"] = repr(error)
                save_state(experiment_dir, state)
                raise
        verify_initial_pairs(state)
        state["status"] = "trained"
        save_state(experiment_dir, state)
    if args.stage in {"evaluate", "all"}:
        verify_initial_pairs(state)
        state["status"] = "evaluating"
        save_state(experiment_dir, state)
        for run in state["runs"]:
            if run["status"] == "locked_evaluated":
                continue
            if run["status"] != "trained":
                raise RuntimeError(f"Cannot evaluate run with status {run['status']}: {run['run_id']}")
            print(f"EVALUATE {run['cell_id']} seed={run['seed']}")
            try:
                evaluate_one(run, cfg)
                save_state(experiment_dir, state)
            except Exception as error:
                run["status"] = "failed_evaluation"
                run["error"] = repr(error)
                save_state(experiment_dir, state)
                raise
        state["status"] = "evaluated"
        save_state(experiment_dir, state)
    if args.stage in {"analyze", "all"}:
        verify_initial_pairs(state)
        analyze_factorial(experiment_dir, state)
        print(f"Factorial experiment complete: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description="Controlled 2x2 image-model factorial pipeline")
    parser.add_argument("--stage", choices=["plan", "smoke", "train", "evaluate", "retest", "analyze", "all"], default=None)
    parser.add_argument("--config", default="config_factorial.json")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage is None:
        if sys.stdin.isatty():
            print(f"1) Plan/check configuration (default)\n2) Smoke test\n3) Train + evaluate + analyze all {len(cfg['seeds']) * 4} runs")
            choice = input("Select operation [1]: ").strip() or "1"
            args.stage = {"1": "plan", "2": "smoke", "3": "all"}.get(choice, "plan")
        else:
            args.stage = "plan"
    execute(args)


if __name__ == "__main__":
    main()

import os
import json
import pathlib
import csv
import argparse
import time
import hashlib
from datetime import datetime, timezone
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pix2pix_wgangp_common import (
    Pix2PixDataset,
    UNetGenerator,
    convert_bw_to_colorjet,
    get_device,
    resolve_path,
    resolve_project_root,
    tensor_density_metrics
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def finalize_evaluation_provenance(run_dir, checkpoint_path, checkpoint, test_dataset, image_size):
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        run_manifest = json.load(f)

    predictions = list((run_dir / "test_results" / "predictions").glob("*.png"))
    floorplan_count = len({name.split("__", 1)[0] for name in test_dataset.filenames})
    checkpoint_cfg = checkpoint.get("config", {})
    dataset_id = checkpoint.get("dataset_id", checkpoint_cfg.get("dataset_id", "unknown"))
    protocol_id = checkpoint_cfg.get("protocol_id", f"image_density_{image_size}_v1")
    inventory = run_manifest.get("dataset_inventory", {})
    overlap = run_manifest.get("plan_overlap", {})
    checks = {
        "dataset_id_matches": dataset_id == "housegan_canonical_imagebase_split_v1",
        "train_case_count_matches": inventory.get("train", {}).get("paired_count") == 2603,
        "validation_case_count_matches": inventory.get("validation", {}).get("paired_count") == 439,
        "test_case_count_matches": len(test_dataset) == 862,
        "test_prediction_count_matches": len(predictions) == 862,
        "test_floorplan_count_matches": floorplan_count == 117,
        "no_plan_overlap": all(value == 0 for value in overlap.values()) and len(overlap) == 3,
        "checkpoint_run_matches": checkpoint.get("run_id") == run_manifest.get("run_id"),
        "checkpoint_resolution_matches": int(checkpoint_cfg.get("image_size", -1)) == image_size == 256,
        "summary_exists": (run_dir / "test_evaluation_summary.csv").exists(),
        "runtime_exists": (run_dir / "test_runtime.csv").exists(),
    }
    research_valid = all(checks.values())
    evaluation_id = f"eval_{dataset_id}_test_{protocol_id}"
    evaluation_dir = run_dir / "evaluations" / evaluation_id
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluated_at = datetime.now(timezone.utc).isoformat()
    write_json(evaluation_dir / "checkpoint_ref.json", {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "validation_l1": checkpoint.get("val_loss"),
        "run_id": checkpoint.get("run_id"),
        "seed": checkpoint.get("seed"),
    })
    write_json(evaluation_dir / "dataset_ref.json", {
        "dataset_id": dataset_id,
        "split": "test",
        "case_count": len(test_dataset),
        "floorplan_count": floorplan_count,
        "dataset_manifest_sha256": run_manifest.get("dataset_manifest_sha256"),
    })
    write_json(evaluation_dir / "evaluation_config.json", {
        "protocol_id": protocol_id,
        "image_size": image_size,
        "batch_size": 1,
        "checkpoint_selection": "minimum validation L1",
        "metrics": ["MAE", "MSE", "RMSE", "PSNR", "SSIM", "walkable metrics", "LPIPS"],
    })
    project_root = resolve_project_root(pathlib.Path(__file__).parent.resolve())
    evaluation_code_files = [
        pathlib.Path(__file__).resolve(),
        pathlib.Path(__file__).with_name("pix2pix_wgangp_common.py").resolve(),
        (project_root / "Tool_utility" / "compute_walkable_metrics.py").resolve(),
    ]
    write_json(evaluation_dir / "evaluation_code_provenance.json", {
        "files": {
            str(path.relative_to(project_root)): sha256_file(path)
            for path in evaluation_code_files
        }
    })
    write_json(evaluation_dir / "evaluation_manifest.json", {
        "evaluation_id": evaluation_id,
        "status": "evaluated",
        "evaluated_at_utc": evaluated_at,
        "method_id": "Method_pix2pix_WGAN-GP",
        "split": "test",
        "case_count": len(test_dataset),
        "floorplan_count": floorplan_count,
        "prediction_count": len(predictions),
        "research_valid": research_valid,
        "validity_checks": checks,
        "summary_metrics_path": str((run_dir / "test_evaluation_summary.csv").resolve()),
        "walkable_metrics_path": str((run_dir / "test_evaluation_walkable_summary.csv").resolve()),
        "runtime_path": str((run_dir / "test_runtime.csv").resolve()),
    })
    run_manifest.update({
        "status": "evaluated",
        "evaluated_at_utc": evaluated_at,
        "evaluation_id": evaluation_id,
        "research_valid": research_valid,
        "research_valid_reason": (
            "Canonical 256 test evaluation completed"
            if research_valid else "One or more canonical evaluation checks failed"
        ),
    })
    write_json(manifest_path, run_manifest)
    return evaluation_dir / "evaluation_manifest.json"


def main():
    parser = argparse.ArgumentParser(description="Test Method_pix2pix_WGAN-GP")
    parser.add_argument("--config", type=str, default="config_test.json", help="Path to config file")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run output directory (optional)")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    config_path = script_dir / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    project_root = resolve_project_root(script_dir)
    dataset_root = resolve_path(cfg["dataset_root"], project_root)

    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
    else:
        outputs_dir = project_root / "AI_GenerateImage" / "AI_Result" / "Method_pix2pix_WGAN-GP" / "outputs"
        runs = [path for path in outputs_dir.glob("run_*") if path.is_dir()]
        if not runs:
            raise FileNotFoundError(f"No run directories found in {outputs_dir}")
        locked_runs = [
            path for path in runs if path.name.endswith("_model_evaluate_256")
        ]
        candidates = locked_runs or runs
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)

    checkpoint_path = run_dir / "checkpoints" / "best_loss.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoints" / "final.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in {run_dir / 'checkpoints'}")

    device = get_device()
    print(f"🧪 Testing Method_pix2pix_WGAN-GP model from: {checkpoint_path}")

    # Load Model
    netG = UNetGenerator(input_nc=3, output_nc=1, num_downs=8, ngf=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint["netG_state_dict"])
    netG.eval()

    # Load Test Dataset
    checkpoint_cfg = checkpoint.get("config", {})
    image_size = int(checkpoint_cfg.get("image_size", cfg.get("image_size", 512)))
    print(f"🖼️ Evaluation resolution from checkpoint: {image_size}x{image_size}")
    test_dataset = Pix2PixDataset(dataset_root, split="test", image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Subdirectories for results
    bw_dir = run_dir / "test_results" / "bw"
    colorjet_dir = run_dir / "test_results" / "colorjet"
    pred_dir = run_dir / "test_results" / "predictions"
    
    bw_dir.mkdir(parents=True, exist_ok=True)
    colorjet_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    metrics_wall_time_s = 0.0
    time_generate_s = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started = time.perf_counter()
    with torch.no_grad():
        for input_a, target_b, filenames in tqdm(test_loader, desc="Testing"):
            fname = filenames[0]
            input_a = input_a.to(device)
            target_b = target_b.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generate_started = time.perf_counter()
            pred_b = netG(input_a)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            time_generate_s += time.perf_counter() - generate_started

            # Conversion to Numpy
            pred_np = pred_b[0, 0].cpu().numpy()
            target_np = target_b[0, 0].cpu().numpy()

            # Save BW
            bw_img_np = (np.clip(pred_np, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(bw_img_np).save(bw_dir / fname)
            Image.fromarray(bw_img_np).save(pred_dir / fname)

            # Save COLORJET
            colorjet_np = convert_bw_to_colorjet(pred_np)
            Image.fromarray(colorjet_np).save(colorjet_dir / fname)

            # Calculate Metrics
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics_started = time.perf_counter()
            m = tensor_density_metrics(target_np, pred_np)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics_wall_time_s += time.perf_counter() - metrics_started
            m["file_name"] = fname
            metrics_rows.append(m)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_wall_time_s = time.perf_counter() - test_started
    runtime_excluding_metrics_s = max(0.0, test_wall_time_s - metrics_wall_time_s)

    runtime_row = {
        "method_id": "Method_pix2pix_WGAN-GP",
        "split": "test",
        "timing_scope": "test_loop_including_data_inference_metrics_postprocess_and_image_write",
        "sample_count": len(metrics_rows),
        "Time Generate": f"{time_generate_s:.6f}",
        "Average Time Generate Per Image": (
            f"{time_generate_s / len(metrics_rows):.9f}" if metrics_rows else "nan"
        ),
        "test_pipeline_wall_time_s": f"{test_wall_time_s:.6f}",
        "metrics_wall_time_s": f"{metrics_wall_time_s:.6f}",
        "runtime_excluding_metrics_s": f"{runtime_excluding_metrics_s:.6f}",
        "mean_runtime_excluding_metrics_per_image_s": (
            f"{runtime_excluding_metrics_s / len(metrics_rows):.9f}" if metrics_rows else "nan"
        ),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "test_runtime.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runtime_row.keys())
        writer.writeheader()
        writer.writerow(runtime_row)

    # Save per-image metrics
    csv_headers = ["file_name", "mae", "mse", "rmse", "psnr", "ssim"]
    per_image_csv = run_dir / "test_evaluation_per_image.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(metrics_rows)

    # Compute Averages
    avg_mae = float(np.mean([r["mae"] for r in metrics_rows]))
    avg_mse = float(np.mean([r["mse"] for r in metrics_rows]))
    avg_rmse = float(np.mean([r["rmse"] for r in metrics_rows]))
    avg_psnr = float(np.mean([r["psnr"] for r in metrics_rows]))
    avg_ssim = float(np.mean([r["ssim"] for r in metrics_rows]))

    summary_csv = run_dir / "test_evaluation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["MAE", avg_mae])
        writer.writerow(["MSE", avg_mse])
        writer.writerow(["RMSE", avg_rmse])
        writer.writerow(["PSNR", avg_psnr])
        writer.writerow(["SSIM", avg_ssim])

    evaluation_manifest_path = finalize_evaluation_provenance(
        run_dir, checkpoint_path, checkpoint, test_dataset, image_size
    )

    print(f"✅ Method_pix2pix_WGAN-GP Test Evaluation Finished!")
    print(f"📊 Average MAE (L1): {avg_mae:.6f}")
    print(f"📊 Average MSE:      {avg_mse:.6f}")
    print(f"📊 Average SSIM:     {avg_ssim:.6f}")
    print(f"⏱️ Full test pipeline: {test_wall_time_s:.3f} s ({len(metrics_rows)} samples)")
    print(f"⏱️ Time Generate:      {time_generate_s:.3f} s")
    print(f"⏱️ Excluding metrics:  {runtime_excluding_metrics_s:.3f} s")
    print(f"🧾 Evaluation manifest: {evaluation_manifest_path}")

    # Trigger compute_walkable_metrics.py
    try:
        sys_path = project_root / "Tool_utility" / "compute_walkable_metrics.py"
        if sys_path.exists():
            import sys
            sys.path.append(str(project_root / "Tool_utility"))
            import compute_walkable_metrics
            compute_walkable_metrics.compute_walkable_for_runs([run_dir])
    except Exception as e:
        print(f"Warning: Walkable metrics auto-trigger failed: {e}")

if __name__ == "__main__":
    main()

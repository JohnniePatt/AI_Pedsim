import os
import json
import pathlib
import csv
import argparse
import time
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
        runs = sorted(list(outputs_dir.glob("run_pix2pix_wgangp_*")))
        if not runs:
            raise FileNotFoundError(f"No run directories found in {outputs_dir}")
        run_dir = runs[-1]

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
    image_size = cfg.get("image_size", 512)
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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started = time.perf_counter()
    with torch.no_grad():
        for input_a, target_b, filenames in tqdm(test_loader, desc="Testing"):
            fname = filenames[0]
            input_a = input_a.to(device)
            target_b = target_b.to(device)

            pred_b = netG(input_a)

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
            m = tensor_density_metrics(target_np, pred_np)
            m["file_name"] = fname
            metrics_rows.append(m)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_wall_time_s = time.perf_counter() - test_started

    runtime_row = {
        "method_id": "Method_pix2pix_WGAN-GP",
        "split": "test",
        "timing_scope": "test_loop_including_data_inference_metrics_postprocess_and_image_write",
        "sample_count": len(metrics_rows),
        "test_wall_time_s": f"{test_wall_time_s:.6f}",
        "mean_wall_time_per_sample_s": f"{test_wall_time_s / len(metrics_rows):.9f}" if metrics_rows else "nan",
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

    print(f"✅ Method_pix2pix_WGAN-GP Test Evaluation Finished!")
    print(f"📊 Average MAE (L1): {avg_mae:.6f}")
    print(f"📊 Average MSE:      {avg_mse:.6f}")
    print(f"📊 Average SSIM:     {avg_ssim:.6f}")
    print(f"⏱️ Test wall time:    {test_wall_time_s:.3f} s ({len(metrics_rows)} samples)")

    # Trigger compute_walkable_metrics.py
    try:
        sys_path = project_root / "Tool_utility" / "compute_walkable_metrics.py"
        if sys_path.exists():
            import sys
            sys.path.append(str(project_root / "Tool_utility"))
            import compute_walkable_metrics
            compute_walkable_metrics.compute_walkable_for_runs()
    except Exception as e:
        print(f"Warning: Walkable metrics auto-trigger failed: {e}")

if __name__ == "__main__":
    main()

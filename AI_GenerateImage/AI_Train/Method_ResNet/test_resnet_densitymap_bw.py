import os
import json
import pathlib
import csv
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet_common import (
    ResNetDataset,
    ResNetGenerator,
    convert_bw_to_colorjet,
    get_device,
    resolve_path,
    resolve_project_root,
    tensor_density_metrics
)

def main():
    parser = argparse.ArgumentParser(description="Test Method_ResNet (9-Block ResNet DensityMap BW)")
    parser.add_argument("--config", type=str, default="config_test.json", help="Path to config file")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to run output directory (optional)")
    parser.add_argument("--save_colorjet", action="store_true", default=True, help="Save COLORJET format alongside BW")
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
        outputs_dir = project_root / "AI_Result" / "Method_ResNet" / "outputs"
        runs = sorted(list(outputs_dir.glob("run_ResNet_*")))
        if not runs:
            raise FileNotFoundError(f"No run directories found in {outputs_dir}")
        run_dir = runs[-1]

    checkpoint_path = run_dir / "checkpoints" / "best_loss.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_result_dir = run_dir / "test_results"
    test_result_bw = test_result_dir / "bw"
    test_result_predictions = test_result_dir / "predictions"
    test_result_colorjet = test_result_dir / "colorjet"

    for d in (test_result_dir, test_result_bw, test_result_predictions, test_result_colorjet):
        d.mkdir(parents=True, exist_ok=True)

    device = get_device()
    image_size = cfg.get("image_size", 256)
    target_channels = cfg.get("target_channels", 1)
    num_resnet_blocks = cfg.get("num_resnet_blocks", 9)

    model = ResNetGenerator(in_ch=3, out_ch=target_channels, num_resnet_blocks=num_resnet_blocks).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = ResNetDataset(dataset_root, "test", image_size, target_channels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()
    except Exception:
        lpips_model = None

    rows = []
    print(f"\n--- Evaluating Method_ResNet Test Set ({len(test_dataset)} samples) ---")

    with torch.no_grad():
        for real_a, real_b, filenames in tqdm(test_loader, desc="Testing"):
            real_a, real_b = real_a.to(device), real_b.to(device)
            pred_b = model(real_a)

            fname = filenames[0]
            b_arr = real_b.cpu().numpy()[0, 0]
            pred_arr = pred_b.cpu().numpy()[0, 0]

            metrics = tensor_density_metrics(b_arr, pred_arr)
            lpips_val = float("nan")
            if lpips_model is not None:
                try:
                    pred_t = (pred_b * 2.0 - 1.0).repeat(1, 3, 1, 1)
                    true_t = (real_b * 2.0 - 1.0).repeat(1, 3, 1, 1)
                    lpips_val = float(lpips_model(pred_t, true_t).mean().item())
                except Exception:
                    pass

            row = {
                "file_name": fname,
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "SSIM": metrics["ssim"],
                "PSNR": metrics["psnr"],
                "LPIPS": lpips_val
            }
            rows.append(row)

            # 1. Save BW image
            p_arr = (pred_arr * 255).clip(0, 255).astype(np.uint8)
            p_img = Image.fromarray(p_arr, mode="L")
            p_img.save(test_result_dir / fname)
            p_img.save(test_result_bw / fname)
            p_img.save(test_result_predictions / fname)

            # 2. Save COLORJET image
            if args.save_colorjet:
                colorjet_arr = convert_bw_to_colorjet(pred_b[0, 0])
                c_img = Image.fromarray(colorjet_arr, mode="RGB")
                c_img.save(test_result_colorjet / fname)

    # Save CSV evaluation files
    summary_dict = {}
    if rows:
        for k in ["MAE", "MSE", "RMSE", "SSIM", "PSNR", "LPIPS"]:
            vals = [r[k] for r in rows if not np.isnan(r[k])]
            summary_dict[k] = float(np.mean(vals)) if vals else float("nan")

        csv_locations = [
            run_dir / "test_evaluation_per_image.csv",
            run_dir / "logs" / "test_evaluation.csv",
            run_dir / "logs" / "test_evaluation_per_image.csv"
        ]
        for loc in csv_locations:
            loc.parent.mkdir(parents=True, exist_ok=True)
            with open(loc, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file_name", "MAE", "MSE", "RMSE", "SSIM", "PSNR", "LPIPS"])
                writer.writeheader()
                writer.writerows(rows)

        summary_locations = [
            run_dir / "test_evaluation_summary.csv",
            run_dir / "logs" / "test_evaluation_summary.csv"
        ]
        for loc in summary_locations:
            with open(loc, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in summary_dict.items():
                    writer.writerow([k, f"{v:.6f}" if not np.isnan(v) else "nan"])

    # Auto-generate walkable area CSVs for instant Streamlit UI rendering
    try:
        from Tool_utility.compute_walkable_metrics import compute_walkable_for_runs
        compute_walkable_for_runs()
    except Exception:
        pass

    print(f"\n✅ Method_ResNet Test Evaluation Finished!")
    print(f"📊 Average MAE (L1): {summary_dict.get('MAE', 0.0):.6f}")
    print(f"📊 Average MSE:      {summary_dict.get('MSE', 0.0):.6f}")
    print(f"📊 Average SSIM:     {summary_dict.get('SSIM', 0.0):.6f}")
    print(f"📁 BW Predictions saved to:       {test_result_bw}")
    if args.save_colorjet:
        print(f"🎨 COLORJET Predictions saved to: {test_result_colorjet}")

if __name__ == "__main__":
    import argparse
    main()

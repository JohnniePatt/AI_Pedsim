import argparse
import pathlib
import torch
import csv
import math
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from plainunet_common import PlainUNetDataset, PlainUNet, get_device, tensor_density_metrics, resolve_path, resolve_project_root

def _to_colorjet(gray_uint8):
    gray_norm = gray_uint8.astype(np.float32) / 255.0
    x = gray_norm
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_loss.pt")
    args = parser.parse_args()

    run_path = pathlib.Path(args.run_path)
    if not run_path.is_absolute(): run_path = pathlib.Path.cwd() / run_path
    
    checkpoint_path = run_path / "checkpoints" / args.checkpoint
    if not checkpoint_path.exists(): raise FileNotFoundError(f"Missing {checkpoint_path}")

    device = get_device()
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state["config"]

    script_dir = pathlib.Path(__file__).parent.resolve()
    dataset_root = resolve_path(cfg["dataset_root"], resolve_project_root(script_dir))

    model = PlainUNet(base=cfg.get("base_filters", 32), drop=cfg.get("dropout", 0.1)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    test_loader = DataLoader(PlainUNetDataset(dataset_root, "test", cfg["image_size"]), batch_size=1, shuffle=False)

    result_dir = run_path / "test_results" / args.checkpoint.replace(".pt", "")
    pred_dir = result_dir / "predictions"
    input_dir = result_dir / "inputs"
    target_dir = result_dir / "targets"
    for d in [pred_dir, input_dir, target_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for i, (a, b, name) in enumerate(tqdm(test_loader, desc="Testing")):
            a = a.to(device)
            pred = torch.sigmoid(model(a))
            pred_arr = pred.cpu().numpy()[0, 0]
            b_arr = b.numpy()[0, 0]
            
            metrics = tensor_density_metrics(b_arr, pred_arr)
            rows.append({
                "file_name": name[0],
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "SSIM": metrics["ssim"],
                "PSNR": metrics["psnr"],
                "LPIPS": float("nan")
            })

            if i < 50:
                # Save input
                a_img = ((a.cpu().numpy()[0].transpose(1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(a_img, mode="RGB").save(input_dir / name[0])
                
                # Save target
                b_img = (b_arr * 255).clip(0, 255).astype(np.uint8)
                _to_colorjet(b_img).save(target_dir / name[0])
                Image.fromarray(b_img, mode="L").convert("RGB").save(target_dir / f"MASK_{name[0]}")
                
                # Save prediction
                p_img = (pred_arr * 255).clip(0, 255).astype(np.uint8)
                _to_colorjet(p_img).save(pred_dir / name[0])
                Image.fromarray(p_img, mode="L").convert("RGB").save(pred_dir / f"MASK_{name[0]}")

    if rows:
        summary = {k: np.mean([r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]) for k in rows[0].keys() if k != "file_name"}
        
        # Save to specific checkpoint dir
        with open(result_dir / "test_evaluation_per_image.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "MAE", "MSE", "RMSE", "SSIM", "PSNR", "LPIPS"])
            writer.writeheader()
            writer.writerows(rows)
            
        with open(result_dir / "test_evaluation_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items(): writer.writerow([k, "nan" if math.isnan(v) else f"{v:.6f}"])

        # Also save to run root for Streamlit global dashboard
        with open(run_path / "test_evaluation_per_image.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "MAE", "MSE", "RMSE", "SSIM", "PSNR", "LPIPS"])
            writer.writeheader()
            writer.writerows(rows)
            
        with open(run_path / "test_evaluation_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in summary.items(): writer.writerow([k, "nan" if math.isnan(v) else f"{v:.6f}"])

if __name__ == "__main__":
    main()

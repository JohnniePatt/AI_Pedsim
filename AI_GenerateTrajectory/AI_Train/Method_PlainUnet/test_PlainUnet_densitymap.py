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

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

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
    parser.add_argument("--image_size", type=int, default=None, help="Custom image size for testing (e.g. 512)")
    args = parser.parse_args()

    run_path = pathlib.Path(args.run_path)
    if not run_path.is_absolute(): run_path = pathlib.Path.cwd() / run_path
    
    checkpoint_path = run_path / "checkpoints" / args.checkpoint
    if not checkpoint_path.exists(): raise FileNotFoundError(f"Missing {checkpoint_path}")

    device = get_device()
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state["config"]

    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_model.eval()
            print("[METRIC] LPIPS enabled (alex)")
        except Exception as e:
            print(f"[WARN] LPIPS unavailable: {e}")
            lpips_model = None

    script_dir = pathlib.Path(__file__).parent.resolve()
    dataset_root = resolve_path(cfg["dataset_root"], resolve_project_root(script_dir))

    model = PlainUNet(base=cfg.get("base_filters", 32), drop=cfg.get("dropout", 0.1)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    image_size = args.image_size if args.image_size is not None else 512
    print(f"[TEST] Using evaluation image size: {image_size}x{image_size}")
    test_loader = DataLoader(PlainUNetDataset(dataset_root, "test", image_size), batch_size=1, shuffle=False)

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
            lpips_val = float("nan")
            if lpips_model is not None:
                try:
                    pred_t = (pred * 2.0 - 1.0).repeat(1, 3, 1, 1)
                    true_t = (b.to(device) * 2.0 - 1.0).repeat(1, 3, 1, 1)
                    lpips_val = float(lpips_model(pred_t, true_t).mean().item())
                except Exception as e:
                    print(f"[WARN] LPIPS calculation failed: {e}")

            rows.append({
                "file_name": name[0],
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "SSIM": metrics["ssim"],
                "PSNR": metrics["psnr"],
                "LPIPS": lpips_val
            })

            if True:
                orig_size = Image.open(dataset_root / "A" / test_loader.dataset.split / name[0]).size
                
                # Save input
                a_img_arr = ((a.cpu().numpy()[0].transpose(1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
                a_img = Image.fromarray(a_img_arr, mode="RGB").resize(orig_size, Image.LANCZOS)
                a_img.save(input_dir / name[0])
                
                # Save target
                b_img_arr = (b_arr * 255).clip(0, 255).astype(np.uint8)
                b_img = Image.fromarray(b_img_arr, mode="L").resize(orig_size, Image.LANCZOS)
                _to_colorjet(np.array(b_img)).save(target_dir / name[0])
                b_img.convert("RGB").save(target_dir / f"MASK_{name[0]}")
                
                # Save prediction
                p_img_arr = (pred_arr * 255).clip(0, 255).astype(np.uint8)
                p_img = Image.fromarray(p_img_arr, mode="L").resize(orig_size, Image.LANCZOS)
                _to_colorjet(np.array(p_img)).save(pred_dir / name[0])
                p_img.convert("RGB").save(pred_dir / f"MASK_{name[0]}")

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

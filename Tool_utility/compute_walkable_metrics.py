import sys
import pathlib
import math
import numpy as np
import pandas as pd
from PIL import Image

def compute_walkable_for_runs():
    script_dir = pathlib.Path(__file__).parent.resolve()
    project_root = script_dir.parent
    sys.path.append(str(project_root / "UI_PerformanceCompare" / "Streamlit"))
    from utils.image_paths import image_triplet

    ds_a = project_root / "Dataset" / "Data_ImageUNet" / "DensityMap_dataset" / "Topo_HouseGAN" / "A" / "test"
    ds_b = project_root / "Dataset" / "Data_ImageUNet" / "DensityMap_dataset" / "Topo_HouseGAN" / "B" / "test"

    result_root = project_root / "AI_GenerateImage" / "AI_Result"

    # Discover active runs for each method
    active_runs = [
        result_root / "Method_ResNet" / "outputs" / "run_ResNet_20260808_025204",
        result_root / "Method_pix2pixHD" / "outputs" / "run_HD_20260517_133538_BestForBW",
        result_root / "Method_PlainUnet" / "outputs" / "run_PlainUNet_20260708_211818",
        result_root / "Method_pix2pixhd_No_D" / "outputs" / "run_HD_NoD_20260709_180550",
        result_root / "Method_CVAE" / "outputs" / "run_CVAE_20260627_193237_config2",
        result_root / "Method_pix2pix" / "outputs" / "run_pix2pix_20260808_015920",
    ]
    for method_dir in result_root.glob("Method_*"):
        latest_runs = sorted(list((method_dir / "outputs").glob("run_*")))
        if latest_runs:
            active_runs.append(latest_runs[-1])

    seen = set()
    runs = []
    for r in active_runs:
        if r.exists() and r not in seen and (r / "test_results").exists():
            seen.add(r)
            runs.append(r)

    files = sorted([f.name for f in ds_b.glob("*.png")])

    masks = {}
    for fname in files:
        pa = ds_a / fname
        if pa.exists():
            img_a = np.array(Image.open(pa))
            masks[fname] = (img_a.sum(axis=-1) > 0)

    import torch
    lpips_model = None
    try:
        import lpips
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()
    except Exception:
        lpips_model = None

    for run_path in runs:
        if not run_path.is_dir():
            continue
        
        # Check if predictions or images exist
        test_res_dir = run_path / "test_results"
        if not test_res_dir.exists():
            continue

        print(f"Processing walkable metrics for {run_path.parent.name} / {run_path.name}...")

        rows = []
        for fname in files:
            pb = ds_b / fname
            if not pb.exists() or fname not in masks:
                continue
            mask = masks[fname]
            if not mask.any():
                continue

            _, pred_p, _ = image_triplet(run_path, fname)
            if pred_p is None or not pred_p.exists():
                continue

            img_b = np.array(Image.open(pb).convert("L"), dtype=np.float32) / 255.0
            pred_img = np.array(Image.open(pred_p).convert("L"), dtype=np.float32) / 255.0
            if pred_img.shape != img_b.shape:
                pred_img = np.array(Image.fromarray((pred_img * 255).astype(np.uint8)).resize((img_b.shape[1], img_b.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0

            t_m = img_b[mask]
            p_m = pred_img[mask]

            diff = p_m - t_m
            mae = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff ** 2))
            rmse = float(math.sqrt(mse))
            psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))

            coords = np.argwhere(mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            sub_b = img_b[y0:y1, x0:x1]
            sub_p = pred_img[y0:y1, x0:x1]

            mu_x, mu_y = float(np.mean(sub_b)), float(np.mean(sub_p))
            var_x, var_y = float(np.var(sub_b)), float(np.var(sub_p))
            cov_xy = float(np.mean((sub_b - mu_x) * (sub_p - mu_y)))
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))

            rows.append({
                "file_name": fname,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "SSIM": ssim,
                "PSNR": psnr,
                "LPIPS": float("nan")
            })

        if not rows:
            continue

        # Batch compute LPIPS on GPU for ultra-fast performance
        if lpips_model is not None:
            try:
                p_tensors = []
                t_tensors = []
                valid_indices = []
                for idx, r in enumerate(rows):
                    fname = r["file_name"]
                    mask = masks[fname]
                    m_f = mask.astype(np.float32)
                    pb = ds_b / fname
                    img_b = np.array(Image.open(pb).convert("L"), dtype=np.float32) / 255.0
                    _, pred_p, _ = image_triplet(run_path, fname)
                    if pred_p is None or not pred_p.exists():
                        continue
                    pred_img = np.array(Image.open(pred_p).convert("L"), dtype=np.float32) / 255.0
                    if pred_img.ndim == 3:
                        pred_img = pred_img[:, :, 0]
                    if img_b.ndim == 3:
                        img_b = img_b[:, :, 0]
                    if pred_img.shape != img_b.shape:
                        pred_img = np.array(Image.fromarray((pred_img * 255).astype(np.uint8)).resize((img_b.shape[1], img_b.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0

                    p_t = torch.from_numpy(pred_img * m_f).unsqueeze(0).repeat(3, 1, 1) * 2.0 - 1.0
                    t_t = torch.from_numpy(img_b * m_f).unsqueeze(0).repeat(3, 1, 1) * 2.0 - 1.0
                    p_tensors.append(p_t)
                    t_tensors.append(t_t)
                    valid_indices.append(idx)

                if p_tensors:
                    bs = 32
                    for b_i in range(0, len(p_tensors), bs):
                        p_batch = torch.stack(p_tensors[b_i:b_i + bs]).to(device)
                        t_batch = torch.stack(t_tensors[b_i:b_i + bs]).to(device)
                        with torch.no_grad():
                            out_lpips = lpips_model(p_batch, t_batch).view(-1).cpu().numpy()
                        for idx_rel, val in enumerate(out_lpips):
                            rows[valid_indices[b_i + idx_rel]]["LPIPS"] = float(val)
            except Exception as e:
                print(f"LPIPS batch error: {e}")

        df_per_img = pd.DataFrame(rows)
        for p in [run_path / "test_evaluation_walkable_per_image.csv", run_path / "logs" / "test_evaluation_walkable_per_image.csv"]:
            p.parent.mkdir(parents=True, exist_ok=True)
            df_per_img.to_csv(p, index=False)

        summary_dict = {
            "MAE": float(df_per_img["MAE"].dropna().mean()) if not df_per_img["MAE"].dropna().empty else float("nan"),
            "MSE": float(df_per_img["MSE"].dropna().mean()) if not df_per_img["MSE"].dropna().empty else float("nan"),
            "RMSE": float(df_per_img["RMSE"].dropna().mean()) if not df_per_img["RMSE"].dropna().empty else float("nan"),
            "SSIM": float(df_per_img["SSIM"].dropna().mean()) if not df_per_img["SSIM"].dropna().empty else float("nan"),
            "PSNR": float(df_per_img["PSNR"].dropna().mean()) if not df_per_img["PSNR"].dropna().empty else float("nan"),
            "LPIPS": float(df_per_img["LPIPS"].dropna().mean()) if not df_per_img["LPIPS"].dropna().empty else float("nan"),
        }
        df_sum = pd.DataFrame([{"metric": k, "value": v} for k, v in summary_dict.items()])
        for p in [run_path / "test_evaluation_walkable_summary.csv", run_path / "logs" / "test_evaluation_walkable_summary.csv"]:
            df_sum.to_csv(p, index=False)

        print(f"✅ Saved walkable CSVs for {run_path.name}: MAE={summary_dict['MAE']:.4f}, SSIM={summary_dict['SSIM']:.4f}, LPIPS={summary_dict['LPIPS']:.4f}")

if __name__ == "__main__":
    compute_walkable_for_runs()

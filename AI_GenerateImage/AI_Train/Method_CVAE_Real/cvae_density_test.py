import argparse
import csv
import json
import math
import pathlib
import time
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cvae_config import TestConfig, write_summary_csv
from cvae_data import BILINEAR, list_pair_files, load_density_target, load_image
from cvae_io import tensor_to_pil
from cvae_losses import tensor_density_metrics
from cvae_model import CVAE

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_backend(cfg):
    use_cudnn = bool(getattr(cfg, "use_cudnn", True))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.enabled = use_cudnn
        print(f"[SYSTEM] cuDNN enabled: {torch.backends.cudnn.enabled}")


def array_to_uint8(chw):
    if torch.is_tensor(chw):
        arr = chw.detach().cpu().numpy()
    else:
        arr = np.asarray(chw)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def tensor_to_image(tensor_chw, out_size):
    arr = array_to_uint8(tensor_chw)
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr, mode=mode).convert("RGB")
    return img.resize(out_size, BILINEAR)


def gray_to_colorjet(gray_uint8: np.ndarray) -> Image.Image:
    gray_norm = gray_uint8.astype(np.float32) / 255.0
    try:
        from matplotlib import colormaps

        rgb = (colormaps["jet"](gray_norm)[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    except Exception:
        x = gray_norm
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_density_display(chw, out_size, path, as_colorjet=False, keep_mask_backup=False):
    arr = array_to_uint8(chw)
    if arr.ndim == 3:
        gray = arr.mean(axis=2).astype(np.uint8)
    else:
        gray = arr

    if as_colorjet:
        gray_img = Image.fromarray(gray, mode="L").resize(out_size, BILINEAR)
        if keep_mask_backup:
            gray_img.convert("RGB").save(path.with_name(f"MASK_{path.name}"))
        colorjet = gray_to_colorjet(np.asarray(gray_img, dtype=np.uint8))
        colorjet.save(path)
        return

    tensor_to_image(chw, out_size).save(path)


def save_error_map(true_chw, pred_chw, path, out_size):
    true_arr = np.asarray(true_chw, dtype=np.float32)
    pred_arr = np.asarray(pred_chw, dtype=np.float32)
    if true_arr.ndim == 3:
        true_arr = true_arr.mean(axis=0)
    if pred_arr.ndim == 3:
        pred_arr = pred_arr.mean(axis=0)
    err = np.abs(pred_arr - true_arr)
    if err.max() > 0:
        err = err / err.max()
    Image.fromarray((err * 255).astype(np.uint8), mode="L").convert("RGB").resize(out_size, BILINEAR).save(path)


def resolve_checkpoint(cfg, args):
    if args.checkpoint:
        checkpoint = pathlib.Path(args.checkpoint)
        checkpoint_label = checkpoint.stem
    else:
        candidates = {
            "best_mae": [cfg.CHECKPOINT_DIR / "best_mae.pt", cfg.CHECKPOINT_DIR / "best.pt"],
            "best_loss": [cfg.CHECKPOINT_DIR / "best_loss.pt"],
            "final": [cfg.CHECKPOINT_DIR / "final.pt"],
        }
        checkpoint = candidates[args.checkpoint_mode][0]
        checkpoint_label = args.checkpoint_mode
        for candidate in candidates[args.checkpoint_mode]:
            if candidate.exists():
                checkpoint = candidate
                break
    if not checkpoint.is_absolute():
        checkpoint = pathlib.Path.cwd() / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint, checkpoint_label


def average_rows(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    res = {}
    for key in keys:
        vals = [row[key] for row in rows if not (isinstance(row[key], float) and math.isnan(row[key]))]
        res[key] = float(np.mean(vals)) if vals else float("nan")
    return res


def write_pix2pix_style_metrics(run_dir, rows, summary):
    per_image_path = run_dir / "test_evaluation_per_image.csv"
    with open(per_image_path, "w", encoding="utf-8") as f:
        f.write("file_name,MAE,MSE,RMSE,SSIM,PSNR,LPIPS,Foreground_MAE,Hotspot_IoU\n")
        for row in rows:
            lpips_txt = "nan" if math.isnan(row.get("lpips", float("nan"))) else f"{row['lpips']:.6f}"
            fg_mae = row.get("foreground_mae", float("nan"))
            hot_iou = row.get("hotspot_iou", float("nan"))
            f.write(
                f"{row['filename']},{row['mae']:.6f},{row['mse']:.6f},{row['rmse']:.6f},"
                f"{row['ssim']:.6f},{row['psnr']:.6f},{lpips_txt},{fg_mae:.6f},{hot_iou:.6f}\n"
            )

    summary_path = run_dir / "test_evaluation_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for metric in ("MAE", "MSE", "RMSE", "SSIM", "PSNR"):
            f.write(f"{metric},{float(summary[metric.lower()]):.6f}\n")
        lpips_score = summary.get("lpips", float("nan"))
        lpips_txt = "nan" if math.isnan(lpips_score) else f"{lpips_score:.6f}"
        f.write(f"LPIPS,{lpips_txt}\n")
        f.write(f"Foreground_MAE,{float(summary.get('foreground_mae', float('nan'))):.6f}\n")
        f.write(f"Hotspot_IoU,{float(summary.get('hotspot_iou', float('nan'))):.6f}\n")


def main(default_config, target_representation, target_channels):
    parser = argparse.ArgumentParser(description=f"Evaluate CVAE density map ({target_representation}).")
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_mode", type=str, default="best_mae", choices=["best_mae", "best_loss", "final"])
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num_samples", type=int, default=None, help="Number of stochastic z samples per input.")
    parser.add_argument("--latent_mode", type=str, default="random", choices=["random", "zero"], help="Latent sampling mode: 'random' or 'zero'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible latent sampling.")
    parser.add_argument("--no_publish_final", action="store_true", help="Do not publish Pix2PixHD-style root final evaluation files.")
    args = parser.parse_args()

    cfg = TestConfig(args.run_path, args.config)
    configure_torch_backend(cfg)
    device = get_device()
    seed = int(args.seed if args.seed is not None else getattr(cfg, "seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    checkpoint, checkpoint_label = resolve_checkpoint(cfg, args)
    state = torch.load(checkpoint, map_location=device)
    state_cfg = state.get("config", {}) if isinstance(state, dict) else {}

    image_size = int(state_cfg.get("image_size", getattr(cfg, "image_size", 256)))
    base_filters = int(state_cfg.get("base_filters", getattr(cfg, "base_filters", 32)))
    latent_dim = int(state_cfg.get("latent_dim", getattr(cfg, "latent_dim", 32)))
    dropout = float(state_cfg.get("dropout", getattr(cfg, "dropout", 0.1)))
    target_representation = str(state_cfg.get("target_representation", getattr(cfg, "target_representation", target_representation)))
    target_channels = int(state_cfg.get("target_channels", getattr(cfg, "target_channels", target_channels)))
    num_samples = int(args.num_samples if args.num_samples is not None else getattr(cfg, "num_samples", 1))
    num_samples = max(num_samples, 1)

    model = CVAE(
        image_size,
        base_filters,
        latent_dim,
        dropout=dropout,
        target_channels=target_channels,
    ).to(device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_model.eval()
            print("[METRIC] LPIPS enabled (alex)")
        except Exception as e:
            print(f"[WARN] LPIPS unavailable: {e}")
            lpips_model = None

    output_name = args.output_name.strip() or checkpoint_label
    result_dir = cfg.TEST_RESULT_DIR / output_name
    pred_dir = result_dir / "predictions"
    input_dir = result_dir / "inputs"
    target_dir = result_dir / "targets"
    error_dir = result_dir / "error_maps"
    for directory in (pred_dir, input_dir, target_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    publish_final = (
        not args.no_publish_final
        and args.split == "test"
        and (output_name == "best_mae" or args.checkpoint_mode == "best_mae")
    )
    final_pred_dir = cfg.TEST_RESULT_DIR / "predictions"
    final_input_dir = cfg.TEST_RESULT_DIR / "inputs"
    final_target_dir = cfg.TEST_RESULT_DIR / "targets"
    if publish_final:
        for directory in (final_pred_dir, final_input_dir, final_target_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for old_png in directory.glob("*.png"):
                old_png.unlink()

    dir_a, dir_b, pair_files = list_pair_files(cfg.DATASET_ROOT, args.split, image_size)
    rows = []
    scalar_rows = []

    metrics_wall_time_s = 0.0
    time_generate_s = 0.0

    print("=" * 70)
    print(f"[RUN] {cfg.CURRENT_RUN_DIR}")
    print(f"[SYSTEM] PyTorch: {torch.__version__} device={device}")
    print(f"[DATA] {cfg.DATASET_ROOT} images={len(pair_files)}")
    print(f"[TARGET] {target_representation} channels={target_channels}")
    print(f"[CKPT] {checkpoint}")
    print(f"[OUTPUT] {result_dir}")
    print(f"[LATENT] mode={args.latent_mode} samples={num_samples} seed={seed}")
    print("=" * 70)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started = time.perf_counter()

    with torch.no_grad():
        progress = tqdm(pair_files, desc=f"Test {output_name}", dynamic_ncols=True)
        for path_a in progress:
            path_b = dir_b / path_a.name
            orig_size = Image.open(path_a).size
            a, _, _ = load_image(path_a, image_size, method="bicubic")
            b = load_density_target(path_b, image_size, target_representation)
            a_device = a.unsqueeze(0).to(device)

            sample_preds = []
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gen_start = time.perf_counter()
            for sample_idx in range(num_samples):
                if args.latent_mode == "random":
                    z = torch.randn((1, latent_dim), dtype=a_device.dtype, device=device)
                else:
                    z = torch.zeros((1, latent_dim), dtype=a_device.dtype, device=device)
                logits = model.forward_infer(a_device, z=z)
                sample_preds.append(torch.sigmoid(logits)[0].detach().cpu().numpy())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            time_generate_s += time.perf_counter() - gen_start

            pred = np.mean(sample_preds, axis=0)
            target = b.numpy()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            met_start = time.perf_counter()
            metrics = tensor_density_metrics(target, pred)
            scalar_metrics = tensor_density_metrics(
                target.mean(axis=0, keepdims=True) if target.shape[0] > 1 else target,
                pred.mean(axis=0, keepdims=True) if pred.shape[0] > 1 else pred,
            )

            # Active region & Hotspot metrics (Reviewer 3)
            fg_threshold = 1.0 / 255.0
            fg_mask = (target > fg_threshold)
            if np.any(fg_mask):
                fg_mae = float(np.mean(np.abs(pred[fg_mask] - target[fg_mask])))
            else:
                fg_mae = 0.0

            hotspot_thresh = 0.20
            target_hotspot = (target >= hotspot_thresh)
            pred_hotspot = (pred >= hotspot_thresh)
            intersection = np.logical_and(target_hotspot, pred_hotspot).sum()
            union = np.logical_or(target_hotspot, pred_hotspot).sum()
            if union > 0:
                hotspot_iou = float(intersection / union)
            else:
                hotspot_iou = 1.0 if not np.any(target_hotspot) else 0.0

            metrics["foreground_mae"] = fg_mae
            metrics["hotspot_iou"] = hotspot_iou
            scalar_metrics["foreground_mae"] = fg_mae
            scalar_metrics["hotspot_iou"] = hotspot_iou

            lpips_val = float("nan")
            if lpips_model is not None:
                try:
                    pred_t = torch.from_numpy(pred).unsqueeze(0).to(device)
                    target_t = torch.from_numpy(target).unsqueeze(0).to(device)
                    # Scale from [0, 1] to [-1, 1]
                    pred_t = pred_t * 2.0 - 1.0
                    target_t = target_t * 2.0 - 1.0
                    if pred_t.shape[1] == 1:
                        pred_t = pred_t.repeat(1, 3, 1, 1)
                    if target_t.shape[1] == 1:
                        target_t = target_t.repeat(1, 3, 1, 1)
                    lpips_val = float(lpips_model(pred_t, target_t).mean().item())
                except Exception as e:
                    print(f"[WARN] LPIPS calculation failed: {e}")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics_wall_time_s += time.perf_counter() - met_start

            rows.append({"filename": path_a.name, "lpips": lpips_val, **metrics})
            scalar_rows.append({"filename": path_a.name, **scalar_metrics})

            tensor_to_pil(a, *orig_size).save(input_dir / path_a.name)
            tensor_to_image(target, orig_size).save(target_dir / path_a.name)
            tensor_to_image(pred, orig_size).save(pred_dir / path_a.name)
            save_error_map(target, pred, error_dir / path_a.name, orig_size)
            if publish_final:
                tensor_to_pil(a, *orig_size).save(final_input_dir / path_a.name)
                as_colorjet = target_channels == 1 or target_representation in {"bw", "gray", "grayscale"}
                save_density_display(
                    pred,
                    orig_size,
                    final_pred_dir / path_a.name,
                    as_colorjet=as_colorjet,
                    keep_mask_backup=as_colorjet,
                )
                save_density_display(
                    target,
                    orig_size,
                    final_target_dir / path_a.name,
                    as_colorjet=as_colorjet,
                    keep_mask_backup=as_colorjet,
                )
            progress.set_postfix(mae=f"{metrics['mae']:.4f}", ssim=f"{metrics['ssim']:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_wall_time_s = time.perf_counter() - test_started
    runtime_excluding_metrics_s = max(0.0, test_wall_time_s - metrics_wall_time_s)

    runtime_row = {
        "method_id": "Method_CVAE_Real",
        "split": args.split,
        "timing_scope": "test_loop_including_data_inference_metrics_postprocess_and_image_write",
        "sample_count": len(rows),
        "Time Generate": f"{time_generate_s:.6f}",
        "Average Time Generate Per Image": (
            f"{time_generate_s / len(rows):.9f}" if rows else "nan"
        ),
        "test_pipeline_wall_time_s": f"{test_wall_time_s:.6f}",
        "metrics_wall_time_s": f"{metrics_wall_time_s:.6f}",
        "runtime_excluding_metrics_s": f"{runtime_excluding_metrics_s:.6f}",
        "mean_runtime_excluding_metrics_per_image_s": (
            f"{runtime_excluding_metrics_s / len(rows):.9f}" if rows else "nan"
        ),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint_path": str(pathlib.Path(checkpoint).resolve()),
        "latent_mode": args.latent_mode,
        "seed": seed,
        "num_samples": num_samples,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(result_dir / "test_runtime.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runtime_row.keys())
        writer.writeheader()
        writer.writerow(runtime_row)
    with open(cfg.CURRENT_RUN_DIR / "test_runtime.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runtime_row.keys())
        writer.writeheader()
        writer.writerow(runtime_row)

    per_image_rows = rows
    numeric_rows = [{k: v for k, v in row.items() if k != "filename"} for row in rows]
    summary = {"split": args.split, "images": len(pair_files), "num_samples": num_samples, **average_rows(numeric_rows)}
    scalar_summary = {
        "split": args.split,
        "images": len(pair_files),
        "num_samples": num_samples,
        **average_rows([{k: v for k, v in row.items() if k != "filename"} for row in scalar_rows]),
    }
    write_summary_csv(result_dir / "test_per_image_metrics.csv", per_image_rows)
    write_summary_csv(result_dir / "test_evaluation_summary.csv", [summary])
    write_summary_csv(result_dir / "test_scalar_density_summary.csv", [scalar_summary])
    with open(result_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump({"image_metrics": summary, "scalar_density_metrics": scalar_summary}, f, indent=4)

    # Always write pix2pix style evaluation metrics for dashboard
    write_pix2pix_style_metrics(result_dir, rows, summary)
    write_pix2pix_style_metrics(cfg.CURRENT_RUN_DIR, rows, summary)
    if publish_final:
        write_pix2pix_style_metrics(cfg.CURRENT_RUN_DIR, rows, summary)
        print(f"[FINAL] Pix2PixHD-style final evaluation published to: {cfg.TEST_RESULT_DIR}")

    print("-" * 70)
    print(
        f"[SUMMARY] mae={summary['mae']:.6f} rmse={summary['rmse']:.6f} "
        f"ssim={summary['ssim']:.4f} psnr={summary['psnr']:.2f}"
    )
    print(
        f"[SCALAR] mae={scalar_summary['mae']:.6f} rmse={scalar_summary['rmse']:.6f} "
        f"ssim={scalar_summary['ssim']:.4f} psnr={scalar_summary['psnr']:.2f}"
    )
    print(f"[DONE] Results: {result_dir}")

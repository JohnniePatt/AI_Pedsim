import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cvae_config import TestConfig, write_summary_csv
from cvae_data import BILINEAR, list_pair_files, load_density_target, load_image
from cvae_io import tensor_to_pil
from cvae_losses import tensor_density_metrics
from cvae_model import CVAE


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
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_pix2pix_style_metrics(run_dir, rows, summary):
    per_image_path = run_dir / "test_evaluation_per_image.csv"
    with open(per_image_path, "w", encoding="utf-8") as f:
        f.write("file_name,MAE,MSE,RMSE,SSIM,PSNR,LPIPS\n")
        for row in rows:
            f.write(
                f"{row['filename']},{row['mae']:.6f},{row['mse']:.6f},{row['rmse']:.6f},"
                f"{row['ssim']:.6f},{row['psnr']:.6f},nan\n"
            )

    summary_path = run_dir / "test_evaluation_summary.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for metric in ("MAE", "MSE", "RMSE", "SSIM", "PSNR"):
            f.write(f"{metric},{float(summary[metric.lower()]):.6f}\n")
        f.write("LPIPS,nan\n")


def main(default_config, target_representation, target_channels):
    parser = argparse.ArgumentParser(description=f"Evaluate CVAE density map ({target_representation}).")
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_mode", type=str, default="best_mae", choices=["best_mae", "best_loss", "final"])
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--num_samples", type=int, default=None, help="Number of stochastic z samples per input. 1 uses z=0.")
    parser.add_argument("--no_publish_final", action="store_true", help="Do not publish Pix2PixHD-style root final evaluation files.")
    args = parser.parse_args()

    cfg = TestConfig(args.run_path, args.config)
    configure_torch_backend(cfg)
    device = get_device()
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

    print("=" * 70)
    print(f"[RUN] {cfg.CURRENT_RUN_DIR}")
    print(f"[SYSTEM] PyTorch: {torch.__version__} device={device}")
    print(f"[DATA] {cfg.DATASET_ROOT} images={len(pair_files)}")
    print(f"[TARGET] {target_representation} channels={target_channels}")
    print(f"[CKPT] {checkpoint}")
    print(f"[OUTPUT] {result_dir}")
    print(f"[SAMPLES] {num_samples}")
    print("=" * 70)

    with torch.no_grad():
        progress = tqdm(pair_files, desc=f"Test {output_name}", dynamic_ncols=True)
        for path_a in progress:
            path_b = dir_b / path_a.name
            orig_size = Image.open(path_a).size
            a, _, _ = load_image(path_a, image_size, method="bicubic")
            b = load_density_target(path_b, image_size, target_representation)
            a_device = a.unsqueeze(0).to(device)

            sample_preds = []
            for sample_idx in range(num_samples):
                z = None
                if num_samples > 1:
                    z = torch.randn((1, latent_dim), dtype=a_device.dtype, device=device)
                logits = model.forward_infer(a_device, z=z)
                sample_preds.append(torch.sigmoid(logits)[0].detach().cpu().numpy())
            pred = np.mean(sample_preds, axis=0)
            target = b.numpy()

            metrics = tensor_density_metrics(target, pred)
            scalar_metrics = tensor_density_metrics(
                target.mean(axis=0, keepdims=True) if target.shape[0] > 1 else target,
                pred.mean(axis=0, keepdims=True) if pred.shape[0] > 1 else pred,
            )
            rows.append({"filename": path_a.name, **metrics})
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

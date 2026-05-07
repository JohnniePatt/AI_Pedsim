import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cvae_config import TestConfig, write_summary_csv
from cvae_data import BILINEAR, NEAREST, list_pair_files, load_image, load_mask
from cvae_io import save_binary_mask, save_probability, tensor_to_pil
from cvae_losses import hard_metrics
from cvae_model import CVAE


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_thresholds(value):
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def threshold_label(threshold):
    return f"t{int(round(float(threshold) * 100)):03d}"


def resize_mask(mask, size):
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize(size, NEAREST)
    return (np.asarray(img) >= 128).astype(np.uint8)


def resize_prob(prob, size):
    img = Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    img = img.resize(size, BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch CVAE trajectory-line masks.")
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_test.json")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_mode", type=str, default="best_dice", choices=["best_dice", "best_loss", "final"])
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--export_thresholds", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of stochastic z samples per input. 1 uses z=0.")
    args = parser.parse_args()

    cfg = TestConfig(args.run_path, args.config)
    device = get_device()
    threshold = float(args.threshold if args.threshold is not None else getattr(cfg, "mask_threshold", 0.65))
    export_thresholds = parse_thresholds(args.export_thresholds) or parse_thresholds(getattr(cfg, "export_thresholds", [0.5, 0.6, 0.65, 0.7, 0.8]))
    if threshold not in export_thresholds:
        export_thresholds.append(threshold)
    export_thresholds = sorted(set(round(float(t), 4) for t in export_thresholds))
    num_samples = int(args.num_samples if args.num_samples is not None else getattr(cfg, "num_samples", 1))
    num_samples = max(num_samples, 1)

    if args.checkpoint:
        checkpoint = pathlib.Path(args.checkpoint)
        checkpoint_label = checkpoint.stem
    else:
        candidates = {
            "best_dice": [cfg.CHECKPOINT_DIR / "best_dice.pt", cfg.CHECKPOINT_DIR / "best.pt"],
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

    state = torch.load(checkpoint, map_location=device)
    state_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    image_size = int(state_cfg.get("image_size", getattr(cfg, "image_size", 256)))
    base_filters = int(state_cfg.get("base_filters", getattr(cfg, "base_filters", 32)))
    latent_dim = int(state_cfg.get("latent_dim", getattr(cfg, "latent_dim", 32)))
    dropout = float(state_cfg.get("dropout", getattr(cfg, "dropout", 0.1)))

    model = CVAE(image_size, base_filters, latent_dim, dropout=dropout).to(device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    output_name = args.output_name.strip() or checkpoint_label
    result_dir = cfg.TEST_RESULT_DIR / output_name
    pred_dir = result_dir / "predictions"
    prob_dir = result_dir / "probability_maps"
    input_dir = result_dir / "inputs"
    target_dir = result_dir / "targets"
    for directory in (pred_dir, prob_dir, input_dir, target_dir):
        directory.mkdir(parents=True, exist_ok=True)
    threshold_dirs = {}
    for export_threshold in export_thresholds:
        directory = result_dir / f"predictions_{threshold_label(export_threshold)}"
        directory.mkdir(parents=True, exist_ok=True)
        threshold_dirs[export_threshold] = directory

    dir_a, dir_b, pair_files = list_pair_files(cfg.DATASET_ROOT, args.split, image_size)

    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    threshold_totals = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in export_thresholds}

    print("=" * 70)
    print(f"[RUN] {cfg.CURRENT_RUN_DIR}")
    print(f"[SYSTEM] PyTorch: {torch.__version__} device={device}")
    print(f"[DATA] {cfg.DATASET_ROOT} images={len(pair_files)}")
    print(f"[CKPT] {checkpoint}")
    print(f"[OUTPUT] {result_dir}")
    print(f"[THRESHOLD] {threshold} export={export_thresholds} samples={num_samples}")
    print("=" * 70)

    with torch.no_grad():
        progress = tqdm(pair_files, desc=f"Test {output_name}", dynamic_ncols=True)
        for index, path_a in enumerate(progress, start=1):
            path_b = dir_b / path_a.name
            orig_size = Image.open(path_a).size
            a, _, _ = load_image(path_a, image_size, method="bicubic")
            b, _, _ = load_mask(path_b, image_size)
            a_device = a.unsqueeze(0).to(device)

            sample_probs = []
            for sample_idx in range(num_samples):
                z = None
                if num_samples > 1:
                    z = torch.randn((1, latent_dim), dtype=a_device.dtype, device=device)
                logits = model.forward_infer(a_device, z=z)
                sample_probs.append(torch.sigmoid(logits)[0, 0].detach().cpu().numpy())
            prob = np.mean(sample_probs, axis=0)
            target = b[0].numpy()
            pred = (prob >= threshold).astype(np.uint8)

            metrics = hard_metrics(target, prob, threshold)
            for key in totals:
                totals[key] += int(metrics[key])
            rows.append({"filename": path_a.name, **metrics})

            tensor_to_pil(a, *orig_size).save(input_dir / path_a.name)
            save_binary_mask((np.asarray(Image.open(path_b).convert("L")) >= 128), target_dir / path_a.name)
            save_binary_mask(resize_mask(pred, orig_size), pred_dir / path_a.name)
            save_probability(resize_prob(prob, orig_size), prob_dir / path_a.name)
            for export_threshold, directory in threshold_dirs.items():
                export_pred = (prob >= export_threshold).astype(np.uint8)
                save_binary_mask(resize_mask(export_pred, orig_size), directory / path_a.name)
                export_metrics = hard_metrics(target, prob, export_threshold)
                for key in threshold_totals[export_threshold]:
                    threshold_totals[export_threshold][key] += int(export_metrics[key])
            progress.set_postfix(dice=f"{metrics['dice']:.4f}", iou=f"{metrics['iou']:.4f}")

    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    eps = 1e-9
    summary = {
        "split": args.split,
        "images": len(pair_files),
        "threshold": threshold,
        "num_samples": num_samples,
        "precision": tp / max(tp + fp, eps),
        "recall": tp / max(tp + fn, eps),
        "dice": (2 * tp) / max(2 * tp + fp + fn, eps),
        "iou": tp / max(tp + fp + fn, eps),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    write_summary_csv(result_dir / "test_per_image_metrics.csv", rows)
    write_summary_csv(result_dir / "test_evaluation_summary.csv", [summary])
    threshold_rows = []
    for export_threshold, export_totals in threshold_totals.items():
        e_tp, e_fp, e_fn, e_tn = export_totals["tp"], export_totals["fp"], export_totals["fn"], export_totals["tn"]
        threshold_rows.append(
            {
                "threshold": export_threshold,
                "precision": e_tp / max(e_tp + e_fp, eps),
                "recall": e_tp / max(e_tp + e_fn, eps),
                "dice": (2 * e_tp) / max(2 * e_tp + e_fp + e_fn, eps),
                "iou": e_tp / max(e_tp + e_fp + e_fn, eps),
                "accuracy": (e_tp + e_tn) / max(e_tp + e_tn + e_fp + e_fn, eps),
                "tp": e_tp,
                "fp": e_fp,
                "fn": e_fn,
                "tn": e_tn,
                "prediction_dir": f"predictions_{threshold_label(export_threshold)}",
            }
        )
    write_summary_csv(result_dir / "test_threshold_metrics.csv", threshold_rows)
    with open(result_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("-" * 70)
    print(f"[SUMMARY] dice={summary['dice']:.4f} iou={summary['iou']:.4f} precision={summary['precision']:.4f} recall={summary['recall']:.4f}")
    print(f"[DONE] Results: {result_dir}")


if __name__ == "__main__":
    main()

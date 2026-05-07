import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from unet_common import (
    BILINEAR,
    NEAREST,
    build_unet,
    get_device,
    hard_metrics,
    list_paired_files,
    load_json_config,
    resolve_path,
    resolve_project_root,
    write_summary_csv,
)


def load_run_config(run_path: pathlib.Path, config_path: pathlib.Path | None) -> dict:
    script_dir = pathlib.Path(__file__).parent.resolve()
    project_root = resolve_project_root(script_dir)
    cfg = {
        "image_size": 256,
        "batch_size": 1,
        "mask_threshold": 0.5,
        "base_filters": 32,
        "dropout": 0.1,
        "dataset_root": "../Dataset/Data_ImageUNet/Trajectory_line_mask_dataset/Topo_HouseGAN",
    }
    if run_path:
        snapshot = run_path / "run_config_snapshot.json"
        if snapshot.exists():
            cfg.update(load_json_config(snapshot))
    if config_path and config_path.exists():
        cfg.update(load_json_config(config_path))
    cfg["PROJECT_ROOT"] = str(project_root)
    if "DATASET_ROOT" not in cfg:
        cfg["DATASET_ROOT"] = str(resolve_path(cfg["dataset_root"], project_root))
    return cfg


def load_rgb_input(path: pathlib.Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_mask(path: pathlib.Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((image_size, image_size), NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr >= 0.5).astype(np.float32)


def save_binary_mask(mask: np.ndarray, path: pathlib.Path):
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")
    img.save(path)


def save_probability(prob: np.ndarray, path: pathlib.Path):
    img = Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    img.save(path)


def resize_mask_to_size(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize(size, NEAREST)
    return (np.asarray(img) >= 128).astype(np.uint8)


def resize_prob_to_size(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    img = img.resize(size, BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def parse_thresholds(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def threshold_label(threshold: float) -> str:
    return f"t{int(round(float(threshold) * 100)):03d}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory-mask PyTorch U-Net.")
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="config_test.json")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint_mode", type=str, default="best_dice", choices=["best_dice", "best_loss", "final"])
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--export_thresholds", type=str, default="")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    run_path = pathlib.Path(args.run_path)
    if not run_path.is_absolute():
        run_path = (pathlib.Path.cwd() / run_path).resolve()
        if not run_path.exists():
            run_path = (script_dir / args.run_path).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = pathlib.Path.cwd() / config_path
        if not config_path.exists():
            config_path = script_dir / args.config

    cfg = load_run_config(run_path, config_path if config_path.exists() else None)
    dataset_root = pathlib.Path(cfg["DATASET_ROOT"])
    image_size = int(cfg["image_size"])
    threshold = float(args.threshold if args.threshold is not None else cfg.get("mask_threshold", 0.5))
    device = get_device()
    export_thresholds = parse_thresholds(args.export_thresholds) or parse_thresholds(cfg.get("export_thresholds"))
    if threshold not in export_thresholds:
        export_thresholds.append(threshold)
    export_thresholds = sorted(set(round(float(t), 4) for t in export_thresholds))

    if args.checkpoint:
        checkpoint = pathlib.Path(args.checkpoint)
        checkpoint_label = pathlib.Path(args.checkpoint).stem
    else:
        checkpoint_candidates = {
            "best_dice": [run_path / "checkpoints" / "best_dice.pt", run_path / "checkpoints" / "best.pt"],
            "best_loss": [run_path / "checkpoints" / "best_loss.pt"],
            "final": [run_path / "checkpoints" / "final.pt"],
        }
        checkpoint = checkpoint_candidates[args.checkpoint_mode][0]
        checkpoint_label = args.checkpoint_mode
        for candidate in checkpoint_candidates[args.checkpoint_mode]:
            if candidate.exists():
                checkpoint = candidate
                break
    if not checkpoint.is_absolute():
        checkpoint = (pathlib.Path.cwd() / checkpoint).resolve()
    if not checkpoint.exists():
        fallback = run_path / "checkpoints" / "final.pt"
        if fallback.exists():
            checkpoint = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = build_unet(
        base_filters=int(cfg.get("base_filters", 32)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    output_name = args.output_name.strip() or checkpoint_label
    result_dir = run_path / "test_results" / output_name
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

    names = list_paired_files(dataset_root, args.split)
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    print("=" * 70)
    print(f"[RUN] {run_path}")
    print(f"[SYSTEM] PyTorch: {torch.__version__} device={device}")
    print(f"[DATA] {dataset_root} split={args.split} images={len(names)}")
    print(f"[CKPT] {checkpoint}")
    print(f"[OUTPUT] {result_dir}")
    print(f"[THRESHOLD] {threshold}")
    print(f"[EXPORT THRESHOLDS] {export_thresholds}")
    print("=" * 70)

    threshold_totals = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in export_thresholds}
    with torch.no_grad():
        progress = tqdm(names, desc=f"Test {args.split}", dynamic_ncols=True)
        for index, name in enumerate(progress, start=1):
            a_path = dataset_root / "A" / args.split / name
            b_path = dataset_root / "B" / args.split / name
            orig_size = Image.open(a_path).size
            a = load_rgb_input(a_path, image_size).unsqueeze(0).to(device)
            b = load_mask(b_path, image_size)
            logits = model(a)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            pred = (prob >= threshold).astype(np.uint8)

            metrics = hard_metrics(b, prob, threshold)
            for key in totals:
                totals[key] += int(metrics[key])
            rows.append(
                {
                    "filename": name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "accuracy": metrics["accuracy"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                }
            )

            Image.open(a_path).convert("RGB").save(input_dir / name)
            save_binary_mask((np.asarray(Image.open(b_path).convert("L")) >= 128), target_dir / name)
            save_binary_mask(resize_mask_to_size(pred, orig_size), pred_dir / name)
            save_probability(resize_prob_to_size(prob, orig_size), prob_dir / name)
            for export_threshold, directory in threshold_dirs.items():
                export_pred = (prob >= export_threshold).astype(np.uint8)
                save_binary_mask(resize_mask_to_size(export_pred, orig_size), directory / name)
                export_metrics = hard_metrics(b, prob, export_threshold)
                for key in threshold_totals[export_threshold]:
                    threshold_totals[export_threshold][key] += int(export_metrics[key])

            progress.set_postfix(
                dice=f"{metrics['dice']:.4f}",
                iou=f"{metrics['iou']:.4f}",
            )

    tp, fp, fn, tn = totals["tp"], totals["fp"], totals["fn"], totals["tn"]
    eps = 1e-9
    summary = {
        "split": args.split,
        "images": len(names),
        "threshold": threshold,
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
        e_tp = export_totals["tp"]
        e_fp = export_totals["fp"]
        e_fn = export_totals["fn"]
        e_tn = export_totals["tn"]
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
    print(
        "[SUMMARY] "
        f"dice={summary['dice']:.4f} iou={summary['iou']:.4f} "
        f"precision={summary['precision']:.4f} recall={summary['recall']:.4f}"
    )
    print(f"[DONE] Results: {result_dir}")


if __name__ == "__main__":
    main()

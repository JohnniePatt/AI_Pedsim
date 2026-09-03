from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
EXPECTED_CASES = {"train": 2603, "validation": 439, "test": 862}
EXPECTED_PLANS = {"train": 412, "validation": 60, "test": 117}
METRICS = ("MAE", "MSE", "RMSE", "PSNR", "SSIM", "LPIPS")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_dataset(dataset_root: Path, snapshot: Path) -> tuple[dict, dict, list[str]]:
    inventory: dict[str, dict] = {}
    plan_sets: dict[str, set[str]] = {}
    snapshot_rows: list[tuple[str, str, str]] = []
    test_names: list[str] = []
    problems: list[str] = []
    for split in ("train", "validation", "test"):
        a_names = {path.name for path in (dataset_root / "A" / split).glob("*.png")}
        b_names = {path.name for path in (dataset_root / "B" / split).glob("*.png")}
        paired = sorted(a_names & b_names)
        plans = {name.split("__", 1)[0] for name in paired}
        inventory[split] = {
            "input_count": len(a_names),
            "target_count": len(b_names),
            "paired_count": len(paired),
            "plan_count": len(plans),
            "input_only_count": len(a_names - b_names),
            "target_only_count": len(b_names - a_names),
        }
        plan_sets[split] = plans
        snapshot_rows.extend((split, name, name.split("__", 1)[0]) for name in paired)
        if split == "test":
            test_names = paired
        if len(paired) != EXPECTED_CASES[split] or len(plans) != EXPECTED_PLANS[split]:
            problems.append(f"{split}: cases={len(paired)}, plans={len(plans)}")
        if a_names != b_names:
            problems.append(f"{split}: A/B filename mismatch")
    overlap = {
        "train_validation": len(plan_sets["train"] & plan_sets["validation"]),
        "train_test": len(plan_sets["train"] & plan_sets["test"]),
        "validation_test": len(plan_sets["validation"] & plan_sets["test"]),
    }
    if any(overlap.values()):
        problems.append(f"plan overlap={overlap}")
    if problems:
        raise RuntimeError("Canonical dataset verification failed: " + "; ".join(problems))
    with snapshot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "file_name", "plan_id"])
        writer.writerows(snapshot_rows)
    return inventory, overlap, test_names


def load_gray(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("L").resize((size, size), BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0


def load_walkable_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((size, size), BILINEAR)
        return np.asarray(resized, dtype=np.uint8).sum(axis=2) > 0


def density_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
    difference = prediction - target
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(difference**2))
    rmse = float(math.sqrt(mse))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))
    mean_t, mean_p = float(target.mean()), float(prediction.mean())
    var_t, var_p = float(target.var()), float(prediction.var())
    covariance = float(np.mean((target - mean_t) * (prediction - mean_p)))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_t * mean_p + c1) * (2 * covariance + c2)) / (
        (mean_t**2 + mean_p**2 + c1) * (var_t + var_p + c2)
    )
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "PSNR": psnr, "SSIM": float(ssim)}


def write_metric_files(output_dir: Path, rows: list[dict], suffix: str = "") -> dict[str, float]:
    per_image = output_dir / f"test_evaluation{suffix}_per_image.csv"
    summary_file = output_dir / f"test_evaluation{suffix}_summary.csv"
    with per_image.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    summary = {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}
    with summary_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())
    return summary


def add_lpips(
    rows: list[dict],
    walkable_rows: list[dict],
    prediction_dir: Path,
    dataset_root: Path,
    names: list[str],
    size: int,
    network: str,
) -> None:
    import torch
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net=network).to(device).eval()
    row_index = {row["file_name"]: row for row in rows}
    walkable_index = {row["file_name"]: row for row in walkable_rows}
    with torch.no_grad():
        for start in range(0, len(names), 32):
            batch_names = names[start:start + 32]
            predictions = torch.from_numpy(np.stack([
                load_gray(prediction_dir / name, size) for name in batch_names
            ])).unsqueeze(1)
            targets = torch.from_numpy(np.stack([
                load_gray(dataset_root / "B" / "test" / name, size) for name in batch_names
            ])).unsqueeze(1)
            masks = torch.from_numpy(np.stack([
                load_walkable_mask(dataset_root / "A" / "test" / name, size) for name in batch_names
            ])).unsqueeze(1).float()
            pred_rgb = predictions.repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
            target_rgb = targets.repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
            walk_pred_rgb = (predictions * masks).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
            walk_target_rgb = (targets * masks).repeat(1, 3, 1, 1).to(device) * 2.0 - 1.0
            full_values = model(pred_rgb, target_rgb).reshape(-1).cpu().numpy()
            walk_values = model(walk_pred_rgb, walk_target_rgb).reshape(-1).cpu().numpy()
            for name, full_value, walk_value in zip(batch_names, full_values, walk_values):
                row_index[name]["LPIPS"] = float(full_value)
                walkable_index[name]["LPIPS"] = float(walk_value)


def evaluate_run(run: dict, dataset_root: Path, names: list[str], output_dir: Path, size: int) -> dict:
    run_dir = resolve_path(run["run_dir"])
    checkpoint = run_dir / run["checkpoint"]
    prediction_dir = run_dir / "test_results" / "predictions"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != run["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch for {run['cell_id']}: {checkpoint_hash}")
    prediction_names = {path.name for path in prediction_dir.glob("*.png")}
    expected_names = set(names)
    if prediction_names != expected_names:
        raise RuntimeError(
            f"Prediction set mismatch for {run['cell_id']}: "
            f"missing={len(expected_names - prediction_names)}, extra={len(prediction_names - expected_names)}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    walkable_rows: list[dict] = []
    prediction_manifest: list[tuple[str, int, str]] = []
    for name in names:
        prediction_path = prediction_dir / name
        target = load_gray(dataset_root / "B" / "test" / name, size)
        prediction = load_gray(prediction_path, size)
        mask = load_walkable_mask(dataset_root / "A" / "test" / name, size)
        if not mask.any():
            raise RuntimeError(f"Empty walkable mask: {name}")
        rows.append({"file_name": name, **density_metrics(target, prediction), "LPIPS": float("nan")})
        walkable_rows.append({
            "file_name": name,
            **density_metrics(target[mask], prediction[mask]),
            "LPIPS": float("nan"),
        })
        prediction_manifest.append((name, prediction_path.stat().st_size, sha256_file(prediction_path)))

    add_lpips(
        rows,
        walkable_rows,
        prediction_dir,
        dataset_root,
        names,
        size,
        run.get("lpips_network", "alex"),
    )
    summary = write_metric_files(output_dir, rows)
    walkable_summary = write_metric_files(output_dir, walkable_rows, suffix="_walkable")
    manifest_path = output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "bytes", "sha256"])
        writer.writerows(prediction_manifest)
    write_json(output_dir / "checkpoint_ref.json", {
        "run_dir": run["run_dir"],
        "checkpoint": run["checkpoint"],
        "checkpoint_sha256": checkpoint_hash,
        "seed": run.get("seed"),
        "provenance_status": run["provenance_status"],
    })
    return {**run, "metrics_dir": str(output_dir.relative_to(project_root())), "summary": summary,
            "walkable_summary": walkable_summary, "prediction_manifest_sha256": sha256_file(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Common post-hoc evaluation for the corrected representative 2x2 set.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_representative_2x2.json")))
    parser.add_argument("--output-dir", help="Unique output directory; defaults to a UTC-stamped directory.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_root = resolve_path(config["dataset_root"])
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(
        f"AI_GenerateImage/AI_Result/RepresentativeComparisons/comparison_{timestamp}_{config['comparison_id']}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = output_dir / "dataset_manifest_snapshot.csv"
    inventory, overlap, names = verify_dataset(dataset_root, snapshot)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    evaluated_runs = []
    for run in config["runs"]:
        print(f"Evaluating {run['display_name']} ({len(names)} cases)", flush=True)
        run_config = {**run, "lpips_network": config.get("lpips_network", "alex")}
        evaluated_runs.append(evaluate_run(run_config, dataset_root, names, output_dir / "runs" / run["cell_id"], int(config["image_size"])))

    results_path = output_dir / "comparison_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["cell_id", "display_name", *METRICS, *(f"walkable_{metric}" for metric in METRICS)]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in evaluated_runs:
            writer.writerow({
                "cell_id": run["cell_id"],
                "display_name": run["display_name"],
                **run["summary"],
                **{f"walkable_{key}": value for key, value in run["walkable_summary"].items()},
            })

    incomplete_legacy_provenance = any(run["provenance_status"] != "complete" for run in evaluated_runs)
    manifest = {
        "comparison_id": config["comparison_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "comparison_type": "corrected_representative_2x2",
        "interpretation": "method-family representative comparison; not a strict component-isolated factorial",
        "dataset_id": config["dataset_id"],
        "split": "test",
        "case_count": len(names),
        "floorplan_count": len({name.split("__", 1)[0] for name in names}),
        "image_size": config["image_size"],
        "prediction_source": config["prediction_source"],
        "metric_protocol": config["metric_protocol"],
        "inventory": inventory,
        "plan_overlap": overlap,
        "dataset_manifest_sha256": sha256_file(snapshot),
        "evaluation_code_sha256": sha256_file(Path(__file__).resolve()),
        "evaluation_config_sha256": sha256_file(output_dir / "evaluation_config.json"),
        "research_valid": not incomplete_legacy_provenance,
        "research_validity_note": (
            "false: the two retained legacy runs do not record seeds in modern provenance manifests; "
            "results are real and complete but limited to descriptive comparison"
            if incomplete_legacy_provenance else "all current research-validity checks passed"
        ),
        "runs": evaluated_runs,
    }
    write_json(output_dir / "comparison_manifest.json", manifest)
    print(f"OUTPUT_DIR={output_dir}")
    print(results_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()

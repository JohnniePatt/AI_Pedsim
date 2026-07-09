import argparse
import pathlib
import sys
import json
from PIL import Image

def upscale_run_images(run_path: pathlib.Path, target_size: int = 1024) -> int:
    run_path = pathlib.Path(run_path).resolve()
    snapshot_path = run_path / "run_config_snapshot.json"
    if not snapshot_path.exists():
        print(f"[ERROR] Snapshot config not found at {snapshot_path}", file=sys.stderr)
        return 1

    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read snapshot: {e}", file=sys.stderr)
        return 1

    dataset_root_raw = snapshot.get("DATASET_ROOT", snapshot.get("dataset_root", ""))
    if not dataset_root_raw:
        print("[ERROR] DATASET_ROOT not defined in snapshot.", file=sys.stderr)
        return 1

    dataset_root = pathlib.Path(dataset_root_raw)
    if not dataset_root.is_absolute():
        project_root = pathlib.Path(__file__).resolve().parent.parent / "AI_GenerateTrajectory"
        dataset_root = (project_root / dataset_root).resolve()

    test_a_dir = dataset_root / "A" / "test"
    if not test_a_dir.exists():
        print(f"[ERROR] Original dataset folder not found at {test_a_dir}", file=sys.stderr)
        return 1

    test_results_path = run_path / "test_results"
    if not test_results_path.exists():
        print(f"[ERROR] test_results folder not found at {test_results_path}", file=sys.stderr)
        return 1

    # Find all 'inputs' subdirectories (handling nested structures like best_loss/inputs)
    input_dirs = []
    if (test_results_path / "inputs").exists():
        input_dirs.append(test_results_path / "inputs")
    for cp in ["best_loss", "best_dice", "best_mae", "final"]:
        if (test_results_path / cp / "inputs").exists():
            input_dirs.append(test_results_path / cp / "inputs")

    if not input_dirs:
        print(f"[ERROR] No inputs directory found under {test_results_path}", file=sys.stderr)
        return 1

    total_replaced = 0
    for input_dir in input_dirs:
        print(f"[SCAN] Processing directory: {input_dir.relative_to(run_path)}")
        pred_dir = input_dir.parent / "predictions"
        target_dir = input_dir.parent / "targets"

        for p_path in input_dir.glob("*.png"):
            if p_path.name.startswith("MASK_"):
                continue
                
            orig_file = test_a_dir / p_path.name
            if not orig_file.exists():
                print(f"[WARN] Original image not found for: {p_path.name}")
                continue

            # Backup the original blurry input image as MASK_<filename> if it doesn't exist
            backup_path = p_path.with_name(f"MASK_{p_path.name}")
            if not backup_path.exists():
                p_path.rename(backup_path)

            # 1. Upscale Input (NEAREST to keep sharp walls and open doors)
            with Image.open(orig_file) as img:
                img_resized = img.resize((target_size, target_size), Image.NEAREST)
                img_resized.save(p_path)

            # 2. Upscale Prediction (LANCZOS for smooth heatmap gradient)
            pred_file = pred_dir / p_path.name
            if pred_file.exists():
                backup_pred = pred_file.with_name(f"MASK_{pred_file.name}")
                if not backup_pred.exists():
                    pred_file.rename(backup_pred)
                with Image.open(backup_pred) as img:
                    img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                    img_resized.save(pred_file)
                    img_resized.save(backup_pred) # Overwrite backup with upscaled version

            # 3. Upscale Target (LANCZOS for smooth heatmap gradient)
            target_file = target_dir / p_path.name
            if target_file.exists():
                backup_target = target_file.with_name(f"MASK_{target_file.name}")
                if not backup_target.exists():
                    target_file.rename(backup_target)
                with Image.open(backup_target) as img:
                    img_resized = img.resize((target_size, target_size), Image.LANCZOS)
                    img_resized.save(target_file)
                    img_resized.save(backup_target) # Overwrite backup with upscaled version

            total_replaced += 1
            print(f"[OK] Upscaled to {target_size}x{target_size}: {p_path.name}")

    print("=" * 60)
    print(f"[SUCCESS] Upscaled {total_replaced} image sets under {run_path.name}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Upscale run inputs/predictions/targets to 1024x1024 using clean dataset images.")
    parser.add_argument("--run_path", required=True, help="Path to the run directory.")
    parser.add_argument("--size", type=int, default=1024, help="Target image dimension (default: 1024).")
    args = parser.parse_args()

    run_path = pathlib.Path(args.run_path)
    sys.exit(upscale_run_images(run_path, args.size))

if __name__ == "__main__":
    main()

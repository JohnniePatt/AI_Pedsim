import argparse
import pathlib
import sys

import numpy as np
from PIL import Image


DEFAULT_DIRS = ("predictions", "targets")


def gray_to_colorjet(gray_uint8: np.ndarray) -> Image.Image:
    gray_norm = gray_uint8.astype(np.float32) / 255.0
    try:
        from matplotlib import colormaps

        rgb = (colormaps["jet"](gray_norm)[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    except Exception:
        # Small dependency-free fallback that approximates the JET color ramp.
        x = gray_norm
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        rgb = (np.stack([r, g, b], axis=-1) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def load_mask_gray(mask_path: pathlib.Path) -> np.ndarray:
    with Image.open(mask_path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)


def format_directory(directory: pathlib.Path, dry_run: bool, overwrite: bool) -> tuple[int, int]:
    if not directory.exists():
        print(f"[SKIP] Missing directory: {directory}")
        return 0, 0

    converted = 0
    skipped = 0
    image_paths = sorted(p for p in directory.glob("*.png") if not p.name.startswith("MASK_"))

    print(f"[SCAN] {directory} | {len(image_paths)} PNG files")
    for image_path in image_paths:
        mask_path = image_path.with_name(f"MASK_{image_path.name}")

        if mask_path.exists():
            if not overwrite:
                print(f"[SKIP] {image_path.name} already has {mask_path.name}")
                skipped += 1
                continue
            source_mask = mask_path
            action = "regenerate"
        else:
            source_mask = mask_path
            action = "rename+convert"

        if dry_run:
            print(f"[DRY] {action}: {image_path.name} -> {mask_path.name}; colorjet -> {image_path.name}")
            converted += 1
            continue

        if not mask_path.exists():
            image_path.rename(mask_path)

        gray = load_mask_gray(source_mask)
        colorjet = gray_to_colorjet(gray)
        colorjet.save(image_path)
        print(f"[OK] {action}: {image_path.name} | mask={mask_path.name}")
        converted += 1

    return converted, skipped


def resolve_run_path(run_path: str) -> pathlib.Path:
    path = pathlib.Path(run_path).expanduser()
    if path.exists():
        return path.resolve()

    project_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [
        project_root / run_path,
        project_root / "AI_GenerateTrajectory" / "AI_Result" / "Method_pix2pixHD" / "outputs" / run_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename BW mask result images to MASK_<filename> and write COLORJET images "
            "back to the original filenames."
        )
    )
    parser.add_argument("--run_path", required=True, help="Path to a training run folder containing test_results.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=list(DEFAULT_DIRS),
        help="Subdirectories under test_results to format. Default: predictions targets",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If MASK_<filename> already exists, regenerate the colorjet image from it.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print actions without modifying files.")
    args = parser.parse_args()

    run_path = resolve_run_path(args.run_path)
    test_results = run_path / "test_results"
    if not test_results.exists():
        print(f"[ERROR] test_results not found: {test_results}", file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"run_path     : {run_path}")
    print(f"test_results : {test_results}")
    print(f"dirs         : {', '.join(args.dirs)}")
    print(f"overwrite    : {args.overwrite}")
    print(f"dry_run      : {args.dry_run}")
    print("=" * 72)

    total_converted = 0
    total_skipped = 0
    for dirname in args.dirs:
        converted, skipped = format_directory(test_results / dirname, args.dry_run, args.overwrite)
        total_converted += converted
        total_skipped += skipped

    print("-" * 72)
    print(f"[SUMMARY] converted={total_converted} skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

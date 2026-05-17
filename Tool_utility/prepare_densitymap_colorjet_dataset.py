import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import colormaps
from tqdm import tqdm


def parse_topologies(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path):
    shutil.copytree(src, dst, dirs_exist_ok=True)


def convert_b_to_colorjet(topology_dst: Path):
    b_dir = topology_dst / "B"
    if not b_dir.exists():
        raise FileNotFoundError(f"Missing B directory: {b_dir}")

    png_files = sorted(b_dir.rglob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {b_dir}")

    jet_cmap = colormaps["jet"]
    for png_path in tqdm(png_files, desc=f"[{topology_dst.name}] B->ColorJet", unit="img"):
        gray = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
        jet_rgb = (jet_cmap(gray / 255.0)[..., :3] * 255.0).astype(np.uint8)
        Image.fromarray(jet_rgb, mode="RGB").save(png_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create DensityMap_COLORJET_dataset from existing DensityMap_dataset."
    )
    parser.add_argument(
        "--source_dataset_root",
        type=str,
        default="Dataset/Data_ImageUNet/DensityMap_dataset",
        help="Source grayscale density dataset root.",
    )
    parser.add_argument(
        "--output_dataset_root",
        type=str,
        default="Dataset/Data_ImageUNet/DensityMap_COLORJET_dataset",
        help="Output ColorJet density dataset root.",
    )
    parser.add_argument(
        "--topologies",
        type=str,
        default="Topo_HouseGAN",
        help="Comma-separated topology names (e.g. Topo_HouseGAN,Topo_bottleneck).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output topology directory if exists.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_dataset_root).resolve()
    output_root = Path(args.output_dataset_root).resolve()
    topologies = parse_topologies(args.topologies)

    print("=" * 70)
    print(f"source_dataset_root : {source_root}")
    print(f"output_dataset_root : {output_root}")
    print(f"topologies          : {', '.join(topologies)}")
    print(f"overwrite           : {args.overwrite}")
    print("=" * 70)

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    prepared = 0
    for topo in topologies:
        src_topo = source_root / topo
        dst_topo = output_root / topo

        if not src_topo.exists():
            print(f"[SKIP] Missing source topology: {src_topo}")
            continue

        ensure_clean_dir(dst_topo, args.overwrite)
        copy_tree(src_topo, dst_topo)
        convert_b_to_colorjet(dst_topo)

        prepared += 1
        print(f"[DONE] {topo} -> {dst_topo}")

    print("-" * 70)
    print(f"[SUMMARY] prepared={prepared} total={len(topologies)}")


if __name__ == "__main__":
    main()

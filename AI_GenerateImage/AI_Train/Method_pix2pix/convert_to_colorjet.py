import argparse
import pathlib
from PIL import Image
from tqdm import tqdm
from pix2pix_common import convert_bw_to_colorjet

def main():
    parser = argparse.ArgumentParser(description="Convert Black & White density map images to COLORJET format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing BW PNG images")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory (default: <input_dir>_colorjet)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    if args.output_dir:
        output_path = pathlib.Path(args.output_dir).resolve()
    else:
        output_path = input_path.parent / f"{input_path.name}_colorjet"

    output_path.mkdir(parents=True, exist_ok=True)

    png_files = sorted(list(input_path.glob("*.png")))
    if not png_files:
        print(f"No PNG files found in {input_path}")
        return

    print(f"🎨 Converting {len(png_files)} BW density maps to COLORJET...")
    print(f"📁 Source: {input_path}")
    print(f"📁 Target: {output_path}")

    for file_p in tqdm(png_files, desc="Converting"):
        img = Image.open(file_p).convert("L")
        colorjet_arr = convert_bw_to_colorjet(img)
        colorjet_img = Image.fromarray(colorjet_arr, mode="RGB")
        colorjet_img.save(output_path / file_p.name)

    print(f"✅ Conversion complete! Images saved to {output_path}")

if __name__ == "__main__":
    main()

import pathlib
from PIL import Image
from tqdm import tqdm
from pix2pix_wgangp_common import convert_bw_to_colorjet

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    bw_dir = script_dir / "test_results" / "bw"
    colorjet_dir = script_dir / "test_results" / "colorjet"
    colorjet_dir.mkdir(parents=True, exist_ok=True)

    if not bw_dir.exists():
        print(f"Directory not found: {bw_dir}")
        return

    files = sorted(list(bw_dir.glob("*.png")))
    print(f"🎨 Converting {len(files)} BW images to COLORJET...")

    for f in tqdm(files):
        img = Image.open(f).convert("L")
        rgb = convert_bw_to_colorjet(img)
        Image.fromarray(rgb).save(colorjet_dir / f.name)

    print("✅ Conversion complete!")

if __name__ == "__main__":
    main()

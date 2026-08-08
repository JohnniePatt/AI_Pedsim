import pathlib
import sys
import numpy as np
from PIL import Image
from resnet_common import convert_bw_to_colorjet

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent
    outputs_dir = project_root / "AI_Result" / "Method_ResNet" / "outputs"

    runs = sorted(list(outputs_dir.glob("run_ResNet_*")))
    if not runs:
        print("No run directories found.")
        return

    latest_run = runs[-1]
    bw_dir = latest_run / "test_results" / "bw"
    colorjet_dir = latest_run / "test_results" / "colorjet"
    colorjet_dir.mkdir(parents=True, exist_ok=True)

    bw_files = sorted(list(bw_dir.glob("*.png")))
    print(f"Converting {len(bw_files)} BW images to COLORJET in {colorjet_dir}...")

    for f in bw_files:
        img = Image.open(f).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        colorjet_np = convert_bw_to_colorjet(arr)
        colorjet_img = Image.fromarray(colorjet_np, mode="RGB")
        colorjet_img.save(colorjet_dir / f.name)

    print("✅ Conversion finished successfully!")

if __name__ == "__main__":
    main()

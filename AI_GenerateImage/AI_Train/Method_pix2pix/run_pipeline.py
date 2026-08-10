import subprocess
import sys
import pathlib

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    train_script = script_dir / "train_pix2pix_densitymap_bw.py"
    test_script = script_dir / "test_pix2pix_densitymap_bw.py"

    print("🚀 [PIPELINE] Starting Method_pix2pix Training...")
    res_train = subprocess.run([sys.executable, str(train_script)], check=True)

    print("\n🧪 [PIPELINE] Starting Method_pix2pix Testing...")
    res_test = subprocess.run([sys.executable, str(test_script)], check=True)

    print("\n🎉 [PIPELINE] Method_pix2pix Pipeline Execution Complete!")

if __name__ == "__main__":
    main()

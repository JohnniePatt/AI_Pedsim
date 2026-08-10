import subprocess
import sys
import pathlib

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    train_script = script_dir / "train_pix2pix_wgangp_densitymap_bw.py"
    test_script = script_dir / "test_pix2pix_wgangp_densitymap_bw.py"

    print("🚀 Starting Method_pix2pix_WGAN-GP Pipeline...")
    
    # Step 1: Train
    res_train = subprocess.run([sys.executable, str(train_script)])
    if res_train.returncode != 0:
        print("❌ Training failed.")
        sys.exit(res_train.returncode)

    # Step 2: Test
    res_test = subprocess.run([sys.executable, str(test_script)])
    if res_test.returncode != 0:
        print("❌ Testing failed.")
        sys.exit(res_test.returncode)

    print("🎉 Method_pix2pix_WGAN-GP Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()

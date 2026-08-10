import subprocess
import sys
import pathlib

def main():
    script_dir = pathlib.Path(__file__).parent.resolve()
    train_script = script_dir / "train_resnet_densitymap_bw.py"
    test_script = script_dir / "test_resnet_densitymap_bw.py"

    print("🚀 [PIPELINE] Starting Method_ResNet Training...")
    res_train = subprocess.run([sys.executable, str(train_script)])
    if res_train.returncode != 0:
        print("❌ Training failed.")
        sys.exit(res_train.returncode)

    print("🧪 [PIPELINE] Starting Method_ResNet Testing...")
    res_test = subprocess.run([sys.executable, str(test_script)])
    if res_test.returncode != 0:
        print("❌ Testing failed.")
        sys.exit(res_test.returncode)

    print("🎉 [PIPELINE] Method_ResNet Training & Testing Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/johnnie/programming/AI_Pedsim/AI_Pedsim")
DATASET_ROOT = PROJECT_ROOT / "Dataset/Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN"

# The 6 selected files
target_files = [
    'plan_141_4cca__45_03_full.png', 'plan_141_4cca__47_05_full.png', 'plan_334_da64__43_01_full.png',
    'plan_310_dab0__200042_00_single.png', 'plan_326_afe5__200042_00_single.png', 'plan_308_0588__200043_01_single.png'
]

# Paths
a_test = DATASET_ROOT / "A/test"
b_test = DATASET_ROOT / "B/test"
a_test_backup = DATASET_ROOT / "A/test_backup"
b_test_backup = DATASET_ROOT / "B/test_backup"

output_dir = PROJECT_ROOT / "AI_GenerateTrajectory/AI_Result/Failure_Case_Analysis"
output_dir.mkdir(parents=True, exist_ok=True)

def backup_and_prepare_dataset():
    print("Backing up dataset...")
    if not a_test_backup.exists():
        shutil.move(str(a_test), str(a_test_backup))
    if not b_test_backup.exists():
        shutil.move(str(b_test), str(b_test_backup))
        
    a_test.mkdir(exist_ok=True)
    b_test.mkdir(exist_ok=True)
    
    print("Copying target files...")
    for f in target_files:
        shutil.copy(a_test_backup / f, a_test / f)
        shutil.copy(b_test_backup / f, b_test / f)

def restore_dataset():
    print("Restoring dataset...")
    if a_test.exists(): shutil.rmtree(a_test)
    if b_test.exists(): shutil.rmtree(b_test)
    shutil.move(str(a_test_backup), str(a_test))
    shutil.move(str(b_test_backup), str(b_test))

def run_inference():
    commands = [
        {
            "name": "Plain U-Net",
            "dir": "Method_PlainUnet",
            "cmd": ["/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3", "test_PlainUnet_densitymap.py", "--run_path", "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_PlainUnet/outputs/run_PlainUNet_20260708_211818"],
            "pred_dir": "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_PlainUnet/outputs/run_PlainUNet_20260708_211818/test_results/best_loss/predictions"
        },
        {
            "name": "CVAE",
            "dir": "Method_CVAE",
            "cmd": ["/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3", "test_CVAE_densitymap_bw.py", "--run_path", "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/run_CVAE_20260627_193237_config2"],
            "pred_dir": "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/run_CVAE_20260627_193237_config2/test_results/best_mae/predictions"
        }
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    for c in commands:
        print(f"Running inference for {c['name']}...")
        work_dir = PROJECT_ROOT / "AI_GenerateTrajectory" / "AI_Train" / c["dir"]
        subprocess.run(c["cmd"], cwd=work_dir, env=env, check=False)
        
        dest_dir = output_dir / c["name"]
        dest_dir.mkdir(exist_ok=True)
        pred_path = Path(c["pred_dir"])
        for f in target_files:
            if (pred_path / f).exists():
                shutil.copy(pred_path / f, dest_dir / f)
            else:
                print(f"WARNING: File {f} missing in {pred_path}")

try:
    backup_and_prepare_dataset()
    run_inference()
finally:
    restore_dataset()
    print("Done!")

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/johnnie/programming/AI_Pedsim/AI_Pedsim")

commands = [
    # {
    #     "name": "pix2pixHD",
    #     "dir": "Method_pix2pixHD",
    #     "cmd": ["/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3", "test_pix2pixHD_densitymap_bw.py", "--run_path", "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_pix2pixHD/outputs/run_HD_20260517_133538_BestForBW", "--config", "config_test_03_bw.json"]
    # },
    {
        "name": "Plain U-Net",
        "dir": "Method_PlainUnet",
        "cmd": ["/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3", "test_PlainUnet_densitymap.py", "--run_path", "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_PlainUnet/outputs/run_PlainUNet_20260708_211818"]
    },
    # {
    #     "name": "pix2pixHD (No D)",
    #     "dir": "Method_pix2pixhd_No_D",
    #     "cmd": ["/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3", "test_pix2pixhd_NoD_densitymap_bw.py", "--run_path", "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_pix2pixhd_No_D/outputs/run_HD_NoD_20260709_180550", "--config", "config_test.json"]
    # },
    # {
    #     "name": "CVAE",
    #     "dir": "Method_CVAE",
    #     "cmd": ["/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3", "test_CVAE_densitymap_bw.py", "--run_path", "/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Result/Method_CVAE/outputs/run_CVAE_20260627_193237_config2"]
    # }
]

env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT)

for c in commands:
    print(f"Running full inference for {c['name']}...")
    work_dir = PROJECT_ROOT / "AI_GenerateTrajectory" / "AI_Train" / c["dir"]
    subprocess.run(c["cmd"], cwd=work_dir, env=env, check=False)

print("Done generating all predictions!")

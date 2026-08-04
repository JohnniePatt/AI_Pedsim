import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pathlib
import argparse
import json
from datetime import datetime

# --- Minimal Config for Testing ---
class TestConfig:
    image_size = 512
    input_channels = 3
    output_channels = 3
    l1_loss_weight = 10.0
    num_discriminators = 2
    mask_threshold = 0.5
    DATASET_ROOT = pathlib.Path(".")
    
    def __init__(self, run_path=None, config_file=None):
        self.SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
        self.CHECKPOINT_DIR = None
        
        # 1. Load from external config file (e.g. config_test.json)
        project_root = self.SCRIPT_DIR.parent.parent  # AI_GenerateImage
        if config_file:
            cf_path = pathlib.Path(config_file)
            if not cf_path.is_absolute() and not cf_path.exists():
                cf_path = self.SCRIPT_DIR / config_file
            if cf_path.exists() and cf_path.is_file():
                with open(cf_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k == "DATASET_ROOT":
                            p = pathlib.Path(v)
                            v = p if p.is_absolute() else (project_root / p).resolve()
                        if k == "checkpoints": self.CHECKPOINT_DIR = pathlib.Path(v)
                        setattr(self, k, v)
                print(f"📖 [CONFIG] Loaded test parameters from {cf_path}")

        # 2. Overwrite with specific RUN snapshot (Architectural parameters)
        # Handle the case where run_path is not provided or is invalid
        detected_run = None
        if run_path:
            rp = pathlib.Path(run_path)
            if not rp.is_absolute() and not rp.exists():
                rp = self.SCRIPT_DIR / run_path
            
            # Use RP if it's a directory AND has checkpoints, OR if it has .pth files directly
            if rp.exists() and rp.is_dir():
                has_ckpt_folder = (rp / "checkpoints").exists()
                has_pth_files = len(list(rp.glob("*.pth"))) > 0
                if has_ckpt_folder or has_pth_files:
                    detected_run = rp.resolve()
                else:
                    print(f"⚠️ [WARNING] '{rp.name}' exists but doesn't look like a valid run folder (no checkpoints found).")
            elif rp.exists() and rp.is_file():
                 if rp.suffix == ".pth":
                     self.CHECKPOINT_DIR = rp.resolve()
                     detected_run = rp.parent.resolve()
                     print(f"🎯 [DIRECT] Using specific checkpoint file: {rp.name}")
                 else:
                     print(f"⚠️ [WARNING] '--run_path' points to a file ({rp.name}). Searching for actual run folders instead...")
        
        if not detected_run and not self.CHECKPOINT_DIR:
            # Try to auto-locate the latest run in common locations
            search_roots = [self.SCRIPT_DIR / "runs_trajectory", self.SCRIPT_DIR / "outputs"]
            all_runs = []
            for root in search_roots:
                if root.exists():
                    all_runs.extend([d for d in root.iterdir() if d.is_dir()])
            
            if all_runs:
                # Sort by name (timestamped) to get the latest
                all_runs.sort(key=lambda x: x.name, reverse=True)
                detected_run = all_runs[0]
                print(f"🔦 [AUTO-DETECT] Using latest run folder: {detected_run}")
            else:
                # Last resort: check if the script directory itself looks like a run folder
                if (self.SCRIPT_DIR / "checkpoints").exists():
                    detected_run = self.SCRIPT_DIR
                    print(f"🔦 [AUTO-DETECT] Using current script directory as run (checkpoints found).")

        if detected_run:
            self.CURRENT_RUN_DIR = detected_run
            snap = self.CURRENT_RUN_DIR / "run_config_snapshot.json"
            if snap.exists():
                with open(snap, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k == "DATASET_ROOT": v = pathlib.Path(v)
                        # Don't overwrite if already set by config or direct path
                        if not hasattr(self, k) or (k == "checkpoints" and not self.CHECKPOINT_DIR):
                             setattr(self, k, v)
            
            if not self.CHECKPOINT_DIR:
                self.CHECKPOINT_DIR = self.CURRENT_RUN_DIR / "checkpoints"
                if not self.CHECKPOINT_DIR.exists():
                    self.CHECKPOINT_DIR = self.CURRENT_RUN_DIR # Fallback to run dir itself
            
            self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
            self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        else:
            if not self.CHECKPOINT_DIR:
                print("❌ [ERROR] Could not find a valid run directory with checkpoints.")
                self.CURRENT_RUN_DIR = pathlib.Path(".")
                self.CHECKPOINT_DIR = pathlib.Path("checkpoints")
            else:
                self.CURRENT_RUN_DIR = self.CHECKPOINT_DIR.parent
            
            self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
            self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Final Validation & Fallback Search
        if not hasattr(self, "DATASET_ROOT") or not (self.DATASET_ROOT / "A" / "test").exists():
            # Try to auto-locate common project structure
            project_root = self.SCRIPT_DIR.parent.parent
            search_paths = [
                project_root / "Model_scenario_case" / "Topo_bottleneck" / "trajectory_line_dataset" / "Cleandata_1",
                project_root / "Prepare_data" / "Topo_bottleneck" / "trajectory_line_dataset" / "Cleandata_1",
                project_root / "Topo_bottleneck" / "trajectory_line_dataset" / "Cleandata_1"
            ]
            for sp in search_paths:
                if (sp / "A" / "test").exists():
                    self.DATASET_ROOT = sp
                    print(f"🔦 [AUTO-DETECT] Dataset found at {sp}")
                    break

# --- Model & Dataset (Duplicated for standalone capability) ---
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.InstanceNorm2d(channels, affine=True)
        )
    def forward(self, x):
        return x + self.block(x)

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True)
        ]
        model += [
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.InstanceNorm2d(512, affine=True), nn.ReLU(True)
        ]
        for _ in range(n_blocks):
            model += [ResNetBlock(512)]
        model += [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
        ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)

class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="test", image_size=256):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        a_names = {f.name for f in self.directory_A.glob("*.png")}
        b_names = {f.name for f in self.directory_B.glob("*.png")}
        self.file_list = sorted(a_names & b_names)
        # Snap to multiple of 32 — must match training resize
        target = ((image_size + 31) // 32) * 32
        self.target_w = target
        self.target_h = target
        self.transform_a = transforms.Compose([
            transforms.Resize((self.target_h, self.target_w), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_b = transforms.Compose([
            transforms.Resize((self.target_h, self.target_w), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_a_raw = Image.open(self.directory_A / name).convert("RGB")
        img_b_raw = Image.open(self.directory_B / name).convert("RGB")
        orig_w, orig_h = img_a_raw.size
        return self.transform_a(img_a_raw), self.transform_b(img_b_raw), torch.tensor([orig_w, orig_h]), name

def run_evaluation(run_path, config_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] HD Evaluation on: {device_status}\n{'='*50}\n")
    config = TestConfig(run_path, config_file)
    print(f"🔍 [TEST] Evaluating Run: {config.CURRENT_RUN_DIR.name}")
    print(f"📂 [DATA] Dataset: {config.DATASET_ROOT}")
    print(f"⚙️ [CONFIG] Params: Size={config.image_size}, In/Out={config.input_channels}/{config.output_channels}")

    # Check if CHECKPOINT_DIR is actually a file
    if config.CHECKPOINT_DIR.is_file():
        best_ckpt = config.CHECKPOINT_DIR
    else:
        best_ckpt = config.CHECKPOINT_DIR / "generator_best.pth"
        if not best_ckpt.exists():
            # Try finding any checkpoint if generator_best.pth is not here
            checkpoints = sorted(list(config.CHECKPOINT_DIR.glob("*.pth")))
            if checkpoints:
                best_ckpt = checkpoints[-1]
                print(f"⚠️ [WARNING] generator_best.pth not found. Using latest: {best_ckpt.name}")
            else:
                print(f"❌ [ERROR] No checkpoint found at {config.CHECKPOINT_DIR}")
                return
            
    try:
        print(f"🔄 [MODEL] Loading weights from {best_ckpt.name}...")
        generator = GeneratorNetwork(int(config.input_channels), int(config.output_channels)).to(device)
        state_dict = torch.load(best_ckpt, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        generator.load_state_dict(state_dict)
        generator.eval()
        print("✅ [MODEL] Model loaded successfully.")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")
        return

    # Data
    test_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", int(config.image_size))
    if len(test_ds) == 0:
        abs_path = (config.DATASET_ROOT).resolve() / "A" / "test"
        print(f"\n❌ [ERROR] No test images found in: {abs_path}")
        print(f"💡 [ADVICE] Please update 'DATASET_ROOT' in config_test.json via the Dashboard's 'Testing model' page.")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Metrics
    mae_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    test_metrics = {"mae": 0.0, "mse": 0.0}

    print(f"🧪 [TEST] Running inference on {len(test_ds)} images...")
    
    # 📁 Prepare subdirectories for cleaner results
    pred_dir = config.TEST_RESULT_DIR / "predictions"
    input_dir = config.TEST_RESULT_DIR / "inputs"
    target_dir = config.TEST_RESULT_DIR / "targets"
    for d in [pred_dir, input_dir, target_dir]: d.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, (ta, tb, orig_size, file_name_batch) in enumerate(test_loader):
            ta, tb = ta.to(device), tb.to(device)
            tfb = generator(ta)
            
            test_metrics["mae"] += mae_criterion(tfb, tb).item()
            test_metrics["mse"] += mse_criterion(tfb, tb).item()
            
            # Save periodic samples (first 50)
            if i < 50:
                ow, oh = orig_size[0].tolist()
                file_name = str(file_name_batch[0])
                def denorm_and_finalize_image(x):
                    arr = ((x.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(arr).resize((int(ow), int(oh)), Image.LANCZOS)
                    return img

                def denorm_and_finalize_mask(x):
                    arr = ((x.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                    gray = Image.fromarray(arr).convert("L").resize((int(ow), int(oh)), Image.NEAREST)
                    mask = np.array(gray) >= int(float(getattr(config, "mask_threshold", 0.5)) * 255)
                    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")
                
                res_a = denorm_and_finalize_image(ta[0])
                res_b = denorm_and_finalize_mask(tb[0])
                res_f = denorm_and_finalize_mask(tfb[0])
                
                # 🖼️ Save files individually for better quality and UX
                # Keep original dataset filename for stable downstream matching (UI/jet lookup).
                res_f.save(pred_dir / file_name)
                res_a.save(input_dir / file_name)
                res_b.save(target_dir / file_name)

    # Scoring
    n_test = len(test_loader)
    if n_test == 0:
        print("❌ [ERROR] No images processed.")
        return
        
    mae_score = test_metrics["mae"] / n_test
    mse_score = test_metrics["mse"] / n_test
    rmse_score = np.sqrt(mse_score)

    score_path = config.CURRENT_RUN_DIR / "test_evaluation_summary.csv"
    with open(score_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"MAE (L1),{mae_score:.6f}\n")
        f.write(f"MSE,{mse_score:.6f}\n")
        f.write(f"RMSE,{rmse_score:.6f}\n")

    print(f"📊 [EVAL] Result: MAE: {mae_score:.4f} | RMSE: {rmse_score:.4f}")
    print(f"✅ [DONE] Evaluation results saved to {config.CURRENT_RUN_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True, help="Path to the training run folder (e.g. outputs/run_xxx)")
    parser.add_argument("--config", type=str, default="config_test.json", help="Path to testing config (e.g. config_test.json)")
    args = parser.parse_args()
    run_evaluation(args.run_path, args.config)

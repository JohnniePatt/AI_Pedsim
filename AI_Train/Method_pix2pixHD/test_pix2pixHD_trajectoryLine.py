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
    DATASET_ROOT = pathlib.Path(".")
    
    def __init__(self, run_path=None, config_file=None):
        # 1. Load from external config file (e.g. config_test.json)
        if config_file:
            cf_path = pathlib.Path(config_file)
            if not cf_path.is_absolute():
                cf_path = pathlib.Path(__file__).parent / config_file
            if cf_path.exists():
                with open(cf_path, "r") as f:
                    data = json.load(f)
                    for k, v in data.items(): 
                        if k == "DATASET_ROOT": v = pathlib.Path(v)
                        setattr(self, k, v)
                print(f"📖 [CONFIG] Loaded test parameters from {cf_path}")

        # 2. Overwrite with specific RUN snapshot (Architectural parameters)
        if run_path:
            self.CURRENT_RUN_DIR = pathlib.Path(run_path).resolve()
            snap = self.CURRENT_RUN_DIR / "run_config_snapshot.json"
            if snap.exists():
                with open(snap, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if k == "DATASET_ROOT": v = pathlib.Path(v)
                        setattr(self, k, v)
                # Re-fix paths
                self.CURRENT_RUN_DIR = pathlib.Path(run_path).resolve()
                if not hasattr(self, "DATASET_ROOT") or not self.DATASET_ROOT or str(self.DATASET_ROOT) == ".":
                    if "DATASET_ROOT" in data: self.DATASET_ROOT = pathlib.Path(data["DATASET_ROOT"])
            
            self.CHECKPOINT_DIR = self.CURRENT_RUN_DIR / "checkpoints"
            self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
            self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)

        # 3. Final Validation & Fallback Search
        if not (self.DATASET_ROOT / "A" / "test").exists():
            # Try to auto-locate common project structure
            project_root = pathlib.Path(__file__).parent.parent.parent
            search_paths = [
                project_root / "Model_scenario_case" / "Topo_2" / "trajectory_line_dataset" / "Cleandata_1",
                project_root / "Prepare_data" / "Topo_2" / "trajectory_line_dataset" / "Cleandata_1",
                project_root / "Topo_2" / "trajectory_line_dataset" / "Cleandata_1"
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
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ]
        model += [
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True)
        ]
        for _ in range(n_blocks):
            model += [ResNetBlock(512)]
        model += [
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
        ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)

class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="test", image_size=512):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        self.file_list = sorted([f.name for f in self.directory_A.glob("*.png")])
        self.image_size = image_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_a_raw = Image.open(self.directory_A / name).convert("RGB")
        img_b_raw = Image.open(self.directory_B / name).convert("RGB")
        
        orig_w, orig_h = img_a_raw.size
        # Dynamic calculation for this specific image
        target_w = ((orig_w + 31) // 32) * 32
        target_h = ((orig_h + 31) // 32) * 32
        
        # Resize to multiple of 32 for model stability
        img_a = img_a_raw.resize((target_w, target_h), Image.BICUBIC)
        img_b = img_b_raw.resize((target_w, target_h), Image.BICUBIC)
        
        return self.transforms(img_a), self.transforms(img_b), torch.tensor([orig_w, orig_h])

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

    # Load Model
    generator = GeneratorNetwork(int(config.input_channels), int(config.output_channels)).to(device)
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
        generator.load_state_dict(torch.load(best_ckpt, map_location=device))
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
    with torch.no_grad():
        for i, (ta, tb, orig_size) in enumerate(test_loader):
            ta, tb = ta.to(device), tb.to(device)
            tfb = generator(ta)
            
            test_metrics["mae"] += mae_criterion(tfb, tb).item()
            test_metrics["mse"] += mse_criterion(tfb, tb).item()
            
            if i < 20:
                ow, oh = orig_size[0].tolist()
                def denorm_and_finalize(x):
                    arr = ((x.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                    # Resize back to EXACT original size of THIS specific image
                    img = Image.fromarray(arr).resize((int(ow), int(oh)), Image.BICUBIC)
                    return np.array(img)
                res_a = denorm_and_finalize(ta[0])
                res_b = denorm_and_finalize(tb[0])
                res_f = denorm_and_finalize(tfb[0])
                combined = np.hstack([res_a, res_b, res_f])
                Image.fromarray(combined).save(config.TEST_RESULT_DIR / f"eval_sample_{i}.png")

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
    parser.add_argument("--run_path", type=str, required=True, help="Path to the training run folder (e.g. runs/run_xxx)")
    parser.add_argument("--config", type=str, default="config_test.json", help="Path to testing config (e.g. config_test.json)")
    args = parser.parse_args()
    run_evaluation(args.run_path, args.config)

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
import sys

# --- Minimal Config for Testing ---
class TestConfig:
    image_size = 512
    input_channels = 3
    output_channels = 3
    l1_loss_weight = 100.0
    
    def __init__(self, run_path=None):
        if run_path:
            self.CURRENT_RUN_DIR = pathlib.Path(run_path).resolve()
            snap = self.CURRENT_RUN_DIR / "run_config_snapshot.json"
            if snap.exists():
                with open(snap, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        setattr(self, k, v)
                self.CURRENT_RUN_DIR = pathlib.Path(run_path).resolve()
                self.DATASET_ROOT = pathlib.Path(data.get("DATASET_ROOT", ""))
            
            self.CHECKPOINT_DIR = self.CURRENT_RUN_DIR / "checkpoints"
            self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
            self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model & Dataset ---
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=True, use_dropout=False):
        super().__init__()
        if downsampling:
            self.operation = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.operation = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.use_dropout = use_dropout
        self.dropout_layer = nn.Dropout(0.5)
    def forward(self, x):
        x = self.operation(x)
        if self.use_dropout: x = self.dropout_layer(x)
        return x

class GeneratorNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.encoder2 = UNetBlock(64, 128, downsampling=True)
        self.encoder3 = UNetBlock(128, 256, downsampling=True)
        self.encoder4 = UNetBlock(256, 512, downsampling=True)
        self.encoder5 = UNetBlock(512, 512, downsampling=True)
        self.encoder6 = UNetBlock(512, 512, downsampling=True)
        self.encoder7 = UNetBlock(512, 512, downsampling=True)
        self.encoder8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        self.decoder1 = UNetBlock(512, 512, downsampling=False, use_dropout=True)
        self.decoder2 = UNetBlock(1024, 512, downsampling=False, use_dropout=True)
        self.decoder3 = UNetBlock(1024, 512, downsampling=False, use_dropout=True)
        self.decoder4 = UNetBlock(1024, 512, downsampling=False)
        self.decoder5 = UNetBlock(1024, 256, downsampling=False)
        self.decoder6 = UNetBlock(512, 128, downsampling=False)
        self.decoder7 = UNetBlock(256, 64, downsampling=False)
        self.decoder8 = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        d1 = self.encoder1(x); d2 = self.encoder2(d1); d3 = self.encoder3(d2); d4 = self.encoder4(d3)
        d5 = self.encoder5(d4); d6 = self.encoder6(d5); d7 = self.encoder7(d6); d8 = self.encoder8(d7)
        u1 = self.decoder1(d8)
        u2 = self.decoder2(torch.cat([u1, d7], 1)); u3 = self.decoder3(torch.cat([u2, d6], 1))
        u4 = self.decoder4(torch.cat([u3, d5], 1)); u5 = self.decoder5(torch.cat([u4, d4], 1))
        u6 = self.decoder6(torch.cat([u5, d3], 1)); u7 = self.decoder7(torch.cat([u6, d2], 1))
        return self.decoder8(torch.cat([u7, d1], 1))

class Pix2PixTrajectoryDataset(Dataset):
    def __init__(self, root_directory, subset="test", image_size=512):
        self.directory_A = pathlib.Path(root_directory) / "A" / subset
        self.directory_B = pathlib.Path(root_directory) / "B" / subset
        self.file_list = sorted([f.name for f in self.directory_A.glob("*.png")])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        name = self.file_list[idx]
        img_a_raw = Image.open(self.directory_A / name).convert("RGB")
        img_b_raw = Image.open(self.directory_B / name).convert("RGB")
        
        ow, oh = img_a_raw.size
        # Nearest multiple of 32 for model
        tw = ((ow + 31) // 32) * 32
        th = ((oh + 31) // 32) * 32
        
        img_a = img_a_raw.resize((tw, th), Image.BICUBIC)
        img_b = img_b_raw.resize((tw, th), Image.BICUBIC)
        
        return self.transforms(img_a), self.transforms(img_b), torch.tensor([ow, oh])

def run_evaluation(run_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 🕵️ Device Reporting
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_status = f"🚀 GPU: {device_name}" if device.type == "cuda" else "💻 CPU"
    print(f"\n{'='*50}\n🛰️ [SYSTEM] Evaluation on: {device_status}\n{'='*50}\n")
    config = TestConfig(run_path)
    
    print(f"🔍 [TEST] Evaluating Run: {config.CURRENT_RUN_DIR.name}")

    generator = GeneratorNetwork(int(config.input_channels), int(config.output_channels)).to(device)
    best_ckpt = config.CHECKPOINT_DIR / "generator_best.pth"
    if not best_ckpt.exists():
        print(f"❌ [ERROR] No checkpoint found at {best_ckpt}"); return
    generator.load_state_dict(torch.load(best_ckpt, map_location=device))
    generator.eval()

    test_ds = Pix2PixTrajectoryDataset(config.DATASET_ROOT, "test", int(config.image_size))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    mae_criterion = nn.L1Loss(); mse_criterion = nn.MSELoss()
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
                    # Resize back to EXACT original size
                    img = Image.fromarray(arr).resize((int(ow), int(oh)), Image.BICUBIC)
                    return np.array(img)
                res_a = denorm_and_finalize(ta[0])
                res_b = denorm_and_finalize(tb[0])
                res_f = denorm_and_finalize(tfb[0])
                combined = np.hstack([res_a, res_b, res_f])
                Image.fromarray(combined).save(config.TEST_RESULT_DIR / f"eval_sample_{i}.png")

    n_test = len(test_loader)
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
    parser.add_argument("--run_path", type=str, required=True, help="Path to run folder")
    args = parser.parse_args()
    run_evaluation(args.run_path)

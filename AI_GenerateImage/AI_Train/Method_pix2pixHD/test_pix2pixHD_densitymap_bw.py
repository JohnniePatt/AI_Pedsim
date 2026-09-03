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
import math
import csv
import time
from datetime import datetime, timezone

try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    LPIPS_AVAILABLE = False

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
    def __init__(self, in_channels=3, out_channels=3, n_blocks=9, decoder_mode="deconv"):
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
        if decoder_mode == "upsample":
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
        else:
            model += [
                nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(256, affine=True), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(128, affine=True), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(64, affine=True), nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x): return self.model(x)



def _compute_ssim_global(img_pred_01, img_true_01):
    # Global SSIM approximation over full image (stable and dependency-free).
    x = img_pred_01.astype(np.float64)
    y = img_true_01.astype(np.float64)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = ((x - mu_x) ** 2).mean()
    sigma_y2 = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)
    if den == 0:
        return 1.0
    return float(num / den)


def _to_bw_01(tensor):
    # BW density maps are stored as RGB copies, but the metric should compare
    # one scalar density field rather than RGB luminance/color channels.
    return ((tensor.mean(dim=1, keepdim=True) * 0.5) + 0.5).clamp(0.0, 1.0)


def _resolve_dataset_from_config_file(config_path, project_root):
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        raw = data.get("DATASET_ROOT") or data.get("dataset_root")
        if not raw:
            return None
        path = pathlib.Path(str(raw))
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path
    except Exception:
        return None


def _compute_segmentation_metrics(gray_pred_01, gray_true_01, threshold):
    pred_mask = gray_pred_01 >= threshold
    true_mask = gray_true_01 >= threshold

    tp = np.logical_and(pred_mask, true_mask).sum(dtype=np.float64)
    fp = np.logical_and(pred_mask, ~true_mask).sum(dtype=np.float64)
    fn = np.logical_and(~pred_mask, true_mask).sum(dtype=np.float64)
    tn = np.logical_and(~pred_mask, ~true_mask).sum(dtype=np.float64)

    eps = 1e-8
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return float(dice), float(iou), float(pixel_acc)


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
        source_a_path = self.directory_A / name
        source_b_path = self.directory_B / name
        img_a_raw = Image.open(source_a_path).convert("RGB")
        img_b_raw = Image.open(source_b_path).convert("RGB")
        orig_w, orig_h = img_a_raw.size
        return (
            self.transform_a(img_a_raw),
            self.transform_b(img_b_raw),
            torch.tensor([orig_w, orig_h]),
            name,
            str(source_b_path),
        )

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
    archived_datasets = []
    for cfg_name in ("config_train_03_bw.json", "config_train_03.json", "config_train.json"):
        cfg_path = config.CURRENT_RUN_DIR / cfg_name
        if cfg_path.exists():
            ds_path = _resolve_dataset_from_config_file(cfg_path, config.SCRIPT_DIR.parent.parent)
            if ds_path is not None:
                archived_datasets.append((cfg_name, ds_path))

    test_dataset = pathlib.Path(config.DATASET_ROOT).resolve()
    archive_matches = any(ds.resolve() == test_dataset for _, ds in archived_datasets)

    snap_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    if snap_path.exists():
        try:
            with open(snap_path, "r", encoding="utf-8-sig") as f:
                snap_data = json.load(f)
            snap_dataset = pathlib.Path(str(snap_data.get("DATASET_ROOT", "")))
            if str(snap_dataset) and snap_dataset.resolve() != test_dataset and not archive_matches:
                print(
                    "⚠️ [WARNING] Run snapshot dataset does not match this test dataset.\n"
                    f"   trained/snapshot DATASET_ROOT: {snap_dataset}\n"
                    f"   testing DATASET_ROOT         : {config.DATASET_ROOT}\n"
                    "   This usually means the checkpoint was trained on a different target domain."
                )
            elif str(snap_dataset) and snap_dataset.resolve() != test_dataset and archive_matches:
                print(
                    "ℹ️ [INFO] Run snapshot dataset is stale, but archived train config matches this BW test dataset."
                )
        except Exception as e:
            print(f"⚠️ [WARNING] Could not inspect run snapshot dataset: {e}")

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
            
    print(f"🔄 [MODEL] Loading weights from {best_ckpt.name}...")
    state_dict = torch.load(best_ckpt, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    generator = None
    load_errors = {}
    for mode in ("deconv", "upsample"):
        try:
            candidate = GeneratorNetwork(
                int(config.input_channels), int(config.output_channels), decoder_mode=mode
            ).to(device)
            candidate.load_state_dict(state_dict)
            generator = candidate
            print(f"✅ [MODEL] Model loaded successfully. decoder_mode={mode}")
            break
        except Exception as e:
            load_errors[mode] = str(e)

    if generator is None:
        print("❌ [ERROR] Failed to load model with both decoder modes.")
        for mode, err in load_errors.items():
            print(f"   - {mode}: {err}")
        return

    generator.eval()

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
    test_metrics = {
        "mae": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "ssim": 0.0,
        "psnr": 0.0,
        "lpips": 0.0,
    }
    per_image_metrics = []
    metrics_wall_time_s = 0.0
    time_generate_s = 0.0

    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_model.eval()
            print("[METRIC] LPIPS enabled (alex)")
        except Exception as e:
            print(f"[WARN] LPIPS unavailable at runtime: {e}")
            lpips_model = None
    else:
        print("[WARN] lpips package not installed. LPIPS will be NaN.")

    print(f"🧪 [TEST] Running inference on {len(test_ds)} images...")
    
    # 📁 Prepare subdirectories for cleaner results
    pred_dir = config.TEST_RESULT_DIR / "predictions"
    input_dir = config.TEST_RESULT_DIR / "inputs"
    target_dir = config.TEST_RESULT_DIR / "targets"
    for d in [pred_dir, input_dir, target_dir]: d.mkdir(parents=True, exist_ok=True)
    for d in [pred_dir, input_dir, target_dir]:
        for old_png in d.glob("*.png"):
            old_png.unlink()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started = time.perf_counter()
    with torch.no_grad():
        for i, (ta, tb, orig_size, file_name_batch, source_b_path_batch) in enumerate(test_loader):
            ta, tb = ta.to(device), tb.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generate_started = time.perf_counter()
            tfb = generator(ta)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            time_generate_s += time.perf_counter() - generate_started
            tfb_bw = _to_bw_01(tfb)
            tb_bw = _to_bw_01(tb)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics_started = time.perf_counter()
            test_metrics["mae"] += mae_criterion(tfb_bw, tb_bw).item()
            mse_val = mse_criterion(tfb_bw, tb_bw).item()
            test_metrics["mse"] += mse_val

            pred_01 = tfb_bw[0, 0].detach().cpu().numpy()
            true_01 = tb_bw[0, 0].detach().cpu().numpy()

            ssim_val = _compute_ssim_global(pred_01, true_01)
            test_metrics["ssim"] += ssim_val

            mse_255 = np.mean(((pred_01 * 255.0) - (true_01 * 255.0)) ** 2)
            psnr_val = 100.0 if mse_255 <= 1e-12 else float(10.0 * np.log10((255.0 ** 2) / mse_255))
            test_metrics["psnr"] += psnr_val

            rmse_val = float(np.sqrt(mse_val))
            test_metrics["rmse"] += rmse_val

            lpips_val = float("nan")
            if lpips_model is not None:
                try:
                    lpips_pred = (tfb_bw * 2.0 - 1.0).repeat(1, 3, 1, 1)
                    lpips_true = (tb_bw * 2.0 - 1.0).repeat(1, 3, 1, 1)
                    lpips_val = float(lpips_model(lpips_pred, lpips_true).mean().item())
                    test_metrics["lpips"] += lpips_val
                except Exception as e:
                    print(f"[WARN] LPIPS compute failed: {e}")
                    lpips_val = float("nan")

            file_name = str(file_name_batch[0])
            per_image_metrics.append({
                "file_name": file_name,
                "mae": float(np.mean(np.abs(pred_01 - true_01))),
                "mse": float(mse_val),
                "rmse": float(rmse_val),
                "ssim": float(ssim_val),
                "psnr": float(psnr_val),
                "lpips": float(lpips_val),
            })
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics_wall_time_s += time.perf_counter() - metrics_started
            
            # Save all samples
            if True:
                ow, oh = orig_size[0].tolist()
                file_name = str(file_name_batch[0])
                def denorm_and_finalize_image(x):
                    arr = ((x.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                    img = Image.fromarray(arr).resize((int(ow), int(oh)), Image.LANCZOS)
                    return img

                def denorm_and_finalize_density_bw(x):
                    arr = ((x.detach().cpu().numpy() * 0.5 + 0.5) * 255).clip(0, 255)
                    gray_arr = arr.mean(axis=0).astype(np.uint8)
                    gray = Image.fromarray(gray_arr, mode="L").resize((int(ow), int(oh)), Image.LANCZOS)
                    return gray.convert("RGB")
                
                res_a = denorm_and_finalize_image(ta[0])
                res_b = Image.open(str(source_b_path_batch[0])).convert("RGB")
                res_f = denorm_and_finalize_density_bw(tfb[0])
                
                # 🖼️ Save files individually for better quality and UX
                # Keep original dataset filename for stable downstream matching (UI/jet lookup).
                res_f.save(pred_dir / file_name)
                res_a.save(input_dir / file_name)
                res_b.save(target_dir / file_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_wall_time_s = time.perf_counter() - test_started
    runtime_excluding_metrics_s = max(0.0, test_wall_time_s - metrics_wall_time_s)

    runtime_row = {
        "method_id": "Method_pix2pixHD",
        "split": "test",
        "timing_scope": "test_loop_including_data_inference_metrics_postprocess_and_image_write",
        "sample_count": len(per_image_metrics),
        "Time Generate": f"{time_generate_s:.6f}",
        "Average Time Generate Per Image": (
            f"{time_generate_s / len(per_image_metrics):.9f}" if per_image_metrics else "nan"
        ),
        "test_pipeline_wall_time_s": f"{test_wall_time_s:.6f}",
        "metrics_wall_time_s": f"{metrics_wall_time_s:.6f}",
        "runtime_excluding_metrics_s": f"{runtime_excluding_metrics_s:.6f}",
        "mean_runtime_excluding_metrics_per_image_s": (
            f"{runtime_excluding_metrics_s / len(per_image_metrics):.9f}" if per_image_metrics else "nan"
        ),
        "device_type": device.type,
        "device_name": device_name,
        "checkpoint_path": str(best_ckpt.resolve()),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(config.CURRENT_RUN_DIR / "test_runtime.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=runtime_row.keys())
        writer.writeheader()
        writer.writerow(runtime_row)

    # Scoring
    n_test = len(test_loader)
    if n_test == 0:
        print("❌ [ERROR] No images processed.")
        return
    mae_score = test_metrics["mae"] / n_test
    mse_score = test_metrics["mse"] / n_test
    rmse_score = test_metrics["rmse"] / n_test
    ssim_score = test_metrics["ssim"] / n_test
    psnr_score = test_metrics["psnr"] / n_test
    lpips_vals = [m["lpips"] for m in per_image_metrics if not math.isnan(m["lpips"])]
    lpips_score = (sum(lpips_vals) / len(lpips_vals)) if lpips_vals else float("nan")

    per_image_path = config.CURRENT_RUN_DIR / "test_evaluation_per_image.csv"
    with open(per_image_path, "w", encoding="utf-8") as f:
        f.write("file_name,MAE,MSE,RMSE,SSIM,PSNR,LPIPS\n")
        for row in per_image_metrics:
            lpips_txt = "nan" if math.isnan(row["lpips"]) else f"{row['lpips']:.6f}"
            f.write(
                f"{row['file_name']},{row['mae']:.6f},{row['mse']:.6f},{row['rmse']:.6f},{row['ssim']:.6f},"
                f"{row['psnr']:.6f},{lpips_txt}\n"
            )

    score_path = config.CURRENT_RUN_DIR / "test_evaluation_summary.csv"
    with open(score_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"MAE,{mae_score:.6f}\n")
        f.write(f"MSE,{mse_score:.6f}\n")
        f.write(f"RMSE,{rmse_score:.6f}\n")
        f.write(f"SSIM,{ssim_score:.6f}\n")
        f.write(f"PSNR,{psnr_score:.6f}\n")
        if math.isnan(lpips_score):
            f.write("LPIPS,nan\n")
        else:
            f.write(f"LPIPS,{lpips_score:.6f}\n")

    lpips_txt = "nan" if math.isnan(lpips_score) else f"{lpips_score:.4f}"
    print(
        f"?? [EVAL] MAE={mae_score:.4f} | MSE={mse_score:.4f} | RMSE={rmse_score:.4f} | "
        f"SSIM={ssim_score:.4f} | PSNR={psnr_score:.2f} | LPIPS={lpips_txt}"
    )
    print(f"? [DONE] Evaluation results saved to {config.CURRENT_RUN_DIR}")
    print(f"✅ [DONE] Evaluation results saved to {config.CURRENT_RUN_DIR}")
    print(f"⏱️ [RUNTIME] Full test pipeline: {test_wall_time_s:.3f} s ({len(per_image_metrics)} samples)")
    print(f"⏱️ [RUNTIME] Time Generate:      {time_generate_s:.3f} s")
    print(f"⏱️ [RUNTIME] Excluding metrics:  {runtime_excluding_metrics_s:.3f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True, help="Path to the training run folder (e.g. outputs/run_xxx)")
    parser.add_argument("--config", type=str, default="config_test_03_bw.json", help="Path to testing config (e.g. config_test_03_bw.json)")
    args = parser.parse_args()
    run_evaluation(args.run_path, args.config)

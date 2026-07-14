import csv
import json
import pathlib
from datetime import datetime


def resolve_project_root(script_dir: pathlib.Path) -> pathlib.Path:
    return script_dir.parent.parent


def resolve_path(path_value: str | pathlib.Path, project_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_json_config(config_path: pathlib.Path) -> dict:
    if not config_path or not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_train_config(config_path: pathlib.Path) -> dict:
    script_dir = pathlib.Path(__file__).parent.resolve()
    project_root = resolve_project_root(script_dir)
    cfg = {
        "epochs": 50,
        "batch_size": 8,
        "learning_rate": 2e-4,
        "image_size": 256,
        "latent_dim": 32,
        "base_filters": 32,
        "dropout": 0.1,
        "use_cudnn": False,
        "target_representation": "bw",
        "target_channels": 1,
        "metric_mode": "density_scalar",
        "l1_loss_weight": 1.0,
        "mse_loss_weight": 0.25,
        "edge_loss_weight": 0.5,
        "density_foreground_weight": 30.0,
        "density_intensity_weight": 10.0,
        "density_foreground_threshold": 1.0 / 255.0,
        "foreground_l1_loss_weight": 0.0,
        "mass_loss_weight": 0.0,
        "gamma_l1_loss_weight": 0.0,
        "density_gamma_loss": 1.0,
        "train_latent_mode": "posterior",
        "kl_weight": 0.01,
        "kl_anneal_epochs": 10,
        "sample_count": 4,
        "sample_every_epochs": 1,
        "checkpoint_every_epochs": 10,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "num_workers": 4,
        "resume_checkpoint_path": "-",
        "run_test_after_train": True,
        "test_checkpoint_modes": ["best_mae", "best_loss"],
        "dataset_root": "../Dataset/Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN",
    }
    cfg.update(load_json_config(config_path))
    cfg["image_size"] = int(((int(cfg["image_size"]) + 31) // 32) * 32)
    cfg["target_channels"] = int(cfg["target_channels"])
    cfg["BASE_DIR"] = str(script_dir)
    cfg["PROJECT_ROOT"] = str(project_root)
    cfg["DATASET_ROOT"] = str(resolve_path(cfg["dataset_root"], project_root))
    return cfg


def make_run_dirs(base_dir: pathlib.Path, method_name: str = "Method_CVAE") -> dict[str, pathlib.Path]:
    project_root = resolve_project_root(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_CVAE_{timestamp}"
    runs_root = project_root / "AI_Result" / method_name / "outputs"
    current_run_dir = runs_root / run_name
    paths = {
        "PROJECT_ROOT": project_root,
        "RUNS_ROOT": runs_root,
        "CURRENT_RUN_DIR": current_run_dir,
        "CHECKPOINT_DIR": current_run_dir / "checkpoints",
        "LOG_DIR": current_run_dir / "logs",
        "SAMPLE_DIR": current_run_dir / "samples",
        "TEST_RESULT_DIR": current_run_dir / "test_results",
        "FINAL_EVALUATION_DIR": current_run_dir / "final_evaluation",
    }
    for key in ("CHECKPOINT_DIR", "LOG_DIR", "SAMPLE_DIR", "TEST_RESULT_DIR", "FINAL_EVALUATION_DIR"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def save_run_snapshot(cfg: dict, config_path: pathlib.Path, run_dirs: dict[str, pathlib.Path]):
    current_run_dir = run_dirs["CURRENT_RUN_DIR"]
    snapshot = dict(cfg)
    snapshot.update({k: str(v) for k, v in run_dirs.items()})
    snapshot["run_name"] = current_run_dir.name
    snapshot["framework"] = "pytorch"
    snapshot["task"] = "density_map_cvae"
    with open(current_run_dir / "run_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4)
    if config_path.exists():
        import shutil
        shutil.copy2(config_path, current_run_dir / config_path.name)
    return snapshot


def write_progress(path: pathlib.Path, epoch: int, total_epochs: int, loss: float, val_total: float):
    data = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "percentage": round((epoch / max(total_epochs, 1)) * 100, 2),
        "loss": round(float(loss), 6),
        "val_total": round(float(val_total), 6),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_summary_csv(path: pathlib.Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class TestConfig:
    def __init__(self, run_path, config_file=None):
        script_dir = pathlib.Path(__file__).parent.resolve()
        project_root = resolve_project_root(script_dir)
        self.SCRIPT_DIR = script_dir
        self.PROJECT_ROOT = project_root
        self.image_size = 256
        self.latent_dim = 32
        self.base_filters = 32
        self.dropout = 0.1
        self.use_cudnn = False
        self.target_representation = "bw"
        self.target_channels = 1
        self.metric_mode = "density_scalar"
        self.num_samples = 1
        self.DATASET_ROOT = resolve_path("../Dataset/Data_ImageUNet/DensityMap_dataset/Topo_HouseGAN", project_root)

        rp = pathlib.Path(run_path)
        if not rp.is_absolute():
            rp = pathlib.Path.cwd() / rp
            if not rp.exists():
                rp = script_dir / run_path
        if not rp.exists():
            raise FileNotFoundError(f"Run path not found: {rp}")
        self.CURRENT_RUN_DIR = rp.resolve()

        snapshot = self.CURRENT_RUN_DIR / "run_config_snapshot.json"
        if snapshot.exists():
            data = load_json_config(snapshot)
            for key, value in data.items():
                if key.endswith("DIR") or key in {"DATASET_ROOT", "PROJECT_ROOT", "BASE_DIR", "RUNS_ROOT", "CURRENT_RUN_DIR", "run_name"}:
                    continue
                setattr(self, key, value)

        if config_file:
            cf = pathlib.Path(config_file)
            if not cf.is_absolute():
                cf = pathlib.Path.cwd() / cf
                if not cf.exists():
                    cf = script_dir / config_file
            if cf.exists():
                data = load_json_config(cf)
                for key, value in data.items():
                    if key in {"DATASET_ROOT", "dataset_root"}:
                        value = resolve_path(value, project_root)
                        key = "DATASET_ROOT"
                    setattr(self, key, value)

        self.target_channels = int(getattr(self, "target_channels", 1))
        self.CHECKPOINT_DIR = self.CURRENT_RUN_DIR / "checkpoints"
        self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
        self.FINAL_EVALUATION_DIR = self.CURRENT_RUN_DIR / "final_evaluation"
        self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self.FINAL_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

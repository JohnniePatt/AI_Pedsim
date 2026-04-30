import json
import os
import pathlib
from datetime import datetime


class TrainingConfiguration:
    # Optimization
    epochs = 80
    batch_size = 4
    learning_rate = 2e-4

    # CVAE
    latent_dim = 32
    base_filters = 48
    kl_weight = 0.02
    kl_anneal_epochs = 12

    # Reconstruction and structure losses
    l1_loss_weight = 10.0
    mask_bce_loss_weight = 1.0
    mask_dice_loss_weight = 1.0
    edge_loss_weight = 2.0
    mask_threshold = 0.08
    dice_smooth = 1e-6

    # Data
    image_size = 256
    input_channels = 3
    output_channels = 3
    dataset_root = ""

    # Runtime
    seed = 42
    sample_count = 4
    sample_every_epochs = 1
    checkpoint_every_epochs = 10
    resume_checkpoint_dir = "-"

    # Paths
    BASE_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = BASE_DIR.parent.parent
    DATASET_ROOT = PROJECT_ROOT / "Topo_bottleneck" / "trajectory_line_dataset" / "Cleandata_1"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_CVAE_{timestamp}"
    METHOD_NAME = BASE_DIR.name
    RUNS_ROOT = PROJECT_ROOT / "AI_Result" / METHOD_NAME / "outputs"
    CURRENT_RUN_DIR = RUNS_ROOT / run_name

    CHECKPOINT_DIR = CURRENT_RUN_DIR / "checkpoints"
    LOG_DIR = CURRENT_RUN_DIR / "logs"
    SAMPLE_DIR = CURRENT_RUN_DIR / "samples"
    TEST_RESULT_DIR = CURRENT_RUN_DIR / "test_results"
    FINAL_EVALUATION_DIR = CURRENT_RUN_DIR / "final_evaluation"


class TestConfig:
    image_size = 256
    latent_dim = 32
    base_filters = 48
    DATASET_ROOT = pathlib.Path(".")

    def __init__(self, run_path=None, config_file=None):
        self.SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
        self.CHECKPOINT_DIR = None
        configured_keys = set()

        project_root = self.SCRIPT_DIR.parent.parent

        if config_file:
            cf_path = pathlib.Path(config_file)
            if not cf_path.is_absolute() and not cf_path.exists():
                cf_path = self.SCRIPT_DIR / config_file
            if cf_path.exists() and cf_path.is_file():
                with open(cf_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    configured_keys.add(k)
                    if k == "DATASET_ROOT":
                        p = pathlib.Path(v)
                        v = p if p.is_absolute() else (project_root / p).resolve()
                    if k == "checkpoints":
                        self.CHECKPOINT_DIR = pathlib.Path(v)
                    setattr(self, k, v)
                print(f"[CONFIG] Loaded test parameters from {cf_path}")

        detected_run = None
        if run_path:
            rp = pathlib.Path(run_path)
            if not rp.is_absolute() and not rp.exists():
                rp = self.SCRIPT_DIR / run_path
            if rp.exists() and rp.is_dir():
                has_ckpt_folder = (rp / "checkpoints").exists()
                has_weight_files = len(list(rp.rglob("*best.weights.h5"))) > 0
                if has_ckpt_folder or has_weight_files:
                    detected_run = rp.resolve()

        if not detected_run:
            outputs_root = self.SCRIPT_DIR / "outputs"
            if outputs_root.exists():
                all_runs = sorted([d for d in outputs_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
                if all_runs:
                    detected_run = all_runs[0]

        if not detected_run:
            raise RuntimeError("No valid run directory found for evaluation.")

        self.CURRENT_RUN_DIR = detected_run
        snap = self.CURRENT_RUN_DIR / "run_config_snapshot.json"
        if snap.exists():
            with open(snap, "r", encoding="utf-8") as f:
                data = json.load(f)
            path_keys = {
                "DATASET_ROOT",
                "BASE_DIR",
                "PROJECT_ROOT",
                "RUNS_ROOT",
                "CURRENT_RUN_DIR",
                "CHECKPOINT_DIR",
                "LOG_DIR",
                "SAMPLE_DIR",
                "TEST_RESULT_DIR",
                "FINAL_EVALUATION_DIR",
            }
            for k, v in data.items():
                # Always keep DATASET_ROOT from config_test when explicitly provided,
                # but force architecture/runtime keys from run snapshot so checkpoints match.
                if k == "DATASET_ROOT" and k in configured_keys:
                    continue
                if k in path_keys and isinstance(v, str):
                    v = pathlib.Path(v)
                setattr(self, k, v)

        if isinstance(self.CURRENT_RUN_DIR, str):
            self.CURRENT_RUN_DIR = pathlib.Path(self.CURRENT_RUN_DIR)

        self.CHECKPOINT_DIR = self.CURRENT_RUN_DIR / "checkpoints"
        self.TEST_RESULT_DIR = self.CURRENT_RUN_DIR / "test_results"
        self.FINAL_EVALUATION_DIR = self.CURRENT_RUN_DIR / "final_evaluation"
        self.TEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        self.FINAL_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def resolve_input_path(path_str, script_dir):
    if not path_str:
        return None

    p = pathlib.Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)

    cwd_candidate = pathlib.Path.cwd() / p
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    script_candidate = pathlib.Path(script_dir) / p
    if script_candidate.exists():
        return str(script_candidate.resolve())

    return None


def load_train_config_from_json(config, json_path):
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if config.dataset_root:
        p = pathlib.Path(config.dataset_root)
        if not p.is_absolute():
            p = config.PROJECT_ROOT / p
        config.DATASET_ROOT = p.resolve()

    config.image_size = int(((int(config.image_size) + 31) // 32) * 32)

    print(f"[CONFIG] Loaded parameters from {json_path}")
    print(f"[CONFIG] DATASET_ROOT = {config.DATASET_ROOT}")
    print(
        "[CONFIG] image_size={} | batch={} | latent_dim={} | kl_weight={}"
        .format(config.image_size, config.batch_size, config.latent_dim, config.kl_weight)
    )


def write_progress(config, epoch, total_epochs, loss, val_total):
    progress_file = config.CURRENT_RUN_DIR / "progress.json"
    data = {
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "percentage": round(((epoch + 1) / total_epochs) * 100, 2),
        "loss": round(float(loss), 6),
        "val_l1": round(float(val_total), 6),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_run_snapshot(config):
    for d in [config.CHECKPOINT_DIR, config.LOG_DIR, config.SAMPLE_DIR, config.TEST_RESULT_DIR, config.FINAL_EVALUATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    run_config_path = config.CURRENT_RUN_DIR / "run_config_snapshot.json"
    config_dict = {
        k: str(v) if isinstance(v, pathlib.Path) else v
        for k, v in config.__class__.__dict__.items()
        if not k.startswith("__")
    }
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    print(f"[CONFIG] Run snapshot saved to {run_config_path}")

    raw_config_path = config.BASE_DIR / "config_train.json"
    if not raw_config_path.exists():
        raw_config_path = config.BASE_DIR / "config_active.json"
    if raw_config_path.exists():
        import shutil

        shutil.copy(raw_config_path, config.CURRENT_RUN_DIR / raw_config_path.name)
        print(f"[ARCHIVE] {raw_config_path.name} copied to {config.CURRENT_RUN_DIR}")

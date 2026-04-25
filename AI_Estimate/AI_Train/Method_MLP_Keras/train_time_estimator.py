import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Runtime CUDA libs bootstrap (for TensorFlow GPU in venv)
# ---------------------------------------------------------------------------

def _append_unique_path(existing, new_paths):
    parts = [p for p in (existing or "").split(":") if p]
    seen = set(parts)
    for path in new_paths:
        if path and path not in seen:
            parts.append(path)
            seen.add(path)
    return ":".join(parts)


def _bootstrap_tf_cuda_ld_library_path():
    candidate_venvs = []
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        candidate_venvs.append(Path(venv_env))
    candidate_venvs.append(Path(sys.executable).resolve().parents[1])
    candidate_venvs.append(Path(__file__).resolve().parents[4] / "AI_Pedsim-env")

    nvidia_root = None
    for venv_root in candidate_venvs:
        if not venv_root:
            continue
        lib_root = venv_root / "lib"
        if not lib_root.exists():
            continue
        for py_dir in lib_root.glob("python*"):
            candidate = py_dir / "site-packages" / "nvidia"
            if candidate.exists():
                nvidia_root = candidate
                break
        if nvidia_root:
            break

    if not nvidia_root:
        return
    lib_dirs = []
    for name in [
        "cuda_runtime",
        "cudnn",
        "cublas",
        "cufft",
        "curand",
        "cusolver",
        "cusparse",
        "nccl",
        "nvjitlink",
    ]:
        lib_dir = nvidia_root / name / "lib"
        if lib_dir.exists():
            lib_dirs.append(str(lib_dir))
    os.environ["LD_LIBRARY_PATH"] = _append_unique_path(os.environ.get("LD_LIBRARY_PATH", ""), lib_dirs)
    _preload_cuda_libs(lib_dirs)
    _set_xla_cuda_data_dir(nvidia_root)


def _preload_cuda_libs(lib_dirs):
    patterns = [
        "libcudart.so*",
        "libcublas.so*",
        "libcublasLt.so*",
        "libcudnn.so*",
        "libcudnn_*.so*",
        "libcusolver.so*",
        "libcusparse.so*",
        "libcurand.so*",
        "libnccl.so*",
        "libnvJitLink.so*",
    ]
    loaded = set()
    for lib_dir in lib_dirs:
        path_obj = Path(lib_dir)
        if not path_obj.exists():
            continue
        for pattern in patterns:
            for so_path in sorted(path_obj.glob(pattern)):
                key = so_path.name
                if key in loaded:
                    continue
                try:
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                    loaded.add(key)
                except OSError:
                    pass


def _set_xla_cuda_data_dir(nvidia_root):
    nvvm_dir = Path(nvidia_root) / "cuda_nvcc"
    if not nvvm_dir.exists():
        return
    xla_flags = os.environ.get("XLA_FLAGS", "")
    token = "--xla_gpu_cuda_data_dir="
    if token in xla_flags:
        return
    extra = f"{token}{nvvm_dir}"
    os.environ["XLA_FLAGS"] = f"{xla_flags} {extra}".strip()


_bootstrap_tf_cuda_ld_library_path()

# ---------------------------------------------------------------------------
# DONE then Import Keras
# ---------------------------------------------------------------------------

import keras

from dataset_keras import (
    build_data_bundle,
    inverse_target_transform,
    ordered_time_predictions,
    read_json,
    write_json,
)
from model import build_model


def print_system_status(model_type="MLP (Keras)"):
    import platform
    import psutil
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    device_type = f"GPU ({len(gpus)})" if gpus else "CPU"
    
    print("-" * 60)
    print(f"🚀 [AI_Estimate] Hardware & Model Status")
    print(f"   • Model Type: {model_type}")
    print(f"   • Processor : {platform.processor()}")
    print(f"   • CPUs      : {psutil.cpu_count(logical=True)} logical cores")
    print(f"   • RAM       : {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
    print(f"   • Device    : {device_type}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def _resolve_output_root(config, config_path):
    raw_root = config.get("output", {}).get("root", "../../AI_result/Method_MLP_Keras/outputs")
    path = Path(raw_root)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def _make_run_dir(config, config_path):
    output_root = _resolve_output_root(config, config_path)
    run_name = config.get("output", {}).get("run_name", "auto")
    if run_name == "auto":
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Metrics (in real seconds)
# ---------------------------------------------------------------------------

def _compute_metrics(pred_scaled, true_scaled, scaler, target_columns):
    pred_s = ordered_time_predictions(inverse_target_transform(pred_scaled, scaler))
    true_s = inverse_target_transform(true_scaled, scaler)
    error = pred_s - true_s
    metrics = {
        "rows": int(len(pred_s)),
        "mae_overall_s": float(np.mean(np.abs(error))),
        "rmse_overall_s": float(np.sqrt(np.mean(error ** 2))),
    }
    for idx, name in enumerate(target_columns):
        metrics[f"mae_{name}"]  = float(np.mean(np.abs(error[:, idx])))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(error[:, idx] ** 2)))
    return metrics


# ---------------------------------------------------------------------------
# Custom callback: real-seconds MAE per epoch
# ---------------------------------------------------------------------------

class RealSecondsMAECallback(keras.callbacks.Callback):
    """Appends val_mae_overall_s / train_mae_overall_s (real seconds) to logs each epoch."""

    def __init__(self, x_train, y_train, x_val, y_val, scaler, target_columns):
        super().__init__()
        self._x_train = x_train
        self._y_train = y_train
        self._x_val   = x_val
        self._y_val   = y_val
        self._scaler  = scaler
        self._target_columns = target_columns

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Only evaluate on validation set for speed
        pred = self.model.predict(self._x_val, verbose=0)
        pred_s = ordered_time_predictions(inverse_target_transform(pred, self._scaler))
        true_s = inverse_target_transform(self._y_val, self._scaler)
        error = pred_s - true_s
        logs["val_mae_overall_s"]  = float(np.mean(np.abs(error)))
        logs["val_rmse_overall_s"] = float(np.sqrt(np.mean(error ** 2)))
        logs["train_mae_overall_s"] = 0.0 # Placeholder for history CSV consistency


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path):
    config_path = Path(config_path).resolve()
    config      = read_json(config_path)
    train_cfg   = config.get("train", {})

    batch_size              = int(train_cfg.get("batch_size", 64))
    epochs                  = int(train_cfg.get("epochs", 200))
    learning_rate           = float(train_cfg.get("learning_rate", 0.001))
    weight_decay            = float(train_cfg.get("weight_decay", 0.0001))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 25))

    run_dir = _make_run_dir(config, config_path)
    bundle  = build_data_bundle(config, config_path, batch_size=batch_size)
    
    print_system_status()

    # Build model
    model = build_model(len(bundle.feature_columns), config)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.summary()

    best_checkpoint_path = str(run_dir / "best_result.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
        RealSecondsMAECallback(
            bundle.x_train, bundle.y_train,
            bundle.x_val,   bundle.y_val,
            bundle.scaler,  bundle.target_columns,
        ),
    ]

    print(f"[AI_Estimate][Keras][Train] run={run_dir}")
    print(f" - Train samples: {len(bundle.train_df)}")
    print(f" - Val samples: {len(bundle.val_df)}")
    print(f" - Test samples: {len(bundle.test_df)}")
    print("-" * 30, flush=True)

    history = model.fit(
        bundle.train_ds,
        validation_data=bundle.val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # Save last-epoch model
    model.save(run_dir / "last_result.keras")

    # Save training history (includes real-seconds MAE from callback)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df = history_df.rename(columns={"loss": "train_loss", "mae": "train_mae_scaled"})
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    # Save config + dataset info
    write_json(run_dir / "config_used.json", config)
    write_json(run_dir / "dataset_manifest.json", {
        "rows":          int(len(bundle.dataframe)),
        "train_rows":    int(len(bundle.train_df)),
        "val_rows":      int(len(bundle.val_df)),
        "test_rows":     int(len(bundle.test_df)),
        "train_plans":   len(bundle.splits["train"]),
        "val_plans":     len(bundle.splits["val"]),
        "test_plans":    len(bundle.splits["test"]),
        "feature_columns": bundle.feature_columns,
        "target_columns":  bundle.target_columns,
        "splits":          bundle.splits,
    })
    write_json(run_dir / "model_bundle.json", {
        "feature_columns": bundle.feature_columns,
        "target_columns":  bundle.target_columns,
        "scaler":          bundle.scaler,
    })
    bundle.dataframe.head(500).to_csv(run_dir / "dataset_preview.csv", index=False)

    # Compute final metrics from best checkpoint
    best_model = keras.models.load_model(best_checkpoint_path)
    train_pred = best_model.predict(bundle.x_train, verbose=0)
    val_pred   = best_model.predict(bundle.x_val,   verbose=0)
    test_pred  = best_model.predict(bundle.x_test,  verbose=0)

    train_metrics = _compute_metrics(train_pred, bundle.y_train, bundle.scaler, bundle.target_columns)
    val_metrics   = _compute_metrics(val_pred,   bundle.y_val,   bundle.scaler, bundle.target_columns)
    test_metrics  = _compute_metrics(test_pred,  bundle.y_test,  bundle.scaler, bundle.target_columns)

    write_json(run_dir / "metrics.json", {
        "best": {"train": train_metrics, "val": val_metrics},
        "final_test": test_metrics,
    })
    print(f"[AI_Estimate][Keras][Train] best_val_mae={val_metrics['mae_overall_s']:.3f}s")
    print(f"[AI_Estimate][Keras][Train] final_test_mae={test_metrics['mae_overall_s']:.3f}s")
    return run_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI_Estimate time estimator (Keras).")
    parser.add_argument("--config", default="AI_Estimate/AI_Train/Method_MLP_Keras/config_train.json")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

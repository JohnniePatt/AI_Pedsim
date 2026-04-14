import argparse
import ctypes
import json
import os
import sys
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

import keras

from dataset_keras import (
    build_data_bundle,
    inverse_target_transform,
    ordered_time_predictions,
    read_json,
    write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_output_root(config, config_path):
    raw_root = config.get("output", {}).get("root", "../../AI_result/Method_MLP_Keras/outputs")
    path = Path(raw_root)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def _latest_checkpoint(output_root):
    output_root = Path(output_root)
    candidates = sorted(output_root.glob("run_*/best_result.keras"))
    if not candidates:
        candidates = sorted(output_root.glob("*/best_result.keras"))
    if not candidates:
        raise FileNotFoundError(f"No best_result.keras found under {output_root}")
    return candidates[-1]


def _compute_metrics(pred_s, true_s, target_columns):
    error = pred_s - true_s
    metrics = {
        "rows":           int(len(pred_s)),
        "mae_overall_s":  float(np.mean(np.abs(error))),
        "rmse_overall_s": float(np.sqrt(np.mean(error ** 2))),
    }
    for idx, name in enumerate(target_columns):
        metrics[f"mae_{name}"]  = float(np.mean(np.abs(error[:, idx])))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(error[:, idx] ** 2)))
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test(config_path, checkpoint_path=None, output_dir=None):
    config_path = Path(config_path).resolve()
    config      = read_json(config_path)

    output_root     = _resolve_output_root(config, config_path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else _latest_checkpoint(output_root)
    run_dir         = checkpoint_path.parent

    bundle_meta = read_json(run_dir / "model_bundle.json")
    bundle      = build_data_bundle(config, config_path)
    model       = keras.models.load_model(checkpoint_path)

    pred_scaled = model.predict(bundle.x_test, verbose=0)
    pred_s      = ordered_time_predictions(inverse_target_transform(pred_scaled, bundle_meta["scaler"]))
    true_s      = inverse_target_transform(bundle.y_test, bundle_meta["scaler"])
    metrics     = _compute_metrics(pred_s, true_s, bundle_meta["target_columns"])

    output_dir = Path(output_dir) if output_dir else run_dir / "test_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = bundle.test_df.reset_index(drop=True).copy()
    for idx, target in enumerate(bundle_meta["target_columns"]):
        result_df[f"true_{target}"]      = true_s[:, idx]
        result_df[f"pred_{target}"]      = pred_s[:, idx]
        result_df[f"abs_error_{target}"] = np.abs(pred_s[:, idx] - true_s[:, idx])
    result_df.to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "test_metrics.json", metrics)

    print(f"[AI_Estimate][Keras][Test] checkpoint={checkpoint_path}")
    print(json.dumps(metrics, indent=2))
    return output_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AI_Estimate time estimator (Keras).")
    parser.add_argument("--config",     default="AI_Estimate/AI_Train/Method_MLP_Keras/config_train.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    test(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()

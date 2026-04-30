import argparse
import math
import re

import numpy as np
import tensorflow as tf
import h5py

from cvae_config import TestConfig
from cvae_data import list_test_pairs, load_image
from cvae_io import denorm_to_pil
from cvae_model import CVAEInference


def _build_inference_model(image_size, base_filters, latent_dim):
    cvae = CVAEInference(int(image_size), int(base_filters), int(latent_dim))
    dummy = tf.zeros((1, int(image_size), int(image_size), 3), dtype=tf.float32)
    _ = cvae.predict(dummy)
    return cvae


def _extract_base_filters_from_load_error(error):
    # Typical mismatch message:
    # variable.shape=(4, 4, 3, 48), Received: value.shape=(4, 4, 3, 40)
    msg = str(error)
    matched = re.search(r"value\.shape=\(\s*4,\s*4,\s*3,\s*(\d+)\s*\)", msg)
    if not matched:
        return None
    try:
        return int(matched.group(1))
    except (TypeError, ValueError):
        return None


def _walk_h5_shapes(path):
    shapes = []
    with h5py.File(path, "r") as f:
        def _visit(_, obj):
            if hasattr(obj, "shape"):
                shapes.append(tuple(int(v) for v in obj.shape))

        f.visititems(_visit)
    return shapes


def _infer_checkpoint_architecture(cond_weights, decoder_weights, fallback_image_size, fallback_base_filters, latent_dim):
    image_size = int(fallback_image_size)
    base_filters = int(fallback_base_filters)

    for shape in _walk_h5_shapes(cond_weights):
        if len(shape) == 4 and shape[0] == 4 and shape[1] == 4 and shape[2] == 3:
            base_filters = int(shape[3])
            break

    dense_units = None
    for shape in _walk_h5_shapes(decoder_weights):
        if len(shape) == 2 and shape[0] == int(latent_dim):
            dense_units = int(shape[1])
            break

    if dense_units and base_filters > 0:
        bottleneck_area = dense_units / float(base_filters * 4)
        bottleneck_side = int(round(math.sqrt(bottleneck_area)))
        if bottleneck_side > 0 and bottleneck_side * bottleneck_side == int(round(bottleneck_area)):
            image_size = bottleneck_side * 32

    return image_size, base_filters


def run_evaluation(run_path, config_file=None):
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        print(f"[SYSTEM] Evaluation on GPU: {gpu_devices[0].name}")
    else:
        print("[SYSTEM] Evaluation on CPU")

    cfg = TestConfig(run_path, config_file)
    print(f"[TEST] Evaluating run: {cfg.CURRENT_RUN_DIR.name}")
    print(f"[DATA] Dataset: {cfg.DATASET_ROOT}")

    cond_w = cfg.CHECKPOINT_DIR / "cond_encoder_best.weights.h5"
    dec_w = cfg.CHECKPOINT_DIR / "decoder_best.weights.h5"
    if not cond_w.exists() or not dec_w.exists():
        raise FileNotFoundError(
            f"Missing best checkpoints in {cfg.CHECKPOINT_DIR}. "
            "Expected cond_encoder_best.weights.h5 and decoder_best.weights.h5"
        )

    ckpt_image_size, ckpt_base_filters = _infer_checkpoint_architecture(
        cond_w, dec_w, cfg.image_size, cfg.base_filters, cfg.latent_dim
    )
    if ckpt_image_size != int(cfg.image_size) or ckpt_base_filters != int(cfg.base_filters):
        print(
            "[WARN] Checkpoint architecture differs from config/snapshot. "
            "Using checkpoint values: image_size={} | base_filters={}".format(
                ckpt_image_size, ckpt_base_filters
            )
        )
    cfg.image_size = ckpt_image_size
    cfg.base_filters = ckpt_base_filters

    cvae = _build_inference_model(int(cfg.image_size), int(cfg.base_filters), int(cfg.latent_dim))
    print(
        "[MODEL] Build config: image_size={} | base_filters={} | latent_dim={}".format(
            int(cfg.image_size), int(cfg.base_filters), int(cfg.latent_dim)
        )
    )

    try:
        cvae.cond_encoder.load_weights(str(cond_w))
        cvae.decoder.load_weights(str(dec_w))
    except ValueError as e:
        inferred_filters = _extract_base_filters_from_load_error(e)
        if inferred_filters and inferred_filters != int(cfg.base_filters):
            print(
                "[WARN] Checkpoint base_filters={} does not match config/snapshot={}. "
                "Retrying with checkpoint value.".format(inferred_filters, int(cfg.base_filters))
            )
            cvae = _build_inference_model(int(cfg.image_size), inferred_filters, int(cfg.latent_dim))
            cvae.cond_encoder.load_weights(str(cond_w))
            cvae.decoder.load_weights(str(dec_w))
            print(
                "[MODEL] Rebuilt config: image_size={} | base_filters={} | latent_dim={}".format(
                    int(cfg.image_size), int(inferred_filters), int(cfg.latent_dim)
                )
            )
        else:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                "Please evaluate with a run folder that matches its own snapshot/checkpoints."
            ) from e
    print("[MODEL] Loaded best CVAE checkpoints")

    dir_a, dir_b, pair_files = list_test_pairs(cfg.DATASET_ROOT)
    if len(pair_files) == 0:
        raise RuntimeError(f"No test images found in {dir_a}")

    pred_dir = cfg.TEST_RESULT_DIR / "predictions"
    input_dir = cfg.TEST_RESULT_DIR / "inputs"
    target_dir = cfg.TEST_RESULT_DIR / "targets"
    final_pred_dir = cfg.FINAL_EVALUATION_DIR / "predictions"
    final_input_dir = cfg.FINAL_EVALUATION_DIR / "inputs"
    final_target_dir = cfg.FINAL_EVALUATION_DIR / "targets"
    for d in [pred_dir, input_dir, target_dir, final_pred_dir, final_input_dir, final_target_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mae_sum = 0.0
    mse_sum = 0.0

    for idx, path_a in enumerate(pair_files):
        path_b = dir_b / path_a.name

        img_a, ow, oh = load_image(path_a, int(cfg.image_size), method="bicubic")
        img_b, _, _ = load_image(path_b, int(cfg.image_size), method="nearest")

        pred = cvae.predict(img_a[None, ...], z=None)[0]

        mae_sum += float(tf.reduce_mean(tf.abs(pred - img_b)).numpy())
        mse_sum += float(tf.reduce_mean(tf.square(pred - img_b)).numpy())

        if idx < 50:
            pil_a = denorm_to_pil(img_a.numpy(), ow, oh)
            pil_b = denorm_to_pil(img_b.numpy(), ow, oh)
            pil_p = denorm_to_pil(pred.numpy(), ow, oh)

            pil_a.save(input_dir / f"input_{idx}.png")
            pil_b.save(target_dir / f"target_{idx}.png")
            pil_p.save(pred_dir / f"result_{idx}.png")
            pil_a.save(final_input_dir / f"input_{idx}.png")
            pil_b.save(final_target_dir / f"target_{idx}.png")
            pil_p.save(final_pred_dir / f"result_{idx}.png")

    n = max(1, len(pair_files))
    mae = mae_sum / n
    mse = mse_sum / n
    rmse = float(np.sqrt(mse))

    score_path = cfg.CURRENT_RUN_DIR / "test_evaluation_summary.csv"
    with open(score_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"MAE (L1),{mae:.6f}\n")
        f.write(f"MSE,{mse:.6f}\n")
        f.write(f"RMSE,{rmse:.6f}\n")
    final_score_path = cfg.FINAL_EVALUATION_DIR / "test_evaluation_summary.csv"
    with open(final_score_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"MAE (L1),{mae:.6f}\n")
        f.write(f"MSE,{mse:.6f}\n")
        f.write(f"RMSE,{rmse:.6f}\n")

    print(f"[EVAL] MAE={mae:.4f} | RMSE={rmse:.4f}")
    print(f"[DONE] Evaluation results saved to {cfg.CURRENT_RUN_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True, help="Path to run folder")
    parser.add_argument("--config", type=str, default="config_test.json", help="Test config path")
    args = parser.parse_args()

    run_evaluation(args.run_path, args.config)

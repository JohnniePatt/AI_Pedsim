import argparse
import os
import pathlib
import subprocess
import sys

import numpy as np
from tqdm import tqdm

import tensorflow as tf

from cvae_config import (
    TrainingConfiguration,
    load_train_config_from_json,
    resolve_input_path,
    save_run_snapshot,
    write_progress,
)
from cvae_data import make_dataset
from cvae_io import save_triptych_sample
from cvae_losses import LossComputer
from cvae_model import CVAE


config = TrainingConfiguration()


def execute_training():
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 60)
    print(f"[SYSTEM] TensorFlow version: {tf.__version__}")
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        print(f"[SYSTEM] Training on GPU: {gpu_devices[0].name}")
    else:
        print("[SYSTEM] Training on CPU")
    print("=" * 60)

    save_run_snapshot(config)
    write_progress(config, -1, config.epochs, 0.0, 0.0)

    train_ds, train_pairs = make_dataset(
        config.DATASET_ROOT,
        "train",
        int(config.batch_size),
        int(config.image_size),
        shuffle=True,
        seed=int(config.seed),
    )
    val_ds, val_pairs = make_dataset(
        config.DATASET_ROOT,
        "validation",
        int(config.batch_size),
        int(config.image_size),
        shuffle=False,
        seed=int(config.seed),
    )
    _test_ds, test_pairs = make_dataset(
        config.DATASET_ROOT,
        "test",
        int(config.batch_size),
        int(config.image_size),
        shuffle=False,
        seed=int(config.seed),
    )

    if len(train_pairs) == 0 or len(val_pairs) == 0 or len(test_pairs) == 0:
        raise RuntimeError("[DATASET] One of train/validation/test splits is empty.")

    cvae = CVAE(int(config.image_size), int(config.base_filters), int(config.latent_dim))
    loss_comp = LossComputer(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=float(config.learning_rate))

    for batch_a, batch_b in train_ds.take(1):
        _ = cvae.forward_train(batch_a, batch_b, training=False)

    if config.resume_checkpoint_dir not in ["-", "", None]:
        ckpt_dir = pathlib.Path(config.resume_checkpoint_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = (config.BASE_DIR / ckpt_dir).resolve()
        cond_w = ckpt_dir / "cond_encoder_best.weights.h5"
        post_w = ckpt_dir / "posterior_encoder_best.weights.h5"
        dec_w = ckpt_dir / "decoder_best.weights.h5"
        if cond_w.exists() and post_w.exists() and dec_w.exists():
            cvae.cond_encoder.load_weights(str(cond_w))
            cvae.posterior_encoder.load_weights(str(post_w))
            cvae.decoder.load_weights(str(dec_w))
            print(f"[RESUME] Loaded weights from {ckpt_dir}")

    log_hist_path = config.LOG_DIR / "training_history.csv"
    with open(log_hist_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_total,train_l1,train_bce,train_dice,train_edge,train_kl,"
            "val_total,val_l1_raw,val_l1,val_bce,val_dice,val_edge,val_kl,kl_weight\n"
        )

    fixed_val_samples = []
    for va, vb in val_ds.unbatch().take(int(config.sample_count)):
        fixed_val_samples.append((va.numpy(), vb.numpy()))

    best_val = float("inf")
    train_steps = max(1, len(train_pairs) // int(config.batch_size) + int(len(train_pairs) % int(config.batch_size) > 0))

    for epoch in range(int(config.epochs)):
        kl_w = float(config.kl_weight)
        if int(config.kl_anneal_epochs) > 0:
            kl_w = float(config.kl_weight) * min(1.0, (epoch + 1) / float(config.kl_anneal_epochs))

        train_metrics = {"total": 0.0, "l1": 0.0, "bce": 0.0, "dice": 0.0, "edge": 0.0, "kl": 0.0}

        pbar = tqdm(train_ds, total=train_steps, desc=f"E{epoch}", leave=False)
        for batch_a, batch_b in pbar:
            with tf.GradientTape() as tape:
                pred_b, mu, logvar = cvae.forward_train(batch_a, batch_b, training=True)
                loss_total, loss_l1_raw, loss_bce, loss_dice, loss_edge, loss_kl = loss_comp.compute(
                    pred_b, batch_b, mu, logvar, kl_w
                )

            grads = tape.gradient(loss_total, cvae.trainable_variables)
            optimizer.apply_gradients(zip(grads, cvae.trainable_variables))

            train_metrics["total"] += float(loss_total.numpy())
            train_metrics["l1"] += float(loss_l1_raw.numpy())
            train_metrics["bce"] += float(loss_bce.numpy())
            train_metrics["dice"] += float(loss_dice.numpy())
            train_metrics["edge"] += float(loss_edge.numpy())
            train_metrics["kl"] += float(loss_kl.numpy())

            pbar.set_postfix(
                total=f"{float(loss_total.numpy()):.4f}",
                l1=f"{float(loss_l1_raw.numpy()):.4f}",
                bce=f"{float(loss_bce.numpy()):.4f}",
                dice=f"{float(loss_dice.numpy()):.4f}",
                edge=f"{float(loss_edge.numpy()):.4f}",
                kl=f"{float(loss_kl.numpy()):.4f}",
            )

        train_batches = max(1, train_steps)
        train_avg_total = train_metrics["total"] / train_batches
        train_avg_l1 = train_metrics["l1"] / train_batches
        train_avg_bce = train_metrics["bce"] / train_batches
        train_avg_dice = train_metrics["dice"] / train_batches
        train_avg_edge = train_metrics["edge"] / train_batches
        train_avg_kl = train_metrics["kl"] / train_batches

        val_metrics = {"total": 0.0, "l1_raw": 0.0, "bce": 0.0, "dice": 0.0, "edge": 0.0, "kl": 0.0}
        val_steps = 0

        for batch_a, batch_b in val_ds:
            pred_b = cvae.forward_infer(batch_a, z=None, training=False)
            posterior_input = tf.concat([batch_a, batch_b], axis=-1)
            mu, logvar = cvae.posterior_encoder(posterior_input, training=False)

            _, loss_l1_raw, loss_bce, loss_dice, loss_edge, loss_kl = loss_comp.compute(
                pred_b, batch_b, mu, logvar, kl_w
            )

            loss_total = (
                config.l1_loss_weight * loss_l1_raw
                + config.mask_bce_loss_weight * loss_bce
                + config.mask_dice_loss_weight * loss_dice
                + config.edge_loss_weight * loss_edge
                + kl_w * loss_kl
            )

            val_metrics["total"] += float(loss_total.numpy())
            val_metrics["l1_raw"] += float(loss_l1_raw.numpy())
            val_metrics["bce"] += float(loss_bce.numpy())
            val_metrics["dice"] += float(loss_dice.numpy())
            val_metrics["edge"] += float(loss_edge.numpy())
            val_metrics["kl"] += float(loss_kl.numpy())
            val_steps += 1

        val_steps = max(1, val_steps)
        val_total = val_metrics["total"] / val_steps
        val_l1_raw = val_metrics["l1_raw"] / val_steps
        val_l1 = val_l1_raw * float(config.l1_loss_weight)
        val_bce = val_metrics["bce"] / val_steps
        val_dice = val_metrics["dice"] / val_steps
        val_edge = val_metrics["edge"] / val_steps
        val_kl = val_metrics["kl"] / val_steps

        with open(log_hist_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_avg_total:.6f},{train_avg_l1:.6f},{train_avg_bce:.6f},{train_avg_dice:.6f},{train_avg_edge:.6f},{train_avg_kl:.6f},"
                f"{val_total:.6f},{val_l1_raw:.6f},{val_l1:.6f},{val_bce:.6f},{val_dice:.6f},{val_edge:.6f},{val_kl:.6f},{kl_w:.6f}\n"
            )

        print(
            f"[EPOCH {epoch}] train_total={train_avg_total:.4f} | "
            f"train_l1/bce/dice/edge/kl={train_avg_l1:.4f}/{train_avg_bce:.4f}/{train_avg_dice:.4f}/{train_avg_edge:.4f}/{train_avg_kl:.4f} | "
            f"val_total={val_total:.4f} | "
            f"val_l1raw/l1/bce/dice/edge/kl={val_l1_raw:.4f}/{val_l1:.4f}/{val_bce:.4f}/{val_dice:.4f}/{val_edge:.4f}/{val_kl:.4f} | "
            f"kl_w={kl_w:.4f}"
        )

        write_progress(config, epoch, int(config.epochs), train_avg_total, val_total)

        if (epoch + 1) % int(config.sample_every_epochs) == 0 and fixed_val_samples:
            for i, (sample_a, sample_b) in enumerate(fixed_val_samples):
                sample_a_batched = tf.convert_to_tensor(sample_a[None, ...], dtype=tf.float32)
                pred = cvae.forward_infer(sample_a_batched, z=None, training=False)[0].numpy()
                out_file = config.SAMPLE_DIR / f"epoch_{epoch+1:04d}_sample_{i:02d}.png"
                save_triptych_sample(sample_a, pred, sample_b, out_file)

        if val_total < best_val:
            best_val = val_total
            cvae.cond_encoder.save_weights(str(config.CHECKPOINT_DIR / "cond_encoder_best.weights.h5"))
            cvae.posterior_encoder.save_weights(str(config.CHECKPOINT_DIR / "posterior_encoder_best.weights.h5"))
            cvae.decoder.save_weights(str(config.CHECKPOINT_DIR / "decoder_best.weights.h5"))
            print(f"  New best model (val_total={best_val:.4f})")

        if (epoch + 1) % int(config.checkpoint_every_epochs) == 0:
            cvae.cond_encoder.save_weights(str(config.CHECKPOINT_DIR / f"cond_encoder_epoch_{epoch+1}.weights.h5"))
            cvae.posterior_encoder.save_weights(str(config.CHECKPOINT_DIR / f"posterior_encoder_epoch_{epoch+1}.weights.h5"))
            cvae.decoder.save_weights(str(config.CHECKPOINT_DIR / f"decoder_epoch_{epoch+1}.weights.h5"))

    print("\n--- Triggering Standalone CVAE Test Evaluation ---")
    test_script = pathlib.Path(__file__).parent / "test_cvae_trajectoryLine.py"
    test_proc = subprocess.run([sys.executable, str(test_script), "--run_path", str(config.CURRENT_RUN_DIR)])
    if test_proc.returncode == 0:
        print(f"Training finished. Results in {config.CURRENT_RUN_DIR}")
    else:
        print(
            f"[WARN] Training finished but standalone test evaluation failed "
            f"(exit code {test_proc.returncode}). Run test script manually after fixing."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.json")
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    cpath = resolve_input_path(args.config, script_dir)
    if cpath:
        load_train_config_from_json(config, cpath)
    else:
        fallback = resolve_input_path("config_active.json", script_dir)
        if fallback:
            print(f"[WARN] Config not found: {args.config}. Falling back to {fallback}")
            load_train_config_from_json(config, fallback)
        else:
            print("[WARN] No config file found. Using in-script defaults.")

    execute_training()

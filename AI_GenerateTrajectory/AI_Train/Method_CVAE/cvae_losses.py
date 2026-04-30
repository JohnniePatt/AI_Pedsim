import tensorflow as tf


def grayscale_01(image_m11):
    image_01 = (image_m11 + 1.0) * 0.5
    r = image_01[..., 0:1]
    g = image_01[..., 1:2]
    b = image_01[..., 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def hard_mask_from_gray(gray, threshold):
    return tf.cast(gray >= threshold, tf.float32)


def soft_mask_from_gray(gray, threshold):
    denom = tf.maximum(1.0 - threshold, 1e-6)
    return tf.clip_by_value((gray - threshold) / denom, 0.0, 1.0)


def dice_loss(pred_prob, target_mask, smooth):
    pred = tf.reshape(pred_prob, [tf.shape(pred_prob)[0], -1])
    target = tf.reshape(target_mask, [tf.shape(target_mask)[0], -1])
    intersection = tf.reduce_sum(pred * target, axis=1)
    denom = tf.reduce_sum(pred, axis=1) + tf.reduce_sum(target, axis=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


def sobel_edge_map(gray):
    # shape: [N, H, W, 1, 2] -> gx, gy
    sobel = tf.image.sobel_edges(gray)
    gx = sobel[..., 0]
    gy = sobel[..., 1]
    return tf.sqrt(tf.maximum(gx * gx + gy * gy, 1e-8))


class LossComputer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def compute(self, pred_b, real_b, mu, logvar, kl_weight):
        loss_l1_raw = tf.reduce_mean(tf.abs(pred_b - real_b))

        real_gray = grayscale_01(real_b)
        pred_gray = grayscale_01(pred_b)

        gt_mask = hard_mask_from_gray(real_gray, self.cfg.mask_threshold)
        pred_prob = soft_mask_from_gray(pred_gray, self.cfg.mask_threshold)

        loss_bce = self.bce(gt_mask, pred_prob)
        loss_dice = dice_loss(pred_prob, gt_mask, self.cfg.dice_smooth)
        loss_edge = tf.reduce_mean(tf.abs(sobel_edge_map(pred_gray) - sobel_edge_map(real_gray)))

        loss_kl = -0.5 * tf.reduce_mean(1.0 + logvar - tf.square(mu) - tf.exp(logvar))

        loss_total = (
            self.cfg.l1_loss_weight * loss_l1_raw
            + self.cfg.mask_bce_loss_weight * loss_bce
            + self.cfg.mask_dice_loss_weight * loss_dice
            + self.cfg.edge_loss_weight * loss_edge
            + kl_weight * loss_kl
        )
        return loss_total, loss_l1_raw, loss_bce, loss_dice, loss_edge, loss_kl

import torch
from torch.nn import functional as F


def dice_from_probs(targets, probs, smooth=1e-6):
    targets = targets.float().reshape(targets.shape[0], -1)
    probs = probs.float().reshape(probs.shape[0], -1)
    intersection = (targets * probs).sum(dim=1)
    denom = targets.sum(dim=1) + probs.sum(dim=1)
    return ((2.0 * intersection + smooth) / (denom + smooth)).mean()


def iou_from_probs(targets, probs, smooth=1e-6):
    targets = targets.float().reshape(targets.shape[0], -1)
    probs = probs.float().reshape(probs.shape[0], -1)
    intersection = (targets * probs).sum(dim=1)
    union = targets.sum(dim=1) + probs.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


def _sobel_kernels(device, dtype):
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return gx, gy


def sobel_edge_map(x):
    gx, gy = _sobel_kernels(x.device, x.dtype)
    px = F.conv2d(x, gx, padding=1)
    py = F.conv2d(x, gy, padding=1)
    return torch.sqrt(torch.clamp(px * px + py * py, min=1e-8))


class LossComputer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.pos_weight = torch.tensor([float(cfg.get("foreground_weight", 8.0))], device=device)

    def compute(self, logits, real_b, mu, logvar, kl_weight):
        pred_prob = torch.sigmoid(logits)
        loss_l1_raw = torch.mean(torch.abs(pred_prob - real_b))
        loss_bce = F.binary_cross_entropy_with_logits(logits, real_b, pos_weight=self.pos_weight)
        loss_dice_value = 1.0 - dice_from_probs(real_b, pred_prob)
        loss_edge = torch.mean(torch.abs(sobel_edge_map(pred_prob) - sobel_edge_map(real_b)))
        loss_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss_total = (
            float(self.cfg.get("l1_loss_weight", 0.5)) * loss_l1_raw
            + float(self.cfg.get("mask_bce_loss_weight", 0.5)) * loss_bce
            + float(self.cfg.get("mask_dice_loss_weight", 2.0)) * loss_dice_value
            + float(self.cfg.get("edge_loss_weight", 1.0)) * loss_edge
            + float(kl_weight) * loss_kl
        )
        return loss_total, loss_l1_raw, loss_bce, loss_dice_value, loss_edge, loss_kl


def hard_metrics(y_true, y_prob, threshold):
    import numpy as np

    y_true = (y_true >= 0.5).astype(np.uint8)
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    eps = 1e-9
    return {
        "precision": tp / max(tp + fp, eps),
        "recall": tp / max(tp + fn, eps),
        "dice": (2 * tp) / max(2 * tp + fp + fn, eps),
        "iou": tp / max(tp + fp + fn, eps),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

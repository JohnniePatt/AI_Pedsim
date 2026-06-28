import torch
from torch.nn import functional as F


def _sobel_kernels(device, dtype):
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return gx, gy


def sobel_edge_map(x):
    gx, gy = _sobel_kernels(x.device, x.dtype)
    if x.shape[1] > 1:
        gx = gx.repeat(x.shape[1], 1, 1, 1)
        gy = gy.repeat(x.shape[1], 1, 1, 1)
    px = F.conv2d(x, gx, padding=1, groups=x.shape[1])
    py = F.conv2d(x, gy, padding=1, groups=x.shape[1])
    return torch.sqrt(torch.clamp(px * px + py * py, min=1e-8))


class DensityLossComputer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _weights(self, real_b):
        weights = torch.ones_like(real_b)
        target_scalar = real_b.mean(dim=1, keepdim=True) if real_b.shape[1] > 1 else real_b
        fg_weight = float(self.cfg.get("density_foreground_weight", 0.0))
        intensity_weight = float(self.cfg.get("density_intensity_weight", 0.0))
        threshold = float(self.cfg.get("density_foreground_threshold", 1.0 / 255.0))
        if fg_weight > 0:
            weights = weights + (target_scalar > threshold).float() * fg_weight
        if intensity_weight > 0:
            weights = weights + target_scalar * intensity_weight
        return weights

    def _foreground_mask(self, real_b):
        target_scalar = real_b.mean(dim=1, keepdim=True) if real_b.shape[1] > 1 else real_b
        threshold = float(self.cfg.get("density_foreground_threshold", 1.0 / 255.0))
        return (target_scalar > threshold).float()

    def compute(self, logits, real_b, mu, logvar, kl_weight):
        pred = torch.sigmoid(logits)
        weights = self._weights(real_b)
        foreground_mask = self._foreground_mask(real_b).expand_as(real_b)
        loss_l1 = torch.mean(torch.abs(pred - real_b) * weights)
        loss_mse = torch.mean(((pred - real_b) ** 2) * weights)
        loss_edge = torch.mean(torch.abs(sobel_edge_map(pred) - sobel_edge_map(real_b)))
        foreground_pixels = torch.clamp(foreground_mask.sum(), min=1.0)
        loss_foreground_l1 = torch.sum(torch.abs(pred - real_b) * foreground_mask) / foreground_pixels
        loss_mass = torch.mean(torch.abs(pred.mean(dim=(2, 3)) - real_b.mean(dim=(2, 3))))
        gamma = float(self.cfg.get("density_gamma_loss", 1.0))
        if gamma != 1.0:
            eps = 1e-6
            loss_gamma_l1 = torch.mean(
                torch.abs(torch.clamp(pred, min=eps).pow(gamma) - torch.clamp(real_b, min=eps).pow(gamma))
                * weights
            )
        else:
            loss_gamma_l1 = pred.new_tensor(0.0)
        loss_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss_total = (
            float(self.cfg.get("l1_loss_weight", 1.0)) * loss_l1
            + float(self.cfg.get("mse_loss_weight", 0.0)) * loss_mse
            + float(self.cfg.get("edge_loss_weight", 0.5)) * loss_edge
            + float(self.cfg.get("foreground_l1_loss_weight", 0.0)) * loss_foreground_l1
            + float(self.cfg.get("mass_loss_weight", 0.0)) * loss_mass
            + float(self.cfg.get("gamma_l1_loss_weight", 0.0)) * loss_gamma_l1
            + float(kl_weight) * loss_kl
        )
        return loss_total, loss_l1, loss_mse, loss_edge, loss_kl


def tensor_density_metrics(y_true, y_pred):
    import math
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.clip(y_true, 0.0, 1.0)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(math.sqrt(mse))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))
    mu_x = float(np.mean(y_true))
    mu_y = float(np.mean(y_pred))
    var_x = float(np.var(y_true))
    var_y = float(np.var(y_pred))
    cov_xy = float(np.mean((y_true - mu_x) * (y_pred - mu_y)))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": float(ssim),
    }

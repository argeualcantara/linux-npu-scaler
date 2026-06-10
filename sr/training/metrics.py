import numpy as np
import torch


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index (SSIM) on Y-channel (luminance).
    Range [0, 1], higher is better. Standard metric for SR evaluation.
    """
    pred_y   = pred.clamp(0, 1)
    target_y = target.clamp(0, 1)

    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    mu1 = torch.nn.functional.avg_pool2d(pred_y,   3, 1, 1)
    mu2 = torch.nn.functional.avg_pool2d(target_y, 3, 1, 1)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(pred_y   ** 2, 3, 1, 1) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(target_y ** 2, 3, 1, 1) - mu2_sq
    sigma12   = torch.nn.functional.avg_pool2d(pred_y * target_y, 3, 1, 1) - mu1_mu2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


def compute_psnr_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR on Y channel in dB. >30 = acceptable, >35 = good.
    Input tensors are (B, 1, H, W) — single Y channel.
    """
    pred_y   = pred.clamp(0, 1)
    target_y = target.clamp(0, 1)
    mse = torch.mean((pred_y - target_y) ** 2).item()
    if mse < 1e-10:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

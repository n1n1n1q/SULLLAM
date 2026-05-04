import torch
import numpy as np


def barron_loss(
    e: torch.Tensor | np.ndarray, alpha: float, c: float
) -> torch.Tensor | float:
    eps = 1e-08
    if abs(alpha - 2) < 1e-06:
        return 0.5 * (e / c) ** 2
    if abs(alpha) < 1e-06:
        if isinstance(e, torch.Tensor):
            return torch.log(0.5 * (e / c) ** 2 + 1)
        else:
            return np.log(0.5 * (e / c) ** 2 + 1)
    a_inv = 1.0 / alpha
    denom = (
        np.maximum(abs(alpha - 2), eps)
        if isinstance(e, (int, float, np.ndarray))
        else abs(alpha - 2) + eps
    )
    x2c2 = (e / c) ** 2 / denom
    return a_inv * abs(alpha - 2) * ((x2c2 + 1) ** (alpha / 2) - 1)


def barron_weight(
    e: torch.Tensor | np.ndarray, alpha: float, c: float
) -> torch.Tensor | np.ndarray:
    eps = 1e-08
    denom = (
        np.maximum(abs(alpha - 2), eps)
        if isinstance(e, (int, float, np.ndarray))
        else abs(alpha - 2) + eps
    )
    x2c2 = (e / c) ** 2 / denom
    return (x2c2 + 1) ** (alpha / 2 - 1)

import numpy as np
from scipy.stats import entropy as scipy_entropy


def image_entropy(image_gray: np.ndarray) -> float:
    if image_gray.size == 0:
        return 0.0
    hist, _ = np.histogram(image_gray.flatten(), bins=256, range=(0, 256))
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return float(scipy_entropy(p))


def map_entropy_to_alpha(
    entropy: float, alpha_min: float = -2.0, alpha_max: float = 2.0
) -> float:
    entropy_normalized = np.clip(entropy / 8.0, 0, 1)
    return alpha_min + entropy_normalized * (alpha_max - alpha_min)

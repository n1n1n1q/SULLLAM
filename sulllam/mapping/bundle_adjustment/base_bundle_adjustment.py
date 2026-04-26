from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseBundleAdjustment(ABC):
    @abstractmethod
    def run(self, mapper, K: np.ndarray) -> None:
        raise NotImplementedError

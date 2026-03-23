from __future__ import annotations

import torch
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Iterable


class BaseExtractor(ABC):
    def __init__(self, *, name: str | None = None) -> None:
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def extract(self, image: np.ndarray | torch.tensor) -> list[str]:
        keypoints, descriptors = self._extract(image)
        return keypoints, descriptors

    @abstractmethod
    def _extract(self, image: np.ndarray | torch.tensor) -> Iterable[str]:
        raise NotImplementedError

    def _normalize_keys(self, keys: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for key in keys:
            if not isinstance(key, str):
                continue

            cleaned = key.strip()
            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            normalized.append(cleaned)

        return normalized

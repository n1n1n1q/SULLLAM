from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
import torch

from lightglue import LightGlue as _LightGlue
from lightglue.utils import rbd

from sulllam.localization.matching.base_matcher import BaseMatcher

if TYPE_CHECKING:
    from sulllam.localization.extraction.superpoint import SuperPointFeatureExtractor


@dataclass
class LightGlueConfig:
    features: str = "superpoint"
    filter_threshold: float = 0.1
    depth_confidence: float = 0.95
    width_confidence: float = 0.99
    n_layers: int = 9
    flash: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LightGlueMatcher(BaseMatcher):
    def __init__(self, config: LightGlueConfig | None = None) -> None:
        super().__init__(name="LightGlue")
        self.config = config or LightGlueConfig()
        self.device = torch.device(self.config.device)

        self._model = (
            _LightGlue(
                features=self.config.features,
                filter_threshold=self.config.filter_threshold,
                depth_confidence=self.config.depth_confidence,
                width_confidence=self.config.width_confidence,
                n_layers=self.config.n_layers,
                flash=self.config.flash,
            )
            .eval()
            .to(self.device)
        )

    def _match(self, query_descriptors, train_descriptors):
        if query_descriptors is None or train_descriptors is None:
            return [], np.array([])
        if len(query_descriptors) == 0 or len(train_descriptors) == 0:
            return [], np.array([])

        n0, n1 = len(query_descriptors), len(train_descriptors)
        dummy_kpts0 = torch.zeros(1, n0, 2, device=self.device)
        dummy_kpts1 = torch.zeros(1, n1, 2, device=self.device)

        feats0 = {
            "keypoints": dummy_kpts0,
            "descriptors": torch.from_numpy(query_descriptors).unsqueeze(0).to(self.device),
            "image_size": torch.tensor([[640.0, 480.0]], device=self.device),
        }
        feats1 = {
            "keypoints": dummy_kpts1,
            "descriptors": torch.from_numpy(train_descriptors).unsqueeze(0).to(self.device),
            "image_size": torch.tensor([[640.0, 480.0]], device=self.device),
        }
        return self._run_matcher(feats0, feats1)

    def match_tensors(self, feats0: dict, feats1: dict) -> tuple[list[cv.DMatch], np.ndarray]:
        feats0_b = {k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in feats0.items()}
        feats1_b = {k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in feats1.items()}
        return self._run_matcher(feats0_b, feats1_b)

    def _run_matcher(self, feats0: dict, feats1: dict) -> tuple[list[cv.DMatch], np.ndarray]:
        with torch.no_grad():
            result = self._model({"image0": feats0, "image1": feats1})

        result = rbd(result)
        matches = result["matches"]
        scores = result["scores"]

        if matches.shape[0] == 0:
            return [], np.array([])

        matches_np = matches.cpu().numpy()
        scores_np = scores.cpu().numpy()

        dm_list = [
            cv.DMatch(int(m[0]), int(m[1]), float(1.0 - s))
            for m, s in zip(matches_np, scores_np)
        ]
        return dm_list, scores_np

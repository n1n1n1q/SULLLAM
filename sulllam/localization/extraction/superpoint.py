from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from lightglue import SuperPoint as _SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

from sulllam.localization.extraction.base_extractor import BaseExtractor


@dataclass
class SuperPointConfig:
    max_num_keypoints: int = 1024
    detection_threshold: float = 0.0005
    nms_radius: int = 4
    remove_borders: int = 4
    weights_path: Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SuperPointFeatureExtractor(BaseExtractor):
    def __init__(self, config: SuperPointConfig | None = None) -> None:
        super().__init__(name="SuperPoint")
        self.config = config or SuperPointConfig()
        self.device = torch.device(self.config.device)

        conf = {
            "max_num_keypoints": self.config.max_num_keypoints,
            "detection_threshold": self.config.detection_threshold,
            "nms_radius": self.config.nms_radius,
            "remove_borders": self.config.remove_borders,
        }
        self._model = _SuperPoint(**conf).eval().to(self.device)

        if self.config.weights_path is not None:
            state = torch.load(str(self.config.weights_path), map_location="cpu")
            self._model.load_state_dict(state, strict=False)

    def _extract(self, image: np.ndarray):
        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        img_tensor = numpy_image_to_torch(image).to(self.device)

        with torch.no_grad():
            feats = self._model.extract(img_tensor)

        feats = rbd(feats)

        kpts_xy = feats["keypoints"].cpu().numpy()       # (N, 2)
        scores = feats["keypoint_scores"].cpu().numpy()  # (N,)
        descs = feats["descriptors"].cpu().numpy()       # (N, 256)

        if len(kpts_xy) == 0:
            return [], []

        # Convert to cv2.KeyPoint list so downstream components are unaffected
        keypoints = [
            cv.KeyPoint(float(x), float(y), 1.0, response=float(s))
            for (x, y), s in zip(kpts_xy, scores)
        ]

        return keypoints, descs.astype(np.float32)

    # Expose raw tensor features for LightGlue matcher to avoid re-running the model
    def extract_tensors(self, image: np.ndarray) -> dict:
        """Return raw LightGlue feature dict (keypoints, descriptors, image_size)."""
        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        img_tensor = numpy_image_to_torch(image).to(self.device)

        with torch.no_grad():
            feats = self._model.extract(img_tensor)

        return rbd(feats)

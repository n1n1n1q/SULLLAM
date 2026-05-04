from __future__ import annotations
import numpy as np
import cv2 as cv
import torch
import torchvision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.transforms.functional import to_tensor

COCO_DYNAMIC_CLASSES: frozenset[int] = frozenset([1, 2, 3, 4, 6, 8])
COCO_CLASS_NAMES: dict[str, int] = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
}


class SAMSegmentor:

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        device: str = "cuda",
        class_ids: set[int] | frozenset[int] | None = None,
        detection_threshold: float = 0.5,
    ):
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights
        )
        self.detector.eval().to(device)
        self.device = device
        self.class_ids: frozenset[int] = (
            frozenset(class_ids) if class_ids is not None else frozenset({1})
        )
        self.threshold = detection_threshold

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        boxes = self._detect(image_rgb)
        if not boxes:
            return np.zeros(image_bgr.shape[:2], dtype=bool)
        self.predictor.set_image(image_rgb)
        combined = np.zeros(image_bgr.shape[:2], dtype=bool)
        for box in boxes:
            masks, _, _ = self.predictor.predict(
                box=np.array(box, dtype=float), multimask_output=False
            )
            combined |= masks[0].astype(bool)
        return combined

    def _detect(self, image_rgb: np.ndarray) -> list[list[float]]:
        img_tensor = to_tensor(image_rgb).to(self.device)
        with torch.no_grad():
            preds = self.detector([img_tensor])[0]
        boxes = []
        for box, label, score in zip(
            preds["boxes"].cpu().numpy(),
            preds["labels"].cpu().numpy(),
            preds["scores"].cpu().numpy(),
        ):
            if int(label) in self.class_ids and score >= self.threshold:
                boxes.append(box.tolist())
        return boxes

    @staticmethod
    def draw_overlay(
        image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 200), alpha: float = 0.45
    ) -> np.ndarray:
        overlay = image_bgr.copy()
        if mask.any():
            colored = np.zeros_like(image_bgr)
            colored[mask] = color
            overlay = cv.addWeighted(image_bgr, 1.0, colored, alpha, 0)
        return overlay

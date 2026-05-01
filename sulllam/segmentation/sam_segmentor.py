from __future__ import annotations

import numpy as np
import cv2 as cv

# COCO class IDs for objects that move and should be masked out of SLAM.
# Full list: https://cocodataset.org/#explore  (labels start at 1)
COCO_DYNAMIC_CLASSES: frozenset[int] = frozenset([
    1,   # person
    2,   # bicycle
    3,   # car
    4,   # motorcycle
    6,   # bus
    8,   # truck
])

# Friendly name → COCO ID mapping for convenience.
COCO_CLASS_NAMES: dict[str, int] = {
    "person":     1,
    "bicycle":    2,
    "car":        3,
    "motorcycle": 4,
    "airplane":   5,
    "bus":        6,
    "train":      7,
    "truck":      8,
    "boat":       9,
    "bird":       16,
    "cat":        17,
    "dog":        18,
    "horse":      19,
    "sheep":      20,
    "cow":        21,
    "elephant":   22,
    "bear":       23,
    "zebra":      24,
    "giraffe":    25,
}


class SAMSegmentor:
    """Segments chosen object classes using SAM2 + Faster-RCNN detector.

    The detector runs on COCO (80 classes).  Pass the COCO class IDs you want
    excluded from feature tracking via ``class_ids``.  You can use the
    ``COCO_CLASS_NAMES`` dict or ``COCO_DYNAMIC_CLASSES`` preset for convenience.

    Requires:
      - sam2 package (https://github.com/facebookresearch/segment-anything-2)
      - torchvision >= 0.13

    Example checkpoint/config for SAM2-tiny:
      checkpoint: "checkpoints/sam2_hiera_tiny.pt"
      model_cfg:  "sam2_hiera_t.yaml"

    Usage examples::

        # Default — only people
        SAMSegmentor(ckpt, cfg)

        # People + cars + trucks
        SAMSegmentor(ckpt, cfg, class_ids={1, 3, 8})

        # All common dynamic objects
        from sulllam.segmentation import SAMSegmentor, COCO_DYNAMIC_CLASSES
        SAMSegmentor(ckpt, cfg, class_ids=COCO_DYNAMIC_CLASSES)

        # By name
        from sulllam.segmentation import SAMSegmentor, COCO_CLASS_NAMES as C
        SAMSegmentor(ckpt, cfg, class_ids={C["person"], C["dog"]})
    """

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        device: str = "cuda",
        class_ids: set[int] | frozenset[int] | None = None,
        detection_threshold: float = 0.5,
    ):
        """
        Args:
            sam2_checkpoint:     Path to the SAM2 model weights (.pt file).
            sam2_model_cfg:      SAM2 model config name (e.g. "sam2_hiera_t.yaml").
            device:              "cuda" or "cpu".
            class_ids:           COCO class IDs to mask out.  Defaults to {1} (person only).
            detection_threshold: Minimum detector confidence to keep a box.
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torchvision

        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self.detector.eval().to(device)

        self.device = device
        self.class_ids: frozenset[int] = frozenset(class_ids) if class_ids is not None else frozenset({1})
        self.threshold = detection_threshold

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run detection + SAM2 and return a boolean exclusion mask (H, W).

        True pixels are inside a detected object and will be excluded from
        feature tracking.
        """
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        boxes = self._detect(image_rgb)

        if not boxes:
            return np.zeros(image_bgr.shape[:2], dtype=bool)

        self.predictor.set_image(image_rgb)
        combined = np.zeros(image_bgr.shape[:2], dtype=bool)
        for box in boxes:
            masks, _, _ = self.predictor.predict(
                box=np.array(box, dtype=float),
                multimask_output=False,
            )
            combined |= masks[0].astype(bool)

        return combined

    def _detect(self, image_rgb: np.ndarray) -> list[list[float]]:
        import torch
        from torchvision.transforms.functional import to_tensor

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
                boxes.append(box.tolist())  # [x1, y1, x2, y2]
        return boxes

    @staticmethod
    def draw_overlay(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 200), alpha: float = 0.45) -> np.ndarray:
        """Return a copy of image with the exclusion mask blended in."""
        overlay = image_bgr.copy()
        if mask.any():
            colored = np.zeros_like(image_bgr)
            colored[mask] = color
            overlay = cv.addWeighted(image_bgr, 1.0, colored, alpha, 0)
        return overlay

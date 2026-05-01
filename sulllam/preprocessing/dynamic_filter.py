from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np
import torch


# COCO class ids treated as dynamic by default.
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck.
DEFAULT_DYNAMIC_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class DynamicFilterConfig:
    enabled: bool = True

    # YOLO
    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.35
    yolo_iou: float = 0.5
    yolo_imgsz: int = 640
    dynamic_classes: tuple[int, ...] = DEFAULT_DYNAMIC_CLASSES

    # Depth model HF Hub id.
    # Default uses a transformers-compatible Depth-Anything V2 model.
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_imgsz: int = 518

    # Filtering geometry
    bbox_shrink: float = 0.85          # shrink bbox before filtering
    foreground_percentile: float = 25.0  # depth percentile = foreground level
    margin_scale: float = 0.05         # margin = scale * (p95 - p05) inside bbox
    median_blur_ksize: int = 5         # 0/1 to disable

    # Throttling: re-run YOLO + depth every N frames, reuse last masks otherwise.
    run_every_n: int = 1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _YOLODetector:
    def __init__(self, cfg: DynamicFilterConfig) -> None:
        from ultralytics import YOLO  # lazy
        self.cfg = cfg
        self.model = YOLO(cfg.yolo_weights)
        self.model.to(cfg.device)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        res = self.model.predict(
            image_bgr,
            conf=cfg.yolo_conf,
            iou=cfg.yolo_iou,
            imgsz=cfg.yolo_imgsz,
            device=cfg.device,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy().astype(float)
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)

        H, W = image_bgr.shape[:2]
        detections: list[Detection] = []
        wanted = set(cfg.dynamic_classes)
        for c, p, (x1, y1, x2, y2) in zip(cls, conf, xyxy):
            if c not in wanted:
                continue
            x1 = max(0, min(W - 1, int(x1)))
            x2 = max(0, min(W, int(x2)))
            y1 = max(0, min(H - 1, int(y1)))
            y2 = max(0, min(H, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(Detection(int(c), float(p), (x1, y1, x2, y2)))
        return detections


class _DepthAnythingV3:
    def __init__(self, cfg: DynamicFilterConfig) -> None:
        from transformers import pipeline  # lazy
        self.cfg = cfg
        device = 0 if cfg.device.startswith("cuda") else -1
        model_id = cfg.depth_model
        try:
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_id,
                device=device,
            )
        except ValueError as exc:
            # Some Depth-Anything v3 repos are not transformers-compatible yet.
            msg = str(exc)
            fallback = "depth-anything/Depth-Anything-V2-Small-hf"
            if "model_type" in msg and model_id != fallback:
                print(
                    f"[WARN] Depth model '{model_id}' is not supported by transformers; "
                    f"falling back to '{fallback}'."
                )
                model_id = fallback
                self.pipe = pipeline(
                    task="depth-estimation",
                    model=model_id,
                    device=device,
                )
            else:
                raise
        self.model_id = model_id

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image
        rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        out = self.pipe(pil)
        depth = np.array(out["predicted_depth"]) if "predicted_depth" in out else np.array(out["depth"])
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)

        H, W = image_bgr.shape[:2]
        if depth.shape != (H, W):
            depth = cv.resize(depth, (W, H), interpolation=cv.INTER_LINEAR)

        if self.cfg.median_blur_ksize and self.cfg.median_blur_ksize >= 3:
            k = self.cfg.median_blur_ksize | 1  # force odd
            depth = cv.medianBlur(depth, k)
        return depth


def _shrink_bbox(bbox: tuple[int, int, int, int], scale: float, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    H, W = shape
    nx1 = int(max(0, cx - w / 2))
    ny1 = int(max(0, cy - h / 2))
    nx2 = int(min(W, cx + w / 2))
    ny2 = int(min(H, cy + h / 2))
    return nx1, ny1, nx2, ny2


class DynamicKeypointFilter:
    """
    Semantic + monocular-depth keypoint filter.

    Pipeline per frame:
        detections = YOLO(image)
        depth      = DepthAnythingV3(image)
        for each kp inside any (shrunk) bbox of a dynamic class:
            if depth(kp) <= foreground_depth + margin:  drop
    """

    def __init__(self, config: DynamicFilterConfig | None = None) -> None:
        self.config = config or DynamicFilterConfig()
        self._yolo: _YOLODetector | None = None
        self._depth: _DepthAnythingV3 | None = None
        self._frame_idx = 0
        self._last_detections: list[Detection] = []
        self._last_depth: np.ndarray | None = None

    # ----- lazy model init so disabled mode pays no import cost -----
    def _ensure_models(self) -> None:
        if self._yolo is None:
            self._yolo = _YOLODetector(self.config)
        if self._depth is None:
            self._depth = _DepthAnythingV3(self.config)

    def run_yolo(self, image_bgr: np.ndarray) -> list[Detection]:
        self._ensure_models()
        return self._yolo.detect(image_bgr)

    def run_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        self._ensure_models()
        return self._depth.predict(image_bgr)

    # ----- core filtering -----
    def _dynamic_mask(
        self,
        keypoints_xy: np.ndarray,
        detections: Sequence[Detection],
        depth: np.ndarray,
    ) -> np.ndarray:
        N = keypoints_xy.shape[0]
        is_dynamic = np.zeros(N, dtype=bool)
        if N == 0 or not detections:
            return is_dynamic

        H, W = depth.shape
        xs = np.clip(keypoints_xy[:, 0].astype(int), 0, W - 1)
        ys = np.clip(keypoints_xy[:, 1].astype(int), 0, H - 1)
        kp_depth = depth[ys, xs]

        for det in detections:
            x1, y1, x2, y2 = _shrink_bbox(det.bbox, self.config.bbox_shrink, (H, W))
            if x2 <= x1 or y2 <= y1:
                continue

            patch = depth[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            flat = patch.reshape(-1)
            fg = float(np.percentile(flat, self.config.foreground_percentile))
            p05 = float(np.percentile(flat, 5.0))
            p95 = float(np.percentile(flat, 95.0))
            margin = self.config.margin_scale * max(p95 - p05, 1e-6)

            in_box = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
            close_enough = kp_depth <= (fg + margin)
            is_dynamic |= in_box & close_enough

        return is_dynamic

    def filter(
        self,
        image_bgr: np.ndarray,
        keypoints: list[cv.KeyPoint],
        descriptors: np.ndarray,
    ) -> tuple[list[cv.KeyPoint], np.ndarray, np.ndarray]:
        """
        Returns (kept_keypoints, kept_descriptors, keep_mask).
        keep_mask is a boolean array over the input keypoints (True = kept).
        """
        if not self.config.enabled or len(keypoints) == 0:
            keep = np.ones(len(keypoints), dtype=bool)
            return keypoints, descriptors, keep

        if self._frame_idx % max(1, self.config.run_every_n) == 0 or self._last_depth is None:
            self._last_detections = self.run_yolo(image_bgr)
            self._last_depth = self.run_depth(image_bgr)
        self._frame_idx += 1

        if not self._last_detections:
            keep = np.ones(len(keypoints), dtype=bool)
            return keypoints, descriptors, keep

        xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        is_dyn = self._dynamic_mask(xy, self._last_detections, self._last_depth)
        keep = ~is_dyn

        kept_kps = [kp for kp, k in zip(keypoints, keep) if k]
        kept_descs = descriptors[keep] if len(descriptors) else descriptors
        return kept_kps, kept_descs, keep

    def filter_tensors(
        self,
        feats: dict,
        keep: np.ndarray,
    ) -> dict:
        """Apply the same keep mask to a LightGlue SuperPoint feature dict."""
        if keep.all():
            return feats
        idx = torch.from_numpy(np.where(keep)[0]).long().to(feats["keypoints"].device)
        out = dict(feats)
        out["keypoints"] = feats["keypoints"].index_select(0, idx)
        if "keypoint_scores" in feats:
            out["keypoint_scores"] = feats["keypoint_scores"].index_select(0, idx)
        if "descriptors" in feats:
            out["descriptors"] = feats["descriptors"].index_select(0, idx)
        return out

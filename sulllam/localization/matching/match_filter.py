from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2 as cv
import numpy as np


class RANSACModel(str, Enum):
    ESSENTIAL = "essential"
    FUNDAMENTAL = "fundamental"
    HOMOGRAPHY = "homography"


@dataclass
class MatchFilterConfig:
    # Spatial dedup: drop matches whose query OR train keypoint is within
    # this radius (pixels) of an already-kept match.
    spatial_dedup_radius: float = 4.0

    # RANSAC geometric verification
    ransac_enabled: bool = True
    ransac_model: RANSACModel = RANSACModel.ESSENTIAL
    ransac_threshold: float = 1.0      # pixels (for F/H); epipolar threshold for E
    ransac_confidence: float = 0.999
    ransac_max_iters: int = 2000

    # Camera intrinsics — required for Essential matrix; ignored for F/H
    K: np.ndarray = field(default=None)


def _spatial_dedup(
    matches: list[cv.DMatch],
    kps0: list[cv.KeyPoint],
    kps1: list[cv.KeyPoint],
    radius: float,
) -> list[cv.DMatch]:
    if not matches or radius <= 0:
        return matches

    kept: list[cv.DMatch] = []
    used_q = np.zeros(len(kps0), dtype=bool)
    used_t = np.zeros(len(kps1), dtype=bool)

    pts_q = np.array([kps0[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts_t = np.array([kps1[m.trainIdx].pt for m in matches], dtype=np.float32)

    r2 = radius * radius
    accepted_q: list[np.ndarray] = []
    accepted_t: list[np.ndarray] = []

    for i, m in enumerate(matches):
        if used_q[m.queryIdx] or used_t[m.trainIdx]:
            continue
        if accepted_q:
            aq = np.array(accepted_q)
            at = np.array(accepted_t)
            dq = ((aq - pts_q[i]) ** 2).sum(axis=1)
            dt = ((at - pts_t[i]) ** 2).sum(axis=1)
            if dq.min() < r2 or dt.min() < r2:
                continue
        kept.append(m)
        used_q[m.queryIdx] = True
        used_t[m.trainIdx] = True
        accepted_q.append(pts_q[i])
        accepted_t.append(pts_t[i])

    return kept


def _ransac_filter(
    matches: list[cv.DMatch],
    kps0: list[cv.KeyPoint],
    kps1: list[cv.KeyPoint],
    cfg: MatchFilterConfig,
) -> list[cv.DMatch]:
    if not matches or len(matches) < 8:
        return matches

    pts0 = np.float32([kps0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kps1[m.trainIdx].pt for m in matches])

    mask: np.ndarray | None = None

    if cfg.ransac_model == RANSACModel.ESSENTIAL:
        if cfg.K is None:
            raise ValueError("MatchFilterConfig.K required for Essential matrix RANSAC")
        _, mask = cv.findEssentialMat(
            pts0, pts1, cfg.K,
            method=cv.RANSAC,
            prob=cfg.ransac_confidence,
            threshold=cfg.ransac_threshold,
            maxIters=cfg.ransac_max_iters,
        )
    elif cfg.ransac_model == RANSACModel.FUNDAMENTAL:
        _, mask = cv.findFundamentalMat(
            pts0, pts1,
            method=cv.FM_RANSAC,
            ransacReprojThreshold=cfg.ransac_threshold,
            confidence=cfg.ransac_confidence,
            maxIters=cfg.ransac_max_iters,
        )
    elif cfg.ransac_model == RANSACModel.HOMOGRAPHY:
        _, mask = cv.findHomography(
            pts0.reshape(-1, 1, 2),
            pts1.reshape(-1, 1, 2),
            method=cv.RANSAC,
            ransacReprojThreshold=cfg.ransac_threshold,
            confidence=cfg.ransac_confidence,
            maxIters=cfg.ransac_max_iters,
        )

    if mask is None:
        return []

    inliers = mask.ravel().astype(bool)
    return [m for m, keep in zip(matches, inliers) if keep]


def apply_match_filters(
    matches: list[cv.DMatch],
    kps0: list[cv.KeyPoint],
    kps1: list[cv.KeyPoint],
    cfg: MatchFilterConfig,
) -> list[cv.DMatch]:
    matches = _spatial_dedup(matches, kps0, kps1, cfg.spatial_dedup_radius)
    if cfg.ransac_enabled:
        matches = _ransac_filter(matches, kps0, kps1, cfg)
    return matches

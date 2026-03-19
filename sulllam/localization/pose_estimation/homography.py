from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from sulllam.localization.pose_estimation.base_estimator import BaseEstimator


@dataclass(slots=True)
class HomographyEstimatorConfig:
    method: int = cv.RANSAC
    ransac_reproj_threshold: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.995
    min_matches: int = 4


class HomographyPoseEstimator(BaseEstimator):
    def __init__(self, config: HomographyEstimatorConfig | None = None) -> None:
        super().__init__(name="Homography")
        self.config = config or HomographyEstimatorConfig()

    def _estimate(self, keypoints_query, keypoints_train, matches):
        total_matches = 0 if matches is None else len(matches)

        if matches is None or len(matches) < self.config.min_matches:
            return self._build_empty_result(total_matches, reason="not_enough_matches")

        src_pts = np.float32([keypoints_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_train[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv.findHomography(
            src_pts,
            dst_pts,
            method=self.config.method,
            ransacReprojThreshold=self.config.ransac_reproj_threshold,
            maxIters=self.config.max_iters,
            confidence=self.config.confidence,
        )

        if homography is None or inlier_mask is None:
            return self._build_empty_result(total_matches, reason="homography_failed")


        R, t = self._decompose_homography(homography, K)

        inlier_mask_flat = inlier_mask.ravel().astype(bool)
        inlier_matches = [m for m, is_inlier in zip(matches, inlier_mask_flat) if is_inlier]
        num_inliers = len(inlier_matches)

        return {
            "success": True,
            "reason": "ok",
            "homography": homography,
            "R": R,
            "t": t,
            "inlier_mask": inlier_mask_flat,
            "inlier_matches": inlier_matches,
            "num_matches": total_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": 0.0 if total_matches == 0 else num_inliers / total_matches,
        }

    def _decompose_homography(self, H: np.ndarray, K: np.ndarray):
        num, rs, ts, ns = cv.decomposeHomographyMat(H, K)

        best_idx = 0
        for i in range(num):
            if ns[i][2] < 0: 
                best_idx = i
                break
                
        return rs[best_idx], ts[best_idx]

    def _build_empty_result(self, total_matches: int, *, reason: str):
        return {
            "success": False,
            "reason": reason,
            "homography": None,
            "R": None,
            "t": None,
            "inlier_mask": None,
            "inlier_matches": [],
            "num_matches": total_matches,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

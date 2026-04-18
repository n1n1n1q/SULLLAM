from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from sulllam.localization.pose_estimation.base_estimator import BaseEstimator


@dataclass(slots=True)
class EightPointEstimatorConfig:
    method: int = cv.RANSAC
    ransac_reproj_threshold: float = 3.0
    max_iters: int = 2000
    confidence: float = 0.995
    K: np.ndarray = None


class EightPointPoseEstimator(BaseEstimator):
    def __init__(self, config: EightPointEstimatorConfig | None = None) -> None:
        super().__init__(name="Homography")
        self.config = config or EightPointEstimatorConfig()

    def _estimate(self, keypoints_query, keypoints_train):

        E, mask = cv.findEssentialMat(keypoints_query,
                                      keypoints_train, 
                                      self.config.K,
                                      method=self.config.method,
                                      prob=self.config.confidence, 
                                      threshold=self.config.ransac_reproj_threshold,
                                      maxIters=self.config.max_iters
                                    )

        points, R, t, mask_pose = cv.recoverPose(E, keypoints_query, keypoints_train, self.config.K)

        # points = points[mask_pose]

        return {
            "success": True,
            "R": R,
            "t": t,
            "points": points,
            "inliers_mask": mask
        }


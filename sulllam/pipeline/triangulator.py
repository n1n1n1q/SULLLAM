from __future__ import annotations

import cv2 as cv
import numpy as np


class Triangulator:
    def __init__(self, K: np.ndarray, max_reproj_error: float, max_depth: float, max_points: int):
        self.K = K
        self.max_reproj_error = max_reproj_error
        self.max_depth = max_depth
        self.max_points = max_points

    def triangulate(
        self,
        R_prev: np.ndarray,
        t_prev: np.ndarray,
        R_curr: np.ndarray,
        t_curr: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        image_rgb: np.ndarray,
        kp_indices_prev: np.ndarray | list[int] | None = None,
        kp_indices_curr: np.ndarray | list[int] | None = None,
    ) -> list[dict]:
        K = self.K
        P1 = K @ np.hstack((R_prev, t_prev.reshape(3, 1)))
        P2 = K @ np.hstack((R_curr, t_curr.reshape(3, 1)))

        points_4d = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T

        rvec_prev, _ = cv.Rodrigues(R_prev)
        rvec_curr, _ = cv.Rodrigues(R_curr)

        proj1, _ = cv.projectPoints(points_3d, rvec_prev, t_prev, K, None)
        proj2, _ = cv.projectPoints(points_3d, rvec_curr, t_curr, K, None)

        err1 = np.linalg.norm(proj1.reshape(-1, 2) - pts1, axis=1)
        err2 = np.linalg.norm(proj2.reshape(-1, 2) - pts2, axis=1)

        pts3d_cam1 = (R_prev @ points_3d.T + t_prev.reshape(3, 1)).T
        pts3d_cam2 = (R_curr @ points_3d.T + t_curr.reshape(3, 1)).T

        candidates = []
        for idx, pt3d in enumerate(points_3d):
            if pts3d_cam1[idx, 2] <= 0.0 or pts3d_cam2[idx, 2] <= 0.0:
                continue
            if pts3d_cam1[idx, 2] >= self.max_depth or pts3d_cam2[idx, 2] >= self.max_depth:
                continue
            if err1[idx] > self.max_reproj_error or err2[idx] > self.max_reproj_error:
                continue

            u = int(pts2[idx, 0])
            v = int(pts2[idx, 1])
            if not (0 <= v < image_rgb.shape[0] and 0 <= u < image_rgb.shape[1]):
                continue

            cand = {
                "error": err1[idx] + err2[idx],
                "pt3d": pt3d,
                "color": image_rgb[v, u],
                "uv_prev": pts1[idx],
                "uv_curr": pts2[idx],
                "depth_prev": float(pts3d_cam1[idx, 2]),
                "depth_curr": float(pts3d_cam2[idx, 2]),
            }
            if kp_indices_prev is not None:
                cand["kp_idx_prev"] = int(kp_indices_prev[idx])
            if kp_indices_curr is not None:
                cand["kp_idx_curr"] = int(kp_indices_curr[idx])
            candidates.append(cand)

        candidates.sort(key=lambda x: x["error"])
        return candidates[: self.max_points]

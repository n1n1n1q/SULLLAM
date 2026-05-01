from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from sulllam.mapping.map import Keyframe, Mapper


@dataclass
class LoopClosureConfig:
    min_matches: int = 50
    min_frame_gap: int = 30
    ransac_reproj_threshold: float = 3.0
    ransac_confidence: float = 0.995
    min_inlier_ratio: float = 0.6
    # Cap how many loop-closure edges may be returned per detector call. Keeps
    # PGO from being swamped by dozens of near-duplicate "loops" produced by a
    # forward-moving camera with overlapping textures.
    max_candidates: int = 1
    max_rotation_deg: float = 45.0
    min_median_parallax_px: float = 5.0


class LoopClosureDetector:
    def __init__(self, config: LoopClosureConfig | None = None) -> None:
        self.config = config or LoopClosureConfig()
        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 50}
        self._flann = cv.FlannBasedMatcher(index_params, search_params)

    def detect(
        self,
        current_kf: Keyframe,
        mapper: Mapper,
        K: np.ndarray,
    ) -> list[dict]:
        cfg = self.config
        candidates = []

        if current_kf.descriptors is None or len(current_kf.descriptors) == 0:
            return candidates

        query_descs = np.asarray(current_kf.descriptors, dtype=np.float32)

        for kf in mapper.keyframes:
            if kf.idx >= current_kf.idx - cfg.min_frame_gap:
                continue
            if kf.descriptors is None or len(kf.descriptors) == 0:
                continue

            train_descs = np.asarray(kf.descriptors, dtype=np.float32)

            try:
                knn_matches = self._flann.knnMatch(query_descs, train_descs, k=2)
            except cv.error:
                continue

            good = [
                m for m, n in knn_matches
                if len([m, n]) == 2 and m.distance < 0.75 * n.distance
            ]

            if len(good) < cfg.min_matches:
                continue

            query_pts = np.array(
                [current_kf.keypoints[m.queryIdx].pt for m in good], dtype=np.float32
            )
            train_pts = np.array(
                [kf.keypoints[m.trainIdx].pt for m in good], dtype=np.float32
            )

            median_parallax = float(
                np.median(np.linalg.norm(query_pts - train_pts, axis=1))
            )
            if median_parallax < cfg.min_median_parallax_px:
                continue

            E, mask = cv.findEssentialMat(
                query_pts,
                train_pts,
                K,
                method=cv.RANSAC,
                prob=cfg.ransac_confidence,
                threshold=cfg.ransac_reproj_threshold,
            )

            if E is None or mask is None:
                continue

            num_inliers = int(mask.sum())
            inlier_ratio = num_inliers / len(good)

            if inlier_ratio < cfg.min_inlier_ratio or num_inliers < cfg.min_matches:
                continue

            _, R, t, _ = cv.recoverPose(E, query_pts, train_pts, K, mask=mask)

            cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
            rot_angle_deg = float(np.degrees(np.arccos(cos_theta)))
            if rot_angle_deg > cfg.max_rotation_deg:
                continue

            T_match_query = np.eye(4)
            T_match_query[:3, :3] = R
            T_match_query[:3, 3] = t.flatten()
            relative_pose = np.linalg.inv(T_match_query)

            candidates.append({
                "query_kf": current_kf,
                "match_kf": kf,
                "relative_pose": relative_pose,
                "num_inliers": num_inliers,
                "inlier_ratio": inlier_ratio,
                "rot_angle_deg": rot_angle_deg,
            })

        candidates.sort(key=lambda c: c["num_inliers"], reverse=True)
        if cfg.max_candidates > 0:
            candidates = candidates[: cfg.max_candidates]

        for c in candidates:
            print(f"[LC] Loop closure: kf {c['query_kf'].idx} <-> kf {c['match_kf'].idx}  "
                  f"({c['num_inliers']} inliers, ratio {c['inlier_ratio']:.2f}, "
                  f"rot {c['rot_angle_deg']:.1f} deg)")

        return candidates

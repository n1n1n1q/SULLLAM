from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from sulllam.localization.global_descriptor import cosine_similarity
from sulllam.mapping.map import Keyframe, Mapper


@dataclass
class LoopClosureConfig:
    # Minimum number of LightGlue/FLANN matches before considering a pair.
    # Higher than the old default — LightGlue is a strong matcher and 30
    # matches is well within "any visually overlapping pair" territory.
    min_matches: int = 80
    # Same in absolute inlier count after RANSAC. We want a real loop, not
    # "two views happen to share a wall".
    min_inliers: int = 60
    # Tighter inlier ratio than the previous 0.5.
    min_inlier_ratio: float = 0.7
    # Distance between keyframes (in keyframe positions, not raw frames)
    # before a pair is even eligible to be a loop closure.
    min_kf_gap: int = 10
    ransac_reproj_threshold: float = 3.0
    ransac_confidence: float = 0.995
    # Appearance gate. Only run the (expensive) geometric check on candidate
    # keyframes whose mean-pooled descriptor is close enough to the query.
    # The mean-pooled SuperPoint descriptor isn't extremely discriminative
    # on its own — this gate's job is to skip obviously dissimilar pairs,
    # not to be the primary filter. The geometric / inlier-count gates do
    # the real work. Disable with a value <= -1.0.
    appearance_threshold: float = 0.4
    # Cap how many loop-closure edges may be returned per detector call.
    max_candidates: int = 1
    # Information matrix tuning. The rotation-only block of the loop closure
    # edge is scaled by `info_scale * min(1.0, num_inliers / ref_inliers)`.
    # This makes data-rich LCs more authoritative and weak ones less so.
    info_scale: float = 4.0
    info_ref_inliers: int = 150


class LoopClosureDetector:
    def __init__(self, config: LoopClosureConfig | None = None) -> None:
        self.config = config or LoopClosureConfig()
        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 50}
        self._flann = cv.FlannBasedMatcher(index_params, search_params)

    def _match_kf_pair(
        self,
        query_kf: Keyframe,
        match_kf: Keyframe,
        matcher_obj=None,
    ) -> list[cv.DMatch]:
        # Prefer LightGlue when feature tensors are cached on both keyframes.
        # Falls back to FLANN ratio test on raw descriptors.
        if matcher_obj is not None and query_kf.feats is not None and match_kf.feats is not None:
            try:
                from sulllam.localization.matching.lightglue import LightGlueMatcher
                if isinstance(matcher_obj, LightGlueMatcher):
                    matches, _ = matcher_obj.match_tensors(query_kf.feats, match_kf.feats)
                    return matches
            except ImportError:
                pass

        if query_kf.descriptors is None or match_kf.descriptors is None:
            return []
        if len(query_kf.descriptors) == 0 or len(match_kf.descriptors) == 0:
            return []

        query_descs = np.asarray(query_kf.descriptors, dtype=np.float32)
        train_descs = np.asarray(match_kf.descriptors, dtype=np.float32)

        try:
            knn_matches = self._flann.knnMatch(query_descs, train_descs, k=2)
        except cv.error:
            return []

        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        return good

    def _build_information(self, num_inliers: int) -> np.ndarray:
        """Build a 6x6 information matrix scaled by inlier count.

        Only the rotation block is meaningful for monocular LC edges, but we
        return a full 6x6 so PGO's per-edge weighting can pick out either
        block consistently with odometry edges.
        """
        cfg = self.config
        scale = cfg.info_scale * min(1.0, num_inliers / max(cfg.info_ref_inliers, 1))
        scale = max(scale, 1e-3)
        info = np.eye(6, dtype=np.float64) * scale
        return info

    def detect(
        self,
        current_kf: Keyframe,
        mapper: Mapper,
        K: np.ndarray,
        matcher_obj=None,
    ) -> list[dict]:
        cfg = self.config
        candidates = []

        if current_kf.descriptors is None or len(current_kf.descriptors) == 0:
            return candidates

        # Keyframe-distance gating (positions in the list, not frame ids).
        kfs = mapper.keyframes
        n = len(kfs)
        if n < cfg.min_kf_gap + 2:
            return candidates
        # Position of `current_kf` in the keyframe list (it's the latest).
        current_pos = n - 1

        for pos, kf in enumerate(kfs):
            if kf.idx == current_kf.idx:
                continue
            if (current_pos - pos) < cfg.min_kf_gap:
                continue
            if kf.descriptors is None or len(kf.descriptors) == 0:
                continue

            # Appearance pre-filter.
            if cfg.appearance_threshold > -1.0:
                sim = cosine_similarity(
                    current_kf.global_descriptor, kf.global_descriptor
                )
                if sim < cfg.appearance_threshold:
                    continue

            good = self._match_kf_pair(current_kf, kf, matcher_obj=matcher_obj)
            if len(good) < cfg.min_matches:
                continue

            query_pts = np.array(
                [current_kf.keypoints[m.queryIdx].pt for m in good], dtype=np.float32
            )
            train_pts = np.array(
                [kf.keypoints[m.trainIdx].pt for m in good], dtype=np.float32
            )

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

            if inlier_ratio < cfg.min_inlier_ratio or num_inliers < cfg.min_inliers:
                continue

            _, R, t, _ = cv.recoverPose(E, query_pts, train_pts, K, mask=mask)

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
                "information": self._build_information(num_inliers),
            })

        # Keep only the strongest candidates (by inlier count).
        candidates.sort(key=lambda c: c["num_inliers"], reverse=True)
        if cfg.max_candidates > 0:
            candidates = candidates[: cfg.max_candidates]

        for c in candidates:
            print(f"[LC] Loop closure: kf {c['query_kf'].idx} ↔ kf {c['match_kf'].idx}  "
                  f"({c['num_inliers']} inliers, ratio {c['inlier_ratio']:.2f})")

        return candidates

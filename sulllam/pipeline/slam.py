from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from sulllam.localization.extraction.superpoint import SuperPointFeatureExtractor
from sulllam.localization.matching.lightglue import LightGlueMatcher
from sulllam.mapping.map import Keyframe, Mapper
from sulllam.mapping.pose_graph import PoseGraph
from sulllam.pipeline.config import SLAMConfig
from sulllam.pipeline.triangulator import Triangulator


def _Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T



_PER_KEYPOINT_KEYS = frozenset({"keypoints", "keypoint_scores", "descriptors", "scales", "oris"})


def _filter_by_mask(
    kps: list,
    descs: np.ndarray,
    feats: dict | None,
    mask: np.ndarray,
) -> tuple[list, np.ndarray, dict | None, list[int]]:
    """Drop keypoints that fall inside the boolean exclusion mask (True = exclude).

    Returns filtered (kps, descs, feats, keep_indices) where keep_indices maps
    new index → original index.

    Note: extract_tensors() returns tensors after rbd(), so shapes are
    (N, D) / (N,) with no batch dim.  match_tensors() adds the batch dim
    back via unsqueeze(0) before passing to LightGlue.  We therefore index
    on dim 0 here, and only for known per-keypoint keys.
    """
    h, w = mask.shape[:2]
    keep = [
        i for i, kp in enumerate(kps)
        if not mask[
            min(int(kp.pt[1]), h - 1),
            min(int(kp.pt[0]), w - 1),
        ]
    ]

    filtered_kps = [kps[i] for i in keep]
    filtered_descs = descs[keep] if descs is not None and len(keep) > 0 else descs

    filtered_feats: dict | None = None
    if feats is not None:
        import torch
        filtered_feats = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor) and k in _PER_KEYPOINT_KEYS and len(keep) > 0:
                filtered_feats[k] = v[keep]
            elif isinstance(v, torch.Tensor) and k in _PER_KEYPOINT_KEYS and len(keep) == 0:
                # Preserve correct shape with 0 keypoints
                filtered_feats[k] = v[:0]
            else:
                filtered_feats[k] = v

    return filtered_kps, filtered_descs, filtered_feats, keep


class SLAMPipeline:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.mapper = Mapper()
        self.triangulator = Triangulator(
            K=config.K,
            max_reproj_error=config.max_reproj_error,
            max_depth=config.max_depth,
            max_points=config.max_points,
            min_parallax_deg=config.min_paralax_deg
        )
        self.pose_graph = PoseGraph()

        self._R_global = np.eye(3)
        self._t_global = np.zeros(3)
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._prev_feats: dict | None = None
        self._current_match_scores: np.ndarray = np.array([])
        self._frame_idx = 0
        self._last_pgo_kf: int = -10**9
        self.trajectory: list[np.ndarray] = []

        self._last_seg_mask: np.ndarray | None = None

        # Keyframe selection state
        self._R_last_kf = np.eye(3)
        self._t_last_kf = np.zeros(3)
        self._last_kf_frame_idx: int = 0
        self._last_kf_n_kps: int = 0

        config.clouds_dir.mkdir(parents=True, exist_ok=True)

    def _initialize(self, image: np.ndarray) -> None:
        kps, descs = self.config.extractor.extract(image)
        feats: dict | None = None
        if isinstance(self.config.extractor, SuperPointFeatureExtractor):
            feats = self.config.extractor.extract_tensors(image)

        if self.config.segmentor is not None:
            seg_mask = self.config.segmentor.segment(image)
            self._last_seg_mask = seg_mask
            kps, descs, feats, _ = _filter_by_mask(kps, descs, feats, seg_mask)
            print(f"[SLAM] Init: {len(kps)} keypoints after segmentation filter")

        self._prev_keypoints = kps
        self._prev_descriptors = descs
        self._prev_feats = feats

        initial_kf = Keyframe(
            idx=0,
            keypoints=kps,
            descriptors=descs,
            pose=_Rt_to_T(self._R_global, self._t_global),
        )
        self.mapper.add_keyframe(initial_kf)
        self.trajectory.append(np.zeros(3))
        self._frame_idx = 1

        self._R_last_kf = np.eye(3)
        self._t_last_kf = np.zeros(3)
        self._last_kf_frame_idx = 0
        self._last_kf_n_kps = len(kps)

    def _is_keyframe(self, n_inliers: int) -> bool:
        cfg = self.config
        frames_since_kf = self._frame_idx - self._last_kf_frame_idx

        if frames_since_kf >= cfg.kf_max_frames:
            return True

        C_curr = -self._R_global.T @ self._t_global
        C_kf = -self._R_last_kf.T @ self._t_last_kf
        translation = np.linalg.norm(C_curr - C_kf)

        R_rel = self._R_global @ self._R_last_kf.T
        cos_angle = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
        rotation_deg = np.degrees(np.arccos(cos_angle))

        tracked_ratio = n_inliers / max(self._last_kf_n_kps, 1)

        return (
            translation >= cfg.kf_min_translation
            or rotation_deg >= cfg.kf_min_rotation_deg
            or tracked_ratio < cfg.kf_max_tracked_ratio
        )

    def _process_frame(self, image: np.ndarray) -> dict:
        cfg = self.config
        i = self._frame_idx

        curr_kps, curr_descs = cfg.extractor.extract(image)
        curr_feats: dict | None = None
        if isinstance(cfg.extractor, SuperPointFeatureExtractor):
            curr_feats = cfg.extractor.extract_tensors(image)

        keep_for_match: list[int] = list(range(len(curr_kps)))
        if cfg.segmentor is not None and self._last_seg_mask is not None:
            curr_kps, curr_descs, curr_feats, keep_for_match = _filter_by_mask(
                curr_kps, curr_descs, curr_feats, self._last_seg_mask
            )

        if (
            isinstance(cfg.matcher, LightGlueMatcher)
            and curr_feats is not None
            and self._prev_feats is not None
        ):
            matches, self._current_match_scores = cfg.matcher.match_tensors(self._prev_feats, curr_feats)
        else:
            matches, self._current_match_scores = cfg.matcher.match(self._prev_descriptors, curr_descs)

        if len(matches) < 8:
            print(f"[SLAM] Frame {i}: too few matches ({len(matches)}), skipping")
            camera_pos = -self._R_global.T @ self._t_global
            self.trajectory.append(camera_pos)
            self._frame_idx += 1
            return {"matches": matches, "prev_keypoints": self._prev_keypoints,
                    "curr_keypoints": curr_kps, "image": image, "seg_mask": None}

        prev_pts = np.array(
            [self._prev_keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        curr_pts = np.array(
            [curr_kps[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        estimate = cfg.pose_estimator.estimate(prev_pts, curr_pts)
        R, t = estimate["R"], estimate["t"].reshape(-1)
        inliers_mask = estimate["inliers_mask"]

        self._R_global = R @ self._R_last_kf
        self._t_global = R @ self._t_last_kf + t

        inlier_mask = inliers_mask.ravel() == 1
        n_inliers = int(inlier_mask.sum())

        camera_pos = -self._R_global.T @ self._t_global
        self.trajectory.append(camera_pos)

        if not self._is_keyframe(n_inliers):
            print(f"[SLAM] Frame {i}: skipped (inliers={n_inliers}, "
                  f"frames_since_kf={i - self._last_kf_frame_idx})")
            self._frame_idx += 1
            return {"matches": matches, "prev_keypoints": self._prev_keypoints,
                    "curr_keypoints": curr_kps, "image": image, "seg_mask": None}


        inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
        prev_inliers = prev_pts[inlier_mask].reshape(-1, 2)
        curr_inliers = curr_pts[inlier_mask].reshape(-1, 2)


        seg_mask: np.ndarray | None = None
        if cfg.segmentor is not None:
            seg_mask = cfg.segmentor.segment(image)
            self._last_seg_mask = seg_mask

            h, w = seg_mask.shape[:2]
            keep_for_kf: list[int] = []  # indices into curr_kps (old-mask-filtered)
            for local_idx, orig_idx in enumerate(keep_for_match):
                kp = curr_kps[local_idx]
                y = min(int(kp.pt[1]), h - 1)
                x = min(int(kp.pt[0]), w - 1)
                if not seg_mask[y, x]:
                    keep_for_kf.append(local_idx)

            old_to_new_kf = {old: new for new, old in enumerate(keep_for_kf)}
            curr_kps    = [curr_kps[j]    for j in keep_for_kf]
            curr_descs  = curr_descs[keep_for_kf] if curr_descs is not None else curr_descs
            if curr_feats is not None:
                import torch
                curr_feats = {
                    k: v[keep_for_kf] if isinstance(v, torch.Tensor) and k in _PER_KEYPOINT_KEYS else v
                    for k, v in curr_feats.items()
                }

            kept = [
                (m, p, c)
                for m, p, c in zip(inlier_matches, prev_inliers, curr_inliers)
                if m.trainIdx in old_to_new_kf
            ]
            if kept:
                inlier_matches, prev_list, curr_list = [], [], []
                for m, p, c in kept:
                    new_m = cv.DMatch()
                    new_m.queryIdx = m.queryIdx
                    new_m.trainIdx = old_to_new_kf[m.trainIdx]
                    new_m.distance = m.distance
                    inlier_matches.append(new_m)
                    prev_list.append(p)
                    curr_list.append(c)
                prev_inliers = np.array(prev_list)
                curr_inliers = np.array(curr_list)
            else:
                inlier_matches = []
                prev_inliers = np.empty((0, 2))
                curr_inliers = np.empty((0, 2))

            print(f"[SLAM] Frame {i}: keyframe — {n_inliers - len(inlier_matches)} "
                  f"keypoints removed by SAM segmentation")

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        curr_kf = Keyframe(
            idx=i,
            keypoints=curr_kps,
            descriptors=curr_descs,
            pose=_Rt_to_T(self._R_global, self._t_global),
            match_scores=self._current_match_scores,
        )
        self.mapper.add_keyframe(curr_kf)

        prev_kf_map = self.mapper.previous_keyframe
        if prev_kf_map is not None:
            rel_pose = _Rt_to_T(R, t)
            self.pose_graph.add_odometry_edge(prev_kf_map.idx, curr_kf.idx, rel_pose)

        # Split inliers: points already tracked (existing 3D point) vs new
        track_list: list[tuple[int, int, int, int]] = []  # (local_idx, query_kp, train_kp, pt_id)
        new_indices: list[int] = []
        for j, m in enumerate(inlier_matches):
            pt_id = prev_kf_map.kp_to_pt.get(m.queryIdx)
            if pt_id is not None:
                track_list.append((j, m.queryIdx, m.trainIdx, pt_id))
            else:
                new_indices.append(j)

        for j, _, train_idx, pt_id in track_list:
            self.mapper.pointmap.add_observation(pt_id, curr_kf.idx, curr_inliers[j])
            curr_kf.kp_to_pt[train_idx] = pt_id

        if new_indices:
            new_prev = prev_inliers[new_indices]
            new_curr = curr_inliers[new_indices]
            candidates = self.triangulator.triangulate(
                R_prev=prev_kf_map.R,
                t_prev=prev_kf_map.t,
                R_curr=curr_kf.R,
                t_curr=curr_kf.t,
                pts1=new_prev,
                pts2=new_curr,
                image_rgb=image_rgb,
            )
            for cand in candidates:
                orig_j = new_indices[cand["orig_idx"]]
                m = inlier_matches[orig_j]
                pt_id = self.mapper.pointmap.add_point(cand["pt3d"], color=cand["color"])
                self.mapper.pointmap.add_observation(pt_id, prev_kf_map.idx, cand["uv_prev"])
                self.mapper.pointmap.add_observation(pt_id, curr_kf.idx, cand["uv_curr"])
                prev_kf_map.kp_to_pt[m.queryIdx] = pt_id
                curr_kf.kp_to_pt[m.trainIdx] = pt_id

        n_kfs = len(self.mapper.keyframes)
        if n_kfs % cfg.ba_frequency == 0 and n_kfs >= cfg.ba_min_frames:
            cfg.bundle_adjustment.run(self.mapper, cfg.K)

        loop_closed = False
        if n_kfs % cfg.lc_frequency == 0:
            lc_candidates = cfg.loop_closure_detector.detect(curr_kf, self.mapper, cfg.K)
            for lc in lc_candidates:
                self.pose_graph.add_loop_closure_edge(
                    from_id=lc["match_kf"].idx,
                    to_id=lc["query_kf"].idx,
                    relative_pose=lc["relative_pose"],
                )
                loop_closed = True

        # Throttle PGO/GBA: even if the LC detector keeps firing on the same
        # revisit, don't re-run PGO unless enough new keyframes have been added
        # since the last optimisation. Avoids pummelling the map with back-to-
        # back PGO+GBA on overlapping detections.
        # run_pgo = loop_closed and (i - self._last_pgo_kf) >= cfg.lc_frequency

        # if run_pgo:
        #     cfg.pose_graph_optimizer.optimize(self.mapper, self.pose_graph)
        #     self._last_pgo_kf = i

        # if run_pgo and i >= cfg.gba_min_frames:
        #     cfg.global_bundle_adjustment.run(self.mapper, cfg.K)

        # Sync from mapper in case BA updated poses.
        self._R_global = self.mapper.current_keyframe.R.copy()
        self._t_global = self.mapper.current_keyframe.t.copy()

        # Advance keyframe selection state.
        self._R_last_kf = self._R_global.copy()
        self._t_last_kf = self._t_global.copy()
        self._last_kf_frame_idx = i
        self._last_kf_n_kps = len(curr_kps)

        # _prev_* always points to the last inserted keyframe (post-filter).
        self._prev_keypoints = curr_kps
        self._prev_descriptors = curr_descs
        self._prev_feats = curr_feats

        self.mapper.pointmap.save_pointcloud(cfg.clouds_dir / f"{i}.ply")
        self._frame_idx += 1

        return {
            "matches": inlier_matches,
            "prev_keypoints": self._prev_keypoints,
            "curr_keypoints": curr_kps,
            "image": image,
            "seg_mask": seg_mask,
        }

    def run(self, images: list[np.ndarray], ros_publisher=None) -> np.ndarray:
        self._initialize(images[0])

        for i, image in enumerate(images[1:], start=1):
            frame_info = self._process_frame(image)

            if ros_publisher is not None:
                self._publish(ros_publisher, frame_info, images[i - 1], i)

        return np.array(self.trajectory)

    def _publish(self, ros_publisher, frame_info: dict, prev_image: np.ndarray, i: int) -> None:
        kf = self.mapper.current_keyframe
        R = kf.R
        t = kf.t

        current_orientation = Rotation.from_matrix(R).as_quat()

        translations, orientations = [], []
        for kf in self.mapper.keyframes:
            translations.append(-kf.R.T @ kf.t)
            orientations.append(Rotation.from_matrix(kf.R.T).as_quat())

        image = frame_info["image"]
        matches = frame_info["matches"]
        curr_kps = frame_info["curr_keypoints"]
        seg_mask: np.ndarray | None = frame_info.get("seg_mask")

        pair = cv.hconcat([prev_image, image])
        match_img = cv.drawMatches(
            prev_image,
            self.mapper.keyframes[-2].keypoints if len(self.mapper.keyframes) >= 2 else [],
            image,
            curr_kps,
            matches,
            None,
        )

        ros_publisher.publish_trajectory(translations, orientations)
        ros_publisher.publish_current_pair(pair)
        ros_publisher.publish_current_matches(match_img)
        ros_publisher.publish_pointcloud(
            self.mapper.pointmap.points_3d[: self.mapper.pointmap.num_points],
            self.mapper.pointmap.point_colors[: self.mapper.pointmap.num_points],
        )

        if seg_mask is not None:
            ros_publisher.publish_segmentation_overlay(image, seg_mask)

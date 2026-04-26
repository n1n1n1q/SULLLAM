from __future__ import annotations

import time
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from sulllam.mapping.map import Keyframe, Mapper
from sulllam.pipeline.config import SLAMConfig
from sulllam.pipeline.triangulator import Triangulator


def _Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


class SLAMPipeline:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.mapper = Mapper()
        self.triangulator = Triangulator(
            K=config.K,
            max_reproj_error=config.max_reproj_error,
            max_depth=config.max_depth,
            max_points=config.max_points,
        )

        self._R_global = np.eye(3)
        self._t_global = np.zeros(3)
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._frame_idx = 0
        self.trajectory: list[np.ndarray] = []

        config.clouds_dir.mkdir(parents=True, exist_ok=True)

    def _initialize(self, image: np.ndarray) -> None:
        kps, descs = self.config.extractor.extract(image)
        self._prev_keypoints = kps
        self._prev_descriptors = descs

        initial_kf = Keyframe(
            idx=0,
            keypoints=kps,
            descriptors=descs,
            pose=_Rt_to_T(self._R_global, self._t_global),
        )
        self.mapper.add_keyframe(initial_kf)
        self.trajectory.append(np.zeros(3))
        self._frame_idx = 1

    def _process_frame(self, image: np.ndarray) -> dict:
        cfg = self.config
        i = self._frame_idx

        curr_kps, curr_descs = cfg.extractor.extract(image)
        matches = cfg.matcher.match(self._prev_descriptors, curr_descs)

        prev_pts = np.array([self._prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.array([curr_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        estimate = cfg.pose_estimator.estimate(prev_pts, curr_pts)
        R, t = estimate["R"], estimate["t"].reshape(-1)
        inliers_mask = estimate["inliers_mask"]

        self._R_global = R @ self._R_global
        self._t_global = R @ self._t_global + t

        curr_kf = Keyframe(
            idx=i,
            keypoints=curr_kps,
            descriptors=curr_descs,
            pose=_Rt_to_T(self._R_global, self._t_global),
        )
        self.mapper.add_keyframe(curr_kf)

        # Triangulate
        inliers = inliers_mask.ravel() == 1
        prev_inliers = prev_pts[inliers].reshape(-1, 2)
        curr_inliers = curr_pts[inliers].reshape(-1, 2)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        candidates = self.triangulator.triangulate(
            R_prev=self.mapper.previous_keyframe.R,
            t_prev=self.mapper.previous_keyframe.t,
            R_curr=self.mapper.current_keyframe.R,
            t_curr=self.mapper.current_keyframe.t,
            pts1=prev_inliers,
            pts2=curr_inliers,
            image_rgb=image_rgb,
        )

        for cand in candidates:
            pt_id = self.mapper.pointmap.add_point(cand["pt3d"], color=cand["color"])
            self.mapper.pointmap.add_observation(pt_id, self.mapper.previous_keyframe.idx, cand["uv_prev"])
            self.mapper.pointmap.add_observation(pt_id, self.mapper.current_keyframe.idx, cand["uv_curr"])

        # Bundle adjustment
        if i % cfg.ba_frequency == 0 and i >= cfg.ba_min_frames:
            cfg.bundle_adjustment.run(self.mapper, cfg.K)
            self._R_global = self.mapper.current_keyframe.R.copy()
            self._t_global = self.mapper.current_keyframe.t.copy()

        # Update accumulators from (possibly LBA-corrected) keyframe
        self._R_global = self.mapper.current_keyframe.R
        self._t_global = self.mapper.current_keyframe.t

        camera_pos = -self._R_global.T @ self._t_global
        self.trajectory.append(camera_pos)

        self.mapper.pointmap.save_pointcloud(cfg.clouds_dir / f"{i}.ply")

        self._prev_keypoints = curr_kps
        self._prev_descriptors = curr_descs
        self._frame_idx += 1

        return {
            "matches": matches,
            "prev_keypoints": self._prev_keypoints,
            "curr_keypoints": curr_kps,
            "image": image,
        }

    def run(self, images: list[np.ndarray], ros_publisher=None) -> np.ndarray:
        self._initialize(images[0])

        for i, image in enumerate(images[1:], start=1):
            time.sleep(1)
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
        prev_kps = frame_info["prev_keypoints"]

        # prev_image is images[i-1], but we stored prev_kps from the previous iteration already
        pair = cv.hconcat([prev_image, image])
        match_img = cv.drawMatches(prev_image, self.mapper.keyframes[-2].keypoints, image, curr_kps, matches, None)

        ros_publisher.publish_trajectory(translations, orientations)
        ros_publisher.publish_current_pair(pair)
        ros_publisher.publish_current_matches(match_img)
        ros_publisher.publish_pointcloud(
            self.mapper.pointmap.points_3d[: self.mapper.pointmap.num_points],
            self.mapper.pointmap.point_colors[: self.mapper.pointmap.num_points],
        )

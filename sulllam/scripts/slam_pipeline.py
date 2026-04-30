from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sulllam.localization.extraction.superpoint import (
    SuperPointFeatureExtractor,
    SuperPointConfig,
)
from sulllam.localization.matching.lightglue import LightGlueMatcher, LightGlueConfig
from sulllam.localization.pose_estimation.eight_point_estimator import (
    EightPointPoseEstimator,
    EightPointEstimatorConfig,
)
from sulllam.mapping.bundle_adjustment.local_bundle_adjustment import (
    LocalBundleAdjustment,
    LocalBundleAdjustmentConfig,
)
from sulllam.mapping.map import Keyframe, Mapper
from sulllam.pipeline.triangulator import Triangulator


def _Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


class MinimalSLAMPipeline:

    def __init__(
        self,
        K: np.ndarray,
        max_reproj_error: float = 2.0,
        max_depth: float = 50.0,
        max_points: int = 100,
        ba_frequency: int = 5,
        ba_min_frames: int = 12,
        clouds_dir: Path = Path("clouds"),
        use_match_confidence: bool = False,
        confidence_gamma: float = 1.0,
        use_adaptive_barron: bool = False,
        barron_alpha_min: float = -2.0,
        barron_alpha_max: float = 2.0,
    ):
        self.K = K
        self.max_reproj_error = max_reproj_error
        self.max_depth = max_depth
        self.max_points = max_points
        self.ba_frequency = ba_frequency
        self.ba_min_frames = ba_min_frames
        self.clouds_dir = clouds_dir

        self.extractor = SuperPointFeatureExtractor(SuperPointConfig())
        self.matcher = LightGlueMatcher(LightGlueConfig())
        self.pose_estimator = EightPointPoseEstimator(
            config=EightPointEstimatorConfig(K=K)
        )
        self.bundle_adjustment = LocalBundleAdjustment(
            LocalBundleAdjustmentConfig(
                use_match_confidence=use_match_confidence,
                confidence_gamma=confidence_gamma,
                use_adaptive_barron=use_adaptive_barron,
                barron_alpha_min=barron_alpha_min,
                barron_alpha_max=barron_alpha_max,
            )
        )
        self.triangulator = Triangulator(
            K=K,
            max_reproj_error=max_reproj_error,
            max_depth=max_depth,
            max_points=max_points,
        )

        self.mapper = Mapper()
        self.trajectory: list[np.ndarray] = []

        self._R_global = np.eye(3)
        self._t_global = np.zeros(3)
        self._prev_keypoints = None
        self._prev_descriptors = None
        self._prev_feats: dict | None = None
        self._frame_idx = 0

        self.clouds_dir.mkdir(parents=True, exist_ok=True)

    def _initialize(self, image: np.ndarray) -> None:
        kps, descs = self.extractor.extract(image)
        self._prev_keypoints = kps
        self._prev_descriptors = descs

        self._prev_feats = self.extractor.extract_tensors(image)

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
        cfg_ba_freq = self.ba_frequency
        cfg_ba_min = self.ba_min_frames
        i = self._frame_idx

        curr_kps, curr_descs = self.extractor.extract(image)
        curr_feats = self.extractor.extract_tensors(image)

        if curr_feats is not None and self._prev_feats is not None:
            matches, match_scores = self.matcher.match_tensors(self._prev_feats, curr_feats)
        else:
            matches, match_scores = self.matcher.match(self._prev_descriptors, curr_descs)

        if len(matches) < 8:
            print(f"[SLAM] Frame {i}: too few matches ({len(matches)}), skipping")
            self._prev_keypoints = curr_kps
            self._prev_descriptors = curr_descs
            self._prev_feats = curr_feats
            self._frame_idx += 1
            return {
                "matches": matches,
                "prev_keypoints": self._prev_keypoints,
                "curr_keypoints": curr_kps,
                "image": image,
            }
        
        prev_pts = np.array(
            [self._prev_keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        curr_pts = np.array([curr_kps[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        estimate = self.pose_estimator.estimate(prev_pts, curr_pts)
        R, t = estimate["R"], estimate["t"].reshape(-1)
        inliers_mask = estimate["inliers_mask"]

        self._R_global = R @ self._R_global
        self._t_global = R @ self._t_global + t

        curr_kf = Keyframe(
            idx=i,
            keypoints=curr_kps,
            descriptors=curr_descs,
            pose=_Rt_to_T(self._R_global, self._t_global),
            match_scores=match_scores,
        )
        self.mapper.add_keyframe(curr_kf)

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
            self.mapper.pointmap.add_observation(
                pt_id, self.mapper.previous_keyframe.idx, cand["uv_prev"]
            )
            self.mapper.pointmap.add_observation(
                pt_id, self.mapper.current_keyframe.idx, cand["uv_curr"]
            )

        if i % cfg_ba_freq == 0 and i >= cfg_ba_min:
            self.bundle_adjustment.run(self.mapper, self.K)

        self._R_global = self.mapper.current_keyframe.R.copy()
        self._t_global = self.mapper.current_keyframe.t.copy()

        camera_pos = -self._R_global.T @ self._t_global
        self.trajectory.append(camera_pos)

        self.mapper.pointmap.save_pointcloud(self.clouds_dir / f"{i}.ply")

        self._prev_keypoints = curr_kps
        self._prev_descriptors = curr_descs
        self._prev_feats = curr_feats
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

        pair = cv.hconcat([prev_image, image])
        match_img = cv.drawMatches(
            prev_image,
            self.mapper.keyframes[-2].keypoints,
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


def _sorted_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(folder.glob(ext))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return sorted(paths)


def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SLAM pipeline (extraction + matching + local BA)")

    parser.add_argument(
        "--images", type=Path, required=True, help="Directory containing sequential images"
    )
    parser.add_argument(
        "--fx", type=float, required=True, help="Focal length x (pixels)"
    )
    parser.add_argument(
        "--fy", type=float, required=True, help="Focal length y (pixels)"
    )
    parser.add_argument(
        "--cx", type=float, required=True, help="Principal point x (pixels)"
    )
    parser.add_argument(
        "--cy", type=float, required=True, help="Principal point y (pixels)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("clouds"), help="Output directory for point clouds"
    )
    parser.add_argument(
        "--skip", type=int, default=1, help="Process every Nth frame (default: 1, no skipping)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after this many frames"
    )
    parser.add_argument(
        "--ba_frequency", type=int, default=5, help="Run local BA every N frames"
    )
    parser.add_argument(
        "--ba_min_frames", type=int, default=12, help="Minimum frames before running local BA"
    )
    parser.add_argument(
        "--use-match-confidence", action="store_true", help="Enable confidence-weighted residuals (Sprint A)"
    )
    parser.add_argument(
        "--confidence-gamma", type=float, default=1.0, help="Match confidence exponent (default 1.0)"
    )
    parser.add_argument(
        "--use-adaptive-barron", action="store_true", help="Enable adaptive Barron loss from entropy (Sprint B)"
    )
    parser.add_argument(
        "--barron-alpha-min", type=float, default=-2.0, help="Min Barron alpha (dark scenes)"
    )
    parser.add_argument(
        "--barron-alpha-max", type=float, default=2.0, help="Max Barron alpha (bright scenes)"
    )
    parser.add_argument(
        "--no-ros", action="store_true", help="Run without ROS publishing"
    )
    args = parser.parse_args()

    image_paths = _sorted_images(args.images)

    if args.skip > 1:
        image_paths = image_paths[::args.skip]
        print(f"[SLAM] Frame skipping: keeping every {args.skip}th frame")

    if args.max_frames:
        image_paths = image_paths[: args.max_frames]

    print(f"[SLAM] Loaded {len(image_paths)} image paths")

    images: list[np.ndarray] = []
    for p in image_paths:
        img = cv.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read {p}, skipping")
            continue
        images.append(img)

    if len(images) < 2:
        sys.exit("[SLAM] Need at least 2 readable images to run.")

    print(f"[SLAM] Loaded {len(images)} valid images")

    K = build_K(args.fx, args.fy, args.cx, args.cy)

    ros_publisher = None
    if not args.no_ros:
        try:
            from sulllam.utils.ros import ROSPublisherWrapper
            ros_publisher = ROSPublisherWrapper()
            print("[ROS] Publisher initialised — streaming to ROS 2 topics")
        except Exception as exc:
            print(f"[WARN] Could not initialise ROS publisher ({exc}). "
                  "Running without ROS. Use --no-ros to suppress this warning.")

    slam = MinimalSLAMPipeline(
        K=K,
        ba_frequency=args.ba_frequency,
        ba_min_frames=args.ba_min_frames,
        clouds_dir=args.output,
        use_match_confidence=args.use_match_confidence,
        confidence_gamma=args.confidence_gamma,
        use_adaptive_barron=args.use_adaptive_barron,
        barron_alpha_min=args.barron_alpha_min,
        barron_alpha_max=args.barron_alpha_max,
    )

    try:
        trajectory = slam.run(images, ros_publisher=ros_publisher)
    finally:
        if ros_publisher is not None:
            ros_publisher.shutdown()

    print(f"\n[SLAM] Pipeline complete")
    print(f"[SLAM] Total frames: {len(images)}")
    print(f"[SLAM] Keyframes: {len(slam.mapper.keyframes)}")
    print(f"[SLAM] 3D points: {slam.mapper.pointmap.num_points}")
    print(f"[SLAM] Observations: {slam.mapper.pointmap.num_observations}")
    print(f"[SLAM] Point clouds saved to: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_file = args.output / "trajectory.npy"
    np.save(trajectory_file, trajectory)
    print(f"[SLAM] Trajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    main()

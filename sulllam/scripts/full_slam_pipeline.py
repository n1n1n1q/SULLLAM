from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import cv2 as cv
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from sulllam.mapping.bundle_adjustment.local_bundle_adjustment import (
    LocalBundleAdjustment,
    LocalBundleAdjustmentConfig,
)
from sulllam.mapping.bundle_adjustment.global_bundle_adjustment import (
    GlobalBundleAdjustment,
    GlobalBundleAdjustmentConfig,
)
from sulllam.mapping.loop_closure import LoopClosureConfig, LoopClosureDetector
from sulllam.mapping.pose_graph import PoseGraphOptimizer, PoseGraphOptimizerConfig
from sulllam.pipeline import SLAMConfig, SLAMPipeline
from sulllam.utils.ros import ROSPublisherWrapper


def _sorted_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(folder.glob(ext))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return sorted(paths)


def _build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("clouds"))
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ba-frequency", type=int, default=5)
    parser.add_argument("--ba-min-frames", type=int, default=12)
    parser.add_argument("--gba-min-frames", type=int, default=30)
    parser.add_argument("--lc-frequency", type=int, default=10)
    parser.add_argument("--lc-min-matches", type=int, default=30)
    parser.add_argument("--lc-min-frame-gap", type=int, default=20)
    parser.add_argument("--lc-min-inlier-ratio", type=float, default=0.3)
    parser.add_argument("--pgo-max-iterations", type=int, default=20)
    parser.add_argument("--use-match-confidence", action="store_true")
    parser.add_argument("--confidence-gamma", type=float, default=1.0)
    parser.add_argument("--use-adaptive-barron", action="store_true")
    parser.add_argument("--barron-alpha-min", type=float, default=-2.0)
    parser.add_argument("--barron-alpha-max", type=float, default=2.0)
    parser.add_argument("--no-ros", action="store_true")
    return parser.parse_args()


def _load_images(image_paths: list[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for p in image_paths:
        img = cv.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read {p}, skipping")
            continue
        images.append(img)
    return images


def _build_config(args: argparse.Namespace, K: np.ndarray) -> SLAMConfig:
    ba_cfg = LocalBundleAdjustmentConfig(
        use_match_confidence=args.use_match_confidence,
        confidence_gamma=args.confidence_gamma,
        use_adaptive_barron=args.use_adaptive_barron,
        barron_alpha_min=args.barron_alpha_min,
        barron_alpha_max=args.barron_alpha_max,
    )
    gba_cfg = GlobalBundleAdjustmentConfig()
    lc_cfg = LoopClosureConfig(
        min_matches=args.lc_min_matches,
        min_frame_gap=args.lc_min_frame_gap,
        min_inlier_ratio=args.lc_min_inlier_ratio,
    )
    pgo_cfg = PoseGraphOptimizerConfig(max_iterations=args.pgo_max_iterations)
    return SLAMConfig(
        K=K,
        bundle_adjustment=LocalBundleAdjustment(ba_cfg),
        global_bundle_adjustment=GlobalBundleAdjustment(gba_cfg),
        loop_closure_detector=LoopClosureDetector(lc_cfg),
        pose_graph_optimizer=PoseGraphOptimizer(pgo_cfg),
        ba_frequency=args.ba_frequency,
        ba_min_frames=args.ba_min_frames,
        gba_min_frames=args.gba_min_frames,
        lc_frequency=args.lc_frequency,
        clouds_dir=args.output,
    )


def main() -> None:
    args = _build_args()
    image_paths = _sorted_images(args.images)
    if args.skip > 1:
        image_paths = image_paths[:: args.skip]
        print(f"[SLAM] Frame skipping: keeping every {args.skip}th frame")
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
    print(f"[SLAM] Loaded {len(image_paths)} image paths")
    images = _load_images(image_paths)
    if len(images) < 2:
        sys.exit("[SLAM] Need at least 2 readable images to run.")
    print(f"[SLAM] Loaded {len(images)} valid images")
    K = _build_K(args.fx, args.fy, args.cx, args.cy)
    config = _build_config(args, K)
    ros_publisher = None
    if not args.no_ros:
        ros_publisher = ROSPublisherWrapper()
        print("[ROS] Publisher initialised — streaming to ROS 2 topics")
    pipeline = SLAMPipeline(config)
    try:
        trajectory = pipeline.run(images, ros_publisher=ros_publisher)
    finally:
        if ros_publisher is not None:
            ros_publisher.shutdown()
    print(f"\n[SLAM] Pipeline complete")
    print(f"[SLAM] Total frames    : {len(images)}")
    print(f"[SLAM] Keyframes       : {len(pipeline.mapper.keyframes)}")
    print(f"[SLAM] 3D points       : {pipeline.mapper.pointmap.num_points}")
    print(f"[SLAM] Observations    : {pipeline.mapper.pointmap.num_observations}")
    print(
        f"[SLAM] Pose-graph edges: {len(pipeline.pose_graph.edges)} ({len(pipeline.pose_graph.loop_closure_edges)} loop closures)"
    )
    print(f"[SLAM] Point clouds saved to: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_file = args.output / "trajectory.npy"
    np.save(trajectory_file, trajectory)
    print(f"[SLAM] Trajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    main()

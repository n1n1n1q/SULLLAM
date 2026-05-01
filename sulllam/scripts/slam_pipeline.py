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
from sulllam.localization.matching.lightglue import LightGlueMatcher, LightGlueConfig
from sulllam.localization.matching.match_filter import MatchFilterConfig, RANSACModel
from sulllam.pipeline import SLAMConfig, SLAMPipeline
from sulllam.preprocessing.dynamic_filter import (
    DynamicFilterConfig,
    DynamicKeypointFilter,
)


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
    parser = argparse.ArgumentParser(
        description=(
            "Full SLAM pipeline (extraction + matching + local BA + loop closure "
            "+ pose-graph optimisation + global BA)."
        )
    )

    parser.add_argument("--images", type=Path, required=True,
                        help="Directory containing sequential images")
    parser.add_argument("--fx", type=float, required=True, help="Focal length x (px)")
    parser.add_argument("--fy", type=float, required=True, help="Focal length y (px)")
    parser.add_argument("--cx", type=float, required=True, help="Principal point x (px)")
    parser.add_argument("--cy", type=float, required=True, help="Principal point y (px)")

    parser.add_argument("--output", type=Path, default=Path("clouds"),
                        help="Output directory for point clouds and trajectory")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (default: 1, no skipping)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames")

    parser.add_argument("--ba-frequency", type=int, default=5,
                        help="Run local BA every N frames")
    parser.add_argument("--ba-min-frames", type=int, default=12,
                        help="Minimum frames before running local BA")
    parser.add_argument("--gba-min-frames", type=int, default=30,
                        help="Minimum frames before running global BA")

    parser.add_argument("--lc-frequency", type=int, default=10,
                        help="Run loop-closure detection every N frames")
    parser.add_argument("--lc-min-matches", type=int, default=30,
                        help="Minimum descriptor matches required for a LC candidate")
    parser.add_argument("--lc-min-frame-gap", type=int, default=20,
                        help="Minimum frame gap to consider a keyframe pair as LC")
    parser.add_argument("--lc-min-inlier-ratio", type=float, default=0.3,
                        help="Minimum essential-matrix inlier ratio for a LC candidate")

    parser.add_argument("--pgo-max-iterations", type=int, default=20,
                        help="Max LM iterations for pose-graph optimisation")

    parser.add_argument("--use-match-confidence", action="store_true",
                        help="Enable confidence-weighted residuals in BA")
    parser.add_argument("--confidence-gamma", type=float, default=1.0,
                        help="Match confidence exponent")
    parser.add_argument("--use-adaptive-barron", action="store_true",
                        help="Enable adaptive Barron loss from entropy in BA")
    parser.add_argument("--barron-alpha-min", type=float, default=-2.0,
                        help="Min Barron alpha (dark scenes)")
    parser.add_argument("--barron-alpha-max", type=float, default=2.0,
                        help="Max Barron alpha (bright scenes)")

    parser.add_argument("--no-ros", action="store_true",
                        help="Run without ROS publishing")

    parser.add_argument("--dynamic-filter", action="store_true",
                        help="Enable YOLO + Depth-Anything dynamic keypoint filter")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt",
                        help="YOLO weights path or model name")
    parser.add_argument("--yolo-conf", type=float, default=0.35,
                        help="YOLO confidence threshold")
    parser.add_argument("--dynamic-classes", type=int, nargs="*", default=None,
                        help="COCO class ids treated as dynamic (default: person/bike/car/moto/bus/truck)")
    parser.add_argument("--depth-model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HuggingFace depth-estimation model id")
    parser.add_argument("--df-bbox-shrink", type=float, default=0.85,
                        help="Bbox shrink ratio before keypoint filtering")
    parser.add_argument("--df-fg-percentile", type=float, default=25.0,
                        help="Foreground depth percentile inside bbox")
    parser.add_argument("--df-margin-scale", type=float, default=0.05,
                        help="Margin = scale * (p95 - p05) of bbox depth")
    parser.add_argument("--df-run-every", type=int, default=1,
                        help="Run YOLO+depth every N frames (reuse last otherwise)")

    # Match filtering
    parser.add_argument("--match-filter", action="store_true",
                        help="Enable spatial dedup + RANSAC match filtering")
    parser.add_argument("--mf-ransac-model", type=str, default="essential",
                        choices=["essential", "fundamental", "homography"],
                        help="RANSAC model for geometric verification")
    parser.add_argument("--mf-ransac-threshold", type=float, default=1.0,
                        help="RANSAC reprojection threshold (px)")
    parser.add_argument("--mf-spatial-radius", type=float, default=4.0,
                        help="Spatial dedup radius (px); 0 to disable")

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
    pgo_cfg = PoseGraphOptimizerConfig(
        max_iterations=args.pgo_max_iterations,
    )

    dynamic_filter = None
    if args.dynamic_filter:
        df_cfg = DynamicFilterConfig(
            enabled=True,
            yolo_weights=args.yolo_weights,
            yolo_conf=args.yolo_conf,
            depth_model=args.depth_model,
            bbox_shrink=args.df_bbox_shrink,
            foreground_percentile=args.df_fg_percentile,
            margin_scale=args.df_margin_scale,
            run_every_n=args.df_run_every,
        )
        if args.dynamic_classes is not None:
            df_cfg.dynamic_classes = tuple(args.dynamic_classes)
        dynamic_filter = DynamicKeypointFilter(df_cfg)
        print(f"[SLAM] Dynamic-keypoint filter ON "
              f"(yolo={args.yolo_weights}, depth={args.depth_model})")

    matcher = None
    mf_cfg = None
    if args.match_filter:
        ransac_model = RANSACModel(args.mf_ransac_model)
        mf_cfg = MatchFilterConfig(
            spatial_dedup_radius=args.mf_spatial_radius,
            ransac_enabled=True,
            ransac_model=ransac_model,
            ransac_threshold=args.mf_ransac_threshold,
            K=K if ransac_model == RANSACModel.ESSENTIAL else None,
        )
        print(f"[SLAM] Match filter ON "
              f"(ransac={args.mf_ransac_model}, threshold={args.mf_ransac_threshold}, "
              f"spatial_radius={args.mf_spatial_radius})")

    if mf_cfg is not None:
        matcher = LightGlueMatcher(LightGlueConfig(match_filter=mf_cfg))

    return SLAMConfig(
        K=K,
        matcher=matcher,
        bundle_adjustment=LocalBundleAdjustment(ba_cfg),
        global_bundle_adjustment=GlobalBundleAdjustment(gba_cfg),
        loop_closure_detector=LoopClosureDetector(lc_cfg),
        pose_graph_optimizer=PoseGraphOptimizer(pgo_cfg),
        dynamic_filter=dynamic_filter,
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
        try:
            from sulllam.utils.ros import ROSPublisherWrapper
            ros_publisher = ROSPublisherWrapper()
            print("[ROS] Publisher initialised — streaming to ROS 2 topics")
        except Exception as exc:
            print(f"[WARN] Could not initialise ROS publisher ({exc}). "
                  "Running without ROS. Use --no-ros to suppress this warning.")

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
    print(f"[SLAM] Pose-graph edges: {len(pipeline.pose_graph.edges)} "
          f"({len(pipeline.pose_graph.loop_closure_edges)} loop closures)")
    print(f"[SLAM] Point clouds saved to: {args.output}")

    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_file = args.output / "trajectory.npy"
    np.save(trajectory_file, trajectory)
    print(f"[SLAM] Trajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    main()

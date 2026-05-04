import argparse
import cv2 as cv
import numpy as np
from pathlib import Path

from sulllam.pipeline import SLAMPipeline, SLAMConfig
from sulllam.mapping.bundle_adjustment.local_bundle_adjustment import LocalBundleAdjustment, LocalBundleAdjustmentConfig
from sulllam.utils.ros import ROSPublisherWrapper
from sulllam.segmentation import SAMSegmentor, COCO_DYNAMIC_CLASSES


K = np.array([
    [1013.535131,    0.0,  638.295572],
    [   0.0,   1060.724658,  399.313666],
    [   0.0,    0.0,    1.0]
])


def parse_args():
    parser = argparse.ArgumentParser(description="Run SULLLAM front-end pipeline")
    parser.add_argument("image_dir", type=Path, help="Directory with input PNG images")
    parser.add_argument("--step", type=int, default=15, help="Frame stride when loading images (default: 15)")
    parser.add_argument("--clouds-dir", type=Path, default=Path("clouds"), help="Output directory for point clouds")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam2_hiera_tiny.pt")
    parser.add_argument("--sam-cfg", default="sam2_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    images = [cv.imread(str(p)) for p in sorted(args.image_dir.glob("*.png"))[::args.step]]
    print(f"Loaded {len(images)} images from {args.image_dir}")

    config = SLAMConfig(
        K=K,
        bundle_adjustment=LocalBundleAdjustment(
            LocalBundleAdjustmentConfig(window_size=10, max_iterations=20, huber_radius=1.0)
        ),
        segmentor=SAMSegmentor(
            sam2_checkpoint=args.sam_checkpoint,
            sam2_model_cfg=args.sam_cfg,
            device=args.device,
            class_ids=COCO_DYNAMIC_CLASSES
        ),
        max_reproj_error=2.0,
        max_depth=50.0,
        max_points=200,
        ba_frequency=5,
        ba_min_frames=15,
        clouds_dir=args.clouds_dir,
    )

    ros_publisher = ROSPublisherWrapper()
    try:
        pipeline = SLAMPipeline(config)
        trajectory = pipeline.run(images, ros_publisher=ros_publisher)
    finally:
        ros_publisher.shutdown()

    return trajectory


if __name__ == "__main__":
    main()

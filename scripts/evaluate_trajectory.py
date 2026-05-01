#!/usr/bin/env python3
"""Evaluate SLAM trajectory quality against COLMAP ground-truth poses.

Loads a COLMAP sparse model (images.txt or images.bin), runs the SLAM pipeline
on the same images, aligns the estimated trajectory to ground-truth via a Sim(3)
transform (Umeyama algorithm), then reports the Absolute Trajectory Error (ATE).

Usage:
    python scripts/evaluate_trajectory.py \
        --images /data/scene/images \
        --colmap-dir /data/scene/sparse/0 \
        --fx 525.0 --fy 525.0 --cx 320.0 --cy 240.0
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent))

from sulllam.mapping.bundle_adjustment.global_bundle_adjustment import (
    GlobalBundleAdjustment,
    GlobalBundleAdjustmentConfig,
)
from sulllam.mapping.bundle_adjustment.local_bundle_adjustment import (
    LocalBundleAdjustment,
    LocalBundleAdjustmentConfig,
)
from sulllam.mapping.loop_closure import LoopClosureConfig, LoopClosureDetector
from sulllam.mapping.pose_graph import PoseGraphOptimizer, PoseGraphOptimizerConfig
from sulllam.pipeline import SLAMConfig, SLAMPipeline


# ---------------------------------------------------------------------------
# COLMAP I/O
# ---------------------------------------------------------------------------

def _parse_images_txt(path: Path) -> dict[str, np.ndarray]:
    """Return {image_name: camera_center_xyz} from a COLMAP images.txt.

    Format (alternating lines):
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      POINTS2D[] (skipped)
    """
    poses: dict[str, np.ndarray] = {}
    with open(path) as f:
        data_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for i in range(0, len(data_lines), 2):
        parts = data_lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        name = parts[9]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy: (x,y,z,w)
        poses[name] = -(R.T @ np.array([tx, ty, tz]))          # world-frame camera center
    return poses


def _parse_images_bin(path: Path) -> dict[str, np.ndarray]:
    """Return {image_name: camera_center_xyz} from a COLMAP images.bin."""
    poses: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            f.read(4)  # image_id
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            f.read(4)  # camera_id
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode()
            (num_pts2d,) = struct.unpack("<Q", f.read(8))
            f.read(num_pts2d * 24)  # (x:f64, y:f64, point3d_id:i64)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses[name] = -(R.T @ np.array([tx, ty, tz]))
    return poses


def load_colmap_poses(colmap_dir: Path) -> dict[str, np.ndarray]:
    txt = colmap_dir / "images.txt"
    bin_ = colmap_dir / "images.bin"
    if txt.exists():
        print(f"[COLMAP] Reading {txt}")
        return _parse_images_txt(txt)
    if bin_.exists():
        print(f"[COLMAP] Reading {bin_}")
        return _parse_images_bin(bin_)
    raise FileNotFoundError(f"No images.txt or images.bin in {colmap_dir}")


# ---------------------------------------------------------------------------
# Sim(3) alignment — Umeyama (1991)
# ---------------------------------------------------------------------------

def umeyama_alignment(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Find s, R, t minimising ||dst - (s*R@src + t)||^2 over N point pairs.

    Args:
        src: (N, 3) SLAM camera centres
        dst: (N, 3) ground-truth camera centres

    Returns:
        s  – scale factor
        R  – (3, 3) rotation matrix
        t  – (3,) translation vector
    """
    n = len(src)
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    var_src = (src_c ** 2).sum() / n                   # scalar variance
    cov = (dst_c.T @ src_c) / n                        # 3×3 cross-covariance

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    s = float(D @ np.diag(S)) / var_src                # trace(D*S) / var_src
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def apply_sim3(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return s * (pts @ R.T) + t


# ---------------------------------------------------------------------------
# ATE
# ---------------------------------------------------------------------------

def compute_ate(aligned: np.ndarray, gt: np.ndarray) -> dict:
    errors = np.linalg.norm(aligned - gt, axis=1)
    return {
        "rmse":      float(np.sqrt((errors ** 2).mean())),
        "mean":      float(errors.mean()),
        "median":    float(np.median(errors)),
        "std":       float(errors.std()),
        "max":       float(errors.max()),
        "min":       float(errors.min()),
        "per_frame": errors,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(folder.glob(ext))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return sorted(paths)


def _load_images(
    image_paths: list[Path],
) -> tuple[list[np.ndarray], list[Path]]:
    images, valid_paths = [], []
    for p in image_paths:
        img = cv.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read {p}, skipping")
        else:
            images.append(img)
            valid_paths.append(p)
    return images, valid_paths


def _build_config(args: argparse.Namespace, K: np.ndarray) -> SLAMConfig:
    return SLAMConfig(
        K=K,
        bundle_adjustment=LocalBundleAdjustment(
            LocalBundleAdjustmentConfig(window_size=10, max_iterations=20, huber_radius=1.0)
        ),
        max_reproj_error=2.0,
        max_depth=50.0,
        max_points=200,
        ba_frequency=5,
        ba_min_frames=15,
        clouds_dir=args.clouds_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SLAM and evaluate ATE against COLMAP ground-truth poses."
    )
    p.add_argument("--images",     type=Path, required=True,
                   help="Directory of sequential images")
    p.add_argument("--colmap-dir", type=Path, required=True,
                   help="COLMAP sparse model dir (contains images.txt / images.bin)")
    p.add_argument("--fx", type=float, required=True)
    p.add_argument("--fy", type=float, required=True)
    p.add_argument("--cx", type=float, required=True)
    p.add_argument("--cy", type=float, required=True)
    p.add_argument("--skip",       type=int,  default=1,
                   help="Use every Nth image (default: 1)")
    p.add_argument("--max-frames", type=int,  default=None)
    p.add_argument("--output",     type=Path, default=Path("eval_output"),
                   help="Output directory for trajectory, plot, and per-frame errors")
    p.add_argument("--no-plot",    action="store_true",
                   help="Skip saving the trajectory comparison plot")
    return p.parse_args()


def main() -> None:
    args = _build_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ---- ground truth ----
    gt_poses = load_colmap_poses(args.colmap_dir)
    print(f"[GT]   {len(gt_poses)} poses loaded from COLMAP")

    # ---- images ----
    image_paths = _sorted_images(args.images)
    if args.skip > 1:
        image_paths = image_paths[:: args.skip]
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]

    images, valid_paths = _load_images(image_paths)
    print(f"[SLAM] {len(images)} images loaded")
    if len(images) < 2:
        sys.exit("[SLAM] Need at least 2 readable images.")

    # ---- match frame index → GT pose ----
    matched_idx: list[int] = []
    matched_gt:  list[np.ndarray] = []
    for i, p in enumerate(valid_paths):
        if p.name in gt_poses:
            matched_idx.append(i)
            matched_gt.append(gt_poses[p.name])
    print(f"[EVAL] {len(matched_idx)} frames matched to COLMAP poses")
    if len(matched_idx) < 3:
        sys.exit("[EVAL] Too few matched frames — verify image filenames match COLMAP names")

    # ---- run SLAM ----
    K = np.array([[args.fx, 0, args.cx],
                  [0, args.fy, args.cy],
                  [0,       0,       1]], dtype=np.float64)
    pipeline = SLAMPipeline(_build_config(args, K))

    print("[SLAM] Running pipeline …")
    trajectory = pipeline.run(images, ros_publisher=None)
    np.save(args.output / "trajectory_slam.npy", trajectory)
    print(f"[SLAM] Trajectory: {trajectory.shape}")

    # ---- Sim(3) alignment ----
    slam_matched = trajectory[matched_idx]   # (M, 3)
    gt_matched   = np.array(matched_gt)      # (M, 3)

    s, R_align, t_align = umeyama_alignment(slam_matched, gt_matched)
    print(f"[ALIGN] Sim(3) scale = {s:.6f}")

    slam_aligned_matched = apply_sim3(s, R_align, t_align, slam_matched)
    slam_aligned_full    = apply_sim3(s, R_align, t_align, trajectory)

    np.save(args.output / "trajectory_slam_aligned.npy", slam_aligned_full)

    # ---- ATE ----
    stats = compute_ate(slam_aligned_matched, gt_matched)

    print()
    print("===== Absolute Trajectory Error (ATE) =====")
    print(f"  Frames evaluated : {len(matched_idx)}")
    print(f"  RMSE   : {stats['rmse']:.4f} m")
    print(f"  Mean   : {stats['mean']:.4f} m")
    print(f"  Median : {stats['median']:.4f} m")
    print(f"  Std    : {stats['std']:.4f} m")
    print(f"  Max    : {stats['max']:.4f} m")
    print(f"  Min    : {stats['min']:.4f} m")
    print("============================================")

    np.save(args.output / "ate_per_frame.npy", stats["per_frame"])

    # ---- optional plot ----
    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Top-down trajectory (X-Z)
        ax = axes[0]
        ax.plot(gt_matched[:, 0], gt_matched[:, 2],
                "g-",  linewidth=1.5, label="Ground truth")
        ax.plot(slam_aligned_matched[:, 0], slam_aligned_matched[:, 2],
                "r--", linewidth=1.5, label="SLAM (aligned)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Trajectory (top-down X-Z)")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Per-frame ATE
        ax2 = axes[1]
        ax2.plot(stats["per_frame"], "b-", linewidth=1, label="Per-frame error")
        ax2.axhline(stats["rmse"], color="r", linestyle="--",
                    label=f"RMSE = {stats['rmse']:.4f} m")
        ax2.set_xlabel("Matched frame index")
        ax2.set_ylabel("Translation error (m)")
        ax2.set_title("Per-frame Absolute Trajectory Error")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = args.output / "trajectory_eval.png"
        plt.savefig(plot_path, dpi=150)
        print(f"[EVAL] Plot saved to {plot_path}")

    except ImportError:
        print("[WARN] matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()

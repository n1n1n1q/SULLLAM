from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from sulllam.localization.extraction.orb import ORBFeatureExtractor
from sulllam.localization.matching.bf import BFFeatureMatcher, BFMatcherConfig
from sulllam.localization.pose_estimation.homography import HomographyPoseEstimator
from sulllam.utils.io import read_image


def estimate_homography_fallback(kp_query, kp_train, matches, ransac_reproj_threshold: float = 3.0):
	if matches is None or len(matches) < 4:
		return {
			"success": False,
			"reason": "not_enough_matches",
			"homography": None,
			"inlier_mask": None,
			"inlier_matches": [],
			"num_matches": 0 if matches is None else len(matches),
			"num_inliers": 0,
			"inlier_ratio": 0.0,
		}

	src_pts = np.float32([kp_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp_train[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

	homography, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_reproj_threshold)
	if homography is None or inlier_mask is None:
		return {
			"success": False,
			"reason": "homography_failed",
			"homography": None,
			"inlier_mask": None,
			"inlier_matches": [],
			"num_matches": len(matches),
			"num_inliers": 0,
			"inlier_ratio": 0.0,
		}

	inlier_mask_flat = inlier_mask.ravel().astype(bool)
	inlier_matches = [m for m, is_inlier in zip(matches, inlier_mask_flat) if is_inlier]
	num_inliers = len(inlier_matches)

	return {
		"success": True,
		"reason": "ok",
		"homography": homography,
		"inlier_mask": inlier_mask_flat,
		"inlier_matches": inlier_matches,
		"num_matches": len(matches),
		"num_inliers": num_inliers,
		"inlier_ratio": 0.0 if len(matches) == 0 else num_inliers / len(matches),
	}


def draw_localization_overlay(train_image, query_image, homography):
	h, w = query_image.shape[:2]
	corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
	projected = cv.perspectiveTransform(corners, homography)
	overlay = train_image.copy()
	cv.polylines(overlay, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)
	return overlay


def main() -> None:
	parser = argparse.ArgumentParser(description="Test ORB + BF + homography localization pipeline")
	parser.add_argument("query_image", type=Path, help="Path to query/object image")
	parser.add_argument("train_image", type=Path, help="Path to train/scene image")
	parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold")
	parser.add_argument("--save", type=Path, default=None, help="Optional output path for localized image")
	parser.add_argument("--show", action="store_true", help="Show match and localization windows")
	args = parser.parse_args()

	query_image = read_image(args.query_image)
	train_image = read_image(args.train_image)

	if query_image is None:
		raise FileNotFoundError(f"Could not read query image: {args.query_image}")
	if train_image is None:
		raise FileNotFoundError(f"Could not read train image: {args.train_image}")

	extractor = ORBFeatureExtractor()

	# The extractor currently returns keypoints and descriptors from _extract.
	kp_query, des_query = extractor._extract(query_image)
	kp_train, des_train = extractor._extract(train_image)

	matcher = BFFeatureMatcher(BFMatcherConfig(ratio_threshold=args.ratio))
	matches = matcher.match(des_query, des_train)

	estimator = HomographyPoseEstimator()
	try:
		result = estimator.estimate(kp_query, kp_train, matches)
	except Exception as exc:
		print(f"Estimator fallback due to runtime error: {exc}")
		result = estimate_homography_fallback(kp_query, kp_train, matches)

	print(f"Keypoints query: {len(kp_query)}")
	print(f"Keypoints train: {len(kp_train)}")
	print(f"Matches: {result.get('num_matches', len(matches))}")
	print(f"Inliers: {result.get('num_inliers', 0)}")
	print(f"Inlier ratio: {result.get('inlier_ratio', 0.0):.4f}")
	print(f"Success: {result.get('success', False)} ({result.get('reason', 'unknown')})")

	if not result.get("success", False):
		return

	homography = result["homography"]
	localized = draw_localization_overlay(train_image, query_image, homography)
	inlier_matches = result.get("inlier_matches", matches)

	match_vis = cv.drawMatches(
		query_image,
		kp_query,
		train_image,
		kp_train,
		inlier_matches,
		None,
		flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
	)

	if args.save is not None:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		cv.imwrite(str(args.save), localized)
		print(f"Saved localization image to: {args.save}")

	if args.show:
		cv.imshow("Localization", localized)
		cv.imshow("Inlier Matches", match_vis)
		cv.waitKey(0)
		cv.destroyAllWindows()


if __name__ == "__main__":
	main()

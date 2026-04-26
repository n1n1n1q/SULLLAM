from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sulllam.localization.extraction.base_extractor import BaseExtractor
from sulllam.localization.matching.base_matcher import BaseMatcher
from sulllam.localization.pose_estimation.base_estimator import BaseEstimator
from sulllam.mapping.bundle_adjustment.base_bundle_adjustment import BaseBundleAdjustment
from sulllam.mapping.bundle_adjustment.local_bundle_adjustment import (
    LocalBundleAdjustment,
    LocalBundleAdjustmentConfig,
)


@dataclass
class SLAMConfig:
    K: np.ndarray

    extractor: BaseExtractor = field(default=None)
    matcher: BaseMatcher = field(default=None)
    pose_estimator: BaseEstimator = field(default=None)
    bundle_adjustment: BaseBundleAdjustment = field(default=None)

    max_reproj_error: float = 2.0
    max_depth: float = 50.0
    max_points: int = 100

    ba_frequency: int = 5
    ba_min_frames: int = 12

    clouds_dir: Path = field(default_factory=lambda: Path("clouds"))

    def __post_init__(self):
        from sulllam.localization.extraction.orb import ORBFeatureExtractor
        from sulllam.localization.matching.bf import BFFeatureMatcher, BFMatcherConfig
        from sulllam.localization.pose_estimation.eight_point_estimator import (
            EightPointPoseEstimator,
            EightPointEstimatorConfig,
        )

        if self.extractor is None:
            self.extractor = ORBFeatureExtractor()

        if self.matcher is None:
            self.matcher = BFFeatureMatcher(config=BFMatcherConfig())

        if self.pose_estimator is None:
            self.pose_estimator = EightPointPoseEstimator(
                config=EightPointEstimatorConfig(K=self.K)
            )

        if self.bundle_adjustment is None:
            self.bundle_adjustment = LocalBundleAdjustment(LocalBundleAdjustmentConfig())

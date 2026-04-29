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
from sulllam.mapping.bundle_adjustment.global_bundle_adjustment import (
    GlobalBundleAdjustment,
    GlobalBundleAdjustmentConfig,
)
from sulllam.mapping.loop_closure import LoopClosureDetector, LoopClosureConfig
from sulllam.mapping.pose_graph import PoseGraphOptimizer, PoseGraphOptimizerConfig


@dataclass
class SLAMConfig:
    K: np.ndarray

    extractor: BaseExtractor = field(default=None)
    matcher: BaseMatcher = field(default=None)
    pose_estimator: BaseEstimator = field(default=None)
    bundle_adjustment: BaseBundleAdjustment = field(default=None)
    global_bundle_adjustment: BaseBundleAdjustment = field(default=None)
    loop_closure_detector: LoopClosureDetector = field(default=None)
    pose_graph_optimizer: PoseGraphOptimizer = field(default=None)

    max_reproj_error: float = 2.0
    max_depth: float = 50.0
    max_points: int = 100

    ba_frequency: int = 5
    ba_min_frames: int = 12
    gba_min_frames: int = 30

    lc_frequency: int = 10

    clouds_dir: Path = field(default_factory=lambda: Path("clouds"))

    def __post_init__(self):
        from sulllam.localization.extraction.superpoint import (
            SuperPointFeatureExtractor,
            SuperPointConfig,
        )
        from sulllam.localization.matching.lightglue import (
            LightGlueMatcher,
            LightGlueConfig,
        )
        from sulllam.localization.pose_estimation.eight_point_estimator import (
            EightPointPoseEstimator,
            EightPointEstimatorConfig,
        )

        if self.extractor is None:
            self.extractor = SuperPointFeatureExtractor(SuperPointConfig())

        if self.matcher is None:
            self.matcher = LightGlueMatcher(LightGlueConfig())

        if self.pose_estimator is None:
            self.pose_estimator = EightPointPoseEstimator(
                config=EightPointEstimatorConfig(K=self.K)
            )

        if self.bundle_adjustment is None:
            self.bundle_adjustment = LocalBundleAdjustment(LocalBundleAdjustmentConfig())

        if self.global_bundle_adjustment is None:
            self.global_bundle_adjustment = GlobalBundleAdjustment(
                GlobalBundleAdjustmentConfig()
            )

        if self.loop_closure_detector is None:
            self.loop_closure_detector = LoopClosureDetector(LoopClosureConfig())

        if self.pose_graph_optimizer is None:
            self.pose_graph_optimizer = PoseGraphOptimizer(PoseGraphOptimizerConfig())

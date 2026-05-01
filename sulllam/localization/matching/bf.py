from __future__ import annotations

from dataclasses import dataclass, field

import cv2 as cv
import numpy as np

from sulllam.localization.matching.base_matcher import BaseMatcher
from sulllam.localization.matching.match_filter import MatchFilterConfig, apply_match_filters


@dataclass(slots=True)
class BFMatcherConfig:
    norm_type: int = cv.NORM_HAMMING
    cross_check: bool = False
    use_ratio_test: bool = True
    ratio_threshold: float = 0.75
    knn_k: int = 2
    sort_by_distance: bool = True
    match_filter: MatchFilterConfig | None = None


class BFFeatureMatcher(BaseMatcher):
    def __init__(self, config: BFMatcherConfig | None = None) -> None:
        super().__init__(name="BF")
        self.config = config or BFMatcherConfig()
        self.matcher = cv.BFMatcher(
            self.config.norm_type,
            crossCheck=self.config.cross_check,
        )

    def _match(self, query_descriptors, train_descriptors):
        if query_descriptors is None or train_descriptors is None:
            return [], np.array([])

        if len(query_descriptors) == 0 or len(train_descriptors) == 0:
            return [], np.array([])

        if self.config.cross_check:
            matches = self.matcher.match(query_descriptors, train_descriptors)
            sorted_matches = sorted(matches, key=lambda m: m.distance) if self.config.sort_by_distance else matches
            return sorted_matches, np.array([])

        if not self.config.use_ratio_test:
            matches = self.matcher.match(query_descriptors, train_descriptors)
            sorted_matches = sorted(matches, key=lambda m: m.distance) if self.config.sort_by_distance else matches
            return sorted_matches, np.array([])

        knn_matches = self.matcher.knnMatch(
            query_descriptors,
            train_descriptors,
            k=self.config.knn_k,
        )

        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue

            m, n = pair
            if m.distance < self.config.ratio_threshold * n.distance:
                good_matches.append(m)

        if self.config.sort_by_distance:
            good_matches.sort(key=lambda m: m.distance)

        return good_matches, np.array([])

    def match_with_keypoints(
        self,
        query_descriptors,
        train_descriptors,
        query_keypoints: list[cv.KeyPoint],
        train_keypoints: list[cv.KeyPoint],
    ) -> tuple[list[cv.DMatch], np.ndarray]:
        """Like match(), but also applies spatial dedup and RANSAC when match_filter is set."""
        matches, scores = self._match(query_descriptors, train_descriptors)
        if self.config.match_filter is not None and matches:
            matches = apply_match_filters(matches, query_keypoints, train_keypoints, self.config.match_filter)
        return matches, scores

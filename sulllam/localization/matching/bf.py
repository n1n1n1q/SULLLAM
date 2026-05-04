from __future__ import annotations
from dataclasses import dataclass
import cv2 as cv
import numpy as np
from sulllam.localization.matching.base_matcher import BaseMatcher


@dataclass(slots=True)
class BFMatcherConfig:
    norm_type: int = cv.NORM_HAMMING
    cross_check: bool = False
    use_ratio_test: bool = True
    ratio_threshold: float = 0.75
    knn_k: int = 2
    sort_by_distance: bool = True


class BFFeatureMatcher(BaseMatcher):

    def __init__(self, config: BFMatcherConfig | None = None) -> None:
        super().__init__(name="BF")
        self.config = config or BFMatcherConfig()
        self.matcher = cv.BFMatcher(
            self.config.norm_type, crossCheck=self.config.cross_check
        )

    def _match(self, query_descriptors, train_descriptors):
        if query_descriptors is None or train_descriptors is None:
            return ([], np.array([]))
        if len(query_descriptors) == 0 or len(train_descriptors) == 0:
            return ([], np.array([]))
        if self.config.cross_check:
            matches = self.matcher.match(query_descriptors, train_descriptors)
            sorted_matches = (
                sorted(matches, key=lambda m: m.distance)
                if self.config.sort_by_distance
                else matches
            )
            return (sorted_matches, np.array([]))
        if not self.config.use_ratio_test:
            matches = self.matcher.match(query_descriptors, train_descriptors)
            sorted_matches = (
                sorted(matches, key=lambda m: m.distance)
                if self.config.sort_by_distance
                else matches
            )
            return (sorted_matches, np.array([]))
        knn_matches = self.matcher.knnMatch(
            query_descriptors, train_descriptors, k=self.config.knn_k
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
        return (good_matches, np.array([]))

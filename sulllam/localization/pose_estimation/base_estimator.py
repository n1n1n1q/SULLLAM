from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEstimator(ABC):
	def __init__(self, *, name: str | None = None) -> None:
		self._name = name or self.__class__.__name__

	@property
	def name(self) -> str:
		return self._name

	def estimate(self, keypoints_query, keypoints_train, matches):
		raw_result = self._estimate(keypoints_query, keypoints_train, matches)
		return self._normalize_result(raw_result)

	@abstractmethod
	def _estimate(self, keypoints_query, keypoints_train, matches):
		raise NotImplementedError

	def _normalize_result(self, result):
		if result is None:
			return {}
		return result

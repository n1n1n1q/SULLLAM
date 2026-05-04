from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable


class BaseMatcher(ABC):

    def __init__(self, *, name: str | None = None) -> None:
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    def match(self, query_descriptors, train_descriptors) -> list:
        raw_matches = self._match(query_descriptors, train_descriptors)
        return self._normalize_matches(raw_matches)

    @abstractmethod
    def _match(self, query_descriptors, train_descriptors) -> Iterable:
        raise NotImplementedError

    def _normalize_matches(self, matches: Iterable) -> list:
        normalized: list = []
        for match in matches:
            if match is None:
                continue
            normalized.append(match)
        return normalized

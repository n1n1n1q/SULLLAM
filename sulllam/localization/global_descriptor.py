"""Lightweight global image descriptors for loop-closure pre-filtering.

The aim is to cheaply reject keyframe pairs that are unlikely to depict the
same place before running the (more expensive) geometric loop-closure check.
This module produces a single fixed-size descriptor per keyframe by pooling
its local descriptors. With L2-normalised SuperPoint descriptors this is
equivalent to a simple "bag of features" centroid that already correlates
well with co-visibility for short trajectories — good enough as a prefilter
without pulling in NetVLAD/DBoW dependencies.
"""
from __future__ import annotations

import numpy as np


def compute_global_descriptor(
    local_descs: np.ndarray | None,
    pooling: str = "mean",
) -> np.ndarray | None:
    """Pool per-keypoint descriptors into a single L2-normalised vector.

    For SuperPoint's L2-normalised 256-D descriptors, simple mean pooling
    followed by re-normalisation yields a centroid that's both cheap and
    reasonably discriminative across visually distinct scenes. We avoid
    GeM-with-clip-to-positive because clipping biases all descriptors
    toward the all-ones direction and collapses cross-image cosine sims.
    """
    if local_descs is None:
        return None
    descs = np.asarray(local_descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0:
        return None

    pooled = descs.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm < 1e-12:
        return None
    return (pooled / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetVladConfig:
    descriptor_dim: int = 256
    num_clusters: int = 64
    similarity_threshold: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NetVLAD(nn.Module):
    """NetVLAD aggregation: (N, C) local descriptors → one global vector."""

    def __init__(self, num_clusters: int = 64, dim: int = 256):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C) local descriptors for one image
        Returns:
            vlad: (num_clusters * C,) normalized global descriptor
        """
        x = x.unsqueeze(0).unsqueeze(3)  # (1, C, N, 1)
        x = F.normalize(x, p=2, dim=1)

        soft_assign = self.conv(x).squeeze(0).squeeze(2)  # (K, N)
        soft_assign = F.softmax(soft_assign, dim=0)

        x_flat = x.squeeze(0).squeeze(2)  # (C, N)

        vlad = torch.zeros(self.num_clusters, self.dim, device=x.device, dtype=x.dtype)
        for k in range(self.num_clusters):
            residual = x_flat - self.centroids[k].unsqueeze(1)  # (C, N)
            residual *= soft_assign[k].unsqueeze(0)              # weight by assignment
            vlad[k] = residual.sum(dim=1)

        vlad = F.normalize(vlad, p=2, dim=1)   # intra-normalization
        vlad = vlad.flatten()
        vlad = F.normalize(vlad, p=2, dim=0)   # L2 normalize
        return vlad


class NetVladEncoder:
    """
    Encodes keyframe descriptors into global place descriptors for retrieval.
    Use this for loop closure candidate pre-selection; not for per-frame matching.
    """

    def __init__(self, config: NetVladConfig | None = None) -> None:
        self.config = config or NetVladConfig()
        self.device = torch.device(self.config.device)
        self._model = NetVLAD(
            num_clusters=self.config.num_clusters,
            dim=self.config.descriptor_dim,
        ).eval().to(self.device)

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """Aggregate local descriptors into one global vector.

        Args:
            descriptors: (N, C) float32 array of local descriptors
        Returns:
            (num_clusters * C,) float32 global descriptor
        """
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.config.num_clusters * self.config.descriptor_dim, dtype=np.float32)

        with torch.no_grad():
            t = torch.from_numpy(np.asarray(descriptors, dtype=np.float32)).to(self.device)
            return self._model(t).cpu().numpy()

    def top_k_candidates(
        self,
        query_global: np.ndarray,
        db_globals: list[np.ndarray],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return indices of top-k most similar database entries, with cosine scores.

        Args:
            query_global: (D,) global descriptor for current frame
            db_globals: list of (D,) global descriptors for database keyframes
            k: number of candidates to return
        Returns:
            sorted list of (index, similarity) pairs, best first
        """
        if not db_globals:
            return []

        db = np.stack(db_globals)                      # (M, D)
        sims = db @ query_global                       # cosine similarity (already L2-normed)

        threshold = self.config.similarity_threshold
        top_k = min(k, len(db))
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        return [(int(i), float(sims[i])) for i in top_idx if sims[i] >= threshold]

from __future__ import annotations

import numpy as np
import open3d as o3d


# Observation schema: (point_id, kf_id, u, v, kp_idx).
# kp_idx is the index of the keypoint in `Keyframe.keypoints` that produced
# this observation. Storing it lets us go from a keyframe + keypoint back to
# the map point that keypoint already corresponds to (used for keyframe
# selection, cross-loop-closure data association, and point fusion).
_OBS_COLS = 5


class PointMap:
    def __init__(self, max_points: int = 1_000_000, max_observations: int = 5_000_000):
        self.points_3d = np.zeros((max_points, 3), dtype=np.float64)
        self.point_colors = np.zeros((max_points, 3), dtype=np.uint8)
        self.num_points = 0

        self.observations = np.zeros((max_observations, _OBS_COLS), dtype=np.float64)
        self.num_observations = 0

        self._kf_to_obs: dict[int, list[int]] = {}
        # kf_id -> {kp_idx -> point_id}. Used for O(1) "does this keyframe
        # already see a map point at this keypoint?" queries during keyframe
        # selection and loop-closure data association.
        self._kf_kp_to_pt: dict[int, dict[int, int]] = {}
        # pt_id -> idx of the keyframe in which this point was triangulated.
        # PGO uses this to anchor each point to a single, well-defined
        # keyframe so post-PGO point updates are unambiguous.
        self.triangulating_kf = np.full(max_points, -1, dtype=np.int64)
        # Points that have been merged into another point (duplicate fusion
        # across loop closures). Their slot in `points_3d` is preserved for
        # index stability; downstream code should skip them.
        self._dead_points: set[int] = set()

    def add_point(
        self,
        xyz: np.ndarray,
        color: np.ndarray | None = None,
        triangulating_kf_id: int | None = None,
    ) -> int:
        pt_id = self.num_points
        self.points_3d[pt_id] = xyz
        if color is not None:
            self.point_colors[pt_id] = np.clip(color, 0, 255).astype(np.uint8)
        if triangulating_kf_id is not None:
            self.triangulating_kf[pt_id] = int(triangulating_kf_id)
        self.num_points += 1
        return pt_id

    def add_observation(
        self,
        point_id: int,
        keyframe_id: int,
        uv_coords: np.ndarray,
        kp_idx: int = -1,
    ) -> int:
        obs_id = self.num_observations
        self.observations[obs_id, 0] = point_id
        self.observations[obs_id, 1] = keyframe_id
        self.observations[obs_id, 2:4] = uv_coords
        self.observations[obs_id, 4] = kp_idx
        self._kf_to_obs.setdefault(keyframe_id, []).append(obs_id)
        if kp_idx >= 0:
            self._kf_kp_to_pt.setdefault(keyframe_id, {})[int(kp_idx)] = int(point_id)
        self.num_observations += 1
        return obs_id

    def observations_for_keyframes(self, kf_ids: set[int]) -> list[int]:
        obs_indices = []
        for kf_id in kf_ids:
            obs_indices.extend(self._kf_to_obs.get(kf_id, []))
        return obs_indices

    def kf_kp_to_pt(self, kf_id: int) -> dict[int, int]:
        return self._kf_kp_to_pt.get(kf_id, {})

    def is_dead(self, pt_id: int) -> bool:
        return pt_id in self._dead_points

    def merge_points(self, survivor_id: int, dead_id: int) -> None:
        """Reassign all observations of `dead_id` to `survivor_id`.

        We keep `points_3d[dead_id]` in place (so global indexing stays
        stable) but mark it dead so it's filtered out of BA, PGO, and saved
        clouds. The survivor inherits the dead point's observations.
        """
        if survivor_id == dead_id:
            return
        if dead_id in self._dead_points:
            return
        # Move all observations of dead_id to point at survivor_id.
        for kf_id, kp_map in self._kf_kp_to_pt.items():
            for kp_idx, pt_id in list(kp_map.items()):
                if pt_id == dead_id:
                    kp_map[kp_idx] = survivor_id
        for obs_id in range(self.num_observations):
            if int(self.observations[obs_id, 0]) == dead_id:
                self.observations[obs_id, 0] = survivor_id
                kf_id = int(self.observations[obs_id, 1])
                # Ensure obs is reachable from survivor's KF index entries.
                # _kf_to_obs is keyed by keyframe so it's already correct.
                _ = kf_id
        self._dead_points.add(dead_id)

    def alive_point_ids(self) -> np.ndarray:
        if self.num_points == 0:
            return np.array([], dtype=np.int64)
        ids = np.arange(self.num_points, dtype=np.int64)
        if not self._dead_points:
            return ids
        mask = np.ones(self.num_points, dtype=bool)
        for d in self._dead_points:
            if 0 <= d < self.num_points:
                mask[d] = False
        return ids[mask]

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        ids = self.alive_point_ids()
        points = self.points_3d[ids]
        colors = self.point_colors[ids].astype(np.float64) / 255.0
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def save_pointcloud(self, filename: str) -> None:
        o3d.io.write_point_cloud(str(filename), self.to_open3d())


class Keyframe:
    def __init__(
        self,
        idx: int,
        keypoints,
        descriptors,
        pose: np.ndarray,
        match_scores: np.ndarray = None,
        global_descriptor: np.ndarray | None = None,
    ):
        self.idx = idx
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.pose = pose
        self.match_scores = match_scores if match_scores is not None else np.array([])
        # Cached LightGlue feature dict, set by the pipeline when available.
        # Keeping it on the keyframe avoids re-extracting features for cross-
        # loop-closure data association.
        self.feats: dict | None = None
        # Compact global descriptor for appearance-based loop-closure
        # pre-filtering. Optional; LC detector falls back to raw descriptor
        # matching when missing.
        self.global_descriptor = global_descriptor

    @property
    def R(self) -> np.ndarray:
        return self.pose[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.pose[:3, 3]


class Mapper:
    def __init__(self):
        self.pointmap = PointMap()
        self.keyframes: list[Keyframe] = []
        # Fast idx -> Keyframe lookup, kept consistent with `keyframes` list.
        self._kf_by_idx: dict[int, Keyframe] = {}

    def add_keyframe(self, keyframe: Keyframe) -> None:
        self.keyframes.append(keyframe)
        self._kf_by_idx[keyframe.idx] = keyframe

    def keyframe_by_idx(self, idx: int) -> Keyframe | None:
        return self._kf_by_idx.get(idx)

    @property
    def current_keyframe(self) -> Keyframe | None:
        return self.keyframes[-1] if self.keyframes else None

    @property
    def previous_keyframe(self) -> Keyframe | None:
        return self.keyframes[-2] if len(self.keyframes) > 1 else None

from __future__ import annotations
import numpy as np
import open3d as o3d


class PointMap:

    def __init__(self, max_points: int = 1000000, max_observations: int = 5000000):
        self.points_3d = np.zeros((max_points, 3), dtype=np.float64)
        self.point_colors = np.zeros((max_points, 3), dtype=np.uint8)
        self.num_points = 0
        self.observations = np.zeros((max_observations, 4), dtype=np.float64)
        self.num_observations = 0
        self._kf_to_obs: dict[int, list[int]] = {}

    def add_point(self, xyz: np.ndarray, color: np.ndarray | None = None) -> int:
        pt_id = self.num_points
        self.points_3d[pt_id] = xyz
        if color is not None:
            self.point_colors[pt_id] = np.clip(color, 0, 255).astype(np.uint8)
        self.num_points += 1
        return pt_id

    def add_observation(
        self, point_id: int, keyframe_id: int, uv_coords: np.ndarray
    ) -> int:
        obs_id = self.num_observations
        self.observations[obs_id, 0] = point_id
        self.observations[obs_id, 1] = keyframe_id
        self.observations[obs_id, 2:] = uv_coords
        self._kf_to_obs.setdefault(keyframe_id, []).append(obs_id)
        self.num_observations += 1
        return obs_id

    def observations_for_keyframes(self, kf_ids: set[int]) -> list[int]:
        obs_indices = []
        for kf_id in kf_ids:
            obs_indices.extend(self._kf_to_obs.get(kf_id, []))
        return obs_indices

    def to_open3d(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        points = self.points_3d[: self.num_points]
        colors = self.point_colors[: self.num_points].astype(np.float64) / 255.0
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
    ):
        self.idx = idx
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.pose = pose
        self.match_scores = match_scores if match_scores is not None else np.array([])
        self.kp_to_pt: dict[int, int] = {}

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

    def add_keyframe(self, keyframe: Keyframe) -> None:
        self.keyframes.append(keyframe)

    @property
    def current_keyframe(self) -> Keyframe | None:
        return self.keyframes[-1] if self.keyframes else None

    @property
    def previous_keyframe(self) -> Keyframe | None:
        return self.keyframes[-2] if len(self.keyframes) > 1 else None

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import theseus as th


@dataclass
class PoseGraphEdge:
    from_id: int
    to_id: int
    relative_pose: np.ndarray
    information: np.ndarray = field(default_factory=lambda: np.eye(6, dtype=np.float64))
    is_loop_closure: bool = False


class PoseGraph:

    def __init__(self) -> None:
        self._edges: list[PoseGraphEdge] = []

    def add_odometry_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        info = information if information is not None else np.eye(6, dtype=np.float64)
        self._edges.append(
            PoseGraphEdge(from_id, to_id, relative_pose, info, is_loop_closure=False)
        )

    def add_loop_closure_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: np.ndarray,
        information: np.ndarray | None = None,
    ) -> None:
        info = (
            information
            if information is not None
            else np.eye(6, dtype=np.float64) * 100.0
        )
        self._edges.append(
            PoseGraphEdge(from_id, to_id, relative_pose, info, is_loop_closure=True)
        )

    @property
    def edges(self) -> list[PoseGraphEdge]:
        return self._edges

    @property
    def loop_closure_edges(self) -> list[PoseGraphEdge]:
        return [e for e in self._edges if e.is_loop_closure]

    def has_loop_closures(self) -> bool:
        return any((e.is_loop_closure for e in self._edges))


def _pose_graph_error(optim_vars, aux_vars):
    pose_from, pose_to = optim_vars
    (rel_pose,) = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()


def _pose_graph_error_fixed_from(optim_vars, aux_vars):
    (pose_to,) = optim_vars
    pose_from, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()


def _pose_graph_error_fixed_to(optim_vars, aux_vars):
    (pose_from,) = optim_vars
    pose_to, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()


def _pose_graph_rot_error(optim_vars, aux_vars):
    pose_from, pose_to = optim_vars
    (rel_pose,) = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()[..., 3:]


def _pose_graph_rot_error_fixed_from(optim_vars, aux_vars):
    (pose_to,) = optim_vars
    pose_from, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()[..., 3:]


def _pose_graph_rot_error_fixed_to(optim_vars, aux_vars):
    (pose_from,) = optim_vars
    pose_to, rel_pose = aux_vars
    predicted = pose_to.inverse().compose(rel_pose).compose(pose_from)
    return predicted.log_map()[..., 3:]


@dataclass
class PoseGraphOptimizerConfig:
    max_iterations: int = 20
    abs_err_tolerance: float = 1e-05
    rel_err_tolerance: float = 1e-05
    step_size: float = 1.0


class PoseGraphOptimizer:

    def __init__(self, config: PoseGraphOptimizerConfig | None = None) -> None:
        self.config = config or PoseGraphOptimizerConfig()

    def optimize(self, mapper, pose_graph: PoseGraph) -> None:
        cfg = self.config
        print(f"\n[PG] --- Starting Pose Graph Optimization ---")
        if len(mapper.keyframes) < 2:
            print("[PG] Not enough keyframes. Skipping.")
            return
        if not pose_graph.has_loop_closures():
            print("[PG] No loop closures. Skipping.")
            return
        node_ids: set[int] = set()
        for edge in pose_graph.edges:
            node_ids.add(edge.from_id)
            node_ids.add(edge.to_id)
        kf_map = {kf.idx: kf for kf in mapper.keyframes if kf.idx in node_ids}
        if len(kf_map) < 2:
            print("[PG] Insufficient keyframes in pose graph. Skipping.")
            return
        print(
            f"[PG] Nodes: {len(kf_map)}, Edges: {len(pose_graph.edges)} ({len(pose_graph.loop_closure_edges)} loop closures)"
        )
        se3_vars: dict[int, th.SE3] = {}
        for kf_id, kf in kf_map.items():
            t = torch.from_numpy(kf.pose[:3, :]).unsqueeze(0).double()
            se3_vars[kf_id] = th.SE3(tensor=t, name=f"pg_cam_{kf_id}")
        anchor_id = min(kf_map.keys())
        print(f"[PG] Anchored keyframe: {anchor_id}")
        objective = th.Objective(dtype=torch.float64)
        for i, edge in enumerate(pose_graph.edges):
            if edge.from_id not in se3_vars or edge.to_id not in se3_vars:
                continue
            if edge.is_loop_closure:
                rel_4x4 = edge.relative_pose
            else:
                T_from_w = np.eye(4)
                T_from_w[:3, :] = kf_map[edge.from_id].pose[:3, :]
                T_to_w = np.eye(4)
                T_to_w[:3, :] = kf_map[edge.to_id].pose[:3, :]
                rel_4x4 = T_to_w @ np.linalg.inv(T_from_w)
            rel_t = torch.from_numpy(rel_4x4[:3, :]).unsqueeze(0).double()
            rel_var = th.SE3(tensor=rel_t, name=f"pg_rel_{i}")
            pose_from = se3_vars[edge.from_id]
            pose_to = se3_vars[edge.to_id]
            if edge.is_loop_closure:
                err_dim = 3
                err_full = _pose_graph_rot_error
                err_fixed_from = _pose_graph_rot_error_fixed_from
                err_fixed_to = _pose_graph_rot_error_fixed_to
                info_block = edge.information[3:, 3:]
                info_mean = float(np.trace(info_block) / 3.0)
            else:
                err_dim = 6
                err_full = _pose_graph_error
                err_fixed_from = _pose_graph_error_fixed_from
                err_fixed_to = _pose_graph_error_fixed_to
                info_mean = float(np.trace(edge.information) / 6.0)
            edge_weight = th.ScaleCostWeight(
                torch.tensor(np.sqrt(max(info_mean, 0.0)), dtype=torch.float64)
            )
            if edge.from_id == anchor_id:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_to],
                    err_fn=err_fixed_from,
                    dim=err_dim,
                    aux_vars=[pose_from, rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            elif edge.to_id == anchor_id:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_from],
                    err_fn=err_fixed_to,
                    dim=err_dim,
                    aux_vars=[pose_to, rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            else:
                cost = th.AutoDiffCostFunction(
                    optim_vars=[pose_from, pose_to],
                    err_fn=err_full,
                    dim=err_dim,
                    aux_vars=[rel_var],
                    cost_weight=edge_weight,
                    name=f"pg_edge_{i}",
                )
            objective.add(cost)
        if objective.size_cost_functions() == 0:
            print("[PG] No valid edges for optimization. Skipping.")
            return
        objective.update()
        initial_error = objective.error_metric().sum().item()
        print(f"[PG] Initial error: {initial_error:.4f}")
        optimizer = th.LevenbergMarquardt(
            objective,
            max_iterations=cfg.max_iterations,
            step_size=cfg.step_size,
            abs_err_tolerance=cfg.abs_err_tolerance,
            rel_err_tolerance=cfg.rel_err_tolerance,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver,
            vectorize=True,
        )
        info = optimizer.optimize()
        final_error = objective.error_metric().sum().item()
        print(f"[PG] Status      : {info.status[0]}")
        print(f"[PG] Iterations  : {info.converged_iter[0].item() + 1}")
        print(
            f"[PG] Final error : {final_error:.4f}  (delta {initial_error - final_error:.4f})"
        )
        pose_corrections: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = kf_map[kf_id]
            old_pose = kf.pose.copy()
            new_3x4 = cam_var.tensor.detach().cpu().numpy()[0]
            new_pose = np.eye(4)
            new_pose[:3, :] = new_3x4
            R_old, t_old = (old_pose[:3, :3], old_pose[:3, 3])
            R_new, t_new = (new_pose[:3, :3], new_pose[:3, 3])
            R_cw_new = R_new.T
            t_cw_new = -R_new.T @ t_new
            R_corr = R_cw_new @ R_old
            t_corr = R_cw_new @ t_old + t_cw_new
            pose_corrections[kf_id] = (R_corr, t_corr)
        sorted_kf_ids = sorted(kf_map.keys())
        pt_anchor: dict[int, int] = {}
        for kf_id in sorted_kf_ids:
            obs_ids = mapper.pointmap._kf_to_obs.get(kf_id, [])
            for obs_id in obs_ids:
                pt_id = int(mapper.pointmap.observations[obs_id, 0])
                if pt_id not in pt_anchor:
                    pt_anchor[pt_id] = kf_id
        corrected_pts = 0
        for pt_id, kf_id in pt_anchor.items():
            if kf_id == anchor_id:
                continue
            R_corr, t_corr = pose_corrections[kf_id]
            p = mapper.pointmap.points_3d[pt_id]
            mapper.pointmap.points_3d[pt_id] = R_corr @ p + t_corr
            corrected_pts += 1
        for kf_id, cam_var in se3_vars.items():
            if kf_id == anchor_id:
                continue
            kf = kf_map[kf_id]
            new_3x4 = cam_var.tensor.detach().cpu().numpy()[0]
            new_pose = np.eye(4)
            new_pose[:3, :] = new_3x4
            kf.pose = new_pose
        print(
            f"[PG] Corrected {corrected_pts} 3D points (single anchor per point, {len(pose_corrections)} keyframes updated)"
        )
        print(f"[PG] --- Pose Graph Optimization Complete ---\n")
